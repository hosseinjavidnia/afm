from __future__ import annotations

import json
from pathlib import Path

import torch

from afmvision.afm.behaviour import behaviour_from_logits, current_behaviour
from afmvision.afm.functional_shield import FunctionalShield
from afmvision.afm.records import CandidateState
from afmvision.afm.trainer import AFMTrainer, CandidateTransferBatch
from afmvision.config import load_config
from afmvision.models.factory import build_model


def _trainer(tmp_path: Path, seed: int = 8) -> AFMTrainer:
    torch.manual_seed(seed)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/smoke.yaml")
    cfg["training"]["device"] = "cpu"
    trainer = AFMTrainer(build_model(cfg), cfg, torch.device("cpu"), tmp_path / "run")
    trainer._initialise_afm_state()
    return trainer


def _random_target_logits(logits: torch.Tensor, offset: float) -> torch.Tensor:
    return logits.flip(1) + torch.tensor([[offset, -offset]], dtype=logits.dtype)


def test_compact_cardinal_shield_interpolates_vector_logits_exactly() -> None:
    torch.manual_seed(1)
    shield = FunctionalShield(feature_dim=5, output_dim=3, max_nodes=20)
    nodes = torch.randn(12, 5, dtype=torch.float64)
    targets = torch.randn(12, 3, dtype=torch.float64)
    guards = torch.randn(32, 5, dtype=torch.float64)
    result = shield.solve_and_replace(
        nodes,
        targets,
        guard_nodes=guards,
        support_multiplier=4.0,
        residual_tolerance=1e-12,
    )
    assert result.available
    assert result.merged_node_count == len(nodes)
    assert result.condition_number == 1.0
    assert result.maximum_guard_leakage <= 1e-12
    predicted = shield(nodes).to(torch.float64)
    assert torch.allclose(predicted, targets, atol=1e-12, rtol=1e-12)


def test_replay_envelope_is_exact_and_support_is_tiny() -> None:
    shield = FunctionalShield(feature_dim=1, output_dim=1, max_nodes=2)
    result = shield.solve_and_replace(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
        feature_match_tolerance=1e-8,
        support_multiplier=4.0,
    )
    assert result.available
    assert result.minimum_support_radius == 4e-8
    within = torch.tensor([[0.5e-8]], dtype=torch.float64)
    outside = torch.tensor([[5.0e-8]], dtype=torch.float64)
    assert torch.equal(shield(within), torch.tensor([[2.0]], dtype=torch.float64))
    assert torch.equal(shield(outside), torch.zeros((1, 1), dtype=torch.float64))


def test_compact_shield_is_exactly_zero_on_nonmatching_guard_addresses() -> None:
    shield = FunctionalShield(feature_dim=2, output_dim=2, max_nodes=8)
    nodes = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0, -1.0], [-0.5, 0.5]], dtype=torch.float64)
    guards = torch.tensor([[0.5, 0.2], [1.0, 1.0], [3.0, 0.0]], dtype=torch.float64)
    result = shield.solve_and_replace(nodes, targets, guard_nodes=guards)
    assert result.available
    assert result.guard_count == 3
    assert torch.equal(shield(guards), torch.zeros_like(targets.new_zeros((3, 2))))


def test_compact_shield_has_no_gaussian_tail_on_far_queries() -> None:
    shield = FunctionalShield(feature_dim=1, output_dim=1, max_nodes=4)
    nodes = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    targets = torch.tensor([[2.0], [-3.0]], dtype=torch.float64)
    result = shield.solve_and_replace(nodes, targets)
    assert result.available
    far = torch.tensor([[10.0], [-10.0], [4.0]], dtype=torch.float64)
    assert torch.equal(shield(far), torch.zeros((3, 1), dtype=torch.float64))


def test_supports_are_disjoint_and_residuals_do_not_accumulate() -> None:
    shield = FunctionalShield(feature_dim=1, output_dim=1, max_nodes=4)
    nodes = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    targets = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
    result = shield.solve_and_replace(nodes, targets)
    assert result.available
    probes = torch.linspace(-1.0, 3.0, 401, dtype=torch.float64).unsqueeze(1)
    values = shield(probes).abs().squeeze(1)
    assert float(values.max().item()) <= 3.0 + 1e-12


def test_too_close_distinct_addresses_abstain_atomically() -> None:
    shield = FunctionalShield(feature_dim=1, output_dim=1, max_nodes=4)
    assert shield.solve_and_replace(
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        torch.tensor([[0.2], [-0.1]], dtype=torch.float64),
    ).available
    before = shield.snapshot()
    result = shield.solve_and_replace(
        torch.tensor([[0.0], [3.0e-8]], dtype=torch.float64),
        torch.tensor([[1.0], [1.0]], dtype=torch.float64),
        duplicate_tolerance=0.0,
        feature_match_tolerance=1.0e-8,
        support_multiplier=4.0,
    )
    assert not result.available
    assert result.obstruction == "functional_shield_address_resolution_obstruction"
    after = shield.snapshot()
    for key in ("centres", "coefficients", "support_radii", "match_radii"):
        assert torch.equal(after[key], before[key])


def test_duplicate_constraint_inconsistency_is_sharp_and_atomic() -> None:
    shield = FunctionalShield(feature_dim=2, output_dim=2, max_nodes=8)
    initial_nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    initial_targets = torch.tensor([[0.2, -0.2], [0.1, 0.3]], dtype=torch.float64)
    assert shield.solve_and_replace(initial_nodes, initial_targets).available
    before = shield.snapshot()

    nodes = torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    result = shield.solve_and_replace(
        nodes,
        targets,
        duplicate_tolerance=0.0,
        target_tolerance=1e-12,
    )
    assert not result.available
    assert result.obstruction == "functional_constraint_inconsistency"
    after = shield.snapshot()
    for key in ("centres", "coefficients", "support_radii", "match_radii"):
        assert torch.equal(after[key], before[key])
    assert after["generation"] == before["generation"]


def test_shield_snapshot_round_trip_is_exact() -> None:
    torch.manual_seed(3)
    shield = FunctionalShield(feature_dim=4, output_dim=2, max_nodes=10)
    nodes = torch.randn(6, 4, dtype=torch.float64)
    targets = torch.randn(6, 2, dtype=torch.float64)
    assert shield.solve_and_replace(nodes, targets).available
    state = shield.snapshot()
    probe = torch.randn(5, 4)
    expected = shield(probe).clone()
    shield.clear()
    shield.restore(state)
    assert torch.equal(shield(probe), expected)


def test_trainer_shield_preserves_full_ordinary_comparator_and_all_candidates(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    current_images = torch.rand(4, 3, 64, 64)
    selected_images = torch.rand(1, 3, 64, 64)
    safeguard_images = torch.rand(1, 3, 64, 64)

    trainer.model.eval()
    with torch.no_grad():
        current_logits = trainer.model(current_images)
        current_labels = (current_logits.argmax(dim=1) + 1) % 2
        selected_logits = trainer.model(selected_images)
        safeguard_logits = trainer.model(safeguard_images)
        selected_target_logits = _random_target_logits(selected_logits, 0.3)
        safeguard_target_logits = _random_target_logits(safeguard_logits, -0.2)
        selected_targets = behaviour_from_logits(selected_target_logits, trainer.behaviour_spec)
        safeguard_targets = behaviour_from_logits(safeguard_target_logits, trainer.behaviour_spec)
        safeguard_before_logits = safeguard_logits.clone()
        safeguard_before_objective = float(
            (0.5 * (current_behaviour(trainer.model, safeguard_images, trainer.behaviour_spec)
                    - safeguard_targets).square().sum(dim=1).mean()).item()
        )

    trainer.candidates[0] = CandidateState(
        slot=0, created_step=0, candidate_id=10, certified=True, certified_step=-1, certified_ucb=0.1
    )
    trainer.candidates[1] = CandidateState(
        slot=1, created_step=0, candidate_id=11, certified=True, certified_step=-1, certified_ucb=0.1
    )
    batches = [
        CandidateTransferBatch(10, 0, selected_images, selected_targets, selected_target_logits, "unit"),
        CandidateTransferBatch(11, 1, safeguard_images, safeguard_targets, safeguard_target_logits, "unit"),
    ]

    result = trainer._safe_update(current_images, current_labels, transfer_batches=batches)
    assert result["accepted"]
    assert result["transfer_joint_step"]
    assert result["functional_shield_update_norm"] > 0.0

    with torch.no_grad():
        selected_after_logits = trainer.model(selected_images)
        safeguard_after_logits = trainer.model(safeguard_images)
        safeguard_after_objective = float(
            (0.5 * (current_behaviour(trainer.model, safeguard_images, trainer.behaviour_spec)
                    - safeguard_targets).square().sum(dim=1).mean()).item()
        )
    assert torch.allclose(selected_after_logits, selected_target_logits, atol=2e-5, rtol=2e-5)
    assert torch.allclose(safeguard_after_logits, safeguard_before_logits, atol=2e-5, rtol=2e-5)
    assert abs(safeguard_after_objective - safeguard_before_objective) <= 2e-6

    event = json.loads((trainer.run_dir / "events.jsonl").read_text().splitlines()[-1])
    assert event["joint_solver_mode"] == "joint_counterfactual_normalized_assimilation"
    assert event["persistent_base_mode"] == "counterfactual_normalized_metaplastic_endpoint"
    assert event["retention_budget_mode"] == "counterfactual_normalized"
    assert event["realised_counterfactual_path_fraction"] >= event["requested_counterfactual_charge_fraction"] - 1e-6
    assert event["persistent_base_progress_ratio"] > 0.0
    assert event["persistent_descent_lower_bound"] >= 0.0
    assert event["exact_counterfactual_restoration_attempted"]
    assert event["exact_counterfactual_restoration_accepted"]
    assert event["exact_counterfactual_decrease"] > 0.0
    assert event["exact_counterfactual_progress_ratio"] >= 0.999999
    assert event["joint_current_certified_decrease"] >= event["joint_current_required"]
    assert event["functional_shield_interpolation_residual"] <= trainer.cfg["afm"]["functional_shield"]["residual_tolerance"]
    assert event["functional_shield_node_count"] <= trainer.model.functional_shield.max_nodes
    assert event["functional_shield_maximum_guard_leakage"] == 0.0


def test_unrelated_predictions_equal_base_logits_after_deployment(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, seed=44)
    current_images = torch.rand(4, 3, 64, 64)
    candidate_images = torch.rand(2, 3, 64, 64)
    probe_images = torch.rand(16, 3, 64, 64)
    with torch.no_grad():
        logits = trainer.model(current_images)
        labels = (logits.argmax(dim=1) + 1) % 2
        candidate_logits = trainer.model(candidate_images)
        target_logits = _random_target_logits(candidate_logits, 0.4)
        targets = behaviour_from_logits(target_logits, trainer.behaviour_spec)
    trainer.candidates[0] = CandidateState(
        slot=0, created_step=0, candidate_id=40, certified=True, certified_step=-1, certified_ucb=0.1
    )
    result = trainer._safe_update(
        current_images,
        labels,
        transfer_batches=[CandidateTransferBatch(40, 0, candidate_images, targets, target_logits, "unit")],
    )
    assert result["accepted"]
    with torch.no_grad():
        features = trainer.model.encode_backbone(probe_images)
        shield_values = trainer.model.functional_shield(features)
    assert torch.equal(shield_values, torch.zeros_like(shield_values))


def test_identical_observable_node_with_conflicting_current_and_candidate_targets_abstains(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, seed=9)
    images = torch.rand(4, 3, 64, 64)
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(images)
        labels = (logits.argmax(dim=1) + 1) % 2
        target_logits = _random_target_logits(logits, 0.5)
        targets = behaviour_from_logits(target_logits, trainer.behaviour_spec)
    candidate = CandidateState(
        slot=0, created_step=0, candidate_id=20, certified=True, certified_step=-1, certified_ucb=0.1
    )
    trainer.candidates[0] = candidate
    batch = CandidateTransferBatch(20, 0, images, targets, target_logits, "collision")
    assert trainer.vectoriser is not None
    vector_before = trainer.vectoriser.flatten(detach=True).clone()
    shield_before = trainer.model.functional_shield.snapshot()

    result = trainer._safe_update(images, labels, transfer_batches=[batch])
    assert not result["accepted"]
    assert torch.equal(trainer.vectoriser.flatten(detach=True), vector_before)
    shield_after = trainer.model.functional_shield.snapshot()
    for key in ("centres", "coefficients", "support_radii", "match_radii"):
        assert torch.equal(shield_after[key], shield_before[key])
    event = json.loads((trainer.run_dir / "events.jsonl").read_text().splitlines()[-1])
    assert event["functional_shield_obstruction"] == "functional_constraint_inconsistency"
    assert event["transfer_priority_obstruction"] == "functional_constraint_inconsistency"


def test_persistent_state_contains_deployed_frozen_and_guard_state(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, seed=10)
    snapshot = trainer.model.functional_shield.snapshot()
    candidate = CandidateState(
        slot=0,
        created_step=0,
        candidate_id=30,
        snapshot=torch.zeros_like(trainer.vectoriser.flatten(detach=True).cpu()),
        snapshot_shield_state=snapshot,
        certified=True,
    )
    trainer.candidates[0] = candidate
    state = trainer._persistent_state_dict()
    assert "functional_shield" in state
    assert "functional_shield_guard_features" in state
    assert state["candidates"][0]["snapshot_shield_state"] is not None


def test_model_state_dict_load_resizes_nonempty_shield_buffers() -> None:
    torch.manual_seed(31)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/smoke.yaml")
    source = build_model(cfg)
    source.functional_shield.max_nodes = 64
    nodes = torch.randn(64, source.functional_shield.feature_dim, dtype=torch.float64)
    targets = torch.randn(64, source.functional_shield.output_dim, dtype=torch.float64)
    result = source.functional_shield.solve_and_replace(nodes, targets)
    assert result.available

    state = source.state_dict()
    restored = build_model(cfg)
    assert restored.functional_shield.node_count == 0
    restored.load_state_dict(state)

    assert restored.functional_shield.node_count == 64
    assert restored.functional_shield.max_nodes >= 64
    probe = torch.randn(7, source.functional_shield.feature_dim)
    assert torch.equal(restored.functional_shield(probe), source.functional_shield(probe))
