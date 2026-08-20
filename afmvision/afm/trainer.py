from __future__ import annotations

import copy
import json
import random
from contextlib import contextmanager
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from afmvision.afm.behaviour import BehaviourSpec, behaviour_from_logits, current_behaviour, stream_behaviour_jacobians
from afmvision.afm.eprocess import HalfNormalMixtureEProcess
from afmvision.afm.error_budget import LifetimeErrorAllocator
from afmvision.afm.frequent_directions import FrequentDirections
from afmvision.afm.functional_shield import ShieldSolveResult, temporary_shield
from afmvision.afm.full_progress_restoration import RestorationBlock, replace_with_endpoint_emulation
from afmvision.afm.metaplastic import (
    MetaplasticController,
    blocked_gradient_fraction,
    make_policy_family,
    spectral_residual,
    top_basis_from_weighted_sketches,
)
from afmvision.afm.parameter_vector import ParameterVector, temporary_parameters
from afmvision.afm.persistent_assimilation import (
    make_counterfactual_normalized_plan,
    persistent_descent_lower_bound,
)
from afmvision.afm.records import CandidateState, ProtectedRecord, ShadowChallenger
from afmvision.afm.router import BoundedCentroidRouter
from afmvision.afm.safe_step import (
    SafeStep,
    joint_progress_protected_step,
    make_safe_step,
    make_priority_safe_step,
    priority_constrained_transfer_direction,
    project_to_allowed_free_subspace,
    quadratic_decrease,
    unconstrained_ball_best,
)
from afmvision.afm.signature import CausalContextSignature
from afmvision.data.stream import load_rows
from afmvision.eval.metrics import ExperimentSummary
from afmvision.instrumentation import RunInstrumentation
from afmvision.models.convnet_adapters import AFMConvNet
from afmvision.utils.io import ensure_dir
from afmvision.utils.logging import JSONLLogger


def error_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) != labels).float().mean().item())


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def consolidation_ucb(count: int, loss_sum: float, alpha: float) -> float:
    if count <= 0:
        return float("inf")
    mean = float(loss_sum) / int(count)
    radius = math.sqrt(math.log(math.pi**2 * count**2 / (6.0 * alpha)) / (2.0 * count))
    return mean + radius


@dataclass
class RenewalTrial:
    slot: int
    adapter_state: dict[str, torch.Tensor]
    previous_gate: float
    zero_change: float
    budget_index: int
    alpha: float


@dataclass
class ReopeningPredictionBatch:
    """Pre-outcome predictions for one immutable record in one optimiser round."""

    record_id: int
    slot: int
    features: torch.Tensor
    record_logits: torch.Tensor
    challenger_logits: torch.Tensor
    signatures: torch.Tensor
    signature_block_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))


@dataclass
class CandidateTransferBatch:
    candidate_id: int
    slot: int
    images: torch.Tensor
    targets: torch.Tensor
    target_logits: torch.Tensor
    source: str


@dataclass
class FunctionalShieldDeployment:
    accepted: bool
    factor: float
    drift: float
    new_loss: float
    selected_after: float | None
    candidate_before: dict[int, float]
    candidate_after: dict[int, float]
    candidate_decreases: tuple[float, ...]
    current_ordinary_best: float
    current_required: float
    current_certified_decrease: float
    selected_certified_decrease: float
    solve: ShieldSolveResult | None
    obstruction: str | None
    shield_update_norm: float
    selected_candidate_id: int | None
    proposal: SafeStep | None = None
    counterfactual_accepted: bool = False
    counterfactual_decrease: float = 0.0
    exact_progress_ratio: float | None = None
    maximum_endpoint_error: float = 0.0
    safe_base_accepted: bool = False
    safe_base_decrease: float = 0.0
    safe_base_drift: float = 0.0
    safe_base_radius: float = 0.0
    safe_base_step_length: float = 0.0
    retention_budget: float = 0.0
    reference_retention_charge: float = 0.0
    requested_charge_fraction: float = 0.0
    selected_path_fraction: float = 0.0
    realised_path_fraction: float = 0.0
    persistent_base_progress_ratio: float | None = None
    persistent_descent_lower_bound: float = 0.0
    projected_counterfactual_alignment_error: float = 0.0
    ordinary_counterfactual_alignment_error: float = 0.0
    projection_idempotence_error: float = 0.0
    compatible_gradient_fraction: float = 0.0
    ordinary_step_size: float = 0.0
    step_size_smoothness_product: float = 0.0
    scalar_comparator_certified: bool = False
    analytic_persistent_progress_ratio_lower_bound: float = 0.0
    certified_persistent_progress_ratio_lower_bound: float | None = None
    shield_residual_progress_fraction: float | None = None


class AFMTrainer:
    """AFM-U nonlinear empirical whole-behaviour research implementation.

    Structural and stochastic state follow the manuscript. ``strict`` mode
    abstains without externally certified pre-step bounds. ``empirical`` mode
    is an explicitly non-theorem-certified potential test with exact endpoint
    checks on every frozen commit block.
    """

    def __init__(self, model: AFMConvNet, config: dict[str, Any], device: torch.device, run_dir: Path):
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.run_dir = ensure_dir(run_dir)
        self.logger = JSONLLogger(self.run_dir / "events.jsonl")
        self.summary = ExperimentSummary()
        balance_cfg = dict(config.get("training", {}).get("diagnostic_class_balance", {}))
        self.diagnostic_class_balance_enabled = bool(balance_cfg.get("enabled", False))
        self.diagnostic_class_balance_after_bootstrap_only = bool(
            balance_cfg.get("after_bootstrap_only", True)
        )
        if self.diagnostic_class_balance_enabled and not self.diagnostic_class_balance_after_bootstrap_only:
            raise ValueError(
                "diagnostic_class_balance is intentionally implemented only after the "
                "representation bootstrap so the frozen backbone is not changed"
            )
        raw_weights = balance_cfg.get("weights")
        if self.diagnostic_class_balance_enabled:
            if raw_weights is None:
                raise ValueError("training.diagnostic_class_balance.weights is required when enabled")
            if len(raw_weights) != int(config["model"]["num_classes"]):
                raise ValueError(
                    "diagnostic class-weight count must equal model.num_classes: "
                    f"{len(raw_weights)} != {int(config['model']['num_classes'])}"
                )
            weights = torch.tensor(raw_weights, dtype=torch.float32, device=device)
            if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
                raise ValueError("diagnostic class weights must be finite and strictly positive")
            self.diagnostic_class_weights: torch.Tensor | None = weights
        else:
            self.diagnostic_class_weights = None
        self.protection_enabled = bool(config.get("afm", {}).get("protection", {}).get("enabled", True))
        self.instrumentation = RunInstrumentation(
            config, self.run_dir, "afm" if self.protection_enabled else "afm_no_protection"
        )
        self.step = 0
        self.afm_round = 0
        self.protection_round = 0
        self.record_counter = 0
        self.candidate_counter = 0
        self.candidate_replacements = 0
        self.candidate_pretest_replacements = 0
        self.candidate_tests_started = 0
        self.candidate_training_runs = 0
        self.candidate_validation_rejections = 0
        self.route_splits = 0
        self.route_split_obstructions = 0
        self.transfer_attempts = 0
        self.transfer_common_descent_steps = 0  # legacy alias
        self.transfer_priority_feasible_steps = 0
        self.transfer_accepted_steps = 0
        self.transfer_fallback_steps = 0
        self.transfer_obstructions = 0
        self.transfer_incompatible_obstructions = 0
        self.transfer_endpoint_obstructions = 0
        self.candidate_certifications = 0
        self.candidate_staleness_rejections = 0
        self.functional_shield_attempts = 0
        self.functional_shield_accepted = 0
        self.functional_shield_obstructions = 0
        self.functional_shield_inconsistencies = 0
        self.functional_shield_numerical_obstructions = 0
        self.exact_restoration_attempts = 0
        self.exact_restoration_accepted = 0
        self.exact_restoration_obstructions = 0
        self._last_split_step_by_slot: dict[int, int] = {}
        self.stall_count = 0
        self.effective_total_steps = int(config["training"]["max_steps"])
        self.effective_afm_horizon = max(self.effective_total_steps - int(config["training"].get("bootstrap_batches", 0)), 1)
        self.last_step_diagnostics: dict[str, float] | None = None
        self.all_nonzero_steps_certified = True

        afm = config["afm"]
        self.behaviour_spec = BehaviourSpec.from_config(config)
        shield_cfg = dict(afm.get("functional_shield", {}))
        self.functional_shield_enabled = bool(shield_cfg.get("enabled", True))
        restoration_cfg = dict(afm.get("exact_counterfactual_restoration", {}))
        self.exact_counterfactual_restoration_enabled = bool(
            restoration_cfg.get("enabled", True)
        ) and self.functional_shield_enabled
        declared_max_nodes = int(
            shield_cfg.get(
                "max_nodes",
                int(config["training"].get("batch_size", 1))
                + int(afm["resources"].get("max_records", 1))
                * int(afm["consolidation"].get("commit_samples", 1))
                + int(afm["resources"].get("max_contexts", 1))
                * int(afm.get("transfer", {}).get("samples", afm["consolidation"].get("commit_samples", 1))),
            )
        )
        self.model.functional_shield.max_nodes = declared_max_nodes
        router_cfg = afm["router"]
        self.router = BoundedCentroidRouter(
            max_slots=int(afm["resources"].get("max_contexts", afm["resources"]["max_records"])),
            threshold=float(router_cfg["threshold"]),
            centroid_rate=float(router_cfg["centroid_rate"]),
            freeze_committed_centroids=bool(router_cfg.get("freeze_committed_centroids", True)),
        )
        signature_cfg = dict(router_cfg.get("signature", {}))
        signature_kind = str(signature_cfg.get("kind", "style_moments"))
        signature_cfg.setdefault(
            "dimension",
            (54 if signature_kind == "block_style_moments" else
             (2 * (27 + int(config["model"]["feature_dim"])) if signature_kind == "block_hybrid_moments" else
              (27 if signature_kind == "style_moments" else int(config["model"]["feature_dim"])))),
        )
        self.context_signature = CausalContextSignature(
            config=signature_cfg,
            fixed_threshold=float(router_cfg["threshold"]),
            centroid_rate=float(router_cfg["centroid_rate"]),
        )
        self.candidates: dict[int, CandidateState] = {}
        # Append-only segment registry keyed by immutable record_id.  Multiple
        # active records may belong to the same routing-context slot.
        self.records: dict[int, ProtectedRecord] = {}

        stats = afm.get("statistics", {})
        self.error_allocator = LifetimeErrorAllocator(
            total_delta=float(stats.get("total_delta", 0.05)),
            category_weights=dict(
                stats.get(
                    "category_weights",
                    {
                        "consolidation": 0.25,
                        "candidate_staleness": 0.10,
                        "reopening": 0.20,
                        "route_split": 0.15,
                        "renewal": 0.10,
                        "policy": 0.10,
                        "numerics": 0.10,
                    },
                )
            ),
        )

        self.bootstrap_batches = int(config["training"].get("bootstrap_batches", 0))
        # Bounded learner-visible references are replayed only after bootstrap
        # through the final frozen backbone.  This avoids mixing features from
        # a sequence of changing bootstrap representations.
        self.signature_prefix_rows: list[dict[str, Any]] = []
        self.functional_shield_guard_capacity = int(
            shield_cfg.get(
                "guard_capacity",
                max(1, self.bootstrap_batches * int(config["training"].get("batch_size", 1))),
            )
        )
        self.functional_shield_guard_features = torch.empty(
            (0, int(config["model"]["feature_dim"])), dtype=torch.float64
        )
        self.bootstrap_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["training"].get("bootstrap_lr", 1e-3)),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        self.vectoriser: ParameterVector | None = None
        self.controller: MetaplasticController | None = None
        self.policy_family = None

        self.certificate_mode = str(afm["certificates"].get("mode", "empirical"))
        if self.certificate_mode not in {"empirical", "strict"}:
            raise ValueError("afm.certificates.mode must be 'empirical' or 'strict'")
        self._validate_configuration()
        self.logger.log(
            "initialised",
            certificate_mode=self.certificate_mode,
            device=str(device),
            theorem_scope="empirical whole commit-block behaviour",
            strict_mode_requested=self.certificate_mode == "strict",
            theorem_certified=False,
            total_delta=self.error_allocator.total_delta,
            behaviour_kind=self.behaviour_spec.kind,
            diagnostic_class_balance_enabled=self.diagnostic_class_balance_enabled,
            diagnostic_class_balance_after_bootstrap_only=self.diagnostic_class_balance_after_bootstrap_only,
            diagnostic_class_weights=(
                self.diagnostic_class_weights.detach().cpu().tolist()
                if self.diagnostic_class_weights is not None else None
            ),
        )

    def _current_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Current-stream scalar loss used by AFM and its no-protection counterpart.

        The optional diagnostic weighting is deliberately applied only after the
        frozen-representation bootstrap.  Per-example weighted losses are averaged
        over the physical batch (rather than PyTorch's weight-normalised mean), so
        inverse-frequency weights actually change aggregate class pressure even in
        temporally homogeneous batches.  The builder normalises the fixed weights so
        their mean over post-bootstrap learner examples is one.
        """

        if (
            not self.diagnostic_class_balance_enabled
            or self.diagnostic_class_weights is None
            or self.step < self.bootstrap_batches
        ):
            return torch.nn.functional.cross_entropy(logits, labels)
        per_example = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        sample_weights = self.diagnostic_class_weights.index_select(0, labels)
        return (per_example * sample_weights).mean()

    def _validate_configuration(self) -> None:
        afm = self.cfg["afm"]
        training = self.cfg["training"]
        resources = afm["resources"]
        consolidation = afm["consolidation"]
        metaplastic = afm["metaplastic"]
        safe_update = afm["safe_update"]
        reopening = afm["reopening"]
        renewal = afm["renewal"]

        max_steps = int(training["max_steps"])
        if max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        if self.bootstrap_batches < 0:
            raise ValueError("training.bootstrap_batches must be nonnegative")
        horizon = max(max_steps - self.bootstrap_batches, 1)

        K = int(metaplastic["timescales"])
        if K <= 0:
            raise ValueError("metaplastic.timescales must be positive")
        required = math.ceil(math.log2(horizon)) + 2
        if bool(metaplastic.get("enforce_horizon_coverage", True)) and K < required:
            raise ValueError(f"timescales={K} does not cover horizon {horizon}; theorem requires at least {required}")
        alphas = [float(value) for value in metaplastic["alphas"]]
        if not alphas or any(not 0.0 < value <= 1.0 for value in alphas):
            raise ValueError("Every scale-free alpha must lie in (0,1]")
        ranks = [int(value) for value in metaplastic["ranks"]]
        if not ranks or any(value < 0 for value in ranks):
            raise ValueError("Metaplastic ranks must be a nonempty list of nonnegative integers")
        if float(metaplastic["zeta"]) < 0.0:
            raise ValueError("metaplastic.zeta must be nonnegative")

        risk_threshold = float(consolidation["risk_threshold"])
        if not 0.0 <= risk_threshold <= 1.0:
            raise ValueError("Consolidation risk threshold must lie in [0,1]")
        legacy_batches = int(consolidation.get("freeze_after_batches", 0))
        training_samples = int(
            consolidation.get(
                "training_samples",
                legacy_batches * int(training.get("batch_size", 1)),
            )
        )
        if training_samples <= 0:
            raise ValueError("consolidation.training_samples must be positive")
        if int(consolidation.get("training_epochs", 1)) <= 0:
            raise ValueError("consolidation.training_epochs must be positive")
        if int(consolidation.get("training_batch_size", training_samples)) <= 0:
            raise ValueError("consolidation.training_batch_size must be positive")
        if float(consolidation.get("training_lr", training.get("bootstrap_lr", 1e-3))) <= 0.0:
            raise ValueError("consolidation.training_lr must be positive")
        if float(consolidation.get("training_weight_decay", 0.0)) < 0.0:
            raise ValueError("consolidation.training_weight_decay must be nonnegative")
        if not 0.0 <= float(consolidation.get("adam_beta1", 0.9)) < 1.0:
            raise ValueError("consolidation.adam_beta1 must lie in [0,1)")
        if not 0.0 <= float(consolidation.get("adam_beta2", 0.999)) < 1.0:
            raise ValueError("consolidation.adam_beta2 must lie in [0,1)")
        if float(consolidation.get("adam_epsilon", 1e-8)) <= 0.0:
            raise ValueError("consolidation.adam_epsilon must be positive")
        min_validation = int(consolidation["min_validation_samples"])
        if min_validation <= 0:
            raise ValueError("min_validation_samples must be positive")
        max_validation = int(consolidation.get("max_validation_samples", min_validation))
        if max_validation < min_validation:
            raise ValueError("max_validation_samples must be at least min_validation_samples")
        if int(consolidation["commit_samples"]) <= 0:
            raise ValueError("commit_samples must be positive")

        if int(resources.get("max_contexts", resources["max_records"])) <= 0:
            raise ValueError("max_contexts must be positive")
        if int(resources["max_records"]) <= 0:
            raise ValueError("max_records must be positive")
        if int(resources.get("max_atoms", resources["max_records"])) <= 0:
            raise ValueError("max_atoms must be positive")
        if int(resources.get("evidence_reference_bytes", 4096)) <= 0:
            raise ValueError("evidence_reference_bytes must be positive")
        if int(resources["total_sketch_rows"]) <= 0:
            raise ValueError("total_sketch_rows must be positive")
        if int(resources["max_rank"]) < 0:
            raise ValueError("max_rank must be nonnegative")
        if str(resources.get("capacity_policy", "release_oldest")) not in {"release_oldest", "overflow"}:
            raise ValueError("capacity_policy must be 'release_oldest' or 'overflow'")
        if int(afm["memory"]["sketch_rows"]) <= 0:
            raise ValueError("sketch_rows must be positive")
        if int(afm["memory"]["sketch_rows"]) > int(resources["total_sketch_rows"]):
            raise ValueError("A single sketch cannot exceed the total sketch-row budget")
        if float(afm["memory"]["declared_sketch_frobenius_sq"]) < 0.0:
            raise ValueError("declared_sketch_frobenius_sq must be nonnegative")

        router = afm["router"]
        if float(router["threshold"]) < 0.0:
            raise ValueError("router.threshold must be nonnegative")
        if not 0.0 < float(router["centroid_rate"]) <= 1.0:
            raise ValueError("router.centroid_rate must lie in (0,1]")
        signature = dict(router.get("signature", {}))
        if str(signature.get("kind", "block_style_moments")) not in {
            "style_moments",
            "block_style_moments",
            "backbone",
            "block_backbone_mean",
            "block_hybrid_moments",
        }:
            raise ValueError(
                "router.signature.kind must be 'style_moments', 'block_style_moments', "
                "'backbone', 'block_backbone_mean', or 'block_hybrid_moments'"
            )
        if int(signature.get("dimension", self.cfg["model"]["feature_dim"])) <= 0:
            raise ValueError("router.signature.dimension must be positive")
        if int(signature.get("block_size", 1)) <= 0:
            raise ValueError("router.signature.block_size must be positive")
        if not 0.0 < float(signature.get("temporal_rate", 0.05)) <= 1.0:
            raise ValueError("router.signature.temporal_rate must lie in (0,1]")
        if float(signature.get("clip", 8.0)) <= 0.0:
            raise ValueError("router.signature.clip must be positive")
        if float(signature.get("scale_floor", 1e-3)) <= 0.0:
            raise ValueError("router.signature.scale_floor must be positive")
        if not 0.0 < float(signature.get("calibration_quantile", 0.995)) <= 1.0:
            raise ValueError("router.signature.calibration_quantile must lie in (0,1]")
        if float(signature.get("calibration_margin", 1.10)) < 1.0:
            raise ValueError("router.signature.calibration_margin must be at least 1")
        calibration_floor = float(signature.get("calibration_floor", 0.05))
        calibration_ceiling = float(signature.get("calibration_ceiling", 2.0))
        if not 0.0 <= calibration_floor <= calibration_ceiling:
            raise ValueError("router.signature calibration floor/ceiling are inconsistent")
        if str(signature.get("calibration_rule", "replace")) not in {"replace", "max_fixed"}:
            raise ValueError("router.signature.calibration_rule must be 'replace' or 'max_fixed'")
        split = dict(router.get("split", {}))
        if int(split.get("min_signature_blocks", split.get("min_observations", 1))) <= 0:
            raise ValueError("router.split.min_signature_blocks must be positive")
        if float(split.get("reference_margin", 0.0)) < 0.0:
            raise ValueError("router.split.reference_margin must be nonnegative")
        if float(split.get("reference_floor", 0.0)) < 0.0:
            raise ValueError("router.split.reference_floor must be nonnegative")
        if float(split.get("min_effect_distance", split.get("min_signature_distance", 0.0))) < 0.0:
            raise ValueError("router.split.min_effect_distance must be nonnegative")
        if float(split.get("signature_sigma", 0.5)) < 0.5:
            raise ValueError(
                "router.split.signature_sigma must be at least 0.5 for the declared "
                "bounded score W=(D-r)/2"
            )
        if float(split.get("signature_prior_scale", 1.0)) <= 0.0:
            raise ValueError("router.split.signature_prior_scale must be positive")
        if "route_split" not in self.error_allocator.category_weights:
            raise ValueError("afm.statistics.category_weights must include route_split")
        if "candidate_staleness" not in self.error_allocator.category_weights:
            raise ValueError("afm.statistics.category_weights must include candidate_staleness")

        transfer = dict(afm.get("transfer", {}))
        if bool(transfer.get("enabled", True)):
            transfer_samples = int(transfer.get("samples", consolidation["commit_samples"]))
            if transfer_samples <= 0:
                raise ValueError("afm.transfer.samples must be positive")
            if transfer_samples < int(consolidation["commit_samples"]):
                raise ValueError(
                    "v0.11.0 requires afm.transfer.samples >= consolidation.commit_samples "
                    "so transfer and activation use the same frozen evidence block"
                )
            if "weight" in transfer:
                raise ValueError(
                    "afm.transfer.weight is unsupported in v0.11.0: transfer is "
                    "scale invariant and has no transfer-weight parameter"
                )
            if float(transfer.get("max_activation_gap", 0.0)) < 0.0:
                raise ValueError("afm.transfer.max_activation_gap must be nonnegative")
            empirical_transfer_L = float(transfer.get("empirical_smoothness", 0.0))
            if empirical_transfer_L <= 0.0 or not math.isfinite(empirical_transfer_L):
                raise ValueError("afm.transfer.empirical_smoothness must be finite and positive")
            if bool(transfer.get("smoothness_certified", False)):
                certified_transfer_L = transfer.get("smoothness_bound")
                if (
                    certified_transfer_L is None
                    or float(certified_transfer_L) <= 0.0
                    or not math.isfinite(float(certified_transfer_L))
                    or transfer.get("smoothness_provenance") is None
                ):
                    raise ValueError(
                        "Certified transfer smoothness requires a finite positive bound and provenance"
                    )
            if float(transfer.get("endpoint_tolerance", 0.0)) < 0.0:
                raise ValueError("afm.transfer.endpoint_tolerance must be nonnegative")
            numerics = dict(afm.get("numerics", {}))
            if bool(numerics.get("endpoint_error_certified", False)):
                endpoint_error = numerics.get("endpoint_error_bound")
                if (
                    endpoint_error is None
                    or float(endpoint_error) < 0.0
                    or not math.isfinite(float(endpoint_error))
                    or numerics.get("endpoint_error_provenance") is None
                ):
                    raise ValueError(
                        "Certified endpoint evaluation requires a finite nonnegative error bound and provenance"
                    )
            if float(transfer.get("priority_tolerance", transfer.get("common_descent_tolerance", 1e-12))) < 0.0:
                raise ValueError("afm.transfer.priority_tolerance must be nonnegative")
            progress_fraction = float(transfer.get("min_progress_fraction", 0.25))
            if not 0.0 <= progress_fraction < 1.0:
                raise ValueError("afm.transfer.min_progress_fraction must lie in [0,1)")
            if int(transfer.get("joint_solver_max_iterations", 1024)) <= 0:
                raise ValueError("afm.transfer.joint_solver_max_iterations must be positive")
            if float(transfer.get("joint_solver_tolerance", 1e-8)) < 0.0:
                raise ValueError("afm.transfer.joint_solver_tolerance must be nonnegative")
            shield = dict(afm.get("functional_shield", {}))
            if bool(shield.get("enabled", True)):
                if int(shield.get("max_nodes", self.model.functional_shield.max_nodes)) <= 0:
                    raise ValueError("afm.functional_shield.max_nodes must be positive")
                if str(shield.get("kind", "compact_cardinal")) != "compact_cardinal":
                    raise ValueError("functional_shield.kind must be compact_cardinal")
                if int(shield.get("guard_capacity", self.functional_shield_guard_capacity)) < 0:
                    raise ValueError("afm.functional_shield.guard_capacity must be nonnegative")
                support_multiplier = float(shield.get("support_multiplier", 4.0))
                if support_multiplier <= 1.0:
                    raise ValueError("functional_shield.support_multiplier must be greater than one")
                for key, default in (
                    ("feature_match_tolerance", 1e-8),
                    ("duplicate_tolerance", 1e-10),
                    ("target_tolerance", 1e-8),
                    ("residual_tolerance", 1e-8),
                ):
                    if float(shield.get(key, default)) < 0.0:
                        raise ValueError(f"afm.functional_shield.{key} must be nonnegative")
                if not 0.0 < float(shield.get("selected_contraction", 1.0)) <= 1.0:
                    raise ValueError("afm.functional_shield.selected_contraction must lie in (0,1]")
                # Strict mode requires certified feature-address comparisons and
                # endpoint arithmetic.  There is no interpolation linear solve.
            restoration = dict(afm.get("exact_counterfactual_restoration", {}))
            if bool(restoration.get("enabled", True)):
                if not bool(shield.get("enabled", True)):
                    # Explicit shield-disabled configurations are the retained
                    # legacy projected/joint-solver mode, not strong mode.
                    restoration["enabled"] = False
                if str(restoration.get("comparator", "exact_afm_no_protection_endpoint")) != "exact_afm_no_protection_endpoint":
                    raise ValueError("Joint endpoint-emulation comparator must be exact_afm_no_protection_endpoint")
                if str(restoration.get("persistent_base", "counterfactual_normalized_metaplastic_endpoint")) != "counterfactual_normalized_metaplastic_endpoint":
                    raise ValueError("v0.11.0 persistent_base must be counterfactual_normalized_metaplastic_endpoint")
                if not bool(restoration.get("no_nonzero_fallback", True)):
                    raise ValueError("v0.11.0 joint mode forbids any nonzero fallback outside the persistent-assimilation-plus-shield transaction")
            service_limit = transfer.get("max_service_attempts")
            if service_limit is not None and int(service_limit) <= 0:
                raise ValueError("afm.transfer.max_service_attempts must be null or positive")
            if str(transfer.get("scheduler", "fair_round_robin")) != "fair_round_robin":
                raise ValueError("afm.transfer.scheduler must be 'fair_round_robin'")
            if float(transfer.get("staleness_sigma", 0.5)) < 0.5:
                raise ValueError("afm.transfer.staleness_sigma must be at least 0.5 for bounded 0/1 losses")
            if float(transfer.get("staleness_prior_scale", 1.0)) <= 0.0:
                raise ValueError("afm.transfer.staleness_prior_scale must be positive")
            if not bool(transfer.get("freeze_certificate_at_first_crossing", True)):
                raise ValueError("v0.11.0 requires freeze_certificate_at_first_crossing=true")

        if float(safe_update["learning_rate"]) <= 0.0:
            raise ValueError("safe_update.learning_rate must be positive")
        if float(safe_update["trust_radius_cap"]) < 0.0:
            raise ValueError("safe_update.trust_radius_cap must be nonnegative")
        if float(safe_update.get("total_budget", 0.0)) < 0.0 or float(safe_update.get("budget_b0", 0.0)) < 0.0:
            raise ValueError("Behaviour budgets must be nonnegative")
        budget_mode = str(safe_update.get("budget_mode", "counterfactual_normalized"))
        if budget_mode not in {"counterfactual_normalized", "legacy_schedule"}:
            raise ValueError("safe_update.budget_mode must be counterfactual_normalized or legacy_schedule")
        charge_fraction = float(safe_update.get("counterfactual_charge_fraction", 1.0))
        if not 0.0 <= charge_fraction <= 1.0:
            raise ValueError("safe_update.counterfactual_charge_fraction must lie in [0,1]")
        if int(safe_update.get("max_backtracks", 0)) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if not 0.0 < float(safe_update.get("backtrack_factor", 0.5)) < 1.0:
            raise ValueError("backtrack_factor must lie in (0,1)")
        if float(safe_update.get("check_tolerance", 0.0)) < 0.0:
            raise ValueError("check_tolerance must be nonnegative")
        if float(safe_update.get("nonzero_step_tolerance", 0.0)) < 0.0:
            raise ValueError("nonzero_step_tolerance must be nonnegative")

        if not 0.0 <= float(reopening["hysteresis"]) <= 1.0:
            raise ValueError("reopening.hysteresis must lie in [0,1]")
        if float(reopening["challenger_lr"]) <= 0.0:
            raise ValueError("reopening.challenger_lr must be positive")
        if int(renewal["patience"]) <= 0:
            raise ValueError("renewal.patience must be positive")
        for name in (
            "loss_threshold",
            "stall_step_threshold",
            "stall_gradient_threshold",
            "useful_gradient_threshold",
            "zero_gate_tolerance",
        ):
            if float(renewal[name]) < 0.0:
                raise ValueError(f"renewal.{name} must be nonnegative")

    def _new_candidate(self, slot: int) -> CandidateState:
        if self.vectoriser is None:
            raise RuntimeError("AFM parameter state must be initialised before creating a candidate")
        self.candidate_counter += 1
        candidate = CandidateState(
            slot=slot,
            created_step=self.step,
            candidate_id=self.candidate_counter,
            initial_parameters=self.vectoriser.flatten(detach=True).cpu(),
        )
        self.logger.log(
            "candidate_created",
            step=self.step,
            slot=slot,
            candidate_id=candidate.candidate_id,
        )
        return candidate

    def _replay_signature_prefix_after_freeze(self) -> None:
        """Calibrate the task-free signature in one fixed representation.

        Only bounded learner-visible observation references collected during
        the declared unprotected prefix are retained.  They are replayed after
        the final bootstrap update and after the backbone is frozen.
        """

        if not self.context_signature.calibrate_from_prefix or not self.signature_prefix_rows:
            return
        batch_size = max(1, int(self.cfg["training"].get("batch_size", 1)))
        self.model.eval()
        for start in range(0, len(self.signature_prefix_rows), batch_size):
            rows = self.signature_prefix_rows[start : start + batch_size]
            images, _ = load_rows(
                rows,
                image_size=int(self.cfg["data"]["image_size"]),
                normalise=True,
            )
            images = images.to(self.device)
            with torch.no_grad():
                backbone_features = self.model.backbone(images).detach()
                features = (
                    backbone_features
                    if self.context_signature.kind
                    in {"backbone", "block_backbone_mean", "block_hybrid_moments"}
                    else None
                )
                if self.functional_shield_enabled and self.functional_shield_guard_capacity > 0:
                    remaining = self.functional_shield_guard_capacity - int(
                        self.functional_shield_guard_features.shape[0]
                    )
                    if remaining > 0:
                        additions = backbone_features[:remaining].to(dtype=torch.float64, device="cpu")
                        self.functional_shield_guard_features = torch.cat(
                            (self.functional_shield_guard_features, additions), dim=0
                        )
            self.context_signature.observe_prefix(images, features)
        self.logger.log(
            "context_signature_prefix_replayed",
            prefix_items=len(self.signature_prefix_rows),
            representation="final_frozen_backbone",
            functional_shield_guard_items=int(self.functional_shield_guard_features.shape[0]),
        )
        self.signature_prefix_rows.clear()

    def _initialise_afm_state(self) -> None:
        if self.vectoriser is not None:
            return
        self.model.freeze_backbone()
        self._replay_signature_prefix_after_freeze()
        calibration = self.context_signature.finalise()
        self.router.threshold = calibration.calibrated_threshold
        self.logger.log("context_signature_frozen", **calibration.as_dict())
        if calibration.calibration_obstruction:
            self.logger.log(
                "signature_calibration_obstruction",
                requested_threshold=calibration.requested_threshold,
                applied_threshold=calibration.calibrated_threshold,
                observed_coverage=calibration.observed_coverage,
                positive_routing_claim_available=False,
            )
        self.vectoriser = ParameterVector(self.model.trainable_named_parameters())
        afm = self.cfg["afm"]
        ranks = [int(r) for r in afm["metaplastic"]["ranks"]]
        declared_max_rank = int(afm["resources"]["max_rank"])
        if declared_max_rank > self.vectoriser.dimension:
            raise ValueError("afm.resources.max_rank cannot exceed the trainable parameter dimension")
        if any(r < 0 or r > declared_max_rank or r > self.vectoriser.dimension for r in ranks):
            raise ValueError("Every metaplastic rank must lie in [0, min(max_rank, parameter_dimension)]")
        policies = make_policy_family(afm["metaplastic"]["alphas"], ranks, int(afm["metaplastic"]["timescales"]))
        self.policy_family = policies
        max_records = int(afm["resources"]["max_records"])
        declared_m = float(afm["memory"]["declared_sketch_frobenius_sq"])
        zeta = float(afm["metaplastic"]["zeta"])
        loss_bound = 2.0 * max_records * declared_m + zeta
        self.controller = MetaplasticController(
            policies=policies,
            horizon=self.effective_afm_horizon,
            seed=int(self.cfg["seed"]),
            zeta=zeta,
            loss_bound=loss_bound,
        )
        policy_index, policy_alpha = self.error_allocator.next("policy")
        self.logger.log(
            "afm_state_initialised",
            dimension=self.vectoriser.dimension,
            policies=len(policies),
            sketch_rows=int(afm["memory"]["sketch_rows"]),
            frontier_loss_bound=loss_bound,
            policy_budget_index=policy_index,
            policy_alpha=policy_alpha,
            effective_afm_horizon=self.effective_afm_horizon,
            max_contexts=self.router.max_slots,
            max_records=max_records,
            max_atoms=int(afm["resources"].get("max_atoms", max_records)),
        )

    def _set_afm_train_mode(self) -> None:
        self.model.train()
        self.model.backbone.eval()

    def _signatures_with_blocks(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-item signatures and one immutable ID per observable block.

        Outcomes remain item-resolved, but route-refinement evidence is updated
        once per distinct observable block.  This is the filtration declared in
        the corrected theorem and prevents repeated microblock signatures from
        being treated as independent shift observations.
        """

        self.model.eval()
        with torch.no_grad():
            features = (
                self.model.backbone(images)
                if self.context_signature.kind in {"backbone", "block_backbone_mean", "block_hybrid_moments"}
                else None
            )
            return self.context_signature.transform_with_block_ids(images, features)

    def _signatures(self, images: torch.Tensor) -> torch.Tensor:
        signatures, _ = self._signatures_with_blocks(images)
        return signatures

    @staticmethod
    def _index_rows(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
        """Select rows with an index tensor on the selected tensor's device.

        Observable routing signatures intentionally remain on CPU while images
        and labels may live on CUDA.  Constructing one shared CUDA index tensor
        for all three tensors is therefore invalid.  This helper makes the
        device boundary explicit and prevents mixed-device ``index_select``
        failures without moving the bounded routing state onto the accelerator.
        """

        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, index_tensor)

    def _active_records(self) -> list[ProtectedRecord]:
        return sorted((record for record in self.records.values() if record.active), key=lambda r: r.record_id)

    def _records_for_slot(self, slot: int, active_only: bool = True) -> list[ProtectedRecord]:
        records = [record for record in self.records.values() if record.slot == slot]
        if active_only:
            records = [record for record in records if record.active]
        return sorted(records, key=lambda r: r.record_id)

    def _refresh_router_commit_flag(self, slot: int) -> None:
        candidate = self.candidates.get(slot)
        reserved_by_candidate = bool(candidate is not None and candidate.certified)
        self.router.set_committed(
            slot,
            bool(self._records_for_slot(slot, active_only=True)) or reserved_by_candidate,
        )


    def _bounded_evidence_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Canonical fixed-field observation reference for empirical checks."""

        if "path" not in row or "label" not in row:
            raise ValueError("Every empirical evidence row requires path and label")
        canonical = {
            "sample_id": str(row.get("sample_id", "")),
            "path": str(row["path"]),
            "label": int(row["label"]),
            "transform": dict(row.get("transform", {})),
            "transform_seed": int(row.get("transform_seed", 0)),
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        limit = int(self.cfg["afm"]["resources"].get("evidence_reference_bytes", 4096))
        if len(encoded) > limit:
            raise ValueError(
                f"Evidence reference uses {len(encoded)} bytes, exceeding declared bound {limit}"
            )
        return canonical

    def _load_evidence(self, record: ProtectedRecord) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = load_rows(record.evidence, image_size=int(self.cfg["data"]["image_size"]), normalise=True)
        return images.to(self.device), labels.to(self.device)

    def _record_current_outputs(self, records: Iterable[ProtectedRecord]) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        self.model.eval()
        with torch.no_grad():
            for record in records:
                images, _ = self._load_evidence(record)
                outputs[record.record_id] = current_behaviour(self.model, images, self.behaviour_spec).detach()
        return outputs

    @contextmanager
    def _candidate_snapshot_context(self, candidate: CandidateState):
        if self.vectoriser is None or candidate.snapshot is None:
            raise RuntimeError("Frozen candidate snapshot is unavailable")
        shield_state = candidate.snapshot_shield_state
        if shield_state is None:
            # Legacy in-memory fixtures predate the compact-cardinal structural state.
            # Production candidates always freeze this field and validity rejects
            # its absence from a scientific checkpoint.
            shield_state = self.model.functional_shield.snapshot()
        with temporary_parameters(self.vectoriser, candidate.snapshot.to(self.device)):
            with temporary_shield(self.model.functional_shield, shield_state):
                yield

    @contextmanager
    def _record_snapshot_context(self, record: ProtectedRecord):
        if self.vectoriser is None:
            raise RuntimeError("AFM parameter state is unavailable")
        shield_state = record.anchor_shield_state
        if shield_state is None:
            shield_state = self.model.functional_shield.snapshot()
        with temporary_parameters(self.vectoriser, record.anchor.to(self.device)):
            with temporary_shield(self.model.functional_shield, shield_state):
                yield

    @staticmethod
    def _stacked_behaviour_drift(before: dict[int, torch.Tensor], after: dict[int, torch.Tensor]) -> float:
        squared = 0.0
        for record_id, old in before.items():
            squared += float((after[record_id] - old).square().sum(dim=1).mean().item())
        return math.sqrt(max(squared, 0.0))

    def _release_record(self, record: ProtectedRecord, reason: str, clear_router: bool = False) -> None:
        if not record.active:
            return
        record.release(self.step, reason)
        self.records.pop(record.record_id, None)
        self.summary.capacity_releases += int(reason.startswith("capacity"))
        if clear_router:
            # A context slot can be recycled only after every active segment
            # attached to it has been explicitly released.
            remaining = [r for r in self._records_for_slot(record.slot) if r.record_id != record.record_id]
            if remaining:
                raise RuntimeError("Cannot clear a context slot while another active segment still uses it")
            self.router.clear(record.slot)
            self.candidates.pop(record.slot, None)
        else:
            self._refresh_router_commit_flag(record.slot)
        self.logger.log(
            "record_released",
            step=self.step,
            record_id=record.record_id,
            slot=record.slot,
            reason=reason,
            cumulative_budget=record.cumulative_budget,
            max_anchor_drift=record.max_anchor_drift,
            activation_gap=record.activation_gap,
        )

    def _capacity_release_record(self, reason: str) -> ProtectedRecord | None:
        active = self._active_records()
        if not active:
            return None
        record = min(active, key=lambda r: (r.last_seen_step, r.record_id))
        self._release_record(record, reason, clear_router=False)
        return record

    def _capacity_release_context(self, reason: str) -> int | None:
        occupied = [
            slot
            for slot, centroid in enumerate(self.router.centroids)
            if centroid is not None
            and not bool(self.candidates.get(slot) is not None and self.candidates[slot].certified)
        ]
        if not occupied:
            return None
        # The router already recycles uncertified slots.  Reaching this path
        # means every occupied context has protected segments, so the declared
        # context-capacity policy must release all segments of one named slot.
        slot = min(occupied, key=lambda i: (self.router.last_seen[i], i))
        records = self._records_for_slot(slot, active_only=True)
        if not records:
            self.router.clear(slot)
            self.candidates.pop(slot, None)
            return slot
        for record in records:
            self._release_record(record, reason, clear_router=False)
        self.router.clear(slot)
        self.candidates.pop(slot, None)
        self.logger.log("context_released", step=self.step, slot=slot, reason=reason, records=[r.record_id for r in records])
        return slot

    def _reset_record_reopening_test(self, record: ProtectedRecord, reason: str) -> None:
        """Restart independent outcome and observable-shift tests."""

        if self.vectoriser is None:
            raise RuntimeError("AFM parameter state is required")
        reopening_index, reopening_alpha = self.error_allocator.next("reopening")
        signature_index, signature_alpha = self.error_allocator.next("route_split")
        with self._record_snapshot_context(record):
            record.challenger = ShadowChallenger.from_model(
                self.model, learning_rate=float(self.cfg["afm"]["reopening"]["challenger_lr"])
            ).to(self.device)
        record.eprocess = HalfNormalMixtureEProcess(
            sigma=float(self.cfg["afm"]["reopening"]["sigma"]),
            prior_scale=float(self.cfg["afm"]["reopening"]["prior_scale"]),
            alpha=reopening_alpha,
        )
        split_cfg = self.cfg["afm"]["router"].get("split", {})
        record.signature_eprocess = HalfNormalMixtureEProcess(
            sigma=float(split_cfg.get("signature_sigma", 0.5)),
            prior_scale=float(split_cfg.get("signature_prior_scale", 1.0)),
            alpha=signature_alpha,
        )
        record.reopening_budget_index = reopening_index
        record.reopening_alpha = reopening_alpha
        record.reopening_start_step = self.step + 1
        record.signature_budget_index = signature_index
        record.signature_alpha = signature_alpha
        record.signature_start_step = self.step + 1
        record.signature_blocks_seen = 0
        record.signature_distance_sum = 0.0
        record.signature_vector_sum = torch.zeros_like(record.signature_vector_sum)
        record.last_signature_block_id = -1
        record.outcome_reopening_crossed = False
        record.outcome_crossing_step = None
        record.outcome_crossing_signature_count = None
        record.outcome_crossing_log_wealth = None
        record.outcome_crossing_observation = None
        self.logger.log(
            "reopening_test_restarted",
            step=self.step,
            record_id=record.record_id,
            slot=record.slot,
            reason=reason,
            reopening_budget_index=reopening_index,
            reopening_alpha=reopening_alpha,
            signature_budget_index=signature_index,
            signature_alpha=signature_alpha,
        )

    def _allocate_route_split(self, source_slot: int, signature: torch.Tensor) -> int | None:
        """Create one bounded evidence-triggered route without deleting the source."""

        target = self.router.available_slot(exclude={source_slot})
        if target is None:
            if self.cfg["afm"]["resources"].get("capacity_policy", "release_oldest") == "overflow":
                self.route_split_obstructions += 1
                return None
            occupied = [
                slot
                for slot, centroid in enumerate(self.router.centroids)
                if centroid is not None
                and slot != source_slot
                and not bool(self.candidates.get(slot) is not None and self.candidates[slot].certified)
            ]
            if not occupied:
                self.route_split_obstructions += 1
                return None
            victim = min(occupied, key=lambda slot: (self.router.last_seen[slot], slot))
            victim_records = self._records_for_slot(victim, active_only=True)
            for record in victim_records:
                self._release_record(record, "capacity_route_split", clear_router=False)
            self.router.clear(victim)
            self.candidates.pop(victim, None)
            self.logger.log(
                "context_released",
                step=self.step,
                slot=victim,
                reason="capacity_route_split",
                records=[record.record_id for record in victim_records],
            )
            target = victim
        source_candidate = self.candidates.get(source_slot)
        if source_candidate is not None and not source_candidate.certified:
            source_candidate = self.candidates.pop(source_slot)
            self.candidate_replacements += 1
            if source_candidate.commit_budget_index is None:
                self.candidate_pretest_replacements += 1
            self.logger.log(
                "candidate_reset_after_route_split",
                step=self.step,
                slot=source_slot,
                candidate_id=source_candidate.candidate_id,
                validation_count=source_candidate.validation_count,
                commit_budget_index=source_candidate.commit_budget_index,
            )
        elif source_candidate is not None and source_candidate.certified:
            self.logger.log(
                "certified_candidate_preserved_during_route_split",
                step=self.step,
                slot=source_slot,
                candidate_id=source_candidate.candidate_id,
            )
        if source_slot not in self.candidates and self._records_for_slot(source_slot, active_only=True):
            self.candidates[source_slot] = self._new_candidate(source_slot)

        old_candidate = self.candidates.get(target)
        if old_candidate is not None and old_candidate.certified:
            self.route_split_obstructions += 1
            self.logger.log(
                "route_split_obstructed_certified_candidate_reservation",
                step=self.step,
                source_slot=source_slot,
                target_slot=target,
                candidate_id=old_candidate.candidate_id,
            )
            return None
        old_candidate = self.candidates.pop(target, None)
        if old_candidate is not None:
            self.candidate_replacements += 1
            if old_candidate.commit_budget_index is None:
                self.candidate_pretest_replacements += 1
        decision = self.router.force_split(signature, self.step, source_slot=source_slot, target_slot=target)
        self.candidates[target] = self._new_candidate(target)
        self._refresh_router_commit_flag(target)
        self.route_splits += 1
        self.summary.route_splits += 1
        self._last_split_step_by_slot[source_slot] = self.step
        self.logger.log(
            "route_split",
            step=self.step,
            source_slot=source_slot,
            target_slot=target,
            distance=decision.distance,
            source_records=[record.record_id for record in self._records_for_slot(source_slot, active_only=True)],
        )
        return target

    def _route_signature(self, signature: torch.Tensor) -> tuple[int | None, bool, float, bool]:
        """Route one observable signature and return auditable final-decision metadata.

        The distance and creation flag are output-only diagnostics.  They are
        never read by the learner and therefore do not alter the task-free
        filtration or the bounded decision state.
        """

        decision = self.router.route(signature, self.step)
        if decision.overflow:
            if self.cfg["afm"]["resources"].get("capacity_policy", "release_oldest") == "overflow":
                self.logger.log("unprotected_overflow", step=self.step)
                return None, True, float(decision.distance), bool(decision.created)
            released_slot = self._capacity_release_context("capacity_new_context")
            if released_slot is None:
                return None, True, float(decision.distance), bool(decision.created)
            decision = self.router.route(signature, self.step)
        if decision.slot is None:
            return None, True, float(decision.distance), bool(decision.created)
        slot = decision.slot
        if decision.created:
            old_candidate = self.candidates.pop(slot, None)
            if old_candidate is not None:
                self.candidate_replacements += 1
                if old_candidate.commit_budget_index is None:
                    self.candidate_pretest_replacements += 1
                old_ucb = None
                if old_candidate.validation_count > 0 and old_candidate.commit_alpha is not None:
                    old_ucb = consolidation_ucb(
                        old_candidate.validation_count,
                        old_candidate.validation_sum,
                        old_candidate.commit_alpha,
                    )
                self.logger.log(
                    "candidate_replaced",
                    step=self.step,
                    slot=slot,
                    candidate_id=old_candidate.candidate_id,
                    old_commit_budget_index=old_candidate.commit_budget_index,
                    old_commit_alpha=old_candidate.commit_alpha,
                    old_test_started=old_candidate.commit_budget_index is not None,
                    old_validation_count=old_candidate.validation_count,
                    old_validation_mean=old_candidate.validation_mean,
                    old_ucb=old_ucb,
                )
            self.candidates[slot] = self._new_candidate(slot)
            self._refresh_router_commit_flag(slot)
        elif slot not in self.candidates and not self._records_for_slot(slot, active_only=True):
            self.candidates[slot] = self._new_candidate(slot)
            self._refresh_router_commit_flag(slot)
        for record in self._records_for_slot(slot, active_only=True):
            record.last_seen_step = self.step
        return slot, False, float(decision.distance), bool(decision.created)

    def _route_batch(
        self,
        images: torch.Tensor,
        learner_rows: list[dict[str, Any]] | None = None,
        include_signatures: bool = False,
    ):
        """Sequentially route each item and bind evidence to immutable identities.

        Candidate evidence is keyed by the candidate token and reopening evidence
        by immutable record_id.  A bounded router slot may be recycled later in
        the same minibatch; grouping only by slot would then mix contexts.
        """

        candidate_groups: dict[tuple[int, tuple[int, int, int]], list[int]] = {}
        record_groups: dict[int, list[int]] = {}
        overflow_count = 0
        signatures, signature_block_ids = self._signatures_with_blocks(images)
        for index, signature in enumerate(signatures):
            route_result = self._route_signature(signature)
            if len(route_result) == 2:  # Backward-compatible test/external override.
                slot, overflow = route_result
                distance, created = float("nan"), False
            else:
                slot, overflow, distance, created = route_result
            overflow_count += int(overflow)
            records = [] if slot is None else self._records_for_slot(slot, active_only=True)
            candidate = None if slot is None else self.candidates.get(slot)
            row = (
                learner_rows[index]
                if learner_rows is not None and index < len(learner_rows)
                else {}
            )
            self.logger.log(
                "routing_assignment",
                step=self.step,
                item_index=index,
                sample_id=row.get("sample_id"),
                slot=slot,
                overflow=overflow,
                distance=distance,
                centroid_created=created,
                active_record_ids=[record.record_id for record in records],
                candidate_id=None if candidate is None else candidate.candidate_id,
                signature_kind=self.context_signature.kind,
            )
            if slot is None:
                continue
            for record in records:
                record_groups.setdefault(record.record_id, []).append(index)
            if candidate is not None:
                candidate_groups.setdefault((slot, candidate.token), []).append(index)
        if include_signatures:
            return candidate_groups, record_groups, overflow_count, signatures, signature_block_ids
        return candidate_groups, record_groups, overflow_count

    def _candidate_training_samples(self) -> int:
        cons = self.cfg["afm"]["consolidation"]
        legacy_batches = int(cons.get("freeze_after_batches", 0))
        return int(
            cons.get(
                "training_samples",
                legacy_batches * int(self.cfg["training"].get("batch_size", 1)),
            )
        )

    def _fit_and_freeze_candidate(self, slot: int) -> None:
        """Fit the declared private candidate on its finite routed training block.

        The deployed model is restored exactly after fitting.  The resulting
        private vector is frozen before any validation outcome is observed.
        Candidate fitting is deterministic conditional on the routed history:
        fixed order, fixed epochs, and a fully declared vector Adam update.
        """

        if self.vectoriser is None or slot not in self.candidates:
            return
        candidate = self.candidates[slot]
        if candidate.snapshot is not None:
            return
        if candidate.initial_parameters is None:
            raise RuntimeError("Candidate is missing its predictable initial parameter vector")
        cons = self.cfg["afm"]["consolidation"]
        required = self._candidate_training_samples()
        if len(candidate.training_evidence) < required:
            return

        train_images, train_labels = load_rows(
            candidate.training_evidence,
            image_size=int(self.cfg["data"]["image_size"]),
            normalise=True,
        )
        train_images = train_images.to(self.device)
        train_labels = train_labels.to(self.device)
        epochs = int(cons.get("training_epochs", 1))
        batch_size = int(cons.get("training_batch_size", required))
        lr = float(cons.get("training_lr", self.cfg["training"].get("bootstrap_lr", 1e-3)))
        weight_decay = float(cons.get("training_weight_decay", 0.0))
        beta1 = float(cons.get("adam_beta1", 0.9))
        beta2 = float(cons.get("adam_beta2", 0.999))
        epsilon = float(cons.get("adam_epsilon", 1e-8))
        initial = candidate.initial_parameters.to(self.device)
        active_slots = set(self.model.adapter_pool.state().active)
        allowed_mask = self.vectoriser.gradient_mask_for_adapter_activity(active_slots)
        first_moment = torch.zeros_like(initial)
        second_moment = torch.zeros_like(initial)
        optimiser_step = 0

        with temporary_parameters(self.vectoriser, initial):
            for _ in range(epochs):
                for start in range(0, len(train_labels), batch_size):
                    stop = min(start + batch_size, len(train_labels))
                    self._set_afm_train_mode()
                    self.model.zero_grad(set_to_none=True)
                    logits = self.model(train_images[start:stop])
                    loss = torch.nn.functional.cross_entropy(logits, train_labels[start:stop])
                    loss.backward()
                    gradient = self.vectoriser.flatten_grads() * allowed_mask
                    if weight_decay > 0.0:
                        current = self.vectoriser.flatten(detach=True)
                        gradient = gradient + weight_decay * current * allowed_mask
                    optimiser_step += 1
                    first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                    first_hat = first_moment / (1.0 - beta1**optimiser_step)
                    second_hat = second_moment / (1.0 - beta2**optimiser_step)
                    delta = -lr * first_hat / (torch.sqrt(second_hat) + epsilon)
                    self.vectoriser.add_(delta * allowed_mask)

            candidate.snapshot = self.vectoriser.flatten(detach=True).cpu()
            candidate.snapshot_shield_state = self.model.functional_shield.snapshot()
            self.model.eval()
            with torch.no_grad():
                fitted_logits = self.model(train_images)
                fitted_behaviour = current_behaviour(self.model, train_images, self.behaviour_spec).detach().cpu()
                candidate.training_loss = float(
                    torch.nn.functional.cross_entropy(fitted_logits, train_labels).item()
                )
                candidate.training_accuracy = accuracy(fitted_logits, train_labels)

        transfer_samples = min(
            int(self.cfg["afm"].get("transfer", {}).get("samples", cons["commit_samples"])),
            len(candidate.training_evidence),
        )
        candidate.transfer_evidence = [dict(row) for row in candidate.training_evidence[:transfer_samples]]
        candidate.transfer_targets = fitted_behaviour[:transfer_samples].clone()
        candidate.transfer_logits = fitted_logits[:transfer_samples].detach().cpu().clone()
        candidate.frozen_step = int(self.step)

        budget_index, alpha = self.error_allocator.next("consolidation")
        candidate.commit_budget_index = budget_index
        candidate.commit_alpha = alpha
        candidate.sketch = FrequentDirections(
            ell=int(self.cfg["afm"]["memory"]["sketch_rows"]),
            dimension=self.vectoriser.dimension,
            device=self.device,
            dtype=torch.float32,
        )
        candidate.initial_parameters = None
        # The finite fit block is no longer decision state once the private
        # snapshot is frozen.  Clearing it enforces the declared bound.
        candidate.training_evidence.clear()
        self.candidate_tests_started += 1
        self.candidate_training_runs += 1
        self.logger.log(
            "candidate_frozen",
            step=self.step,
            slot=slot,
            candidate_id=candidate.candidate_id,
            training_count=candidate.training_count,
            training_epochs=epochs,
            training_batch_size=batch_size,
            training_lr=lr,
            training_loss=candidate.training_loss,
            training_accuracy=candidate.training_accuracy,
            transfer_evidence_size=len(candidate.transfer_evidence),
            frozen_step=candidate.frozen_step,
            commit_budget_index=budget_index,
            commit_alpha=alpha,
        )

    def _reject_candidate(self, slot: int, reason: str, ucb: float | None = None) -> None:
        candidate = self.candidates.get(slot)
        if candidate is None:
            return
        self.candidate_validation_rejections += 1
        self.logger.log(
            "candidate_rejected",
            step=self.step,
            slot=slot,
            candidate_id=candidate.candidate_id,
            reason=reason,
            validation_count=candidate.validation_count,
            validation_mean=candidate.validation_mean,
            ucb=ucb,
            risk_threshold=float(self.cfg["afm"]["consolidation"]["risk_threshold"]),
            commit_budget_index=candidate.commit_budget_index,
            commit_alpha=candidate.commit_alpha,
        )
        del self.candidates[slot]
        if not self._records_for_slot(slot, active_only=True) or bool(
            self.cfg["afm"]["consolidation"].get("start_successor_candidate", True)
        ):
            self.candidates[slot] = self._new_candidate(slot)
        self._refresh_router_commit_flag(slot)

    def _candidate_transfer_batch(self, candidate: CandidateState) -> CandidateTransferBatch | None:
        """Return the immutable first-crossing block used for safe transfer."""

        if (
            not candidate.certified
            or candidate.snapshot is None
            or candidate.frozen_step is None
            or candidate.certified_step is None
            or self.step < candidate.certified_step
        ):
            return None
        if candidate.evidence and candidate.anchor_output_chunks:
            rows = candidate.evidence
            targets = torch.cat(candidate.anchor_output_chunks, dim=0)
            if candidate.anchor_logit_chunks:
                target_logits = torch.cat(candidate.anchor_logit_chunks, dim=0)
            else:
                images_for_logits, _ = load_rows(
                    rows, image_size=int(self.cfg["data"]["image_size"]), normalise=True
                )
                with self._candidate_snapshot_context(candidate):
                    self.model.eval()
                    with torch.no_grad():
                        target_logits = self.model(images_for_logits.to(self.device)).detach().cpu()
            source = "first_crossing_certification_block"
        else:
            return None
        limit = min(int(self.cfg["afm"].get("transfer", {}).get("samples", len(rows))), len(rows), len(targets))
        if limit <= 0:
            return None
        images, _ = load_rows(rows[:limit], image_size=int(self.cfg["data"]["image_size"]), normalise=True)
        return CandidateTransferBatch(
            candidate_id=candidate.candidate_id,
            slot=candidate.slot,
            images=images.to(self.device),
            targets=targets[:limit].to(self.device),
            target_logits=target_logits[:limit].to(self.device),
            source=source,
        )

    def _candidate_transfer_objective(self, candidate: CandidateState) -> float | None:
        batch = self._candidate_transfer_batch(candidate)
        if batch is None:
            return None
        self.model.eval()
        with torch.no_grad():
            outputs = current_behaviour(self.model, batch.images, self.behaviour_spec)
            value = 0.5 * (outputs - batch.targets).square().sum(dim=1).mean()
        return float(value.item())

    def _candidate_activation_gap(self, candidate: CandidateState) -> float:
        """Exact empirical gap on the immutable candidate commit block."""

        if not candidate.evidence or not candidate.anchor_output_chunks:
            return float("inf")
        images, _ = load_rows(
            candidate.evidence,
            image_size=int(self.cfg["data"]["image_size"]),
            normalise=True,
        )
        images = images.to(self.device)
        targets = torch.cat(candidate.anchor_output_chunks, dim=0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            deployed = current_behaviour(self.model, images, self.behaviour_spec)
        gap = math.sqrt(
            max(float((deployed - targets).square().sum(dim=1).mean().item()), 0.0)
        )
        candidate.last_activation_gap = gap
        return gap

    def _eligible_transfer_batches(self) -> list[CandidateTransferBatch]:
        if not bool(self.cfg["afm"].get("transfer", {}).get("enabled", True)):
            return []
        batches: list[CandidateTransferBatch] = []
        service_limit = self.cfg["afm"].get("transfer", {}).get("max_service_attempts")
        maximum = None if service_limit is None else int(service_limit)
        for candidate in self.candidates.values():
            if not candidate.certified:
                continue
            if maximum is not None and candidate.transfer_service_rounds >= maximum:
                continue
            batch = self._candidate_transfer_batch(candidate)
            if batch is None:
                raise RuntimeError(
                    "Certified candidate is missing its immutable transfer block; "
                    "refusing an update that could damage an unrepresented objective"
                )
            batches.append(batch)
        # The first batch is the predictably selected least-recently-served
        # candidate.  Every remaining certified batch is returned as a frozen
        # safeguard objective, so no deployed update may erase its accumulated
        # transfer progress.  All ties are deterministic.
        if not batches:
            return []
        batches.sort(
            key=lambda item: (
                int(
                    self.candidates[item.slot].transfer_last_service_step
                    if self.candidates[item.slot].transfer_last_service_step is not None
                    else (
                        self.candidates[item.slot].certified_step
                        if self.candidates[item.slot].certified_step is not None
                        else self.candidates[item.slot].created_step
                    )
                ),
                int(self.candidates[item.slot].certified_step or self.candidates[item.slot].created_step),
                item.candidate_id,
                item.slot,
            )
        )
        return batches

    def _freeze_candidate_certificate(self, candidate: CandidateState, ucb: float) -> None:
        if candidate.certified:
            return
        transfer_cfg = self.cfg["afm"].get("transfer", {})
        budget_index, alpha = self.error_allocator.next("candidate_staleness")
        candidate.certified = True
        candidate.certified_step = int(self.step)
        candidate.certified_validation_count = int(candidate.validation_count)
        candidate.certified_validation_sum = float(candidate.validation_sum)
        candidate.certified_ucb = float(ucb)
        candidate.staleness_budget_index = budget_index
        candidate.staleness_alpha = alpha
        candidate.staleness_eprocess = HalfNormalMixtureEProcess(
            sigma=float(transfer_cfg.get("staleness_sigma", 0.5)),
            prior_scale=float(transfer_cfg.get("staleness_prior_scale", 1.0)),
            alpha=alpha,
        )
        candidate.transfer_initial_objective = self._candidate_transfer_objective(candidate)
        candidate.transfer_last_objective = candidate.transfer_initial_objective
        # Virtual service time at certification makes least-recently-served
        # scheduling non-starving even when later candidates arrive.
        candidate.transfer_last_service_step = int(self.step)
        self._refresh_router_commit_flag(candidate.slot)
        self.candidate_certifications += 1
        self.logger.log(
            "candidate_certified",
            step=self.step,
            slot=candidate.slot,
            candidate_id=candidate.candidate_id,
            validation_count=candidate.validation_count,
            validation_mean=candidate.validation_mean,
            ucb=ucb,
            risk_threshold=float(self.cfg["afm"]["consolidation"]["risk_threshold"]),
            certification_rule="first_anytime_valid_crossing",
            evidence_size=len(candidate.evidence),
            transfer_initial_objective=candidate.transfer_initial_objective,
            staleness_budget_index=budget_index,
            staleness_alpha=alpha,
        )

    def _update_candidate_staleness(
        self,
        candidate: CandidateState,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> bool:
        """Update the separately allocated post-certification relevance test."""

        if not candidate.certified or candidate.snapshot is None or candidate.staleness_eprocess is None:
            return False
        with self._candidate_snapshot_context(candidate):
            self.model.eval()
            with torch.no_grad():
                logits = self.model(images)
        errors = (logits.argmax(dim=1) != labels).to(torch.float64).cpu().tolist()
        threshold = float(self.cfg["afm"]["consolidation"]["risk_threshold"])
        wealth = candidate.staleness_eprocess.state.wealth
        for error in errors:
            wealth = candidate.staleness_eprocess.update(float(error) - threshold)
            candidate.staleness_observations += 1
            if candidate.staleness_eprocess.crossed:
                candidate.staleness_crossed_step = int(self.step)
                self.candidate_staleness_rejections += 1
                self.logger.log(
                    "candidate_staleness_crossed",
                    step=self.step,
                    slot=candidate.slot,
                    candidate_id=candidate.candidate_id,
                    observations=candidate.staleness_observations,
                    wealth=wealth,
                    log_wealth=candidate.staleness_eprocess.state.log_wealth,
                    threshold=1.0 / max(float(candidate.staleness_alpha or 1.0), 1e-300),
                    staleness_budget_index=candidate.staleness_budget_index,
                    staleness_alpha=candidate.staleness_alpha,
                )
                return True
        return False

    def _update_transfer_ledgers(
        self,
        before: dict[int, float],
        *,
        enforce_nonincrease: bool = False,
    ) -> None:
        """Atomically validate and then update every frozen-candidate ledger.

        Validation is a first pass: if any candidate increased, no ledger is
        mutated.  The caller can therefore roll back the model vector without
        leaving partially advanced accounting state.
        """

        endpoint_tolerance = float(
            self.cfg["afm"].get("transfer", {}).get(
                "endpoint_tolerance", self.cfg["afm"]["safe_update"].get("check_tolerance", 1e-6)
            )
        )
        rows: list[tuple[CandidateState, float, float, float]] = []
        for candidate in list(self.candidates.values()):
            if not candidate.certified or candidate.candidate_id not in before:
                continue
            after = self._candidate_transfer_objective(candidate)
            if after is None:
                raise RuntimeError(
                    "Certified candidate is missing its immutable transfer objective during ledger verification"
                )
            previous = float(before[candidate.candidate_id])
            delta = previous - float(after)
            rows.append((candidate, previous, float(after), delta))

        strict_margin = (
            2.0 * float(self.cfg["afm"].get("numerics", {}).get("endpoint_error_bound") or 0.0)
            if self.certificate_mode == "strict" else 0.0
        )
        allowed_negative = endpoint_tolerance + strict_margin
        violations = [
            row for row in rows
            if enforce_nonincrease and row[3] < -allowed_negative
        ]
        if violations:
            candidate, previous, after, delta = max(violations, key=lambda row: -row[3])
            self.logger.log(
                "candidate_progress_invariant_violation",
                step=self.step,
                slot=candidate.slot,
                candidate_id=candidate.candidate_id,
                objective_before=previous,
                objective_after=after,
                violation=-delta,
                tolerance=endpoint_tolerance,
                strict_numerical_margin=strict_margin,
            )
            raise RuntimeError(
                "A deployed update increased a frozen certified candidate objective "
                f"by {-delta:.6g}; v0.11.0 forbids any increment beyond the declared numerical envelope"
            )

        for candidate, previous, after, delta in rows:
            candidate.transfer_progress_total += max(delta, 0.0)
            certified_damage = (
                max(-delta - allowed_negative, 0.0)
                if enforce_nonincrease else max(-delta, 0.0)
            )
            candidate.transfer_damage_total += certified_damage
            candidate.transfer_ledger_updates += 1
            candidate.transfer_last_objective = after
            self.logger.log(
                "candidate_transfer_ledger",
                step=self.step,
                slot=candidate.slot,
                candidate_id=candidate.candidate_id,
                objective_before=previous,
                objective_after=after,
                realised_progress=delta,
                certified_nonincrease=delta >= -allowed_negative,
                numerical_envelope=allowed_negative,
                cumulative_progress=candidate.transfer_progress_total,
                cumulative_damage=candidate.transfer_damage_total,
                telescoping_value=(
                    None
                    if candidate.transfer_initial_objective is None
                    else candidate.transfer_initial_objective
                    - candidate.transfer_progress_total
                    + candidate.transfer_damage_total
                ),
            )

    def _commit_ready_certified_candidates(self) -> None:
        maximum_gap = float(self.cfg["afm"].get("transfer", {}).get("max_activation_gap", float("inf")))
        for slot, candidate in sorted(
            list(self.candidates.items()),
            key=lambda item: (int(item[1].certified_step or 10**18), item[1].candidate_id),
        ):
            if not candidate.certified:
                continue
            gap = self._candidate_activation_gap(candidate)
            if gap <= maximum_gap:
                self._commit_candidate(slot, ucb=float(candidate.certified_ucb or float("inf")))

    def _reject_exhausted_certified_candidates(self) -> None:
        service_limit = self.cfg["afm"].get("transfer", {}).get("max_service_attempts")
        if service_limit is None:
            return
        maximum = int(service_limit)
        for slot, candidate in list(self.candidates.items()):
            if not candidate.certified or candidate.transfer_service_rounds < maximum:
                continue
            self._reject_candidate(
                slot,
                "predeclared_transfer_service_limit_reached",
                ucb=candidate.certified_ucb,
            )

    def _candidate_update_before_training(
        self,
        slot: int,
        images: torch.Tensor,
        labels: torch.Tensor,
        learner_rows: list[dict[str, Any]],
        signatures: torch.Tensor | None = None,
        signature_block_ids: torch.Tensor | None = None,
    ) -> None:
        if self.vectoriser is None or slot not in self.candidates:
            return
        candidate = self.candidates[slot]
        cons = self.cfg["afm"]["consolidation"]

        # Phase I: collect one finite routed fitting block.  Outcomes in the
        # batch that completes the block are not reused for validation because
        # their labels were observed before the snapshot was frozen.
        if candidate.snapshot is None:
            required = self._candidate_training_samples()
            take = min(required - len(candidate.training_evidence), len(learner_rows))
            if take > 0:
                candidate.training_evidence.extend(
                    self._bounded_evidence_row(row) for row in learner_rows[:take]
                )
                candidate.training_count += take
            if len(candidate.training_evidence) >= required:
                self._fit_and_freeze_candidate(slot)
            return

        if candidate.sketch is None or candidate.commit_alpha is None:
            raise RuntimeError("Frozen candidate is missing its validation state")

        # After the first anytime-valid crossing the original certificate and
        # transfer block are immutable. Future routed outcomes are consumed by
        # a separately allocated relevance/staleness e-process only.
        if candidate.certified:
            if self._update_candidate_staleness(candidate, images, labels):
                self._reject_candidate(slot, "post_certification_staleness")
                return
            if candidate.staleness_observations and candidate.staleness_observations % 256 == 0:
                assert candidate.staleness_eprocess is not None
                self.logger.log(
                    "candidate_staleness",
                    step=self.step,
                    slot=slot,
                    candidate_id=candidate.candidate_id,
                    observations=candidate.staleness_observations,
                    wealth=candidate.staleness_eprocess.state.wealth,
                    log_wealth=candidate.staleness_eprocess.state.log_wealth,
                    crossed=False,
                )
            return

        max_validation = int(cons.get("max_validation_samples", cons["min_validation_samples"]))
        remaining = max_validation - candidate.validation_count
        if remaining <= 0:
            ucb = consolidation_ucb(
                candidate.validation_count, candidate.validation_sum, candidate.commit_alpha
            )
            self._reject_candidate(slot, "validation_horizon_exhausted", ucb=ucb)
            return
        take = min(remaining, len(learner_rows))
        if take <= 0:
            return
        validation_images = images[:take]
        validation_labels = labels[:take]
        validation_rows = learner_rows[:take]

        with self._candidate_snapshot_context(candidate):
            self.model.eval()
            with torch.no_grad():
                snapshot_logits = self.model(validation_images)
        errors = (snapshot_logits.argmax(dim=1) != validation_labels).float().cpu().tolist()
        candidate.add_validation([float(x) for x in errors])

        if signatures is not None and signature_block_ids is not None:
            for local_index in range(min(take, len(signature_block_ids))):
                block_id = int(signature_block_ids[local_index].item())
                if block_id <= candidate.last_validation_signature_block_id:
                    continue
                candidate.validation_signatures.append(signatures[local_index].detach().cpu().to(torch.float64))
                candidate.last_validation_signature_block_id = block_id

        limit = int(cons["commit_samples"])
        if len(candidate.evidence) < limit:
            evidence_take = min(limit - len(candidate.evidence), take)
            if evidence_take > 0:
                selected_rows = [
                    self._bounded_evidence_row(row) for row in validation_rows[:evidence_take]
                ]
                selected_images = validation_images[:evidence_take]
                with self._candidate_snapshot_context(candidate):
                    outputs = stream_behaviour_jacobians(
                        self.model,
                        self.vectoriser,
                        selected_images,
                        candidate.sketch,
                        self.behaviour_spec,
                        anchor=None,
                    )
                    self.model.eval()
                    with torch.no_grad():
                        selected_logits = self.model(selected_images).detach()
                candidate.evidence.extend(selected_rows)
                candidate.anchor_output_chunks.append(outputs.detach().cpu())
                candidate.anchor_logit_chunks.append(selected_logits.cpu())

        ucb = consolidation_ucb(candidate.validation_count, candidate.validation_sum, candidate.commit_alpha)
        min_validation = int(cons["min_validation_samples"])
        if candidate.validation_count >= min_validation and (
            candidate.last_logged_validation_count < min_validation
            or candidate.validation_count >= max(2 * candidate.last_logged_validation_count, min_validation)
        ):
            candidate.last_logged_validation_count = candidate.validation_count
            self.logger.log(
                "candidate_validation",
                step=self.step,
                slot=slot,
                candidate_id=candidate.candidate_id,
                validation_count=candidate.validation_count,
                validation_mean=candidate.validation_mean,
                ucb=ucb,
                risk_threshold=float(cons["risk_threshold"]),
                commit_alpha=candidate.commit_alpha,
                commit_budget_index=candidate.commit_budget_index,
            )
        statistically_ready = (
            candidate.validation_count >= min_validation
            and len(candidate.evidence) >= max(1, int(cons["commit_samples"]))
            and ucb <= float(cons["risk_threshold"])
        )
        if statistically_ready:
            self._freeze_candidate_certificate(candidate, ucb)
            activation_gap = self._candidate_activation_gap(candidate)
            maximum_gap = float(self.cfg["afm"].get("transfer", {}).get("max_activation_gap", float("inf")))
            if activation_gap <= maximum_gap:
                self._commit_candidate(slot, ucb=ucb)
                return
            self.logger.log(
                "candidate_transfer_pending",
                step=self.step,
                slot=slot,
                candidate_id=candidate.candidate_id,
                activation_gap=activation_gap,
                max_activation_gap=maximum_gap,
                validation_count=candidate.validation_count,
                validation_mean=candidate.validation_mean,
                ucb=ucb,
                transfer_attempts=candidate.transfer_attempts,
                transfer_accepted_steps=candidate.transfer_accepted_steps,
                transfer_obstructions=candidate.transfer_obstructions,
                certified_step=candidate.certified_step,
                certificate_frozen=True,
            )
            return

        # The candidate test has a fixed finite horizon.  If even zero loss on
        # every remaining validation outcome could not meet the declared UCB,
        # rejection is exact and may occur early; otherwise reject at the cap.
        best_case_ucb = consolidation_ucb(
            max_validation,
            candidate.validation_sum,
            candidate.commit_alpha,
        )
        if best_case_ucb > float(cons["risk_threshold"]):
            self._reject_candidate(slot, "certification_impossible_within_horizon", ucb=ucb)
        elif candidate.validation_count >= max_validation:
            self._reject_candidate(slot, "validation_horizon_exhausted", ucb=ucb)

    def _candidate_signature_reference_radius(self, slot: int, candidate: CandidateState) -> float:
        """Predictable observable-stability reference for a committed record.

        The value is fixed at commit from distinct validation-block signatures
        and the final router centroid.  It is not an evaluator label and it is
        never updated after commitment.  The separate signature e-process tests
        the future conditional mean distance against this reference.
        """

        split_cfg = self.cfg["afm"]["router"].get("split", {})
        floor = float(split_cfg.get("reference_floor", 0.05))
        margin = float(split_cfg.get("reference_margin", 0.12))
        centroid = self.router.centroids[slot]
        if centroid is None or not candidate.validation_signatures:
            return min(2.0, max(floor, float(self.router.threshold)))
        centre = centroid.detach().cpu().to(torch.float64)
        distances = [
            float(torch.linalg.vector_norm(signature.to(torch.float64) - centre).item())
            for signature in candidate.validation_signatures
        ]
        empirical_mean = float(sum(distances) / max(len(distances), 1))
        return min(2.0, max(floor, empirical_mean + margin))

    def _build_candidate_record(self, slot: int) -> ProtectedRecord | None:
        assert self.vectoriser is not None
        candidate = self.candidates[slot]
        if (
            not candidate.certified
            or candidate.snapshot is None
            or candidate.snapshot_shield_state is None
            or candidate.sketch is None
            or not candidate.anchor_output_chunks
            or not candidate.anchor_logit_chunks
        ):
            return None
        afm = self.cfg["afm"]
        n = len(candidate.evidence)
        snap = candidate.sketch.snapshot(scale=n ** -0.5, delta_scale=1.0 / n)
        frob_sq = float(snap.matrix.square().sum().item())
        declared = float(afm["memory"]["declared_sketch_frobenius_sq"])
        numerical_tol = float(afm.get("numerics", {}).get("bound_tolerance", 1e-6))
        if frob_sq > declared + numerical_tol:
            self.logger.log(
                "commit_rejected_sketch_bound",
                step=self.step,
                slot=slot,
                frobenius_sq=frob_sq,
                declared_bound=declared,
            )
            return None
        evidence_images, _ = load_rows(candidate.evidence, image_size=int(self.cfg["data"]["image_size"]), normalise=True)
        evidence_images = evidence_images.to(self.device)
        self.model.eval()
        with torch.no_grad():
            activation_outputs = current_behaviour(self.model, evidence_images, self.behaviour_spec).detach().cpu()
        reopening_index, reopening_alpha = self.error_allocator.next("reopening")
        signature_index, signature_alpha = self.error_allocator.next("route_split")
        signature_reference_radius = self._candidate_signature_reference_radius(slot, candidate)
        # Initialise the challenger exactly at the frozen committed snapshot.
        # It is therefore predictable and tied with the record initially; any
        # later advantage is learned from post-commit routed outcomes.
        with self._candidate_snapshot_context(candidate):
            challenger = ShadowChallenger.from_model(
                self.model, learning_rate=float(afm["reopening"]["challenger_lr"])
            ).to(self.device)
        return ProtectedRecord(
            record_id=self.record_counter + 1,
            slot=slot,
            created_step=candidate.created_step,
            committed_step=self.step,
            anchor=candidate.snapshot.detach().cpu(),
            anchor_shield_state=copy.deepcopy(candidate.snapshot_shield_state),
            sketch=snap.matrix.detach().cpu(),
            fd_delta=float(snap.delta),
            evidence=[dict(row) for row in candidate.evidence],
            anchor_outputs=torch.cat(candidate.anchor_output_chunks, dim=0).detach().cpu(),
            activation_outputs=activation_outputs,
            traces=torch.zeros(int(afm["metaplastic"]["timescales"]), dtype=torch.float64),
            challenger=challenger,
            eprocess=HalfNormalMixtureEProcess(
                sigma=float(afm["reopening"]["sigma"]),
                prior_scale=float(afm["reopening"]["prior_scale"]),
                alpha=reopening_alpha,
            ),
            reopening_budget_index=reopening_index,
            reopening_alpha=reopening_alpha,
            reopening_start_step=self.step + 1,
            signature_eprocess=HalfNormalMixtureEProcess(
                sigma=float(afm["router"].get("split", {}).get("signature_sigma", 0.5)),
                prior_scale=float(afm["router"].get("split", {}).get("signature_prior_scale", 1.0)),
                alpha=signature_alpha,
            ),
            signature_budget_index=signature_index,
            signature_alpha=signature_alpha,
            signature_start_step=self.step + 1,
            signature_reference_radius=signature_reference_radius,
            signature_blocks_seen=0,
            signature_distance_sum=0.0,
            signature_vector_sum=torch.zeros(int(self.context_signature.declared_dimension), dtype=torch.float64),
            last_signature_block_id=-1,
            last_seen_step=self.step,
        )

    def _commit_candidate(self, slot: int, ucb: float, current_error: float | None = None) -> None:
        candidate = self.candidates[slot]
        if (
            not candidate.certified
            or candidate.commit_budget_index is None
            or candidate.commit_alpha is None
            or candidate.certified_ucb is None
        ):
            raise RuntimeError("Only an instantiated frozen consolidation test may commit")
        record = self._build_candidate_record(slot)
        if record is None:
            return
        afm = self.cfg["afm"]
        active = list(self._active_records())
        max_records = int(afm["resources"]["max_records"])
        max_atoms = int(afm["resources"].get("max_atoms", max_records))
        max_active_segments = min(max_records, max_atoms)  # record-level atomisation
        total_rows = int(afm["resources"]["total_sketch_rows"])
        required_rows = int(record.sketch.shape[0])
        release_plan: list[ProtectedRecord] = []
        survivors = list(active)
        while (
            len(survivors) + 1 > max_active_segments
            or sum(int(r.sketch.shape[0]) for r in survivors) + required_rows > total_rows
        ):
            if afm["resources"].get("capacity_policy", "release_oldest") == "overflow" or not survivors:
                self.logger.log("commit_rejected_capacity", step=self.step, slot=slot)
                return
            victim = min(survivors, key=lambda r: (r.last_seen_step, r.record_id))
            release_plan.append(victim)
            survivors.remove(victim)

        # Atomic transaction: candidate is fully built and checked before any release.
        for victim in release_plan:
            self._release_record(victim, "capacity_commit", clear_router=False)
        self.record_counter += 1
        record.record_id = self.record_counter
        self.records[record.record_id] = record
        del self.candidates[slot]
        self._refresh_router_commit_flag(slot)
        self.summary.commits += 1
        evidence_labels = torch.tensor([int(row["label"]) for row in record.evidence], dtype=torch.long)
        anchor_evidence_error = error_rate(record.anchor_outputs, evidence_labels) if len(evidence_labels) else 0.0
        activation_error = error_rate(record.activation_outputs, evidence_labels) if len(evidence_labels) else 0.0
        anchor_predictions = record.anchor_outputs.argmax(dim=1) if record.anchor_outputs.numel() else torch.empty(0)
        activation_predictions = (
            record.activation_outputs.argmax(dim=1) if record.activation_outputs.numel() else torch.empty(0)
        )
        activation_prediction_disagreement = (
            float((anchor_predictions != activation_predictions).to(torch.float32).mean().item())
            if len(evidence_labels)
            else 0.0
        )
        activation_anchor_gap = record.activation_gap
        self.summary.max_activation_gap = max(self.summary.max_activation_gap, activation_anchor_gap)
        self.logger.log(
            "record_committed",
            step=self.step,
            slot=slot,
            record_id=record.record_id,
            ucb=ucb,
            anchor_evidence_error=anchor_evidence_error,
            activation_error=activation_error,
            activation_prediction_disagreement=activation_prediction_disagreement,
            activation_anchor_gap=activation_anchor_gap,
            validation_count=candidate.certified_validation_count,
            validation_mean=(
                None
                if not candidate.certified_validation_count
                else float(candidate.certified_validation_sum or 0.0)
                / int(candidate.certified_validation_count)
            ),
            validation_accuracy=(
                None
                if not candidate.certified_validation_count
                else 1.0
                - float(candidate.certified_validation_sum or 0.0)
                / int(candidate.certified_validation_count)
            ),
            certified_step=candidate.certified_step,
            certificate_frozen=True,
            commit_alpha=candidate.commit_alpha,
            commit_budget_index=candidate.commit_budget_index,
            candidate_id=candidate.candidate_id,
            reopening_alpha=record.reopening_alpha,
            signature_alpha=record.signature_alpha,
            signature_budget_index=record.signature_budget_index,
            signature_reference_radius=record.signature_reference_radius,
            validation_signature_blocks=len(candidate.validation_signatures),
            evidence_size=len(record.evidence),
            sketch_rows=int(record.sketch.shape[0]),
            sketch_frobenius_sq=float(record.sketch.square().sum().item()),
            fd_delta=record.fd_delta,
            transfer_initial_objective=candidate.transfer_initial_objective,
            transfer_final_objective=candidate.transfer_last_objective,
            transfer_progress_total=candidate.transfer_progress_total,
            transfer_damage_total=candidate.transfer_damage_total,
            transfer_service_rounds=candidate.transfer_service_rounds,
            active_segments_for_slot=len(self._records_for_slot(slot, active_only=True)),
        )
        if bool(afm["consolidation"].get("start_successor_candidate", True)):
            # Re-anchoring is append-only: the next candidate coexists with all
            # active segments and can only add a new immutable record.
            self.candidates[slot] = self._new_candidate(slot)

    def _policy_bases(
        self, records: list[ProtectedRecord], allowed_mask: torch.Tensor
    ) -> tuple[int, list[torch.Tensor], list[float]]:
        assert self.vectoriser is not None and self.controller is not None and self.policy_family is not None
        probabilities = self.controller.probabilities().tolist()
        selected = self.controller.sample()
        matrices = [record.sketch for record in records]
        traces = [record.traces for record in records]
        bases: list[torch.Tensor | None] = [None] * len(self.policy_family)
        groups: dict[tuple[float, ...], list[int]] = {}
        for index, policy in enumerate(self.policy_family):
            groups.setdefault(policy.beta, []).append(index)
        for indices in groups.values():
            representative = indices[0]
            predicted = self.controller.predicted_record_weights(representative, traces)
            max_rank = max(self.policy_family[index].rank for index in indices)
            full_basis = top_basis_from_weighted_sketches(
                matrices=matrices,
                weights=predicted,
                rank=max_rank,
                dimension=self.vectoriser.dimension,
                device=self.device,
                allowed_mask=allowed_mask,
            )
            for index in indices:
                rank = min(self.policy_family[index].rank, full_basis.shape[1])
                bases[index] = full_basis[:, :rank]
        return selected, [basis for basis in bases if basis is not None], probabilities

    def _renewal_triggered(self) -> bool:
        renewal = self.cfg["afm"]["renewal"]
        if not bool(renewal.get("enabled", True)) or self.last_step_diagnostics is None:
            self.stall_count = 0
            return False
        d = self.last_step_diagnostics
        stalled = (
            d["loss"] > float(renewal["loss_threshold"])
            and (
                d["realised_step_length"] <= float(renewal["stall_step_threshold"])
                or d["projected_gradient_norm"] <= float(renewal["stall_gradient_threshold"])
            )
        )
        self.stall_count = self.stall_count + 1 if stalled else 0
        return self.stall_count >= int(renewal["patience"])

    def _prepare_renewal(self, images: torch.Tensor) -> RenewalTrial | None:
        if not self._renewal_triggered():
            return None
        dormant = list(self.model.adapter_pool.state().dormant)
        if not dormant:
            self.summary.renewal_capacity_obstructions += 1
            self.logger.log(
                "renewal_obstruction",
                step=self.step,
                reason="no_dormant_slot",
                active_slots=list(self.model.adapter_pool.state().active),
            )
            return None
        slot = dormant[self.summary.renewals_attempted % len(dormant)]
        self.summary.renewals_attempted += 1
        self.stall_count = 0
        budget_index, alpha = self.error_allocator.next("renewal")
        records = self._active_records()
        before_behaviour = self._record_current_outputs(records)
        self.model.eval()
        with torch.no_grad():
            before_current = self.model(images).detach()
        adapter_state = copy.deepcopy(self.model.adapter_pool.adapters[slot].state_dict())
        previous_gate = float(self.model.adapter_pool.gates[slot].detach().item())
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.cfg["seed"]) + 100003 * self.summary.renewals_attempted
        )
        self.model.adapter_pool.reset_dormant(slot, generator=generator)
        with torch.no_grad():
            after_current = self.model(images).detach()
        after_behaviour = self._record_current_outputs(records)
        zero_change = max(
            float((after_current - before_current).abs().max().item()),
            self._stacked_behaviour_drift(before_behaviour, after_behaviour),
        )
        tolerance = float(self.cfg["afm"]["renewal"]["zero_gate_tolerance"])
        if zero_change > tolerance:
            self.model.adapter_pool.adapters[slot].load_state_dict(adapter_state)
            with torch.no_grad():
                self.model.adapter_pool.gates[slot].fill_(previous_gate)
            raise RuntimeError(f"Zero-gated renewal changed the realised function by {zero_change}")
        self.logger.log(
            "renewal_reset",
            step=self.step,
            slot=slot,
            zero_change=zero_change,
            renewal_budget_index=budget_index,
            renewal_alpha=alpha,
        )
        return RenewalTrial(slot, adapter_state, previous_gate, zero_change, budget_index, alpha)

    def _rollback_renewal(self, trial: RenewalTrial) -> None:
        self.model.adapter_pool.adapters[trial.slot].load_state_dict(trial.adapter_state)
        with torch.no_grad():
            self.model.adapter_pool.gates[trial.slot].fill_(trial.previous_gate)
            self.model.adapter_pool.active_mask[trial.slot] = False

    def _record_reopening_predictions(
        self,
        record_id: int,
        images: torch.Tensor,
        signatures: torch.Tensor | None = None,
        signature_block_ids: torch.Tensor | None = None,
    ) -> ReopeningPredictionBatch | None:
        """Freeze the record and challenger predictions before outcome use.

        The threshold decision is deliberately applied only after the current
        safe update, so a crossing observation remains protected in that round.
        Computing both predictions here makes their predictability explicit
        rather than relying on the fact that later code does not inspect labels.
        """

        if self.vectoriser is None or images.numel() == 0:
            return None
        record = self.records.get(int(record_id))
        if record is None or not record.active or self.step < record.reopening_start_step:
            return None
        self.model.eval()
        with torch.no_grad():
            features = self.model.backbone(images).detach()
        with self._record_snapshot_context(record):
            self.model.eval()
            with torch.no_grad():
                record_logits = self.model.forward_from_backbone(features).detach()
        with torch.no_grad():
            challenger_logits = record.challenger.predict(features).detach()
        if signatures is None or signature_block_ids is None:
            signatures, signature_block_ids = self._signatures_with_blocks(images)
        return ReopeningPredictionBatch(
            record_id=record.record_id,
            slot=record.slot,
            features=features,
            record_logits=record_logits,
            challenger_logits=challenger_logits,
            signatures=signatures.detach().cpu(),
            signature_block_ids=signature_block_ids.detach().cpu(),
        )

    def _update_signature_shift_test(
        self, record: ProtectedRecord, prediction: ReopeningPredictionBatch
    ) -> dict[str, Any]:
        """Update one separately allocated test per distinct signature block."""

        if prediction.signature_block_ids.numel() == 0:
            prediction.signature_block_ids = torch.arange(len(prediction.signatures), dtype=torch.long)
        centroid = self.router.centroids[record.slot]
        if centroid is None:
            return {"processed_blocks": 0, "crossing_block_id": None}
        centre = centroid.detach().cpu().to(torch.float64)
        processed = 0
        crossing_block_id: int | None = None
        seen_in_batch: set[int] = set()
        for index in range(len(prediction.signature_block_ids)):
            block_id = int(prediction.signature_block_ids[index].item())
            if block_id in seen_in_batch or block_id <= record.last_signature_block_id:
                continue
            seen_in_batch.add(block_id)
            signature = prediction.signatures[index].detach().cpu().to(torch.float64)
            if record.signature_vector_sum.numel() == 0:
                record.signature_vector_sum = torch.zeros_like(signature)
            distance = float(torch.linalg.vector_norm(signature - centre).item())
            # Unit signatures and convex-combination centroids have distance in
            # [0,2].  Dividing by two gives a bounded increment interval of
            # length one, so sigma=1/2 is the universal Hoeffding scale.
            increment = 0.5 * (distance - record.signature_reference_radius)
            record.signature_eprocess.update(increment)
            record.signature_blocks_seen += 1
            record.signature_distance_sum += distance
            record.signature_vector_sum += signature
            record.last_signature_block_id = block_id
            processed += 1
            if record.signature_eprocess.crossed and crossing_block_id is None:
                crossing_block_id = block_id

        if processed:
            mean_distance = record.signature_distance_sum / max(record.signature_blocks_seen, 1)
            mean_signature = record.signature_vector_sum / max(record.signature_blocks_seen, 1)
            mean_vector_distance = float(torch.linalg.vector_norm(mean_signature - centre).item())
            self.logger.log(
                "signature_shift_score",
                step=self.step,
                record_id=record.record_id,
                slot=record.slot,
                processed_blocks=processed,
                cumulative_blocks=record.signature_blocks_seen,
                reference_radius=record.signature_reference_radius,
                mean_observed_distance=mean_distance,
                mean_signature_distance=mean_vector_distance,
                wealth=record.signature_eprocess.state.wealth,
                log_wealth=record.signature_eprocess.state.log_wealth,
                alpha=record.signature_alpha,
                crossed=bool(record.signature_eprocess.crossed),
                crossing_block_id=crossing_block_id,
                block_resolved=True,
            )
        return {"processed_blocks": processed, "crossing_block_id": crossing_block_id}

    def _apply_reopening_update(self, prediction: ReopeningPredictionBatch, labels: torch.Tensor) -> None:
        """Apply independent outcome-reopening and observable-shift tests.

        The outcome test is item-resolved.  The signature test is block-resolved
        and has a separate lifetime allocation.  A route split requires both
        crossings; an outcome-only crossing eventually releases the record
        without inventing an observable context.
        """

        record = self.records.get(int(prediction.record_id))
        if record is None or not record.active:
            return
        if record.slot != prediction.slot:
            raise RuntimeError("Reopening evidence identity changed between prediction and update")
        if self._last_split_step_by_slot.get(record.slot) == self.step:
            self.logger.log(
                "reopening_update_skipped_after_route_split",
                step=self.step,
                record_id=record.record_id,
                slot=record.slot,
                observations=int(len(labels)),
            )
            return

        self._update_signature_shift_test(record, prediction)
        slot = record.slot
        wealth = record.eprocess.state.wealth
        crossing_observation: int | None = None
        processed_observations = 0

        if not record.outcome_reopening_crossed:
            hysteresis = float(self.cfg["afm"]["reopening"]["hysteresis"])
            rec_losses = (prediction.record_logits.argmax(dim=1) != labels).to(torch.float64)
            chal_losses = (prediction.challenger_logits.argmax(dim=1) != labels).to(torch.float64)
            xs = rec_losses - chal_losses - hysteresis
            for observation, x in enumerate(xs.detach().cpu().tolist(), start=1):
                wealth = record.eprocess.update(float(x))
                processed_observations = observation
                if record.eprocess.crossed:
                    crossing_observation = observation
                    break
            if processed_observations == 0:
                processed_observations = len(xs)
            challenger_loss = record.challenger.update(
                prediction.features[:processed_observations], labels[:processed_observations]
            )
            self.logger.log(
                "reopening_score",
                step=self.step,
                record_id=record.record_id,
                observations=int(len(xs)),
                processed_observations=int(processed_observations),
                rec_error=float(rec_losses.mean().item()),
                challenger_error=float(chal_losses.mean().item()),
                mean_x=float(xs.mean().item()),
                wealth=wealth,
                log_wealth=record.eprocess.state.log_wealth,
                alpha=record.reopening_alpha,
                crossed=bool(record.eprocess.crossed),
                crossing_observation=crossing_observation,
                challenger_training_loss=challenger_loss,
                predictions_fixed_before_safe_update=True,
            )
            if record.eprocess.crossed:
                record.outcome_reopening_crossed = True
                record.outcome_crossing_step = self.step
                record.outcome_crossing_signature_count = record.signature_blocks_seen
                record.outcome_crossing_log_wealth = record.eprocess.state.log_wealth
                record.outcome_crossing_observation = crossing_observation
                self.logger.log(
                    "reopening_threshold_crossed",
                    step=self.step,
                    record_id=record.record_id,
                    slot=record.slot,
                    log_wealth=record.outcome_crossing_log_wealth,
                    alpha=record.reopening_alpha,
                    crossing_observation=crossing_observation,
                    signature_blocks_seen=record.signature_blocks_seen,
                )

        if record.outcome_reopening_crossed:
            split_cfg = self.cfg["afm"]["router"].get("split", {})
            centroid = self.router.centroids[slot]
            if centroid is None or record.signature_blocks_seen <= 0:
                mean_signature = torch.zeros_like(record.signature_vector_sum)
                mean_signature_distance = float("inf")
            else:
                mean_signature = record.signature_vector_sum / record.signature_blocks_seen
                mean_signature_distance = float(
                    torch.linalg.vector_norm(
                        mean_signature - centroid.detach().cpu().to(torch.float64)
                    ).item()
                )
            min_blocks = int(split_cfg.get("min_signature_blocks", split_cfg.get("min_observations", 1)))
            min_effect = float(
                split_cfg.get(
                    "min_effect_distance",
                    split_cfg.get("min_signature_distance", 0.0),
                )
            )
            signature_crossed = bool(record.signature_eprocess.crossed)
            split_requested = (
                bool(split_cfg.get("enabled", True))
                and signature_crossed
                and record.signature_blocks_seen >= min_blocks
                and mean_signature_distance >= min_effect
            )
            split_target: int | None = None
            preserved_by_split = False
            if split_requested:
                split_target = self._allocate_route_split(slot, mean_signature)
                if split_target is not None:
                    preserved_by_split = True
                    outcome_log_threshold = -math.log(record.reopening_alpha)
                    signature_log_threshold = -math.log(record.signature_alpha)
                    self.logger.log(
                        "record_route_split",
                        step=self.step,
                        record_id=record.record_id,
                        source_slot=slot,
                        target_slot=split_target,
                        outcome_log_wealth=record.outcome_crossing_log_wealth,
                        outcome_log_threshold=outcome_log_threshold,
                        outcome_alpha=record.reopening_alpha,
                        signature_log_wealth=record.signature_eprocess.state.log_wealth,
                        signature_log_threshold=signature_log_threshold,
                        signature_alpha=record.signature_alpha,
                        signature_blocks=record.signature_blocks_seen,
                        signature_reference_radius=record.signature_reference_radius,
                        mean_signature_distance=mean_signature_distance,
                        outcome_crossing_observation=record.outcome_crossing_observation,
                        dual_evidence_required=True,
                    )
                    for source_record in self._records_for_slot(slot, active_only=True):
                        self._reset_record_reopening_test(source_record, "dual_evidence_route_split")

            if not preserved_by_split:
                crossed_at = int(record.outcome_crossing_signature_count or 0)
                waited_blocks = max(record.signature_blocks_seen - crossed_at, 0)
                max_wait = int(split_cfg.get("max_wait_signature_blocks", 64))
                should_release = (
                    not bool(split_cfg.get("enabled", True))
                    or waited_blocks >= max_wait
                    or (split_requested and split_target is None)
                )
                if should_release:
                    outcome_log_threshold = -math.log(record.reopening_alpha)
                    self._release_record(record, "outcome_reopening_without_signature_shift", clear_router=False)
                    if slot not in self.candidates:
                        self.candidates[slot] = self._new_candidate(slot)
                    self.summary.reopenings += 1
                    self.logger.log(
                        "record_reopened",
                        step=self.step,
                        record_id=record.record_id,
                        slot=slot,
                        wealth=wealth,
                        crossing_log_wealth=record.outcome_crossing_log_wealth,
                        threshold=math.exp(min(outcome_log_threshold, 700.0)),
                        log_threshold=outcome_log_threshold,
                        threshold_display_capped=outcome_log_threshold > 700.0,
                        crossing_observation=record.outcome_crossing_observation,
                        signature_crossed=signature_crossed,
                        signature_log_wealth=record.signature_eprocess.state.log_wealth,
                        signature_alpha=record.signature_alpha,
                        signature_blocks=record.signature_blocks_seen,
                        signature_wait_blocks=waited_blocks,
                        signature_reference_radius=record.signature_reference_radius,
                        mean_signature_distance=mean_signature_distance,
                        split_requested=split_requested,
                        split_obstructed=split_requested and split_target is None,
                    )
                else:
                    self.logger.log(
                        "reopening_split_pending",
                        step=self.step,
                        record_id=record.record_id,
                        slot=slot,
                        signature_crossed=signature_crossed,
                        signature_blocks=record.signature_blocks_seen,
                        signature_wait_blocks=waited_blocks,
                        max_wait_signature_blocks=max_wait,
                        signature_log_wealth=record.signature_eprocess.state.log_wealth,
                        signature_alpha=record.signature_alpha,
                        mean_signature_distance=mean_signature_distance,
                    )
        self._refresh_router_commit_flag(slot)

    def _record_reopening_update(self, record_id: int, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Compatibility wrapper used by focused tests and external callers."""

        prediction = self._record_reopening_predictions(record_id, images)
        if prediction is not None:
            self._apply_reopening_update(prediction, labels)

    def _certificate_values(self, has_active_behaviour: bool) -> tuple[float, float, float, float, bool]:
        cert = self.cfg["afm"]["certificates"]
        if self.certificate_mode == "strict":
            required = ["loss_smoothness_bound", "trust_radius_cap", "provenance"]
            if has_active_behaviour:
                required += ["behaviour_curvature_bound", "jacobian_lipschitz_bound"]
            numerics = self.cfg["afm"].get("numerics", {})
            if (
                any(cert.get(name) is None for name in required)
                or not bool(cert.get("certified", False))
                or not bool(numerics.get("certified", False))
                or numerics.get("provenance") is None
                or not bool(numerics.get("endpoint_error_certified", False))
                or numerics.get("endpoint_error_bound") is None
                or numerics.get("endpoint_error_provenance") is None
            ):
                return float("inf"), float("inf"), float("inf"), 0.0, False
            L = float(cert["loss_smoothness_bound"])
            H = float(cert["behaviour_curvature_bound"]) if has_active_behaviour else 0.0
            LJ = float(cert["jacobian_lipschitz_bound"]) if has_active_behaviour else 0.0
            cap = float(cert["trust_radius_cap"])
            return L, H, LJ, cap, True
        L = float(cert.get("empirical_loss_smoothness", 100.0))
        H = float(cert.get("empirical_behaviour_curvature", 100.0)) if has_active_behaviour else 0.0
        LJ = float(cert.get("empirical_jacobian_lipschitz", 1e-3)) if has_active_behaviour else 0.0
        cap = float(self.cfg["afm"]["safe_update"]["trust_radius_cap"])
        return L, H, LJ, cap, False

    def _behaviour_budget(self, has_active_behaviour: bool) -> float:
        if not has_active_behaviour:
            return 0.0
        schedule = self.cfg["afm"]["safe_update"]
        kind = str(schedule.get("budget_schedule", "finite_power"))
        horizon = self.effective_afm_horizon
        t = self.protection_round
        if kind == "finite_uniform":
            return float(schedule["total_budget"]) / horizon
        if kind == "finite_power":
            exponent = float(schedule["budget_exponent"])
            normalizer = sum((i + 1.0) ** (-exponent) for i in range(horizon))
            return float(schedule["total_budget"]) * (t + 1.0) ** (-exponent) / normalizer
        return float(schedule["budget_b0"]) / ((t + 1.0) ** float(schedule["budget_exponent"]))

    def _numerical_leakage(self, basis: torch.Tensor, total_sketch_energy: float) -> tuple[float, float]:
        numerics = self.cfg["afm"].get("numerics", {})
        if basis.numel() == 0:
            orth_error = 0.0
        else:
            work = basis.detach().to(device="cpu", dtype=torch.float64)
            eye = torch.eye(work.shape[1], device=work.device, dtype=work.dtype)
            orth_error = float(torch.linalg.matrix_norm(work.T @ work - eye, ord=2).item())
        declared = float(numerics.get("arithmetic_error_bound", 0.0))
        measured = orth_error * math.sqrt(max(total_sketch_energy, 0.0))
        return declared + measured, orth_error

    def _attempt_functional_shield_transfer(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        records: list[ProtectedRecord],
        transfer_batches: list[CandidateTransferBatch],
        old_vector: torch.Tensor,
        old_loss: float,
        ordinary_proposal: SafeStep,
        projected_current_gradient: torch.Tensor,
        current_smoothness: float,
        before_behaviour: dict[int, torch.Tensor],
        behaviour_budget: float,
        endpoint_tolerance: float,
        max_backtracks: int,
        nonzero_tolerance: float,
    ) -> FunctionalShieldDeployment:
        """Deploy the ordinary protected update through an exact finite logit shield.

        The base-parameter proposal is the same ordinary protected step used by
        AFM without candidate transfer.  The structural solve then preserves its
        current-batch logits, restores every protected and unselected-candidate
        logit, and contracts the predictably selected candidate toward its frozen
        target.  Inconsistent observable constraints or uncertified numerics
        produce an obstruction and leave both parameter and shield state intact.
        """

        if not transfer_batches:
            raise ValueError("Functional-shield transfer requires at least one candidate")
        shield_cfg = dict(self.cfg["afm"].get("functional_shield", {}))
        transfer_cfg = dict(self.cfg["afm"].get("transfer", {}))
        selected = transfer_batches[0]
        contraction = float(shield_cfg.get("selected_contraction", 1.0))
        progress_fraction = float(transfer_cfg.get("min_progress_fraction", 0.25))
        ordinary_best = unconstrained_ball_best(
            projected_current_gradient,
            current_smoothness,
            ordinary_proposal.radius,
        )
        current_required = progress_fraction * ordinary_best
        original_shield = self.model.functional_shield.snapshot()

        self.model.eval()
        with torch.no_grad():
            current_features = self.model.encode_backbone(images).detach()
            current_pre_logits = self.model.forward_from_backbone(current_features).detach()

        record_features: list[torch.Tensor] = []
        record_pre_logits: list[torch.Tensor] = []
        for record in records:
            record_images, _ = self._load_evidence(record)
            with torch.no_grad():
                features = self.model.encode_backbone(record_images).detach()
                logits = self.model.forward_from_backbone(features).detach()
            record_features.append(features)
            record_pre_logits.append(logits)

        candidate_features: list[torch.Tensor] = []
        candidate_pre_logits: list[torch.Tensor] = []
        candidate_before: dict[int, float] = {}
        for batch in transfer_batches:
            with torch.no_grad():
                features = self.model.encode_backbone(batch.images).detach()
                logits = self.model.forward_from_backbone(features).detach()
                behaviour = behaviour_from_logits(logits, self.behaviour_spec)
                objective = 0.5 * (behaviour - batch.targets).square().sum(dim=1).mean()
            candidate_features.append(features)
            candidate_pre_logits.append(logits)
            candidate_before[batch.candidate_id] = float(objective.item())

        solve_result: ShieldSolveResult | None = None
        last_obstruction: str | None = None
        local_factor = 1.0
        for _ in range(max_backtracks + 1):
            self.vectoriser.assign(old_vector.to(self.device) + local_factor * ordinary_proposal.delta)
            self.model.functional_shield.restore(original_shield)
            self._set_afm_train_mode()
            with torch.no_grad():
                current_ordinary_logits = self.model.forward_from_backbone(current_features).detach()

                nodes: list[torch.Tensor] = [current_features]
                desired_logits: list[torch.Tensor] = [current_ordinary_logits]
                pre_constraint_logits: list[torch.Tensor] = [current_pre_logits]
                nodes.extend(record_features)
                desired_logits.extend(record_pre_logits)
                pre_constraint_logits.extend(record_pre_logits)
                for batch, features, pre_logits in zip(
                    transfer_batches, candidate_features, candidate_pre_logits
                ):
                    nodes.append(features)
                    if batch.candidate_id == selected.candidate_id:
                        target = (1.0 - contraction) * pre_logits + contraction * batch.target_logits
                    else:
                        target = pre_logits
                    desired_logits.append(target)
                    pre_constraint_logits.append(pre_logits)

                all_nodes = torch.cat(nodes, dim=0)
                desired = torch.cat(desired_logits, dim=0)
                pre_constraints = torch.cat(pre_constraint_logits, dim=0)
                base_after = self.model.base_logits_from_backbone(all_nodes).detach()
                residual_targets = desired - base_after
                shield_update_norm = float((desired - pre_constraints).abs().max().item())

            feature_match_tolerance = float(shield_cfg.get("feature_match_tolerance", 1e-8))
            if self.certificate_mode == "strict":
                certified_address_error = shield_cfg.get("feature_distance_error_bound")
                if certified_address_error is not None:
                    feature_match_tolerance = max(
                        feature_match_tolerance, float(certified_address_error)
                    )
            solve_result = self.model.functional_shield.solve_and_replace(
                all_nodes,
                residual_targets,
                guard_nodes=self.functional_shield_guard_features,
                support_multiplier=float(shield_cfg.get("support_multiplier", 4.0)),
                feature_match_tolerance=feature_match_tolerance,
                duplicate_tolerance=float(shield_cfg.get("duplicate_tolerance", 1e-10)),
                target_tolerance=float(shield_cfg.get("target_tolerance", endpoint_tolerance)),
                residual_tolerance=float(shield_cfg.get("residual_tolerance", endpoint_tolerance)),
                coefficient_norm_limit=float(shield_cfg.get("coefficient_norm_limit", float("inf"))),
            )
            if not solve_result.available:
                last_obstruction = solve_result.obstruction
                if last_obstruction in {
                    "functional_shield_capacity_exceeded",
                    "functional_shield_nonfinite_target_obstruction",
                    "functional_shield_guard_leakage_obstruction",
                    "functional_shield_address_resolution_obstruction",
                }:
                    break
                local_factor *= float(self.cfg["afm"]["safe_update"].get("backtrack_factor", 0.5))
                continue

            self.model.eval()
            candidate_after: dict[int, float] = {}
            with torch.no_grad():
                current_logits = self.model.forward_from_backbone(current_features)
                new_loss = float(self._current_loss(current_logits, labels).item())
                after_behaviour: dict[int, torch.Tensor] = {}
                for record, features in zip(records, record_features):
                    logits = self.model.forward_from_backbone(features)
                    after_behaviour[record.record_id] = behaviour_from_logits(
                        logits, self.behaviour_spec
                    ).detach()
                for batch, features in zip(transfer_batches, candidate_features):
                    logits = self.model.forward_from_backbone(features)
                    behaviour = behaviour_from_logits(logits, self.behaviour_spec)
                    value = 0.5 * (behaviour - batch.targets).square().sum(dim=1).mean()
                    candidate_after[batch.candidate_id] = float(value.item())

            drift = self._stacked_behaviour_drift(before_behaviour, after_behaviour)
            current_decrease = old_loss - new_loss
            candidate_decreases = tuple(
                candidate_before[batch.candidate_id] - candidate_after[batch.candidate_id]
                for batch in transfer_batches
            )
            selected_decrease = candidate_before[selected.candidate_id] - candidate_after[selected.candidate_id]
            strict_error = 0.0
            if self.certificate_mode == "strict":
                strict_error = float(shield_cfg.get("feature_distance_error_bound", 0.0)) + 2.0 * float(
                    self.cfg["afm"].get("numerics", {}).get("endpoint_error_bound") or 0.0
                )
            current_ok = current_decrease + endpoint_tolerance >= current_required + strict_error
            retention_ok = drift <= behaviour_budget + endpoint_tolerance
            candidate_ok = all(value >= -endpoint_tolerance - strict_error for value in candidate_decreases)
            selected_ok = selected_decrease > endpoint_tolerance + strict_error
            nonzero = (
                local_factor * ordinary_proposal.step_length > nonzero_tolerance
                or shield_update_norm > nonzero_tolerance
            )
            if current_ok and retention_ok and candidate_ok and selected_ok and nonzero:
                return FunctionalShieldDeployment(
                    accepted=True,
                    factor=local_factor,
                    drift=drift,
                    new_loss=new_loss,
                    selected_after=candidate_after[selected.candidate_id],
                    candidate_before=candidate_before,
                    candidate_after=candidate_after,
                    candidate_decreases=candidate_decreases,
                    current_ordinary_best=ordinary_best,
                    current_required=current_required,
                    current_certified_decrease=current_decrease - strict_error,
                    selected_certified_decrease=selected_decrease - strict_error,
                    solve=solve_result,
                    obstruction=None,
                    shield_update_norm=shield_update_norm,
                    selected_candidate_id=selected.candidate_id,
                )

            if not current_ok:
                last_obstruction = "functional_shield_current_endpoint_obstruction"
            elif not retention_ok:
                last_obstruction = "functional_shield_retention_endpoint_obstruction"
            elif not candidate_ok:
                last_obstruction = "functional_shield_candidate_endpoint_obstruction"
            elif not selected_ok:
                last_obstruction = "functional_shield_selected_progress_obstruction"
            else:
                last_obstruction = "functional_shield_zero_update"
            local_factor *= float(self.cfg["afm"]["safe_update"].get("backtrack_factor", 0.5))

        self.vectoriser.assign(old_vector.to(self.device))
        self.model.functional_shield.restore(original_shield)
        return FunctionalShieldDeployment(
            accepted=False,
            factor=0.0,
            drift=0.0,
            new_loss=old_loss,
            selected_after=candidate_before.get(selected.candidate_id),
            candidate_before=candidate_before,
            candidate_after=dict(candidate_before),
            candidate_decreases=tuple(0.0 for _ in transfer_batches),
            current_ordinary_best=ordinary_best,
            current_required=current_required,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            solve=solve_result,
            obstruction=last_obstruction or "functional_shield_abstention",
            shield_update_norm=0.0,
            selected_candidate_id=selected.candidate_id,
        )


    def _proposal_endpoint(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        proposal: SafeStep,
        projected_gradient_norm: float,
        old_vector: torch.Tensor,
        old_loss: float,
        old_logits: torch.Tensor,
        records: list[ProtectedRecord],
        before_behaviour: dict[int, torch.Tensor],
        behaviour_budget: float,
        nonzero_tolerance: float,
        max_backtracks: int,
        check_tolerance: float,
        require_retention: bool,
    ) -> dict[str, Any]:
        """Evaluate one proposal transactionally from the same pre-step state.

        Endpoint probes never commit parameters. The returned realised delta is
        the unique accepted backtracking endpoint under the declared current-loss
        and, when requested, complete bounded behaviour checks.
        """

        assert self.vectoriser is not None
        schedule = self.cfg["afm"]["safe_update"]
        probe_snapshot = self._snapshot_joint_predictive_state(
            old_vector=old_vector,
            shield_state=self.model.functional_shield.snapshot(),
        )
        accepted = False
        factor = 0.0
        new_loss = old_loss
        drift = 0.0
        endpoint_logits = old_logits.detach().clone()
        accepted_predictive_state: dict[str, Any] | None = None
        if proposal.step_length > nonzero_tolerance:
            local_factor = 1.0
            try:
                for _ in range(max_backtracks + 1):
                    # Every candidate factor starts from the identical pre-step
                    # parameters, buffers, shield, module modes, and random state.
                    # This makes the comparator and safe endpoint true same-state
                    # transactions rather than sequential probes with hidden
                    # BatchNorm or random-state carry-over.
                    self._restore_joint_predictive_state(probe_snapshot)
                    delta = local_factor * proposal.delta
                    self.vectoriser.assign(old_vector.to(self.device) + delta)
                    self._set_afm_train_mode()
                    with torch.no_grad():
                        candidate_logits = self.model(images)
                        candidate_loss = float(
                            self._current_loss(candidate_logits, labels).item()
                        )
                    effective_s = local_factor * proposal.step_length
                    rhs = old_loss - 0.5 * effective_s * projected_gradient_norm
                    current_ok = candidate_loss <= rhs + check_tolerance
                    local_drift = 0.0
                    retention_ok = True
                    if require_retention and records:
                        after_behaviour = self._record_current_outputs(records)
                        local_drift = self._stacked_behaviour_drift(
                            before_behaviour, after_behaviour
                        )
                        retention_ok = local_drift <= behaviour_budget + check_tolerance
                    if current_ok and retention_ok and effective_s > nonzero_tolerance:
                        accepted = True
                        factor = local_factor
                        new_loss = candidate_loss
                        drift = local_drift
                        endpoint_logits = candidate_logits.detach().clone()
                        accepted_predictive_state = self._snapshot_joint_predictive_state(
                            old_vector=self.vectoriser.flatten(detach=True),
                            shield_state=self.model.functional_shield.snapshot(),
                        )
                        break
                    local_factor *= float(schedule.get("backtrack_factor", 0.5))
            finally:
                self._restore_joint_predictive_state(probe_snapshot)
        realised_delta = factor * proposal.delta if accepted else torch.zeros_like(proposal.delta)
        return {
            "accepted": accepted,
            "factor": float(factor),
            "old_loss": float(old_loss),
            "new_loss": float(new_loss),
            "decrease": float(old_loss - new_loss),
            "projected_gradient_norm": float(projected_gradient_norm),
            "safe_radius": float(proposal.radius),
            "proposed_step_length": float(proposal.step_length),
            "realised_step_length": float(factor * proposal.step_length),
            "delta": realised_delta.detach().clone(),
            "endpoint_logits": endpoint_logits,
            "proposal": proposal,
            "drift": float(drift),
            "predictive_state": accepted_predictive_state,
        }

    def _exact_no_protection_endpoint(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        gradient: torch.Tensor,
        allowed_mask: torch.Tensor,
        old_vector: torch.Tensor,
        old_loss: float,
        old_logits: torch.Tensor,
        nonzero_tolerance: float,
        max_backtracks: int,
        check_tolerance: float,
    ) -> dict[str, Any]:
        """Compute the genuine AFM-no-protection comparator endpoint.

        The comparator uses the same pre-step deployed state, current gradient,
        active-coordinate mask, learning-rate rule and backtracking rule, but no
        protected basis or retention radius. It is evaluated transactionally.
        """

        assert self.vectoriser is not None
        schedule = self.cfg["afm"]["safe_update"]
        ordinary_L, _, _, ordinary_cap, theorem_certified = self._certificate_values(False)
        eta = float(schedule["learning_rate"])
        if math.isfinite(ordinary_L):
            eta = min(eta, 1.0 / max(ordinary_L, 1e-12))
        else:
            ordinary_cap = 0.0
        empty_basis = torch.zeros(
            (self.vectoriser.dimension, 0), device=gradient.device, dtype=gradient.dtype
        )
        proposal = make_safe_step(
            gradient=gradient, basis=empty_basis, learning_rate=eta, E=0.0, H=0.0,
            budget=0.0, cap=ordinary_cap, allowed_mask=allowed_mask,
        )
        q = project_to_allowed_free_subspace(gradient, empty_basis, allowed_mask)
        result = self._proposal_endpoint(
            images=images, labels=labels, proposal=proposal,
            projected_gradient_norm=float(torch.linalg.vector_norm(q).item()),
            old_vector=old_vector, old_loss=old_loss, old_logits=old_logits,
            records=[], before_behaviour={}, behaviour_budget=0.0,
            nonzero_tolerance=nonzero_tolerance, max_backtracks=max_backtracks,
            check_tolerance=check_tolerance, require_retention=False,
        )
        result["theorem_certified"] = theorem_certified
        return result

    def _safe_metaplastic_endpoint(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        proposal: SafeStep,
        projected_gradient_norm: float,
        records: list[ProtectedRecord],
        before_behaviour: dict[int, torch.Tensor],
        behaviour_budget: float,
        old_vector: torch.Tensor,
        old_loss: float,
        old_logits: torch.Tensor,
        nonzero_tolerance: float,
        max_backtracks: int,
        check_tolerance: float,
    ) -> dict[str, Any]:
        """Compute the persistent budget-controlled metaplastic base endpoint."""

        return self._proposal_endpoint(
            images=images, labels=labels, proposal=proposal,
            projected_gradient_norm=projected_gradient_norm,
            old_vector=old_vector, old_loss=old_loss, old_logits=old_logits,
            records=records, before_behaviour=before_behaviour,
            behaviour_budget=behaviour_budget, nonzero_tolerance=nonzero_tolerance,
            max_backtracks=max_backtracks, check_tolerance=check_tolerance,
            require_retention=bool(records),
        )

    def _snapshot_joint_predictive_state(
        self,
        *,
        old_vector: torch.Tensor,
        shield_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Snapshot every mutable predictive component touched by a joint probe.

        Audit/service counters intentionally remain monotone and record failed
        transactions. The endpoint probes do not step an optimiser, so optimiser
        state is outside this transaction.
        """

        return {
            "parameters": old_vector.detach().clone(),
            "shield": copy.deepcopy(shield_state),
            "buffers": {
                name: buffer.detach().clone()
                for name, buffer in self.model.named_buffers()
                if not name.startswith("functional_shield.")
            },
            "module_training": {name: module.training for name, module in self.model.named_modules()},
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
            "numpy_rng": copy.deepcopy(np.random.get_state()),
            "python_rng": random.getstate(),
        }

    def _restore_joint_predictive_state(self, snapshot: dict[str, Any]) -> None:
        assert self.vectoriser is not None
        self.vectoriser.assign(snapshot["parameters"].to(self.device))
        self.model.functional_shield.restore(snapshot["shield"])
        current_buffers = dict(self.model.named_buffers())
        with torch.no_grad():
            for name, value in snapshot["buffers"].items():
                if name in current_buffers:
                    current_buffers[name].copy_(value.to(current_buffers[name].device))
        for name, module in self.model.named_modules():
            module.train(bool(snapshot["module_training"].get(name, module.training)))
        torch.set_rng_state(snapshot["torch_rng"])
        if snapshot["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        random.setstate(snapshot["python_rng"])

    def _attempt_exact_counterfactual_restoration(
        self,
        **kwargs: Any,
    ) -> FunctionalShieldDeployment:
        """Atomic wrapper for the joint safe-base/endpoint-emulation transaction."""

        assert self.vectoriser is not None
        old_vector = kwargs["old_vector"]
        shield_state = self.model.functional_shield.snapshot()
        snapshot = self._snapshot_joint_predictive_state(
            old_vector=old_vector, shield_state=shield_state
        )
        try:
            result = self._attempt_exact_counterfactual_restoration_impl(**kwargs)
        except Exception:
            self._restore_joint_predictive_state(snapshot)
            raise
        if not result.accepted:
            self._restore_joint_predictive_state(snapshot)
        return result

    def _attempt_exact_counterfactual_restoration_impl(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        records: list[ProtectedRecord],
        transfer_batches: list[CandidateTransferBatch],
        gradient: torch.Tensor,
        allowed_mask: torch.Tensor,
        old_vector: torch.Tensor,
        old_loss: float,
        old_logits: torch.Tensor,
        before_behaviour: dict[int, torch.Tensor],
        protected_basis: torch.Tensor,
        leakage_bound: float,
        curvature_bound: float,
        retention_cap: float,
        loss_smoothness: float,
        charge_fraction: float,
        endpoint_tolerance: float,
        max_backtracks: int,
        nonzero_tolerance: float,
    ) -> FunctionalShieldDeployment:
        """Commit a safe metaplastic base endpoint plus exact endpoint emulation.

        The genuine ``afm_no_protection`` endpoint supplies the current-batch
        comparator logits and displacement. Persistent parameters take the
        maximal certified fraction of its protected-space projection under a
        predeclared fraction of the full projected retention charge. A compact-cardinal
        shield then emulates the comparator on current data, restores active
        protected and unselected-candidate logits, and applies the selected
        target. Any failed base or shield certificate rolls back atomically.
        """

        assert self.vectoriser is not None
        shield_cfg = dict(self.cfg["afm"].get("functional_shield", {}))
        transfer_cfg = dict(self.cfg["afm"].get("transfer", {}))
        progress_fraction = float(transfer_cfg.get("min_progress_fraction", 0.25))
        original_shield = self.model.functional_shield.snapshot()

        self.model.eval()
        with torch.no_grad():
            current_features = self.model.encode_backbone(images).detach()
            current_pre_logits = self.model.forward_from_backbone(current_features).detach()

        record_features: list[torch.Tensor] = []
        record_pre_logits: list[torch.Tensor] = []
        for record in records:
            record_images, _ = self._load_evidence(record)
            with torch.no_grad():
                features = self.model.encode_backbone(record_images).detach()
                logits = self.model.forward_from_backbone(features).detach()
            record_features.append(features)
            record_pre_logits.append(logits)

        candidate_features: list[torch.Tensor] = []
        candidate_pre_logits: list[torch.Tensor] = []
        candidate_before: dict[int, float] = {}
        for batch in transfer_batches:
            with torch.no_grad():
                features = self.model.encode_backbone(batch.images).detach()
                logits = self.model.forward_from_backbone(features).detach()
                behaviour = behaviour_from_logits(logits, self.behaviour_spec)
                objective = 0.5 * (behaviour - batch.targets).square().sum(dim=1).mean()
            candidate_features.append(features)
            candidate_pre_logits.append(logits)
            candidate_before[batch.candidate_id] = float(objective.item())

        counterfactual = self._exact_no_protection_endpoint(
            images=images,
            labels=labels,
            gradient=gradient,
            allowed_mask=allowed_mask,
            old_vector=old_vector,
            old_loss=old_loss,
            old_logits=old_logits,
            nonzero_tolerance=nonzero_tolerance,
            max_backtracks=max_backtracks,
            check_tolerance=endpoint_tolerance,
        )
        if not counterfactual["accepted"]:
            return FunctionalShieldDeployment(
                accepted=False,
                factor=0.0,
                drift=0.0,
                new_loss=old_loss,
                selected_after=(None if not transfer_batches else candidate_before.get(transfer_batches[0].candidate_id)),
                candidate_before=candidate_before,
                candidate_after=dict(candidate_before),
                candidate_decreases=tuple(0.0 for _ in transfer_batches),
                current_ordinary_best=max(float(counterfactual["decrease"]), 0.0),
                current_required=progress_fraction * max(float(counterfactual["decrease"]), 0.0),
                current_certified_decrease=0.0,
                selected_certified_decrease=0.0,
                solve=None,
                obstruction="exact_counterfactual_endpoint_unavailable",
                shield_update_norm=0.0,
                selected_candidate_id=(None if not transfer_batches else transfer_batches[0].candidate_id),
                proposal=counterfactual["proposal"],
                counterfactual_accepted=False,
                counterfactual_decrease=float(counterfactual["decrease"]),
                exact_progress_ratio=None,
            )

        projected_gradient = project_to_allowed_free_subspace(
            gradient, protected_basis, allowed_mask
        )
        projected_gradient_norm = float(torch.linalg.vector_norm(projected_gradient).item())
        alignment_relative_tolerance = float(
            shield_cfg.get("alignment_relative_tolerance", 1e-5)
        )
        try:
            plan = make_counterfactual_normalized_plan(
                counterfactual_delta=counterfactual["delta"],
                ordinary_gradient=gradient,
                protected_basis=protected_basis,
                allowed_mask=allowed_mask,
                E=leakage_bound,
                H=curvature_bound,
                charge_fraction=charge_fraction,
                cap=retention_cap,
                active_protection=bool(records),
                loss_smoothness=loss_smoothness,
                tolerance=alignment_relative_tolerance,
            )
        except ValueError:
            return FunctionalShieldDeployment(
                accepted=False, factor=0.0, drift=0.0, new_loss=old_loss,
                selected_after=(None if not transfer_batches else candidate_before.get(transfer_batches[0].candidate_id)),
                candidate_before=candidate_before, candidate_after=dict(candidate_before),
                candidate_decreases=tuple(0.0 for _ in transfer_batches),
                current_ordinary_best=max(float(counterfactual["decrease"]), 0.0),
                current_required=progress_fraction * max(float(counterfactual["decrease"]), 0.0),
                current_certified_decrease=0.0, selected_certified_decrease=0.0,
                solve=None, obstruction="counterfactual_normalization_obstruction",
                shield_update_norm=0.0,
                selected_candidate_id=(None if not transfer_batches else transfer_batches[0].candidate_id),
                proposal=counterfactual["proposal"], counterfactual_accepted=True,
                counterfactual_decrease=float(counterfactual["decrease"]),
                exact_progress_ratio=None, safe_base_accepted=False,
                requested_charge_fraction=float(charge_fraction),
                shield_residual_progress_fraction=1.0,
            )
        if plan.reference_step_length <= nonzero_tolerance:
            return FunctionalShieldDeployment(
                accepted=False, factor=0.0, drift=0.0, new_loss=old_loss,
                selected_after=(None if not transfer_batches else candidate_before.get(transfer_batches[0].candidate_id)),
                candidate_before=candidate_before, candidate_after=dict(candidate_before),
                candidate_decreases=tuple(0.0 for _ in transfer_batches),
                current_ordinary_best=max(float(counterfactual["decrease"]), 0.0),
                current_required=progress_fraction * max(float(counterfactual["decrease"]), 0.0),
                current_certified_decrease=0.0, selected_certified_decrease=0.0,
                solve=None, obstruction="zero_compatible_counterfactual_projection",
                shield_update_norm=0.0,
                selected_candidate_id=(None if not transfer_batches else transfer_batches[0].candidate_id),
                proposal=plan.proposal, counterfactual_accepted=True,
                counterfactual_decrease=float(counterfactual["decrease"]),
                exact_progress_ratio=None, safe_base_accepted=False,
                retention_budget=plan.retention_budget,
                reference_retention_charge=plan.reference_charge,
                requested_charge_fraction=plan.requested_charge_fraction,
                selected_path_fraction=plan.selected_path_fraction,
                realised_path_fraction=0.0,
                persistent_base_progress_ratio=0.0,
                compatible_gradient_fraction=plan.compatibility_fraction,
                ordinary_step_size=plan.ordinary_step_size,
                step_size_smoothness_product=plan.step_size_smoothness_product,
                scalar_comparator_certified=False,
                shield_residual_progress_fraction=1.0,
            )
        alignment_scale = max(
            float(torch.linalg.vector_norm(counterfactual["delta"]).item()),
            plan.reference_step_length,
            1e-12,
        )
        alignment_tolerance = alignment_relative_tolerance * alignment_scale
        zero_compatibility = (
            plan.reference_step_length <= nonzero_tolerance
            or plan.compatibility_fraction <= nonzero_tolerance
        )
        if (
            zero_compatibility
            or not plan.scalar_comparator_certified
            or plan.projected_counterfactual_alignment_error > alignment_tolerance
            or plan.ordinary_counterfactual_alignment_error > alignment_tolerance
            or plan.projection_idempotence_error > alignment_tolerance
        ):
            alignment_obstruction = (
                "persistent_assimilation_zero_compatibility_obstruction"
                if zero_compatibility
                else "counterfactual_projection_alignment_obstruction"
            )
            return FunctionalShieldDeployment(
                accepted=False, factor=0.0, drift=0.0, new_loss=old_loss,
                selected_after=(None if not transfer_batches else candidate_before.get(transfer_batches[0].candidate_id)),
                candidate_before=candidate_before, candidate_after=dict(candidate_before),
                candidate_decreases=tuple(0.0 for _ in transfer_batches),
                current_ordinary_best=max(float(counterfactual["decrease"]), 0.0),
                current_required=progress_fraction * max(float(counterfactual["decrease"]), 0.0),
                current_certified_decrease=0.0, selected_certified_decrease=0.0,
                solve=None, obstruction=alignment_obstruction,
                shield_update_norm=0.0,
                selected_candidate_id=(None if not transfer_batches else transfer_batches[0].candidate_id),
                proposal=plan.proposal, counterfactual_accepted=True,
                counterfactual_decrease=float(counterfactual["decrease"]),
                exact_progress_ratio=None, safe_base_accepted=False,
                retention_budget=plan.retention_budget,
                reference_retention_charge=plan.reference_charge,
                requested_charge_fraction=plan.requested_charge_fraction,
                selected_path_fraction=plan.selected_path_fraction,
                realised_path_fraction=0.0,
                persistent_base_progress_ratio=0.0,
                persistent_descent_lower_bound=0.0,
                projected_counterfactual_alignment_error=plan.projected_counterfactual_alignment_error,
                ordinary_counterfactual_alignment_error=plan.ordinary_counterfactual_alignment_error,
                projection_idempotence_error=plan.projection_idempotence_error,
                compatible_gradient_fraction=plan.compatibility_fraction,
                ordinary_step_size=plan.ordinary_step_size,
                step_size_smoothness_product=plan.step_size_smoothness_product,
                scalar_comparator_certified=plan.scalar_comparator_certified,
                analytic_persistent_progress_ratio_lower_bound=0.0,
                certified_persistent_progress_ratio_lower_bound=0.0,
                shield_residual_progress_fraction=1.0,
            )
        planned_descent_lower_bound = persistent_descent_lower_bound(
            projected_gradient_norm=projected_gradient_norm,
            reference_step_length=plan.reference_step_length,
            selected_path_fraction=plan.selected_path_fraction,
            smoothness=loss_smoothness,
        )

        safe_endpoint = self._safe_metaplastic_endpoint(
            images=images,
            labels=labels,
            proposal=plan.proposal,
            projected_gradient_norm=projected_gradient_norm,
            records=records,
            before_behaviour=before_behaviour,
            behaviour_budget=plan.retention_budget,
            old_vector=old_vector,
            old_loss=old_loss,
            old_logits=old_logits,
            nonzero_tolerance=nonzero_tolerance,
            max_backtracks=max_backtracks,
            check_tolerance=endpoint_tolerance,
        )
        if not safe_endpoint["accepted"]:
            return FunctionalShieldDeployment(
                accepted=False, factor=0.0, drift=0.0, new_loss=old_loss,
                selected_after=(None if not transfer_batches else candidate_before.get(transfer_batches[0].candidate_id)),
                candidate_before=candidate_before, candidate_after=dict(candidate_before),
                candidate_decreases=tuple(0.0 for _ in transfer_batches),
                current_ordinary_best=max(float(counterfactual["decrease"]), 0.0),
                current_required=progress_fraction * max(float(counterfactual["decrease"]), 0.0),
                current_certified_decrease=0.0, selected_certified_decrease=0.0,
                solve=None, obstruction="safe_metaplastic_base_endpoint_unavailable",
                shield_update_norm=0.0,
                selected_candidate_id=(None if not transfer_batches else transfer_batches[0].candidate_id),
                proposal=plan.proposal, counterfactual_accepted=True,
                counterfactual_decrease=float(counterfactual["decrease"]),
                exact_progress_ratio=None, safe_base_accepted=False,
                safe_base_decrease=float(safe_endpoint["decrease"]),
                safe_base_drift=float(safe_endpoint["drift"]),
                safe_base_radius=float(safe_endpoint["safe_radius"]),
                safe_base_step_length=float(safe_endpoint["realised_step_length"]),
                retention_budget=plan.retention_budget,
                reference_retention_charge=plan.reference_charge,
                requested_charge_fraction=plan.requested_charge_fraction,
                selected_path_fraction=plan.selected_path_fraction,
                realised_path_fraction=0.0,
                persistent_base_progress_ratio=0.0,
                persistent_descent_lower_bound=planned_descent_lower_bound,
                projected_counterfactual_alignment_error=plan.projected_counterfactual_alignment_error,
                ordinary_counterfactual_alignment_error=plan.ordinary_counterfactual_alignment_error,
                projection_idempotence_error=plan.projection_idempotence_error,
                compatible_gradient_fraction=plan.compatibility_fraction,
                ordinary_step_size=plan.ordinary_step_size,
                step_size_smoothness_product=plan.step_size_smoothness_product,
                scalar_comparator_certified=plan.scalar_comparator_certified,
                analytic_persistent_progress_ratio_lower_bound=plan.analytic_persistent_progress_ratio_lower_bound,
                certified_persistent_progress_ratio_lower_bound=(
                    planned_descent_lower_bound / float(counterfactual["decrease"])
                    if float(counterfactual["decrease"]) > 0.0 else None
                ),
                shield_residual_progress_fraction=1.0,
            )

        realised_path_fraction = (
            float(safe_endpoint["realised_step_length"]) / plan.reference_step_length
            if plan.reference_step_length > nonzero_tolerance
            else 0.0
        )
        persistent_progress_ratio = (
            float(safe_endpoint["decrease"]) / float(counterfactual["decrease"])
            if float(counterfactual["decrease"]) > 0.0
            else None
        )
        descent_lower_bound = persistent_descent_lower_bound(
            projected_gradient_norm=projected_gradient_norm,
            reference_step_length=plan.reference_step_length,
            selected_path_fraction=realised_path_fraction,
            smoothness=loss_smoothness,
        )
        certified_progress_ratio_lower_bound = (
            descent_lower_bound / float(counterfactual["decrease"])
            if float(counterfactual["decrease"]) > 0.0
            else None
        )
        analytic_progress_ratio_lower_bound = (
            realised_path_fraction
            * plan.compatibility_fraction
            * max(
                1.0
                - 0.5
                * plan.step_size_smoothness_product
                * realised_path_fraction,
                0.0,
            )
            / (1.0 + 0.5 * plan.step_size_smoothness_product)
            if plan.scalar_comparator_certified
            else 0.0
        )
        fraction_tolerance = max(1e-8, alignment_relative_tolerance)
        assimilation_fraction_ok = (
            not records
            or realised_path_fraction + fraction_tolerance >= plan.requested_charge_fraction
        )
        persistent_descent_ok = (
            float(safe_endpoint["decrease"]) + endpoint_tolerance >= descent_lower_bound
        )
        persistent_positive_ok = float(safe_endpoint["decrease"]) > 0.0
        if not assimilation_fraction_ok or not persistent_descent_ok or not persistent_positive_ok:
            return FunctionalShieldDeployment(
                accepted=False, factor=0.0, drift=0.0, new_loss=old_loss,
                selected_after=(None if not transfer_batches else candidate_before.get(transfer_batches[0].candidate_id)),
                candidate_before=candidate_before, candidate_after=dict(candidate_before),
                candidate_decreases=tuple(0.0 for _ in transfer_batches),
                current_ordinary_best=max(float(counterfactual["decrease"]), 0.0),
                current_required=progress_fraction * max(float(counterfactual["decrease"]), 0.0),
                current_certified_decrease=0.0, selected_certified_decrease=0.0,
                solve=None,
                obstruction=(
                    "persistent_assimilation_fraction_obstruction"
                    if not assimilation_fraction_ok
                    else (
                        "persistent_assimilation_descent_obstruction"
                        if not persistent_descent_ok
                        else "persistent_assimilation_nonpositive_progress_obstruction"
                    )
                ),
                shield_update_norm=0.0,
                selected_candidate_id=(None if not transfer_batches else transfer_batches[0].candidate_id),
                proposal=plan.proposal, counterfactual_accepted=True,
                counterfactual_decrease=float(counterfactual["decrease"]),
                exact_progress_ratio=None, safe_base_accepted=False,
                safe_base_decrease=float(safe_endpoint["decrease"]),
                safe_base_drift=float(safe_endpoint["drift"]),
                safe_base_radius=float(safe_endpoint["safe_radius"]),
                safe_base_step_length=float(safe_endpoint["realised_step_length"]),
                retention_budget=plan.retention_budget,
                reference_retention_charge=plan.reference_charge,
                requested_charge_fraction=plan.requested_charge_fraction,
                selected_path_fraction=plan.selected_path_fraction,
                realised_path_fraction=realised_path_fraction,
                persistent_base_progress_ratio=persistent_progress_ratio,
                persistent_descent_lower_bound=descent_lower_bound,
                projected_counterfactual_alignment_error=plan.projected_counterfactual_alignment_error,
                ordinary_counterfactual_alignment_error=plan.ordinary_counterfactual_alignment_error,
                projection_idempotence_error=plan.projection_idempotence_error,
                compatible_gradient_fraction=plan.compatibility_fraction,
                ordinary_step_size=plan.ordinary_step_size,
                step_size_smoothness_product=plan.step_size_smoothness_product,
                scalar_comparator_certified=plan.scalar_comparator_certified,
                analytic_persistent_progress_ratio_lower_bound=analytic_progress_ratio_lower_bound,
                certified_persistent_progress_ratio_lower_bound=certified_progress_ratio_lower_bound,
                shield_residual_progress_fraction=(
                    None if persistent_progress_ratio is None
                    else max(1.0 - persistent_progress_ratio, 0.0)
                ),
            )
        shield_residual_fraction = (
            None
            if persistent_progress_ratio is None
            else max(1.0 - persistent_progress_ratio, 0.0)
        )

        safe_predictive_state = safe_endpoint.get("predictive_state")
        if safe_predictive_state is None:
            raise RuntimeError("Accepted safe endpoint is missing its predictive-state transaction")
        self._restore_joint_predictive_state(safe_predictive_state)
        self.model.functional_shield.restore(original_shield)
        selected = transfer_batches[0] if transfer_batches else None
        contraction = float(shield_cfg.get("selected_contraction", 1.0))
        protected_blocks = [
            RestorationBlock(features, logits, f"record_{record.record_id}")
            for record, features, logits in zip(records, record_features, record_pre_logits)
        ]
        safeguard_blocks: list[RestorationBlock] = []
        selected_blocks: list[RestorationBlock] = []
        for batch, features, pre_logits in zip(transfer_batches, candidate_features, candidate_pre_logits):
            if selected is not None and batch.candidate_id == selected.candidate_id:
                target = (1.0 - contraction) * pre_logits + contraction * batch.target_logits
                selected_blocks.append(RestorationBlock(features, target, f"selected_{batch.candidate_id}"))
            else:
                safeguard_blocks.append(RestorationBlock(features, pre_logits, f"safeguard_{batch.candidate_id}"))

        feature_match_tolerance = float(shield_cfg.get("feature_match_tolerance", 1e-8))
        if self.certificate_mode == "strict":
            certified_address_error = shield_cfg.get("feature_distance_error_bound")
            if certified_address_error is not None:
                feature_match_tolerance = max(feature_match_tolerance, float(certified_address_error))
        residual_tolerance = float(shield_cfg.get("residual_tolerance", endpoint_tolerance))
        if self.certificate_mode == "strict":
            executable_endpoint_tolerance = residual_tolerance + float(
                shield_cfg.get("feature_distance_error_bound") or 0.0
            ) + 2.0 * float(
                self.cfg["afm"].get("numerics", {}).get("endpoint_error_bound") or 0.0
            )
        else:
            executable_endpoint_tolerance = max(
                residual_tolerance,
                float(self.cfg["afm"].get("numerics", {}).get("arithmetic_error_bound", 0.0)),
            )

        restoration = replace_with_endpoint_emulation(
            shield=self.model.functional_shield,
            base_logits_from_features=self.model.base_logits_from_backbone,
            current_features=current_features,
            counterfactual_current_logits=counterfactual["endpoint_logits"],
            protected_blocks=protected_blocks,
            safeguard_blocks=safeguard_blocks,
            selected_blocks=selected_blocks,
            guard_features=self.functional_shield_guard_features,
            support_multiplier=float(shield_cfg.get("support_multiplier", 4.0)),
            feature_match_tolerance=feature_match_tolerance,
            duplicate_tolerance=float(shield_cfg.get("duplicate_tolerance", 1e-10)),
            target_tolerance=float(shield_cfg.get("target_tolerance", endpoint_tolerance)),
            residual_tolerance=residual_tolerance,
            executable_endpoint_tolerance=executable_endpoint_tolerance,
            coefficient_norm_limit=float(shield_cfg.get("coefficient_norm_limit", float("inf"))),
        )
        if not restoration.available:
            self.vectoriser.assign(old_vector.to(self.device))
            self.model.functional_shield.restore(original_shield)
            return FunctionalShieldDeployment(
                accepted=False,
                factor=0.0,
                drift=0.0,
                new_loss=old_loss,
                selected_after=(None if selected is None else candidate_before.get(selected.candidate_id)),
                candidate_before=candidate_before,
                candidate_after=dict(candidate_before),
                candidate_decreases=tuple(0.0 for _ in transfer_batches),
                current_ordinary_best=float(counterfactual["decrease"]),
                current_required=progress_fraction * float(counterfactual["decrease"]),
                current_certified_decrease=0.0,
                selected_certified_decrease=0.0,
                solve=restoration.solve,
                obstruction=restoration.obstruction or "exact_counterfactual_restoration_obstruction",
                shield_update_norm=0.0,
                selected_candidate_id=(None if selected is None else selected.candidate_id),
                proposal=safe_endpoint["proposal"],
                counterfactual_accepted=True,
                counterfactual_decrease=float(counterfactual["decrease"]),
                exact_progress_ratio=0.0,
                maximum_endpoint_error=float(restoration.maximum_endpoint_error),
                safe_base_accepted=True,
                safe_base_decrease=float(safe_endpoint["decrease"]),
                safe_base_drift=float(safe_endpoint["drift"]),
                safe_base_radius=float(safe_endpoint["safe_radius"]),
                safe_base_step_length=float(safe_endpoint["realised_step_length"]),
                retention_budget=plan.retention_budget,
                reference_retention_charge=plan.reference_charge,
                requested_charge_fraction=plan.requested_charge_fraction,
                selected_path_fraction=plan.selected_path_fraction,
                realised_path_fraction=realised_path_fraction,
                persistent_base_progress_ratio=persistent_progress_ratio,
                persistent_descent_lower_bound=descent_lower_bound,
                projected_counterfactual_alignment_error=plan.projected_counterfactual_alignment_error,
                ordinary_counterfactual_alignment_error=plan.ordinary_counterfactual_alignment_error,
                projection_idempotence_error=plan.projection_idempotence_error,
                compatible_gradient_fraction=plan.compatibility_fraction,
                ordinary_step_size=plan.ordinary_step_size,
                step_size_smoothness_product=plan.step_size_smoothness_product,
                scalar_comparator_certified=plan.scalar_comparator_certified,
                analytic_persistent_progress_ratio_lower_bound=analytic_progress_ratio_lower_bound,
                certified_persistent_progress_ratio_lower_bound=certified_progress_ratio_lower_bound,
                shield_residual_progress_fraction=shield_residual_fraction,
            )

        self.model.eval()
        candidate_after: dict[int, float] = {}
        with torch.no_grad():
            current_logits = self.model.forward_from_backbone(current_features)
            new_loss = float(self._current_loss(current_logits, labels).item())
            after_behaviour: dict[int, torch.Tensor] = {}
            for record, features in zip(records, record_features):
                logits_after = self.model.forward_from_backbone(features)
                after_behaviour[record.record_id] = behaviour_from_logits(
                    logits_after, self.behaviour_spec
                ).detach()
            for batch, features in zip(transfer_batches, candidate_features):
                logits_after = self.model.forward_from_backbone(features)
                behaviour = behaviour_from_logits(logits_after, self.behaviour_spec)
                objective = 0.5 * (behaviour - batch.targets).square().sum(dim=1).mean()
                candidate_after[batch.candidate_id] = float(objective.item())

        drift = self._stacked_behaviour_drift(before_behaviour, after_behaviour)
        current_decrease = old_loss - new_loss
        counterfactual_decrease = float(counterfactual["decrease"])
        ratio = current_decrease / counterfactual_decrease if counterfactual_decrease > 0.0 else None
        candidate_decreases = tuple(
            candidate_before[batch.candidate_id] - candidate_after[batch.candidate_id]
            for batch in transfer_batches
        )
        selected_decrease = (
            0.0 if selected is None else candidate_before[selected.candidate_id] - candidate_after[selected.candidate_id]
        )
        strict_error = 0.0
        if self.certificate_mode == "strict":
            strict_error = float(shield_cfg.get("feature_distance_error_bound", 0.0)) + 2.0 * float(
                self.cfg["afm"].get("numerics", {}).get("endpoint_error_bound") or 0.0
            )
        current_required = progress_fraction * counterfactual_decrease
        current_ok = current_decrease + endpoint_tolerance >= current_required + strict_error
        exact_endpoint_ok = abs(new_loss - float(counterfactual["new_loss"])) <= endpoint_tolerance + strict_error
        retention_ok = drift <= endpoint_tolerance + strict_error
        candidate_ok = all(value >= -endpoint_tolerance - strict_error for value in candidate_decreases)
        selected_ok = selected is None or selected_decrease > endpoint_tolerance + strict_error
        shield_update_norm = 0.0
        if self.model.functional_shield.coefficients.numel():
            shield_update_norm = float(torch.linalg.vector_norm(self.model.functional_shield.coefficients).item())
        nonzero = float(safe_endpoint["realised_step_length"]) > nonzero_tolerance or shield_update_norm > nonzero_tolerance
        if current_ok and exact_endpoint_ok and retention_ok and candidate_ok and selected_ok and nonzero:
            return FunctionalShieldDeployment(
                accepted=True,
                factor=float(safe_endpoint["factor"]),
                drift=drift,
                new_loss=new_loss,
                selected_after=(None if selected is None else candidate_after[selected.candidate_id]),
                candidate_before=candidate_before,
                candidate_after=candidate_after,
                candidate_decreases=candidate_decreases,
                current_ordinary_best=counterfactual_decrease,
                current_required=current_required,
                current_certified_decrease=current_decrease - strict_error,
                selected_certified_decrease=selected_decrease - strict_error,
                solve=restoration.solve,
                obstruction=None,
                shield_update_norm=shield_update_norm,
                selected_candidate_id=(None if selected is None else selected.candidate_id),
                proposal=safe_endpoint["proposal"],
                counterfactual_accepted=True,
                counterfactual_decrease=counterfactual_decrease,
                exact_progress_ratio=ratio,
                maximum_endpoint_error=float(restoration.maximum_endpoint_error),
                safe_base_accepted=True,
                safe_base_decrease=float(safe_endpoint["decrease"]),
                safe_base_drift=float(safe_endpoint["drift"]),
                safe_base_radius=float(safe_endpoint["safe_radius"]),
                safe_base_step_length=float(safe_endpoint["realised_step_length"]),
                retention_budget=plan.retention_budget,
                reference_retention_charge=plan.reference_charge,
                requested_charge_fraction=plan.requested_charge_fraction,
                selected_path_fraction=plan.selected_path_fraction,
                realised_path_fraction=realised_path_fraction,
                persistent_base_progress_ratio=persistent_progress_ratio,
                persistent_descent_lower_bound=descent_lower_bound,
                projected_counterfactual_alignment_error=plan.projected_counterfactual_alignment_error,
                ordinary_counterfactual_alignment_error=plan.ordinary_counterfactual_alignment_error,
                projection_idempotence_error=plan.projection_idempotence_error,
                compatible_gradient_fraction=plan.compatibility_fraction,
                ordinary_step_size=plan.ordinary_step_size,
                step_size_smoothness_product=plan.step_size_smoothness_product,
                scalar_comparator_certified=plan.scalar_comparator_certified,
                analytic_persistent_progress_ratio_lower_bound=analytic_progress_ratio_lower_bound,
                certified_persistent_progress_ratio_lower_bound=certified_progress_ratio_lower_bound,
                shield_residual_progress_fraction=shield_residual_fraction,
            )

        self.vectoriser.assign(old_vector.to(self.device))
        self.model.functional_shield.restore(original_shield)
        if not exact_endpoint_ok:
            obstruction = "exact_counterfactual_current_endpoint_obstruction"
        elif not current_ok:
            obstruction = "exact_counterfactual_progress_obstruction"
        elif not retention_ok:
            obstruction = "exact_counterfactual_retention_obstruction"
        elif not candidate_ok:
            obstruction = "exact_counterfactual_candidate_obstruction"
        elif not selected_ok:
            obstruction = "exact_counterfactual_selected_progress_obstruction"
        else:
            obstruction = "exact_counterfactual_zero_update"
        return FunctionalShieldDeployment(
            accepted=False,
            factor=0.0,
            drift=0.0,
            new_loss=old_loss,
            selected_after=(None if selected is None else candidate_before.get(selected.candidate_id)),
            candidate_before=candidate_before,
            candidate_after=dict(candidate_before),
            candidate_decreases=tuple(0.0 for _ in transfer_batches),
            current_ordinary_best=counterfactual_decrease,
            current_required=current_required,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            solve=restoration.solve,
            obstruction=obstruction,
            shield_update_norm=0.0,
            selected_candidate_id=(None if selected is None else selected.candidate_id),
            proposal=safe_endpoint["proposal"],
            counterfactual_accepted=True,
            counterfactual_decrease=counterfactual_decrease,
            exact_progress_ratio=ratio,
            maximum_endpoint_error=float(restoration.maximum_endpoint_error),
            safe_base_accepted=True,
            safe_base_decrease=float(safe_endpoint["decrease"]),
            safe_base_drift=float(safe_endpoint["drift"]),
            safe_base_radius=float(safe_endpoint["safe_radius"]),
            safe_base_step_length=float(safe_endpoint["realised_step_length"]),
            retention_budget=plan.retention_budget,
            reference_retention_charge=plan.reference_charge,
            requested_charge_fraction=plan.requested_charge_fraction,
            selected_path_fraction=plan.selected_path_fraction,
            realised_path_fraction=realised_path_fraction,
            persistent_base_progress_ratio=persistent_progress_ratio,
            persistent_descent_lower_bound=descent_lower_bound,
            projected_counterfactual_alignment_error=plan.projected_counterfactual_alignment_error,
            ordinary_counterfactual_alignment_error=plan.ordinary_counterfactual_alignment_error,
            projection_idempotence_error=plan.projection_idempotence_error,
            compatible_gradient_fraction=plan.compatibility_fraction,
            ordinary_step_size=plan.ordinary_step_size,
            step_size_smoothness_product=plan.step_size_smoothness_product,
            scalar_comparator_certified=plan.scalar_comparator_certified,
            analytic_persistent_progress_ratio_lower_bound=analytic_progress_ratio_lower_bound,
            certified_persistent_progress_ratio_lower_bound=certified_progress_ratio_lower_bound,
            shield_residual_progress_fraction=shield_residual_fraction,
        )

    def _safe_update(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        transfer_batches: list[CandidateTransferBatch] | None = None,
    ) -> dict[str, float | int | bool]:
        """Perform one ordinary or constructively shielded safe-transfer update.

        A frozen candidate is never copied into the deployed parameter vector.
        In v0.11.0 strong mode, every round with an active protected record or
        certified candidate computes both the genuine same-state
        ``afm_no_protection`` comparator and the counterfactual-normalized metaplastic persistent endpoint. Persistent parameters take the largest certified projected comparator fraction; a bounded
        compact-cardinal residual emulates the comparator on the current batch,
        restores protected and unselected-candidate logits, and applies the
        selected target.
        Inconsistent constraints, unavailable counterfactuals, capacity failure,
        address-resolution failure, or uncertified numerics produce a typed
        obstruction and atomic zero update.  No nonzero projected safe-radius
        fallback is permitted in strong mode.
        """

        assert self.vectoriser is not None and self.controller is not None and self.policy_family is not None
        records = self._active_records()
        transfer_batches = list(transfer_batches or [])
        transfer_batch = transfer_batches[0] if transfer_batches else None
        transfer_candidate = None if transfer_batch is None else self.candidates.get(transfer_batch.slot)

        # Renewal is a predictable parameterisation-equivalence move based only
        # on the previous round. Structural availability and the controller's
        # private policy draw are fixed before the current gradients are used.
        renewal_trial = self._prepare_renewal(images)
        trial_slot = None if renewal_trial is None else renewal_trial.slot
        active_slots = set(self.model.adapter_pool.state().active)
        allowed_mask = self.vectoriser.gradient_mask_for_adapter_activity(active_slots, trial_slot=trial_slot)
        selected, bases, policy_probabilities = self._policy_bases(records, allowed_mask)
        basis = bases[selected]
        allowed_coordinate_count = int(torch.count_nonzero(allowed_mask).item())
        protected_basis_rank = int(basis.shape[1]) if basis.ndim == 2 else 0
        feasible_subspace_dimension = max(allowed_coordinate_count - protected_basis_rank, 0)

        self._set_afm_train_mode()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        loss = self._current_loss(logits, labels)
        loss.backward()
        gradient = self.vectoriser.flatten_grads()
        effective_gradient = gradient * allowed_mask
        current_q = project_to_allowed_free_subspace(gradient, basis, allowed_mask)
        current_qnorm = float(torch.linalg.vector_norm(current_q).item())

        renewal_trial_qnorm = 0.0
        renewal_useful = False
        if renewal_trial is not None:
            trial_mask = self.vectoriser.mask_for_adapter_slot(renewal_trial.slot)
            renewal_trial_qnorm = float(torch.linalg.vector_norm(current_q * trial_mask).item())
            renewal_useful = renewal_trial_qnorm >= float(self.cfg["afm"]["renewal"]["useful_gradient_threshold"])

        matrices = [record.sketch for record in records]
        grad_norm_sq = float(effective_gradient.square().sum().item())
        realised_weights: list[float] = []
        chis: list[float] = []
        for record in records:
            Bg = record.sketch.to(self.device) @ effective_gradient
            chi = min(1.0, float(Bg.square().sum().item()) / (1.0 + grad_norm_sq))
            chis.append(chi)
            realised_weights.append(1.0 + chi)

        evaluations: list[tuple[float, float, float, float]] = []
        zeta = float(self.cfg["afm"]["metaplastic"]["zeta"])
        for policy_basis in bases:
            residual = spectral_residual(matrices, realised_weights, policy_basis, allowed_mask=allowed_mask)
            blocked = blocked_gradient_fraction(effective_gradient, policy_basis)
            cost = residual + zeta * blocked
            bounded = cost / self.controller.loss_bound
            if bounded > 1.0 + 1e-5:
                raise RuntimeError(
                    f"Frontier loss {cost} exceeded declared bound {self.controller.loss_bound}; "
                    "tighten record admission or increase the predeclared bound"
                )
            evaluations.append((residual, blocked, cost, min(max(bounded, 0.0), 1.0)))
        self.controller.update([item[3] for item in evaluations])

        rates = [2.0 ** (-(k + 1)) for k in range(int(self.cfg["afm"]["metaplastic"]["timescales"]))]
        for record, chi in zip(records, chis):
            for k, rate in enumerate(rates):
                record.traces[k] = (1.0 - rate) * record.traces[k] + rate * chi

        selected_residual = evaluations[selected][0] if evaluations else 0.0
        delta_sum = sum(w * record.fd_delta for w, record in zip(realised_weights, records))
        Lbar, Hbar, LJ, cap, theorem_certified = self._certificate_values(bool(records))
        theta = self.vectoriser.flatten(detach=True)
        anchor_term = 0.0
        for weight, record in zip(realised_weights, records):
            distance = torch.linalg.vector_norm(theta.cpu() - record.anchor).item()
            anchor_term += weight * LJ * LJ * distance * distance
        d_anc = math.sqrt(max(anchor_term, 0.0))
        total_energy = 0.0
        for weight, record in zip(realised_weights, records):
            masked_sketch = record.sketch.to(self.device) * allowed_mask.unsqueeze(0)
            total_energy += weight * float(masked_sketch.square().sum().item())
        if records:
            arithmetic_E, orth_error = self._numerical_leakage(basis, total_energy)
        else:
            arithmetic_E, orth_error = 0.0, 0.0
        E = math.sqrt(max(selected_residual + delta_sum, 0.0)) + d_anc + arithmetic_E

        budget = self._behaviour_budget(bool(records))
        schedule = self.cfg["afm"]["safe_update"]
        transfer_cfg = self.cfg["afm"].get("transfer", {})
        lr = float(schedule["learning_rate"])
        transfer_bound_certified = not bool(transfer_batches)
        transfer_bound_available = True
        if transfer_batches and self.certificate_mode == "strict" and not self.functional_shield_enabled:
            certified_transfer_L = transfer_cfg.get("smoothness_bound")
            certified_transfer_provenance = transfer_cfg.get("smoothness_provenance")
            if (
                certified_transfer_L is None
                or not bool(transfer_cfg.get("smoothness_certified", False))
                or certified_transfer_provenance is None
                or float(certified_transfer_L) <= 0.0
                or not math.isfinite(float(certified_transfer_L))
            ):
                transfer_L = float("inf")
                transfer_bound_available = False
                transfer_bound_certified = False
            else:
                transfer_L = float(certified_transfer_L)
                transfer_bound_certified = True
        else:
            transfer_L = float(
                transfer_cfg.get("empirical_smoothness", Lbar if math.isfinite(Lbar) else 100.0)
            )
            if transfer_batches:
                if self.certificate_mode == "strict" and self.functional_shield_enabled:
                    shield_cfg = self.cfg["afm"].get("functional_shield", {})
                    transfer_bound_certified = bool(
                        shield_cfg.get("feature_distance_error_certified", False)
                        and shield_cfg.get("feature_distance_error_bound") is not None
                        and shield_cfg.get("feature_distance_error_provenance") is not None
                    )
                else:
                    transfer_bound_certified = False
        step_theorem_certified = theorem_certified and transfer_bound_certified
        ordinary_eta = lr
        if math.isfinite(Lbar):
            ordinary_eta = min(ordinary_eta, 1.0 / max(Lbar, 1e-12))
        if not math.isfinite(E) or not math.isfinite(Hbar):
            cap = 0.0

        ordinary_proposal = make_safe_step(
            gradient=gradient,
            basis=basis,
            learning_rate=ordinary_eta,
            E=E,
            H=Hbar,
            budget=budget,
            cap=cap,
            allowed_mask=allowed_mask,
        )

        transfer_before: float | None = None
        transfer_before_all: dict[int, float] = {}
        transfer_gradients: list[torch.Tensor] = []
        priority = None  # legacy log compatibility; v0.11.0 joint mode uses exact restoration by default.
        joint_result = None
        joint_proposal: SafeStep | None = None
        candidate_safe_proposal: SafeStep | None = None
        transfer_source: str | None = None
        transfer_actual_slope = 0.0
        transfer_progress_floor = 0.0
        transfer_progress_qualified = False
        transfer_unprojected_cosine: float | None = None
        selected_candidate_index: int | None = None
        selected_candidate_decrease = 0.0
        joint_current_required = 0.0
        joint_current_certified = 0.0
        candidate_safe_current_certified = 0.0
        joint_candidate_decreases: tuple[float, ...] = tuple()
        joint_external_obstruction: str | None = None

        if transfer_batches and not self.functional_shield_enabled:
            # The first batch is the predictable service target; every remaining
            # batch is an immutable safeguard objective.  All candidate
            # objectives are differentiated at the same deployed pre-step state.
            transfer_batch = transfer_batches[0]
            transfer_candidate = self.candidates.get(transfer_batch.slot)
            transfer_source = transfer_batch.source
            for index, batch in enumerate(transfer_batches):
                candidate = self.candidates.get(batch.slot)
                if candidate is None or not candidate.certified:
                    continue
                self._set_afm_train_mode()
                self.model.zero_grad(set_to_none=True)
                outputs = current_behaviour(self.model, batch.images, self.behaviour_spec)
                objective = 0.5 * (outputs - batch.targets).square().sum(dim=1).mean()
                transfer_before_all[batch.candidate_id] = float(objective.item())
                objective.backward()
                transfer_gradients.append(self.vectoriser.flatten_grads())
                if batch.candidate_id == transfer_batch.candidate_id:
                    selected_candidate_index = len(transfer_gradients) - 1
                    transfer_before = float(objective.item())

            if transfer_candidate is not None and selected_candidate_index is not None:
                transfer_candidate.transfer_attempts += 1
                transfer_candidate.transfer_service_rounds += 1
                transfer_candidate.transfer_last_service_step = int(self.step)
                self.transfer_attempts += 1
                selected_gradient = transfer_gradients[selected_candidate_index]
                current_full_norm = float(torch.linalg.vector_norm(gradient).item())
                transfer_full_norm = float(torch.linalg.vector_norm(selected_gradient).item())
                if current_full_norm > 0.0 and transfer_full_norm > 0.0:
                    transfer_unprojected_cosine = float(
                        torch.dot(gradient, selected_gradient).item()
                        / (current_full_norm * transfer_full_norm)
                    )
                progress_fraction = float(transfer_cfg.get("min_progress_fraction", 0.25))
                current_L_joint = float(Lbar)
                if not math.isfinite(current_L_joint) or current_L_joint <= 0.0:
                    if self.certificate_mode == "strict":
                        transfer_bound_available = False
                        joint_external_obstruction = "missing_certified_current_smoothness"
                    else:
                        current_L_joint = float(
                            self.cfg["afm"]["certificates"].get("empirical_loss_smoothness", 40.0)
                        )
                if not transfer_bound_available:
                    if joint_external_obstruction is None:
                        joint_external_obstruction = "missing_certified_transfer_smoothness"
                    transfer_candidate.transfer_obstructions += 1
                    self.transfer_obstructions += 1
                    self.transfer_incompatible_obstructions += 1
                else:
                    joint_result = joint_progress_protected_step(
                        current_gradient=gradient,
                        candidate_gradients=transfer_gradients,
                        selected_index=selected_candidate_index,
                        protected_basis=basis,
                        allowed_mask=allowed_mask,
                        current_smoothness=current_L_joint,
                        candidate_smoothness=[transfer_L for _ in transfer_gradients],
                        radius=float(ordinary_proposal.radius),
                        current_fraction=progress_fraction,
                        max_iterations=int(transfer_cfg.get("joint_solver_max_iterations", 1024)),
                        tolerance=float(transfer_cfg.get("joint_solver_tolerance", 1e-8)),
                    )
                if joint_result is not None:
                    joint_current_required = float(joint_result.current_required)
                    joint_current_certified = float(joint_result.current_certified_decrease)
                    candidate_safe_current_certified = float(joint_result.current_safe_best)
                    joint_candidate_decreases = tuple(joint_result.candidate_certified_decreases)
                    selected_candidate_decrease = float(joint_result.selected_certified_decrease)
                    transfer_progress_floor = joint_current_required
                    transfer_progress_qualified = bool(joint_result.mode == "joint_transfer" and joint_result.available)
                    if transfer_progress_qualified:
                        transfer_candidate.transfer_common_descent_steps += 1
                        transfer_candidate.transfer_priority_feasible_steps += 1
                        self.transfer_common_descent_steps += 1
                        self.transfer_priority_feasible_steps += 1
                        displacement = joint_result.displacement
                        displacement_norm = float(torch.linalg.vector_norm(displacement).item())
                        joint_proposal = SafeStep(
                            radius=float(ordinary_proposal.radius),
                            step_length=displacement_norm,
                            projected_gradient_norm=displacement_norm,
                            delta=-displacement,
                        )
                        if displacement_norm > 0.0:
                            transfer_actual_slope = selected_candidate_decrease / displacement_norm
                    else:
                        transfer_candidate.transfer_obstructions += 1
                        self.transfer_obstructions += 1
                        self.transfer_incompatible_obstructions += 1

                    safe_displacement = joint_result.candidate_safe_displacement
                    safe_norm = float(torch.linalg.vector_norm(safe_displacement).item())
                    if safe_norm > 0.0 and joint_result.current_safe_best + 1e-12 >= joint_result.current_required:
                        candidate_safe_proposal = SafeStep(
                            radius=float(ordinary_proposal.radius),
                            step_length=safe_norm,
                            projected_gradient_norm=safe_norm,
                            delta=-safe_displacement,
                        )
        old_vector = theta
        old_loss = float(loss.item())
        before_behaviour = self._record_current_outputs(records)
        nonzero_tol = float(schedule.get("nonzero_step_tolerance", 1e-12))
        max_backtracks = int(schedule.get("max_backtracks", 12))
        tolerance = float(schedule.get("check_tolerance", 1e-6))
        transfer_tolerance = float(transfer_cfg.get("endpoint_tolerance", tolerance))
        endpoint_error_bound = (
            float(self.cfg["afm"].get("numerics", {}).get("endpoint_error_bound") or 0.0)
            if self.certificate_mode == "strict"
            else 0.0
        )
        endpoint_two_sided_margin = 2.0 * endpoint_error_bound

        shield_deployment: FunctionalShieldDeployment | None = None
        shield_update_norm = 0.0
        accepted = False
        accepted_joint = False
        factor = 0.0
        drift = 0.0
        new_loss = old_loss
        transfer_after = transfer_before
        final_proposal = ordinary_proposal
        final_current_slope = current_qnorm
        strong_restoration = bool(
            self.exact_counterfactual_restoration_enabled
            and self.functional_shield_enabled
            and (records or transfer_batches)
        )
        exact_progress_ratio: float | None = None
        counterfactual_decrease = 0.0
        maximum_restoration_endpoint_error = 0.0
        safe_base_decrease = 0.0
        safe_base_drift = 0.0
        safe_base_radius = 0.0
        safe_base_step_length = 0.0
        reference_retention_charge = 0.0
        requested_charge_fraction = float(schedule.get("counterfactual_charge_fraction", 1.0))
        selected_path_fraction = 0.0
        realised_path_fraction = 0.0
        persistent_base_progress_ratio: float | None = None
        persistent_lower_bound = 0.0
        projected_counterfactual_alignment_error = 0.0
        ordinary_counterfactual_alignment_error = 0.0
        projection_idempotence_error = 0.0
        compatible_gradient_fraction = 0.0
        ordinary_step_size = 0.0
        step_size_smoothness_product = 0.0
        scalar_comparator_certified = False
        analytic_persistent_progress_ratio_lower_bound = 0.0
        certified_persistent_progress_ratio_lower_bound: float | None = None
        shield_residual_progress_fraction: float | None = None

        if strong_restoration:
            self.exact_restoration_attempts += 1
            self.functional_shield_attempts += 1
            restoration_cfg = self.cfg["afm"].get("functional_shield", {})
            restoration_numerics_available = True
            if self.certificate_mode == "strict":
                restoration_numerics_available = bool(
                    restoration_cfg.get("feature_distance_error_certified", False)
                    and restoration_cfg.get("feature_distance_error_bound") is not None
                    and restoration_cfg.get("feature_distance_error_provenance") is not None
                    and self.cfg["afm"].get("numerics", {}).get("endpoint_error_certified", False)
                )
            if transfer_batches:
                transfer_batch = transfer_batches[0]
                transfer_candidate = self.candidates.get(transfer_batch.slot)
                transfer_source = transfer_batch.source
                self.transfer_attempts += 1
                if transfer_candidate is not None:
                    transfer_candidate.transfer_attempts += 1
                    transfer_candidate.transfer_service_rounds += 1
                    transfer_candidate.transfer_last_service_step = int(self.step)
            if not restoration_numerics_available:
                joint_external_obstruction = "missing_certified_exact_restoration_numerics"
                self.exact_restoration_obstructions += 1
                self.functional_shield_obstructions += 1
                self.functional_shield_numerical_obstructions += 1
                if transfer_candidate is not None:
                    transfer_candidate.transfer_obstructions += 1
                    self.transfer_obstructions += 1
            elif transfer_batches and (transfer_candidate is None or not transfer_candidate.certified):
                joint_external_obstruction = "exact_restoration_missing_selected_candidate"
                self.exact_restoration_obstructions += 1
                self.functional_shield_obstructions += 1
                self.transfer_obstructions += 1
            else:
                shield_deployment = self._attempt_exact_counterfactual_restoration(
                    images=images,
                    labels=labels,
                    records=records,
                    transfer_batches=transfer_batches,
                    gradient=gradient,
                    allowed_mask=allowed_mask,
                    old_vector=old_vector,
                    old_loss=old_loss,
                    old_logits=logits.detach(),
                    before_behaviour=before_behaviour,
                    protected_basis=basis,
                    leakage_bound=E,
                    curvature_bound=Hbar,
                    retention_cap=cap,
                    loss_smoothness=Lbar,
                    charge_fraction=requested_charge_fraction,
                    endpoint_tolerance=transfer_tolerance,
                    max_backtracks=max_backtracks,
                    nonzero_tolerance=nonzero_tol,
                )
                transfer_before_all = dict(shield_deployment.candidate_before)
                if transfer_batch is not None:
                    transfer_before = transfer_before_all.get(transfer_batch.candidate_id)
                    transfer_after = shield_deployment.selected_after
                joint_current_required = shield_deployment.current_required
                joint_current_certified = shield_deployment.current_certified_decrease
                candidate_safe_current_certified = shield_deployment.current_certified_decrease
                joint_candidate_decreases = shield_deployment.candidate_decreases
                selected_candidate_decrease = shield_deployment.selected_certified_decrease
                transfer_progress_floor = joint_current_required
                transfer_progress_qualified = bool(transfer_batches and shield_deployment.accepted)
                accepted = shield_deployment.accepted
                accepted_joint = bool(transfer_batches and shield_deployment.accepted)
                factor = shield_deployment.factor
                drift = shield_deployment.drift
                new_loss = shield_deployment.new_loss
                shield_update_norm = shield_deployment.shield_update_norm
                exact_progress_ratio = shield_deployment.exact_progress_ratio
                counterfactual_decrease = shield_deployment.counterfactual_decrease
                maximum_restoration_endpoint_error = shield_deployment.maximum_endpoint_error
                safe_base_decrease = shield_deployment.safe_base_decrease
                safe_base_drift = shield_deployment.safe_base_drift
                safe_base_radius = shield_deployment.safe_base_radius
                safe_base_step_length = shield_deployment.safe_base_step_length
                budget = shield_deployment.retention_budget
                reference_retention_charge = shield_deployment.reference_retention_charge
                requested_charge_fraction = shield_deployment.requested_charge_fraction
                selected_path_fraction = shield_deployment.selected_path_fraction
                realised_path_fraction = shield_deployment.realised_path_fraction
                persistent_base_progress_ratio = shield_deployment.persistent_base_progress_ratio
                persistent_lower_bound = shield_deployment.persistent_descent_lower_bound
                projected_counterfactual_alignment_error = (
                    shield_deployment.projected_counterfactual_alignment_error
                )
                ordinary_counterfactual_alignment_error = (
                    shield_deployment.ordinary_counterfactual_alignment_error
                )
                projection_idempotence_error = shield_deployment.projection_idempotence_error
                compatible_gradient_fraction = shield_deployment.compatible_gradient_fraction
                ordinary_step_size = shield_deployment.ordinary_step_size
                step_size_smoothness_product = shield_deployment.step_size_smoothness_product
                scalar_comparator_certified = shield_deployment.scalar_comparator_certified
                analytic_persistent_progress_ratio_lower_bound = (
                    shield_deployment.analytic_persistent_progress_ratio_lower_bound
                )
                certified_persistent_progress_ratio_lower_bound = (
                    shield_deployment.certified_persistent_progress_ratio_lower_bound
                )
                shield_residual_progress_fraction = (
                    shield_deployment.shield_residual_progress_fraction
                )
                if shield_deployment.proposal is not None:
                    final_proposal = shield_deployment.proposal
                final_current_slope = (
                    joint_current_certified / max(factor * final_proposal.step_length, 1e-30)
                    if accepted else current_qnorm
                )
                if accepted:
                    self.exact_restoration_accepted += 1
                    self.functional_shield_accepted += 1
                    if transfer_candidate is not None:
                        transfer_candidate.transfer_common_descent_steps += 1
                        transfer_candidate.transfer_priority_feasible_steps += 1
                        transfer_candidate.transfer_accepted_steps += 1
                        self.transfer_common_descent_steps += 1
                        self.transfer_priority_feasible_steps += 1
                        self.transfer_accepted_steps += 1
                else:
                    self.exact_restoration_obstructions += 1
                    self.functional_shield_obstructions += 1
                    joint_external_obstruction = shield_deployment.obstruction
                    if transfer_candidate is not None:
                        transfer_candidate.transfer_obstructions += 1
                        self.transfer_obstructions += 1
                    if joint_external_obstruction == "functional_constraint_inconsistency":
                        self.functional_shield_inconsistencies += 1
                    else:
                        self.functional_shield_numerical_obstructions += 1

        if not strong_restoration and transfer_batches and self.functional_shield_enabled:
            transfer_batch = transfer_batches[0]
            transfer_candidate = self.candidates.get(transfer_batch.slot)
            transfer_source = transfer_batch.source
            # A scheduled certified-candidate service counts as an attempt even when
            # strict numerical provenance, current smoothness, or candidate state
            # forces immediate abstention. This keeps event logs and summary counters
            # on the same predictable service clock.
            self.transfer_attempts += 1
            self.functional_shield_attempts += 1
            if transfer_candidate is not None:
                transfer_candidate.transfer_attempts += 1
                transfer_candidate.transfer_service_rounds += 1
                transfer_candidate.transfer_last_service_step = int(self.step)
            if self.certificate_mode == "strict" and not transfer_bound_certified:
                joint_external_obstruction = "missing_certified_functional_shield_numerics"
                self.transfer_obstructions += 1
                self.functional_shield_obstructions += 1
                self.functional_shield_numerical_obstructions += 1
            elif transfer_candidate is None or not transfer_candidate.certified:
                joint_external_obstruction = "functional_shield_missing_selected_candidate"
                self.transfer_obstructions += 1
                self.functional_shield_obstructions += 1
            elif not math.isfinite(float(Lbar)) or float(Lbar) <= 0.0:
                joint_external_obstruction = "missing_certified_current_smoothness"
                transfer_candidate.transfer_obstructions += 1
                self.transfer_obstructions += 1
                self.functional_shield_obstructions += 1
            else:
                shield_deployment = self._attempt_functional_shield_transfer(
                    images=images,
                    labels=labels,
                    records=records,
                    transfer_batches=transfer_batches,
                    old_vector=old_vector,
                    old_loss=old_loss,
                    ordinary_proposal=ordinary_proposal,
                    projected_current_gradient=project_to_allowed_free_subspace(
                        gradient, basis, allowed_mask
                    ),
                    current_smoothness=float(Lbar),
                    before_behaviour=before_behaviour,
                    behaviour_budget=budget,
                    endpoint_tolerance=transfer_tolerance,
                    max_backtracks=max_backtracks,
                    nonzero_tolerance=nonzero_tol,
                )
                transfer_before_all = dict(shield_deployment.candidate_before)
                transfer_before = transfer_before_all.get(transfer_batch.candidate_id)
                transfer_after = shield_deployment.selected_after
                joint_current_required = shield_deployment.current_required
                joint_current_certified = shield_deployment.current_certified_decrease
                candidate_safe_current_certified = shield_deployment.current_certified_decrease
                joint_candidate_decreases = shield_deployment.candidate_decreases
                selected_candidate_decrease = shield_deployment.selected_certified_decrease
                transfer_progress_floor = joint_current_required
                transfer_progress_qualified = shield_deployment.accepted
                accepted = shield_deployment.accepted
                accepted_joint = shield_deployment.accepted
                factor = shield_deployment.factor
                drift = shield_deployment.drift
                new_loss = shield_deployment.new_loss
                shield_update_norm = shield_deployment.shield_update_norm
                final_current_slope = (
                    joint_current_certified / max(factor * ordinary_proposal.step_length, 1e-30)
                    if accepted else current_qnorm
                )
                if accepted:
                    transfer_candidate.transfer_common_descent_steps += 1
                    transfer_candidate.transfer_priority_feasible_steps += 1
                    transfer_candidate.transfer_accepted_steps += 1
                    self.transfer_common_descent_steps += 1
                    self.transfer_priority_feasible_steps += 1
                    self.transfer_accepted_steps += 1
                    self.functional_shield_accepted += 1
                else:
                    transfer_candidate.transfer_obstructions += 1
                    self.transfer_obstructions += 1
                    self.functional_shield_obstructions += 1
                    joint_external_obstruction = shield_deployment.obstruction
                    if joint_external_obstruction == "functional_constraint_inconsistency":
                        self.functional_shield_inconsistencies += 1
                    else:
                        self.functional_shield_numerical_obstructions += 1

        def attempt(
            proposal: SafeStep,
            current_slope: float,
            require_transfer: bool,
            transfer_slope: float = 0.0,
        ) -> tuple[bool, float, float, float, float | None]:
            if proposal.step_length <= nonzero_tol:
                return False, 0.0, 0.0, old_loss, transfer_before
            local_factor = 1.0
            local_drift = 0.0
            local_new_loss = old_loss
            local_transfer = transfer_before
            for _ in range(max_backtracks + 1):
                delta = local_factor * proposal.delta
                self.vectoriser.assign(old_vector.to(self.device) + delta)
                self._set_afm_train_mode()
                local_transfer_all: dict[int, float] = {}
                with torch.no_grad():
                    candidate_logits = self.model(images)
                    local_new_loss = float(self._current_loss(candidate_logits, labels).item())
                    for batch in transfer_batches:
                        outputs = current_behaviour(self.model, batch.images, self.behaviour_spec)
                        local_transfer_all[batch.candidate_id] = float(
                            (0.5 * (outputs - batch.targets).square().sum(dim=1).mean()).item()
                        )
                    if transfer_batch is not None:
                        local_transfer = local_transfer_all.get(transfer_batch.candidate_id, transfer_before)
                after_behaviour = self._record_current_outputs(records)
                local_drift = self._stacked_behaviour_drift(before_behaviour, after_behaviour)
                effective_s = local_factor * proposal.step_length
                retention_ok = local_drift <= budget + tolerance
                if transfer_batches and joint_result is not None:
                    # The current-progress requirement is an absolute certified
                    # decrease, not a raw-gradient slope scaled by a backtracked
                    # step.  A smaller factor is accepted only if it still meets
                    # the same predeclared fraction of ordinary safe progress.
                    current_rhs = old_loss - joint_current_required
                else:
                    current_rhs = old_loss - 0.5 * effective_s * current_slope
                current_ok = (
                    local_new_loss <= current_rhs - endpoint_two_sided_margin
                    if self.certificate_mode == "strict"
                    else local_new_loss <= current_rhs + tolerance
                )
                candidate_ok = True
                for candidate_id, before_value in transfer_before_all.items():
                    after_value = local_transfer_all.get(candidate_id)
                    if (
                        after_value is None
                        or after_value > before_value - endpoint_two_sided_margin
                    ):
                        candidate_ok = False
                        break
                transfer_ok = True
                if require_transfer:
                    assert transfer_before is not None and local_transfer is not None
                    transfer_rhs = transfer_before - selected_candidate_decrease
                    transfer_ok = (
                        local_transfer < transfer_before
                        and (
                            local_transfer <= transfer_rhs - endpoint_two_sided_margin
                            if self.certificate_mode == "strict"
                            else local_transfer <= transfer_rhs + transfer_tolerance
                        )
                    )
                if retention_ok and current_ok and candidate_ok and transfer_ok and effective_s > nonzero_tol:
                    return True, local_factor, local_drift, local_new_loss, local_transfer
                local_factor *= float(schedule.get("backtrack_factor", 0.5))
            self.vectoriser.assign(old_vector.to(self.device))
            return False, 0.0, 0.0, old_loss, transfer_before

        if not strong_restoration and joint_proposal is not None and transfer_progress_qualified:
            accepted, factor, drift, new_loss, transfer_after = attempt(
                joint_proposal,
                current_slope=(joint_current_certified / max(joint_proposal.step_length, 1e-30)),
                require_transfer=True,
                transfer_slope=transfer_actual_slope,
            )
            if accepted:
                accepted_joint = True
                final_proposal = joint_proposal
                final_current_slope = joint_current_certified / max(joint_proposal.step_length, 1e-30)
                assert transfer_candidate is not None
                transfer_candidate.transfer_accepted_steps += 1
                self.transfer_accepted_steps += 1
            else:
                assert transfer_candidate is not None
                transfer_candidate.transfer_obstructions += 1
                self.transfer_obstructions += 1
                self.transfer_endpoint_obstructions += 1

        if not strong_restoration and not accepted and transfer_batches:
            # Once a candidate is certified there is no ordinary damaging
            # fallback.  A current-loss update is permitted only if the joint
            # oracle certified nonincrease of every frozen candidate objective.
            if candidate_safe_proposal is not None and candidate_safe_proposal is not joint_proposal:
                accepted, factor, drift, new_loss, transfer_after = attempt(
                    candidate_safe_proposal,
                    current_slope=(candidate_safe_current_certified / max(candidate_safe_proposal.step_length, 1e-30)),
                    require_transfer=False,
                )
                final_proposal = candidate_safe_proposal
                final_current_slope = candidate_safe_current_certified / max(candidate_safe_proposal.step_length, 1e-30)
            else:
                self.transfer_fallback_steps += 1
        elif not strong_restoration and not accepted:
            accepted, factor, drift, new_loss, _ = attempt(
                ordinary_proposal,
                current_slope=current_qnorm,
                require_transfer=False,
            )
            final_proposal = ordinary_proposal
            final_current_slope = current_qnorm

        if not accepted:
            self.vectoriser.assign(old_vector.to(self.device))
            if max(ordinary_proposal.step_length, 0.0 if joint_proposal is None else joint_proposal.step_length) > nonzero_tol:
                self.summary.rejected_steps += 1
            else:
                self.summary.zero_steps += 1
            factor = 0.0
            drift = 0.0
            new_loss = old_loss
            transfer_after = transfer_before
        else:
            self.summary.optimizer_steps += 1
            self.summary.accepted_steps += 1
            self.summary.nonzero_accepted_steps += 1

        renewal_activated = False
        if renewal_trial is not None:
            slot_mask = self.vectoriser.mask_for_adapter_slot(renewal_trial.slot)
            trial_delta = (factor * final_proposal.delta) * slot_mask
            trial_motion = float(torch.linalg.vector_norm(trial_delta).item())
            trial_has_motion = bool(torch.count_nonzero(trial_delta).item())
            if accepted and trial_has_motion:
                self.model.adapter_pool.mark_active(renewal_trial.slot)
                self.summary.renewals_activated += 1
                renewal_activated = True
            else:
                self._rollback_renewal(renewal_trial)
            self.logger.log(
                "renewal_trial",
                step=self.step,
                slot=renewal_trial.slot,
                zero_change=renewal_trial.zero_change,
                projected_gradient_norm=final_proposal.projected_gradient_norm,
                renewal_projected_gradient_norm=renewal_trial_qnorm,
                trial_motion=trial_motion,
                trial_has_motion=trial_has_motion,
                useful=renewal_useful,
                activated=renewal_activated,
                accepted=accepted,
                renewal_alpha=renewal_trial.alpha,
            )

        violation = max(drift - budget, 0.0)
        self.summary.max_empirical_drift_violation = max(self.summary.max_empirical_drift_violation, violation)
        if records:
            for record in records:
                record.cumulative_budget += budget
            current_outputs = self._record_current_outputs(records)
            for record in records:
                active_interval_drift = math.sqrt(
                    max(float((current_outputs[record.record_id].cpu() - record.activation_outputs).square().sum(dim=1).mean().item()), 0.0)
                )
                snapshot_drift = math.sqrt(
                    max(float((current_outputs[record.record_id].cpu() - record.anchor_outputs).square().sum(dim=1).mean().item()), 0.0)
                )
                record.max_anchor_drift = max(record.max_anchor_drift, active_interval_drift)
                snapshot_bound = record.activation_gap + record.cumulative_budget
                snapshot_violation = max(snapshot_drift - snapshot_bound, 0.0)
                self.summary.max_snapshot_drift_violation = max(
                    self.summary.max_snapshot_drift_violation, snapshot_violation
                )
                self.logger.log(
                    "record_retention",
                    step=self.step,
                    record_id=record.record_id,
                    anchor_drift=active_interval_drift,
                    activation_gap=record.activation_gap,
                    snapshot_drift=snapshot_drift,
                    cumulative_budget=record.cumulative_budget,
                    snapshot_bound=snapshot_bound,
                    within_cumulative_budget=active_interval_drift <= record.cumulative_budget + tolerance,
                    within_snapshot_bound=snapshot_drift <= snapshot_bound + tolerance,
                )
            self.protection_round += 1

        realised_step_length = factor * final_proposal.step_length
        functional_nonzero = accepted and (
            realised_step_length > nonzero_tol or shield_update_norm > nonzero_tol
        )
        if functional_nonzero and not step_theorem_certified:
            self.all_nonzero_steps_certified = False
        self.last_step_diagnostics = {
            "loss": old_loss,
            "realised_step_length": realised_step_length,
            "projected_gradient_norm": final_proposal.projected_gradient_norm,
            "safe_radius": final_proposal.radius,
            "functional_shield_update_norm": shield_update_norm,
        }
        self.afm_round += 1

        self.logger.log(
            "afm_step",
            step=self.step,
            afm_round=self.afm_round - 1,
            protection_round=self.protection_round - int(bool(records)),
            selected_policy=selected,
            policy_probabilities=policy_probabilities,
            selected_alpha=self.policy_family[selected].alpha,
            selected_rank=self.policy_family[selected].rank,
            all_policy_frontier_costs=[item[2] for item in evaluations],
            all_policy_bounded_losses=[item[3] for item in evaluations],
            old_loss=old_loss,
            new_loss=new_loss,
            accepted=accepted,
            nonzero_accepted=functional_nonzero,
            backtrack_factor=factor,
            projected_gradient_norm=final_proposal.projected_gradient_norm,
            current_projected_gradient_norm=current_qnorm,
            current_directional_slope=final_current_slope,
            safe_radius=final_proposal.radius,
            proposed_step_length=final_proposal.step_length,
            realised_step_length=realised_step_length,
            behaviour_budget=budget,
            empirical_behaviour_drift=drift,
            epsilon=E,
            spectral_residual=selected_residual,
            fd_delta=delta_sum,
            anchor_drift=d_anc,
            arithmetic_leakage=arithmetic_E,
            projector_orthogonality_error=orth_error,
            blocked_fraction=evaluations[selected][1] if evaluations else 0.0,
            active_records=len(records),
            allowed_coordinate_count=allowed_coordinate_count,
            protected_basis_rank=protected_basis_rank,
            feasible_subspace_dimension=feasible_subspace_dimension,
            exact_zero_dim_transfer_obstruction=bool(
                transfer_batch is not None
                and len(records) > 0
                and feasible_subspace_dimension == 0
                and shield_deployment is None
                and (priority is None or not priority.available)
            ),
            renewal_slot=None if renewal_trial is None else renewal_trial.slot,
            renewal_activated=renewal_activated,
            transfer_candidate_id=None if transfer_batch is None else transfer_batch.candidate_id,
            transfer_slot=None if transfer_batch is None else transfer_batch.slot,
            transfer_source=transfer_source,
            transfer_attempted=transfer_batch is not None,
            transfer_common_descent=bool(
                accepted_joint if shield_deployment is not None
                else joint_result is not None and joint_result.mode == "joint_transfer" and joint_result.available
            ),
            transfer_priority_feasible=bool(
                accepted_joint if shield_deployment is not None
                else joint_result is not None and joint_result.mode == "joint_transfer" and joint_result.available
            ),
            transfer_progress_qualified=transfer_progress_qualified,
            transfer_progress_floor=transfer_progress_floor,
            transfer_joint_step=accepted_joint,
            transfer_mixture_weight=None,
            transfer_projected_gradient_norm=(
                None
                if selected_candidate_index is None
                else float(torch.linalg.vector_norm(project_to_allowed_free_subspace(
                    transfer_gradients[selected_candidate_index], basis, allowed_mask
                )).item())
            ),
            transfer_projected_cosine=None,
            transfer_priority_fraction=(None if transfer_batch is None else float(transfer_cfg.get("min_progress_fraction", 0.25))),
            transfer_compatibility=(
                None
                if transfer_batch is None
                else float(selected_candidate_decrease / max(joint_current_certified, 1e-30))
                if shield_deployment is not None
                else (
                    None
                    if joint_result is None or joint_result.current_safe_best <= 0.0
                    else float(joint_result.selected_certified_decrease / max(joint_result.current_safe_best, 1e-30))
                )
            ),
            transfer_priority_obstruction=(
                joint_external_obstruction if joint_result is None else joint_result.obstruction
            ),
            transfer_priority_current_slope=(
                None
                if transfer_batch is None or final_proposal.step_length <= 0.0
                else float(joint_current_certified / max(factor * final_proposal.step_length, 1e-30))
            ),
            joint_candidate_count=len(transfer_before_all),
            joint_current_ordinary_best=(
                shield_deployment.current_ordinary_best if shield_deployment is not None
                else None if joint_result is None else joint_result.current_ordinary_best
            ),
            joint_current_safe_best=(
                shield_deployment.current_certified_decrease if shield_deployment is not None
                else None if joint_result is None else joint_result.current_safe_best
            ),
            joint_current_required=(
                joint_current_required if transfer_batch is not None else None
            ),
            joint_current_certified_decrease=(
                joint_current_certified if transfer_batch is not None else None
            ),
            candidate_safe_current_certified_decrease=(
                candidate_safe_current_certified if transfer_batch is not None else None
            ),
            joint_selected_certified_decrease=(
                selected_candidate_decrease if transfer_batch is not None else None
            ),
            joint_candidate_certified_decreases=list(joint_candidate_decreases),
            joint_solver_iterations=(
                1 if shield_deployment is not None and shield_deployment.solve is not None
                else None if joint_result is None else joint_result.iterations
            ),
            joint_solver_max_constraint_violation=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.interpolation_residual
            ) if joint_result is None else joint_result.max_constraint_violation,
            joint_solver_converged=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.available
            ) if joint_result is None else joint_result.solver_converged,
            joint_solver_mode=(
                "joint_counterfactual_normalized_assimilation" if strong_restoration and shield_deployment is not None
                else "functional_shield" if shield_deployment is not None
                else None if joint_result is None else joint_result.mode
            ),
            functional_shield_enabled=self.functional_shield_enabled,
            functional_shield_attempted=bool(strong_restoration or (transfer_batch is not None and self.functional_shield_enabled)),
            exact_counterfactual_restoration_attempted=strong_restoration,
            exact_counterfactual_restoration_accepted=bool(strong_restoration and accepted),
            exact_counterfactual_decrease=(counterfactual_decrease if strong_restoration else None),
            exact_counterfactual_progress_ratio=(exact_progress_ratio if strong_restoration else None),
            exact_counterfactual_endpoint_error=(maximum_restoration_endpoint_error if strong_restoration else None),
            safe_base_decrease=(safe_base_decrease if strong_restoration else None),
            safe_base_drift=(safe_base_drift if strong_restoration else None),
            safe_base_radius=(safe_base_radius if strong_restoration else None),
            safe_base_step_length=(safe_base_step_length if strong_restoration else None),
            reference_retention_charge=(reference_retention_charge if strong_restoration else None),
            requested_counterfactual_charge_fraction=(requested_charge_fraction if strong_restoration else None),
            selected_counterfactual_path_fraction=(selected_path_fraction if strong_restoration else None),
            realised_counterfactual_path_fraction=(realised_path_fraction if strong_restoration else None),
            persistent_base_progress_ratio=(persistent_base_progress_ratio if strong_restoration else None),
            persistent_descent_lower_bound=(persistent_lower_bound if strong_restoration else None),
            projected_counterfactual_alignment_error=(
                projected_counterfactual_alignment_error if strong_restoration else None
            ),
            ordinary_counterfactual_alignment_error=(
                ordinary_counterfactual_alignment_error if strong_restoration else None
            ),
            projection_idempotence_error=(projection_idempotence_error if strong_restoration else None),
            compatible_gradient_fraction=(compatible_gradient_fraction if strong_restoration else None),
            ordinary_counterfactual_step_size=(ordinary_step_size if strong_restoration else None),
            counterfactual_step_size_smoothness_product=(
                step_size_smoothness_product if strong_restoration else None
            ),
            scalar_counterfactual_certified=(
                scalar_comparator_certified if strong_restoration else None
            ),
            analytic_persistent_progress_ratio_lower_bound=(
                analytic_persistent_progress_ratio_lower_bound if strong_restoration else None
            ),
            certified_persistent_progress_ratio_lower_bound=(
                certified_persistent_progress_ratio_lower_bound if strong_restoration else None
            ),
            shield_residual_progress_fraction=(
                shield_residual_progress_fraction if strong_restoration else None
            ),
            retention_budget_mode=(
                "counterfactual_normalized" if strong_restoration else str(schedule.get("budget_mode", "legacy_schedule"))
            ),
            persistent_base_mode=("counterfactual_normalized_metaplastic_endpoint" if strong_restoration else "ordinary"),
            functional_shield_update_norm=shield_update_norm,
            functional_shield_node_count=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.merged_node_count
            ),
            functional_shield_condition_number=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.condition_number
            ),
            functional_shield_minimum_eigenvalue=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.minimum_eigenvalue
            ),
            functional_shield_interpolation_residual=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.interpolation_residual
            ),
            functional_shield_minimum_support_radius=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.minimum_support_radius
            ),
            functional_shield_maximum_support_radius=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.maximum_support_radius
            ),
            functional_shield_guard_count=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.guard_count
            ),
            functional_shield_maximum_guard_leakage=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.maximum_guard_leakage
            ),
            functional_shield_minimum_address_separation=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.minimum_address_separation
            ),
            functional_shield_support_multiplier=(
                None if shield_deployment is None or shield_deployment.solve is None
                else shield_deployment.solve.support_multiplier
            ),
            functional_shield_obstruction=(
                joint_external_obstruction
                if shield_deployment is None
                else shield_deployment.obstruction
            ),
            transfer_unprojected_cosine=transfer_unprojected_cosine,
            transfer_directional_slope=transfer_actual_slope,
            transfer_loss_before=transfer_before,
            transfer_loss_after=transfer_after,
            certificate_mode=self.certificate_mode,
            theorem_certified=step_theorem_certified,
        )
        return {
            "loss": old_loss,
            "accepted": accepted,
            "drift": drift,
            "budget": budget,
            "qnorm": final_proposal.projected_gradient_norm,
            "realised_step_length": realised_step_length,
            "functional_shield_update_norm": shield_update_norm,
            "transfer_joint_step": accepted_joint,
        }

    def _bootstrap_step(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, torch.Tensor]:
        self.model.train()
        self.bootstrap_optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        pre_accuracy = accuracy(logits.detach(), labels)
        loss.backward()
        self.bootstrap_optimizer.step()
        self.summary.optimizer_steps += 1
        return float(loss.item()), pre_accuracy, logits.detach()

    def _persistent_state_dict(self) -> dict[str, Any]:
        candidates: dict[int, Any] = {}
        for slot, candidate in self.candidates.items():
            candidates[slot] = {
                "slot": candidate.slot,
                "created_step": candidate.created_step,
                "candidate_id": candidate.candidate_id,
                "initial_parameters": candidate.initial_parameters,
                "training_evidence": candidate.training_evidence,
                "training_count": candidate.training_count,
                "training_loss": candidate.training_loss,
                "training_accuracy": candidate.training_accuracy,
                "commit_budget_index": candidate.commit_budget_index,
                "commit_alpha": candidate.commit_alpha,
                "snapshot": candidate.snapshot,
                "snapshot_shield_state": candidate.snapshot_shield_state,
                "validation_count": candidate.validation_count,
                "validation_sum": candidate.validation_sum,
                "last_logged_validation_count": candidate.last_logged_validation_count,
                "certified": candidate.certified,
                "certified_step": candidate.certified_step,
                "certified_validation_count": candidate.certified_validation_count,
                "certified_validation_sum": candidate.certified_validation_sum,
                "certified_ucb": candidate.certified_ucb,
                "staleness_eprocess": (
                    None if candidate.staleness_eprocess is None else candidate.staleness_eprocess.state_dict()
                ),
                "staleness_budget_index": candidate.staleness_budget_index,
                "staleness_alpha": candidate.staleness_alpha,
                "staleness_observations": candidate.staleness_observations,
                "staleness_crossed_step": candidate.staleness_crossed_step,
                "evidence": candidate.evidence,
                "sketch": None if candidate.sketch is None else candidate.sketch.state_dict(),
                "anchor_output_chunks": candidate.anchor_output_chunks,
                "anchor_logit_chunks": candidate.anchor_logit_chunks,
                "transfer_evidence": candidate.transfer_evidence,
                "transfer_targets": candidate.transfer_targets,
                "transfer_logits": candidate.transfer_logits,
                "frozen_step": candidate.frozen_step,
                "transfer_attempts": candidate.transfer_attempts,
                "transfer_common_descent_steps": candidate.transfer_common_descent_steps,
                "transfer_priority_feasible_steps": candidate.transfer_priority_feasible_steps,
                "transfer_accepted_steps": candidate.transfer_accepted_steps,
                "transfer_obstructions": candidate.transfer_obstructions,
                "last_activation_gap": candidate.last_activation_gap,
                "transfer_initial_objective": candidate.transfer_initial_objective,
                "transfer_last_objective": candidate.transfer_last_objective,
                "transfer_progress_total": candidate.transfer_progress_total,
                "transfer_damage_total": candidate.transfer_damage_total,
                "transfer_ledger_updates": candidate.transfer_ledger_updates,
                "transfer_service_rounds": candidate.transfer_service_rounds,
                "transfer_last_service_step": candidate.transfer_last_service_step,
                "validation_signatures": candidate.validation_signatures,
                "last_validation_signature_block_id": candidate.last_validation_signature_block_id,
            }
        records: dict[int, Any] = {}
        for record_id, record in self.records.items():
            records[record_id] = {
                "record_id": record.record_id,
                "slot": record.slot,
                "created_step": record.created_step,
                "committed_step": record.committed_step,
                "anchor": record.anchor,
                "anchor_shield_state": record.anchor_shield_state,
                "sketch": record.sketch,
                "fd_delta": record.fd_delta,
                "evidence": record.evidence,
                "anchor_outputs": record.anchor_outputs,
                "activation_outputs": record.activation_outputs,
                "traces": record.traces,
                "challenger": record.challenger.state_dict(),
                "challenger_optimizer": record.challenger.optimizer.state_dict(),
                "eprocess": record.eprocess.state_dict(),
                "reopening_budget_index": record.reopening_budget_index,
                "reopening_alpha": record.reopening_alpha,
                "reopening_start_step": record.reopening_start_step,
                "signature_eprocess": record.signature_eprocess.state_dict(),
                "signature_budget_index": record.signature_budget_index,
                "signature_alpha": record.signature_alpha,
                "signature_start_step": record.signature_start_step,
                "signature_reference_radius": record.signature_reference_radius,
                "signature_blocks_seen": record.signature_blocks_seen,
                "signature_distance_sum": record.signature_distance_sum,
                "signature_vector_sum": record.signature_vector_sum,
                "last_signature_block_id": record.last_signature_block_id,
                "outcome_reopening_crossed": record.outcome_reopening_crossed,
                "outcome_crossing_step": record.outcome_crossing_step,
                "outcome_crossing_signature_count": record.outcome_crossing_signature_count,
                "outcome_crossing_log_wealth": record.outcome_crossing_log_wealth,
                "outcome_crossing_observation": record.outcome_crossing_observation,
                "last_seen_step": record.last_seen_step,
                "cumulative_budget": record.cumulative_budget,
                "max_anchor_drift": record.max_anchor_drift,
                "released_step": record.released_step,
                "release_reason": record.release_reason,
            }
        return {
            "step": self.step,
            "afm_round": self.afm_round,
            "protection_round": self.protection_round,
            "record_counter": self.record_counter,
            "candidate_counter": self.candidate_counter,
            "candidate_replacements": self.candidate_replacements,
            "candidate_pretest_replacements": self.candidate_pretest_replacements,
            "candidate_tests_started": self.candidate_tests_started,
            "candidate_training_runs": self.candidate_training_runs,
            "candidate_validation_rejections": self.candidate_validation_rejections,
            "route_splits": self.route_splits,
            "route_split_obstructions": self.route_split_obstructions,
            "transfer_attempts": self.transfer_attempts,
            "transfer_common_descent_steps": self.transfer_common_descent_steps,
            "transfer_priority_feasible_steps": self.transfer_priority_feasible_steps,
            "transfer_accepted_steps": self.transfer_accepted_steps,
            "transfer_fallback_steps": self.transfer_fallback_steps,
            "transfer_obstructions": self.transfer_obstructions,
            "transfer_incompatible_obstructions": self.transfer_incompatible_obstructions,
            "transfer_endpoint_obstructions": self.transfer_endpoint_obstructions,
            "functional_shield_attempts": self.functional_shield_attempts,
            "functional_shield_accepted": self.functional_shield_accepted,
            "functional_shield_obstructions": self.functional_shield_obstructions,
            "functional_shield_inconsistencies": self.functional_shield_inconsistencies,
            "functional_shield_numerical_obstructions": self.functional_shield_numerical_obstructions,
            "exact_restoration_attempts": self.exact_restoration_attempts,
            "exact_restoration_accepted": self.exact_restoration_accepted,
            "exact_restoration_obstructions": self.exact_restoration_obstructions,
            "functional_shield": self.model.functional_shield.snapshot(),
            "functional_shield_guard_features": self.functional_shield_guard_features.clone(),
            "candidate_certifications": self.candidate_certifications,
            "candidate_staleness_rejections": self.candidate_staleness_rejections,
            "signature_prefix_rows": self.signature_prefix_rows,
            "last_split_step_by_slot": dict(self._last_split_step_by_slot),
            "stall_count": self.stall_count,
            "last_step_diagnostics": self.last_step_diagnostics,
            "effective_total_steps": self.effective_total_steps,
            "effective_afm_horizon": self.effective_afm_horizon,
            "router": self.router.state_dict(),
            "context_signature": self.context_signature.state_dict(),
            "error_allocator": self.error_allocator.state_dict(),
            "controller": None if self.controller is None else self.controller.state_dict(),
            "candidates": candidates,
            "records": records,
        }

    def fit(self, loader: DataLoader) -> dict[str, Any]:
        max_steps = int(self.cfg["training"]["max_steps"])
        try:
            stream_steps = int(len(loader))
        except TypeError:
            stream_steps = max_steps
        self.effective_total_steps = min(max_steps, stream_steps)
        self.effective_afm_horizon = max(self.effective_total_steps - self.bootstrap_batches, 1)
        log_interval = int(self.cfg["training"].get("log_interval", 20))
        start = time.time()
        self.logger.log(
            "horizon_declared",
            configured_max_steps=max_steps,
            finite_stream_steps=stream_steps,
            effective_total_steps=self.effective_total_steps,
            effective_afm_horizon=self.effective_afm_horizon,
        )
        for images, labels, learner_rows in loader:
            if self.step >= max_steps:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.step < self.bootstrap_batches:
                if self.context_signature.calibrate_from_prefix:
                    self.signature_prefix_rows.extend(
                        self._bounded_evidence_row(row) for row in learner_rows
                    )
                loss_value, pre_accuracy, pre_logits = self._bootstrap_step(images, labels)
                self.instrumentation.log_predictions(
                    step=self.step, logits=pre_logits, labels=labels, metadata=learner_rows
                )
                self.summary.online_loss.update(loss_value, len(labels))
                self.summary.online_accuracy.update(pre_accuracy, len(labels))
                if self.step % log_interval == 0:
                    self.logger.log("bootstrap_step", step=self.step, loss=loss_value, accuracy=pre_accuracy)
                self.step += 1
                self.instrumentation.maybe_checkpoint(
                    completed_steps=self.step, model=self.model, config=self.cfg, summary=self.summary.as_dict()
                )
                continue

            self._initialise_afm_state()
            if self.protection_enabled:
                candidate_groups, record_groups, overflow_count, signatures, signature_block_ids = self._route_batch(
                    images, learner_rows, include_signatures=True
                )
            else:
                candidate_groups, record_groups, overflow_count = {}, {}, 0
                signatures = torch.empty((len(images), 0), dtype=torch.float64)
                signature_block_ids = torch.empty((len(images),), dtype=torch.long)
            reopening_predictions: list[tuple[ReopeningPredictionBatch, torch.Tensor]] = []
            for record_id, indices in sorted(record_groups.items(), key=lambda item: min(item[1])):
                prediction = self._record_reopening_predictions(
                    record_id,
                    self._index_rows(images, indices),
                    self._index_rows(signatures, indices),
                    self._index_rows(signature_block_ids, indices),
                )
                if prediction is not None:
                    reopening_predictions.append((prediction, self._index_rows(labels, indices)))
            with torch.no_grad():
                pre_logits = self.model(images)
            self.summary.online_accuracy.update(accuracy(pre_logits, labels), len(labels))
            self.summary.online_loss.update(float(self._current_loss(pre_logits, labels).item()), len(labels))
            self.instrumentation.log_predictions(
                step=self.step, logits=pre_logits, labels=labels, metadata=learner_rows
            )
            for (slot, token), indices in sorted(candidate_groups.items(), key=lambda item: min(item[1])):
                candidate = self.candidates.get(slot)
                if candidate is None or candidate.token != token:
                    continue
                subset_rows = [learner_rows[i] for i in indices]
                self._candidate_update_before_training(
                    slot,
                    self._index_rows(images, indices),
                    self._index_rows(labels, indices),
                    subset_rows,
                    self._index_rows(signatures, indices),
                    self._index_rows(signature_block_ids, indices),
                )
            self._commit_ready_certified_candidates()
            # Expired candidates are released before any further deployed
            # update, so an ordinary step can never damage a still-certified
            # objective that has merely reached its service horizon.
            self._reject_exhausted_certified_candidates()
            ledger_before = {
                candidate.candidate_id: value
                for candidate in self.candidates.values()
                if candidate.certified
                for value in [self._candidate_transfer_objective(candidate)]
                if value is not None
            }
            transfer_batches = self._eligible_transfer_batches()
            pre_update_vector = (
                None
                if not ledger_before or self.vectoriser is None
                else self.vectoriser.flatten(detach=True).cpu()
            )
            pre_update_shield = (
                None if not ledger_before else self.model.functional_shield.snapshot()
            )
            result = self._safe_update(images, labels, transfer_batches=transfer_batches)
            try:
                self._update_transfer_ledgers(ledger_before, enforce_nonincrease=True)
            except RuntimeError:
                if pre_update_vector is not None and self.vectoriser is not None:
                    self.vectoriser.assign(pre_update_vector.to(self.device))
                    if pre_update_shield is not None:
                        self.model.functional_shield.restore(pre_update_shield)
                    self.logger.log(
                        "candidate_progress_invariant_rollback",
                        step=self.step,
                        candidate_count=len(ledger_before),
                    )
                raise
            self._commit_ready_certified_candidates()
            self._reject_exhausted_certified_candidates()
            # Predictions were frozen before the current loss update; threshold
            # crossings and releases take effect only after that protected step.
            for prediction, routed_labels in reopening_predictions:
                self._apply_reopening_update(prediction, routed_labels)
            if self.step % log_interval == 0:
                self.logger.log(
                    "progress",
                    step=self.step,
                    overflow=overflow_count > 0,
                    overflow_items=overflow_count,
                    online_accuracy=self.summary.online_accuracy.mean,
                    online_loss=self.summary.online_loss.mean,
                    active_records=len(self._active_records()),
                    candidates=len(self.candidates),
                    elapsed_seconds=time.time() - start,
                    **result,
                )
            self.step += 1
            self.instrumentation.maybe_checkpoint(
                completed_steps=self.step, model=self.model, config=self.cfg, summary=self.summary.as_dict()
            )

        summary = self.summary.as_dict()
        summary.update(
            {
                "method": "afm" if self.protection_enabled else "afm_no_protection",
                "steps": self.step,
                "afm_rounds": self.afm_round,
                "protection_rounds": self.protection_round,
                "active_records": len(self._active_records()),
                "candidate_slots": len(self.candidates),
                "candidate_creations": self.candidate_counter,
                "candidate_replacements": self.candidate_replacements,
                "candidate_pretest_replacements": self.candidate_pretest_replacements,
                "candidate_tests_started": self.candidate_tests_started,
                "candidate_training_runs": self.candidate_training_runs,
                "candidate_validation_rejections": self.candidate_validation_rejections,
                "route_splits": self.route_splits,
                "route_split_obstructions": self.route_split_obstructions,
                "transfer_attempts": self.transfer_attempts,
                "transfer_common_descent_steps": self.transfer_common_descent_steps,
                "transfer_priority_feasible_steps": self.transfer_priority_feasible_steps,
                "transfer_accepted_steps": self.transfer_accepted_steps,
                "transfer_fallback_steps": self.transfer_fallback_steps,
                "transfer_obstructions": self.transfer_obstructions,
                "transfer_incompatible_obstructions": self.transfer_incompatible_obstructions,
                "transfer_endpoint_obstructions": self.transfer_endpoint_obstructions,
                "functional_shield_enabled": self.functional_shield_enabled,
                "functional_shield_attempts": self.functional_shield_attempts,
                "functional_shield_accepted": self.functional_shield_accepted,
                "functional_shield_obstructions": self.functional_shield_obstructions,
                "functional_shield_inconsistencies": self.functional_shield_inconsistencies,
                "functional_shield_numerical_obstructions": self.functional_shield_numerical_obstructions,
                "exact_restoration_attempts": self.exact_restoration_attempts,
                "exact_restoration_accepted": self.exact_restoration_accepted,
                "exact_restoration_obstructions": self.exact_restoration_obstructions,
                "functional_shield_nodes": self.model.functional_shield.node_count,
                "functional_shield_generation": int(self.model.functional_shield.generation.item()),
                "candidate_certifications": self.candidate_certifications,
                "candidate_staleness_rejections": self.candidate_staleness_rejections,
                "router_threshold": self.router.threshold,
                "signature_calibration_obstruction": bool(
                    self.context_signature.calibration
                    and self.context_signature.calibration.calibration_obstruction
                ),
                "signature_positive_routing_claim_available": bool(
                    self.context_signature.calibration
                    and self.context_signature.calibration.positive_routing_claim_available
                ),
                "effective_total_steps": self.effective_total_steps,
                "effective_afm_horizon": self.effective_afm_horizon,
                "elapsed_seconds": time.time() - start,
                "certificate_mode": self.certificate_mode,
                "theorem_certified_run": (
                    self.certificate_mode == "strict"
                    and self.summary.nonzero_accepted_steps > 0
                    and self.all_nonzero_steps_certified
                ),
                "total_delta": self.error_allocator.total_delta,
                "total_error_budget_allocated": self.error_allocator.total_allocated,
                "protection_enabled": self.protection_enabled,
                "diagnostic_class_balance_enabled": self.diagnostic_class_balance_enabled,
                "diagnostic_class_balance_after_bootstrap_only": self.diagnostic_class_balance_after_bootstrap_only,
                "diagnostic_class_weights": (
                    self.diagnostic_class_weights.detach().cpu().tolist()
                    if self.diagnostic_class_weights is not None else None
                ),
                **RunInstrumentation.resource_summary(self.device),
            }
        )
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.cfg,
                "summary": summary,
                "afm_state": self._persistent_state_dict(),
            },
            self.run_dir / "final.pt",
        )
        self.logger.log("finished", **summary)
        return summary


class SGDTrainer:
    def __init__(self, model: nn.Module, config: dict[str, Any], device: torch.device, run_dir: Path, matched: bool = False):
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.run_dir = ensure_dir(run_dir)
        self.logger = JSONLLogger(self.run_dir / "events.jsonl")
        self.summary = ExperimentSummary()
        self.matched = matched
        self.method = "matched_sgd" if matched else "sgd"
        self.instrumentation = RunInstrumentation(config, self.run_dir, self.method)
        self.bootstrap_batches = int(config["training"].get("bootstrap_batches", 0)) if matched else 0
        self.bootstrap_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["training"].get("bootstrap_lr", 1e-3)),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        self.optimizer: torch.optim.Optimizer | None = None
        self.vectoriser: ParameterVector | None = None

    def _initialise_matched_state(self) -> None:
        if self.vectoriser is not None:
            return
        assert isinstance(self.model, AFMConvNet)
        self.model.freeze_backbone()
        self.vectoriser = ParameterVector(self.model.trainable_named_parameters())

    def _matched_step(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, torch.Tensor]:
        self._initialise_matched_state()
        assert self.vectoriser is not None and isinstance(self.model, AFMConvNet)
        self.model.train()
        self.model.backbone.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        pre_acc = accuracy(logits.detach(), labels)
        loss.backward()
        gradient = self.vectoriser.flatten_grads()
        active_slots = set(self.model.adapter_pool.state().active)
        mask = self.vectoriser.gradient_mask_for_adapter_activity(active_slots)
        q = gradient * mask
        lr = float(self.cfg["training"].get("baseline_lr", 1e-3))
        cap = float(self.cfg["afm"]["safe_update"].get("trust_radius_cap", float("inf")))
        delta = -lr * q
        norm = float(torch.linalg.vector_norm(delta).item())
        if norm > cap > 0.0:
            delta *= cap / norm
        self.vectoriser.add_(delta)
        return float(loss.item()), pre_acc, logits.detach()

    def fit(self, loader: DataLoader) -> dict[str, Any]:
        max_steps = int(self.cfg["training"]["max_steps"])
        start = time.time()
        for step, (images, labels, _) in enumerate(loader):
            if step >= max_steps:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.matched and step < self.bootstrap_batches:
                self.model.train()
                self.bootstrap_optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                pre_acc = accuracy(logits.detach(), labels)
                loss.backward()
                self.bootstrap_optimizer.step()
                loss_value = float(loss.item())
                pre_logits = logits.detach()
            elif self.matched:
                loss_value, pre_acc, pre_logits = self._matched_step(images, labels)
            else:
                self.model.train()
                if self.optimizer is None:
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=float(self.cfg["training"].get("baseline_lr", 1e-3)),
                        weight_decay=float(self.cfg["training"].get("weight_decay", 0.0)),
                    )
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                pre_acc = accuracy(logits.detach(), labels)
                loss.backward()
                self.optimizer.step()
                loss_value = float(loss.item())
                pre_logits = logits.detach()
            self.instrumentation.log_predictions(step=step, logits=pre_logits, labels=labels, metadata=_)
            self.summary.optimizer_steps += 1
            self.summary.online_accuracy.update(pre_acc, len(labels))
            self.summary.online_loss.update(loss_value, len(labels))
            self.instrumentation.maybe_checkpoint(
                completed_steps=step + 1, model=self.model, config=self.cfg, summary=self.summary.as_dict()
            )
            if step % int(self.cfg["training"].get("log_interval", 20)) == 0:
                self.logger.log("matched_sgd_step" if self.matched else "sgd_step", step=step, loss=loss_value, accuracy=pre_acc)
        method = "matched_sgd" if self.matched else "sgd"
        summary = self.summary.as_dict()
        summary.update({
            "elapsed_seconds": time.time() - start,
            "method": method,
            **RunInstrumentation.resource_summary(self.device),
        })
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        torch.save({"model": self.model.state_dict(), "config": self.cfg, "summary": summary}, self.run_dir / "final.pt")
        return summary
