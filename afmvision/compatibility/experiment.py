from __future__ import annotations

import json
import math
import random
import time

import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from afmvision.afm.parameter_vector import ParameterVector
from afmvision.compatibility.data import (
    Batch,
    CharNextTokenDataset,
    IndexedCIFAR10,
    Reservoir,
    deterministic_order,
    make_causal_current_batch,
    stack_indices,
)
from afmvision.compatibility.geometry import (
    build_protected_geometry,
    functional_jacobian,
    make_controlled_targets,
)
from afmvision.compatibility.methods import (
    calibrate_comparator_to_decrease,
    endpoint_metrics_for_vector,
    method_proposal_delta,
    retention_frontier_grid,
    run_method,
    unrestricted_comparator,
)
from afmvision.compatibility.models import build_compatibility_model, describe_model
from afmvision.compatibility.shield import StaticAddressEncoder
from afmvision.utils.seed import seed_everything


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _safe_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing compatibility run: {run_dir}. "
            "Choose a new run directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)


def _archive_for_resume(path: Path) -> None:
    if not path.exists():
        return
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.pre_resume_{index}")
        if not candidate.exists():
            path.rename(candidate)
            return
        index += 1


def _parent_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that determine the trained pre-probe parent state.

    v1.4 deliberately changes only the causal probe/evaluation protocol.  Saved
    preprobe parents may therefore be reused when this signature is unchanged.
    """
    sweep = dict(config["compatibility_sweep"])
    training = dict(sweep["training"])
    causal = dict(sweep["causal"])
    return {
        "seed": int(config["seed"]),
        "deterministic": bool(sweep.get("deterministic", True)),
        "dataset": sweep["dataset"],
        "model": sweep["model"],
        "training": {
            k: training.get(k)
            for k in (
                "batch_size", "parent_lr", "weight_decay",
                "pretrain_steps", "stream_steps", "log_interval"
            )
        },
        "reservoir_capacity": causal.get("reservoir_capacity"),
    }


def _prepare_resume_run_dir(run_dir: Path, config: dict[str, Any]) -> None:
    checkpoint = run_dir / "preprobe_parent.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Cannot resume {run_dir}: missing preprobe_parent.pt. "
            "A normal fresh run is required for this job."
        )
    resolved = run_dir / "resolved_config.json"
    if not resolved.is_file():
        raise FileNotFoundError(f"Cannot resume {run_dir}: missing resolved_config.json")
    existing = json.loads(resolved.read_text(encoding="utf-8"))
    if _parent_signature(existing) != _parent_signature(config):
        raise RuntimeError(
            f"Cannot resume {run_dir}: parent-training configuration differs from saved run"
        )
    # Preserve prior diagnostics while ensuring the v1.4 analysis sees one clean
    # probe/frontier stream.  The trained preprobe parent is never overwritten.
    for name in (
        "events.jsonl", "compatibility_points.jsonl", "retention_frontier_points.jsonl",
        "retention_frontier_grid.jsonl", "afm_native_points.jsonl",
        "summary.json", "final_parent.pt", "resolved_config.json"
    ):
        _archive_for_resume(run_dir / name)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )


def _batch_accuracy(model: nn.Module, batch: Batch, device: torch.device) -> float:
    with torch.no_grad():
        logits = model(batch.inputs.to(device))
        labels = batch.labels.to(device)
        return float((logits.argmax(dim=1) == labels).float().mean().item())


def _train_parent_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    model.train()
    x = batch.inputs.to(device)
    y = batch.labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    acc = float((logits.detach().argmax(dim=1) == y).float().mean().item())
    loss.backward()
    optimizer.step()
    return float(loss.item()), acc, logits.detach()


def _select_disjoint(ids: list[int], forbidden: set[int], count: int) -> list[int]:
    out: list[int] = []
    for item in ids:
        if item in forbidden:
            continue
        out.append(int(item))
        if len(out) >= int(count):
            break
    return out


class CompatibilitySweepRunner:
    """Matched-state causal compatibility sweep.

    A single end-to-end parent network follows an ordinary supervised stream.
    At declared states the exact same pre-update model, protected bank, and
    current inputs are forked conceptually across all compatibility levels and
    learning methods.  No causal branch is committed to the parent trajectory.
    This gives matched interventions on compatibility rather than a correlation
    across unrelated examples or method-specific states.
    """

    def __init__(
        self,
        config: dict[str, Any],
        run_dir: str | Path,
        device: torch.device,
        *,
        resume_preprobe: bool = False,
    ) -> None:
        self.cfg = config
        self.sweep = dict(config["compatibility_sweep"])
        self.seed = int(config["seed"])
        self.device = device
        self.run_dir = Path(run_dir)
        self.resume_preprobe = bool(resume_preprobe)
        if self.resume_preprobe:
            _prepare_resume_run_dir(self.run_dir, config)
        else:
            _safe_run_dir(self.run_dir)
            (self.run_dir / "resolved_config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
            )
        seed_everything(self.seed, deterministic=bool(self.sweep.get("deterministic", True)))
        self.events = JsonlWriter(self.run_dir / "events.jsonl")
        self.points = JsonlWriter(self.run_dir / "compatibility_points.jsonl")
        self.frontier_points = JsonlWriter(self.run_dir / "retention_frontier_points.jsonl")
        self.frontier_grid = JsonlWriter(self.run_dir / "retention_frontier_grid.jsonl")
        self.afm_native_points = JsonlWriter(self.run_dir / "afm_native_points.jsonl")

        data_cfg = dict(self.sweep["dataset"])
        self.modality = str(data_cfg["modality"])
        kind = str(data_cfg["kind"])
        if kind == "cifar10":
            self.dataset = IndexedCIFAR10(
                data_cfg["root"], train=True, download=bool(data_cfg.get("download", False))
            )
            vocab_size = None
        elif kind == "char_text":
            self.dataset = CharNextTokenDataset(
                data_cfg["text_path"],
                context_length=int(self.sweep["model"].get("context_length", 64)),
                stride=int(data_cfg.get("stride", 1)),
            )
            vocab_size = self.dataset.vocab_size
        else:
            raise ValueError(f"unknown compatibility dataset kind: {kind}")
        self.model = build_compatibility_model(config, vocab_size=vocab_size).to(device)
        # Full end-to-end trainability is a hard contract for this experiment.
        frozen = [name for name, p in self.model.named_parameters() if not p.requires_grad]
        if frozen:
            raise RuntimeError(f"compatibility experiment forbids frozen parameters: {frozen}")
        self.vectoriser = ParameterVector(self.model.named_parameters())
        self.description = describe_model(
            self.model, self.modality, str(self.sweep["model"]["architecture"])
        )
        self.vocab_size = vocab_size
        first_x, _, _ = self.dataset[0]
        self.address_encoder = StaticAddressEncoder(
            modality=self.modality,
            input_shape=tuple(first_x.shape),
            address_dim=int(self.sweep["causal"].get("address_dim", 64)),
            seed=int(self.sweep["causal"].get("address_seed", 20260817)),
            vocab_size=vocab_size,
        )

    def _probe_state(
        self,
        *,
        state_index: int,
        parent_step: int,
        protected: Batch,
        novel: Batch,
        guards: Batch | None,
    ) -> int:
        causal = dict(self.sweep["causal"])
        methods = list(causal["methods"])
        requested = [float(x) for x in causal["requested_kappas"]]
        current = make_causal_current_batch(
            modality=self.modality,
            protected=protected,
            novel=novel,
            near_count=int(causal["near_protected_count"]),
            seed=self.seed * 1000003 + state_index,
            vocab_size=self.vocab_size,
        )
        current_inputs = current.inputs.to(self.device)
        current_labels = current.labels.to(self.device)
        protected_inputs = protected.inputs.to(self.device)
        protected_labels = protected.labels.to(self.device)
        guard_inputs = None if guards is None else guards.inputs.to(self.device)

        self.model.eval()
        pre_vector = self.vectoriser.flatten(detach=True)
        with torch.no_grad():
            protected_full_before = self.model(protected_inputs).detach().clone()
            replay_logits = protected_full_before.clone()
        protected_measure, Jp = functional_jacobian(self.model, protected_inputs)
        current_measure, Jc = functional_jacobian(self.model, current_inputs)
        geometry = build_protected_geometry(Jp, ridge=float(causal.get("geometry_ridge", 1e-7)))
        targets = make_controlled_targets(
            model=self.model,
            current_inputs=current_inputs,
            current_measurements=current_measure.detach(),
            current_jacobian=Jc.detach(),
            geometry=geometry,
            requested_kappas=requested,
            gradient_norm=float(causal.get("gradient_norm", 1.0)),
        )
        if float(targets[0].achievable_max - targets[0].achievable_min) < float(
            causal.get("minimum_achievable_span", 0.5)
        ):
            self.events.write(
                {
                    "event": "state_skipped_insufficient_compatibility_span",
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "achievable_min": targets[0].achievable_min,
                    "achievable_max": targets[0].achievable_max,
                }
            )
            self.vectoriser.assign(pre_vector)
            return 0

        point_count = 0
        comparator_decreases: list[float] = []
        state_payload = {
            "event": "causal_state",
            "state_index": state_index,
            "parent_step": parent_step,
            "protected_count": len(protected.inputs),
            "current_count": len(current.inputs),
            "guard_count": 0 if guards is None else len(guards.inputs),
            "achievable_min": targets[0].achievable_min,
            "achievable_max": targets[0].achievable_max,
        }
        self.events.write(state_payload)

        # Construct every same-state unrestricted comparator before running any
        # method.  A causal state is retained only when all requested levels have
        # a genuine positive no-protection decrease and the unrestricted decreases
        # satisfy the predeclared matching criterion.  This prevents partial states
        # from entering the analysis and turns a numerically unsuitable state into
        # a documented skip rather than a job-level failure.
        comparator_rows: list[tuple[Any, Any]] = []
        for target in targets:
            self.vectoriser.assign(pre_vector)
            try:
                comparator = unrestricted_comparator(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    current_inputs=current_inputs,
                    current_targets=target.teacher_measurements,
                    current_gradient=target.gradient,
                    initial_alpha=float(causal.get("comparator_lr", 1e-2)),
                    max_backtracks=int(causal.get("comparator_max_backtracks", 20)),
                    backtrack_factor=float(causal.get("comparator_backtrack_factor", 0.5)),
                )
            except RuntimeError as exc:
                self.events.write(
                    {
                        "event": "state_skipped_unrestricted_comparator",
                        "state_index": state_index,
                        "parent_step": parent_step,
                        "requested_kappa": target.requested_kappa,
                        "measured_kappa": target.measured_kappa,
                        "reason": str(exc),
                    }
                )
                self.vectoriser.assign(pre_vector)
                return 0
            comparator_rows.append((target, comparator))
            comparator_decreases.append(comparator.decrease)

        if comparator_decreases:
            raw_mean = sum(comparator_decreases) / len(comparator_decreases)
            raw_variance = sum((x - raw_mean) ** 2 for x in comparator_decreases) / len(comparator_decreases)
            raw_cv = math.sqrt(raw_variance) / raw_mean if raw_mean > 0 else float("inf")

            # v1.3 causal normalization: do not discard a state merely because
            # finite curvature makes equal-norm gradients yield different Delta_0
            # at a common trial step.  Instead, preserve each genuine same-state
            # unrestricted descent direction and calibrate its endpoint to one
            # common finite decrease.  The common target is chosen below the
            # smallest already-positive comparator decrease, so every requested
            # compatibility level is guaranteed to bracket that target between
            # alpha=0 and its native positive endpoint.
            match_fraction = float(causal.get("comparator_match_fraction", 0.5))
            match_rtol = float(causal.get("comparator_match_rtol", 2.0e-3))
            match_bisections = int(causal.get("comparator_match_bisections", 40))
            common_target = match_fraction * min(comparator_decreases)
            if not (0.0 < match_fraction < 1.0) or not (common_target > 0.0):
                raise RuntimeError(
                    f"invalid comparator matching target: fraction={match_fraction}, target={common_target}"
                )

            matched_rows: list[tuple[Any, Any]] = []
            try:
                for target, comparator in comparator_rows:
                    self.vectoriser.assign(pre_vector)
                    matched = calibrate_comparator_to_decrease(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        current_inputs=current_inputs,
                        current_targets=target.teacher_measurements,
                        base=comparator,
                        target_decrease=common_target,
                        relative_tolerance=match_rtol,
                        max_bisections=match_bisections,
                    )
                    matched_rows.append((target, matched))
            except RuntimeError as exc:
                self.events.write(
                    {
                        "event": "state_skipped_comparator_calibration",
                        "state_index": state_index,
                        "parent_step": parent_step,
                        "raw_cv_delta0": raw_cv,
                        "common_target_delta0": common_target,
                        "reason": str(exc),
                    }
                )
                self.vectoriser.assign(pre_vector)
                return 0

            comparator_rows = matched_rows
            comparator_decreases = [float(c.decrease) for _, c in comparator_rows]
            mean = sum(comparator_decreases) / len(comparator_decreases)
            variance = sum((x - mean) ** 2 for x in comparator_decreases) / len(comparator_decreases)
            cv = math.sqrt(variance) / mean if mean > 0 else float("inf")
            max_cv = float(causal.get("maximum_comparator_cv", 0.15))
            self.events.write(
                {
                    "event": "comparator_matching",
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "normalization": "matched_finite_delta0_bisection",
                    "raw_mean_delta0": raw_mean,
                    "raw_cv_delta0": raw_cv,
                    "common_target_delta0": common_target,
                    "match_fraction": match_fraction,
                    "match_rtol": match_rtol,
                    "mean_delta0": mean,
                    "cv_delta0": cv,
                    "max_allowed_cv": max_cv,
                    "pass": cv <= max_cv,
                }
            )
            if cv > max_cv:
                self.events.write(
                    {
                        "event": "state_skipped_comparator_mismatch",
                        "state_index": state_index,
                        "parent_step": parent_step,
                        "normalization": "matched_finite_delta0_bisection",
                        "raw_cv_delta0": raw_cv,
                        "common_target_delta0": common_target,
                        "mean_delta0": mean,
                        "cv_delta0": cv,
                        "max_allowed_cv": max_cv,
                    }
                )
                self.vectoriser.assign(pre_vector)
                return 0

        retention_betas = [float(x) for x in causal.get(
            "retention_budget_betas", [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
        )]
        retention_epsilon = float(causal.get("retention_numeric_epsilon", 1e-8))
        frontier_grid_points = int(causal.get("retention_frontier_grid_points", 33))
        retention_reference_rule = str(
            causal.get("retention_reference_rule", "max_unrestricted_drift_across_kappa")
        )
        if retention_reference_rule != "max_unrestricted_drift_across_kappa":
            raise RuntimeError(
                f"unsupported retention_reference_rule: {retention_reference_rule}"
            )

        # Evaluate all six matched unrestricted endpoints first.  The state-level
        # reference drift is then frozen before any method frontier is selected.
        # Thus every method and every compatibility level receives the same
        # absolute protected-function budget for a given beta.
        state_rows: list[tuple[Any, Any, float, float]] = []
        for target, comparator in comparator_rows:
            self.vectoriser.assign(pre_vector)
            _, unrestricted_drift, unrestricted_rms = endpoint_metrics_for_vector(
                model=self.model,
                vectoriser=self.vectoriser,
                vector_after=comparator.vector_after,
                current_inputs=current_inputs,
                current_targets=target.teacher_measurements,
                protected_inputs=protected_inputs,
                protected_logits_before=protected_full_before,
            )
            state_rows.append((target, comparator, unrestricted_drift, unrestricted_rms))

        retention_reference_drift = max(float(row[2]) for row in state_rows)
        self.events.write(
            {
                "event": "retention_reference_fixed",
                "state_index": state_index,
                "parent_step": parent_step,
                "rule": retention_reference_rule,
                "retention_reference_drift": retention_reference_drift,
                "unrestricted_drifts_by_requested_kappa": {
                    str(float(t.requested_kappa)): float(d)
                    for t, _, d, _ in state_rows
                },
            }
        )

        for target, comparator, unrestricted_drift, unrestricted_rms in state_rows:
            # Preserve one AFM-specific native transaction audit per kappa.  This
            # is separate from the method-neutral common-budget frontier and is
            # the correct source for the persistent-vs-finite AFM figure.
            self.vectoriser.assign(pre_vector)
            native_afm = run_method(
                method="afm",
                model=self.model,
                vectoriser=self.vectoriser,
                comparator=comparator,
                current_gradient=target.gradient,
                geometry=geometry,
                current_inputs=current_inputs,
                current_targets=target.teacher_measurements,
                protected_inputs=protected_inputs,
                protected_labels=protected_labels,
                protected_logits_before=protected_full_before,
                replay_stored_logits=replay_logits,
                guard_inputs=guard_inputs,
                retention_tolerance=float(causal.get("retention_tolerance", 0.005)),
                method_config=causal,
                address_encoder=self.address_encoder,
            )
            self.afm_native_points.write(
                {
                    "dataset": str(self.sweep["dataset"]["kind"]),
                    "modality": self.modality,
                    "architecture": str(self.sweep["model"]["architecture"]),
                    "seed": self.seed,
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "requested_kappa": target.requested_kappa,
                    "measured_kappa": target.measured_kappa,
                    "delta0": comparator.decrease,
                    "unrestricted_protected_drift": unrestricted_drift,
                    "retention_reference_drift": retention_reference_drift,
                    "retention_reference_rule": retention_reference_rule,
                    "persistent_decrease": native_afm.persistent_decrease,
                    "persistent_ratio": native_afm.persistent_ratio,
                    "retention_max_abs_drift": native_afm.protected_max_abs_drift,
                    "retention_rms_drift": native_afm.protected_rms_drift,
                    "retention_pass": native_afm.retention_pass,
                    "accepted": native_afm.accepted,
                    "update_norm": native_afm.update_norm,
                    "backtracking_steps": native_afm.backtracking_steps,
                    "afm_lambda_hat": native_afm.afm_lambda_hat,
                    "deployed_ratio": native_afm.deployed_ratio,
                    "finite_completion_available": native_afm.finite_completion_available,
                    "finite_endpoint_error": native_afm.finite_endpoint_error,
                    "finite_current_error": native_afm.finite_current_error,
                    "finite_protected_error": native_afm.finite_protected_error,
                    "obstruction": native_afm.obstruction,
                }
            )

            for method in methods:
                self.vectoriser.assign(pre_vector)
                proposal_delta = method_proposal_delta(
                    method=method,
                    model=self.model,
                    vectoriser=self.vectoriser,
                    comparator=comparator,
                    current_gradient=target.gradient,
                    geometry=geometry,
                    protected_inputs=protected_inputs,
                    protected_labels=protected_labels,
                    replay_stored_logits=replay_logits,
                    method_config=causal,
                )
                proposal_vector = comparator.vector_before + proposal_delta
                proposal_loss, proposal_drift, proposal_rms = endpoint_metrics_for_vector(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    vector_after=proposal_vector,
                    current_inputs=current_inputs,
                    current_targets=target.teacher_measurements,
                    protected_inputs=protected_inputs,
                    protected_logits_before=protected_full_before,
                )
                proposal_decrease = float(comparator.loss_before) - float(proposal_loss)
                proposal_ratio = (
                    proposal_decrease / float(comparator.decrease)
                    if comparator.decrease > 0 else float("nan")
                )
                payload = {
                    "dataset": str(self.sweep["dataset"]["kind"]),
                    "modality": self.modality,
                    "architecture": str(self.sweep["model"]["architecture"]),
                    "seed": self.seed,
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "method": method,
                    "requested_kappa": target.requested_kappa,
                    "measured_kappa": target.measured_kappa,
                    "achievable_min": target.achievable_min,
                    "achievable_max": target.achievable_max,
                    "target_clipped": target.clipped,
                    "gradient_norm": target.gradient_norm,
                    "comparator_alpha": comparator.alpha,
                    "delta0": comparator.decrease,
                    "comparator_loss_before": comparator.loss_before,
                    "comparator_loss_after": comparator.loss_after,
                    "unrestricted_protected_drift": unrestricted_drift,
                    "unrestricted_protected_rms_drift": unrestricted_rms,
                    "retention_reference_drift": retention_reference_drift,
                    "retention_reference_rule": retention_reference_rule,
                    "persistent_decrease": proposal_decrease,
                    "persistent_ratio": proposal_ratio,
                    "retention_max_abs_drift": proposal_drift,
                    "retention_rms_drift": proposal_rms,
                    "retention_pass": proposal_drift <= float(causal.get("retention_tolerance", 0.005)),
                    "accepted": proposal_decrease > 0.0,
                    "update_norm": float(torch.linalg.vector_norm(proposal_delta).item()),
                }
                self.points.write(payload)
                point_count += 1

                frontier, grid_rows, frontier_audit = retention_frontier_grid(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    comparator=comparator,
                    proposal_delta=proposal_delta,
                    current_inputs=current_inputs,
                    current_targets=target.teacher_measurements,
                    protected_inputs=protected_inputs,
                    protected_logits_before=protected_full_before,
                    retention_reference_drift=retention_reference_drift,
                    betas=retention_betas,
                    epsilon_num=retention_epsilon,
                    grid_points=frontier_grid_points,
                )

                # Save all actual nonlinear grid endpoints.  Future budget
                # summaries can therefore be recomputed on CPU without another
                # compatibility reprobe.
                for gp in grid_rows:
                    self.frontier_grid.write(
                        {
                            "dataset": str(self.sweep["dataset"]["kind"]),
                            "modality": self.modality,
                            "architecture": str(self.sweep["model"]["architecture"]),
                            "seed": self.seed,
                            "state_index": state_index,
                            "parent_step": parent_step,
                            "method": method,
                            "requested_kappa": target.requested_kappa,
                            "measured_kappa": target.measured_kappa,
                            "delta0": comparator.decrease,
                            "unrestricted_protected_drift": unrestricted_drift,
                            "retention_reference_drift": retention_reference_drift,
                            "retention_reference_rule": retention_reference_rule,
                            "scale": float(gp.scale),
                            "update_norm": float(gp.update_norm),
                            "current_loss_after": float(gp.current_loss_after),
                            "persistent_decrease": float(gp.persistent_decrease),
                            "persistent_ratio": float(gp.persistent_ratio),
                            "retention_max_abs_drift": float(gp.protected_max_abs_drift),
                            "retention_rms_drift": float(gp.protected_rms_drift),
                        }
                    )

                for fp in frontier:
                    frontier_payload = {
                        "dataset": str(self.sweep["dataset"]["kind"]),
                        "modality": self.modality,
                        "architecture": str(self.sweep["model"]["architecture"]),
                        "seed": self.seed,
                        "state_index": state_index,
                        "parent_step": parent_step,
                        "method": method,
                        "requested_kappa": target.requested_kappa,
                        "measured_kappa": target.measured_kappa,
                        "target_clipped": target.clipped,
                        "delta0": comparator.decrease,
                        "unrestricted_protected_drift": unrestricted_drift,
                        "retention_reference_drift": retention_reference_drift,
                        "retention_reference_rule": retention_reference_rule,
                        "retention_beta": float(fp.beta),
                        "retention_budget": float(fp.budget),
                        "retention_numeric_epsilon": retention_epsilon,
                        "frontier_scale": float(fp.scale),
                        "frontier_grid_points": frontier_grid_points,
                        "frontier_drift_monotone_on_grid": frontier_audit["drift_monotone_on_grid"],
                        "frontier_monotonic_drift_violations": frontier_audit["monotonic_drift_violations"],
                        "persistent_decrease": float(fp.persistent_decrease),
                        "persistent_ratio": float(fp.persistent_ratio),
                        "retention_max_abs_drift": float(fp.protected_max_abs_drift),
                        "retention_rms_drift": float(fp.protected_rms_drift),
                        "retention_pass": float(fp.protected_max_abs_drift) <= float(fp.budget) + 1e-12,
                        "accepted": float(fp.persistent_decrease) > 0.0,
                        "update_norm": float(fp.update_norm),
                    }
                    self.frontier_points.write(frontier_payload)

        self.vectoriser.assign(pre_vector)
        return point_count

    def run(self) -> dict[str, Any]:
        training = dict(self.sweep["training"])
        causal = dict(self.sweep["causal"])
        batch_size = int(training.get("batch_size", 64))
        order = deterministic_order(len(self.dataset), self.seed)
        if len(order) < batch_size * 4:
            raise RuntimeError("dataset is too small for configured compatibility sweep")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training.get("parent_lr", 1e-3)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        reservoir = Reservoir(int(causal.get("reservoir_capacity", 512)), self.seed + 991)
        cursor = 0
        start = time.time()
        parent_steps = 0
        pretrain_steps = int(training.get("pretrain_steps", 1000))
        stream_steps = int(training.get("stream_steps", 1000))
        total_required = pretrain_steps + stream_steps
        # Recycle the deterministic order by reshuffling with epoch-specific seeds.
        epoch = 0

        def next_batch() -> Batch:
            nonlocal cursor, order, epoch
            if cursor + batch_size > len(order):
                epoch += 1
                order = deterministic_order(len(self.dataset), self.seed + 1009 * epoch)
                cursor = 0
            ids = order[cursor : cursor + batch_size]
            cursor += batch_size
            return stack_indices(self.dataset, ids)

        if self.resume_preprobe:
            checkpoint_path = self.run_dir / "preprobe_parent.pt"
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )
            except TypeError:
                # Compatibility with PyTorch versions predating weights_only.
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if int(checkpoint.get("seed", self.seed)) != self.seed:
                raise RuntimeError(f"resume checkpoint seed mismatch in {checkpoint_path}")
            checkpoint_cfg = checkpoint.get("config", self.cfg)
            if _parent_signature(checkpoint_cfg) != _parent_signature(self.cfg):
                raise RuntimeError(f"resume checkpoint parent configuration mismatch in {checkpoint_path}")
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            recovery = checkpoint.get("recovery_state")
            if recovery is not None:
                order = [int(x) for x in recovery["order"]]
                cursor = int(recovery["cursor"])
                epoch = int(recovery["epoch"])
                reservoir.ids = [int(x) for x in recovery["reservoir_ids"]]
                reservoir.seen = int(recovery["reservoir_seen"])
                reservoir.rng.setstate(recovery["reservoir_rng_state"])
                parent_steps = int(recovery["parent_steps"])
                if "python_random_state" in recovery:
                    random.setstate(recovery["python_random_state"])
                if "numpy_random_state" in recovery:
                    np.random.set_state(recovery["numpy_random_state"])
                if "torch_rng_state" in recovery:
                    torch.set_rng_state(recovery["torch_rng_state"].cpu())
                if torch.cuda.is_available() and "cuda_rng_state_all" in recovery:
                    torch.cuda.set_rng_state_all([state.cpu() for state in recovery["cuda_rng_state_all"]])
                recovery_mode = "checkpoint_state"
            else:
                # v1.1 checkpoints predate full recovery-state capture.  The
                # parent input order and reservoir are generated by local seeded
                # RNGs, so their exact pre-probe state can be reconstructed
                # without repeating expensive parent forward/backward passes.
                order = deterministic_order(len(self.dataset), self.seed)
                cursor = 0
                epoch = 0
                reservoir = Reservoir(int(causal.get("reservoir_capacity", 512)), self.seed + 991)
                parent_steps = 0
                for _ in range(total_required):
                    if cursor + batch_size > len(order):
                        epoch += 1
                        order = deterministic_order(len(self.dataset), self.seed + 1009 * epoch)
                        cursor = 0
                    ids = order[cursor : cursor + batch_size]
                    cursor += batch_size
                    reservoir.add_many(ids)
                    parent_steps += 1
                expected_steps = int(checkpoint.get("parent_steps", total_required))
                if parent_steps != expected_steps:
                    raise RuntimeError(
                        f"reconstructed parent step mismatch: {parent_steps} != {expected_steps}"
                    )
                recovery_mode = "reconstructed_v1_1_state"

            self.events.write(
                {
                    "event": "resumed_from_preprobe_parent",
                    "checkpoint": str(checkpoint_path),
                    "parent_steps": parent_steps,
                    "recovery_mode": recovery_mode,
                }
            )
        else:
            for step in range(total_required):
                batch = next_batch()
                loss, acc, _ = _train_parent_step(
                    model=self.model, optimizer=optimizer, batch=batch, device=self.device
                )
                reservoir.add_many(batch.ids)
                parent_steps += 1
                if step % int(training.get("log_interval", 100)) == 0:
                    self.events.write(
                        {
                            "event": "parent_step",
                            "step": step,
                            "loss": loss,
                            "accuracy": acc,
                            "reservoir": len(reservoir.ids),
                        }
                    )

            # Persist the parent state before any causal probes.  Probe branches
            # are observational and never committed.  v1.2 additionally stores
            # the exact stream/reservoir/RNG state so any future probe-side
            # interruption can resume without repeating parent training.
            recovery_state = {
                "order": list(order),
                "cursor": int(cursor),
                "epoch": int(epoch),
                "reservoir_ids": list(reservoir.ids),
                "reservoir_seen": int(reservoir.seen),
                "reservoir_rng_state": reservoir.rng.getstate(),
                "parent_steps": int(parent_steps),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                recovery_state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "seed": self.seed,
                    "parent_steps": parent_steps,
                    "config": self.cfg,
                    "recovery_state": recovery_state,
                },
                self.run_dir / "preprobe_parent.pt",
            )

        # The causal probe stream is separate from the pretraining pass so all
        # probe states occur after the model has learned a nontrivial function.
        causal_states = 0
        probe_attempts = 0
        points = 0
        probe_interval = max(int(training.get("probe_interval", 10)), 1)
        max_states = int(training.get("causal_states", 50))
        probe_steps = int(training.get("probe_stream_steps", max_states * probe_interval * 3 + 10))
        protected_count = int(causal.get("protected_count", 12))
        guard_count = int(causal.get("guard_count", 12))
        current_count = int(causal.get("current_count", 12))

        for probe_step in range(probe_steps):
            batch = next_batch()
            if probe_step % probe_interval == 0 and causal_states < max_states and len(reservoir.ids) >= protected_count + guard_count:
                candidates = reservoir.sample(protected_count + guard_count + 16)
                protected_ids = candidates[:protected_count]
                guard_ids = _select_disjoint(candidates[protected_count:], set(protected_ids), guard_count)
                if len(guard_ids) < guard_count:
                    extra = _select_disjoint(reservoir.ids, set(protected_ids) | set(guard_ids), guard_count - len(guard_ids))
                    guard_ids.extend(extra)
                protected = stack_indices(self.dataset, protected_ids)
                guards = stack_indices(self.dataset, guard_ids) if guard_ids else None
                novel_ids = batch.ids[:current_count]
                if len(novel_ids) < current_count:
                    novel_ids += next_batch().ids[: current_count - len(novel_ids)]
                novel = stack_indices(self.dataset, novel_ids)
                produced = self._probe_state(
                    state_index=probe_attempts,
                    parent_step=parent_steps,
                    protected=protected,
                    novel=novel,
                    guards=guards,
                )
                probe_attempts += 1
                points += produced
                if produced > 0:
                    causal_states += 1
            loss, acc, _ = _train_parent_step(model=self.model, optimizer=optimizer, batch=batch, device=self.device)
            reservoir.add_many(batch.ids)
            parent_steps += 1
            if causal_states >= max_states:
                break

        full_coverage = causal_states == max_states
        summary = {
            "status": "complete" if full_coverage else "incomplete",
            "seed": self.seed,
            "dataset": str(self.sweep["dataset"]["kind"]),
            "modality": self.modality,
            "architecture": str(self.sweep["model"]["architecture"]),
            "model": asdict(self.description),
            "parent_steps": parent_steps,
            "causal_states": causal_states,
            "probe_attempts": probe_attempts,
            "compatibility_points": points,
            "retention_frontier_points": points * len(causal.get("retention_budget_betas", [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0])),
            "retention_frontier_grid_points": points * int(causal.get("retention_frontier_grid_points", 33)),
            "afm_native_points": causal_states * len(causal["requested_kappas"]),
            "requested_kappas": [float(x) for x in causal["requested_kappas"]],
            "retention_budget_betas": [float(x) for x in causal.get("retention_budget_betas", [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0])],
            "methods": list(causal["methods"]),
            "comparator_normalization": "matched_finite_delta0_bisection",
            "retention_reference_rule": str(causal.get("retention_reference_rule", "max_unrestricted_drift_across_kappa")),
            "elapsed_seconds": time.time() - start,
            "device": str(self.device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        torch.save(
            {"model": self.model.state_dict(), "config": self.cfg, "summary": summary},
            self.run_dir / "final_parent.pt",
        )
        self.events.write({"event": "finished", **summary})
        self.events.close()
        self.points.close()
        self.frontier_points.close()
        self.frontier_grid.close()
        self.afm_native_points.close()
        if bool(training.get("require_full_causal_states", True)) and not full_coverage:
            raise RuntimeError(
                f"compatibility sweep incomplete: obtained {causal_states}/{max_states} causal states "
                f"after {probe_attempts} probe attempts"
            )
        return summary
