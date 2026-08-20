from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from afmvision.afm.parameter_vector import temporary_parameters
from afmvision.compatibility.data import Batch, deterministic_order, make_causal_current_batch, stack_indices
from afmvision.compatibility.extension_common import (
    JsonlWriter,
    SavedParentProbeBase,
    coefficient_of_variation,
    relative_range,
    train_parent_step,
)
from afmvision.compatibility.geometry import build_protected_geometry, functional_jacobian
from afmvision.compatibility.independent_directions import DirectionCandidate, make_independent_target_candidates
from afmvision.compatibility.methods import (
    ComparatorResult,
    endpoint_metrics_for_vector,
    method_proposal_delta,
    retention_frontier_grid,
    run_method,
)


DEFAULT_SYSTEMS = ["cifar10_cnn", "cifar10_vit"]
DEFAULT_REQUESTED_KAPPAS = [0.10, 0.25, 0.50, 0.75]
DEFAULT_NATURAL_NORM_FRACTIONS = [0.01, 0.10, 0.50, 1.00]
DEFAULT_METHODS = ["projection", "unrestricted", "linearized_distillation", "ewc_prox"]
DEFAULT_BETAS = [0.05, 0.10, 0.25, 0.50]


@dataclass
class FixedNormCandidate:
    direction: DirectionCandidate
    comparator: ComparatorResult
    norm_relative_error: float


def _teacher_loss_value64(model, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        values = model.functional_logits(inputs)
        diff = (values - targets).to(dtype=torch.float64)
        return float((0.5 * diff.square().sum()).item())


def fixed_norm_comparator(
    *,
    model,
    vectoriser,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    current_gradient: torch.Tensor,
    target_update_norm: float,
    update_norm_rtol: float,
) -> tuple[ComparatorResult | None, str | None, float]:
    """Evaluate the genuine unrestricted endpoint at an exact requested displacement norm.

    The teacher-gradient direction is unchanged.  Its scalar step is chosen as
    alpha = target_norm / ||g||, so scaling changes magnitude but not compatibility.
    A candidate is usable only if the realised finite endpoint is distinct,
    norm-matched, finite, and gives positive teacher-loss decrease.
    """
    target_norm = float(target_update_norm)
    if target_norm <= 0.0:
        raise ValueError("target_update_norm must be positive")
    vector_before = vectoriser.flatten(detach=True)
    qnorm = float(torch.linalg.vector_norm(current_gradient).item())
    if not math.isfinite(qnorm) or qnorm <= 1e-20:
        return None, "zero_or_nonfinite_gradient", float("inf")
    alpha = target_norm / qnorm
    candidate = vector_before - alpha * current_gradient
    if torch.equal(candidate, vector_before):
        return None, "parameter_resolution_zero_step", float("inf")
    realised_delta = candidate - vector_before
    realised_norm = float(torch.linalg.vector_norm(realised_delta).item())
    rel_norm_error = abs(realised_norm - target_norm) / target_norm
    if not math.isfinite(realised_norm) or rel_norm_error > float(update_norm_rtol):
        return None, "update_norm_mismatch", rel_norm_error

    loss_before = _teacher_loss_value64(model, current_inputs, current_targets)
    with temporary_parameters(vectoriser, candidate):
        loss_after = _teacher_loss_value64(model, current_inputs, current_targets)
        with torch.no_grad():
            current_full = model(current_inputs).detach().clone()
    decrease = float(loss_before - loss_after)
    if not math.isfinite(decrease):
        return None, "nonfinite_delta0", rel_norm_error
    if decrease <= 0.0:
        return None, "nonpositive_delta0", rel_norm_error
    return (
        ComparatorResult(
            alpha=float(alpha),
            vector_before=vector_before,
            vector_after=candidate,
            delta=realised_delta,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            decrease=decrease,
            current_full_logits_after=current_full,
        ),
        None,
        rel_norm_error,
    )


def select_best_delta0_candidates(
    by_kappa: dict[float, list[FixedNormCandidate]],
    *,
    max_cv: float,
    max_relative_range: float,
) -> tuple[dict[float, FixedNormCandidate] | None, dict[str, float | int | bool]]:
    """Pick the tightest available finite-Delta0 set without making it an admission rule.

    Every returned candidate already has the requested fixed update norm and a
    positive genuine unrestricted endpoint.  Directional freedom is used only to
    minimise finite-Delta0 imbalance across kappa.  The declared CV/range
    thresholds are reported in ``audit['matched']`` but do not suppress the best
    available selection.  This distinction is required by the natural-scale
    repair experiment, where finite-Delta0 spread is diagnostic rather than a
    hard feasibility criterion.
    """
    kappas = sorted(by_kappa)
    if not kappas or any(not by_kappa[k] for k in kappas):
        return None, {
            "delta0_cv": float("nan"),
            "delta0_relative_range": float("nan"),
            "matched": False,
            "candidate_anchor_count": 0,
        }

    anchors = sorted({float(c.comparator.decrease) for k in kappas for c in by_kappa[k]})
    best_sel: dict[float, FixedNormCandidate] | None = None
    best_score: tuple[float, float, float] | None = None
    best_cv = float("inf")
    best_range = float("inf")
    for anchor in anchors:
        selected: dict[float, FixedNormCandidate] = {}
        for k in kappas:
            selected[k] = min(
                by_kappa[k],
                key=lambda c: abs(math.log(float(c.comparator.decrease)) - math.log(anchor)),
            )
        vals = [float(selected[k].comparator.decrease) for k in kappas]
        cv = coefficient_of_variation(vals)
        rr = relative_range(vals)
        # First minimise cross-kappa mismatch; among ties prefer larger progress.
        score = (rr, cv, -sum(vals) / len(vals))
        if best_score is None or score < best_score:
            best_score = score
            best_sel = selected
            best_cv = cv
            best_range = rr

    matched = (
        best_sel is not None
        and math.isfinite(best_cv)
        and math.isfinite(best_range)
        and best_cv <= float(max_cv)
        and best_range <= float(max_relative_range)
    )
    audit = {
        "delta0_cv": float(best_cv),
        "delta0_relative_range": float(best_range),
        "matched": bool(matched),
        "candidate_anchor_count": len(anchors),
    }
    return best_sel, audit


def select_delta0_matched_candidates(
    by_kappa: dict[float, list[FixedNormCandidate]],
    *,
    max_cv: float,
    max_relative_range: float,
) -> tuple[dict[float, FixedNormCandidate] | None, dict[str, float | int | bool]]:
    """Backward-compatible strict selector used by the original bridge-v1 run."""
    selected, audit = select_best_delta0_candidates(
        by_kappa, max_cv=max_cv, max_relative_range=max_relative_range
    )
    return (selected if bool(audit.get("matched")) else None), audit


class NaturalScaleBridgeRunner(SavedParentProbeBase):
    """Causal compatibility bridge at parameter-update norms tied to natural training.

    States are sampled on a fixed parent-step schedule with no rejection.  At each
    state and each predeclared natural-norm fraction, the experiment attempts to
    find one direction for every interior kappa such that:

      * realised compatibility is within the declared tolerance,
      * unrestricted displacement norm is the same target norm,
      * genuine finite unrestricted Delta0 is matched across kappa,
      * the absolute retention reference D_ref is common across kappa and methods.

    A state x target-norm condition that cannot satisfy these constraints is
    recorded as infeasible; it is never replaced by another parent state and is
    never silently shrunk to an easier norm.
    """

    schema = "causal_compatibility_natural_scale_bridge_v1"

    def __init__(
        self,
        *,
        source_run_dir: str | Path,
        run_dir: str | Path,
        system: str,
        device: torch.device,
        natural_median_update_norm: float,
        natural_norm_fractions: list[float] | None = None,
        requested_kappas: list[float] | None = None,
        candidate_pool: int = 64,
        kappa_tolerance: float = 0.01,
        update_norm_rtol: float = 5e-3,
        delta0_cv_tolerance: float = 0.02,
        delta0_range_tolerance: float = 0.04,
        hard_delta0_match: bool = True,
        states: int = 50,
        probe_interval: int = 10,
        methods: list[str] | None = None,
        retention_betas: list[float] | None = None,
        probe_rng_offset: int = 939391,
    ) -> None:
        if str(system) not in DEFAULT_SYSTEMS:
            raise ValueError(f"bridge v1 is restricted to {DEFAULT_SYSTEMS}, got {system!r}")
        # Base init is used for audited parent/data/model reconstruction; its
        # admission-style trajectory is intentionally not used below.
        super().__init__(
            source_run_dir=source_run_dir,
            run_dir=run_dir,
            system=system,
            device=device,
            max_states=int(states),
            max_probe_steps=max(int(states) * int(probe_interval), 1),
        )
        self.states = int(states)
        self.probe_interval = int(probe_interval)
        self.natural_median_update_norm = float(natural_median_update_norm)
        self.natural_norm_fractions = [float(x) for x in (natural_norm_fractions or DEFAULT_NATURAL_NORM_FRACTIONS)]
        self.requested_kappas = [float(x) for x in (requested_kappas or DEFAULT_REQUESTED_KAPPAS)]
        self.candidate_pool = int(candidate_pool)
        self.kappa_tolerance = float(kappa_tolerance)
        self.update_norm_rtol = float(update_norm_rtol)
        self.delta0_cv_tolerance = float(delta0_cv_tolerance)
        self.delta0_range_tolerance = float(delta0_range_tolerance)
        self.hard_delta0_match = bool(hard_delta0_match)
        self.methods = [str(x) for x in (methods or DEFAULT_METHODS)]
        self.retention_betas = [float(x) for x in (retention_betas or DEFAULT_BETAS)]
        self.probe_rng_offset = int(probe_rng_offset)

        if self.states <= 0 or self.probe_interval <= 0:
            raise ValueError("states and probe_interval must be positive")
        if not self.natural_median_update_norm > 0.0:
            raise ValueError("natural_median_update_norm must be positive")
        if not self.natural_norm_fractions or any(x <= 0.0 for x in self.natural_norm_fractions):
            raise ValueError("natural norm fractions must all be positive")
        if not self.requested_kappas or any(not (0.0 < x < 1.0) for x in self.requested_kappas):
            raise ValueError("bridge v1 uses interior kappas strictly between 0 and 1")
        if self.candidate_pool < 4:
            raise ValueError("candidate_pool must be at least 4")
        allowed_methods = {"projection", "unrestricted", "linearized_distillation", "ewc_prox"}
        if set(self.methods) - allowed_methods:
            raise ValueError(f"bridge methods must be a subset of {sorted(allowed_methods)}")

        self.feasibility = JsonlWriter(self.run_dir / "bridge_feasibility.jsonl")
        self.points = JsonlWriter(self.run_dir / "bridge_points.jsonl")
        self.frontier = JsonlWriter(self.run_dir / "bridge_frontier_points.jsonl")
        self.native = JsonlWriter(self.run_dir / "bridge_afm_native_points.jsonl")
        (self.run_dir / "bridge_config.json").write_text(
            json.dumps(
                {
                    "schema": self.schema,
                    "source_run_dir": str(self.source_run_dir.resolve()),
                    "system": self.system,
                    "seed": self.seed,
                    "states": self.states,
                    "probe_interval": self.probe_interval,
                    "state_rule": "fixed pre-update parent states 0, interval, ..., (states-1)*interval; never filter by feasibility",
                    "protected_sampling": "separate deterministic probe RNG; does not consume reservoir replacement RNG",
                    "natural_median_update_norm": self.natural_median_update_norm,
                    "natural_norm_fractions": self.natural_norm_fractions,
                    "target_update_norms": [self.natural_median_update_norm * x for x in self.natural_norm_fractions],
                    "requested_kappas": self.requested_kappas,
                    "candidate_pool": self.candidate_pool,
                    "kappa_tolerance": self.kappa_tolerance,
                    "update_norm_rtol": self.update_norm_rtol,
                    "delta0_cv_tolerance": self.delta0_cv_tolerance,
                    "delta0_range_tolerance": self.delta0_range_tolerance,
                    "hard_delta0_match": self.hard_delta0_match,
                    "methods": self.methods,
                    "retention_betas": self.retention_betas,
                    "retention_reference": "D_ref(state,target_norm)=max selected unrestricted protected drift across kappa",
                    "frontier_grid_points": int(self.causal.get("retention_frontier_grid_points", 33)),
                    "grid_storage": "all nonlinear grid endpoints are evaluated; only selected frontier points and monotonicity audit are stored",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def probe_state(self, **kwargs) -> int:  # pragma: no cover - fixed-schedule runner calls _probe_fixed_state directly.
        raise RuntimeError("NaturalScaleBridgeRunner uses its fixed-schedule trajectory, not admission sampling")

    def _write_infeasible(
        self,
        *,
        state_id: int,
        local_step: int,
        parent_step: int,
        fraction: float,
        target_norm: float,
        reason: str,
        candidate_counts: dict[float, int] | None = None,
        audit: dict[str, Any] | None = None,
    ) -> None:
        self.feasibility.write(
            {
                "system": self.system,
                "seed": self.seed,
                "state_id": int(state_id),
                "local_step": int(local_step),
                "parent_step": int(parent_step),
                "natural_norm_fraction": float(fraction),
                "natural_median_update_norm": self.natural_median_update_norm,
                "target_update_norm": float(target_norm),
                "feasible": False,
                "reason": str(reason),
                "candidate_counts_by_kappa": {} if candidate_counts is None else {str(k): int(v) for k, v in candidate_counts.items()},
                **({} if audit is None else audit),
            }
        )

    def _probe_fixed_state(
        self,
        *,
        state_id: int,
        local_step: int,
        parent_step: int,
        protected: Batch,
        novel: Batch,
        guards: Batch | None,
    ) -> tuple[int, int]:
        current = make_causal_current_batch(
            modality=self.modality,
            protected=protected,
            novel=novel,
            near_count=int(self.causal["near_protected_count"]),
            seed=self.seed * 1000003 + state_id,
            vocab_size=self.vocab_size,
        )
        current_inputs = current.inputs.to(self.device)
        protected_inputs = protected.inputs.to(self.device)
        protected_labels = protected.labels.to(self.device)
        guard_inputs = None if guards is None else guards.inputs.to(self.device)

        self.model.eval()
        pre_vector = self.vectoriser.flatten(detach=True)
        with torch.no_grad():
            protected_before = self.model(protected_inputs).detach().clone()
            replay_logits = protected_before.clone()
        _, Jp = functional_jacobian(self.model, protected_inputs)
        current_measure, Jc = functional_jacobian(self.model, current_inputs)
        geometry = build_protected_geometry(Jp, ridge=float(self.causal.get("geometry_ridge", 1e-7)))

        # Generate the direction bank once per state; it is then evaluated at all
        # four predeclared natural-norm targets.
        candidates_by_kappa: dict[float, list[DirectionCandidate]] = {}
        lo_all = None
        hi_all = None
        for kidx, kappa in enumerate(self.requested_kappas):
            candidates = make_independent_target_candidates(
                model=self.model,
                current_inputs=current_inputs,
                current_measurements=current_measure.detach(),
                current_jacobian=Jc.detach(),
                geometry=geometry,
                requested_kappa=kappa,
                gradient_norm=float(self.causal.get("gradient_norm", 1.0)),
                max_candidates=self.candidate_pool,
                seed=self.seed * 10000019 + state_id * 1009 + kidx,
                kappa_tolerance=self.kappa_tolerance,
            )
            candidates_by_kappa[kappa] = candidates
            if candidates:
                lo_all = candidates[0].target.achievable_min if lo_all is None else min(lo_all, candidates[0].target.achievable_min)
                hi_all = candidates[0].target.achievable_max if hi_all is None else max(hi_all, candidates[0].target.achievable_max)

        counts = {k: len(v) for k, v in candidates_by_kappa.items()}
        if any(v == 0 for v in counts.values()):
            for fraction in self.natural_norm_fractions:
                self._write_infeasible(
                    state_id=state_id,
                    local_step=local_step,
                    parent_step=parent_step,
                    fraction=fraction,
                    target_norm=fraction * self.natural_median_update_norm,
                    reason="insufficient_fixed_kappa_direction_candidates",
                    candidate_counts=counts,
                    audit={"achievable_min": lo_all, "achievable_max": hi_all},
                )
            self.vectoriser.assign(pre_vector)
            return 0, len(self.natural_norm_fractions)

        feasible_conditions = 0
        attempted_conditions = 0
        for fraction in self.natural_norm_fractions:
            attempted_conditions += 1
            target_norm = float(fraction * self.natural_median_update_norm)
            usable_by_kappa: dict[float, list[FixedNormCandidate]] = {}
            failure_counts: dict[str, int] = {}
            max_norm_error_seen = 0.0
            for kappa in self.requested_kappas:
                usable: list[FixedNormCandidate] = []
                for cand in candidates_by_kappa[kappa]:
                    self.vectoriser.assign(pre_vector)
                    comp, reason, norm_err = fixed_norm_comparator(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        current_inputs=current_inputs,
                        current_targets=cand.target.teacher_measurements,
                        current_gradient=cand.target.gradient,
                        target_update_norm=target_norm,
                        update_norm_rtol=self.update_norm_rtol,
                    )
                    if math.isfinite(norm_err):
                        max_norm_error_seen = max(max_norm_error_seen, float(norm_err))
                    if comp is None:
                        failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + 1
                        continue
                    usable.append(FixedNormCandidate(cand, comp, float(norm_err)))
                usable_by_kappa[kappa] = usable

            usable_counts = {k: len(v) for k, v in usable_by_kappa.items()}
            if any(v == 0 for v in usable_counts.values()):
                self._write_infeasible(
                    state_id=state_id,
                    local_step=local_step,
                    parent_step=parent_step,
                    fraction=fraction,
                    target_norm=target_norm,
                    reason="no_positive_same_norm_endpoint_for_one_or_more_kappas",
                    candidate_counts=usable_counts,
                    audit={
                        "raw_candidate_counts_by_kappa": {str(k): counts[k] for k in counts},
                        "endpoint_failure_counts": failure_counts,
                        "max_candidate_norm_relative_error": max_norm_error_seen,
                    },
                )
                continue

            selected, match_audit = select_best_delta0_candidates(
                usable_by_kappa,
                max_cv=self.delta0_cv_tolerance,
                max_relative_range=self.delta0_range_tolerance,
            )
            if selected is None:
                raise RuntimeError("positive fixed-norm candidates existed for every kappa but no selection was produced")
            if self.hard_delta0_match and not bool(match_audit.get("matched")):
                self._write_infeasible(
                    state_id=state_id,
                    local_step=local_step,
                    parent_step=parent_step,
                    fraction=fraction,
                    target_norm=target_norm,
                    reason="finite_delta0_match_tolerance_not_met",
                    candidate_counts=usable_counts,
                    audit={
                        **match_audit,
                        "max_candidate_norm_relative_error": max_norm_error_seen,
                    },
                )
                continue

            # Compute the common absolute retention reference from the selected
            # same-norm unrestricted endpoints.  In repair mode finite-Delta0
            # alignment is diagnostic only, not an admission criterion.
            unrestricted_rows: dict[float, tuple[FixedNormCandidate, float, float]] = {}
            for kappa in self.requested_kappas:
                chosen = selected[kappa]
                self.vectoriser.assign(pre_vector)
                _, drift, rms = endpoint_metrics_for_vector(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    vector_after=chosen.comparator.vector_after,
                    current_inputs=current_inputs,
                    current_targets=chosen.direction.target.teacher_measurements,
                    protected_inputs=protected_inputs,
                    protected_logits_before=protected_before,
                )
                unrestricted_rows[kappa] = (chosen, float(drift), float(rms))
            d_ref = max(v[1] for v in unrestricted_rows.values())
            selected_delta0s = [float(unrestricted_rows[k][0].comparator.decrease) for k in self.requested_kappas]
            selected_norms = [float(torch.linalg.vector_norm(unrestricted_rows[k][0].comparator.delta).item()) for k in self.requested_kappas]
            selected_kappas = [float(unrestricted_rows[k][0].direction.target.measured_kappa) for k in self.requested_kappas]
            feasible_conditions += 1
            self.feasibility.write(
                {
                    "system": self.system,
                    "seed": self.seed,
                    "state_id": state_id,
                    "local_step": local_step,
                    "parent_step": parent_step,
                    "natural_norm_fraction": fraction,
                    "natural_median_update_norm": self.natural_median_update_norm,
                    "target_update_norm": target_norm,
                    "feasible": True,
                    "reason": "matched" if bool(match_audit.get("matched")) else "positive_same_norm_delta0_diagnostic_only",
                    "candidate_counts_by_kappa": {str(k): usable_counts[k] for k in self.requested_kappas},
                    "delta0_match_within_declared_tolerance": bool(match_audit.get("matched")),
                    "delta0_matching_used_as_admission": bool(self.hard_delta0_match),
                    "delta0_cv": coefficient_of_variation(selected_delta0s),
                    "delta0_relative_range": relative_range(selected_delta0s),
                    "mean_delta0": sum(selected_delta0s) / len(selected_delta0s),
                    "max_update_norm_relative_error": max(abs(x - target_norm) / target_norm for x in selected_norms),
                    "max_abs_kappa_error": max(abs(x - k) for x, k in zip(selected_kappas, self.requested_kappas)),
                    "retention_reference_drift": d_ref,
                    "achieved_to_natural_median_norm_ratio": sum(selected_norms) / len(selected_norms) / self.natural_median_update_norm,
                }
            )

            betas = self.retention_betas
            eps = float(self.causal.get("retention_budget_epsilon", 1e-8))
            grid_n = int(self.causal.get("retention_frontier_grid_points", 33))
            for kappa in self.requested_kappas:
                chosen, unrestricted_drift, unrestricted_rms = unrestricted_rows[kappa]
                target = chosen.direction.target
                comp = chosen.comparator
                realised_norm = float(torch.linalg.vector_norm(comp.delta).item())
                self.vectoriser.assign(pre_vector)
                native = run_method(
                    method="afm",
                    model=self.model,
                    vectoriser=self.vectoriser,
                    comparator=comp,
                    current_gradient=target.gradient,
                    geometry=geometry,
                    current_inputs=current_inputs,
                    current_targets=target.teacher_measurements,
                    protected_inputs=protected_inputs,
                    protected_labels=protected_labels,
                    protected_logits_before=protected_before,
                    replay_stored_logits=replay_logits,
                    guard_inputs=guard_inputs,
                    retention_tolerance=float(self.causal.get("retention_tolerance", 0.005)),
                    method_config=self.causal,
                    address_encoder=self.address_encoder,
                )
                self.native.write(
                    {
                        "system": self.system,
                        "seed": self.seed,
                        "state_id": state_id,
                        "local_step": local_step,
                        "parent_step": parent_step,
                        "natural_norm_fraction": fraction,
                        "natural_median_update_norm": self.natural_median_update_norm,
                        "target_update_norm": target_norm,
                        "requested_kappa": kappa,
                        "measured_kappa": target.measured_kappa,
                        "gradient_norm": target.gradient_norm,
                        "delta0": comp.decrease,
                        "unrestricted_update_norm": realised_norm,
                        "unrestricted_protected_drift": unrestricted_drift,
                        "retention_reference_drift": d_ref,
                        "persistent_decrease": native.persistent_decrease,
                        "persistent_ratio": native.persistent_ratio,
                        "retention_max_abs_drift": native.protected_max_abs_drift,
                        "retention_pass": native.retention_pass,
                        "accepted": native.accepted,
                        "afm_lambda_hat": native.afm_lambda_hat,
                        "deployed_ratio": native.deployed_ratio,
                        "finite_completion_available": native.finite_completion_available,
                        "finite_endpoint_error": native.finite_endpoint_error,
                        "finite_current_error": native.finite_current_error,
                        "finite_protected_error": native.finite_protected_error,
                        "obstruction": native.obstruction,
                    }
                )

                for method in self.methods:
                    self.vectoriser.assign(pre_vector)
                    proposal = method_proposal_delta(
                        method=method,
                        model=self.model,
                        vectoriser=self.vectoriser,
                        comparator=comp,
                        current_gradient=target.gradient,
                        geometry=geometry,
                        protected_inputs=protected_inputs,
                        protected_labels=protected_labels,
                        replay_stored_logits=replay_logits,
                        method_config=self.causal,
                    )
                    base = {
                        "system": self.system,
                        "dataset": str(self.sweep["dataset"]["kind"]),
                        "modality": self.modality,
                        "architecture": str(self.sweep["model"]["architecture"]),
                        "seed": self.seed,
                        "state_id": state_id,
                        "local_step": local_step,
                        "parent_step": parent_step,
                        "natural_norm_fraction": fraction,
                        "natural_median_update_norm": self.natural_median_update_norm,
                        "target_update_norm": target_norm,
                        "requested_kappa": kappa,
                        "measured_kappa": target.measured_kappa,
                        "gradient_norm": target.gradient_norm,
                        "method": method,
                        "delta0": comp.decrease,
                        "comparator_alpha": comp.alpha,
                        "unrestricted_update_norm": realised_norm,
                        "unrestricted_protected_drift": unrestricted_drift,
                        "unrestricted_protected_rms": unrestricted_rms,
                        "retention_reference_drift": d_ref,
                    }
                    self.points.write(
                        {
                            **base,
                            "proposal_update_norm": float(torch.linalg.vector_norm(proposal).item()),
                        }
                    )
                    frontier, _grid, audit = retention_frontier_grid(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        comparator=comp,
                        proposal_delta=proposal,
                        current_inputs=current_inputs,
                        current_targets=target.teacher_measurements,
                        protected_inputs=protected_inputs,
                        protected_logits_before=protected_before,
                        retention_reference_drift=d_ref,
                        betas=betas,
                        epsilon_num=eps,
                        grid_points=grid_n,
                    )
                    for fp in frontier:
                        self.frontier.write(
                            {
                                **base,
                                "retention_beta": fp.beta,
                                "retention_budget": fp.budget,
                                "frontier_scale": fp.scale,
                                "grid_points_evaluated": audit["grid_points"],
                                "frontier_drift_monotone_on_grid": audit["drift_monotone_on_grid"],
                                "frontier_monotonic_drift_violations": audit["monotonic_drift_violations"],
                                "persistent_decrease": fp.persistent_decrease,
                                "persistent_ratio": fp.persistent_ratio,
                                "retention_max_abs_drift": fp.protected_max_abs_drift,
                                "retention_rms_drift": fp.protected_rms_drift,
                                "retention_pass": fp.protected_max_abs_drift <= fp.budget + 1e-12,
                                "accepted": fp.persistent_decrease > 0.0,
                                "update_norm": fp.update_norm,
                            }
                        )

        self.vectoriser.assign(pre_vector)
        return feasible_conditions, attempted_conditions

    def run_fixed_schedule(self) -> dict[str, Any]:
        state = self.load_parent()
        batch_size = int(self.training.get("batch_size", 64))
        protected_count = int(self.causal.get("protected_count", 16))
        guard_count = int(self.causal.get("guard_count", 16))
        current_count = int(self.causal.get("current_count", 16))
        probe_rng = random.Random(self.seed + self.probe_rng_offset)

        def next_batch() -> Batch:
            if state.cursor + batch_size > len(state.order):
                state.epoch += 1
                state.order = deterministic_order(len(self.dataset), self.seed + 1009 * state.epoch)
                state.cursor = 0
            ids = state.order[state.cursor : state.cursor + batch_size]
            state.cursor += batch_size
            return stack_indices(self.dataset, ids)

        self.events.write(
            {
                "event": "resumed_from_preprobe_parent",
                "source": str(self.checkpoint_path.resolve()),
                "parent_steps": state.parent_steps,
                "recovery_mode": state.recovery_mode,
            }
        )
        state_id = 0
        feasible_conditions = 0
        attempted_conditions = 0
        max_local_step = (self.states - 1) * self.probe_interval
        for local_step in range(max_local_step + 1):
            batch = next_batch()
            if local_step % self.probe_interval == 0:
                if len(state.reservoir.ids) < protected_count + guard_count:
                    raise RuntimeError(
                        f"bridge reservoir has {len(state.reservoir.ids)} ids; needs {protected_count + guard_count}"
                    )
                sampled = probe_rng.sample(list(state.reservoir.ids), protected_count + guard_count)
                protected_ids = sampled[:protected_count]
                guard_ids = sampled[protected_count:]
                protected = stack_indices(self.dataset, protected_ids)
                guards = stack_indices(self.dataset, guard_ids) if guard_ids else None
                novel_ids = list(batch.ids[:current_count])
                if len(novel_ids) != current_count:
                    raise RuntimeError("ordinary parent batch shorter than current_count")
                novel = stack_indices(self.dataset, novel_ids)
                feasible, attempted = self._probe_fixed_state(
                    state_id=state_id,
                    local_step=local_step,
                    parent_step=state.parent_steps,
                    protected=protected,
                    novel=novel,
                    guards=guards,
                )
                feasible_conditions += feasible
                attempted_conditions += attempted
                self.events.write(
                    {
                        "event": "bridge_state",
                        "state_id": state_id,
                        "local_step": local_step,
                        "parent_step": state.parent_steps,
                        "feasible_target_norm_conditions": feasible,
                        "attempted_target_norm_conditions": attempted,
                        "current_protected_id_overlap": len(set(novel_ids) & set(protected_ids)),
                    }
                )
                state_id += 1

            # Probe branches are discarded.  Only the ordinary parent update is committed.
            self.vectoriser.assign(self.vectoriser.flatten(detach=True))
            train_parent_step(model=self.model, optimizer=state.optimizer, batch=batch, device=self.device)
            state.reservoir.add_many(batch.ids)
            state.parent_steps += 1

        if state_id != self.states:
            raise RuntimeError(f"bridge collected {state_id} fixed states, expected {self.states}")
        expected_attempts = self.states * len(self.natural_norm_fractions)
        if attempted_conditions != expected_attempts:
            raise RuntimeError(f"bridge attempted {attempted_conditions} state-target conditions, expected {expected_attempts}")
        return {
            "states": state_id,
            "attempted_target_norm_conditions": attempted_conditions,
            "feasible_target_norm_conditions": feasible_conditions,
            "parent_steps": state.parent_steps,
            "recovery_mode": state.recovery_mode,
        }

    def run(self) -> dict[str, Any]:
        start = time.time()
        traj = self.run_fixed_schedule()
        summary = {
            "schema": self.schema,
            "status": "complete",
            "system": self.system,
            "seed": self.seed,
            "states": traj["states"],
            "attempted_target_norm_conditions": traj["attempted_target_norm_conditions"],
            "feasible_target_norm_conditions": traj["feasible_target_norm_conditions"],
            "natural_median_update_norm": self.natural_median_update_norm,
            "natural_norm_fractions": self.natural_norm_fractions,
            "requested_kappas": self.requested_kappas,
            "candidate_pool": self.candidate_pool,
            "methods": self.methods,
            "retention_betas": self.retention_betas,
            "elapsed_seconds": time.time() - start,
            "device": str(self.device),
            "recovery_mode": traj["recovery_mode"],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self.events.write({"event": "finished", **summary})
        for writer in (self.events, self.feasibility, self.points, self.frontier, self.native):
            writer.close()
        return summary
