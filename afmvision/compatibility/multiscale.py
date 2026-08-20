from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from afmvision.afm.parameter_vector import temporary_parameters
from afmvision.compatibility.data import Batch, make_causal_current_batch
from afmvision.compatibility.extension_common import (
    JsonlWriter,
    SavedParentProbeBase,
    coefficient_of_variation,
)
from afmvision.compatibility.geometry import build_protected_geometry, functional_jacobian, make_controlled_targets
from afmvision.compatibility.methods import (
    ComparatorResult,
    calibrate_comparator_to_decrease,
    endpoint_metrics_for_vector,
    method_proposal_delta,
    retention_frontier_grid,
    run_method,
    unrestricted_comparator,
)


DEFAULT_SCALE_FRACTIONS = [0.05, 0.20, 0.50, 0.90]

def _teacher_loss_value64(model, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Teacher loss with float64 scalar accumulation for comparator scans."""
    with torch.no_grad():
        values = model.functional_logits(inputs)
        diff = (values - targets).to(dtype=torch.float64)
        return float((0.5 * diff.square().sum()).item())


def best_positive_comparator_on_backtracking_grid(
    *,
    model,
    vectoriser,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    current_gradient: torch.Tensor,
    initial_alpha: float,
    max_backtracks: int,
    backtrack_factor: float,
) -> ComparatorResult:
    """Return the largest positive finite decrease on the declared alpha grid.

    The v1.5 unrestricted comparator intentionally stops at the *first* positive
    endpoint while backtracking from ``initial_alpha``.  That is appropriate for
    constructing one genuine comparator, but it is not an estimate of the
    maximum attainable finite decrease along the direction.  A multi-scale
    experiment needs the latter: otherwise a near-zero crossing can be mistaken
    for the usable scale ceiling.

    We therefore evaluate the same deterministic backtracking grid without
    changing the direction and select the endpoint with the greatest positive
    decrease.  No proposal or protected-retention information enters this scan.
    """
    vector_before = vectoriser.flatten(detach=True)
    loss_before = _teacher_loss_value64(model, current_inputs, current_targets)
    alpha = float(initial_alpha)
    factor = float(backtrack_factor)
    if not (0.0 < factor < 1.0):
        raise ValueError(f"backtrack_factor must lie in (0,1), got {factor!r}")

    best = None
    evaluated = 0
    for _ in range(int(max_backtracks) + 1):
        if alpha <= 0.0:
            break
        candidate = vector_before - alpha * current_gradient
        realised_delta = candidate - vector_before
        if torch.equal(candidate, vector_before):
            break
        with temporary_parameters(vectoriser, candidate):
            loss_after = _teacher_loss_value64(model, current_inputs, current_targets)
            with torch.no_grad():
                current_full = model(current_inputs).detach().clone()
        decrease = float(loss_before - loss_after)
        evaluated += 1
        if decrease > 0.0 and (best is None or decrease > float(best.decrease)):
            best = ComparatorResult(
                alpha=float(alpha),
                vector_before=vector_before,
                vector_after=candidate,
                delta=realised_delta,
                loss_before=float(loss_before),
                loss_after=float(loss_after),
                decrease=decrease,
                current_full_logits_after=current_full,
            )
        alpha *= factor

    if best is None:
        raise RuntimeError(
            "same-state unrestricted scale scan found no positive endpoint; "
            f"initial_alpha={float(initial_alpha):.9g}, "
            f"backtrack_factor={factor:.9g}, max_backtracks={int(max_backtracks)}, "
            f"evaluated={evaluated}, "
            f"gradient_norm={float(torch.linalg.vector_norm(current_gradient).item()):.9g}"
        )
    return best


def multiscale_target_delta0(*, local_delta0: float, peak_delta0: float, expansion_fraction: float) -> float:
    """Log-interpolate from the proven v1.5 local target toward the common peak.

    ``expansion_fraction=0`` reproduces the v1.5 matched comparator target.
    ``expansion_fraction=1`` reaches the common maximum positive decrease on the
    declared unrestricted alpha grid.  Intermediate values are spaced in log
    Delta0 so the experiment can span orders of magnitude without introducing
    an artificial sub-v1.5 scale that may be below float parameter resolution.
    """
    local = float(local_delta0)
    peak = float(peak_delta0)
    frac = float(expansion_fraction)
    if not (local > 0.0 and peak >= local):
        raise ValueError(f"invalid multiscale anchors: local={local!r}, peak={peak!r}")
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"expansion_fraction must lie in [0,1], got {frac!r}")
    if peak == local:
        return local
    return math.exp(math.log(local) + frac * (math.log(peak) - math.log(local)))


def calibrate_native_gradient_to_decrease(
    *,
    model,
    vectoriser,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    current_gradient: torch.Tensor,
    vector_before: torch.Tensor,
    loss_before: float,
    hi_alpha: float,
    hi_decrease: float,
    target_decrease: float,
    relative_tolerance: float = 2.0e-3,
    max_bisections: int = 60,
) -> ComparatorResult:
    """Match finite Delta0 using the *native gradient* and a certified bracket.

    This avoids reconstructing the direction from a quantised endpoint delta.
    The high endpoint is one actually evaluated by the peak scan, so its finite
    decrease is known to be at least the requested target.  Bisection preserves
    the bracket ``decrease(lo) < target <= decrease(hi)``; monotonicity of the
    nonlinear loss is not assumed.
    """
    target = float(target_decrease)
    hi = float(hi_alpha)
    if not target > 0.0:
        raise ValueError(f"target_decrease must be positive, got {target!r}")
    if not hi > 0.0:
        raise ValueError(f"hi_alpha must be positive, got {hi!r}")
    if float(hi_decrease) + 1e-18 < target:
        raise ValueError(
            f"high endpoint decrease {float(hi_decrease):.9g} below target {target:.9g}"
        )

    lo = 0.0
    best = None
    best_error = float('inf')

    def evaluate(alpha: float):
        candidate = vector_before - float(alpha) * current_gradient
        if torch.equal(candidate, vector_before):
            return None
        realised_delta = candidate - vector_before
        with temporary_parameters(vectoriser, candidate):
            loss_after = _teacher_loss_value64(model, current_inputs, current_targets)
            with torch.no_grad():
                current_full = model(current_inputs).detach().clone()
        decrease = float(loss_before - loss_after)
        return ComparatorResult(
            alpha=float(alpha),
            vector_before=vector_before,
            vector_after=candidate,
            delta=realised_delta,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            decrease=decrease,
            current_full_logits_after=current_full,
        )

    # Re-evaluate the declared high endpoint using exactly the same native
    # direction formula used during the peak scan.
    high_result = evaluate(hi)
    if high_result is None or float(high_result.decrease) + 1e-18 < target:
        raise RuntimeError(
            "native-gradient multiscale bracket lost its high endpoint; "
            f"target={target:.9g}, hi_alpha={hi:.9g}, "
            f"hi_decrease_scan={float(hi_decrease):.9g}, "
            f"hi_decrease_recomputed={float('nan') if high_result is None else float(high_result.decrease):.9g}"
        )
    best = high_result
    best_error = abs(float(high_result.decrease) - target)
    if best_error <= float(relative_tolerance) * target:
        return best

    for _ in range(int(max_bisections)):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0 or mid == lo or mid == hi:
            break
        result = evaluate(mid)
        if result is None:
            lo = mid
            continue
        decrease = float(result.decrease)
        error = abs(decrease - target)
        if decrease > 0.0 and error < best_error:
            best = result
            best_error = error
        if error <= float(relative_tolerance) * target:
            return result
        if decrease >= target:
            hi = mid
        else:
            lo = mid

    rel = best_error / target
    if best is None or rel > float(relative_tolerance):
        raise RuntimeError(
            "failed to match multiscale Delta0 on native gradient; "
            f"target={target:.9g}, realised={float('nan') if best is None else float(best.decrease):.9g}, "
            f"relative_error={rel:.9g}, tolerance={float(relative_tolerance):.9g}, "
            f"hi_alpha={float(hi_alpha):.9g}"
        )
    return best


class MultiScaleCompatibilityRunner(SavedParentProbeBase):
    """Matched-kappa causal intervention repeated at several finite Delta0 scales.

    Each retained neural state must support every requested kappa and every
    declared scale fraction.  Raw same-state unrestricted directions are built
    once, and each is calibrated by bracketed bisection to

        Delta0(scale) = log-interpolation between the original v1.5 matched
        local target and the common maximum positive unrestricted decrease on
        the declared alpha grid.

    A separate common retention reference is frozen at each state x scale:

        D_ref(state, scale) = max_kappa D_unrestricted(state, scale, kappa).

    Thus update magnitude is an explicit causal axis while compatibility and the
    absolute retention allowance remain matched within each scale.
    """

    schema = "causal_compatibility_multiscale_v1"

    def __init__(
        self,
        *,
        source_run_dir: str | Path,
        run_dir: str | Path,
        system: str,
        device: torch.device,
        scale_fractions: list[float] | None = None,
        max_states: int = 50,
        max_probe_steps: int = 5000,
    ) -> None:
        super().__init__(
            source_run_dir=source_run_dir,
            run_dir=run_dir,
            system=system,
            device=device,
            max_states=max_states,
            max_probe_steps=max_probe_steps,
        )
        self.scale_fractions = sorted(float(x) for x in (scale_fractions or DEFAULT_SCALE_FRACTIONS))
        if not self.scale_fractions or any(not (0.0 <= x <= 1.0) for x in self.scale_fractions):
            raise ValueError("multi-scale expansion fractions must lie in [0,1]")
        if len(set(self.scale_fractions)) != len(self.scale_fractions):
            raise ValueError("multi-scale fractions must be unique")
        self.points = JsonlWriter(self.run_dir / "multiscale_points.jsonl")
        self.frontier = JsonlWriter(self.run_dir / "multiscale_frontier_points.jsonl")
        self.grid = JsonlWriter(self.run_dir / "multiscale_frontier_grid.jsonl")
        self.native = JsonlWriter(self.run_dir / "multiscale_afm_native_points.jsonl")
        provenance = {
            "schema": self.schema,
            "source_run_dir": str(self.source_run_dir.resolve()),
            "system": self.system,
            "seed": self.seed,
            "scale_fractions": self.scale_fractions,
            "scale_definition": "log interpolation from the original v1.5 matched local Delta0 target toward the common peak unrestricted Delta0 on the declared backtracking alpha grid",
            "scale_fraction_semantics": "0 reproduces the v1.5 local target; 1 reaches the common discrete peak; intermediate values are log-Delta0 interpolation fractions",
            "retention_reference": "separate max unrestricted protected drift across kappa at each state x scale",
            "requested_kappas": [float(x) for x in self.causal["requested_kappas"]],
            "methods": list(self.causal["methods"]),
        }
        (self.run_dir / "multiscale_config.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )

    def probe_state(
        self,
        *,
        state_index: int,
        parent_step: int,
        protected: Batch,
        novel: Batch,
        guards: Batch | None,
    ) -> int:
        requested = [float(x) for x in self.causal["requested_kappas"]]
        methods = [str(x) for x in self.causal["methods"]]
        current = make_causal_current_batch(
            modality=self.modality,
            protected=protected,
            novel=novel,
            near_count=int(self.causal["near_protected_count"]),
            seed=self.seed * 1000003 + state_index,
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
        protected_measure, Jp = functional_jacobian(self.model, protected_inputs)
        current_measure, Jc = functional_jacobian(self.model, current_inputs)
        geometry = build_protected_geometry(Jp, ridge=float(self.causal.get("geometry_ridge", 1e-7)))
        targets = make_controlled_targets(
            model=self.model,
            current_inputs=current_inputs,
            current_measurements=current_measure.detach(),
            current_jacobian=Jc.detach(),
            geometry=geometry,
            requested_kappas=requested,
            gradient_norm=float(self.causal.get("gradient_norm", 1.0)),
        )
        span = float(targets[0].achievable_max - targets[0].achievable_min)
        if span < float(self.causal.get("minimum_achievable_span", 0.5)):
            self.events.write({
                "event": "state_skipped_insufficient_compatibility_span",
                "state_index": state_index,
                "parent_step": parent_step,
                "achievable_min": targets[0].achievable_min,
                "achievable_max": targets[0].achievable_max,
            })
            self.vectoriser.assign(pre_vector)
            return 0

        # Establish two common state-level Delta0 anchors.
        #
        # LOCAL: reproduce the exact v1.5 construction -- first positive
        # unrestricted comparator for each kappa, followed by the declared
        # comparator_match_fraction of the weakest such endpoint.  This is the
        # already-audited numerically representable local regime.
        #
        # PEAK: scan the same deterministic alpha grid and take the largest
        # positive unrestricted decrease for each kappa; the weakest of these is
        # the common attainable upper anchor.
        local_rows = []
        peak_rows = []
        local_positive_decreases = []
        peak_decreases = []
        for target in targets:
            self.vectoriser.assign(pre_vector)
            try:
                local_comp = unrestricted_comparator(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    current_inputs=current_inputs,
                    current_targets=target.teacher_measurements,
                    current_gradient=target.gradient,
                    initial_alpha=float(self.causal.get("comparator_lr", 1e-2)),
                    max_backtracks=int(self.causal.get("comparator_max_backtracks", 20)),
                    backtrack_factor=float(self.causal.get("comparator_backtrack_factor", 0.5)),
                )
                peak_comp = best_positive_comparator_on_backtracking_grid(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    current_inputs=current_inputs,
                    current_targets=target.teacher_measurements,
                    current_gradient=target.gradient,
                    initial_alpha=float(self.causal.get("comparator_lr", 1e-2)),
                    max_backtracks=int(self.causal.get("comparator_max_backtracks", 20)),
                    backtrack_factor=float(self.causal.get("comparator_backtrack_factor", 0.5)),
                )
            except RuntimeError as exc:
                self.events.write({
                    "event": "state_skipped_unrestricted_multiscale_anchor",
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "requested_kappa": target.requested_kappa,
                    "reason": str(exc),
                })
                self.vectoriser.assign(pre_vector)
                return 0
            local_rows.append((target, local_comp))
            peak_rows.append((target, peak_comp))
            local_positive_decreases.append(float(local_comp.decrease))
            peak_decreases.append(float(peak_comp.decrease))

        match_fraction = float(self.causal.get("comparator_match_fraction", 0.5))
        if not (0.0 < match_fraction < 1.0):
            raise RuntimeError(f"invalid comparator_match_fraction={match_fraction}")
        local_anchor = match_fraction * min(local_positive_decreases)
        common_peak = min(peak_decreases)
        if not (0.0 < local_anchor <= common_peak):
            self.events.write({
                "event": "state_skipped_invalid_multiscale_anchor_range",
                "state_index": state_index,
                "parent_step": parent_step,
                "v15_local_delta0": local_anchor,
                "common_peak_delta0": common_peak,
            })
            self.vectoriser.assign(pre_vector)
            return 0

        match_rtol = float(self.causal.get("comparator_match_rtol", 2e-3))
        max_bisections = max(int(self.causal.get("comparator_match_bisections", 40)), 60)
        max_cv = float(self.causal.get("maximum_comparator_cv", 0.15))
        matched_by_scale: dict[float, list[tuple[Any, Any]]] = {}
        scale_targets: dict[float, float] = {}

        peak_by_requested = {float(t.requested_kappa): c for t, c in peak_rows}
        for fraction in self.scale_fractions:
            target_delta0 = multiscale_target_delta0(
                local_delta0=local_anchor,
                peak_delta0=common_peak,
                expansion_fraction=float(fraction),
            )
            scale_targets[float(fraction)] = float(target_delta0)
            rows = []
            try:
                for target, _local in local_rows:
                    self.vectoriser.assign(pre_vector)
                    peak = peak_by_requested[float(target.requested_kappa)]
                    matched = calibrate_native_gradient_to_decrease(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        current_inputs=current_inputs,
                        current_targets=target.teacher_measurements,
                        current_gradient=target.gradient,
                        vector_before=pre_vector,
                        loss_before=float(peak.loss_before),
                        hi_alpha=float(peak.alpha),
                        hi_decrease=float(peak.decrease),
                        target_decrease=target_delta0,
                        relative_tolerance=match_rtol,
                        max_bisections=max_bisections,
                    )
                    rows.append((target, matched))
            except RuntimeError as exc:
                self.events.write({
                    "event": "state_skipped_multiscale_calibration",
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "scale_fraction": fraction,
                    "scale_semantics": "log_expansion_from_v15_local_to_common_peak",
                    "target_delta0": target_delta0,
                    "v15_local_delta0": local_anchor,
                    "common_peak_delta0": common_peak,
                    "reason": str(exc),
                })
                self.vectoriser.assign(pre_vector)
                return 0
            cv = coefficient_of_variation([float(c.decrease) for _, c in rows])
            if not math.isfinite(cv) or cv > max_cv:
                self.events.write({
                    "event": "state_skipped_multiscale_comparator_mismatch",
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "scale_fraction": fraction,
                    "scale_semantics": "log_expansion_from_v15_local_to_common_peak",
                    "target_delta0": target_delta0,
                    "v15_local_delta0": local_anchor,
                    "common_peak_delta0": common_peak,
                    "cv_delta0": cv,
                    "max_allowed_cv": max_cv,
                })
                self.vectoriser.assign(pre_vector)
                return 0
            matched_by_scale[fraction] = rows

        self.events.write({
            "event": "multiscale_causal_state",
            "state_index": state_index,
            "parent_step": parent_step,
            "v15_local_delta0": local_anchor,
            "common_attainable_max_delta0": common_peak,
            "first_positive_delta0_by_kappa": {
                str(float(t.requested_kappa)): float(c.decrease) for t, c in local_rows
            },
            "peak_delta0_by_kappa": {
                str(float(t.requested_kappa)): float(c.decrease) for t, c in peak_rows
            },
            "peak_alpha_by_kappa": {
                str(float(t.requested_kappa)): float(c.alpha) for t, c in peak_rows
            },
            "scale_fractions": self.scale_fractions,
            "scale_targets_delta0": {str(float(k)): float(v) for k, v in scale_targets.items()},
            "scale_semantics": "log_expansion_from_v15_local_to_common_peak",
        })

        betas = [float(x) for x in self.causal.get("retention_budget_betas", [0, .01, .05, .1, .25, .5, 1])]
        eps = float(self.causal.get("retention_numeric_epsilon", 1e-8))
        grid_n = int(self.causal.get("retention_frontier_grid_points", 33))
        emitted = 0

        for fraction in self.scale_fractions:
            state_rows = []
            for target, comp in matched_by_scale[fraction]:
                self.vectoriser.assign(pre_vector)
                _, drift, rms = endpoint_metrics_for_vector(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    vector_after=comp.vector_after,
                    current_inputs=current_inputs,
                    current_targets=target.teacher_measurements,
                    protected_inputs=protected_inputs,
                    protected_logits_before=protected_before,
                )
                state_rows.append((target, comp, float(drift), float(rms)))
            d_ref = max(row[2] for row in state_rows)
            self.events.write({
                "event": "multiscale_retention_reference_fixed",
                "state_index": state_index,
                "parent_step": parent_step,
                "scale_fraction": fraction,
                "retention_reference_drift": d_ref,
                "delta0_target": scale_targets[fraction],
                "v15_local_delta0": local_anchor,
                "common_peak_delta0": common_peak,
                "scale_semantics": "log_expansion_from_v15_local_to_common_peak",
            })

            for target, comp, unrestricted_drift, unrestricted_rms in state_rows:
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
                self.native.write({
                    "system": self.system,
                    "dataset": str(self.sweep["dataset"]["kind"]),
                    "modality": self.modality,
                    "architecture": str(self.sweep["model"]["architecture"]),
                    "seed": self.seed,
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "scale_fraction": fraction,
                    "delta0_target": scale_targets[fraction],
                    "v15_local_delta0": local_anchor,
                    "common_peak_delta0": common_peak,
                    "target_fraction_of_common_peak": scale_targets[fraction] / common_peak,
                    "target_multiple_of_v15_local": scale_targets[fraction] / local_anchor,
                    "requested_kappa": target.requested_kappa,
                    "measured_kappa": target.measured_kappa,
                    "gradient_norm": target.gradient_norm,
                    "comparator_alpha": comp.alpha,
                    "delta0": comp.decrease,
                    "unrestricted_update_norm": float(torch.linalg.vector_norm(comp.delta).item()),
                    "retention_reference_drift": d_ref,
                    "unrestricted_protected_drift": unrestricted_drift,
                    "persistent_decrease": native.persistent_decrease,
                    "persistent_ratio": native.persistent_ratio,
                    "retention_max_abs_drift": native.protected_max_abs_drift,
                    "retention_pass": native.retention_pass,
                    "accepted": native.accepted,
                    "update_norm": native.update_norm,
                    "afm_lambda_hat": native.afm_lambda_hat,
                    "backtracking_steps": native.backtracking_steps,
                    "finite_completion_available": native.finite_completion_available,
                    "finite_endpoint_error": native.finite_endpoint_error,
                    "finite_current_error": native.finite_current_error,
                    "finite_protected_error": native.finite_protected_error,
                    "deployed_ratio": native.deployed_ratio,
                    "obstruction": native.obstruction,
                })

                for method in methods:
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
                    proposal_vector = comp.vector_before + proposal
                    proposal_loss, proposal_drift, proposal_rms = endpoint_metrics_for_vector(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        vector_after=proposal_vector,
                        current_inputs=current_inputs,
                        current_targets=target.teacher_measurements,
                        protected_inputs=protected_inputs,
                        protected_logits_before=protected_before,
                    )
                    proposal_decrease = float(comp.loss_before) - float(proposal_loss)
                    proposal_ratio = proposal_decrease / float(comp.decrease)
                    base_payload = {
                        "system": self.system,
                        "dataset": str(self.sweep["dataset"]["kind"]),
                        "modality": self.modality,
                        "architecture": str(self.sweep["model"]["architecture"]),
                        "seed": self.seed,
                        "state_index": state_index,
                        "parent_step": parent_step,
                        "scale_fraction": fraction,
                        "delta0_target": scale_targets[fraction],
                        "v15_local_delta0": local_anchor,
                        "common_peak_delta0": common_peak,
                        "target_fraction_of_common_peak": scale_targets[fraction] / common_peak,
                        "target_multiple_of_v15_local": scale_targets[fraction] / local_anchor,
                        "method": method,
                        "requested_kappa": target.requested_kappa,
                        "measured_kappa": target.measured_kappa,
                        "gradient_norm": target.gradient_norm,
                        "comparator_alpha": comp.alpha,
                        "delta0": comp.decrease,
                        "unrestricted_update_norm": float(torch.linalg.vector_norm(comp.delta).item()),
                        "unrestricted_protected_drift": unrestricted_drift,
                        "unrestricted_protected_rms_drift": unrestricted_rms,
                        "retention_reference_drift": d_ref,
                        "proposal_update_norm": float(torch.linalg.vector_norm(proposal).item()),
                        "proposal_persistent_decrease": proposal_decrease,
                        "proposal_persistent_ratio": proposal_ratio,
                        "proposal_retention_drift": proposal_drift,
                        "proposal_retention_rms_drift": proposal_rms,
                    }
                    self.points.write(base_payload)
                    emitted += 1
                    frontier, grid, audit = retention_frontier_grid(
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
                    for gp in grid:
                        self.grid.write({
                            **{k: base_payload[k] for k in (
                                "system", "dataset", "modality", "architecture", "seed", "state_index",
                                "parent_step", "scale_fraction", "delta0_target", "v15_local_delta0",
                                "common_peak_delta0", "target_fraction_of_common_peak",
                                "target_multiple_of_v15_local", "method", "requested_kappa",
                                "measured_kappa", "gradient_norm", "comparator_alpha", "delta0",
                                "unrestricted_update_norm", "unrestricted_protected_drift", "retention_reference_drift"
                            )},
                            "scale": gp.scale,
                            "update_norm": gp.update_norm,
                            "current_loss_after": gp.current_loss_after,
                            "persistent_decrease": gp.persistent_decrease,
                            "persistent_ratio": gp.persistent_ratio,
                            "retention_max_abs_drift": gp.protected_max_abs_drift,
                            "retention_rms_drift": gp.protected_rms_drift,
                        })
                    for fp in frontier:
                        self.frontier.write({
                            **{k: base_payload[k] for k in (
                                "system", "dataset", "modality", "architecture", "seed", "state_index",
                                "parent_step", "scale_fraction", "delta0_target", "v15_local_delta0",
                                "common_peak_delta0", "target_fraction_of_common_peak",
                                "target_multiple_of_v15_local", "method", "requested_kappa",
                                "measured_kappa", "gradient_norm", "comparator_alpha", "delta0",
                                "unrestricted_update_norm", "unrestricted_protected_drift", "retention_reference_drift"
                            )},
                            "retention_beta": fp.beta,
                            "retention_budget": fp.budget,
                            "frontier_scale": fp.scale,
                            "frontier_grid_points": grid_n,
                            "frontier_drift_monotone_on_grid": audit["drift_monotone_on_grid"],
                            "frontier_monotonic_drift_violations": audit["monotonic_drift_violations"],
                            "persistent_decrease": fp.persistent_decrease,
                            "persistent_ratio": fp.persistent_ratio,
                            "retention_max_abs_drift": fp.protected_max_abs_drift,
                            "retention_rms_drift": fp.protected_rms_drift,
                            "retention_pass": fp.protected_max_abs_drift <= fp.budget + 1e-12,
                            "accepted": fp.persistent_decrease > 0.0,
                            "update_norm": fp.update_norm,
                        })
        self.vectoriser.assign(pre_vector)
        return emitted

    def run(self) -> dict[str, Any]:
        start = time.time()
        traj = self.run_probe_trajectory()
        requested = [float(x) for x in self.causal["requested_kappas"]]
        methods = [str(x) for x in self.causal["methods"]]
        betas = [float(x) for x in self.causal.get("retention_budget_betas", [0, .01, .05, .1, .25, .5, 1])]
        grid_n = int(self.causal.get("retention_frontier_grid_points", 33))
        states = int(traj["accepted_states"])
        summary = {
            "schema": self.schema,
            "status": "complete" if traj["full_coverage"] else "incomplete",
            "system": self.system,
            "seed": self.seed,
            "causal_states": states,
            "probe_attempts": traj["probe_attempts"],
            "scale_fractions": self.scale_fractions,
            "scale_fraction_semantics": "log_expansion_from_v15_local_to_common_peak",
            "requested_kappas": requested,
            "methods": methods,
            "retention_budget_betas": betas,
            "multiscale_points": states * len(self.scale_fractions) * len(requested) * len(methods),
            "frontier_rows": states * len(self.scale_fractions) * len(requested) * len(methods) * len(betas),
            "grid_rows": states * len(self.scale_fractions) * len(requested) * len(methods) * grid_n,
            "afm_native_rows": states * len(self.scale_fractions) * len(requested),
            "elapsed_seconds": time.time() - start,
            "device": str(self.device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self.events.write({"event": "finished", **summary})
        for writer in (self.events, self.points, self.frontier, self.grid, self.native):
            writer.close()
        if not traj["full_coverage"]:
            raise RuntimeError(
                f"multi-scale sweep incomplete: {states}/{self.max_states} matched states after {traj['probe_attempts']} attempts"
            )
        return summary
