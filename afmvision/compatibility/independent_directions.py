from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from afmvision.compatibility.data import Batch, make_causal_current_batch
from afmvision.compatibility.extension_common import JsonlWriter, SavedParentProbeBase, coefficient_of_variation, relative_range
from afmvision.compatibility.geometry import (
    CompatibilityTarget,
    _generalised_modes,
    _native_teacher_gradient,
    build_protected_geometry,
    compatibility_fraction,
    functional_jacobian,
)
from afmvision.compatibility.methods import (
    calibrate_comparator_to_decrease,
    endpoint_metrics_for_vector,
    method_proposal_delta,
    retention_frontier_grid,
    run_method,
    unrestricted_comparator,
)


DEFAULT_INTERIOR_KAPPAS = [0.10, 0.25, 0.50, 0.75]


@dataclass
class DirectionCandidate:
    target: CompatibilityTarget
    pair_low: int
    pair_high: int
    sign_high: int
    raw_comparator: Any | None = None
    comparator: Any | None = None


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(torch.linalg.vector_norm(a).item())
    nb = float(torch.linalg.vector_norm(b).item())
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item()) / (na * nb)


def _pair_coefficients(lo: float, hi: float, target: float) -> tuple[float, float]:
    if hi <= lo:
        raise ValueError("pair eigenvalues must satisfy hi > lo")
    t = min(max(float(target), float(lo)), float(hi))
    w_hi = (t - lo) / (hi - lo)
    w_hi = min(max(w_hi, 0.0), 1.0)
    return math.sqrt(1.0 - w_hi), math.sqrt(w_hi)


def make_independent_target_candidates(
    *,
    model,
    current_inputs: torch.Tensor,
    current_measurements: torch.Tensor,
    current_jacobian: torch.Tensor,
    geometry,
    requested_kappa: float,
    gradient_norm: float,
    max_candidates: int,
    seed: int,
    kappa_tolerance: float,
) -> list[DirectionCandidate]:
    """Generate many distinct function-space directions at approximately fixed kappa.

    Generalised residual modes are G-orthonormal and diagonalise compatibility.
    Any two modes bracketing kappa can therefore be mixed with squared
    coefficients whose weighted eigenvalue equals the target Rayleigh quotient.
    Different mode pairs and signs provide genuinely different parameter-gradient
    directions without injecting arbitrary parameter vectors.
    """

    lambdas, modes = _generalised_modes(current_jacobian, geometry)
    vals = [float(x) for x in lambdas.detach().cpu().tolist()]
    lo_all, hi_all = min(vals), max(vals)
    target = float(requested_kappa)
    if target < lo_all - float(kappa_tolerance) or target > hi_all + float(kappa_tolerance):
        return []
    clipped = min(max(target, lo_all), hi_all)
    pairs = []
    for i, li in enumerate(vals):
        if li > clipped + 1e-12:
            continue
        for j, lj in enumerate(vals):
            if lj < clipped - 1e-12 or lj - li <= 1e-10:
                continue
            pairs.append((i, j, li, lj))
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    measurement_shape = current_measurements.shape
    candidates: list[DirectionCandidate] = []
    seen_axis: list[torch.Tensor] = []
    for i, j, li, lj in pairs:
        c_lo, c_hi = _pair_coefficients(li, lj, clipped)
        for sign in (1, -1):
            residual = c_lo * modes[:, i] + float(sign) * c_hi * modes[:, j]
            q64 = current_jacobian.to(dtype=torch.float64).T @ residual
            qn = float(torch.linalg.vector_norm(q64).item())
            if qn <= 1e-20:
                continue
            residual = residual * (float(gradient_norm) / qn)
            teacher = current_measurements.detach().to(dtype=torch.float64).reshape(-1) - residual
            teacher = teacher.reshape(measurement_shape).to(
                device=current_measurements.device, dtype=current_measurements.dtype
            )
            q_native = _native_teacher_gradient(model, current_inputs, teacher)
            native_norm = float(torch.linalg.vector_norm(q_native).item())
            if native_norm <= 1e-20:
                continue
            residual = residual * (float(gradient_norm) / native_norm)
            teacher = current_measurements.detach().to(dtype=torch.float64).reshape(-1) - residual
            teacher = teacher.reshape(measurement_shape).to(
                device=current_measurements.device, dtype=current_measurements.dtype
            )
            q_native = _native_teacher_gradient(model, current_inputs, teacher)
            native_norm = float(torch.linalg.vector_norm(q_native).item())
            measured = compatibility_fraction(q_native, geometry)
            if abs(measured - target) > float(kappa_tolerance):
                continue
            # Do not store numerically duplicate gradient axes from different
            # algebraic pairs.  This is only a generation filter; the stricter
            # pairwise cosine requirement is applied after Delta0 matching.
            duplicate = any(abs(_cosine(q_native, q)) > 0.999999 for q in seen_axis)
            if duplicate:
                continue
            seen_axis.append(q_native.detach().clone())
            ct = CompatibilityTarget(
                requested_kappa=target,
                measured_kappa=measured,
                achievable_min=lo_all,
                achievable_max=hi_all,
                clipped=abs(clipped - target) > 1e-9,
                residual=residual.to(device=current_measurements.device, dtype=current_measurements.dtype),
                teacher_measurements=teacher,
                gradient=q_native,
                gradient_norm=native_norm,
                low_mode_kappa=li,
                high_mode_kappa=lj,
            )
            candidates.append(DirectionCandidate(ct, i, j, sign))
            if len(candidates) >= int(max_candidates):
                return candidates
    return candidates


def select_matched_diverse_candidates(
    candidates: list[DirectionCandidate],
    *,
    count: int,
    update_norm_rtol: float,
    max_abs_cosine: float,
) -> list[DirectionCandidate]:
    """Select a direction set matched on comparator update norm and diverse in angle."""

    usable = [c for c in candidates if c.comparator is not None]
    if len(usable) < int(count):
        return []
    usable = sorted(usable, key=lambda c: float(torch.linalg.vector_norm(c.comparator.delta).item()))
    best: list[DirectionCandidate] = []
    best_spread = float("inf")
    for anchor in usable:
        anchor_norm = float(torch.linalg.vector_norm(anchor.comparator.delta).item())
        if anchor_norm <= 0.0:
            continue
        eligible = [
            c for c in usable
            if abs(float(torch.linalg.vector_norm(c.comparator.delta).item()) - anchor_norm)
            <= float(update_norm_rtol) * anchor_norm
        ]
        eligible.sort(
            key=lambda c: abs(float(torch.linalg.vector_norm(c.comparator.delta).item()) - anchor_norm)
        )
        selected: list[DirectionCandidate] = []
        for cand in eligible:
            if all(abs(_cosine(cand.target.gradient, prev.target.gradient)) <= float(max_abs_cosine) for prev in selected):
                selected.append(cand)
            if len(selected) >= int(count):
                break
        if len(selected) < int(count):
            continue
        norms = [float(torch.linalg.vector_norm(c.comparator.delta).item()) for c in selected]
        spread = relative_range(norms)
        if spread < best_spread:
            best = selected
            best_spread = spread
    return best


class IndependentDirectionRunner(SavedParentProbeBase):
    """Fixed-compatibility test across several genuinely different directions."""

    schema = "causal_compatibility_independent_directions_v1"

    def __init__(
        self,
        *,
        source_run_dir: str | Path,
        run_dir: str | Path,
        system: str,
        device: torch.device,
        requested_kappas: list[float] | None = None,
        directions_per_kappa: int = 4,
        candidate_pool: int = 24,
        kappa_tolerance: float = 0.01,
        update_norm_rtol: float = 0.05,
        max_abs_cosine: float = 0.95,
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
        self.requested_kappas = [float(x) for x in (requested_kappas or DEFAULT_INTERIOR_KAPPAS)]
        if not self.requested_kappas or any(not (0.0 < x < 1.0) for x in self.requested_kappas):
            raise ValueError("independent-direction test defaults to interior kappa values strictly between 0 and 1")
        self.directions_per_kappa = int(directions_per_kappa)
        self.candidate_pool = int(candidate_pool)
        self.kappa_tolerance = float(kappa_tolerance)
        self.update_norm_rtol = float(update_norm_rtol)
        self.max_abs_cosine = float(max_abs_cosine)
        if self.directions_per_kappa < 2:
            raise ValueError("directions_per_kappa must be >=2")
        if self.candidate_pool < self.directions_per_kappa:
            raise ValueError("candidate_pool must be >= directions_per_kappa")
        self.points = JsonlWriter(self.run_dir / "independent_direction_points.jsonl")
        self.frontier = JsonlWriter(self.run_dir / "independent_direction_frontier_points.jsonl")
        self.grid = JsonlWriter(self.run_dir / "independent_direction_frontier_grid.jsonl")
        self.native = JsonlWriter(self.run_dir / "independent_direction_afm_native_points.jsonl")
        (self.run_dir / "independent_direction_config.json").write_text(
            json.dumps({
                "schema": self.schema,
                "source_run_dir": str(self.source_run_dir.resolve()),
                "system": self.system,
                "seed": self.seed,
                "requested_kappas": self.requested_kappas,
                "directions_per_kappa": self.directions_per_kappa,
                "candidate_pool": self.candidate_pool,
                "kappa_tolerance": self.kappa_tolerance,
                "update_norm_rtol": self.update_norm_rtol,
                "max_abs_cosine": self.max_abs_cosine,
                "delta0_match_fraction": float(self.causal.get("comparator_match_fraction", 0.5)),
                "retention_reference": "max unrestricted drift across selected directions at fixed state x kappa",
            }, indent=2, sort_keys=True), encoding="utf-8"
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
        _, Jp = functional_jacobian(self.model, protected_inputs)
        current_measure, Jc = functional_jacobian(self.model, current_inputs)
        geometry = build_protected_geometry(Jp, ridge=float(self.causal.get("geometry_ridge", 1e-7)))

        gradient_norm_target = float(self.causal.get("gradient_norm", 1.0))
        delta_fraction = float(self.causal.get("comparator_match_fraction", 0.5))
        delta_rtol = float(self.causal.get("comparator_match_rtol", 2e-3))
        max_bisect = int(self.causal.get("comparator_match_bisections", 40))
        selected_by_kappa: dict[float, list[DirectionCandidate]] = {}

        for kidx, kappa in enumerate(self.requested_kappas):
            candidates = make_independent_target_candidates(
                model=self.model,
                current_inputs=current_inputs,
                current_measurements=current_measure.detach(),
                current_jacobian=Jc.detach(),
                geometry=geometry,
                requested_kappa=kappa,
                gradient_norm=gradient_norm_target,
                max_candidates=self.candidate_pool,
                seed=self.seed * 10000019 + state_index * 1009 + kidx,
                kappa_tolerance=self.kappa_tolerance,
            )
            if len(candidates) < self.directions_per_kappa:
                self.events.write({
                    "event": "state_skipped_insufficient_direction_candidates",
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "requested_kappa": kappa,
                    "candidates": len(candidates),
                })
                self.vectoriser.assign(pre_vector)
                return 0
            raw_decreases = []
            viable: list[DirectionCandidate] = []
            for cand in candidates:
                self.vectoriser.assign(pre_vector)
                try:
                    raw = unrestricted_comparator(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        current_inputs=current_inputs,
                        current_targets=cand.target.teacher_measurements,
                        current_gradient=cand.target.gradient,
                        initial_alpha=float(self.causal.get("comparator_lr", 1e-2)),
                        max_backtracks=int(self.causal.get("comparator_max_backtracks", 20)),
                        backtrack_factor=float(self.causal.get("comparator_backtrack_factor", 0.5)),
                    )
                except RuntimeError:
                    continue
                cand.raw_comparator = raw
                raw_decreases.append(float(raw.decrease))
                viable.append(cand)
            if len(viable) < self.directions_per_kappa:
                self.events.write({
                    "event": "state_skipped_insufficient_positive_direction_comparators",
                    "state_index": state_index,
                    "requested_kappa": kappa,
                    "viable": len(viable),
                })
                self.vectoriser.assign(pre_vector)
                return 0
            target_delta0 = delta_fraction * min(raw_decreases)
            calibrated: list[DirectionCandidate] = []
            for cand in viable:
                self.vectoriser.assign(pre_vector)
                try:
                    cand.comparator = calibrate_comparator_to_decrease(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        current_inputs=current_inputs,
                        current_targets=cand.target.teacher_measurements,
                        base=cand.raw_comparator,
                        target_decrease=target_delta0,
                        relative_tolerance=delta_rtol,
                        max_bisections=max_bisect,
                    )
                    calibrated.append(cand)
                except RuntimeError:
                    continue
            selected = select_matched_diverse_candidates(
                calibrated,
                count=self.directions_per_kappa,
                update_norm_rtol=self.update_norm_rtol,
                max_abs_cosine=self.max_abs_cosine,
            )
            if len(selected) != self.directions_per_kappa:
                self.events.write({
                    "event": "state_skipped_direction_matching",
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "requested_kappa": kappa,
                    "calibrated": len(calibrated),
                    "required": self.directions_per_kappa,
                    "update_norm_rtol": self.update_norm_rtol,
                    "max_abs_cosine": self.max_abs_cosine,
                })
                self.vectoriser.assign(pre_vector)
                return 0
            delta_cv = coefficient_of_variation([float(c.comparator.decrease) for c in selected])
            update_rr = relative_range([float(torch.linalg.vector_norm(c.comparator.delta).item()) for c in selected])
            max_pair_cos = max(
                abs(_cosine(selected[i].target.gradient, selected[j].target.gradient))
                for i in range(len(selected)) for j in range(i + 1, len(selected))
            )
            if delta_cv > max(float(self.causal.get("maximum_comparator_cv", .15)), delta_rtol * 2.0):
                self.vectoriser.assign(pre_vector)
                return 0
            self.events.write({
                "event": "direction_set_matched",
                "state_index": state_index,
                "parent_step": parent_step,
                "requested_kappa": kappa,
                "target_delta0": target_delta0,
                "delta0_cv": delta_cv,
                "update_norm_relative_range": update_rr,
                "max_pair_abs_cosine": max_pair_cos,
                "measured_kappas": [c.target.measured_kappa for c in selected],
            })
            selected_by_kappa[kappa] = selected

        betas = [float(x) for x in self.causal.get("retention_budget_betas", [0, .01, .05, .1, .25, .5, 1])]
        eps = float(self.causal.get("retention_numeric_epsilon", 1e-8))
        grid_n = int(self.causal.get("retention_frontier_grid_points", 33))
        emitted = 0

        for kappa, selected in selected_by_kappa.items():
            rows = []
            for cand in selected:
                comp = cand.comparator
                self.vectoriser.assign(pre_vector)
                _, drift, rms = endpoint_metrics_for_vector(
                    model=self.model,
                    vectoriser=self.vectoriser,
                    vector_after=comp.vector_after,
                    current_inputs=current_inputs,
                    current_targets=cand.target.teacher_measurements,
                    protected_inputs=protected_inputs,
                    protected_logits_before=protected_before,
                )
                rows.append((cand, float(drift), float(rms)))
            d_ref = max(r[1] for r in rows)

            for direction_id, (cand, unrestricted_drift, unrestricted_rms) in enumerate(rows):
                target = cand.target
                comp = cand.comparator
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
                direction_update_norm = float(torch.linalg.vector_norm(comp.delta).item())
                self.native.write({
                    "system": self.system,
                    "seed": self.seed,
                    "state_index": state_index,
                    "parent_step": parent_step,
                    "requested_kappa": kappa,
                    "direction_id": direction_id,
                    "measured_kappa": target.measured_kappa,
                    "gradient_norm": target.gradient_norm,
                    "delta0": comp.decrease,
                    "comparator_alpha": comp.alpha,
                    "unrestricted_update_norm": direction_update_norm,
                    "unrestricted_protected_drift": unrestricted_drift,
                    "retention_reference_drift": d_ref,
                    "pair_low_mode": cand.pair_low,
                    "pair_high_mode": cand.pair_high,
                    "pair_high_sign": cand.sign_high,
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
                    base = {
                        "system": self.system,
                        "dataset": str(self.sweep["dataset"]["kind"]),
                        "modality": self.modality,
                        "architecture": str(self.sweep["model"]["architecture"]),
                        "seed": self.seed,
                        "state_index": state_index,
                        "parent_step": parent_step,
                        "requested_kappa": kappa,
                        "direction_id": direction_id,
                        "measured_kappa": target.measured_kappa,
                        "gradient_norm": target.gradient_norm,
                        "method": method,
                        "delta0": comp.decrease,
                        "comparator_alpha": comp.alpha,
                        "unrestricted_update_norm": direction_update_norm,
                        "unrestricted_protected_drift": unrestricted_drift,
                        "retention_reference_drift": d_ref,
                        "pair_low_mode": cand.pair_low,
                        "pair_high_mode": cand.pair_high,
                        "pair_high_sign": cand.sign_high,
                    }
                    self.points.write({**base, "proposal_update_norm": float(torch.linalg.vector_norm(proposal).item())})
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
                            **base,
                            "scale": gp.scale,
                            "update_norm": gp.update_norm,
                            "persistent_decrease": gp.persistent_decrease,
                            "persistent_ratio": gp.persistent_ratio,
                            "retention_max_abs_drift": gp.protected_max_abs_drift,
                            "retention_rms_drift": gp.protected_rms_drift,
                        })
                    for fp in frontier:
                        self.frontier.write({
                            **base,
                            "retention_beta": fp.beta,
                            "retention_budget": fp.budget,
                            "frontier_scale": fp.scale,
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
            "requested_kappas": self.requested_kappas,
            "directions_per_kappa": self.directions_per_kappa,
            "methods": methods,
            "retention_budget_betas": betas,
            "point_rows": states * len(self.requested_kappas) * self.directions_per_kappa * len(methods),
            "frontier_rows": states * len(self.requested_kappas) * self.directions_per_kappa * len(methods) * len(betas),
            "grid_rows": states * len(self.requested_kappas) * self.directions_per_kappa * len(methods) * grid_n,
            "afm_native_rows": states * len(self.requested_kappas) * self.directions_per_kappa,
            "elapsed_seconds": time.time() - start,
            "device": str(self.device),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self.events.write({"event": "finished", **summary})
        for writer in (self.events, self.points, self.frontier, self.grid, self.native):
            writer.close()
        if not traj["full_coverage"]:
            raise RuntimeError(
                f"independent-direction sweep incomplete: {states}/{self.max_states} states after {traj['probe_attempts']} attempts"
            )
        return summary
