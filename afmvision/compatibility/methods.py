from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from afmvision.afm.parameter_vector import ParameterVector, temporary_parameters
from afmvision.compatibility.geometry import (
    ProtectedGeometry,
    compatible_projection,
    teacher_loss,
)
from afmvision.compatibility.models import CompatibilityModel
from afmvision.compatibility.shield import StaticAddressEncoder, finite_endpoint_completion


@dataclass(frozen=True)
class ComparatorResult:
    alpha: float
    vector_before: torch.Tensor
    vector_after: torch.Tensor
    delta: torch.Tensor
    loss_before: float
    loss_after: float
    decrease: float
    current_full_logits_after: torch.Tensor


@dataclass(frozen=True)
class MethodResult:
    method: str
    accepted: bool
    vector_after: torch.Tensor
    update_norm: float
    current_loss_after: float
    persistent_decrease: float
    persistent_ratio: float
    protected_max_abs_drift: float
    protected_rms_drift: float
    retention_pass: bool
    deployed_ratio: float | None = None
    finite_completion_available: bool | None = None
    finite_endpoint_error: float | None = None
    finite_current_error: float | None = None
    finite_protected_error: float | None = None
    obstruction: str | None = None
    backtracking_steps: int = 0
    afm_lambda_hat: float | None = None



def _teacher_loss_value(
    model: CompatibilityModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Evaluate teacher loss with float64 accumulation for line-search decisions.

    Model execution remains in its configured/native precision.  Only the scalar
    reduction is accumulated in float64 so a genuine small decrease is not lost
    to float32 summation when teacher residuals are large.
    """

    with torch.no_grad():
        values = model.functional_logits(inputs)
        diff = (values - targets).to(dtype=torch.float64)
        return float((0.5 * diff.square().sum()).item())


def full_logits_and_loss(
    model: CompatibilityModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        logits = model(inputs)
    loss = _teacher_loss_value(model, inputs, targets)
    return logits, loss


def _endpoint_metrics(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    vector_after: torch.Tensor,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_logits_before: torch.Tensor,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    with temporary_parameters(vectoriser, vector_after):
        with torch.no_grad():
            current_full = model(current_inputs)
            protected_full = model(protected_inputs)
            drift = protected_full - protected_logits_before
            max_abs = float(drift.abs().max().item()) if drift.numel() else 0.0
            rms = float(torch.sqrt(torch.mean(drift.square())).item()) if drift.numel() else 0.0
        current_loss = _teacher_loss_value(model, current_inputs, current_targets)
    return current_loss, max_abs, rms, current_full, protected_full


def calibrate_comparator_to_decrease(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    base: ComparatorResult,
    target_decrease: float,
    relative_tolerance: float = 2.0e-3,
    max_bisections: int = 40,
) -> ComparatorResult:
    """Calibrate a genuine same-state comparator to a common finite decrease.

    ``base`` supplies a positive endpoint along the native unrestricted descent
    direction.  Because loss decrease is zero at alpha=0 and at least
    ``target_decrease`` at ``base.alpha``, continuity guarantees a crossing in
    that interval.  A bracketed bisection therefore matches the *actual finite*
    no-protection decrease without redefining Delta_0 or changing the gradient
    direction.  This is the causal normalization required by the compatibility
    sweep: gradient norm is matched in target construction, while finite
    unrestricted progress is matched here.
    """

    target = float(target_decrease)
    if not (target > 0.0):
        raise ValueError(f"target_decrease must be positive, got {target!r}")
    if float(base.decrease) + 1e-18 < target:
        raise ValueError(
            f"base comparator decrease {base.decrease:.9g} is below requested target {target:.9g}"
        )

    vector_before = base.vector_before
    direction = base.delta / float(base.alpha)
    loss_before = float(base.loss_before)
    lo = 0.0
    hi = float(base.alpha)
    best = base
    best_error = abs(float(base.decrease) - target)

    for _ in range(int(max_bisections)):
        alpha = 0.5 * (lo + hi)
        if alpha <= 0.0:
            break
        delta = alpha * direction
        candidate = vector_before + delta
        if torch.equal(candidate, vector_before):
            lo = alpha
            continue
        with temporary_parameters(vectoriser, candidate):
            loss_after = _teacher_loss_value(model, current_inputs, current_targets)
            with torch.no_grad():
                current_full = model(current_inputs).detach().clone()
        decrease = loss_before - loss_after
        error = abs(decrease - target)
        if decrease > 0.0 and error < best_error:
            best = ComparatorResult(
                alpha=float(alpha),
                vector_before=vector_before,
                vector_after=candidate,
                delta=candidate - vector_before,
                loss_before=loss_before,
                loss_after=float(loss_after),
                decrease=float(decrease),
                current_full_logits_after=current_full,
            )
            best_error = error
        if error <= float(relative_tolerance) * target:
            return best
        if decrease >= target:
            hi = alpha
        else:
            lo = alpha

    relative_error = best_error / target
    if relative_error > float(relative_tolerance):
        raise RuntimeError(
            "failed to match same-state unrestricted decrease by bracketed bisection; "
            f"target={target:.9g}, realised={best.decrease:.9g}, "
            f"relative_error={relative_error:.9g}, tolerance={float(relative_tolerance):.9g}"
        )
    return best


def unrestricted_comparator(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    current_gradient: torch.Tensor,
    initial_alpha: float,
    max_backtracks: int,
    backtrack_factor: float,
    min_decrease: float = 0.0,
) -> ComparatorResult:
    """Construct the genuine same-state unrestricted endpoint by backtracking.

    ``current_gradient`` is the realised native-precision teacher-loss gradient.
    The search therefore follows a genuine descent direction at the same state.
    The line search is allowed to continue until either a positive finite
    decrease is observed or parameter precision makes the next candidate
    identical to the current parameter vector.
    """

    vector_before = vectoriser.flatten(detach=True)
    loss_before = _teacher_loss_value(model, current_inputs, current_targets)
    alpha = float(initial_alpha)
    selected = None
    smallest_alpha = alpha
    last_loss = loss_before
    last_update_norm = 0.0

    for attempt in range(int(max_backtracks) + 1):
        delta = -alpha * current_gradient
        candidate = vector_before + delta
        realised_delta = candidate - vector_before
        last_update_norm = float(torch.linalg.vector_norm(realised_delta).item())
        smallest_alpha = alpha

        # Once the candidate is bitwise identical to the current vector, further
        # backtracking cannot produce a distinct float parameter endpoint.
        if torch.equal(candidate, vector_before):
            break

        with temporary_parameters(vectoriser, candidate):
            loss_after = _teacher_loss_value(model, current_inputs, current_targets)
            with torch.no_grad():
                current_full = model(current_inputs).detach().clone()
        last_loss = loss_after
        decrease = loss_before - loss_after
        if decrease > float(min_decrease):
            selected = (candidate, realised_delta, loss_after, decrease, current_full)
            break
        alpha *= float(backtrack_factor)

    if selected is None:
        raise RuntimeError(
            "same-state unrestricted comparator failed to produce positive decrease; "
            f"loss_before={loss_before:.17g}, last_loss={last_loss:.17g}, "
            f"initial_alpha={float(initial_alpha):.9g}, smallest_alpha={smallest_alpha:.9g}, "
            f"gradient_norm={float(torch.linalg.vector_norm(current_gradient).item()):.9g}, "
            f"last_update_norm={last_update_norm:.9g}, max_backtracks={int(max_backtracks)}"
        )

    candidate, delta, loss_after, decrease, current_full = selected
    return ComparatorResult(
        alpha=alpha,
        vector_before=vector_before,
        vector_after=candidate,
        delta=delta,
        loss_before=loss_before,
        loss_after=loss_after,
        decrease=decrease,
        current_full_logits_after=current_full,
    )


def _gradient_for_ce(
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    return vectoriser.flatten_grads().detach().clone()


def _gradient_for_logit_mse(
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    inputs: torch.Tensor,
    stored_logits: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = 0.5 * torch.nn.functional.mse_loss(logits, stored_logits, reduction="sum")
    loss.backward()
    return vectoriser.flatten_grads().detach().clone()


def fisher_diagonal(
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    grad = _gradient_for_ce(model, vectoriser, inputs, labels)
    return grad.square()


def linearised_distillation_direction(
    current_gradient: torch.Tensor,
    geometry: ProtectedGeometry,
    *,
    alpha: float,
    strength: float,
) -> torch.Tensor:
    """Return (I + alpha*lambda*Jp^T Jp)^-1 q via Woodbury."""
    lam = float(strength)
    if lam <= 0.0 or geometry.jacobian.numel() == 0:
        return current_gradient
    J = geometry.jacobian.to(device=current_gradient.device, dtype=current_gradient.dtype)
    scale = float(alpha) * lam
    gram = J @ J.T
    system = gram + (1.0 / scale) * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    correction = J.T @ torch.linalg.solve(system, J @ current_gradient)
    return current_gradient - correction


def _result_from_delta(
    *,
    method: str,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    comparator: ComparatorResult,
    delta: torch.Tensor,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_logits_before: torch.Tensor,
    retention_tolerance: float,
    accepted: bool = True,
    obstruction: str | None = None,
    backtracking_steps: int = 0,
) -> MethodResult:
    vector_after = comparator.vector_before + delta
    current_loss, max_abs, rms, _, _ = _endpoint_metrics(
        model=model,
        vectoriser=vectoriser,
        vector_after=vector_after,
        current_inputs=current_inputs,
        current_targets=current_targets,
        protected_inputs=protected_inputs,
        protected_logits_before=protected_logits_before,
    )
    decrease = comparator.loss_before - current_loss
    ratio = decrease / comparator.decrease if comparator.decrease > 0 else float("nan")
    return MethodResult(
        method=method,
        accepted=bool(accepted),
        vector_after=vector_after,
        update_norm=float(torch.linalg.vector_norm(delta).item()),
        current_loss_after=current_loss,
        persistent_decrease=decrease,
        persistent_ratio=ratio,
        protected_max_abs_drift=max_abs,
        protected_rms_drift=rms,
        retention_pass=max_abs <= float(retention_tolerance),
        obstruction=obstruction,
        backtracking_steps=int(backtracking_steps),
    )


def run_method(
    *,
    method: str,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    comparator: ComparatorResult,
    current_gradient: torch.Tensor,
    geometry: ProtectedGeometry,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_labels: torch.Tensor,
    protected_logits_before: torch.Tensor,
    replay_stored_logits: torch.Tensor,
    guard_inputs: torch.Tensor | None,
    retention_tolerance: float,
    method_config: dict,
    address_encoder: StaticAddressEncoder,
) -> MethodResult:
    method = str(method)
    alpha = float(comparator.alpha)

    if method == "unrestricted":
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=comparator.delta,
            current_inputs=current_inputs,
            current_targets=current_targets,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
        )

    if method == "projection":
        direction = compatible_projection(current_gradient, geometry)
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=-alpha * direction,
            current_inputs=current_inputs,
            current_targets=current_targets,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
        )

    if method == "replay":
        replay_weight = float(method_config.get("replay_weight", 1.0))
        g_replay = _gradient_for_ce(model, vectoriser, protected_inputs, protected_labels)
        direction = current_gradient + replay_weight * g_replay
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=-alpha * direction,
            current_inputs=current_inputs,
            current_targets=current_targets,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
        )

    if method == "derpp":
        alpha_der = float(method_config.get("derpp_alpha", 0.5))
        beta_der = float(method_config.get("derpp_beta", 0.5))
        g_logits = _gradient_for_logit_mse(model, vectoriser, protected_inputs, replay_stored_logits)
        g_labels = _gradient_for_ce(model, vectoriser, protected_inputs, protected_labels)
        direction = current_gradient + alpha_der * g_logits + beta_der * g_labels
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=-alpha * direction,
            current_inputs=current_inputs,
            current_targets=current_targets,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
        )

    if method == "linearized_distillation":
        direction = linearised_distillation_direction(
            current_gradient,
            geometry,
            alpha=alpha,
            strength=float(method_config.get("distillation_strength", 10.0)),
        )
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=-alpha * direction,
            current_inputs=current_inputs,
            current_targets=current_targets,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
        )

    if method == "ewc_prox":
        strength = float(method_config.get("ewc_strength", 10.0))
        fisher = fisher_diagonal(model, vectoriser, protected_inputs, protected_labels)
        direction = current_gradient / (1.0 + alpha * strength * fisher)
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=-alpha * direction,
            current_inputs=current_inputs,
            current_targets=current_targets,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
        )

    if method == "afm":
        direction = compatible_projection(current_gradient, geometry)
        factor = 1.0
        backtrack_factor = float(method_config.get("afm_backtrack_factor", 0.5))
        max_backtracks = int(method_config.get("afm_max_backtracks", 12))
        chosen: MethodResult | None = None
        for attempt in range(max_backtracks + 1):
            delta = -alpha * factor * direction
            candidate = _result_from_delta(
                method=method,
                model=model,
                vectoriser=vectoriser,
                comparator=comparator,
                delta=delta,
                current_inputs=current_inputs,
                current_targets=current_targets,
                protected_inputs=protected_inputs,
                protected_logits_before=protected_logits_before,
                retention_tolerance=retention_tolerance,
                backtracking_steps=attempt,
            )
            if candidate.retention_pass and candidate.persistent_decrease > 0.0:
                chosen = candidate
                break
            factor *= backtrack_factor
        if chosen is None:
            zero = torch.zeros_like(current_gradient)
            return _result_from_delta(
                method=method,
                model=model,
                vectoriser=vectoriser,
                comparator=comparator,
                delta=zero,
                current_inputs=current_inputs,
                current_targets=current_targets,
                protected_inputs=protected_inputs,
                protected_logits_before=protected_logits_before,
                retention_tolerance=retention_tolerance,
                accepted=False,
                obstruction="persistent_retention_backtracking_exhausted",
                backtracking_steps=max_backtracks,
            )

        with temporary_parameters(vectoriser, chosen.vector_after):
            with torch.no_grad():
                base_current = model(current_inputs).detach().clone()
                base_protected = model(protected_inputs).detach().clone()
            finite = finite_endpoint_completion(
                address_encoder=address_encoder,
                base_current_logits=base_current,
                base_protected_logits=base_protected,
                desired_current_logits=comparator.current_full_logits_after,
                desired_protected_logits=protected_logits_before,
                current_inputs=current_inputs,
                protected_inputs=protected_inputs,
                guard_inputs=guard_inputs,
                support_multiplier=float(method_config.get("shield_support_multiplier", 4.0)),
                feature_match_tolerance=float(method_config.get("shield_feature_match_tolerance", 1e-8)),
                target_tolerance=float(method_config.get("shield_target_tolerance", 1e-8)),
                residual_tolerance=float(method_config.get("shield_residual_tolerance", 1e-8)),
            )
            if finite.available and finite.deployed_current_logits is not None:
                current_measure = finite.deployed_current_logits
                if hasattr(model, "functional_projection"):
                    projection = getattr(model, "functional_projection")
                    current_measure = current_measure @ projection
                deployed_diff = (current_measure - current_targets).to(dtype=torch.float64)
                deployed_loss = float((0.5 * deployed_diff.square().sum()).item())
                deployed_decrease = comparator.loss_before - deployed_loss
                deployed_ratio = deployed_decrease / comparator.decrease if comparator.decrease > 0 else float("nan")
            else:
                deployed_ratio = None

        return MethodResult(
            method=chosen.method,
            accepted=chosen.accepted,
            vector_after=chosen.vector_after,
            update_norm=chosen.update_norm,
            current_loss_after=chosen.current_loss_after,
            persistent_decrease=chosen.persistent_decrease,
            persistent_ratio=chosen.persistent_ratio,
            protected_max_abs_drift=chosen.protected_max_abs_drift,
            protected_rms_drift=chosen.protected_rms_drift,
            retention_pass=chosen.retention_pass,
            deployed_ratio=deployed_ratio,
            finite_completion_available=finite.available,
            finite_endpoint_error=finite.endpoint_error,
            finite_current_error=finite.current_error,
            finite_protected_error=finite.protected_error,
            obstruction=finite.solve.obstruction if not finite.available else None,
            backtracking_steps=chosen.backtracking_steps,
            afm_lambda_hat=float(backtrack_factor ** chosen.backtracking_steps),
        )

    raise ValueError(f"unknown compatibility method: {method}")


@dataclass(frozen=True)
class RetentionFrontierPoint:
    beta: float
    budget: float
    scale: float
    vector_after: torch.Tensor
    update_norm: float
    current_loss_after: float
    persistent_decrease: float
    persistent_ratio: float
    protected_max_abs_drift: float
    protected_rms_drift: float


@dataclass(frozen=True)
class RetentionGridPoint:
    scale: float
    update_norm: float
    current_loss_after: float
    persistent_decrease: float
    persistent_ratio: float
    protected_max_abs_drift: float
    protected_rms_drift: float


@dataclass(frozen=True)
class FiniteCompletionResult:
    available: bool
    deployed_ratio: float | None
    endpoint_error: float | None
    current_error: float | None
    protected_error: float | None
    obstruction: str | None


def endpoint_metrics_for_vector(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    vector_after: torch.Tensor,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_logits_before: torch.Tensor,
) -> tuple[float, float, float]:
    current_loss, max_abs, rms, _, _ = _endpoint_metrics(
        model=model,
        vectoriser=vectoriser,
        vector_after=vector_after,
        current_inputs=current_inputs,
        current_targets=current_targets,
        protected_inputs=protected_inputs,
        protected_logits_before=protected_logits_before,
    )
    return current_loss, max_abs, rms


def method_proposal_delta(
    *,
    method: str,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    comparator: ComparatorResult,
    current_gradient: torch.Tensor,
    geometry: ProtectedGeometry,
    protected_inputs: torch.Tensor,
    protected_labels: torch.Tensor,
    replay_stored_logits: torch.Tensor,
    method_config: dict,
) -> torch.Tensor:
    """Return the method's native one-step proposal before a common retention cap.

    The causal-frontier experiment applies one *method-neutral* scalar retention
    constraint after this proposal is constructed.  Thus the compatibility
    intervention, method proposal, and retention budget remain separate axes.
    AFM's persistent proposal is its compatible projected base direction; finite
    endpoint completion is audited separately and is not folded into this vector.
    """

    method = str(method)
    alpha = float(comparator.alpha)
    if method == "unrestricted":
        return comparator.delta.detach().clone()
    if method in {"projection", "afm"}:
        direction = compatible_projection(current_gradient, geometry)
        return (-alpha * direction).detach().clone()
    if method == "replay":
        replay_weight = float(method_config.get("replay_weight", 1.0))
        g_replay = _gradient_for_ce(model, vectoriser, protected_inputs, protected_labels)
        return (-alpha * (current_gradient + replay_weight * g_replay)).detach().clone()
    if method == "derpp":
        alpha_der = float(method_config.get("derpp_alpha", 0.5))
        beta_der = float(method_config.get("derpp_beta", 0.5))
        g_logits = _gradient_for_logit_mse(model, vectoriser, protected_inputs, replay_stored_logits)
        g_labels = _gradient_for_ce(model, vectoriser, protected_inputs, protected_labels)
        return (-alpha * (current_gradient + alpha_der * g_logits + beta_der * g_labels)).detach().clone()
    if method == "linearized_distillation":
        direction = linearised_distillation_direction(
            current_gradient,
            geometry,
            alpha=alpha,
            strength=float(method_config.get("distillation_strength", 10.0)),
        )
        return (-alpha * direction).detach().clone()
    if method == "ewc_prox":
        strength = float(method_config.get("ewc_strength", 10.0))
        fisher = fisher_diagonal(model, vectoriser, protected_inputs, protected_labels)
        direction = current_gradient / (1.0 + alpha * strength * fisher)
        return (-alpha * direction).detach().clone()
    raise ValueError(f"unknown compatibility method: {method}")


def retention_frontier_grid(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    comparator: ComparatorResult,
    proposal_delta: torch.Tensor,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_logits_before: torch.Tensor,
    retention_reference_drift: float,
    betas: list[float],
    epsilon_num: float,
    grid_points: int = 33,
) -> tuple[list[RetentionFrontierPoint], list[RetentionGridPoint], dict[str, float | int | bool]]:
    """Evaluate a finite common-budget retention frontier for one proposal.

    ``retention_reference_drift`` is fixed *once per matched causal state* and
    shared by every requested compatibility level and every method.  v1.5 uses
    the maximum protected drift of the six matched unrestricted comparators,

        D_ref = max_kappa D_unrestricted(kappa),

    so each beta defines one identical absolute functional-retention allowance

        D <= max(epsilon_num, beta * D_ref)

    across the entire compatibility intervention.  This avoids the v1.4 error
    in which the allowed budget changed with kappa and therefore partially
    normalized away the causal manipulation.

    All ``grid_points`` actual nonlinear endpoints are returned as an audit
    table.  Frontier selection is therefore reproducible later without another
    GPU probe if alternative predeclared beta summaries are desired.
    """

    n = int(grid_points)
    if n < 2:
        raise ValueError("retention frontier grid_points must be >= 2")
    eps = float(epsilon_num)
    if eps < 0.0:
        raise ValueError("retention epsilon must be non-negative")
    beta_values = sorted(float(x) for x in betas)
    if any(x < 0.0 for x in beta_values):
        raise ValueError("retention beta values must be non-negative")

    d_ref = max(float(retention_reference_drift), 0.0)
    base_vector = comparator.vector_before
    delta_norm = float(torch.linalg.vector_norm(proposal_delta).item())
    evaluations: list[tuple[float, torch.Tensor, float, float, float, float, float]] = []
    grid: list[RetentionGridPoint] = []
    for i in range(n):
        scale = float(i) / float(n - 1)
        if i == 0 or delta_norm == 0.0:
            candidate = base_vector
            current_loss = float(comparator.loss_before)
            max_abs = 0.0
            rms = 0.0
        else:
            candidate = base_vector + scale * proposal_delta
            current_loss, max_abs, rms = endpoint_metrics_for_vector(
                model=model,
                vectoriser=vectoriser,
                vector_after=candidate,
                current_inputs=current_inputs,
                current_targets=current_targets,
                protected_inputs=protected_inputs,
                protected_logits_before=protected_logits_before,
            )
        decrease = float(comparator.loss_before) - float(current_loss)
        ratio = decrease / float(comparator.decrease) if comparator.decrease > 0 else float("nan")
        evaluations.append((scale, candidate, current_loss, decrease, ratio, max_abs, rms))
        grid.append(
            RetentionGridPoint(
                scale=scale,
                update_norm=scale * delta_norm,
                current_loss_after=float(current_loss),
                persistent_decrease=float(decrease),
                persistent_ratio=float(ratio),
                protected_max_abs_drift=float(max_abs),
                protected_rms_drift=float(rms),
            )
        )

    # Diagnostic only: frontier selection scans every finite grid endpoint and
    # does not assume monotonic protected drift.
    drift_values = [row[5] for row in evaluations]
    monotonic_violations = sum(
        1 for a, b in zip(drift_values, drift_values[1:]) if b + 1e-12 < a
    )

    out: list[RetentionFrontierPoint] = []
    for beta in beta_values:
        budget = max(eps, beta * d_ref)
        feasible = [row for row in evaluations if row[5] <= budget + 1e-12]
        selected = max(feasible, key=lambda row: row[0]) if feasible else evaluations[0]
        scale, candidate, current_loss, decrease, ratio, max_abs, rms = selected
        out.append(
            RetentionFrontierPoint(
                beta=beta,
                budget=budget,
                scale=scale,
                vector_after=candidate.detach().clone(),
                update_norm=scale * delta_norm,
                current_loss_after=float(current_loss),
                persistent_decrease=float(decrease),
                persistent_ratio=float(ratio),
                protected_max_abs_drift=float(max_abs),
                protected_rms_drift=float(rms),
            )
        )

    audit = {
        "grid_points": n,
        "retention_reference_drift": d_ref,
        "monotonic_drift_violations": int(monotonic_violations),
        "drift_monotone_on_grid": bool(monotonic_violations == 0),
    }
    return out, grid, audit

def afm_finite_completion_for_vector(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    vector_after: torch.Tensor,
    comparator: ComparatorResult,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_logits_before: torch.Tensor,
    guard_inputs: torch.Tensor | None,
    address_encoder: StaticAddressEncoder,
    method_config: dict,
) -> FiniteCompletionResult:
    """Audit AFM finite endpoint completion on a chosen persistent base vector."""

    with temporary_parameters(vectoriser, vector_after):
        with torch.no_grad():
            base_current = model(current_inputs).detach().clone()
            base_protected = model(protected_inputs).detach().clone()
        finite = finite_endpoint_completion(
            address_encoder=address_encoder,
            base_current_logits=base_current,
            base_protected_logits=base_protected,
            desired_current_logits=comparator.current_full_logits_after,
            desired_protected_logits=protected_logits_before,
            current_inputs=current_inputs,
            protected_inputs=protected_inputs,
            guard_inputs=guard_inputs,
            support_multiplier=float(method_config.get("shield_support_multiplier", 4.0)),
            feature_match_tolerance=float(method_config.get("shield_feature_match_tolerance", 1e-8)),
            target_tolerance=float(method_config.get("shield_target_tolerance", 1e-8)),
            residual_tolerance=float(method_config.get("shield_residual_tolerance", 1e-8)),
        )
        if finite.available and finite.deployed_current_logits is not None:
            current_measure = finite.deployed_current_logits
            if hasattr(model, "functional_projection"):
                projection = getattr(model, "functional_projection")
                current_measure = current_measure @ projection
            deployed_diff = (current_measure - current_targets).to(dtype=torch.float64)
            deployed_loss = float((0.5 * deployed_diff.square().sum()).item())
            deployed_decrease = float(comparator.loss_before) - deployed_loss
            deployed_ratio = (
                deployed_decrease / float(comparator.decrease)
                if comparator.decrease > 0
                else float("nan")
            )
        else:
            deployed_ratio = None
    return FiniteCompletionResult(
        available=bool(finite.available),
        deployed_ratio=deployed_ratio,
        endpoint_error=finite.endpoint_error,
        current_error=finite.current_error,
        protected_error=finite.protected_error,
        obstruction=finite.solve.obstruction if not finite.available else None,
    )
