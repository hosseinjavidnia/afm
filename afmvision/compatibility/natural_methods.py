from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from afmvision.afm.parameter_vector import ParameterVector, temporary_parameters
from afmvision.compatibility.geometry import (
    ProtectedGeometry,
    compatible_projection,
    compatibility_fraction,
)
from afmvision.compatibility.methods import (
    fisher_diagonal,
    linearised_distillation_direction,
)
from afmvision.compatibility.models import CompatibilityModel


@dataclass(frozen=True)
class NaturalComparatorResult:
    alpha: float
    vector_before: torch.Tensor
    vector_after: torch.Tensor
    delta: torch.Tensor
    loss_before: float
    loss_after: float
    decrease: float
    current_full_logits_after: torch.Tensor
    backtracking_steps: int


@dataclass(frozen=True)
class NaturalMethodResult:
    method: str
    accepted: bool
    vector_after: torch.Tensor
    delta: torch.Tensor
    update_norm: float
    current_loss_after: float
    persistent_decrease: float
    persistent_ratio: float
    protected_max_abs_drift: float
    protected_rms_drift: float
    retention_pass: bool
    proposal_kappa: float
    obstruction: str | None = None
    backtracking_steps: int = 0
    afm_lambda_hat: float | None = None


def supervised_loss_value(
    model: CompatibilityModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Evaluate ordinary supervised CE with float64 scalar accumulation.

    Model execution remains in native precision.  Only the finite endpoint loss
    reduction is evaluated in float64, matching the numerical philosophy used by
    the causal comparator while keeping the objective the true-label CE loss.
    """

    with torch.no_grad():
        logits = model(inputs)
        return float(F.cross_entropy(logits.to(dtype=torch.float64), labels, reduction="mean").item())


def supervised_gradient(
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Realised native-precision gradient of the ordinary supervised objective."""

    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    return vectoriser.flatten_grads().detach().clone()


def _gradient_for_ce(
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return supervised_gradient(model, vectoriser, inputs, labels)


def _gradient_for_logit_mse(
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    inputs: torch.Tensor,
    stored_logits: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = 0.5 * F.mse_loss(logits, stored_logits, reduction="sum")
    loss.backward()
    return vectoriser.flatten_grads().detach().clone()


def supervised_unrestricted_comparator(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    current_inputs: torch.Tensor,
    current_labels: torch.Tensor,
    current_gradient: torch.Tensor,
    initial_alpha: float,
    max_backtracks: int,
    backtrack_factor: float,
    min_decrease: float = 0.0,
) -> NaturalComparatorResult:
    """Genuine same-state no-protection endpoint for the natural CE objective.

    The direction is exactly ``-current_gradient``.  Only scalar step length is
    backtracked.  Search stops if the next float parameter vector is bitwise
    identical to the current vector.
    """

    vector_before = vectoriser.flatten(detach=True)
    loss_before = supervised_loss_value(model, current_inputs, current_labels)
    alpha = float(initial_alpha)

    for attempt in range(int(max_backtracks) + 1):
        delta = -alpha * current_gradient
        candidate = vector_before + delta
        if torch.equal(candidate, vector_before):
            break
        with temporary_parameters(vectoriser, candidate):
            loss_after = supervised_loss_value(model, current_inputs, current_labels)
            with torch.no_grad():
                current_full = model(current_inputs).detach().clone()
        decrease = float(loss_before - loss_after)
        if decrease > float(min_decrease):
            return NaturalComparatorResult(
                alpha=float(alpha),
                vector_before=vector_before,
                vector_after=candidate,
                delta=(candidate - vector_before).detach().clone(),
                loss_before=float(loss_before),
                loss_after=float(loss_after),
                decrease=decrease,
                current_full_logits_after=current_full,
                backtracking_steps=int(attempt),
            )
        alpha *= float(backtrack_factor)

    raise RuntimeError(
        "natural same-state unrestricted CE comparator failed to produce a positive finite decrease"
    )


def _endpoint_metrics(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    vector_after: torch.Tensor,
    current_inputs: torch.Tensor,
    current_labels: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_logits_before: torch.Tensor,
) -> tuple[float, float, float]:
    with temporary_parameters(vectoriser, vector_after):
        current_loss = supervised_loss_value(model, current_inputs, current_labels)
        with torch.no_grad():
            protected_after = model(protected_inputs)
            drift = protected_after - protected_logits_before
            max_abs = float(drift.abs().max().item()) if drift.numel() else 0.0
            rms = float(torch.sqrt(torch.mean(drift.square())).item()) if drift.numel() else 0.0
    return current_loss, max_abs, rms


def _result_from_delta(
    *,
    method: str,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    comparator: NaturalComparatorResult,
    delta: torch.Tensor,
    geometry: ProtectedGeometry,
    current_inputs: torch.Tensor,
    current_labels: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_logits_before: torch.Tensor,
    retention_tolerance: float,
    accepted: bool = True,
    obstruction: str | None = None,
    backtracking_steps: int = 0,
    afm_lambda_hat: float | None = None,
) -> NaturalMethodResult:
    vector_after = comparator.vector_before + delta
    current_loss, max_abs, rms = _endpoint_metrics(
        model=model,
        vectoriser=vectoriser,
        vector_after=vector_after,
        current_inputs=current_inputs,
        current_labels=current_labels,
        protected_inputs=protected_inputs,
        protected_logits_before=protected_logits_before,
    )
    decrease = float(comparator.loss_before - current_loss)
    ratio = decrease / comparator.decrease if comparator.decrease > 0.0 else float("nan")
    proposal_kappa = compatibility_fraction(delta, geometry) if float(torch.dot(delta, delta).item()) > 0.0 else 0.0
    return NaturalMethodResult(
        method=str(method),
        accepted=bool(accepted),
        vector_after=vector_after,
        delta=delta.detach().clone(),
        update_norm=float(torch.linalg.vector_norm(delta).item()),
        current_loss_after=float(current_loss),
        persistent_decrease=decrease,
        persistent_ratio=float(ratio),
        protected_max_abs_drift=float(max_abs),
        protected_rms_drift=float(rms),
        retention_pass=bool(max_abs <= float(retention_tolerance)),
        proposal_kappa=float(proposal_kappa),
        obstruction=obstruction,
        backtracking_steps=int(backtracking_steps),
        afm_lambda_hat=afm_lambda_hat,
    )


def run_natural_method(
    *,
    method: str,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    comparator: NaturalComparatorResult,
    current_gradient: torch.Tensor,
    geometry: ProtectedGeometry,
    current_inputs: torch.Tensor,
    current_labels: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_labels: torch.Tensor,
    protected_logits_before: torch.Tensor,
    replay_stored_logits: torch.Tensor,
    retention_tolerance: float,
    method_config: dict,
) -> NaturalMethodResult:
    """Evaluate one natural-state method branch without committing it.

    This preserves the seven local proposal mechanisms used by the causal
    experiment while replacing the controlled teacher objective with the
    ordinary true-label supervised objective.  AFM here reports the persistent
    compatible base transaction only; finite compact-cardinal completion is not
    part of the requested natural-state response variable.
    """

    method = str(method)
    alpha = float(comparator.alpha)

    if method == "unrestricted":
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=comparator.delta,
            geometry=geometry,
            current_inputs=current_inputs,
            current_labels=current_labels,
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
            geometry=geometry,
            current_inputs=current_inputs,
            current_labels=current_labels,
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
            geometry=geometry,
            current_inputs=current_inputs,
            current_labels=current_labels,
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
            geometry=geometry,
            current_inputs=current_inputs,
            current_labels=current_labels,
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
            geometry=geometry,
            current_inputs=current_inputs,
            current_labels=current_labels,
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
            geometry=geometry,
            current_inputs=current_inputs,
            current_labels=current_labels,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
        )

    if method == "afm":
        direction = compatible_projection(current_gradient, geometry)
        factor = 1.0
        backtrack_factor = float(method_config.get("afm_backtrack_factor", 0.5))
        max_backtracks = int(method_config.get("afm_max_backtracks", 20))
        for attempt in range(max_backtracks + 1):
            delta = -alpha * factor * direction
            candidate = _result_from_delta(
                method=method,
                model=model,
                vectoriser=vectoriser,
                comparator=comparator,
                delta=delta,
                geometry=geometry,
                current_inputs=current_inputs,
                current_labels=current_labels,
                protected_inputs=protected_inputs,
                protected_logits_before=protected_logits_before,
                retention_tolerance=retention_tolerance,
                backtracking_steps=attempt,
                afm_lambda_hat=float(factor),
            )
            if candidate.retention_pass and candidate.persistent_decrease > 0.0:
                return candidate
            factor *= backtrack_factor

        zero = torch.zeros_like(current_gradient)
        return _result_from_delta(
            method=method,
            model=model,
            vectoriser=vectoriser,
            comparator=comparator,
            delta=zero,
            geometry=geometry,
            current_inputs=current_inputs,
            current_labels=current_labels,
            protected_inputs=protected_inputs,
            protected_logits_before=protected_logits_before,
            retention_tolerance=retention_tolerance,
            accepted=False,
            obstruction="persistent_retention_backtracking_exhausted",
            backtracking_steps=max_backtracks,
            afm_lambda_hat=0.0,
        )

    raise ValueError(f"unknown natural compatibility method: {method}")
