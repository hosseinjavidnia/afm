from __future__ import annotations

from dataclasses import dataclass

import torch

from afmvision.afm.safe_step import SafeStep, project_to_allowed_free_subspace, safe_radius


@dataclass(frozen=True)
class PersistentAssimilationPlan:
    """Certified projected assimilation of the exact ordinary comparator path.

    ``reference_delta`` is the exact ordinary counterfactual displacement after
    orthogonal projection into the selected metaplastic feasible subspace.  The
    round budget is a predeclared fraction of the full linear--quadratic charge
    of that projected reference.  The selected displacement is the largest
    point on the reference segment certified by the same retention envelope.
    """

    proposal: SafeStep
    reference_delta: torch.Tensor
    reference_step_length: float
    reference_charge: float
    retention_budget: float
    requested_charge_fraction: float
    guaranteed_path_fraction: float
    selected_path_fraction: float
    projected_counterfactual_alignment_error: float
    ordinary_counterfactual_alignment_error: float
    projection_idempotence_error: float
    ordinary_gradient_norm: float
    compatibility_fraction: float
    ordinary_step_size: float
    step_size_smoothness_product: float
    scalar_comparator_certified: bool
    analytic_persistent_progress_ratio_lower_bound: float


def retention_charge(E: float, H: float, step_length: float) -> float:
    """Return ``E s + H s^2 / 2`` with all certificate inputs clipped at zero."""

    e = max(float(E), 0.0)
    h = max(float(H), 0.0)
    s = max(float(step_length), 0.0)
    return e * s + 0.5 * h * s * s


def normalized_retention_budget(
    E: float,
    H: float,
    reference_step_length: float,
    charge_fraction: float,
) -> tuple[float, float]:
    """Return the full projected-reference charge and its declared fraction."""

    fraction = float(charge_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("charge_fraction must lie in [0,1]")
    full_charge = retention_charge(E, H, reference_step_length)
    return full_charge, fraction * full_charge


def make_counterfactual_normalized_plan(
    *,
    counterfactual_delta: torch.Tensor,
    ordinary_gradient: torch.Tensor,
    protected_basis: torch.Tensor | None,
    allowed_mask: torch.Tensor | None,
    E: float,
    H: float,
    charge_fraction: float,
    cap: float,
    active_protection: bool,
    loss_smoothness: float,
    tolerance: float = 1e-12,
) -> PersistentAssimilationPlan:
    """Construct the maximal certified fraction of the projected comparator.

    Let ``v`` be the feasible projection of the exact ordinary comparator
    displacement and ``C(s)=E s + H s^2/2``.  On a protected round the budget is
    ``eta C(||v||)``.  Convexity and ``C(0)=0`` imply
    ``C(eta ||v||) <= eta C(||v||)``, hence the certified path fraction is at
    least ``eta`` whenever the reference is nonzero and the supplied comparator
    is within the declared cap.  With no protected behaviour the whole feasible
    comparator displacement is selected.

    The exact ordinary operator used by AFM-U is a scalar backtracked gradient
    step.  Therefore its feasible projection must be collinear with the negative
    feasible gradient.  The executable alignment residual below makes that
    premise auditable rather than implicit.
    """

    fraction = float(charge_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("charge_fraction must lie in [0,1]")
    cap_value = max(float(cap), 0.0)

    reference = project_to_allowed_free_subspace(
        counterfactual_delta, protected_basis, allowed_mask
    )
    ordinary_gradient = project_to_allowed_free_subspace(
        ordinary_gradient, None, allowed_mask
    )
    feasible_gradient = project_to_allowed_free_subspace(
        ordinary_gradient, protected_basis, allowed_mask
    )
    ordinary_delta = project_to_allowed_free_subspace(
        counterfactual_delta, None, allowed_mask
    )
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    gradient_norm = float(torch.linalg.vector_norm(feasible_gradient).item())
    ordinary_gradient_norm = float(torch.linalg.vector_norm(ordinary_gradient).item())
    ordinary_delta_norm = float(torch.linalg.vector_norm(ordinary_delta).item())
    compatibility_fraction = (
        (gradient_norm / ordinary_gradient_norm) ** 2
        if ordinary_gradient_norm > tolerance
        else 0.0
    )
    if ordinary_gradient_norm > tolerance:
        ordinary_step_size = float(
            -torch.dot(ordinary_delta, ordinary_gradient).item()
            / (ordinary_gradient_norm * ordinary_gradient_norm)
        )
    else:
        ordinary_step_size = 0.0
    L = max(float(loss_smoothness), 0.0)
    step_size_smoothness_product = ordinary_step_size * L

    if reference_norm <= tolerance:
        zero = torch.zeros_like(counterfactual_delta)
        return PersistentAssimilationPlan(
            proposal=SafeStep(0.0, 0.0, gradient_norm, zero),
            reference_delta=zero,
            reference_step_length=0.0,
            reference_charge=0.0,
            retention_budget=0.0,
            requested_charge_fraction=fraction,
            guaranteed_path_fraction=0.0,
            selected_path_fraction=0.0,
            projected_counterfactual_alignment_error=0.0,
            ordinary_counterfactual_alignment_error=0.0,
            projection_idempotence_error=0.0,
            ordinary_gradient_norm=ordinary_gradient_norm,
            compatibility_fraction=compatibility_fraction,
            ordinary_step_size=ordinary_step_size,
            step_size_smoothness_product=step_size_smoothness_product,
            scalar_comparator_certified=False,
            analytic_persistent_progress_ratio_lower_bound=0.0,
        )

    # The production ordinary comparator is already trust-capped and orthogonal
    # projection cannot increase its norm.  Keep an explicit guard for malformed
    # external comparator operators rather than silently changing their path.
    if cap_value <= 0.0 or reference_norm > cap_value + tolerance * max(1.0, cap_value):
        raise ValueError("projected counterfactual displacement exceeds its declared trust cap")

    if gradient_norm <= tolerance or ordinary_step_size <= 0.0:
        alignment_error = reference_norm
    else:
        ideal = -ordinary_step_size * feasible_gradient
        alignment_error = float(torch.linalg.vector_norm(reference - ideal).item())

    if ordinary_gradient_norm <= tolerance or ordinary_step_size <= 0.0:
        ordinary_alignment_error = ordinary_delta_norm
    else:
        ordinary_ideal = -ordinary_step_size * ordinary_gradient
        ordinary_alignment_error = float(
            torch.linalg.vector_norm(ordinary_delta - ordinary_ideal).item()
        )

    reprojection = project_to_allowed_free_subspace(
        reference, protected_basis, allowed_mask
    )
    idempotence_error = float(torch.linalg.vector_norm(reprojection - reference).item())

    full_charge, budget = normalized_retention_budget(E, H, reference_norm, fraction)
    if active_protection:
        radius = safe_radius(E=E, H=H, budget=budget, cap=reference_norm)
        guaranteed_fraction = fraction
    else:
        radius = reference_norm
        budget = 0.0
        guaranteed_fraction = 1.0

    step_length = min(reference_norm, radius)
    selected_fraction = step_length / reference_norm
    delta = reference * selected_fraction
    proposal = SafeStep(
        radius=float(radius),
        step_length=float(step_length),
        projected_gradient_norm=gradient_norm,
        delta=delta,
    )
    alignment_scale = max(ordinary_delta_norm, reference_norm, tolerance)
    scalar_comparator_certified = bool(
        ordinary_step_size > 0.0
        and step_size_smoothness_product <= 1.0 + tolerance
        and ordinary_alignment_error <= tolerance * alignment_scale
        and alignment_error <= tolerance * alignment_scale
        and idempotence_error <= tolerance * alignment_scale
    )
    # For an L-smooth loss and an ordinary scalar gradient step d0=-a g with
    # 0<aL<=1, the selected protected segment lambda Pi d0 obeys
    #   Delta_safe/Delta_ordinary >=
    #   lambda*kappa*(1-aL*lambda/2)/(1+aL/2),
    # where kappa=||Pi g||^2/||g||^2.  This is the exact smoothness quotient;
    # lambda*kappa/3 is only its coarser universal consequence.
    if scalar_comparator_certified:
        analytic_ratio_lower_bound = (
            selected_fraction
            * compatibility_fraction
            * max(1.0 - 0.5 * step_size_smoothness_product * selected_fraction, 0.0)
            / (1.0 + 0.5 * step_size_smoothness_product)
        )
    else:
        analytic_ratio_lower_bound = 0.0
    return PersistentAssimilationPlan(
        proposal=proposal,
        reference_delta=reference,
        reference_step_length=reference_norm,
        reference_charge=float(full_charge),
        retention_budget=float(budget),
        requested_charge_fraction=fraction,
        guaranteed_path_fraction=float(guaranteed_fraction),
        selected_path_fraction=float(selected_fraction),
        projected_counterfactual_alignment_error=alignment_error,
        ordinary_counterfactual_alignment_error=ordinary_alignment_error,
        projection_idempotence_error=idempotence_error,
        ordinary_gradient_norm=ordinary_gradient_norm,
        compatibility_fraction=compatibility_fraction,
        ordinary_step_size=ordinary_step_size,
        step_size_smoothness_product=step_size_smoothness_product,
        scalar_comparator_certified=scalar_comparator_certified,
        analytic_persistent_progress_ratio_lower_bound=float(analytic_ratio_lower_bound),
    )


def persistent_descent_lower_bound(
    *,
    projected_gradient_norm: float,
    reference_step_length: float,
    selected_path_fraction: float,
    smoothness: float,
) -> float:
    """Return the descent-lemma guarantee on the selected comparator segment."""

    q = max(float(projected_gradient_norm), 0.0)
    s = max(float(reference_step_length), 0.0)
    lam = min(max(float(selected_path_fraction), 0.0), 1.0)
    L = max(float(smoothness), 0.0)
    step = lam * s
    return max(step * q - 0.5 * L * step * step, 0.0)
