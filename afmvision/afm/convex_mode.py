from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


class ExactProjectionOracle(Protocol):
    """Exact Euclidean projection oracle for a declared closed convex set."""

    def __call__(self, point: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ExactConvexStep:
    available: bool
    previous: torch.Tensor
    gradient: torch.Tensor
    unconstrained: torch.Tensor
    endpoint: torch.Tensor
    displacement: torch.Tensor
    step_size: float
    projection_residual: float
    obstruction: str | None = None


@torch.no_grad()
def exact_projected_gradient_step(
    *,
    point: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
    project_exact: ExactProjectionOracle,
    feasibility_residual: Callable[[torch.Tensor], float] | None = None,
    tolerance: float = 0.0,
) -> ExactConvexStep:
    """Execute the manuscript's exact-convex projected-gradient specialization.

    The generic nonlinear vision learner cannot infer an exact projector for an
    arbitrary closed convex retention set. The admissible exact-convex mode
    therefore requires the specification to supply a deterministic exact
    Euclidean projection oracle and, optionally, a conservative feasibility
    residual. Failure or uncertified feasibility returns a typed obstruction.
    """

    if point.shape != gradient.shape:
        raise ValueError("point and gradient must have the same shape")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    unconstrained = point - float(step_size) * gradient
    try:
        endpoint = project_exact(unconstrained.detach().clone()).detach().clone()
    except Exception as error:  # explicit oracle boundary
        return ExactConvexStep(
            available=False,
            previous=point.detach().clone(),
            gradient=gradient.detach().clone(),
            unconstrained=unconstrained.detach().clone(),
            endpoint=point.detach().clone(),
            displacement=torch.zeros_like(point),
            step_size=float(step_size),
            projection_residual=float("inf"),
            obstruction=f"exact_convex_projection_oracle_failure:{type(error).__name__}",
        )
    if endpoint.shape != point.shape or not torch.isfinite(endpoint).all():
        return ExactConvexStep(
            available=False,
            previous=point.detach().clone(),
            gradient=gradient.detach().clone(),
            unconstrained=unconstrained.detach().clone(),
            endpoint=point.detach().clone(),
            displacement=torch.zeros_like(point),
            step_size=float(step_size),
            projection_residual=float("inf"),
            obstruction="exact_convex_projection_invalid_endpoint",
        )
    residual = 0.0 if feasibility_residual is None else float(feasibility_residual(endpoint))
    if not torch.isfinite(torch.tensor(residual)) or residual > tolerance:
        return ExactConvexStep(
            available=False,
            previous=point.detach().clone(),
            gradient=gradient.detach().clone(),
            unconstrained=unconstrained.detach().clone(),
            endpoint=point.detach().clone(),
            displacement=torch.zeros_like(point),
            step_size=float(step_size),
            projection_residual=residual,
            obstruction="exact_convex_projection_uncertified",
        )
    return ExactConvexStep(
        available=True,
        previous=point.detach().clone(),
        gradient=gradient.detach().clone(),
        unconstrained=unconstrained.detach().clone(),
        endpoint=endpoint,
        displacement=endpoint - point,
        step_size=float(step_size),
        projection_residual=residual,
        obstruction=None,
    )


@torch.no_grad()
def project_box(lower: torch.Tensor, upper: torch.Tensor) -> ExactProjectionOracle:
    """Return the exact componentwise projector onto a closed box."""

    if lower.shape != upper.shape or torch.any(lower > upper):
        raise ValueError("invalid box bounds")
    lo = lower.detach().clone()
    hi = upper.detach().clone()

    def projector(point: torch.Tensor) -> torch.Tensor:
        if point.shape != lo.shape:
            raise ValueError("point shape does not match box")
        return torch.maximum(torch.minimum(point, hi.to(point)), lo.to(point))

    return projector
