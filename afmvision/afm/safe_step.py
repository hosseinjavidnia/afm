from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SafeStep:
    radius: float
    step_length: float
    projected_gradient_norm: float
    delta: torch.Tensor




@dataclass
class PriorityTransferDirection:
    """Exact solution of the priority-constrained transfer direction problem.

    The unit-ball problem maximises transfer slope while preserving at least
    ``current_fraction`` of the ordinary projected-current-gradient slope.
    Eligibility is invariant to positive rescaling of the transfer objective.
    """

    available: bool
    direction: torch.Tensor
    current_projected_norm: float
    transfer_projected_norm: float
    direction_norm: float
    current_slope: float
    transfer_slope: float
    current_fraction: float
    compatibility: float
    projected_cosine: float
    obstruction: str | None = None


@dataclass
class CommonDescentDirection:
    """Minimum-norm point in the convex hull of two compatible gradients.

    If ``available`` is true, ``-direction`` is a common first-order descent
    direction for both objectives in the declared feasible subspace.  The two
    returned slopes are directional derivatives in the normalised descent
    direction and are used to choose a step valid for both quadratic models.
    """

    available: bool
    direction: torch.Tensor
    current_projected_norm: float
    transfer_projected_norm: float
    direction_norm: float
    current_slope: float
    transfer_slope: float
    mixture_weight: float
    projected_cosine: float = 0.0


def safe_radius(E: float, H: float, budget: float, cap: float) -> float:
    E = max(float(E), 0.0)
    H = max(float(H), 0.0)
    budget = max(float(budget), 0.0)
    cap = max(float(cap), 0.0)
    if cap == 0.0:
        return 0.0
    if E == 0.0 and H == 0.0:
        return cap
    if budget == 0.0:
        return 0.0
    if H > 0.0:
        raw = 2.0 * budget / (E + math.sqrt(E * E + 2.0 * H * budget))
    elif E > 0.0:
        raw = budget / E
    else:
        raw = cap
    return min(cap, max(raw, 0.0))


def orthonormal_basis(matrix: torch.Tensor, tolerance: float = 1e-10) -> torch.Tensor:
    if matrix.numel() == 0 or matrix.shape[1] == 0:
        return torch.zeros((matrix.shape[0], 0), device=matrix.device, dtype=matrix.dtype)

    # Bases produced by the metaplastic sketch decomposition are already
    # orthonormal in the structurally allowed space.  Check the small Gram
    # matrix first and avoid any decomposition on the common path.
    gram = matrix.T @ matrix
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    residual = float((gram - eye).abs().max().item())
    if residual <= max(100.0 * tolerance, 2e-5):
        return matrix.contiguous()

    # For externally supplied or roundoff-degraded bases, diagonalise only the
    # bounded column Gram matrix on CPU float64 and reconstruct the left basis.
    gram_cpu = gram.detach().to(device="cpu", dtype=torch.float64)
    eigenvalues, vectors = torch.linalg.eigh(gram_cpu)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues.index_select(0, order), min=0.0)
    vectors = vectors.index_select(1, order)
    singular = torch.sqrt(eigenvalues)
    threshold = max(float(singular.max().item()) * tolerance, tolerance)
    rank = int((singular > threshold).sum().item())
    if rank <= 0:
        return torch.zeros((matrix.shape[0], 0), device=matrix.device, dtype=matrix.dtype)
    vectors = vectors[:, :rank].to(device=matrix.device, dtype=matrix.dtype)
    singular = singular[:rank].to(device=matrix.device, dtype=matrix.dtype)
    basis = matrix @ (vectors / singular.unsqueeze(0))
    return basis.contiguous()


def project_to_allowed_free_subspace(
    vector: torch.Tensor,
    protected_basis: torch.Tensor | None,
    allowed_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Project onto allowed coordinates intersected with the protected nullspace.

    For coordinate subspace A=Range(diag(mask)), d in A satisfies U^T d=0 iff
    (diag(mask) U)^T d=0. We therefore orthonormalise the masked protected
    columns inside A, project there, and reapply the mask for roundoff safety.
    """

    if allowed_mask is None:
        allowed = vector
        mask = None
    else:
        mask = allowed_mask.to(device=vector.device, dtype=vector.dtype)
        allowed = vector * mask
    if protected_basis is None or protected_basis.numel() == 0:
        return allowed if mask is None else allowed * mask
    U = protected_basis.to(device=vector.device, dtype=vector.dtype)
    restricted = U if mask is None else U * mask.unsqueeze(1)
    V = orthonormal_basis(restricted)
    projected = allowed if V.numel() == 0 else allowed - V @ (V.T @ allowed)
    return projected if mask is None else projected * mask


def minimum_norm_common_descent(
    current_gradient: torch.Tensor,
    transfer_gradient: torch.Tensor,
    protected_basis: torch.Tensor | None,
    allowed_mask: torch.Tensor | None,
    tolerance: float = 1e-12,
) -> CommonDescentDirection:
    """Return a two-objective common descent direction in the same feasible space.

    Let ``a`` and ``b`` be the projected gradients.  The minimum-norm point
    ``q`` of the line segment ``conv{a,b}`` is computed in closed form.  The
    projection optimality condition gives ``a^T q >= ||q||^2`` and
    ``b^T q >= ||q||^2`` whenever ``q`` is nonzero, so ``-q`` is a common
    first-order descent direction.  Numerical checks are explicit; otherwise
    the caller must fall back to the ordinary current-loss direction.
    """

    a = project_to_allowed_free_subspace(current_gradient, protected_basis, allowed_mask)
    b = project_to_allowed_free_subspace(transfer_gradient, protected_basis, allowed_mask)
    a_norm = float(torch.linalg.vector_norm(a).item())
    b_norm = float(torch.linalg.vector_norm(b).item())
    diff = a - b
    denom = float(torch.dot(diff, diff).item())
    if denom <= tolerance:
        weight = 0.5
    else:
        # q(w)=w a +(1-w)b; minimise ||q(w)||^2 over w in [0,1].
        weight = float(torch.clamp((torch.dot(b, b) - torch.dot(a, b)) / denom, 0.0, 1.0).item())
    q = weight * a + (1.0 - weight) * b
    q_norm = float(torch.linalg.vector_norm(q).item())
    if q_norm <= tolerance:
        return CommonDescentDirection(
            available=False,
            direction=torch.zeros_like(current_gradient),
            current_projected_norm=a_norm,
            transfer_projected_norm=b_norm,
            direction_norm=q_norm,
            current_slope=0.0,
            transfer_slope=0.0,
            mixture_weight=weight,
            projected_cosine=(float(torch.dot(a, b).item()) / max(a_norm * b_norm, tolerance) if a_norm > tolerance and b_norm > tolerance else 0.0),
        )
    current_slope = float(torch.dot(a, q).item()) / q_norm
    transfer_slope = float(torch.dot(b, q).item()) / q_norm
    scale = max(a_norm, b_norm, q_norm, 1.0)
    available = current_slope > tolerance * scale and transfer_slope > tolerance * scale
    return CommonDescentDirection(
        available=available,
        direction=q if available else torch.zeros_like(current_gradient),
        current_projected_norm=a_norm,
        transfer_projected_norm=b_norm,
        direction_norm=q_norm,
        current_slope=max(current_slope, 0.0),
        transfer_slope=max(transfer_slope, 0.0),
        mixture_weight=weight,
        projected_cosine=(float(torch.dot(a, b).item()) / max(a_norm * b_norm, tolerance) if a_norm > tolerance and b_norm > tolerance else 0.0),
    )



def priority_constrained_transfer_direction(
    current_gradient: torch.Tensor,
    transfer_gradient: torch.Tensor,
    protected_basis: torch.Tensor | None,
    allowed_mask: torch.Tensor | None,
    current_fraction: float = 0.25,
    tolerance: float = 1e-12,
) -> PriorityTransferDirection:
    """Solve the scale-invariant priority-constrained direction exactly.

    With ``a`` and ``b`` the projected gradients, solve

        maximise <b, q>
        subject to ||q|| <= 1 and <a, q> >= rho ||a||.

    The parameter update is along ``-q``.  When the returned compatibility is
    nonpositive, no direction in the declared feasible subspace can preserve
    the requested current-loss fraction and strictly reduce transfer to first
    order.
    """

    rho = float(current_fraction)
    if not 0.0 <= rho < 1.0:
        raise ValueError("current_fraction must lie in [0,1)")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    a = project_to_allowed_free_subspace(current_gradient, protected_basis, allowed_mask)
    b = project_to_allowed_free_subspace(transfer_gradient, protected_basis, allowed_mask)
    a_norm = float(torch.linalg.vector_norm(a).item())
    b_norm = float(torch.linalg.vector_norm(b).item())
    scale = max(a_norm, b_norm, 1.0)

    if b_norm <= tolerance * scale:
        return PriorityTransferDirection(
            available=False,
            direction=torch.zeros_like(current_gradient),
            current_projected_norm=a_norm,
            transfer_projected_norm=b_norm,
            direction_norm=0.0,
            current_slope=0.0,
            transfer_slope=0.0,
            current_fraction=rho,
            compatibility=0.0,
            projected_cosine=0.0,
            obstruction="zero_projected_transfer_gradient",
        )

    if a_norm <= tolerance * scale:
        q = b / b_norm
        return PriorityTransferDirection(
            available=True,
            direction=q,
            current_projected_norm=a_norm,
            transfer_projected_norm=b_norm,
            direction_norm=1.0,
            current_slope=0.0,
            transfer_slope=b_norm,
            current_fraction=rho,
            compatibility=1.0,
            projected_cosine=0.0,
            obstruction=None,
        )

    dot = float(torch.dot(a, b).item())
    cosine = max(-1.0, min(1.0, dot / (a_norm * b_norm)))
    e = a / a_norm

    if cosine <= -1.0 + max(tolerance, 1e-14):
        # The exact program optimum is attained by q=rho*e and equals
        # -rho*||b||. It is never a strict transfer-descent direction.
        return PriorityTransferDirection(
            available=False,
            direction=torch.zeros_like(current_gradient),
            current_projected_norm=a_norm,
            transfer_projected_norm=b_norm,
            direction_norm=rho,
            current_slope=rho * a_norm,
            transfer_slope=-rho * b_norm,
            current_fraction=rho,
            compatibility=-rho,
            projected_cosine=-1.0,
            obstruction="priority_transfer_incompatible",
        )

    if cosine + tolerance >= rho:
        q = b / b_norm
    else:
        unit_b = b / b_norm
        orthogonal = unit_b - cosine * e
        orthogonal_norm = float(torch.linalg.vector_norm(orthogonal).item())
        if orthogonal_norm <= tolerance:
            q = rho * e
        else:
            v = orthogonal / orthogonal_norm
            q = rho * e + math.sqrt(max(1.0 - rho * rho, 0.0)) * v

    q_norm = float(torch.linalg.vector_norm(q).item())
    if q_norm <= tolerance:
        return PriorityTransferDirection(
            available=False,
            direction=torch.zeros_like(current_gradient),
            current_projected_norm=a_norm,
            transfer_projected_norm=b_norm,
            direction_norm=q_norm,
            current_slope=0.0,
            transfer_slope=0.0,
            current_fraction=rho,
            compatibility=0.0,
            projected_cosine=cosine,
            obstruction="zero_priority_direction",
        )

    # Endpoint models use slopes in the normalised direction.
    q_unit = q / q_norm
    current_slope = float(torch.dot(a, q_unit).item())
    transfer_slope = float(torch.dot(b, q_unit).item())
    compatibility = transfer_slope / b_norm
    current_ok = current_slope + tolerance * scale >= rho * a_norm
    transfer_ok = transfer_slope > tolerance * scale
    available = bool(current_ok and transfer_ok)
    obstruction = None
    if not current_ok:
        obstruction = "current_priority_constraint_failed_numerically"
    elif not transfer_ok:
        obstruction = "priority_transfer_incompatible"

    return PriorityTransferDirection(
        available=available,
        direction=q_unit if available else torch.zeros_like(current_gradient),
        current_projected_norm=a_norm,
        transfer_projected_norm=b_norm,
        direction_norm=q_norm,
        current_slope=max(current_slope, 0.0) if available else current_slope,
        transfer_slope=max(transfer_slope, 0.0) if available else transfer_slope,
        current_fraction=rho,
        compatibility=compatibility,
        projected_cosine=cosine,
        obstruction=obstruction,
    )


def make_priority_safe_step(
    direction: torch.Tensor,
    current_projected_norm: float,
    current_slope: float,
    transfer_slope: float,
    current_smoothness: float,
    transfer_smoothness: float,
    E: float,
    H: float,
    budget: float,
    cap: float,
    learning_rate: float,
    allowed_mask: torch.Tensor | None = None,
) -> SafeStep:
    """Construct the theorem-aligned priority-transfer step.

    The step is bounded by the existing retention radius, the nominal ordinary
    current-gradient step length, and both certified quadratic descent caps.
    """

    qnorm = float(torch.linalg.vector_norm(direction).item())
    radius = safe_radius(E=E, H=H, budget=budget, cap=cap)
    if (
        qnorm == 0.0
        or radius == 0.0
        or current_slope < 0.0
        or transfer_slope <= 0.0
        or learning_rate <= 0.0
    ):
        return SafeStep(
            radius=radius,
            step_length=0.0,
            projected_gradient_norm=qnorm,
            delta=torch.zeros_like(direction),
        )

    caps = [radius]
    # Match the ordinary current-gradient nominal scale when the current
    # projected gradient is nonzero.  At a current stationary point the
    # priority constraint is vacuous, so a zero ordinary nominal scale must not
    # suppress a safe transfer step; the retention and transfer-smoothness caps
    # remain in force.
    if float(current_projected_norm) > 0.0:
        caps.append(float(learning_rate) * float(current_projected_norm))
    if current_smoothness > 0.0 and math.isfinite(current_smoothness):
        caps.append(float(current_slope) / float(current_smoothness))
    if transfer_smoothness > 0.0 and math.isfinite(transfer_smoothness):
        caps.append(float(transfer_slope) / float(transfer_smoothness))
    step_length = max(0.0, min(caps))
    delta = -step_length * direction / qnorm
    if allowed_mask is not None:
        delta = delta * allowed_mask.to(device=delta.device, dtype=delta.dtype)
    return SafeStep(
        radius=radius,
        step_length=step_length,
        projected_gradient_norm=qnorm,
        delta=delta,
    )
@dataclass
class JointProgressStep:
    """Candidate-safe step from the joint smoothness-envelope program."""

    available: bool
    displacement: torch.Tensor
    candidate_safe_displacement: torch.Tensor
    mode: str
    current_ordinary_best: float
    current_safe_best: float
    current_required: float
    current_certified_decrease: float
    selected_certified_decrease: float
    candidate_certified_decreases: tuple[float, ...]
    iterations: int
    max_constraint_violation: float
    solver_converged: bool
    obstruction: str | None = None


def _project_ball(point: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    radius = max(float(radius), 0.0)
    diff = point - center
    norm = float(torch.linalg.vector_norm(diff).item())
    if norm <= radius or norm == 0.0:
        return point
    return center + (radius / norm) * diff


def _dual_coordinate_project_balls(
    point: torch.Tensor,
    balls: list[tuple[torch.Tensor, float]],
    *,
    max_iterations: int = 512,
    tolerance: float = 1e-9,
) -> tuple[torch.Tensor, int, float, float, bool]:
    """Project onto a finite intersection of Euclidean balls.

    This solves the Lagrange dual of

        min_x 1/2 ||x-point||^2  subject to ||x-c_i|| <= r_i.

    With nonnegative multipliers ``lambda_i``, stationarity gives

        x(lambda) = (point + sum_i lambda_i c_i) / (1 + sum_i lambda_i).

    Holding the other multipliers fixed, the exact coordinate maximiser is

        lambda_i = max(0, ||v_0-S_0 c_i||/r_i - S_0).

    Cyclic exact coordinate ascent is scale free, uses only the bounded number
    of declared candidate constraints, and is substantially more reliable near
    tangent ball intersections than a fixed-iteration primal projection.  The
    caller still verifies every quadratic constraint and every exact endpoint
    before deployment, so nonconvergence can only cause abstention.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if not balls:
        return point.clone(), 0, 0.0, 0.0, True

    # A zero-radius ball fixes the solution to one point.  Handle it directly
    # because the dual coordinate optimum would otherwise require an infinite
    # multiplier.
    zero_indices = [i for i, (_, radius) in enumerate(balls) if float(radius) <= tolerance]
    if zero_indices:
        fixed = balls[zero_indices[0]][0].clone()
        mismatch = 0.0
        for index in zero_indices[1:]:
            mismatch = max(
                mismatch,
                float(torch.linalg.vector_norm(fixed - balls[index][0]).item()),
            )
        violation = mismatch
        for center, radius in balls:
            violation = max(
                violation,
                float(torch.linalg.vector_norm(fixed - center).item()) - max(float(radius), 0.0),
            )
        residual = max(violation, 0.0)
        return fixed, 0, residual, residual, residual <= tolerance * max(1.0, float(torch.linalg.vector_norm(fixed).item()))

    multipliers = [0.0 for _ in balls]
    total = 1.0
    weighted = point.clone()
    iterations = 0
    dtype_floor = 64.0 * torch.finfo(point.dtype).eps if point.dtype.is_floating_point else tolerance
    effective_tolerance = max(float(tolerance), float(dtype_floor))

    for iteration in range(1, max_iterations + 1):
        max_change = 0.0
        for index, (center, radius) in enumerate(balls):
            old = multipliers[index]
            subtotal = total - old
            partial = weighted - old * center
            distance_numerator = float(
                torch.linalg.vector_norm(partial - subtotal * center).item()
            )
            new = max(0.0, distance_numerator / float(radius) - subtotal)
            if new != old:
                weighted = partial + new * center
                total = subtotal + new
                multipliers[index] = new
                max_change = max(max_change, abs(new - old))

        x = weighted / total
        scale = max(1.0, float(torch.linalg.vector_norm(x).item()))
        violation = 0.0
        complementarity = 0.0
        for multiplier, (center, radius) in zip(multipliers, balls):
            distance = float(torch.linalg.vector_norm(x - center).item())
            violation = max(violation, distance - float(radius))
            complementarity = max(
                complementarity,
                abs(multiplier * (distance * distance - float(radius) * float(radius))),
            )
        iterations = iteration
        multiplier_scale = max(1.0, total, max(multipliers, default=0.0))
        if (
            violation <= effective_tolerance * scale
            and max_change <= effective_tolerance * multiplier_scale
            and complementarity <= effective_tolerance * multiplier_scale * scale
        ):
            optimality_residual = max(
                max(violation, 0.0) / max(scale, 1e-300),
                max_change / max(multiplier_scale, 1e-300),
                complementarity / max(multiplier_scale * scale, 1e-300),
            )
            return x, iterations, max(violation, 0.0), optimality_residual, True

    x = weighted / total
    violation = 0.0
    for center, radius in balls:
        violation = max(
            violation,
            float(torch.linalg.vector_norm(x - center).item()) - max(float(radius), 0.0),
        )
    scale = max(1.0, float(torch.linalg.vector_norm(x).item()))
    optimality_residual = max(max(violation, 0.0) / scale, max_change / max(1.0, total, max(multipliers, default=0.0)))
    return x, iterations, max(violation, 0.0), optimality_residual, False


def quadratic_decrease(gradient: torch.Tensor, smoothness: float, displacement: torch.Tensor) -> float:
    return float(torch.dot(gradient, displacement).item()) - 0.5 * float(smoothness) * float(
        torch.dot(displacement, displacement).item()
    )


def unconstrained_ball_best(gradient: torch.Tensor, smoothness: float, radius: float) -> float:
    norm = float(torch.linalg.vector_norm(gradient).item())
    radius = max(float(radius), 0.0)
    smoothness = float(smoothness)
    if norm == 0.0 or radius == 0.0:
        return 0.0
    if smoothness <= 0.0 or not math.isfinite(smoothness):
        return radius * norm if smoothness == 0.0 else 0.0
    step = min(radius, norm / smoothness)
    return step * norm - 0.5 * smoothness * step * step


def joint_progress_protected_step(
    current_gradient: torch.Tensor,
    candidate_gradients: list[torch.Tensor],
    selected_index: int | None,
    protected_basis: torch.Tensor | None,
    allowed_mask: torch.Tensor | None,
    current_smoothness: float,
    candidate_smoothness: list[float],
    radius: float,
    current_fraction: float = 0.25,
    max_iterations: int = 512,
    tolerance: float = 1e-8,
) -> JointProgressStep:
    """Solve the joint candidate-progress-protection program.

    Let x denote the descent displacement, so the deployed update is theta-x.
    For each smooth objective f with projected gradient g and smoothness L, the
    certified decrease is g^T x - L||x||^2/2.  The oracle:

    1. finds the best current-loss decrease that does not increase any frozen
       certified candidate objective;
    2. requires at least ``current_fraction`` of the ordinary protected current
       decrease; and
    3. among those steps, maximises the selected candidate's certified decrease.

    The feasible sets are intersections of Euclidean balls, and the selected
    objective is equivalent to Euclidean projection of g/L onto that
    intersection.  Positive rescaling of any objective leaves the step
    unchanged.
    """

    rho = float(current_fraction)
    if not 0.0 <= rho < 1.0:
        raise ValueError("current_fraction must lie in [0,1)")
    if len(candidate_gradients) != len(candidate_smoothness):
        raise ValueError("candidate gradients and smoothness lists must have equal length")
    if selected_index is not None and not 0 <= selected_index < len(candidate_gradients):
        raise IndexError("selected_index is outside the candidate list")
    if current_smoothness <= 0.0 or not math.isfinite(float(current_smoothness)):
        raise ValueError("joint progress protection requires a finite positive current smoothness bound")
    if any(value <= 0.0 or not math.isfinite(float(value)) for value in candidate_smoothness):
        raise ValueError("joint progress protection requires finite positive candidate smoothness bounds")

    a = project_to_allowed_free_subspace(current_gradient, protected_basis, allowed_mask)
    bs = [project_to_allowed_free_subspace(item, protected_basis, allowed_mask) for item in candidate_gradients]
    radius = max(float(radius), 0.0)
    zero = torch.zeros_like(a)
    if radius == 0.0:
        return JointProgressStep(
            available=False,
            displacement=zero,
            candidate_safe_displacement=zero,
            mode="zero",
            current_ordinary_best=0.0,
            current_safe_best=0.0,
            current_required=0.0,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            candidate_certified_decreases=tuple(0.0 for _ in bs),
            iterations=0,
            max_constraint_violation=0.0,
            solver_converged=True,
            obstruction="zero_safe_radius",
        )

    current_L = float(current_smoothness)
    candidate_L = [float(value) for value in candidate_smoothness]
    ordinary_best = unconstrained_ball_best(a, current_L, radius)
    required = rho * ordinary_best

    trust_ball = (zero, radius)
    candidate_balls: list[tuple[torch.Tensor, float]] = []
    for gradient, smoothness in zip(bs, candidate_L):
        center = gradient / smoothness
        candidate_balls.append((center, float(torch.linalg.vector_norm(gradient).item()) / smoothness))

    current_center = a / current_L
    current_safe, current_iterations, current_violation, current_optimality, current_converged = _dual_coordinate_project_balls(
        current_center,
        [trust_ball, *candidate_balls],
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    current_safe_best = quadratic_decrease(a, current_L, current_safe)
    candidate_at_current = tuple(
        quadratic_decrease(gradient, smoothness, current_safe)
        for gradient, smoothness in zip(bs, candidate_L)
    )
    geometry_scale = max(
        1.0,
        radius,
        float(torch.linalg.vector_norm(current_center).item()),
        *(float(torch.linalg.vector_norm(center).item()) for center, _ in candidate_balls),
    )
    geometry_tol = tolerance * geometry_scale
    current_objective_scale = max(
        float(torch.linalg.vector_norm(a).item()) * radius,
        current_L * radius * radius,
        1e-300,
    )
    current_tol = tolerance * current_objective_scale
    candidate_tols = [
        tolerance
        * max(
            float(torch.linalg.vector_norm(gradient).item()) * radius,
            smoothness * radius * radius,
            1e-300,
        )
        for gradient, smoothness in zip(bs, candidate_L)
    ]
    if not current_converged:
        return JointProgressStep(
            available=False,
            displacement=zero,
            candidate_safe_displacement=zero,
            mode="zero",
            current_ordinary_best=ordinary_best,
            current_safe_best=current_safe_best,
            current_required=required,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            candidate_certified_decreases=tuple(0.0 for _ in bs),
            iterations=current_iterations,
            max_constraint_violation=max(current_violation, current_optimality),
            solver_converged=False,
            obstruction="candidate_safe_current_numerical_abstention",
        )
    if current_violation > 10.0 * geometry_tol or any(
        value < -10.0 * objective_tol
        for value, objective_tol in zip(candidate_at_current, candidate_tols)
    ):
        return JointProgressStep(
            available=False,
            displacement=zero,
            candidate_safe_displacement=zero,
            mode="zero",
            current_ordinary_best=ordinary_best,
            current_safe_best=current_safe_best,
            current_required=required,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            candidate_certified_decreases=tuple(0.0 for _ in bs),
            iterations=current_iterations,
            max_constraint_violation=max(current_violation, max((-v for v in candidate_at_current), default=0.0)),
            solver_converged=True,
            obstruction="candidate_safe_current_solver_residual",
        )
    if current_safe_best + current_tol < required:
        return JointProgressStep(
            available=False,
            displacement=zero,
            candidate_safe_displacement=zero,
            mode="zero",
            current_ordinary_best=ordinary_best,
            current_safe_best=current_safe_best,
            current_required=required,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            candidate_certified_decreases=tuple(0.0 for _ in bs),
            iterations=current_iterations,
            max_constraint_violation=max(current_violation, required - current_safe_best),
            solver_converged=True,
            obstruction="joint_current_progress_incompatible",
        )

    # The current-progress superlevel set is a ball around a/L.
    current_radius_sq = float(torch.dot(a, a).item()) / (current_L * current_L) - 2.0 * required / current_L
    current_ball = (current_center, math.sqrt(max(current_radius_sq, 0.0)))

    if selected_index is None or not bs:
        return JointProgressStep(
            available=current_safe_best > current_tol,
            displacement=current_safe,
            candidate_safe_displacement=current_safe,
            mode="candidate_safe_current" if current_safe_best > current_tol else "zero",
            current_ordinary_best=ordinary_best,
            current_safe_best=current_safe_best,
            current_required=required,
            current_certified_decrease=current_safe_best,
            selected_certified_decrease=0.0,
            candidate_certified_decreases=candidate_at_current,
            iterations=current_iterations,
            max_constraint_violation=current_violation,
            solver_converged=True,
            obstruction=None if current_safe_best > current_tol else "zero_candidate_safe_current_progress",
        )

    selected_gradient = bs[selected_index]
    selected_L = candidate_L[selected_index]
    selected_center = selected_gradient / selected_L
    transfer_step, transfer_iterations, transfer_violation, transfer_optimality, transfer_converged = _dual_coordinate_project_balls(
        selected_center,
        [trust_ball, *candidate_balls, current_ball],
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    current_decrease = quadratic_decrease(a, current_L, transfer_step)
    candidate_decreases = tuple(
        quadratic_decrease(gradient, smoothness, transfer_step)
        for gradient, smoothness in zip(bs, candidate_L)
    )
    selected_decrease = candidate_decreases[selected_index]
    if not transfer_converged:
        return JointProgressStep(
            available=False,
            displacement=zero,
            candidate_safe_displacement=current_safe,
            mode="zero",
            current_ordinary_best=ordinary_best,
            current_safe_best=current_safe_best,
            current_required=required,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            candidate_certified_decreases=tuple(0.0 for _ in bs),
            iterations=current_iterations + transfer_iterations,
            max_constraint_violation=max(transfer_violation, transfer_optimality),
            solver_converged=False,
            obstruction="joint_projection_numerical_abstention",
        )
    objective_violations = [max(required - current_decrease, 0.0) / max(current_objective_scale, 1e-300)]
    objective_violations.extend(
        max(-value, 0.0)
        / max(
            float(torch.linalg.vector_norm(gradient).item()) * radius
            + smoothness * radius * radius,
            1e-300,
        )
        for value, gradient, smoothness in zip(candidate_decreases, bs, candidate_L)
    )
    max_violation = max(
        transfer_violation / max(geometry_scale, 1e-300),
        max(objective_violations, default=0.0),
    )
    constraint_failed = (
        transfer_violation > 10.0 * geometry_tol
        or current_decrease + current_tol < required
        or any(
            value < -10.0 * objective_tol
            for value, objective_tol in zip(candidate_decreases, candidate_tols)
        )
    )
    if constraint_failed:
        return JointProgressStep(
            available=False,
            displacement=zero,
            candidate_safe_displacement=zero,
            mode="zero",
            current_ordinary_best=ordinary_best,
            current_safe_best=current_safe_best,
            current_required=required,
            current_certified_decrease=0.0,
            selected_certified_decrease=0.0,
            candidate_certified_decreases=tuple(0.0 for _ in bs),
            iterations=current_iterations + transfer_iterations,
            max_constraint_violation=max_violation,
            solver_converged=True,
            obstruction="joint_projection_solver_residual",
        )

    selected_tol = candidate_tols[selected_index]
    if selected_decrease <= selected_tol:
        return JointProgressStep(
            available=current_safe_best > current_tol,
            displacement=current_safe,
            candidate_safe_displacement=current_safe,
            mode="candidate_safe_current" if current_safe_best > current_tol else "zero",
            current_ordinary_best=ordinary_best,
            current_safe_best=current_safe_best,
            current_required=required,
            current_certified_decrease=current_safe_best,
            selected_certified_decrease=selected_decrease,
            candidate_certified_decreases=candidate_at_current,
            iterations=current_iterations + transfer_iterations,
            max_constraint_violation=max_violation,
            solver_converged=True,
            obstruction="joint_transfer_incompatible",
        )

    return JointProgressStep(
        available=True,
        displacement=transfer_step,
        candidate_safe_displacement=current_safe,
        mode="joint_transfer",
        current_ordinary_best=ordinary_best,
        current_safe_best=current_safe_best,
        current_required=required,
        current_certified_decrease=current_decrease,
        selected_certified_decrease=selected_decrease,
        candidate_certified_decreases=candidate_decreases,
        iterations=current_iterations + transfer_iterations,
        max_constraint_violation=max_violation,
        solver_converged=True,
        obstruction=None,
    )


def make_safe_step_from_direction(
    direction: torch.Tensor,
    descent_scale: float,
    E: float,
    H: float,
    budget: float,
    cap: float,
    learning_rate: float,
    allowed_mask: torch.Tensor | None = None,
) -> SafeStep:
    """Construct a radius-limited step along a precomputed feasible direction."""

    qnorm = float(torch.linalg.vector_norm(direction).item())
    radius = safe_radius(E=E, H=H, budget=budget, cap=cap)
    if qnorm == 0.0 or radius == 0.0 or descent_scale <= 0.0:
        return SafeStep(
            radius=radius,
            step_length=0.0,
            projected_gradient_norm=qnorm,
            delta=torch.zeros_like(direction),
        )
    step_length = min(float(learning_rate) * float(descent_scale), radius)
    delta = -step_length * direction / qnorm
    if allowed_mask is not None:
        delta = delta * allowed_mask.to(device=delta.device, dtype=delta.dtype)
    return SafeStep(
        radius=radius,
        step_length=step_length,
        projected_gradient_norm=qnorm,
        delta=delta,
    )


def make_safe_step(
    gradient: torch.Tensor,
    basis: torch.Tensor | None,
    learning_rate: float,
    E: float,
    H: float,
    budget: float,
    cap: float,
    allowed_mask: torch.Tensor | None = None,
) -> SafeStep:
    q = project_to_allowed_free_subspace(gradient, basis, allowed_mask)
    qnorm = float(torch.linalg.vector_norm(q).item())
    return make_safe_step_from_direction(
        direction=q,
        descent_scale=qnorm,
        E=E,
        H=H,
        budget=budget,
        cap=cap,
        learning_rate=learning_rate,
        allowed_mask=allowed_mask,
    )
