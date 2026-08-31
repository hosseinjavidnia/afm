from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from afmvision.afm.parameter_vector import ParameterVector, temporary_parameters
from afmvision.compatibility.models import CompatibilityModel


@dataclass(frozen=True)
class ProtectedGeometry:
    jacobian: torch.Tensor
    gram: torch.Tensor
    ridge: float

    @property
    def constraint_count(self) -> int:
        return int(self.jacobian.shape[0])

    @property
    def parameter_dim(self) -> int:
        return int(self.jacobian.shape[1])


@dataclass(frozen=True)
class CompatibilityTarget:
    requested_kappa: float
    measured_kappa: float
    achievable_min: float
    achievable_max: float
    clipped: bool
    residual: torch.Tensor
    teacher_measurements: torch.Tensor
    gradient: torch.Tensor
    gradient_norm: float
    low_mode_kappa: float
    high_mode_kappa: float


def _flatten_grad_tuple(params: list[nn.Parameter], grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            parts.append(torch.zeros_like(param).reshape(-1))
        else:
            parts.append(grad.reshape(-1))
    return torch.cat(parts)


def functional_jacobian(
    model: CompatibilityModel,
    inputs: torch.Tensor,
    *,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return functional outputs and their parameter Jacobian.

    The Jacobian rows correspond to the flattened declared functional outputs.
    All trainable model parameters are included; no representation prefix is
    frozen or omitted.
    """

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("compatibility model has no trainable parameters")
    model.zero_grad(set_to_none=True)
    outputs = model.functional_logits(inputs)
    flat = outputs.reshape(-1)
    rows: list[torch.Tensor] = []
    for index in range(flat.numel()):
        grads = torch.autograd.grad(
            flat[index],
            params,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        rows.append(_flatten_grad_tuple(params, grads))
    out = outputs if create_graph else outputs.detach()
    jac = torch.stack(rows, dim=0)
    if not create_graph:
        jac = jac.detach()
    return out, jac


def build_protected_geometry(jacobian: torch.Tensor, ridge: float = 1e-8) -> ProtectedGeometry:
    if jacobian.ndim != 2:
        raise ValueError("protected jacobian must be a matrix")
    work = jacobian.to(dtype=torch.float64)
    gram = work @ work.T
    if gram.numel():
        gram = gram + float(ridge) * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return ProtectedGeometry(jacobian=jacobian, gram=gram, ridge=float(ridge))


def _solve_protected_gram(
    geometry: ProtectedGeometry,
    rhs: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Solve the declared ridge-regularised protected Gram system robustly.

    ``geometry.gram`` is deliberately constructed in float64.  Do not cast it
    down to the native model precision before solving: for redundant protected
    constraints, a small declared ridge can disappear after a float32 cast and
    turn an analytically positive-definite system into a numerically singular
    one.  The primary path therefore solves the exact declared matrix in
    float64.  The eigensolver fallback is only a numerical recovery path for an
    SPD system whose mathematical eigenvalues are bounded below by ``ridge``.
    """

    gram = geometry.gram.to(device=device, dtype=torch.float64)
    rhs64 = rhs.to(device=device, dtype=torch.float64)
    try:
        return torch.linalg.solve(gram, rhs64)
    except torch._C._LinAlgError:
        sym = 0.5 * (gram + gram.T)
        evals, evecs = torch.linalg.eigh(sym)
        scale = max(float(evals.abs().max().item()), 1.0)
        numerical_floor = torch.finfo(torch.float64).eps * scale * max(int(gram.shape[0]), 1)
        floor = max(float(geometry.ridge), float(numerical_floor))
        evals = torch.clamp(evals, min=floor)
        projected_rhs = evecs.T @ rhs64
        if projected_rhs.ndim == 1:
            return evecs @ (projected_rhs / evals)
        return evecs @ (projected_rhs / evals.unsqueeze(1))


def protected_projection(vector: torch.Tensor, geometry: ProtectedGeometry) -> torch.Tensor:
    """Project a parameter vector into the declared protected row-space operator.

    The linear solve is carried out in float64 even when the network itself is
    float32.  This preserves the predeclared ridge term and prevents redundant
    protected functional constraints from creating a false singular-matrix
    failure.  The projected vector is returned in the caller's native dtype.
    """

    J64 = geometry.jacobian.to(device=vector.device, dtype=torch.float64)
    if J64.numel() == 0:
        return torch.zeros_like(vector)
    vector64 = vector.to(device=vector.device, dtype=torch.float64)
    rhs = J64 @ vector64
    coeff = _solve_protected_gram(geometry, rhs, device=vector.device)
    projected = J64.T @ coeff
    return projected.to(dtype=vector.dtype)


def compatible_projection(vector: torch.Tensor, geometry: ProtectedGeometry) -> torch.Tensor:
    return vector - protected_projection(vector, geometry)


def compatibility_fraction(vector: torch.Tensor, geometry: ProtectedGeometry, eps: float = 1e-20) -> float:
    denom = float(torch.dot(vector, vector).item())
    if denom <= eps:
        return 0.0
    allowed = compatible_projection(vector, geometry)
    return float(torch.dot(allowed, allowed).item()) / denom


def _generalised_modes(
    current_jacobian: torch.Tensor,
    geometry: ProtectedGeometry,
    *,
    eig_tolerance: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return G-orthonormal residual modes and their compatibility eigenvalues.

    For a current residual r, q=J_c^T r.  The ratio
    ||P q||^2 / ||q||^2 is a generalised Rayleigh quotient.  The returned
    residual modes diagonalise that quotient inside the current functional
    gradient subspace, allowing exact causal interpolation between low- and
    high-compatibility directions without injecting an arbitrary parameter
    gradient.
    """

    Jc = current_jacobian.to(dtype=torch.float64)
    Jp = geometry.jacobian.to(device=Jc.device, dtype=torch.float64)
    G = Jc @ Jc.T
    if Jp.numel() == 0:
        C = G.clone()
    else:
        K = Jp @ Jc.T
        C = G - K.T @ _solve_protected_gram(geometry, K, device=Jc.device)
    G = 0.5 * (G + G.T)
    C = 0.5 * (C + C.T)

    eval_g, evec_g = torch.linalg.eigh(G)
    scale = max(float(eval_g.max().item()), 1.0)
    keep = eval_g > float(eig_tolerance) * scale
    if int(keep.sum().item()) < 2:
        raise RuntimeError("current functional Jacobian has fewer than two independent gradient modes")
    U = evec_g[:, keep]
    s = eval_g[keep]
    W = U / torch.sqrt(s).unsqueeze(0)
    B = W.T @ C @ W
    B = 0.5 * (B + B.T)
    lambdas, Z = torch.linalg.eigh(B)
    lambdas = torch.clamp(lambdas, min=0.0, max=1.0)
    residual_modes = W @ Z
    return lambdas, residual_modes


def _native_teacher_gradient(
    model: CompatibilityModel,
    current_inputs: torch.Tensor,
    teacher_measurements: torch.Tensor,
) -> torch.Tensor:
    """Return the realised native-precision gradient of the teacher loss.

    The causal intervention is constructed from the explicit functional
    Jacobian, but the experiment must use the gradient that PyTorch actually
    realises for the model and precision being profiled.  This avoids relying
    on cancellation-sensitive J^T r reconstruction in large networks.
    """

    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    values = model.functional_logits(current_inputs)
    diff = values - teacher_measurements
    loss = 0.5 * diff.square().sum()
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
    return _flatten_grad_tuple(params, grads).detach().clone()


def make_controlled_targets(
    *,
    model: CompatibilityModel,
    current_inputs: torch.Tensor,
    current_measurements: torch.Tensor,
    current_jacobian: torch.Tensor,
    geometry: ProtectedGeometry,
    requested_kappas: Iterable[float],
    gradient_norm: float,
) -> list[CompatibilityTarget]:
    """Construct teacher targets that causally intervene on compatibility.

    The intervention is performed in function space.  Generalised residual
    eigenmodes of the current functional Jacobian are mixed so that the
    resulting teacher loss gradient has the requested compatibility whenever
    that value lies inside the achievable range.  Gradient norm is matched
    across requested levels.  The measured compatibility is always recomputed
    from the realised gradient and is the quantity intended for the x-axis.
    """

    lambdas, modes = _generalised_modes(current_jacobian, geometry)
    low_index = int(torch.argmin(lambdas).item())
    high_index = int(torch.argmax(lambdas).item())
    lo = float(lambdas[low_index].item())
    hi = float(lambdas[high_index].item())
    r_lo = modes[:, low_index]
    r_hi = modes[:, high_index]
    if hi - lo <= 1e-12:
        raise RuntimeError(f"compatibility geometry has no usable sweep range: [{lo:.6g}, {hi:.6g}]")

    Jc64 = current_jacobian.to(dtype=torch.float64)
    measurement_shape = current_measurements.shape
    results: list[CompatibilityTarget] = []

    for raw in requested_kappas:
        target = float(raw)
        clipped_target = min(max(target, lo), hi)
        clipped = abs(clipped_target - target) > 1e-9
        weight_hi = (clipped_target - lo) / (hi - lo)
        weight_hi = min(max(weight_hi, 0.0), 1.0)
        residual = (weight_hi ** 0.5) * r_hi + ((1.0 - weight_hi) ** 0.5) * r_lo
        q = Jc64.T @ residual
        q_norm = float(torch.linalg.vector_norm(q).item())
        if q_norm <= 1e-20:
            raise RuntimeError("constructed compatibility target has zero gradient")
        # First normalise analytically, then recompute the gradient through the
        # actual model in native precision.  Large end-to-end networks can make
        # J^T r cancellation-sensitive; the realised gradient is the quantity
        # that defines both the comparator direction and the reported kappa.
        scale = float(gradient_norm) / q_norm
        residual = residual * scale

        teacher = current_measurements.detach().to(dtype=torch.float64).reshape(-1) - residual
        teacher = teacher.reshape(measurement_shape).to(
            device=current_measurements.device, dtype=current_measurements.dtype
        )
        q_native = _native_teacher_gradient(model, current_inputs, teacher)
        realised_norm = float(torch.linalg.vector_norm(q_native).item())
        if realised_norm <= 1e-20:
            raise RuntimeError("constructed compatibility target has zero realised gradient")

        # Match the *realised* native gradient norm, not only the reconstructed
        # Jacobian product.  For squared teacher loss this scaling is linear at
        # the current state.  Recompute once after scaling to record the actual
        # direction used by all methods.
        realised_scale = float(gradient_norm) / realised_norm
        residual = residual * realised_scale
        teacher = current_measurements.detach().to(dtype=torch.float64).reshape(-1) - residual
        teacher = teacher.reshape(measurement_shape).to(
            device=current_measurements.device, dtype=current_measurements.dtype
        )
        q_native = _native_teacher_gradient(model, current_inputs, teacher)
        realised_norm = float(torch.linalg.vector_norm(q_native).item())
        if realised_norm <= 1e-20:
            raise RuntimeError("constructed compatibility target has zero realised gradient after scaling")
        realised = compatibility_fraction(q_native, geometry)
        results.append(
            CompatibilityTarget(
                requested_kappa=target,
                measured_kappa=realised,
                achievable_min=lo,
                achievable_max=hi,
                clipped=clipped,
                residual=residual.to(device=current_measurements.device, dtype=current_measurements.dtype),
                teacher_measurements=teacher,
                gradient=q_native,
                gradient_norm=float(torch.linalg.vector_norm(q_native).item()),
                low_mode_kappa=lo,
                high_mode_kappa=hi,
            )
        )
    return results


def teacher_loss(model: CompatibilityModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    values = model.functional_logits(inputs)
    if values.shape != targets.shape:
        raise ValueError(f"teacher target shape mismatch: {values.shape} vs {targets.shape}")
    diff = values - targets
    return 0.5 * diff.square().sum()


def gradient_of_loss(model: nn.Module, loss: torch.Tensor, vectoriser: ParameterVector) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return vectoriser.flatten_grads().detach().clone()


def evaluate_parameter_endpoint(
    *,
    model: CompatibilityModel,
    vectoriser: ParameterVector,
    vector: torch.Tensor,
    current_inputs: torch.Tensor,
    current_targets: torch.Tensor,
    protected_inputs: torch.Tensor,
    protected_full_logits: torch.Tensor,
) -> dict[str, float]:
    with temporary_parameters(vectoriser, vector):
        with torch.no_grad():
            current_loss = float(teacher_loss(model, current_inputs, current_targets).item())
            protected_after = model(protected_inputs)
            drift = protected_after - protected_full_logits
            max_abs = float(drift.abs().max().item()) if drift.numel() else 0.0
            rms = float(torch.sqrt(torch.mean(drift.square())).item()) if drift.numel() else 0.0
    return {
        "current_loss": current_loss,
        "protected_max_abs_drift": max_abs,
        "protected_rms_drift": rms,
    }
