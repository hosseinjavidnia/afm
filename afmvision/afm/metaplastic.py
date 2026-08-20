from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class Policy:
    alpha: float
    rank: int
    beta: tuple[float, ...]


@dataclass
class PolicyEvaluation:
    basis: torch.Tensor
    spectral_residual: float
    blocked_fraction: float
    frontier_cost: float
    bounded_loss: float


def scale_free_beta(alpha: float, K: int) -> tuple[float, ...]:
    rates = np.array([2.0 ** (-(k + 1)) for k in range(K)], dtype=np.float64)
    weights = rates ** (alpha - 1.0)
    weights /= weights.sum()
    return tuple(float(x) for x in weights)


def make_policy_family(alphas: Sequence[float], ranks: Sequence[int], K: int) -> list[Policy]:
    return [
        Policy(alpha=float(alpha), rank=int(rank), beta=scale_free_beta(float(alpha), K))
        for alpha in alphas
        for rank in ranks
    ]


def math_sqrt_tensor(weight: float, matrix: torch.Tensor) -> torch.Tensor:
    return matrix * float(weight) ** 0.5


def _canonicalise_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return basis
    out = basis.clone()
    for column in range(out.shape[1]):
        vector = out[:, column]
        pivot = int(torch.argmax(vector.abs()).item())
        if float(vector[pivot].item()) < 0.0:
            out[:, column].neg_()
    return out


def top_basis_from_weighted_sketches(
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float],
    rank: int,
    dimension: int,
    device: torch.device,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target_device = torch.device(device)
    target_dtype = (
        allowed_mask.dtype
        if allowed_mask is not None and allowed_mask.is_floating_point()
        else torch.float32
    )
    if rank <= 0 or not matrices:
        return torch.zeros((dimension, 0), device=target_device, dtype=target_dtype)

    # Protected sketches are bounded and stored on CPU.  Build the weighted
    # matrix and perform the small-row SVD in CPU float64, then move only the
    # resulting basis to the training device.  The mathematical operator is
    # unchanged; this removes a CUDA solver dependency from the first
    # protected update.
    mask_cpu = None if allowed_mask is None else allowed_mask.detach().to(device="cpu", dtype=torch.float64)
    rows: list[torch.Tensor] = []
    for matrix, weight in zip(matrices, weights):
        if not matrix.numel() or weight <= 0:
            continue
        B = matrix.detach().to(device="cpu", dtype=torch.float64)
        if mask_cpu is not None:
            B = B * mask_cpu.unsqueeze(0)
        rows.append(math_sqrt_tensor(weight, B))
    if not rows:
        return torch.zeros((dimension, 0), device=target_device, dtype=target_dtype)
    stacked = torch.cat(rows, dim=0)
    max_rank = min(rank, stacked.shape[0], stacked.shape[1])
    if max_rank <= 0:
        return torch.zeros((dimension, 0), device=target_device, dtype=target_dtype)
    gram = stacked @ stacked.T
    eigenvalues, left = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues.index_select(0, order), min=0.0)
    left = left.index_select(1, order)
    singular_values = torch.sqrt(eigenvalues)
    if singular_values.numel() == 0:
        return torch.zeros((dimension, 0), device=target_device, dtype=target_dtype)
    tolerance = max(float(singular_values.max().item()) * 1e-10, 1e-10)
    numerical_rank = int((singular_values > tolerance).sum().item())
    retained_rank = min(max_rank, numerical_rank)
    if retained_rank <= 0:
        return torch.zeros((dimension, 0), device=target_device, dtype=target_dtype)
    left = left[:, :retained_rank]
    singular_values = singular_values[:retained_rank]
    basis = stacked.T @ (left / singular_values.unsqueeze(0))
    basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True).clamp_min(1e-15)
    basis = _canonicalise_basis_signs(basis.contiguous())
    return basis.to(device=target_device, dtype=target_dtype)


def spectral_residual(
    matrices: Sequence[torch.Tensor],
    realised_weights: Sequence[float],
    basis: torch.Tensor,
    allowed_mask: torch.Tensor | None = None,
) -> float:
    total = 0.0
    protected = 0.0
    mask = None if allowed_mask is None else allowed_mask.to(device=basis.device)
    for matrix, weight in zip(matrices, realised_weights):
        B = matrix.to(device=basis.device)
        if mask is not None:
            B = B * mask.unsqueeze(0)
        total += float(weight) * float(B.square().sum().item())
        if basis.numel():
            protected += float(weight) * float((B @ basis).square().sum().item())
    return max(total - protected, 0.0)


def blocked_gradient_fraction(gradient: torch.Tensor, basis: torch.Tensor) -> float:
    if basis.numel() == 0:
        return 0.0
    blocked = basis.T @ gradient
    return float(blocked.square().sum().item() / (1.0 + gradient.square().sum().item()))


class MetaplasticController:
    def __init__(self, policies: list[Policy], horizon: int, seed: int, zeta: float, loss_bound: float):
        if not policies:
            raise ValueError("At least one policy is required")
        self.policies = policies
        self.horizon = max(int(horizon), 1)
        self.zeta = float(zeta)
        self.loss_bound = max(float(loss_bound), 1e-12)
        self.log_weights = np.zeros(len(policies), dtype=np.float64)
        self.eta = (8.0 * np.log(max(len(policies), 2)) / self.horizon) ** 0.5
        self.rng = np.random.default_rng(seed)
        self.cumulative_losses = np.zeros(len(policies), dtype=np.float64)
        self.selection_counts = np.zeros(len(policies), dtype=np.int64)
        self.selected_cumulative_loss = 0.0
        self.last_selected: int | None = None
        self.rounds = 0

    def probabilities(self) -> np.ndarray:
        shifted = self.log_weights - self.log_weights.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        return probs

    def sample(self) -> int:
        probs = self.probabilities()
        index = int(self.rng.choice(len(self.policies), p=probs))
        self.selection_counts[index] += 1
        self.last_selected = index
        return index

    def predicted_record_weights(self, policy_index: int, traces: Sequence[torch.Tensor]) -> list[float]:
        beta = torch.tensor(self.policies[policy_index].beta, dtype=torch.float64)
        out: list[float] = []
        for trace in traces:
            z = trace.detach().cpu().to(torch.float64)
            out.append(1.0 + float(torch.dot(beta, z).item()))
        return out

    def update(self, bounded_losses: Sequence[float]) -> None:
        losses = np.asarray(bounded_losses, dtype=np.float64)
        if losses.shape != self.log_weights.shape:
            raise ValueError("One bounded loss is required for every policy")
        if np.any(losses < -1e-12) or np.any(losses > 1.0 + 1e-12):
            raise ValueError("Hedge losses must lie in [0,1]; tighten the declared frontier bound")
        losses = np.clip(losses, 0.0, 1.0)
        self.cumulative_losses += losses
        if self.last_selected is None:
            raise RuntimeError("A policy must be sampled before updating Hedge")
        self.selected_cumulative_loss += float(losses[self.last_selected])
        self.log_weights -= self.eta * losses
        self.log_weights -= self.log_weights.max()
        self.rounds += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "zeta": self.zeta,
            "loss_bound": self.loss_bound,
            "log_weights": self.log_weights.copy(),
            "eta": self.eta,
            "rng_state": self.rng.bit_generator.state,
            "cumulative_losses": self.cumulative_losses.copy(),
            "selection_counts": self.selection_counts.copy(),
            "selected_cumulative_loss": self.selected_cumulative_loss,
            "last_selected": self.last_selected,
            "rounds": self.rounds,
        }

    @property
    def empirical_regret(self) -> float:
        if self.rounds == 0:
            return 0.0
        return float(self.selected_cumulative_loss - self.cumulative_losses.min())
