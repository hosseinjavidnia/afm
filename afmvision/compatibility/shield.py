from __future__ import annotations

from dataclasses import dataclass

import torch

from afmvision.afm.functional_shield import FunctionalShield, ShieldSolveResult


class StaticAddressEncoder:
    """Immutable input-address map used by finite AFM completion.

    Unlike the original frozen-feature experiment, the causal sweep fully
    updates every learned model parameter.  Finite residuals are therefore
    indexed by a deterministic, nonlearned address of the raw input so their
    centres do not move when the representation changes.
    """

    def __init__(
        self,
        *,
        modality: str,
        input_shape: tuple[int, ...],
        address_dim: int = 64,
        seed: int = 20260817,
        vocab_size: int | None = None,
    ) -> None:
        self.modality = str(modality)
        self.address_dim = int(address_dim)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        if self.modality == "vision":
            source_dim = 1
            for x in input_shape:
                source_dim *= int(x)
            matrix = torch.randn(source_dim, self.address_dim, generator=gen, dtype=torch.float64)
            matrix = matrix / torch.sqrt(torch.tensor(float(source_dim), dtype=torch.float64))
            self.projection = matrix
            self.token_table = None
        elif self.modality == "text":
            if vocab_size is None:
                raise ValueError("vocab_size is required for text static addresses")
            token_dim = max(self.address_dim // 2, 8)
            self.token_table = torch.randn(int(vocab_size), token_dim, generator=gen, dtype=torch.float64)
            self.projection = torch.randn(2 * token_dim, self.address_dim, generator=gen, dtype=torch.float64)
            self.projection /= torch.sqrt(torch.tensor(float(2 * token_dim), dtype=torch.float64))
        else:
            raise ValueError(f"unknown modality: {modality}")

    @torch.no_grad()
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.modality == "vision":
            flat = inputs.detach().to(device="cpu", dtype=torch.float64).reshape(inputs.shape[0], -1)
            return (flat @ self.projection).to(device=inputs.device, dtype=torch.float32)
        ids = inputs.detach().to(device="cpu", dtype=torch.long)
        table = self.token_table
        assert table is not None
        emb = table[ids]
        mean = emb.mean(dim=1)
        weights = torch.linspace(0.5, 1.5, ids.shape[1], dtype=torch.float64).view(1, -1, 1)
        weighted = (emb * weights).mean(dim=1)
        features = torch.cat((mean, weighted), dim=1)
        return (features @ self.projection).to(device=inputs.device, dtype=torch.float32)


@dataclass(frozen=True)
class FiniteCompletionResult:
    available: bool
    solve: ShieldSolveResult
    endpoint_error: float
    protected_error: float
    current_error: float
    deployed_current_logits: torch.Tensor | None
    deployed_protected_logits: torch.Tensor | None


@torch.no_grad()
def finite_endpoint_completion(
    *,
    address_encoder: StaticAddressEncoder,
    base_current_logits: torch.Tensor,
    base_protected_logits: torch.Tensor,
    desired_current_logits: torch.Tensor,
    desired_protected_logits: torch.Tensor,
    current_inputs: torch.Tensor,
    protected_inputs: torch.Tensor,
    guard_inputs: torch.Tensor | None,
    support_multiplier: float = 4.0,
    feature_match_tolerance: float = 1e-8,
    target_tolerance: float = 1e-8,
    residual_tolerance: float = 1e-8,
) -> FiniteCompletionResult:
    current_addr = address_encoder(current_inputs)
    protected_addr = address_encoder(protected_inputs)
    nodes = torch.cat((current_addr, protected_addr), dim=0)
    desired = torch.cat((desired_current_logits, desired_protected_logits), dim=0)
    base = torch.cat((base_current_logits, base_protected_logits), dim=0)
    residual = desired - base
    guards = None if guard_inputs is None or guard_inputs.numel() == 0 else address_encoder(guard_inputs)
    shield = FunctionalShield(
        feature_dim=int(nodes.shape[1]),
        output_dim=int(desired.shape[1]),
        max_nodes=int(nodes.shape[0]),
        bandwidth=1.0,
    ).to(device=nodes.device)
    solve = shield.solve_and_replace(
        nodes,
        residual,
        guard_nodes=guards,
        support_multiplier=float(support_multiplier),
        feature_match_tolerance=float(feature_match_tolerance),
        duplicate_tolerance=1e-12,
        target_tolerance=float(target_tolerance),
        residual_tolerance=float(residual_tolerance),
    )
    if not solve.available:
        return FiniteCompletionResult(False, solve, float("inf"), float("inf"), float("inf"), None, None)
    deployed = base + shield(nodes)
    n_current = len(current_inputs)
    deployed_current = deployed[:n_current]
    deployed_protected = deployed[n_current:]
    current_error = float((deployed_current - desired_current_logits).abs().max().item())
    protected_error = float((deployed_protected - desired_protected_logits).abs().max().item())
    return FiniteCompletionResult(
        available=True,
        solve=solve,
        endpoint_error=max(current_error, protected_error),
        protected_error=protected_error,
        current_error=current_error,
        deployed_current_logits=deployed_current,
        deployed_protected_logits=deployed_protected,
    )
