from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional

import math
import torch

from .functional_shield import FunctionalShield, ShieldSolveResult


@dataclass(frozen=True)
class RestorationBlock:
    features: torch.Tensor
    desired_logits: torch.Tensor
    name: str


@dataclass(frozen=True)
class FullProgressRestorationResult:
    available: bool
    solve: ShieldSolveResult
    current_count: int
    protected_count: int
    safeguard_count: int
    selected_count: int
    maximum_endpoint_error: float
    obstruction: str | None = None


@torch.no_grad()
def replace_with_endpoint_emulation(
    *,
    shield: FunctionalShield,
    base_logits_from_features,
    current_features: torch.Tensor,
    counterfactual_current_logits: torch.Tensor,
    protected_blocks: Iterable[RestorationBlock] = (),
    safeguard_blocks: Iterable[RestorationBlock] = (),
    selected_blocks: Iterable[RestorationBlock] = (),
    guard_features: torch.Tensor | None = None,
    support_multiplier: float = 4.0,
    feature_match_tolerance: float = 1e-8,
    duplicate_tolerance: float = 1e-10,
    target_tolerance: float = 1e-8,
    residual_tolerance: float = 1e-8,
    executable_endpoint_tolerance: Optional[float] = None,
    coefficient_norm_limit: float = float("inf"),
) -> FullProgressRestorationResult:
    """Atomically replace a compact shield around a certified safe base endpoint.

    The base parameters are assumed to have already been moved to the declared
    metaplastic safe endpoint. This function constructs one bounded finite
    shield that emulates the genuine no-protection current-batch logits while
    restoring every protected/safeguard block and applying any selected target
    block.  On failure, ``FunctionalShield.solve_and_replace`` is atomic and the
    previously deployed shield is left unchanged.  The executable endpoint
    check is a second transaction boundary: if finite-precision composition of
    the base logits and shield exceeds its separately declared envelope, the
    old shield is restored and a typed obstruction is returned.  The caller can
    then roll back the provisionally assigned base parameters.
    """

    if executable_endpoint_tolerance is None:
        executable_endpoint_tolerance = residual_tolerance
    if executable_endpoint_tolerance < 0.0:
        raise ValueError("Executable endpoint tolerance must be nonnegative")

    protected = list(protected_blocks)
    safeguards = list(safeguard_blocks)
    selected = list(selected_blocks)
    blocks = [
        RestorationBlock(current_features, counterfactual_current_logits, "current"),
        *protected,
        *safeguards,
        *selected,
    ]
    for block in blocks:
        if block.features.ndim != 2:
            raise ValueError(f"{block.name} features must be a matrix")
        if block.desired_logits.ndim != 2:
            raise ValueError(f"{block.name} desired logits must be a matrix")
        if block.features.shape[0] != block.desired_logits.shape[0]:
            raise ValueError(f"{block.name} feature/logit row mismatch")

    nodes = torch.cat([block.features for block in blocks], dim=0)
    desired = torch.cat([block.desired_logits for block in blocks], dim=0)
    base_after = base_logits_from_features(nodes).detach()
    residual_targets = desired - base_after
    previous_shield = shield.snapshot()
    solve = shield.solve_and_replace(
        nodes,
        residual_targets,
        guard_nodes=guard_features,
        support_multiplier=support_multiplier,
        feature_match_tolerance=feature_match_tolerance,
        duplicate_tolerance=duplicate_tolerance,
        target_tolerance=target_tolerance,
        residual_tolerance=residual_tolerance,
        coefficient_norm_limit=coefficient_norm_limit,
    )
    if not solve.available:
        return FullProgressRestorationResult(
            available=False,
            solve=solve,
            current_count=int(current_features.shape[0]),
            protected_count=sum(int(x.features.shape[0]) for x in protected),
            safeguard_count=sum(int(x.features.shape[0]) for x in safeguards),
            selected_count=sum(int(x.features.shape[0]) for x in selected),
            maximum_endpoint_error=float("inf"),
            obstruction=solve.obstruction,
        )

    realised = base_logits_from_features(nodes).detach() + shield(nodes).detach()
    error = float((realised - desired).abs().max().item()) if desired.numel() else 0.0
    if not math.isfinite(error) or error > executable_endpoint_tolerance:
        shield.restore(previous_shield)
        failed_solve = replace(
            solve,
            available=False,
            interpolation_residual=(
                error if not math.isfinite(error)
                else max(float(solve.interpolation_residual), error)
            ),
            obstruction="exact_counterfactual_executable_endpoint_obstruction",
        )
        return FullProgressRestorationResult(
            available=False,
            solve=failed_solve,
            current_count=int(current_features.shape[0]),
            protected_count=sum(int(x.features.shape[0]) for x in protected),
            safeguard_count=sum(int(x.features.shape[0]) for x in safeguards),
            selected_count=sum(int(x.features.shape[0]) for x in selected),
            maximum_endpoint_error=error,
            obstruction=failed_solve.obstruction,
        )
    return FullProgressRestorationResult(
        available=True,
        solve=solve,
        current_count=int(current_features.shape[0]),
        protected_count=sum(int(x.features.shape[0]) for x in protected),
        safeguard_count=sum(int(x.features.shape[0]) for x in safeguards),
        selected_count=sum(int(x.features.shape[0]) for x in selected),
        maximum_endpoint_error=error,
        obstruction=None,
    )


# Backward-compatible API name retained for v0.9 diagnostic consumers.
# v0.10+ uses ``replace_with_endpoint_emulation`` because the provisionally
# installed base endpoint is the budget-controlled metaplastic endpoint, not
# the no-protection endpoint.
replace_with_counterfactual_restoration = replace_with_endpoint_emulation
