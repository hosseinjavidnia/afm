from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class ShieldSolveResult:
    available: bool
    node_count: int
    merged_node_count: int
    # Retained compatibility fields.  The compact-cardinal construction has no
    # Gram solve, so these have exact neutral values on success.
    bandwidth: float
    condition_number: float
    minimum_eigenvalue: float
    interpolation_residual: float
    coefficient_norm: float
    obstruction: str | None = None
    conflicting_group_size: int = 0
    conflicting_target_spread: float = 0.0
    minimum_support_radius: float = 0.0
    maximum_support_radius: float = 0.0
    guard_count: int = 0
    maximum_guard_leakage: float = 0.0
    feature_match_tolerance: float = 0.0
    minimum_address_separation: float = 0.0
    support_multiplier: float = 0.0


class FunctionalShield(nn.Module):
    """Bounded compact-cardinal residual over frozen backbone features.

    The v0.7 Gaussian representer had infinite support: exact finite-node
    interpolation could therefore perturb every unrelated input and all later
    gradients.  This module replaces it with disjoint compact cardinal bumps.

    Frozen features are first mapped injectively into the open unit ball by
    ``a(z)=z/sqrt(1+||z||^2)``.  After consistent duplicate rows are merged, a
    centre receives a radius equal to a fixed multiple of the predeclared
    feature-replay error envelope.  Deployment is allowed only when twice that
    radius is smaller than every nonmatching centre-or-guard separation.  Hence
    supports are pairwise disjoint, exclude every nonmatching guard, and remain
    confined to a certified numerical neighbourhood of the finite evidence.
    The coefficient at each centre is its requested residual itself; no linear
    system, bandwidth search, jitter, or post-hoc threshold is used.

    Exact or numerically replayed centre addresses are snapped to their stored
    coefficient under the declared feature-match tolerance.  Away from the
    union of compact supports the shield is identically zero.  At any address
    at most one bump is active, so residuals cannot accumulate across centres.
    """

    KIND_CODE = 8

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        *,
        max_nodes: int = 1,
        bandwidth: float = 1.0,  # legacy constructor compatibility; unused
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or output_dim <= 0:
            raise ValueError("feature_dim and output_dim must be positive")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.max_nodes = int(max_nodes)
        self.register_buffer("centres", torch.empty((0, self.feature_dim), dtype=torch.float64))
        self.register_buffer("coefficients", torch.empty((0, self.output_dim), dtype=torch.float64))
        self.register_buffer("support_radii", torch.empty((0,), dtype=torch.float64))
        self.register_buffer("match_radii", torch.empty((0,), dtype=torch.float64))
        # Kept so older evaluators that inspect the key do not fail.  It is not
        # used by the compact-cardinal construction.
        self.register_buffer("bandwidth", torch.zeros((), dtype=torch.float64))
        self.register_buffer("generation", torch.zeros((), dtype=torch.long))
        self.register_buffer("kind_code", torch.tensor(self.KIND_CODE, dtype=torch.long))

    @property
    def node_count(self) -> int:
        return int(self.centres.shape[0])

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, output_dim={self.output_dim}, "
            f"max_nodes={self.max_nodes}, nodes={self.node_count}, kind=compact_cardinal"
        )

    @staticmethod
    def _address(features: torch.Tensor) -> torch.Tensor:
        work = features.to(dtype=torch.float64)
        scale = torch.sqrt(1.0 + work.square().sum(dim=1, keepdim=True))
        return work / scale

    @staticmethod
    def _distances(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Direct Euclidean distances with exact zero on identical rows.

        ``torch.cdist`` may return a small positive diagonal under its matrix-
        multiplication path.  Cardinal replay requires identical addresses to
        have exactly zero numerical distance, so the finite bounded matrices are
        evaluated by direct subtraction.
        """
        differences = left.unsqueeze(1) - right.unsqueeze(0)
        return torch.sqrt(torch.clamp(differences.square().sum(dim=2), min=0.0))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        centres_key = prefix + "centres"
        coefficients_key = prefix + "coefficients"
        support_key = prefix + "support_radii"
        match_key = prefix + "match_radii"
        kind_key = prefix + "kind_code"
        saved_centres = state_dict.get(centres_key)
        saved_coefficients = state_dict.get(coefficients_key)
        saved_support = state_dict.get(support_key)
        saved_match = state_dict.get(match_key)
        saved_kind = state_dict.get(kind_key)

        if saved_centres is not None and saved_coefficients is not None:
            rows = int(saved_centres.shape[0]) if saved_centres.ndim == 2 else -1
            if saved_centres.ndim != 2 or saved_centres.shape[1] != self.feature_dim:
                error_msgs.append(
                    f"size mismatch for {centres_key}: expected (*, {self.feature_dim}), "
                    f"found {tuple(saved_centres.shape)}"
                )
            elif saved_coefficients.shape != (rows, self.output_dim):
                error_msgs.append(
                    f"size mismatch for {coefficients_key}: expected ({rows}, {self.output_dim}), "
                    f"found {tuple(saved_coefficients.shape)}"
                )
            elif saved_support is None or tuple(saved_support.shape) != (rows,):
                error_msgs.append(
                    f"{support_key} is missing or incompatible. A v0.7 Gaussian checkpoint "
                    "cannot be resumed as v0.8 compact-cardinal state; start a fresh v0.8 run."
                )
            elif saved_match is None or tuple(saved_match.shape) != (rows,):
                error_msgs.append(f"{match_key} is missing or incompatible")
            elif saved_kind is None or int(saved_kind.item()) != self.KIND_CODE:
                error_msgs.append(
                    f"{kind_key} does not identify a v0.8 compact-cardinal shield; "
                    "cross-version checkpoint resume is forbidden"
                )
            else:
                device = self.centres.device
                self.centres = torch.empty((rows, self.feature_dim), device=device, dtype=torch.float64)
                self.coefficients = torch.empty((rows, self.output_dim), device=device, dtype=torch.float64)
                self.support_radii = torch.empty((rows,), device=device, dtype=torch.float64)
                self.match_radii = torch.empty((rows,), device=device, dtype=torch.float64)
                self.max_nodes = max(self.max_nodes, rows)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _compact_bump(
        distances: torch.Tensor,
        radii: torch.Tensor,
        plateau_radii: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """C1 plateau-cardinal bump supported on the declared radii.

        The value is exactly one throughout the numerical replay envelope and
        then decays as ``(1-s^2)^2`` to zero at the support boundary.
        """
        plateau = torch.zeros_like(radii) if plateau_radii is None else plateau_radii
        positive = radii > plateau
        widths = torch.where(positive, radii - plateau, torch.ones_like(radii))
        scaled = torch.clamp(
            (distances - plateau.unsqueeze(0)) / widths.unsqueeze(0), min=0.0
        )
        values = torch.clamp(1.0 - scaled.square(), min=0.0).square()
        return values * positive.to(values.dtype).unsqueeze(0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.node_count == 0:
            return torch.zeros(
                (features.shape[0], self.output_dim),
                device=features.device,
                dtype=features.dtype,
            )
        addresses = self._address(features)
        centres = self.centres.to(device=features.device, dtype=torch.float64)
        distances = self._distances(addresses, centres)
        support = self.support_radii.to(device=features.device, dtype=torch.float64)
        match = self.match_radii.to(device=features.device, dtype=torch.float64)
        weights = self._compact_bump(distances, support, match)

        values = weights @ self.coefficients.to(device=features.device, dtype=torch.float64)
        return values.to(dtype=features.dtype)

    @torch.no_grad()
    def clear(self) -> None:
        device = self.centres.device
        self.centres = torch.empty((0, self.feature_dim), device=device, dtype=torch.float64)
        self.coefficients = torch.empty((0, self.output_dim), device=device, dtype=torch.float64)
        self.support_radii = torch.empty((0,), device=device, dtype=torch.float64)
        self.match_radii = torch.empty((0,), device=device, dtype=torch.float64)
        self.bandwidth.zero_()
        self.generation.add_(1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "compact_cardinal",
            "kind_code": int(self.kind_code.item()),
            "centres": self.centres.detach().cpu().clone(),
            "coefficients": self.coefficients.detach().cpu().clone(),
            "support_radii": self.support_radii.detach().cpu().clone(),
            "match_radii": self.match_radii.detach().cpu().clone(),
            "bandwidth": 0.0,
            "generation": int(self.generation.item()),
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "max_nodes": self.max_nodes,
        }

    @torch.no_grad()
    def restore(self, state: dict[str, Any]) -> None:
        if str(state.get("kind", "")) != "compact_cardinal" or int(
            state.get("kind_code", -1)
        ) != self.KIND_CODE:
            raise ValueError(
                "Shield snapshot is not v0.8 compact-cardinal state; cross-version restore is forbidden"
            )
        if int(state.get("feature_dim", self.feature_dim)) != self.feature_dim:
            raise ValueError("Shield snapshot feature dimension mismatch")
        if int(state.get("output_dim", self.output_dim)) != self.output_dim:
            raise ValueError("Shield snapshot output dimension mismatch")
        centres = torch.as_tensor(state["centres"], dtype=torch.float64, device=self.centres.device)
        coefficients = torch.as_tensor(
            state["coefficients"], dtype=torch.float64, device=self.coefficients.device
        )
        support_radii = torch.as_tensor(
            state["support_radii"], dtype=torch.float64, device=self.support_radii.device
        )
        match_radii = torch.as_tensor(
            state["match_radii"], dtype=torch.float64, device=self.match_radii.device
        )
        if centres.ndim != 2 or centres.shape[1] != self.feature_dim:
            raise ValueError("Invalid shield centre shape")
        if coefficients.shape != (centres.shape[0], self.output_dim):
            raise ValueError("Invalid shield coefficient shape")
        if support_radii.shape != (centres.shape[0],) or match_radii.shape != (centres.shape[0],):
            raise ValueError("Invalid shield radius shape")
        if centres.shape[0] > self.max_nodes:
            raise ValueError("Shield snapshot exceeds max_nodes")
        if bool((support_radii < 0.0).any().item()) or bool((match_radii < 0.0).any().item()):
            raise ValueError("Shield radii must be nonnegative")
        self.centres = centres.clone()
        self.coefficients = coefficients.clone()
        self.support_radii = support_radii.clone()
        self.match_radii = match_radii.clone()
        self.bandwidth.zero_()
        self.generation.fill_(int(state.get("generation", 0)))
        self.kind_code.fill_(self.KIND_CODE)

    @staticmethod
    def _merge_consistent_nodes(
        nodes: torch.Tensor,
        targets: torch.Tensor,
        *,
        duplicate_tolerance: float,
        target_tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int, float] | ShieldSolveResult:
        nodes64 = nodes.detach().to(dtype=torch.float64, device="cpu")
        targets64 = targets.detach().to(dtype=torch.float64, device="cpu")
        remaining = set(range(len(nodes64)))
        merged_nodes: list[torch.Tensor] = []
        merged_targets: list[torch.Tensor] = []
        largest_group = 0
        largest_spread = 0.0
        while remaining:
            first = min(remaining)
            centre = nodes64[first]
            group = [
                index
                for index in sorted(remaining)
                if float(torch.linalg.vector_norm(nodes64[index] - centre).item())
                <= duplicate_tolerance
            ]
            group_targets = targets64[group]
            target_mean = group_targets.mean(dim=0)
            spread = float((group_targets - target_mean).abs().max().item())
            largest_group = max(largest_group, len(group))
            largest_spread = max(largest_spread, spread)
            if spread > target_tolerance:
                return ShieldSolveResult(
                    available=False,
                    node_count=len(nodes64),
                    merged_node_count=len(merged_nodes),
                    bandwidth=0.0,
                    condition_number=1.0,
                    minimum_eigenvalue=1.0,
                    interpolation_residual=float("inf"),
                    coefficient_norm=0.0,
                    obstruction="functional_constraint_inconsistency",
                    conflicting_group_size=len(group),
                    conflicting_target_spread=spread,
                )
            merged_nodes.append(nodes64[group].mean(dim=0))
            merged_targets.append(target_mean)
            remaining.difference_update(group)
        return (
            torch.stack(merged_nodes, dim=0),
            torch.stack(merged_targets, dim=0),
            largest_group,
            largest_spread,
        )

    @staticmethod
    def _radii_from_replay_envelope(
        centres: torch.Tensor,
        guards: torch.Tensor,
        *,
        support_multiplier: float,
        feature_match_tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        """Build tiny supports from the declared replay-error envelope.

        The support radius is ``support_multiplier * feature_match_tolerance``.
        It is not enlarged to fill the available separation.  A solve is
        admissible only when every nonmatching centre-or-guard address is more
        than twice that radius away.  The factor two is stronger than needed
        for guard exclusion and gives pairwise-disjoint closed supports.
        """
        count = int(centres.shape[0])
        nearest = torch.full((count,), float("inf"), dtype=torch.float64)
        if count > 1:
            pairwise = FunctionalShield._distances(centres, centres)
            pairwise.fill_diagonal_(float("inf"))
            nearest = torch.minimum(nearest, pairwise.min(dim=1).values)
        if guards.numel() > 0:
            guard_distances = FunctionalShield._distances(centres, guards)
            # A guard inside the declared replay envelope is the same observable
            # address for certification purposes; it cannot simultaneously be
            # required to carry a zero residual.
            guard_distances = torch.where(
                guard_distances <= feature_match_tolerance,
                torch.full_like(guard_distances, float("inf")),
                guard_distances,
            )
            nearest = torch.minimum(nearest, guard_distances.min(dim=1).values)

        radius_value = support_multiplier * feature_match_tolerance
        support = torch.full((count,), radius_value, dtype=torch.float64)
        match = torch.full((count,), feature_match_tolerance, dtype=torch.float64)
        finite = nearest[torch.isfinite(nearest)]
        minimum_separation = float(finite.min().item()) if finite.numel() else float("inf")
        separated = bool(finite.numel() == 0 or (finite > (2.0 * radius_value)).all().item())
        return support, match, minimum_separation, separated

    @torch.no_grad()
    def solve_and_replace(
        self,
        nodes: torch.Tensor,
        residual_targets: torch.Tensor,
        *,
        guard_nodes: torch.Tensor | None = None,
        support_multiplier: float = 4.0,
        feature_match_tolerance: float = 1e-8,
        duplicate_tolerance: float = 1e-10,
        target_tolerance: float = 1e-8,
        residual_tolerance: float = 1e-8,
        coefficient_norm_limit: float = float("inf"),
        # Accepted only to make accidental old call sites fail semantically
        # rather than syntactically during migration. They are ignored.
        bandwidth_rule: str | None = None,
        fixed_bandwidth: float | None = None,
        bandwidth_floor: float | None = None,
        bandwidth_ceiling: float | None = None,
        condition_limit: float | None = None,
        bandwidth_shrink: float | None = None,
        max_bandwidth_trials: int | None = None,
    ) -> ShieldSolveResult:
        if nodes.ndim != 2 or nodes.shape[1] != self.feature_dim:
            raise ValueError("Functional-shield nodes have the wrong shape")
        if residual_targets.shape != (nodes.shape[0], self.output_dim):
            raise ValueError("Functional-shield target shape mismatch")
        if support_multiplier <= 1.0:
            raise ValueError("functional_shield.support_multiplier must be greater than one")
        if feature_match_tolerance <= 0.0:
            raise ValueError("functional_shield.feature_match_tolerance must be positive")
        if duplicate_tolerance < 0.0:
            raise ValueError("Functional-shield duplicate tolerance must be nonnegative")
        if len(nodes) == 0:
            self.clear()
            return ShieldSolveResult(
                available=True,
                node_count=0,
                merged_node_count=0,
                bandwidth=0.0,
                condition_number=1.0,
                minimum_eigenvalue=1.0,
                interpolation_residual=0.0,
                coefficient_norm=0.0,
                feature_match_tolerance=feature_match_tolerance,
            )
        if len(nodes) > self.max_nodes:
            return ShieldSolveResult(
                available=False,
                node_count=len(nodes),
                merged_node_count=0,
                bandwidth=0.0,
                condition_number=1.0,
                minimum_eigenvalue=1.0,
                interpolation_residual=float("inf"),
                coefficient_norm=0.0,
                obstruction="functional_shield_capacity_exceeded",
            )

        addressed_nodes = self._address(nodes.detach()).to(device="cpu", dtype=torch.float64)
        merged = self._merge_consistent_nodes(
            addressed_nodes,
            residual_targets,
            duplicate_tolerance=float(duplicate_tolerance),
            target_tolerance=float(target_tolerance),
        )
        if isinstance(merged, ShieldSolveResult):
            return merged
        merged_nodes, merged_targets, largest_group, largest_spread = merged
        if len(merged_nodes) > self.max_nodes:
            return ShieldSolveResult(
                available=False,
                node_count=len(nodes),
                merged_node_count=len(merged_nodes),
                bandwidth=0.0,
                condition_number=1.0,
                minimum_eigenvalue=1.0,
                interpolation_residual=float("inf"),
                coefficient_norm=0.0,
                obstruction="functional_shield_capacity_exceeded",
            )

        if guard_nodes is None or guard_nodes.numel() == 0:
            guards = torch.empty((0, self.feature_dim), dtype=torch.float64)
        else:
            if guard_nodes.ndim != 2 or guard_nodes.shape[1] != self.feature_dim:
                raise ValueError("Functional-shield guard nodes have the wrong shape")
            guards = self._address(guard_nodes.detach()).to(device="cpu", dtype=torch.float64)

        support, match, minimum_separation, separated = self._radii_from_replay_envelope(
            merged_nodes,
            guards,
            support_multiplier=float(support_multiplier),
            feature_match_tolerance=float(feature_match_tolerance),
        )
        if not separated:
            return ShieldSolveResult(
                available=False,
                node_count=len(nodes),
                merged_node_count=len(merged_nodes),
                bandwidth=0.0,
                condition_number=1.0,
                minimum_eigenvalue=1.0,
                interpolation_residual=float("inf"),
                coefficient_norm=0.0,
                obstruction="functional_shield_address_resolution_obstruction",
                minimum_support_radius=float(support.min().item()),
                maximum_support_radius=float(support.max().item()),
                guard_count=int(guards.shape[0]),
                feature_match_tolerance=float(feature_match_tolerance),
                minimum_address_separation=minimum_separation,
                support_multiplier=float(support_multiplier),
            )
        coefficient_norm = float(torch.linalg.vector_norm(merged_targets).item())
        if not torch.isfinite(merged_targets).all():
            return ShieldSolveResult(
                available=False,
                node_count=len(nodes),
                merged_node_count=len(merged_nodes),
                bandwidth=0.0,
                condition_number=1.0,
                minimum_eigenvalue=1.0,
                interpolation_residual=float("inf"),
                coefficient_norm=coefficient_norm,
                obstruction="functional_shield_nonfinite_target_obstruction",
            )
        if coefficient_norm > coefficient_norm_limit:
            return ShieldSolveResult(
                available=False,
                node_count=len(nodes),
                merged_node_count=len(merged_nodes),
                bandwidth=0.0,
                condition_number=1.0,
                minimum_eigenvalue=1.0,
                interpolation_residual=0.0,
                coefficient_norm=coefficient_norm,
                obstruction="functional_shield_coefficient_norm_obstruction",
            )

        # Direct coefficients plus cardinal supports give exact interpolation;
        # verify the executable map before atomic replacement.
        old = self.snapshot()
        self.centres = merged_nodes.to(device=self.centres.device, dtype=torch.float64)
        self.coefficients = merged_targets.to(device=self.coefficients.device, dtype=torch.float64)
        self.support_radii = support.to(device=self.support_radii.device, dtype=torch.float64)
        self.match_radii = match.to(device=self.match_radii.device, dtype=torch.float64)
        self.bandwidth.zero_()
        predicted = self(nodes.detach().to(device=self.centres.device)).to(dtype=torch.float64, device="cpu")
        requested = residual_targets.detach().to(dtype=torch.float64, device="cpu")
        residual = float((predicted - requested).abs().max().item())
        if residual > residual_tolerance or not torch.isfinite(predicted).all():
            self.restore(old)
            return ShieldSolveResult(
                available=False,
                node_count=len(nodes),
                merged_node_count=len(merged_nodes),
                bandwidth=0.0,
                condition_number=1.0,
                minimum_eigenvalue=1.0,
                interpolation_residual=residual,
                coefficient_norm=coefficient_norm,
                obstruction="functional_shield_cardinality_residual",
            )

        maximum_guard_leakage = 0.0
        if guards.numel() > 0:
            guard_distances = self._distances(guards, self.centres.detach().cpu())
            nearest, index = guard_distances.min(dim=1)
            nonmatching = nearest > self.match_radii.detach().cpu()[index]
            if bool(nonmatching.any().item()):
                guard_weights = self._compact_bump(
                    guard_distances[nonmatching],
                    self.support_radii.detach().cpu(),
                    self.match_radii.detach().cpu(),
                )
                guard_values = guard_weights @ self.coefficients.detach().cpu()
                maximum_guard_leakage = float(guard_values.abs().max().item())
            if maximum_guard_leakage > residual_tolerance:
                self.restore(old)
                return ShieldSolveResult(
                    available=False,
                    node_count=len(nodes),
                    merged_node_count=len(merged_nodes),
                    bandwidth=0.0,
                    condition_number=1.0,
                    minimum_eigenvalue=1.0,
                    interpolation_residual=residual,
                    coefficient_norm=coefficient_norm,
                    obstruction="functional_shield_guard_leakage_obstruction",
                    maximum_guard_leakage=maximum_guard_leakage,
                )

        self.generation.add_(1)
        positive_support = support[support > 0.0]
        return ShieldSolveResult(
            available=True,
            node_count=len(nodes),
            merged_node_count=len(merged_nodes),
            bandwidth=0.0,
            condition_number=1.0,
            minimum_eigenvalue=1.0,
            interpolation_residual=residual,
            coefficient_norm=coefficient_norm,
            conflicting_group_size=largest_group,
            conflicting_target_spread=largest_spread,
            minimum_support_radius=(
                float(positive_support.min().item()) if positive_support.numel() else 0.0
            ),
            maximum_support_radius=float(support.max().item()) if support.numel() else 0.0,
            guard_count=int(guards.shape[0]),
            maximum_guard_leakage=maximum_guard_leakage,
            feature_match_tolerance=float(feature_match_tolerance),
            minimum_address_separation=minimum_separation,
            support_multiplier=float(support_multiplier),
        )


@contextmanager
def temporary_shield(shield: FunctionalShield, state: dict[str, Any]) -> Iterator[None]:
    original = shield.snapshot()
    shield.restore(state)
    try:
        yield
    finally:
        shield.restore(original)
