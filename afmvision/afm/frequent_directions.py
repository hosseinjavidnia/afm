from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FDSnapshot:
    matrix: torch.Tensor
    delta: float
    rows_seen: int


class FrequentDirections:
    """Deterministic Frequent Directions sketch for row streams.

    The sketch keeps 2*ell rows internally and compresses to ell rows. The
    accumulated shrinkage delta satisfies ||Cv||^2 - ||Bv||^2 <= delta ||v||^2.
    """

    def __init__(self, ell: int, dimension: int, device: torch.device | str = "cpu", dtype=torch.float32):
        if ell <= 0:
            raise ValueError("ell must be positive")
        self.ell = int(ell)
        self.dimension = int(dimension)
        self.device = torch.device(device)
        self.dtype = dtype
        self._rows: list[torch.Tensor] = []
        self.delta = 0.0
        self.rows_seen = 0

    def append(self, row: torch.Tensor) -> None:
        row = row.detach().reshape(-1).to(device=self.device, dtype=self.dtype)
        if row.numel() != self.dimension:
            raise ValueError(f"Expected row length {self.dimension}, got {row.numel()}")
        self._rows.append(row)
        self.rows_seen += 1
        if len(self._rows) >= 2 * self.ell:
            self._compress()

    def extend(self, rows: torch.Tensor) -> None:
        for row in rows:
            self.append(row)

    def _compress(self) -> None:
        if not self._rows:
            return
        matrix = torch.stack(self._rows, dim=0)
        # The sketch has at most 2*ell rows.  Work with its small row Gram
        # matrix on CPU float64 instead of invoking a CUDA SVD on a very wide
        # matrix.  This is the same singular system used by Frequent
        # Directions, reconstructed from B B^T.
        work = matrix.detach().to(device="cpu", dtype=torch.float64)
        gram = work @ work.T
        eigenvalues, left = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = torch.clamp(eigenvalues.index_select(0, order), min=0.0)
        left = left.index_select(1, order)
        singular = torch.sqrt(eigenvalues)
        if singular.numel() <= self.ell:
            self._rows = [row.to(device=self.device, dtype=self.dtype) for row in work]
            return
        scale = max(float(singular.max().item()), 1.0)
        tolerance = max(scale * 1e-12, 1e-14)
        denominator = torch.where(singular > tolerance, singular, torch.ones_like(singular))
        vh = (left.T @ work) / denominator.unsqueeze(1)
        vh = torch.where((singular > tolerance).unsqueeze(1), vh, torch.zeros_like(vh))
        shrink = float(singular[self.ell].square().item())
        kept_sq = torch.clamp(singular[: self.ell].square() - shrink, min=0.0)
        reduced = torch.sqrt(kept_sq).unsqueeze(1) * vh[: self.ell]
        self.delta += shrink
        self._rows = [row.to(device=self.device, dtype=self.dtype) for row in reduced]


    def state_dict(self) -> dict[str, Any]:
        return {
            "ell": self.ell,
            "dimension": self.dimension,
            "dtype": str(self.dtype),
            "rows": [row.detach().cpu() for row in self._rows],
            "delta": float(self.delta),
            "rows_seen": int(self.rows_seen),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], device: torch.device | str = "cpu") -> "FrequentDirections":
        obj = cls(int(state["ell"]), int(state["dimension"]), device=device, dtype=torch.float32)
        obj._rows = [row.to(device=obj.device, dtype=obj.dtype) for row in state.get("rows", [])]
        obj.delta = float(state.get("delta", 0.0))
        obj.rows_seen = int(state.get("rows_seen", len(obj._rows)))
        return obj

    def snapshot(self, scale: float = 1.0, delta_scale: float | None = None) -> FDSnapshot:
        self._compress()
        if self._rows:
            matrix = torch.stack(self._rows, dim=0)
        else:
            matrix = torch.zeros((0, self.dimension), device=self.device, dtype=self.dtype)
        if matrix.shape[0] < self.ell:
            pad = torch.zeros((self.ell - matrix.shape[0], self.dimension), device=self.device, dtype=self.dtype)
            matrix = torch.cat([matrix, pad], dim=0)
        matrix = matrix[: self.ell] * scale
        ds = scale * scale if delta_scale is None else delta_scale
        return FDSnapshot(matrix=matrix.detach().clone(), delta=float(self.delta * ds), rows_seen=self.rows_seen)
