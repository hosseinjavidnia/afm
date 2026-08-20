from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LifetimeErrorAllocator:
    """Summable lifetime failure-budget allocator.

    Each category receives a fixed share of ``total_delta``. Event ``n`` in a
    category receives share * 6/(pi^2 (n+1)^2), so all allocations across an
    unbounded event sequence sum to at most the declared category share.
    """

    total_delta: float
    category_weights: dict[str, float]
    counters: dict[str, int] = field(default_factory=dict)
    allocated: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.total_delta) < 1.0:
            raise ValueError("total_delta must lie in (0,1)")
        if not self.category_weights:
            raise ValueError("At least one error-budget category is required")
        if any(float(v) < 0.0 for v in self.category_weights.values()):
            raise ValueError("Category weights must be nonnegative")
        total = sum(float(v) for v in self.category_weights.values())
        if total <= 0.0:
            raise ValueError("Category weights must have positive sum")
        self.category_weights = {str(k): float(v) / total for k, v in self.category_weights.items()}
        for category in self.category_weights:
            self.counters.setdefault(category, 0)
            self.allocated.setdefault(category, 0.0)

    def next(self, category: str) -> tuple[int, float]:
        if category not in self.category_weights:
            raise KeyError(f"Unknown error-budget category: {category}")
        index = int(self.counters[category])
        share = self.total_delta * self.category_weights[category]
        alpha = share * 6.0 / (math.pi**2 * (index + 1) ** 2)
        self.counters[category] = index + 1
        self.allocated[category] = float(self.allocated.get(category, 0.0) + alpha)
        return index, float(alpha)

    @property
    def total_allocated(self) -> float:
        return float(sum(self.allocated.values()))

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_delta": float(self.total_delta),
            "category_weights": dict(self.category_weights),
            "counters": dict(self.counters),
            "allocated": dict(self.allocated),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "LifetimeErrorAllocator":
        obj = cls(float(state["total_delta"]), dict(state["category_weights"]))
        obj.counters = {str(k): int(v) for k, v in state.get("counters", {}).items()}
        obj.allocated = {str(k): float(v) for k, v in state.get("allocated", {}).items()}
        return obj
