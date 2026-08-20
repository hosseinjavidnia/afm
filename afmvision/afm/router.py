from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RouteDecision:
    slot: int | None
    created: bool
    overflow: bool
    distance: float


class BoundedCentroidRouter:
    """Predictable bounded centroid router with explicit evidence-driven splits.

    Ordinary routing is still nearest-centroid threshold routing.  A committed
    centroid may be frozen so unrelated later observations cannot drag it across
    signature space.  ``force_split`` is only called by the trainer after an
    anytime-valid predictive-evidence crossing and creates a new bounded route
    without rewriting or deleting the source route.
    """

    def __init__(
        self,
        max_slots: int,
        threshold: float,
        centroid_rate: float,
        freeze_committed_centroids: bool = True,
    ):
        self.max_slots = int(max_slots)
        self.threshold = float(threshold)
        self.centroid_rate = float(centroid_rate)
        self.freeze_committed_centroids = bool(freeze_committed_centroids)
        self.centroids: list[torch.Tensor | None] = [None] * self.max_slots
        self.last_seen: list[int] = [-1] * self.max_slots
        self.birth_step: list[int] = [-1] * self.max_slots
        self.committed: list[bool] = [False] * self.max_slots
        self.parent_slot: list[int | None] = [None] * self.max_slots

    def set_committed(self, slot: int, committed: bool) -> None:
        self.committed[slot] = committed

    def clear(self, slot: int) -> None:
        self.centroids[slot] = None
        self.last_seen[slot] = -1
        self.birth_step[slot] = -1
        self.committed[slot] = False
        self.parent_slot[slot] = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_slots": self.max_slots,
            "threshold": self.threshold,
            "centroid_rate": self.centroid_rate,
            "freeze_committed_centroids": self.freeze_committed_centroids,
            "centroids": [None if c is None else c.clone() for c in self.centroids],
            "last_seen": list(self.last_seen),
            "birth_step": list(self.birth_step),
            "committed": list(self.committed),
            "parent_slot": list(self.parent_slot),
        }

    def available_slot(self, exclude: set[int] | None = None) -> int | None:
        excluded = set() if exclude is None else set(exclude)
        empty = next(
            (i for i, centroid in enumerate(self.centroids) if centroid is None and i not in excluded),
            None,
        )
        if empty is not None:
            return empty
        uncertified = [
            i for i in range(self.max_slots) if i not in excluded and not self.committed[i]
        ]
        if uncertified:
            return min(uncertified, key=lambda i: (self.last_seen[i], i))
        return None

    def force_split(
        self,
        signature: torch.Tensor,
        step: int,
        source_slot: int,
        target_slot: int,
    ) -> RouteDecision:
        if target_slot == source_slot:
            raise ValueError("A route split requires a distinct target slot")
        z = signature.detach().flatten().cpu()
        self.centroids[target_slot] = z.clone()
        self.last_seen[target_slot] = int(step)
        self.birth_step[target_slot] = int(step)
        self.committed[target_slot] = False
        self.parent_slot[target_slot] = int(source_slot)
        return RouteDecision(slot=target_slot, created=True, overflow=False, distance=0.0)

    def route(self, signature: torch.Tensor, step: int) -> RouteDecision:
        z = signature.detach().flatten().cpu()
        occupied = [i for i, c in enumerate(self.centroids) if c is not None]
        if occupied:
            distances = [(i, float(torch.linalg.vector_norm(z - self.centroids[i]).item())) for i in occupied]
            # Newer evidence-created split routes win exact numerical ties.  The
            # rule is fixed and predictable before the stream.
            slot, distance = min(distances, key=lambda pair: (pair[1], -self.birth_step[pair[0]], pair[0]))
            if distance <= self.threshold:
                old = self.centroids[slot]
                assert old is not None
                if not (self.freeze_committed_centroids and self.committed[slot]):
                    self.centroids[slot] = (1.0 - self.centroid_rate) * old + self.centroid_rate * z
                self.last_seen[slot] = int(step)
                return RouteDecision(slot=slot, created=False, overflow=False, distance=distance)
        target = self.available_slot()
        if target is not None:
            self.centroids[target] = z.clone()
            self.last_seen[target] = int(step)
            self.birth_step[target] = int(step)
            self.committed[target] = False
            self.parent_slot[target] = None
            return RouteDecision(slot=target, created=True, overflow=False, distance=float("inf"))
        return RouteDecision(slot=None, created=False, overflow=True, distance=float("inf"))
