from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * int(weight)
        self.count += int(weight)

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class ExperimentSummary:
    online_accuracy: RunningMean = field(default_factory=RunningMean)
    online_loss: RunningMean = field(default_factory=RunningMean)
    optimizer_steps: int = 0
    accepted_steps: int = 0
    nonzero_accepted_steps: int = 0
    zero_steps: int = 0
    rejected_steps: int = 0
    commits: int = 0
    reopenings: int = 0
    route_splits: int = 0
    capacity_releases: int = 0
    renewals_attempted: int = 0
    renewals_activated: int = 0
    renewal_capacity_obstructions: int = 0
    max_empirical_drift_violation: float = 0.0
    max_activation_gap: float = 0.0
    max_snapshot_drift_violation: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "online_accuracy": self.online_accuracy.mean,
            "online_loss": self.online_loss.mean,
            "optimizer_steps": self.optimizer_steps,
            "accepted_steps": self.accepted_steps,
            "nonzero_accepted_steps": self.nonzero_accepted_steps,
            "zero_steps": self.zero_steps,
            "rejected_steps": self.rejected_steps,
            "commits": self.commits,
            "reopenings": self.reopenings,
            "route_splits": self.route_splits,
            "capacity_releases": self.capacity_releases,
            "renewals_attempted": self.renewals_attempted,
            "renewals_activated": self.renewals_activated,
            "renewal_capacity_obstructions": self.renewal_capacity_obstructions,
            "max_empirical_drift_violation": self.max_empirical_drift_violation,
            "max_activation_gap": self.max_activation_gap,
            "max_snapshot_drift_violation": self.max_snapshot_drift_violation,
        }
