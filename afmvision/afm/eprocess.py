from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class EProcessState:
    n: int = 0
    sum_x: float = 0.0
    log_wealth: float = 0.0
    wealth: float = 1.0


class HalfNormalMixtureEProcess:
    """Fixed-state half-normal mixture e-process from the AFM manuscript.

    Wealth is evaluated in the log domain with ``torch.special.log_ndtr``.
    Directly clamping an underflowed normal CDF can spuriously inflate wealth in
    the far negative tail, which would invalidate false-reopening control.
    """

    def __init__(self, sigma: float = 1.0, prior_scale: float = 1.0, alpha: float = 0.01):
        if sigma <= 0 or prior_scale <= 0 or not 0 < alpha < 1:
            raise ValueError("Invalid e-process parameters")
        self.sigma = float(sigma)
        self.prior_scale = float(prior_scale)
        self.alpha = float(alpha)
        self.state = EProcessState()

    @staticmethod
    def _log_normal_cdf(x: float) -> float:
        value = torch.special.log_ndtr(torch.tensor(float(x), dtype=torch.float64))
        return float(value.item())

    def update(self, x: float) -> float:
        self.state.n += 1
        self.state.sum_x += float(x)
        n = self.state.n
        s = self.state.sum_x
        tau = self.prior_scale
        A = self.sigma * self.sigma * n + tau ** -2
        log_wealth = (
            math.log(2.0)
            - math.log(tau)
            - 0.5 * math.log(A)
            + s * s / (2.0 * A)
            + self._log_normal_cdf(s / math.sqrt(A))
        )
        self.state.log_wealth = log_wealth
        self.state.wealth = math.exp(min(log_wealth, 700.0))
        return self.state.wealth

    @property
    def crossed(self) -> bool:
        return self.state.log_wealth >= -math.log(self.alpha)

    def reset(self) -> None:
        self.state = EProcessState()

    def state_dict(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma,
            "prior_scale": self.prior_scale,
            "alpha": self.alpha,
            "state": asdict(self.state),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "HalfNormalMixtureEProcess":
        obj = cls(float(state["sigma"]), float(state["prior_scale"]), float(state["alpha"]))
        state_values = dict(state.get("state", {}))
        # Backward compatibility with checkpoints created before log-wealth was
        # stored explicitly.  Those checkpoints remain numerically auditable.
        if "log_wealth" not in state_values:
            state_values["log_wealth"] = math.log(max(float(state_values.get("wealth", 1.0)), 1e-300))
        obj.state = EProcessState(**state_values)
        return obj
