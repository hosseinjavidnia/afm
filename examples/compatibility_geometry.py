"""Minimal same-state AFM compatibility calculation.

This example is deliberately model-free.  It demonstrates the core persistent
assimilation operator from an ordinary comparator displacement, an ordinary
gradient, and a protected functional basis.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from afmvision.afm.persistent_assimilation import make_counterfactual_normalized_plan


def main() -> None:
    gradient = torch.tensor([3.0, 4.0], dtype=torch.float64)
    ordinary_delta = -0.02 * gradient

    # Protect the first coordinate. The compatible component is therefore the
    # second coordinate, giving kappa = 4^2 / (3^2 + 4^2) = 16/25.
    protected_basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)

    plan = make_counterfactual_normalized_plan(
        counterfactual_delta=ordinary_delta,
        ordinary_gradient=gradient,
        protected_basis=protected_basis,
        allowed_mask=torch.ones(2, dtype=torch.float64),
        E=2.0,
        H=3.0,
        charge_fraction=0.5,
        cap=0.2,
        active_protection=True,
        loss_smoothness=25.0,
    )

    result = {
        "compatibility_kappa": plan.compatibility_fraction,
        "requested_charge_fraction_eta": plan.requested_charge_fraction,
        "selected_path_fraction_lambda": plan.selected_path_fraction,
        "lambda_ge_eta": plan.selected_path_fraction + 1e-12 >= plan.requested_charge_fraction,
        "analytic_ratio_lower_bound": plan.analytic_persistent_progress_ratio_lower_bound,
        "persistent_delta": plan.proposal.delta.tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
