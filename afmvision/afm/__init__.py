"""Public AFM primitives.

These exports cover the persistent-assimilation geometry and finite functional
shield without requiring users to import internal helper modules directly.
"""

from .functional_shield import FunctionalShield, ShieldSolveResult
from .persistent_assimilation import (
    PersistentAssimilationPlan,
    make_counterfactual_normalized_plan,
    normalized_retention_budget,
    persistent_descent_lower_bound,
    retention_charge,
)
from .safe_step import SafeStep, project_to_allowed_free_subspace, safe_radius

__all__ = [
    "FunctionalShield",
    "PersistentAssimilationPlan",
    "SafeStep",
    "ShieldSolveResult",
    "make_counterfactual_normalized_plan",
    "normalized_retention_budget",
    "persistent_descent_lower_bound",
    "project_to_allowed_free_subspace",
    "retention_charge",
    "safe_radius",
]
