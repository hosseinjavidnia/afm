from __future__ import annotations

import math

import pytest
import torch

from afmvision.afm.persistent_assimilation import (
    make_counterfactual_normalized_plan,
    normalized_retention_budget,
    persistent_descent_lower_bound,
    retention_charge,
)


def test_normalized_charge_selects_at_least_declared_path_fraction() -> None:
    generator = torch.Generator().manual_seed(1101)
    for _ in range(500):
        dimension = 24
        gradient = torch.randn(dimension, generator=generator, dtype=torch.float64)
        step_scale = 0.001 + 0.049 * float(torch.rand((), generator=generator).item())
        delta = -step_scale * gradient / torch.linalg.vector_norm(gradient)
        E = 20.0 * float(torch.rand((), generator=generator).item())
        H = 100.0 * float(torch.rand((), generator=generator).item())
        eta = float(torch.rand((), generator=generator).item())
        plan = make_counterfactual_normalized_plan(
            counterfactual_delta=delta,
            ordinary_gradient=gradient,
            protected_basis=torch.zeros((dimension, 0), dtype=torch.float64),
            allowed_mask=torch.ones(dimension, dtype=torch.float64),
            E=E,
            H=H,
            charge_fraction=eta,
            cap=0.05,
            active_protection=True,
            loss_smoothness=20.0,
        )
        assert plan.selected_path_fraction + 2e-12 >= eta
        assert plan.selected_path_fraction <= 1.0 + 2e-12
        selected_charge = retention_charge(E, H, plan.proposal.step_length)
        assert selected_charge <= plan.retention_budget + 2e-12
        assert plan.projected_counterfactual_alignment_error <= 1e-12
        assert plan.projection_idempotence_error <= 1e-12


def test_full_charge_fraction_commits_full_projected_comparator() -> None:
    gradient = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    delta = -0.01 * gradient
    plan = make_counterfactual_normalized_plan(
        counterfactual_delta=delta,
        ordinary_gradient=gradient,
        protected_basis=torch.zeros((3, 0), dtype=torch.float64),
        allowed_mask=torch.ones(3, dtype=torch.float64),
        E=5.0,
        H=20.0,
        charge_fraction=1.0,
        cap=0.05,
        active_protection=True,
        loss_smoothness=20.0,
    )
    assert plan.selected_path_fraction == pytest.approx(1.0, abs=1e-12)
    assert torch.allclose(plan.proposal.delta, delta, atol=1e-12, rtol=0.0)
    assert plan.retention_budget == pytest.approx(plan.reference_charge)


def test_projection_and_mask_determine_persistent_reference() -> None:
    gradient = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    delta = -0.01 * gradient
    basis = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    plan = make_counterfactual_normalized_plan(
        counterfactual_delta=delta,
        ordinary_gradient=gradient,
        protected_basis=basis,
        allowed_mask=mask,
        E=1.0,
        H=1.0,
        charge_fraction=1.0,
        cap=0.05,
        active_protection=True,
        loss_smoothness=20.0,
    )
    expected = torch.tensor([0.0, -0.02, 0.0, -0.04], dtype=torch.float64)
    assert torch.allclose(plan.reference_delta, expected, atol=1e-12, rtol=0.0)
    assert torch.allclose(plan.proposal.delta, expected, atol=1e-12, rtol=0.0)


def test_malformed_non_gradient_comparator_is_auditable() -> None:
    gradient = torch.tensor([1.0, 0.0], dtype=torch.float64)
    delta = torch.tensor([0.0, -0.01], dtype=torch.float64)
    plan = make_counterfactual_normalized_plan(
        counterfactual_delta=delta,
        ordinary_gradient=gradient,
        protected_basis=torch.zeros((2, 0), dtype=torch.float64),
        allowed_mask=torch.ones(2, dtype=torch.float64),
        E=1.0,
        H=1.0,
        charge_fraction=0.5,
        cap=0.05,
        active_protection=True,
        loss_smoothness=20.0,
    )
    assert plan.projected_counterfactual_alignment_error >= 0.01
    assert not plan.scalar_comparator_certified


def test_persistent_descent_bound_matches_quadratic_model() -> None:
    q = 3.0
    s = 0.02
    lam = 0.6
    L = 40.0
    expected = lam * s * q - 0.5 * L * (lam * s) ** 2
    assert persistent_descent_lower_bound(
        projected_gradient_norm=q,
        reference_step_length=s,
        selected_path_fraction=lam,
        smoothness=L,
    ) == pytest.approx(expected)
    assert expected > 0.0


def test_normalized_budget_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        normalized_retention_budget(1.0, 1.0, 1.0, -0.1)
    with pytest.raises(ValueError):
        normalized_retention_budget(1.0, 1.0, 1.0, 1.1)


def test_exact_analytic_ratio_bound_is_stronger_than_universal_bound() -> None:
    gradient = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    delta = -0.01 * gradient
    plan = make_counterfactual_normalized_plan(
        counterfactual_delta=delta, ordinary_gradient=gradient,
        protected_basis=torch.zeros((3, 0), dtype=torch.float64),
        allowed_mask=torch.ones(3, dtype=torch.float64), E=3.0, H=8.0,
        charge_fraction=0.4, cap=0.05, active_protection=True,
        loss_smoothness=20.0,
    )
    coarse = plan.selected_path_fraction * plan.compatibility_fraction / 3.0
    assert plan.analytic_persistent_progress_ratio_lower_bound + 1e-15 >= coarse
    assert plan.analytic_persistent_progress_ratio_lower_bound > 0.0


def test_compatibility_uses_true_ordinary_gradient_energy() -> None:
    gradient = torch.tensor([3.0, 4.0], dtype=torch.float64)
    delta = -0.02 * gradient
    # Protect the first coordinate: only the second component is compatible.
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    plan = make_counterfactual_normalized_plan(
        counterfactual_delta=delta,
        ordinary_gradient=gradient,
        protected_basis=basis,
        allowed_mask=torch.ones(2, dtype=torch.float64),
        E=2.0,
        H=3.0,
        charge_fraction=0.5,
        cap=0.2,
        active_protection=True,
        loss_smoothness=25.0,
    )
    assert plan.compatibility_fraction == pytest.approx(16.0 / 25.0, abs=1e-12)
    assert plan.ordinary_step_size == pytest.approx(0.02, abs=1e-12)
    assert plan.step_size_smoothness_product == pytest.approx(0.5, abs=1e-12)
    assert plan.scalar_comparator_certified
    expected = (
        plan.selected_path_fraction
        * (16.0 / 25.0)
        * (1.0 - 0.25 * plan.selected_path_fraction)
        / 1.25
    )
    assert plan.analytic_persistent_progress_ratio_lower_bound == pytest.approx(
        expected, abs=1e-12
    )


def test_zero_compatible_reference_is_not_certified() -> None:
    gradient = torch.tensor([1.0, 0.0], dtype=torch.float64)
    delta = -0.01 * gradient
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    plan = make_counterfactual_normalized_plan(
        counterfactual_delta=delta,
        ordinary_gradient=gradient,
        protected_basis=basis,
        allowed_mask=torch.ones(2, dtype=torch.float64),
        E=1.0,
        H=1.0,
        charge_fraction=1.0,
        cap=0.05,
        active_protection=True,
        loss_smoothness=20.0,
    )
    assert plan.reference_step_length == pytest.approx(0.0)
    assert plan.compatibility_fraction == pytest.approx(0.0)
    assert not plan.scalar_comparator_certified
    assert plan.analytic_persistent_progress_ratio_lower_bound == pytest.approx(0.0)
