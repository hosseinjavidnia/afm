from __future__ import annotations

import math

import torch

from afmvision.afm.eprocess import HalfNormalMixtureEProcess
from afmvision.afm.frequent_directions import FrequentDirections
from afmvision.afm.safe_step import make_safe_step, safe_radius


def test_safe_radius_saturates_budget():
    E, H, b = 0.3, 1.4, 0.08
    radius = safe_radius(E, H, b, cap=100.0)
    assert E * radius + 0.5 * H * radius * radius <= b + 1e-12
    assert math.isclose(E * radius + 0.5 * H * radius * radius, b, rel_tol=1e-9)


def test_projection_step_is_orthogonal_to_protected_basis():
    gradient = torch.tensor([1.0, 2.0, 3.0])
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    step = make_safe_step(gradient, basis, 0.1, E=0.0, H=0.0, budget=1.0, cap=1.0)
    assert torch.allclose(basis.T @ step.delta, torch.zeros(1), atol=1e-7)


def test_frequent_directions_certificate_random_vectors():
    torch.manual_seed(0)
    C = torch.randn(30, 7)
    fd = FrequentDirections(ell=4, dimension=7)
    fd.extend(C)
    snapshot = fd.snapshot()
    for _ in range(50):
        v = torch.randn(7)
        gap = torch.linalg.vector_norm(C @ v).square() - torch.linalg.vector_norm(snapshot.matrix @ v).square()
        assert gap >= -1e-4
        assert gap <= snapshot.delta * torch.linalg.vector_norm(v).square() + 2e-3


def test_eprocess_detects_persistent_positive_advantage():
    process = HalfNormalMixtureEProcess(alpha=0.05)
    for _ in range(500):
        process.update(0.25)
        if process.crossed:
            break
    assert process.crossed
