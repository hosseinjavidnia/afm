from __future__ import annotations

import torch
from torch import nn

from afmvision.afm.parameter_vector import ParameterVector
from afmvision.compatibility.geometry import (
    build_protected_geometry,
    compatibility_fraction,
    compatible_projection,
    functional_jacobian,
    make_controlled_targets,
    protected_projection,
    teacher_loss,
)
from afmvision.compatibility.models import CompatibilityModel


class ToyLinear(CompatibilityModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.output_dim = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_controlled_intervention_hits_requested_compatibility() -> None:
    torch.manual_seed(3)
    model = ToyLinear().double()
    protected = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    current = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    _, protected_jacobian = functional_jacobian(model, protected)
    current_out, current_jacobian = functional_jacobian(model, current)
    geometry = build_protected_geometry(protected_jacobian, ridge=1e-12)

    targets = make_controlled_targets(
        model=model,
        current_inputs=current,
        current_measurements=current_out.detach(),
        current_jacobian=current_jacobian.detach(),
        geometry=geometry,
        requested_kappas=[0.1, 0.25, 0.5, 0.75, 1.0],
        gradient_norm=1.0,
    )

    vectoriser = ParameterVector(model.named_parameters())
    for target in targets:
        assert abs(target.measured_kappa - target.requested_kappa) < 1e-6
        model.zero_grad(set_to_none=True)
        teacher_loss(model, current, target.teacher_measurements).backward()
        realised = vectoriser.flatten_grads()
        assert torch.allclose(realised, target.gradient, atol=1e-8, rtol=1e-7)


def test_projection_and_kappa_are_consistent() -> None:
    jacobian = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    geometry = build_protected_geometry(jacobian, ridge=1e-12)
    vector = torch.tensor([3.0, 4.0], dtype=torch.float64)
    protected = protected_projection(vector, geometry)
    compatible = compatible_projection(vector, geometry)
    assert torch.allclose(protected, torch.tensor([3.0, 0.0], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(compatible, torch.tensor([0.0, 4.0], dtype=torch.float64), atol=1e-9)
    assert abs(compatibility_fraction(vector, geometry) - 16.0 / 25.0) < 1e-9
