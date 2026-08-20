from __future__ import annotations

import torch
from torch import nn

from .parameter_vector import ParameterVector, temporary_parameters


def exact_logit_jacobian_rows(
    model: nn.Module,
    vectoriser: ParameterVector,
    images: torch.Tensor,
    anchor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all logit Jacobian rows for every image at the requested anchor.

    Shape is [batch * classes, d]. This is the complete empirical behaviour map
    for the commit block, not a probe subset.
    """

    def compute() -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        logits = model(images)
        rows: list[torch.Tensor] = []
        for sample in range(logits.shape[0]):
            for cls in range(logits.shape[1]):
                grads = torch.autograd.grad(
                    logits[sample, cls],
                    vectoriser.params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                parts = []
                for parameter, grad in zip(vectoriser.params, grads):
                    parts.append((torch.zeros_like(parameter) if grad is None else grad).reshape(-1))
                rows.append(torch.cat(parts).detach())
        return torch.stack(rows), logits.detach()

    if anchor is None:
        return compute()
    with temporary_parameters(vectoriser, anchor.to(device=vectoriser.params[0].device)):
        return compute()
