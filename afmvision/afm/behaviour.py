from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from .frequent_directions import FrequentDirections
from .parameter_vector import ParameterVector, temporary_parameters


@dataclass(frozen=True)
class BehaviourSpec:
    kind: str = "probabilities"
    temperature: float = 1.0
    output_scale: float = 1.0

    @classmethod
    def from_config(cls, config: dict) -> "BehaviourSpec":
        spec = config.get("afm", {}).get("behaviour", {})
        kind = str(spec.get("kind", "probabilities"))
        if kind not in {"logits", "centered_logits", "probabilities"}:
            raise ValueError("afm.behaviour.kind must be logits, centered_logits, or probabilities")
        temperature = float(spec.get("temperature", 1.0))
        output_scale = float(spec.get("output_scale", 1.0))
        if temperature <= 0.0 or output_scale <= 0.0:
            raise ValueError("Behaviour temperature and output_scale must be positive")
        return cls(kind=kind, temperature=temperature, output_scale=output_scale)


def behaviour_from_logits(logits: torch.Tensor, spec: BehaviourSpec) -> torch.Tensor:
    scaled = logits / spec.temperature
    if spec.kind == "probabilities":
        value = torch.softmax(scaled, dim=-1)
    elif spec.kind == "centered_logits":
        value = scaled - scaled.mean(dim=-1, keepdim=True)
    else:
        value = scaled
    return value * spec.output_scale


def current_behaviour(model: nn.Module, images: torch.Tensor, spec: BehaviourSpec) -> torch.Tensor:
    return behaviour_from_logits(model(images), spec)


def iter_behaviour_jacobian_rows(
    model: nn.Module,
    vectoriser: ParameterVector,
    images: torch.Tensor,
    spec: BehaviourSpec,
) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
    """Yield complete empirical behaviour Jacobian rows without materialising them."""

    model.eval()
    logits = model(images)
    outputs = behaviour_from_logits(logits, spec)

    def iterator() -> Iterator[torch.Tensor]:
        for sample in range(outputs.shape[0]):
            for component in range(outputs.shape[1]):
                grads = torch.autograd.grad(
                    outputs[sample, component],
                    vectoriser.params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                parts = [
                    (torch.zeros_like(parameter) if grad is None else grad).reshape(-1)
                    for parameter, grad in zip(vectoriser.params, grads)
                ]
                yield torch.cat(parts).detach()

    return iterator(), outputs.detach()


def stream_behaviour_jacobians(
    model: nn.Module,
    vectoriser: ParameterVector,
    images: torch.Tensor,
    fd: FrequentDirections,
    spec: BehaviourSpec,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    def compute() -> torch.Tensor:
        rows, outputs = iter_behaviour_jacobian_rows(model, vectoriser, images, spec)
        for row in rows:
            fd.append(row)
        return outputs

    if anchor is None:
        return compute()
    with temporary_parameters(vectoriser, anchor.to(device=vectoriser.params[0].device)):
        return compute()
