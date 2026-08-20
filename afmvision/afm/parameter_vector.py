from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class SliceInfo:
    name: str
    start: int
    stop: int
    shape: torch.Size


class ParameterVector:
    def __init__(self, named_parameters: Iterable[tuple[str, nn.Parameter]]):
        items = [(name, p) for name, p in named_parameters if p.requires_grad]
        if not items:
            raise ValueError("No trainable parameters found")
        self.names = [name for name, _ in items]
        self.params = [p for _, p in items]
        self.slices: list[SliceInfo] = []
        cursor = 0
        for name, p in items:
            count = p.numel()
            self.slices.append(SliceInfo(name, cursor, cursor + count, p.shape))
            cursor += count
        self.dimension = cursor

    def flatten(self, detach: bool = True) -> torch.Tensor:
        parts = [p.reshape(-1) for p in self.params]
        vector = torch.cat(parts)
        return vector.detach().clone() if detach else vector

    def flatten_grads(self, fill_none: float = 0.0) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for p in self.params:
            if p.grad is None:
                parts.append(torch.full_like(p, fill_none).reshape(-1))
            else:
                parts.append(p.grad.reshape(-1))
        return torch.cat(parts)

    @torch.no_grad()
    def assign(self, vector: torch.Tensor) -> None:
        if vector.numel() != self.dimension:
            raise ValueError(f"Expected vector of length {self.dimension}, got {vector.numel()}")
        for p, info in zip(self.params, self.slices):
            p.copy_(vector[info.start : info.stop].view(info.shape).to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def add_(self, delta: torch.Tensor) -> None:
        if delta.numel() != self.dimension:
            raise ValueError(f"Expected delta of length {self.dimension}, got {delta.numel()}")
        for p, info in zip(self.params, self.slices):
            p.add_(delta[info.start : info.stop].view(info.shape).to(device=p.device, dtype=p.dtype))

    def mask_for_parameter_prefixes(self, prefixes: tuple[str, ...], value: float = 1.0) -> torch.Tensor:
        ref = self.params[0]
        mask = torch.zeros(self.dimension, device=ref.device, dtype=ref.dtype)
        for info in self.slices:
            if info.name.startswith(prefixes):
                mask[info.start : info.stop] = value
        return mask

    def slice_for_exact_name(self, name: str) -> SliceInfo:
        for info in self.slices:
            if info.name == name:
                return info
        raise KeyError(name)

    def mask_for_adapter_slot(self, slot: int) -> torch.Tensor:
        ref = self.params[0]
        mask = torch.zeros(self.dimension, device=ref.device, dtype=ref.dtype)
        prefix = f"adapter_pool.adapters.{int(slot)}."
        for info in self.slices:
            if info.name.startswith(prefix):
                mask[info.start : info.stop] = 1.0
        try:
            gates = self.slice_for_exact_name("adapter_pool.gates")
            if 0 <= int(slot) < gates.stop - gates.start:
                mask[gates.start + int(slot)] = 1.0
        except KeyError:
            pass
        return mask

    def gradient_mask_for_adapter_activity(self, active_slots: set[int], trial_slot: int | None = None) -> torch.Tensor:
        ref = self.params[0]
        mask = torch.ones(self.dimension, device=ref.device, dtype=ref.dtype)
        allowed_slots = set(active_slots)
        if trial_slot is not None:
            allowed_slots.add(trial_slot)
        for info in self.slices:
            if not info.name.startswith("adapter_pool.adapters."):
                continue
            parts = info.name.split(".")
            slot = int(parts[2])
            if slot not in allowed_slots:
                mask[info.start : info.stop] = 0.0
        try:
            gates = self.slice_for_exact_name("adapter_pool.gates")
        except KeyError:
            return mask
        gate_mask = torch.zeros(gates.stop - gates.start, device=ref.device, dtype=ref.dtype)
        for slot in allowed_slots:
            if 0 <= slot < gate_mask.numel():
                gate_mask[slot] = 1.0
        mask[gates.start : gates.stop] = gate_mask
        return mask

from contextlib import contextmanager


@contextmanager
def temporary_parameters(vectoriser: ParameterVector, vector: torch.Tensor):
    original = vectoriser.flatten(detach=True)
    vectoriser.assign(vector)
    try:
        yield
    finally:
        vectoriser.assign(original)
