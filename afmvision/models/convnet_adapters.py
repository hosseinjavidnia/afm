from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from afmvision.afm.functional_shield import FunctionalShield


class ConvBackbone(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        channels = [3, 32, 64, 96, 128]
        blocks: list[nn.Module] = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(channels[-1], feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def randomise_for_renewal(self, generator: torch.Generator | None = None) -> None:
        with torch.no_grad():
            self.down.weight.normal_(0.0, 0.05, generator=generator)
            self.down.bias.zero_()
            self.up.weight.normal_(0.0, 0.05, generator=generator)
            self.up.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


@dataclass
class RenewalState:
    active: tuple[int, ...]
    dormant: tuple[int, ...]


class ZeroGatedAdapterPool(nn.Module):
    """Fixed adapter pool with exactly zero functional effect for dormant slots."""

    def __init__(self, dim: int, bottleneck: int, slots: int, initially_active: int = 1):
        super().__init__()
        if not 0 <= initially_active <= slots:
            raise ValueError("initially_active must be between 0 and slots")
        self.adapters = nn.ModuleList([BottleneckAdapter(dim, bottleneck) for _ in range(slots)])
        gates = torch.zeros(slots)
        if initially_active:
            gates[:initially_active] = 0.1
        self.gates = nn.Parameter(gates)
        active = torch.zeros(slots, dtype=torch.bool)
        active[:initially_active] = True
        self.register_buffer("active_mask", active)

    @property
    def slots(self) -> int:
        return len(self.adapters)

    def state(self) -> RenewalState:
        active = tuple(torch.where(self.active_mask)[0].tolist())
        dormant = tuple(torch.where(~self.active_mask)[0].tolist())
        return RenewalState(active=active, dormant=dormant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, adapter in enumerate(self.adapters):
            out = out + self.gates[idx] * adapter(x)
        return out

    @torch.no_grad()
    def reset_dormant(self, slot: int, generator: torch.Generator | None = None) -> None:
        if bool(self.active_mask[slot]):
            raise ValueError(f"Adapter slot {slot} is active and cannot be reset")
        self.gates[slot].zero_()
        self.adapters[slot].randomise_for_renewal(generator=generator)

    @torch.no_grad()
    def mark_active(self, slot: int) -> None:
        self.active_mask[slot] = True

    @torch.no_grad()
    def deactivate(self, slot: int) -> None:
        self.gates[slot].zero_()
        self.active_mask[slot] = False

    def dormant_gate_names(self) -> set[str]:
        return {f"adapter_pool.gates[{i}]" for i in self.state().dormant}


class AFMFeatureHead(nn.Module):
    """The feature-to-logit map, including frozen structural shield state."""

    def __init__(
        self,
        adapter_pool: ZeroGatedAdapterPool,
        norm: nn.Module,
        classifier: nn.Module,
        functional_shield: FunctionalShield | None = None,
    ):
        super().__init__()
        self.adapter_pool = adapter_pool
        self.norm = norm
        self.classifier = classifier
        self.functional_shield = functional_shield

    def base_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(self.adapter_pool(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.base_logits(features)
        if self.functional_shield is not None:
            logits = logits + self.functional_shield(features)
        return logits


class AFMConvNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 128,
        adapter_bottleneck: int = 32,
        adapter_slots: int = 6,
        initially_active_adapters: int = 1,
    ):
        super().__init__()
        self.backbone = ConvBackbone(feature_dim=feature_dim)
        self.adapter_pool = ZeroGatedAdapterPool(
            dim=feature_dim,
            bottleneck=adapter_bottleneck,
            slots=adapter_slots,
            initially_active=initially_active_adapters,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.functional_shield = FunctionalShield(
            feature_dim=feature_dim,
            output_dim=num_classes,
            max_nodes=1,
            bandwidth=1.0,
        )

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.adapter_pool(self.encode_backbone(x)))

    def base_logits_from_backbone(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(self.adapter_pool(features)))

    def forward_from_backbone(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_logits_from_backbone(features) + self.functional_shield(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_backbone(self.encode_backbone(x))

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

    def trainable_named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        return ((name, p) for name, p in self.named_parameters() if p.requires_grad)


class ConstantScalarBackbone(nn.Module):
    """Validation-only backbone returning one constant observable feature.

    The transfer-conflict benchmark uses this finite neural model so the
    protected feasible geometry can be certified exactly.  The class is not
    selected by production configurations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)


class AntiSymmetricBinaryClassifier(nn.Module):
    """One-parameter binary classifier with logits ``(theta, -theta)``."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        theta = self.theta.to(dtype=features.dtype)
        logits = torch.stack((theta, -theta), dim=0)
        return logits.unsqueeze(0).expand(features.shape[0], -1)


class ScalarTransferConflictNet(AFMConvNet):
    """Exact one-dimensional model for the transfer-obstruction control.

    After one nonconstant behaviour is protected, its empirical Jacobian spans
    the sole trainable coordinate.  The protected feasible subspace therefore
    has dimension zero.  An opposite-policy candidate cannot be transferred
    without releasing protection; this is an analytic obstruction rather than
    a stochastic label-based proxy.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.backbone = ConstantScalarBackbone()
        self.adapter_pool = ZeroGatedAdapterPool(
            dim=1, bottleneck=1, slots=0, initially_active=0
        )
        self.adapter_pool.gates.requires_grad_(False)
        self.norm = nn.Identity()
        self.classifier = AntiSymmetricBinaryClassifier()
        self.functional_shield = FunctionalShield(
            feature_dim=1,
            output_dim=2,
            max_nodes=1,
            bandwidth=1.0,
        )
