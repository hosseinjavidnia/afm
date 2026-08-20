from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class CompatibilityModel(nn.Module):
    """Interface used by the causal compatibility experiment.

    Every parameter in these models remains trainable throughout the experiment.
    ``functional_logits`` returns the finite function whose Jacobian defines
    compatibility.  Vision models use all class logits.  The language model can
    use a deterministic fixed projection of next-token logits to keep the
    protected Jacobian bounded while retention is still audited on full logits.
    """

    output_dim: int

    def functional_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


class FullyTrainableCNN(CompatibilityModel):
    """Compact end-to-end CNN with no frozen representation prefix."""

    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        w = int(width)
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1, groups=w, bias=False),
            nn.Conv2d(w, 2 * w, 1, bias=False),
            nn.BatchNorm2d(2 * w),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1, groups=2 * w, bias=False),
            nn.Conv2d(2 * w, 3 * w, 1, bias=False),
            nn.BatchNorm2d(3 * w),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(3 * w, 3 * w, 3, padding=1, groups=3 * w, bias=False),
            nn.Conv2d(3 * w, 4 * w, 1, bias=False),
            nn.BatchNorm2d(4 * w),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.norm = nn.LayerNorm(4 * w)
        self.classifier = nn.Linear(4 * w, int(num_classes))
        self.output_dim = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.classifier(self.norm(h))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        a, _ = self.attn(q, q, q, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TinyVisionTransformer(CompatibilityModel):
    """Small ViT used as a geometry-generalisation architecture."""

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 48,
        depth: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, int(num_classes))
        self.output_dim = int(num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat((cls, h), dim=1)
        h = h + self.pos[:, : h.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.classifier(self.norm(h[:, 0]))


class TinyCharTransformer(CompatibilityModel):
    """Small fully trainable character-level next-token transformer.

    The model predicts the next character from a fixed-length context.  The
    compatibility geometry may use a fixed random orthonormal projection of
    the full next-token logits.  Full logits remain available for retention and
    AFM finite-endpoint verification.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 64,
        dim: int = 64,
        depth: int = 3,
        heads: int = 4,
        functional_dim: int = 16,
        projection_seed: int = 20260817,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.token = nn.Embedding(self.vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.context_length, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.vocab_size)
        self.output_dim = self.vocab_size
        fd = min(int(functional_dim), self.vocab_size)
        gen = torch.Generator(device="cpu").manual_seed(int(projection_seed))
        random_matrix = torch.randn(self.vocab_size, fd, generator=gen, dtype=torch.float64)
        q, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("functional_projection", q.to(dtype=torch.float32), persistent=True)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("TinyCharTransformer expects [batch, context] token ids")
        if x.shape[1] > self.context_length:
            raise ValueError("input context exceeds configured context_length")
        h = self.token(x) + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h[:, -1]))

    def functional_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x) @ self.functional_projection


@dataclass(frozen=True)
class ModelDescription:
    modality: str
    architecture: str
    trainable_parameters: int
    output_dim: int
    functional_dim: int


def describe_model(model: CompatibilityModel, modality: str, architecture: str) -> ModelDescription:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if isinstance(model, TinyCharTransformer):
        functional_dim = int(model.functional_projection.shape[1])
    else:
        functional_dim = int(model.output_dim)
    return ModelDescription(
        modality=str(modality),
        architecture=str(architecture),
        trainable_parameters=int(trainable),
        output_dim=int(model.output_dim),
        functional_dim=functional_dim,
    )


def build_compatibility_model(config: dict, *, vocab_size: int | None = None) -> CompatibilityModel:
    model = dict(config["compatibility_sweep"]["model"])
    architecture = str(model["architecture"])
    if architecture == "cnn":
        return FullyTrainableCNN(
            num_classes=int(model.get("num_classes", 10)),
            width=int(model.get("width", 32)),
        )
    if architecture == "vit":
        return TinyVisionTransformer(
            num_classes=int(model.get("num_classes", 10)),
            image_size=int(model.get("image_size", 32)),
            patch_size=int(model.get("patch_size", 4)),
            dim=int(model.get("dim", 48)),
            depth=int(model.get("depth", 3)),
            heads=int(model.get("heads", 4)),
        )
    if architecture == "char_transformer":
        if vocab_size is None:
            raise ValueError("vocab_size is required for char_transformer")
        return TinyCharTransformer(
            vocab_size=int(vocab_size),
            context_length=int(model.get("context_length", 64)),
            dim=int(model.get("dim", 64)),
            depth=int(model.get("depth", 3)),
            heads=int(model.get("heads", 4)),
            functional_dim=int(model.get("functional_dim", 16)),
            projection_seed=int(model.get("projection_seed", 20260817)),
        )
    raise ValueError(f"Unknown compatibility architecture: {architecture}")
