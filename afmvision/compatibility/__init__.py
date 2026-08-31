"""Causal functional-compatibility sweep for fully trainable networks."""

from .models import (
    CompatibilityModel,
    FullyTrainableCNN,
    TinyVisionTransformer,
    TinyCharTransformer,
    build_compatibility_model,
)

__all__ = [
    "CompatibilityModel",
    "FullyTrainableCNN",
    "TinyVisionTransformer",
    "TinyCharTransformer",
    "build_compatibility_model",
]
