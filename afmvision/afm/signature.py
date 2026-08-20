from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


@dataclass(frozen=True)
class SignatureCalibration:
    kind: str
    dimension: int
    prefix_items: int
    fixed_threshold: float
    requested_threshold: float
    calibrated_threshold: float
    calibration_quantile: float
    calibration_margin: float
    observed_radius_quantile: float
    observed_radius_max: float
    observed_coverage: float
    ceiling_hit: bool
    calibration_obstruction: bool
    positive_routing_claim_available: bool
    representation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dimension": self.dimension,
            "prefix_items": self.prefix_items,
            "fixed_threshold": self.fixed_threshold,
            "requested_threshold": self.requested_threshold,
            "calibrated_threshold": self.calibrated_threshold,
            "calibration_quantile": self.calibration_quantile,
            "calibration_margin": self.calibration_margin,
            "observed_radius_quantile": self.observed_radius_quantile,
            "observed_radius_max": self.observed_radius_max,
            "observed_coverage": self.observed_coverage,
            "ceiling_hit": self.ceiling_hit,
            "calibration_obstruction": self.calibration_obstruction,
            "positive_routing_claim_available": self.positive_routing_claim_available,
            "representation": self.representation,
        }


class CausalContextSignature:
    """Fixed task-free observable context signature.

    ``block_hybrid_moments`` is the default CORe50 instantiation.  Each fixed
    observable microblock is represented by the mean and dispersion of bounded
    low-level style moments and frozen-backbone features.  Block aggregation
    suppresses shuffled object identity while retaining observable acquisition
    and feature-distribution shifts.  The same label-free signature is assigned
    to each item in the block, exactly as allowed by the manuscript.  A finite
    unprotected prefix fixes centring, scale, and the nearest-centroid threshold;
    no evaluator context, session, episode, target, or intervention is used.
    """

    _KINDS = {
        "style_moments",
        "block_style_moments",
        "backbone",
        "block_backbone_mean",
        "block_hybrid_moments",
    }

    def __init__(self, config: dict[str, Any], fixed_threshold: float, centroid_rate: float):
        self.kind = str(config.get("kind", "block_style_moments"))
        if self.kind not in self._KINDS:
            raise ValueError(
                "router.signature.kind must be one of " + ", ".join(sorted(self._KINDS))
            )
        default_dimension = 54 if self.kind == "block_style_moments" else (27 if self.kind == "style_moments" else 0)
        self.declared_dimension = int(config.get("dimension", default_dimension))
        if self.declared_dimension < 0:
            raise ValueError("router.signature.dimension must be nonnegative")
        self.block_size = int(config.get("block_size", 1))
        if self.block_size <= 0:
            raise ValueError("router.signature.block_size must be positive")
        self.temporal_rate = float(config.get("temporal_rate", 1.0 if self.kind.startswith("block_") else 0.05))
        if not 0.0 < self.temporal_rate <= 1.0:
            raise ValueError("router.signature.temporal_rate must lie in (0,1]")
        self.clip = float(config.get("clip", 8.0))
        if self.clip <= 0.0:
            raise ValueError("router.signature.clip must be positive")
        self.scale_floor = float(config.get("scale_floor", 1e-3))
        if self.scale_floor <= 0.0:
            raise ValueError("router.signature.scale_floor must be positive")
        self.calibrate_from_prefix = bool(config.get("calibrate_from_prefix", self.kind in {"style_moments", "block_style_moments", "block_hybrid_moments"}))
        self.calibration_quantile = float(config.get("calibration_quantile", 0.995))
        if not 0.0 < self.calibration_quantile <= 1.0:
            raise ValueError("router.signature.calibration_quantile must lie in (0,1]")
        self.calibration_margin = float(config.get("calibration_margin", 1.10))
        if self.calibration_margin < 1.0:
            raise ValueError("router.signature.calibration_margin must be at least 1")
        self.calibration_floor = float(config.get("calibration_floor", 0.05))
        self.calibration_ceiling = float(config.get("calibration_ceiling", 2.0))
        if not 0.0 <= self.calibration_floor <= self.calibration_ceiling:
            raise ValueError("signature calibration floor/ceiling are inconsistent")
        self.calibration_rule = str(config.get("calibration_rule", "replace"))
        if self.calibration_rule not in {"replace", "max_fixed"}:
            raise ValueError("signature.calibration_rule must be 'replace' or 'max_fixed'")
        self.fixed_threshold = float(fixed_threshold)
        self.centroid_rate = float(centroid_rate)

        self._prefix: list[torch.Tensor] = []
        self.center: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        self.finalised = False
        self.calibration: SignatureCalibration | None = None
        # Monotone observable-block identity. One fixed microblock contributes
        # one signature observation even when that signature is repeated for
        # several outcome-resolved items.
        self.next_block_id = 0

    @staticmethod
    def _style_moments(images: torch.Tensor) -> torch.Tensor:
        mean = _IMAGENET_MEAN.to(images.device, images.dtype)
        std = _IMAGENET_STD.to(images.device, images.dtype)
        pixels = torch.clamp(images * std + mean, 0.0, 1.0)

        channel_mean = pixels.mean(dim=(2, 3))
        channel_std = pixels.std(dim=(2, 3), unbiased=False)
        horizontal = (pixels[:, :, :, 1:] - pixels[:, :, :, :-1]).abs().mean(dim=(2, 3))
        vertical = (pixels[:, :, 1:, :] - pixels[:, :, :-1, :]).abs().mean(dim=(2, 3))
        spatial = F.adaptive_avg_pool2d(pixels, (2, 2)).flatten(1)

        centered = pixels - channel_mean[:, :, None, None]
        cov_rg = (centered[:, 0] * centered[:, 1]).mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        cov_rb = (centered[:, 0] * centered[:, 2]).mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        cov_gb = (centered[:, 1] * centered[:, 2]).mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        covariance = torch.cat([cov_rg, cov_rb, cov_gb], dim=1)
        return torch.cat([channel_mean, channel_std, horizontal, vertical, spatial, covariance], dim=1)

    def raw(self, images: torch.Tensor, backbone_features: torch.Tensor | None = None) -> torch.Tensor:
        style = self._style_moments(images)
        if self.kind in {"style_moments", "block_style_moments"}:
            return style
        if backbone_features is None:
            raise ValueError("backbone_features are required for backbone or hybrid signatures")
        norms = torch.clamp(torch.linalg.vector_norm(backbone_features, dim=1, keepdim=True), min=1e-12)
        backbone = backbone_features / norms
        if self.kind == "block_hybrid_moments":
            return torch.cat([style, backbone], dim=1)
        return backbone

    def observe_prefix(self, images: torch.Tensor, backbone_features: torch.Tensor | None = None) -> None:
        if self.finalised:
            raise RuntimeError("Cannot modify a frozen context signature")
        raw = self.raw(images, backbone_features).detach().cpu().to(torch.float64)
        self._prefix.append(raw)

    def _standardise(self, raw: torch.Tensor) -> torch.Tensor:
        if self.center is None or self.scale is None:
            raise RuntimeError("Context signature has not been finalised")
        z = (raw.to(torch.float64).cpu() - self.center) / self.scale
        return torch.clamp(z, -self.clip, self.clip)

    @staticmethod
    def _unit(vector: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) <= 1e-12:
            return torch.zeros_like(vector)
        return vector / norm

    def _causal_sequence(self, rows: torch.Tensor, initial: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs: list[torch.Tensor] = []
        state = None if initial is None else initial.clone()
        for row in rows:
            if state is None:
                state = row.clone()
            else:
                state = (1.0 - self.temporal_rate) * state + self.temporal_rate * row
            outputs.append(self._unit(state))
        if not outputs:
            return torch.empty((0, rows.shape[1]), dtype=torch.float64), state
        return torch.stack(outputs), state

    def _block_rows(self, standardised: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        for start in range(0, len(standardised), self.block_size):
            stop = min(start + self.block_size, len(standardised))
            block = standardised[start:stop]
            if self.kind in {"block_style_moments", "block_hybrid_moments"}:
                # Mean and dispersion suppress item identity while retaining
                # acquisition/style and, for the hybrid kind, the distribution
                # of frozen semantic features across the observable block.
                descriptor = torch.cat(
                    [block.mean(dim=0), block.std(dim=0, unbiased=False)], dim=0
                )
            elif self.kind == "block_backbone_mean":
                descriptor = block.mean(dim=0)
            else:
                raise RuntimeError("_block_rows called for a non-block signature")
            rows.append(descriptor)
            lengths.append(stop - start)
        if not rows:
            dim = self.declared_dimension
            return torch.empty((0, dim), dtype=torch.float64), lengths
        return torch.stack(rows), lengths

    def _signature_rows(self, standardised: torch.Tensor, initial: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None, list[int]]:
        if self.kind in {"block_style_moments", "block_backbone_mean", "block_hybrid_moments"}:
            block_rows, lengths = self._block_rows(standardised)
            signatures, state = self._causal_sequence(block_rows, initial=initial)
            return signatures, state, lengths
        signatures, state = self._causal_sequence(standardised, initial=initial)
        return signatures, state, [1] * len(signatures)

    def finalise(self) -> SignatureCalibration:
        if self.finalised:
            assert self.calibration is not None
            return self.calibration
        if self._prefix:
            prefix = torch.cat(self._prefix, dim=0)
        else:
            prefix = torch.empty((0, 0), dtype=torch.float64)

        if prefix.numel() == 0:
            dimension = self.declared_dimension
            self.center = torch.empty((0,), dtype=torch.float64)
            self.scale = torch.empty((0,), dtype=torch.float64)
            threshold = self.fixed_threshold
            requested_threshold = self.fixed_threshold
            radius_q = 0.0
            radius_max = 0.0
            coverage = 1.0
            ceiling_hit = False
            obstruction = False
        else:
            self.center = prefix.mean(dim=0)
            self.scale = torch.clamp(prefix.std(dim=0, unbiased=False), min=self.scale_floor)
            standardised = self._standardise(prefix)
            signatures, final_state, _ = self._signature_rows(standardised, initial=None)
            self.state = final_state
            dimension = int(signatures.shape[1]) if signatures.numel() else self.declared_dimension

            distances: list[float] = []
            if len(signatures):
                centroid = signatures[0].clone()
                for signature in signatures[1:]:
                    distance = float(torch.linalg.vector_norm(signature - centroid).item())
                    distances.append(distance)
                    centroid = (1.0 - self.centroid_rate) * centroid + self.centroid_rate * signature
            if distances:
                distance_tensor = torch.tensor(distances, dtype=torch.float64)
                radius_q = float(torch.quantile(distance_tensor, self.calibration_quantile).item())
                radius_max = float(distance_tensor.max().item())
            else:
                radius_q = 0.0
                radius_max = 0.0
            if self.calibrate_from_prefix:
                calibrated = self.calibration_margin * radius_q
                if self.calibration_rule == "max_fixed":
                    calibrated = max(self.fixed_threshold, calibrated)
                requested_threshold = max(self.calibration_floor, calibrated)
                threshold = min(self.calibration_ceiling, requested_threshold)
                ceiling_hit = requested_threshold > self.calibration_ceiling + 1e-15
                obstruction = ceiling_hit
            else:
                threshold = self.fixed_threshold
                requested_threshold = self.fixed_threshold
                ceiling_hit = False
                obstruction = False
            coverage = 1.0 if not distances else float(
                (torch.tensor(distances, dtype=torch.float64) <= threshold).to(torch.float64).mean().item()
            )

        self.finalised = True
        self._prefix.clear()
        self.calibration = SignatureCalibration(
            kind=self.kind,
            dimension=dimension,
            prefix_items=int(prefix.shape[0]),
            fixed_threshold=self.fixed_threshold,
            requested_threshold=float(requested_threshold),
            calibrated_threshold=float(threshold),
            calibration_quantile=self.calibration_quantile,
            calibration_margin=self.calibration_margin,
            observed_radius_quantile=radius_q,
            observed_radius_max=radius_max,
            observed_coverage=coverage,
            ceiling_hit=bool(ceiling_hit),
            calibration_obstruction=bool(obstruction),
            positive_routing_claim_available=not bool(obstruction),
            representation="final_frozen_backbone_prefix_replay",
        )
        return self.calibration

    def transform_with_block_ids(
        self, images: torch.Tensor, backbone_features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-item signatures and immutable observable-block IDs.

        Statistical route-refinement evidence is block-resolved, whereas
        consolidation and reopening outcomes remain item-resolved.  Returning
        explicit IDs prevents one microblock signature repeated across eight
        labels from being counted as eight independent signature observations.
        """

        if not self.finalised:
            raise RuntimeError("Context signature must be frozen before protected routing")
        raw = self.raw(images, backbone_features)
        if self.center is None or self.center.numel() == 0:
            standardised = raw.detach().cpu().to(torch.float64)
        else:
            standardised = self._standardise(raw.detach().cpu())
        signatures, final_state, lengths = self._signature_rows(standardised, initial=self.state)
        self.state = final_state

        per_item: list[torch.Tensor] = []
        block_ids: list[int] = []
        for signature, length in zip(signatures, lengths):
            block_id = self.next_block_id
            self.next_block_id += 1
            per_item.extend([signature] * int(length))
            block_ids.extend([block_id] * int(length))
        if not per_item:
            dimension = signatures.shape[1] if signatures.ndim == 2 else self.declared_dimension
            return (
                torch.empty((0, dimension), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )
        return (
            torch.stack(per_item).to(dtype=torch.float32),
            torch.tensor(block_ids, dtype=torch.long),
        )

    def transform(self, images: torch.Tensor, backbone_features: torch.Tensor | None = None) -> torch.Tensor:
        signatures, _ = self.transform_with_block_ids(images, backbone_features)
        return signatures

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "declared_dimension": self.declared_dimension,
            "temporal_rate": self.temporal_rate,
            "block_size": self.block_size,
            "clip": self.clip,
            "scale_floor": self.scale_floor,
            "calibrate_from_prefix": self.calibrate_from_prefix,
            "calibration_quantile": self.calibration_quantile,
            "calibration_margin": self.calibration_margin,
            "calibration_floor": self.calibration_floor,
            "calibration_ceiling": self.calibration_ceiling,
            "calibration_rule": self.calibration_rule,
            "fixed_threshold": self.fixed_threshold,
            "centroid_rate": self.centroid_rate,
            "center": self.center,
            "scale": self.scale,
            "state": self.state,
            "finalised": self.finalised,
            "calibration": None if self.calibration is None else self.calibration.as_dict(),
            "next_block_id": int(self.next_block_id),
        }
