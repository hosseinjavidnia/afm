from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF


def _deterministic_transform(image: Image.Image, spec: dict[str, Any], seed: int) -> Image.Image:
    rng = random.Random(seed)
    brightness = float(spec.get("brightness", 1.0))
    contrast = float(spec.get("contrast", 1.0))
    saturation = float(spec.get("saturation", 1.0))
    blur_sigma = float(spec.get("blur_sigma", 0.0))
    rotation = float(spec.get("rotation", 0.0))
    noise_std = float(spec.get("noise_std", 0.0))
    if brightness != 1.0:
        image = TF.adjust_brightness(image, brightness)
    if contrast != 1.0:
        image = TF.adjust_contrast(image, contrast)
    if saturation != 1.0:
        image = TF.adjust_saturation(image, saturation)
    if rotation:
        image = TF.rotate(image, rotation)
    if blur_sigma > 0:
        kernel = max(3, int(math.ceil(blur_sigma * 4)) | 1)
        image = TF.gaussian_blur(image, [kernel, kernel], [blur_sigma, blur_sigma])
    tensor = TF.to_tensor(image)
    if noise_std > 0:
        generator = torch.Generator().manual_seed(seed)
        tensor = torch.clamp(tensor + torch.randn(tensor.shape, generator=generator) * noise_std, 0.0, 1.0)
    return TF.to_pil_image(tensor)


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size: int = 128, normalise: bool = True):
        self.manifest = Path(manifest)
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.image_size = int(image_size)
        self.normalise = bool(normalise)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        row = self.rows[index]
        image = Image.open(row["path"]).convert("RGB")
        image = _deterministic_transform(image, row.get("transform", {}), int(row.get("transform_seed", index)))
        if image.size != (self.image_size, self.image_size):
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        tensor = TF.to_tensor(image)
        if self.normalise:
            tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        metadata = dict(row)
        return tensor, int(row["label"]), metadata


class PrecomputedBatchSampler(Sampler[list[int]]):
    """Read exact batch indices prepared with a streaming dataset protocol."""

    def __init__(self, path: str | Path, dataset_size: int):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Batch-index file must contain a non-empty list: {self.path}")
        self.batches: list[list[int]] = []
        flattened: list[int] = []
        for batch_number, raw in enumerate(payload):
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"Batch {batch_number} is empty or invalid in {self.path}")
            batch = [int(index) for index in raw]
            if any(index < 0 or index >= dataset_size for index in batch):
                raise IndexError(f"Batch {batch_number} contains an index outside 0..{dataset_size-1}")
            self.batches.append(batch)
            flattened.extend(batch)
        if flattened != list(range(dataset_size)):
            raise ValueError(
                "Precomputed streaming batches must cover every manifest row exactly once "
                "and in chronological order"
            )

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def collate_manifest(batch: list[tuple[torch.Tensor, int, dict[str, Any]]]):
    images, labels, metadata = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(metadata)


def load_rows(rows: list[dict[str, Any]], image_size: int = 128, normalise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    samples = []
    labels = []
    for index, row in enumerate(rows):
        image = Image.open(row["path"]).convert("RGB")
        image = _deterministic_transform(image, row.get("transform", {}), int(row.get("transform_seed", index)))
        if image.size != (image_size, image_size):
            image = TF.resize(image, [image_size, image_size], antialias=True)
        tensor = TF.to_tensor(image)
        if normalise:
            tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        samples.append(tensor)
        labels.append(int(row["label"]))
    if not samples:
        return torch.empty((0, 3, image_size, image_size)), torch.empty((0,), dtype=torch.long)
    return torch.stack(samples), torch.tensor(labels, dtype=torch.long)
