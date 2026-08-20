from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Batch:
    inputs: torch.Tensor
    labels: torch.Tensor
    ids: list[int]


class IndexedCIFAR10(Dataset):
    def __init__(self, root: str | Path, *, train: bool, download: bool) -> None:
        root = Path(root)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.dataset = datasets.CIFAR10(root=str(root), train=train, transform=transform, download=download)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[int(index)]
        return x, int(y), int(index)


class CharNextTokenDataset(Dataset):
    def __init__(self, text_path: str | Path, context_length: int = 64, stride: int = 1) -> None:
        self.path = Path(text_path)
        text = self.path.read_text(encoding="utf-8")
        if len(text) <= context_length + 1:
            raise ValueError("text corpus is too short")
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars
        encoded = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.tokens = encoded
        self.context_length = int(context_length)
        self.stride = max(int(stride), 1)
        self.count = (len(encoded) - self.context_length - 1) // self.stride + 1

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return int(self.count)

    def __getitem__(self, index: int):
        start = int(index) * self.stride
        x = self.tokens[start : start + self.context_length].clone()
        y = int(self.tokens[start + self.context_length].item())
        return x, y, int(index)

    def vocabulary_payload(self) -> dict[str, Any]:
        return {"stoi": self.stoi, "itos": self.itos, "context_length": self.context_length, "stride": self.stride}


def stack_indices(dataset: Dataset, indices: list[int]) -> Batch:
    items = [dataset[int(i)] for i in indices]
    xs, ys, ids = zip(*items)
    return Batch(inputs=torch.stack(xs), labels=torch.tensor(ys, dtype=torch.long), ids=[int(i) for i in ids])


def deterministic_order(length: int, seed: int) -> list[int]:
    order = list(range(int(length)))
    random.Random(int(seed)).shuffle(order)
    return order


def perturb_vision_batch(inputs: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Produce deterministic nonidentical views of protected images.

    These near-protected views create strongly shared functional geometry while
    remaining distinct finite addresses, so low compatibility does not rely on
    an impossible duplicate-address target conflict.
    """

    out = inputs.detach().clone()
    rng = random.Random(int(seed))
    for i in range(len(out)):
        shift_x = 1 if rng.random() < 0.5 else -1
        shift_y = 1 if rng.random() < 0.5 else -1
        out[i] = torch.roll(out[i], shifts=(shift_y, shift_x), dims=(1, 2))
        scale = 0.97 + 0.06 * rng.random()
        out[i] = out[i] * scale
        # A deterministic tiny perturbation guarantees a distinct raw address.
        out[i, 0, 0, 0] = out[i, 0, 0, 0] + (i + 1) * 1e-4
    return out


def perturb_text_batch(inputs: torch.Tensor, vocab_size: int, *, seed: int) -> torch.Tensor:
    out = inputs.detach().clone()
    rng = random.Random(int(seed))
    for i in range(len(out)):
        pos = rng.randrange(out.shape[1])
        old = int(out[i, pos].item())
        replacement = (old + 1 + rng.randrange(max(vocab_size - 1, 1))) % vocab_size
        out[i, pos] = replacement
    return out


def make_causal_current_batch(
    *,
    modality: str,
    protected: Batch,
    novel: Batch,
    near_count: int,
    seed: int,
    vocab_size: int | None = None,
) -> Batch:
    near_count = min(int(near_count), len(protected.inputs), len(novel.inputs))
    if near_count <= 0:
        return novel
    protected_slice = protected.inputs[:near_count]
    if modality == "vision":
        near = perturb_vision_batch(protected_slice, seed=seed)
    elif modality == "text":
        if vocab_size is None:
            raise ValueError("vocab_size is required for text perturbation")
        near = perturb_text_batch(protected_slice, int(vocab_size), seed=seed)
    else:
        raise ValueError(f"unknown modality: {modality}")
    near_labels = protected.labels[:near_count]
    novel_count = max(len(novel.inputs) - near_count, 0)
    novel_inputs = novel.inputs[:novel_count]
    novel_labels = novel.labels[:novel_count]
    inputs = torch.cat((near, novel_inputs), dim=0)
    labels = torch.cat((near_labels, novel_labels), dim=0)
    ids = [-(int(x) + 1) for x in protected.ids[:near_count]] + novel.ids[:novel_count]
    return Batch(inputs=inputs, labels=labels, ids=ids)


class Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = random.Random(int(seed))
        self.ids: list[int] = []
        self.seen = 0

    def add_many(self, ids: list[int]) -> None:
        for item in ids:
            self.seen += 1
            if len(self.ids) < self.capacity:
                self.ids.append(int(item))
            elif self.capacity > 0:
                j = self.rng.randrange(self.seen)
                if j < self.capacity:
                    self.ids[j] = int(item)

    def sample(self, count: int) -> list[int]:
        n = min(int(count), len(self.ids))
        if n <= 0:
            return []
        return [self.ids[i] for i in self.rng.sample(range(len(self.ids)), n)]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_dataset_metadata(dataset: CharNextTokenDataset, output: str | Path) -> None:
    payload = dataset.vocabulary_payload()
    payload.update({"source": str(dataset.path.resolve()), "sha256": file_sha256(dataset.path), "samples": len(dataset)})
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
