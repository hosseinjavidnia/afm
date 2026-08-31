#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from afmvision.data.stream import ManifestDataset, collate_manifest
from afmvision.models.factory import build_model


def ratio(correct: dict[str, int], total: dict[str, int]) -> dict[str, float]:
    return {key: correct[key] / max(total[key], 1) for key in sorted(total)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed run on evaluator-only manifests")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = ManifestDataset(args.manifest, image_size=int(cfg["data"]["image_size"]), normalise=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_manifest,
    )

    correct_by_context: dict[str, int] = defaultdict(int)
    total_by_context: dict[str, int] = defaultdict(int)
    correct_by_episode: dict[str, int] = defaultdict(int)
    total_by_episode: dict[str, int] = defaultdict(int)
    correct_by_regime: dict[str, int] = defaultdict(int)
    total_by_regime: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            matches = predictions.eq(labels).cpu().tolist()
            for match, row in zip(matches, metadata):
                context = str(row.get("context_id", "unknown"))
                episode = str(row.get("episode_name", row.get("episode", "unknown")))
                regime = str(row.get("semantic_regime", episode))
                correct_by_context[context] += int(match)
                total_by_context[context] += 1
                correct_by_episode[episode] += int(match)
                total_by_episode[episode] += 1
                correct_by_regime[regime] += int(match)
                total_by_regime[regime] += 1

    total_correct = sum(correct_by_episode.values())
    total_count = sum(total_by_episode.values())
    report = {
        "manifest": str(args.manifest),
        "overall_accuracy": total_correct / max(total_count, 1),
        "per_context_accuracy": ratio(correct_by_context, total_by_context),
        "per_episode_accuracy": ratio(correct_by_episode, total_by_episode),
        "per_semantic_regime_accuracy": ratio(correct_by_regime, total_by_regime),
        "context_counts": dict(sorted(total_by_context.items())),
        "episode_counts": dict(sorted(total_by_episode.items())),
        "semantic_regime_counts": dict(sorted(total_by_regime.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
