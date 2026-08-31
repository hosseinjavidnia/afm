#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny task-free image stream for installation checks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    root = args.output.resolve()
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    learner_rows = []
    evaluator_rows = []
    sidecar_rows = []
    for index in range(args.count):
        label = index % 2
        array = np.zeros((64, 64, 3), dtype=np.uint8)
        array[:, :, label] = 180
        noise = rng.integers(0, 40, size=array.shape, dtype=np.uint8)
        array = np.clip(array.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        path = image_dir / f"{index:03d}.png"
        Image.fromarray(array).save(path)
        sample_id = f"smoke-{index}"
        learner_rows.append(
            {
                "sample_id": sample_id,
                "path": str(path),
                "label": label,
                "transform": {},
                "transform_seed": index,
            }
        )
        metadata = {
            "sample_id": sample_id,
            "context_id": 1 if index < args.count // 2 else 2,
            "session": 1 if index < args.count // 2 else 2,
            "episode": 0 if index < args.count // 2 else 1,
            "episode_name": "smoke_a" if index < args.count // 2 else "smoke_b",
            "semantic_regime": "smoke_original",
            "intervention": "none",
            "original_label": label,
        }
        sidecar_rows.append(metadata)
        evaluator_rows.append({**learner_rows[-1], **metadata})
    for name, rows in [
        ("train.jsonl", learner_rows),
        ("eval.jsonl", evaluator_rows[:8]),
        ("sidecar.jsonl", sidecar_rows),
    ]:
        with (root / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(root)


if __name__ == "__main__":
    main()
