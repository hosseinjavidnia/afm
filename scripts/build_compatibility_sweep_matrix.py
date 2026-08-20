from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

SEEDS = [11, 29, 47, 71, 101]
SYSTEMS = [
    ("cifar10_cnn", "configs/compatibility/causal_sweep_cifar_cnn.yaml"),
    ("cifar10_vit", "configs/compatibility/causal_sweep_cifar_vit.yaml"),
    ("text_transformer", "configs/compatibility/causal_sweep_text_transformer.yaml"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs_compatibility_causal_v1")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = (root / args.output_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing compatibility suite: {out}")
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    matrix = []
    index = 0
    for system, rel in SYSTEMS:
        base = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        for seed in SEEDS:
            cfg = deepcopy(base)
            cfg["seed"] = int(seed)
            name = f"{system}_seed{seed}"
            cfg_path = out / "configs" / f"{name}.yaml"
            run_dir = out / "runs" / name
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            matrix.append(
                {
                    "index": index,
                    "system": system,
                    "seed": seed,
                    "config": str(cfg_path),
                    "run_dir": str(run_dir),
                }
            )
            index += 1
    (out / "job_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE: {out / 'job_matrix.json'}")
    print(f"jobs: {len(matrix)}")
    for row in matrix:
        print(row["index"], row["system"], row["seed"], row["run_dir"])


if __name__ == "__main__":
    main()
