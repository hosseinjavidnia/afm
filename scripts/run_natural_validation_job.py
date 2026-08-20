from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from afmvision.compatibility.natural_validation import NaturalStateValidationRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default=os.environ.get("NATURAL_MATRIX", "runs_compatibility_natural_v1/job_matrix.json"),
    )
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    index = args.index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise SystemExit("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(raw)
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    row = next((x for x in matrix if int(x["index"]) == int(index)), None)
    if row is None:
        raise SystemExit(f"job index {index} not found")
    if not torch.cuda.is_available():
        raise SystemExit("natural compatibility validation requires CUDA")
    runner = NaturalStateValidationRunner(
        source_run_dir=row["source_run_dir"],
        run_dir=row["run_dir"],
        system=row["system"],
        device=torch.device("cuda"),
        states=int(row["states"]),
        probe_interval=int(row["probe_interval"]),
        probe_steps=int(row["probe_steps"]),
    )
    summary = runner.run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
