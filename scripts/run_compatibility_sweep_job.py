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

from afmvision.compatibility.experiment import CompatibilitySweepRunner
from afmvision.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=os.environ.get("COMPAT_MATRIX", "runs_compatibility_causal_v1/job_matrix.json"))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument(
        "--resume-preprobe",
        action="store_true",
        help="Resume an existing failed run from preprobe_parent.pt without repeating parent training",
    )
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
        raise SystemExit("compatibility sweep requires CUDA")
    cfg = load_config(row["config"])
    resume_preprobe = args.resume_preprobe or os.environ.get("COMPAT_RESUME_PREPROBE", "0") == "1"
    runner = CompatibilitySweepRunner(
        cfg, row["run_dir"], torch.device("cuda"), resume_preprobe=resume_preprobe
    )
    summary = runner.run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
