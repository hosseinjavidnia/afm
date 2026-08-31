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
from afmvision.compatibility.natural_scale_bridge_repair import NaturalScaleBridgeRepairRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        default=os.environ.get(
            "AFM_NATURAL_BRIDGE_REPAIR_MATRIX",
            "runs_compatibility_natural_scale_bridge_v1_delta0_repair/job_matrix.json",
        ),
    )
    ap.add_argument("--index", type=int, default=None)
    args = ap.parse_args()
    idx = args.index
    if idx is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise SystemExit("--index or SLURM_ARRAY_TASK_ID is required")
        idx = int(raw)

    rows = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    row = next((r for r in rows if int(r["index"]) == int(idx)), None)
    if row is None:
        raise SystemExit(f"matrix index {idx} not found")
    if not torch.cuda.is_available():
        raise SystemExit("natural-scale bridge repair requires CUDA")

    repair_targets = {
        int(state): [float(x) for x in fracs]
        for state, fracs in row["repair_targets"].items()
    }
    runner = NaturalScaleBridgeRepairRunner(
        source_run_dir=row["source_run_dir"],
        original_bridge_run_dir=row["original_bridge_run_dir"],
        run_dir=row["run_dir"],
        system=row["system"],
        device=torch.device("cuda"),
        repair_targets=repair_targets,
        natural_median_update_norm=float(row["natural_median_update_norm"]),
        requested_kappas=[float(x) for x in row["requested_kappas"]],
        candidate_pool=int(row["candidate_pool"]),
        kappa_tolerance=float(row["kappa_tolerance"]),
        update_norm_rtol=float(row["update_norm_rtol"]),
        delta0_cv_tolerance=float(row["delta0_cv_tolerance"]),
        delta0_range_tolerance=float(row["delta0_range_tolerance"]),
        states=int(row["states"]),
        probe_interval=int(row["probe_interval"]),
        methods=[str(x) for x in row["methods"]],
        retention_betas=[float(x) for x in row["retention_betas"]],
    )
    print(json.dumps(runner.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
