from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from afmvision.compatibility.independent_directions import IndependentDirectionRunner

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.environ.get("AFM_DIRECTION_MATRIX", "runs_compatibility_independent_directions_v1/job_matrix.json"))
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
        raise SystemExit("independent-direction compatibility probe requires CUDA")
    runner = IndependentDirectionRunner(
        source_run_dir=row["source_run_dir"],
        run_dir=row["run_dir"],
        system=row["system"],
        device=torch.device("cuda"),
        requested_kappas=[float(x) for x in row["requested_kappas"]],
        directions_per_kappa=int(row["directions_per_kappa"]),
        candidate_pool=int(row["candidate_pool"]),
        kappa_tolerance=float(row["kappa_tolerance"]),
        update_norm_rtol=float(row["update_norm_rtol"]),
        max_abs_cosine=float(row["max_abs_cosine"]),
    )
    print(json.dumps(runner.run(), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
