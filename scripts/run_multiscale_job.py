from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from afmvision.compatibility.multiscale import MultiScaleCompatibilityRunner

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.environ.get("AFM_MULTISCALE_MATRIX", "runs_compatibility_multiscale_v1/job_matrix.json"))
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
        raise SystemExit("multi-scale causal compatibility probe requires CUDA")
    runner = MultiScaleCompatibilityRunner(
        source_run_dir=row["source_run_dir"],
        run_dir=row["run_dir"],
        system=row["system"],
        device=torch.device("cuda"),
        scale_fractions=[float(x) for x in row["scale_fractions"]],
    )
    print(json.dumps(runner.run(), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
