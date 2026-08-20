from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-suite-root", default="runs_compatibility_causal_v1")
    ap.add_argument("--output-root", default="runs_compatibility_independent_directions_v1")
    ap.add_argument("--requested-kappas", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75])
    ap.add_argument("--directions", type=int, default=4)
    ap.add_argument("--candidate-pool", type=int, default=24)
    ap.add_argument("--kappa-tolerance", type=float, default=0.01)
    ap.add_argument("--update-norm-rtol", type=float, default=0.05)
    ap.add_argument("--max-abs-cosine", type=float, default=0.95)
    args = ap.parse_args()
    source = Path(args.source_suite_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty suite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir()
    matrix = json.loads((source / "job_matrix.json").read_text(encoding="utf-8"))
    rows = []
    for i, row in enumerate(matrix):
        source_run = Path(row["run_dir"]).resolve()
        if not (source_run / "preprobe_parent.pt").is_file():
            raise FileNotFoundError(source_run / "preprobe_parent.pt")
        name = f"{row['system']}_seed{row['seed']}"
        rows.append({
            "index": i,
            "system": row["system"],
            "seed": int(row["seed"]),
            "source_run_dir": str(source_run),
            "run_dir": str((output / "runs" / name).resolve()),
            "requested_kappas": [float(x) for x in args.requested_kappas],
            "directions_per_kappa": int(args.directions),
            "candidate_pool": int(args.candidate_pool),
            "kappa_tolerance": float(args.kappa_tolerance),
            "update_norm_rtol": float(args.update_norm_rtol),
            "max_abs_cosine": float(args.max_abs_cosine),
        })
    (output / "job_matrix.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE: {output / 'job_matrix.json'}")
    print(f"jobs: {len(rows)}")
    print(f"directions per kappa: {args.directions}")

if __name__ == "__main__":
    main()
