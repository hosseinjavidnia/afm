from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-suite-root", default="runs_compatibility_causal_v1")
    ap.add_argument("--output-root", default="runs_compatibility_multiscale_v1")
    ap.add_argument("--scale-fractions", nargs="+", type=float, default=[0.05, 0.20, 0.50, 0.90])
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
            "scale_fractions": [float(x) for x in args.scale_fractions],
        })
    (output / "job_matrix.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE: {output / 'job_matrix.json'}")
    print(f"jobs: {len(rows)}")
    print(f"scale fractions: {[float(x) for x in args.scale_fractions]}")

if __name__ == "__main__":
    main()
