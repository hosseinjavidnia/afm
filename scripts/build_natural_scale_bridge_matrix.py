from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

SYSTEMS = ["cifar10_cnn", "cifar10_vit"]
TEN_SEEDS = [11, 29, 47, 71, 101, 131, 149, 167, 191, 223]


def natural_medians(path: Path) -> dict[str, float]:
    vals = {s: [] for s in SYSTEMS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            system = str(row.get("system", ""))
            if system not in vals or str(row.get("method", "")) != "unrestricted":
                continue
            raw = row.get("update_norm")
            if raw in (None, "", "None", "nan"):
                continue
            vals[system].append(float(raw))
    out = {}
    for system, xs in vals.items():
        if not xs:
            raise RuntimeError(f"no natural unrestricted update norms found for {system} in {path}")
        out[system] = float(statistics.median(xs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-generality-root", default="runs_compatibility_generality_v1")
    ap.add_argument("--natural-rows", default="runs_compatibility_natural_v1/analysis/natural_state_validation_rows.csv")
    ap.add_argument("--output-root", default="runs_compatibility_natural_scale_bridge_v1")
    ap.add_argument("--norm-fractions", nargs="+", type=float, default=[0.01, 0.10, 0.50, 1.00])
    ap.add_argument("--requested-kappas", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75])
    ap.add_argument("--candidate-pool", type=int, default=64)
    ap.add_argument("--kappa-tolerance", type=float, default=0.01)
    ap.add_argument("--update-norm-rtol", type=float, default=0.005)
    ap.add_argument("--delta0-cv-tolerance", type=float, default=0.02)
    ap.add_argument("--delta0-range-tolerance", type=float, default=0.04)
    ap.add_argument("--states", type=int, default=50)
    ap.add_argument("--probe-interval", type=int, default=10)
    args = ap.parse_args()

    root = Path.cwd().resolve()
    source = (root / args.source_generality_root).resolve()
    natural = (root / args.natural_rows).resolve()
    out = (root / args.output_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty bridge suite: {out}")
    matrix_path = source / "job_matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"missing generality combined matrix: {matrix_path}")
    if not natural.is_file():
        raise FileNotFoundError(f"missing natural validation rows: {natural}")

    medians = natural_medians(natural)
    source_rows = json.loads(matrix_path.read_text(encoding="utf-8"))
    lookup = {}
    for row in source_rows:
        key = (str(row["system"]), int(row["seed"]))
        if key[0] in SYSTEMS and key[1] in TEN_SEEDS:
            lookup[key] = row
    expected = {(s, seed) for s in SYSTEMS for seed in TEN_SEEDS}
    if set(lookup) != expected:
        raise RuntimeError(f"generality matrix does not contain exact bridge 2x10 grid; missing={sorted(expected-set(lookup))}")

    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    rows = []
    for system in SYSTEMS:
        for seed in TEN_SEEDS:
            source_row = lookup[(system, seed)]
            source_run = Path(source_row["run_dir"])
            if not source_run.is_absolute():
                source_run = (root / source_run).resolve()
            if not (source_run / "preprobe_parent.pt").is_file():
                raise FileNotFoundError(source_run / "preprobe_parent.pt")
            if not (source_run / "resolved_config.json").is_file():
                raise FileNotFoundError(source_run / "resolved_config.json")
            name = f"{system}_seed{seed}"
            rows.append(
                {
                    "index": len(rows),
                    "system": system,
                    "seed": seed,
                    "source_run_dir": str(source_run.resolve()),
                    "run_dir": str((out / "runs" / name).resolve()),
                    "natural_median_update_norm": medians[system],
                    "natural_norm_fractions": [float(x) for x in args.norm_fractions],
                    "requested_kappas": [float(x) for x in args.requested_kappas],
                    "candidate_pool": int(args.candidate_pool),
                    "kappa_tolerance": float(args.kappa_tolerance),
                    "update_norm_rtol": float(args.update_norm_rtol),
                    "delta0_cv_tolerance": float(args.delta0_cv_tolerance),
                    "delta0_range_tolerance": float(args.delta0_range_tolerance),
                    "states": int(args.states),
                    "probe_interval": int(args.probe_interval),
                    "methods": ["projection", "unrestricted", "linearized_distillation", "ewc_prox"],
                    "retention_betas": [0.05, 0.10, 0.25, 0.50],
                }
            )
    (out / "job_matrix.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (out / "design.json").write_text(
        json.dumps(
            {
                "schema": "causal_compatibility_natural_scale_bridge_v1",
                "systems": SYSTEMS,
                "seeds": TEN_SEEDS,
                "jobs": len(rows),
                "states_per_job": int(args.states),
                "probe_interval": int(args.probe_interval),
                "state_sampling": "fixed schedule; no feasibility-conditioned state replacement",
                "requested_kappas": [float(x) for x in args.requested_kappas],
                "natural_norm_fractions": [float(x) for x in args.norm_fractions],
                "natural_median_update_norm_by_system": medians,
                "natural_rows_source": str(natural),
                "candidate_pool": int(args.candidate_pool),
                "matching_tolerances": {
                    "kappa_abs": float(args.kappa_tolerance),
                    "update_norm_relative": float(args.update_norm_rtol),
                    "delta0_cv": float(args.delta0_cv_tolerance),
                    "delta0_relative_range": float(args.delta0_range_tolerance),
                },
                "methods": ["projection", "unrestricted", "linearized_distillation", "ewc_prox"],
                "native_afm": True,
                "retention_betas": [0.05, 0.10, 0.25, 0.50],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"WROTE: {out / 'job_matrix.json'}")
    print(f"GPU jobs: {len(rows)}")
    for system in SYSTEMS:
        print(f"{system} median natural unrestricted update norm: {medians[system]:.12g}")
        print("  targets:", ", ".join(f"{f:g}x={f*medians[system]:.12g}" for f in args.norm_fractions))


if __name__ == "__main__":
    main()
