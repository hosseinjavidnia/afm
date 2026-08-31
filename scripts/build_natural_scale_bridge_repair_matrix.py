from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPAIR_REASON = "finite_delta0_match_tolerance_not_met"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-suite-root", default="runs_compatibility_natural_scale_bridge_v1")
    ap.add_argument("--output-root", default="runs_compatibility_natural_scale_bridge_v1_delta0_repair")
    args = ap.parse_args()

    root = Path.cwd().resolve()
    source = (root / args.source_suite_root).resolve()
    out = (root / args.output_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty repair suite: {out}")
    matrix_path = source / "job_matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)

    original_jobs = json.loads(matrix_path.read_text(encoding="utf-8"))
    repair_rows = []
    manifest_targets = []

    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)

    for original in original_jobs:
        original_run = Path(original["run_dir"])
        if not original_run.is_absolute():
            original_run = (root / original_run).resolve()
        feasibility_path = original_run / "bridge_feasibility.jsonl"
        if not feasibility_path.is_file():
            raise FileNotFoundError(feasibility_path)

        targets = []
        by_state: dict[int, list[float]] = defaultdict(list)
        for row in read_jsonl(feasibility_path):
            if str(row.get("reason")) != REPAIR_REASON:
                continue
            if bool(row.get("feasible")):
                raise RuntimeError(f"repair reason appears on feasible row: {original_run}: {row}")
            if str(row.get("system")) != str(original["system"]) or int(row.get("seed")) != int(original["seed"]):
                raise RuntimeError(f"feasibility identity mismatch in {feasibility_path}")
            state_id = int(row["state_id"])
            frac = float(row["natural_norm_fraction"])
            by_state[state_id].append(frac)
            target_record = {
                "system": str(original["system"]),
                "seed": int(original["seed"]),
                "state_id": state_id,
                "natural_norm_fraction": frac,
                "target_update_norm": float(row["target_update_norm"]),
                "original_reason": REPAIR_REASON,
                "original_delta0_cv": float(row.get("delta0_cv", float("nan"))),
                "original_delta0_relative_range": float(row.get("delta0_relative_range", float("nan"))),
                "original_candidate_counts_by_kappa": row.get("candidate_counts_by_kappa", {}),
            }
            targets.append(target_record)
            manifest_targets.append(target_record)

        if not targets:
            continue

        system = str(original["system"])
        seed = int(original["seed"])
        run_name = f"{system}_seed{seed}"
        repair_rows.append(
            {
                "index": len(repair_rows),
                "system": system,
                "seed": seed,
                "source_run_dir": str(Path(original["source_run_dir"]).resolve()),
                "original_bridge_run_dir": str(original_run),
                "run_dir": str((out / "runs" / run_name).resolve()),
                "natural_median_update_norm": float(original["natural_median_update_norm"]),
                "natural_norm_fractions": sorted({float(t["natural_norm_fraction"]) for t in targets}),
                "requested_kappas": [float(x) for x in original["requested_kappas"]],
                "candidate_pool": int(original["candidate_pool"]),
                "kappa_tolerance": float(original["kappa_tolerance"]),
                "update_norm_rtol": float(original["update_norm_rtol"]),
                "delta0_cv_tolerance": float(original["delta0_cv_tolerance"]),
                "delta0_range_tolerance": float(original["delta0_range_tolerance"]),
                "states": int(original["states"]),
                "probe_interval": int(original["probe_interval"]),
                "methods": [str(x) for x in original["methods"]],
                "retention_betas": [float(x) for x in original["retention_betas"]],
                "repair_targets": {
                    str(state): sorted({float(x) for x in fracs}) for state, fracs in sorted(by_state.items())
                },
                "repair_target_count": len(targets),
            }
        )

    if not repair_rows:
        raise RuntimeError(f"No {REPAIR_REASON!r} rows found under {source}")

    # Duplicate target keys would make merge semantics ambiguous.
    keys = [
        (r["system"], r["seed"], r["state_id"], r["natural_norm_fraction"])
        for r in manifest_targets
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate repair state-scale targets detected")

    (out / "job_matrix.json").write_text(
        json.dumps(repair_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "repair_manifest.json").write_text(
        json.dumps(
            {
                "schema": "causal_compatibility_natural_scale_bridge_v1_delta0_admission_repair",
                "source_suite_root": str(source),
                "repair_reason": REPAIR_REASON,
                "gpu_jobs": len(repair_rows),
                "repair_target_conditions": len(manifest_targets),
                "systems": sorted({r["system"] for r in manifest_targets}),
                "selection_rule": "same best available finite-Delta0 alignment as original bridge-v1",
                "changed_rule": "Delta0 CV/range tolerance is diagnostic only; it is no longer an admission criterion",
                "targets": sorted(
                    manifest_targets,
                    key=lambda r: (r["system"], r["seed"], r["state_id"], r["natural_norm_fraction"]),
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"WROTE: {out / 'job_matrix.json'}")
    print(f"WROTE: {out / 'repair_manifest.json'}")
    print(f"GPU jobs: {len(repair_rows)}")
    print(f"state-scale conditions to repair: {len(manifest_targets)}")
    for row in repair_rows:
        print(f"  {row['system']} seed={row['seed']}: {row['repair_target_count']}")


if __name__ == "__main__":
    main()
