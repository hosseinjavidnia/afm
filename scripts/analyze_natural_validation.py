from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_SYSTEMS = ["cifar10_cnn", "cifar10_vit", "text_transformer"]
EXPECTED_SEEDS = [11, 29, 47, 71, 101]
EXPECTED_METHODS = [
    "unrestricted",
    "replay",
    "projection",
    "linearized_distillation",
    "ewc_prox",
    "derpp",
    "afm",
]
CORE_COLUMNS = [
    "system",
    "seed",
    "state_id",
    "method",
    "natural_kappa",
    "gradient_norm",
    "delta0",
    "delta_persistent",
    "rho_persistent",
    "retention_drift",
    "retention_pass",
]
AUDIT_COLUMNS = [
    "dataset",
    "modality",
    "local_step",
    "parent_step",
    "comparator_valid",
    "comparator_alpha",
    "comparator_backtracking_steps",
    "proposal_kappa",
    "update_norm",
    "accepted",
    "obstruction",
    "method_backtracking_steps",
    "afm_lambda_hat",
    "retention_tolerance",
    "current_count",
    "protected_count",
    "current_protected_id_overlap",
    "source_run_dir",
]


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", default="runs_compatibility_natural_v1")
    args = parser.parse_args()
    suite = Path(args.suite_root).resolve()
    matrix = json.loads((suite / "job_matrix.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    failures: list[str] = []
    summaries = []

    for job in matrix:
        run_dir = Path(job["run_dir"])
        summary_path = run_dir / "summary.json"
        points_path = run_dir / "natural_state_points.jsonl"
        if not summary_path.is_file() or not points_path.is_file():
            failures.append(f"missing outputs: {run_dir}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        if summary.get("status") != "complete":
            failures.append(f"incomplete summary: {run_dir}")
        for row in _read_jsonl(points_path):
            row["source_run_dir"] = summary.get("source_run_dir")
            rows.append(row)

    expected_rows = 3 * 5 * 50 * 7
    expected_states = 3 * 5 * 50
    group = defaultdict(list)
    for row in rows:
        group[(row.get("system"), int(row.get("seed")), int(row.get("state_id")))].append(row)
    state_method_failures = []
    for key, gr in group.items():
        methods = sorted(str(x.get("method")) for x in gr)
        if methods != sorted(EXPECTED_METHODS):
            state_method_failures.append({"state": key, "methods": methods})

    system_counts = Counter(str(r.get("system")) for r in rows)
    comparator_invalid = sum(not bool(r.get("comparator_valid")) for r in rows)
    finite_core_rows = sum(
        all(r.get(k) is not None for k in ("natural_kappa", "gradient_norm", "delta0", "delta_persistent", "rho_persistent", "retention_drift"))
        for r in rows
    )
    validation = {
        "pass": (
            not failures
            and len(matrix) == 15
            and len(rows) == expected_rows
            and len(group) == expected_states
            and not state_method_failures
            and comparator_invalid == 0
            and finite_core_rows == expected_rows
        ),
        "jobs_expected": 15,
        "jobs_observed": len(summaries),
        "states_expected": expected_states,
        "states_observed": len(group),
        "rows_expected": expected_rows,
        "rows_observed": len(rows),
        "comparator_invalid_rows": comparator_invalid,
        "rows_with_complete_requested_numeric_core": finite_core_rows,
        "system_row_counts": dict(system_counts),
        "failures": failures,
        "state_method_failures": state_method_failures,
    }

    out = suite / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    headers = CORE_COLUMNS + AUDIT_COLUMNS
    _write_csv(out / "natural_state_validation_rows.csv", headers, rows)
    (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    # Minimal descriptive table; independent inferential analysis can use the row-level CSV.
    desc_groups = defaultdict(list)
    for r in rows:
        desc_groups[(r["system"], r["method"])].append(r)
    desc = []
    for (system, method), gr in sorted(desc_groups.items()):
        valid = [r for r in gr if r.get("rho_persistent") is not None]
        desc.append(
            {
                "system": system,
                "method": method,
                "n": len(gr),
                "mean_natural_kappa": sum(float(r["natural_kappa"]) for r in valid) / len(valid),
                "mean_gradient_norm": sum(float(r["gradient_norm"]) for r in valid) / len(valid),
                "mean_delta0": sum(float(r["delta0"]) for r in valid) / len(valid),
                "mean_rho_persistent": sum(float(r["rho_persistent"]) for r in valid) / len(valid),
                "retention_pass_rate": sum(bool(r["retention_pass"]) for r in valid) / len(valid),
            }
        )
    _write_csv(
        out / "descriptive_summary.csv",
        ["system", "method", "n", "mean_natural_kappa", "mean_gradient_norm", "mean_delta0", "mean_rho_persistent", "retention_pass_rate"],
        desc,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    print(f"WROTE: {out / 'natural_state_validation_rows.csv'}")


if __name__ == "__main__":
    main()
