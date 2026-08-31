from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

SYSTEMS = ["cifar10_cnn", "cifar10_vit"]
SEEDS = [11, 29, 47, 71, 101, 131, 149, 167, 191, 223]
KAPPAS = [0.10, 0.25, 0.50, 0.75]
FRACTIONS = [0.01, 0.10, 0.50, 1.00]
METHODS = ["projection", "unrestricted", "linearized_distillation", "ewc_prox"]
BETAS = [0.05, 0.10, 0.25, 0.50]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slope_xy(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def mc_ci(values: list[float], *, resamples: int, seed: int) -> tuple[float, float]:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(int(seed))
    n = len(vals)
    means = []
    for _ in range(int(resamples)):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", default="runs_compatibility_natural_scale_bridge_v1")
    ap.add_argument("--bootstrap-resamples", type=int, default=100000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260819)
    args = ap.parse_args()

    suite = Path(args.suite_root).resolve()
    analysis = suite / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((suite / "job_matrix.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    summaries = []
    feasibility: list[dict] = []
    frontier: list[dict] = []
    native: list[dict] = []

    for job in matrix:
        rd = Path(job["run_dir"])
        sp = rd / "summary.json"
        fp = rd / "bridge_feasibility.jsonl"
        fr = rd / "bridge_frontier_points.jsonl"
        np = rd / "bridge_afm_native_points.jsonl"
        if not sp.is_file() or not fp.is_file() or not fr.is_file() or not np.is_file():
            failures.append(f"missing outputs: {rd}")
            continue
        summary = json.loads(sp.read_text(encoding="utf-8"))
        summaries.append(summary)
        if summary.get("status") != "complete":
            failures.append(f"incomplete summary: {rd}")
        feasibility.extend(read_jsonl(fp))
        frontier.extend(read_jsonl(fr))
        native.extend(read_jsonl(np))

    # Validation: fixed 2x10x50x4 attempted state-target grid must be complete.
    expected_jobs = len(SYSTEMS) * len(SEEDS)
    expected_feasibility = expected_jobs * 50 * len(FRACTIONS)
    expected_keys = {
        (s, seed, state, frac)
        for s in SYSTEMS
        for seed in SEEDS
        for state in range(50)
        for frac in FRACTIONS
    }
    observed_keys = {
        (str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"]))
        for r in feasibility
    }
    if observed_keys != expected_keys:
        failures.append(
            f"fixed state-target grid mismatch: observed={len(observed_keys)} expected={len(expected_keys)} missing={len(expected_keys-observed_keys)} extra={len(observed_keys-expected_keys)}"
        )
    if len(feasibility) != expected_feasibility:
        failures.append(f"feasibility rows {len(feasibility)} != expected {expected_feasibility}")

    feasible_rows = [r for r in feasibility if bool(r.get("feasible"))]
    max_delta0_cv = max((float(r["delta0_cv"]) for r in feasible_rows), default=float("nan"))
    max_delta0_range = max((float(r["delta0_relative_range"]) for r in feasible_rows), default=float("nan"))
    max_norm_error = max((float(r["max_update_norm_relative_error"]) for r in feasible_rows), default=float("nan"))
    max_kappa_error = max((float(r["max_abs_kappa_error"]) for r in feasible_rows), default=float("nan"))

    # Use declared tolerances from matrix, and verify every feasible row actually satisfies them.
    if matrix:
        cv_tol = max(float(r["delta0_cv_tolerance"]) for r in matrix)
        range_tol = max(float(r["delta0_range_tolerance"]) for r in matrix)
        norm_tol = max(float(r["update_norm_rtol"]) for r in matrix)
        kappa_tol = max(float(r["kappa_tolerance"]) for r in matrix)
        if any(float(r["delta0_cv"]) > cv_tol + 1e-12 for r in feasible_rows):
            failures.append("one or more feasible rows exceed declared Delta0 CV tolerance")
        if any(float(r["delta0_relative_range"]) > range_tol + 1e-12 for r in feasible_rows):
            failures.append("one or more feasible rows exceed declared Delta0 relative-range tolerance")
        if any(float(r["max_update_norm_relative_error"]) > norm_tol + 1e-12 for r in feasible_rows):
            failures.append("one or more feasible rows exceed declared update-norm tolerance")
        if any(float(r["max_abs_kappa_error"]) > kappa_tol + 1e-12 for r in feasible_rows):
            failures.append("one or more feasible rows exceed declared kappa tolerance")
    else:
        cv_tol = range_tol = norm_tol = kappa_tol = float("nan")

    feasible_count = len(feasible_rows)
    expected_frontier = feasible_count * len(KAPPAS) * len(METHODS) * len(BETAS)
    expected_native = feasible_count * len(KAPPAS)
    if len(frontier) != expected_frontier:
        failures.append(f"frontier rows {len(frontier)} != feasible-derived expected {expected_frontier}")
    if len(native) != expected_native:
        failures.append(f"native rows {len(native)} != feasible-derived expected {expected_native}")
    if any(not bool(r.get("retention_pass")) for r in frontier):
        failures.append("selected frontier contains retention-pass violation")

    # Feasibility summary, including how often each true natural-scale target is exactly attainable.
    feasibility_summary = []
    fg = defaultdict(list)
    for r in feasibility:
        fg[(str(r["system"]), float(r["natural_norm_fraction"]))].append(r)
    for (system, frac), rows in sorted(fg.items()):
        feas = [r for r in rows if bool(r.get("feasible"))]
        reasons = Counter(str(r.get("reason")) for r in rows if not bool(r.get("feasible")))
        feasibility_summary.append(
            {
                "system": system,
                "natural_norm_fraction": frac,
                "target_update_norm": float(rows[0]["target_update_norm"]),
                "states_attempted": len(rows),
                "states_feasible": len(feas),
                "feasibility_rate": len(feas) / len(rows) if rows else float("nan"),
                "seeds_with_any_feasible_state": len({int(r["seed"]) for r in feas}),
                "mean_feasible_delta0_cv": statistics.mean(float(r["delta0_cv"]) for r in feas) if feas else float("nan"),
                "max_feasible_delta0_cv": max((float(r["delta0_cv"]) for r in feas), default=float("nan")),
                "mean_feasible_delta0_relative_range": statistics.mean(float(r["delta0_relative_range"]) for r in feas) if feas else float("nan"),
                "max_feasible_delta0_relative_range": max((float(r["delta0_relative_range"]) for r in feas), default=float("nan")),
                "max_update_norm_relative_error": max((float(r["max_update_norm_relative_error"]) for r in feas), default=float("nan")),
                "max_abs_kappa_error": max((float(r["max_abs_kappa_error"]) for r in feas), default=float("nan")),
                "infeasible_reason_counts": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
            }
        )
    write_csv(analysis / "bridge_feasibility_summary.csv", feasibility_summary)

    # State-level matched slopes for common-frontier methods.
    state_rows = []
    by_state = defaultdict(list)
    for r in frontier:
        key = (
            str(r["system"]), int(r["seed"]), int(r["state_id"]),
            float(r["natural_norm_fraction"]), str(r["method"]), float(r["retention_beta"]),
        )
        by_state[key].append(r)
    for key, rows in sorted(by_state.items()):
        rows = sorted(rows, key=lambda r: float(r["requested_kappa"]))
        if len(rows) != len(KAPPAS):
            failures.append(f"incomplete matched frontier kappa group: {key} rows={len(rows)}")
            continue
        xs = [float(r["measured_kappa"]) for r in rows]
        ys = [float(r["persistent_ratio"]) for r in rows]
        state_rows.append(
            {
                "system": key[0], "seed": key[1], "state_id": key[2],
                "natural_norm_fraction": key[3], "method": key[4], "retention_beta": key[5],
                "matched_kappa_slope": slope_xy(xs, ys),
                "mean_delta0": statistics.mean(float(r["delta0"]) for r in rows),
                "delta0_cv": statistics.pstdev([float(r["delta0"]) for r in rows]) / statistics.mean(float(r["delta0"]) for r in rows),
                "mean_unrestricted_update_norm": statistics.mean(float(r["unrestricted_update_norm"]) for r in rows),
            }
        )
    write_csv(analysis / "bridge_state_level_matched_slopes.csv", state_rows)

    seed_rows = []
    sg = defaultdict(list)
    for r in state_rows:
        sg[(r["system"], r["seed"], r["natural_norm_fraction"], r["method"], r["retention_beta"])].append(r)
    for key, rows in sorted(sg.items()):
        slopes = [float(r["matched_kappa_slope"]) for r in rows if math.isfinite(float(r["matched_kappa_slope"]))]
        seed_rows.append(
            {
                "system": key[0], "seed": key[1], "natural_norm_fraction": key[2],
                "method": key[3], "retention_beta": key[4],
                "feasible_states": len(rows),
                "mean_matched_kappa_slope": statistics.mean(slopes) if slopes else float("nan"),
            }
        )
    write_csv(analysis / "bridge_seed_level_matched_slopes.csv", seed_rows)

    summary_rows = []
    ag = defaultdict(list)
    for r in seed_rows:
        ag[(r["system"], r["natural_norm_fraction"], r["method"], r["retention_beta"])].append(r)
    for key, rows in sorted(ag.items()):
        vals = [float(r["mean_matched_kappa_slope"]) for r in rows if math.isfinite(float(r["mean_matched_kappa_slope"]))]
        lo, hi = mc_ci(vals, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + zlib.crc32(repr(key).encode("utf-8")) % 1000000)
        summary_rows.append(
            {
                "system": key[0], "natural_norm_fraction": key[1], "method": key[2], "retention_beta": key[3],
                "defined_seeds": len(vals),
                "total_feasible_states": sum(int(r["feasible_states"]) for r in rows),
                "mean_matched_kappa_slope": statistics.mean(vals) if vals else float("nan"),
                "ci95_low": lo, "ci95_high": hi,
                "ci_strictly_positive": bool(math.isfinite(lo) and lo > 0.0),
                "bootstrap_resamples": int(args.bootstrap_resamples),
            }
        )
    write_csv(analysis / "bridge_matched_slopes.csv", summary_rows)

    # Native AFM state/seed/aggregate slopes.
    native_state = []
    ng = defaultdict(list)
    for r in native:
        ng[(str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"]))].append(r)
    for key, rows in sorted(ng.items()):
        rows = sorted(rows, key=lambda r: float(r["requested_kappa"]))
        if len(rows) != len(KAPPAS):
            failures.append(f"incomplete native kappa group: {key} rows={len(rows)}")
            continue
        xs = [float(r["measured_kappa"]) for r in rows]
        ys = [float(r["persistent_ratio"]) for r in rows]
        native_state.append(
            {
                "system": key[0], "seed": key[1], "state_id": key[2], "natural_norm_fraction": key[3],
                "matched_kappa_slope": slope_xy(xs, ys),
                "accepted_fraction": statistics.mean(1.0 if bool(r.get("accepted")) else 0.0 for r in rows),
                "finite_completion_fraction": statistics.mean(1.0 if bool(r.get("finite_completion_available")) else 0.0 for r in rows),
            }
        )
    write_csv(analysis / "bridge_native_afm_state_slopes.csv", native_state)

    nseed = []
    nsg = defaultdict(list)
    for r in native_state:
        nsg[(r["system"], r["seed"], r["natural_norm_fraction"])].append(r)
    for key, rows in sorted(nsg.items()):
        vals = [float(r["matched_kappa_slope"]) for r in rows if math.isfinite(float(r["matched_kappa_slope"]))]
        nseed.append(
            {
                "system": key[0], "seed": key[1], "natural_norm_fraction": key[2],
                "feasible_states": len(rows),
                "mean_matched_kappa_slope": statistics.mean(vals) if vals else float("nan"),
                "mean_accepted_fraction": statistics.mean(float(r["accepted_fraction"]) for r in rows) if rows else float("nan"),
                "mean_finite_completion_fraction": statistics.mean(float(r["finite_completion_fraction"]) for r in rows) if rows else float("nan"),
            }
        )
    write_csv(analysis / "bridge_native_afm_seed_slopes.csv", nseed)

    nsummary = []
    nag = defaultdict(list)
    for r in nseed:
        nag[(r["system"], r["natural_norm_fraction"])].append(r)
    for key, rows in sorted(nag.items()):
        vals = [float(r["mean_matched_kappa_slope"]) for r in rows if math.isfinite(float(r["mean_matched_kappa_slope"]))]
        lo, hi = mc_ci(vals, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 17 + zlib.crc32(repr(key).encode("utf-8")) % 1000000)
        nsummary.append(
            {
                "system": key[0], "natural_norm_fraction": key[1],
                "defined_seeds": len(vals),
                "total_feasible_states": sum(int(r["feasible_states"]) for r in rows),
                "mean_matched_kappa_slope": statistics.mean(vals) if vals else float("nan"),
                "ci95_low": lo, "ci95_high": hi,
                "ci_strictly_positive": bool(math.isfinite(lo) and lo > 0.0),
                "mean_accepted_fraction": statistics.mean(float(r["mean_accepted_fraction"]) for r in rows) if rows else float("nan"),
                "mean_finite_completion_fraction": statistics.mean(float(r["mean_finite_completion_fraction"]) for r in rows) if rows else float("nan"),
                "bootstrap_resamples": int(args.bootstrap_resamples),
            }
        )
    write_csv(analysis / "bridge_native_afm_slopes.csv", nsummary)

    validation = {
        "pass": not failures,
        "failures": failures,
        "runs_expected": expected_jobs,
        "runs_complete": len(summaries),
        "systems": SYSTEMS,
        "seeds": SEEDS,
        "states_per_run": 50,
        "state_target_conditions_expected": expected_feasibility,
        "state_target_conditions_observed": len(feasibility),
        "feasible_state_target_conditions": feasible_count,
        "frontier_rows_observed": len(frontier),
        "frontier_rows_expected_from_feasibility": expected_frontier,
        "native_rows_observed": len(native),
        "native_rows_expected_from_feasibility": expected_native,
        "max_feasible_delta0_cv": max_delta0_cv,
        "declared_delta0_cv_tolerance": cv_tol,
        "max_feasible_delta0_relative_range": max_delta0_range,
        "declared_delta0_relative_range_tolerance": range_tol,
        "max_feasible_update_norm_relative_error": max_norm_error,
        "declared_update_norm_relative_tolerance": norm_tol,
        "max_feasible_abs_kappa_error": max_kappa_error,
        "declared_abs_kappa_tolerance": kappa_tol,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (analysis / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(validation, indent=2, sort_keys=True))
    print(f"WROTE: {analysis / 'bridge_feasibility_summary.csv'}")
    print(f"WROTE: {analysis / 'bridge_matched_slopes.csv'}")
    print(f"WROTE: {analysis / 'bridge_native_afm_slopes.csv'}")


if __name__ == "__main__":
    main()
