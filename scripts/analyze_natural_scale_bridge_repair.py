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

REPAIR_REASON = "finite_delta0_match_tolerance_not_met"


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
        w = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def slope_xy(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def kappa_coef_adjusted_for_log_delta0(kappas: list[float], delta0s: list[float], ys: list[float]) -> float:
    """OLS coefficient on kappa in y ~ 1 + kappa + log(Delta0), within one state."""
    if not (len(kappas) == len(delta0s) == len(ys)) or len(ys) < 3:
        return float("nan")
    if any((not math.isfinite(d)) or d <= 0.0 for d in delta0s):
        return float("nan")
    zs = [math.log(d) for d in delta0s]
    mx = statistics.mean(kappas)
    mz = statistics.mean(zs)
    my = statistics.mean(ys)
    xc = [x - mx for x in kappas]
    zc = [z - mz for z in zs]
    yc = [y - my for y in ys]
    sxx = sum(x * x for x in xc)
    szz = sum(z * z for z in zc)
    sxz = sum(x * z for x, z in zip(xc, zc))
    sxy = sum(x * y for x, y in zip(xc, yc))
    szy = sum(z * y for z, y in zip(zc, yc))
    det = sxx * szz - sxz * sxz
    scale = max(sxx * szz, 1.0)
    if not math.isfinite(det) or abs(det) <= 1e-12 * scale:
        return float("nan")
    return (sxy * szz - szy * sxz) / det


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
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def state_key(r: dict) -> tuple[str, int, int, float]:
    return str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"])


def frontier_key(r: dict) -> tuple:
    return (
        str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"]),
        str(r["method"]), float(r["retention_beta"]), float(r["requested_kappa"]),
    )


def native_key(r: dict) -> tuple:
    return (
        str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"]),
        float(r["requested_kappa"]),
    )


def load_suite_outputs(matrix: list[dict], *, repair: bool = False):
    summaries, feasibility, frontier, native, failures = [], [], [], [], []
    for job in matrix:
        rd = Path(job["run_dir"])
        required = [
            rd / "summary.json",
            rd / "bridge_feasibility.jsonl",
            rd / "bridge_frontier_points.jsonl",
            rd / "bridge_afm_native_points.jsonl",
        ]
        if any(not p.is_file() for p in required):
            failures.append(f"missing {'repair ' if repair else ''}outputs: {rd}")
            continue
        summary = json.loads(required[0].read_text(encoding="utf-8"))
        summaries.append(summary)
        if summary.get("status") != "complete":
            failures.append(f"incomplete {'repair ' if repair else ''}summary: {rd}")
        feasibility.extend(read_jsonl(required[1]))
        frontier.extend(read_jsonl(required[2]))
        native.extend(read_jsonl(required[3]))
    return summaries, feasibility, frontier, native, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-suite-root", default="runs_compatibility_natural_scale_bridge_v1")
    ap.add_argument("--repair-suite-root", default="runs_compatibility_natural_scale_bridge_v1_delta0_repair")
    ap.add_argument("--bootstrap-resamples", type=int, default=100000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260819)
    args = ap.parse_args()

    source = Path(args.source_suite_root).resolve()
    repair = Path(args.repair_suite_root).resolve()
    analysis = repair / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    source_matrix = json.loads((source / "job_matrix.json").read_text(encoding="utf-8"))
    repair_matrix = json.loads((repair / "job_matrix.json").read_text(encoding="utf-8"))
    manifest = json.loads((repair / "repair_manifest.json").read_text(encoding="utf-8"))

    source_summaries, source_feas, source_frontier, source_native, fs = load_suite_outputs(source_matrix)
    repair_summaries, repair_feas, repair_frontier, repair_native, fr = load_suite_outputs(repair_matrix, repair=True)
    failures.extend(fs)
    failures.extend(fr)

    target_keys = {
        (str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"]))
        for r in manifest["targets"]
    }
    source_by_key = {state_key(r): r for r in source_feas}
    if len(source_by_key) != len(source_feas):
        failures.append("duplicate source feasibility keys")
    for key in sorted(target_keys):
        old = source_by_key.get(key)
        if old is None:
            failures.append(f"repair target absent from original feasibility: {key}")
        elif str(old.get("reason")) != REPAIR_REASON or bool(old.get("feasible")):
            failures.append(f"repair target was not an original Delta0-admission rejection: {key}")

    repair_by_key = {state_key(r): r for r in repair_feas}
    if len(repair_by_key) != len(repair_feas):
        failures.append("duplicate repair feasibility keys")
    if set(repair_by_key) != target_keys:
        failures.append(
            f"repair feasibility target mismatch: observed={len(repair_by_key)} expected={len(target_keys)} "
            f"missing={len(target_keys-set(repair_by_key))} extra={len(set(repair_by_key)-target_keys)}"
        )
    repair_infeasible = [k for k, r in repair_by_key.items() if not bool(r.get("feasible"))]
    if repair_infeasible:
        failures.append(f"{len(repair_infeasible)} targeted repair conditions unexpectedly remained infeasible")

    # Replace only the over-constrained original rejection rows.  Every other
    # original feasibility decision, including no-positive-endpoint failures, is retained.
    merged_feas_by_key = dict(source_by_key)
    for key, row in repair_by_key.items():
        merged_feas_by_key[key] = {**row, "repair_applied": True, "original_reason": REPAIR_REASON}
    merged_feas = list(merged_feas_by_key.values())

    # Original repaired targets have no frontier/native rows, so unioning is safe;
    # still enforce uniqueness as an audit.
    merged_frontier = source_frontier + repair_frontier
    fk = [frontier_key(r) for r in merged_frontier]
    if len(fk) != len(set(fk)):
        failures.append("duplicate merged frontier keys")
    merged_native = source_native + repair_native
    nk = [native_key(r) for r in merged_native]
    if len(nk) != len(set(nk)):
        failures.append("duplicate merged native keys")

    systems = sorted({str(r["system"]) for r in source_feas})
    seeds = sorted({int(r["seed"]) for r in source_feas})
    kappas = sorted({float(r["requested_kappa"]) for r in merged_frontier})
    methods = sorted({str(r["method"]) for r in merged_frontier})
    betas = sorted({float(r["retention_beta"]) for r in merged_frontier})
    fractions = sorted({float(r["natural_norm_fraction"]) for r in source_feas})

    # Corrected feasibility: Delta0 thresholds are diagnostic only.
    feas_summary = []
    fg = defaultdict(list)
    for r in merged_feas:
        fg[(str(r["system"]), float(r["natural_norm_fraction"]))].append(r)
    for (system, frac), rows in sorted(fg.items()):
        feas = [r for r in rows if bool(r.get("feasible"))]
        reasons = Counter(str(r.get("reason")) for r in rows if not bool(r.get("feasible")))
        outside_old = [
            r for r in feas if not bool(r.get("delta0_match_within_declared_tolerance", str(r.get("reason")) == "matched"))
        ]
        feas_summary.append(
            {
                "system": system,
                "natural_norm_fraction": frac,
                "target_update_norm": float(rows[0]["target_update_norm"]),
                "states_attempted": len(rows),
                "states_feasible_corrected": len(feas),
                "feasibility_rate_corrected": len(feas) / len(rows) if rows else float("nan"),
                "seeds_with_any_feasible_state": len({int(r["seed"]) for r in feas}),
                "feasible_within_original_delta0_tolerance": len(feas) - len(outside_old),
                "feasible_beyond_original_delta0_tolerance": len(outside_old),
                "mean_delta0_cv_feasible": statistics.mean(float(r["delta0_cv"]) for r in feas) if feas else float("nan"),
                "max_delta0_cv_feasible": max((float(r["delta0_cv"]) for r in feas), default=float("nan")),
                "mean_delta0_relative_range_feasible": statistics.mean(float(r["delta0_relative_range"]) for r in feas) if feas else float("nan"),
                "max_delta0_relative_range_feasible": max((float(r["delta0_relative_range"]) for r in feas), default=float("nan")),
                "max_update_norm_relative_error": max((float(r["max_update_norm_relative_error"]) for r in feas), default=float("nan")),
                "max_abs_kappa_error": max((float(r["max_abs_kappa_error"]) for r in feas), default=float("nan")),
                "remaining_infeasible_reason_counts": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
            }
        )
    write_csv(analysis / "bridge_corrected_feasibility_summary.csv", feas_summary)

    # State-level fixed-norm effects.  Raw slope is primary because persistent_ratio
    # already normalises by each kappa direction's own finite unrestricted Delta0.
    # Delta0-adjusted coefficient is a robustness diagnostic, not an admission rule.
    state_rows = []
    bg = defaultdict(list)
    for r in merged_frontier:
        bg[(str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"]), str(r["method"]), float(r["retention_beta"]))].append(r)
    for key, rows in sorted(bg.items()):
        rows = sorted(rows, key=lambda r: float(r["requested_kappa"]))
        if len(rows) != len(kappas):
            failures.append(f"incomplete corrected frontier kappa group: {key} rows={len(rows)}")
            continue
        xs = [float(r["measured_kappa"]) for r in rows]
        ds = [float(r["delta0"]) for r in rows]
        ys = [float(r["persistent_ratio"]) for r in rows]
        state_rows.append(
            {
                "system": key[0], "seed": key[1], "state_id": key[2], "natural_norm_fraction": key[3],
                "method": key[4], "retention_beta": key[5],
                "raw_fixed_norm_kappa_slope": slope_xy(xs, ys),
                "delta0_adjusted_kappa_coefficient": kappa_coef_adjusted_for_log_delta0(xs, ds, ys),
                "mean_delta0": statistics.mean(ds),
                "delta0_cv": statistics.pstdev(ds) / statistics.mean(ds),
                "delta0_relative_range": (max(ds) - min(ds)) / statistics.mean(ds),
                "mean_unrestricted_update_norm": statistics.mean(float(r["unrestricted_update_norm"]) for r in rows),
                "all_state_slopes_positive": None,
            }
        )
    write_csv(analysis / "bridge_corrected_state_level_slopes.csv", state_rows)

    seed_rows = []
    sg = defaultdict(list)
    for r in state_rows:
        sg[(r["system"], r["seed"], r["natural_norm_fraction"], r["method"], r["retention_beta"])].append(r)
    for key, rows in sorted(sg.items()):
        raw = [float(r["raw_fixed_norm_kappa_slope"]) for r in rows if math.isfinite(float(r["raw_fixed_norm_kappa_slope"]))]
        adj = [float(r["delta0_adjusted_kappa_coefficient"]) for r in rows if math.isfinite(float(r["delta0_adjusted_kappa_coefficient"]))]
        seed_rows.append(
            {
                "system": key[0], "seed": key[1], "natural_norm_fraction": key[2], "method": key[3], "retention_beta": key[4],
                "feasible_states": len(rows),
                "mean_raw_fixed_norm_kappa_slope": statistics.mean(raw) if raw else float("nan"),
                "fraction_state_raw_slopes_positive": statistics.mean(1.0 if x > 0.0 else 0.0 for x in raw) if raw else float("nan"),
                "defined_delta0_adjusted_states": len(adj),
                "mean_delta0_adjusted_kappa_coefficient": statistics.mean(adj) if adj else float("nan"),
            }
        )
    write_csv(analysis / "bridge_corrected_seed_level_slopes.csv", seed_rows)

    summary_rows = []
    ag = defaultdict(list)
    for r in seed_rows:
        ag[(r["system"], r["natural_norm_fraction"], r["method"], r["retention_beta"])].append(r)
    for key, rows in sorted(ag.items()):
        raw = [float(r["mean_raw_fixed_norm_kappa_slope"]) for r in rows if math.isfinite(float(r["mean_raw_fixed_norm_kappa_slope"]))]
        adj = [float(r["mean_delta0_adjusted_kappa_coefficient"]) for r in rows if math.isfinite(float(r["mean_delta0_adjusted_kappa_coefficient"]))]
        salt = zlib.crc32(repr(key).encode("utf-8")) % 1000000
        rlo, rhi = mc_ci(raw, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + salt)
        alo, ahi = mc_ci(adj, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 1000003 + salt)
        summary_rows.append(
            {
                "system": key[0], "natural_norm_fraction": key[1], "method": key[2], "retention_beta": key[3],
                "defined_seeds": len(raw),
                "total_feasible_states": sum(int(r["feasible_states"]) for r in rows),
                "mean_raw_fixed_norm_kappa_slope": statistics.mean(raw) if raw else float("nan"),
                "raw_ci95_low": rlo, "raw_ci95_high": rhi,
                "raw_ci_strictly_positive": bool(math.isfinite(rlo) and rlo > 0.0),
                "defined_delta0_adjusted_seeds": len(adj),
                "mean_delta0_adjusted_kappa_coefficient": statistics.mean(adj) if adj else float("nan"),
                "adjusted_ci95_low": alo, "adjusted_ci95_high": ahi,
                "adjusted_ci_strictly_positive": bool(math.isfinite(alo) and alo > 0.0),
                "bootstrap_resamples": int(args.bootstrap_resamples),
            }
        )
    write_csv(analysis / "bridge_corrected_fixed_norm_slopes.csv", summary_rows)

    # Native AFM, using the same fixed-norm comparator set.
    native_state = []
    ng = defaultdict(list)
    for r in merged_native:
        ng[(str(r["system"]), int(r["seed"]), int(r["state_id"]), float(r["natural_norm_fraction"]))].append(r)
    for key, rows in sorted(ng.items()):
        rows = sorted(rows, key=lambda r: float(r["requested_kappa"]))
        if len(rows) != len(kappas):
            failures.append(f"incomplete corrected native kappa group: {key} rows={len(rows)}")
            continue
        xs = [float(r["measured_kappa"]) for r in rows]
        ds = [float(r["delta0"]) for r in rows]
        ys = [float(r["persistent_ratio"]) for r in rows]
        native_state.append(
            {
                "system": key[0], "seed": key[1], "state_id": key[2], "natural_norm_fraction": key[3],
                "raw_fixed_norm_kappa_slope": slope_xy(xs, ys),
                "delta0_adjusted_kappa_coefficient": kappa_coef_adjusted_for_log_delta0(xs, ds, ys),
                "accepted_fraction": statistics.mean(1.0 if bool(r.get("accepted")) else 0.0 for r in rows),
                "finite_completion_fraction": statistics.mean(1.0 if bool(r.get("finite_completion_available")) else 0.0 for r in rows),
                "delta0_cv": statistics.pstdev(ds) / statistics.mean(ds),
            }
        )
    write_csv(analysis / "bridge_corrected_native_afm_state_slopes.csv", native_state)

    nseed = []
    nsg = defaultdict(list)
    for r in native_state:
        nsg[(r["system"], r["seed"], r["natural_norm_fraction"])].append(r)
    for key, rows in sorted(nsg.items()):
        raw = [float(r["raw_fixed_norm_kappa_slope"]) for r in rows if math.isfinite(float(r["raw_fixed_norm_kappa_slope"]))]
        adj = [float(r["delta0_adjusted_kappa_coefficient"]) for r in rows if math.isfinite(float(r["delta0_adjusted_kappa_coefficient"]))]
        nseed.append(
            {
                "system": key[0], "seed": key[1], "natural_norm_fraction": key[2],
                "feasible_states": len(rows),
                "mean_raw_fixed_norm_kappa_slope": statistics.mean(raw) if raw else float("nan"),
                "fraction_state_raw_slopes_positive": statistics.mean(1.0 if x > 0.0 else 0.0 for x in raw) if raw else float("nan"),
                "defined_delta0_adjusted_states": len(adj),
                "mean_delta0_adjusted_kappa_coefficient": statistics.mean(adj) if adj else float("nan"),
                "mean_accepted_fraction": statistics.mean(float(r["accepted_fraction"]) for r in rows),
                "mean_finite_completion_fraction": statistics.mean(float(r["finite_completion_fraction"]) for r in rows),
            }
        )
    write_csv(analysis / "bridge_corrected_native_afm_seed_slopes.csv", nseed)

    nsummary = []
    nag = defaultdict(list)
    for r in nseed:
        nag[(r["system"], r["natural_norm_fraction"])].append(r)
    for key, rows in sorted(nag.items()):
        raw = [float(r["mean_raw_fixed_norm_kappa_slope"]) for r in rows if math.isfinite(float(r["mean_raw_fixed_norm_kappa_slope"]))]
        adj = [float(r["mean_delta0_adjusted_kappa_coefficient"]) for r in rows if math.isfinite(float(r["mean_delta0_adjusted_kappa_coefficient"]))]
        salt = zlib.crc32(repr(key).encode("utf-8")) % 1000000
        rlo, rhi = mc_ci(raw, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 17 + salt)
        alo, ahi = mc_ci(adj, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 1000020 + salt)
        nsummary.append(
            {
                "system": key[0], "natural_norm_fraction": key[1],
                "defined_seeds": len(raw),
                "total_feasible_states": sum(int(r["feasible_states"]) for r in rows),
                "mean_raw_fixed_norm_kappa_slope": statistics.mean(raw) if raw else float("nan"),
                "raw_ci95_low": rlo, "raw_ci95_high": rhi,
                "raw_ci_strictly_positive": bool(math.isfinite(rlo) and rlo > 0.0),
                "defined_delta0_adjusted_seeds": len(adj),
                "mean_delta0_adjusted_kappa_coefficient": statistics.mean(adj) if adj else float("nan"),
                "adjusted_ci95_low": alo, "adjusted_ci95_high": ahi,
                "adjusted_ci_strictly_positive": bool(math.isfinite(alo) and alo > 0.0),
                "mean_accepted_fraction": statistics.mean(float(r["mean_accepted_fraction"]) for r in rows),
                "mean_finite_completion_fraction": statistics.mean(float(r["mean_finite_completion_fraction"]) for r in rows),
                "bootstrap_resamples": int(args.bootstrap_resamples),
            }
        )
    write_csv(analysis / "bridge_corrected_native_afm_slopes.csv", nsummary)

    # Validation of corrected merge. Delta0 CV/range are intentionally absent from
    # admission checks; they are reported only as diagnostics.
    corrected_feasible = [r for r in merged_feas if bool(r.get("feasible"))]
    expected_frontier = len(corrected_feasible) * len(kappas) * len(methods) * len(betas)
    expected_native = len(corrected_feasible) * len(kappas)
    if len(merged_frontier) != expected_frontier:
        failures.append(f"merged frontier rows {len(merged_frontier)} != expected {expected_frontier}")
    if len(merged_native) != expected_native:
        failures.append(f"merged native rows {len(merged_native)} != expected {expected_native}")
    if any(not bool(r.get("retention_pass")) for r in merged_frontier):
        failures.append("corrected selected frontier contains retention-pass violation")

    norm_tol = max(float(r["update_norm_rtol"]) for r in source_matrix)
    kappa_tol = max(float(r["kappa_tolerance"]) for r in source_matrix)
    max_norm_error = max((float(r["max_update_norm_relative_error"]) for r in corrected_feasible), default=float("nan"))
    max_kappa_error = max((float(r["max_abs_kappa_error"]) for r in corrected_feasible), default=float("nan"))
    if math.isfinite(max_norm_error) and max_norm_error > norm_tol + 1e-12:
        failures.append("corrected feasible row exceeds update-norm tolerance")
    if math.isfinite(max_kappa_error) and max_kappa_error > kappa_tol + 1e-12:
        failures.append("corrected feasible row exceeds kappa tolerance")

    old_cv_tol = max(float(r["delta0_cv_tolerance"]) for r in source_matrix)
    old_range_tol = max(float(r["delta0_range_tolerance"]) for r in source_matrix)
    beyond_old = [
        r for r in corrected_feasible
        if float(r.get("delta0_cv", 0.0)) > old_cv_tol + 1e-12
        or float(r.get("delta0_relative_range", 0.0)) > old_range_tol + 1e-12
    ]

    validation = {
        "pass": not failures,
        "failures": failures,
        "source_runs_complete": len(source_summaries),
        "repair_jobs_expected": len(repair_matrix),
        "repair_jobs_complete": len(repair_summaries),
        "repair_target_conditions_expected": len(target_keys),
        "repair_target_conditions_observed": len(repair_feas),
        "repair_target_conditions_feasible": sum(1 for r in repair_feas if bool(r.get("feasible"))),
        "corrected_state_target_conditions": len(merged_feas),
        "corrected_feasible_state_target_conditions": len(corrected_feasible),
        "corrected_feasible_beyond_original_delta0_tolerance": len(beyond_old),
        "delta0_admission_rule": "removed; finite Delta0 CV/range are diagnostics only",
        "original_delta0_cv_tolerance_for_reference": old_cv_tol,
        "original_delta0_relative_range_tolerance_for_reference": old_range_tol,
        "max_corrected_update_norm_relative_error": max_norm_error,
        "declared_update_norm_relative_tolerance": norm_tol,
        "max_corrected_abs_kappa_error": max_kappa_error,
        "declared_abs_kappa_tolerance": kappa_tol,
        "merged_frontier_rows_observed": len(merged_frontier),
        "merged_frontier_rows_expected": expected_frontier,
        "merged_native_rows_observed": len(merged_native),
        "merged_native_rows_expected": expected_native,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (analysis / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))
    for name in [
        "bridge_corrected_feasibility_summary.csv",
        "bridge_corrected_fixed_norm_slopes.csv",
        "bridge_corrected_native_afm_slopes.csv",
    ]:
        print(f"WROTE: {analysis / name}")


if __name__ == "__main__":
    main()
