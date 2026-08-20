from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    t = pos - lo
    return xs[lo] * (1 - t) + xs[hi] * t


def exact_bootstrap_ci(values: list[float]) -> tuple[float, float]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    n = len(xs)
    if not xs:
        return float("nan"), float("nan")
    means = []
    for draw in itertools.product(range(n), repeat=n):
        means.append(sum(xs[i] for i in draw) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def slope(x, y):
    if len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    den = sum((a - mx) ** 2 for a in x)
    if den <= 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [a - mx for a in x]
    dy = [b - my for b in y]
    den = (sum(a * a for a in dx) * sum(b * b for b in dy)) ** 0.5
    if den <= 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def system_name(row: dict) -> str:
    return str(row["system"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", default="runs_compatibility_causal_v1")
    args = ap.parse_args()
    suite = Path(args.suite_root).resolve()
    analysis = suite / "analysis_v15_common_budget"
    analysis.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((suite / "job_matrix.json").read_text(encoding="utf-8"))

    points: list[dict] = []
    native: list[dict] = []
    summaries: list[dict] = []
    failures: list[dict] = []
    grid_rows_observed = 0
    grid_scale_sets: dict[tuple, set[float]] = defaultdict(set)

    for row in matrix:
        run = Path(row["run_dir"])
        sp = run / "summary.json"
        fp = run / "retention_frontier_points.jsonl"
        gp = run / "retention_frontier_grid.jsonl"
        np = run / "afm_native_points.jsonl"
        missing = [str(p.name) for p in (sp, fp, gp, np) if not p.is_file()]
        if missing:
            failures.append({
                "system": row["system"], "seed": row["seed"],
                "reason": "missing " + ",".join(missing),
            })
            continue
        s = json.loads(sp.read_text())
        s["system"] = system_name(row)
        summaries.append(s)
        if s.get("status") != "complete" or int(s.get("causal_states", -1)) != 50:
            failures.append({
                "system": row["system"], "seed": row["seed"],
                "reason": f"status={s.get('status')} causal_states={s.get('causal_states')}",
            })
        for p in read_jsonl(fp):
            p["system"] = system_name(row)
            points.append(p)
        for p in read_jsonl(np):
            p["system"] = system_name(row)
            native.append(p)
        for g in read_jsonl(gp):
            grid_rows_observed += 1
            key = (
                system_name(row), int(g["seed"]), int(g["state_index"]),
                str(g["method"]), float(g["requested_kappa"]),
            )
            grid_scale_sets[key].add(float(g["scale"]))

    methods = sorted({p["method"] for p in points})
    betas = sorted({float(p["retention_beta"]) for p in points})
    requested = sorted({float(p["requested_kappa"]) for p in points})
    systems = sorted({p["system"] for p in points})
    grid_n_values = sorted({int(s.get("retention_frontier_grid_points", 0)) // max(int(s.get("compatibility_points", 1)), 1) for s in summaries if int(s.get("compatibility_points", 0)) > 0})
    grid_points = grid_n_values[0] if len(grid_n_values) == 1 else 33

    expected_states = len(matrix) * 50
    expected_frontier = expected_states * len(requested) * len(methods) * len(betas) if methods and betas and requested else 0
    expected_grid = expected_states * len(requested) * len(methods) * grid_points if methods and requested else 0
    expected_native = expected_states * len(requested) if requested else 0

    # ------------------------------------------------------------------
    # State-level common-reference audit.
    # Use one unrestricted frontier row per kappa (beta=0) to avoid method/beta
    # duplication.  The stored state reference must be invariant across kappa and
    # equal max_kappa D_unrestricted(kappa).
    # ------------------------------------------------------------------
    ref_groups = defaultdict(list)
    for p in points:
        if p["method"] == "unrestricted" and abs(float(p["retention_beta"])) <= 1e-15:
            ref_groups[(p["system"], int(p["seed"]), int(p["state_index"]))].append(p)
    ref_audit = []
    reference_failures = []
    for key, rows in sorted(ref_groups.items()):
        refs = [float(r["retention_reference_drift"]) for r in rows]
        drifts = [float(r["unrestricted_protected_drift"]) for r in rows]
        unique_k = sorted({float(r["requested_kappa"]) for r in rows})
        ref = refs[0] if refs else float("nan")
        max_drift = max(drifts) if drifts else float("nan")
        invariant = bool(refs) and max(refs) - min(refs) <= max(1e-12, 1e-10 * max(abs(ref), 1.0))
        matches_max = math.isfinite(ref) and math.isfinite(max_drift) and abs(ref - max_drift) <= max(1e-12, 1e-9 * max(abs(max_drift), 1.0))
        complete_k = len(unique_k) == len(requested)
        row = {
            "system": key[0], "seed": key[1], "state_index": key[2],
            "kappa_levels": len(unique_k), "retention_reference_drift": ref,
            "max_unrestricted_drift": max_drift,
            "reference_invariant_across_kappa": invariant,
            "reference_equals_max_unrestricted_drift": matches_max,
            "all_kappa_levels_present": complete_k,
        }
        ref_audit.append(row)
        if not (invariant and matches_max and complete_k):
            reference_failures.append(row)
    write_csv(analysis / "retention_reference_audit.csv", ref_audit)

    # ------------------------------------------------------------------
    # Common-budget frontier summaries.
    # ------------------------------------------------------------------
    seed_groups = defaultdict(list)
    for p in points:
        seed_groups[(p["system"], int(p["seed"]), p["method"], float(p["retention_beta"]), float(p["requested_kappa"]))].append(p)
    seed_rows = []
    for key, rows in sorted(seed_groups.items()):
        system, seed, method, beta, kappa = key
        ratios = [float(r["persistent_ratio"]) for r in rows]
        scales = [float(r["frontier_scale"]) for r in rows]
        kappas = [float(r["measured_kappa"]) for r in rows]
        drifts = [float(r["retention_max_abs_drift"]) for r in rows]
        budgets = [float(r["retention_budget"]) for r in rows]
        refs = [float(r["retention_reference_drift"]) for r in rows]
        seed_rows.append({
            "system": system, "seed": seed, "method": method, "retention_beta": beta,
            "requested_kappa": kappa, "states": len(rows),
            "mean_measured_kappa": mean(kappas), "median_measured_kappa": median(kappas),
            "mean_persistent_ratio": mean(ratios), "median_persistent_ratio": median(ratios),
            "mean_frontier_scale": mean(scales), "median_frontier_scale": median(scales),
            "mean_retention_drift": mean(drifts), "mean_retention_budget": mean(budgets),
            "mean_retention_reference_drift": mean(refs),
            "retention_pass_rate": mean(1.0 if bool(r["retention_pass"]) else 0.0 for r in rows),
            "positive_progress_rate": mean(1.0 if float(r["persistent_decrease"]) > 0 else 0.0 for r in rows),
        })
    write_csv(analysis / "seed_level_frontier.csv", seed_rows)

    agg_groups = defaultdict(list)
    for r in seed_rows:
        agg_groups[(r["system"], r["method"], r["retention_beta"], r["requested_kappa"])].append(r)
    aggregate = []
    for key, rows in sorted(agg_groups.items()):
        system, method, beta, kappa = key
        vals = [float(r["mean_persistent_ratio"]) for r in rows]
        lo, hi = exact_bootstrap_ci(vals)
        aggregate.append({
            "system": system, "method": method, "retention_beta": beta,
            "requested_kappa": kappa, "seeds": len(rows),
            "mean_seed_measured_kappa": mean(float(r["mean_measured_kappa"]) for r in rows),
            "mean_seed_persistent_ratio": mean(vals), "ci95_low": lo, "ci95_high": hi,
            "mean_seed_frontier_scale": mean(float(r["mean_frontier_scale"]) for r in rows),
            "mean_seed_retention_pass_rate": mean(float(r["retention_pass_rate"]) for r in rows),
            "mean_seed_positive_progress_rate": mean(float(r["positive_progress_rate"]) for r in rows),
            "mean_seed_retention_reference_drift": mean(float(r["mean_retention_reference_drift"]) for r in rows),
        })
    write_csv(analysis / "aggregate_frontier.csv", aggregate)

    trend_groups = defaultdict(list)
    for p in points:
        trend_groups[(p["system"], int(p["seed"]), p["method"], float(p["retention_beta"]))].append(p)
    trends = []
    for key, rows in sorted(trend_groups.items()):
        system, seed, method, beta = key
        x = [float(r["measured_kappa"]) for r in rows]
        y = [float(r["persistent_ratio"]) for r in rows]
        trends.append({
            "system": system, "seed": seed, "method": method, "retention_beta": beta,
            "points": len(rows), "slope_rho_on_kappa": slope(x, y),
            "spearman_rho": spearman(x, y),
        })
    write_csv(analysis / "seed_level_trends.csv", trends)

    tgroups = defaultdict(list)
    for r in trends:
        tgroups[(r["system"], r["method"], r["retention_beta"])].append(r)
    trend_summary = []
    for key, rows in sorted(tgroups.items()):
        system, method, beta = key
        sv = [float(r["slope_rho_on_kappa"]) for r in rows if math.isfinite(float(r["slope_rho_on_kappa"]))]
        cv = [float(r["spearman_rho"]) for r in rows if math.isfinite(float(r["spearman_rho"]))]
        slo, shi = exact_bootstrap_ci(sv)
        clo, chi = exact_bootstrap_ci(cv)
        trend_summary.append({
            "system": system, "method": method, "retention_beta": beta, "seeds": len(rows),
            "mean_slope": mean(sv) if sv else float("nan"),
            "slope_ci95_low": slo, "slope_ci95_high": shi,
            "mean_spearman": mean(cv) if cv else float("nan"),
            "spearman_ci95_low": clo, "spearman_ci95_high": chi,
        })
    write_csv(analysis / "trend_summary.csv", trend_summary)

    central = [{k: p.get(k) for k in [
        "system", "seed", "state_index", "method", "requested_kappa", "measured_kappa",
        "retention_beta", "retention_budget", "retention_reference_drift",
        "unrestricted_protected_drift", "frontier_scale", "persistent_ratio",
        "retention_max_abs_drift", "delta0", "frontier_drift_monotone_on_grid",
        "frontier_monotonic_drift_violations",
    ]} for p in points]
    write_csv(analysis / "central_frontier_points.csv", central)

    # ------------------------------------------------------------------
    # AFM-native transaction: separate from method-neutral frontier.
    # ------------------------------------------------------------------
    write_csv(analysis / "afm_native_finite_vs_persistent.csv", native)
    ngroups = defaultdict(list)
    for p in native:
        ngroups[(p["system"], int(p["seed"]), float(p["requested_kappa"]))].append(p)
    native_seed = []
    for key, rows in sorted(ngroups.items()):
        system, seed, kappa = key
        native_seed.append({
            "system": system, "seed": seed, "requested_kappa": kappa, "states": len(rows),
            "mean_measured_kappa": mean(float(r["measured_kappa"]) for r in rows),
            "mean_persistent_ratio": mean(float(r["persistent_ratio"]) for r in rows),
            "mean_deployed_ratio": mean(float(r["deployed_ratio"]) for r in rows if r.get("deployed_ratio") is not None),
            "finite_completion_rate": mean(1.0 if r.get("finite_completion_available") else 0.0 for r in rows),
            "max_finite_endpoint_error": max([float(r["finite_endpoint_error"]) for r in rows if r.get("finite_endpoint_error") is not None] or [float("nan")]),
        })
    write_csv(analysis / "afm_native_seed_level.csv", native_seed)

    nagg = defaultdict(list)
    for r in native_seed:
        nagg[(r["system"], r["requested_kappa"])].append(r)
    native_summary = []
    for key, rows in sorted(nagg.items()):
        system, kappa = key
        vals = [float(r["mean_persistent_ratio"]) for r in rows]
        lo, hi = exact_bootstrap_ci(vals)
        native_summary.append({
            "system": system, "requested_kappa": kappa, "seeds": len(rows),
            "mean_seed_measured_kappa": mean(float(r["mean_measured_kappa"]) for r in rows),
            "mean_seed_persistent_ratio": mean(vals), "persistent_ci95_low": lo,
            "persistent_ci95_high": hi,
            "mean_seed_deployed_ratio": mean(float(r["mean_deployed_ratio"]) for r in rows),
            "mean_seed_finite_completion_rate": mean(float(r["finite_completion_rate"]) for r in rows),
            "max_seed_finite_endpoint_error": max(float(r["max_finite_endpoint_error"]) for r in rows),
        })
    write_csv(analysis / "afm_native_summary.csv", native_summary)

    ntrend_groups = defaultdict(list)
    for p in native:
        ntrend_groups[(p["system"], int(p["seed"]))].append(p)
    native_trends = []
    for key, rows in sorted(ntrend_groups.items()):
        x = [float(r["measured_kappa"]) for r in rows]
        y = [float(r["persistent_ratio"]) for r in rows]
        native_trends.append({
            "system": key[0], "seed": key[1], "points": len(rows),
            "slope_rho_on_kappa": slope(x, y), "spearman_rho": spearman(x, y),
        })
    write_csv(analysis / "afm_native_seed_trends.csv", native_trends)
    ntg = defaultdict(list)
    for r in native_trends:
        ntg[r["system"]].append(r)
    native_trend_summary = []
    for system, rows in sorted(ntg.items()):
        sv = [float(r["slope_rho_on_kappa"]) for r in rows]
        cv = [float(r["spearman_rho"]) for r in rows]
        slo, shi = exact_bootstrap_ci(sv)
        clo, chi = exact_bootstrap_ci(cv)
        native_trend_summary.append({
            "system": system, "seeds": len(rows), "mean_slope": mean(sv),
            "slope_ci95_low": slo, "slope_ci95_high": shi,
            "mean_spearman": mean(cv), "spearman_ci95_low": clo, "spearman_ci95_high": chi,
        })
    write_csv(analysis / "afm_native_trend_summary.csv", native_trend_summary)

    # Grid completeness audit: every method/kappa/state must have the same full
    # 33-point predeclared scale set.
    expected_scale_set = {i / float(grid_points - 1) for i in range(grid_points)}
    grid_incomplete = []
    for key, scales in grid_scale_sets.items():
        if len(scales) != grid_points or any(min(abs(x - y) for y in expected_scale_set) > 1e-12 for x in scales):
            grid_incomplete.append({"key": key, "scales": len(scales)})

    validation = {
        "schema": "causal_compatibility_v1_5_common_budget_frontier",
        "runs_expected": len(matrix), "runs_with_outputs": len(summaries), "failures": failures,
        "systems": systems, "methods": methods, "retention_budget_betas": betas,
        "requested_kappas": requested,
        "causal_states_expected": expected_states,
        "causal_states_observed": sum(int(s.get("causal_states", 0)) for s in summaries),
        "frontier_rows_expected": expected_frontier, "frontier_rows_observed": len(points),
        "grid_rows_expected": expected_grid, "grid_rows_observed": grid_rows_observed,
        "afm_native_rows_expected": expected_native, "afm_native_rows_observed": len(native),
        "retention_reference_states_expected": expected_states,
        "retention_reference_states_observed": len(ref_audit),
        "retention_reference_failures": reference_failures,
        "grid_incomplete_groups": grid_incomplete,
        "retention_reference_rule": "D_ref(state)=max over requested kappa of unrestricted comparator protected max-absolute logit drift",
        "frontier_definition": "largest scale on predeclared 33-point [0,1] proposal grid satisfying protected max-absolute logit drift <= max(epsilon_num, beta * D_ref(state)); D_ref is identical across all kappa and methods within a causal state",
        "afm_native_definition": "AFM-specific compatible persistent transaction and finite endpoint completion, recorded separately from the method-neutral retention frontier",
        "independent_statistical_unit": "seed",
        "ci": "exact bootstrap over seed-level summaries; all n^n resamples; 2.5/97.5 percentiles with linear interpolation",
    }
    validation["full_coverage"] = bool(
        not failures
        and not reference_failures
        and not grid_incomplete
        and validation["causal_states_observed"] == expected_states
        and len(points) == expected_frontier
        and grid_rows_observed == expected_grid
        and len(native) == expected_native
        and len(ref_audit) == expected_states
    )
    (analysis / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    print(json.dumps(validation, indent=2, sort_keys=True))
    print("WROTE", analysis)


if __name__ == "__main__":
    main()
