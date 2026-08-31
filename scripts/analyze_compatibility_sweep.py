from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def percentile_linear(values: list[float], q: float) -> float:
    xs = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = float(q) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return (1.0 - w) * xs[lo] + w * xs[hi]


def exact_seed_bootstrap_ci(values: list[float]) -> tuple[float, float]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    means = [sum(xs[i] for i in draw) / n for draw in itertools.product(range(n), repeat=n)]
    return percentile_linear(means, 0.025), percentile_linear(means, 0.975)


def median(values: Iterable[float]) -> float:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.median(xs) if xs else float("nan")


def mean(values: Iterable[float]) -> float:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    dx = [a - mx for a in x]
    dy = [b - my for b in y]
    denom = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom > 0 else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return correlation(rankdata(x), rankdata(y))


def slope(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    denom = sum((a - mx) ** 2 for a in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom if denom > 0 else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", default="runs_compatibility_causal_v1")
    args = parser.parse_args()
    suite = Path(args.suite_root)
    matrix = json.loads((suite / "job_matrix.json").read_text(encoding="utf-8"))
    analysis = suite / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    points: list[dict] = []
    summaries: list[dict] = []
    failures: list[str] = []
    for row in matrix:
        run = Path(row["run_dir"])
        p = run / "compatibility_points.jsonl"
        s = run / "summary.json"
        if not p.is_file() or not s.is_file():
            failures.append(f"missing output: {run}")
            continue
        summary = json.loads(s.read_text(encoding="utf-8"))
        summaries.append(summary)
        with p.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    item["system"] = row["system"]
                    points.append(item)

    # Seed-level summaries are the independent units for inferential aggregation.
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for p in points:
        grouped[(p["system"], int(p["seed"]), p["method"], float(p["requested_kappa"]))].append(p)

    seed_rows: list[dict] = []
    for (system, seed, method, requested), rows in sorted(grouped.items()):
        retained = [r for r in rows if bool(r["retention_pass"]) and bool(r["accepted"])]
        seed_rows.append(
            {
                "system": system,
                "seed": seed,
                "method": method,
                "requested_kappa": requested,
                "trials": len(rows),
                "median_measured_kappa": median(r["measured_kappa"] for r in rows),
                "median_delta0": median(r["delta0"] for r in rows),
                "median_persistent_ratio_all": median(r["persistent_ratio"] for r in rows),
                "median_persistent_ratio_retention_qualified": median(r["persistent_ratio"] for r in retained),
                "retention_pass_rate": len(retained) / len(rows) if rows else float("nan"),
                "median_retention_drift": median(r["retention_max_abs_drift"] for r in rows),
                "median_deployed_ratio": median(r.get("deployed_ratio") for r in rows),
                "finite_completion_rate": mean(1.0 if r.get("finite_completion_available") else 0.0 for r in rows if r["method"] == "afm"),
                "max_finite_endpoint_error": max([float(r["finite_endpoint_error"]) for r in rows if r.get("finite_endpoint_error") is not None and math.isfinite(float(r["finite_endpoint_error"]))] or [float("nan")]),
                "min_afm_floor_margin": min([float(r["afm_coarse_floor_margin"]) for r in rows if r.get("afm_coarse_floor_margin") is not None and math.isfinite(float(r["afm_coarse_floor_margin"]))] or [float("nan")]),
                "target_clipped_rate": mean(1.0 if r["target_clipped"] else 0.0 for r in rows),
            }
        )
    write_csv(analysis / "seed_level_summary.csv", seed_rows)

    across: dict[tuple, list[dict]] = defaultdict(list)
    for r in seed_rows:
        across[(r["system"], r["method"], r["requested_kappa"])].append(r)
    aggregate_rows: list[dict] = []
    for (system, method, requested), rows in sorted(across.items()):
        values = [float(r["median_persistent_ratio_retention_qualified"]) for r in rows]
        finite = [v for v in values if math.isfinite(v)]
        lo, hi = exact_seed_bootstrap_ci(finite)
        aggregate_rows.append(
            {
                "system": system,
                "method": method,
                "requested_kappa": requested,
                "seeds": len(rows),
                "median_measured_kappa": median(r["median_measured_kappa"] for r in rows),
                "mean_seed_persistent_ratio_retention_qualified": mean(finite),
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_seed_retention_pass_rate": mean(r["retention_pass_rate"] for r in rows),
                "median_seed_delta0": median(r["median_delta0"] for r in rows),
                "median_deployed_ratio": median(r["median_deployed_ratio"] for r in rows),
                "mean_finite_completion_rate": mean(r["finite_completion_rate"] for r in rows),
                "max_finite_endpoint_error": max([float(r["max_finite_endpoint_error"]) for r in rows if math.isfinite(float(r["max_finite_endpoint_error"]))] or [float("nan")]),
                "min_afm_floor_margin": min([float(r["min_afm_floor_margin"]) for r in rows if math.isfinite(float(r["min_afm_floor_margin"]))] or [float("nan")]),
                "mean_target_clipped_rate": mean(r["target_clipped_rate"] for r in rows),
            }
        )
    write_csv(analysis / "aggregate_by_kappa_method.csv", aggregate_rows)

    # Seed-level monotonicity and causal slope, avoiding update-level pseudoreplication.
    trend_rows: list[dict] = []
    trend_group: dict[tuple, list[dict]] = defaultdict(list)
    for p in points:
        if bool(p["accepted"]) and bool(p["retention_pass"]):
            trend_group[(p["system"], int(p["seed"]), p["method"])].append(p)
    for (system, seed, method), rows in sorted(trend_group.items()):
        x = [float(r["measured_kappa"]) for r in rows]
        y = [float(r["persistent_ratio"]) for r in rows]
        trend_rows.append(
            {
                "system": system,
                "seed": seed,
                "method": method,
                "retention_qualified_points": len(rows),
                "slope_rho_on_kappa": slope(x, y),
                "spearman_rho": spearman(x, y),
            }
        )
    write_csv(analysis / "seed_level_trends.csv", trend_rows)

    trend_agg: dict[tuple, list[dict]] = defaultdict(list)
    for r in trend_rows:
        trend_agg[(r["system"], r["method"])].append(r)
    trend_summary: list[dict] = []
    for (system, method), rows in sorted(trend_agg.items()):
        slopes = [float(r["slope_rho_on_kappa"]) for r in rows if math.isfinite(float(r["slope_rho_on_kappa"]))]
        cors = [float(r["spearman_rho"]) for r in rows if math.isfinite(float(r["spearman_rho"]))]
        slo, shi = exact_seed_bootstrap_ci(slopes)
        clo, chi = exact_seed_bootstrap_ci(cors)
        trend_summary.append(
            {
                "system": system,
                "method": method,
                "seeds": len(rows),
                "mean_slope": mean(slopes),
                "slope_ci95_low": slo,
                "slope_ci95_high": shi,
                "mean_spearman": mean(cors),
                "spearman_ci95_low": clo,
                "spearman_ci95_high": chi,
            }
        )
    write_csv(analysis / "trend_summary.csv", trend_summary)

    # Central-figure point file retains all observations but flags seed nesting.
    figure_rows = [
        {
            "system": p["system"],
            "seed": p["seed"],
            "state_index": p["state_index"],
            "method": p["method"],
            "requested_kappa": p["requested_kappa"],
            "measured_kappa": p["measured_kappa"],
            "persistent_ratio": p["persistent_ratio"],
            "retention_pass": p["retention_pass"],
            "retention_max_abs_drift": p["retention_max_abs_drift"],
            "delta0": p["delta0"],
            "deployed_ratio": p.get("deployed_ratio"),
            "afm_coarse_kappa_floor": p.get("afm_coarse_kappa_floor"),
        }
        for p in points
    ]
    write_csv(analysis / "central_figure_points.csv", figure_rows)

    # Retention-qualified cross-method upper envelope.  The seed remains the
    # independent unit: within each seed/bin we first take the best persistent
    # ratio attained by any method that passed the common retention criterion.
    bin_edges = [i / 10.0 for i in range(11)]
    envelope_seed = []
    for system in sorted({p["system"] for p in points}):
        for seed in sorted({int(p["seed"]) for p in points if p["system"] == system}):
            rows = [p for p in points if p["system"] == system and int(p["seed"]) == seed and bool(p["accepted"]) and bool(p["retention_pass"])]
            for b in range(10):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                inside = [p for p in rows if (lo <= float(p["measured_kappa"]) < hi) or (b == 9 and float(p["measured_kappa"]) == 1.0)]
                if not inside:
                    continue
                envelope_seed.append({
                    "system": system,
                    "seed": seed,
                    "bin_low": lo,
                    "bin_high": hi,
                    "bin_mid": 0.5 * (lo + hi),
                    "best_retention_qualified_ratio": max(float(p["persistent_ratio"]) for p in inside),
                    "points": len(inside),
                })
    write_csv(analysis / "retention_qualified_envelope_seed.csv", envelope_seed)
    envelope_groups = defaultdict(list)
    for r in envelope_seed:
        envelope_groups[(r["system"], r["bin_low"], r["bin_high"], r["bin_mid"])].append(r)
    envelope_agg = []
    for (system, lo, hi, mid), rows in sorted(envelope_groups.items()):
        vals = [float(r["best_retention_qualified_ratio"]) for r in rows]
        ci_lo, ci_hi = exact_seed_bootstrap_ci(vals)
        envelope_agg.append({
            "system": system,
            "bin_low": lo,
            "bin_high": hi,
            "bin_mid": mid,
            "seeds": len(rows),
            "mean_best_retention_qualified_ratio": mean(vals),
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
        })
    write_csv(analysis / "retention_qualified_envelope.csv", envelope_agg)

    # Validation gates for the intended broad causal claim.
    systems = sorted({p["system"] for p in points})
    afm = [p for p in points if p["method"] == "afm"]
    coverage = {}
    for system in systems:
        rows = [p for p in afm if p["system"] == system]
        coverage[system] = {
            "min_measured_kappa": min([float(p["measured_kappa"]) for p in rows] or [float("nan")]),
            "max_measured_kappa": max([float(p["measured_kappa"]) for p in rows] or [float("nan")]),
            "clipped_fraction": mean(1.0 if p["target_clipped"] else 0.0 for p in rows),
            "finite_completion_fraction": mean(1.0 if p.get("finite_completion_available") else 0.0 for p in rows),
            "max_endpoint_error": max([float(p["finite_endpoint_error"]) for p in rows if p.get("finite_endpoint_error") is not None and math.isfinite(float(p["finite_endpoint_error"]))] or [float("nan")]),
            "min_coarse_floor_margin": min([float(p["afm_coarse_floor_margin"]) for p in rows if p.get("afm_coarse_floor_margin") is not None and math.isfinite(float(p["afm_coarse_floor_margin"]))] or [float("nan")]),
        }
    validation = {
        "runs_expected": len(matrix),
        "runs_complete": len(summaries),
        "failures": failures,
        "systems": systems,
        "coverage": coverage,
        "independent_unit": "seed",
        "ci": "exact paired/seed bootstrap over available seed-level summaries; all n^n resamples; 2.5/97.5 percentiles with linear interpolation",
    }
    (analysis / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    # Manuscript-oriented compact LaTeX table by system for AFM.
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Causal functional-compatibility sweep. Values summarize retention-qualified AFM persistent progress across five seeds; measured $\\kappa$ is reported because nominal interventions are clipped only when the local geometry cannot realise the requested endpoint.}",
        "\\label{tab:causal-compatibility-sweep}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "System & Target $\\kappa$ & Measured $\\kappa$ & $\\rho_{\\mathrm{persistent}}$ & 95\\% CI & Retention pass & Deployed $\\rho$ & Finite completion " + r"\\",
        "\\midrule",
    ]
    for r in aggregate_rows:
        if r["method"] != "afm":
            continue
        ci = f"[{r['ci95_low']:.3f},{r['ci95_high']:.3f}]" if math.isfinite(float(r["ci95_low"])) else "--"
        lines.append(
            f"{r['system']} & {r['requested_kappa']:.2f} & {r['median_measured_kappa']:.3f} & "
            f"{r['mean_seed_persistent_ratio_retention_qualified']:.3f} & {ci} & "
            f"{r['mean_seed_retention_pass_rate']:.3f} & {r['median_deployed_ratio']:.3f} & "
            f"{r['mean_finite_completion_rate']:.3f} " + r"\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (analysis / "afm_compatibility_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(validation, indent=2, sort_keys=True))
    print(f"WROTE analysis to {analysis}")


if __name__ == "__main__":
    main()
