from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

SEEDS_EXPECTED = [11, 29, 47, 71, 101]
SYSTEMS_EXPECTED = ["cifar10_cnn", "cifar10_vit", "text_transformer"]
KAPPAS_EXPECTED = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
BETAS_EXPECTED = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
RAW_METHODS_EXPECTED = [
    "afm",
    "projection",
    "unrestricted",
    "linearized_distillation",
    "ewc_prox",
    "replay",
    "derpp",
]

# AFM and projection are identical in the method-neutral frontier.
SUMMARY_FAMILY = {
    "afm": "projection_afm_base",
    "projection": "projection_afm_base",
    "unrestricted": "unrestricted",
    "linearized_distillation": "linearized_distillation",
    "ewc_prox": "ewc_prox",
    "replay": "replay",
    "derpp": "derpp",
}
SUMMARY_FAMILIES = [
    "projection_afm_base",
    "unrestricted",
    "linearized_distillation",
    "ewc_prox",
    "replay",
    "derpp",
]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    xs = sorted(float(x) for x in values if math.isfinite(float(x)))
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
    return xs[lo] * (1.0 - t) + xs[hi] * t


def exact_bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Exact ordered n^n bootstrap of the mean."""
    xs = [float(x) for x in values if math.isfinite(float(x))]
    n = len(xs)
    if not xs:
        return float("nan"), float("nan")
    boot = []
    for draw in itertools.product(range(n), repeat=n):
        boot.append(sum(xs[i] for i in draw) / n)
    return percentile(boot, 0.025), percentile(boot, 0.975)


def finite_mean(values) -> float:
    """Mean of finite values; NaN when no finite values exist."""
    xs = [float(x) for x in values if math.isfinite(float(x))]
    return mean(xs) if xs else float("nan")


def complete_five_seed_summary(values: list[float]) -> tuple[float, float, float]:
    """
    Mean and exact 5^5 bootstrap CI only when all five seed summaries are finite.
    This preserves the declared five-seed replication basis for sensitivity analyses.
    """
    xs = [float(x) for x in values]
    if len(xs) != 5 or not all(math.isfinite(x) for x in xs):
        return float("nan"), float("nan"), float("nan")
    lo, hi = exact_bootstrap_ci(xs)
    return mean(xs), lo, hi


def slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        r = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = r
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = (sum(x*x for x in dx) * sum(y*y for y in dy)) ** 0.5
    if den <= 0.0:
        return float("nan")
    return sum(x*y for x, y in zip(dx, dy)) / den


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def fbool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes"}


def finite_float(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def cv(values: list[float]) -> float:
    xs = [float(x) for x in values]
    if not xs:
        return float("nan")
    m = mean(xs)
    if m == 0.0:
        return 0.0 if all(x == 0.0 for x in xs) else float("inf")
    var = mean((x-m)**2 for x in xs)
    return math.sqrt(var) / abs(m)


def family_row_filter(method: str) -> bool:
    # Keep only projection as the representative of the duplicate AFM/projection
    # proposal family in publication summaries. Raw long/wide outputs retain both.
    return method != "afm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-root", default="runs_compatibility_causal_v1")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Default: <suite-root>/analysis_v15_common_budget_reporting_v2",
    )
    args = ap.parse_args()

    suite = Path(args.suite_root).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else suite / "analysis_v15_common_budget_reporting_v2"
    out.mkdir(parents=True, exist_ok=True)

    source_validation_path = suite / "analysis_v15_common_budget" / "validation.json"
    source_validation = None
    if source_validation_path.is_file():
        source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
        if not bool(source_validation.get("full_coverage")):
            raise RuntimeError(f"Existing v1.5 validation is not full coverage: {source_validation_path}")

    matrix_path = suite / "job_matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    points: list[dict] = []
    native: list[dict] = []
    grid_rows_observed = 0
    full_endpoint = {}
    grid_scale_sets = defaultdict(set)
    summaries: list[dict] = []
    failures: list[dict] = []

    for job in matrix:
        run = Path(job["run_dir"])
        files = {
            "summary": run / "summary.json",
            "frontier": run / "retention_frontier_points.jsonl",
            "grid": run / "retention_frontier_grid.jsonl",
            "native": run / "afm_native_points.jsonl",
        }
        missing = [name for name, p in files.items() if not p.is_file()]
        if missing:
            failures.append({"system": job["system"], "seed": job["seed"], "missing": missing})
            continue

        s = json.loads(files["summary"].read_text(encoding="utf-8"))
        s["system"] = str(job["system"])
        summaries.append(s)

        for p in read_jsonl(files["frontier"]):
            p["system"] = str(job["system"])
            points.append(p)
        for g in read_jsonl(files["grid"]):
            g["system"] = str(job["system"])
            grid_rows_observed += 1
            key5 = (
                str(g["system"]), int(g["seed"]), int(g["state_index"]),
                str(g["method"]), float(g["requested_kappa"]),
            )
            scale_value = float(g["scale"])
            grid_scale_sets[key5].add(scale_value)
            if abs(scale_value - 1.0) <= 1e-12:
                if key5 in full_endpoint:
                    raise RuntimeError(f"Duplicate scale=1 grid endpoint for {key5}")
                full_endpoint[key5] = g
        for n in read_jsonl(files["native"]):
            n["system"] = str(job["system"])
            native.append(n)

    if failures:
        raise RuntimeError(f"Missing v1.5 outputs: {failures}")

    # ------------------------------------------------------------------
    # Basic observed dimensions.
    # ------------------------------------------------------------------
    systems = sorted({str(p["system"]) for p in points})
    seeds = sorted({int(p["seed"]) for p in points})
    methods = sorted({str(p["method"]) for p in points})
    betas = sorted({float(p["retention_beta"]) for p in points})
    requested = sorted({float(p["requested_kappa"]) for p in points})

    # ------------------------------------------------------------------
    # AFM/projection identity audit in method-neutral frontier.
    # ------------------------------------------------------------------
    pindex = {
        (
            str(p["system"]), int(p["seed"]), int(p["state_index"]),
            str(p["method"]), float(p["requested_kappa"]), float(p["retention_beta"]),
        ): p
        for p in points
    }
    duplicate_mismatches = []
    for system in systems:
        for seed in seeds:
            for state in range(50):
                # state_index need not be 0..49, so discover below instead.
                pass
    state_keys = sorted({(str(p["system"]), int(p["seed"]), int(p["state_index"])) for p in points})
    for system, seed, state in state_keys:
        for kappa in requested:
            for beta in betas:
                a = pindex.get((system, seed, state, "afm", kappa, beta))
                q = pindex.get((system, seed, state, "projection", kappa, beta))
                if a is None or q is None:
                    duplicate_mismatches.append({
                        "system": system, "seed": seed, "state_index": state,
                        "requested_kappa": kappa, "beta": beta, "reason": "missing_row",
                    })
                    continue
                diffs = {
                    "scale_diff": abs(float(a["frontier_scale"]) - float(q["frontier_scale"])),
                    "rho_diff": abs(float(a["persistent_ratio"]) - float(q["persistent_ratio"])),
                    "drift_diff": abs(float(a["retention_max_abs_drift"]) - float(q["retention_max_abs_drift"])),
                }
                if max(diffs.values()) > 1e-12:
                    duplicate_mismatches.append({
                        "system": system, "seed": seed, "state_index": state,
                        "requested_kappa": kappa, "beta": beta, **diffs,
                    })

    # ------------------------------------------------------------------
    # Long row-level output: keep all seven raw method labels and all betas.
    # ------------------------------------------------------------------
    native_index = {
        (str(n["system"]), int(n["seed"]), int(n["state_index"]), float(n["requested_kappa"])): n
        for n in native
    }

    long_rows = []
    for p in points:
        system = str(p["system"])
        seed = int(p["seed"])
        state = int(p["state_index"])
        method = str(p["method"])
        kappa = float(p["requested_kappa"])
        beta = float(p["retention_beta"])
        key5 = (system, seed, state, method, kappa)
        full = full_endpoint[key5]
        budget = float(p["retention_budget"])
        selected_drift = float(p["retention_max_abs_drift"])
        full_drift = float(full["retention_max_abs_drift"])
        row = {
            "system": system,
            "seed": seed,
            "state_index": state,
            "parent_step": p.get("parent_step"),
            "method": method,
            "proposal_family": SUMMARY_FAMILY[method],
            "duplicate_of_projection_family": method == "afm",
            "requested_kappa": kappa,
            "realized_current_gradient_kappa": float(p["measured_kappa"]),
            "retention_beta": beta,
            "retention_reference_drift": float(p["retention_reference_drift"]),
            "retention_budget": budget,
            "unrestricted_protected_drift": float(p["unrestricted_protected_drift"]),
            "delta0": float(p["delta0"]),
            "frontier_scale": float(p["frontier_scale"]),
            "rho_persistent": float(p["persistent_ratio"]),
            "persistent_decrease": float(p["persistent_decrease"]),
            "selected_retention_drift": selected_drift,
            "selected_retention_audit_pass": fbool(p["retention_pass"]),
            "full_proposal_retention_drift": full_drift,
            "full_proposal_pass": full_drift <= budget + 1e-12,
            "nonzero_feasible": float(p["frontier_scale"]) > 0.0,
            "budget_utilization": None if beta == 0.0 else selected_drift / budget,
            "frontier_drift_monotone_on_grid": fbool(p["frontier_drift_monotone_on_grid"]),
            "frontier_monotonic_drift_violations": int(p["frontier_monotonic_drift_violations"]),
        }
        if method == "afm":
            n = native_index[(system, seed, state, kappa)]
            lam = finite_float(n.get("afm_lambda_hat"))
            nk = float(n["measured_kappa"])
            nrho = float(n["persistent_ratio"])
            tref = lam * nk / 3.0 if lam is not None else None
            row.update({
                "afm_native_accepted": fbool(n.get("accepted")),
                "afm_native_lambda_hat": lam,
                "afm_native_realized_kappa": nk,
                "afm_native_rho_persistent": nrho,
                "afm_native_theorem_aligned_lambda_kappa_over_3": tref,
                "afm_native_empirical_margin": nrho - tref if tref is not None else None,
                "afm_native_retention_drift": float(n["retention_max_abs_drift"]),
                "afm_native_finite_completion_available": fbool(n.get("finite_completion_available")),
                "afm_native_finite_current_error": finite_float(n.get("finite_current_error")),
                "afm_native_finite_endpoint_error": finite_float(n.get("finite_endpoint_error")),
                "afm_native_finite_protected_error": finite_float(n.get("finite_protected_error")),
                "afm_native_deployed_ratio": finite_float(n.get("deployed_ratio")),
                "afm_native_obstruction": str(n.get("obstruction") or ""),
            })
        long_rows.append(row)

    write_csv(out / "causal_compatibility_v15_reporting_v2_long.csv", long_rows)

    # ------------------------------------------------------------------
    # Wide row-level output: one row per system*seed*state*kappa*raw method.
    # ------------------------------------------------------------------
    wide_groups = defaultdict(list)
    for r in long_rows:
        wide_groups[(r["system"], r["seed"], r["state_index"], r["method"], r["requested_kappa"])].append(r)
    wide_rows = []
    beta_suffix = {0.0:"0", 0.01:"0p01", 0.05:"0p05", 0.10:"0p10", 0.25:"0p25", 0.50:"0p50", 1.0:"1"}
    for key, rows in sorted(wide_groups.items()):
        rows = sorted(rows, key=lambda r: r["retention_beta"])
        first = rows[0]
        w = {
            "system": key[0], "seed": key[1], "state_index": key[2], "method": key[3],
            "proposal_family": first["proposal_family"],
            "duplicate_of_projection_family": first["duplicate_of_projection_family"],
            "requested_kappa": key[4],
            "realized_current_gradient_kappa": first["realized_current_gradient_kappa"],
            "retention_reference_drift": first["retention_reference_drift"],
            "unrestricted_protected_drift": first["unrestricted_protected_drift"],
            "delta0": first["delta0"],
        }
        for r in rows:
            sfx = beta_suffix[float(r["retention_beta"])]
            w[f"rho_beta_{sfx}"] = r["rho_persistent"]
            w[f"scale_beta_{sfx}"] = r["frontier_scale"]
            w[f"drift_beta_{sfx}"] = r["selected_retention_drift"]
            w[f"budget_beta_{sfx}"] = r["retention_budget"]
            w[f"budget_utilization_beta_{sfx}"] = r["budget_utilization"]
            w[f"full_proposal_pass_beta_{sfx}"] = r["full_proposal_pass"]
            w[f"nonzero_feasible_beta_{sfx}"] = r["nonzero_feasible"]
        if key[3] == "afm":
            for name in [
                "afm_native_accepted", "afm_native_lambda_hat", "afm_native_realized_kappa",
                "afm_native_rho_persistent", "afm_native_theorem_aligned_lambda_kappa_over_3",
                "afm_native_empirical_margin", "afm_native_retention_drift",
                "afm_native_finite_completion_available", "afm_native_finite_current_error",
                "afm_native_finite_endpoint_error", "afm_native_finite_protected_error",
                "afm_native_deployed_ratio", "afm_native_obstruction",
            ]:
                w[name] = first.get(name)
        wide_rows.append(w)
    write_csv(out / "causal_compatibility_v15_reporting_v2_wide.csv", wide_rows)

    # ------------------------------------------------------------------
    # Main publication causal table. Collapse AFM/projection to projection row.
    # ------------------------------------------------------------------
    summary_rows_source = [r for r in long_rows if family_row_filter(r["method"])]
    main_groups = defaultdict(list)
    for r in summary_rows_source:
        main_groups[(r["system"], r["proposal_family"], r["requested_kappa"], r["retention_beta"])].append(r)

    main_rows = []
    for key, rows in sorted(main_groups.items()):
        system, family, kappa, beta = key
        by_seed = defaultdict(list)
        for r in rows:
            by_seed[int(r["seed"])].append(r)
        seed_mean_rho = [mean(float(x["rho_persistent"]) for x in rs) for _, rs in sorted(by_seed.items())]
        lo, hi = exact_bootstrap_ci(seed_mean_rho)
        kappas = [float(r["realized_current_gradient_kappa"]) for r in rows]
        rhos = [float(r["rho_persistent"]) for r in rows]
        utils = [float(r["budget_utilization"]) for r in rows if r["budget_utilization"] is not None]
        main_rows.append({
            "system": system,
            "method": family,
            "requested_kappa": kappa,
            "retention_beta": beta,
            "n_states": len(rows),
            "mean_realized_current_gradient_kappa": mean(kappas),
            "median_realized_current_gradient_kappa": median(kappas),
            "mean_rho_persistent": mean(rhos),
            "median_rho_persistent": median(rhos),
            "rho_ci95_low": lo,
            "rho_ci95_high": hi,
            "full_proposal_pass_rate": mean(1.0 if r["full_proposal_pass"] else 0.0 for r in rows),
            "nonzero_feasible_rate": mean(1.0 if r["nonzero_feasible"] else 0.0 for r in rows),
            "mean_frontier_scale": mean(float(r["frontier_scale"]) for r in rows),
            "mean_retention_drift": mean(float(r["selected_retention_drift"]) for r in rows),
            "mean_budget_utilization": mean(utils) if utils else None,
        })
    write_csv(out / "main_causal_table_all_beta.csv", main_rows)

    # ------------------------------------------------------------------
    # Matched within-state slopes and Spearman. One coefficient per state,
    # then average 50 states per seed; seed is independent unit.
    # ------------------------------------------------------------------
    state_groups = defaultdict(list)
    for r in summary_rows_source:
        state_groups[(r["system"], r["seed"], r["state_index"], r["proposal_family"], r["retention_beta"])].append(r)

    state_effects = []
    for key, rows in sorted(state_groups.items()):
        rows = sorted(rows, key=lambda r: r["requested_kappa"])
        xs = [float(r["realized_current_gradient_kappa"]) for r in rows]
        ys = [float(r["rho_persistent"]) for r in rows]
        if len(rows) != len(requested):
            raise RuntimeError(f"Incomplete matched kappa set: {key}: {len(rows)}")
        state_effects.append({
            "system": key[0], "seed": key[1], "state_index": key[2],
            "method": key[3], "retention_beta": key[4],
            "matched_slope": slope(xs, ys),
            "matched_spearman": spearman(xs, ys),
        })
    write_csv(out / "state_level_matched_effects_all_beta.csv", state_effects)

    seed_effect_groups = defaultdict(list)
    for r in state_effects:
        seed_effect_groups[(r["system"], r["seed"], r["method"], r["retention_beta"])].append(r)
    seed_effects = []
    for key, rows in sorted(seed_effect_groups.items()):
        slope_values = [float(r["matched_slope"]) for r in rows]
        spearman_values = [float(r["matched_spearman"]) for r in rows]
        seed_effects.append({
            "system": key[0], "seed": key[1], "method": key[2], "retention_beta": key[3],
            "states": len(rows),
            "defined_matched_slope_states": sum(math.isfinite(v) for v in slope_values),
            "defined_matched_spearman_states": sum(math.isfinite(v) for v in spearman_values),
            "mean_matched_slope": finite_mean(slope_values),
            "mean_matched_spearman": finite_mean(spearman_values),
        })
    write_csv(out / "seed_level_matched_effects_all_beta.csv", seed_effects)

    seed_effect_index = {
        (r["system"], int(r["seed"]), r["method"], float(r["retention_beta"])): r for r in seed_effects
    }
    slope_rows = []
    spear_rows = []
    for family in SUMMARY_FAMILIES:
        for beta in betas:
            sr = {"method": family, "retention_beta": beta}
            rr = {"method": family, "retention_beta": beta}
            for system, prefix in [
                ("cifar10_cnn", "cnn"),
                ("cifar10_vit", "vit"),
                ("text_transformer", "text"),
            ]:
                vals = [float(seed_effect_index[(system, seed, family, beta)]["mean_matched_slope"]) for seed in seeds]
                slope_mean, lo, hi = complete_five_seed_summary(vals)
                sr[f"{prefix}_slope"] = slope_mean
                sr[f"{prefix}_ci95_low"] = lo
                sr[f"{prefix}_ci95_high"] = hi

                rv = [float(seed_effect_index[(system, seed, family, beta)]["mean_matched_spearman"]) for seed in seeds]
                spear_mean, rlo, rhi = complete_five_seed_summary(rv)
                rr[f"{prefix}_spearman"] = spear_mean
                rr[f"{prefix}_ci95_low"] = rlo
                rr[f"{prefix}_ci95_high"] = rhi
                rr[f"{prefix}_defined_seeds"] = sum(math.isfinite(v) for v in rv)

            pooled_slope = []
            for seed in seeds:
                vals = [
                    float(seed_effect_index[(system, seed, family, beta)]["mean_matched_slope"])
                    for system in systems
                ]
                pooled_slope.append(mean(vals) if all(math.isfinite(v) for v in vals) else float("nan"))
            pooled_slope_mean, plo, phi = complete_five_seed_summary(pooled_slope)
            sr["pooled_slope"] = pooled_slope_mean
            sr["pooled_ci95_low"] = plo
            sr["pooled_ci95_high"] = phi

            pooled_spear = []
            for seed in seeds:
                rv = [
                    float(seed_effect_index[(system, seed, family, beta)]["mean_matched_spearman"])
                    for system in systems
                ]
                pooled_spear.append(mean(rv) if all(math.isfinite(v) for v in rv) else float("nan"))
            pooled_spear_mean, rlo, rhi = complete_five_seed_summary(pooled_spear)
            rr["pooled_spearman"] = pooled_spear_mean
            rr["pooled_ci95_low"] = rlo
            rr["pooled_ci95_high"] = rhi
            rr["pooled_defined_seeds"] = sum(math.isfinite(v) for v in pooled_spear)
            slope_rows.append(sr)
            spear_rows.append(rr)

    write_csv(out / "cross_system_matched_slopes_all_beta.csv", slope_rows)
    write_csv(out / "cross_system_matched_spearman_all_beta.csv", spear_rows)

    # ------------------------------------------------------------------
    # Native AFM mechanism table and obstruction/theorem diagnostics.
    # ------------------------------------------------------------------
    native_groups = defaultdict(list)
    for n in native:
        native_groups[(str(n["system"]), float(n["requested_kappa"]))].append(n)

    mechanism_rows = []
    obstruction_rows = []
    negative_margin_rows = []
    theorem_diag_rows = []

    for n in native:
        accepted = fbool(n.get("accepted"))
        lam = finite_float(n.get("afm_lambda_hat")) if accepted else None
        kappa = float(n["measured_kappa"])
        rho = float(n["persistent_ratio"])
        ref = lam * kappa / 3.0 if lam is not None else None
        margin = rho - ref if ref is not None else None
        drow = {
            "system": str(n["system"]), "seed": int(n["seed"]), "state_index": int(n["state_index"]),
            "requested_kappa": float(n["requested_kappa"]), "realized_kappa": kappa,
            "accepted": accepted, "lambda_hat": lam, "rho_persistent": rho,
            "theorem_aligned_lambda_kappa_over_3": ref,
            "empirical_margin": margin,
            "margin_nonnegative": None if margin is None else margin >= 0.0,
            "obstruction": str(n.get("obstruction") or ""),
        }
        theorem_diag_rows.append(drow)
        if margin is not None and margin < 0.0:
            negative_margin_rows.append(drow)
        if str(n.get("obstruction") or ""):
            obstruction_rows.append({
                "system": str(n["system"]), "seed": int(n["seed"]), "state_index": int(n["state_index"]),
                "requested_kappa": float(n["requested_kappa"]), "realized_kappa": kappa,
                "accepted": accepted,
                "finite_completion_available": fbool(n.get("finite_completion_available")),
                "obstruction": str(n.get("obstruction") or ""),
            })

    for key, rows in sorted(native_groups.items()):
        system, requested_kappa = key
        accepted_rows = [r for r in rows if fbool(r.get("accepted"))]
        persistent_fail = [r for r in rows if not fbool(r.get("accepted"))]
        finite_success = [r for r in rows if fbool(r.get("finite_completion_available"))]
        finite_attempts = accepted_rows
        margins = []
        refs = []
        lambdas = []
        for r in accepted_rows:
            lam = finite_float(r.get("afm_lambda_hat"))
            if lam is None:
                continue
            rv = lam * float(r["measured_kappa"]) / 3.0
            refs.append(rv)
            lambdas.append(lam)
            margins.append(float(r["persistent_ratio"]) - rv)

        obs = Counter(str(r.get("obstruction") or "") for r in rows if str(r.get("obstruction") or ""))
        current_errors = [finite_float(r.get("finite_current_error")) for r in finite_success]
        endpoint_errors = [finite_float(r.get("finite_endpoint_error")) for r in finite_success]
        protected_errors = [finite_float(r.get("finite_protected_error")) for r in finite_success]
        deployed = [finite_float(r.get("deployed_ratio")) for r in finite_success]
        current_errors = [x for x in current_errors if x is not None]
        endpoint_errors = [x for x in endpoint_errors if x is not None]
        protected_errors = [x for x in protected_errors if x is not None]
        deployed = [x for x in deployed if x is not None]

        mechanism_rows.append({
            "system": system,
            "requested_kappa": requested_kappa,
            "n_total": len(rows),
            "n_persistent_accepted": len(accepted_rows),
            "persistent_acceptance_rate": len(accepted_rows) / len(rows),
            "n_persistent_backtracking_failures": obs.get("persistent_retention_backtracking_exhausted", 0),
            "mean_rho_persistent_all": mean(float(r["persistent_ratio"]) for r in rows),
            "mean_rho_persistent_accepted": mean(float(r["persistent_ratio"]) for r in accepted_rows) if accepted_rows else None,
            "mean_lambda_hat_accepted": mean(lambdas) if lambdas else None,
            "mean_realized_kappa": mean(float(r["measured_kappa"]) for r in rows),
            "mean_theorem_aligned_lambda_kappa_over_3": mean(refs) if refs else None,
            "median_theorem_aligned_lambda_kappa_over_3": median(refs) if refs else None,
            "mean_empirical_margin": mean(margins) if margins else None,
            "median_empirical_margin": median(margins) if margins else None,
            "minimum_empirical_margin": min(margins) if margins else None,
            "fraction_margin_nonnegative": mean(1.0 if x >= 0.0 else 0.0 for x in margins) if margins else None,
            "n_theorem_margin_rows": len(margins),
            "n_finite_completion_attempts": len(finite_attempts),
            "n_finite_completion_success": len(finite_success),
            "finite_completion_success_rate_overall": len(finite_success) / len(rows),
            "finite_completion_success_rate_given_persistent_acceptance": len(finite_success) / len(accepted_rows) if accepted_rows else None,
            "functional_constraint_inconsistency_count": obs.get("functional_constraint_inconsistency", 0),
            "other_obstruction_count": sum(v for k, v in obs.items() if k not in {"persistent_retention_backtracking_exhausted", "functional_constraint_inconsistency"}),
            "other_obstruction_types": ";".join(sorted(k for k in obs if k not in {"persistent_retention_backtracking_exhausted", "functional_constraint_inconsistency"})),
            "mean_successful_finite_current_error": mean(current_errors) if current_errors else None,
            "max_successful_finite_current_error": max(current_errors) if current_errors else None,
            "mean_successful_finite_endpoint_error": mean(endpoint_errors) if endpoint_errors else None,
            "max_successful_finite_endpoint_error": max(endpoint_errors) if endpoint_errors else None,
            "mean_successful_protected_endpoint_error": mean(protected_errors) if protected_errors else None,
            "max_successful_protected_endpoint_error": max(protected_errors) if protected_errors else None,
            "mean_successful_deployed_ratio": mean(deployed) if deployed else None,
            "min_successful_deployed_ratio": min(deployed) if deployed else None,
            "max_successful_deployed_ratio": max(deployed) if deployed else None,
        })

    write_csv(out / "afm_native_mechanism_table.csv", mechanism_rows)
    write_csv(out / "afm_native_obstructions.csv", obstruction_rows)
    write_csv(out / "afm_theorem_margin_diagnostics.csv", theorem_diag_rows)
    write_csv(out / "afm_negative_theorem_margin_rows.csv", negative_margin_rows)

    # ------------------------------------------------------------------
    # Validation: common reference, coverage, grid, comparator matching,
    # successful finite errors, output counts.
    # ------------------------------------------------------------------
    ref_groups = defaultdict(list)
    for p in points:
        if str(p["method"]) == "unrestricted" and abs(float(p["retention_beta"])) <= 1e-15:
            ref_groups[(str(p["system"]), int(p["seed"]), int(p["state_index"]))].append(p)
    ref_failures = []
    max_ref_abs_error = 0.0
    max_comparator_cv = 0.0
    comparator_cv_failures = []
    for key, rows in sorted(ref_groups.items()):
        refs = [float(r["retention_reference_drift"]) for r in rows]
        drifts = [float(r["unrestricted_protected_drift"]) for r in rows]
        deltas = [float(r["delta0"]) for r in rows]
        err = abs(refs[0] - max(drifts))
        max_ref_abs_error = max(max_ref_abs_error, err)
        invariant = max(refs) - min(refs) <= max(1e-12, 1e-10 * max(abs(refs[0]), 1.0))
        matches = err <= max(1e-12, 1e-9 * max(abs(max(drifts)), 1.0))
        complete = len({float(r["requested_kappa"]) for r in rows}) == len(requested)
        if not (invariant and matches and complete):
            ref_failures.append({"key": key, "invariant": invariant, "matches_max": matches, "complete_kappa": complete})
        c = cv(deltas)
        max_comparator_cv = max(max_comparator_cv, c)
        if c > 0.15:
            comparator_cv_failures.append({"key": key, "cv": c})

    grid_incomplete = []
    expected_scales = {i/32.0 for i in range(33)}
    for key, scales in grid_scale_sets.items():
        if len(scales) != 33 or any(min(abs(s-e) for e in expected_scales) > 1e-12 for s in scales):
            grid_incomplete.append({"key": list(key), "n_scales": len(scales)})

    successful_nonfinite_errors = []
    for n in native:
        if not fbool(n.get("finite_completion_available")):
            continue
        for field in ["finite_current_error", "finite_endpoint_error", "finite_protected_error", "deployed_ratio"]:
            if finite_float(n.get(field)) is None:
                successful_nonfinite_errors.append({
                    "system": n["system"], "seed": n["seed"], "state_index": n["state_index"],
                    "requested_kappa": n["requested_kappa"], "field": field, "value": n.get(field),
                })

    # Exact 50 states per system*seed.
    state_counts = Counter((str(s["system"]), int(s.get("seed", -1))) for s in summaries for _ in [0])
    bad_summary_states = [
        {"system": s["system"], "seed": s.get("seed"), "causal_states": s.get("causal_states")}
        for s in summaries if int(s.get("causal_states", -1)) != 50
    ]

    validation = {
        "schema": "causal_compatibility_v1_5_reporting_v2",
        "source_suite": str(suite),
        "output_directory": str(out),
        "read_only_reporting": True,
        "source_v15_validation_present": source_validation is not None,
        "source_v15_full_coverage": None if source_validation is None else bool(source_validation.get("full_coverage")),
        "runs_expected": 15,
        "runs_observed": len(summaries),
        "systems_observed": systems,
        "seeds_observed": seeds,
        "raw_methods_observed": methods,
        "requested_kappas_observed": requested,
        "retention_betas_observed": betas,
        "causal_states_expected": 750,
        "causal_states_observed": sum(int(s.get("causal_states", 0)) for s in summaries),
        "frontier_rows_expected": 220500,
        "frontier_rows_observed": len(points),
        "grid_rows_expected": 1039500,
        "grid_rows_observed": grid_rows_observed,
        "native_afm_rows_expected": 4500,
        "native_afm_rows_observed": len(native),
        "long_rows_expected": 220500,
        "long_rows_observed": len(long_rows),
        "wide_rows_expected": 31500,
        "wide_rows_observed": len(wide_rows),
        "main_table_rows_expected_after_afm_projection_collapse": 756,
        "main_table_rows_observed": len(main_rows),
        "cross_system_effect_rows_expected": 42,
        "cross_system_slope_rows_observed": len(slope_rows),
        "cross_system_spearman_rows_observed": len(spear_rows),
        "afm_mechanism_rows_expected": 18,
        "afm_mechanism_rows_observed": len(mechanism_rows),
        "reference_state_count": len(ref_groups),
        "reference_failures": ref_failures,
        "max_reference_abs_error": max_ref_abs_error,
        "max_matched_comparator_delta0_cv": max_comparator_cv,
        "comparator_cv_failures_over_0p15": comparator_cv_failures,
        "grid_incomplete_groups": grid_incomplete,
        "afm_projection_identity_mismatches": duplicate_mismatches,
        "successful_completion_nonfinite_error_rows": successful_nonfinite_errors,
        "bad_summary_state_counts": bad_summary_states,
        "negative_theorem_margin_rows": len(negative_margin_rows),
        "statistical_unit": "seed",
        "matched_effect_definition": "within each causal state fit effect across six realized-current-gradient-kappa interventions; average 50 state effects within seed; exact 5^5 bootstrap over five seed summaries",
        "pooled_effect_definition": "equal-weight mean of CNN, ViT, and text seed-level effects within each seed; exact 5^5 bootstrap across five seeds",
        "kappa_note": "For replay and DER++, kappa is the realized compatibility of the manipulated current-gradient component, not the final composite proposal.",
        "theorem_reference_note": "lambda_hat * realized_kappa / 3 is reported as a theorem-aligned empirical reference, not a newly certified theorem bound.",
    }
    validation["pass"] = bool(
        len(summaries) == 15
        and systems == SYSTEMS_EXPECTED
        and seeds == SEEDS_EXPECTED
        and methods == sorted(RAW_METHODS_EXPECTED)
        and requested == KAPPAS_EXPECTED
        and betas == BETAS_EXPECTED
        and validation["causal_states_observed"] == 750
        and len(points) == 220500
        and grid_rows_observed == 1039500
        and len(native) == 4500
        and len(long_rows) == 220500
        and len(wide_rows) == 31500
        and len(main_rows) == 756
        and len(slope_rows) == 42
        and len(spear_rows) == 42
        and len(mechanism_rows) == 18
        and not ref_failures
        and not comparator_cv_failures
        and not grid_incomplete
        and not duplicate_mismatches
        and not successful_nonfinite_errors
        and not bad_summary_states
    )
    (out / "reporting_validation_v2.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    readme = f"""# AFM causal compatibility v1.5 reporting v2

This directory is CPU-only reporting derived from immutable v1.5 outputs.

Key corrections relative to the original analyzer:

1. All seven predeclared retention budgets are retained; beta=0.10 is not privileged.
2. Selected-frontier retention pass is treated only as an audit because the selected point is chosen to satisfy the budget.
3. Full-proposal pass, nonzero-feasible rate, selected scale, retention drift, and budget utilization are reported instead.
4. Primary causal slopes are matched within causal state across the six realized-kappa interventions, then averaged within seed. Seed is the independent unit.
5. Exact 5^5 = 3125 ordered seed bootstrap resamples provide 95% CIs.
6. Cross-system pooled effects give CNN, ViT, and text equal weight within seed.
7. AFM and projection are one identical method-neutral proposal family (`projection_afm_base`) in publication summaries; raw long/wide files retain both source labels and mark AFM as a duplicate.
8. For Replay and DER++, reported kappa is compatibility of the manipulated current-gradient component, not the final composite proposal.
9. Native-AFM finite-completion obstructions are categorical and excluded from successful endpoint-error aggregates; no obstruction sentinel is turned into an infinite error statistic.
10. lambda_hat * realized_kappa / 3 is a theorem-aligned empirical reference, not a newly certified theorem bound. Negative empirical margins are preserved in diagnostics.
11. The original v1.5 analyzer's duplicated `trend_groups.append(...)` reporting bug is avoided entirely by recomputing matched effects from raw frontier rows.

Validation pass: {validation['pass']}
Negative accepted theorem-margin rows: {len(negative_margin_rows)}
"""
    (out / "REPORTING_V2_README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(validation, indent=2, sort_keys=True))
    print("WROTE", out)


if __name__ == "__main__":
    main()
