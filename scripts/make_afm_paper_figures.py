#!/usr/bin/env python3
"""
Generate publication-ready AFM compatibility paper figures from existing experiments.

Creates every requested panel except Fig. 1a, plus assembled vector-PDF figures:
  Figure1_bcd.pdf
  Figure2_abcd.pdf
  Figure3_abcd.pdf
  Figure4_abcd.pdf
and individual panel PDFs in <out>/panels/.

Run from the project root:
    python3 scripts/make_afm_paper_figures.py --root . --out paper_figures

Dependencies: numpy, pandas, matplotlib (no seaborn).

The script intentionally reads experiment outputs rather than hard-coding the headline
numbers. If a required output is absent, it raises a clear error naming the file(s).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# -----------------------------------------------------------------------------
# Publication defaults
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

SYSTEM_ORDER = ["cifar10_cnn", "cifar10_vit", "text_transformer"]
SYSTEM_LABEL = {
    "cifar10_cnn": "CNN",
    "cifar10_vit": "ViT",
    "cifar10_vit_strong": "Strong ViT",
    "text_transformer": "Text transformer",
}
METHOD_LABEL = {
    "projection_afm_base": "Projection / AFM-base",
    "projection": "Projection / AFM-base",
    "afm": "AFM",
    "unrestricted": "Unrestricted",
    "linearized_distillation": "Linearized distill.",
    "ewc_prox": "EWC-prox",
    "replay": "Replay",
    "derpp": "DER++",
}
METHOD_ORDER_MAIN = [
    "projection_afm_base", "unrestricted", "linearized_distillation",
    "ewc_prox", "replay", "derpp",
]
BETA_COL = {
    0.0: "rho_beta_0",
    0.01: "rho_beta_0p01",
    0.05: "rho_beta_0p05",
    0.1: "rho_beta_0p10",
    0.25: "rho_beta_0p25",
    0.5: "rho_beta_0p50",
    1.0: "rho_beta_1",
}


def _clean_ax(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _first_existing(root: Path, candidates: Sequence[str], description: str) -> Path:
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p
    # Last-resort basename search inside likely analysis trees.
    basenames = {Path(x).name for x in candidates}
    found = []
    for b in basenames:
        found.extend(root.glob(f"**/{b}"))
    found = [p for p in found if p.is_file()]
    if len(found) == 1:
        return found[0]
    msg = [f"Missing required {description}.", "Tried:"]
    msg += [f"  - {root / c}" for c in candidates]
    if found:
        msg += ["Ambiguous basename matches:"] + [f"  - {p}" for p in found]
    raise FileNotFoundError("\n".join(msg))


def paths(root: Path) -> dict[str, Path]:
    """Locate existing experiment analysis outputs."""
    return {
        "v15_wide": _first_existing(root, [
            "runs_compatibility_causal_v1/analysis_v15_common_budget_reporting_v2/causal_compatibility_v15_reporting_v2_wide.csv",
            "analysis_v15_common_budget_reporting_v2/causal_compatibility_v15_reporting_v2_wide.csv",
            "causal_compatibility_v15_reporting_v2_wide.csv",
        ], "v1.5 wide reporting table"),
        "v15_slopes": _first_existing(root, [
            "runs_compatibility_causal_v1/analysis_v15_common_budget_reporting_v2/cross_system_matched_slopes_all_beta.csv",
            "analysis_v15_common_budget_reporting_v2/cross_system_matched_slopes_all_beta.csv",
            "cross_system_matched_slopes_all_beta.csv",
            "table2_cross_system_matched_slopes_all_beta.csv",
        ], "v1.5 cross-system slope table"),
        "indep_long": _first_existing(root, [
            "runs_compatibility_independent_directions_v1/analysis/independent_direction_frontier_rows.csv",
        ], "independent-direction long rows"),
        "indep_summary": _first_existing(root, [
            "runs_compatibility_independent_directions_v1/analysis/direction_independence_summary.csv",
        ], "independent-direction summary"),
        "generality": _first_existing(root, [
            "runs_compatibility_generality_v1/analysis/generality_matched_slopes.csv",
        ], "10-seed generality slopes"),
        "multiscale_slopes": _first_existing(root, [
            "runs_compatibility_multiscale_v1/analysis/matched_slopes_by_scale.csv",
        ], "multiscale slope table"),
        "multiscale_aggregate": _first_existing(root, [
            "runs_compatibility_multiscale_v1/analysis/aggregate_multiscale.csv",
        ], "multiscale aggregate table"),
        "natural_rows": _first_existing(root, [
            "runs_compatibility_natural_v1/analysis/natural_state_validation_rows.csv",
            "natural_state_validation_rows.csv",
        ], "natural-state row table"),
        "natural_summary": _first_existing(root, [
            "runs_compatibility_natural_v1/analysis/descriptive_summary.csv",
            "descriptive_summary.csv",
        ], "natural-state descriptive summary"),
        "bridge_feasibility": _first_existing(root, [
            "runs_compatibility_natural_scale_bridge_v1_delta0_repair/analysis/bridge_corrected_feasibility_summary.csv",
        ], "corrected bridge feasibility table"),
        "bridge_decomp": _first_existing(root, [
            "runs_compatibility_natural_scale_bridge_v1_delta0_repair/analysis/bridge_decomposition_summary.csv",
        ], "bridge decomposition summary"),
        "bridge_heterogeneity": _first_existing(root, [
            "runs_compatibility_natural_scale_bridge_v1_delta0_repair/analysis/bridge_decomposition_by_delta0_heterogeneity.csv",
            "bridge_decomposition_by_delta0_heterogeneity.csv",
        ], "bridge Delta0-heterogeneity table"),
    }


# -----------------------------------------------------------------------------
# Small statistics helpers
# -----------------------------------------------------------------------------
def slope_xy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or np.var(x) <= 0:
        return np.nan
    return float(np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2))


def seed_bootstrap_ci(seed_values: Sequence[float], n: int = 100000,
                      seed: int = 20260819) -> tuple[float, float]:
    vals = np.asarray([v for v in seed_values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    # Chunk to avoid a giant (n x nseed) allocation.
    means = np.empty(n, dtype=float)
    chunk = 10000
    for start in range(0, n, chunk):
        m = min(chunk, n-start)
        idx = rng.integers(0, len(vals), size=(m, len(vals)))
        means[start:start+m] = vals[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def _mean_seed_ci(df: pd.DataFrame, value: str, group_cols: list[str]) -> pd.DataFrame:
    seed = df.groupby(group_cols + ["seed"], as_index=False)[value].mean()
    rows = []
    for key, g in seed.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        vals = g[value].to_numpy(float)
        lo, hi = seed_bootstrap_ci(vals)
        r = dict(zip(group_cols, key))
        r.update(mean=float(np.mean(vals)), lo=lo, hi=hi, nseed=len(vals))
        rows.append(r)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------
def panel_1b(ax: plt.Axes, P: dict[str, Path]) -> None:
    """Main causal dose response, projection/AFM-compatible family, beta=.5."""
    df = _read_csv(P["v15_wide"])
    # AFM and projection are duplicate on the common frontier. Pick projection if present.
    method = "projection" if "projection" in set(df.method) else "afm"
    d = df[df.method.eq(method)].copy()
    d["rho"] = pd.to_numeric(d[BETA_COL[0.5]], errors="coerce")
    d = d[np.isfinite(d.rho)]

    # Seed-level uncertainty at each requested intervention, plotted at mean realized kappa.
    seed = d.groupby(["system", "seed", "requested_kappa"], as_index=False).agg(
        realized_kappa=("realized_current_gradient_kappa", "mean"),
        rho=("rho", "mean"),
    )
    agg = _mean_seed_ci(seed, "rho", ["system", "requested_kappa"])
    xr = seed.groupby(["system", "requested_kappa"], as_index=False).realized_kappa.mean()
    agg = agg.merge(xr, on=["system", "requested_kappa"], how="left")

    slopes = _read_csv(P["v15_slopes"])
    sm = slopes[(slopes.retention_beta.eq(0.5)) & slopes.method.eq("projection_afm_base")]
    if sm.empty:
        sm = slopes[(slopes.retention_beta.eq(0.5)) & slopes.method.isin(["projection", "afm"])].head(1)

    for system in SYSTEM_ORDER:
        g = agg[agg.system.eq(system)].sort_values("realized_kappa")
        line, = ax.plot(g.realized_kappa, g["mean"], marker="o", label=SYSTEM_LABEL[system])
        ax.fill_between(g.realized_kappa, g.lo, g.hi, alpha=0.13, color=line.get_color(), linewidth=0)

    if not sm.empty:
        r = sm.iloc[0]
        text = (f"matched slopes, β=0.5\n"
                f"CNN {r.cnn_slope:.3f}   ViT {r.vit_slope:.3f}   Text {r.text_slope:.3f}")
        ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.92))
    ax.set_xlabel("Realized compatibility, κ")
    ax.set_ylabel("Persistent-progress ratio, ρ")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    _clean_ax(ax)


def panel_1c(ax: plt.Axes, P: dict[str, Path]) -> None:
    df = _read_csv(P["v15_slopes"])
    d = df[df.method.eq("projection_afm_base")].copy()
    if d.empty:
        d = df[df.method.isin(["projection", "afm"])].copy()
    specs = [
        ("cnn", "CNN"), ("vit", "ViT"), ("text", "Text transformer")
    ]
    budgets = [0, .01, .05, .1, .25, .5, 1]
    xpos = np.arange(len(budgets))
    d = d.set_index("retention_beta").reindex(budgets).reset_index()
    for prefix, label in specs:
        ax.plot(xpos, d[f"{prefix}_slope"], marker="o", label=label)
        ax.fill_between(xpos,
                        d[f"{prefix}_ci95_low"], d[f"{prefix}_ci95_high"], alpha=0.12)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["0", ".01", ".05", ".10", ".25", ".50", "1"])
    ax.set_xlabel("Retention budget, β")
    ax.set_ylabel("Matched κ→ρ slope")
    ax.legend(frameon=False)
    _clean_ax(ax)


def panel_1d(ax: plt.Axes, P: dict[str, Path]) -> None:
    df = _read_csv(P["v15_slopes"])
    d = df[np.isclose(df.retention_beta, .5)].copy()
    aliases = {
        "projection_afm_base": "projection_afm_base",
        "unrestricted": "unrestricted",
        "linearized_distillation": "linearized_distillation",
        "ewc_prox": "ewc_prox",
        "replay": "replay",
        "derpp": "derpp",
    }
    methods = [m for m in METHOD_ORDER_MAIN if m in set(d.method)]
    y = np.arange(len(methods), dtype=float)
    offsets = [-0.20, 0.0, 0.20]
    specs = [("cnn", "CNN"), ("vit", "ViT"), ("text", "Text")]
    for off, (prefix, label) in zip(offsets, specs):
        vals, lo, hi = [], [], []
        for m in methods:
            r = d[d.method.eq(m)].iloc[0]
            vals.append(r[f"{prefix}_slope"])
            lo.append(r[f"{prefix}_ci95_low"])
            hi.append(r[f"{prefix}_ci95_high"])
        vals, lo, hi = map(np.asarray, (vals, lo, hi))
        ax.errorbar(vals, y+off, xerr=[vals-lo, hi-vals], fmt="o", capsize=2, label=label)
    ax.axvline(0, color="0.55", lw=.7)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Matched κ→ρ slope at β=0.5")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    _clean_ax(ax, grid=False)
    ax.grid(axis="x", alpha=.18, linewidth=.6)


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------
def panel_2a(fig: plt.Figure, subspec, P: dict[str, Path], label: str | None = None) -> None:
    """Three mini-panels: four independent direction IDs at each kappa, beta=.5."""
    df = _read_csv(P["indep_long"])
    method = "projection" if "projection" in set(df.method) else "afm"
    d = df[(df.method.eq(method)) & np.isclose(df.retention_beta, .5)].copy()
    # Average each independently generated direction ID over states within seed, then over seeds.
    sd = d.groupby(["system", "seed", "requested_kappa", "direction_id"], as_index=False).agg(
        rho=("persistent_ratio", "mean")
    )
    direction_mean = sd.groupby(["system", "requested_kappa", "direction_id"], as_index=False).rho.mean()
    grand = _mean_seed_ci(
        d.groupby(["system", "seed", "requested_kappa"], as_index=False).persistent_ratio.mean(),
        "persistent_ratio", ["system", "requested_kappa"]
    )
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=subspec, wspace=.34)
    for i, system in enumerate(SYSTEM_ORDER):
        ax = fig.add_subplot(gs[0, i])
        g = direction_mean[direction_mean.system.eq(system)]
        for did, gd in g.groupby("direction_id"):
            gd = gd.sort_values("requested_kappa")
            ax.plot(gd.requested_kappa, gd.rho, marker="o", alpha=.42, lw=.85)
        gg = grand[grand.system.eq(system)].sort_values("requested_kappa")
        ax.errorbar(gg.requested_kappa, gg["mean"],
                    yerr=[gg["mean"]-gg.lo, gg.hi-gg["mean"]],
                    fmt="o-", lw=1.8, capsize=2.0, zorder=10)
        ax.set_title(SYSTEM_LABEL[system])
        ax.set_xticks([.1,.25,.5,.75])
        ax.set_xlabel("κ")
        if i == 0:
            ax.set_ylabel("Persistent ratio, ρ")
        _clean_ax(ax)
        if i == 0 and label:
            _panel_label(ax, label)


def panel_2b(ax: plt.Axes, P: dict[str, Path]) -> None:
    df = _read_csv(P["indep_summary"])
    method = "projection" if "projection" in set(df.method) else "afm"
    d = df[(df.method.eq(method)) & np.isclose(df.retention_beta, .5) & df.system.isin(SYSTEM_ORDER)].copy()
    d["order"] = d.system.map({s:i for i,s in enumerate(SYSTEM_ORDER)})
    d = d.sort_values("order")
    x = np.arange(len(d))
    v = d.mean_direction_to_kappa_sd_ratio.to_numpy(float)
    lo = d.ratio_ci95_low.to_numpy(float)
    hi = d.ratio_ci95_high.to_numpy(float)
    ax.errorbar(x, v, yerr=[v-lo, hi-v], fmt="o", capsize=3)
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.text(xi, hi[list(x).index(xi)] + max(.004, .07*np.nanmax(hi)),
                f"slope {r.mean_matched_kappa_slope:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1, color="0.6", ls="--", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABEL[s] for s in d.system])
    ax.set_ylabel(r"Direction / between-κ variability  $R_{dir/κ}$")
    ax.set_ylim(bottom=0)
    _clean_ax(ax)


def panel_2c(ax: plt.Axes, P: dict[str, Path]) -> None:
    df = _read_csv(P["generality"])
    method = "projection" if "projection" in set(df.method) else "afm"
    systems = ["cifar10_cnn", "cifar10_vit", "cifar10_vit_strong", "text_transformer"]
    d = df[(df.method.eq(method)) & np.isclose(df.retention_beta,.5) & df.system.isin(systems)].copy()
    d["order"] = d.system.map({s:i for i,s in enumerate(systems)})
    d = d.sort_values("order")
    x = np.arange(len(d))
    v = d.mean_matched_slope.to_numpy(float)
    lo, hi = d.ci95_low.to_numpy(float), d.ci95_high.to_numpy(float)
    ax.errorbar(x, v, yerr=[v-lo, hi-v], fmt="o", capsize=3)
    ax.axhline(1, color="0.55", lw=.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABEL[s] for s in d.system], rotation=20, ha="right")
    ax.set_ylabel("Matched κ→ρ slope, β=0.5")
    for xi, vv in zip(x, v):
        ax.text(xi, vv + .018, f"{vv:.3f}", ha="center", va="bottom", fontsize=7)
    _clean_ax(ax)


def panel_2d(ax: plt.Axes, P: dict[str, Path]) -> None:
    slopes = _read_csv(P["multiscale_slopes"])
    agg = _read_csv(P["multiscale_aggregate"])
    method = "projection" if "projection" in set(slopes.method) else "afm"
    d = slopes[(slopes.method.eq(method)) & np.isclose(slopes.retention_beta,.5)].copy()
    # Actual scale displayed as multiple of the original v1.5 local intervention.
    a = agg[(agg.method.eq(method)) & np.isclose(agg.retention_beta,.5)].copy()
    xm = a.groupby(["system","scale_fraction"], as_index=False).mean_target_multiple_of_v15_local.mean()
    d = d.merge(xm, on=["system","scale_fraction"], how="left")
    for system in SYSTEM_ORDER:
        g = d[d.system.eq(system)].sort_values("mean_target_multiple_of_v15_local")
        if g.empty:
            continue
        alpha = 1.0 if system != "text_transformer" else .55
        line = ax.plot(g.mean_target_multiple_of_v15_local, g.matched_slope,
                       marker="o", label=SYSTEM_LABEL[system], alpha=alpha)[0]
        ax.fill_between(g.mean_target_multiple_of_v15_local, g.ci95_low, g.ci95_high,
                        alpha=.10*alpha, color=line.get_color(), linewidth=0)
    ax.axhline(1, color="0.55", ls="--", lw=.8)
    ax.set_xlabel("Intervention magnitude / original v1.5 local magnitude")
    ax.set_ylabel("Matched κ→ρ slope, β=0.5")
    ax.legend(frameon=False)
    ax.text(.02,.03,"Designed enlarged local regime — not % of a natural update",
            transform=ax.transAxes, fontsize=6.8, va="bottom")
    _clean_ax(ax)


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------
def panel_3a(ax: plt.Axes, P: dict[str, Path]) -> None:
    df = _read_csv(P["natural_rows"])
    # natural_kappa repeats by method; AFM gives one row per natural state.
    d = df[df.method.eq("afm")].copy()
    data = [d[d.system.eq(s)].natural_kappa.dropna().to_numpy(float) for s in SYSTEM_ORDER]
    vp = ax.violinplot(data, positions=np.arange(3), showmeans=False, showmedians=False, widths=.72)
    # Overlay box-like robust summaries without requiring seaborn.
    for i, vals in enumerate(data):
        q1, med, q3 = np.quantile(vals, [.25,.5,.75])
        ax.plot([i,i],[q1,q3], lw=5, solid_capstyle="butt")
        ax.plot(i, med, marker="o", ms=3.5)
        ax.text(i, min(1.02, vals.mean()+.08), f"mean {vals.mean():.3f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([SYSTEM_LABEL[s] for s in SYSTEM_ORDER])
    ax.set_ylabel("Natural compatibility, κ")
    ax.set_ylim(0,1.06)
    _clean_ax(ax)


def panel_3b(ax: plt.Axes, P: dict[str, Path]) -> None:
    df = _read_csv(P["natural_rows"])
    d = df[df.method.eq("afm")].copy()
    rows=[]
    for s,g in d.groupby("system"):
        rows.append({
            "system":s,
            "mean_kappa":g.natural_kappa.mean(),
            "mean_lambda":g.afm_lambda_hat.mean(),
            "median_rho":g.rho_persistent.median(),
        })
    q=pd.DataFrame(rows)
    q["order"]=q.system.map({s:i for i,s in enumerate(SYSTEM_ORDER)})
    q=q.sort_values("order")
    x=np.arange(3); w=.23
    ax.bar(x-w, q.mean_kappa, width=w, label="Mean κ")
    ax.bar(x, q.mean_lambda, width=w, label="Mean accepted path fraction λ")
    ax.bar(x+w, q.median_rho, width=w, label="Median persistent ratio ρ")
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABEL[s] for s in q.system])
    ax.set_ylim(0,1.05)
    ax.set_ylabel("Fraction / ratio")
    ax.legend(frameon=False, fontsize=6.7)
    _clean_ax(ax)


def panel_3c(ax: plt.Axes, P: dict[str, Path]) -> None:
    df = _read_csv(P["natural_summary"])
    methods=["afm","projection","linearized_distillation","ewc_prox","replay","derpp","unrestricted"]
    systems=SYSTEM_ORDER
    mat=np.full((len(methods),len(systems)),np.nan)
    for i,m in enumerate(methods):
        for j,s in enumerate(systems):
            z=df[(df.method.eq(m))&(df.system.eq(s))]
            if not z.empty: mat[i,j]=float(z.iloc[0].retention_pass_rate)
    im=ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="Blues")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v=mat[i,j]
            txt="—" if not np.isfinite(v) else f"{100*v:.1f}%".replace("100.0%","100%").replace("0.0%","0%")
            ax.text(j,i,txt,ha="center",va="center",fontsize=7,
                    color="white" if np.isfinite(v) and v>.55 else "black")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([SYSTEM_LABEL[s] for s in systems])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in methods])
    ax.set_title(r"Retention feasibility: $D\leq0.005$")
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)


def panel_3d(ax: plt.Axes) -> None:
    """Simplified AFM mechanism: persistent learning is distinct from finite completion."""
    ax.set_axis_off()
    boxes = [
        (.06,.62,.36,.23,"Same-state comparator","unprotected learning opportunity, Δ₀"),
        (.58,.62,.36,.23,"Compatible component","measure κ"),
        (.06,.18,.36,.23,"Persistent assimilation","ordinary network parameters"),
        (.58,.18,.36,.23,"Finite completion","restore protected outputs"),
    ]
    for x,y,w,h,title,sub in boxes:
        box=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.014,rounding_size=0.018",
                           transform=ax.transAxes,ec="0.25",fc="white",lw=1.0)
        ax.add_patch(box)
        ax.text(x+w/2,y+h*.64,title,ha="center",va="center",transform=ax.transAxes,
                fontsize=7.8,fontweight="bold")
        ax.text(x+w/2,y+h*.29,sub,ha="center",va="center",transform=ax.transAxes,
                fontsize=6.5,color="0.35")
    ax.add_patch(FancyArrowPatch((.42,.735),(.58,.735),transform=ax.transAxes,
                                 arrowstyle="-|>",mutation_scale=10,lw=1,color="0.35"))
    ax.add_patch(FancyArrowPatch((.76,.62),(.24,.41),transform=ax.transAxes,
                                 connectionstyle="arc3,rad=.24",arrowstyle="-|>",
                                 mutation_scale=10,lw=1,color="0.35"))
    ax.add_patch(FancyArrowPatch((.42,.295),(.58,.295),transform=ax.transAxes,
                                 arrowstyle="-|>",mutation_scale=10,lw=1,color="0.35"))
    ax.text(.24,.06,"PERSISTENT LEARNING",ha="center",transform=ax.transAxes,
            fontsize=7,fontweight="bold")
    ax.text(.76,.06,"FINITE OUTPUT RESTORATION",ha="center",transform=ax.transAxes,
            fontsize=7,fontweight="bold")


# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------
def panel_4a(ax: plt.Axes, P: dict[str, Path]) -> None:
    natural=_read_csv(P["natural_rows"])
    multi=_read_csv(P["multiscale_aggregate"])
    method="projection" if "projection" in set(multi.method) else "afm"
    m=multi[(multi.method.eq(method)) & np.isclose(multi.retention_beta,.5)]
    rows=[]
    for s in SYSTEM_ORDER:
        nat=natural[(natural.system.eq(s))&(natural.method.eq("unrestricted"))].update_norm.dropna().to_numpy(float)
        med=float(np.median(nat))
        z=m[m.system.eq(s)]
        sf=float(z.scale_fraction.max())
        # Unrestricted update norm can vary slightly with kappa; use the reported mean at largest scale.
        causal=float(z[np.isclose(z.scale_fraction,sf)].mean_unrestricted_update_norm.mean())
        rows.append((s,causal,med,causal/med))
    x=np.arange(3); w=.32
    causal=np.array([r[1] for r in rows]); nat=np.array([r[2] for r in rows])
    ax.bar(x-w/2,causal,width=w,label="Largest multiscale causal update")
    ax.bar(x+w/2,nat,width=w,label="Median natural update")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABEL[r[0]] for r in rows])
    ax.set_ylabel("Parameter-update norm (log scale)")
    ax.legend(frameon=False,fontsize=6.7)
    for i,r in enumerate(rows):
        ax.text(i, max(r[1],r[2])*1.35, f"{100*r[3]:.3g}%",ha="center",va="bottom",fontsize=7)
    _clean_ax(ax)


def panel_4b(ax: plt.Axes, P: dict[str, Path]) -> None:
    df=_read_csv(P["bridge_feasibility"])
    d=df[df.system.isin(["cifar10_cnn","cifar10_vit"])].copy()
    for s in ["cifar10_cnn","cifar10_vit"]:
        g=d[d.system.eq(s)].sort_values("natural_norm_fraction")
        ax.plot(100*g.natural_norm_fraction,100*g.feasibility_rate_corrected,
                marker="o",label=SYSTEM_LABEL[s])
    ax.set_xscale("log")
    ax.set_xticks([1,10,50,100])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("Target update norm (% of median natural update)")
    ax.set_ylabel("States supporting all four positive\nsame-norm κ endpoints (%)")
    ax.set_ylim(-1,50)
    ax.legend(frameon=False)
    ax.text(.02,.03,"Zero = intervention infeasible, not zero causal effect",
            transform=ax.transAxes,fontsize=6.8)
    _clean_ax(ax)


def _read_jsonl(path: Path) -> list[dict]:
    out=[]
    with path.open() as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def _load_bridge_frontier(root: Path) -> pd.DataFrame:
    """Merge original valid + repaired-only frontier JSONLs."""
    rows=[]
    for suite in [
        root/"runs_compatibility_natural_scale_bridge_v1",
        root/"runs_compatibility_natural_scale_bridge_v1_delta0_repair",
    ]:
        matrix=suite/"job_matrix.json"
        if not matrix.is_file():
            continue
        for job in json.loads(matrix.read_text()):
            run=Path(job["run_dir"])
            # Matrix paths are normally absolute. If copied, try basename under suite/runs.
            if not run.exists():
                alt=suite/"runs"/run.name
                if alt.exists(): run=alt
            p=run/"bridge_frontier_points.jsonl"
            if p.is_file(): rows.extend(_read_jsonl(p))
    if not rows:
        raise FileNotFoundError(
            "Could not locate bridge_frontier_points.jsonl via the original/repair job matrices. "
            "Run this script inside the project containing the completed bridge suites."
        )
    df=pd.DataFrame(rows)
    keys=["system","seed","state_id","natural_norm_fraction","method","retention_beta","requested_kappa"]
    if all(k in df.columns for k in keys):
        if df.duplicated(keys).any():
            dup=df[df.duplicated(keys,keep=False)][keys].head()
            raise RuntimeError(f"Duplicate merged bridge keys detected:\n{dup}")
    return df


def _bridge_projection_seed_curve(root: Path) -> pd.DataFrame:
    df=_load_bridge_frontier(root)
    method="projection" if "projection" in set(df.method) else "afm"
    d=df[(df.system.eq("cifar10_vit")) & np.isclose(df.natural_norm_fraction,.01)
         & df.method.eq(method) & np.isclose(df.retention_beta,.5)].copy()
    if d.empty: raise RuntimeError("No ViT 1% projection bridge frontier rows found")
    seed=d.groupby(["seed","requested_kappa"],as_index=False).agg(
        kappa=("measured_kappa","mean"),
        persistent=("persistent_decrease","mean"),
        rho=("persistent_ratio","mean"),
    )
    rows=[]
    for k,g in seed.groupby("requested_kappa"):
        for val in ["persistent","rho"]:
            lo,hi=seed_bootstrap_ci(g[val].to_numpy(float))
            rows.append({"requested_kappa":k,"metric":val,"kappa":g.kappa.mean(),
                         "mean":g[val].mean(),"lo":lo,"hi":hi,"nseed":len(g)})
    return pd.DataFrame(rows)


def panel_4c(fig: plt.Figure, subspec, root: Path, P: dict[str, Path], label: str | None=None) -> None:
    curve=_bridge_projection_seed_curve(root)
    decomp=_read_csv(P["bridge_decomp"])
    r=decomp[(decomp.system.eq("cifar10_vit")) & np.isclose(decomp.natural_norm_fraction,.01)
             & decomp.method.eq("projection") & np.isclose(decomp.retention_beta,.5)]
    gs=GridSpecFromSubplotSpec(1,2,subplot_spec=subspec,wspace=.35,width_ratios=[1.25,1])
    ax=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1])
    p=curve[curve.metric.eq("persistent")].sort_values("kappa")
    ax.errorbar(p.kappa,p["mean"],yerr=[p["mean"]-p.lo,p.hi-p["mean"]],fmt="o-",capsize=2)
    ax.set_xlabel("Realized compatibility, κ")
    ax.set_ylabel(r"Absolute persistent progress, $\Delta_{persistent}$")
    _clean_ax(ax)
    q=curve[curve.metric.eq("rho")].sort_values("kappa")
    ax2.errorbar(q.kappa,q["mean"],yerr=[q["mean"]-q.lo,q.hi-q["mean"]],fmt="o-",capsize=2)
    ax2.set_xlabel("Realized κ")
    ax2.set_ylabel("Persistent ratio, ρ")
    ax2.axhline(0,color="0.55",lw=.7)
    _clean_ax(ax2)
    if not r.empty:
        rr=r.iloc[0]
        ax.text(.03,.97,
                f"κ→Δpersistent slope\n{rr.mean_kappa_persistent_decrease_slope:.3e}\n"
                f"[{rr.kappa_persistent_decrease_slope_ci95_low:.3e}, {rr.kappa_persistent_decrease_slope_ci95_high:.3e}]",
                transform=ax.transAxes,ha="left",va="top",fontsize=6.7,
                bbox=dict(boxstyle="round,pad=.25",fc="white",ec="0.82"))
        ax2.text(.03,.97,
                 f"κ→ρ slope\n{rr.mean_kappa_persistent_ratio_slope:.3f}\n"
                 f"[{rr.kappa_persistent_ratio_slope_ci95_low:.2f}, {rr.kappa_persistent_ratio_slope_ci95_high:.2f}]",
                 transform=ax2.transAxes,ha="left",va="top",fontsize=6.7,
                 bbox=dict(boxstyle="round,pad=.25",fc="white",ec="0.82"))
    if label: _panel_label(ax,label)


def panel_4d(fig: plt.Figure, subspec, P: dict[str, Path], label: str | None=None) -> None:
    df=_read_csv(P["bridge_heterogeneity"])
    d=df[(df.system.eq("cifar10_vit")) & np.isclose(df.natural_norm_fraction,.01)
         & df.method.eq("projection") & np.isclose(df.retention_beta,.5)].copy()
    order=["low","medium","high"]
    d["order"]=d.delta0_cv_band.map({x:i for i,x in enumerate(order)})
    d=d.sort_values("order")
    gs=GridSpecFromSubplotSpec(2,1,subplot_spec=subspec,hspace=.18)
    ax=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[1,0],sharex=ax)
    x=np.arange(3)
    v=d.mean_kappa_persistent_decrease_slope.to_numpy(float)*1e4
    lo=d.kappa_persistent_decrease_slope_ci95_low.to_numpy(float)*1e4
    hi=d.kappa_persistent_decrease_slope_ci95_high.to_numpy(float)*1e4
    ax.errorbar(x,v,yerr=[v-lo,hi-v],fmt="o-",capsize=2)
    ax.set_ylabel(r"κ→Δpersistent slope  ($\times10^{-4}$)")
    ax.tick_params(labelbottom=False)
    _clean_ax(ax)
    vr=d.mean_kappa_persistent_ratio_slope.to_numpy(float)
    lor=d.kappa_persistent_ratio_slope_ci95_low.to_numpy(float)
    hir=d.kappa_persistent_ratio_slope_ci95_high.to_numpy(float)
    ax2.errorbar(x,vr,yerr=[vr-lor,hir-vr],fmt="o-",capsize=2)
    ax2.axhline(0,color="0.55",lw=.7)
    ax2.set_ylabel("κ→ρ slope")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Low Δ₀ CV","Medium Δ₀ CV","High Δ₀ CV"])
    _clean_ax(ax2)
    if label: _panel_label(ax,label)


# -----------------------------------------------------------------------------
# Standalone panel wrappers and assembled figures
# -----------------------------------------------------------------------------
def save_single_axis_panel(func, out: Path, P: dict[str, Path], figsize=(3.35,2.55)):
    fig,ax=plt.subplots(figsize=figsize,constrained_layout=True)
    func(ax,P)
    _save(fig,out)


def save_2a(out: Path, P: dict[str, Path]):
    fig=plt.figure(figsize=(7.0,2.35),constrained_layout=True)
    gs=GridSpec(1,1,figure=fig)
    panel_2a(fig,gs[0,0],P)
    _save(fig,out)


def save_3d(out: Path):
    fig,ax=plt.subplots(figsize=(7.0,2.2),constrained_layout=True)
    panel_3d(ax)
    _save(fig,out)


def save_4c(out: Path, root: Path, P: dict[str, Path]):
    fig=plt.figure(figsize=(6.8,2.6),constrained_layout=True)
    gs=GridSpec(1,1,figure=fig)
    panel_4c(fig,gs[0,0],root,P)
    _save(fig,out)


def save_4d(out: Path, P: dict[str, Path]):
    fig=plt.figure(figsize=(3.6,4.0),constrained_layout=True)
    gs=GridSpec(1,1,figure=fig)
    panel_4d(fig,gs[0,0],P)
    _save(fig,out)


def assembled_figure1(out: Path, P: dict[str, Path]):
    fig=plt.figure(figsize=(10.0,3.0),constrained_layout=True)
    gs=GridSpec(1,3,figure=fig,width_ratios=[1.05,.95,1.25])
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    panel_1b(axes[0],P); _panel_label(axes[0],"b")
    panel_1c(axes[1],P); _panel_label(axes[1],"c")
    panel_1d(axes[2],P); _panel_label(axes[2],"d")
    _save(fig,out)


def assembled_figure2(out: Path, P: dict[str, Path]):
    fig=plt.figure(figsize=(10.0,6.7),constrained_layout=True)
    gs=GridSpec(2,2,figure=fig,width_ratios=[1.35,1],height_ratios=[1,1])
    panel_2a(fig,gs[0,0],P,label="a")
    ax=fig.add_subplot(gs[0,1]); panel_2b(ax,P); _panel_label(ax,"b")
    ax=fig.add_subplot(gs[1,0]); panel_2c(ax,P); _panel_label(ax,"c")
    ax=fig.add_subplot(gs[1,1]); panel_2d(ax,P); _panel_label(ax,"d")
    _save(fig,out)


def assembled_figure3(out: Path, P: dict[str, Path]):
    fig=plt.figure(figsize=(10.0,6.8),constrained_layout=True)
    gs=GridSpec(2,2,figure=fig,width_ratios=[1,1.15])
    ax=fig.add_subplot(gs[0,0]); panel_3a(ax,P); _panel_label(ax,"a")
    ax=fig.add_subplot(gs[0,1]); panel_3b(ax,P); _panel_label(ax,"b")
    ax=fig.add_subplot(gs[1,0]); panel_3c(ax,P); _panel_label(ax,"c")
    ax=fig.add_subplot(gs[1,1]); panel_3d(ax); _panel_label(ax,"d")
    _save(fig,out)


def assembled_figure4(out: Path, root: Path, P: dict[str, Path]):
    fig=plt.figure(figsize=(10.0,7.1),constrained_layout=True)
    gs=GridSpec(2,2,figure=fig,width_ratios=[1,1.25])
    ax=fig.add_subplot(gs[0,0]); panel_4a(ax,P); _panel_label(ax,"a")
    ax=fig.add_subplot(gs[0,1]); panel_4b(ax,P); _panel_label(ax,"b")
    panel_4c(fig,gs[1,0],root,P,label="c")
    panel_4d(fig,gs[1,1],P,label="d")
    _save(fig,out)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".",help="Project root (default: current directory)")
    ap.add_argument("--out",default="paper_figures",help="Output directory")
    args=ap.parse_args()
    root=Path(args.root).resolve()
    out=(root/args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    panels=out/"panels"; panels.mkdir(parents=True,exist_ok=True)

    print("Locating existing experiment outputs...")
    P=paths(root)
    for k,v in P.items(): print(f"  {k:24s} {v.relative_to(root) if root in v.parents else v}")

    print("\nGenerating individual vector-PDF panels...")
    single=[
        ("Fig1b.pdf",panel_1b,(3.5,2.7)),
        ("Fig1c.pdf",panel_1c,(3.3,2.7)),
        ("Fig1d.pdf",panel_1d,(4.6,3.0)),
        ("Fig2b.pdf",panel_2b,(3.4,2.7)),
        ("Fig2c.pdf",panel_2c,(3.6,2.7)),
        ("Fig2d.pdf",panel_2d,(3.8,2.8)),
        ("Fig3a.pdf",panel_3a,(3.5,2.7)),
        ("Fig3b.pdf",panel_3b,(4.0,2.8)),
        ("Fig3c.pdf",panel_3c,(4.0,3.1)),
        ("Fig4a.pdf",panel_4a,(3.6,2.8)),
        ("Fig4b.pdf",panel_4b,(3.6,2.8)),
    ]
    for name,fn,size in single:
        save_single_axis_panel(fn,panels/name,P,size)
        print("  wrote",panels/name)
    save_2a(panels/"Fig2a.pdf",P); print("  wrote",panels/"Fig2a.pdf")
    save_3d(panels/"Fig3d.pdf"); print("  wrote",panels/"Fig3d.pdf")
    save_4c(panels/"Fig4c.pdf",root,P); print("  wrote",panels/"Fig4c.pdf")
    save_4d(panels/"Fig4d.pdf",P); print("  wrote",panels/"Fig4d.pdf")

    print("\nAssembling figures...")
    assembled_figure1(out/"Figure1_bcd.pdf",P)
    assembled_figure2(out/"Figure2_abcd.pdf",P)
    assembled_figure3(out/"Figure3_abcd.pdf",P)
    assembled_figure4(out/"Figure4_abcd.pdf",root,P)
    for p in [out/"Figure1_bcd.pdf",out/"Figure2_abcd.pdf",out/"Figure3_abcd.pdf",out/"Figure4_abcd.pdf"]:
        print("  wrote",p)

    print("\nDone. All outputs are vector PDFs. Fig. 1a is intentionally not generated.")


if __name__ == "__main__":
    main()
