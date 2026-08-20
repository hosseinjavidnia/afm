from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs_compatibility_causal_v1/analysis/central_figure_points.csv")
    parser.add_argument("--output", default="runs_compatibility_causal_v1/analysis/compatibility_collapse.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required only for plotting; the CSV analysis is already complete") from exc
    rows = list(csv.DictReader(Path(args.input).open(encoding="utf-8")))
    groups = defaultdict(list)
    for r in rows:
        if r["retention_pass"].lower() != "true":
            continue
        groups[r["method"]].append((float(r["measured_kappa"]), float(r["persistent_ratio"])))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method, pts in sorted(groups.items()):
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=8, alpha=0.22, label=method)
    ax.set_xlabel(r"Measured functional compatibility $\\kappa$")
    ax.set_ylabel(r"Persistent progress ratio $\\Delta_{persistent}/\\Delta_0$")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
