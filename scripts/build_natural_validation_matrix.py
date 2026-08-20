from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_SYSTEMS = ["cifar10_cnn", "cifar10_vit", "text_transformer"]
EXPECTED_SEEDS = [11, 29, 47, 71, 101]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite-root", default="runs_compatibility_causal_v1")
    parser.add_argument("--output-root", default="runs_compatibility_natural_v1")
    parser.add_argument("--states", type=int, default=50)
    parser.add_argument("--probe-interval", type=int, default=10)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    source = (root / args.source_suite_root).resolve()
    out = (root / args.output_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing natural validation suite: {out}")
    source_matrix = source / "job_matrix.json"
    if not source_matrix.is_file():
        raise FileNotFoundError(f"missing source compatibility matrix: {source_matrix}")
    rows = json.loads(source_matrix.read_text(encoding="utf-8"))
    if len(rows) != 15:
        raise RuntimeError(f"expected 15 source compatibility jobs, found {len(rows)}")

    (out / "runs").mkdir(parents=True, exist_ok=True)
    matrix = []
    seen = set()
    for source_row in rows:
        system = str(source_row["system"])
        seed = int(source_row["seed"])
        if system not in EXPECTED_SYSTEMS or seed not in EXPECTED_SEEDS:
            raise RuntimeError(f"unexpected source row: {system} seed {seed}")
        key = (system, seed)
        if key in seen:
            raise RuntimeError(f"duplicate source row: {key}")
        seen.add(key)
        source_run = Path(source_row["run_dir"])
        if not source_run.is_absolute():
            source_run = (root / source_run).resolve()
        checkpoint = source_run / "preprobe_parent.pt"
        resolved = source_run / "resolved_config.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing preprobe parent for {key}: {checkpoint}")
        if not resolved.is_file():
            raise FileNotFoundError(f"missing resolved config for {key}: {resolved}")
        name = f"{system}_seed{seed}"
        matrix.append(
            {
                "index": len(matrix),
                "system": system,
                "seed": seed,
                "source_run_dir": str(source_run),
                "source_preprobe_parent": str(checkpoint),
                "run_dir": str((out / "runs" / name).resolve()),
                "states": int(args.states),
                "probe_interval": int(args.probe_interval),
                "probe_steps": int(args.states) * int(args.probe_interval),
            }
        )

    expected = {(s, seed) for s in EXPECTED_SYSTEMS for seed in EXPECTED_SEEDS}
    if seen != expected:
        raise RuntimeError(f"source matrix does not contain the exact 3x5 system/seed grid: missing={sorted(expected-seen)}")
    (out / "job_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    (out / "sampling_rule.json").write_text(
        json.dumps(
            {
                "schema": "natural_compatibility_validation_v1",
                "source_suite_root": str(source),
                "states_per_system_seed": int(args.states),
                "probe_interval_parent_steps": int(args.probe_interval),
                "probe_steps": int(args.states) * int(args.probe_interval),
                "state_rule": "sample pre-update states at local steps 0, interval, ..., (states-1)*interval; never filter by kappa or outcome",
                "current_batch_rule": "entire ordinary parent-stream minibatch with true labels",
                "branch_commit_rule": "no validation method branch is committed; only the ordinary supervised parent update is committed",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"WROTE: {out / 'job_matrix.json'}")
    print(f"jobs: {len(matrix)}")
    print(f"expected method outcomes: {len(matrix) * int(args.states) * 7}")


if __name__ == "__main__":
    main()
