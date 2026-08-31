#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from afmvision.eval.routing import analyse_routing, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise AFM theorem quantities from a completed run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    events = [
        json.loads(line)
        for line in (args.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [e for e in events if e.get("event") == "afm_step"]
    if not steps:
        raise RuntimeError("No afm_step events found")

    policy_count = max((len(e.get("all_policy_frontier_costs", [])) for e in steps), default=0)
    policy_cumulative = [0.0] * policy_count
    selected_cost = 0.0
    selected_bounded = 0.0
    for event in steps:
        costs = [float(x) for x in event.get("all_policy_frontier_costs", [])]
        bounded = [float(x) for x in event.get("all_policy_bounded_losses", [])]
        for i, value in enumerate(costs):
            policy_cumulative[i] += value
        selected = int(event.get("selected_policy", 0))
        if selected < len(costs):
            selected_cost += costs[selected]
        if selected < len(bounded):
            selected_bounded += bounded[selected]

    record_max_drift: dict[int, float] = defaultdict(float)
    record_max_snapshot_drift: dict[int, float] = defaultdict(float)
    record_activation_gap: dict[int, float] = defaultdict(float)
    record_last_budget: dict[int, float] = defaultdict(float)
    record_last_snapshot_bound: dict[int, float] = defaultdict(float)
    for event in events:
        if event.get("event") == "record_retention":
            rid = int(event["record_id"])
            record_max_drift[rid] = max(record_max_drift[rid], float(event["anchor_drift"]))
            record_max_snapshot_drift[rid] = max(
                record_max_snapshot_drift[rid], float(event.get("snapshot_drift", event["anchor_drift"]))
            )
            record_activation_gap[rid] = float(event.get("activation_gap", 0.0))
            record_last_budget[rid] = float(event["cumulative_budget"])
            record_last_snapshot_bound[rid] = float(
                event.get("snapshot_bound", record_activation_gap[rid] + record_last_budget[rid])
            )

    commit_events = [event for event in events if event.get("event") == "record_committed"]
    activation_transfer = {
        str(int(event["record_id"])): {
            "slot": int(event["slot"]),
            "step": int(event["step"]),
            "validation_count": int(event.get("validation_count", 0)),
            "validation_error": float(event.get("validation_mean", 0.0)),
            "validation_accuracy": float(
                event.get("validation_accuracy", 1.0 - float(event.get("validation_mean", 0.0)))
            ),
            "anchor_evidence_error": (
                None if event.get("anchor_evidence_error") is None else float(event["anchor_evidence_error"])
            ),
            "activation_error": float(event.get("activation_error", 0.0)),
            "activation_anchor_gap": float(event.get("activation_anchor_gap", 0.0)),
            "activation_prediction_disagreement": (
                None
                if event.get("activation_prediction_disagreement") is None
                else float(event["activation_prediction_disagreement"])
            ),
        }
        for event in commit_events
    }

    routing_report = None
    if any(event.get("event") == "routing_assignment" for event in events):
        resolved_config_path = args.run_dir / "resolved_config.json"
        if resolved_config_path.exists():
            resolved = json.loads(resolved_config_path.read_text(encoding="utf-8"))
            configured_sidecar_raw = str(resolved.get("data", {}).get("evaluator_sidecar", "")).strip()
            train_manifest_raw = str(resolved.get("data", {}).get("train_manifest", "")).strip()
            sidecar_candidates: list[Path] = []
            if configured_sidecar_raw:
                sidecar_candidates.append(Path(configured_sidecar_raw))
            if train_manifest_raw:
                sidecar_candidates.append(Path(train_manifest_raw).parent / "evaluator_sidecar.jsonl")
            sidecar_path = next((path for path in sidecar_candidates if path.is_file()), None)
            if sidecar_path is not None:
                routing_report = analyse_routing(events, read_jsonl(sidecar_path))

    report = {
        "rounds": len(steps),
        "nonzero_accepted_steps": sum(
            bool(e.get("accepted")) and float(e.get("realised_step_length", 0.0)) > 0.0 for e in steps
        ),
        "mean_projected_gradient_norm": sum(float(e.get("projected_gradient_norm", 0.0)) for e in steps) / len(steps),
        "mean_safe_radius": sum(float(e.get("safe_radius", 0.0)) for e in steps) / len(steps),
        "mean_epsilon": sum(float(e.get("epsilon", 0.0)) for e in steps) / len(steps),
        "mean_spectral_residual": sum(float(e.get("spectral_residual", 0.0)) for e in steps) / len(steps),
        "mean_blocked_fraction": sum(float(e.get("blocked_fraction", 0.0)) for e in steps) / len(steps),
        "selected_frontier_cost": selected_cost,
        "best_fixed_policy_frontier_cost": min(policy_cumulative) if policy_cumulative else 0.0,
        "empirical_policy_regret": selected_cost - min(policy_cumulative) if policy_cumulative else 0.0,
        "selected_bounded_loss": selected_bounded,
        "policy_cumulative_frontier_costs": policy_cumulative,
        "record_retention": {
            str(rid): {
                "max_active_interval_drift": record_max_drift[rid],
                "activation_gap": record_activation_gap[rid],
                "cumulative_budget": record_last_budget[rid],
                "max_validated_snapshot_drift": record_max_snapshot_drift[rid],
                "validated_snapshot_bound": record_last_snapshot_bound[rid],
                "within_budget": record_max_drift[rid] <= record_last_budget[rid] + 1e-5,
                "within_validated_snapshot_bound": (
                    record_max_snapshot_drift[rid] <= record_last_snapshot_bound[rid] + 1e-5
                ),
            }
            for rid in sorted(record_max_drift)
        },
        "reopening_score_batches": sum(e.get("event") == "reopening_score" for e in events),
        "reopening_routed_outcomes": sum(
            int(e.get("processed_observations", e.get("observations", 1)))
            for e in events
            if e.get("event") == "reopening_score"
        ),
        "reopenings": sum(e.get("event") == "record_reopened" for e in events),
        "renewal_trials": sum(e.get("event") == "renewal_trial" for e in events),
        "renewal_activations": sum(e.get("event") == "renewal_trial" and bool(e.get("activated")) for e in events),
        "renewal_capacity_obstructions": sum(e.get("event") == "renewal_obstruction" for e in events),
        "capacity_releases": sum(
            e.get("event") == "record_released" and str(e.get("reason", "")).startswith("capacity") for e in events
        ),
        "commits": sum(e.get("event") == "record_committed" for e in events),
        "protected_nonzero_steps": sum(
            bool(e.get("accepted"))
            and float(e.get("realised_step_length", 0.0)) > 0.0
            and int(e.get("active_records", 0)) > 0
            for e in steps
        ),
        "method_activated": any(e.get("event") == "record_committed" for e in events)
        and any(
            bool(e.get("accepted"))
            and float(e.get("realised_step_length", 0.0)) > 0.0
            and int(e.get("active_records", 0)) > 0
            for e in steps
        ),
        "candidate_creations": sum(e.get("event") == "candidate_created" for e in events),
        "candidate_replacements": sum(e.get("event") == "candidate_replaced" for e in events),
        "candidate_pretest_replacements": sum(
            e.get("event") == "candidate_replaced" and not bool(e.get("old_test_started", False)) for e in events
        ),
        "candidate_tests_started": sum(e.get("event") == "candidate_frozen" for e in events),
        "candidate_training_runs": sum(e.get("event") == "candidate_frozen" for e in events),
        "candidate_validation_rejections": sum(e.get("event") == "candidate_rejected" for e in events),
        "candidate_training_accuracy": [
            float(e["training_accuracy"])
            for e in events
            if e.get("event") == "candidate_frozen" and e.get("training_accuracy") is not None
        ],
        "candidate_training_loss": [
            float(e["training_loss"])
            for e in events
            if e.get("event") == "candidate_frozen" and e.get("training_loss") is not None
        ],
        "candidate_validation_reports": sum(e.get("event") == "candidate_validation" for e in events),
        "candidate_transfer": {
            "attempts": sum(e.get("event") == "afm_step" and bool(e.get("transfer_attempted")) for e in events),
            "common_descent_steps": sum(e.get("event") == "afm_step" and bool(e.get("transfer_common_descent")) for e in events),
            "accepted_joint_steps": sum(e.get("event") == "afm_step" and bool(e.get("transfer_joint_step")) for e in events),
            "fallback_steps": sum(
                e.get("event") == "afm_step"
                and bool(e.get("transfer_attempted"))
                and not bool(e.get("transfer_joint_step"))
                for e in events
            ),
            "mean_exact_loss_reduction": (
                sum(
                    float(e["transfer_loss_before"]) - float(e["transfer_loss_after"])
                    for e in events
                    if e.get("event") == "afm_step"
                    and bool(e.get("transfer_joint_step"))
                    and e.get("transfer_loss_before") is not None
                    and e.get("transfer_loss_after") is not None
                )
                / max(1, sum(e.get("event") == "afm_step" and bool(e.get("transfer_joint_step")) for e in events))
            ),
        },
        "route_splits": [
            {
                "step": int(e["step"]),
                "record_id": int(e["record_id"]),
                "source_slot": int(e["source_slot"]),
                "target_slot": int(e["target_slot"]),
                "signature_distance": float(e.get("signature_distance", 0.0)),
                "crossing_observation": int(e.get("crossing_observation", 0)),
            }
            for e in events if e.get("event") == "record_route_split"
        ],
        "context_signature": next(
            (e for e in events if e.get("event") == "context_signature_frozen"), None
        ),
        "activation_transfer": activation_transfer,
        "routing_analysis": routing_report,
        "all_nonzero_steps_theorem_certified": bool(
            [e for e in steps if float(e.get("realised_step_length", 0.0)) > 0.0]
        ) and all(
            bool(e.get("theorem_certified"))
            for e in steps
            if float(e.get("realised_step_length", 0.0)) > 0.0
        ),
    }
    output = args.output or (args.run_dir / "afm_analysis.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
