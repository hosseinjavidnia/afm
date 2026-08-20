#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _advance_reopening_epoch_count(counts: dict[int, int], event: dict) -> None:
    """Track observations in the currently active predictable reopening test.

    A preserved route split restarts the challenger and e-process.  Persistent
    checkpoint state therefore contains the count for the latest test epoch,
    not the lifetime total across superseded tests.
    """

    kind = event.get("event")
    if kind == "reopening_score":
        rid = int(event["record_id"])
        counts[rid] = counts.get(rid, 0) + int(
            event.get("processed_observations", event.get("observations", 1))
        )
    elif kind == "reopening_test_restarted":
        counts[int(event["record_id"])] = 0



def _advance_signature_epoch_count(counts: dict[int, int], event: dict) -> None:
    kind = event.get("event")
    if kind == "signature_shift_score":
        counts[int(event["record_id"])] = int(event.get("cumulative_blocks", 0))
    elif kind == "reopening_test_restarted":
        counts[int(event["record_id"])] = 0


def _exact_restoration_executable_endpoint_tolerance(cfg: dict, certificate_mode: str) -> float:
    """Reproduce the trainer's declared base-plus-shield arithmetic envelope.

    ``functional_shield.residual_tolerance`` is the compact-cardinal solve
    tolerance.  It is not, by itself, the tolerance for recomposing float32
    base logits with the shield.  The validity checker must use the same
    executable endpoint envelope that the trainer used when deciding whether
    to accept the restoration transaction.
    """

    afm_cfg = cfg.get("afm", {})
    shield_cfg = afm_cfg.get("functional_shield", {})
    numerics_cfg = afm_cfg.get("numerics", {})
    residual_tolerance = float(shield_cfg.get("residual_tolerance", 0.0))
    if str(certificate_mode) == "strict":
        return (
            residual_tolerance
            + float(shield_cfg.get("feature_distance_error_bound") or 0.0)
            + 2.0 * float(numerics_cfg.get("endpoint_error_bound") or 0.0)
        )
    return max(
        residual_tolerance,
        float(numerics_cfg.get("arithmetic_error_bound", 0.0)),
    )



def _ratio_tolerance_from_loss_tolerance(loss_tolerance: float, counterfactual_decrease: float) -> float:
    """Translate an additive loss-space numerical envelope into ratio space."""
    tol = max(float(loss_tolerance), 0.0)
    delta0 = float(counterfactual_decrease)
    if math.isfinite(delta0) and delta0 > 0.0:
        return tol / delta0
    return tol


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if an AFM run violates implementation-level invariants")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--zero-gate-tolerance", type=float, default=1e-6)
    parser.add_argument("--drift-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = json.loads((args.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (args.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    failures: list[str] = []

    nonzero_steps = [
        e for e in events
        if e.get("event") == "afm_step"
        and bool(e.get("accepted"))
        and (
            float(e.get("realised_step_length", 0.0)) > 0.0
            or float(e.get("functional_shield_update_norm", 0.0)) > 0.0
        )
    ]
    if not nonzero_steps:
        failures.append("No nonzero AFM step was accepted")
    protected_nonzero_steps = [e for e in nonzero_steps if int(e.get("active_records", 0)) > 0]
    requirements = dict(cfg.get("afm", {}).get("run_requirements", {}))
    min_commits = int(requirements.get("min_commits", 0))
    min_protected_nonzero = int(requirements.get("min_protected_nonzero_steps", 0))
    if summary.get("nonzero_accepted_steps", 0) != len(nonzero_steps):
        failures.append("Nonzero accepted-step summary does not match event log")
    if summary.get("max_empirical_drift_violation", 0.0) > args.drift_tolerance:
        failures.append("Empirical behaviour drift exceeded its per-step budget")
    if summary.get("max_snapshot_drift_violation", 0.0) > args.drift_tolerance:
        failures.append("A committed snapshot drift exceeded activation gap plus active budget")
    if float(summary.get("total_error_budget_allocated", 0.0)) > float(summary.get("total_delta", 0.0)) + 1e-12:
        failures.append("Summable statistical allocations exceeded total delta")

    last_log_wealth_by_record: dict[int, tuple[float, float]] = {}
    committed_step: dict[int, int] = {}
    commit_ids: list[int] = []
    released_ids: set[int] = set()
    reopening_processed: dict[int, int] = {}
    signature_processed: dict[int, int] = {}
    last_signature_log_wealth_by_record: dict[int, tuple[float, float]] = {}
    afm_event_index_by_step: dict[int, int] = {}
    activated_slots: set[int] = set()
    frozen_consolidation_indices: list[int] = []
    frozen_candidates: list[dict] = []
    rejected_candidates: list[dict] = []
    signature_events: list[dict] = []
    routing_assignments: list[dict] = []
    route_split_events: list[dict] = []
    transfer_joint_steps: list[dict] = []
    transfer_attempt_steps: list[dict] = []
    candidate_ledger_events: list[dict] = []
    candidate_invariant_violations: list[dict] = []
    max_transfer_gap = float(cfg.get("afm", {}).get("transfer", {}).get("max_activation_gap", float("inf")))
    for event_index, event in enumerate(events):
        _advance_reopening_epoch_count(reopening_processed, event)
        _advance_signature_epoch_count(signature_processed, event)
        kind = event.get("event")
        if kind == "record_committed":
            rid = int(event["record_id"])
            committed_step[rid] = int(event["step"])
            commit_ids.append(rid)
            activation_gap = float(event.get("activation_anchor_gap", float("inf")))
            if activation_gap > max_transfer_gap + args.drift_tolerance:
                failures.append(
                    f"Record {rid} committed with activation gap {activation_gap} above "
                    f"the predeclared limit {max_transfer_gap}"
                )
        elif kind == "candidate_frozen":
            frozen_candidates.append(event)
            if event.get("commit_budget_index") is None or event.get("commit_alpha") is None:
                failures.append("A frozen candidate omitted its predeclared consolidation budget")
            else:
                frozen_consolidation_indices.append(int(event["commit_budget_index"]))
            training_samples = int(cfg["afm"]["consolidation"].get("training_samples", 0))
            if int(event.get("training_count", -1)) < training_samples:
                failures.append("A candidate was frozen before its declared routed training block was complete")
            if event.get("training_loss") is None or event.get("training_accuracy") is None:
                failures.append("A frozen candidate omitted its private fitting diagnostics")
        elif kind == "candidate_rejected":
            rejected_candidates.append(event)
            if event.get("commit_budget_index") is None or event.get("commit_alpha") is None:
                failures.append("A candidate was validation-rejected before a test existed")
        elif kind == "candidate_created":
            if event.get("commit_budget_index") is not None or event.get("commit_alpha") is not None:
                failures.append("A provisional candidate spent consolidation budget before validation existed")
        elif kind == "context_signature_frozen":
            signature_events.append(event)
        elif kind == "routing_assignment":
            routing_assignments.append(event)
            forbidden_route_fields = {
                "label", "original_label", "session", "context_id", "episode",
                "episode_name", "semantic_regime", "intervention", "object_id",
            }
            leaked_route_fields = sorted(forbidden_route_fields.intersection(event))
            if leaked_route_fields:
                failures.append(
                    "Evaluator metadata leaked into learner routing transcript: "
                    f"{leaked_route_fields}"
                )
            if event.get("sample_id") is None:
                failures.append("A routing assignment omitted its immutable sample_id")
        elif kind == "reopening_score":
            rid = int(event["record_id"])
            if int(event["step"]) <= committed_step.get(rid, -1):
                failures.append(f"Record {rid} used its commit outcome as reopening evidence")
            if not bool(event.get("predictions_fixed_before_safe_update", False)):
                failures.append(f"Record {rid} reopening predictions were not fixed before the safe update")
            last_log_wealth_by_record[rid] = (float(event.get("log_wealth", float("-inf"))), float(event["alpha"]))
        elif kind == "signature_shift_score":
            rid = int(event["record_id"])
            if not bool(event.get("block_resolved", False)):
                failures.append(f"Record {rid} signature evidence was not block-resolved")
            last_signature_log_wealth_by_record[rid] = (
                float(event.get("log_wealth", float("-inf"))),
                float(event.get("alpha", 1.0)),
            )
        elif kind == "record_released":
            released_ids.add(int(event["record_id"]))
        elif kind == "record_reopened":
            rid = int(event["record_id"])
            log_wealth, alpha = last_log_wealth_by_record.get(rid, (float("-inf"), 1.0))
            if log_wealth + 1e-12 < -__import__("math").log(alpha):
                failures.append(f"Record {rid} reopened before its allocated e-process threshold")
            if afm_event_index_by_step.get(int(event["step"]), -1) >= event_index or int(event["step"]) not in afm_event_index_by_step:
                failures.append(f"Record {rid} reopened before the same round's protected AFM update")
        elif kind == "record_route_split":
            route_split_events.append(event)
            rid = int(event["record_id"])
            outcome_log = float(event.get("outcome_log_wealth", float("-inf")))
            outcome_alpha = float(event.get("outcome_alpha", 1.0))
            signature_log = float(event.get("signature_log_wealth", float("-inf")))
            signature_alpha = float(event.get("signature_alpha", 1.0))
            if outcome_log + 1e-12 < -__import__("math").log(outcome_alpha):
                failures.append(f"Record {rid} split before its outcome-reopening threshold")
            if signature_log + 1e-12 < -__import__("math").log(signature_alpha):
                failures.append(f"Record {rid} split before its independent signature-shift threshold")
            if not bool(event.get("dual_evidence_required", False)):
                failures.append(f"Record {rid} split without declaring dual evidence")
            if afm_event_index_by_step.get(int(event["step"]), -1) >= event_index or int(event["step"]) not in afm_event_index_by_step:
                failures.append(f"Record {rid} caused a route split before the same round's protected AFM update")
            if int(event.get("source_slot", -1)) == int(event.get("target_slot", -1)):
                failures.append(f"Record {rid} route split reused the source slot")
        elif kind == "renewal_reset":
            if float(event["zero_change"]) > args.zero_gate_tolerance:
                failures.append(f"Renewal slot {event['slot']} changed the realised function")
        elif kind == "renewal_trial" and bool(event.get("activated")):
            if not bool(event.get("accepted")) or float(event.get("trial_motion", 0.0)) <= 0.0:
                failures.append(f"Renewal slot {event['slot']} activated without an accepted nonzero activation")
            activated_slots.add(int(event["slot"]))
        elif kind == "candidate_transfer_ledger":
            candidate_ledger_events.append(event)
            envelope = float(event.get("numerical_envelope", 0.0))
            if float(event.get("realised_progress", 0.0)) < -envelope:
                failures.append(
                    "A deployed v0.11.0 update increased a frozen certified candidate objective "
                    "beyond the declared numerical envelope"
                )
            if not bool(event.get("certified_nonincrease", False)):
                failures.append("A candidate ledger omitted certified nonincrease")
        elif kind == "candidate_progress_invariant_violation":
            candidate_invariant_violations.append(event)
            failures.append("A candidate-progress invariant violation was logged")
        elif kind == "record_retention":
            if not bool(event.get("within_cumulative_budget", False)):
                failures.append(f"Record {event['record_id']} exceeded its cumulative active-interval budget")
            if not bool(event.get("within_snapshot_bound", False)):
                failures.append(f"Record {event['record_id']} exceeded activation gap plus cumulative budget")
        elif kind == "afm_step":
            afm_event_index_by_step[int(event["step"])] = event_index
            losses = event.get("all_policy_bounded_losses", [])
            if any(float(x) < -1e-12 or float(x) > 1.0 + 1e-12 for x in losses):
                failures.append("A Hedge loss fell outside [0,1]")
            if bool(event.get("exact_counterfactual_restoration_attempted", False)):
                if bool(event.get("accepted", False)):
                    ratio = event.get("exact_counterfactual_progress_ratio")
                    rho = float(cfg.get("afm", {}).get("transfer", {}).get("min_progress_fraction", 0.25))
                    if ratio is None or not math.isfinite(float(ratio)) or float(ratio) + 1e-12 < rho:
                        failures.append("An accepted exact-restoration step violated the genuine no-protection progress ratio")
                    endpoint_error = float(event.get("exact_counterfactual_endpoint_error", float("inf")))
                    endpoint_limit = _exact_restoration_executable_endpoint_tolerance(
                        cfg, str(event.get("certificate_mode", "empirical"))
                    )
                    if endpoint_error > endpoint_limit:
                        failures.append(
                            "An accepted exact-restoration step exceeded its executable endpoint envelope"
                        )
                    if event.get("joint_solver_mode") != "joint_counterfactual_normalized_assimilation":
                        failures.append("An accepted exact-restoration step reported the wrong solver mode")
                    if event.get("persistent_base_mode") != "counterfactual_normalized_metaplastic_endpoint":
                        failures.append("An accepted exact-restoration step did not commit the metaplastic safe base endpoint")
                    safe_radius = float(event.get("safe_base_radius", float("nan")))
                    safe_step = float(event.get("safe_base_step_length", float("nan")))
                    safe_drift = float(event.get("safe_base_drift", float("inf")))
                    budget = float(event.get("behaviour_budget", 0.0))
                    tolerance = float(cfg.get("afm", {}).get("safe_update", {}).get("check_tolerance", 0.0))
                    if not math.isfinite(safe_radius) or not math.isfinite(safe_step) or safe_step <= 0.0:
                        failures.append("An accepted exact-restoration step lacked a nonzero certified safe-base endpoint")
                    if safe_step > safe_radius + tolerance:
                        failures.append("An accepted exact-restoration safe-base step exceeded its certified radius")
                    if safe_drift > budget + tolerance:
                        failures.append("An accepted exact-restoration safe-base step exceeded its behavioural budget")
                    if event.get("retention_budget_mode") != "counterfactual_normalized":
                        failures.append("An accepted v0.11.0 step did not use counterfactual-normalized retention budgeting")
                    charge_fraction = float(event.get("requested_counterfactual_charge_fraction", float("nan")))
                    selected_fraction = float(event.get("selected_counterfactual_path_fraction", float("nan")))
                    realised_fraction = float(event.get("realised_counterfactual_path_fraction", float("nan")))
                    reference_charge = float(event.get("reference_retention_charge", float("nan")))
                    if not math.isfinite(charge_fraction) or not 0.0 <= charge_fraction <= 1.0:
                        failures.append("An accepted v0.11.0 step reported an invalid counterfactual charge fraction")
                    if not math.isfinite(selected_fraction) or not 0.0 <= selected_fraction <= 1.0 + tolerance:
                        failures.append("An accepted v0.11.0 step reported an invalid selected comparator-path fraction")
                    if not math.isfinite(realised_fraction) or not 0.0 <= realised_fraction <= 1.0 + tolerance:
                        failures.append("An accepted v0.11.0 step reported an invalid realised comparator-path fraction")
                    if int(event.get("active_records", 0)) > 0:
                        if selected_fraction + tolerance < charge_fraction:
                            failures.append("An accepted v0.11.0 step selected less than its declared comparator-path fraction")
                        if realised_fraction + tolerance < charge_fraction:
                            failures.append("An accepted v0.11.0 step committed less than its declared comparator-path fraction")
                        expected_budget = charge_fraction * reference_charge
                        scale = max(1.0, abs(expected_budget), abs(budget))
                        if not math.isfinite(reference_charge) or abs(budget - expected_budget) > tolerance * scale:
                            failures.append("An accepted v0.11.0 step used a budget inconsistent with its reference charge")
                    persistent_decrease = float(event.get("safe_base_decrease", float("nan")))
                    persistent_lower = float(event.get("persistent_descent_lower_bound", float("nan")))
                    persistent_ratio = float(event.get("persistent_base_progress_ratio", float("nan")))
                    certified_ratio = float(event.get("certified_persistent_progress_ratio_lower_bound", float("nan")))
                    if not math.isfinite(persistent_decrease) or persistent_decrease <= 0.0:
                        failures.append("An accepted v0.11.0 step lacked positive persistent-base progress")
                    if not math.isfinite(persistent_lower) or persistent_decrease + tolerance < persistent_lower:
                        failures.append("An accepted v0.11.0 step violated its persistent descent lower bound")
                    if not math.isfinite(persistent_ratio) or persistent_ratio <= 0.0:
                        failures.append("An accepted v0.11.0 step omitted a positive persistent progress ratio")
                    counterfactual_decrease = float(
                        event.get("exact_counterfactual_decrease", float("nan"))
                    )
                    # The transaction certifies the descent inequality in loss
                    # space with an additive endpoint tolerance.  Dividing the
                    # decrease and lower bound by Delta^0 must divide that same
                    # numerical envelope by Delta^0 as well.  Using the raw loss
                    # tolerance directly in ratio space is spuriously stricter
                    # whenever 0 < Delta^0 < 1.
                    ratio_tolerance = _ratio_tolerance_from_loss_tolerance(
                        tolerance, counterfactual_decrease
                    )
                    if not math.isfinite(certified_ratio) or persistent_ratio + ratio_tolerance < certified_ratio:
                        failures.append("An accepted v0.11.0 step violated its certified persistent progress-ratio lower bound")
                    alignment_error = float(event.get("projected_counterfactual_alignment_error", float("inf")))
                    ordinary_alignment_error = float(event.get("ordinary_counterfactual_alignment_error", float("inf")))
                    idempotence_error = float(event.get("projection_idempotence_error", float("inf")))
                    compatibility = float(event.get("compatible_gradient_fraction", float("nan")))
                    ordinary_step_size = float(event.get("ordinary_counterfactual_step_size", float("nan")))
                    smoothness_product = float(
                        event.get("counterfactual_step_size_smoothness_product", float("nan"))
                    )
                    scalar_certified = bool(event.get("scalar_counterfactual_certified", False))
                    analytic_ratio = float(event.get("analytic_persistent_progress_ratio_lower_bound", float("nan")))
                    if (
                        alignment_error > tolerance
                        or ordinary_alignment_error > tolerance
                        or idempotence_error > tolerance
                    ):
                        failures.append("An accepted v0.11.0 step failed its scalar-comparator projection audit")
                    if not scalar_certified:
                        failures.append("An accepted v0.11.0 step lacked a certified scalar-gradient comparator")
                    if not math.isfinite(ordinary_step_size) or ordinary_step_size <= 0.0:
                        failures.append("An accepted v0.11.0 step reported an invalid ordinary comparator step size")
                    if (
                        not math.isfinite(smoothness_product)
                        or smoothness_product < -tolerance
                        or smoothness_product > 1.0 + tolerance
                    ):
                        failures.append("An accepted v0.11.0 step violated the scalar comparator smoothness range")
                    if not math.isfinite(compatibility) or not 0.0 <= compatibility <= 1.0 + tolerance:
                        failures.append("An accepted v0.11.0 step reported an invalid compatible-gradient fraction")
                    expected_analytic_ratio = (
                        realised_fraction
                        * compatibility
                        * max(1.0 - 0.5 * smoothness_product * realised_fraction, 0.0)
                        / (1.0 + 0.5 * smoothness_product)
                        if math.isfinite(realised_fraction)
                        and math.isfinite(compatibility)
                        and math.isfinite(smoothness_product)
                        else float("nan")
                    )
                    analytic_scale = max(1.0, abs(expected_analytic_ratio), abs(analytic_ratio))
                    if (
                        not math.isfinite(analytic_ratio)
                        or analytic_ratio < -tolerance
                        or not math.isfinite(expected_analytic_ratio)
                        or abs(analytic_ratio - expected_analytic_ratio) > tolerance * analytic_scale
                        or persistent_ratio + tolerance < analytic_ratio
                    ):
                        failures.append("An accepted v0.11.0 step violated its analytic persistent-progress ratio bound")
                    if event.get("functional_shield_obstruction") is not None:
                        failures.append("An accepted exact-restoration step carried an obstruction")
            if bool(event.get("transfer_attempted", False)):
                transfer_attempt_steps.append(event)
            if bool(event.get("transfer_joint_step", False)):
                transfer_joint_steps.append(event)
                if not bool(event.get("accepted", False)) or (
                    float(event.get("realised_step_length", 0.0)) <= 0.0
                    and float(event.get("functional_shield_update_norm", 0.0)) <= 0.0
                ):
                    failures.append("A candidate transfer was reported without an accepted functional update")
                before = event.get("transfer_loss_before")
                after = event.get("transfer_loss_after")
                if before is None or after is None or float(after) >= float(before) - 1e-12:
                    failures.append("An accepted candidate-transfer step did not reduce the exact transfer objective")
                if float(event.get("new_loss", float("inf"))) >= float(event.get("old_loss", float("-inf"))) - 1e-12:
                    failures.append("An accepted candidate-transfer step did not reduce the exact current loss")
                if not bool(event.get("transfer_progress_qualified", False)):
                    failures.append("An accepted candidate-transfer step bypassed the joint feasibility test")
                if not bool(event.get("transfer_priority_feasible", False)):
                    failures.append("An accepted candidate-transfer step lacked a feasible joint solution")
                if event.get("joint_solver_converged") is not True:
                    failures.append("An accepted candidate-transfer step used an unconverged joint solver")
                required = float(event.get("joint_current_required", 0.0))
                certified_current = float(event.get("joint_current_certified_decrease", 0.0))
                if certified_current + 1e-12 < required:
                    failures.append("An accepted candidate-transfer step violated the certified current-loss requirement")
                selected = float(event.get("joint_selected_certified_decrease", 0.0))
                if selected <= 0.0:
                    failures.append("An accepted candidate-transfer step had nonpositive selected certified progress")
                candidate_decreases = [float(x) for x in event.get("joint_candidate_certified_decreases", [])]
                endpoint_tolerance = float(
                    cfg.get("afm", {}).get("transfer", {}).get(
                        "endpoint_tolerance", cfg.get("afm", {}).get("safe_update", {}).get("check_tolerance", 0.0)
                    )
                )
                strict_margin = (
                    2.0 * float(cfg.get("afm", {}).get("numerics", {}).get("endpoint_error_bound") or 0.0)
                    if str(event.get("certificate_mode", "empirical")) == "strict" else 0.0
                )
                numerical_envelope = endpoint_tolerance + strict_margin
                if not candidate_decreases or any(value < -numerical_envelope for value in candidate_decreases):
                    failures.append("An accepted candidate-transfer step lacked all-candidate certified nonincrease")
                solver_mode = event.get("joint_solver_mode")
                if solver_mode in {"functional_shield", "exact_counterfactual_restoration", "joint_counterfactual_normalized_assimilation"}:
                    shield_cfg = cfg.get("afm", {}).get("functional_shield", {})
                    if event.get("functional_shield_obstruction") is not None:
                        failures.append("An accepted functional-shield step carried an obstruction")
                    nodes = int(event.get("functional_shield_node_count", -1))
                    declared_max_nodes = int(
                        shield_cfg.get(
                            "max_nodes",
                            int(cfg.get("training", {}).get("batch_size", 0))
                            + int(cfg.get("afm", {}).get("resources", {}).get("max_records", 0))
                            * int(cfg.get("afm", {}).get("consolidation", {}).get("commit_samples", 0))
                            + int(cfg.get("afm", {}).get("resources", {}).get("max_contexts", 0))
                            * int(cfg.get("afm", {}).get("transfer", {}).get("samples", 0)),
                        )
                    )
                    if nodes < 0 or nodes > declared_max_nodes:
                        failures.append("An accepted functional shield exceeded its declared node capacity")
                    condition = float(event.get("functional_shield_condition_number", float("inf")))
                    if abs(condition - 1.0) > 1e-12:
                        failures.append("An accepted compact-cardinal shield reported a nontrivial solve condition")
                    residual = float(event.get("functional_shield_interpolation_residual", float("inf")))
                    residual_limit = float(shield_cfg.get("residual_tolerance", 0.0))
                    if residual > residual_limit:
                        failures.append("An accepted compact-cardinal shield exceeded its cardinal residual limit")
                    guard_leakage = float(event.get("functional_shield_maximum_guard_leakage", float("inf")))
                    if guard_leakage > residual_limit:
                        failures.append("An accepted compact-cardinal shield leaked onto its frozen guard set")
                    min_radius = float(event.get("functional_shield_minimum_support_radius", 0.0))
                    max_radius = float(event.get("functional_shield_maximum_support_radius", 0.0))
                    if min_radius < 0.0 or max_radius < min_radius:
                        failures.append("An accepted compact-cardinal shield reported invalid support radii")
                    match_tolerance = float(shield_cfg.get("feature_match_tolerance", 1e-8))
                    support_multiplier = float(shield_cfg.get("support_multiplier", 4.0))
                    expected_radius = support_multiplier * match_tolerance
                    radius_tolerance = max(1e-15, 1e-9 * expected_radius)
                    if abs(min_radius - expected_radius) > radius_tolerance or abs(max_radius - expected_radius) > radius_tolerance:
                        failures.append("An accepted compact-cardinal shield enlarged support beyond its replay-error envelope")
                    event_multiplier = float(event.get("functional_shield_support_multiplier", float("nan")))
                    if not math.isfinite(event_multiplier) or abs(event_multiplier - support_multiplier) > 1e-12:
                        failures.append("An accepted compact-cardinal shield reported the wrong support multiplier")
                    separation = float(event.get("functional_shield_minimum_address_separation", float("inf")))
                    if math.isfinite(separation) and separation <= 2.0 * max_radius:
                        failures.append("An accepted compact-cardinal shield violated address resolution")
                elif solver_mode not in {"joint_transfer"}:
                    failures.append("An accepted candidate transfer used an unknown certified solver mode")
                if float(event.get("new_loss", float("inf"))) > float(event.get("old_loss", float("-inf"))) - required + 1e-6:
                    failures.append("An accepted candidate-transfer step failed the exact current-progress endpoint")

    if len(commit_ids) != len(set(commit_ids)):
        failures.append("A protected record_id was reused, violating append-only segment identity")
    if len(frozen_consolidation_indices) != len(set(frozen_consolidation_indices)):
        failures.append("A consolidation-test error-budget index was reused")
    if len(signature_events) != 1:
        failures.append("The context signature was not frozen exactly once before protected routing")
    if not routing_assignments:
        failures.append("No item-level routing audit assignments were emitted")
    routed_sample_ids = [str(event.get("sample_id")) for event in routing_assignments]
    if len(routed_sample_ids) != len(set(routed_sample_ids)):
        failures.append("A sample_id appeared more than once in the routing audit transcript")
    if int(summary.get("candidate_training_runs", 0)) != len(frozen_candidates):
        failures.append("Candidate-training summary does not match frozen-candidate events")
    if int(summary.get("candidate_tests_started", 0)) != len(frozen_candidates):
        failures.append("Candidate-test summary does not match frozen-candidate events")
    if int(summary.get("candidate_validation_rejections", 0)) != len(rejected_candidates):
        failures.append("Candidate-rejection summary does not match event log")
    if int(summary.get("route_splits", 0)) != len(route_split_events):
        failures.append("Route-split summary does not match threshold-crossing split events")
    if int(summary.get("transfer_attempts", 0)) != len(transfer_attempt_steps):
        failures.append("Candidate-transfer attempt summary does not match AFM step events")
    if int(summary.get("transfer_accepted_steps", 0)) != len(transfer_joint_steps):
        failures.append("Accepted candidate-transfer summary does not match AFM step events")
    afm_steps = [e for e in events if e.get("event") == "afm_step"]
    shield_attempts = [
        e for e in afm_steps if bool(e.get("functional_shield_attempted", False))
    ]
    shield_accepted = [
        e for e in shield_attempts
        if e.get("joint_solver_mode") in {"functional_shield", "exact_counterfactual_restoration", "joint_counterfactual_normalized_assimilation"}
        and bool(e.get("accepted", False))
    ]
    exact_attempts = [
        e for e in afm_steps if bool(e.get("exact_counterfactual_restoration_attempted", False))
    ]
    exact_accepted = [e for e in exact_attempts if bool(e.get("accepted", False))]
    if int(summary.get("functional_shield_attempts", 0)) != len(shield_attempts):
        failures.append("Functional-shield attempt summary does not match AFM step events")
    if int(summary.get("functional_shield_accepted", 0)) != len(shield_accepted):
        failures.append("Functional-shield accepted summary does not match AFM step events")
    if int(summary.get("exact_restoration_attempts", 0)) != len(exact_attempts):
        failures.append("Exact-restoration attempt summary does not match AFM step events")
    if int(summary.get("exact_restoration_accepted", 0)) != len(exact_accepted):
        failures.append("Exact-restoration accepted summary does not match AFM step events")
    if candidate_invariant_violations:
        failures.append("Candidate invariant violations are forbidden in v0.11.0")
    if any(float(event.get("cumulative_damage", 0.0)) > 1e-15 for event in candidate_ledger_events):
        failures.append("A v0.11.0 candidate ledger accumulated damage beyond its numerical envelope")
    if len(commit_ids) < min_commits:
        failures.append(f"Run requirement failed: commits={len(commit_ids)} < min_commits={min_commits}")
    if len(protected_nonzero_steps) < min_protected_nonzero:
        failures.append(
            "Run requirement failed: protected nonzero steps="
            f"{len(protected_nonzero_steps)} < min_protected_nonzero_steps={min_protected_nonzero}"
        )

    renewal_obstructions = [e for e in events if e.get("event") == "renewal_obstruction"]
    if int(summary.get("renewal_capacity_obstructions", 0)) != len(renewal_obstructions):
        failures.append("Renewal-capacity obstruction summary does not match event log")

    capacity_events = [
        e for e in events if e.get("event") == "record_released" and str(e.get("reason", "")).startswith("capacity")
    ]
    if int(summary.get("capacity_releases", 0)) != len(capacity_events):
        failures.append("Capacity-release summary does not match explicit release events")

    checkpoint = __import__("torch").load(args.run_dir / "final.pt", map_location="cpu", weights_only=False)
    if "afm_state" not in checkpoint:
        failures.append("Checkpoint omits AFM persistent state")
    else:
        afm_state = checkpoint["afm_state"]
        candidate_states = afm_state.get("candidates", {})
        record_states = afm_state.get("records", {})
        shield_state = afm_state.get("functional_shield")
        if shield_state is None:
            failures.append("Checkpoint AFM state omits the deployed functional shield")
        else:
            max_nodes = int(cfg.get("afm", {}).get("functional_shield", {}).get(
                "max_nodes",
                int(cfg.get("training", {}).get("batch_size", 0))
                + int(cfg.get("afm", {}).get("resources", {}).get("max_records", 0))
                * int(cfg.get("afm", {}).get("consolidation", {}).get("commit_samples", 0))
                + int(cfg.get("afm", {}).get("resources", {}).get("max_contexts", 0))
                * int(cfg.get("afm", {}).get("transfer", {}).get("samples", 0)),
            ))
            centres = shield_state.get("centres")
            coefficients = shield_state.get("coefficients")
            support_radii = shield_state.get("support_radii")
            match_radii = shield_state.get("match_radii")
            if shield_state.get("kind") != "compact_cardinal" or int(shield_state.get("kind_code", -1)) != 8:
                failures.append("Checkpoint functional shield is not the v0.8 compact-cardinal construction")
            if centres is None or coefficients is None or support_radii is None or match_radii is None:
                failures.append("Checkpoint compact-cardinal shield omits centres, coefficients, or radii")
            elif (
                len(centres) > max_nodes
                or len(coefficients) != len(centres)
                or len(support_radii) != len(centres)
                or len(match_radii) != len(centres)
            ):
                failures.append("Checkpoint compact-cardinal shield violates its bounded node declaration")
            guards = afm_state.get("functional_shield_guard_features")
            guard_capacity = int(cfg.get("afm", {}).get("functional_shield", {}).get("guard_capacity", 0))
            if guards is None:
                failures.append("Checkpoint AFM state omits compact-shield guard features")
            elif len(guards) > guard_capacity:
                failures.append("Checkpoint compact-shield guard bank exceeds its declared capacity")
        if any("validation_losses" in c for c in candidate_states.values()):
            failures.append("Candidate checkpoint contains an unbounded validation-loss list")
        checkpoint_ids = {int(key) for key in record_states}
        expected_active_ids = set(commit_ids) - released_ids
        if checkpoint_ids != expected_active_ids:
            failures.append(
                "Checkpoint record registry is not exactly the active committed set "
                "after explicit releases"
            )
        for key, record in record_states.items():
            rid = int(record.get("record_id", -1))
            if int(key) != rid:
                failures.append(f"Checkpoint record key {key} does not equal immutable record_id {rid}")
            e_n = int(record.get("eprocess", {}).get("state", {}).get("n", 0))
            if e_n != reopening_processed.get(rid, 0):
                failures.append(f"Record {rid} e-process count does not match routed-outcome evidence")
            signature_n = int(record.get("signature_eprocess", {}).get("state", {}).get("n", 0))
            if signature_n != signature_processed.get(rid, 0):
                failures.append(f"Record {rid} signature e-process count does not match distinct block evidence")
            if int(record.get("signature_blocks_seen", 0)) != signature_n:
                failures.append(f"Record {rid} signature sufficient-state count is inconsistent")
        max_records = int(cfg["afm"]["resources"]["max_records"])
        active_records = list(record_states.values())
        if any(record.get("released_step") is not None for record in active_records):
            failures.append("Released records remain in persistent checkpoint state")
        if len(active_records) > max_records:
            failures.append("Active protected-segment registry exceeded max_records")
        total_rows = sum(int(record["sketch"].shape[0]) for record in active_records)
        if total_rows > int(cfg["afm"]["resources"]["total_sketch_rows"]):
            failures.append("Active sketch rows exceeded total_sketch_rows")
        router_state = afm_state.get("router", {})
        max_contexts = int(cfg["afm"]["resources"].get("max_contexts", max_records))
        if int(router_state.get("max_slots", -1)) != max_contexts:
            failures.append("Router context capacity does not match max_contexts")
        if len(candidate_states) > max_contexts:
            failures.append("Candidate context registry exceeded max_contexts")
        consolidation = cfg["afm"]["consolidation"]
        commit_samples = int(consolidation["commit_samples"])
        training_samples = int(consolidation.get("training_samples", 0))
        max_validation_samples = int(consolidation.get("max_validation_samples", consolidation["min_validation_samples"]))
        sketch_rows = int(cfg["afm"]["memory"]["sketch_rows"])
        reference_bytes = int(cfg["afm"]["resources"].get("evidence_reference_bytes", 4096))
        for slot, candidate in candidate_states.items():
            training_evidence = candidate.get("training_evidence", [])
            evidence = candidate.get("evidence", [])
            if len(training_evidence) > training_samples:
                failures.append(f"Candidate slot {slot} exceeded training_samples fitting capacity")
            if int(candidate.get("training_count", 0)) > training_samples:
                failures.append(f"Candidate slot {slot} exceeded its declared fitting count")
            if int(candidate.get("validation_count", 0)) > max_validation_samples:
                failures.append(f"Candidate slot {slot} exceeded max_validation_samples")
            snapshot = candidate.get("snapshot")
            if snapshot is None:
                if candidate.get("initial_parameters") is None:
                    failures.append(f"Unfrozen candidate slot {slot} omitted its private initial vector")
                if candidate.get("commit_budget_index") is not None or candidate.get("commit_alpha") is not None:
                    failures.append(f"Unfrozen candidate slot {slot} spent consolidation budget")
                if evidence or candidate.get("sketch") is not None:
                    failures.append(f"Unfrozen candidate slot {slot} contains validation state")
            else:
                if candidate.get("snapshot_shield_state") is None:
                    failures.append(f"Frozen candidate slot {slot} omitted its shield snapshot")
                if candidate.get("initial_parameters") is not None or training_evidence:
                    failures.append(f"Frozen candidate slot {slot} retained obsolete fitting state")
                if candidate.get("commit_budget_index") is None or candidate.get("commit_alpha") is None:
                    failures.append(f"Frozen candidate slot {slot} omitted its consolidation allocation")
            if len(evidence) > commit_samples:
                failures.append(f"Candidate slot {slot} exceeded commit_samples evidence capacity")
            transfer_evidence = candidate.get("transfer_evidence", [])
            transfer_targets = candidate.get("transfer_targets")
            transfer_capacity = int(cfg.get("afm", {}).get("transfer", {}).get("samples", commit_samples))
            if len(transfer_evidence) > transfer_capacity:
                failures.append(f"Candidate slot {slot} exceeded its bounded transfer evidence capacity")
            transfer_logits = candidate.get("transfer_logits")
            if transfer_targets is not None and len(transfer_targets) != len(transfer_evidence):
                failures.append(f"Candidate slot {slot} transfer targets do not match transfer evidence")
            if transfer_logits is not None and len(transfer_logits) != len(transfer_evidence):
                failures.append(f"Candidate slot {slot} transfer logits do not match transfer evidence")
            for row in list(training_evidence) + list(evidence):
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                if len(encoded) > reference_bytes:
                    failures.append(f"Candidate slot {slot} has an oversized evidence reference")
            sketch_state = candidate.get("sketch")
            if sketch_state is not None and len(sketch_state.get("rows", [])) >= 2 * sketch_rows:
                failures.append(f"Candidate slot {slot} exceeded bounded staging-sketch state")
        max_atoms = int(cfg["afm"]["resources"].get("max_atoms", max_records))
        if len(active_records) > max_atoms:
            failures.append("Active protected atomisation exceeded max_atoms")
        for key, record in record_states.items():
            if record.get("anchor_shield_state") is None:
                failures.append(f"Record {key} omitted its immutable shield snapshot")
            evidence = record.get("evidence", [])
            if len(evidence) > commit_samples:
                failures.append(f"Record {key} exceeded commit_samples evidence capacity")
            for row in evidence:
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                if len(encoded) > reference_bytes:
                    failures.append(f"Record {key} has an oversized evidence reference")
    learner_manifest = Path(cfg["data"]["train_manifest"])
    if learner_manifest.exists():
        forbidden = {"session", "context_id", "episode", "episode_name", "intervention", "original_label", "object_id"}
        with learner_manifest.open("r", encoding="utf-8") as handle:
            first = next((json.loads(line) for line in handle if line.strip()), {})
        leaked = sorted(forbidden.intersection(first))
        if leaked:
            failures.append(f"Task/evaluator metadata leaked into learner manifest: {leaked}")

    report = {
        "pass": not failures,
        "failures": failures,
        "nonzero_accepted_steps": len(nonzero_steps),
        "commits": len(commit_ids),
        "protected_nonzero_steps": len(protected_nonzero_steps),
        "method_activated": len(commit_ids) > 0 and len(protected_nonzero_steps) > 0,
        "run_requirements": {
            "min_commits": min_commits,
            "min_protected_nonzero_steps": min_protected_nonzero,
        },
        "context_signature": signature_events[0] if len(signature_events) == 1 else None,
        "routing_assignment_events": len(routing_assignments),
        "append_only_record_ids": commit_ids,
        "explicitly_released_record_ids": sorted(released_ids),
        "active_checkpoint_record_ids": sorted(set(commit_ids) - released_ids),
        "reopening_routed_outcomes": sum(reopening_processed.values()),
        "signature_shift_blocks": sum(signature_processed.values()),
        "reopenings": int(summary.get("reopenings", 0)),
        "route_splits": len(route_split_events),
        "transfer_attempts": len(transfer_attempt_steps),
        "transfer_accepted_steps": len(transfer_joint_steps),
        "functional_shield_attempts": int(summary.get("functional_shield_attempts", 0)),
        "functional_shield_accepted": int(summary.get("functional_shield_accepted", 0)),
        "functional_shield_obstructions": int(summary.get("functional_shield_obstructions", 0)),
        "exact_restoration_attempts": len(exact_attempts),
        "exact_restoration_accepted": len(exact_accepted),
        "exact_restoration_obstructions": int(summary.get("exact_restoration_obstructions", 0)),
        "minimum_exact_counterfactual_progress_ratio": min(
            [float(e.get("exact_counterfactual_progress_ratio")) for e in exact_accepted
             if e.get("exact_counterfactual_progress_ratio") is not None] + [1.0]
        ),
        "maximum_exact_counterfactual_endpoint_error": max(
            [float(e.get("exact_counterfactual_endpoint_error", 0.0) or 0.0) for e in exact_accepted] + [0.0]
        ),
        "minimum_requested_counterfactual_charge_fraction": (
            min(float(e.get("requested_counterfactual_charge_fraction")) for e in exact_accepted
                if e.get("requested_counterfactual_charge_fraction") is not None)
            if any(e.get("requested_counterfactual_charge_fraction") is not None for e in exact_accepted)
            else None
        ),
        "minimum_realised_counterfactual_path_fraction": min(
            [float(e.get("realised_counterfactual_path_fraction")) for e in exact_accepted
             if e.get("realised_counterfactual_path_fraction") is not None] + [1.0]
        ),
        "minimum_persistent_base_progress_ratio": (
            min(float(e.get("persistent_base_progress_ratio")) for e in exact_accepted
                if e.get("persistent_base_progress_ratio") is not None)
            if any(e.get("persistent_base_progress_ratio") is not None for e in exact_accepted)
            else None
        ),
        "minimum_certified_persistent_progress_ratio_lower_bound": (
            min(float(e.get("certified_persistent_progress_ratio_lower_bound")) for e in exact_accepted
                if e.get("certified_persistent_progress_ratio_lower_bound") is not None)
            if any(e.get("certified_persistent_progress_ratio_lower_bound") is not None for e in exact_accepted)
            else None
        ),
        "maximum_projected_counterfactual_alignment_error": max(
            [float(e.get("projected_counterfactual_alignment_error", 0.0) or 0.0) for e in exact_accepted] + [0.0]
        ),
        "maximum_projection_idempotence_error": max(
            [float(e.get("projection_idempotence_error", 0.0) or 0.0) for e in exact_accepted] + [0.0]
        ),
        "exact_counterfactual_executable_endpoint_tolerance": max(
            [
                _exact_restoration_executable_endpoint_tolerance(
                    cfg, str(e.get("certificate_mode", "empirical"))
                )
                for e in exact_accepted
            ]
            + [
                _exact_restoration_executable_endpoint_tolerance(
                    cfg, str(cfg.get("afm", {}).get("certificates", {}).get("mode", "empirical"))
                )
            ]
        ),
        "candidate_training_runs": len(frozen_candidates),
        "candidate_validation_rejections": len(rejected_candidates),
        "renewal_trials": len([e for e in events if e.get("event") == "renewal_trial"]),
        "renewal_activated_slots": sorted(activated_slots),
        "renewal_capacity_obstructions": len(renewal_obstructions),
        "max_empirical_drift_violation": float(summary.get("max_empirical_drift_violation", 0.0)),
        "max_activation_gap": float(summary.get("max_activation_gap", 0.0)),
        "max_snapshot_drift_violation": float(summary.get("max_snapshot_drift_violation", 0.0)),
        "total_delta": float(summary.get("total_delta", 0.0)),
        "total_error_budget_allocated": float(summary.get("total_error_budget_allocated", 0.0)),
    }
    (args.run_dir / "validity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
