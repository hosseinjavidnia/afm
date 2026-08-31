from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]



def _binomial_upper_99(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    z = 2.3263478740408408
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = p + z2 / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return min(1.0, (centre + radius) / denominator)

def analyse_integrated_job(job: dict[str, Any]) -> dict[str, Any]:
    afm_dir = Path(job["afm_run_dir"])
    baseline_dir = Path(job["baseline_run_dir"])
    status = _read_json(Path(job.get("status_path", ""))) if job.get("status_path") else None
    summary = _read_json(afm_dir / "summary.json")
    evaluation = _read_json(afm_dir / "evaluation.json")
    routing = _read_json(afm_dir / "routing_analysis.json")
    baseline_eval = _read_json(baseline_dir / "evaluation.json")
    events = _read_events(afm_dir / "events.jsonl")
    sidecar_rows = _read_events(Path(job["sidecar"])) if job.get("sidecar") else []
    sidecar_by_sample = {str(row.get("sample_id")): row for row in sidecar_rows}
    expected = dict(job["expected"])
    config_path = Path(job["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    indexed_events = list(enumerate(events))
    route_split_events = [event for event in events if event.get("event") == "record_route_split"]
    indexed_route_splits = [
        (index, event) for index, event in indexed_events if event.get("event") == "record_route_split"
    ]
    routing_assignments = [
        (index, event) for index, event in indexed_events if event.get("event") == "routing_assignment"
    ]
    route_split_budget = float(config.get("afm", {}).get("statistics", {}).get("total_delta", 0.0)) * float(
        config.get("afm", {}).get("statistics", {}).get("category_weights", {}).get("route_split", 0.0)
    )
    preflight_failures: list[str] = []
    if status is None:
        preflight_failures.append("job status missing")
    elif not bool(status.get("pass", False)):
        preflight_failures.extend(str(item) for item in status.get("failures", ["job command failed"]))
    if summary is None:
        preflight_failures.append("AFM summary missing")
    if evaluation is None:
        preflight_failures.append("AFM evaluation missing")
    if routing is None:
        preflight_failures.append("AFM routing analysis missing")
    if baseline_eval is None:
        preflight_failures.append("matched-SGD evaluation missing")
    if not events:
        preflight_failures.append("AFM event log missing or empty")
    if summary is None:
        return {
            "scenario": job["scenario"],
            "seed": job["seed"],
            "complete": False,
            "pass": False,
            "failures": preflight_failures,
        }

    afm_steps = [e for e in events if e.get("event") == "afm_step"]
    protected_nonzero = [
        e for e in afm_steps
        if bool(e.get("accepted"))
        and float(e.get("realised_step_length", 0.0)) > 0.0
        and int(e.get("active_records", 0)) > 0
    ]
    releases = [e for e in events if e.get("event") == "record_released"]
    capacity_events = [
        e for e in events
        if (e.get("event") in {"record_released", "context_released"} and str(e.get("reason", "")).startswith("capacity"))
        or e.get("event") == "unprotected_overflow"
    ]
    record_ids = [int(e["record_id"]) for e in events if e.get("event") == "record_committed"]
    transfer_fallbacks = int(summary.get("transfer_fallback_steps", 0))
    transfer_obstructions = int(summary.get("transfer_obstructions", 0))
    transfer_events = [e for e in afm_steps if bool(e.get("transfer_attempted", False))]
    transfer_cosines = [
        float(e["transfer_projected_cosine"])
        for e in transfer_events
        if e.get("transfer_projected_cosine") is not None
    ]
    noncommon_transfer_attempts = sum(
        1 for e in transfer_events if not bool(e.get("transfer_common_descent", False))
    )
    exact_zero_dim_transfer_obstructions = sum(
        1
        for e in transfer_events
        if bool(e.get("exact_zero_dim_transfer_obstruction", False))
        and int(e.get("allowed_coordinate_count", -1)) == 1
        and int(e.get("protected_basis_rank", -1)) == 1
        and int(e.get("feasible_subspace_dimension", -1)) == 0
        and not bool(e.get("transfer_common_descent", False))
        and not bool(e.get("transfer_joint_step", False))
        and e.get("transfer_unprojected_cosine") is not None
        and float(e["transfer_unprojected_cosine"]) <= -0.999
    )
    route_purity = None
    context_collision_rate = None
    centroid_creations = None
    if routing is not None:
        route_purity = float(routing.get("context", {}).get("slot_to_context_purity", 0.0))
        context_collision_rate = float(
            routing.get("context", {}).get("external_context_collision_rate", 1.0)
        )
        centroid_creations = int(routing.get("centroid_creations", 0))

    within_route_split_verified = False
    if expected.get("require_within_route_split", False) and indexed_route_splits:
        split_index, split_event = indexed_route_splits[0]
        source_slot = int(split_event.get("source_slot", -1))
        pre_split_shift_assignments = []
        for event_index, event in routing_assignments:
            if event_index >= split_index:
                continue
            metadata = sidecar_by_sample.get(str(event.get("sample_id")), {})
            if metadata.get("intervention") == "observable_shift_within_route":
                pre_split_shift_assignments.append(event)
        within_route_split_verified = bool(pre_split_shift_assignments) and all(
            int(event.get("slot", -1)) == source_slot
            and not bool(event.get("centroid_created", False))
            for event in pre_split_shift_assignments
        )
    failures: list[str] = list(preflight_failures)

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(float(summary.get("max_empirical_drift_violation", 0.0)) <= 1e-5, "active-interval retention violation")
    require(float(summary.get("max_snapshot_drift_violation", 0.0)) <= 1e-5, "validated-snapshot retention violation")
    require(len(record_ids) == len(set(record_ids)), "protected record identity was reused")

    if "min_commits" in expected:
        require(int(summary.get("commits", 0)) >= int(expected["min_commits"]), "minimum commits not reached")
    if "min_protected_nonzero_steps" in expected:
        require(len(protected_nonzero) >= int(expected["min_protected_nonzero_steps"]), "minimum protected nonzero steps not reached")
    if "min_route_purity" in expected:
        require(route_purity is not None and route_purity >= float(expected["min_route_purity"]), "positive-control route purity too low")
    if "min_centroid_creations" in expected:
        require(
            centroid_creations is not None
            and centroid_creations >= int(expected["min_centroid_creations"]),
            "direct observable shift did not create the required ordinary route",
        )
    if "max_context_collision_rate" in expected:
        require(
            context_collision_rate is not None
            and context_collision_rate <= float(expected["max_context_collision_rate"]),
            "direct observable discovery produced excessive context collision",
        )
    if "max_reopenings" in expected:
        require(int(summary.get("reopenings", 0)) <= int(expected["max_reopenings"]), "false reopening in unchanged control")
    if "min_reopenings" in expected:
        require(int(summary.get("reopenings", 0)) >= int(expected["min_reopenings"]), "semantic change did not trigger reopening")
    if "max_route_splits" in expected:
        require(int(summary.get("route_splits", 0)) <= int(expected["max_route_splits"]), "identical-signature conflict triggered route split")
    if expected.get("route_split_null", False):
        require(
            all(bool(event.get("dual_evidence_required", False)) for event in route_split_events),
            "a semantic-conflict split bypassed the independent signature test",
        )
    if "min_reopenings_or_splits" in expected:
        require(
            int(summary.get("reopenings", 0)) + int(summary.get("route_splits", 0)) >= int(expected["min_reopenings_or_splits"]),
            "observable change triggered neither reopening nor route refinement",
        )
    if "min_route_splits" in expected:
        require(
            int(summary.get("route_splits", 0)) >= int(expected["min_route_splits"]),
            "within-route observable change did not trigger dual-evidence route refinement",
        )
    if expected.get("require_within_route_split", False):
        require(
            within_route_split_verified,
            "the route-refinement control was separated by ordinary routing before dual-evidence splitting",
        )
    if "min_explicit_capacity_events" in expected:
        require(len(capacity_events) >= int(expected["min_explicit_capacity_events"]), "capacity saturation was not handled explicitly")
    if "min_transfer_attempts" in expected:
        require(int(summary.get("transfer_attempts", 0)) >= int(expected["min_transfer_attempts"]), "transfer-conflict control produced no transfer attempt")
    if "min_transfer_obstructions_or_fallbacks" in expected:
        require(
            transfer_fallbacks + transfer_obstructions >= int(expected["min_transfer_obstructions_or_fallbacks"]),
            "transfer conflict was not exposed as fallback or obstruction",
        )
    if "max_transfer_projected_cosine" in expected:
        require(
            bool(transfer_cosines)
            and min(transfer_cosines) <= float(expected["max_transfer_projected_cosine"]),
            "legacy transfer-conflict control did not create measured opposed projected gradients",
        )
    if "min_exact_zero_dim_obstructions" in expected:
        require(
            exact_zero_dim_transfer_obstructions >= int(expected["min_exact_zero_dim_obstructions"]),
            "exact transfer control did not expose a zero-dimensional protected feasible subspace",
        )
    if "analytic_model_kind" in expected:
        require(
            str(config.get("model", {}).get("kind", "convnet_adapters"))
            == str(expected["analytic_model_kind"]),
            "transfer obstruction was not run with the declared analytic validation model",
        )
    if "min_noncommon_transfer_attempts" in expected:
        require(
            noncommon_transfer_attempts >= int(expected["min_noncommon_transfer_attempts"]),
            "transfer-conflict control produced no unavailable common-descent attempt",
        )

    # Impossibility controls pass by safe, explicit behaviour, not by semantic recovery.
    if expected.get("kind") == "impossibility":
        require(float(summary.get("max_empirical_drift_violation", 0.0)) <= 1e-5, "impossible control caused hidden forgetting")
        require(float(summary.get("max_snapshot_drift_violation", 0.0)) <= 1e-5, "impossible control violated snapshot bound")

    afm_accuracy = None if evaluation is None else float(evaluation.get("overall_accuracy", 0.0))
    baseline_accuracy = None if baseline_eval is None else float(baseline_eval.get("overall_accuracy", 0.0))
    return {
        "scenario": job["scenario"],
        "seed": job["seed"],
        "purpose": job["purpose"],
        "expected_kind": expected.get("kind"),
        "complete": not preflight_failures,
        "pass": not failures,
        "failures": failures,
        "summary": summary,
        "protected_nonzero_steps": len(protected_nonzero),
        "record_ids": record_ids,
        "explicit_releases": len(releases),
        "explicit_capacity_events": len(capacity_events),
        "route_purity": route_purity,
        "context_collision_rate": context_collision_rate,
        "centroid_creations": centroid_creations,
        "within_route_split_verified": within_route_split_verified,
        "route_split_null": bool(expected.get("route_split_null", False)),
        "route_split_budget": route_split_budget,
        "route_split_occurred": bool(route_split_events),
        "minimum_transfer_projected_cosine": None if not transfer_cosines else min(transfer_cosines),
        "noncommon_transfer_attempts": noncommon_transfer_attempts,
        "exact_zero_dim_transfer_obstructions": exact_zero_dim_transfer_obstructions,
        "afm_accuracy": afm_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_difference": None if afm_accuracy is None or baseline_accuracy is None else afm_accuracy - baseline_accuracy,
    }


def aggregate_full_claim(matrix: dict[str, Any], mechanism_report: dict[str, Any] | None) -> dict[str, Any]:
    jobs = [analyse_integrated_job(job) for job in matrix["jobs"]]
    positive = [job for job in jobs if job.get("expected_kind") in {"positive", "positive_with_change"}]
    obstruction = [job for job in jobs if job.get("expected_kind") in {"obstruction", "impossibility"}]
    mechanism_pass = bool(mechanism_report and mechanism_report.get("mechanism_alignment_pass", False))
    positive_pass = bool(positive) and all(job["pass"] for job in positive)
    obstruction_pass = bool(obstruction) and all(job["pass"] for job in obstruction)
    semantic_null = [job for job in jobs if job.get("route_split_null") and job.get("complete")]
    semantic_false_splits = sum(int(job.get("route_split_occurred", False)) for job in semantic_null)
    semantic_null_trials = len(semantic_null)
    semantic_null_upper = _binomial_upper_99(semantic_false_splits, semantic_null_trials)
    semantic_nominal = max((float(job.get("route_split_budget", 0.0)) for job in semantic_null), default=0.0)
    semantic_null_report = {
        "runs": semantic_null_trials,
        "runs_with_split": semantic_false_splits,
        "empirical_rate": None if semantic_null_trials == 0 else semantic_false_splits / semantic_null_trials,
        "upper_99": semantic_null_upper,
        "declared_per_run_lifetime_budget": semantic_nominal,
        "minimum_recommended_runs": 100,
        "status": (
            "insufficient_runs" if semantic_null_trials < 100
            else "pass" if semantic_null_upper <= semantic_nominal + 0.02
            else "fail"
        ),
    }
    return {
        "suite": matrix["suite"],
        "mechanism_alignment_pass": mechanism_pass,
        "controlled_positive_pass": positive_pass,
        "controlled_obstruction_pass": obstruction_pass,
        "controlled_integration_complete": all(job.get("complete", False) for job in jobs),
        "real_world_evidence_complete": False,
        "cross_dataset_evidence_complete": False,
        "full_ai_validation_supported": False,
        "paper_proved": False,
        "semantic_conflict_false_split_diagnostic": semantic_null_report,
        "jobs": jobs,
        "interpretation": (
            "Full AI validation remains false until mechanism, controlled positive, controlled obstruction, "
            "multi-seed real-world, and cross-dataset gates all pass. Experimental success never replaces proof."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# AFM-U full-claim validation report",
        "",
        f"- Mechanism alignment: **{report['mechanism_alignment_pass']}**",
        f"- Controlled positive suite: **{report['controlled_positive_pass']}**",
        f"- Controlled obstruction suite: **{report['controlled_obstruction_pass']}**",
        f"- Full AI validation supported: **{report['full_ai_validation_supported']}**",
        "",
        "## Semantic-conflict false-split diagnostic",
        "",
        ("- " + json.dumps(report.get("semantic_conflict_false_split_diagnostic", {}), sort_keys=True)),
        "",
        "## Integrated jobs",
        "",
        "| Scenario | Seed | Pass | Commits | Protected steps | Reopenings | Splits | Route purity | AFM acc. | Baseline acc. |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for job in report["jobs"]:
        summary = job.get("summary", {})
        lines.append(
            "| {scenario} | {seed} | {passed} | {commits} | {protected} | {reopenings} | {splits} | {purity} | {afm} | {base} |".format(
                scenario=job["scenario"],
                seed=job["seed"],
                passed="yes" if job["pass"] else "no",
                commits=summary.get("commits", "—"),
                protected=job.get("protected_nonzero_steps", "—"),
                reopenings=summary.get("reopenings", "—"),
                splits=summary.get("route_splits", "—"),
                purity="—" if job.get("route_purity") is None else f"{job['route_purity']:.3f}",
                afm="—" if job.get("afm_accuracy") is None else f"{job['afm_accuracy']:.3f}",
                base="—" if job.get("baseline_accuracy") is None else f"{job['baseline_accuracy']:.3f}",
            )
        )
    lines.extend(["", "## Failures", ""])
    for job in report["jobs"]:
        if job["failures"]:
            lines.append(f"### {job['scenario']} / seed {job['seed']}")
            lines.extend(f"- {failure}" for failure in job["failures"])
            lines.append("")
    lines.append(report["interpretation"])
    return "\n".join(lines) + "\n"
