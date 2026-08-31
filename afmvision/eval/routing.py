from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _entropy(counts: Iterable[int]) -> float:
    values = [int(value) for value in counts if int(value) > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values)


def _normalised_mutual_information(contingency: dict[str, Counter[int]]) -> float:
    context_totals = {context: sum(counter.values()) for context, counter in contingency.items()}
    slot_totals: Counter[int] = Counter()
    for counter in contingency.values():
        slot_totals.update(counter)
    total = sum(context_totals.values())
    if total <= 0:
        return 0.0

    mutual_information = 0.0
    for context, counter in contingency.items():
        for slot, count in counter.items():
            if count <= 0:
                continue
            p_cs = count / total
            p_c = context_totals[context] / total
            p_s = slot_totals[slot] / total
            mutual_information += p_cs * math.log(p_cs / (p_c * p_s))

    h_context = _entropy(context_totals.values())
    h_slot = _entropy(slot_totals.values())
    denominator = math.sqrt(h_context * h_slot)
    return 0.0 if denominator <= 0.0 else mutual_information / denominator


def _contingency_report(contingency: dict[str, Counter[int]]) -> dict[str, Any]:
    context_totals = {context: sum(counter.values()) for context, counter in contingency.items()}
    slot_contexts: dict[int, Counter[str]] = defaultdict(Counter)
    for context, counter in contingency.items():
        for slot, count in counter.items():
            slot_contexts[int(slot)][str(context)] += int(count)
    total = sum(context_totals.values())
    context_consistent = sum(max(counter.values(), default=0) for counter in contingency.values())
    slot_pure = sum(max(counter.values(), default=0) for counter in slot_contexts.values())

    per_context: dict[str, Any] = {}
    for context, counter in sorted(contingency.items(), key=lambda item: item[0]):
        n = sum(counter.values())
        dominant_slot, dominant_count = (None, 0)
        if counter:
            dominant_slot, dominant_count = min(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        per_context[str(context)] = {
            "count": n,
            "dominant_slot": dominant_slot,
            "dominant_slot_fraction": 0.0 if n == 0 else dominant_count / n,
            "slot_counts": {str(slot): count for slot, count in sorted(counter.items())},
        }

    per_slot: dict[str, Any] = {}
    for slot, counter in sorted(slot_contexts.items()):
        n = sum(counter.values())
        dominant_context, dominant_count = (None, 0)
        if counter:
            dominant_context, dominant_count = min(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        per_slot[str(slot)] = {
            "count": n,
            "dominant_context": dominant_context,
            "dominant_context_fraction": 0.0 if n == 0 else dominant_count / n,
            "context_counts": dict(sorted(counter.items())),
        }

    return {
        "assigned_items": total,
        "context_to_slot_consistency": 0.0 if total == 0 else context_consistent / total,
        "slot_to_context_purity": 0.0 if total == 0 else slot_pure / total,
        "external_context_collision_rate": 0.0 if total == 0 else 1.0 - slot_pure / total,
        "normalised_mutual_information": _normalised_mutual_information(contingency),
        "per_context": per_context,
        "per_slot": per_slot,
    }


def analyse_routing(
    events: list[dict[str, Any]], sidecar_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute evaluator-only routing diagnostics without changing learner state.

    The sidecar is joined only after the run.  It contains semantic fields that
    are forbidden from the learner process, so this function must remain an
    analysis-only utility.
    """

    sidecar = {
        str(row["sample_id"]): row
        for row in sidecar_rows
        if row.get("sample_id") is not None
    }
    assignments = [event for event in events if event.get("event") == "routing_assignment"]
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_sidecar = 0
    for event in assignments:
        sample_id = event.get("sample_id")
        metadata = sidecar.get(str(sample_id)) if sample_id is not None else None
        if metadata is None:
            missing_sidecar += 1
            continue
        joined.append((event, metadata))

    assigned = [(event, meta) for event, meta in joined if event.get("slot") is not None]
    overflow = sum(bool(event.get("overflow")) for event, _ in joined)
    created = sum(bool(event.get("centroid_created")) for event, _ in joined)
    protected = sum(bool(event.get("active_record_ids")) for event, _ in assigned)
    finite_distances = [
        float(event["distance"])
        for event, _ in assigned
        if event.get("distance") is not None and math.isfinite(float(event["distance"]))
    ]

    context_contingency: dict[str, Counter[int]] = defaultdict(Counter)
    regime_contingency: dict[str, Counter[int]] = defaultdict(Counter)
    episode_contingency: dict[str, Counter[int]] = defaultdict(Counter)
    protected_by_context: Counter[str] = Counter()
    assigned_by_context: Counter[str] = Counter()
    for event, metadata in assigned:
        slot = int(event["slot"])
        context = str(metadata.get("context_id", "unknown"))
        regime = str(metadata.get("semantic_regime", "unknown"))
        episode = str(metadata.get("episode_name", metadata.get("episode", "unknown")))
        context_contingency[context][slot] += 1
        regime_contingency[regime][slot] += 1
        episode_contingency[episode][slot] += 1
        assigned_by_context[context] += 1
        if event.get("active_record_ids"):
            protected_by_context[context] += 1

    context_report = _contingency_report(context_contingency)
    context_report["protected_fraction_by_context"] = {
        context: protected_by_context[context] / count
        for context, count in sorted(assigned_by_context.items())
        if count > 0
    }

    return {
        "routing_assignment_events": len(assignments),
        "joined_assignments": len(joined),
        "missing_sidecar_assignments": missing_sidecar,
        "assigned_items": len(assigned),
        "overflow_items": overflow,
        "centroid_creations": created,
        "protected_assignment_fraction": 0.0 if not assigned else protected / len(assigned),
        "finite_distance_mean": (
            0.0 if not finite_distances else sum(finite_distances) / len(finite_distances)
        ),
        "finite_distance_max": 0.0 if not finite_distances else max(finite_distances),
        "context": context_report,
        "semantic_regime": _contingency_report(regime_contingency),
        "episode": _contingency_report(episode_contingency),
        "analysis_scope": "posthoc evaluator-only; never read by AFMTrainer",
    }
