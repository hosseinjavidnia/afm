from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Phase:
    name: str
    context_id: str
    count: int
    style: str
    label_rule: str = "normal"
    intervention: str = "none"


@dataclass(frozen=True)
class Scenario:
    name: str
    purpose: str
    phases: tuple[Phase, ...]
    expected: dict[str, Any]


STYLES: dict[str, tuple[int, int, int]] = {
    "red": (115, 20, 20),
    "blue": (20, 35, 120),
    "green": (20, 105, 35),
    "amber": (120, 80, 15),
    "purple": (90, 25, 105),
    "red_shift": (20, 120, 120),
    "neutral": (65, 65, 65),
}


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            name="favourable_recurrence",
            purpose="Positive control: separated observable contexts, recurrence, multiple commits and protected learning.",
            phases=(
                Phase("a_initial", "A", 160, "red"),
                Phase("b_new", "B", 160, "blue"),
                Phase("a_recurrence", "A", 160, "red"),
            ),
            expected={
                "kind": "positive",
                "min_commits": 2,
                "min_protected_nonzero_steps": 4,
                "min_route_purity": 0.95,
                "max_reopenings": 0,
            },
        ),
        Scenario(
            name="semantic_conflict",
            purpose="Same observable context later requires the opposite label policy; tests evidence-based reopening without oracle routing.",
            phases=(
                Phase("a_initial", "A", 192, "red"),
                Phase("b_new", "B", 160, "blue"),
                Phase("a_return", "A", 160, "red"),
                Phase("a_conflict", "A", 768, "red", label_rule="swap", intervention="semantic_conflict"),
            ),
            expected={
                "kind": "positive_with_change",
                "min_commits": 1,
                "min_protected_nonzero_steps": 4,
                "min_reopenings": 1,
                "route_split_null": True,
            },
        ),
        Scenario(
            name="observable_shift",
            purpose=(
                "Direct-discovery positive control: a large observable acquisition shift "
                "is immediately separated by ordinary bounded routing before reopening is needed."
            ),
            phases=(
                Phase("a_initial", "A", 256, "red"),
                Phase(
                    "a_shifted_policy",
                    "A_shift",
                    768,
                    "red_shift",
                    label_rule="swap",
                    intervention="observable_shift_direct",
                ),
            ),
            expected={
                "kind": "positive_with_change",
                "min_commits": 1,
                "min_protected_nonzero_steps": 4,
                "min_centroid_creations": 2,
                "min_route_purity": 0.95,
                "max_context_collision_rate": 0.05,
                "max_route_splits": 0,
                "direct_discovery_control": True,
            },
        ),
        Scenario(
            name="within_route_observable_shift",
            purpose=(
                "Dual-evidence route-refinement positive control: the observable shift is "
                "deliberately kept inside the ordinary routing threshold, so a new route may "
                "be created only after outcome and signature e-process crossings."
            ),
            phases=(
                Phase("a_initial", "A", 256, "red"),
                Phase(
                    "a_shifted_policy",
                    "A_shift",
                    768,
                    "red_shift",
                    label_rule="swap",
                    intervention="observable_shift_within_route",
                ),
            ),
            expected={
                "kind": "positive_with_change",
                "min_commits": 1,
                "min_protected_nonzero_steps": 4,
                "min_route_splits": 1,
                "require_within_route_split": True,
            },
        ),
        Scenario(
            name="capacity_pressure",
            purpose="More observable contexts than bounded router and record capacity; tests explicit release or overflow.",
            phases=tuple(
                Phase(f"context_{name.lower()}", name, 160, style)
                for name, style in zip("ABCDE", ("red", "blue", "green", "amber", "purple"))
            ),
            expected={
                "kind": "obstruction",
                "min_commits": 1,
                "min_explicit_capacity_events": 1,
                "silent_overwrite_allowed": False,
            },
        ),
        Scenario(
            name="transfer_conflict",
            purpose=(
                "Exact protected-feasibility control. A one-parameter binary neural predictor "
                "first commits the constant-zero policy. Its nonconstant protected behaviour "
                "spans the sole trainable coordinate. A successor candidate is fitted to the "
                "constant-one policy on the same observable image law, while current learning "
                "returns to constant zero. The protected feasible subspace is therefore exactly "
                "zero-dimensional, so no nonzero direction can reduce both objectives while "
                "preserving the active record."
            ),
            phases=(
                # 32 bootstrap + 32 private fit + at least 32 validation outcomes.
                Phase("zero_policy_commit", "A", 96, "red", label_rule="constant_zero"),
                # One successor fitting block on the opposite policy.
                Phase(
                    "one_policy_candidate_fit",
                    "A",
                    32,
                    "red",
                    label_rule="constant_one",
                    intervention="transfer_candidate_fit",
                ),
                # Current learning returns to the protected policy while the
                # frozen opposite candidate supplies the transfer objective.
                Phase(
                    "zero_policy_current_objective",
                    "A",
                    128,
                    "red",
                    label_rule="constant_zero",
                    intervention="transfer_conflict",
                ),
            ),
            expected={
                "kind": "obstruction",
                "min_commits": 1,
                "min_transfer_attempts": 1,
                "min_noncommon_transfer_attempts": 1,
                "min_exact_zero_dim_obstructions": 1,
                "min_transfer_obstructions_or_fallbacks": 1,
                "max_retention_violation": 0.0,
                "analytic_model_kind": "scalar_transfer_conflict",
            },
        ),
        Scenario(
            name="identical_observation_impossibility",
            purpose="Evaluator semantics change while the observable image law is identical; tests routing and compatibility lower bounds.",
            phases=(
                Phase("same_law_policy_0", "H0", 256, "neutral"),
                Phase("same_law_policy_1", "H1", 256, "neutral", label_rule="swap", intervention="identical_law_conflict"),
            ),
            expected={
                "kind": "impossibility",
                "semantic_oracle_recovery_required": False,
                "no_hidden_retention_violation": True,
            },
        ),
    )


def _base_label(index: int) -> int:
    return index % 2


def _render_image(style: str, base_label: int, seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = np.asarray(STYLES[style], dtype=np.int16)
    image = np.broadcast_to(background, (size, size, 3)).copy()
    texture = rng.normal(0.0, 7.0, size=image.shape)
    image = np.clip(image + texture, 0, 255).astype(np.uint8)
    if style == "red_shift":
        # Predeclared class-independent acquisition shift.  It is visible in
        # pixels and therefore admissible to the task-free signature, but it
        # carries no label, context ID, or evaluator metadata.
        tile = max(4, size // 8)
        yy, xx = np.indices((size, size))
        checker = ((xx // tile + yy // tile) % 2).astype(bool)
        image[checker] = np.array([20, 160, 160], dtype=np.uint8)
        image[~checker] = np.array([145, 25, 25], dtype=np.uint8)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    foreground = (235, 235, 235)
    margin = size // 5
    width = max(4, size // 9)
    jitter = int(rng.integers(-2, 3))
    if base_label == 0:
        x = size // 2 + jitter
        draw.rectangle((x - width, margin, x + width, size - margin), fill=foreground)
    else:
        y = size // 2 + jitter
        draw.rectangle((margin, y - width, size - margin, y + width), fill=foreground)
    # A small class-independent corner marker prevents exact pixel duplicates.
    radius = max(2, size // 18)
    cx = int(rng.integers(radius, size - radius))
    cy = int(rng.integers(radius, size - radius))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(150, 150, 150))
    return np.asarray(pil, dtype=np.uint8)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def generate_scenario(root: Path, scenario: Scenario, seed: int, image_size: int = 64) -> dict[str, Any]:
    scenario_root = root / scenario.name / f"seed_{seed}"
    image_root = scenario_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    learner: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    evaluator: list[dict[str, Any]] = []
    item_index = 0
    for phase_index, phase in enumerate(scenario.phases):
        for local_index in range(phase.count):
            base_label = _base_label(local_index)
            if phase.label_rule == "normal":
                observed_label = base_label
            elif phase.label_rule == "swap":
                observed_label = 1 - base_label
            elif phase.label_rule == "constant_zero":
                observed_label = 0
            elif phase.label_rule == "constant_one":
                observed_label = 1
            else:
                raise ValueError(f"Unknown label rule: {phase.label_rule}")
            sample_seed = seed * 1_000_003 + phase_index * 10_007 + local_index
            array = _render_image(phase.style, base_label, sample_seed, size=image_size)
            sample_id = f"{scenario.name}-{seed}-{item_index:06d}"
            path = image_root / f"{item_index:06d}.png"
            Image.fromarray(array).save(path)
            learner_row = {
                "sample_id": sample_id,
                "path": str(path.resolve()),
                "label": int(observed_label),
                "transform": {},
                "transform_seed": sample_seed,
            }
            metadata = {
                "sample_id": sample_id,
                "context_id": phase.context_id,
                "episode": phase_index,
                "episode_name": phase.name,
                "semantic_regime": phase.label_rule,
                "intervention": phase.intervention,
                "style": phase.style,
                "base_label": base_label,
                "observed_label": observed_label,
                "stream_index": item_index,
            }
            learner.append(learner_row)
            sidecar.append(metadata)
            evaluator.append({**learner_row, **metadata})
            item_index += 1
    train = scenario_root / "train.jsonl"
    eval_path = scenario_root / "evaluator.jsonl"
    sidecar_path = scenario_root / "sidecar.jsonl"
    _write_jsonl(train, learner)
    _write_jsonl(eval_path, evaluator)
    _write_jsonl(sidecar_path, sidecar)
    descriptor = {
        "name": scenario.name,
        "purpose": scenario.purpose,
        "seed": seed,
        "items": len(learner),
        "phases": [asdict(phase) for phase in scenario.phases],
        "expected": scenario.expected,
        "train_manifest": str(train.resolve()),
        "evaluator_manifest": str(eval_path.resolve()),
        "sidecar": str(sidecar_path.resolve()),
    }
    (scenario_root / "scenario.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
    )
    return descriptor


def generate_suite(
    root: Path,
    seeds: list[int],
    image_size: int = 64,
    scenario_names: list[str] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    declared = scenarios()
    if scenario_names is None:
        selected = declared
    else:
        requested = list(dict.fromkeys(str(name) for name in scenario_names))
        by_name = {scenario.name: scenario for scenario in declared}
        unknown = sorted(set(requested) - set(by_name))
        if unknown:
            raise ValueError(f"Unknown controlled scenarios: {unknown}")
        selected = tuple(by_name[name] for name in requested)
    entries = [
        generate_scenario(root, scenario, seed, image_size=image_size)
        for scenario in selected
        for seed in seeds
    ]
    index = {
        "suite": "AFM-U controlled AI full-claim integration",
        "image_size": image_size,
        "seeds": seeds,
        "selected_scenarios": [scenario.name for scenario in selected],
        "scenarios": entries,
    }
    (root / "suite_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    return index
