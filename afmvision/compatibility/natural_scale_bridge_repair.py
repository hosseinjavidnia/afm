from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from afmvision.compatibility.data import Batch, deterministic_order, stack_indices
from afmvision.compatibility.extension_common import train_parent_step
from afmvision.compatibility.natural_scale_bridge import NaturalScaleBridgeRunner


class NaturalScaleBridgeRepairRunner(NaturalScaleBridgeRunner):
    """Targeted continuation for bridge-v1 finite-Delta0 admission rejections.

    The original bridge generated positive same-norm endpoints for every requested
    kappa but discarded some state-scale conditions because the *best available*
    finite-Delta0 set exceeded a predeclared CV/range tolerance.  This repair
    replays only those predeclared conditions, preserves the exact original
    direction bank and best-Delta0-alignment selection rule, and changes only one
    design choice: finite-Delta0 spread is recorded as a diagnostic rather than
    used as a hard admission criterion.

    Non-target states are still replayed and their protected-sampling RNG draws are
    consumed so that every repaired target reconstructs the same parent state,
    reservoir and probe sample as bridge-v1.
    """

    schema = "causal_compatibility_natural_scale_bridge_v1_delta0_admission_repair"

    def __init__(
        self,
        *,
        repair_targets: dict[int, list[float]],
        original_bridge_run_dir: str | Path,
        **kwargs: Any,
    ) -> None:
        clean: dict[int, list[float]] = {}
        for raw_state, raw_fracs in repair_targets.items():
            state_id = int(raw_state)
            fracs = sorted({float(x) for x in raw_fracs})
            if state_id < 0 or not fracs or any(x <= 0.0 for x in fracs):
                raise ValueError(f"invalid repair target {raw_state!r}: {raw_fracs!r}")
            clean[state_id] = fracs
        if not clean:
            raise ValueError("repair_targets must be nonempty")
        self.repair_targets = clean
        self.original_bridge_run_dir = Path(original_bridge_run_dir).resolve()

        # Only fractions that actually need repair are enabled.  hard_delta0_match
        # is deliberately false; all other bridge-v1 tolerances/configuration are
        # inherited unchanged.
        union_fractions = sorted({f for fs in clean.values() for f in fs})
        kwargs["natural_norm_fractions"] = union_fractions
        kwargs["hard_delta0_match"] = False
        super().__init__(**kwargs)

        (self.run_dir / "repair_config.json").write_text(
            json.dumps(
                {
                    "schema": self.schema,
                    "original_bridge_run_dir": str(self.original_bridge_run_dir),
                    "repair_targets": [
                        {"state_id": state, "natural_norm_fraction": frac}
                        for state, fracs in sorted(self.repair_targets.items())
                        for frac in fracs
                    ],
                    "repair_condition": "original reason == finite_delta0_match_tolerance_not_met",
                    "selection_rule": "same best available finite-Delta0 alignment as bridge-v1",
                    "changed_admission_rule": "finite Delta0 CV/range are diagnostic only; positive same-norm endpoint for every kappa remains required",
                    "hard_delta0_match": False,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def probe_state(self, **kwargs) -> int:  # pragma: no cover
        raise RuntimeError("NaturalScaleBridgeRepairRunner uses targeted fixed-schedule replay")

    def run_fixed_schedule(self) -> dict[str, Any]:
        state = self.load_parent()
        batch_size = int(self.training.get("batch_size", 64))
        protected_count = int(self.causal.get("protected_count", 16))
        guard_count = int(self.causal.get("guard_count", 16))
        current_count = int(self.causal.get("current_count", 16))
        probe_rng = random.Random(self.seed + self.probe_rng_offset)

        def next_batch() -> Batch:
            if state.cursor + batch_size > len(state.order):
                state.epoch += 1
                state.order = deterministic_order(len(self.dataset), self.seed + 1009 * state.epoch)
                state.cursor = 0
            ids = state.order[state.cursor : state.cursor + batch_size]
            state.cursor += batch_size
            return stack_indices(self.dataset, ids)

        self.events.write(
            {
                "event": "repair_resumed_from_preprobe_parent",
                "source": str(self.checkpoint_path.resolve()),
                "original_bridge_run_dir": str(self.original_bridge_run_dir),
                "parent_steps": state.parent_steps,
                "recovery_mode": state.recovery_mode,
            }
        )

        max_target_state = max(self.repair_targets)
        max_local_step = max_target_state * self.probe_interval
        state_id = 0
        repaired_states = 0
        feasible_conditions = 0
        attempted_conditions = 0

        for local_step in range(max_local_step + 1):
            batch = next_batch()
            if local_step % self.probe_interval == 0:
                if len(state.reservoir.ids) < protected_count + guard_count:
                    raise RuntimeError(
                        f"bridge repair reservoir has {len(state.reservoir.ids)} ids; needs {protected_count + guard_count}"
                    )

                # Consume this draw for *every* scheduled state, including states
                # not being repaired, to reproduce the original probe RNG stream.
                sampled = probe_rng.sample(list(state.reservoir.ids), protected_count + guard_count)
                protected_ids = sampled[:protected_count]
                guard_ids = sampled[protected_count:]

                if state_id in self.repair_targets:
                    protected = stack_indices(self.dataset, protected_ids)
                    guards = stack_indices(self.dataset, guard_ids) if guard_ids else None
                    novel_ids = list(batch.ids[:current_count])
                    if len(novel_ids) != current_count:
                        raise RuntimeError("ordinary parent batch shorter than current_count")
                    novel = stack_indices(self.dataset, novel_ids)

                    requested_here = list(self.repair_targets[state_id])
                    old_fractions = self.natural_norm_fractions
                    self.natural_norm_fractions = requested_here
                    try:
                        feasible, attempted = self._probe_fixed_state(
                            state_id=state_id,
                            local_step=local_step,
                            parent_step=state.parent_steps,
                            protected=protected,
                            novel=novel,
                            guards=guards,
                        )
                    finally:
                        self.natural_norm_fractions = old_fractions
                    if attempted != len(requested_here):
                        raise RuntimeError(
                            f"repair state {state_id} attempted {attempted} conditions, expected {len(requested_here)}"
                        )
                    feasible_conditions += feasible
                    attempted_conditions += attempted
                    repaired_states += 1
                    self.events.write(
                        {
                            "event": "bridge_repair_state",
                            "state_id": state_id,
                            "local_step": local_step,
                            "parent_step": state.parent_steps,
                            "requested_fractions": requested_here,
                            "feasible_target_norm_conditions": feasible,
                            "attempted_target_norm_conditions": attempted,
                        }
                    )
                state_id += 1

            # Preserve the exact ordinary parent trajectory.  Probe branches were
            # restored by _probe_fixed_state; only this update is committed.
            train_parent_step(model=self.model, optimizer=state.optimizer, batch=batch, device=self.device)
            state.reservoir.add_many(batch.ids)
            state.parent_steps += 1

        expected_attempts = sum(len(v) for v in self.repair_targets.values())
        if attempted_conditions != expected_attempts:
            raise RuntimeError(
                f"repair attempted {attempted_conditions} target conditions, expected {expected_attempts}"
            )
        if repaired_states != len(self.repair_targets):
            raise RuntimeError(f"repair visited {repaired_states} target states, expected {len(self.repair_targets)}")

        return {
            "repair_states": repaired_states,
            "replayed_scheduled_states": max_target_state + 1,
            "attempted_target_norm_conditions": attempted_conditions,
            "feasible_target_norm_conditions": feasible_conditions,
            "parent_steps": state.parent_steps,
            "recovery_mode": state.recovery_mode,
        }

    def run(self) -> dict[str, Any]:
        start = time.time()
        traj = self.run_fixed_schedule()
        summary = {
            "schema": self.schema,
            "status": "complete",
            "system": self.system,
            "seed": self.seed,
            "repair_states": traj["repair_states"],
            "replayed_scheduled_states": traj["replayed_scheduled_states"],
            "attempted_target_norm_conditions": traj["attempted_target_norm_conditions"],
            "feasible_target_norm_conditions": traj["feasible_target_norm_conditions"],
            "repair_target_conditions": sum(len(v) for v in self.repair_targets.values()),
            "natural_median_update_norm": self.natural_median_update_norm,
            "natural_norm_fractions": self.natural_norm_fractions,
            "requested_kappas": self.requested_kappas,
            "candidate_pool": self.candidate_pool,
            "hard_delta0_match": False,
            "methods": self.methods,
            "retention_betas": self.retention_betas,
            "elapsed_seconds": time.time() - start,
            "device": str(self.device),
            "recovery_mode": traj["recovery_mode"],
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.events.write({"event": "finished", **summary})
        for writer in (self.events, self.feasibility, self.points, self.frontier, self.native):
            writer.close()
        return summary
