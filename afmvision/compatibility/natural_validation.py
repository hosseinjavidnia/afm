from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from afmvision.afm.parameter_vector import ParameterVector
from afmvision.compatibility.data import (
    Batch,
    CharNextTokenDataset,
    IndexedCIFAR10,
    Reservoir,
    deterministic_order,
    stack_indices,
)
from afmvision.compatibility.geometry import (
    build_protected_geometry,
    compatibility_fraction,
    functional_jacobian,
)
from afmvision.compatibility.models import build_compatibility_model, describe_model
from afmvision.compatibility.natural_methods import (
    run_natural_method,
    supervised_gradient,
    supervised_unrestricted_comparator,
)
from afmvision.utils.seed import seed_everything


EXPECTED_METHODS = [
    "unrestricted",
    "replay",
    "projection",
    "linearized_distillation",
    "ewc_prox",
    "derpp",
    "afm",
]


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _safe_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite natural-state validation run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def _parent_signature(config: dict[str, Any]) -> dict[str, Any]:
    sweep = dict(config["compatibility_sweep"])
    training = dict(sweep["training"])
    causal = dict(sweep["causal"])
    return {
        "seed": int(config["seed"]),
        "deterministic": bool(sweep.get("deterministic", True)),
        "dataset": sweep["dataset"],
        "model": sweep["model"],
        "training": {
            k: training.get(k)
            for k in (
                "batch_size",
                "parent_lr",
                "weight_decay",
                "pretrain_steps",
                "stream_steps",
                "log_interval",
            )
        },
        "reservoir_capacity": causal.get("reservoir_capacity"),
    }


def _train_parent_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    x = batch.inputs.to(device)
    y = batch.labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    acc = float((logits.detach().argmax(dim=1) == y).float().mean().item())
    loss.backward()
    optimizer.step()
    return float(loss.item()), acc


def _independent_sample(ids: list[int], count: int, rng: random.Random) -> list[int]:
    """Sample protected IDs without consuming the reservoir replacement RNG."""

    n = min(int(count), len(ids))
    if n <= 0:
        return []
    return [ids[i] for i in rng.sample(range(len(ids)), n)]


class NaturalStateValidationRunner:
    """Observational natural-compatibility validation from a saved preprobe parent.

    The parent model continues along the ordinary supervised stream.  Exactly
    every ``probe_interval``-th pre-update state is measured, with no filtering
    or selection by compatibility.  The current batch and true labels are used
    directly: no causal current-batch construction and no teacher targets are
    created.  Seven method branches are evaluated from the identical pre-update
    model/history and then discarded; only the ordinary parent CE update is
    committed.
    """

    def __init__(
        self,
        *,
        source_run_dir: str | Path,
        run_dir: str | Path,
        system: str,
        device: torch.device,
        states: int = 50,
        probe_interval: int = 10,
        probe_steps: int | None = None,
        probe_rng_offset: int = 424242,
    ) -> None:
        self.source_run_dir = Path(source_run_dir)
        self.run_dir = Path(run_dir)
        self.system = str(system)
        if self.system not in {"cifar10_cnn", "cifar10_vit", "text_transformer"}:
            raise ValueError(f"unknown natural-validation system: {self.system}")
        self.device = device
        self.states = int(states)
        self.probe_interval = int(probe_interval)
        if self.states <= 0:
            raise ValueError("states must be positive")
        if self.probe_interval <= 0:
            raise ValueError("probe_interval must be positive")
        self.probe_steps = int(probe_steps) if probe_steps is not None else self.states * self.probe_interval
        if self.probe_steps < (self.states - 1) * self.probe_interval + 1:
            raise ValueError("probe_steps is too short to collect the requested fixed-schedule states")
        self.probe_rng_offset = int(probe_rng_offset)

        resolved = self.source_run_dir / "resolved_config.json"
        checkpoint_path = self.source_run_dir / "preprobe_parent.pt"
        if not resolved.is_file():
            raise FileNotFoundError(f"missing source resolved config: {resolved}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing source preprobe checkpoint: {checkpoint_path}")
        self.cfg = json.loads(resolved.read_text(encoding="utf-8"))
        self.sweep = dict(self.cfg["compatibility_sweep"])
        self.seed = int(self.cfg["seed"])
        self.training = dict(self.sweep["training"])
        self.method_cfg = dict(self.sweep["causal"])
        methods = [str(x) for x in self.method_cfg["methods"]]
        if methods != EXPECTED_METHODS:
            raise RuntimeError(f"natural validation expects the seven audited methods {EXPECTED_METHODS}, got {methods}")
        self.methods = methods

        _safe_run_dir(self.run_dir)
        provenance = {
            "schema": "natural_compatibility_validation_v1",
            "source_run_dir": str(self.source_run_dir.resolve()),
            "system": self.system,
            "source_preprobe_parent": str(checkpoint_path.resolve()),
            "source_resolved_config": str(resolved.resolve()),
            "sampling_rule": {
                "description": "probe every fixed interval before the ordinary parent update; no state rejection",
                "states": self.states,
                "probe_interval": self.probe_interval,
                "probe_steps": self.probe_steps,
                "current_batch": "entire ordinary parent-stream minibatch with true labels",
                "protected_sampling": "separate deterministic probe RNG; does not consume reservoir replacement RNG",
            },
        }
        (self.run_dir / "natural_validation_config.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.events = JsonlWriter(self.run_dir / "events.jsonl")
        self.points = JsonlWriter(self.run_dir / "natural_state_points.jsonl")

        seed_everything(self.seed, deterministic=bool(self.sweep.get("deterministic", True)))
        data_cfg = dict(self.sweep["dataset"])
        self.modality = str(data_cfg["modality"])
        kind = str(data_cfg["kind"])
        if kind == "cifar10":
            self.dataset = IndexedCIFAR10(
                data_cfg["root"], train=True, download=bool(data_cfg.get("download", False))
            )
            vocab_size = None
        elif kind == "char_text":
            self.dataset = CharNextTokenDataset(
                data_cfg["text_path"],
                context_length=int(self.sweep["model"].get("context_length", 64)),
                stride=int(data_cfg.get("stride", 1)),
            )
            vocab_size = self.dataset.vocab_size
        else:
            raise ValueError(f"unknown compatibility dataset kind: {kind}")

        self.model = build_compatibility_model(self.cfg, vocab_size=vocab_size).to(self.device)
        frozen = [name for name, p in self.model.named_parameters() if not p.requires_grad]
        if frozen:
            raise RuntimeError(f"natural compatibility validation forbids frozen parameters: {frozen}")
        self.vectoriser = ParameterVector(self.model.named_parameters())
        self.description = describe_model(
            self.model, self.modality, str(self.sweep["model"]["architecture"])
        )
        self.checkpoint_path = checkpoint_path

    def _load_parent(self):
        batch_size = int(self.training.get("batch_size", 64))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training.get("parent_lr", 1e-3)),
            weight_decay=float(self.training.get("weight_decay", 0.0)),
        )
        reservoir = Reservoir(int(self.method_cfg.get("reservoir_capacity", 512)), self.seed + 991)
        order = deterministic_order(len(self.dataset), self.seed)
        cursor = 0
        epoch = 0
        parent_steps = 0

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        checkpoint_cfg = checkpoint.get("config", self.cfg)
        if _parent_signature(checkpoint_cfg) != _parent_signature(self.cfg):
            raise RuntimeError("source preprobe checkpoint parent signature differs from resolved source config")
        if int(checkpoint.get("seed", self.seed)) != self.seed:
            raise RuntimeError("source preprobe checkpoint seed mismatch")
        self.model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        recovery = checkpoint.get("recovery_state")
        if recovery is not None:
            order = [int(x) for x in recovery["order"]]
            cursor = int(recovery["cursor"])
            epoch = int(recovery["epoch"])
            reservoir.ids = [int(x) for x in recovery["reservoir_ids"]]
            reservoir.seen = int(recovery["reservoir_seen"])
            reservoir.rng.setstate(recovery["reservoir_rng_state"])
            parent_steps = int(recovery["parent_steps"])
            if "python_random_state" in recovery:
                random.setstate(recovery["python_random_state"])
            if "numpy_random_state" in recovery:
                np.random.set_state(recovery["numpy_random_state"])
            if "torch_rng_state" in recovery:
                torch.set_rng_state(recovery["torch_rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state_all" in recovery:
                torch.cuda.set_rng_state_all([state.cpu() for state in recovery["cuda_rng_state_all"]])
            recovery_mode = "checkpoint_state"
        else:
            # Exact deterministic reconstruction path for older v1.1 preprobe parents.
            pretrain_steps = int(self.training.get("pretrain_steps", 1000))
            stream_steps = int(self.training.get("stream_steps", 1000))
            total_required = pretrain_steps + stream_steps
            for _ in range(total_required):
                if cursor + batch_size > len(order):
                    epoch += 1
                    order = deterministic_order(len(self.dataset), self.seed + 1009 * epoch)
                    cursor = 0
                ids = order[cursor : cursor + batch_size]
                cursor += batch_size
                reservoir.add_many(ids)
                parent_steps += 1
            expected = int(checkpoint.get("parent_steps", total_required))
            if parent_steps != expected:
                raise RuntimeError(f"reconstructed parent step mismatch: {parent_steps} != {expected}")
            recovery_mode = "reconstructed_v1_1_state"

        return optimizer, reservoir, order, cursor, epoch, parent_steps, recovery_mode

    def run(self) -> dict[str, Any]:
        optimizer, reservoir, order, cursor, epoch, parent_steps, recovery_mode = self._load_parent()
        batch_size = int(self.training.get("batch_size", 64))
        protected_count = int(self.method_cfg.get("protected_count", 12))
        geometry_ridge = float(self.method_cfg.get("geometry_ridge", 1e-7))
        retention_tolerance = float(self.method_cfg.get("retention_tolerance", 0.005))
        probe_rng = random.Random(self.seed + self.probe_rng_offset)
        start = time.time()
        state_id = 0
        method_rows = 0

        def next_batch() -> Batch:
            nonlocal cursor, order, epoch
            if cursor + batch_size > len(order):
                epoch += 1
                order = deterministic_order(len(self.dataset), self.seed + 1009 * epoch)
                cursor = 0
            ids = order[cursor : cursor + batch_size]
            cursor += batch_size
            return stack_indices(self.dataset, ids)

        self.events.write(
            {
                "event": "natural_validation_started",
                "seed": self.seed,
                "recovery_mode": recovery_mode,
                "parent_steps": parent_steps,
                "states": self.states,
                "probe_interval": self.probe_interval,
                "probe_steps": self.probe_steps,
            }
        )

        for local_step in range(self.probe_steps):
            batch = next_batch()
            should_probe = (local_step % self.probe_interval == 0) and (state_id < self.states)
            if should_probe:
                if len(reservoir.ids) < protected_count:
                    raise RuntimeError(
                        f"natural state {state_id}: reservoir has {len(reservoir.ids)} ids, needs {protected_count}"
                    )
                protected_ids = _independent_sample(reservoir.ids, protected_count, probe_rng)
                protected = stack_indices(self.dataset, protected_ids)
                current_inputs = batch.inputs.to(self.device)
                current_labels = batch.labels.to(self.device)
                protected_inputs = protected.inputs.to(self.device)
                protected_labels = protected.labels.to(self.device)

                self.model.eval()
                pre_vector = self.vectoriser.flatten(detach=True)
                with torch.no_grad():
                    protected_logits_before = self.model(protected_inputs).detach().clone()
                _, Jp = functional_jacobian(self.model, protected_inputs)
                geometry = build_protected_geometry(Jp, ridge=geometry_ridge)
                current_gradient = supervised_gradient(
                    self.model, self.vectoriser, current_inputs, current_labels
                )
                gradient_norm = float(torch.linalg.vector_norm(current_gradient).item())
                natural_kappa = float(compatibility_fraction(current_gradient, geometry))
                comparator_valid = True
                comparator_error = None
                try:
                    comparator = supervised_unrestricted_comparator(
                        model=self.model,
                        vectoriser=self.vectoriser,
                        current_inputs=current_inputs,
                        current_labels=current_labels,
                        current_gradient=current_gradient,
                        initial_alpha=float(self.method_cfg.get("comparator_lr", 1e-2)),
                        max_backtracks=int(self.method_cfg.get("comparator_max_backtracks", 20)),
                        backtrack_factor=float(self.method_cfg.get("comparator_backtrack_factor", 0.5)),
                    )
                except RuntimeError as exc:
                    comparator_valid = False
                    comparator_error = str(exc)
                    comparator = None

                self.events.write(
                    {
                        "event": "natural_state",
                        "state_id": state_id,
                        "local_step": local_step,
                        "parent_step": parent_steps,
                        "current_count": len(batch.ids),
                        "protected_count": len(protected_ids),
                        "current_protected_id_overlap": len(set(batch.ids) & set(protected_ids)),
                        "natural_kappa": natural_kappa,
                        "gradient_norm": gradient_norm,
                        "comparator_valid": comparator_valid,
                        "comparator_error": comparator_error,
                    }
                )

                for method in self.methods:
                    self.vectoriser.assign(pre_vector)
                    if comparator is None:
                        payload = {
                            "system": self.system,
                            "dataset": str(self.sweep["dataset"]["kind"]),
                            "modality": self.modality,
                            "seed": self.seed,
                            "state_id": state_id,
                            "local_step": local_step,
                            "parent_step": parent_steps,
                            "method": method,
                            "natural_kappa": natural_kappa,
                            "gradient_norm": gradient_norm,
                            "delta0": None,
                            "delta_persistent": None,
                            "rho_persistent": None,
                            "retention_drift": None,
                            "retention_pass": None,
                            "comparator_valid": False,
                            "comparator_error": comparator_error,
                            "accepted": False,
                            "obstruction": "unrestricted_comparator_failed",
                        }
                    else:
                        result = run_natural_method(
                            method=method,
                            model=self.model,
                            vectoriser=self.vectoriser,
                            comparator=comparator,
                            current_gradient=current_gradient,
                            geometry=geometry,
                            current_inputs=current_inputs,
                            current_labels=current_labels,
                            protected_inputs=protected_inputs,
                            protected_labels=protected_labels,
                            protected_logits_before=protected_logits_before,
                            replay_stored_logits=protected_logits_before,
                            retention_tolerance=retention_tolerance,
                            method_config=self.method_cfg,
                        )
                        payload = {
                            "system": self.system,
                            "dataset": str(self.sweep["dataset"]["kind"]),
                            "modality": self.modality,
                            "seed": self.seed,
                            "state_id": state_id,
                            "local_step": local_step,
                            "parent_step": parent_steps,
                            "method": method,
                            "natural_kappa": natural_kappa,
                            "gradient_norm": gradient_norm,
                            "delta0": comparator.decrease,
                            "delta_persistent": result.persistent_decrease,
                            "rho_persistent": result.persistent_ratio,
                            "retention_drift": result.protected_max_abs_drift,
                            "retention_pass": result.retention_pass,
                            "comparator_valid": True,
                            "comparator_alpha": comparator.alpha,
                            "comparator_backtracking_steps": comparator.backtracking_steps,
                            "proposal_kappa": result.proposal_kappa,
                            "update_norm": result.update_norm,
                            "accepted": result.accepted,
                            "obstruction": result.obstruction,
                            "method_backtracking_steps": result.backtracking_steps,
                            "afm_lambda_hat": result.afm_lambda_hat,
                            "retention_tolerance": retention_tolerance,
                            "current_count": len(batch.ids),
                            "protected_count": len(protected_ids),
                            "current_protected_id_overlap": len(set(batch.ids) & set(protected_ids)),
                        }
                    self.points.write(payload)
                    method_rows += 1

                self.vectoriser.assign(pre_vector)
                state_id += 1

            # Only this ordinary supervised parent update is committed.
            loss, acc = _train_parent_step(
                model=self.model, optimizer=optimizer, batch=batch, device=self.device
            )
            reservoir.add_many(batch.ids)
            parent_steps += 1
            if local_step % int(self.training.get("log_interval", 100)) == 0:
                self.events.write(
                    {
                        "event": "natural_parent_step",
                        "local_step": local_step,
                        "parent_step": parent_steps,
                        "loss": loss,
                        "accuracy": acc,
                        "reservoir": len(reservoir.ids),
                    }
                )
            if state_id >= self.states and local_step >= (self.states - 1) * self.probe_interval:
                # Finish immediately after committing the ordinary update at the
                # last sampled state.  No post-hoc state selection is performed.
                break

        if state_id != self.states:
            raise RuntimeError(f"natural validation collected {state_id} states, expected exactly {self.states}")
        expected_rows = self.states * len(self.methods)
        if method_rows != expected_rows:
            raise RuntimeError(f"natural validation wrote {method_rows} rows, expected {expected_rows}")

        summary = {
            "status": "complete",
            "schema": "natural_compatibility_validation_v1",
            "seed": self.seed,
            "system": self.system,
            "dataset": str(self.sweep["dataset"]["kind"]),
            "modality": self.modality,
            "architecture": str(self.sweep["model"]["architecture"]),
            "model": asdict(self.description),
            "source_run_dir": str(self.source_run_dir.resolve()),
            "source_preprobe_parent": str(self.checkpoint_path.resolve()),
            "recovery_mode": recovery_mode,
            "states": state_id,
            "methods": self.methods,
            "method_rows": method_rows,
            "probe_interval": self.probe_interval,
            "parent_steps_final": parent_steps,
            "elapsed_seconds": time.time() - start,
            "device": str(self.device),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "source_config": self.cfg,
                "summary": summary,
            },
            self.run_dir / "final_parent.pt",
        )
        self.events.write({"event": "finished", **summary})
        self.events.close()
        self.points.close()
        return summary
