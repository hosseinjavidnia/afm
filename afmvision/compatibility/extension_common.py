from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
from afmvision.compatibility.models import build_compatibility_model, describe_model
from afmvision.compatibility.shield import StaticAddressEncoder
from afmvision.utils.seed import seed_everything


SYSTEMS = {
    "cifar10_cnn": "configs/compatibility/causal_sweep_cifar_cnn.yaml",
    "cifar10_vit": "configs/compatibility/causal_sweep_cifar_vit.yaml",
    "text_transformer": "configs/compatibility/causal_sweep_text_transformer.yaml",
}
DEFAULT_SEEDS = [11, 29, 47, 71, 101]
EXTRA_SEEDS = [131, 149, 167, 191, 223]
TEN_SEEDS = DEFAULT_SEEDS + EXTRA_SEEDS


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def safe_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty extension run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def parent_signature(config: dict[str, Any]) -> dict[str, Any]:
    sweep = dict(config["compatibility_sweep"])
    training = dict(sweep["training"])
    causal = dict(sweep["causal"])
    return {
        "seed": int(config["seed"]),
        "deterministic": bool(sweep.get("deterministic", True)),
        "dataset": sweep["dataset"],
        "model": sweep["model"],
        "training": {
            key: training.get(key)
            for key in (
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


def train_parent_step(
    *, model: nn.Module, optimizer: torch.optim.Optimizer, batch: Batch, device: torch.device
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


def select_disjoint(ids: list[int], forbidden: set[int], count: int) -> list[int]:
    out: list[int] = []
    for item in ids:
        if int(item) in forbidden:
            continue
        out.append(int(item))
        if len(out) >= int(count):
            break
    return out


@dataclass
class ParentState:
    optimizer: torch.optim.Optimizer
    reservoir: Reservoir
    order: list[int]
    cursor: int
    epoch: int
    parent_steps: int
    recovery_mode: str


class SavedParentProbeBase:
    """Common deterministic replay of the audited v1.5 preprobe parent.

    Subclasses implement ``probe_state``.  Probe branches are never committed;
    after each attempted state, the ordinary supervised parent update is the only
    model update committed to the trajectory.  Reservoir sampling follows the
    original v1.5 probe path so state attempts are reproducible from the saved
    recovery state.
    """

    schema = "compatibility_extension_v1"

    def __init__(
        self,
        *,
        source_run_dir: str | Path,
        run_dir: str | Path,
        system: str,
        device: torch.device,
        max_states: int = 50,
        max_probe_steps: int = 5000,
    ) -> None:
        self.source_run_dir = Path(source_run_dir)
        self.run_dir = Path(run_dir)
        self.system = str(system)
        self.device = device
        self.max_states = int(max_states)
        self.max_probe_steps = int(max_probe_steps)
        if self.max_states <= 0 or self.max_probe_steps <= 0:
            raise ValueError("max_states and max_probe_steps must be positive")

        resolved = self.source_run_dir / "resolved_config.json"
        checkpoint = self.source_run_dir / "preprobe_parent.pt"
        if not resolved.is_file():
            raise FileNotFoundError(f"missing source resolved config: {resolved}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing source preprobe parent: {checkpoint}")
        self.cfg = json.loads(resolved.read_text(encoding="utf-8"))
        self.sweep = dict(self.cfg["compatibility_sweep"])
        self.training = dict(self.sweep["training"])
        self.causal = dict(self.sweep["causal"])
        self.seed = int(self.cfg["seed"])
        self.checkpoint_path = checkpoint

        safe_run_dir(self.run_dir)
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
            raise RuntimeError(f"extension experiment forbids frozen parameters: {frozen}")
        self.vectoriser = ParameterVector(self.model.named_parameters())
        self.description = describe_model(
            self.model, self.modality, str(self.sweep["model"]["architecture"])
        )
        self.vocab_size = vocab_size
        first_x, _, _ = self.dataset[0]
        self.address_encoder = StaticAddressEncoder(
            modality=self.modality,
            input_shape=tuple(first_x.shape),
            address_dim=int(self.causal.get("address_dim", 64)),
            seed=int(self.causal.get("address_seed", 20260817)),
            vocab_size=vocab_size,
        )
        self.events = JsonlWriter(self.run_dir / "events.jsonl")

    def load_parent(self) -> ParentState:
        batch_size = int(self.training.get("batch_size", 64))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training.get("parent_lr", 1e-3)),
            weight_decay=float(self.training.get("weight_decay", 0.0)),
        )
        reservoir = Reservoir(int(self.causal.get("reservoir_capacity", 512)), self.seed + 991)
        order = deterministic_order(len(self.dataset), self.seed)
        cursor = 0
        epoch = 0
        parent_steps = 0
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        ckpt_cfg = ckpt.get("config", self.cfg)
        if parent_signature(ckpt_cfg) != parent_signature(self.cfg):
            raise RuntimeError("preprobe checkpoint parent signature mismatch")
        if int(ckpt.get("seed", self.seed)) != self.seed:
            raise RuntimeError("preprobe checkpoint seed mismatch")
        self.model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        recovery = ckpt.get("recovery_state")
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
            expected = int(ckpt.get("parent_steps", total_required))
            if parent_steps != expected:
                raise RuntimeError(f"reconstructed parent steps {parent_steps} != {expected}")
            recovery_mode = "reconstructed_v1_1_state"
        return ParentState(optimizer, reservoir, order, cursor, epoch, parent_steps, recovery_mode)

    def probe_state(
        self,
        *,
        state_index: int,
        parent_step: int,
        protected: Batch,
        novel: Batch,
        guards: Batch | None,
    ) -> int:
        raise NotImplementedError

    def run_probe_trajectory(self) -> dict[str, Any]:
        state = self.load_parent()
        batch_size = int(self.training.get("batch_size", 64))
        probe_interval = max(int(self.training.get("probe_interval", 1)), 1)
        protected_count = int(self.causal.get("protected_count", 16))
        guard_count = int(self.causal.get("guard_count", 16))
        current_count = int(self.causal.get("current_count", 16))

        def next_batch() -> Batch:
            if state.cursor + batch_size > len(state.order):
                state.epoch += 1
                state.order = deterministic_order(len(self.dataset), self.seed + 1009 * state.epoch)
                state.cursor = 0
            ids = state.order[state.cursor : state.cursor + batch_size]
            state.cursor += batch_size
            return stack_indices(self.dataset, ids)

        accepted_states = 0
        probe_attempts = 0
        emitted_rows = 0
        self.events.write(
            {
                "event": "resumed_from_preprobe_parent",
                "source": str(self.checkpoint_path.resolve()),
                "parent_steps": state.parent_steps,
                "recovery_mode": state.recovery_mode,
            }
        )
        for probe_step in range(self.max_probe_steps):
            batch = next_batch()
            if (
                probe_step % probe_interval == 0
                and accepted_states < self.max_states
                and len(state.reservoir.ids) >= protected_count + guard_count
            ):
                candidates = state.reservoir.sample(protected_count + guard_count + 16)
                protected_ids = candidates[:protected_count]
                guard_ids = select_disjoint(candidates[protected_count:], set(protected_ids), guard_count)
                if len(guard_ids) < guard_count:
                    guard_ids.extend(
                        select_disjoint(
                            state.reservoir.ids,
                            set(protected_ids) | set(guard_ids),
                            guard_count - len(guard_ids),
                        )
                    )
                protected = stack_indices(self.dataset, protected_ids)
                guards = stack_indices(self.dataset, guard_ids) if guard_ids else None
                novel_ids = list(batch.ids[:current_count])
                if len(novel_ids) < current_count:
                    novel_ids += list(next_batch().ids[: current_count - len(novel_ids)])
                novel = stack_indices(self.dataset, novel_ids)
                produced = self.probe_state(
                    state_index=probe_attempts,
                    parent_step=state.parent_steps,
                    protected=protected,
                    novel=novel,
                    guards=guards,
                )
                probe_attempts += 1
                emitted_rows += int(produced)
                if produced > 0:
                    accepted_states += 1
            train_parent_step(
                model=self.model,
                optimizer=state.optimizer,
                batch=batch,
                device=self.device,
            )
            state.reservoir.add_many(batch.ids)
            state.parent_steps += 1
            if accepted_states >= self.max_states:
                break
        return {
            "accepted_states": accepted_states,
            "probe_attempts": probe_attempts,
            "emitted_rows": emitted_rows,
            "parent_steps": state.parent_steps,
            "full_coverage": accepted_states == self.max_states,
        }


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = sum(values) / len(values)
    if mu <= 0.0:
        return float("inf")
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(var) / mu


def relative_range(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = sum(values) / len(values)
    if mu == 0.0:
        return float("inf")
    return (max(values) - min(values)) / abs(mu)
