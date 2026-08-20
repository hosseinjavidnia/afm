from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from afmvision.afm.parameter_vector import ParameterVector
from afmvision.data.stream import load_rows
from afmvision.eval.metrics import ExperimentSummary
from afmvision.instrumentation import RunInstrumentation
from afmvision.models.convnet_adapters import AFMConvNet
from afmvision.utils.io import ensure_dir
from afmvision.utils.logging import JSONLLogger


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


class ReservoirReferenceBuffer:
    """Bounded reservoir of learner-visible sample references.

    The buffer stores exactly the same fields available to the learner in the
    training manifest.  It never stores evaluator metadata or task boundaries.
    """

    def __init__(self, capacity: int, seed: int):
        self.capacity = max(int(capacity), 0)
        self.rows: list[dict[str, Any]] = []
        self.seen = 0
        self.rng = random.Random(int(seed))

    def add_many(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.seen += 1
            learner_row = {
                key: row[key]
                for key in ("sample_id", "path", "label", "transform", "transform_seed")
                if key in row
            }
            if self.capacity <= 0:
                continue
            if len(self.rows) < self.capacity:
                self.rows.append(learner_row)
                continue
            index = self.rng.randrange(self.seen)
            if index < self.capacity:
                self.rows[index] = learner_row

    def sample(self, count: int) -> list[dict[str, Any]]:
        count = min(max(int(count), 0), len(self.rows))
        if count == 0:
            return []
        return [self.rows[i] for i in self.rng.sample(range(len(self.rows)), count)]

    @property
    def estimated_reference_bytes(self) -> int:
        return sum(len(json.dumps(row, sort_keys=True).encode("utf-8")) for row in self.rows)


class _FrozenBackboneBaseline:
    def __init__(self, model: AFMConvNet, config: dict[str, Any], device: torch.device, run_dir: Path, method: str):
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.run_dir = ensure_dir(run_dir)
        self.method = str(method)
        self.logger = JSONLLogger(self.run_dir / "events.jsonl")
        self.summary = ExperimentSummary()
        self.bootstrap_batches = int(config["training"].get("bootstrap_batches", 0))
        self.bootstrap_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["training"].get("bootstrap_lr", 1e-3)),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        self.vectoriser: ParameterVector | None = None
        self.instrumentation = RunInstrumentation(config, self.run_dir, self.method)

    def _initialise_state(self) -> None:
        if self.vectoriser is not None:
            return
        self.model.freeze_backbone()
        self.vectoriser = ParameterVector(self.model.trainable_named_parameters())

    def _allowed_mask(self) -> torch.Tensor:
        assert self.vectoriser is not None
        active_slots = set(self.model.adapter_pool.state().active)
        return self.vectoriser.gradient_mask_for_adapter_activity(active_slots)

    def _apply_gradient(self, gradient: torch.Tensor) -> float:
        assert self.vectoriser is not None
        mask = self._allowed_mask()
        delta = -float(self.cfg["training"].get("baseline_lr", 1e-3)) * gradient * mask
        cap = float(self.cfg["afm"]["safe_update"].get("trust_radius_cap", float("inf")))
        norm = float(torch.linalg.vector_norm(delta).item())
        if norm > cap > 0.0:
            delta *= cap / norm
            norm = cap
        self.vectoriser.add_(delta)
        return norm

    def _bootstrap_step(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, torch.Tensor]:
        self.model.train()
        self.bootstrap_optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = _accuracy(logits.detach(), labels)
        loss.backward()
        self.bootstrap_optimizer.step()
        return float(loss.item()), acc, logits.detach()

    def _checkpoint(self, completed_steps: int) -> None:
        self.instrumentation.maybe_checkpoint(
            completed_steps=completed_steps,
            model=self.model,
            config=self.cfg,
            summary=self.summary.as_dict(),
        )

    def _finish(self, start: float, extra: dict[str, Any]) -> dict[str, Any]:
        summary = self.summary.as_dict()
        summary.update(
            {
                "elapsed_seconds": time.time() - start,
                "method": self.method,
                **RunInstrumentation.resource_summary(self.device),
                **extra,
            }
        )
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        torch.save({"model": self.model.state_dict(), "config": self.cfg, "summary": summary}, self.run_dir / "final.pt")
        self.logger.log("finished", **summary)
        return summary


class ReplayTrainer(_FrozenBackboneBaseline):
    def __init__(self, model: AFMConvNet, config: dict[str, Any], device: torch.device, run_dir: Path):
        super().__init__(model, config, device, run_dir, method="replay")
        section = dict(config.get("baseline", {}).get("replay", {}))
        self.buffer = ReservoirReferenceBuffer(int(section.get("memory_items", 128)), int(config["seed"]) + 41)
        self.replay_batch_size = int(section.get("batch_size", config["training"]["batch_size"]))
        self.replay_samples_used = 0

    def fit(self, loader: DataLoader) -> dict[str, Any]:
        max_steps = int(self.cfg["training"]["max_steps"])
        start = time.time()
        for step, (images, labels, metadata) in enumerate(loader):
            if step >= max_steps:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if step < self.bootstrap_batches:
                loss_value, pre_acc, pre_logits = self._bootstrap_step(images, labels)
            else:
                self._initialise_state()
                assert self.vectoriser is not None
                self.model.train()
                self.model.backbone.eval()
                with torch.no_grad():
                    pre_logits = self.model(images)
                pre_acc = _accuracy(pre_logits, labels)
                replay_rows = self.buffer.sample(self.replay_batch_size)
                if replay_rows:
                    replay_images, replay_labels = load_rows(
                        replay_rows,
                        image_size=int(self.cfg["data"]["image_size"]),
                        normalise=True,
                    )
                    replay_images = replay_images.to(self.device, non_blocking=True)
                    replay_labels = replay_labels.to(self.device, non_blocking=True)
                    train_images = torch.cat((images, replay_images), dim=0)
                    train_labels = torch.cat((labels, replay_labels), dim=0)
                    self.replay_samples_used += len(replay_rows)
                else:
                    train_images, train_labels = images, labels
                self.model.zero_grad(set_to_none=True)
                logits = self.model(train_images)
                loss = torch.nn.functional.cross_entropy(logits, train_labels)
                loss.backward()
                gradient = self.vectoriser.flatten_grads()
                self._apply_gradient(gradient)
                loss_value = float(loss.item())
            self.instrumentation.log_predictions(step=step, logits=pre_logits, labels=labels, metadata=metadata)
            self.buffer.add_many(metadata)
            self.summary.optimizer_steps += 1
            self.summary.online_accuracy.update(pre_acc, len(labels))
            self.summary.online_loss.update(loss_value, len(labels))
            self._checkpoint(step + 1)
            if step % int(self.cfg["training"].get("log_interval", 20)) == 0:
                self.logger.log("replay_step", step=step, loss=loss_value, accuracy=pre_acc, buffer_items=len(self.buffer.rows))
        return self._finish(
            start,
            {
                "buffer_items": len(self.buffer.rows),
                "buffer_seen": self.buffer.seen,
                "buffer_reference_bytes": self.buffer.estimated_reference_bytes,
                "replay_samples_used": self.replay_samples_used,
            },
        )


class AGEMTrainer(_FrozenBackboneBaseline):
    def __init__(self, model: AFMConvNet, config: dict[str, Any], device: torch.device, run_dir: Path):
        super().__init__(model, config, device, run_dir, method="agem")
        section = dict(config.get("baseline", {}).get("agem", {}))
        self.buffer = ReservoirReferenceBuffer(int(section.get("memory_items", 128)), int(config["seed"]) + 73)
        self.reference_batch_size = int(section.get("batch_size", config["training"]["batch_size"]))
        self.projection_steps = 0
        self.reference_samples_used = 0

    def fit(self, loader: DataLoader) -> dict[str, Any]:
        max_steps = int(self.cfg["training"]["max_steps"])
        start = time.time()
        for step, (images, labels, metadata) in enumerate(loader):
            if step >= max_steps:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if step < self.bootstrap_batches:
                loss_value, pre_acc, pre_logits = self._bootstrap_step(images, labels)
            else:
                self._initialise_state()
                assert self.vectoriser is not None
                self.model.train()
                self.model.backbone.eval()
                self.model.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                pre_logits = logits.detach()
                pre_acc = _accuracy(pre_logits, labels)
                loss.backward()
                current_gradient = self.vectoriser.flatten_grads()
                gradient = current_gradient
                reference_rows = self.buffer.sample(self.reference_batch_size)
                if reference_rows:
                    reference_images, reference_labels = load_rows(
                        reference_rows,
                        image_size=int(self.cfg["data"]["image_size"]),
                        normalise=True,
                    )
                    reference_images = reference_images.to(self.device, non_blocking=True)
                    reference_labels = reference_labels.to(self.device, non_blocking=True)
                    self.model.zero_grad(set_to_none=True)
                    reference_loss = torch.nn.functional.cross_entropy(self.model(reference_images), reference_labels)
                    reference_loss.backward()
                    reference_gradient = self.vectoriser.flatten_grads()
                    dot = torch.dot(current_gradient, reference_gradient)
                    denominator = torch.dot(reference_gradient, reference_gradient)
                    if float(dot.item()) < 0.0 and float(denominator.item()) > 1e-20:
                        gradient = current_gradient - dot / denominator * reference_gradient
                        self.projection_steps += 1
                    self.reference_samples_used += len(reference_rows)
                self._apply_gradient(gradient)
                loss_value = float(loss.item())
            self.instrumentation.log_predictions(step=step, logits=pre_logits, labels=labels, metadata=metadata)
            self.buffer.add_many(metadata)
            self.summary.optimizer_steps += 1
            self.summary.online_accuracy.update(pre_acc, len(labels))
            self.summary.online_loss.update(loss_value, len(labels))
            self._checkpoint(step + 1)
            if step % int(self.cfg["training"].get("log_interval", 20)) == 0:
                self.logger.log(
                    "agem_step",
                    step=step,
                    loss=loss_value,
                    accuracy=pre_acc,
                    buffer_items=len(self.buffer.rows),
                    projection_steps=self.projection_steps,
                )
        return self._finish(
            start,
            {
                "buffer_items": len(self.buffer.rows),
                "buffer_seen": self.buffer.seen,
                "buffer_reference_bytes": self.buffer.estimated_reference_bytes,
                "agem_projection_steps": self.projection_steps,
                "agem_reference_samples_used": self.reference_samples_used,
            },
        )


class OnlineEWCTrainer(_FrozenBackboneBaseline):
    def __init__(
        self,
        model: AFMConvNet,
        config: dict[str, Any],
        device: torch.device,
        run_dir: Path,
        *,
        oracle_boundaries: bool = False,
    ):
        method = "oracle_ewc" if oracle_boundaries else "online_ewc"
        super().__init__(model, config, device, run_dir, method=method)
        section = dict(config.get("baseline", {}).get("ewc", {}))
        self.strength = float(section.get("lambda", 10.0))
        self.decay = float(section.get("decay", 0.9))
        self.interval = int(section.get("interval", 50))
        self.oracle_boundaries = bool(oracle_boundaries)
        self.consolidation_steps = {int(x) for x in section.get("consolidation_steps", [])}
        self.fisher: torch.Tensor | None = None
        self.anchor: torch.Tensor | None = None
        self.fisher_accumulator: torch.Tensor | None = None
        self.fisher_count = 0
        self.consolidations = 0

    def _initialise_state(self) -> None:
        super()._initialise_state()
        assert self.vectoriser is not None
        if self.fisher is None:
            vector = self.vectoriser.flatten(detach=True)
            self.fisher = torch.zeros_like(vector)
            self.anchor = vector.clone()
            self.fisher_accumulator = torch.zeros_like(vector)

    def _flatten_grad_list(self, grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
        assert self.vectoriser is not None
        parts = []
        for parameter, grad in zip(self.vectoriser.params, grads):
            parts.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1))
        return torch.cat(parts)

    def _should_consolidate(self, completed_steps: int) -> bool:
        if self.oracle_boundaries:
            return int(completed_steps) in self.consolidation_steps
        return self.interval > 0 and int(completed_steps) % self.interval == 0

    def _consolidate(self) -> None:
        assert self.vectoriser is not None and self.fisher is not None and self.anchor is not None
        assert self.fisher_accumulator is not None
        if self.fisher_count <= 0:
            return
        estimate = self.fisher_accumulator / float(self.fisher_count)
        self.fisher.mul_(self.decay).add_(estimate)
        self.anchor = self.vectoriser.flatten(detach=True)
        self.fisher_accumulator.zero_()
        self.fisher_count = 0
        self.consolidations += 1

    def fit(self, loader: DataLoader) -> dict[str, Any]:
        max_steps = int(self.cfg["training"]["max_steps"])
        start = time.time()
        for step, (images, labels, metadata) in enumerate(loader):
            if step >= max_steps:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if step < self.bootstrap_batches:
                loss_value, pre_acc, pre_logits = self._bootstrap_step(images, labels)
            else:
                self._initialise_state()
                assert self.vectoriser is not None
                assert self.fisher is not None and self.anchor is not None and self.fisher_accumulator is not None
                self.model.train()
                self.model.backbone.eval()
                self.model.zero_grad(set_to_none=True)
                logits = self.model(images)
                ce_loss = torch.nn.functional.cross_entropy(logits, labels)
                pre_logits = logits.detach()
                pre_acc = _accuracy(pre_logits, labels)
                ce_grads = torch.autograd.grad(
                    ce_loss,
                    tuple(self.vectoriser.params),
                    retain_graph=True,
                    allow_unused=True,
                )
                current_vector = self.vectoriser.flatten(detach=False)
                penalty = 0.5 * self.strength * torch.sum(self.fisher * (current_vector - self.anchor).square())
                total_loss = ce_loss + penalty
                total_loss.backward()
                gradient = self.vectoriser.flatten_grads()
                self._apply_gradient(gradient)
                fisher_gradient = self._flatten_grad_list(ce_grads).detach()
                mask = self._allowed_mask()
                self.fisher_accumulator.add_((fisher_gradient * mask).square())
                self.fisher_count += 1
                loss_value = float(total_loss.item())
            self.instrumentation.log_predictions(step=step, logits=pre_logits, labels=labels, metadata=metadata)
            self.summary.optimizer_steps += 1
            self.summary.online_accuracy.update(pre_acc, len(labels))
            self.summary.online_loss.update(loss_value, len(labels))
            completed = step + 1
            if step >= self.bootstrap_batches and self._should_consolidate(completed):
                self._consolidate()
                self.logger.log("ewc_consolidation", step=step, completed_steps=completed, oracle=self.oracle_boundaries)
            self._checkpoint(completed)
            if step % int(self.cfg["training"].get("log_interval", 20)) == 0:
                self.logger.log(
                    "oracle_ewc_step" if self.oracle_boundaries else "online_ewc_step",
                    step=step,
                    loss=loss_value,
                    accuracy=pre_acc,
                    consolidations=self.consolidations,
                )
        if self.fisher_count:
            self._consolidate()
        fisher_norm = 0.0 if self.fisher is None else float(torch.linalg.vector_norm(self.fisher).item())
        return self._finish(
            start,
            {
                "ewc_lambda": self.strength,
                "ewc_decay": self.decay,
                "ewc_consolidations": self.consolidations,
                "ewc_fisher_norm": fisher_norm,
                "oracle_boundaries": self.oracle_boundaries,
            },
        )
