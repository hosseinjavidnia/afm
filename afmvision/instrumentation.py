from __future__ import annotations

import json
import resource
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from afmvision.utils.io import ensure_dir


class RunInstrumentation:
    """Evaluation-only instrumentation that never changes learner decisions.

    Checkpoint times are integer optimiser-step counts.  The learner receives no
    episode names, task labels, context identifiers, or boundary flags.  Those
    names remain in the evaluator-side schedule used after training.
    """

    def __init__(self, config: dict[str, Any], run_dir: Path, method: str):
        section = dict(config.get("instrumentation", {}))
        self.enabled = bool(section.get("enabled", False))
        self.log_predictions_enabled = bool(section.get("log_predictions", False))
        self.checkpoint_steps = {int(x) for x in section.get("checkpoint_steps", [])}
        self.run_dir = ensure_dir(run_dir)
        self.method = str(method)
        self.prediction_path = self.run_dir / "online_predictions.jsonl"
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        if self.log_predictions_enabled:
            self.prediction_path.write_text("", encoding="utf-8")

    def log_predictions(
        self,
        *,
        step: int,
        logits: torch.Tensor,
        labels: torch.Tensor,
        metadata: Iterable[dict[str, Any]],
    ) -> None:
        if not self.enabled or not self.log_predictions_enabled:
            return
        probabilities = torch.softmax(logits.detach(), dim=1)
        confidence, predictions = probabilities.max(dim=1)
        labels_cpu = labels.detach().cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        confidence_cpu = confidence.cpu().tolist()
        rows = []
        for row, label, prediction, conf in zip(metadata, labels_cpu, predictions_cpu, confidence_cpu):
            rows.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "step": int(step),
                    "label": int(label),
                    "prediction": int(prediction),
                    "confidence": float(conf),
                    "correct": bool(int(label) == int(prediction)),
                }
            )
        with self.prediction_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def maybe_checkpoint(
        self,
        *,
        completed_steps: int,
        model: nn.Module,
        config: dict[str, Any],
        summary: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled or int(completed_steps) not in self.checkpoint_steps:
            return None
        path = self.checkpoint_dir / f"step_{int(completed_steps):06d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "summary": dict(summary or {}),
                "instrumentation": {
                    "method": self.method,
                    "completed_steps": int(completed_steps),
                },
            },
            path,
        )
        return path

    @staticmethod
    def resource_summary(device: torch.device) -> dict[str, Any]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports ru_maxrss in KiB.
        out: dict[str, Any] = {
            "peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        }
        if device.type == "cuda" and torch.cuda.is_available():
            out.update(
                {
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
        else:
            out.update({"peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0})
        return out
