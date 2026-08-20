from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .eprocess import HalfNormalMixtureEProcess
from .frequent_directions import FrequentDirections


@dataclass
class CandidateState:
    """Bounded state for one context's next append-only protected segment.

    A candidate first owns a finite routed training block and one private
    parameter initialisation.  The declared deterministic candidate trainer
    fits that private vector, freezes it as ``snapshot``, and only subsequent
    routed outcomes enter the anytime-valid validation process.  This keeps
    fitting and certification disjoint while using bounded persistent state.
    """

    slot: int
    created_step: int
    candidate_id: int
    initial_parameters: torch.Tensor | None = None
    training_evidence: list[dict[str, Any]] = field(default_factory=list)
    training_count: int = 0
    training_loss: float | None = None
    training_accuracy: float | None = None
    commit_budget_index: int | None = None
    commit_alpha: float | None = None
    snapshot: torch.Tensor | None = None
    snapshot_shield_state: dict[str, Any] | None = None
    validation_count: int = 0
    validation_sum: float = 0.0
    last_logged_validation_count: int = 0
    # The anytime-valid certificate is frozen at its first threshold crossing.
    # Later observations are handled by a separately allocated staleness
    # e-process; they never rewrite the original certificate or transfer block.
    certified: bool = False
    certified_step: int | None = None
    certified_validation_count: int | None = None
    certified_validation_sum: float | None = None
    certified_ucb: float | None = None
    staleness_eprocess: HalfNormalMixtureEProcess | None = None
    staleness_budget_index: int | None = None
    staleness_alpha: float | None = None
    staleness_observations: int = 0
    staleness_crossed_step: int | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    sketch: FrequentDirections | None = None
    anchor_output_chunks: list[torch.Tensor] = field(default_factory=list)
    anchor_logit_chunks: list[torch.Tensor] = field(default_factory=list)
    # Bounded distillation state used by the theorem-level safe transfer.
    # It contains only a fixed subset of the routed fitting block and the
    # frozen candidate behaviour on those observations.
    transfer_evidence: list[dict[str, Any]] = field(default_factory=list)
    transfer_targets: torch.Tensor | None = None
    transfer_logits: torch.Tensor | None = None
    frozen_step: int | None = None
    transfer_attempts: int = 0
    # Legacy counter retained for checkpoint/report compatibility.
    transfer_common_descent_steps: int = 0
    transfer_priority_feasible_steps: int = 0
    transfer_accepted_steps: int = 0
    transfer_obstructions: int = 0
    last_activation_gap: float | None = None
    # Exact pathwise transfer ledger for the frozen transfer objective.
    transfer_initial_objective: float | None = None
    transfer_last_objective: float | None = None
    transfer_progress_total: float = 0.0
    transfer_damage_total: float = 0.0
    transfer_ledger_updates: int = 0
    transfer_service_rounds: int = 0
    transfer_last_service_step: int | None = None
    # Distinct observable signature blocks seen during the frozen validation
    # process. The list is bounded by the finite validation horizon.
    validation_signatures: list[torch.Tensor] = field(default_factory=list)
    last_validation_signature_block_id: int = -1

    @property
    def token(self) -> tuple[int, int, int]:
        """Stable identity used when a router slot is recycled within a minibatch."""

        return (int(self.slot), int(self.created_step), int(self.candidate_id))

    def add_validation(self, losses: list[float]) -> None:
        self.validation_count += len(losses)
        self.validation_sum += float(sum(float(x) for x in losses))

    @property
    def validation_mean(self) -> float:
        return self.validation_sum / max(self.validation_count, 1)


class _LinearFeatureHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class ShadowChallenger(nn.Module):
    """Predictable bounded-state challenger operating on frozen-backbone features.

    The production path initialises ``feature_head`` as an exact deep copy of the
    committed snapshot's adapter/norm/classifier head.  This makes the challenger
    equal to the record at activation and lets subsequent routed outcomes create
    genuine observable advantage.  The legacy linear constructor remains for
    small unit tests and external callers.
    """

    def __init__(
        self,
        feature_dim: int | None = None,
        num_classes: int | None = None,
        learning_rate: float = 0.05,
        feature_head: nn.Module | None = None,
    ):
        super().__init__()
        if feature_head is None:
            if feature_dim is None or num_classes is None:
                raise ValueError("feature_dim and num_classes are required without feature_head")
            feature_head = _LinearFeatureHead(int(feature_dim), int(num_classes))
        self.feature_head = feature_head
        self.learning_rate = float(learning_rate)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    @classmethod
    def from_model(cls, model: nn.Module, learning_rate: float = 0.05) -> "ShadowChallenger":
        from afmvision.models.convnet_adapters import AFMFeatureHead, AFMConvNet

        if not isinstance(model, AFMConvNet):
            raise TypeError("Snapshot challenger requires AFMConvNet")
        head = AFMFeatureHead(
            adapter_pool=copy.deepcopy(model.adapter_pool),
            norm=copy.deepcopy(model.norm),
            classifier=copy.deepcopy(model.classifier),
            functional_shield=copy.deepcopy(model.functional_shield),
        )
        return cls(learning_rate=learning_rate, feature_head=head)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.feature_head(features)

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.feature_head(features.detach())
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


@dataclass
class ProtectedRecord:
    record_id: int
    slot: int
    created_step: int
    committed_step: int
    anchor: torch.Tensor
    sketch: torch.Tensor
    fd_delta: float
    evidence: list[dict[str, Any]]
    anchor_outputs: torch.Tensor
    activation_outputs: torch.Tensor
    traces: torch.Tensor
    challenger: ShadowChallenger
    eprocess: HalfNormalMixtureEProcess
    reopening_budget_index: int
    reopening_alpha: float
    reopening_start_step: int
    # Present for every compact-cardinal-era record. Optional only to read legacy
    # in-memory fixtures/checkpoints; the validity checker rejects its absence
    # in a v0.11.0 scientific run.
    anchor_shield_state: dict[str, Any] | None = None
    # A separate anytime-valid test controls observable route splitting.
    # It is not the outcome-reopening e-process.
    signature_eprocess: HalfNormalMixtureEProcess = field(
        default_factory=lambda: HalfNormalMixtureEProcess(sigma=0.5, prior_scale=1.0, alpha=0.5)
    )
    signature_budget_index: int = -1
    signature_alpha: float = 0.5
    signature_start_step: int = 0
    signature_reference_radius: float = 0.0
    signature_blocks_seen: int = 0
    signature_distance_sum: float = 0.0
    signature_vector_sum: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    last_signature_block_id: int = -1
    outcome_reopening_crossed: bool = False
    outcome_crossing_step: int | None = None
    outcome_crossing_signature_count: int | None = None
    outcome_crossing_log_wealth: float | None = None
    outcome_crossing_observation: int | None = None
    last_seen_step: int = 0
    cumulative_budget: float = 0.0
    max_anchor_drift: float = 0.0
    released_step: int | None = None
    release_reason: str | None = None

    @property
    def activation_gap(self) -> float:
        if self.anchor_outputs.numel() == 0:
            return 0.0
        return float(
            torch.sqrt(
                torch.clamp(
                    (self.activation_outputs - self.anchor_outputs).square().sum(dim=1).mean(),
                    min=0.0,
                )
            ).item()
        )

    @property
    def active(self) -> bool:
        return self.released_step is None

    def release(self, step: int, reason: str) -> None:
        if self.released_step is not None:
            return
        self.released_step = int(step)
        self.release_reason = reason


def evidence_paths(record: ProtectedRecord) -> list[Path]:
    return [Path(item["path"]) for item in record.evidence]
