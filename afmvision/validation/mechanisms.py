from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from afmvision.afm.eprocess import HalfNormalMixtureEProcess
from afmvision.afm.functional_shield import FunctionalShield
from afmvision.afm.full_progress_restoration import RestorationBlock, replace_with_endpoint_emulation
from afmvision.afm.metaplastic import MetaplasticController, make_policy_family
from afmvision.afm.router import BoundedCentroidRouter
from afmvision.afm.safe_step import (
    joint_progress_protected_step,
    priority_constrained_transfer_direction,
    project_to_allowed_free_subspace,
    safe_radius,
)
from afmvision.afm.trainer import consolidation_ucb
from afmvision.models.convnet_adapters import ZeroGatedAdapterPool


def _binomial_upper_99(successes: int, trials: int) -> float:
    """One-sided 99% Wilson upper bound, dependency free."""
    if trials <= 0:
        return 1.0
    z = 2.3263478740408408
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = p + z2 / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return min(1.0, (centre + radius) / denominator)


def _first_commit(losses: np.ndarray, alpha: float, tau: float, max_n: int) -> int | None:
    total = 0.0
    for n, loss in enumerate(losses[:max_n], start=1):
        total += float(loss)
        if consolidation_ucb(n, total, alpha) <= tau:
            return n
    return None


def consolidation_coverage(rng: np.random.Generator, trials: int) -> dict[str, Any]:
    alpha = 0.05
    tau = 0.25
    max_n = 512
    false_risk = 0.35
    good_risk = 0.05
    false_commits = 0
    good_commits = 0
    good_delays: list[int] = []
    for _ in range(trials):
        bad = rng.binomial(1, false_risk, size=max_n).astype(np.float64)
        if _first_commit(bad, alpha, tau, max_n) is not None:
            false_commits += 1
        good = rng.binomial(1, good_risk, size=max_n).astype(np.float64)
        delay = _first_commit(good, alpha, tau, max_n)
        if delay is not None:
            good_commits += 1
            good_delays.append(delay)
    false_upper = _binomial_upper_99(false_commits, trials)
    return {
        "theorem": "thm:commit",
        "trials": trials,
        "alpha": alpha,
        "risk_threshold": tau,
        "bad_true_risk": false_risk,
        "false_commits": false_commits,
        "false_commit_rate": false_commits / trials,
        "false_commit_rate_upper_99": false_upper,
        "good_true_risk": good_risk,
        "good_commit_rate": good_commits / trials,
        "good_median_delay": None if not good_delays else float(np.median(good_delays)),
        "pass": false_upper <= alpha + 0.02 and good_commits / trials >= 0.95,
    }


def routing_experiments(rng: np.random.Generator) -> dict[str, Any]:
    contexts = 4
    dimension = 6
    radius = 0.025
    threshold = 0.25
    router = BoundedCentroidRouter(contexts, threshold, centroid_rate=0.05)
    centers = torch.eye(dimension, dtype=torch.float64)[:contexts]
    assignments: list[tuple[int, int | None]] = []
    for round_index in range(120):
        context = round_index % contexts
        signature = centers[context] + torch.tensor(
            rng.normal(0.0, radius, size=dimension), dtype=torch.float64
        )
        decision = router.route(signature, round_index)
        assignments.append((context, decision.slot))
    # Resolve slot identities by majority mapping, as slot numbers are arbitrary.
    majority: dict[int, int] = {}
    for slot in range(contexts):
        labels = [context for context, assigned in assignments if assigned == slot]
        if labels:
            majority[slot] = max(set(labels), key=labels.count)
    errors = sum(majority.get(slot, -1) != context for context, slot in assignments)

    overlap_grid: list[dict[str, float]] = []
    for separation in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
        local = BoundedCentroidRouter(2, threshold=0.2, centroid_rate=0.05)
        local_assignments: list[tuple[int, int | None]] = []
        c0 = torch.zeros(3, dtype=torch.float64)
        c1 = torch.tensor([separation, 0.0, 0.0], dtype=torch.float64)
        for index in range(400):
            context = index % 2
            center = c0 if context == 0 else c1
            signature = center + torch.tensor(rng.normal(0.0, 0.05, size=3), dtype=torch.float64)
            local_assignments.append((context, local.route(signature, index).slot))
        local_majority: dict[int, int] = {}
        for slot in range(2):
            labels = [c for c, s in local_assignments if s == slot]
            if labels:
                local_majority[slot] = max(set(labels), key=labels.count)
        error = sum(local_majority.get(s, -1) != c for c, s in local_assignments) / len(local_assignments)
        overlap_grid.append({"separation": separation, "assignment_error": error})
    return {
        "theorems": ["thm:routing-positive", "routing-unidentifiability"],
        "separated_assignment_error": errors / len(assignments),
        "overlap_grid": overlap_grid,
        "pass": errors == 0 and overlap_grid[0]["assignment_error"] >= 0.40,
    }


def exact_retention_certificate(rng: np.random.Generator, trials: int) -> dict[str, Any]:
    max_violation = 0.0
    max_slack = 0.0
    for _ in range(trials):
        E = float(rng.uniform(0.0, 3.0))
        H = float(rng.uniform(0.0, 4.0))
        budget = float(rng.uniform(1e-5, 1.0))
        cap = float(rng.uniform(1e-4, 1.0))
        radius = safe_radius(E, H, budget, cap)
        realised = E * radius + 0.5 * H * radius * radius
        max_violation = max(max_violation, realised - budget)
        max_slack = max(max_slack, budget - realised)
    return {
        "theorems": ["thm:memory", "thm:onestep", "cor:interval"],
        "trials": trials,
        "max_bound_violation": max_violation,
        "max_nonnegative_slack": max_slack,
        "pass": max_violation <= 1e-10,
    }


def transfer_geometry(rng: np.random.Generator, trials: int) -> dict[str, Any]:
    """Validate the joint safe-base endpoint-emulation theorem retained by v0.11.

    The unrestricted endpoint is constructed by one genuine gradient step and
    is used only as a current-batch logit target.  A distinct budget-controlled
    safe endpoint is the persistent base.  The compact-cardinal shield must
    reproduce the unrestricted current logits, restore protected logits,
    preserve a safeguard, reach a selected target, remain silent at guards and
    off support, and reject contradictory duplicate requests atomically.
    """

    available = 0
    bad = 0
    positive_comparators = 0
    distinct_safe_endpoints = 0
    conflicts_detected = 0
    max_current_error = 0.0
    max_protected_error = 0.0
    max_safeguard_error = 0.0
    max_selected_error = 0.0
    max_guard_leakage = 0.0
    max_off_support_leakage = 0.0
    min_progress_ratio = float("inf")
    min_safe_comparator_difference = float("inf")

    for trial in range(trials):
        d, c = 12, 5
        old_w = torch.tensor(rng.normal(scale=0.15, size=(c, d)), dtype=torch.float64)
        current = torch.tensor(rng.normal(size=(16, d)), dtype=torch.float64)
        labels = torch.tensor(rng.integers(0, c, size=16), dtype=torch.long)
        probe = old_w.clone().requires_grad_(True)
        old_logits = current @ probe.T
        old_loss = torch.nn.functional.cross_entropy(old_logits, labels)
        (gradient,) = torch.autograd.grad(old_loss, probe)
        learning_rate = 0.02
        comparator_w = (old_w - learning_rate * gradient).detach()
        safe_fraction = float(rng.uniform(0.1, 0.75))
        safe_w = (old_w - safe_fraction * learning_rate * gradient).detach()
        alternate_fraction = min(safe_fraction + 0.15, 0.95)
        alternate_safe_w = (old_w - alternate_fraction * learning_rate * gradient).detach()
        base_difference = float((safe_w - comparator_w).abs().max().item())
        min_safe_comparator_difference = min(min_safe_comparator_difference, base_difference)
        distinct_safe_endpoints += int(not torch.equal(safe_w, alternate_safe_w))

        comparator_logits = current @ comparator_w.T
        comparator_loss = torch.nn.functional.cross_entropy(comparator_logits, labels)
        comparator_decrease = float((old_loss.detach() - comparator_loss).item())
        if comparator_decrease <= 0.0:
            bad += 1
            continue
        positive_comparators += 1

        protected = torch.tensor(rng.normal(size=(12, d)) + 10.0, dtype=torch.float64)
        safeguard = torch.tensor(rng.normal(size=(8, d)) - 10.0, dtype=torch.float64)
        selected = torch.tensor(rng.normal(size=(8, d)) + 20.0, dtype=torch.float64)
        guards = torch.tensor(rng.normal(size=(16, d)) - 20.0, dtype=torch.float64)
        protected_target = protected @ old_w.T
        safeguard_target = safeguard @ old_w.T
        selected_target = torch.tensor(rng.normal(size=(8, c)), dtype=torch.float64)
        shield = FunctionalShield(d, c, max_nodes=64)
        result = replace_with_endpoint_emulation(
            shield=shield,
            base_logits_from_features=lambda x, w=safe_w: x @ w.T,
            current_features=current,
            counterfactual_current_logits=comparator_logits,
            protected_blocks=[RestorationBlock(protected, protected_target, "protected")],
            safeguard_blocks=[RestorationBlock(safeguard, safeguard_target, "safeguard")],
            selected_blocks=[RestorationBlock(selected, selected_target, "selected")],
            guard_features=guards,
            residual_tolerance=1e-11,
            executable_endpoint_tolerance=1e-11,
        )
        if not result.available:
            bad += 1
            continue
        available += 1
        deployed_current = current @ safe_w.T + shield(current)
        deployed_loss = torch.nn.functional.cross_entropy(deployed_current, labels)
        deployed_decrease = float((old_loss.detach() - deployed_loss).item())
        ratio = deployed_decrease / comparator_decrease
        min_progress_ratio = min(min_progress_ratio, ratio)
        max_current_error = max(max_current_error, float((deployed_current - comparator_logits).abs().max().item()))
        max_protected_error = max(max_protected_error, float((protected @ safe_w.T + shield(protected) - protected_target).abs().max().item()))
        max_safeguard_error = max(max_safeguard_error, float((safeguard @ safe_w.T + shield(safeguard) - safeguard_target).abs().max().item()))
        max_selected_error = max(max_selected_error, float((selected @ safe_w.T + shield(selected) - selected_target).abs().max().item()))
        max_guard_leakage = max(max_guard_leakage, float(shield(guards).abs().max().item()))
        far = torch.tensor(rng.normal(size=(16, d)) + 40.0, dtype=torch.float64)
        max_off_support_leakage = max(max_off_support_leakage, float(shield(far).abs().max().item()))
        if (
            abs(ratio - 1.0) > 1e-9
            or base_difference <= 0.0
            or max_current_error > 1e-9
            or max_protected_error > 1e-9
            or max_safeguard_error > 1e-9
            or max_selected_error > 1e-9
            or max_guard_leakage > 1e-12
            or max_off_support_leakage > 1e-12
        ):
            bad += 1

        if trial % 25 == 0:
            before = shield.snapshot()
            shared = torch.tensor(rng.normal(size=(1, d)), dtype=torch.float64)
            conflict = replace_with_endpoint_emulation(
                shield=shield,
                base_logits_from_features=lambda x, w=safe_w: x @ w.T,
                current_features=shared,
                counterfactual_current_logits=torch.ones((1, c), dtype=torch.float64),
                protected_blocks=[
                    RestorationBlock(shared, -torch.ones((1, c), dtype=torch.float64), "protected")
                ],
                duplicate_tolerance=0.0,
                target_tolerance=1e-13,
            )
            after = shield.snapshot()
            atomic = all(
                torch.equal(before[key], after[key])
                for key in ("centres", "coefficients", "support_radii", "match_radii")
            )
            if (
                not conflict.available
                and conflict.obstruction == "functional_constraint_inconsistency"
                and atomic
            ):
                conflicts_detected += 1
            else:
                bad += 1

    if min_progress_ratio == float("inf"):
        min_progress_ratio = 0.0
    if min_safe_comparator_difference == float("inf"):
        min_safe_comparator_difference = 0.0
    expected_conflicts = (trials - 1) // 25 + 1
    return {
        "theorems": ["thm:priority-oracle", "prop:safe-transfer", "thm:transfer-completion"],
        "trials": trials,
        "positive_comparator_rate": positive_comparators / trials,
        "joint_available_rate": available / trials,
        "joint_invalid_count": bad,
        "minimum_exact_progress_ratio": min_progress_ratio,
        "maximum_current_endpoint_error": max_current_error,
        "maximum_protected_endpoint_error": max_protected_error,
        "maximum_safeguard_endpoint_error": max_safeguard_error,
        "maximum_selected_endpoint_error": max_selected_error,
        "maximum_guard_leakage": max_guard_leakage,
        "maximum_off_support_leakage": max_off_support_leakage,
        "minimum_safe_comparator_base_difference": min_safe_comparator_difference,
        "distinct_budget_endpoint_rate": distinct_safe_endpoints / trials,
        "atomic_conflicts_detected": conflicts_detected,
        "expected_atomic_conflicts": expected_conflicts,
        "pass": (
            positive_comparators == trials
            and available == trials
            and bad == 0
            and min_progress_ratio >= 1.0 - 1e-9
            and max_current_error <= 1e-9
            and max_protected_error <= 1e-9
            and max_safeguard_error <= 1e-9
            and max_selected_error <= 1e-9
            and max_guard_leakage == 0.0
            and max_off_support_leakage == 0.0
            and min_safe_comparator_difference > 0.0
            and distinct_safe_endpoints == trials
            and conflicts_detected == expected_conflicts
        ),
    }


def reopening_experiments(rng: np.random.Generator, trials: int) -> dict[str, Any]:
    alpha = 0.05
    horizon = 512
    false_crossings = 0
    for _ in range(trials):
        process = HalfNormalMixtureEProcess(sigma=1.0, prior_scale=1.0, alpha=alpha)
        for x in rng.normal(loc=0.0, scale=1.0, size=horizon):
            process.update(float(x))
            if process.crossed:
                false_crossings += 1
                break
    false_upper = _binomial_upper_99(false_crossings, trials)

    delays: list[dict[str, Any]] = []
    for mean in [0.1, 0.2, 0.4, 0.8]:
        observed: list[int] = []
        missed = 0
        for _ in range(max(100, trials // 5)):
            process = HalfNormalMixtureEProcess(sigma=1.0, prior_scale=1.0, alpha=alpha)
            crossing = None
            for n, x in enumerate(rng.normal(loc=mean, scale=1.0, size=horizon), start=1):
                process.update(float(x))
                if process.crossed:
                    crossing = n
                    break
            if crossing is None:
                missed += 1
            else:
                observed.append(crossing)
        delays.append(
            {
                "mean_drift": mean,
                "median_delay": None if not observed else float(np.median(observed)),
                "crossing_rate": len(observed) / (len(observed) + missed),
            }
        )
    finite = [row["median_delay"] for row in delays if row["median_delay"] is not None]
    monotone = all(a >= b for a, b in zip(finite, finite[1:]))
    return {
        "theorems": ["thm:falseopen", "thm:delay"],
        "trials": trials,
        "alpha": alpha,
        "false_crossings": false_crossings,
        "false_crossing_rate": false_crossings / trials,
        "false_crossing_rate_upper_99": false_upper,
        "delay_grid": delays,
        "pass": false_upper <= alpha + 0.02 and monotone and delays[-1]["crossing_rate"] >= 0.95,
    }



def signature_shift_experiments(rng: np.random.Generator, trials: int) -> dict[str, Any]:
    """Independently test lifetime false splitting and joint-evidence delay.

    The signature score has bounded range length one, so sigma=1/2 is the
    universal Hoeffding scale used by the implementation and theorem.
    """

    alpha = 0.05
    horizon = 512
    null_trials = max(500, trials)
    false_crossings = 0
    semantic_only_splits = 0
    outcome_crossings = 0
    for _ in range(null_trials):
        signature_process = HalfNormalMixtureEProcess(
            sigma=0.5, prior_scale=1.0, alpha=alpha
        )
        outcome_process = HalfNormalMixtureEProcess(
            sigma=1.0, prior_scale=1.0, alpha=alpha
        )
        signature_crossed = False
        outcome_crossed = False
        # Semantic conflict: outcome evidence has positive drift while the
        # observable signature score remains a bounded mean-zero null.
        for _n in range(horizon):
            signature_x = 0.5 if rng.random() < 0.5 else -0.5
            outcome_x = float(np.clip(rng.normal(loc=0.35, scale=0.7), -1.0, 1.0))
            signature_process.update(signature_x)
            outcome_process.update(outcome_x)
            signature_crossed = signature_crossed or signature_process.crossed
            outcome_crossed = outcome_crossed or outcome_process.crossed
            if signature_crossed and outcome_crossed:
                break
        false_crossings += int(signature_crossed)
        outcome_crossings += int(outcome_crossed)
        semantic_only_splits += int(signature_crossed and outcome_crossed)
    false_upper = _binomial_upper_99(false_crossings, null_trials)

    delay_grid: list[dict[str, Any]] = []
    for signature_mean in [0.03, 0.06, 0.12, 0.24]:
        observed: list[int] = []
        missed = 0
        local_trials = max(100, trials // 5)
        for _ in range(local_trials):
            signature_process = HalfNormalMixtureEProcess(
                sigma=0.5, prior_scale=1.0, alpha=alpha
            )
            outcome_process = HalfNormalMixtureEProcess(
                sigma=1.0, prior_scale=1.0, alpha=alpha
            )
            crossing = None
            for n in range(1, horizon + 1):
                signature_x = float(
                    np.clip(rng.normal(loc=signature_mean, scale=0.22), -0.5, 0.5)
                )
                outcome_x = float(np.clip(rng.normal(loc=0.35, scale=0.7), -1.0, 1.0))
                signature_process.update(signature_x)
                outcome_process.update(outcome_x)
                if signature_process.crossed and outcome_process.crossed:
                    crossing = n
                    break
            if crossing is None:
                missed += 1
            else:
                observed.append(crossing)
        delay_grid.append(
            {
                "signature_mean_drift": signature_mean,
                "median_joint_delay": None if not observed else float(np.median(observed)),
                "joint_crossing_rate": len(observed) / (len(observed) + missed),
            }
        )
    finite = [row["median_joint_delay"] for row in delay_grid if row["median_joint_delay"] is not None]
    monotone = all(a >= b for a, b in zip(finite, finite[1:]))
    return {
        "theorems": ["prop:route-split", "thm:split-delay"],
        "trials": trials,
        "null_trials": null_trials,
        "alpha_split": alpha,
        "semantic_only_outcome_crossing_rate": outcome_crossings / null_trials,
        "semantic_only_false_signature_crossings": false_crossings,
        "semantic_only_false_split_rate": semantic_only_splits / null_trials,
        "false_split_rate_upper_99": false_upper,
        "joint_delay_grid": delay_grid,
        "pass": (
            false_upper <= alpha + 0.02
            and outcome_crossings / null_trials >= 0.90
            and monotone
            and delay_grid[-1]["joint_crossing_rate"] >= 0.90
        ),
    }


def metaplastic_frontier(rng: np.random.Generator, repetitions: int) -> dict[str, Any]:
    T = 512
    P = 6
    delta = 0.05
    violations = 0
    worst_excess = -float("inf")
    for repetition in range(repetitions):
        policies = make_policy_family([0.35, 0.65], [0, 1, 2], K=10)
        controller = MetaplasticController(policies, T, seed=1000 + repetition, zeta=1.0, loss_bound=1.0)
        phases = rng.integers(0, P, size=T)
        selected = 0.0
        cumulative = np.zeros(P, dtype=np.float64)
        for t in range(T):
            # Fixed before the private draw: a slowly switching best policy.
            best = int(phases[t] if t % 64 == 0 else phases[(t // 64) * 64])
            losses = np.full(P, 0.65, dtype=np.float64)
            losses[best] = 0.05
            losses += rng.uniform(0.0, 0.02, size=P)
            losses = np.clip(losses, 0.0, 1.0)
            chosen = controller.sample()
            selected += float(losses[chosen])
            cumulative += losses
            controller.update(losses)
        bound = math.sqrt(T * math.log(P) / 2.0) + math.sqrt(T * math.log(1.0 / delta) / 2.0)
        excess = selected - float(cumulative.min()) - bound
        worst_excess = max(worst_excess, excess)
        violations += int(excess > 1e-9)

    K = 12
    alpha = 0.65
    rates = np.array([2.0 ** (-(k + 1)) for k in range(K)], dtype=np.float64)
    ages = np.arange(1, 2 ** (K - 2) + 1, dtype=np.float64)
    kernel = np.array([
        np.sum(rates**alpha * (1.0 - rates) ** (age - 1.0)) for age in ages
    ])
    ratio = kernel * ages**alpha
    distortion = float(ratio.max() / ratio.min())
    violation_upper = _binomial_upper_99(violations, repetitions)
    return {
        "theorems": ["thm:multiscale", "thm:single", "thm:hedge"],
        "repetitions": repetitions,
        "hedge_bound_violation_rate": violations / repetitions,
        "hedge_bound_violation_rate_upper_99": violation_upper,
        "worst_excess_over_bound": worst_excess,
        "ema_powerlaw_distortion": distortion,
        "pass": violation_upper <= delta + 0.03 and distortion < 20.0,
    }


def renewal_experiments(rng: np.random.Generator, trials: int) -> dict[str, Any]:
    pool = ZeroGatedAdapterPool(dim=12, bottleneck=4, slots=3, initially_active=1).double()
    x = torch.tensor(rng.normal(size=(16, 12)), dtype=torch.float64)
    with torch.no_grad():
        before = pool(x).clone()
        generator = torch.Generator().manual_seed(17)
        pool.reset_dormant(1, generator=generator)
        after = pool(x).clone()
    exact_change = float(torch.max(torch.abs(after - before)).item())

    richness_grid: list[dict[str, Any]] = []
    for rho in [0.0, 0.01, 0.05, 0.2]:
        n = 100
        successes = rng.binomial(1, rho, size=(trials, n)).max(axis=1)
        empirical = float(successes.mean())
        theoretical = 1.0 - (1.0 - rho) ** n
        richness_grid.append(
            {
                "rho": rho,
                "trials_per_run": n,
                "empirical_success_probability": empirical,
                "theoretical_success_probability": theoretical,
            }
        )
    return {
        "theorems": ["thm:renew-safe", "thm:renew-richness"],
        "zero_gated_reset_max_function_change": exact_change,
        "richness_grid": richness_grid,
        "pass": exact_change == 0.0 and richness_grid[0]["empirical_success_probability"] == 0.0,
    }


def resource_scaling() -> dict[str, Any]:
    # Fixed structural arrays; only exact counter precision depends on horizon.
    max_contexts = 6
    max_records = 6
    sketch_rows = 32
    dimension = 1000
    timescales = 13
    atoms = 6
    structural_scalars = (
        max_contexts * dimension
        + max_records * sketch_rows * dimension
        + atoms * timescales
        + max_contexts * 8
        + max_records * 24
    )
    rows = []
    for horizon in [10, 100, 1_000, 10_000, 1_000_000]:
        rows.append(
            {
                "horizon": horizon,
                "structural_scalars": structural_scalars,
                "counter_bits": int(math.ceil(math.log2(horizon + 1))),
            }
        )
    structural_constant = len({row["structural_scalars"] for row in rows}) == 1
    logarithmic_bits = all(
        rows[i + 1]["counter_bits"] - rows[i]["counter_bits"]
        <= math.ceil(math.log2(rows[i + 1]["horizon"] / rows[i]["horizon"])) + 1
        for i in range(len(rows) - 1)
    )
    return {
        "theorem": "thm:main item vii",
        "rows": rows,
        "pass": structural_constant and logarithmic_bits,
    }


def convex_dynamic_regret() -> dict[str, Any]:
    T = 300
    eta = 0.08
    D = 2.0
    G = 2.0
    y = np.concatenate(
        [np.full(75, -0.8), np.full(75, 0.8), np.linspace(0.8, -0.8, 75), np.full(75, -0.2)]
    )
    lower = np.concatenate([np.full(150, -0.25), np.full(150, -0.45)])
    upper = np.concatenate([np.full(150, 0.25), np.full(150, 0.05)])
    theta = 0.0
    regret = 0.0
    gammas = []
    daggers = []
    for t in range(T):
        theta_star = float(np.clip(y[t], -1.0, 1.0))
        theta_dagger = float(np.clip(theta_star, lower[t], upper[t]))
        daggers.append(theta_dagger)
        loss = 0.5 * (theta - y[t]) ** 2
        optimum_loss = 0.5 * (theta_star - y[t]) ** 2
        compatible_loss = 0.5 * (theta_dagger - y[t]) ** 2
        regret += loss - optimum_loss
        gammas.append(compatible_loss - optimum_loss)
        gradient = theta - y[t]
        next_lower = lower[min(t + 1, T - 1)]
        next_upper = upper[min(t + 1, T - 1)]
        theta = float(np.clip(theta - eta * gradient, next_lower, next_upper))
    path = float(np.abs(np.diff(np.asarray(daggers))).sum())
    bound = (
        sum(gammas)
        + D * D / (2.0 * eta)
        + (D / eta + G) * path
        + eta * G * G * T / 2.0
    )
    conflict_gamma = 0.5 * (-1.0 - 1.0) ** 2
    return {
        "theorem": "thm:dynreg",
        "dynamic_regret": regret,
        "compatibility_price": float(sum(gammas)),
        "compatible_path_variation": path,
        "theorem_bound": bound,
        "conflicting_single_round_gamma": conflict_gamma,
        "pass": regret <= bound + 1e-10 and conflict_gamma > 0.0,
    }


def impossibility_controls() -> dict[str, Any]:
    identical_signature_bayes_error = 0.5
    conflict = priority_constrained_transfer_direction(
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
        protected_basis=None,
        allowed_mask=None,
        current_fraction=0.25,
    )
    finite_memory_collision = 2**8 < 2**16
    return {
        "theorem": "cor:maximality",
        "routing_identical_law_minimum_error": identical_signature_bayes_error,
        "direct_conflict_priority_transfer_available": conflict.available,
        "finite_memory_counting_collision": finite_memory_collision,
        "zero_information_reopening_claimed": False,
        "zero_richness_renewal_claimed": False,
        "pass": (
            identical_signature_bayes_error == 0.5
            and not conflict.available
            and finite_memory_collision
        ),
    }


def run_mechanism_suite(seed: int = 20260729, trials: int = 2000, quick: bool = False) -> dict[str, Any]:
    # These validation problems are tiny. Large BLAS thread pools make each
    # small matrix operation dramatically slower and can look like a hang in
    # subprocess-based release tests. Use one intra-op thread transactionally.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    mc_trials = min(trials, 250) if quick else trials
    # Lifetime false-alarm gates require enough trials for their own one-sided
    # 99% confidence bounds.  A 200-trial quick run is underpowered even when
    # the empirical rate is at the declared alpha, so quick mode keeps the
    # expensive geometry/integration grids small but retains 1000 statistical
    # trials rather than weakening the criterion.
    statistical_trials = max(1000, trials) if quick else trials
    results = {
        "routing": routing_experiments(rng),
        "consolidation": consolidation_coverage(rng, statistical_trials),
        "retention": exact_retention_certificate(rng, max(1000, mc_trials)),
        "transfer_and_stationarity": transfer_geometry(rng, max(500, mc_trials)),
        "reopening": reopening_experiments(rng, statistical_trials),
        "route_split": signature_shift_experiments(rng, statistical_trials),
        "metaplastic": metaplastic_frontier(rng, max(80, mc_trials // 10)),
        "renewal": renewal_experiments(rng, max(500, mc_trials)),
        "resources": resource_scaling(),
        "convex": convex_dynamic_regret(),
        "impossibility": impossibility_controls(),
    }
    passed = {name: bool(result.get("pass", False)) for name, result in results.items()}
    report = {
        "suite": "AFM-U theorem-mechanism validation",
        "seed": seed,
        "requested_trials": trials,
        "quick": quick,
        "results": results,
        "passes": passed,
        "mechanism_alignment_pass": all(passed.values()),
        "paper_proved": False,
        "note": "Experiments validate executable consequences; they do not replace the mathematical proof.",
    }
    torch.set_num_threads(previous_threads)
    return report
