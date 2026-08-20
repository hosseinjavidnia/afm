# AFM causal functional-compatibility sweep

## Scientific question

This experiment is designed to test the causal statement:

> The amount of new learning that can be stored persistently without violating prior behaviour is governed by measurable functional compatibility with the protected model.

It is not another ordinary continual-learning benchmark.  It intervenes on the compatibility of the incoming learning signal while holding the pre-update network state, protected evidence, current inputs, gradient norm, and learning protocol fixed within each matched causal bundle.

## Systems and seeds

The full suite contains three end-to-end trainable systems and five frozen seeds per system:

1. CIFAR-10 + fully trainable compact CNN.
2. CIFAR-10 + fully trainable tiny Vision Transformer.
3. WikiText-2 training text + fully trainable character-level transformer language model.

Seeds: 11, 29, 47, 71, 101.

No learned representation prefix is frozen.  The runner aborts if any learned model parameter has `requires_grad=False`.

## Matched-state causal intervention

A parent network first learns normally from the supervised stream.  At each declared causal state:

1. The exact pre-update parameter vector is fixed.
2. A protected bank is sampled from previously seen evidence.
3. A current batch is built from both novel examples and deterministic nonidentical perturbations of protected examples.  This creates shared functional geometry without making current and protected finite addresses identical.
4. The protected functional Jacobian `J_p` and current functional Jacobian `J_c` are measured over all trainable parameters.
5. The protected feasible projector is

   `P = I - J_p^T (J_p J_p^T + ridge I)^(-1) J_p`.

6. For current teacher residual `r`, the unrestricted parameter gradient is `q = J_c^T r`, and measured compatibility is

   `kappa = ||P q||^2 / ||q||^2`.

7. Rather than injecting an arbitrary parameter gradient, the code solves the generalized function-space eigenproblem

   `J_c P J_c^T r = lambda J_c J_c^T r`.

   Low- and high-compatibility residual eigenmodes are mixed to create teacher targets at nominal compatibility levels

   `0, 0.1, 0.25, 0.5, 0.75, 1.0`.

   The overall parameter-gradient norm is matched across nominal levels.  The actually measured `kappa` is always logged and is the primary x-axis value.  If a local geometry cannot realize a nominal endpoint, the target is clipped to the achievable interval and explicitly marked.

8. Every method is evaluated from the same pre-update model state and the same teacher target.  No causal branch is committed to the parent trajectory.  The parent then continues ordinary supervised learning, producing the next independent matched causal state.

This design makes compatibility an intervention, not a post-hoc correlation with naturally hard examples.

## Genuine no-protection comparator

For every compatibility target, the runner computes the genuine same-state unrestricted endpoint using the current teacher gradient.  A bounded backtracking line search only reduces the common initial step size if the unrestricted endpoint fails to decrease the teacher loss.

`Delta_0 = L_before - L_unrestricted_after`.

The experiment matches the native teacher-gradient norm across the six compatibility interventions and then calibrates each genuine same-state no-protection comparator along that same native descent direction to a common finite `Delta_0` by bracketed step-length bisection.  The common target is set below the smallest already-positive unrestricted decrease at that causal state, so each compatibility level brackets the same finite progress target without redefining `Delta_0` as a linearized surrogate.  The experiment records the coefficient of variation of the calibrated `Delta_0` values as an explicit audit of the causal matching.

## Cross-method persistent outcome

For every method, persistent progress is measured only in ordinary trainable model parameters:

`rho_persistent = (L_before - L_method_base_after) / Delta_0`.

Protected behaviour is audited on the same protected full-logit evidence for every method.  Each point records maximum absolute protected-logit drift, RMS drift, and whether it passes the predeclared common retention tolerance.

The cross-method scientific question is therefore not whether AFM wins every point.  It is whether low measured compatibility suppresses persistent progress for methods that actually satisfy the common retention criterion.

## Methods

Every matched causal bundle evaluates:

- `unrestricted`: genuine no-protection learning.
- `replay`: current teacher objective plus replay of protected labels.
- `projection`: current gradient projected into the protected functional nullspace.
- `linearized_distillation`: local logit-distillation proximal update using the same protected Jacobian.
- `ewc_prox`: diagonal-Fisher proximal update.
- `derpp`: local DER++-style replay of stored logits plus protected labels.
- `afm`: compatibility-projected persistent base update with retention backtracking, followed by a separate finite endpoint-completion transaction.

The replay/distillation/proximal methods are part of this causal apparatus; they do not modify the public challenger packages used in the separate benchmark study.

## AFM finite completion under an unfrozen representation

The original benchmark shield is indexed by frozen learned features.  That is not valid when every representation parameter is trainable.  This experiment therefore keeps the AFM distinction between persistent base adaptation and finite endpoint completion while indexing the finite residual by an immutable deterministic address of the raw input.

- Vision: a fixed random projection of normalized pixels.
- Text: a fixed random token-address map.

The address map has no learned parameters and never changes as the network representation learns.

After an accepted AFM persistent base step, the finite transaction requests:

- the unrestricted no-protection current-batch logits at current addresses; and
- the pre-update protected logits at protected addresses.

Guard addresses are included.  The existing compact-cardinal shield implementation is used with support multiplier 4 and feature/address replay envelope `1e-8`.  The code logs finite completion availability and exact endpoint error separately from persistent progress.

This is the central conceptual distinction the experiment is designed to expose: a finite endpoint can be reproduced without turning that finite residual into persistent parametric learning.

## Scale of the full experiment

Default suite:

- 3 systems.
- 5 seeds per system.
- 50 accepted causal states per seed when the local geometry has the configured minimum compatibility span.
- 6 nominal compatibility interventions per state.
- 7 learning methods per intervention.

At full completion this yields up to 31,500 method-level causal measurements, nested inside 750 independent seed/state bundles.  Inferential analysis treats the seed as the independent experimental unit; update-level points are not treated as independent replicates.

## Primary outputs

Each run writes:

- `compatibility_points.jsonl`: every method/intervention measurement.
- `events.jsonl`: parent training, geometry coverage, and comparator-matching audits.
- `summary.json`: run-level counts and provenance.
- `final_parent.pt`: final parent network checkpoint.
- `resolved_config.json`: exact frozen run configuration.

The suite analyzer writes:

- `seed_level_summary.csv`
- `aggregate_by_kappa_method.csv`
- `seed_level_trends.csv`
- `trend_summary.csv`
- `central_figure_points.csv`
- `retention_qualified_envelope_seed.csv`
- `retention_qualified_envelope.csv`
- `validation.json`
- `afm_compatibility_table.tex`

Seed-level 95% intervals use the exact bootstrap over the available seed-level statistics.  With five seeds, all `5^5 = 3125` resamples are enumerated and the 2.5th/97.5th percentile endpoints are computed with linear interpolation.

## Central figure

The primary plot uses:

- x-axis: measured functional compatibility `kappa`;
- y-axis: persistent progress ratio `Delta_persistent / Delta_0`;
- point identity: method, architecture/system, seed, and matched causal state;
- retention qualification: explicit marker/filter from the common protected-logit tolerance.

AFM points additionally carry:

- finite deployed progress ratio;
- finite completion success/error;
- selected persistent backtracking fraction;
- the theorem-aligned coarse reference `lambda_hat * kappa / 3` and its empirical margin.

The latter is reported as a theorem-aligned empirical audit, not as a new proof.

## v1.4 full retention-frontier protocol (supersedes the single-tolerance cross-method analysis above)

The final causal design requires **exactly 50 accepted causal states for every one of the 15 system/seed runs**. A run with fewer than 50 states is marked incomplete and fails. The probe stream is scanned at every parent-stream step for up to 5000 attempts; probing remains observational and no causal branch is committed to the parent.

The six compatibility interventions and seven method proposals are unchanged. The genuine same-state unrestricted finite decrease remains matched across the six compatibility levels before any method comparison.

For the broad cross-method claim, a single absolute protected-logit tolerance is no longer the primary retention control. At each matched state and compatibility level, let `D0` be the maximum absolute protected-logit drift caused by that compatibility level's genuine same-state unrestricted comparator. The common retention budgets are predeclared as

`D <= max(1e-8, beta * D0)` for `beta in {0, .01, .05, .10, .25, .50, 1}`.

Each method first constructs its native one-step persistent proposal from the identical pre-update state. The proposal is then evaluated at 33 equally spaced scalar fractions in `[0,1]` using actual finite network endpoints. For each beta, the reported frontier point is the **largest proposal fraction on that predeclared grid** satisfying the common protected-logit budget. This is a method-neutral causal retention cap; it is not a method-specific tuning threshold.

The full design therefore yields:

- 750 causal states = 3 systems x 5 seeds x 50 states;
- 31,500 native method/intervention proposals = 750 x 6 compatibility levels x 7 methods;
- 220,500 retention-frontier measurements = 31,500 x 7 relative retention budgets.

AFM finite endpoint completion is evaluated separately at the strict `beta=0` frontier point. The scalar frontier fraction is **not** relabelled as AFM's certified theorem `lambda_hat`; theorem validation remains the separate theorem-aligned audit from the main AFM study.

The v1.4 primary outputs add `retention_frontier_points.jsonl`, and the final analyzer writes to `analysis_v14_frontier/`. Statistical inference still aggregates within run/seed first and uses the seed as the independent unit.

## v1.5 common state-level retention frontier (supersedes the v1.4 budget definition)

The compatibility intervention must not change the retention allowance itself.
For every matched causal state, first evaluate the six genuine unrestricted
comparators and freeze

`D_ref = max_kappa D_unrestricted(kappa)`.

The seven predeclared retention levels then use the same absolute budget for all
six compatibility interventions and all seven methods:

`D <= max(1e-8, beta * D_ref)` for `beta in {0, .01, .05, .10, .25, .50, 1}`.

The 33 actual finite endpoint evaluations for every proposal are saved, rather
than only the selected frontier point. AFM's native persistent transaction and
finite endpoint completion are also saved separately; beta=0 of the method-
neutral frontier is not used as a surrogate for AFM's native persistent update.

At complete coverage this produces 750 causal states, 220,500 selected frontier
rows, 1,039,500 raw proposal-grid rows, and 4,500 AFM-native transaction rows.
