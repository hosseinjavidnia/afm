# Controlled causal functional-compatibility protocol

## Scientific question

The controlled experiment tests whether functional compatibility causally
determines how much incoming learning can persist while a common prior-function
retention requirement is enforced.

Compatibility is manipulated from the same pre-update neural state. It is not
inferred from naturally occurring easy or difficult examples.

## Systems and prespecified seeds

The primary suite contains three fully trainable systems:

1. CIFAR-10 with a compact CNN.
2. CIFAR-10 with a compact Vision Transformer.
3. WikiText-2 training text with a character-level transformer language model.

The five prespecified seeds are `11, 29, 47, 71, 101`. No learned
representation prefix is frozen; the runner aborts if a learned model parameter
has `requires_grad=False`.

## Same-state compatibility intervention

A parent network learns normally from the supervised stream. At each accepted
causal state:

1. The exact pre-update parameter vector is fixed.
2. A protected bank is sampled from previously seen evidence.
3. The current batch contains novel examples and deterministic nonidentical
   perturbations of protected examples, creating shared functional geometry
   without making current and protected finite addresses identical.
4. The protected and current functional Jacobians, `J_p` and `J_c`, are measured
   over all trainable parameters.
5. The protected feasible projector is

```math
P = I - J_p^\top\left(J_pJ_p^\top + \mathrm{ridge}\,I\right)^{-1}J_p.
```

6. For current teacher residual `r`, the unrestricted parameter gradient is

```math
q = J_c^\top r,
```

and realized functional compatibility is

```math
\kappa = \frac{\lVert Pq\rVert^2}{\lVert q\rVert^2}.
```

7. The intervention is constructed in function space from the generalized
   eigenproblem

```math
J_c P J_c^\top r = \lambda J_c J_c^\top r.
```

   Low- and high-compatibility residual modes are mixed to target nominal levels
   `{0, 0.1, 0.25, 0.5, 0.75, 1.0}`. The native parameter-gradient norm is
   matched across the six interventions. Analysis uses the measured realized
   κ. If a local geometry cannot realize a requested endpoint, the achieved
   value and clipping status are recorded.
8. Every method is evaluated from the identical pre-update model state and the
   same teacher target. Causal branches are observational experiments and are
   not committed to the parent trajectory.

Exactly 50 accepted causal states are required for each system/seed run. A run
with fewer than 50 accepted states is incomplete.

## Genuine unrestricted comparator and matched finite opportunity

For each κ intervention, the runner first evaluates the genuine same-state
unrestricted endpoint along that intervention's native teacher-gradient
direction:

```math
\Delta_{0,\mathrm{raw}}(\kappa)
=
L_{\mathrm{before}}-L_{\mathrm{unrestricted,after}}.
```

After a positive unrestricted decrease is established for all six
interventions, the common finite target is

```math
\Delta_{0,\mathrm{target}}
=
0.5\,\min_{\kappa}\Delta_{0,\mathrm{raw}}(\kappa).
```

Each unrestricted comparator is then calibrated to this same target by
one-dimensional bracketed bisection **along its already constructed direction**.
The direction is not rotated during matching. The resulting finite decreases
are recorded as a matching audit.

## Common retention frontier

The compatibility intervention must not change the retention allowance. For
each causal state, evaluate the six genuine unrestricted comparators and define
one state-level reference

```math
D_{\mathrm{ref}}
=
\max_{\kappa}D_{\mathrm{unrestricted}}(\kappa),
```

where `D` is maximum absolute drift on the common protected full-logit evidence.
All six κ interventions and all methods use the same seven budgets

```math
D \le \max\!\left(10^{-8},\,\beta D_{\mathrm{ref}}\right),
\qquad
\beta \in \{0,.01,.05,.10,.25,.50,1\}.
```

Each method constructs its native persistent proposal from the same pre-update
state. The proposal is evaluated at 33 prespecified scalar fractions in
`[0,1]` using actual finite network endpoints. For each β, the selected frontier
point is the largest evaluated fraction satisfying the common absolute
retention budget.

This scalar frontier is method-neutral. It is not a method-specific tuning
threshold and is not relabeled as AFM's native retention fraction.

## Persistent outcome

For each selected method endpoint, persistent progress is measured only in
ordinary trainable model parameters:

```math
\rho_{\mathrm{persistent}}
=
\frac{L_{\mathrm{before}}-L_{\mathrm{method\ base,after}}}{\Delta_0}.
```

The protected evidence and retention metric are the same across methods.

## Methods

Each matched causal bundle evaluates:

- `unrestricted`: genuine no-protection learning;
- `replay`: current teacher objective plus replay of protected labels;
- `projection`: current gradient projected into the protected functional
  nullspace;
- `linearized_distillation`: local logit-distillation proximal update using the
  same protected Jacobian;
- `ewc_prox`: diagonal-Fisher proximal update;
- `derpp`: local DER++-style replay of stored logits plus protected labels; and
- `afm`: compatibility-projected persistent base update with native retention
  backtracking, followed by a separate finite endpoint-completion transaction.

On the **common method-neutral frontier**, `projection` and the AFM base proposal
are the same compatible projection and are reported as
`projection_afm_base`. Native AFM is analyzed separately because it includes
its own retention backtracking and finite-completion transaction.

## Native AFM finite completion

The causal experiment keeps persistent parameter adaptation separate from
finite requested-output restoration. Because all representation parameters are
trainable, finite residuals are indexed by an immutable deterministic address
of the raw input rather than by a learned feature prefix:

- vision: fixed random projection of normalized pixels;
- text: fixed random token-address map.

After an accepted native AFM persistent base step, the finite transaction
requests the unrestricted current-batch logits at current addresses and the
pre-update protected logits at protected addresses. Guard addresses are
included. Completion availability and finite endpoint error are logged
separately from persistent progress.

## Full-suite scale

At complete coverage the primary suite contains:

- 3 systems × 5 seeds × 50 causal states = 750 causal states;
- 750 × 6 κ interventions × 7 methods = 31,500 native method proposals;
- 31,500 × 7 retention budgets = 220,500 selected frontier rows;
- 1,039,500 raw proposal-grid rows; and
- 4,500 native AFM transaction rows.

The 750 states are nested within 15 system/seed runs. They are not treated as
750 independent inferential replicates.

## Primary statistical hierarchy

For each system, method family, and retention budget:

1. fit one ordinary least-squares slope of persistent ratio on **realized κ**
   across the six interventions within each causal state;
2. average the 50 state-level slopes within each seed;
3. average the five seed-level means for the reported effect; and
4. form the two-sided 95% Student-t confidence interval over the five seed means
   with `df=4`.

The primary β=0.5 projection/AFM-base slopes are therefore based on five
independent seed summaries per system. A separate cross-system descriptive
summary uses the 15 system-seed slope values and a two-sided 95% Student-t
interval with `df=14`.

Follow-up experiments use their own declared bootstrap conventions; those do
not replace the primary statistical hierarchy above.

## Output wiring

Each primary run writes the causal records needed by the two analysis stages,
including:

- `compatibility_points.jsonl`;
- `retention_frontier_points.jsonl`;
- `retention_frontier_grid.jsonl`;
- `afm_native_points.jsonl`;
- `events.jsonl`;
- `summary.json`;
- `preprobe_parent.pt` and/or final parent checkpoint outputs as produced by the
  runner; and
- `resolved_config.json`.

Run the common-budget frontier analyzer first:

```bash
python scripts/analyze_compatibility_frontier_v15.py \
  --suite-root runs_compatibility_causal_v15
```

It writes `analysis_v15_common_budget/`, including retention-reference audits,
seed-level and aggregate frontier tables, trend tables, native-AFM summaries,
and `validation.json`.

Then run the manuscript reporting analyzer:

```bash
python scripts/analyze_compatibility_reporting.py \
  --suite-root runs_compatibility_causal_v15
```

It writes `analysis_v15_common_budget_reporting_v2/`, including:

- `causal_compatibility_v15_reporting_v2_long.csv`;
- `causal_compatibility_v15_reporting_v2_wide.csv`;
- `main_causal_table_all_beta.csv`;
- `state_level_matched_effects_all_beta.csv`;
- `seed_level_matched_effects_all_beta.csv`;
- `cross_system_matched_slopes_all_beta.csv`;
- `cross_system_matched_spearman_all_beta.csv`;
- native-AFM mechanism and diagnostic tables; and
- `reporting_validation_v2.json`.

`reporting_validation_v2.json` must pass before the tables are used for the
paper figures.

## Figure 1 uncertainty

`make_afm_paper_figures.py` reconstructs Figure 1B uncertainty from seed-level
values with the same two-sided 95% Student-t (`df=4`) convention and reads the
Student-t matched-slope intervals produced above for Figure 1C-D. The figure
script does not rerun the experiment.
