# Fixed-norm natural-scale bridge

## Purpose

The fixed-norm bridge is a finite-scale follow-up to the matched-Δ0 causal
experiment. It asks whether compatibility can still be manipulated while the
genuine unrestricted update norm is held at a common fraction of a natural
training-update scale.

It is a different estimand from the primary matched-Δ0 experiment and should be
reported separately.

## Inputs

The bridge uses:

- the completed 10-seed generality suite for parent states and model
  configurations; and
- the natural-state validation table to define the system-specific median
  unrestricted update norm.

The implemented bridge covers the CNN and baseline ViT systems over ten seeds.
The default natural-norm fractions are `0.01, 0.10, 0.50, 1.00`, and the default
requested κ values are `0.10, 0.25, 0.50, 0.75`.

## Fixed-norm comparator

For a requested displacement norm, the unrestricted teacher-gradient direction
is kept fixed and only its scalar step is chosen so that the realized parameter
displacement has the requested norm. A candidate is usable only when the finite
endpoint is distinct, finite, norm-matched within the declared tolerance, and
produces a positive genuine unrestricted teacher-loss decrease.

Directional candidates are used to realize the requested κ levels. Among
available candidates, the code selects the set with the tightest finite-Δ0
alignment across κ while preserving the fixed-norm construction.

## Feasibility and finite-Δ0 diagnostics

The final bridge interpretation distinguishes **feasibility requirements** from
**diagnostics**:

- positive genuine unrestricted endpoints at every requested κ are required;
- the requested update-norm and κ tolerances are required;
- finite-Δ0 coefficient of variation and relative range are recorded to
  describe residual heterogeneity, but are not final admission criteria.

Accordingly, an infeasible state-scale condition means that the required all-κ
same-norm comparison could not be constructed. It is not assigned a zero
compatibility effect.

## Targeted completion pass

The public workflow includes a targeted completion runner for state-scale
conditions that satisfy the core fixed-norm construction but should be evaluated
without using finite-Δ0 CV/range diagnostics as admission thresholds. It uses
the same parent state, fixed schedule, candidate construction, and fixed-norm
requirements.

The filenames retain `repair` as an implementation identifier:

- `build_natural_scale_bridge_repair_matrix.py`;
- `run_natural_scale_bridge_repair_job.py`; and
- `analyze_natural_scale_bridge_repair.py`.

The merged analyzer substitutes the targeted completion results only for those
finite-Δ0-diagnostic exclusions. Other feasibility decisions, including missing
positive endpoints, are retained.

## Statistical summary

The final bridge analysis uses the seed as the replication unit and the
deterministic Monte Carlo seed bootstrap implemented by
`analyze_natural_scale_bridge_repair.py` (100,000 resamples by default).

## Reproduction order

1. Complete the 10-seed generality suite.
2. Complete the natural-state validation analysis.
3. Build and run `runs_compatibility_natural_scale_bridge_v1`.
4. Build and run the targeted bridge completion matrix.
5. Run `analyze_natural_scale_bridge_repair.py` to create the merged final
   bridge tables.

Exact commands are given in
[`../reproducing_compatibility.md`](../reproducing_compatibility.md).
