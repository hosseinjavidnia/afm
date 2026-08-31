# Reproducing the compatibility experiments

This page gives the dependency order for the compatibility study. Full suites
are GPU-intensive. Each runner accepts a matrix path and one integer job index,
so the generated matrix can be mapped to a local loop or a cluster scheduler.

## A. Main controlled causal suite

Prepare the datasets and build the 3-system × 5-seed matrix:

```bash
python scripts/prepare_compatibility_cifar10.py --root data_compatibility/cifar10
python scripts/prepare_compatibility_text.py --output data_compatibility/text/input.txt

python scripts/build_compatibility_sweep_matrix.py \
  --output-root runs_compatibility_causal_v15
```

Run all 15 entries, shown here sequentially:

```bash
for i in $(seq 0 14); do
  python scripts/run_compatibility_sweep_job.py \
    --matrix runs_compatibility_causal_v15/job_matrix.json \
    --index "$i"
done
```

Analyze the common retention frontier and primary reporting tables:

```bash
python scripts/analyze_compatibility_frontier_v15.py \
  --suite-root runs_compatibility_causal_v15

python scripts/analyze_compatibility_reporting.py \
  --suite-root runs_compatibility_causal_v15
```

The primary five-seed matched-effect intervals produced by the reporting path
are two-sided 95% Student-t intervals with `df=4`.

## B. Independent directions

```bash
python scripts/build_independent_direction_matrix.py \
  --source-suite-root runs_compatibility_causal_v15 \
  --output-root runs_compatibility_independent_directions_v1 \
  --requested-kappas 0.10 0.25 0.50 0.75 \
  --directions 4
```

Execute each row in the generated `job_matrix.json` with
`run_independent_direction_job.py`, then analyze:

```bash
python scripts/analyze_independent_directions.py \
  --suite-root runs_compatibility_independent_directions_v1 \
  --plot-method projection \
  --plot-beta 0.5
```

This five-seed follow-up uses the exact `5^5 = 3125` seed bootstrap implemented
by the analysis script.

## C. Multiscale local robustness

```bash
python scripts/build_multiscale_matrix.py \
  --source-suite-root runs_compatibility_causal_v15 \
  --output-root runs_compatibility_multiscale_v1 \
  --scale-fractions 0.05 0.20 0.50 0.90
```

Execute generated rows with `run_multiscale_job.py`, then:

```bash
python scripts/analyze_multiscale.py \
  --suite-root runs_compatibility_multiscale_v1 \
  --plot-method projection \
  --plot-beta 0.5
```

The scale fractions are log-expansion coordinates from the validated local
reference toward a common attainable peak; they are not percentages of a
natural training update. This five-seed follow-up uses the exact
`5^5 = 3125` seed bootstrap.

## D. Ten-seed and stronger-ViT generality

```bash
python scripts/build_generality_matrix.py \
  --source-suite-root runs_compatibility_causal_v15 \
  --output-root runs_compatibility_generality_v1 \
  --extra-seeds 131 149 167 191 223 \
  --strong-vit-seeds 11 29 47 71 101 131 149 167 191 223 \
  --strong-vit-dim 96 \
  --strong-vit-depth 6 \
  --strong-vit-heads 6
```

Run every row in `runs_compatibility_generality_v1/new_job_matrix.json` with
`run_generality_job.py`. The combined `job_matrix.json` also references the
five primary-suite runs and is used by the analyzer.

```bash
python scripts/analyze_generality.py \
  --suite-root runs_compatibility_generality_v1 \
  --bootstrap-resamples 100000 \
  --bootstrap-seed 20260819
```

The stronger ViT changes embedding dimension 48→96, depth 3→6, and heads 4→6;
the input size, patch size, and remaining experiment configuration are kept the
same. The generality intervals use the deterministic 100,000-resample seed
bootstrap.

## E. κ≈0 boundary audit

Build the audit from the completed primary suite:

```bash
python scripts/build_kappa_zero_audit_matrix.py \
  --source-suite-root runs_compatibility_causal_v15 \
  --output-root runs_kappa_zero_boundary_audit_v1
```

Execute each generated row with `run_kappa_zero_audit_job.py`, then:

```bash
python scripts/analyze_kappa_zero_audit.py \
  --suite-root runs_kappa_zero_boundary_audit_v1
```

The κ≈0 audit is kept separate from rows where the finite theorem conditions are
certified.

## F. Natural-state validation

Natural-state validation uses ordinary parent-stream states rather than
manipulated κ states. It is built from the primary 3-system × 5-seed causal
suite, not from the 10-seed generality matrix.

```bash
python scripts/build_natural_validation_matrix.py \
  --source-suite-root runs_compatibility_causal_v15 \
  --output-root runs_compatibility_natural_v1 \
  --states 50 \
  --probe-interval 10
```

Execute matrix rows with `run_natural_validation_job.py`, then:

```bash
python scripts/analyze_natural_validation.py \
  --suite-root runs_compatibility_natural_v1
```

Treat these results as observational associations, not as manipulated causal
effects.

## G. Fixed-norm natural-scale bridge

Read
[`research_protocols/natural_scale_bridge.md`](research_protocols/natural_scale_bridge.md)
before interpreting this experiment. The bridge depends on both the 10-seed
generality suite and the natural-state validation table.

Build the base fixed-norm bridge matrix:

```bash
python scripts/build_natural_scale_bridge_matrix.py \
  --source-generality-root runs_compatibility_generality_v1 \
  --natural-rows runs_compatibility_natural_v1/analysis/natural_state_validation_rows.csv \
  --output-root runs_compatibility_natural_scale_bridge_v1
```

Execute each generated row with `run_natural_scale_bridge_job.py`.

The final analysis includes a targeted completion pass for state-scale
conditions that satisfy the core fixed-norm construction but require evaluation
without treating finite-Δ0 CV/range diagnostics as admission thresholds. Build
that matrix from the completed base bridge:

```bash
python scripts/build_natural_scale_bridge_repair_matrix.py \
  --source-suite-root runs_compatibility_natural_scale_bridge_v1 \
  --output-root runs_compatibility_natural_scale_bridge_v1_delta0_repair
```

Execute its rows with `run_natural_scale_bridge_repair_job.py`, then produce the
merged final bridge analysis:

```bash
python scripts/analyze_natural_scale_bridge_repair.py \
  --source-suite-root runs_compatibility_natural_scale_bridge_v1 \
  --repair-suite-root runs_compatibility_natural_scale_bridge_v1_delta0_repair \
  --bootstrap-resamples 100000 \
  --bootstrap-seed 20260819
```

The completion pass changes only the admission treatment of finite-Δ0 spread:
CV/range remain reported diagnostics, while positive genuine unrestricted
endpoints and the fixed-norm/κ requirements remain required. The final bridge
analysis uses the deterministic 100,000-resample seed bootstrap.

## H. Regenerate figures

After the analyses above are present under their standard directories:

```bash
python scripts/make_afm_paper_figures.py \
  --root . \
  --out paper_figures
```

The figure script reads the saved outputs and does not rerun the experiments.
