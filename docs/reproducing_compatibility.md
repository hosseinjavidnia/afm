# Reproducing the compatibility experiments

This page gives the dependency order for the compatibility study.  Full suites
are GPU-intensive; every runner accepts a matrix path and one integer job index,
so users can map the matrix to their own scheduler.

## A. Main v1.5 controlled causal suite

```bash
python scripts/prepare_compatibility_cifar10.py --root data_compatibility/cifar10
python scripts/prepare_compatibility_text.py --output data_compatibility/text/input.txt

python scripts/build_compatibility_sweep_matrix.py \
  --output-root runs_compatibility_causal_v15
```

Run all 15 matrix entries (shown here sequentially):

```bash
for i in $(seq 0 14); do
  python scripts/run_compatibility_sweep_job.py \
    --matrix runs_compatibility_causal_v15/job_matrix.json \
    --index "$i"
done
```

Analyze:

```bash
python scripts/analyze_compatibility_frontier_v15.py \
  --suite-root runs_compatibility_causal_v15
python scripts/analyze_compatibility_reporting.py \
  --suite-root runs_compatibility_causal_v15
```

## B. Independent directions

```bash
python scripts/build_independent_direction_matrix.py \
  --source-suite-root runs_compatibility_causal_v15 \
  --output-root runs_compatibility_independent_directions_v1 \
  --requested-kappas 0.10 0.25 0.50 0.75 \
  --directions 4
```

Then execute each row in the generated `job_matrix.json` with
`run_independent_direction_job.py`, followed by:

```bash
python scripts/analyze_independent_directions.py \
  --suite-root runs_compatibility_independent_directions_v1 \
  --plot-method projection \
  --plot-beta 0.5
```

## C. Multiscale local robustness

```bash
python scripts/build_multiscale_matrix.py \
  --source-suite-root runs_compatibility_causal_v15 \
  --output-root runs_compatibility_multiscale_v1 \
  --scale-fractions 0.05 0.20 0.50 0.90
```

Execute generated matrix rows with `run_multiscale_job.py`, then:

```bash
python scripts/analyze_multiscale.py \
  --suite-root runs_compatibility_multiscale_v1 \
  --plot-method projection \
  --plot-beta 0.5
```

The scale fractions are log-expansion coordinates from the validated v1.5 local
reference toward a common attainable peak.  They are **not percentages of a
natural training update**.

## D. Generality

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

Execute generated new-job matrix rows with `run_generality_job.py`, then:

```bash
python scripts/analyze_generality.py \
  --suite-root runs_compatibility_generality_v1 \
  --bootstrap-resamples 100000 \
  --bootstrap-seed 20260819
```

## E. Natural-state validation

This uses ordinary parent-stream states rather than manipulated κ states.

```bash
python scripts/build_natural_validation_matrix.py \
  --source-suite-root runs_compatibility_generality_v1 \
  --output-root runs_compatibility_natural_v1 \
  --states 50 \
  --probe-interval 10
```

Execute matrix rows with `run_natural_validation_job.py`, then:

```bash
python scripts/analyze_natural_validation.py \
  --suite-root runs_compatibility_natural_v1
```

## F. Natural-scale bridge and Δ0 repair

The bridge is intentionally separate from the matched-Δ0 causal estimand.  See
`docs/research_protocols/natural_scale_bridge.md` before interpreting it.

Build/run/analyze the bridge with:

```bash
python scripts/build_natural_scale_bridge_matrix.py --help
python scripts/run_natural_scale_bridge_job.py --help
python scripts/analyze_natural_scale_bridge.py --help
```

The published corrected analysis removes the old finite-Δ0 CV/range **admission
threshold** by rerunning only rows whose sole rejection reason was
`finite_delta0_match_tolerance_not_met`:

```bash
python scripts/build_natural_scale_bridge_repair_matrix.py \
  --source-suite-root runs_compatibility_natural_scale_bridge_v1 \
  --output-root runs_compatibility_natural_scale_bridge_v1_delta0_repair
```

Execute the generated repair matrix with
`run_natural_scale_bridge_repair_job.py`, then merge/analyze with:

```bash
python scripts/analyze_natural_scale_bridge_repair.py \
  --source-suite-root runs_compatibility_natural_scale_bridge_v1 \
  --repair-suite-root runs_compatibility_natural_scale_bridge_v1_delta0_repair \
  --bootstrap-resamples 100000 \
  --bootstrap-seed 20260819
```
