# Reproducibility checklist

Use this checklist when producing results from the public AFM repository.

## Environment

Record:

- git commit hash;
- Python version;
- PyTorch version;
- CUDA version and GPU model where applicable;
- resolved YAML configuration;
- random seed.

`run_experiment.py` writes the resolved configuration and runtime metadata into
each run directory.

## Same-state causal experiments

For causal functional-compatibility experiments:

1. start all κ interventions from the identical serialized parent state;
2. use **realized κ** in analysis;
3. keep the unrestricted comparator direction/finite calibration rule fixed;
4. use the common absolute retention reference declared by the v1.5 protocol;
5. keep seed as the independent unit in cross-state bootstrap summaries;
6. do not merge native AFM finite-completion diagnostics with the common
   persistent-frontier estimand.

## Original matched-Δ0 experiment

The main v1.5 experiment matches genuine finite unrestricted progress across κ.
Its normalized persistent ratio is therefore a controlled matched-progress
estimand.

## Natural-scale bridge

The corrected fixed-norm bridge removes finite-Δ0 CV/range as **admission
criteria**.  It still requires a positive unrestricted endpoint for every
requested κ.  Δ0 heterogeneity is diagnostic rather than a reason to discard an
otherwise valid fixed-norm state.

Zeros in bridge feasibility at larger scales mean that the all-κ same-norm
comparison could not be constructed; they are not zero causal-effect estimates.

## Natural-state validation

Natural-state compatibility is observed rather than manipulated.  Report its
conditional association separately from the controlled causal experiment.

## Generated outputs

Do not commit:

- downloaded datasets;
- prepared data caches;
- `.pt`/`.ckpt` checkpoints;
- `runs*` directories;
- scheduler logs;
- generated figures.

Instead, publish a release/archive of the exact compact analysis tables required
for the paper if journal policy requires direct numerical reproduction.
