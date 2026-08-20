# Adaptive Functional Metaplasticity

Research code for **AFM-U**, the continual-learning implementation used in the
AFM functional-compatibility experiments.

This repository is organized as a public/reproducible release. It contains:

- the AFM persistent-assimilation and functional-protection implementation;
- the vision model and task-free streaming trainer used in the AFM study;
- the controlled causal functional-compatibility experiment;
- the independent-direction, multiscale, generality, natural-state, and
  natural-scale bridge analyses;
- a synthetic smoke test that does not require external data;
- unit tests for the core geometry and functional shield;
- the paper-figure generator.

Generated datasets, checkpoints, cluster logs, and completed run directories are
**not** included.

> **Research-code status.** AFM contains both theorem-aligned quantities and
> empirical finite-endpoint checks.  A run in empirical certificate mode must
> not be described as externally theorem-certified.  The code records this
> distinction explicitly.

## 1. Installation

Python 3.9+ and PyTorch 2.2+ are supported by the package metadata.  A CUDA GPU
is recommended for the full compatibility experiments; the smoke test and unit
tests run on CPU.

```bash
git clone https://github.com/hosseinjavidnia/afm
cd afm

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,figures]"
```

For the text-transformer reproduction, also install the experiment extras:

```bash
pip install -e ".[all]"
```

## 2. Thirty-second geometry check

The smallest AFM example does not train a neural network.  It constructs a
same-state ordinary comparator, projects it through a protected functional
subspace, allocates a normalized retention budget, and prints the resulting
compatibility and persistent path fraction.

```bash
python examples/compatibility_geometry.py
```

Expected qualitative checks:

- `compatibility_kappa` is `0.64` for the supplied toy geometry;
- `selected_path_fraction_lambda >= requested_charge_fraction_eta`;
- the persistent displacement has zero component in the protected coordinate.

## 3. End-to-end synthetic smoke run

This creates a tiny two-class image stream, runs AFM, evaluates the checkpoint,
checks implementation invariants, and writes a mechanism summary.

```bash
bash examples/quickstart.sh
```

Outputs are written to `smoke_output/run/` by default.  Important files are:

```text
resolved_config.json   exact configuration used by the run
runtime.json           runtime / device metadata
events.jsonl           transaction-level AFM log
final.pt               final checkpoint
summary.json           run summary (when produced by the trainer)
evaluation.json        evaluator result
```

You can choose another output directory:

```bash
bash examples/quickstart.sh /tmp/afm_smoke
```

## 4. Core AFM transaction

The protected update is deliberately separated into two stages.

1. **Persistent assimilation.**  Start from the same-state unrestricted
   comparator displacement `d0`, project it into the protected feasible
   subspace, and install the largest point on that projected path admitted by
   the normalized retention charge.
2. **Finite functional completion.**  After a persistent safe base is accepted,
   the functional shield can restore the requested finite current/protected
   outputs without replacing the persistent base by the unrestricted endpoint.

The main implementation modules are:

```text
afmvision/afm/persistent_assimilation.py  normalized retention budget and path
afmvision/afm/safe_step.py                protected geometry and safe steps
afmvision/afm/functional_shield.py        finite compact-cardinal completion
afmvision/afm/trainer.py                  full AFM transaction / streaming logic
afmvision/models/convnet_adapters.py      AFMConvNet and zero-gated adapters
```

The compact public API is available from `afmvision.afm`:

```python
from afmvision.afm import (
    FunctionalShield,
    make_counterfactual_normalized_plan,
    persistent_descent_lower_bound,
    retention_charge,
)
```

For the exact equations implemented by these functions, see
[`docs/algorithm.md`](docs/algorithm.md).

## 5. Run AFM on a manifest stream

The generic runner expects a YAML configuration and a JSONL image manifest.
The smoke configuration is a complete minimal example.

```bash
python scripts/run_experiment.py \
  --config configs/smoke.yaml \
  --method afm \
  --run-dir runs/my_afm_run \
  --set 'data.train_manifest="/path/to/train.jsonl"' \
  --set 'data.evaluator_manifest="/path/to/eval.jsonl"'
```

Each manifest row minimally contains:

```json
{"sample_id":"example-0","path":"/absolute/or/relative/image.png","label":3,"transform":{},"transform_seed":0}
```

The full trainer supports the AFM protected run and several baselines through
`--method`; use `python scripts/run_experiment.py --help` for the current list.

## 6. Reproduce the main causal compatibility experiment

The main experiment uses three fully trainable systems: CIFAR-10 CNN,
CIFAR-10 tiny ViT, and a character-level text transformer.  It manipulates the
realized functional compatibility of the current learning signal from the same
parent state, matches the genuine finite unrestricted learning opportunity, and
then evaluates the nonlinear retention frontier under a common absolute
retention reference.

Prepare CIFAR-10:

```bash
python scripts/prepare_compatibility_cifar10.py \
  --root data_compatibility/cifar10
```

Prepare the text corpus using WikiText-2:

```bash
python scripts/prepare_compatibility_text.py \
  --output data_compatibility/text/input.txt
```

or use your own UTF-8 corpus:

```bash
python scripts/prepare_compatibility_text.py \
  --input-file /path/to/corpus.txt \
  --output data_compatibility/text/input.txt
```

Build the frozen 3-system × 5-seed job matrix:

```bash
python scripts/build_compatibility_sweep_matrix.py \
  --output-root runs_compatibility_causal_v15
```

Run a single job by matrix index:

```bash
python scripts/run_compatibility_sweep_job.py \
  --matrix runs_compatibility_causal_v15/job_matrix.json \
  --index 0
```

The full matrix contains 15 independent GPU jobs.  On a workstation you may
run the indices sequentially; on a cluster, map one matrix index to one GPU job
using your own scheduler.

Analyze the completed suite:

```bash
python scripts/analyze_compatibility_frontier_v15.py \
  --suite-root runs_compatibility_causal_v15

python scripts/analyze_compatibility_reporting.py \
  --suite-root runs_compatibility_causal_v15
```

The reporting script treats **seed as the independent unit** for matched-state
slope summaries.

More protocol detail is preserved in
[`docs/research_protocols/causal_compatibility_design.md`](docs/research_protocols/causal_compatibility_design.md).

## 7. Extension experiments

The public release includes the experiment code used to test alternative
explanations of the compatibility result:

| Experiment | Build | Run | Analyze |
|---|---|---|---|
| Independent directions | `build_independent_direction_matrix.py` | `run_independent_direction_job.py` | `analyze_independent_directions.py` |
| Multiscale local interventions | `build_multiscale_matrix.py` | `run_multiscale_job.py` | `analyze_multiscale.py` |
| 10-seed / stronger-ViT generality | `build_generality_matrix.py` | `run_generality_job.py` | `analyze_generality.py` |
| κ≈0 boundary audit | `build_kappa_zero_audit_matrix.py` | `run_kappa_zero_audit_job.py` | `analyze_kappa_zero_audit.py` |
| Natural-state validation | `build_natural_validation_matrix.py` | `run_natural_validation_job.py` | `analyze_natural_validation.py` |
| Natural-scale bridge | `build_natural_scale_bridge_matrix.py` | `run_natural_scale_bridge_job.py` | `analyze_natural_scale_bridge.py` |
| Δ0 bridge repair | `build_natural_scale_bridge_repair_matrix.py` | `run_natural_scale_bridge_repair_job.py` | `analyze_natural_scale_bridge_repair.py` |

These experiments intentionally depend on outputs from the main v1.5 causal
suite.  See [`docs/reproducing_compatibility.md`](docs/reproducing_compatibility.md)
for the dependency order and exact commands.

## 8. Figures

Once the experiment analyses are present under their standard run directories,
regenerate the paper figures with:

```bash
pip install -e ".[figures]"
python scripts/make_afm_paper_figures.py \
  --root . \
  --out paper_figures
```

The figure builder reads stored analysis tables / row-level outputs.  It does
not rerun GPU experiments.

## 9. Tests

Run the curated public test suite with:

```bash
python -m pytest -q
```

The tests cover:

- safe-radius saturation and protected projection;
- Frequent Directions sketch guarantees;
- normalized retention-budget/path-fraction behavior;
- compatibility calculation using the true ordinary gradient energy;
- the analytic persistent-progress reference;
- compact-cardinal functional-shield behavior;
- controlled compatibility interventions on an exact toy model.

## 10. Repository layout

```text
afmvision/
  afm/               core AFM implementation
  compatibility/     causal compatibility and extension experiment kernels
  models/            AFMConvNet and compatibility-study models
  data/               manifest / stream utilities
  baselines/          continual-learning baselines used by the study
  validation/         theorem/mechanism validation utilities
configs/
  smoke.yaml
  compatibility/     frozen causal-study configurations
examples/
  compatibility_geometry.py
  quickstart.sh
scripts/              runnable experiment / analysis entry points
docs/                 algorithm and reproducibility documentation
tests/                curated public CPU test suite
```

## 11. Reproducibility rules

A few distinctions are essential when interpreting AFM outputs:

- **Requested κ vs realized κ:** causal analyses use realized compatibility.
- **Persistent base vs finite completion:** finite endpoint restoration is not
  relabeled as persistent parameter learning.
- **Matched-Δ0 experiment vs fixed-norm bridge:** these answer different causal
  questions and must not be pooled as one estimand.
- **κ≈0 boundary:** theorem-aligned coarse references at the zero boundary are
  reported separately from rows where finite step/curvature conditions are
  certified.
- **Natural-state validation:** conditional natural compatibility analyses are
  observational; they are not described as manipulated causal effects.

See [`docs/reproducibility.md`](docs/reproducibility.md) for the full checklist.

