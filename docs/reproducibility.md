# Reproducibility checklist

Use this checklist when producing results from the public AFM repository.

## Environment and provenance

Record the following for every run:

- git commit hash;
- Python version;
- PyTorch version;
- CUDA version and GPU model where applicable;
- resolved configuration; and
- random seed.

The runners write resolved configuration and runtime metadata into the run
outputs. Keep those files with any derived tables or figures.

## Controlled same-state causal experiment

For the primary functional-compatibility experiment:

1. start all κ interventions from the identical serialized parent state;
2. analyze **realized κ**, not only the requested intervention label;
3. preserve the genuine same-state unrestricted direction and finite comparator
   calibration rule;
4. match the finite unrestricted decrease at each state to
   `0.5 × minκ Δ0,raw(κ)` by one-dimensional bisection along each constructed
   unrestricted direction;
5. use the state-level common absolute retention reference
   `D_ref = maxκ D_unrestricted(κ)` for all κ interventions and methods;
6. keep seed as the independent inferential unit; and
7. keep native AFM finite-completion diagnostics separate from the common
   method-neutral persistent frontier.

The primary matched-effect hierarchy is:

- fit one κ→persistent-ratio slope across the six realized-κ interventions
  within each causal state;
- average the 50 state slopes within each seed;
- summarize the five seed means with a two-sided 95% Student-t interval
  (`df=4`).

The cross-system descriptive summary pools the 15 system-seed effects only for
that descriptive calculation and uses a two-sided 95% Student-t interval with
`df=14`.

## Follow-up uncertainty conventions

Do not silently substitute one uncertainty convention for another.

- Independent-direction and multiscale five-seed follow-ups use the exact
  `5^5 = 3125` ordered seed bootstrap implemented by their analysis scripts.
- The 10-seed generality analysis uses a deterministic 100,000-resample seed
  bootstrap.
- The fixed-norm bridge uses its declared deterministic 100,000-resample seed
  bootstrap.

These are follow-up analyses and do not replace the primary five-seed Student-t
analysis.

## Fixed-norm natural-scale bridge

The bridge asks a different question from the matched-Δ0 experiment. It holds
the unrestricted update norm fixed at fractions of a natural-update reference
and tests whether the requested all-κ comparison can be constructed.

A bridge state-scale condition requires a positive genuine unrestricted
endpoint at every requested κ and the declared update-norm and κ tolerances.
Finite-Δ0 CV/range values are diagnostics, not admission criteria in the final
merged bridge analysis. A feasibility value of zero at larger scales therefore
means that the all-κ same-norm comparison could not be constructed; it is not a
zero causal-effect estimate.

See
[`research_protocols/natural_scale_bridge.md`](research_protocols/natural_scale_bridge.md)
for the bridge interpretation and reproduction sequence.

## Natural-state validation

Natural-state compatibility is observed rather than manipulated. Report its
conditional association separately from the controlled causal experiment.

## Figures

`make_afm_paper_figures.py` reads completed analysis outputs. It does not rerun
GPU experiments. Before using a generated figure, confirm that the expected
analysis directory is the one resolved by the script and that the corresponding
validation file passes.
