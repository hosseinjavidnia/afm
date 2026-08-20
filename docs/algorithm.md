# AFM algorithm notes

This document describes the core quantities implemented in the public code.  It
is a code guide, not a replacement for the paper.

## Same-state unrestricted comparator

For ordinary current loss gradient `g_t`, step size `alpha_t`, and trainable
coordinate mask/preconditioner `A_t`, the same-state no-protection comparator is

\[
d_t^0=-\alpha_t A_t g_t.
\]

The protected feasible component is

\[
v_t=\Pi_t d_t^0,
\]

where `Pi_t` is the projection induced by the protected functional geometry and
any active coordinate mask.

In code, the exact ordinary comparator is generated first.  Persistent
assimilation projects that already-defined comparator; it does not install the
unrestricted endpoint and then try to repair it.

## Compatibility

The compatibility statistic used by the same-state theory is

\[
\kappa_t=\frac{\|\Pi_t g_t\|^2}{\|A_t g_t\|^2}.
\]

`make_counterfactual_normalized_plan` computes this from the **ordinary gradient
energy**, not from the final protected proposal.

## Normalized retention charge

For a protected linear/curvature envelope

\[
C_t(s)=E_t s+\frac{\overline H_t}{2}s^2,
\]

and reference length `s_t^0 = ||v_t||`, AFM allocates

\[
b_t^{norm}=\eta_t C_t(s_t^0).
\]

It then selects the largest path fraction `lambda_t` satisfying

\[
C_t(\lambda_t s_t^0)\le b_t^{norm}.
\]

Because `C_t` is convex with `C_t(0)=0`, the selected fraction obeys

\[
\lambda_t\ge\eta_t
\]

whenever the nonzero reference is inside the declared trust cap.

The implementation is in
`afmvision/afm/persistent_assimilation.py`.

## Persistent-progress reference

For a scalar ordinary gradient comparator satisfying the executable alignment
and step-size/smoothness conditions, the implementation records the smoothness
quotient

\[
\frac{\Delta_t^{base}}{\Delta_t^0}
\ge
\widehat\lambda_t\kappa_t
\frac{1-\frac12\alpha_t\overline L_t\widehat\lambda_t}
     {1+\frac12\alpha_t\overline L_t}.
\]

The code also uses the coarser quantity

\[
\widehat\lambda_t\kappa_t/3
\]

as a **theorem-aligned coarse reference** when the corresponding conditions are
being audited.  Do not describe every numerically computed coarse-reference row
as independently theorem-certified.

## Finite functional completion

AFM keeps persistent parameter learning separate from finite requested-output
restoration.  After an accepted persistent base update, the compact-cardinal
`FunctionalShield` may store a bounded finite residual that restores requested
current/protected outputs.  The unrestricted comparator endpoint is never
installed as the protected persistent base.

Relevant code:

- `afmvision/afm/functional_shield.py`
- `afmvision/afm/full_progress_restoration.py`
- `afmvision/afm/trainer.py`

## Strict vs empirical certificate mode

The trainer supports research configurations where external smoothness / error
bounds are unavailable.  In empirical mode, exact finite endpoint checks are
performed, but this does **not** turn empirical estimates into theorem
certificates.  Strict/certified language should be reserved for runs supplied
with the required externally justified bounds.
