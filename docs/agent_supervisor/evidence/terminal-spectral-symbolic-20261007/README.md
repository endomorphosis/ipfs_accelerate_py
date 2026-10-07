# Spectral operator qualification, 2026-10-07

The reviewed dominant-eigenpair operator computes and checks authored cases
with **zero runtime provider calls**. It handles maximum modulus rather than
maximum real value, including complex outputs for general real matrices.
The [implementation and improvement plan](../../spectral_symbolic_operator.md)
describes the contract and integration.

The fresh combined suite executed **113 tests**, with zero failures, errors or
skips. The earlier combined run passed 55 tests and reused 58 AST-seal results;
both records are retained. Required sealing remained enabled for the fresh run
with a new DuckDB catalog. Independent review passed 24 controls and bound the
five runtime/analysis/dispatch source files. The tests exercised an actual
isolated interpreter, 13 conditional Z3 lemmas, and the existing allocated
native candidate materializer. Candidate preparation left original source and
task state unchanged.

| Evidence | Result |
| --- | --- |
| [Final qualification](qualification.json) | Source bindings and scope of the fresh suite |
| [Fresh JUnit record](spectral-integration-tests-02.xml) | 113 executed tests; no skips or failures |
| [Independent review](independent-spectral-review-01.json) | Both identified acceptance gaps corrected |
| [Conditional algebra](conditional-spectral-algebra-01.json) | 13 ideal adapter lemmas; explicit assumptions |
| [Isolated target probe](isolated-spectral-target-01.json) | 22 authored cases, Python 3.12.3, NumPy 1.26.4, SciPy 1.11.4 |
| [Workflow qualification](doctor-spectral-contract-qualification-02.json) | Signed scope, exact evidence binding, staged materialization |
| [Performance gate](spectral-performance-not-qualified-01.json) | Failed consistent speed improvement |
| [Publication manifest](manifest.json) | Byte counts and digests for all 18 retained artifact bodies |

Both local 31-case timing attempts are retained. Only the two 1 by 1 cases
were faster; the final sizes 2–10 took 1.018–1.597 times the NumPy reference.
These timing runs used SciPy 1.17.1 outside the isolated probe and recorded
setup separately. The official Python 3.13/NumPy 2.3 target initially lacks
SciPy and remains unqualified. No official benchmark was launched.

Conditional SMT lemmas are not a Python, LAPACK, IEEE floating-point,
all-matrix or timing proof. The numerical cross-check is finite evidence, and
NumPy and SciPy may share LAPACK. The workflow has no completion or publication
authority. Explicit Doctor dispatch retains a residual model route while speed
is unqualified; the general planner and benchmark CLI defaults are unchanged.
There is no measured total-token saving claim. Runtime accounting excludes
the Codex implementation conversation and subagents; benchmark planning was
not rerun. No model weights were produced in this experiment.
