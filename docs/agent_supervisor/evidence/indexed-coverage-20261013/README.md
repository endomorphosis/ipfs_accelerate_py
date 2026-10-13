# Indexed supervisor readiness evidence — 2026-10-13

These records qualify the output observer and optional Intent384 checkpoint
route. They contain no new full-supervisor benchmark, official reward, token
score or discharged numerical proof. The [implementation report](../../indexed_coverage_readiness.md)
explains the changes and remaining coverage work.

- [Output-observer diagnosis](output-observation/diagnosis.json) records the
  production-writer permission mismatch and four reproduced failures before
  the fix. The owner event log may be `0644` inside its private `0700` directory.
  The [network-disabled Docker fixture](output-observation/docker/run-02/qualification.json)
  then captures [candidate and canonical hashes](output-observation/docker/run-02/results/observation.json)
  using real binding/event writers and authored lifecycle records. This does
  not recover the removed containers' historical outputs or establish a full
  supervisor START. Its first attempt stopped on fixture extension-file
  permissions; the second reused the image's hash-verified existing extension.
- [Real Intent384 qualification](intent384/qualification-03/qualification-result.json)
  consumes unchanged released checkpoint
  `4f3fd17ea2d908fe36c57a444517f3cc0a983cab26f0f67f2e32371e25def3cd`
  and local pinned GTE assets. The authored scalar instruction produces a
  source-audited candidate with four actual embedding/inference calls across
  preparation and replay. The original benchmark instruction remains outside
  the grammar and makes zero such calls. No training, LLM calls, downloads,
  logic projection or Lake build ran in this qualification.
- [Earlier checkpoint diagnosis](intent384/compatibility-diagnosis.json) and
  [initial failed qualification](intent384/qualification-result.json) precede
  the exact logging-pin compatibility fix. The
  [second qualification](intent384/qualification-02/qualification-result.json)
  loaded the checkpoint but encountered incompatible installed tokenizer
  versions. The successful third run selected already-installed matching
  dependencies through its recorded import path; it installed nothing.
- [Real-asset readiness](intent384/readiness-qualification-result.json)
  separates selection, current local load eligibility and complete-instruction
  grammar support, with zero embedding or decoder calls. An archive binding
  without an explicit local config leaves runtime load eligibility unchecked.
- [Numerical coverage audit](four-function-coverage-audit.json) matches all
  four public source spans, with two structurally complete functions, four
  assertion requirements and 45 pending obligations. Zero learned candidates
  match the supported scalar source grammar and zero source obligations are
  discharged. The [audit helper](audit_four_numerical_functions.py) retains the
  bounded next steps and exact-source replay procedure.

The [deduplicated test ledger](combined-test-results.json) contains **742
distinct passing cases**: observer/progress/comparison (295), readiness and
preparation (291), and shared Intent plus supervisor wiring (182), with 26
overlapping transport cases counted once. The
[two explicit real-checkpoint tests](intent384/optional-tests-01-result.json)
close the initial optional skips. They exercise real model inference, goal
preparation and saved-advice replay with an authored local planner response;
they do not call a live provider. The [Intent closeout](intent384/final-summary.json)
records that distinction and the final source hashes.

[Observer review](terminal-output-observer-independent-review.json) and
[Intent/readiness review](intent-action-readiness-independent-review.json)
record the independent checks and their limits. Successful regression suites
retain commands and reports under `output-observation`, `intent384` and
`readiness`. Failed exploratory attempts are historical evidence, not counted
as passing validation. Local test logs remain outside this publication when
they add no content-free evidence; source-bearing advice, checkpoints, model
assets, private owner state, credentials and databases are excluded.

The local qualification uses accelerator source based on `49b01d2515f338af73f8c14a96bbf235e339e4bd`
with the accompanying committed changes and datasets
`afadfbcf8d33a67568c133bb761c43d42a9a133c`. Hashes in the records bind
the reviewed code and checkpoint bytes. Saved paths describe the original
local environment; reproducing a helper requires equivalent explicit assets
and dependency paths. A passing grammar check, source agreement or compiled
syntax must remain separate from a proof of benchmark behavior.
