# Frozen formula initialization boundary

The container's selected formula decoder and reviewed header protocol now reach
the native initial-context builder through `terminal_indexed_preparation.initial_context`.
The wrapper previously raised `TypeError` for those arguments before inference.
The change adds two optional parameters and forwards them unchanged; existing
selection validation remains in the native builder.

The new boundary tests reproduced three pre-fix failures. After the change,
four relevant suites passed **39 cases in 90.62 seconds**, with no skips. The
positive boundary test builds explicitly authored checkpoint fixtures, then uses
frozen inference through the actual runtime-selection and preparation wrappers.
It verifies formula registration, learned formula output, native DuckLake
hydration, exact saved selections and persisted artifact replay. Inference cannot
retrain the formula checkpoint; provider/download/training call counts stay zero
in the frozen advice receipt. Negative controls reject a decoder without its
security checkpoint and a protocol without its decoder before index writes.
The surrounding suites cover frozen security advice, archived formula assets,
selection drift, and initial index/planner boundaries.

These are authored local integration controls. They do not run an official
Terminal-Bench verifier, establish holdout generalization, qualify a new native
supervisor START, or measure token savings. The three failures are retained as
pre-fix diagnostics, not counted as successes. Native registry databases,
training/temp directories and private capability material are excluded from
this bounded evidence package. Source snapshots preserve the changed files and
the directly relevant tested producers and neighboring tests.

The separately qualified finite source384 supervisor uses three CPU/process
slots and a 6-GiB parent envelope. The current Harbor task remains 1 CPU and
2 GiB and uses the older Terminal-Bench composition. Source384 preparation
selection, immutable model/index binding and its observer still need integration
into that container path with a qualified common resource profile. This wrapper
repair does not imply that the new finite methodology ran on Terminal-Bench.
