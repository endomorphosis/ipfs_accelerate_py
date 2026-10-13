# Guarded numerical qualification — 2026-10-13

This bundle records the explicit `python-numerical-guarded-structure@2` profile.
The [implementation report](../../numerical_guarded_coverage.md) describes the
scope. [manifest.json](manifest.json) hashes every included file except itself.
Raw transcripts and receipts are copied byte-for-byte; owner-private state,
database files, prompts and raw capability/session descriptors are excluded.

| Observation | Evidence | Result |
| --- | --- | --- |
| Historical v1 baseline | [baseline-v1.json](baseline-v1.json) | Exact public reports and structured-checkpoint implementation pins retained |
| CPython evaluation order | [authored-evaluation-observation.json](authored-evaluation-observation.json) | 17 authored cases passed; benchmark source never executed |
| Public source and Lake | [public-01/qualification.json](public-01/qualification.json) | 3/4 complete structures, 75 pending obligations, 3 successful builds and 1 expected rejection |
| Source/Lean review | [public-evidence-audit.json](public-evidence-audit.json) | 59 independent checks passed |
| Docker worker retrieval | [docker-01/qualification.json](docker-01/qualification.json) | Exact 75-row retrieval, custody/refusal/cleanup checks passed |
| Docker review | [docker-evidence-audit.json](docker-evidence-audit.json) | 42 independent checks passed |
| Regression ledger | [test-ledger.json](test-ledger.json) | 947 distinct passes, 2 explicit checkpoint tests skipped, no final failures |
| Upstream integration | [publication-inputs.json](publication-inputs.json) | Concurrent LegalIR change preserved; all 44 qualification source bindings unchanged |

The shared-library final run contains 338 passes with all six selected optional
native-Lake checks enabled. The accelerator run contains 609 passes and two
skipped learned-checkpoint integration cases whose explicit local assets were
not selected. The ledger counts only these two final runs, not overlapping
development runs. The new profile itself requires no learned checkpoint.

The network-disabled container used the unchanged generated production owner
entrypoint, worker identity transition, actual `llm_router` runtime, obligation
IPC and native DuckLake hydration. The fixture replaced route discovery and
model response generation with authored callbacks. It did not execute a live
provider, learned decoder, full supervisor START, official verifier or benchmark
solution. Provider labels in the routing receipt are fixture metadata.

Initial retrieval returned 75 unique obligations across 21 pages totaling
147,671 UTF-8 response bytes, with a maximum of 8,006 bytes. The complete broker
session returned 55 responses totaling 314,709 bytes, including filters,
recovery and seven refusals. Repeated queries delivered 157 row instances;
that is not 157 distinct obligations. Response sizes exclude four-byte IPC
frame prefixes. The worker could not read five owner-private artifacts. The
fixture checked restored source bytes before deleting its own container;
those container workspaces are no longer available for a fresh external check.

Native Lean receipts include the exact generated input, toolchain and executable
hashes. Guard and requirement parameters remain unresolved. Successful builds
establish declaration compilation; **zero source obligations were discharged**.
Loops, learned numerical composition and source-to-library refinement remain
uncovered. There is no new reward, token score or evidence of token savings.

Development observations are retained: an initial frontend smoke found an empty
format component with a zero-width AST span; the fix preserves its byte span
without inventing column ranges. Four accelerator mutation fixtures initially
used the exclusive-create writer to overwrite a receipt; the fixtures were
corrected. A later planner fixture correctly rejected a concurrent producer
change. The frozen final runs passed these cases. Independent reviews are in
[logic-integration-audit.json](logic-integration-audit.json) and
[guarded-consumer-independent-review.json](guarded-consumer-independent-review.json).

Helpers retain original local worktree paths and require the installed, pinned
dependencies. The Docker runner needs the recorded existing image and native
DuckDB extensions. These are historical execution inputs, not a claim that
arbitrary hosts can reproduce the result without provisioning those assets.
