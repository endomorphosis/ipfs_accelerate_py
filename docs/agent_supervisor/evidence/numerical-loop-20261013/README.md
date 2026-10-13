# Numerical loop qualification evidence

The [report](../../numerical_loop_coverage.md) describes the explicitly selected
v3 control-flow profile. [manifest.json](manifest.json) seals the exact files in
this directory. Results are copied without rewriting. Private owner keys,
session descriptors, prompts, raw broker accounting, full worker response dumps
and DuckDB/DuckLake catalogs remain outside the publication.

- [Public qualification](public-02/qualification.json): four complete function
  structures, 105 pending obligations, two compounds, four assertions and two
  loops. `eval.py` remains partial at module scope. Three actual Lake builds
  passed and an authored false theorem failed. Source code was not executed.
- [Regression ledger](test-ledger.json): 1,096 distinct passes (426 datasets,
  670 accelerator), two explicit learned-checkpoint skips. Focused runs are
  retained but not counted twice. All six selected native-Lake tests executed.
- [Docker qualification](docker-03/qualification.json) and
  [independent audit](docker-evidence-audit.json): real owner/worker/router/IPC,
  56 initial pages containing all 105 rows, 377,548 initial response bytes,
  maximum 8,180 bytes. The full 64-request session used 385,404 response bytes,
  including seven refusals and recovery. Initial rows 105 differ from 107
  delivered row instances because recovery repeats two rows. Frame prefixes
  are excluded from response-byte totals. No live provider or token meter ran.
- [Earlier fixture diagnosis](docker-fixture-failure-diagnosis.json): two
  attempts exhausted the unchanged 64-request quota through redundant reads.
  The completed full-scan fixture omits successful filter repeats; those
  combinations passed the separate regression tests. It has no spare request.
- [Frontend review](frontend-cross-layer-audit.json),
  [logic review](logic-independent-review.json), and
  [public evidence review](public-evidence-audit.json) retain independent
  source/CFG/projection checks and current file hashes. The
  [authored CPython probe](authored-loop-protocol-observation.json) executed
  only small authored examples, not the benchmark.
- [Planning-summary review](consumer-summary-independent-review.json): the
  7,614-byte summary retains both loop/CFG inventories and three assertions. All
  105 pending rows are omitted inline with exact counters; full scopes are
  available through retrieval.
- [Publication inputs](publication-inputs.json) bind the selected closure,
  frozen historical reports and checkpoint implementation pins, including
  preserved concurrent upstream Intent checkpoint compatibility tests.

The replay baseline and final public reports are source-bound artifacts, not
proof certificates. Reachability, effects, protocol semantics, invariants,
termination and all 105 source obligations remain unresolved. Guarded and control
sidecars are mandatory; standalone ProgramIR is insufficient. FOL extracted
expressions retain proof scope as well as their enclosing report. No operational
TLA+/transition-system interpretation is claimed.

Docker used a writable base and root setup for the existing sudo worker
transition, with networking disabled and the worker constrained to zero effective,
permitted and ambient capabilities plus no-new-privileges. Five owner-private
reads were refused. These bounded checks do not establish general sandbox
security or full supervisor START qualification.

No training, checkpoint upload, learned Source384/IntentIR inference, official
verifier, benchmark score or token-saving measurement occurred. Compact planner
summaries are bounded; exhaustive retrieval has a larger transport cost than v2.
The new metadata needs a live selective-retrieval ablation before making an
efficiency claim.
