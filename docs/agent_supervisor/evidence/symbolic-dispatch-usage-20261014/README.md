# Symbolic dispatch and bounded usage observations

Implementation commit `480effcfbb` follows the three retained live compaction
runs. No new live benchmark or external model call was made for this change.

Supported model-input treatments no longer reject an independently checked
Doctor candidate. The selected candidate still passes the existing source,
publication, validation and typed-completion gates. Requested model controls
remain visible as not applicable on the provider-free route. Observers require
a matching successful materialization, successful cleanup and no coding-provider
attempt before using that designation; missing evidence remains unavailable.

Grok progress version 2 retains the first 128 usage events as closed numeric
observations with explicit missing values and omitted-event counts. Version 1
records remain accepted without fabricated histories. Repeated progress updates
use the count-only version 1 shape; the final receipt retains version 2 history
once. Final-envelope accounting is unchanged, and event values are not summed.

The [qualification](qualification.json) and [per-file ledger](tests-ledger.json)
record **691 distinct passing tests** across 14 files:

| Selected evidence | Passing cases |
| --- | ---: |
| Doctor routing, excluding superseded dispatch file | 229 |
| Final complete Doctor dispatch file | 27 |
| Final Grok observer, stream and router tests | 156 |
| Comparison, archive and context-audit integration | 279 |

All selected test-file hashes and their relevant production hashes match the
final implementation. The initial Doctor run's one failed fixture is retained:
its setup tried to materialize an already-created task, which production
correctly rejected. The corrected test claims that existing task and then checks
actual Z3/Lean repair, candidate materialization, Quack claim, failed pre-repair
validation, native Git/queue publication, successful owner validation and typed
completion. The fixture forbids LLM entrypoints. This is an integration test,
not a fresh full-daemon START or official benchmark result.

The first Doctor run also captured unrelated concurrent Grok source changes.
Its selected Doctor files and production dependencies remained unchanged;
the complete affected file was rerun, and all final suites recorded unchanged
sources. Earlier successful runs and the redundant one-test lifecycle rerun
are not added to the final count.

The [independent review](independent-review.json) binds seven production files.
The [size check](grok-usage-size-check.json) measures a 68,274-byte detailed
snapshot and 2,798-byte count-only progress line using authored large numeric
values. At the stated 1,800-second budget and five-second update interval,
362 updates plus that snapshot occupy 1,081,150 bytes. This describes the
telemetry contribution, not an upper bound on all worker output or the full
invocation receipt.

The [producer reference](grok-usage-producer-reference.json) identifies locally
installed Grok 1.0.52 documentation. Tests use authored frames matching that
documented shape, not a new live trace from the earlier pinned 1.0.46 trials.
Counter additivity, cache semantics, auxiliary-model coverage and complete
billing remain unknown. Event ordinals are not authenticated turn identifiers.

The manifest binds these records and retained command, source-hash, exit and
JUnit files. No credentials, private databases, raw provider transcripts or
runtime archive are included. The [benchmark report](../../compact_worker_benchmark.md)
keeps the measured results separate from this implementation qualification.
