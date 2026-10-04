# Extended native-startup retry, 2026-10-04

The fresh Docker qualification passed, but the single official
`fix-code-vulnerability__FMKFoso` trial scored **0**. The supervisor did not
complete the task. This trial did **not** exercise the new native START allowance:
resource admission failed during runtime construction, before START.

Built revisions: accelerator `a8d1e15e91ad9ec47622ddcb36ed4dfde2bebaec`, datasets
`d5aaf256618c53b009695e8d5618436fba1c0419`, kit
`a9b98beac1ef14278b4adf4f2289cd509530fb86`.
Archive SHA-256:
`63560e7bb7f7376e9b39663d395d45206f81c0839232e869c6a43f972aa190be`.
All 27,214 archive members and 12,013 repository member pins were audited.
Checkpoint, embedding and Torch assets were unchanged from the prior generation.

The explicit profile uses 5 CPUs, 16 GiB, a 900-second driver limit, an
840-second work cutoff, and a 60-second cleanup reserve. Requested START was
120 seconds and STOP 20 seconds. Qualification and the full trial both ran with
Leanstral paused under the user's authorization. The separate service record
requires successful restart and HTTP model readiness before this export.

Qualification took 382.750 seconds. Actual checkpoint/native worker execution
ran; Source384 took 77.118 seconds, initial context 133.234 seconds, and the warm
observation 8.914 seconds. Public task inputs, archived source pins, and
qualification cleanup checks passed.

The full-trial controller took 493.706 seconds. Agent setup took 196.427 seconds;
driver work ended after 269.311 seconds, below its work cutoff. Initial indexing
produced 426 symbols and 531 semantic capsules; Source384 took 70.730 seconds.
Intent-symbolic planning qualified with two goals, one task, and zero provider
calls. The Doctor produced a candidate and verified the bounded local property
`reject-control-preserve-safe`. This grants neither whole-program proof nor
SecurityIR obligation discharge, source publication, or completion authority.

`AdmittedBenchmarkRuntime.create` replayed planning admission and raised
`LeaseTimeoutError` while acquiring resources for header applicability. The
retained request-primary observation reports `proof_memory_stall`: host pressure
27.62% exceeded the explicit 10% limit, while visible container pressure was 0%.
The last sample showed 12,209 MiB available for a 1,024 MiB request plus 3,277 MiB
headroom. The pressure observation explains the admission refusal, not its
underlying cause or the complete admission decision. Later host samples and
service-restoration activity cannot retroactively establish that cause.

No START, STOP, task-state or startup-phase timing receipt exists. Worker cleanup
returned zero, and the trial container was removed. Remaining-process count is
absent; it is not represented as zero. Provider invocation count is zero for this
failed prefix, not a completed-task token score. Reviewed intent-contract costs
remain outside runtime timing and have no measured token attribution. Matched
Codex/no-index/indexed comparisons and any efficiency claim remain pending.

The result export preserves bounded admission and startup metadata and binds its
inspector and production-projector hashes. It excludes credential contents,
hidden verifier bodies, source bodies, and model output. The preceding B/C/N
failures remain retained as separate observations.

GitHub's documentation job did not start: its annotation reports an account
billing lock. Other workflow failure causes were not audited.
