# Grok worker cleanup and native failure settlement — 2026-10-06

**Fresh full-indexed trial12 timed out with official reward 0.0, but its callback
settled and native shutdown completed cleanly.** This closes the reproduced
timeout-custody gap; it does not establish task correctness or token savings.

The isolated router worker now reaps its own descendants before returning from
the provider call. An actual Docker replay reproduced adopted zombies with the
old worker and observed none with the patched worker. Timeouts still return a
failure; cleanup does not authorize task completion or a retry.

| Disconnected Docker fixture | Old worker: adopted zombies | Patched worker |
| --- | ---: | ---: |
| No descendants | 0 | 0 |
| Detached `setsid` child | 1 | 0 |
| Double fork | 3 | 0 |

The [six-case receipt](offline-worker-custody-02.json) covers the real
owner → sudo → UID1001 worker and shared `llm_router` Grok adapter, using an
authored executable and synthetic credentials. Setup used network access;
all six cases ran after network disconnection. No remote model calls occurred.
The observer reaped the baseline zombies, and the exact container and network
were removed. This is process-boundary qualification, not an official task score.

## Fresh live result

Trial12 used clean source `092a467a560d150e904fc6f680720c781bdb7698` and a newly
audited archive with capability `@2`. The original `tune-mjcf` verifier returned
**0.0**. Grok 4.7 ran through `llm_router` with the same 600-second coding cap,
20 GiB limit, five CPUs, retrieval model and SecurityIR checkpoint as trial11.
There was one planning call and one coding call, with no fallback or retry.

| Observation | Trial11, before this fix | Trial12 |
| --- | --- | --- |
| Official reward | 0.0 | 0.0 |
| Coding router time | 600.25 s, timeout | 600.23 s, timeout |
| Callback | Outcome unknown | **Failed outcome settled** |
| Native exit receipt | Absent | Present; all four custody gates passed |
| Task | In progress, revision 3 | Blocked, revision 4 |
| START / STOP / runtime close | Succeeded | Succeeded |
| Remaining tracked processes | 0 | 0 |
| Planning native tokens | 22,806 | 22,072 |
| Coding / combined tokens | Unknown | Unknown |

The [live summary](live-trial-summary-12.json),
[closure receipt](trial-closure-observation-12.json) and
[performance observation](performance-observation-12.json) retain these results.
The new native exit receipt reports the reaped process, absent process group,
absent subreaper children and finalized lifecycle. Worker cleanup returned zero;
all three observers exited successfully, and an independent exact-container
inspection confirmed its absence. The monitor observed no OOM kills and a peak
of 7,460,016,128 bytes under the 21,474,836,480-byte limit.

Planning took 69.25 router seconds; the driver took 734.99 seconds, and the
recorded execution recipe took 947.45 seconds including the surrounding harness.
These are separate scopes. Planning's 22,072 tokens comprise 13,293 uncached
input, 1,536 cached input and 7,243 output; reasoning tokens are a subset of
output. The timeout emitted no final coding usage envelope, so coding and
aggregate tokens and billing remain unknown. These single runs are not a
matched performance-advantage experiment.

The [indexed-path record](indexed-path-observation-12.json) confirms actual
learned retrieval, four indexed symbols, eleven semantic capsules and fresh
Source384 checkpoint inference. Initial indexing took 17.20 seconds; the
9.51-second inference measurement is nested in that work. The admitted context
reused the index and supplied three worker capsules totaling 29,875 bytes.
Planning independently admitted two goals and one task. No training or parallel
task execution occurred. Doctor retained one residual with six coverage gaps;
it did not produce a symbolic repair or local contract proof for this task.

## Runtime changes

The fresh, dedicated worker verifies Linux subreaper support and an empty,
stable child census before invoking the router. Its `finally` cleanup reuses
the existing orphan-reaping primitive under the worker's own Unix identity.
The outer watchdog remains responsible for a worker that cannot finish cleanup.
No global daemon cleanup was added.

Bridge diagnostic `@2` retains up to four ordered observations: issuance,
native exit, durable bridge join, and one-use capability consumption. Fixed
reason codes distinguish refused checks without exporting process identifiers,
raw exceptions or model output. Diagnostic `@1` remains readable. These fields
are observations only; the native process receipt, signed admission, original
claim, lease/fence, protected-path checks and settlement CAS remain authoritative.
Late process cleanup alone does not settle an unknown callback.

Every supervisor resource profile now requires archive capability
`terminal-router-worker-capability@2`, which binds the exact installer, router
and child-custody helper bytes. Missing, stale and `@1` capabilities are rejected
before planning or deployment, including for the original 300-second profiles.
Existing supervisor archives must be rebuilt. Resource budgets are unchanged.

## Targeted qualification

The [qualification summary](qualification-summary.json) retains twelve separate
runs, including failures, a nonexistent initial target and an intentionally
interrupted duplicate collection. Repeated cases are not summed.

| Final component check | Result | Recorded wall time | Source |
| --- | --- | ---: | --- |
| Native lifecycle, custody, router and reply contracts | 220 passed | 411.30 s | `370dff420` |
| Merged archive, profiles, IntentIR and harness checks | 566 passed, 6 historical-fixture failures | 201.24 s | `1510686be` |
| Corrected historical comparison controls | 25 passed | 14.51 s | `092a467a5` |

The six historical digests describe GPT-5.6 with CLI 0.158 and its original
retrieval declaration. The corrected fixture reproduces all six original
digests, preserves that history and separately verifies the deliberate current
GPT-6.1/CLI 0.160 changes. The
[final source review](worker-custody-capability-final-review-01.json) joins 42
unchanged recorded source files across the latter two runs; only that test file
changed. This is not a claim that one final run passed all 572 cases.

The [native review](merged-native-custody-independent-review-05.json) verifies
clean, unchanged source and signed START/owner/claim/Portal/STOP cases for a
leaf timeout, post-decode rejection, and a real Grok adapter timeout with an
authored detached child. Failed attempts released their claims only through
the existing native settlement checks, with no published edits, completion
effects or automatic retries. The negative process-custody case stays unknown
after later fixture cleanup. None of these tests call a remote model; historical
`provider_calls: 0` fields in these replay records refer to remote calls, not to
the authored local adapter invocations.

Qualification uses datasets `5171a632c6b9f0ecb2939d29d2ad74992cbfeb11` and kit
`a9b98beac1ef14278b4adf4f2289cd509530fb86`. The parent repository's newer pins
are preserved separately and are not represented as tested by these runs.

## Evidence boundaries

Docker tested the worker/helper bytes committed at
`53d0a9cde1c4126d9b2cec3666be611c7893c5c0`. The source snapshot was taken before
that commit; exact recorded hashes match its committed files. The installed
worker was read back in each case. The helper upload was bound to its source
hash, but no separate in-container helper digest was read back. A later merge
adds an independent Codex reply-mode option to the worker/router, so the earlier
Docker receipt must not be relabeled as a test of those later template bytes.

The [historical trial11 review](retained-trial11-custody-review-01.json) could
not recover its exact refused gate. The new reproductions establish cleanup
defects and qualify their repair, without retroactively identifying that gate.
The [same-group baseline](native-grok-custody-baseline-review-02.json) failed
before normal outer-process return; trial11 retained a router diagnostic after
that return. These are distinct observations.

Earlier test and setup failures remain in the published evidence. In particular,
the first Docker setup selected an incompatible prebuilt architecture and ran
zero cases; the [cleanup/correction record](offline-worker-custody-setup-correction-01.json)
preserves that failure and confirms exact-resource cleanup. Two signed test
assertions initially looked for sidecar databases beside the task rather than
in their canonical runtime storage. The tests now resolve the native storage
paths and retain all settlement assertions. IntentIR fail-open tests now expect
the current signed instruction and task-contract fields while still requiring
exact equality of the request without optional advice.

These changes do not add numerical/MJCF proof or repair operators. The previous
[live results and remaining symbolic gaps](../grok-recovery-20261006/README.md)
remain applicable. Learned index/checkpoint consumption, a syntactically valid
projection and passing runtime tests are distinct from a proof of task behavior
or measured token savings against a matched baseline.

## Broader symbolic coverage

The [source-bound operator audit](symbolic-coverage-next-operators-review-01.json)
identifies three concrete next additions. These are proposed work, with no
measured coverage gain yet:

1. Finite NDJSON copy/rename transformations with an independent input/output
   correspondence checker and support for one declared created output.
2. Package-aware Python import/alias resolution over a complete signed source
   graph, preserving the existing narrow call-binding proof contract.
3. Finite UTC/integer-tick scheduling with independently replayed solver
   witnesses for duration, deadline, overlap and capacity constraints.

The generic Doctor currently leaves task-data dependencies unresolved and
limits its generic repairs to modifying existing Python files. Format support
alone does not close those obligations. Shared logic and checker owners should
remain in `ipfs_datasets_py`, consumed through the supervisor's existing signed
admission, staged candidate and publication checks. Numerical/MJCF optimization
also needs an explicit numerical or simulator contract; XML syntax or a Lean
build alone cannot prove fitness.
