# LegalIR parallel supervisor operations

The `legal-ir-autoencoder-v1-workers.target` user target owns four independent
implementation supervisor services. Lane 0 uses `legal-ir-autoencoder-v1.service`;
lanes 1–3 use `legal-ir-autoencoder-v1-lane-N.service`. Each supervises one active
implementation at a time. Strict SHA256 task sharding assigns each task to one
lane. Each lane has separate state, process ownership, worktrees, watchdogs and
restart budgets. All lanes merge through the same fenced merge queue onto
`legal-ir-plaintext-definition-landing`.

The durable repository is
`/home/barberb/lift_coding/.worktrees/legal-ir-autoencoder-v1`.
The old `/tmp/ipfs-datasets-definition-landing` path is a compatibility symlink.
Lane state lives under `data/legal-ir-autoencoder-v1/state` and `state/lane-N`.
Worker worktree roots are siblings: `worktrees`, `worktrees-lane-1`, and so on.

The user timer `legal-ir-autoencoder-v1-monitor.timer` runs the fleet monitor
every minute. Its authoritative configuration is
`config/legal_ir_autoencoder_fleet_monitor.json`, with one explicit per-service
monitor configuration for each lane. It writes aggregate observations to
`data/legal-ir-autoencoder-v1/fleet-health.json` in the durable repository.

```sh
systemctl --user status legal-ir-autoencoder-v1-workers.target
systemctl --user status legal-ir-autoencoder-v1.service legal-ir-autoencoder-v1-lane-{1,2,3}.service
systemctl --user list-timers legal-ir-autoencoder-v1-monitor.timer
journalctl --user -u legal-ir-autoencoder-v1-monitor.service -n 10
python3 -m json.tool /home/barberb/lift_coding/.worktrees/legal-ir-autoencoder-v1/data/legal-ir-autoencoder-v1/fleet-health.json
python3 /home/barberb/lift_coding/artifacts/legal-ir-parallel-20261020/fleet_brief.py
```

The compact status command shows deferred task IDs and missing tools, and marks
a cached fleet observation stale after 180 seconds.

An alive process or polling heartbeat does not establish useful progress.
Monitors inspect actual implementation events, worker log freshness, native
state and absolute task duration. A lane with no selectable tasks is idle while
its peers finish. Cross-lane observations detect duplicate active task claims,
repeated structured failures and stale merge stage receipts. Aggregate counts
do not add together each lane's copy of the global board count.

Before restarting active work for quiet provider logs, the monitor checks the
children of every Linux thread and binds the supervisor, daemon and complete
owned subprocess tree to their process births. The tree is limited to 48
processes and eight descendant levels. An initial or uncertain sample cannot
authorize an early restart. Consecutive CPU or I/O advancement establishes
subprocess liveness; unchanged counters and an empty tree remain diagnosable.
These observations never renew the original absolute task deadline or establish
task completion. Quiet validation is reported separately from a stalled worker.

Validation timeouts retain up to 65,536 characters of partial stdout and stderr,
including the beginning and final progress when truncation is necessary. Invalid
UTF8 is replaced and shortened captures are marked. A timeout still returns 124,
keeps its failure status, and cannot create a successful validation cache entry.
The scheduler is included in the existing control-plane source identity, so its
changes use the normal reload path: active work defers reload until cleanup.

The [timeout-output qualification](../../workspace/legal-ir-watchdog-20261026-v01/timeout-output-v01/candidate-review-v01.json)
passed nine focused cases. Eleven unrelated failures from the relocated broad
run reproduced on the unchanged scheduler; that comparison remains recorded.
The [reload dependency qualification](../../workspace/legal-ir-watchdog-20261026-v01/timeout-output-v01/reload-adoption-v01/candidate-review-v01.json)
passed all thirteen cases, including active-work deferral. These are separate
from the earlier fleet-monitor qualification; no deadline or retry cap changed.

Lane diagnostics have their own error boundary. If one diagnostic read fails,
the aggregate report still includes every sibling and marks the affected lane
`monitor_error` with restart eligibility disabled. Deeply nested JSON is
reported as unreadable evidence; valid neighboring events remain available.
Unreadable durable restart history stays intact and cannot grant a fresh
restart budget. A later successful observation clears the diagnostic error.

Monitor configuration, JSON/event inputs, locks and action logs must reference
regular files directly. Named pipes, devices and symlinks are rejected without
waiting for a peer. Event reads stop at the initial bounded tail even if a
writer appends concurrently. Observation-log failures still publish a fresh
unhealthy health report. Restart-intent logging stays mandatory: failure
prevents the restart and preserves its already charged attempt and cooldown.

An empty lane is healthy only when native selection confirms that remaining
ready tasks belong to peers. Unresolved no-dispatch deferrals retain their task
identity, missing executable names, contract fingerprint and retry time in
health evidence; polling does not erase them. The monitor reports these fences
as backpressure and does not restart a service to bypass missing inputs. Native
child crashes and exhausted inner restart loops are immediately unhealthy while
the parent performs its own recovery, including a child that exited cleanly.
Fresh polling never turns an unavailable child into a completed/healthy lane.

The external watchdog gives a continuously unavailable child a 600-second
recovery grace, bound to the parent boot ID, PID and birth ticks. Failed launch
run IDs and restart counters are diagnostic and cannot renew this interval.
Observed child recovery or an actual work event clears it. The first typed
native maintenance deadline is also respected; later failed attempts cannot
keep extending that deadline. Common outer-loop publication covers all four
lanes, including nonleaders that skip shared maintenance.

Recovery consumes a dedicated native safety description, independently of the
idle-ready recycle flag. The parent reads its real event-driven checkpoint
through the existing authenticated, read-only checkpoint owner, binds its
state/event/strategy paths to the configured lane, and checks current workers,
validation, its in-memory checkout lease and the original checkout fences.
Missing, foreign, malformed or changed state cannot become a default empty
board. Database-authority deployments remain fenced without a separately
qualified canonical-readiness proof. The external monitor also verifies the
same parent birth and no physical children, then repeats those observations
immediately before any budgeted exact-service restart. A missing proof remains
unhealthy and cannot authorize a restart. Healthy idle, dependency waits and
provider/resource backoff keep their existing behavior.

Some native ready counts include tasks in retry cooldown. The monitor compares
the latest ready-task identities and canonical CIDs with the lane's exact shard
and retained deferrals. When every counted task is fenced, it reports
backpressure immediately. A partial fence cannot hide other stalled ready work.

Recovery only restarts the exact configured service. It has a durable cooldown
and an hourly restart limit; native supervisors separately fence owned-child
recycling against active workers, validation and checkout transactions. A fleet
observation never authorizes a whole-fleet restart or task completion. Dependency
gaps, quota backpressure and exhausted recovery budgets remain visible and do
not become artificial completion evidence.

All writing tasks require editable scope and actual validation commands before
provider dispatch. A malformed task receives a non-consuming, bounded cooldown
and other ready work can advance. The repaired board declares task-specific
receipt outputs, source-bound evidence validation and family regression tests.
The board, plan, validator, validator tests and retained Constitution input are
protected exact files. Only supervisor maintenance may update board statuses.
Missing scientific assets must stay unavailable; recorded metrics from a
different source cohort cannot satisfy a born-alive sample task.

The 8D and 4096D WIDTH comparisons still lack authentic compatible source
inputs. Their prerequisite remains `legal-ir-born-alive-width-replay-v1`;
the native guard defers before provider dispatch or attempt consumption when
it is absent. The protected validator rejects self-reported metrics even if an
executable with that name appears. The 4096 task forbids encoder execution and
requires saved vectors. Neither missing-input task is marked completed.

The reviewed 384D and 768D jobs use frozen local encoders and the exact selected
decoder checkpoints. They authenticate both complete parent sources and literal
subsection byte ranges, retain actual tokens below 512 without clipping, and
independently repeat numerical production and inference in sealed validation.
Their cosine and MSE measure a frozen identity projection: values of 1 and 0
are by construction and establish no learned reconstruction or legal quality.
Raw predictions and false quality/fidelity flags remain in the observations.
These local jobs do execute neural models and make no LLM/provider requests.

Optional `required validation file pins` metadata lists exact canonical files,
byte sizes and SHA256 values. With the existing contract policy enabled, native
preflight checks these for local and provider jobs before attempts or workspaces.
Missing or changed inputs receive a non-consuming cooldown and automatically
become eligible when the exact bytes return. Limits are 64 pins, 1 GiB per file
and 2 GiB total. This check grants no executable or scientific authority;
the protected checker still independently authenticates the actual replay.
When every eligible canonical task is in cooldown, native selection reports
backpressure and its parent clears the idle-ready recycle timer.

The separate `legal-ir-validation-input-recovery.timer` checks every five
minutes after its previous run finishes. Its service can restore only missing
files from the 23 literal 384D/768D input pins compiled into
`rescue/legal_ir_validation_input_recovery.py`. The verified local mirror is
`data/legal-ir-autoencoder-v1/validation-input-mirror` in the durable repository.
The manifest cannot add destinations. Restores authenticate saved bytes and
publish a complete file without replacing any existing destination. Existing
ordinary files are left untouched; native task preflight checks their hashes.
Missing parents, symlinks, changed files and unknown scientific inputs require
separate resolution. This service cannot produce the missing 8D/4096D inputs.

Each run allows at most eight attempted files and 1 GiB of attempted input
bytes, with a 90-second deadline. Durable per-path cooldowns and hourly limits
also charge failed copies. Anonymous staging prevents a killed copy from
leaving a partial input. Unresolved recovery is visible in the service result
and journal; the timer continues retrying within its limits. Recovery has no
task completion, executable installation or worker restart authority.

```sh
systemctl --user list-timers legal-ir-validation-input-recovery.timer
systemctl --user show legal-ir-validation-input-recovery.service -p Result -p ExecMainStatus
journalctl --user -u legal-ir-validation-input-recovery.service -n 5 --no-pager
```

Before intentional input removal or replacement, stop both this recovery timer
and its running service so maintenance does not race a restore. Preserve the
verified mirror and its retry history.

```sh
systemctl --user stop legal-ir-validation-input-recovery.timer legal-ir-validation-input-recovery.service
```

A shared protected-file change requires explicit, exact native approval; a
clean checkout alone is insufficient. Interrupted clearances may replay only a
retained private native receipt matching the same incident, task, attempt,
workspace, HEAD transition and approved commits. Foreign or changed receipts,
live workers and changed protected inputs remain fenced. Perform planned
protected-file updates after current attempts finish, with workers quiescent
and the native checkout mutation lease held, to avoid unnecessary incidents.

The parent watches nine exact control-plane modules, including proposal
validation, native proof collection and the shared DuckDB lock owner. Same-HEAD
edits and deletions of those dependencies trigger a bounded reload after active
work becomes quiescent.
Reload waits through active validation and consumes no task or provider attempt.

The shared DuckDB file lock releases its in-process lock even if opening the
lock file, unlocking it or closing its handle fails. A later thread can retry
after the underlying I/O problem clears; cleanup failures cannot leave the
process permanently fenced by its own abandoned lock. Cross-process locking
and the existing acquisition timeout still apply.

Validation uses the existing root-owned Python package deployment
`/opt/ipfs-accelerate-legal-validation-7ffe92439767/site-packages` through the
native sealed launcher and private HOME. It does not inherit the operator's
writable Python packages or treat skipped source/toolchain tests as completion.
Lean and Lake use the existing root-owned, read-only deployment at
`/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/lean-v4.31.0/bin`,
explicitly included in the approved validation PATH. Tasks can declare
`required validation executables` to defer before provider dispatch if an
approved tool is unavailable. The method validator authenticates both real
source pins and executes the real compiler, deriving its result from execution.
Provider routing remains Grok, with the existing typed hard-quota fallback.

Pending USC inventory tasks use the reviewed `deterministic-only` task role
with context budget zero, the native value for local execution. They run
`materialize_legal_ir_inventory.py` and the protected checker in one command,
alongside the unchanged family test command. This keeps the checker after
materialization even when independent validation nodes run concurrently.
Ordinary proposal admission, sealed validation, task custody and merge receipts
still determine completion. Each nonempty source section has a full-section
inventory span and a citing section entity; this is explicitly inventory and
does not claim sentence-level formalization.

Large USC evidence uses deterministic single-member JSON ZIP files for sections,
spans and entities, plus a separate receipt. The exact three ZIP paths have
explicit task artifact authority. Global binary/archive permissions remain off
and the existing 16,000,000-byte materialization and 24,000,000-byte serialized
proposal limits still apply. Native collection binds opaque artifacts to their
actual Git byte sizes, hashes and type prefixes. The protected checker bounds
decompression and verifies the original source identities and full plaintext.
Completed contracts retain their previous routing and output format.

The original 35 Constitution tasks cover 153 of the retained source's 236 spans.
The add-only `LIRAE-CONST-ANCILLARY` task retains the remaining 83 notes and
signature spans through the same native queue. It preserves their actual
non-operative status and all false formalization flags. To independently
authenticate every completed Constitution receipt and require each of the 236
source spans exactly once, run in the durable dataset repository:

```sh
IPFS_DATASETS_PY_MINIMAL_IMPORTS=1 PYTHONPATH=. python3 scripts/ops/legal_ir/plan_constitution_ancillary.py --check-coverage
```

The aggregate command returns failure while any source span lacks a completed,
authenticated task; files alone never establish completion. The planner and its
tests are also exact protected paths in all four lane units.

Provider containers use the qualified profile in
`config/legal_ir_autoencoder_research_toolchain.json`, supplied through
`config/legal_ir_autoencoder_provider.env`. Its exact manifest authenticates a
read-only CPU/checker snapshot with the Python dependencies and Lean tooling.
The Python entrypoints retain admitted dependencies when task commands set
`PYTHONPATH=.` and supply private temporary HOME directories when needed.
Qualification evidence is retained at
`/home/barberb/lift_coding/artifacts/legal-ir-provider-toolchain-20261020-v1/qualification.json`.
The native sealed validator continues using its root-owned deployments.

Merge completion preserves request identity and candidate ancestry when
reconciling concurrent results. Released pooled worktrees receive a non-mutating
cleanup acknowledgment. Status updates recognize lowercase and duplicate
declarations within one task section. The existing train lease and ancestry
checks automatically recover already-integrated quarantines after these
repairs; workspace quarantine and authority denials remain enforced.

To pause the fleet deliberately, stop the monitor timer and any running monitor
oneshot before stopping the worker target; otherwise monitoring may recover a
stopped worker. Preserve active work by waiting for an attempt to finish before
maintenance whenever practical.

```sh
systemctl --user stop legal-ir-autoencoder-v1-monitor.timer legal-ir-autoencoder-v1-monitor.service
systemctl --user stop legal-ir-autoencoder-v1-workers.target
```

To resume after a validated configuration update:

```sh
systemctl --user daemon-reload
systemctl --user start legal-ir-autoencoder-v1-workers.target legal-ir-autoencoder-v1-monitor.timer
systemctl --user start legal-ir-autoencoder-v1-monitor.service
systemctl --user start legal-ir-validation-input-recovery.timer
```


## Source reconciliation on 2026-11-14

The publication worktree reconciles the primary checkout, the retained native
metadata executor worktree, the source-lease recovery worktree, and the older
home checkout. Historical source trees and their immutable receipts remain in
place. Source-lease recovery shares the existing executor and health owners;
current catalogue, capacity, admission, and retention controls remain available.
The home host-capability gate uses the current stale-claim quarantine checks.
The current synchronized DuckDB owner and abandoned-wrapper cleanup supersede
the older home close/destructor implementation; the corresponding cleanup
regressions exercise that maintained owner instead of restoring a removed path.

Focused software validation recorded 176 passing protocol tests. A subsequent
nine-file reconciliation run recorded 673 passed and 81 skipped under the
existing pytest reuse policy. Collecting all 45 affected test owners exceeded a
120-second bound; that broader collection did not complete and is not a passing
result. Other unexecuted paths remain unqualified. An earlier broader run recorded 538 passed and
two failures: one outdated authorization fixture was corrected, and the other
identified incomplete dependency-release work. These results establish software
regression coverage only. They do not activate a service, renew a resource or
repair authorization, qualify an original native executor, admit legal meaning,
or establish a checked statutory corpus.

The initial publication preserved the incomplete dependency-release scope on
`preserve/incomplete-dependency-release-20261114`. Its real watchdog integration
failed with `AttributeError` / `release_failed` because `IntentRepository` had no
`release_dependency_satisfied_blocks` operation. That historical failure and
exact original source remain preserved.

The follow-up implements that operation in the maintained repository, reusing
its write transaction, task upsert, revision CAS, current-evidence gate, revision
history, completion receipts, and domain events. The pure decision owner and
board watchdog now call the real operation. An eligible manual dependency gate
requires explicit acceptance and its own current recorded evidence. Every
eligible evidence row must explicitly bind the exact `task_authority_spec_cid`
and `dependency_snapshot` (task CID, revision, status, and authority spec CID
for every dependency); declared source/artifact/version bindings are also
retained. The repository never stamps an old receipt with a new binding.
Changed source fields, unknown identity fields, acceptance, or dependency
revisions leave old evidence pending. Exact evidence IDs are filtered before
checking acceptance, so a shared digest cannot import another record's kind.
Bare notes cannot replace validation without an explicit kind/digest policy.

Eligible high-level goals can gain deterministic, blocked obligations through
the existing pure decomposition contract. Decomposition preserves the original
blocked reason and dependencies and is idempotent; it does not prove a clause.
A source-bound target is not split into weaker targets. Active blockers,
review/publication holds, explicit runtime permission/budget contracts, and
persisted claims/leases remain with their existing authority owners, including
expired leases and exhausted retry histories. No ready status, admission,
permission, hold removal, retry reset, or scientific execution is granted.
Borrowed connections require an active owner transaction and the caller owns
rollback. Unbound local calls own one transaction; child identity conflicts
roll back the entire change.

Compatibility is intentionally stricter for this new automatic path: legacy
unbound metadata receipts stay pending, and a dependency chain needs newly
recorded downstream evidence after a predecessor's revision changes. Existing
manual completion and packet-admission APIs retain their existing behavior.
The focused source-only regression owner is
`test/api/test_agent_supervisor_dependency_release_repository.py`; the original
watchdog integration now exercises the real repository. Final qualification
recorded 56 actual passes, zero skips/failures, in 92.61 seconds across these two
owners and selected existing evidence, CAS, control-receipt, event-rebuild,
manual-completion and dependency-packet regressions. The private evidence record
is `accelerate-dependency-release-adversarial-20261114-v07.xml`. Earlier checks
retain their actual outcomes: the first adversarial pass had seven test-fixture
errors from implicit ready status, corrected to explicit blocked updates; one
later invocation hit the existing test-seal database lock and ran no tests. The
owner process was allowed to finish before the successful sequential rerun;
no seal, lock, retry history, or budget was reset. These are software results,
not a native Lean, scientific training, legal-review, or live activation result.

Thirty original native/scientific probe and full historical target-suite paths
remain on the preservation branch rather than being treated as qualified
automatic completion routes. The full preserved suite also reproduces a
collection error for missing `modality_error_report_probe`; it is not a passing
result. These preserved routes require producer-specific integration and their
normal resource admission before qualification. Source-marker matches, fixed
probe constants, and old receipts do not establish scientific or legal success.
No service, live supervisor task, native Lean process, model, or paid provider
was started by this follow-up. The next smallest source-only task is to adapt
one existing producer to the exact binding contract and test that its original
and changed dependency receipts are accepted and rejected respectively, without
automatic rebinding or source-marker completion.

Runtime databases, attestation keys, test caches, generated logs, and historical
large-blob branch ancestry are excluded from Git publication. Their local files
and branch references are preserved; no cleanup or history rewrite is implied.


The independent parallel-lane planning owner from the patent/legal worktree is
also retained through the maintained planning package. Its three software tests
passed on 2026-11-14; it proposes conflict-graph slices and grants no execution or
completion authority. The older daemon variant that completed protected outputs
without validation, and resolved code add/add conflicts by larger content, is
preserved on `preserve/patent-parallel-pending-20261114` rather than overwriting
current main safeguards. The exact legacy home cleanup/configuration variant is
preserved on `preserve/home-legacy-cleanup-20261114` for the same reason.
