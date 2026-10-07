# Candidate rejection recovery migration audit

Read-only audit on 2026-10-07. Current source is the isolated `supervisor-gaps-20261007b` worktree, baseline `c733fa1240e3e31c5172ebc35e53f65da4b93891`; concurrent restoration edits were not treated as qualified. Donor initial closure is `7311b3f6d`; later journal integration is present in `fleet-supervisor-maintenance-20260910` at `f7194e65775f000c8d8924e700a54b4067d83298`. This report performs no Docker operation, callback update, claim release, lifecycle deletion, or source change.

## Immediate safe boundary

Restore the runner's durable provider cleanup producer and independent read-only terminal CAS observer. The Portal protected-provider audit may require that evidence before reporting cleanup complete. That observation is not outer callback settlement, candidate acceptance, retry admission, or claim-release authority.

Keep current `DatabaseImplementationDaemon.run_provider` and `_run_provider_impl` semantics intact for this restoration. `run_provider` is now a shutdown-aware wrapper with the native Doctor callback lock; Source384 retry, native provider failure, and unapplied-router-proposal handling live below it. Replacing this method with the older donor discards these contracts. Current `DatabasePortalExecutionBridge.run_provider` bounds retries by the maximum of database and local Portal attempts. Candidate retry exceptions currently leave the outer `started_outcome_unknown` callback in custody. Preserve that behavior until the complete new receipt chain is implemented.

The current source contains candidate closure/journal helper modules and historical tests/docs, but the daemon/Bridge/native-store joins are absent. Their presence is not implementation evidence.

## Native lifecycle and Portal producer seams

The following donor interfaces form a relatively narrow restoration seam, but must retain newer current checks:

| Owner | Required interface/change | Constraint |
|---|---|---|
| `WorktreeLifecycleStore` | `compare_and_delete_observed(expected)` | Exact present record/index, held regular inodes, successful unlink and directory persistence; absence alone is not observation. |
| `WorktreeLifecycleStore` | `delete_candidate_observed`, `resume_candidate_observed_delete`, `observe_candidate_deletion` | Thin calls to existing `worktree_lifecycle_delete_journal.operate` with `delete`, `resume`, `observe`; resume cannot prepare a new operation, observer cannot create or fsync state. |
| `PortalImplementationDaemon._finalize_exact_worktree_lifecycle` | Optional terminal/released callback pair; journal-aware deletion | Preserve current `expected_record` checks for both terminal CAS and ordinary delete. Do not copy the donor's weaker ordinary branches. Existing terminal records cannot reconstruct a missing prior callback. |
| `PortalImplementationDaemon._cleanup_merged_worktree` | Pass optional callbacks; capture actual nonpooled cleanup disposition before journal terminal callback | Current method is already a guarded mutation boundary. Donor `_cleanup_merged_worktree_guarded` is not a reason to replace the current wrapper/guard. Require actual worktree and branch removal, no pooled release. |
| `PortalImplementationDaemon._preserve_interrupted_worktree` | Instantiate `CandidateLifecycleHandoff` only for exact cleanup proof and retained rescue ref/commit | Preservation precedes cleanup; emit no release event before actual deletion observation. |
| `_preserve_failed_validation_worktree` and callers | Carry `candidate_cleanup_required` and verified `candidate_cleanup_evidence` | An expected protected cleanup proof that is unavailable raises the distinct uncertainty exception. |
| `_run_once`, `_run_implementation`, ephemeral implementation finalizers | Retain custody on `CandidateClosureObservationUnknown` | Generic exception/finally cleanup must not release implementation/resource/task claims, discard selected dispatch intent, or mark unfinished work finished. |

The native journal module already supplies the difficult no-replace/retained-inode persistence mechanics. Its helpers must remain under exact task-index then workspace guards and same-filesystem constraints. No automatic migration of old absent lifecycle rows is sound.

## Outer callback migration must be future-only

Current predispatch `database-portal-callback-intent@1` records `started_outcome_unknown` and a subset of attempt identity. It does not contain the older donor's sealed unknown-callback fingerprint, task-contract digest, repository-tree identity, original owner session, and explicit callback-key binding as a complete receipt. Existing unknown rows cannot be retroactively upgraded from cleanup evidence.

Introduce a separately reviewed future callback-intent version (name/schema to be chosen during implementation) before provider entry. It should bind exact original attempt/task/claim/lease/session/fences, canonical idempotency key, bound Bridge identity and immutable attempt directory/projection, task contract digest, repository tree, and an exact receipt fingerprint. The row must be committed before dispatch and become immutable input to the later compare-and-swap. Historical rows retain their current unknown behavior.

Then add a distinct candidate-rejected callback receipt that nests that exact original intent and independently verified closure. Publication performs CAS against the exact original serialized value plus current row/attempt/key/owner bindings. Receipt replay must revalidate native provider CAS, immutable Portal evidence, rescue ref, and journal; it must not invoke the provider again. Failed or unavailable classification for a future journal handoff retains custody.

Donor methods showing the algorithm, not drop-in replacements:

- `_bound_candidate_rejection_closure`, `_recover_bound_candidate_rejection_closure`.
- `_candidate_callback_receipt`, `_commit_candidate_callback_closure`.
- `_release_closed_candidate_claim`, `_reconcile_closed_candidate_claim`.

Current source lacks donor `_uses_quack_command_gateway`, `_require_execution_authority`, `_require_typed_attempt_admission`, `_release_exact_attempt_lease`, and ordinary terminal retry reconciliation. Do not create permissive stubs for these names. Adapt to current `_require_provider_admission`, `_protect_attempt_write`, live typed owner/claim attestation and exact coordinator primitives, while keeping the new path unavailable for any authority mode not independently qualified.

Claim release must occur only after the normal FAILED disposition and exact canonical `retrying` receipt. It must verify the same accepted unexpired claim/fence and absence of prepared completion. An expired claim remains with the existing expiry/successor mechanism. Merely obtaining cleanup evidence or a candidate-rejection code never releases it.

## Bridge dependencies and schema adaptation

The donor closure verifier needs these absent methods:

- `_verified_event_chain`: bounded sequence/stream/snapshot/hash/predecessor verification. Use current stable descriptor/duplicate-key defenses rather than restoring donor unguarded path reads.
- `_recovery_attempt_binding`: compare immutable attempt/projection against the current exact task and admitted claim, allowing only explicitly permitted status-revision advancement.
- `_portal_completion_event_identity`: independently derive the projection-local Portal CID/key and join it to the database binding.
- `_preserved_commit_exists`: exact rescue ref plus commit observation.
- `_candidate_rejection_closure_receipt`, `verify_candidate_rejection_closure`, `recover_candidate_rejection_closure`, and `DatabasePortalCandidateRejectedClosed`.

The current `_binding`/projection format must be mapped explicitly; donor stable-field exclusions cannot simply be copied. Existing `candidate_journal_recovery._context` expects these Bridge methods plus exact binding fields. Its `observe` is read-only; `resume` accepts a guarded callback and rechecks before and after the actual existing prepared operation. The outer admitted daemon must call mutating recovery under heartbeat with current claim protection. A verifier must never silently resume a journal.

## Focused verification population

1. Keep existing ordinary native Doctor, Source384 deferral, router proposal refusal, callback unknown-outcome, native failure settlement and shutdown tests as regression coverage.
2. Run durable producer and observer tests with fresh compatible fixtures: `test_protected_normal_cleanup_producers.py`, `test_protected_watchdog_cleanup_handoff.py`, exact terminal observer cases from `test_candidate_rejection_closure.py`. Historical fixtures have old Grok/Codex model IDs and removed wrapper exports; qualify adapted fixtures rather than relaxing current routing policy.
3. Native store tests: `test_candidate_lifecycle_delete_journal.py`, `test_lifecycle_terminal_callback_replay.py`, exact-owned worktree lifecycle tests. Exercise crash after prepared publication, each move, committed publication; fsync failure, replaced inode, collision, noncreating observation, already-terminal callback refusal, and ordinary delete compatibility.
4. Once future callback contracts exist: adapt `test_candidate_closure_caller_custody.py`, `test_candidate_rejection_closure.py`, `test_candidate_journal_recovery.py` to current typed admission. Require one provider invocation across interruption after finish/callback CAS/FAILED/retry CAS/release, fresh-owner replay, changed binding/contract/rescue ref, expired claim, and actual public supervisor tick. Explicitly test old unknown rows remain unknown.
5. Candidate suites use real signed CAS/Git/native lifecycle/typed gateway fixtures but double Docker absence, accepted source capsules and TCP Quack. They cannot qualify live Docker by themselves.

## Existing authored no-model Docker probe

The best matching donor probe is:

`fleet-supervisor-maintenance-20260910/test/api/test_agent_supervisor_grok_quota_terra_gate.py::test_strict_fence_waits_for_detached_container_cleanup` (starts at donor line 9298).

It creates one disposable Docker container running `/bin/sleep 300`, captures its real running termination fence, verifies durable cleanup bindings, asks the actual managed-process owner to terminate it, and checks zero owned process members, retired lease root, exact container absence, and detached-cleanup absence verification. It makes no model request and needs no provider credential. `finally` only targets its own captured process group and unique container name.

Prerequisites: Linux `/proc` process identity and signals; trusted `/usr/bin/docker`; access to `unix:///var/run/docker.sock`; the existing locally pinned image (never pull); restored runtime `_DockerContainerLease`/watchdog and `multi_supervisor_runner` durable binding/termination APIs; a new isolated private state/temporary root. The fixed image currently declared by `agent_implementation_route.py` is:

`sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6`, label `2026-08-03-v2`.

The current checkout does not contain this exact test. Its legacy test module imports the thin public `agent_supervisor.grok_cli_runner` and other historical fixtures, so collecting the whole donor module against current code is not reliable. Restore only this authored function and its `_skip_or_fail_live_cleanup_validation` helper into a new qualification module, explicitly importing the current `runtime.grok_cli_runner`. Preserve the original positive command grammar, running fence and owner termination assertions. Do not replace missing production APIs with test stubs. A command for that new module, once added and reviewed, is:

```sh
IPFS_ACCELERATE_AGENT_REQUIRE_LIVE_DOCKER_CLEANUP_VALIDATION=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
/home/barberb/.local/bin/python -B -m pytest -q -o addopts= \
  -p no:cacheprovider -p ipfs_accelerate_py.testing.pytest_ast_seal \
  test/integration/test_native_durable_docker_cleanup.py
```

Use an explicit current supervisor/datasets `PYTHONPATH`, fresh seal database and fresh artifact/basetemp paths as for other qualifications. The required-live flag turns a missing Docker/image prerequisite into failure rather than an apparent qualified skip. This command was not run and that proposed new test file was not created by this audit.

The current `test_real_disposable_codex_container_and_board_toolchain_probe` (current legacy test file line 2621) only performs `codex --version` and an in-container authored pytest suite, with exact container removal. It is a useful smaller smoke after adapting its stale helper imports/model shape, but it does not establish durable watchdog cleanup or terminal provider CAS progress. The historical R16 CLI wrapper is tied to an old accepted source/authorization chain and should not be reused as current-root authority.

## Follow-up producer review: lifecycle probe is currently blocked

A subsequent exact source check found that current `runtime/multi_supervisor_runner.py` lacks `_durable_docker_cleanup_bindings`, `_detached_docker_cleanup_bindings`, and `_detached_docker_cleanup_absence_verified`. Its `_terminate_managed_process` returns success when the managed process tree is absent, without independent Docker cleanup observation. Therefore the authored strict-fence probe above is not runnable merely by extracting the test and changing its import. Real owner-side durable binding discovery, exact detached-watchdog custody, and container/kernel absence joins are prerequisites. Do not replace these production joins with test stubs. The native watchdog carries lifecycle markers and can otherwise be included in a generic force-fenced tree before its cleanup finishes.

The probe's exact positive command uses `--entrypoint=/usr/bin/env -i <fixed environment> /bin/sh -c <native start script> aseh-provider-start /bin/sleep 300`. It directly exercises `lease.create_inert_container`, then anonymous start custody and running-fence capture; it does not call the protected network/route adapter and cannot qualify that adapter.

Read-only AST review of the producer restoration found no changes to the preexisting model defaults, Grok/Codex command builders, quota classification, or environment construction. Twelve existing top-level definitions changed; the explicit legacy lease, watchdog, removal and created-Grok helpers reproduce the original ASTs after renaming. The producer manifest records 82 restored donor definitions. The large addition is the dependency closure for native cleanup/create/removal custody, not evidence that end-to-end launch integration is complete.

Review findings sent to the producer/root for correction or explicit retained limitation:

- Scoped primary Grok could enter the preserved legacy lease after successful preflight. Add explicit denial before legacy construction until the scoped native launch profile is qualified; keep unsigned ordinary behavior.
- Protected Codex caller does not supply a network profile. Separately, the current adapter demands a `/bin/sh` entrypoint while the restored Codex positive grammar requires `/usr/bin/env` followed by the shell gate. A native protected launch remains unavailable. Supplying a profile alone does not fix the grammar mismatch.
- Old `_start_recorded_codex_effect` wrote raw prompt bytes to a pipe without the restored running-fence/start-marker protocol. It must not consume a native gated receipt; created-effect recovery requires an exact adapter or explicit denial. A prompt starting with the public marker could otherwise release a gated provider before durable fence capture.
- One concurrent-terminal branch in `run_authorized_preflight_fallback` returned a terminal result without invoking the exact store/reservation cleanup replay. It needs the same scoped join as the other terminal replay paths.
- Rootless endpoint acceptance was inconsistent with fixed local-endpoint create/attestation/removal schemas. Deny that unsupported endpoint until it is bound end-to-end.

These observations were made while the producer was applying narrow follow-up fixes. They are not a final frozen-source approval or a claim that the fixes passed. No Docker or model call was made by this audit.

## Bounded publication recommendation

Publish the independently qualified read-only terminal observer and daemon audit first. Retain the full producer restoration and its donor manifests as a separate patch/artifact until owner STOP joins and a live no-model protocol test can be qualified. Protected route denial is necessary but does not disable the directly callable native lease factory or capability-bearing internal CLI modes. If source is retained as a prerequisite, the native factory and CLI effect entrypoints should remain explicitly disabled until an exact reviewed owner contract exists; an ambient environment opt-in is not such a contract. Do not describe the current detached watchdog as lifecycle-safe or a synchronous STOP barrier.

The parent independently found the pinned image absent, so there is no live Docker protocol qualification in this turn. No substitute image or model invocation was attempted by this audit.

A concrete ordinary compatibility issue also requires correction: renaming the preserved ordinary watchdog mode to `--internal-legacy-docker-cleanup-watchdog` changes the process classifier. Current `todo_daemon/supervisor.py::_is_internal_docker_cleanup_watchdog_argv` originally recognized only the old mode, so an ordinary reaper could be mistaken for an implementation worker. The parent owns the exact-mode classifier fix and regression coverage.

The producer subsequently froze guards rejecting created-effect restart, rootless endpoints, and scoped-primary legacy dispatch, and added exact cleanup replay for concurrent terminal observation. During freeze review the new primary guard was found to read `route_plan` on ordinary no-fallback requests, where the variable had not been initialized. The producer was notified to fix this ordinary-path regression and requalify before final publication. This audit does not certify a frozen result before that correction.

## Final gated-source review

The parent chose to retain the protocol prerequisite with unconditional native gates. Final read-only reachability review confirms `_DockerContainerLease.create` raises before effects; all four native internal CLI modes exit 125 before helper invocation; a repository-wide call-site scan found no alternate production dispatch around these gates. There is no environment escape. The preserved ordinary legacy watchdog mode remains reachable and the parent updated its exact classifier. `route_plan` is now initialized before the ordinary branch. The scoped-primary guard, created-effect recovery denial, fixed-engine restriction and exact concurrent-terminal replay remain in place.

This bounded disabled/offline protocol deliverable is acceptable subject to final focused tests passing. It is not approval of native launch, managed STOP Docker cleanup, candidate callback settlement, claim release or benchmark completion. The factory/internal CLI block must remain until the actual owner barrier and live no-model qualification are implemented. Existing immutable completed cleanup may be observed/replayed only through the restored exact terminal CAS joins; unknown ownership stays retained.

Reviewed runner SHA-256: `c28d1e05179adf186ba26701b9a5ff4a61f5270e3aa212ed4583a7cd2ad6ee5b`. No tests, Docker operations or source mutations were performed by this reviewer; producer and parent own the reported validation runs.
