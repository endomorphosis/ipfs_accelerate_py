# Independent STOP and native image audit

Read-only audit against `/home/barberb/lift_coding/.worktrees/supervisor-stop-20261007`, base `07eed982c7150406480eefdf0c57cdd2f772b671`, on 2026-10-07. No Docker command, service change, model call, hidden benchmark read, or source mutation was performed. The only write is this audit artifact. Parent/owner changes may be concurrent; final source review is recorded separately below.

## Current bounded delivery

Restore and qualify an independent read-only container-cleanup barrier for the managed owner. Keep the native lease factory and all four native internal CLI modes unconditionally disabled. Do not turn the existence of helper functions into admission for a new launch. Candidate callback settlement, claim release, signed network/start-profile migration and live container qualification remain separate work.

## Alternate process-only authority paths

`control/lifecycle_orchestrator.py` is independent of `multi_supervisor_runner._terminate_managed_process`:

- `_prove_absent` checks old process identities and an empty process snapshot only.
- `_stop_old` uses that result to record `OLD_FENCED`, `old_tree_fenced=True`, and `run_fenced`.
- `execute(RESTART)` may then create a new process; replay of a cached receipt returns without a fresh cleanup check.
- `_start_new` health-timeout compensation calls `_prove_absent` and records process termination.
- `repair_start_cleanup` signals the launched tree and reports `process_tree_absent=True` from empty process observations and dead captured identities.

A minimal safe migration is explicit refusal for profiles with retained native cleanup custody until these paths consume the shared independent barrier. Apply the guard before signalling, before process absence becomes an authority result, before cached STOP/RESTART success, before a new launch, and at final commit/repair success. Treat malformed, inaccessible or aliased cleanup directories as unknown. Presence-based refusal is not positive Docker absence evidence. Existing ordinary process-only profiles should retain their behavior, and negative tests should use actual private bound records.

`runtime/multi_supervisor_runner.py::_strict_plan_bound_process_fence_observation` is a second independent authority seam. Its existing double `/proc` scan preserves UNKNOWN on unstable observations but says nothing about Docker. Four callers consume it: slice reassignment; successor process-birth admission; terminal-missing disposition; global process-birth exhaustion. The successor branch can accept an alive tree containing only its new launch gate. A common cleanup guard/barrier must run before this early ALIVE branch as well as before DEAD, otherwise that branch can bypass cleanup. Preserve the current strict `/proc` uncertainty behavior rather than replacing it with `LinuxProcessAdapter.identity_alive`.

`stop_tracks` currently accepts `_terminate_managed_process(None)` as `(True,())` for a track missing from its process dictionary. A restarted owner may lack a Popen handle while durable cleanup records remain. That case requires exact track/profile recovery or explicit unknown/refusal; a missing dictionary entry alone is not a cleanup receipt.

`_fence_unreleased_plan_bound_process` is intentionally different: it is limited to an exact Popen gate whose authorization writer was never released and whose accepted helper cannot fork or exec before release. Preserve this before-dispatch exception and its source-bound custody; do not generalize it to a launched provider.

## Capability and lifecycle constraints

Native cleanup bindings must retain exact run/profile/target/configuration/repository/state/run-root identity, fencing epoch, boot and process births, executable/config/private-path identity, command and container/image identity, and kernel scope. The owner may observe exact existing terminal CAS and cleanup completion. It must not mint terminal authority, issue a fresh `docker rm`, infer completion from PID disappearance, or treat a missing/corrupt CAS store as absence. Unknown create/removal outcomes remain retained.

The current factory hard guard and native CLI hard guards are prerequisites while owner semantics are incomplete. They have no ambient opt-in escape. The root's lifecycle alternative-path work should use the same helper API or a conservative explicit refusal, and must not substitute daemon-local state for an independent terminal observation.

## Approved image and source recipe

The current task toolchain admission is exact:

- Image ID: `sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6`.
- OS/architecture: `linux/arm64`.
- Label `org.ipfs-accelerate.authority-validation=2026-08-03-v2`.
- Local tag used by build inputs: `ipfs-accelerate-authority-validation:20260803-v2`.

`runtime/grok_cli_runner.py::_docker_codex_task_toolchain_image_id` enforces that exact tuple. Both task-Grok and protected Codex use this task-toolchain authority. The prior parent inspection reported the image absent.

The committed `containers/external-agent/implementation-worker.Containerfile`, `bootstrap-reconciliation.Containerfile`, and their qualification scripts consume this existing local image. The minimal worker recipe uses pinned Ubuntu `sha256:ea17ec341c4211d1dd7f184a0dedf7dcb7945e92db20a5dde20544262214b84f` but still copies its tool closure from the same missing `74c4...` image. They do not reproduce the missing original image. Their output is an unsigned zero-capacity candidate; it is not an automatic replacement for production admission.

Other recorded images are not substitutes:

- `scripts/ops/agent_supervisor/pcpc_external_runtime_image_v3.manifest.json` records `sha256:ca52183d6e3f6d472b36092fc07a76fde0b7962da92b84dad2dc1038d93009ad`, tag `ipfs-accelerate-pcpc-runtime:2026-08-20-v3`, with different labels/runtime. Its own qualification note says exact rebuilding additionally requires the original Ubuntu archive snapshot or an OCI export.
- `docs/architecture/external_agent_autonomous_execution_fabric/receipts/host_admission/worker_image.json` records `sha256:3c2bec0ebd89f6427cc1e31ab650fa252d40d1f05f2c4a9f2ca52d1609e31b3b` under a separate EAAEF image/profile and rootless endpoint. It explicitly reports no live dispatch or configured board launch.

No original base-image build recipe or OCI archive was found in the reviewed committed recipes or image/build artifacts. Read-only filesystem checks of the three exact image config paths were permission-denied for rootful Docker and did not observe them under standard user-rootless storage. These checks do not establish current engine inventories. No Docker operation or permission bypass was attempted. Nothing found authorizes substituting an image, changing a pin or injecting an unvalidated sealed-image environment value.

A future live no-model qualification must first restore the original approved image from a verifiable archive/registry source, or explicitly review a new image/profile contract as separate work. The authored sleep-container probe can then exercise real create/start fence/owner STOP cleanup; offline fixtures cannot replace that result.

## Exact historical no-model probe

`scripts/run_agent_supervisor_efficiency_state_hardening.py::_r16_production_lifecycle_live_fixture` is the retained authored Docker fixture. It creates the pinned `74c4...` image with a provider-start gate and `/bin/sleep 300`; it does not invoke a model. The hidden child command is `internal-r16-live-docker-fixture`. `_run_r16_production_lifecycle_live_validation` and `_admit_r16_production_lifecycle_live_executor_contract` own its outer admission and validation. The hidden child is not an independently authorized operator command: it requires exact candidate commit/tree, parent process birth, lifecycle environment, private temporary-root custody, an admitted outer executor contract and dispatch release. The current native factory/CLI guards intentionally prevent it from running. The image and missing cold-owner/launch contracts must be resolved before adapting and admitting this qualification; bypassing the factory or invoking the hidden child directly would not qualify the current owner.

## Final source review, 2026-10-07

The root and owner changes resolve the bounded paths identified above:

- `LifecycleOrchestrator._require_process_only_cleanup` refuses current cleanup custody or observation uncertainty before cached receipt replay, old-tree signalling/absence, successor launch, failed START compensation, cleanup repair and commit. It persists `provider_cleanup_owner_required` for a nonterminal saga before refusing. Reopening the lifecycle store after record deletion does not clear that refusal. Existing historical committed receipts are not rewritten, and their replay is guarded by current custody.
- The strict plan-bound observer refuses cleanup before its early ALIVE return and before DEAD. A missing-Popen track refuses visible or unknown cleanup custody. These are stateless negative gates, not cold-owner cleanup evidence.
- Managed launch retains the exact private cleanup directory before child birth and rejects unresolved prior entries. An owned STOP retains its observer and uncertainty latch on the Popen across retries. Unknown scans, record removal or namespace replacement cannot become an empty-directory success on that same Popen.
- The observer joins exact lifecycle, boot, watchdog and detached removal-issuer births to private binding and removal-journal records. Protected births and their observed descendants are excluded from ordinary termination; PID reuse cannot inherit that exclusion. It independently reads the terminal CAS, compares complete launch/resource/container/image identity, verifies canonical producer completion and resource retirement, and observes Docker plus kernel absence with `issue_removal=False`. No owner cleanup/recovery mutation or new removal authority is added. Command-bound, unfenced records are deliberately incomplete.
- The first reviewed manager revision had a final-observation race: `complete()` followed by a final empty process snapshot could miss a binding published immediately before the last cleanup process exited. The reviewed revision now rechecks completion after that empty snapshot using the same absolute STOP deadline. The focused test owner is qualifying publication of an actual unresolved binding at that boundary.
- The native factory and four native internal CLI mode blocks remain byte-for-byte unchanged. Ordinary legacy route selection was not modified by this turn.

No additional blocking finding remains for this bounded delivery. This is **not admission for native provider dispatch** or proof of crash-safe STOP across owner loss. The new retained observer is Popen-lifetime custody. A restarted owner has no deletion-resistant historical admission to reconstruct missing records. The separate lifecycle orchestrator explicitly refuses container custody rather than consuming positive cleanup authority. Signed launch/network-profile wiring, cold-owner recovery, callback/claim-release joins, original image restoration and live no-model qualification remain prerequisites before the native guards can be removed.

This review did not run pytest, Docker or services. Test suites are run by the owner/test/root agents; their source-bound final artifacts must be used for verification claims. `git diff --check` passed during this read-only review. Source hashes below were calculated after the owner's declared final production edit and verified directly from the workspace:

| File, relative to the reviewed checkout | SHA256 |
| --- | --- |
| `ipfs_accelerate_py/agent_supervisor/runtime/durable_cleanup_observer.py` | `7a4b0329e6d5b1b4dc79649415cdf0972ac79c8fbb956845ba2e56bbfd588526` |
| `ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py` | `a3f71ba82885162f2fbd67124d072fc4ab370e1d0bb4806506a5b90cdcf8aa61` |
| `ipfs_accelerate_py/agent_supervisor/control/lifecycle_orchestrator.py` | `ba775d1ece31e6af9635128bec1fd771469ac8277cc5e10ad105e7fbf9d41767` |
| `ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py` (unchanged native guards) | `da85f716215df06edd5fd3664e510f1e10dae3534b3da2174663a4c69e7397f8` |

Final focused evidence from the test owner: `final-snapshot-06.xml` / `final-snapshot-06.log` reports 44 passing cases in 34.84 seconds (36 actual private-record/CAS observer cases and 8 manager-algorithm cases using authored doubles). The reviewer read the XML and confirmed zero errors/failures/skips and the passing `test_final_empty_process_snapshot_cannot_hide_last_cleanup_publication` case. That regression publishes an actual unresolved binding on the final empty process snapshot and requires STOP to remain incomplete under the same deadline. These are offline protocol checks; Docker/process observations are authored doubles, and no live container result is claimed. Root's broader combined qualification remains separate.

## Concurrent datasets pin advancement

Parent `origin/main` advanced to `01cac3bff5`, carrying datasets `75f57f9eeff052a5718fd1f2c6a4b73de3784af8`. This reviewer compared that immutable commit with the previously tested datasets commit `271ad796288e17039f036af6a0d003a4628da6ed` without changing a checkout or pin.

The delta is 27 files: 26 additions and one documentation modification. Its only installed-package addition is the 409-line opt-in `ipfs_datasets_py/logic/formalization/autoencoder/legal_scope_support_action.py`. Other additions contain documentation/evidence, standalone checkpoint-release/reconciliation drivers and their tests. Existing runtime Python files, package initializers, packaging, pytest configuration, routers, resource schedulers and storage code are unchanged. The autoencoder package initializer remains a docstring with no module discovery or implicit imports. The added support/action module is not referenced from another package module at the new commit. No direct datasets import occurs in the reviewed STOP observer, manager, lifecycle, provider-attempt store or native runner.

No changed runtime dependency in this delta warrants repeating the complete STOP/lifecycle suite solely because of the datasets advance. Preserve the concurrent pin. Existing result artifacts still identify their original datasets commit and must not be relabeled as a new exact-source run. If an exact new-parent/new-datasets qualification is desired, a fresh focused run provides that new provenance; this is a source-binding qualification rather than evidence of a changed STOP implementation. The reviewer performed no test execution or source/pin mutation for this comparison.
