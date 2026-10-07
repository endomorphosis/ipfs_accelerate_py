# Durable provider cleanup: current status and remaining integration

Audit date: 2026-10-07. Starting supervisor revision:
`c733fa1240e3e31c5172ebc35e53f65da4b93891`.

## Working scope

The native provider-attempt CAS can record terminal cleanup intent and completion.
The restored runner protocol supplies the matching private bindings, socket
watchdog, exact container/process fences, resource retirement, and scoped cleanup
replay. These are prerequisites, not an enabled native launch profile.

`observe_agent_implementation_terminal_cleanup` independently verifies an existing
terminal CAS and its original signed authority. It can observe the exact disposed
workspace, but cannot create a state store, return new launch authority, settle a
task, or release a claim. The ordinary public effect APIs still require their
strict live workspace checks. The Portal diagnostic audit attaches this cleanup
observation only after its existing route/outcome checks. A companion capacity
receipt prevents successful candidate-cleanup projection.

Ordinary unsigned Grok/Codex container routes retain their separate compatibility
implementation. Their watchdog dispatch mode is excluded from implementation
worker counts. Duplicate identical `.git` mounts in standalone workspaces are
removed before command validation. Defaults remain Grok 4.7 and GPT-6.1-sol,
with route selection owned by `llm_router`.

## Disabled paths and benchmark blockers

Native lease creation and native internal watchdog/removal CLI entrypoints are
disabled before effects until the process owner implements durable STOP. There
is no environment override. Existing private protocol functions are exercised by
offline fixtures; their presence does not establish live lifecycle qualification.

1. **STOP owner:** `runtime/multi_supervisor_runner.py` does not yet join durable
   cleanup records with detached watchdogs and independently observed Docker
   absence. Process-tree absence alone cannot certify cleanup. Its current
   termination loop can signal the cleanup watchdog together with the provider.
2. **Signed launch agreement:** the current Codex builder and authority validator
   use a direct `/usr/bin/env` suffix. The native adapter expects a different
   gated launch, and its own positive command grammar is incompatible with that
   adapter. The caller also has no admitted `WorkerNetworkProfile`. Protected
   routes must not fall back to compatibility leases. Created-container recovery
   must not feed a raw prompt to a native start gate before recording a fence.
3. **Outer callback:** current `database-portal-callback-intent@1` lacks the
   immutable contract/tree/fingerprint inputs required by the retained candidate
   recovery protocol. The Bridge and daemon do not implement its complete joins.
   A terminal provider CAS is insufficient to settle this callback.
4. **Live fixture image:** the approved image
   `sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6`
   was absent locally at this audit. No substitute image was treated as approved.

## Required migration order

1. Implement exact durable binding observation in the process owner. STOP must
   retain watchdog custody until independently verified container and kernel
   absence; failed or unavailable observation keeps STOP incomplete. Cover
   runner/watchdog death, PID reuse, record replacement, delayed removal and
   interrupted persistence. Then qualify the authored disposable `/bin/sleep`
   container probe using the approved local image, with no provider calls.
2. Version the signed native launch contract explicitly. Preserve the old command
   receipt's meaning; bind the exact start gate, image, environment, mounts and
   network policy in the new version. Acquire stdin custody, record the running
   fence, then release input. Test fresh launch, adoption and all terminal races.
   Enable factory/CLI dispatch only after the owner and route agree.
3. Introduce a future-only callback intent before dispatch, binding the exact
   task/attempt/claim/lease/session/fences, callback key, Bridge, contract digest,
   repository tree and immutable fingerprint. Do not synthesize these for old
   unknown rows.
4. Connect native observed lifecycle deletion and its prepared journal to the
   Portal preservation/finalization hooks. Reuse the journal's retained-inode,
   exact-index and directory persistence checks. Recovery may resume a prepared
   operation; it may not infer deletion from missing rows. Preserve current
   Doctor/shutdown wrappers and exact-record guards.
5. Add Bridge verification of the hash-linked event prefix, immutable attempt
   binding, exact rescue commit and current cleanup CAS. Only an admitted outer
   daemon may resume mutation under heartbeat. CAS the exact original callback
   into a distinct rejected-candidate receipt; record ordinary FAILED/retry
   disposition before releasing the exact still-live claim. Expired claims stay
   with the existing expiry/successor mechanism.
6. Qualify interruption/replay at every boundary through a public supervisor
   tick. Require one provider invocation, no queue publication/completion for a
   rejected candidate, and unchanged custody for old/incomplete receipts. Only
   then rerun Terminal Bench and publish new correctness/token measurements.

## Evidence boundaries

Current offline tests use real signed current-model routes, Git workspaces,
private CAS files and terminal resource retirement. Docker absence/dispatch are
explicit doubles. They validate the cleanup protocol and observation joins, not
live Docker STOP, a successful protected provider launch, callback settlement,
or a new Terminal Bench score. Historical helper tests using retired model
profiles or missing fixture exports require migration and are not counted as
passing current qualification.

The detailed migration audit and donor provenance are retained with this change's
qualification artifacts. No live task board, accepted source pin, existing claim,
foreign service or container is rewritten by these repairs.
