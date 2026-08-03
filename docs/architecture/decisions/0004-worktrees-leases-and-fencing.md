# ADR-0004: Isolate concurrent implementation with worktrees, leases, and fencing

- **Status:** Accepted
- **Date:** 2026-08-03
- **Last verified:** 2026-08-03
- **Deciders:** Agent supervisor maintainers; documentation-refresh runtime-decisions track (DOC-018)
- **Scope:** How concurrent implementation lanes isolate filesystem checkouts, claim exclusive ownership of a task or lane, and reject stale or duplicate terminal effects. Covers the composed lease/fence path, worktree pool lifecycle, fenced worktree ownership records, heartbeats, and checkout serialization at merge boundaries.
- **Non-goals:** Mutable coordination store authority versus immutable replicas (ADR-0005); merge-versus-authoritative-completion gate recomputation beyond isolation (see `EXECUTION_AND_RECOVERY.md`); transport-neutral control authorization; provider selection; objective or taskboard projection design (ADR-0001).
- **Supersedes:** none
- **Superseded-by:** none
- **Related guides:**
  - [`docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md`](../agent_supervisor/EXECUTION_AND_RECOVERY.md) — lane lifecycle, lease/fence/worktree invariants, recovery taxonomy
  - [`docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md`](../AGENT_SUPERVISOR_ARCHITECTURE.md) — “Leases, fencing, and worktrees are a distributed-systems protocol”
  - [`docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md`](../AGENT_SUPERVISOR_PHILOSOPHY.md) — isolation as a safety boundary
- **Source anchors:**
  - `ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py` — `LeaseCoordinator`, `LeaseGrant`, `StaleFencingTokenError`, `LeaseExpiredError`, heartbeats
  - `ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py` — lane-scoped lease adaptation
  - `ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py` — serialized Git checkout mutations
  - `ipfs_accelerate_py/agent_supervisor/todo_daemon/worktrees.py` — `WorktreePool`, `WorktreeLease`
  - `ipfs_accelerate_py/agent_supervisor/worktree_lifecycle.py` — fenced cross-lane worktree ownership, `ProcessBirthIdentity`, stale reclamation
  - `ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py` — `ResourceAdmissionLease` (capacity, not task authority)
  - `test/api/test_agent_supervisor_lease_coordination.py`
  - `test/api/test_agent_supervisor_worktree_lifecycle.py`
  - `test/api/test_agent_supervisor_process_tree_fencing.py`

## Context

The agent supervisor runs multiple implementation daemons, lanes, and provider
processes against one repository and shared task population. Processes pause,
crash, restart, or resume after another owner has taken over. On Linux, process
IDs are reused. Git working trees are mutable and can be left dirty, reused, or
partially cleaned by a peer lane.

Without a distributed-systems safety boundary, concurrent work produces:

- **Stale effects** — a delayed worker publishes merge, receipt, or cleanup
  actions after its claim was superseded.
- **Duplicate effects** — two live owners edit the same scope, land two
  implementations, or both attempt cleanup of one worktree.
- **Cross-lane races** — a peer classifies a freshly created worktree as
  already-merged or disposable before the owner is discoverable (the six-lane
  race closed by fenced worktree lifecycle).
- **False health** — operators or watchdogs treat a live PID, a dirty checkout,
  or a heartbeat alone as proof of healthy exclusive ownership.

Resource capacity (`ResourceAdmissionLease`) and filesystem checkouts answer
different questions from task ownership. Capacity may be free while no fence
authorizes mutation; a worktree path may exist while its lifecycle fence is
stale. The design must keep isolation, claim ownership, and terminal
publication authority distinct and jointly enforced.

## Decision

Concurrent implementation uses **isolated worktrees plus leases, fencing
tokens, and heartbeats** as a single distributed-systems safety boundary.
Filesystem isolation alone is not authority; liveness alone is not ownership.

### Normative protocol

1. **Worktree isolation.** Each claimed task (or managed lane attempt) mutates
   repository state only inside an exclusive checkout from `WorktreePool` /
   `WorktreeLease` (or an equivalently fenced lifecycle workspace). Concurrent
   lanes must not share a dirty main working tree for implementation edits.
2. **Single live owner.** At most one accepted live claim exists per task (or
   lane/workspace scope) for the current fencing epoch. Claims are persisted by
   `LeaseCoordinator` (DuckDB coordination store) on the composed path.
3. **Monotonic fencing.** Every accepted claim carries a `logical_epoch` /
   fencing generation and a fencing token. Terminal operations (receipts, merge
   boundary publishes, lifecycle CAS transitions, cleanup compare-and-delete)
   require the **current** token. Superseded tokens fail closed with
   `StaleFencingTokenError` / `FenceMismatchError` rather than silently
   committing.
4. **Heartbeat window.** Owners refresh heartbeats before lease expiry. Expiry
   reclaims capacity and advances the epoch so a recovered owner receives a
   **higher** fence; expired workers cannot publish under the old token.
5. **Process identity is multi-factor.** Lifecycle liveness uses
   `ProcessBirthIdentity` (PID + start-time ticks + boot id, and parent when
   available). A matching PID after reuse or reboot is treated as a dead or
   unknown owner—never as continued exclusive ownership.
6. **Worktree is isolation, not authority.** A dirty, reused, or path-present
   worktree proves nothing without a current lease, fence, and bound receipts.
   Cleanup and reuse require terminal or provably stale fenced records; peer
   cleaners must not delete nonterminal claims merely because a branch tip
   matches the merge target.
7. **Checkout serialization.** Concurrent Git checkout mutations are locked
   (`checkout_lock`) so merge and worktree operations do not corrupt shared
   repository metadata even when leases are correctly fenced.
8. **Retries re-claim.** A retry attempt re-obtains a current lease and fence
   (or worktree lifecycle claim on the plain daemon path) and re-validates on
   the current worktree tip. Prior provider exit codes and prior fence tokens
   are not cached success.
9. **Capacity ≠ mutation authority.** `ResourceAdmissionLease` authorizes host
   or provider capacity use only. Holding resource capacity without a valid
   task/lane fence must not authorize Git or board mutation.

### Ownership boundary

| Concern | Authoritative package / symbol |
| --- | --- |
| Task claim, fence, heartbeat, expiry | `merge/lease_coordination.LeaseCoordinator` |
| Lane adaptation of the same protocol | `merge/leased_lane` |
| Filesystem exclusive checkout | `todo_daemon/worktrees.WorktreePool` / `WorktreeLease` |
| Cross-lane worktree publish/cleanup fence | `worktree_lifecycle` |
| Host/provider capacity reservation | `runtime/resource_scheduler.ResourceAdmissionLease` |
| Git mutation serialization | `merge/checkout_lock` |

## Alternatives

### Alternative A: Direct shared-checkout concurrency

- **Summary:** Multiple workers (or one shared daemon) edit the same repository
  working tree in place. File locks, branch names, or informal “do not touch
  paths claimed by peers” conventions separate scopes. No per-task git worktree.
- **Expected benefits:** Lower disk use; simpler mental model; fewer
  `git worktree` operations and less pool management.
- **Why not chosen:** A shared checkout does **not** prevent stale or
  duplicate effects.
  - Two processes can stage, commit, or rewrite the same index and working
    tree; last writer wins with unmerged cross-task dirt.
  - A delayed worker that resumes after crash or preemption still holds open
    file handles and can commit under an old attempt while a peer has already
    claimed the same paths—**duplicate** patches and **stale** commits relative
    to the current claim epoch.
  - Dirty residual files from a failed attempt poison the next task’s validation
    unless every exit path is perfect; isolation fails open on partial cleanup.
  - Merge serialization alone (`checkout_lock`) can order Git metadata
    operations but cannot give each task a private tip for validation, so
    validation green on a shared dirty tree is not attributable to one claim.
  - Protected-path policy and conflict graphs reduce *planned* overlap but do
    not stop a live process that still mutates files after its logical claim
    ended.

Shared checkout therefore fails both the **stale-worker** and **duplicate-owner**
failure modes that multi-lane execution must close.

### Alternative B: PID-only ownership (process liveness as the lock)

- **Summary:** Record the owner process ID (and perhaps a boolean `locked`
  flag or a file lock). If `kill -0` / `/proc/<pid>` shows the process alive,
  treat the claim as current; if dead, reclaim. No fencing tokens, no lease
  epochs, no multi-factor process birth identity.
- **Expected benefits:** Minimal coordination state; easy operator inspection
  (“is PID 12345 still running?”); no DuckDB fencing tables.
- **Why not chosen:** PID-only ownership does **not** prevent stale or
  duplicate effects.
  - **PID reuse:** After exit, the kernel can assign the same PID to an
    unrelated process. A reclaim check that only compares PID numbers falsely
    believes the original owner is still alive—or, conversely, kills the wrong
    process. The lifecycle layer therefore binds `ProcessBirthIdentity`
    (start-time ticks + boot id), not PID alone.
  - **Stale resume after takeover:** Process A pauses (debugger, SIGSTOP, long
    GC, network stall) while still “alive.” Process B correctly takes over
    after timeout or operator action. A still-live A then publishes merge or
    cleanup under the old claim—a **stale** effect—because liveness never
    invalidated A’s authority. Fencing tokens make that publish fail closed
    even when A’s PID remains alive.
  - **Duplicate owners under races:** Two claimers that both observe “no live
    PID” (or a heartbeat file lag) both start work. Without a single-writer
    lease registry and monotonic fence, both can land **duplicate** effects.
  - **Wrong health signal:** A live PID proves a process exists, not that it
    holds the current fence, the correct worktree, or healthy progress
    (`EXECUTION_AND_RECOVERY.md` non-authoritative signals table). Operators
    must not treat PID alone as exclusive ownership.

PID liveness remains a **probe** for watchdogs and cleanup eligibility when
combined with birth identity and fence state. It is never sufficient
authorization for terminal mutation.

### Alternative C: Boolean lock flag or lease without fencing epochs

- **Summary:** Persist `locked=true` with an owner string, or a lease expiry
  without a monotonic fencing generation. On expiry, the next worker sets
  itself as owner without invalidating in-flight work of the previous owner.
- **Expected benefits:** Simpler schema than epoch + token pairs; fewer error
  types.
- **Why not chosen:** Classic split-brain under delayed workers. Expiry
  reclaims capacity, but without a fencing token the previous owner can still
  complete a terminal write. The observed protocol uses expiry **and** fence
  advancement so stale workers are harmless (`test_expired_lane_is_recovered_with_higher_epoch_and_token`).

### Alternative D: Worktrees without lease/fence (isolation-only)

- **Summary:** Always use git worktrees for isolation, but treat worktree
  presence or branch tip ancestry as ownership (e.g. “if branch is merged,
  delete worktree”).
- **Expected benefits:** Stronger filesystem isolation than Alternative A
  without coordination store complexity.
- **Why not chosen:** Worktrees answer “where may I write?” not “may I still
  write?” Peer cleanup that keys on branch-merged status races with freshly
  prepared worktrees whose tips still match the merge target (ASI-171).
  Isolation without fencing reintroduces stale cleanup and duplicate attempt
  ownership under concurrent lanes.

## Consequences

### Positive

- Stale workers with superseded fences cannot publish terminal effects; delayed
  resume after takeover fails closed instead of double-landing.
- At most one accepted live claim per task/lane epoch reduces duplicate
  implementation and duplicate cleanup.
- Isolated worktrees keep concurrent dirty state out of the integration
  checkout and make validation attributable to one claim tip.
- Heartbeat + expiry reclaim capacity after crashes; fence advancement makes
  reclaim safe.
- PID reuse and missing `/proc` fail closed via birth identity rather than
  false ownership.
- Operators gain a clear taxonomy: stale isolation (`StaleFencingTokenError`,
  expired lease) is distinct from dependency idle, resource block, and
  validation failure.

### Negative

- Disk and setup cost for worktree pools; need reclaim and reuse policy.
- Coordination depends on DuckDB (or equivalent) for the composed lease path;
  optional dependency failure must fail closed rather than degrade to
  unfenced mutation.
- More failure modes for agents to handle (`StaleFencingTokenError`,
  `LeaseExpiredError`, `FenceMismatchError`, ownership denials).
- Lifecycle and lease layers both use fence vocabulary; implementers must not
  confuse resource leases, task leases, and worktree lifecycle fences.
- Startup grace and heartbeat intervals add tuning surface; too-short TTLs
  thrash reclaim, too-long TTLs delay recovery.

### Neutral / residual risks

- Plain `runtime.multi_supervisor_runner` paths may not construct
  `LeaseCoordinator`; those runners still require their own task/worktree
  lifecycle claims and must not be described as implicitly fenced by the
  bundle supervisor.
- Correct fencing does not replace protected-path policy, validation, or
  authoritative completion gates.
- Clock skew and wall-clock expiry remain residual; fencing and CAS reduce
  harm when expiry races occur.
- Foreign or operator-created worktrees outside the managed lifecycle remain
  out of automatic cleanup scope by design.

## Evidence

| Claim in Decision | Evidence (path, test, or operational check) | Notes |
| --- | --- | --- |
| Single live owner + fencing at claim/heartbeat/receipt | `merge/lease_coordination.py` (`LeaseGrant`, `StaleFencingTokenError`); `test_claim_renew_heartbeat_release_and_receipt_are_fenced` | Terminal ops re-validate token |
| Expired lane advances epoch/token | `test_expired_lane_is_recovered_with_higher_epoch_and_token` | Higher fence after expiry |
| Worktree pool exclusive lease | `todo_daemon/worktrees.py` — `WorktreeLease` / `WorktreePool` | Dirty checkouts not silently reused as clean |
| Fenced worktree lifecycle before visibility | `worktree_lifecycle.py`; `test_begin_preparing_publishes_before_worktree_visibility` | Closes six-lane premature-cleanup race |
| Stale fence rejected on CAS | `FenceMismatchError`; `test_owner_transitions_and_only_owner_may_advance`, `test_compare_and_delete_requires_matching_fence` | Peer cannot advance or delete on stale view |
| Stale reclamation requires expiry | `test_stale_reclamation_requires_expiry_and_advances_fence` | Fence advances on reclaim |
| PID reuse not treated as live owner | `ProcessBirthIdentity`; `test_pid_reuse_treated_as_dead_owner` | Multi-factor identity |
| Missing `/proc` fails closed | `test_missing_proc_fails_closed` | Unknown ≠ alive |
| Duplicate attempt rejected while owner alive | `test_duplicate_attempt_rejected_while_owner_alive` | Single nonterminal claim |
| Process tree fencing for descendants | `test_agent_supervisor_process_tree_fencing.py` | Child sessions included |
| Resource lease ≠ task fence | `runtime/resource_scheduler.py`; `EXECUTION_AND_RECOVERY.md` §5.3 | Capacity only |
| Product narrative of the protocol | `EXECUTION_AND_RECOVERY.md` §5; architecture “distributed-systems protocol” section | Operator-facing invariants |

## Verification

From the repository root:

```text
# Focused isolation tests
python -m pytest \
  test/api/test_agent_supervisor_lease_coordination.py \
  test/api/test_agent_supervisor_worktree_lifecycle.py \
  test/api/test_agent_supervisor_process_tree_fencing.py -q

# Protocol symbols still present
rg -n 'StaleFencingTokenError|LeaseCoordinator|ProcessBirthIdentity|WorktreePool' \
  ipfs_accelerate_py/agent_supervisor

# Guide still documents non-authoritative PID and fence monoticity
rg -n 'stale fencing|PID alone|Worktree is isolation' \
  docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
```

Pass signals: lease/fence tests green; superseded tokens still raise; worktree
lifecycle still requires matching fence for cleanup; documentation continues to
treat PID and dirty worktrees as non-authoritative.

Fail signals (ADR stale): terminal publish without fence check; reclaim of
non-expired live owners by PID alone; shared main-checkout implementation path
without worktree or equivalent isolation; removal of fencing epochs in favor of
boolean locks.

## Review triggers

- [ ] Source anchors no longer match the Decision statement
- [ ] A recorded negative consequence becomes unacceptable
- [ ] A rejected alternative becomes viable without those costs (e.g. kernel
      or VFS primitives that provide fence-equivalent exclusive mutation without
      worktrees)
- [ ] Security, isolation, lease/fence, or trust-tier changes touch this scope
- [ ] Related guide or package ownership is restructured
- [ ] Superseding design is Accepted under a new ADR number
- [ ] Plain multi-supervisor runners gain or lose implicit lease composition
- [ ] Process identity model changes (containers without stable boot_id, remote
      executors without `/proc`)

## Notes (optional)

- **Resource vs task vs lifecycle fences.** Three related tokens exist:
  resource capacity leases, task/lane fencing epochs in `LeaseCoordinator`,
  and worktree lifecycle fence integers. They must not be substituted for each
  other in APIs or docs.
- **Merge serialization.** `checkout_lock` is necessary for Git safety under
  concurrency but is not a substitute for per-task fencing or worktree
  isolation (see Alternatives A and D).
- **Composition caveat.** Describe only the path a runner actually wires:
  bundle/`leased_lane` composition includes `LeaseCoordinator`; plain
  implementation-daemon shards still need explicit worktree lifecycle claims.
- Implementation detail sketches and tuning defaults (heartbeat intervals,
  startup grace, lease TTLs) belong in operator guides and daemon config, not
  as hard ADR constants unless tests pin them as protocol invariants.
