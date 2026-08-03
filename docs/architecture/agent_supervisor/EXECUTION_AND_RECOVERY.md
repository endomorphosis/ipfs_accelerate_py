# Supervisor execution, landing, and recovery

**Status:** Current
**Audience:** Operators, developers, and implementation agents diagnosing
blocked work, scheduling multi-lane execution, or interpreting merge and
acceptance outcomes
**Scope:** Dependency and conflict admission, resource and provider scheduling,
leases and fencing, worktree isolation, implementation providers, validation,
merge queue/train, authoritative completion (separate from merge), heartbeats,
retries, reconciliation, rescue, and quarantine
**Non-goals:** Transport-neutral control operations and authorization policy
([CONTROL_PLANE.md](CONTROL_PLANE.md)); planning, proof tiers, and assurance
pipelines ([PLANNING_AND_ASSURANCE.md](PLANNING_AND_ASSURANCE.md)); prompt-first
entrypoint facade status ([PROMPT_FIRST_RUNTIME.md](PROMPT_FIRST_RUNTIME.md) / DOC-014); package
DAG placement rules ([PACKAGE_MAP.md](PACKAGE_MAP.md)); full operator runbooks
([Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md)). This guide does not
invent new daemons, leases, or completion authority.
**Last verified:** `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03);
lane lifecycle, lease/fence/worktree invariants, merge-versus-acceptance gates,
resource admission, and rescue dispositions checked against `todo_daemon/`,
`runtime/`, `merge/`, `validation/`, `rescue/`, and focused tests listed under
Source anchors.

---

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Implementation daemon | `ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py` | Claim, worktree, provider, validation, merge admission |
| Daemon runners | `todo_daemon/implementation_daemon_runner.py`, `implementation_supervisor_runner.py` | Process entry and loop helpers |
| Worktree pool / lease | `todo_daemon/worktrees.py` — `WorktreePool`, `WorktreeLease` | Filesystem isolation; not completion authority |
| Task / status phases | `todo_daemon/status.py` | Heartbeats, phase keys, acceptance projections |
| Authoritative completion | `todo_daemon/authoritative_completion.py` — `ImplementationReceipt`, `AuthoritativeCompletionGate`, `evaluate_authoritative_completion_gate`, `promote_authoritative_completion`, `reopen_acceptance_for_stale_post_merge_validation` | Merge ≠ completion |
| Execution policy | `todo_daemon/task_execution_policy.py` | Attempt budgets, protected paths, provider routing inputs |
| Conflict / work contracts | `ipfs_accelerate_py/agent_supervisor/core/conflict_graph.py` — `TaskWorkContract`, `ConflictSurface`, `ConflictEdge` | Who may run together |
| Resource admission | `runtime/resource_scheduler.py` — `ResourceScheduler`, `ResourceAdmissionLease`, `AdmissionDecision` | Host/provider/pool capacity |
| Provider batching | `runtime/provider_batch_scheduler.py` | Capacity-checked batch dispatch; receipts |
| Multi-lane runtime | `runtime/multi_supervisor_runner.py`, `event_log.py`, `artifact_store.py` | Lanes, events, large artifacts |
| Lease coordination | `merge/lease_coordination.py` — `LeaseCoordinator`, `LeaseGrant`, `DependencyNotReadyError`, `StaleFencingTokenError` | DuckDB claims, heartbeats, fences |
| Leased lanes | `merge/leased_lane.py` | Lane-scoped lease adaptation |
| Checkout lock | `merge/checkout_lock.py` | Serialize Git checkout mutations |
| Merge train / queue | `merge/merge_train.py`, `merge_queue.py`, `merge_checkpoint.py`, `merge_resolver.py`, `merge_conflict_repair.py` | Landing; not acceptance |
| Validation | `validation/proposal_validation.py`, `validation_commands.py`, `validation_runtime.py`, `validation_scheduler.py` | Pre- and post-merge deterministic gates |
| Rescue | `rescue/rescue_orchestrator.py` — `RescueReceiptDisposition` | Bound recovery under control |
| Recovery store / quarantine | `rescue/supervisor_recovery.py`, `supervisor_watchdog.py`, `recovery_diagnostics.py` | Restart reconcile; quarantine bounds |
| External operational receipts | `core/external_completion.py` | Distinct from task authoritative completion |
| Package pages | `packages/runtime.md`, `merge.md`, `rescue.md`, `validation.md`, `todo_daemon.md` | Domain ownership |
| Focused tests | `test/api/test_agent_supervisor_authoritative_task_completion.py`, `test_agent_supervisor_post_merge_evidence.py`, `test_agent_supervisor_lease_coordination.py`, `test_agent_supervisor_merge_{queue,train,resolver}.py` | Gate, lease, and merge evidence |

Related product narrative: [Architecture — leases and scheduling](../AGENT_SUPERVISOR_ARCHITECTURE.md#leases-fencing-and-worktrees-are-a-distributed-systems-protocol),
[Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md),
[Developer guide — daemons](DEVELOPER_GUIDE.md).

---

## 1. Purpose

This guide is the maintained product narrative for **how admitted work runs,
lands, and recovers** inside the agent supervisor.

It answers:

1. What is a **lane lifecycle**, and which packages own each phase?
2. How do **dependency, conflict, and resource admission** decide whether a
   task may run *now* versus wait?
3. What **lease / fence / worktree** invariants stop stale workers from
   mutating state?
4. Why is **merge** not **authoritative completion**, and when does **stale
   evidence reopen acceptance**?
5. How do operators distinguish **legitimate dependency idle** from
   **provider, resource, validation, merge, or recovery blocks**?

**Non-authoritative signals (never call these completion):**

| Signal | What it proves | What it does not prove |
| --- | --- | --- |
| Process **PID** alive | A daemon process exists | Healthy progress, correct task, or acceptance |
| **Provider exit** 0 / model “success” | The child process returned | Scope-valid patch, passed validation, or completion |
| **Merge** / merge-queue ancestry | Code landed on a target branch | Post-merge gates, fresh evidence, or board completion |
| Board status `done` without gate packet | A projection was written | Recomputed `AuthoritativeCompletionGate` admission |
| Heartbeat freshness alone | Owner still holds a lease window | Terminal success |

Models propose; **leases, validation, merge receipts, and recomputed completion
gates** decide. Callers cannot self-assert `completion_authoritative` on an
`ImplementationReceipt` (`build_implementation_receipt` discards that flag).

---

## 2. Context and component map

Execution spans the ops and edge packages on the acyclic domain DAG. Higher
layers orchestrate; lower layers own isolation and landing. The diagram below
is the **fully composed bundle/leased-lane path**. The plain
`runtime.multi_supervisor_runner` path instead starts sharded implementation
daemon processes; it does not itself construct `LeaseCoordinator` or
`ResourceScheduler`.

```text
 Taskboard projection (task_sources/) + control admission
                    │
                    ▼
         runtime/  resource + provider admission
         (ResourceAdmissionLease, batch scheduler)
                    │
                    ▼
         todo_daemon/  implementation / supervisor loops
              │ claim task  ·  open WorktreeLease
              │ invoke provider  ·  run validation
              │ emit events / status / heartbeats
              ▼
         merge/  LeaseCoordinator · checkout_lock
              │ merge_queue / merge_train / resolver
              │ ImplementationReceipt (merged, pending gates)
              ▼
         authoritative_completion  (separate acceptance)
              │ promote only if all gates recompute clean
              │ reopen if post-merge validation goes stale
              ▼
         rescue/  watchdog · reconcile · quarantine
```

| Layer | Responsibility | Owning packages |
| --- | --- | --- |
| **Admission (composed path)** | Dependencies ready? Conflicts free? Capacity free? | `objectives.bundle_supervisor`, `core/conflict_graph`, `runtime/resource_scheduler`, `merge/lease_coordination` |
| **Actuation** | Claim, worktree, provider, validation commands | `todo_daemon/`, `validation/` |
| **Landing** | Queue/train, conflict repair, Git hygiene | `merge/` |
| **Acceptance** | Recompute gates; board mutation only when admitted | `todo_daemon/authoritative_completion` |
| **Recovery** | Watchdog, rescue orchestration, quarantine | `rescue/` |
| **Observation** | Events, artifacts, scheduler metrics | `runtime/event_log`, `artifact_store`, `scheduler_metrics` |

Control-plane authorization and operation vocabulary remain in
[CONTROL_PLANE.md](CONTROL_PLANE.md). This page starts **after** a task is a
schedulable, identity-bound unit of work.

---

## 3. Lane lifecycle

A **lane** is one unit of concurrent execution: typically a claimed task, an
isolated worktree, a provider process, validation, and optional merge
submission. In the bundle-supervisor composition it is also lease/fence and
resource-admission bound. The plain `runtime.multi_supervisor_runner` supplies
process supervision and shard filters around implementation/supervisor daemon
runners; those runners must not be described as implicitly acquiring the
separate bundle supervisor's `LeaseCoordinator` or `ResourceAdmissionLease`.

### 3.1 Lifecycle phases (conceptual)

Phases are recorded in daemon status (`status.py` phase keys and active
snapshots). Exact string vocabularies may vary by runner; the **order and
authority** of transitions do not.

```text
  ready / waiting_deps
        │  dependency + conflict + resource + lease claim
        ▼
  claimed  ──► worktree open  ──► implementing (provider)
        │                              │
        │                              ▼
        │                         validating
        │                              │
        │              ┌───────────────┼────────────────┐
        │              ▼               ▼                ▼
        │         failed/retry    merge_pending    (no merge policy)
        │              │               │
        │              ▼               ▼
        │         release/rescue  merged_pending_acceptance
        │                              │
        │              ┌───────────────┼────────────────┐
        │              ▼               ▼                ▼
        │     authoritatively_   acceptance_reopened   denied
        │         completed      (stale evidence)
        ▼
  terminal (release lease / fence advance / quarantine)
```

| Phase family | Operator meaning | Not completion? |
| --- | --- | --- |
| Waiting deps / not claimable | Legitimate **dependency idle** or conflict wait | Yes — healthy scheduler wait |
| Claimed / implementing | Live owner with lease; provider may be running | PID/provider progress ≠ done |
| Validating | Deterministic commands on the worktree tip | Command green is not acceptance |
| Merge pending / merged | Landing path active or branch advanced | **Merge ≠ authoritative completion** |
| Merged pending acceptance | Receipt exists; gates still open | Correct intermediate state |
| Authoritatively completed | Gate admitted **and** board mutation authorized | Only this is task completion |
| Acceptance reopened | Freshness or bound evidence invalidated | Prior acceptance is void |
| Quarantined / failed terminal | Recovery or budget exhaustion | Requires operator or rescue |

### 3.2 Heartbeats and process liveness

- Lease **heartbeats** (`lease_coordination` heartbeat table) prove the **claim
  owner** is still within the lease window for the current fencing token.
- Daemon **status PID** fields prove a process exists for ops tooling.
- A live PID with a **stale fencing token**, expired lease, or mismatched
  worktree is **not** healthy progress: terminal operations must fail closed
  (`StaleFencingTokenError`, `LeaseExpiredError`).

Watchdogs (`supervisor_watchdog`) distinguish a live lifecycle from a **dead
owner** by combining heartbeat age, lifecycle state, and process probes—not PID
alone. They do not implement a generic per-phase age timer.

### 3.3 Attempts and retries

Attempt budgets and repair tasks are policy inputs
(`task_execution_policy`, retry-budget helpers on the implementation daemon).
Retries:

- must re-obtain a current **lease and fence** before mutation on the fully
  composed leased/control path; plain implementation-daemon retries instead
  reacquire their own task/worktree lifecycle claim,
- must re-run **validation** on the current worktree tip,
- must not treat a previous provider exit as cached success,
- terminate into **rescue**, **quarantine**, or a bounded repair task when
  budgets exhaust—not unbounded silent loops.

---

## 4. Scheduling admission

Scheduling is **admission control**, not only priority sorting
([Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md#scheduling-is-admission-control-not-just-priority-sorting)).

### 4.1 Admission layers

| Layer | Question | Primary owner | Typical fail / wait |
| --- | --- | --- | --- |
| **Dependency** | Are prerequisites accepted / ready? | Board metadata + `LeaseCoordinator.claimability` / `DependencyNotReadyError` | Wait — **dependency idle** |
| **Conflict** | Do file/symbol/scope surfaces overlap live work? | `conflict_graph` surfaces and edges | Wait or recolor lanes |
| **Resource** | Host CPU/memory/disk/process slots free? | `ResourceScheduler` / `ResourceAdmissionLease` | Wait — **resource block** |
| **Provider** | Model route capacity / batch slots free? | `provider_batch_scheduler`, provider slots on `AdmissionDecision` | Wait — **provider block** |
| **Lease / fence** | Can this claimant own the task/lane now? | `LeaseCoordinator.claim`, control lease validators | Deny / conflict — reclaim or retry |
| **Policy / protected paths** | Is the edit scope allowlisted? | Control + daemon protected-path policy | Fail closed — not idle |

Priority ranks **eligible** work. Admission answers whether that work may
consume capacity **right now**. A high-priority task without a kernel or
provider slot waits with an explicit reason; it must not permanently fail solely
because a peer holds capacity.

### 4.2 Conflict and work contracts

`TaskWorkContract` / `ConflictSurface` normalize:

- dependency edges (`depends_on` and related fields),
- conflict keys (paths, symbols, interfaces, submodules),
- shard and bundle membership for multi-lane coloring.

Conflict-free coloring produces concurrent lanes; overlapping scopes serialize
or wait. Merge-conflict receipts feed learned weights
(`ConflictWeightHistory`) for later planning—they do not by themselves complete
tasks.

### 4.3 Resource and provider admission

`ResourceScheduler.evaluate()` evaluates a request and `acquire()` reserves an
admitted lease; these APIs produce an `AdmissionDecision` and, on acquisition,
a `ResourceAdmissionLease`:

- `admitted: bool` plus structured `reasons`,
- host vs provider available slots, resource class/pool, reserved process and
  memory bounds,
- queue depth, merge age, active lease counts for backpressure signals.

`ResourceAdmissionLease` is **reclaimable capacity**, distinct from a **task
lease** / fencing epoch. Holding a resource lease without a valid task fence
must not authorize repository mutation. Provider batches check health and
capacity immediately before dispatch and emit content-addressed batch receipts;
adapter “success” remains a proposal-class input to validation.

### 4.4 Task sharding

Boards and multi-supervisor runners apply **shard filters** so parallel lanes
drain disjoint task populations (by index, track, bundle, or conflict color).
Sharding is a **scheduling partition**, not a second authority plane.
Validation, merge, and completion rules remain shared; lease/fence and resource
admission apply when the runner is composed through
`objectives.bundle_supervisor` / `merge.leased_lane`, not merely because a
plain implementation-daemon process received a shard filter.

---

## 5. Leases, fencing, and worktrees

Leases, fencing tokens, and worktrees form a **distributed-systems protocol**
for multi-process execution. They are not cosmetic locks.

### 5.1 Invariants

| Invariant | Meaning | Enforced by |
| --- | --- | --- |
| **Single live owner** | At most one accepted live claim per task (or lane scope) for the current epoch | `LeaseCoordinator`, fencing tables |
| **Fence monoticity** | Terminal ops require the **current** fencing token; superseded tokens fail closed | `StaleFencingTokenError` |
| **Heartbeat window** | Owner must refresh before expiry; expiry reclaims capacity | Heartbeat rows + expiry |
| **Worktree is isolation, not authority** | A dirty or reused worktree proves nothing without lease+fence+receipts | `WorktreePool` + claim checks |
| **Checkout serialization** | Concurrent Git checkout mutations are locked | `checkout_lock` |
| **DuckDB owner authority** | Mutable lease/claim state is single-writer; IPFS/Parquet replicas never grant leases | `lease_coordination` design |

```text
 claim(task, claimant) → LeaseGrant(fencing_token, expires_at)
        │
        ├─ heartbeat(fencing_token) while working
        │
        ├─ mutate only if token still current
        │
        └─ terminal publish / merge boundary re-validates token
              stale token → reject (no silent commit)
```

### 5.2 Worktree lifecycle

`WorktreePool.acquire` / `WorktreeLease`:

1. Creates or reuses an isolated checkout for the claimed task,
2. binds metadata for cache reuse when inputs match,
3. releases with optional reuse or invalidation.

Worktrees keep concurrent lanes from sharing a dirty main working tree. They do
**not** replace:

- control authorization,
- protected-path policy,
- lease/fence checks at commit and merge boundaries,
- validation command success,
- authoritative completion gates.

### 5.3 Lease vs resource lease vs completion

| Token | Authorizes | Does not authorize |
| --- | --- | --- |
| Task / lane **lease** + **fence** | Isolated execution and terminal publish under claim | Goal completion; board acceptance |
| **ResourceAdmissionLease** | Using host/provider capacity | Any Git or board mutation |
| **ImplementationReceipt** (merged) | Evidence that code landed | Authoritative completion |
| **AuthoritativeCompletionGate** (admitted) | Board completion mutation when bound evidence recomputes | Future tasks or other goals |

---

## 6. Provider execution and validation

### 6.1 Implementation providers

The implementation daemon selects a ready task, holds isolation, and invokes a
configured provider (Grok, Codex, Goose, deterministic fallback, etc.) with a
bounded prompt, timeout, and workspace path. Provider routing and capacity
classification (`classify_provider_capacity_failure`, batch scheduler) treat
quota and overload as **provider blocks**, not dependency idle.

| Outcome | Classification | Next step |
| --- | --- | --- |
| Provider capacity / rate limit | **Provider block** | Wait / backoff; do not burn attempt budget as “logic failure” without policy |
| Provider crash / non-zero exit | Execution failure | Retry under budget or rescue |
| Provider “success” text / patch | Proposal only | Validate; never complete |
| Deterministic-only policy + model call | Policy denial | `deterministic_only` gate / reject event |

### 6.2 Validation

`validation/` owns deterministic gates: command selection, runtime execution,
scope adjudication, and proposal validation policy. Validation runs against the
**worktree tip** (pre-merge) and again as **post-merge** evidence bound to the
merge commit when acceptance is evaluated.

Rules:

- Model prose never substitutes for configured validation commands.
- A green pre-merge run that is later **stale** relative to the merge commit
  fails the **freshness** gate.
- Pytest green does not bypass admission, protected paths, or completion gates
  (see program admission policy in objectives; enforced in daemon admission).

---

## 7. Merge versus authoritative completion

This separation is a **hard product invariant**.

### 7.1 Merge lands code

The merge package (`merge_queue`, `merge_train`, `merge_resolver`,
`merge_conflict_repair`, checkpoints) lands implementation branches onto the
configured target. Merge may produce:

- a merge commit and ancestry relationship,
- conflict repair attempts with bounded evidence,
- an `ImplementationReceipt` with `merged=True` and
  `acceptance_state=implemented_merged_but_pending`.

Merge success is a **MergeReceipt-class** fact: code is on the branch. It is
**not** task or goal completion.

### 7.2 Authoritative completion admits acceptance

`authoritative_completion.py` keeps acceptance auditable and independent of the
merge path. Gate kinds (`AUTHORITATIVE_COMPLETION_GATE_KINDS`):

| Gate kind | Role |
| --- | --- |
| `merge` | Bound merge commit verified; `merged` true |
| `freshness` | Post-merge validation not stale |
| `semantic` | Semantic / scope evidence bound to commits |
| `proof` | Required proof evidence when policy demands it |
| `provider_review` | Required review evidence when policy demands it |
| `deterministic_only` | No forbidden model invocation under deterministic-only policy |

`evaluate_authoritative_completion_gate` **recomputes every gate from bound
evidence** and never trusts a cached “all green” list. Promotion
(`promote_authoritative_completion`) only sets
`completion_authoritative=True` after a fresh clean recomputation. Board
mutation requires `authorize_completion_mutation` to match the **exact**
recomputed gate packet.

Acceptance states:

| State | Meaning |
| --- | --- |
| `implemented_merged_but_pending` | Landed; gates incomplete or denied |
| `authoritatively_completed` | Gates admitted; completion may be recorded |
| `acceptance_reopened` | Prior acceptance invalidated (e.g. stale post-merge validation) |

Events such as `implementation_merged_pending_acceptance`,
`authoritative_task_completion_admitted` /
`authoritative_task_completion_denied`, and
`acceptance_reopened_stale_post_merge_validation` make the distinction
observable in logs and status projections (`status.py`).

### 7.3 Stale evidence reopens acceptance

`reopen_acceptance_for_stale_post_merge_validation`:

- marks validation stale / failed,
- forces the freshness gate pending,
- clears `completion_authoritative`,
- sets `acceptance_state=acceptance_reopened`,
- **retains** exact implementation and merge commit bindings for audit.

Operators must treat reopened acceptance as **open work**, not as a historical
“still done” board line. Tree identity drift, lost post-merge receipts, or
failed revalidation all reopen or deny—never silently keep acceptance.

```text
  merge_train success
        │
        ▼
  ImplementationReceipt (merged, completion_authoritative=false)
        │
        ▼
  evaluate_authoritative_completion_gate  ──denied──► merged_pending / repair
        │ admitted
        ▼
  promote + authorize board mutation
        │
        ▼
  authoritatively_completed
        │
        │  post_merge validation becomes stale
        ▼
  reopen_acceptance_for_stale_post_merge_validation
        │
        ▼
  acceptance_reopened  (must re-validate / re-promote)
```

### 7.4 External and goal completion remain distinct

- `core/external_completion.py` records **external operational** receipts; they
  do not auto-promote task-board authoritative completion.
- Objective / goal completion
  ([CONTROL_PLANE](CONTROL_PLANE.md) / objectives package) requires goal-level
  evidence and policy; draining the task queue is only one observation.

---

## 8. Operator block taxonomy

When a task is not advancing, classify the **block class** before changing
retry budgets or restarting processes.

| Block class | Typical signals | Operator interpretation | Not this |
| --- | --- | --- | --- |
| **Dependency idle** | `DependencyNotReadyError`; claimability shows unmet deps; board `depends_on` still open | **Legitimate wait** — do not treat as failure; fix or wait for upstream acceptance | Provider outage |
| **Conflict wait** | Conflict surface overlap; lane recolor; scope conflict errors | Wait for peer lane release or reshard | Permanent logic bug |
| **Resource block** | `AdmissionDecision.admitted=false` with host/process/memory reasons; queue capacity | Capacity pressure; scale or reduce concurrency | Task incorrectness |
| **Provider block** | Provider slots zero; capacity/rate-limit classification; batch health fail | Provider or quota issue; backoff | Validation failure |
| **Validation block** | Validation command non-zero; scope adjudication deny; protected path | Patch or command wrong; repair attempt | Merge queue backlog |
| **Merge block** | Checkout lock; dirty tree; train conflict; lease missing at merge boundary | Landing path stuck; resolve conflict / lock | Already completed |
| **Acceptance / evidence block** | `merged_pending`; gate reason codes; `completion_authoritative_false` | Need bound post-merge evidence or promotion | “Green merge” alone |
| **Recovery / quarantine** | Rescue disposition `quarantined` / `denied`; recovery store quarantine; exhausted retry | Incident path; inspect quarantine and receipts | Healthy dependency idle |
| **Stale isolation** | `StaleFencingTokenError`; expired lease; crash fence reconcile | Reclaim and re-claim; discard stale worker output | Ignore and force-merge |

### 8.1 Quick decision tree

```text
 Is the task missing prerequisites on the board / claimability?
   yes → DEPENDENCY IDLE (wait or finish upstream)
   no  ↓
 Does conflict surface overlap an active lane?
   yes → CONFLICT WAIT
   no  ↓
 Is ResourceScheduler / provider admission denying slots?
   yes → RESOURCE or PROVIDER BLOCK (capacity)
   no  ↓
 Is there a live claim with fresh heartbeat and current fence?
   no  → RECLAIM / RE-CLAIM (stale isolation or crash)
   yes ↓
 Is the provider still running or in capacity backoff?
   yes → PROVIDER progress or PROVIDER BLOCK
   no  ↓
 Did validation fail on the current tip?
   yes → VALIDATION BLOCK
   no  ↓
 Is merge queue/train blocked (lock, conflict, dirty tree)?
   yes → MERGE BLOCK
   no  ↓
 Is code merged but acceptance pending / reopened / denied?
   yes → ACCEPTANCE / EVIDENCE BLOCK (not incomplete merge)
   no  ↓
 Rescue/watchdog quarantine or budget exhaustion?
   yes → RECOVERY BLOCK
   no  → Inspect events/metrics; avoid calling PID or provider exit "done"
```

---

## 9. Recovery, rescue, and quarantine

### 9.1 Crash and restart reconciliation

On restart, the recovery-composed leased path can **pause new admission** until
tree, state, fence, and last terminal receipts reconcile. The plain
implementation-daemon path performs its own bounded status, worktree, and merge
reconciliation but does not inherit that global admission pause simply by
running under `multi_supervisor_runner`. In every composition, recovery must
not invent authoritative completion or re-apply an already proven exact
successful mutation.

### 9.2 Rescue orchestration

`rescue_orchestrator` runs under the control-plane `rescue` operation with
effect, lease, and root bindings. `RescueReceiptDisposition`:

| Disposition | Meaning |
| --- | --- |
| `applied` | Rescue actions applied under authority |
| `recovered` | Target returned to a healthy schedulable state |
| `stopped` | Bound stop (budget, policy, or explicit halt) |
| `denied` | Authorization or schema denial |
| `partial` | Some actions applied; residual remains |
| `quarantined` | Material isolated; do not continue blind retries |

Rescue **must not** grant completion without fresh evidence
([packages/rescue.md](packages/rescue.md)). Self-authorization, stale roots,
changed arguments after bind, and exhaustion drift fail closed.

### 9.3 Quarantine

Quarantine paths isolate corrupt status, oversized projections, or incident
artifacts under byte budgets (`max_quarantine_bytes`). Quarantine is a
**safety boundary**, not completion and not dependency idle. Operators inspect
quarantine, fix root cause, then re-admit with new leases/fences.

### 9.4 Recovery decision tree (summary)

```text
 Detect stall / crash / corrupt status
        │
        ▼
 Preserve journals, receipts, leases; stop new automatic admission
        │
        ▼
 Reconcile fence + tree + last terminal receipt
        │
        ├─ incomplete work, budget remains → bounded retry / rescue_preview → rescue
        ├─ authority or schema fail → DENIED (no mutation)
        ├─ corrupt / unbounded artifact → QUARANTINE
        └─ clean reconcile, capacity free → re-admit scheduling
        │
        ▼
 Never mark authoritative completion from recovery alone
```

---

## 10. Trust, authorization, and failure semantics

### 10.1 Fail-closed conditions

Work **must not** advance state-changing effects when:

- lease is missing, expired, or scope-mismatched,
- fencing token is stale,
- repository `tree_id` is stale relative to policy,
- validation required by policy failed or is stale post-merge,
- protected or undeclared paths appear in the patch,
- authoritative completion gate recomputation denies or mismatches,
- rescue/control authorization is absent or self-granted.

### 10.2 Degradation

| Missing / degraded dependency | Behavior |
| --- | --- |
| Optional provider binary / quota | Provider block or deterministic fallback (recorded, lower assurance) |
| Solver / kernel capacity | Resource wait; no fake proof pass |
| DuckDB lease backend unavailable | Fail closed on claims; no IPFS replica authority |
| Event/artifact store pressure | Bound projections; quarantine oversized material |

### 10.3 Non-authoritative signals (recap)

Chat logs, board status alone, cache hits without re-derivation, import success,
capability probes, **PID liveness**, **provider exit codes**, and **merge
ancestry** do not authorize completion mutations.

---

## 11. Rationale

1. **Multi-process safety** — Daemons crash and restart; fencing makes late
   workers harmless without blocking reclaim via lease expiry.
2. **Capacity fairness** — Separating priority from resource/provider admission
   prevents a hot model route from starving proof or validation lanes.
3. **Merge ≠ truth** — Landing code is necessary but insufficient; post-merge
   freshness and bound gates catch tip drift and incomplete evidence.
4. **Observable blocks** — Explicit idle vs block classes prevent operators from
   “fixing” healthy dependency waits or treating capacity as task failure.
5. **Recovery without authority expansion** — Rescue and quarantine restore
   schedulability; they do not mint completion.

---

## 12. Alternatives considered

| Alternative | Breakage |
| --- | --- |
| Treat provider exit 0 as task complete | Hallucinated or out-of-scope patches land as done |
| Treat merge ancestry as acceptance | Stale or incomplete post-merge evidence stays green |
| PID heartbeat as health sole signal | Zombie or wrong-phase processes look healthy |
| Global `locked=true` without fencing | Delayed workers commit after takeover |
| IPFS / Parquet replica as lease authority | Split-brain claims; stale publishers steal work |
| Collapse dependency wait into failure | False retries thrash upstream incompleteness |
| Rescue auto-completes tasks | Recovery becomes a backdoor past validation gates |
| Single shared working tree for lanes | Cross-lane dirt and non-reproducible validation |

---

## 13. Consequences

**Positive**

- Operators can map a stuck task to a named admission or acceptance boundary.
- Concurrent lanes scale under leases without trusting worktrees alone.
- Acceptance can reopen when evidence goes stale—correctness over vanity
  “done” status.
- Merge train and completion gates evolve independently.

**Negative / costs**

- More states to learn (`merged_pending`, `acceptance_reopened`).
- Capacity waits require metrics literacy (`AdmissionDecision`, provider slots).
- Dual isolation (task lease + resource lease + worktree) adds operational
  surface.
- Forced revalidation after reopen can delay board closeout.

---

## 14. Extension and compatibility

1. New schedulers must emit **typed wait/deny reasons** (dependency vs resource
   vs provider vs policy)—do not overload a single `blocked` bit.
2. New merge strategies still produce non-authoritative
   `ImplementationReceipt`s until completion gates recompute.
3. New providers plug in behind validation; they must not write completion
   flags.
4. Rescue actions remain control-bound operations with effect lists and leases.
5. Compatibility facades for flat module imports do not create alternate lease
   or completion authorities ([PACKAGE_MAP.md](PACKAGE_MAP.md)).

---

## 15. Operational signals

| Signal surface | Use for |
| --- | --- |
| Daemon status JSON (`status.py`) | Phase, PID, active task, acceptance projection |
| Lease claimability / heartbeat rows | Dependency idle vs expired claim |
| `AdmissionDecision` / scheduler metrics | Resource and provider pressure |
| Event log (`runtime/event_log`) | Compact operational history |
| Artifact store | Large validation / provider outputs by content id |
| Implementation receipt + gate packet | Merge vs pending vs authoritative vs reopened |
| Rescue run / action receipts | Disposition and reason codes |
| Quarantine directory | Corrupt or oversized isolated artifacts |

Health for execution means **valid lease/fence, admitted capacity, progressing
phase, and evidence-consistent acceptance**—not merely a running PID.

---

## 16. Verification

Deterministic checks for this guide’s claims (run from repository root):

```bash
# Guide present and key vocabulary locked for DOC-013
test -f docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
rg -qi 'lease' docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
rg -qi 'authoritative completion' docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
git diff --check

# Focused behavioral suites (optional; stronger evidence)
python -m pytest \
  test/api/test_agent_supervisor_authoritative_task_completion.py \
  test/api/test_agent_supervisor_post_merge_evidence.py \
  test/api/test_agent_supervisor_lease_coordination.py \
  test/api/test_agent_supervisor_merge_queue.py \
  test/api/test_agent_supervisor_merge_train.py -q
```

Review checklist:

- [ ] PID, provider exit, and merge are never described as completion.
- [ ] Stale post-merge / freshness path reopens acceptance.
- [ ] Dependency idle is distinct from provider/resource/validation/merge/recovery blocks.
- [ ] Source anchors resolve to live modules.
- [ ] No claims of planned high-level facades as current runtime API.

---

## 17. Related guides

| Doc | Relationship |
| --- | --- |
| [CONTROL_PLANE.md](CONTROL_PLANE.md) | Intent, operations, authz, lease fields on requests |
| [PLANNING_AND_ASSURANCE.md](PLANNING_AND_ASSURANCE.md) | Evidence tiers before execution |
| [PACKAGE_MAP.md](PACKAGE_MAP.md) | Domain package DAG |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Daemon loop and debug symptoms |
| [README.md](README.md) | Supervisor docs hub |
| [AGENT_SUPERVISOR_ARCHITECTURE.md](../AGENT_SUPERVISOR_ARCHITECTURE.md) | Deep contracts and theory |
| [AGENT_SUPERVISOR_PHILOSOPHY.md](../AGENT_SUPERVISOR_PHILOSOPHY.md) | Authority ladder pillars |
| [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) | Runbooks and profiles |
| ADR (planned DOC-018) `0004-worktrees-leases-and-fencing.md` | Decision record for isolation |

---

*Interfaces referenced for program identity (documentation, not separate wire
schemas): LaneLifecycle@1, LeaseFence@1, MergeReceipt@1,
AuthoritativeCompletion@1, RescueDisposition@1 — realized by the modules in
Source anchors.*
