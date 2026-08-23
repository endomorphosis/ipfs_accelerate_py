# Agent Supervisor Causal Event Federation Plan

Program identifier: `agent-supervisor-causal-event-federation-v1`
Primary subsystem: `CausalAbstractionSupervisorFederation`
Task prefix: `CASF-`
Root objective: `CASF-G000`
Plan revision: `CASF-PLAN-R1`
Execution branch: `codex/causal-event-supervisor-federation-v1`

## 1. Decision and mission

This program extends the existing `ipfs_accelerate_py.agent_supervisor`; it
does not create a second control plane. The intended end state is one
authenticated, event-driven federation that shares a transactional DuckDB
memory through the exclusive Quack state owner, projects optional history to
DuckLake, and coordinates roughly twelve supervisors plus hundreds of bounded
logical subagents without allowing any model, projection, or process to grant
itself authority.

The transformation is:

```text
authenticated FederationRequest
  -> typed gateway and transactional admission
  -> DuckDB state mutation + event + outbox + generation
  -> exact causal frontier
  -> only affected supervisor shards wake
  -> bounded agents reuse semantic/proof/context objects
  -> isolated effects, validation, merge, and recovery
  -> event-driven fixed point
  -> non-authoritative DuckLake projection
```

Optimization maximizes safely parallel accepted work, accepted criteria per
wall-clock hour, autonomous recovery, deterministic assignment, evidence and
context reuse, and useful specialization. It minimizes repeated model calls,
tokens, repository scans, duplicate work/tests/proofs, idle CPU, polling, lock
contention, merge conflicts, and human interventions.

The following invariants are non-compensable:

- zero unauthorized supervisor, subagent, or mutation creation;
- zero duplicate committed effects, stale-fence completion, or lost
  authoritative transitions;
- zero simulated-as-live evidence, model-created authority, model-created
  policy permission, model-created completion, or false completion;
- zero DuckLake-derived scheduling, lease, policy, or completion authority;
- zero direct multi-process DuckDB file mutation and zero implicit Quack to
  embedded/file fallback;
- zero arbitrary SQL from an agent, unbounded event fanout, hidden validation
  reduction, cross-tenant leakage, or raw credential propagation.

## 2. Preserved authorities and prerequisite classification

CASF-000 and CASF-001 produce the exact current-tree inventory. Every required
surface is classified `available`, `available_with_caveats`, `stale`,
`incompatible`, or `missing`, with source and test identities. Names, imports,
Markdown state, generated reports, historical receipts, fixtures, embedded
tests, Quack imports, DuckLake imports, and similarly named tables are never
sufficient authority evidence.

Minimum accelerator surfaces inspected and extended are:

- `task_sources.control_plane_contracts`, `control_plane_migrations`,
  `control_plane_schema`, `control_plane_repository`,
  `control_plane_transactions`, `quack_capabilities`, and
  `quack_state_client`;
- `runtime.quack_state_server`, `runtime.multi_supervisor_runner`, control,
  runtime, planning, proof, verification, semantic-governor, and adversarial
  assurance surfaces;
- `semantic_state.world_snapshot_builder`,
  `analysis.doctor_causal_localization`, and
  `integrations.ducklake_history_projection`;
- the existing DuckDB/Quack, logic-governed fabric, qualification, and general
  supervisor architecture documents.

Missing authority is handled narrowly: use a compatibility adapter only when
its semantics and effect ceiling are exact; otherwise emit a typed capability
blocker and continue independent work. Missing functionality is never reported
as available.

Canonical ownership remains:

| Owner | Canonical responsibility | Prohibited CASF reinterpretation |
| --- | --- | --- |
| `ipfs_datasets_py` | Semantic meaning, repository/AST/symbol identities, versions, semantic capsules, relationship graphs, contract/proof references, KG/BM25/vector semantics, formal IR | Accelerator may persist/query records but not redefine their meaning |
| `ipfs_accelerate_py` | Federations, supervisors, agents, plans, tasks, claims, leases, fences, budgets, attempts, effects, validation, merge, events, causal propagation, recovery | No competing operational authority |
| DuckDB | Authoritative normalized transactional operational records | One exclusive state-owner boundary only |
| Quack | Qualified multi-client transport and exclusive state-owner service | No direct file access or implicit fallback |
| DuckLake | Rebuildable append-only history, analytics, benchmark, and training projections | Never a scheduling, lease, policy, acceptance, or completion prerequisite |
| `ipfs_kit_py` | Durable artifact/content storage through published interfaces | No duplicate VFS, WAL, proof-seal, or current-pointer authority |
| MCP++ | Existing shared wire/profile contracts | No new MCP++ profile for this program |

## 3. Topology and state ownership

```text
Triggering Agent
      |
      v
Federation Control Gateway -- authenticated typed command
      |
      v
Quack Exclusive State Owner
      |
      v
Authoritative DuckDB
  | transactional state | causal network | event/outbox/router |
      v
Supervisor Federation
  coordinator | repository | domain | verification | merge supervisors
      |
      v
bounded logical subagent pools and isolated worktrees
      |
      v
merge / proof / release
      |
      v
DuckLake append-only projection
```

All authoritative mutations pass through one typed state-owner gateway. In
Quack mode every supervisor/subagent read and write goes through the typed
client and owner. Embedded mode is limited to hermetic tests, single-process
development, and exclusive offline recovery when policy explicitly admits it.
Multi-supervisor availability fails closed when Quack is unavailable or
incompatible.

Direct prompt-to-prompt supervisor communication is noncanonical. Supervisors
exchange typed state, events, causal dependencies, content-addressed evidence,
claims/leases, and world snapshots.

## 4. Closed contracts and external trigger

CASF-002 through CASF-004 define immutable versioned contracts for federation,
supervisor, subagent, shard/rebalance, hierarchical budgets, commands,
idempotency, world snapshots, fixed points, events, subscriptions, causal
nodes/edges/evidence, abstraction maps/interventions, frontiers, and receipts.

Every authority-bearing federation record binds repository and tree identities,
program/objective/policy revisions, operation catalog, control-plane generation,
causal graph revision, semantic roots, population, budget hierarchy, expiry,
issuer, and admitted signature/delegation evidence. Decoders reject unknown
normative fields, empty identities, nonfinite values, unbounded strings/arrays,
arbitrary paths/SQL, raw credentials, executable callbacks, model-authored
authority, and self-promoting definitions.

The typed gateway accepts a bounded `FederationRequest` containing caller DID,
delegation chain, audience, program, repository roots, objective reference,
requested profile, population ceilings, resource/token budgets, effect scope,
policy reference, expiry, nonce, and idempotency key. It authenticates and
authorizes server-side, resolves roots, reserves resources, creates the
federation transactionally, emits the first event, and returns an identity and
receipt. The caller cannot select a database path, submit SQL, grant itself
authority, supply a policy result/completion, select unqualified providers,
reuse foreign leases/fences, or change promotion policy. CLI and MCP adapters
call the gateway directly and never shell out.

## 5. Supervisor hierarchy, lifecycle, and agents

The hierarchy contains a coordinator; repository, domain, verification, and
merge supervisors; and bounded subagent pools. Every supervisor binds exact
identity and parentage, repositories/goals/tasks, allowed task families and
operations, effect/risk/resource/token ceilings, provider routes, proof/merge
policy, subscriptions, lease/fence, checkpoint, and lifecycle.

The closed lifecycle is:

```text
DECLARED -> ADMITTED -> STARTING -> IDLE <-> ACTIVE
ACTIVE/IDLE -> PAUSED -> STARTING
ACTIVE/IDLE/PAUSED -> DRAINING -> COMPLETED or STOPPED
eligible state -> RECOVERING -> IDLE/ACTIVE/QUARANTINED/FAILED
eligible state -> QUARANTINED/FAILED/STOPPED
```

Illegal transitions fail closed. Child supervisors require a current policy and
new bounded assignment; no unrestricted child creation exists.

Logical registration, concurrent admission, OS workers, and provider requests
are separate capacities. The benchmark target is 12 supervisors, 256
registered logical agents, and at most 64 concurrent subagents, never a
universal default. Effective capacity is the minimum of policy, host,
provider, proof, merge, storage, and live-telemetry ceilings; missing or stale
telemetry adds no capacity. Pools support async workers, provider batches,
resource-class queues, fairness, work stealing, per-agent identity/effect/token
scope, cancellation, leases, and receipts.

## 6. Authoritative schema and transactions

CASF-005 extends the existing migration system and normalized database. It
keeps join-critical identities in columns for:

- generations, migrations, identities, federations, supervisors, subagents,
  sessions, process births, repositories/trees, policies, capabilities;
- programs/revisions, objectives/revisions, goals/subgoals/edges, formal plans,
  branches/revisions, acceptance criteria;
- tasks/revisions/dependencies/conflicts/claims/resolutions, leases/fences,
  attempts/checkpoints, resource/provider reservations, merge queue;
- repository semantic records: files, symbols, AST/import/call/effect graphs,
  versions, capsules/dependencies, contracts, obligations, bindings, roots,
  world snapshots;
- proof/test/validation attempts and receipts, caches/seals, counterexamples,
  adversarial findings;
- documents/chunks, BM25 terms/postings, vectors/metadata, KG nodes/edges,
  retrieval receipts and index roots/revisions;
- causal nodes/edges/evidence, abstraction maps/validations, intervention
  tests, slices/frontiers/invalidations;
- domain events, transactional outbox, inboxes/subscriptions/cursors,
  deliveries/acknowledgements/coalescing/dead letters;
- commands/idempotency, authorization/policy/confirmation decisions, effect
  reservations/observations, and audit receipts.

Large immutable bodies live in managed content-addressed storage, while DuckDB
retains identity, type, owner, revision, content reference, source root,
provenance, join fields, status, and freshness.

Every cross-component mutation atomically writes state, one or more domain
events, outbox rows, and a new generation/cursor. Events bind ID/CID/type,
stream/global sequence, causal parents, correlation/causation, federation,
supervisor/task/repository/tree IDs, compact payload and changed-fact refs,
effect class, timestamp, and deduplication key. Delivery is at least once;
authoritative effects are exactly-once through idempotency, CAS, leases, and
fencing. The program never claims exactly-once network delivery.

## 7. Event waiting, routing, and storm control

CASF-010 implements a state-owner `wait_for_events(consumer_id, after_cursor,
subscription_revision, deadline, maximum_events)` boundary. It checks and
registers the waiter without a lost-wakeup window, returns immediately for
matching committed events, otherwise blocks on an owner-controlled condition,
descriptor, or equivalent service signal until event/deadline/cancellation/
shutdown. Remote Quack clients use bounded long-poll or event-stream semantics.
Adaptive polling is an explicitly unqualified compatibility mode only: bounded,
backing off, and never reported as event-driven.

Closed event classes cover repository/tree/symbol/contract/semantic/capsule,
proof/test/task/goal/subgoal/plan/policy/capability/provider/lease/merge,
counterexample/human/health/resource/rebalance, and DuckLake projection
changes. Subscriptions use bounded selectors over event class and typed
repository/tree/intent/task/symbol/contract/proof/supervisor/resource/causal
identities; arbitrary SQL is forbidden.

Coalescing supports debounce, supersession, batches, per-consumer limits,
backpressure, retries, dead letters, quarantine, circuit breaking, and safe
expiry. Intermediate audit, payment, lease/fence, irreversible effect, proof
lineage, and legal/security transitions are never coalesced away.

## 8. Causal abstraction network

The multilevel network is:

- L0 runtime observations: processes, tools, provider results, file changes,
  tests/proofs;
- L1 code/artifact state: files, symbols, AST, contracts, capsules, proofs,
  indexes, effects;
- L2 work state: tasks, attempts, claims, validation/merge units, procedures;
- L3 intent state: subgoals, goals, plans, objectives, assignments;
- L4 federation state: programs, federations, portfolio policy, releases.

The closed edge vocabulary is `CAUSES`, `ENABLES`, `DISABLES`, `BLOCKS`,
`INVALIDATES`, `PRODUCES`, `OBSERVES`, `CONSUMES`, `DEPENDS_ON`,
`CONFLICTS_WITH`, `REFINES`, `ABSTRACTS`, `IMPLEMENTS`, `DELEGATES_TO`,
`COMPENSATES`, and `SUPERSEDES`. Authoritative edges require static, trace,
contract, proof/test, effect, event, counterexample, delta-debugging, unsat-core,
or human-reviewed causal evidence. Retrieval/similarity/model output may only
nominate candidates.

Abstraction maps bind low/high models and variables, abstraction/intervention
functions, admitted/excluded domains, validation, and faithfulness status:
`EXACT`, `CONSERVATIVE`, `EMPIRICALLY_SUPPORTED`, `HEURISTIC`, `REFUTED`, or
`UNKNOWN`. Only exact and separately policy-admitted conservative maps may
control scheduling/invalidation. Representative interventions compare the
abstracted low-level outcome against the high-level mapped intervention;
mismatches and excluded domains are durable evidence.

For every event the frontier computes changed facts, exact descendants,
admitted abstraction projections, affected work/proofs/goals/supervisors, and
classifies `must_wake`, `may_wake`, or `do_not_wake`. Unknown dependencies widen
rather than suppress the frontier. `do_not_wake` requires proved/admitted
independence.

On wake a supervisor validates batch/cursor identities, loads only the minimal
causal slice, refreshes affected capsules, recomputes affected tasks/proofs,
reuses unchanged receipts, reserves work, executes or remains idle, commits
results/events, and advances its cursor transactionally. It never scans the
complete board on every wake.

## 9. Shared semantic, retrieval, proof, and context state

All supervisors use one tree-bound semantic root for a repository generation:
AST/symbol/call/dependency/effect/contract/proof graphs, capsules, bindings,
selection data, KG, vector, and BM25 indexes. Tree changes update only changed
files/symbols, exact dependencies, affected capsules/index records and
proof/test dependencies, then emit causal invalidations.

BM25 retrieves exact lexical/symbol evidence; vectors rank semantic
nominations; KG traversal exposes bounded relationships. Every retrieval binds
index revision, source CID, tree, score, method, and partitions. Retrieval
proposes; exact analysis disposes. It cannot establish cause, independence,
authority, policy, proof, or completion.

Federation-wide content-addressed context objects cover policy/objective
prefixes, repository/symbol/task/proof capsules, failure signatures, and
procedures. Reuse requires compatible model/provider, policy/authority,
identical tree and context dependency root, and privacy permission. Each call
contains the unresolved question, mandatory authority core, causal ancestors,
required evidence/symbols/tests/proofs, and bounded alternatives—not the whole
federation/board/repository/history. Token ledgers record use and avoided tokens
by federation, supervisor, agent, goal, task, question, and provider.

## 10. Deduplication, scheduling, sharding, and budgets

Task-intent identity includes tree, goal/subgoal, operation, targets,
acceptance, effect class, and validation. Exact duplicates share one task and
result; subsumed work depends/splits; overlap gets explicit boundaries;
conflicts serialize or select a plan; only proved/conservatively admitted
independence runs concurrently.

The parallel frontier combines causal/task/proof dependencies, effect sets,
file/symbol/state ownership, resource/provider/merge capacity, and merge
pressure. Every admitted task has supervisor/subagent assignment, resource and
token reservation, worktree, lease/fence, merge lane, and validation plan.
Shared reads do not require exclusivity; authoritative writes serialize;
disjoint worktrees may run concurrently; merge order and global proof/model
budgets remain explicit.

`ShardRebalancePlanner` reacts to load, idleness, graph/resource/provider
changes, hotspots, merge pressure, task shifts, and failure. It freezes the
assignment revision, stops claims, drains/transfers safely, preserves task and
attempt identities/checkpoints/cursors, increments fencing, activates the new
assignment, and emits a receipt. Irreversible in-flight effects never move.
Work stealing is allowed only for unclaimed in-ceiling work with current
semantic state and atomic budget/assignment/fence transfer; it never bypasses
repository, policy, proof, merge, privacy, or human review.

Budgets form `Federation -> Supervisor -> Subagent -> OperationReservation`
over CPU, memory, GPU, processes, temporary/durable bytes, model calls/tokens/
spend, proof/validation time, merge slots, human questions, and wall time.
Children cannot exceed parents. Return and transfer use CAS plus events, and
validation reserves cannot fund speculative reasoning.

## 11. Merge, recovery, fixed point, and DuckLake

All tasks execute in isolated worktrees under explicit effect ownership and
merge lanes. Merge/proof/release consumes current-tree receipts and preserves
failed attempts. Recovery covers owner/transaction/supervisor/subagent/
consumer crashes, duplicate/out-of-order events, connection/partition/cursor/
lease/fence faults, storms/dead letters, provider/DuckLake outages, partial
projections, merge conflicts, proof timeouts, and unknown external effects.
Recovery reconnects through the owner, validates generation, resumes durable
cursors, replays idempotently, rejects stale fences, reconciles unknown effects,
and rebuilds derived projections. Process exit is not completion.

Fixed point requires no ready unclaimed task, active attempt, unresolved effect,
required validation/proof, merge-ready item, answerable human decision, due
recovery action, unprocessed authoritative event, or stale required semantic
component. Its receipt binds watermark, population, claims/leases/attempts,
merge/proof/test state, semantic roots, snapshot/frontier, policy/capabilities,
and budgets. Quiet queues and completed boards alone prove nothing.

DuckLake projection is:

```text
DuckDB transaction/outbox
  -> idempotent projection worker
  -> immutable bounded partition files/catalog
  -> projection receipt back in DuckDB
```

Receipts bind event range, control generation, checksums/CIDs, cursor, schema,
redaction, tenant/privacy policy, and recovery. DuckLake outage leaves the
control plane live with typed unavailable/lagging projection state. Analytics
may measure duplicates, invalidation paths, communication/tokens/wakeups,
merge conflicts, cache reuse, and abstraction error, but cannot mutate
production state.

## 12. Security and public control

Default transport is loopback, authenticated, opaque-handle based, audited,
and secret-free in argv/logs/prompts/receipts/provider environments. Remote
access stays unavailable until mutual authentication, encryption, caller DID
and delegation/audience/tenant authorization, rate limiting, revocation, and
audit are separately qualified. Every query/command binds verified tenant,
federation, supervisor, and repository scope. Events contain compact redacted
identifiers; private bodies and retrieval indexes obey the same authorization.

The existing `SupervisorControlService` gains read operations for federation
capabilities/list/get/status/health/metrics/snapshot/causal graph/frontier/
events/lag/supervisors/subagents/shards/tasks/leases/resources/budgets/proofs/
indexes/DuckLake, and authorized mutation operations create/start/pause/resume/
drain/stop/cancel/rebalance/scale/quarantine/replay/retry/rebuild. Mutations
require exact roots, idempotency, expected generation, lease/fence, dry run,
expected effects, authorization, and audit receipt. Python, `ipfs-accelerate
agent federation ...`, and canonical MCP agent-supervisor adapters must have
contract parity.

## 13. Formal verification, tests, and observability

Existing formal tooling models event delivery, claims/leases/fencing,
supervisor lifecycle, shard transfer/budget conservation, and causal
propagation. Required properties include no lost committed event/lost wakeup,
idempotent duplicate delivery, durable cursor ordering, one current task owner,
stale-owner rejection, no pre-admission work, safe draining/recovery, no
double/disappearing shard ownership, no irreversible transfer, exact descendant
notification, nomination non-authority, stale-map non-suppression, and explicit
fixed-point groups for cycles.

Focused hermetic tests live under `test/api/causal_federation/` and cover every
required contract, authentication, lifecycle, population/budget bound, Quack
authority/file rejection, outbox/wakeup/replay/cursors/dead letters/coalescing/
storms/subscriptions, causal/abstraction/intervention/frontier behavior,
incremental semantic/proof/index invalidation, dedup/frontier/claims/leases/
fences/stealing/rebalance/recovery, DuckLake outage/replay, effects/fixed point,
CLI/MCP parity, tenant/secret isolation, idle stability, 12-supervisor and
256-agent load, token efficiency, and parallel throughput. Import/collection
must not install extensions, use networks, or start services. Live Quack,
DuckLake, provider, multiprocess, and multihost tests are separately marked.

Metrics observe active populations/work states, duplicate suppression, event
rate/lag/coalescing/dead letters, frontier/wakeup accuracy, tokens/context reuse,
model/provider calls, proof/validation reuse, merge conflicts, idle CPU, and
DuckLake lag. Metrics alone are never completion/release evidence.

## 14. Frozen benchmarks and promotion gates

Scale target: 12 supervisors, 256 registered agents, 64 concurrent slots,
1,000 bounded tasks, and 100,000 event deliveries including replay using real
independent processes where possible. Event measures include commit-to-wakeup,
outbox drain, lag, duplicates/loss, coalescing, and idle query/wakeup counts.
Parallel comparison uses identical host/tasks/providers/tests/proofs/budgets
against one qualified supervisor and targets at least 3x accepted-task
throughput without lower assurance or duplicate effects.

Token targets against an equivalent baseline are at least 50% fewer repeated
context tokens, 40% fewer model input tokens per accepted criterion, 60% fewer
duplicate model calls, 70% eligible semantic-capsule reuse, and 80% fewer full
board scans. No-event intervals require zero model calls, board scans, context
recompilations, and unchanged writes; only bounded heartbeat/deadline activity
is allowed.

Promotion is conjunctive. Required safety results are zero direct multiprocess
file writes, store ambiguity, event loss, duplicate effects, stale-fence
completion, unauthorized creation, tenant leakage, agent SQL, or secret leaks;
100% exact-descendant notification; zero nomination/stale-map authority; zero
cycle/shard corruption; zero busy loops/idle models/full scans/writes; replay
idempotency; benchmark populations within live capacity; zero ownership/effect/
merge corruption; and the token targets without reduced evidence. DuckLake
promotion additionally requires current compatible DuckLake/httpfs, typed
catalog, idempotent secure recoverable projection, event-range binding, and
receipt. Its failure never blocks DuckDB/Quack promotion.

## 15. Tranches and safe execution order

| Tranche | Tasks | Deliverable | Admission restriction |
| --- | --- | --- | --- |
| 1 authoritative federation core | CASF-000..012 | inventory, closed contracts, schema, registries, trigger, atomic outbox, wait, subscriptions, backpressure | no event-driven claim until no-lost-wakeup and idle tests pass |
| 2 causal/semantic coordination | CASF-013..021 | causal graph/maps/interventions/frontier, snapshots, semantic/proof/index projection, wake/cursor | retrieval remains nomination-only |
| 3 parallel federation | CASF-022..030 | dedup/frontier, budgets, sharding/steal/rebalance, merge, recovery, fixed point | no high concurrency until owner, outbox, fencing, budgets, frontier, recovery qualify |
| 4 history/product/qualification | CASF-031..043 | DuckLake, drift, controls, formal/chaos suites, benchmarks, gates, final report | no unsupported readiness or efficiency claim |

The protected scheduler projection records all 44 tasks as `todo` and makes
CASF-000 the sole dependency-ready task. Existing source and test observations
do not substitute for canonical current-tree producer receipts, so no pending
final-result identity is materialized as completed. Bootstrap capacity remains
one bounded coordinator and one registered logical subagent.
Scaling to 12 supervisors and 64 agent slots is a later authorized transition
after the non-compensable high-concurrency gate. Blocked tasks emit typed
blockers while independent DAG branches continue.

## 16. Required artifacts and execution discipline

This campaign creates the plan/objectives/task board, current-tree inventory,
board validator, frozen benchmark directory, and scheduler config requested by
the program. Large AST graphs, traces, vector/proof corpora, DuckLake files, and
load results remain managed content-addressed artifacts; Git contains compact
manifests, schemas, fixtures, and references only.

Run focused tests after every bounded task. Keep task/commit ownership narrow,
protect board controls from workers, never modify sibling repositories, never
weaken tests/proofs/policy/authorization, never store private chain-of-thought,
and requalify the actual final merged tree. Rebase/merge current main often
enough to avoid an unreviewable campaign, but never treat conflict-free Git as
semantic acceptance.

## 17. Final qualification report

CASF-043 produces machine and human reports binding starting/ending commit,
control generation/schema fingerprint, Quack/DuckLake capability and state,
federation/supervisor/agent populations and concurrency, task/dedup/causal/
abstraction/intervention/event/dead-letter/wakeup/idle/throughput/merge/model/
token/context/proof/validation/projection results, recovery/failures, executed
and unexecuted tests/models, safety gates, blockers, qualification/promotion,
and rollback target.

The report may claim `event driven`, `causally coordinated`,
`multi-supervisor`, `parallel`, `token efficient`, or `production ready` only
when exact current-tree evidence proves that specific property. Task-board
status, process exit, quiet queues, historical receipts, model statements, and
DuckLake projections are never sufficient.
