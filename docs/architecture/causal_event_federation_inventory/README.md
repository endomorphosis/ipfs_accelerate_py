# Causal Event Federation Authority Inventory

This directory seals the CASF-000/CASF-001 starting baseline for
agent-supervisor-causal-event-federation-v1. It is an inventory of the
committed current tree and the current local extension probes. It is not a
task-board completion claim, a policy decision, or a promotion receipt.

CASF-000 seals the exact commit/tree, migrations, operation catalog, state
owner, sibling contracts, non-compensable constraints, and fail-closed
missing-capability policy. CASF-001 classifies the mandated DuckDB, Quack,
DuckLake, runner, event, and causal surfaces. Later worktree files after the
starting commit are not this baseline.

## Baseline

| Field | Value |
|---|---|
| Repository | endomorphosis/ipfs_accelerate_py |
| Branch | codex/causal-event-supervisor-federation-v1 |
| Starting commit | 84a056e41e48a81d4484be43840196578d6c87da |
| Starting tree | 40f0771e77d394ac91d92cc1edb02f7860f6131b |
| Rollback target | 84a056e41e48a81d4484be43840196578d6c87da |
| Program | agent-supervisor-causal-event-federation-v1 |
| Root objective | CASF-G000 |
| Plan revision | CASF-PLAN-R1 |
| Package | ipfs_accelerate_py 0.0.45 |
| Canonical interpreter | /usr/bin/python3.12 |
| Validation PATH | /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin |
| Inventory tasks | CASF-000, CASF-001 |
| Authority class | evidence-only; no operational authority |

The exact machine-readable baseline is in starting_tree.json. Concurrent
implementation work may make the worktree dirty after this committed tree was
sealed; that does not change the starting commit or tree identity.

## Closed status vocabulary

| Status | Meaning |
|---|---|
| available | Present and suitable as a current canonical primitive for its declared scope. |
| available_with_caveats | Present and reusable, but incomplete, limited, or requiring a compatibility extension for CASF. |
| stale | Present, but historical, placeholder-only, or not current-tree authority. |
| incompatible | Present, but unsafe as the canonical CASF path because its behavior conflicts with a non-compensable constraint. |
| missing | No qualifying current-tree implementation was found. |

Missing functionality is never reported as available. A compatibility adapter
is used only when its semantics and effect ceiling are exact; otherwise the
gap is a typed blocker and independent work continues.

## Fail-closed missing capability

| Missing or incomplete surface | Disposition |
|---|---|
| DuckDB import/connect | Unavailable; no federation or multi-supervisor claim. |
| Quack missing, incompatible, or unhealthy | Multi-supervisor availability fails closed. No implicit embedded/file fallback. |
| Quack compatible but polling-limited | available_with_caveats. Event-driven qualification remains closed. |
| DuckLake extension or projection | Typed unavailable/lagging. Never blocks DuckDB/Quack scheduling or core completion. |
| httpfs | Transport only. Never scheduling, lease, completion, or policy authority. |
| Sibling published contract | typed_blocker. Sibling repositories remain read-only. |
| Missing or stale telemetry | Adds no capacity. |
| Federation package, contracts, outbox, or wait_for_events | Typed blocker. Independent contract/migration/inventory/hermetic-test work may continue. |

Network installation is disabled during ordinary probes. Names, imports,
Markdown state, generated reports, historical receipts, fixtures, embedded
tests, Quack imports, DuckLake imports, and similarly named tables are never
sufficient authority evidence.

## Non-compensable constraints

These constraints are not traded for throughput, convenience, or missing
capability:

- zero unauthorized supervisor, subagent, or mutation creation
- zero duplicate committed effects, stale-fence completion, or lost
  authoritative transitions
- zero simulated-as-live evidence, model-created authority, model-created
  policy permission, model-created completion, or false completion
- zero DuckLake-derived scheduling, lease, policy, or completion authority
- zero direct multi-process DuckDB file mutation and zero implicit Quack to
  embedded/file fallback
- zero arbitrary SQL from an agent, unbounded event fanout, hidden validation
  reduction, cross-tenant leakage, or raw credential propagation

Unknown normative fields, empty identities, nonfinite values, unbounded
strings/arrays, arbitrary paths/SQL, raw credentials, executable callbacks,
model-authored authority, and self-promoting definitions fail closed.

## Current capability snapshot

| Capability | Inventory result | Qualification boundary |
|---|---|---|
| DuckDB | available; version 1.5.5 | Runtime presence is not federation qualification. |
| Quack | available_with_caveats; core extension c154811 loaded; quack_serve and quack_query present; compatible and health check passed | experimental_usable is false. The pinned beta profile includes no_server_push_clients_must_poll. It does not qualify the required event-wait gate. |
| DuckLake | available_with_caveats; core extension d8a1881e loaded | Extension load is not a typed, idempotent projection pipeline or a promotion receipt. |
| httpfs | available_with_caveats; core extension 827222f loaded | Transport capability does not grant scheduling or policy authority. |
| Python ducklake package | missing | This is not a DuckDB-extension blocker, but no standalone-package behavior is claimed. |

Network installation was disabled during the probes. Full probe facts,
negative paths, and nonclaims are recorded in capability_snapshot.json.

## Control-plane and catalog baseline

| Surface | Starting-tree fact |
|---|---|
| Base migration | ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql version 1 |
| Normalized domain tables | 110 |
| Schema authority | ControlPlaneSchema@1 in control_plane_schema.py |
| Repository opener | explicit open_state_repository; Quack refuses embedded fallback |
| Operation catalog | control-operation-catalog@2, 35 operations, requirement 294719425747343997526263348545558645762 |
| federation.* operations | missing |
| Runtime DDL | prohibited |
| Later additive migrations | not this baseline |

## Named authority disposition

| Named authority | Status | CASF disposition |
|---|---|---|
| task_sources.control_plane_contracts | available | Reuse closed store identities, generations, commands, snapshots, and export receipts. |
| task_sources.control_plane_migrations | available | Extend the existing migration catalog and runner. |
| task_sources.control_plane_schema | available_with_caveats | Reuse normalized schema authority; add missing CASF populations through migrations. |
| task_sources.control_plane_repository | available_with_caveats | Reuse the repository boundary and explicit Quack/no-fallback selection; extend its typed operation surface. |
| task_sources.control_plane_transactions | available_with_caveats | Reuse transaction, CAS, and idempotency primitives; add atomic mutation/event/outbox semantics. |
| task_sources.quack_capabilities | available_with_caveats | Current profile is compatible but explicitly polling-limited. |
| task_sources.quack_state_client | available_with_caveats | Reuse registered statements and raw-SQL rejection; add typed event waiting and scoped federation calls. |
| runtime.quack_state_server | available_with_caveats | Reuse exclusive-owner machinery; add a no-lost-wakeup server-owned wait path. |
| runtime.multi_supervisor_runner | available_with_caveats | Reuse bounded process-management pieces only; the current coordinator polls and its live seal is NO-GO. |
| semantic_state.world_snapshot_builder | available_with_caveats | Reuse observed state inputs; it is not yet FederationWorldSnapshot. |
| analysis.doctor_causal_localization | available_with_caveats | Reuse report-only evidence and nomination separation; do not promote it to causal authority. |
| integrations.ducklake_history_projection | stale | Replace the starting-tree non-authoritative placeholder with a typed projection pipeline. |
| agent_supervisor.control | available_with_caveats | Extend the canonical control service and operation catalog; do not create a second control plane. |
| agent_supervisor.runtime | available_with_caveats | Reuse schedulers, provider queues, CAS, and bounded workers subject to the state-owner boundary. |
| agent_supervisor.planning | available_with_caveats | Reuse plan/frontier primitives; exact causal independence and federation assignment remain missing. |
| agent_supervisor.proof | available_with_caveats | Reuse proof contracts/cache semantics; prevent direct multi-process control-database mutation. |
| agent_supervisor.verification | available_with_caveats | Reuse verification planning and receipts under current-tree evidence rules. |
| agent_supervisor.semantic_governor | available_with_caveats | Reuse operational governance while retaining ipfs_datasets_py semantic ownership. |
| agent_supervisor.adversarial_assurance | available_with_caveats | Reuse campaign and worker primitives; evidence remains non-promotional until admitted. |
| AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md | available_with_caveats | Current architectural input; it documents polling and one-writer constraints. |
| agent_supervisor_duckdb_quack_control_plane.todo.md | stale | Historical completed board; Markdown state is not completion evidence. |
| LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_PLAN.md | stale | Historical program plan and useful gap record, not CASF authority. |
| LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_QUALIFICATION.md | stale | Exact-tree historical research-demo result, not current-tree qualification. |
| AGENT_SUPERVISOR_ARCHITECTURE.md | available_with_caveats | Useful implementation map with explicit caveats; it predates the CASF surface. |

authorities.json contains exact paths, key symbols, evidence, related
incompatible surfaces, and missing target surfaces.

## Canonical ownership retained

- ipfs_datasets_py owns semantic meaning and immutable semantic contracts.
  The accelerator may persist/query those identities but may not reinterpret
  them.
- ipfs_accelerate_py owns operational federation coordination.
- DuckDB owns authoritative transactional operational records behind one state
  owner.
- Quack owns the qualified multi-client transport and exclusive state-owner
  boundary. No implicit fallback to direct embedded file mutation is allowed.
- DuckLake is optional, append-only, rebuildable, eventually consistent, and
  non-authoritative.
- ipfs_kit_py artifact/VFS/proof-seal/WAL/current-pointer interfaces are reused,
  not duplicated.
- Existing MCP++ wire profiles are reused when applicable; this program does
  not create a new profile.

Declared gitlinks remain sibling pins. CASF-000 does not initialize or rewrite
them. Missing sibling capability is a typed blocker.

## Typed blockers

- CASF-BLOCKER-FEDERATION-SURFACE-MISSING: no qualifying federation package,
  contracts, registry, gateway, or CausalAbstractionSupervisorFederation exists
  at the starting tree.
- CASF-BLOCKER-OUTBOX-MISSING: no normalized transactional_outbox and no atomic
  mutation plus domain-event plus outbox helper exists.
- CASF-BLOCKER-EVENT-WAIT-MISSING: the state owner has no server-owned
  wait_for_events/no-lost-wakeup path; the current Quack profile says clients
  must poll.
- CASF-BLOCKER-QUACK-EVENT-QUALIFICATION: Quack loads and passes its health
  probe, but event-driven multi-supervisor qualification is not established.
- CASF-BLOCKER-LIFECYCLE-VOCABULARY: the existing lifecycle vocabulary differs
  from the required versioned CASF closed state machine.
- CASF-BLOCKER-DUCKLAKE-PROJECTION-MISSING: extension load is available, but
  the starting-tree history projection is a placeholder without typed catalog,
  cursor, source range, recovery, or receipt.
- CASF-BLOCKER-MULTI-SUPERVISOR-QUALIFICATION: the runner polls, its configured
  live-seal gate is NO-GO, and there is no current-tree 12-supervisor evidence.
- CASF-BLOCKER-SCHEMA-COVERAGE: the normalized base schema lacks the required
  federation, subagent, shard, causal, retrieval, outbox, subscription, cursor,
  and projection populations.
- CASF-BLOCKER-CURRENT-TREE-QUALIFICATION: historical receipts and Markdown
  boards cannot qualify this starting tree.

These blockers do not prevent independent contract, migration, inventory, or
hermetic-test work. They do prevent the affected capability and promotion
claims.

## Explicit nonclaims

This inventory does not claim that the starting tree is event driven, causally
coordinated, multi-supervisor qualified, parallel qualified, token efficient,
production ready, exactly-once-delivery capable, DuckLake-promotion qualified,
or Quack event-wait qualified. It also does not infer authority from an import,
module name, report, fixture, task-board state, table name, or historical
receipt.
