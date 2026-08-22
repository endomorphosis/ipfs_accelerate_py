# Causal Event Federation Authority Inventory

This directory seals the CASF-000/CASF-001 starting baseline for
agent-supervisor-causal-event-federation-v1. It is an inventory of the
committed current tree and the current local extension probes. It is not a
task-board completion claim, a policy decision, or a promotion receipt.

## Baseline

| Field | Value |
|---|---|
| Repository | endomorphosis/ipfs_accelerate_py |
| Branch | codex/causal-event-supervisor-federation-v1 |
| Starting commit | 84a056e41e48a81d4484be43840196578d6c87da |
| Starting tree | 40f0771e77d394ac91d92cc1edb02f7860f6131b |
| Program | agent-supervisor-causal-event-federation-v1 |
| Root objective | CASF-G000 |
| Inventory tasks | CASF-000, CASF-001 |

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

## Current capability snapshot

| Capability | Inventory result | Qualification boundary |
|---|---|---|
| DuckDB | available; version 1.5.5 | Runtime presence is not federation qualification. |
| Quack | available_with_caveats; core extension c154811 loaded; quack_serve and quack_query present; compatible and health check passed | experimental_usable is false and the declared beta limitation is no_server_push_clients_must_poll. It does not qualify the required event-wait gate. |
| DuckLake | available_with_caveats; core extension d8a1881e loaded | Extension load is not a typed, idempotent projection pipeline or a promotion receipt. |
| httpfs | available_with_caveats; core extension 827222f loaded | Transport capability does not grant scheduling or policy authority. |
| Python ducklake package | missing | This is not a DuckDB-extension blocker, but no standalone-package behavior is claimed. |

Network installation was disabled during the probes. Full probe facts and
nonclaims are recorded in capability_snapshot.json.

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
| integrations.ducklake_history_projection | stale | Replace the four-line non-authoritative placeholder with a typed projection pipeline. |
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
  the current history projection is a placeholder without typed catalog,
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
