# Causal Event Federation Authority Inventory

This directory seals the CASF-000/CASF-001 starting baseline for
agent-supervisor-causal-event-federation-v1. It is an inventory of the
committed current tree and the current local extension probes. It is not a
task-board completion claim, a policy decision, or a promotion receipt.

CASF-000 owns the starting-tree seal and capability snapshot. CASF-001 owns
the named-authority classification. Both tasks remain evidence-only: they
extend current canonical authorities narrowly, fail closed on missing
capability, and preserve every non-compensable constraint.

## Baseline

| Field | Value |
|---|---|
| Repository | endomorphosis/ipfs_accelerate_py |
| Branch | codex/causal-event-supervisor-federation-v1 |
| Starting commit | 84a056e41e48a81d4484be43840196578d6c87da |
| Starting tree | 40f0771e77d394ac91d92cc1edb02f7860f6131b |
| Rollback target | 84a056e41e48a81d4484be43840196578d6c87da |
| Plan revision | CASF-PLAN-R1 |
| Program | agent-supervisor-causal-event-federation-v1 |
| Root objective | CASF-G000 |
| Inventory tasks | CASF-000, CASF-001 |
| Authority class | evidence-only |

The exact machine-readable baseline is in starting_tree.json. Concurrent
implementation work may make the worktree dirty after this committed tree was
sealed; that does not change the starting commit or tree identity. Later
overlay paths are not this baseline.

## Validation environment

Capability and validation assertions are bound to the sealed environment, not
a provider-side PATH or operator toolchain:

| Field | Value |
|---|---|
| PATH | `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin` |
| Canonical interpreter | `/usr/bin/python3.12` |
| HOME | fresh private directory named `ipfs-accelerate-validation-home-*` |
| XDG paths | `$HOME/.cache`, `$HOME/.config`, `$HOME/.local/share`, `$HOME/.local/state` |
| Network | deny |
| Extension INSTALL | disabled |

## Closed status vocabulary

| Status | Meaning |
|---|---|
| available | Present and suitable as a current canonical primitive for its declared scope. |
| available_with_caveats | Present and reusable, but incomplete, limited, or requiring a compatibility extension for CASF. |
| stale | Present, but historical, placeholder-only, or not current-tree authority. |
| incompatible | Present, but unsafe as the canonical CASF path because its behavior conflicts with a non-compensable constraint. |
| missing | No qualifying current-tree implementation was found. |

Unknown, unprobed, or unnamed statuses fail closed. They are never rewritten
as available.

## Closed contracts

- Status vocabulary is closed. No extra labels are admitted.
- The sealed starting commit/tree identities are exact and cannot be replaced
  by a dirty overlay, branch name, or planning document.
- This inventory is compact evidence. It cannot create operational authority,
  policy outcomes, leases, fences, completion, or promotion.
- Current canonical owners (store identity, migrations, schema, repository,
  transactions, control catalog) are extended later, not replaced here.
- Missing capability emits a typed blocker and allows independent work. It is
  never classified available.
- Authoritative mutation remains behind one exclusive Quack state-owner
  boundary. Embedded mode cannot claim multi-supervisor parallelism.
- DuckLake remains optional, append-only, rebuildable, eventually consistent,
  and never a scheduling, lease, policy, acceptance, or completion
  prerequisite.

## Fail-closed negative paths

| Negative path | Disposition |
|---|---|
| Status outside the closed vocabulary | reject |
| Capability claimed available without source identity or probe | reject |
| Missing prerequisite reported as available | typed blocker; reject the available claim |
| Module name, import, fixture, table name, Markdown board, generated report, or historical receipt used as live authority | reject |
| Later overlay treated as the sealed starting tree | reject |
| DuckLake or httpfs LOAD used as scheduling, lease, policy, or completion authority | reject |
| Quack compatibility or health used as event-driven or multi-supervisor qualification | reject |
| Direct multi-process `control.duckdb` mutation or implicit Quack-to-file fallback | reject |
| Agent-supplied SQL, database path, endpoint, or raw credentials | reject |
| Process exit, board status, quiet queue, or this inventory used as completion | reject; final result identity remains pending |
| Weakening a non-compensable constraint | reject |
| Write to sibling `ipfs_datasets_py`, `ipfs_kit_py`, or MCP++ sources | reject |
| Network `INSTALL` during an ordinary probe | reject |
| `import duckdb` treated as a health-check pass | reject |
| Starting-tree DuckLake placeholder treated as a typed projection pipeline | reject; status remains stale |
| Client polling treated as server-owned `wait_for_events` | reject |

## Non-compensable constraints

These invariants are preserved by this baseline and cannot be traded away:

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

## Current capability snapshot

| Capability | Inventory result | Qualification boundary |
|---|---|---|
| Python | available; canonical `/usr/bin/python3.12` | Interpreter presence is not federation qualification. |
| DuckDB | available; version 1.5.5; pin `>=1.5.0,<1.6.0` | Runtime presence is not federation qualification. Direct multi-process file mutation is prohibited. |
| Quack | available_with_caveats; core extension c154811 loaded; quack_serve and quack_query present; compatible and health check passed | experimental_usable is false and the declared beta limitation includes no_server_push_clients_must_poll. It does not qualify the required event-wait gate. |
| DuckLake | available_with_caveats; core extension d8a1881e loaded | Extension load is not a typed, idempotent projection pipeline or a promotion receipt. |
| httpfs | available_with_caveats; core extension 827222f loaded | Transport capability does not grant scheduling or policy authority. |
| Python ducklake package | missing | This is not a DuckDB-extension blocker, but no standalone-package behavior is claimed. |
| DuckLake projection pipeline | stale | Starting-tree adapter is a non-authoritative placeholder. |
| Event-driven wakeup | missing | No starting-tree server-owned `wait_for_events` path. |
| Qualified parallel federation | missing | Runner polls and the configured live-seal gate is NO-GO. |

Network installation was disabled during the probes. Full probe facts, closed
contracts, and explicit nonclaims are recorded in capability_snapshot.json.

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
