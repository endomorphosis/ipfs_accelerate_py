# Causal Event Federation Authority Inventory

This directory seals the CASF-000/CASF-001 starting baseline for
agent-supervisor-causal-event-federation-v1. It is an inventory of the
committed current tree and the current local extension probes. It is not a
task-board completion claim, a policy decision, or a promotion receipt.

CASF-000 seals the exact commit/tree and capability probes. CASF-001 inventories
DuckDB, Quack, DuckLake, runner, event, and causal surfaces against current
source and test identities, extends those canonical authorities narrowly, and
fails closed on missing capability. Current-tree implementation is not
qualification.

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
sealed; that does not change the starting commit or tree identity. CASF-001
records current-tree source identities for the six surfaces without promoting
those identities to completion.

## Closed status vocabulary

| Status | Meaning |
|---|---|
| available | Present and suitable as a current canonical primitive for its declared scope. |
| available_with_caveats | Present and reusable, but incomplete, limited, or requiring a compatibility extension for CASF. |
| stale | Present, but historical, placeholder-only, or not current-tree authority. |
| incompatible | Present, but unsafe as the canonical CASF path because its behavior conflicts with a non-compensable constraint. |
| missing | No qualifying current-tree implementation was found. |

Names, imports, Markdown state, generated reports, historical receipts,
fixtures, embedded tests, Quack imports, DuckLake imports, and similarly named
tables are never sufficient authority evidence.

## Current capability snapshot

| Capability | Inventory result | Qualification boundary |
|---|---|---|
| DuckDB | available; version 1.5.5; pin duckdb>=1.5.0,<1.6.0 | Runtime presence is not federation qualification. Direct multi-process file mutation is prohibited. |
| Quack | available_with_caveats; core extension c154811 loaded; quack_serve and quack_query present; compatible and health check passed | experimental_usable is false and the declared beta limitation is no_server_push_clients_must_poll. A server-owned typed wait exists as a bootstrap capability. It does not qualify federation-wide event-driven execution. |
| DuckLake | available_with_caveats; core extension d8a1881e loaded | Extension load is not a promotion receipt. A typed non-authoritative projection worker exists; DuckLake-specific gates remain open. |
| httpfs | available_with_caveats; core extension 827222f loaded | Transport capability does not grant scheduling or policy authority. |
| Python ducklake package | missing | This is not a DuckDB-extension blocker, but no standalone-package behavior is claimed. |

Network installation was disabled during the probes. Full probe facts and
nonclaims are recorded in capability_snapshot.json.

## CASF-001 surface inventory

### DuckDB

DuckDB remains the exclusive authoritative transactional store. Canonical
primitives are `control_plane_schema`, the 0001/0002/0003 SQL catalog,
`duckdb_state.open_duckdb_connection`, and the repository/transaction boundary.
ControlPlaneSchema@1 still means the 110-table 0001 profile.
CausalEventFederationSchemaExtension@1 adds 109 federation tables at schema
revision 2. Embedded or exclusive-owner file opens are policy-limited;
multi-process workers must speak Quack. Runtime DDL and agent SQL fail closed.

### Quack

Quack remains the exclusive live state-owner transport. `quack_capabilities`
still records the polling beta limitation. `QuackStateServer` owns the DuckDB
file, lease, and `wait_for_events` condition. `QuackStateClient` and
`TypedStateOwnerGateway` reject arbitrary SQL and database paths. Adaptive
long-poll is an unqualified compatibility fallback. Multi-supervisor
availability fails closed when Quack is unavailable or incompatible.

### DuckLake

DuckLake is optional, append-only, rebuildable, eventually consistent, and
non-authoritative. `DuckLakeProjectionWorker` binds event ranges, partitions,
checksums, and cursors and hard-codes `projection_establishes_authority()` and
`projection_establishes_completion()` as false. The integrations adapter
observes receipts only. Missing or lagging DuckLake cannot block DuckDB/Quack
qualification and cannot schedule, lease, or complete work.

### Runner

`runtime.multi_supervisor_runner` remains a bounded process/track helper whose
coordinator polls and whose configured live-seal gate is NO-GO.
`federation.supervisor_runtime` and `bootstrap_runtime` exist as current-tree
modules. They do not establish 12-supervisor, parallel, or production
qualification. High concurrency stays closed.

### Event

Closed event/outbox/subscription/cursor contracts, `transactional_outbox`,
`materialize_event`, and `StateOwnerEventWait` exist. Network delivery is
at-least-once; authoritative effects are exactly-once only through idempotency,
CAS, leases, and fencing. `DatabaseEventLog` is incompatible as the canonical
CASF event authority because it opens a DuckDB file and applies runtime DDL.
Federation event-driven execution remains unqualified until CASF-021.

### Causal

`CausalGraphStore`, abstraction maps, frontiers, and
`FederationWorldSnapshot` exist over the sealed Quack catalog. Doctor
localization remains report-only and maps kinds without granting authority.
Retrieval and model output may only nominate. Unknown dependency widens the
frontier. The inventory does not claim causal coordination.

## Named authority disposition

| Named authority | Status | CASF disposition |
|---|---|---|
| task_sources.control_plane_contracts | available | Reuse closed store identities, generations, commands, snapshots, and export receipts. |
| task_sources.control_plane_migrations | available | Extend the existing migration catalog and runner; 0002/0003 are additive. |
| task_sources.control_plane_schema | available_with_caveats | Reuse ControlPlaneSchema@1; federation populations are the 0002 extension. Live fingerprint remains pending. |
| task_sources.control_plane_repository | available_with_caveats | Reuse the repository boundary and explicit Quack/no-fallback selection; federation operations stay on this store. |
| task_sources.control_plane_transactions | available_with_caveats | Reuse transaction, CAS, and idempotency primitives; mutation/event/outbox share one generation. |
| task_sources.duckdb_state | available_with_caveats | Connection policy for exclusive owner/embedded/recovery only. |
| task_sources.quack_capabilities | available_with_caveats | Current profile is compatible but explicitly polling-limited. |
| task_sources.quack_state_client | available_with_caveats | Reuse registered statements and raw-SQL rejection; owner wait is preferred over adaptive poll. |
| task_sources.typed_state_owner | available_with_caveats | Authenticated owner catalog and wait boundary; not remote event-driven qualification. |
| runtime.quack_state_server | available_with_caveats | Reuse exclusive-owner machinery and the server-owned wait path. |
| runtime.multi_supervisor_runner | available_with_caveats | Reuse bounded process-management pieces only; the current coordinator polls and its live seal is NO-GO. |
| runtime.database_event_log | incompatible | Direct-file open plus runtime DDL; not the CASF event authority. |
| semantic_state.world_snapshot_builder | available_with_caveats | Reuse observed state inputs; FederationWorldSnapshot is the federation assembler. |
| analysis.doctor_causal_localization | available_with_caveats | Reuse report-only evidence and nomination separation; do not promote it to causal authority. |
| integrations.ducklake_history_projection | available_with_caveats | Observational adapter over the typed non-authoritative worker. |
| agent_supervisor.federation | available_with_caveats | Current-tree CausalAbstractionSupervisorFederation@1 package. Implementation is not completion. |
| federation.events / outbox / event_wait | available_with_caveats | Closed contracts and owner wait exist; federation event-driven execution is unqualified. |
| federation.causal_graph / frontier / world_snapshot | available_with_caveats | Graph primitives exist; causal coordination is not claimed. |
| federation.ducklake_projection | available_with_caveats | Typed non-authoritative worker; promotion remains closed. |
| federation.lifecycle | available_with_caveats | CASF closed state machine. Control-plane SupervisorLifecycleState is a separate incompatible vocabulary. |
| agent_supervisor.control | available_with_caveats | Extend the canonical control service and operation catalog; do not create a second control plane. |
| agent_supervisor.runtime | available_with_caveats | Reuse schedulers, provider queues, CAS, and bounded workers subject to the state-owner boundary. |
| agent_supervisor.planning | available_with_caveats | Reuse plan/frontier primitives; model independence remains nomination-only. |
| agent_supervisor.proof | available_with_caveats | Reuse proof contracts/cache semantics; prevent direct multi-process control-database mutation. |
| agent_supervisor.verification | available_with_caveats | Reuse verification planning and receipts under current-tree evidence rules. |
| agent_supervisor.semantic_governor | available_with_caveats | Reuse operational governance while retaining ipfs_datasets_py semantic ownership. |
| agent_supervisor.adversarial_assurance | available_with_caveats | Reuse campaign and worker primitives; evidence remains non-promotional until admitted. |
| AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md | available_with_caveats | Current architectural input; it documents polling and one-writer constraints. |
| agent_supervisor_duckdb_quack_control_plane.todo.md | stale | Historical completed board; Markdown state is not completion evidence. |
| LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_PLAN.md | stale | Historical program plan and useful gap record, not CASF authority. |
| LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_QUALIFICATION.md | stale | Exact-tree historical research-demo result, not current-tree qualification. |
| AGENT_SUPERVISOR_ARCHITECTURE.md | available_with_caveats | Useful implementation map with explicit caveats; it predates the CASF surface. |

authorities.json contains exact paths, key symbols, test identities, related
incompatible surfaces, and remaining missing qualification surfaces.

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

- CASF-BLOCKER-FEDERATION-SURFACE-MISSING: the federation package now exists;
  live admission, producer receipts, and promotion remain pending.
- CASF-BLOCKER-OUTBOX-MISSING: transactional_outbox and materialize_event exist;
  live drain/replay receipts remain pending.
- CASF-BLOCKER-EVENT-WAIT-MISSING: server-owned wait_for_events exists as a
  bootstrap typed-wait capability; federation event-driven execution remains
  unqualified until CASF-021.
- CASF-BLOCKER-QUACK-EVENT-QUALIFICATION: Quack loads and passes its health
  probe, but event-driven multi-supervisor qualification is not established.
- CASF-BLOCKER-LIFECYCLE-VOCABULARY: FederationLifecycleState matches the CASF
  machine; control-plane SupervisorLifecycleState remains a separate
  incompatible vocabulary.
- CASF-BLOCKER-DUCKLAKE-PROJECTION-MISSING: a typed non-authoritative worker
  exists; DuckLake promotion gates remain open.
- CASF-BLOCKER-MULTI-SUPERVISOR-QUALIFICATION: the runner polls, its configured
  live-seal gate is NO-GO, and there is no current-tree 12-supervisor evidence.
- CASF-BLOCKER-SCHEMA-COVERAGE: 0002 adds the required populations; live
  fingerprint and producer receipts remain pending.
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
