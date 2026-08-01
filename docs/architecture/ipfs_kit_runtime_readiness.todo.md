# IPFS Kit Runtime Readiness Taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `KITA-`.

Companion artifacts:

- plan: `docs/architecture/IPFS_KIT_RUNTIME_READINESS_PLAN.md`
- objective heap: `docs/architecture/ipfs_kit_runtime_readiness.objectives.md`
- scheduler: `config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json`
- validator: `scripts/validate_ipfs_kit_runtime_readiness_board.py`

This is a sealed first projection with 48 append-only tasks (`KITA-000`
through `KITA-047`) over 12 goals. `KITA-000` is the completed planning/control
seal. Exactly `KITA-001` through `KITA-004` are initially dependency-ready and
map one-to-one to four strict shards.

Implementation tasks may change the nested `ipfs_kit_py` repository only from
isolated initialized worktrees. Each task must land a reviewed nested commit
and then advance the parent gitlink through the serialized merge queue. A
task that also changes `ipfs_datasets_py` must use the same two-commit
protocol. Advisory retrieval, tests, solvers, or models never confer semantic
or write authority.

## KITA-000 Seal the runtime-readiness control program

- Status: completed
- Completion: manual
- Completion evidence: plan, objective heap, taskboard, scheduler, validator, parser test, and JSON allowlist present and validated on agent/ipfs-kit-runtime-readiness
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: control
- Depends on:
- Goal id: KITA-G000
- Outputs: .gitignore, docs/architecture/IPFS_KIT_RUNTIME_READINESS_PLAN.md, docs/architecture/ipfs_kit_runtime_readiness.objectives.md, docs/architecture/ipfs_kit_runtime_readiness.todo.md, config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json, scripts/validate_ipfs_kit_runtime_readiness_board.py, test/api/test_ipfs_kit_runtime_readiness_board.py
- Validation: python scripts/validate_ipfs_kit_runtime_readiness_board.py --check-all && python -m pytest -q test/api/test_ipfs_kit_runtime_readiness_board.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/control
- Parallel lane: control
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 24000
- Implementation timeout seconds: 1800
- Predicted files: .gitignore, docs/architecture/IPFS_KIT_RUNTIME_READINESS_PLAN.md, docs/architecture/ipfs_kit_runtime_readiness.objectives.md, docs/architecture/ipfs_kit_runtime_readiness.todo.md, config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json, scripts/validate_ipfs_kit_runtime_readiness_board.py, test/api/test_ipfs_kit_runtime_readiness_board.py
- Interfaces: IPFSKitRuntimeReadinessPlan@1
- Allow concurrent with:
- Conflict policy: These seven normative artifacts are protected after this task; later tasks emit runtime evidence elsewhere and may not rewrite task identities, dependencies, or acceptance.
- Preconditions: The accelerator branch is isolated and the bound ipfs_kit_py and ipfs_datasets_py gitlinks are known.
- Effects: A parseable, acyclic, four-lane, bounded and fail-closed implementation program is available.
- Evidence subset: architecture, goals, tasks, dependencies, ownership, scheduler, validation, release gates
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Validator proves 48 exact unique tasks, 12 exact unique goals, all local references, acyclicity, protected ownership, scheduler/config consistency, and exactly KITA-001 through KITA-004 ready after this completion; the production parser consumes all tasks and normalized task identities are stable; scheduler JSON is not hidden by the repository JSON ignore policy; no later task owns a protected control artifact.
- Embedding query: seal ipfs kit runtime readiness supervisor goals tasks parallel scheduler

## KITA-001 Freeze the repository, capability, backend, and test-gate inventory

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundations-inventory
- Depends on: KITA-000
- Goal id: KITA-G010
- Outputs: ipfs_kit_py/docs/runtime_readiness/capability_manifest.json, ipfs_kit_py/docs/runtime_readiness/surface_inventory.md, ipfs_kit_py/tests/runtime_readiness/foundations/test_capability_manifest.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/foundations/test_capability_manifest.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/foundations/inventory
- Parallel lane: kita-inventory
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/docs/runtime_readiness/capability_manifest.json, ipfs_kit_py/docs/runtime_readiness/surface_inventory.md, ipfs_kit_py/tests/runtime_readiness/foundations/test_capability_manifest.py
- Interfaces: CapabilityManifest@1, RepositoryForestDescriptor@1, BackendSupportTier@1
- Allow concurrent with: KITA-002, KITA-003, KITA-004
- Conflict policy: Own only new inventory/manifest/test files; read existing implementations and manifests without refactoring them.
- Preconditions: Exact recursively initialized repository descriptors and dirty-overlay policies are available.
- Effects: Every duplicate implementation, interface, backend name/alias/schema/factory, dependency, test exclusion, daemon/provider, and confirmed defect has a content-bound disposition.
- Evidence subset: git tree, public exports, CLI/MCP registries, VFS/bucket/WAL/ARC/replica/GraphRAG variants, 23 backend types, pytest collection
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Inventory is exhaustive under a checked-in policy; records the confirmed VFS no-op rename/journal mismatch, overlapping bucket planes, WAL defects, ARC accounting/concurrency defects, shadowed replica methods, backend registry/factory fracture, MCP++ construction failure, GraphRAG persistence/safety drift, lazy-import/dependency/version drift, and default-test exclusions; every advertised item is assigned production, conditional, configuration-only, experimental, unsupported, or unknown-pending-proof without inferring correctness from presence.
- Embedding query: ipfs kit capability inventory backend registry duplicate implementation test gate

## KITA-002 Define canonical operation, result, error, state, and evidence contracts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundations-contracts
- Depends on: KITA-000
- Goal id: KITA-G010
- Outputs: ipfs_kit_py/ipfs_kit_py/core/operation_contracts.py, ipfs_kit_py/tests/runtime_readiness/foundations/test_operation_contracts.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/foundations/test_operation_contracts.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/foundations/contracts
- Parallel lane: kita-contracts
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/operation_contracts.py, ipfs_kit_py/tests/runtime_readiness/foundations/test_operation_contracts.py
- Interfaces: OperationRequest@1, OperationResult@1, StorageError@1, StateTransitionReceipt@1
- Allow concurrent with: KITA-001, KITA-003, KITA-004
- Conflict policy: Own one new inert contract module/test; do not cut over callers or import optional storage providers.
- Preconditions: Existing result/error/signature/schema variants are available as observations from KITA-001 or a fresh local scan.
- Effects: All later subsystems exchange finite canonical records with explicit acknowledgement, durability, consistency, authorization, retry, partial-effect and content/version semantics.
- Evidence subset: schemas, canonical serialization, CIDs, states, errors, deadlines, cancellation, idempotency, policy, backend capability
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Records distinguish accepted/queued/committed/verified/converged and every failure/partial-effect state; bind request/idempotency/principal/policy/backend/WAL/cache/index/replica/environment identities as applicable; reject secrets, bodies, cycles, non-finite/unbounded fields, forged IDs, inconsistent states, and success without required effect/durability evidence; type/resource/memory facets remain distinct.
- Embedding query: canonical storage operation request result error receipt durability authorization

## KITA-003 Build hermetic adversarial state-machine and fault fixtures

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundations-fixtures
- Depends on: KITA-000
- Goal id: KITA-G010
- Outputs: ipfs_kit_py/tests/runtime_readiness/fixtures, ipfs_kit_py/tests/runtime_readiness/foundations/test_fixture_manifest.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/foundations/test_fixture_manifest.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/foundations/fixtures
- Parallel lane: kita-fixtures
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/tests/runtime_readiness/fixtures, ipfs_kit_py/tests/runtime_readiness/foundations/test_fixture_manifest.py
- Interfaces: RuntimeReadinessFixture@1, FaultSchedule@1, ExpectedStateTrace@1
- Allow concurrent with: KITA-001, KITA-002, KITA-004
- Conflict policy: Own only new hermetic fixtures and manifest tests; no live credentials, network, user paths, executable untrusted payloads, or production logic.
- Preconditions: Confirmed baseline defects and the canonical record vocabulary are available by reference.
- Effects: VFS, bucket, WAL, ARC, GraphRAG, replica, authorization, interface, backend, crash, corruption and resource work share exact positive and adversarial expectations.
- Evidence subset: state-machine traces, crash points, torn records, corrupt caches, invalid tokens, path escapes, backend failures, concurrency, cancellation
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Fixtures cover every confirmed blocker plus rename/move, multi-store bucket saga, commit/replay/checkpoint, ARC growth/ghost hits, index restart/history, replica drift, absent/forged/revoked/replayed UCAN, interface error drift, missing optional extras, backend retry/partial effects, resource exhaustion and nondeterministic ordering; expected traces and faults are finite, safe and content identified.
- Embedding query: adversarial storage fixture state machine crash fault corruption concurrency authorization

## KITA-004 Establish install, cold-import, workload, TPS, and resource baselines

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundations-baseline
- Depends on: KITA-000
- Goal id: KITA-G010
- Outputs: ipfs_kit_py/benchmarks/runtime_readiness/baseline.py, ipfs_kit_py/benchmarks/runtime_readiness/workloads.json, ipfs_kit_py/benchmarks/runtime_readiness/reference_floors.json, ipfs_kit_py/tests/runtime_readiness/foundations/test_install_import_baseline.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/foundations/test_install_import_baseline.py && python benchmarks/runtime_readiness/baseline.py --profile ci-reference --check-schema
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/foundations/baseline
- Parallel lane: kita-baseline
- Resource class: cpu-io-large
- Resource stage: benchmark
- Estimated tokens: 24000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/benchmarks/runtime_readiness/baseline.py, ipfs_kit_py/benchmarks/runtime_readiness/workloads.json, ipfs_kit_py/benchmarks/runtime_readiness/reference_floors.json, ipfs_kit_py/tests/runtime_readiness/foundations/test_install_import_baseline.py
- Interfaces: RuntimeBenchmarkManifest@1, WorkloadProfile@1, ImportTrace@1
- Allow concurrent with: KITA-001, KITA-002, KITA-003
- Conflict policy: Own new benchmark/baseline files only; measure the bound revision without optimizing production paths or lowering durability.
- Preconditions: A clean reproducible environment can build the minimal wheel and collect platform/resource identity.
- Effects: Initial import/module-action traces and transaction throughput/latency/resource distributions are pinned for honest later comparison.
- Evidence subset: minimal/core/extras wheels, version/dependency drift, cold imports, metadata/small-object/mixed-VFS/WAL/ARC/GraphRAG/replica/interface workloads
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Baseline records hardware/OS/Python/dependencies/revision/dataset/seed/concurrency/durability/warmup/samples/confidence; captures current root and MCP eager imports, runtime/metadata version mismatch, dependency projection drift and no transaction-specific SLO; measures committed not merely accepted TPS; distinguishes cold/warm/cache paths; defines immutable comparison rules and leaves absolute floors explicitly provisional until reviewed.
- Embedding query: ipfs kit baseline cold import dependency wheel transaction tps latency resource

## KITA-005 Specify VFS namespace, path, mount, and operation semantics

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vfs-contract
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G020
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/contracts.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_contracts.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/vfs/test_vfs_contracts.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/vfs/contracts
- Parallel lane: kita-vfs-contract
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/vfs/contracts.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_contracts.py
- Interfaces: VFSPathPolicy@1, VFSOperation@1, VFSStat@1, VFSMount@1
- Allow concurrent with: KITA-006, KITA-010, KITA-014, KITA-018, KITA-022, KITA-026, KITA-030, KITA-034, KITA-038
- Conflict policy: Own new VFS contract/test files; existing managers remain read-only observations.
- Preconditions: Capability inventory and canonical operation records exist.
- Effects: Every supported path, stat, list, stream/range, create, replace, rename/move, delete, mount and cross-boundary disposition is explicit.
- Evidence subset: normalization, Unicode/case, traversal, symlink, pagination/order, atomicity, CAS, errors, effects
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Contract rejects absolute/traversing/escaping paths under configured roots, defines Unicode/case and symlink policy, stable listing/stat/error semantics, atomic boundary and typed unsupported cases, and makes success contingent on an observed state transition.
- Embedding query: vfs namespace path mount symlink rename list stat operation contract

## KITA-006 Build the canonical VFS service and reference model

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vfs-service
- Depends on: KITA-002, KITA-003
- Goal id: KITA-G020
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/service.py, ipfs_kit_py/tests/runtime_readiness/vfs/reference_model.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_state_machine.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/vfs/test_vfs_state_machine.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/vfs/service
- Parallel lane: kita-vfs-service
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/vfs/service.py, ipfs_kit_py/tests/runtime_readiness/vfs/reference_model.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_state_machine.py
- Interfaces: CanonicalVFSService@1, VFSReferenceModel@1
- Allow concurrent with: KITA-005, KITA-010, KITA-014, KITA-018, KITA-022, KITA-026, KITA-030, KITA-034, KITA-038
- Conflict policy: Own new service/reference files; do not edit legacy adapters until KITA-007.
- Preconditions: Canonical operation records and hermetic state-machine fixtures exist.
- Effects: A deterministic in-memory/reference implementation exposes the exact VFS state machine without daemon or backend side effects.
- Evidence subset: full CRUD, directories, streams, ranges, rename/move, versions, errors, ordering, idempotency
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Generated traces match the reference model for all supported operations; failure creates no success event; rename/move changes state; return/error types are stable; operations are bounded, cancellation aware and side-effect-free outside the injected storage boundary.
- Embedding query: canonical vfs service reference model state machine deterministic

## KITA-007 Migrate legacy VFS managers and journals through the canonical service

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vfs-migration
- Depends on: KITA-005, KITA-006
- Goal id: KITA-G020
- Outputs: ipfs_kit_py/ipfs_kit_py/vfs_manager.py, ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/vfs.py, ipfs_kit_py/ipfs_kit_py/core/vfs/adapters.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_legacy_adapters.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/vfs/test_vfs_legacy_adapters.py tests/test_vfs_contract_hardening.py tests/test_mcp_vfs_adapter_contract.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/vfs/migration
- Parallel lane: kita-vfs-migration
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/vfs_manager.py, ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/vfs.py, ipfs_kit_py/ipfs_kit_py/core/vfs/adapters.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_legacy_adapters.py
- Interfaces: CanonicalVFSService@1, LegacyVFSAdapter@1, FilesystemJournal
- Allow concurrent with: KITA-008, KITA-011, KITA-015, KITA-019, KITA-023, KITA-027, KITA-031, KITA-035, KITA-039
- Conflict policy: Own listed VFS managers/adapters only; journal implementation changes belong to KITA-018 through KITA-021.
- Preconditions: VFS contracts/service pass; static call closure identifies every resolved manager/journal caller.
- Effects: Existing public VFS paths delegate through one service and use actual journal method names with mutation/result/event ordering preserved.
- Evidence subset: caller migration, differential traces, record_operation/get_entries, failure ordering, compatibility
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: All resolved legacy callers are migrated atomically; no `log_operation`/`get_recent_entries` mismatch remains; underlying false/error results cannot become success; event/dataset buffer publication follows committed state; unavailable dataset flush retains retryable work with collision-safe identity; unresolved dynamic callers block cutover.
- Embedding query: migrate vfs manager journal adapter contract caller closure

## KITA-008 Add VFS versions, snapshots, isolation, locking, and cancellation

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vfs-transactions
- Depends on: KITA-005, KITA-006
- Goal id: KITA-G020
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/transactions.py, ipfs_kit_py/ipfs_kit_py/core/vfs/snapshots.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_transactions.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/vfs/test_vfs_transactions.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/vfs/transactions
- Parallel lane: kita-vfs-transactions
- Resource class: cpu-io-large
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/vfs/transactions.py, ipfs_kit_py/ipfs_kit_py/core/vfs/snapshots.py, ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_transactions.py
- Interfaces: VFSTransaction@1, VFSSnapshot@1, VFSVersion@1
- Allow concurrent with: KITA-007, KITA-011, KITA-015, KITA-019, KITA-023, KITA-027, KITA-031, KITA-035, KITA-039
- Conflict policy: Own new VFS transaction/snapshot modules and tests; WAL integration belongs to KITA-021.
- Preconditions: Canonical VFS service and reference model pass.
- Effects: Conditional versions, snapshots and declared isolation/locking/cancellation semantics become explicit and testable.
- Evidence subset: CAS, lost update, deadlock, cancellation, snapshot identity, concurrent rename/delete
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Version/CID preconditions reject stale writes; declared isolation prevents lost updates; lock ordering is deterministic and bounded; cancellation has an explicit pre/post-commit disposition; snapshots are immutable and reproducible; concurrent generated schedules match the reference model or report typed unsupported boundaries.
- Embedding query: vfs transaction version snapshot cas isolation locking cancellation

## KITA-009 Join VFS backend, WAL, crash, and interface conformance

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: vfs-conformance
- Depends on: KITA-007, KITA-008, KITA-021, KITA-037, KITA-040, KITA-041
- Goal id: KITA-G020
- Outputs: ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_joined_conformance.py, ipfs_kit_py/docs/runtime_readiness/vfs_conformance.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/vfs/test_vfs_joined_conformance.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/vfs/conformance
- Parallel lane: kita-vfs-conformance
- Resource class: io-large
- Resource stage: validation
- Estimated tokens: 24000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/tests/runtime_readiness/vfs/test_vfs_joined_conformance.py, ipfs_kit_py/docs/runtime_readiness/vfs_conformance.json
- Interfaces: VFSConformanceReceipt@1
- Allow concurrent with: KITA-013, KITA-017, KITA-025, KITA-029, KITA-033
- Conflict policy: Own joined VFS tests/report only; discovered defects return to owning tasks and do not get patched in the validator.
- Preconditions: Canonical/legacy VFS, WAL integration, interface adapters and backend-family certifications are current.
- Effects: One evidence bundle proves supported VFS semantics across reference, filesystem, IPFS fixture and Iroh plus every interface and restart boundary.
- Evidence subset: differential, crash, path security, concurrency, backend capability, interface parity
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Required VFS operations have identical canonical results/errors/effects across declared backends and Python/CLI/MCP; every crash point recovers to pre-commit or committed state; path escape and false-success rates are zero; unavailable backend capabilities reject explicitly; no required test skips or print-only paths remain.
- Embedding query: joined vfs conformance backend wal crash interface parity

## KITA-010 Define one backend-scoped bucket catalog and policy contract

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bucket-contract
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G030
- Outputs: ipfs_kit_py/ipfs_kit_py/core/buckets/contracts.py, ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_contracts.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/buckets/test_bucket_contracts.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/buckets/contracts
- Parallel lane: kita-bucket-contract
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 20000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/buckets/contracts.py, ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_contracts.py
- Interfaces: BucketIdentity@1, BucketCatalog@1, BucketPolicy@1, BucketManifest@1
- Allow concurrent with: KITA-005, KITA-006, KITA-014, KITA-018, KITA-022, KITA-026, KITA-030, KITA-034, KITA-038
- Conflict policy: Own new bucket contract/test files; existing five management planes remain read-only observations.
- Preconditions: Inventory and canonical operation contracts exist.
- Effects: Backend-scoped identity, lifecycle states, catalog generation and all quota/retention/encryption/tiering/replica/import/export/query policy fields become finite and versioned.
- Evidence subset: identity, catalog, metadata, policy validation, quotas, encryption, retention, tiering, CAR, query, deletion
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Equal names on distinct backends are distinct; canonical names/aliases are validated; policy cross-field invariants and backend capability sufficiency are checked; exactly-one-primary and verified-replica terminology is defined; unknown fields, secrets, cycles and invalid transitions reject; configured policy cannot claim enforced state.
- Embedding query: bucket catalog backend scoped identity policy quota retention tiering

## KITA-011 Implement transactional bucket lifecycle and object operations

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bucket-service
- Depends on: KITA-006, KITA-010
- Goal id: KITA-G030
- Outputs: ipfs_kit_py/ipfs_kit_py/core/buckets/service.py, ipfs_kit_py/ipfs_kit_py/core/buckets/catalog.py, ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_state_machine.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/buckets/test_bucket_state_machine.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/buckets/service
- Parallel lane: kita-bucket-service
- Resource class: cpu-io-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/buckets/service.py, ipfs_kit_py/ipfs_kit_py/core/buckets/catalog.py, ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_state_machine.py
- Interfaces: BucketService@1, BucketCatalog@1, BucketTransaction@1
- Allow concurrent with: KITA-007, KITA-008, KITA-015, KITA-019, KITA-023, KITA-027, KITA-031, KITA-035, KITA-039
- Conflict policy: Own new canonical bucket service/catalog/tests; legacy manager migration belongs to KITA-012.
- Preconditions: Bucket contract and canonical VFS service are current.
- Effects: Create/update/delete/list/object CRUD/pagination/CAS/quota/deletion fences execute as one state machine with explicit partial effects.
- Evidence subset: state machine, catalog transaction, object operations, quota, pagination, concurrency, cancellation
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Multi-store create/update/delete either commit atomically or persist recoverable compensation; writes cannot race deletion; false backend results never become success; pagination/order and content/version identities are stable; quota holds under concurrent schedules; retries with the same idempotency key do not duplicate effects.
- Embedding query: transactional bucket service catalog object crud quota deletion fence

## KITA-012 Consolidate bucket managers, tiering, import/export, and compensation

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bucket-migration
- Depends on: KITA-010, KITA-011
- Goal id: KITA-G030
- Outputs: ipfs_kit_py/ipfs_kit_py/core/buckets/adapters.py, ipfs_kit_py/ipfs_kit_py/core/buckets/transfer.py, ipfs_kit_py/ipfs_kit_py/unified_bucket_interface.py, ipfs_kit_py/ipfs_kit_py/iroh/bucket_tiering.py, ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_migration_and_sagas.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/buckets/test_bucket_migration_and_sagas.py tests/test_iroh_bucket_tiering.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/buckets/migration
- Parallel lane: kita-bucket-migration
- Resource class: io-large
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/buckets/adapters.py, ipfs_kit_py/ipfs_kit_py/core/buckets/transfer.py, ipfs_kit_py/ipfs_kit_py/unified_bucket_interface.py, ipfs_kit_py/ipfs_kit_py/iroh/bucket_tiering.py, ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_migration_and_sagas.py
- Interfaces: LegacyBucketAdapter@1, BucketTransfer@1, CompensationReceipt@1
- Allow concurrent with: KITA-016, KITA-020, KITA-024, KITA-028, KITA-032, KITA-036, KITA-040, KITA-041
- Conflict policy: Own listed bucket/tiering/transfer surfaces; do not edit replica reconciler or backend adapters.
- Preconditions: Canonical bucket service passes and static call/state-format closure is complete.
- Effects: Five management planes migrate to one catalog; Iroh placement/policy receipts generalize while external effects gain compensation.
- Evidence subset: format migration, duplicate methods, placement handler, policy rollback, CAR, snapshot, clone, cross-bucket query
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Dead duplicate methods and global bucket-name collisions are removed; old formats migrate idempotently with rollback; external placement before catalog commit has compensation/recovery; failed policy reconciliation restores or marks recoverable prior desired state; export binds a snapshot/content manifest; import validates then atomically publishes; cross-bucket query enforces per-bucket authorization/consistency.
- Embedding query: consolidate bucket managers tiering import export car saga compensation

## KITA-013 Join bucket WAL, replica, authorization, backend, and interface conformance

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bucket-conformance
- Depends on: KITA-011, KITA-012, KITA-021, KITA-029, KITA-033, KITA-037, KITA-040, KITA-041
- Goal id: KITA-G030
- Outputs: ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_joined_conformance.py, ipfs_kit_py/docs/runtime_readiness/bucket_conformance.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/buckets/test_bucket_joined_conformance.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/buckets/conformance
- Parallel lane: kita-bucket-conformance
- Resource class: io-network-large
- Resource stage: validation
- Estimated tokens: 24000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/tests/runtime_readiness/buckets/test_bucket_joined_conformance.py, ipfs_kit_py/docs/runtime_readiness/bucket_conformance.json
- Interfaces: BucketConformanceReceipt@1
- Allow concurrent with: KITA-009, KITA-017, KITA-025, KITA-029
- Conflict policy: Own joined bucket tests/report only; fixes return to owning implementation tasks.
- Preconditions: Bucket lifecycle/migration, WAL, replica, authorization, interface and backend evidence is current.
- Effects: Full bucket behavior is certified across declared backends, interfaces, failures, restarts and policy transitions.
- Evidence subset: lifecycle, object, transfer, query, quota, placement, crash, auth, parity, backend
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: All advertised bucket operations pass the common state-machine/differential corpus; crash/retry preserves catalog and external-effect consistency; unauthorized calls dispatch no effects; only verified replicas count; CLI/MCP/Python results match; unavailable capabilities reject; no required skip or success/no-op fallback remains.
- Embedding query: joined bucket conformance wal replica authorization backend interface

## KITA-014 Select one GraphRAG schema, engine, safe persistence, and lazy capability boundary

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: graphrag-contract
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/graphrag/contracts.py, ipfs_kit_py/ipfs_kit_py/graphrag/storage.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_contracts_and_safe_storage.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/graphrag/test_contracts_and_safe_storage.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/graphrag/contracts
- Parallel lane: kita-graphrag-contract
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/graphrag/contracts.py, ipfs_kit_py/ipfs_kit_py/graphrag/storage.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_contracts_and_safe_storage.py
- Interfaces: GraphRAGContent@1, GraphRAGRelation@1, GraphRAGIndexManifest@1, GraphRAGQuery@1
- Allow concurrent with: KITA-005, KITA-006, KITA-010, KITA-018, KITA-022, KITA-026, KITA-030, KITA-034, KITA-038
- Conflict policy: Own new GraphRAG contract/storage/test files; do not modify historic engines until KITA-017.
- Preconditions: Inventory identifies all competing schemas and import-time side effects.
- Effects: One finite schema and non-executable atomic persistence format bind model/dimension/metric/source/index identities and optional capability states.
- Evidence subset: schema, provenance, versions, tombstones, serializer, ownership/mode/symlink, lazy providers, generation CIDs
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Pickle/executable deserialization is forbidden; storage checks type/schema/size/ownership/mode/symlink and atomically publishes generations; imports do not load models or optional ML stacks; relation/history/result contracts are unique and constructor mode cannot change semantic return types; model/dimension/index mismatches reject.
- Embedding query: graphrag canonical schema safe storage lazy model index manifest

## KITA-015 Implement durable rehydration, version history, incremental updates, and clean rebuild

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: graphrag-durability
- Depends on: KITA-003, KITA-014
- Goal id: KITA-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/graphrag/service.py, ipfs_kit_py/ipfs_kit_py/graphrag/projections.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_restart_and_rebuild.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/graphrag/test_restart_and_rebuild.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/graphrag/durability
- Parallel lane: kita-graphrag-durability
- Resource class: cpu-io-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/graphrag/service.py, ipfs_kit_py/ipfs_kit_py/graphrag/projections.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_restart_and_rebuild.py
- Interfaces: GraphRAGService@1, GraphProjection@1, IndexGeneration@1
- Allow concurrent with: KITA-007, KITA-008, KITA-011, KITA-019, KITA-023, KITA-027, KITA-031, KITA-035, KITA-039
- Conflict policy: Own new canonical GraphRAG service/projection/tests; vector index belongs to KITA-016.
- Preconditions: Safe canonical schema/storage and adversarial fixtures exist.
- Effects: Durable records become the source of truth and graph/RDF/vector projections are reproducibly reconstructed after restart or damage.
- Evidence subset: add/update/delete, old versions, relations, tombstones, restart, corrupt generation, incremental versus clean
- Symbolic first: true
- LLM context budget bytes: 16000
- Acceptance: Old content is captured before replacement; restart restores 100% admitted nodes/edges/versions and projection identities; repeated incremental sequences equal a clean rebuild; crash during generation leaves the previous generation readable; corrupt/stale projections rebuild without executing data; deletions/tombstones cannot resurrect.
- Embedding query: graphrag durable restart rehydrate version history incremental clean rebuild

## KITA-016 Add pluggable ANN, exact baseline, deterministic hybrid retrieval, and bounds

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: graphrag-vector-index
- Depends on: KITA-014, KITA-015
- Goal id: KITA-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/graphrag/vector_index.py, ipfs_kit_py/ipfs_kit_py/graphrag/retrieval.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_vector_and_hybrid_retrieval.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/graphrag/test_vector_and_hybrid_retrieval.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/graphrag/vector
- Parallel lane: kita-graphrag-vector
- Resource class: cpu-ml-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/graphrag/vector_index.py, ipfs_kit_py/ipfs_kit_py/graphrag/retrieval.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_vector_and_hybrid_retrieval.py
- Interfaces: VectorIndex@1, ExactVectorIndex@1, HybridRetriever@1
- Allow concurrent with: KITA-012, KITA-020, KITA-024, KITA-028, KITA-032, KITA-036, KITA-040, KITA-041
- Conflict policy: Own canonical vector/retrieval modules/tests; packaging extras belong to KITA-035.
- Preconditions: Canonical durable index generation and pinned corpus/model descriptors exist.
- Effects: Exact and ANN implementations share one interface; bounded deterministic hybrid search replaces full SQLite embedding scans.
- Evidence subset: recall, latency, model/dimension, add/delete, filters, weights, ties, provenance, fallback
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: ANN recall@10 is at least 0.95 against exact search on the pinned representative corpus; query p95 floor is recorded; index/model/dimension/metric identity is enforced; add/update/delete and rebuild are equivalent; weights are finite/nonnegative/normalized; filters and tie-breaks are deterministic; exact fallback is bounded and explicit; retrieval never gains semantic or authorization authority.
- Embedding query: ann vector index exact recall hybrid retrieval deterministic provenance

## KITA-017 Retire competing GraphRAG engines and prove package CLI MCP parity

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: graphrag-parity
- Depends on: KITA-015, KITA-016, KITA-037
- Goal id: KITA-G040
- Outputs: ipfs_kit_py/ipfs_kit_py/graphrag.py, ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/graphrag.py, ipfs_kit_py/ipfs_kit_py/mcp_server/tools/graphrag_tools.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_interface_parity.py, ipfs_kit_py/docs/runtime_readiness/graphrag_conformance.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/graphrag/test_interface_parity.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/graphrag/parity
- Parallel lane: kita-graphrag-parity
- Resource class: cpu-ml-large
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/graphrag.py, ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/graphrag.py, ipfs_kit_py/ipfs_kit_py/mcp_server/tools/graphrag_tools.py, ipfs_kit_py/tests/runtime_readiness/graphrag/test_interface_parity.py, ipfs_kit_py/docs/runtime_readiness/graphrag_conformance.json
- Interfaces: GraphRAGService@1, MCPGraphRAGTools@1, GraphRAGConformanceReceipt@1
- Allow concurrent with: KITA-009, KITA-013, KITA-025, KITA-029, KITA-033
- Conflict policy: Own listed compatibility wrappers/tool files and joined tests; do not reimplement engine behavior in an adapter.
- Preconditions: Canonical GraphRAG and operation/interface registries pass.
- Effects: Historic GraphRAG implementations become thin adapters or are retired; one registry exposes the same operations everywhere.
- Evidence subset: imports, schemas, tool registry, result/error/CID parity, restart, poisoning, missing extras
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: No wrapper returns success/no-op on import failure; canonical MCP++ includes GraphRAG tools; CLI vector/hybrid paths are not stubs; package/CLI/MCP request schemas and normalized results/errors/CIDs are byte-equivalent; restart and poisoning fixtures pass through every interface; required tests assert outcomes and never accept either success or failure.
- Embedding query: graphrag package cli mcp tool parity retire duplicate engine

## KITA-018 Define canonical WAL records, durability states, and compatibility mappings

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-contract
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/core/wal/contracts.py, ipfs_kit_py/ipfs_kit_py/core/wal/compatibility.py, ipfs_kit_py/tests/runtime_readiness/wal/test_wal_contracts.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/wal/test_wal_contracts.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/wal/contracts
- Parallel lane: kita-wal-contract
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/wal/contracts.py, ipfs_kit_py/ipfs_kit_py/core/wal/compatibility.py, ipfs_kit_py/tests/runtime_readiness/wal/test_wal_contracts.py
- Interfaces: WALRecord@1, WALTransaction@1, WALSegment@1, WALCheckpoint@1
- Allow concurrent with: KITA-005, KITA-006, KITA-010, KITA-014, KITA-022, KITA-026, KITA-030, KITA-034, KITA-038
- Conflict policy: Own new WAL contracts/mappings/tests; existing WAL implementations remain observations.
- Preconditions: Inventory and canonical operation-state vocabulary exist.
- Effects: All journal/WAL variants map to one finite framing, sequence, transaction, acknowledgement, replay and corruption model.
- Evidence subset: generation, sequence, checksum, transaction IDs, prepare/commit/abort, payload CID, fsync, checkpoint, archive
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Record identities are collision-safe and monotonic within a generation; states reject impossible transitions; committed/durable is distinct from buffered/queued; payloads are bounded references; compatibility mappings preserve unknown/legacy state explicitly; secrets and unsafe executable encodings reject; fsync/parent-directory and backend-effect requirements are declared per acknowledgement mode.
- Embedding query: wal record durability state transaction sequence checksum compatibility

## KITA-019 Implement append, fsync, group commit, segment, and clean shutdown

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-writer
- Depends on: KITA-003, KITA-018
- Goal id: KITA-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/core/wal/writer.py, ipfs_kit_py/ipfs_kit_py/core/wal/segments.py, ipfs_kit_py/tests/runtime_readiness/wal/test_wal_writer.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/wal/test_wal_writer.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/wal/writer
- Parallel lane: kita-wal-writer
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/wal/writer.py, ipfs_kit_py/ipfs_kit_py/core/wal/segments.py, ipfs_kit_py/tests/runtime_readiness/wal/test_wal_writer.py
- Interfaces: WALWriter@1, WALSegment@1, GroupCommitPolicy@1
- Allow concurrent with: KITA-007, KITA-008, KITA-011, KITA-015, KITA-023, KITA-027, KITA-031, KITA-035, KITA-039
- Conflict policy: Own new writer/segment/test files; legacy adapters and transaction integration belong to KITA-021.
- Preconditions: WAL contracts and fault fixtures pass.
- Effects: Durable append and bounded group commit publish immutable segments with checksums, collision-safe IDs, explicit flush/fsync and stoppable workers.
- Evidence subset: torn write, short write, fsync failure, parent directory, rotation, concurrency, cancellation, shutdown
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Acknowledgement obeys selected flush/fsync policy; file and parent directory durability is injected/tested; concurrent appends receive unique ordered sequence IDs; rotation never appends to a sealed/checkpointed segment; torn/corrupt tails preserve valid prefix; queues are bounded; cancellation is typed; all worker threads/tasks stop and flush or report incomplete shutdown.
- Embedding query: wal writer append fsync group commit segment rotation shutdown

## KITA-020 Implement replay, idempotency, checkpoint, compaction, archive, and corruption recovery

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-recovery
- Depends on: KITA-018, KITA-019
- Goal id: KITA-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/core/wal/recovery.py, ipfs_kit_py/ipfs_kit_py/core/wal/checkpoint.py, ipfs_kit_py/tests/runtime_readiness/wal/test_wal_recovery.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/wal/test_wal_recovery.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/wal/recovery
- Parallel lane: kita-wal-recovery
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/wal/recovery.py, ipfs_kit_py/ipfs_kit_py/core/wal/checkpoint.py, ipfs_kit_py/tests/runtime_readiness/wal/test_wal_recovery.py
- Interfaces: WALRecovery@1, WALCheckpoint@1, WALRecoveryReceipt@1
- Allow concurrent with: KITA-012, KITA-016, KITA-024, KITA-028, KITA-032, KITA-036, KITA-040, KITA-041
- Conflict policy: Own new recovery/checkpoint/tests; no direct legacy-WAL rewrites.
- Preconditions: Immutable segment writer and canonical records pass.
- Effects: Valid committed transactions replay idempotently; checkpoints/compaction/archive preserve post-checkpoint appends and recoverable prior generations.
- Evidence subset: repeated replay, non-idempotent effect, checkpoint, compaction swap, archive failure, corrupt/torn segment, migration
- Symbolic first: true
- LLM context budget bytes: 16000
- Acceptance: Only fully committed transactions replay; repeated recovery has identical result and no duplicate effect; non-idempotent handlers require a verified key/reconciliation path; checkpoint identity covers exact sealed segments; append-after-checkpoint cannot be skipped; compacted state publishes atomically; completed records are not deleted until archive durability; corruption is bounded/reported and valid prior data remains readable.
- Embedding query: wal replay idempotency checkpoint compaction archive corruption recovery

## KITA-021 Cut over VFS, bucket, backend, cache, index, and replica transactions to the WAL

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-integration
- Depends on: KITA-007, KITA-019, KITA-020
- Goal id: KITA-G050
- Outputs: ipfs_kit_py/ipfs_kit_py/core/wal/coordinator.py, ipfs_kit_py/ipfs_kit_py/filesystem_journal.py, ipfs_kit_py/ipfs_kit_py/storage_wal.py, ipfs_kit_py/ipfs_kit_py/enhanced_wal_durability.py, ipfs_kit_py/tests/runtime_readiness/wal/test_joined_crash_matrix.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/wal/test_joined_crash_matrix.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/wal/integration
- Parallel lane: kita-wal-integration
- Resource class: io-large
- Resource stage: integration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/wal/coordinator.py, ipfs_kit_py/ipfs_kit_py/filesystem_journal.py, ipfs_kit_py/ipfs_kit_py/storage_wal.py, ipfs_kit_py/ipfs_kit_py/enhanced_wal_durability.py, ipfs_kit_py/tests/runtime_readiness/wal/test_joined_crash_matrix.py
- Interfaces: TransactionCoordinator@1, WALRecoveryReceipt@1
- Allow concurrent with: KITA-023, KITA-027, KITA-031, KITA-035, KITA-039
- Conflict policy: Own canonical coordinator and listed legacy WAL/journal cutovers; coordinate VFS adapter changes through dependency and do not edit bucket/cache/index/replica modules owned elsewhere.
- Preconditions: Canonical VFS call path and WAL writer/recovery are current; impact closure covers every WAL transaction caller.
- Effects: Real mutations are fenced by durable intent and commit/abort; legacy variants delegate or migrate; downstream generation invalidation is transaction aware.
- Evidence subset: begin/commit/rollback IDs, mutation order, false backend results, crash points, cache/index/replica invalidation
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Begin/commit/abort markers persist correct transaction IDs; rollback is not metadata-only for an already performed effect; callers check commit failure; random/mock production handlers are removed; failed storage append cannot return accepted success; no in-place Parquet rewrite or delete-before-archive remains; crash injection at every named boundary yields pre-commit or committed state with zero acknowledged loss and repeated replay duplicates zero effects.
- Embedding query: transaction coordinator wal vfs bucket cache index replica crash cutover

## KITA-022 Extract an ARC core and formalize reference-model invariants

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: arc-contract
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G060
- Outputs: ipfs_kit_py/ipfs_kit_py/cache/arc/contracts.py, ipfs_kit_py/ipfs_kit_py/cache/arc/reference.py, ipfs_kit_py/tests/runtime_readiness/arc/test_arc_reference_model.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/arc/test_arc_reference_model.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/arc/contracts
- Parallel lane: kita-arc-contract
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/cache/arc/contracts.py, ipfs_kit_py/ipfs_kit_py/cache/arc/reference.py, ipfs_kit_py/tests/runtime_readiness/arc/test_arc_reference_model.py
- Interfaces: AdaptiveReplacementCache@1, ARCReferenceModel@1, CacheKey@1
- Allow concurrent with: KITA-005, KITA-006, KITA-010, KITA-014, KITA-018, KITA-026, KITA-030, KITA-034, KITA-038
- Conflict policy: Own new ARC contract/reference/tests; the 6,488-line legacy module remains read-only until KITA-024.
- Preconditions: Inventory and canonical operation/content/version contracts exist.
- Effects: T1/T2/B1/B2, byte/entry size, adaptive target, admission, eviction, ghost and metric transitions have a deterministic oracle.
- Evidence subset: invariants, entry/byte budgets, growth, ghost hits, eviction, invalid capacities, deterministic traces
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Reference enforces current_size equals live T1 plus T2 size and capacity, pairwise-disjoint lists, no ghost values, bounded adaptive target, exact update/growth/ghost accounting and deterministic eviction; invalid keys/sizes/capacities and unbounded values reject; property strategy emits reproducible minimal traces.
- Embedding query: arc cache reference model t1 t2 b1 b2 size ghost invariants

## KITA-023 Implement concurrency-safe ARC, byte accounting, and single-flight fills

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: arc-core
- Depends on: KITA-003, KITA-022
- Goal id: KITA-G060
- Outputs: ipfs_kit_py/ipfs_kit_py/cache/arc/cache.py, ipfs_kit_py/ipfs_kit_py/cache/arc/concurrency.py, ipfs_kit_py/tests/runtime_readiness/arc/test_arc_concurrency.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/arc/test_arc_concurrency.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/arc/core
- Parallel lane: kita-arc-core
- Resource class: cpu-memory-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/cache/arc/cache.py, ipfs_kit_py/ipfs_kit_py/cache/arc/concurrency.py, ipfs_kit_py/tests/runtime_readiness/arc/test_arc_concurrency.py
- Interfaces: AdaptiveReplacementCache@1, CacheFillCoordinator@1
- Allow concurrent with: KITA-007, KITA-008, KITA-011, KITA-015, KITA-019, KITA-027, KITA-031, KITA-035, KITA-039
- Conflict policy: Own new ARC core/concurrency/tests; compatibility migration belongs to KITA-024.
- Preconditions: ARC reference model and adversarial schedules exist.
- Effects: One synchronized or single-owner ARC implementation preserves invariants under concurrent get/put/delete/fill/cancel operations.
- Evidence subset: linearizability, locks/owner, replacement growth, ghost pressure, stampede, oversized values, cancellation
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Random sequential traces equal the reference model; admitted concurrent histories are linearizable; updates and ghost hits maintain exact byte accounting and evict as necessary; ghost lists are bounded/pruned; single-flight prevents duplicate fill without deadlock; cancelled/failing fillers wake waiters with typed results; no unguarded thread dispatch mutates shared dictionaries.
- Embedding query: concurrent arc cache byte accounting single flight linearizable cancellation

## KITA-024 Add generation-bound invalidation, persistence, metrics, and legacy migration

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: arc-integration
- Depends on: KITA-008, KITA-022, KITA-023
- Goal id: KITA-G060
- Outputs: ipfs_kit_py/ipfs_kit_py/cache/arc/persistence.py, ipfs_kit_py/ipfs_kit_py/cache/arc/metrics.py, ipfs_kit_py/ipfs_kit_py/arc_cache.py, ipfs_kit_py/ipfs_kit_py/arc_cache_anyio.py, ipfs_kit_py/tests/runtime_readiness/arc/test_arc_persistence_and_migration.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/arc/test_arc_persistence_and_migration.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/arc/integration
- Parallel lane: kita-arc-integration
- Resource class: cpu-io-large
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/cache/arc/persistence.py, ipfs_kit_py/ipfs_kit_py/cache/arc/metrics.py, ipfs_kit_py/ipfs_kit_py/arc_cache.py, ipfs_kit_py/ipfs_kit_py/arc_cache_anyio.py, ipfs_kit_py/tests/runtime_readiness/arc/test_arc_persistence_and_migration.py
- Interfaces: CacheGeneration@1, CachePersistence@1, CacheMetrics@1
- Allow concurrent with: KITA-012, KITA-016, KITA-020, KITA-028, KITA-032, KITA-036, KITA-040, KITA-041
- Conflict policy: Own ARC compatibility modules plus new persistence/metrics; other cache packages remain untouched.
- Preconditions: Concurrency-safe ARC and VFS content/version identities pass.
- Effects: Cache keys bind content/version/namespace/policy/serializer/generation, safe restart is possible, and legacy exports delegate to the core.
- Evidence subset: safe serialization, corrupt/stale entry, restart, invalidation, authorization scope, heat/recency, telemetry
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Persistence uses bounded non-executable atomic format; stale/corrupt/schema-mismatched entries miss safely; content/version/policy changes invalidate exact dependents; hits never bypass authorization/consistency; recency uses pre-access time rather than a constant; metrics distinguish live/ghost/stale/admission/eviction/fill states; legacy and AnyIO surfaces contain no copied unsynchronized implementation.
- Embedding query: arc cache persistence generation invalidation metrics legacy migration

## KITA-025 Prove ARC coherence, randomized invariants, restart, and performance

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: arc-conformance
- Depends on: KITA-022, KITA-023, KITA-024
- Goal id: KITA-G060
- Outputs: ipfs_kit_py/tests/runtime_readiness/arc/test_arc_joined_conformance.py, ipfs_kit_py/docs/runtime_readiness/arc_conformance.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/arc/test_arc_joined_conformance.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/arc/conformance
- Parallel lane: kita-arc-conformance
- Resource class: cpu-memory-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/tests/runtime_readiness/arc/test_arc_joined_conformance.py, ipfs_kit_py/docs/runtime_readiness/arc_conformance.json
- Interfaces: ARCConformanceReceipt@1
- Allow concurrent with: KITA-009, KITA-013, KITA-017, KITA-029, KITA-033
- Conflict policy: Own joined ARC tests/report only; implementation fixes return to KITA-022 through KITA-024.
- Preconditions: ARC core, invalidation/persistence and metrics pass.
- Effects: Default CI gains assertion-backed randomized, concurrent, restart, corrupt-entry, coherence and benchmark coverage.
- Evidence subset: invariant exhaustion, concurrency, restart, VFS/WAL/index/replica coherence, memory, throughput
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Thousands of seeded traces and concurrent schedules preserve every invariant; restart identity and corrupt-entry rejection pass; VFS write/delete, WAL replay, index rebuild and replica convergence invalidate or retain exact entries; no excluded-only ARC gate remains; hit/miss speedup and memory overhead meet the pinned floor without hiding cold-path failures.
- Embedding query: arc cache conformance randomized invariant restart coherence performance

## KITA-026 Define valid replica policy, placement, and lifecycle states

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: replica-contract
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G070
- Outputs: ipfs_kit_py/ipfs_kit_py/core/replication/contracts.py, ipfs_kit_py/ipfs_kit_py/core/replication/placement.py, ipfs_kit_py/tests/runtime_readiness/replication/test_replica_contracts.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/replication/test_replica_contracts.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/replication/contracts
- Parallel lane: kita-replica-contract
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/replication/contracts.py, ipfs_kit_py/ipfs_kit_py/core/replication/placement.py, ipfs_kit_py/tests/runtime_readiness/replication/test_replica_contracts.py
- Interfaces: ReplicaPolicy@1, PlacementPlan@1, ReplicaState@1
- Allow concurrent with: KITA-005, KITA-006, KITA-010, KITA-014, KITA-018, KITA-022, KITA-030, KITA-034, KITA-038
- Conflict policy: Own new replication contract/placement/tests; historic policy/reconciler code remains read-only.
- Preconditions: Backend capability and canonical state vocabularies exist.
- Effects: Desired count, eligible failure domains/backends, durability/consistency/encryption/retention/cost/locality and replica lifecycle states become closed and validated.
- Evidence subset: cross-field policy, distinct placement, capability sufficiency, deterministic inventory snapshot, desired/planned/pending/verified
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Enforce min <= max <= critical and all declared ordering, preferred/excluded disjointness, distinct writable failure domains and sufficient capability before mutation; placement is deterministic for an exact inventory snapshot; only integrity-verified durable replicas satisfy desired count; invalid/unsatisfiable policy fails with zero effects.
- Embedding query: replica policy placement failure domain backend capability lifecycle

## KITA-027 Implement idempotent replica reconciliation and anti-entropy

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: replica-reconciler
- Depends on: KITA-003, KITA-026
- Goal id: KITA-G070
- Outputs: ipfs_kit_py/ipfs_kit_py/core/replication/reconciler.py, ipfs_kit_py/ipfs_kit_py/core/replication/integrity.py, ipfs_kit_py/tests/runtime_readiness/replication/test_reconciler.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/replication/test_reconciler.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/replication/reconciler
- Parallel lane: kita-replica-reconciler
- Resource class: io-network-large
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/replication/reconciler.py, ipfs_kit_py/ipfs_kit_py/core/replication/integrity.py, ipfs_kit_py/tests/runtime_readiness/replication/test_reconciler.py
- Interfaces: ReplicaReconciler@1, IntegrityVerifier@1, ReconciliationReceipt@1
- Allow concurrent with: KITA-007, KITA-008, KITA-011, KITA-015, KITA-019, KITA-023, KITA-031, KITA-035, KITA-039
- Conflict policy: Own new reconciler/integrity/tests; legacy integration belongs to KITA-028.
- Preconditions: Replica contracts and backend-fault fixtures exist.
- Effects: Bounded reconciliation schedules, verifies, repairs, removes and rebalances replicas idempotently with explicit retry/partial state.
- Evidence subset: backend loss, partition, stale listing, corrupt/divergent version, retry, cancellation, rate limit, anti-entropy
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Repeated reconciliation converges to the same plan/state; pending/queued work never counts; copied content is integrity/version verified; network/backend failure preserves recoverable state; cancellation and backpressure are bounded; anti-entropy detects and repairs missing/corrupt/divergent replicas under policy; removal never drops below verified minimum without explicit blocked receipt.
- Embedding query: replica reconciler anti entropy integrity retry convergence backpressure

## KITA-028 Migrate legacy replication and tiered-cache paths

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: replica-migration
- Depends on: KITA-026, KITA-027
- Goal id: KITA-G070
- Outputs: ipfs_kit_py/ipfs_kit_py/backend_policies.py, ipfs_kit_py/ipfs_kit_py/tiered_cache_manager.py, ipfs_kit_py/ipfs_kit_py/fs_journal_replication.py, ipfs_kit_py/tests/runtime_readiness/replication/test_legacy_replication_adapters.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/replication/test_legacy_replication_adapters.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/replication/migration
- Parallel lane: kita-replica-migration
- Resource class: io-large
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/backend_policies.py, ipfs_kit_py/ipfs_kit_py/tiered_cache_manager.py, ipfs_kit_py/ipfs_kit_py/fs_journal_replication.py, ipfs_kit_py/tests/runtime_readiness/replication/test_legacy_replication_adapters.py
- Interfaces: ReplicaReconciler@1, LegacyReplicationAdapter@1
- Allow concurrent with: KITA-012, KITA-016, KITA-020, KITA-024, KITA-032, KITA-036, KITA-040, KITA-041
- Conflict policy: Own listed legacy replication/policy files; bucket-tiering edits coordinate through KITA-012 dependency outputs without overlapping them.
- Preconditions: Canonical policy/reconciler passes and impact closure identifies every shadowed method/caller.
- Effects: Historic replication paths delegate to one reconciler, duplicate definitions disappear, and fabricated/test-specific success metadata is removed.
- Evidence subset: duplicate ensure_replication, pending count, test-key branches, policy migration, caller compatibility
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: The reporting-only duplicate no longer shadows implementation; method signatures have one caller-complete migration; pending work never augments verified redundancy; test-key-specific production behavior is removed; legacy policies migrate with explicit unsupported fields; no mock-injected metadata/health is required for conformance; unresolved dynamic callers block destructive cleanup.
- Embedding query: migrate legacy replication tiered cache duplicate method pending verified

## KITA-029 Join bucket, WAL, index, cache, backend, and replica convergence

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: replica-conformance
- Depends on: KITA-012, KITA-016, KITA-020, KITA-024, KITA-028, KITA-040, KITA-041
- Goal id: KITA-G070
- Outputs: ipfs_kit_py/tests/runtime_readiness/replication/test_joined_convergence.py, ipfs_kit_py/docs/runtime_readiness/replica_conformance.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/replication/test_joined_convergence.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/replication/conformance
- Parallel lane: kita-replica-conformance
- Resource class: io-network-large
- Resource stage: validation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/tests/runtime_readiness/replication/test_joined_convergence.py, ipfs_kit_py/docs/runtime_readiness/replica_conformance.json
- Interfaces: ReplicaConformanceReceipt@1
- Allow concurrent with: KITA-009, KITA-013, KITA-017, KITA-025, KITA-033
- Conflict policy: Own joined convergence tests/report only; fixes return to owning tasks.
- Preconditions: Bucket/WAL/GraphRAG/ARC/replication and backend evidence is current.
- Effects: Cross-subsystem failure, replay, deletion, rebalancing and recovery converge without false replica or cache/index state.
- Evidence subset: policy change, backend loss, partition, bucket delete, WAL replay, index lag, cache invalidation, integrity
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Seeded and generated chaos schedules converge to one desired state; verified replica counts are correct; duplicate effects are zero; deleted/tombstoned content does not resurrect; cache and GraphRAG reflect the committed version; partial placement has compensation or a recoverable receipt; unsupported provider capability blocks rather than falls back.
- Embedding query: joined replica convergence bucket wal index cache backend chaos

## KITA-030 Repair MCP++ construction, registry, transports, and protocol advertisement

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcplusplus-bootstrap
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G080
- Outputs: ipfs_kit_py/ipfs_kit_py/mcp_server/server.py, ipfs_kit_py/ipfs_kit_py/mcp_server/__init__.py, ipfs_kit_py/ipfs_kit_py/mcp_server/tools/__init__.py, ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_server_bootstrap.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/mcplusplus/test_server_bootstrap.py ipfs_kit_py/mcp_server/tests_e2e_interop.py -k 'profile_c or profile_d or all_five_profiles_smoke or mcppp_envelope'
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/mcplusplus/bootstrap
- Parallel lane: kita-mcplusplus-bootstrap
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/mcp_server/server.py, ipfs_kit_py/ipfs_kit_py/mcp_server/__init__.py, ipfs_kit_py/ipfs_kit_py/mcp_server/tools/__init__.py, ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_server_bootstrap.py
- Interfaces: MCPPlusPlusServer@1, EventDAGStore@1, MCPProfileRegistry@1
- Allow concurrent with: KITA-005, KITA-006, KITA-010, KITA-014, KITA-018, KITA-022, KITA-026, KITA-034, KITA-038
- Conflict policy: Own MCP++ server/bootstrap/tool-registry files; authorization implementation belongs to KITA-031/KITA-032.
- Preconditions: Inventory proves current undefined EventDAGStore and missing profile/tool registrations.
- Effects: MCP++ constructs inertly and exposes one deterministic registry/profile map on stdio, HTTP and P2P.
- Evidence subset: EventDAGStore import, constructor, initialize, profiles, REST map, stdio, HTTP, P2P, tool registry
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Import and construction raise no NameError; EventDAGStore is explicitly injected/imported; MCPServer is exported lazily; initialize advertises canonical supported profiles including deontic policy when available; routes and tools derive from one registry; minimal MCP wheel starts all three transports without importing FastAPI unless the HTTP extra/path is used; the six focused baseline tests pass.
- Embedding query: mcp plus plus server EventDAGStore profile registry stdio http p2p bootstrap

## KITA-031 Implement signed UCAN verification, attenuation, revocation, replay, and key lifecycle

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ucan-verifier
- Depends on: KITA-003, KITA-030
- Goal id: KITA-G080
- Outputs: ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/ucan.py, ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/revocation.py, ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_ucan_verifier.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/mcplusplus/test_ucan_verifier.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/mcplusplus/ucan
- Parallel lane: kita-ucan
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/ucan.py, ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/revocation.py, ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_ucan_verifier.py
- Interfaces: UCANVerifier@1, UCANDelegation@1, RevocationLedger@1
- Allow concurrent with: KITA-007, KITA-008, KITA-011, KITA-015, KITA-019, KITA-023, KITA-027, KITA-035, KITA-039
- Conflict policy: Own canonical UCAN/revocation modules/tests; dispatcher and Profile D integration belong to KITA-032.
- Preconditions: MCP++ bootstrap and adversarial token fixtures pass.
- Effects: Signed tokens and proof chains produce finite decisions with exact resource/ability/audience/time/revocation/replay identity.
- Evidence subset: signatures, issuer/audience, nbf/exp/skew, proof chain, attenuation, wildcard rules, nonce, revocation, rotation
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Forged/tampered/unsigned/downgraded/wrong-audience/expired/not-yet-valid/revoked/replayed/cyclic/over-broad/cross-tenant chains reject; delegation can only attenuate resource, ability, time and bounds; omitted resource/ability cannot become wildcard; durable nonce/revocation/key-rotation semantics survive restart; unavailable requested crypto/ledger fails closed; secrets/private keys never enter receipts.
- Embedding query: signed ucan verification delegation attenuation revocation replay key rotation

## KITA-032 Wire canonical datasets Profile D and UCAN into a fail-closed dispatcher

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authorization-gate
- Depends on: KITA-030, KITA-031
- Goal id: KITA-G080
- Outputs: ipfs_kit_py/ipfs_kit_py/mcp_server/authorization.py, ipfs_kit_py/ipfs_kit_py/mcp_server/server.py, ipfs_kit_py/ipfs_kit_py/mcp/profile_d_policy.py, ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_authorization_dispatch_gate.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/mcplusplus/test_authorization_dispatch_gate.py tests/test_profile_d_policy.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/mcplusplus/authorization
- Parallel lane: kita-auth-gate
- Resource class: cpu-security
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/mcp_server/authorization.py, ipfs_kit_py/ipfs_kit_py/mcp_server/server.py, ipfs_kit_py/ipfs_kit_py/mcp/profile_d_policy.py, ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_authorization_dispatch_gate.py
- Interfaces: AuthorizationDecision@1, ProfileDPolicyProvider@1, ProtectedDispatcher@1
- Allow concurrent with: KITA-012, KITA-016, KITA-020, KITA-024, KITA-028, KITA-036, KITA-040, KITA-041
- Conflict policy: Own authorization module, canonical Profile D bridge and one serialized server dispatcher cutover; do not modify storage handlers.
- Preconditions: MCP++ registry and canonical UCAN verifier pass; datasets provider capability is probed lazily.
- Effects: Every protected tool resolves exact resource/ability, verifies UCAN, evaluates canonical Profile D, records audit as required, then and only then dispatches.
- Evidence subset: tool map, policy route/profile, provider unavailable, decision CID, audit, deadline/rate bounds, handler spy
- Symbolic first: true
- LLM context budget bytes: 16000
- Acceptance: The simplistic competing evaluator is retired/adapted; `/mcp/policy/evaluate` and profile advertisement use one provider; `tools/call` has no bypass; all negative UCAN/policy cases produce zero handler calls; requested unavailable validator/provider/ledger denies; allowed decisions bind exact envelope/tool/resource/ability/policy roots; audit is durable/redacted; authorization and effect share request/transaction identity.
- Embedding query: profile d ucan fail closed dispatcher tool resource ability policy gate

## KITA-033 Prove MCP++ authorization and result parity across transports

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authorization-conformance
- Depends on: KITA-011, KITA-016, KITA-021, KITA-029, KITA-031, KITA-032, KITA-037
- Goal id: KITA-G080
- Outputs: ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_transport_security_parity.py, ipfs_kit_py/docs/runtime_readiness/mcplusplus_conformance.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/mcplusplus/test_transport_security_parity.py ipfs_kit_py/mcp_server/tests_e2e_interop.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/mcplusplus/conformance
- Parallel lane: kita-auth-conformance
- Resource class: cpu-security
- Resource stage: validation
- Estimated tokens: 26000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_transport_security_parity.py, ipfs_kit_py/docs/runtime_readiness/mcplusplus_conformance.json
- Interfaces: MCPPlusPlusSecurityReceipt@1
- Allow concurrent with: KITA-009, KITA-013, KITA-017, KITA-025, KITA-029
- Conflict policy: Own joined security/transport tests/report only; fixes return to owning auth/interface/storage tasks.
- Preconditions: Canonical dispatcher, operation adapters and representative bucket/GraphRAG/WAL/replica operations pass.
- Effects: One adversarial corpus proves identical security decisions and canonical storage results over stdio, HTTP and P2P.
- Evidence subset: absent/forged/tampered/expired/revoked/replayed/attenuation/confused deputy/downgrade, transport decision/result parity
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Every denial yields the same canonical code/decision CID and zero dispatch on every transport; allowed operations yield semantically identical result/error/content/version/transaction CIDs after transport normalization; signed-to-unsigned downgrade and confused-deputy attempts fail; restart preserves revocation/replay; no MagicMock-only security evidence satisfies the gate.
- Embedding query: mcp plus plus ucan transport stdio http p2p security result parity

## KITA-034 Build the versioned operation registry and canonical service router

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: interface-registry
- Depends on: KITA-002, KITA-004
- Goal id: KITA-G090
- Outputs: ipfs_kit_py/ipfs_kit_py/core/operation_registry.py, ipfs_kit_py/ipfs_kit_py/core/service_router.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_operation_registry.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/interfaces/test_operation_registry.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/interfaces/registry
- Parallel lane: kita-interface-registry
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/operation_registry.py, ipfs_kit_py/ipfs_kit_py/core/service_router.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_operation_registry.py
- Interfaces: OperationRegistry@1, CanonicalStorageService@1, ServiceRouter@1
- Allow concurrent with: KITA-005, KITA-006, KITA-010, KITA-014, KITA-018, KITA-022, KITA-026, KITA-030, KITA-038
- Conflict policy: Own new registry/router/tests; no existing CLI/MCP/package adapters are edited.
- Preconditions: Canonical operation records and interface/import baseline exist.
- Effects: One inert registry maps operation IDs to schemas, capabilities, authorization, service handlers and transport projections without loading providers.
- Evidence subset: operation IDs, versions, request/result/error schemas, capabilities, handler routes, public/ protected classification
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: IDs and aliases are unique; every registered operation has one request/result/error contract, service handler/capability and auth classification; unsupported operations reject before dispatch; registry import is inert/lazy; generated projections are deterministic; no adapter-specific fallback or business logic is admitted.
- Embedding query: operation registry canonical service router schema capability interface

## KITA-035 Make version, exports, dependencies, extras, setup, requirements, and imports coherent

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: packaging-lazy-imports
- Depends on: KITA-004, KITA-034
- Goal id: KITA-G090
- Outputs: ipfs_kit_py/pyproject.toml, ipfs_kit_py/setup.py, ipfs_kit_py/requirements.txt, ipfs_kit_py/ipfs_kit_py/__init__.py, ipfs_kit_py/ipfs_kit_py/core/__init__.py, ipfs_kit_py/ipfs_kit_py/mcp_server/__init__.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_packaging_and_lazy_imports.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/interfaces/test_packaging_and_lazy_imports.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/interfaces/packaging
- Parallel lane: kita-packaging
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/pyproject.toml, ipfs_kit_py/setup.py, ipfs_kit_py/requirements.txt, ipfs_kit_py/ipfs_kit_py/__init__.py, ipfs_kit_py/ipfs_kit_py/core/__init__.py, ipfs_kit_py/ipfs_kit_py/mcp_server/__init__.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_packaging_and_lazy_imports.py
- Interfaces: PackageMetadata@1, LazyFeatureRegistry@1
- Allow concurrent with: KITA-007, KITA-008, KITA-011, KITA-015, KITA-019, KITA-023, KITA-027, KITA-031, KITA-039
- Conflict policy: Own listed package/manifest/init files; do not modify feature implementations or MCP tools.
- Preconditions: Bound install/import baseline and operation registry exist.
- Effects: Pyproject is the dependency/version source; setup/requirements are generated or mechanically checked; optional feature imports occur only on use.
- Evidence subset: dynamic version, __all__, entry points, extras, markers, setup metadata purity, cold imports, installer JIT recursion
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Runtime version equals wheel metadata; reviewed public exports include canonical APIs; normalized pyproject/setup/requirements core and markers have zero unexplained drift; dedicated GraphRAG and MCP extras contain exact dependencies; setup metadata performs no dpkg/system probe; root/registry import does not import installers or heavy optional stacks, start processes/network/models or write user state; core JIT recursion is fixed; missing extras fail only on use with actionable typed errors; cold-import floors are checked.
- Embedding query: package version exports dependency extras setup requirements lazy cold import

## KITA-036 Generate or validate Python sync async and CLI adapters

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: python-cli-parity
- Depends on: KITA-034, KITA-035
- Goal id: KITA-G090
- Outputs: ipfs_kit_py/ipfs_kit_py/high_level_api, ipfs_kit_py/ipfs_kit_py/cli, ipfs_kit_py/ipfs_kit_py/bucket_vfs_cli.py, ipfs_kit_py/ipfs_kit_py/backend_cli.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_python_cli_parity.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/interfaces/test_python_cli_parity.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/interfaces/python-cli
- Parallel lane: kita-python-cli
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/high_level_api, ipfs_kit_py/ipfs_kit_py/cli, ipfs_kit_py/ipfs_kit_py/bucket_vfs_cli.py, ipfs_kit_py/ipfs_kit_py/backend_cli.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_python_cli_parity.py
- Interfaces: PythonAdapter@1, AsyncPythonAdapter@1, CLIAdapter@1
- Allow concurrent with: KITA-012, KITA-016, KITA-020, KITA-024, KITA-028, KITA-032, KITA-040, KITA-041
- Conflict policy: Own listed high-level/CLI adapter directories/files; do not edit canonical subsystem handlers.
- Preconditions: Operation registry and coherent package metadata/lazy exports pass.
- Effects: Public Python sync/async calls and CLI commands translate through one registry/service and canonical error/result serializer.
- Evidence subset: signatures, parameters/defaults, async cancellation, streams, JSON/stdout/stderr, exit codes, side effects
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Every advertised registry operation has the intended Python/CLI projection or explicit non-applicable reason; sync/async results and cancellation states match; CLI option/default/schema and exit codes derive from canonical metadata; no stub/print-only/pass-on-error path remains; identical fixtures produce equal canonical result/error/content/version/effect records.
- Embedding query: python sync async cli adapter signature result error exit code parity

## KITA-037 Generate or validate MCP and MCP++ tools and prove all-interface parity

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp-interface-parity
- Depends on: KITA-032, KITA-034, KITA-035, KITA-036
- Goal id: KITA-G090
- Outputs: ipfs_kit_py/ipfs_kit_py/mcp_server/tools, ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/mcp_tools.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_all_interface_parity.py, ipfs_kit_py/docs/runtime_readiness/interface_manifest.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/interfaces/test_all_interface_parity.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/interfaces/mcp
- Parallel lane: kita-mcp-interface
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/mcp_server/tools, ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/mcp_tools.py, ipfs_kit_py/tests/runtime_readiness/interfaces/test_all_interface_parity.py, ipfs_kit_py/docs/runtime_readiness/interface_manifest.json
- Interfaces: MCPToolAdapter@1, MCPPlusPlusToolAdapter@1, InterfaceManifest@1
- Allow concurrent with: KITA-039, KITA-040, KITA-041
- Conflict policy: Own canonical MCP tool adapters/manifest/tests; server authorization remains owned by KITA-032.
- Preconditions: Authorized dispatcher, operation registry, packaging and Python/CLI adapters pass.
- Effects: One tool registry/schema projection serves MCP/MCP++ transports and is differentially checked against Python and CLI.
- Evidence subset: tool IDs, JSON schemas, required/default fields, results, errors, CIDs, effects, auth, stdio/HTTP/P2P
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Tool sets and schemas equal the reviewed interface manifest; no duplicate competing registration silently wins; package, Python, CLI, MCP stdio, HTTP and P2P fixtures yield 100% semantic parity after removal of request ID/timing; error/retry/partial-effect meanings match; every protected tool requires the same authorization; missing optional feature rejects identically rather than disappearing or succeeding/no-op.
- Embedding query: mcp mcplusplus tool registry schema package cli python parity

## KITA-038 Reconcile backend names, aliases, schemas, factories, capabilities, and support tiers

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backend-contract
- Depends on: KITA-001, KITA-002
- Goal id: KITA-G100
- Outputs: ipfs_kit_py/ipfs_kit_py/backends/spec.py, ipfs_kit_py/ipfs_kit_py/backend_registry.py, ipfs_kit_py/ipfs_kit_py/backend_schemas.py, ipfs_kit_py/tests/runtime_readiness/backends/test_backend_spec_registry.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/backends/test_backend_spec_registry.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/backends/contracts
- Parallel lane: kita-backend-contract
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/ipfs_kit_py/backends/spec.py, ipfs_kit_py/ipfs_kit_py/backend_registry.py, ipfs_kit_py/ipfs_kit_py/backend_schemas.py, ipfs_kit_py/tests/runtime_readiness/backends/test_backend_spec_registry.py
- Interfaces: BackendSpec@1, BackendPlugin@2, BackendCapability@1, BackendSupportTier@1
- Allow concurrent with: KITA-005, KITA-006, KITA-010, KITA-014, KITA-018, KITA-022, KITA-026, KITA-030, KITA-034
- Conflict policy: Own backend spec/registry/schema/tests; actual adapter implementations belong to KITA-039 through KITA-041.
- Preconditions: Exhaustive backend inventory and canonical operation records exist.
- Effects: Every registered/documented type and alias has one schema/plugin/optional factory/capability/health/secret/tier mapping.
- Evidence subset: 22 legacy types plus Iroh, aliases, schema hyphen/underscore drift, unregistered implementations, missing create_filesystem
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Registry, schemas, runtime factories, CLI/MCP names and docs are bijective; BackendPlugin requires only correctly capability-gated operations including an optional runtime factory contract; legacy configuration-only plugins cannot be invoked as storage; ipfs_cluster/ipfs-cluster and other aliases normalize unambiguously; Saturn/Synapse/Lotus/Arrow-like surfaced implementations are registered honestly or explicitly excluded; support tier is never inferred from registry presence.
- Embedding query: backend spec registry alias schema factory capability support tier

## KITA-039 Build the common backend conformance kit and hermetic reference adapters

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backend-conformance-kit
- Depends on: KITA-003, KITA-038
- Goal id: KITA-G100
- Outputs: ipfs_kit_py/tests/runtime_readiness/backends/conformance.py, ipfs_kit_py/ipfs_kit_py/backends/filesystem_backend.py, ipfs_kit_py/ipfs_kit_py/backends/ipfs_backend.py, ipfs_kit_py/tests/runtime_readiness/backends/test_hermetic_adapters.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/backends/test_hermetic_adapters.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/backends/hermetic
- Parallel lane: kita-backend-hermetic
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/tests/runtime_readiness/backends/conformance.py, ipfs_kit_py/ipfs_kit_py/backends/filesystem_backend.py, ipfs_kit_py/ipfs_kit_py/backends/ipfs_backend.py, ipfs_kit_py/tests/runtime_readiness/backends/test_hermetic_adapters.py
- Interfaces: BackendConformanceKit@1, BackendAdapter@2, BackendCertificationReceipt@1
- Allow concurrent with: KITA-007, KITA-008, KITA-011, KITA-015, KITA-019, KITA-023, KITA-027, KITA-031, KITA-035, KITA-037
- Conflict policy: Own conformance kit and hermetic filesystem/IPFS-fixture adapter files; family adapters belong to KITA-040/KITA-041.
- Preconditions: BackendSpec registry and adversarial backend fixtures pass.
- Effects: One capability-driven suite tests lifecycle, CRUD/stream/range/list, metadata, integrity, conditional/idempotent operations, failures, retry, cancellation, security and parity.
- Evidence subset: fake/reference backend, path safety, pagination, streams, partial effect, credentials, health, retry, cancellation
- Symbolic first: true
- LLM context budget bytes: 12000
- Acceptance: Required core CI executes assertion-backed conformance for filesystem/local and hermetic IPFS fixture; suite skips only non-declared operations and verifies typed unsupported with zero effects; adapters preserve canonical errors/results/CIDs; path and secret boundaries pass; retry does not duplicate; cancellation/close are bounded; fixture cannot be mistaken for live provider certification.
- Embedding query: backend conformance kit filesystem local ipfs fixture crud streaming failure

## KITA-040 Certify local-service and object-store backend families

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backend-service-lanes
- Depends on: KITA-038, KITA-039
- Goal id: KITA-G100
- Outputs: ipfs_kit_py/ipfs_kit_py/backends/s3_backend.py, ipfs_kit_py/ipfs_kit_py/iroh/backend.py, ipfs_kit_py/ipfs_kit_py/backends/ipfs_backend.py, ipfs_kit_py/tests/runtime_readiness/backends/test_service_backends.py, ipfs_kit_py/docs/runtime_readiness/backend_service_receipts
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/backends/test_service_backends.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/backends/services
- Parallel lane: kita-backend-services
- Resource class: io-network-large
- Resource stage: validation
- Estimated tokens: 32000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_kit_py/ipfs_kit_py/backends/s3_backend.py, ipfs_kit_py/ipfs_kit_py/iroh/backend.py, ipfs_kit_py/ipfs_kit_py/backends/ipfs_backend.py, ipfs_kit_py/tests/runtime_readiness/backends/test_service_backends.py, ipfs_kit_py/docs/runtime_readiness/backend_service_receipts
- Interfaces: BackendAdapter@2, BackendCertificationReceipt@1
- Allow concurrent with: KITA-012, KITA-016, KITA-020, KITA-024, KITA-028, KITA-032, KITA-036, KITA-041
- Conflict policy: Own IPFS/cluster, S3-compatible and Iroh adapter/test/receipt scopes; family aliases share implementation but get distinct capability evidence.
- Preconditions: Common conformance kit and pinned Kubo/MinIO/Iroh service fixtures are available.
- Effects: IPFS/IPFS Cluster, S3/MinIO/DigitalOcean aliases and Iroh receive current pinned-service certification at their declared capability tiers.
- Evidence subset: Kubo, cluster, MinIO/S3 aliases, Iroh, multipart/range, CAS/idempotency, consistency, reconnect, credentials
- Symbolic first: true
- LLM context budget bytes: 20000
- Acceptance: Each promoted adapter instantiates inertly then passes declared live operations against a pinned service; aliases do not overclaim provider-specific behavior; MCP default manager no longer silently omits IPFS due to constructor mismatch; multipart/range/pagination/retry/cancellation/integrity/reconnect/secret-redaction pass; service unavailability produces blocked certification, not pass or fallback.
- Embedding query: certify ipfs cluster s3 minio digitalocean iroh backend service

## KITA-041 Implement or honestly classify decentralized, remote, archive, and legacy backend families

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backend-provider-lanes
- Depends on: KITA-038, KITA-039
- Goal id: KITA-G100
- Outputs: ipfs_kit_py/ipfs_kit_py/backends/provider_adapters.py, ipfs_kit_py/tests/runtime_readiness/backends/test_provider_backend_contracts.py, ipfs_kit_py/docs/runtime_readiness/backend_external_receipts
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/backends/test_provider_backend_contracts.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/backends/providers
- Parallel lane: kita-backend-providers
- Resource class: io-network-large
- Resource stage: implementation
- Estimated tokens: 36000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_kit_py/ipfs_kit_py/backends/provider_adapters.py, ipfs_kit_py/tests/runtime_readiness/backends/test_provider_backend_contracts.py, ipfs_kit_py/docs/runtime_readiness/backend_external_receipts
- Interfaces: BackendAdapter@2, BackendCertificationReceipt@1
- Allow concurrent with: KITA-012, KITA-016, KITA-020, KITA-024, KITA-028, KITA-032, KITA-036, KITA-040
- Conflict policy: Own provider adapter aggregate/new modules, provider-family fixtures and external receipts; do not edit shared registry after KITA-038.
- Preconditions: Common conformance kit exists; credentialed external runs are separately authorized and secretless.
- Effects: Filecoin/filecoin-pin/Lotus/Lassie, Storacha, HuggingFace, SSHFS/FTP, GDrive/GitHub, Parquet/Arrow, Estuary, local aliases and other inventoried providers gain an implementation or explicit narrower tier.
- Evidence subset: named provider families, runtime factory, configuration-only, external credentials, retries, rate limits, consistency, secrets
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Every inventoried type has an explicit runtime or typed unsupported/configuration-only result; no standalone kit or MCP client is mistaken for a canonical adapter; production/conditional promotion requires current provider receipt; credentials come only from authorized secret references and are absent from prompts/logs/receipts; rate/timeout/retry/idempotency/consistency semantics pass; Estuary-like registry-only entries cannot advertise storage.
- Embedding query: filecoin storacha huggingface sshfs ftp gdrive github parquet estuary backend

## KITA-042 Publish the joined backend support matrix and all-interface certification

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backend-release-matrix
- Depends on: KITA-013, KITA-017, KITA-021, KITA-025, KITA-029, KITA-033, KITA-037, KITA-040, KITA-041
- Goal id: KITA-G100
- Outputs: ipfs_kit_py/docs/runtime_readiness/backend_support_matrix.md, ipfs_kit_py/docs/runtime_readiness/backend_support_manifest.json, ipfs_kit_py/tests/runtime_readiness/backends/test_joined_backend_matrix.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/backends/test_joined_backend_matrix.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/backends/matrix
- Parallel lane: kita-backend-matrix
- Resource class: io-network-large
- Resource stage: validation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/docs/runtime_readiness/backend_support_matrix.md, ipfs_kit_py/docs/runtime_readiness/backend_support_manifest.json, ipfs_kit_py/tests/runtime_readiness/backends/test_joined_backend_matrix.py
- Interfaces: BackendSupportManifest@1, BackendCertificationReceipt@1
- Allow concurrent with:
- Conflict policy: Own support matrix/manifest/joined tests only; cannot promote a tier by editing prose and cannot patch adapters.
- Preconditions: Every subsystem, interface, auth and backend-family receipt dependency is current.
- Effects: Users and routing logic consume one machine-readable honest backend capability/tier manifest.
- Evidence subset: all registry types/aliases, operations, live tier, receipt freshness, interfaces, auth, WAL, replication, GraphRAG, ARC
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Every registry/documented name appears exactly once with canonical name, aliases, schema, factory, capabilities, tier, limitations, evidence CIDs and freshness; all advertised operations pass Python/CLI/MCP/MCP++ parity and required durability/auth/integrity semantics; stale/missing external evidence demotes or blocks rather than silently passes; routing never selects an unsupported capability or hidden fallback.
- Embedding query: joined backend support matrix capability tier interface certification

## KITA-043 Build the canonical benchmark harness, SLO manifest, and regression gate

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: performance-harness
- Depends on: KITA-004, KITA-007, KITA-018, KITA-022, KITA-026, KITA-034, KITA-038
- Goal id: KITA-G110
- Outputs: ipfs_kit_py/benchmarks/runtime_readiness/run.py, ipfs_kit_py/benchmarks/runtime_readiness/slo.py, ipfs_kit_py/tests/runtime_readiness/release/test_benchmark_harness.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/release/test_benchmark_harness.py && python benchmarks/runtime_readiness/run.py --profile ci-reference --check-schema
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/release/benchmark
- Parallel lane: kita-performance
- Resource class: cpu-io-large
- Resource stage: benchmark
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/benchmarks/runtime_readiness/run.py, ipfs_kit_py/benchmarks/runtime_readiness/slo.py, ipfs_kit_py/tests/runtime_readiness/release/test_benchmark_harness.py
- Interfaces: RuntimeBenchmarkManifest@1, RuntimeSLO@1, RegressionDecision@1
- Allow concurrent with:
- Conflict policy: Own benchmark/SLO/test files; benchmark may observe but never modify production configuration or relax durability.
- Preconditions: Initial baseline and representative canonical subsystem paths exist.
- Effects: Reproducible workloads measure committed TPS, p50/p95/p99, queues, memory, descriptors, tasks/threads, amplification and recovery across resource profiles.
- Evidence subset: metadata, small-object, mixed VFS, WAL group commit, ARC, GraphRAG, replica, interface, cold/warm, profile identity
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Harness pins environment/workload/seed/capability/durability and confidence; distinguishes accepted/committed/converged and cold/warm/cache paths; compares against immutable baseline with default 5% throughput and 10% p99 tolerances; absolute floors are reviewed and cannot be lowered in the failing change; benchmark errors/partial samples cannot pass; metrics have bounded cardinality and no secrets.
- Embedding query: benchmark harness committed tps p99 slo regression resource profile

## KITA-044 Optimize canonical transaction hot paths with bounded backpressure

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: performance-optimization
- Depends on: KITA-009, KITA-013, KITA-017, KITA-021, KITA-025, KITA-029, KITA-033, KITA-037, KITA-042, KITA-043
- Goal id: KITA-G110
- Outputs: ipfs_kit_py/ipfs_kit_py/core/performance.py, ipfs_kit_py/benchmarks/runtime_readiness/optimized_results.json, ipfs_kit_py/tests/runtime_readiness/release/test_backpressure_and_resources.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/release/test_backpressure_and_resources.py && python benchmarks/runtime_readiness/run.py --profile ci-reference --check
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/release/optimization
- Parallel lane: kita-optimization
- Resource class: cpu-io-large
- Resource stage: implementation
- Estimated tokens: 36000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/performance.py, ipfs_kit_py/benchmarks/runtime_readiness/optimized_results.json, ipfs_kit_py/tests/runtime_readiness/release/test_backpressure_and_resources.py
- Interfaces: BackpressureController@1, RuntimeSLO@1
- Allow concurrent with:
- Conflict policy: Cross-cutting hot-file optimization is serialized after all correctness joins; changes outside the new performance module require exact predicted-file amendment and renewed owning-subsystem tests.
- Preconditions: All joined correctness/security/backend/interface gates and benchmark harness pass before optimization.
- Effects: Measured WAL batching, I/O copies, lock contention, ARC fills, index updates, replica scheduling, connection pooling and adapter overhead improve under bounded queues.
- Evidence subset: profiles, batching, zero-copy/streaming, locks, async tasks, pools, queue bounds, fairness, cancellation, before/after
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Committed reference-profile transaction TPS is at least 2x bound-revision baseline or a reviewed evidence-backed ceiling/alternative target exists; no accepted workload regresses beyond policy; queues/memory/descriptors/tasks/threads remain bounded; overload returns explicit backpressure/deadline results; cancellation and fairness pass; fsync/auth/integrity/replication/consistency settings are identical before and after.
- Embedding query: optimize transaction tps wal batching arc index replica backpressure bounded queues

## KITA-045 Run soak, chaos, crash, leak, security, and resource-exhaustion qualification

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-soak-chaos
- Depends on: KITA-042, KITA-044
- Goal id: KITA-G110
- Outputs: ipfs_kit_py/tests/runtime_readiness/release/test_soak_chaos.py, ipfs_kit_py/docs/runtime_readiness/soak_chaos_receipt.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/release/test_soak_chaos.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/release/soak
- Parallel lane: kita-soak
- Resource class: cpu-io-network-large
- Resource stage: validation
- Estimated tokens: 28000
- Implementation timeout seconds: 21600
- Predicted files: ipfs_kit_py/tests/runtime_readiness/release/test_soak_chaos.py, ipfs_kit_py/docs/runtime_readiness/soak_chaos_receipt.json
- Interfaces: SoakReceipt@1, ChaosSchedule@1
- Allow concurrent with:
- Conflict policy: Own release qualification tests/receipt only; discovered implementation defects reopen owning tasks.
- Preconditions: Joined backend matrix and optimized benchmark gates pass.
- Effects: Long mixed workloads exercise process kill, torn write, backend loss, partition, corrupt data, auth attacks, overload and restart while monitoring leaks and convergence.
- Evidence subset: soak duration/seed, crash, partition, backend loss, corrupt WAL/cache/index/replica, UCAN attacks, overload, leaks
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Zero acknowledged loss, duplicate non-idempotent effects, authorization bypass, path escape, unsafe execution, secret leak or false convergence occurs; recovery is bounded; all queues/resources return within reviewed tolerance after load; no unbounded thread/task/fd/memory growth; backend outage remains explicit; repeated seeded run has identity-equivalent semantic receipts.
- Embedding query: soak chaos crash leak security resource exhaustion recovery convergence

## KITA-046 Validate compatibility migration, wheel matrix, rollout, and rollback

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-candidate
- Depends on: KITA-009, KITA-013, KITA-017, KITA-021, KITA-025, KITA-029, KITA-033, KITA-037, KITA-042, KITA-045
- Goal id: KITA-G110
- Outputs: ipfs_kit_py/tests/runtime_readiness/release/test_release_candidate.py, ipfs_kit_py/docs/runtime_readiness/migration_and_rollback.md, ipfs_kit_py/docs/runtime_readiness/release_candidate_receipt.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/release/test_release_candidate.py
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/release/candidate
- Parallel lane: kita-release-candidate
- Resource class: cpu-io-network-large
- Resource stage: validation
- Estimated tokens: 30000
- Implementation timeout seconds: 21600
- Predicted files: ipfs_kit_py/tests/runtime_readiness/release/test_release_candidate.py, ipfs_kit_py/docs/runtime_readiness/migration_and_rollback.md, ipfs_kit_py/docs/runtime_readiness/release_candidate_receipt.json
- Interfaces: ReleaseCandidateReceipt@1, MigrationReceipt@1, RollbackReceipt@1
- Allow concurrent with:
- Conflict policy: Own release tests/docs/receipt only; cannot patch production behavior or alter prior evidence.
- Preconditions: Every subsystem join, backend matrix and soak/chaos gate passes on current tree.
- Effects: Old VFS/bucket/WAL/cache/GraphRAG/replica state migrates forward, minimal/extras wheels pass, staged rollout and rollback/forward-recovery are rehearsed.
- Evidence subset: schema/data migration, old fixtures, WAL replay, index rebuild, cache discard, policy migration, Python versions, minimal/extras wheels, rollback
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Supported old state migrates idempotently with preserved content/version/policy semantics; unsupported state fails before mutation with backup/recovery instructions; minimal core and each extra wheel pass supported Python matrix; rollback restores executable prior state or documented forward recovery without acknowledged loss; support manifest and docs match actual registry; no required lane skips or stale receipt satisfies candidate.
- Embedding query: release candidate compatibility migration wheel extras rollout rollback

## KITA-047 Emit the independent current-tree runtime-readiness release receipt

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: release-closeout
- Depends on: KITA-046
- Goal id: KITA-G110
- Outputs: ipfs_kit_py/docs/runtime_readiness/KITA_RELEASE_RECEIPT.json, ipfs_kit_py/tests/runtime_readiness/release/test_joined_release_receipt.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/release/test_joined_release_receipt.py && cd .. && python scripts/validate_ipfs_kit_runtime_readiness_board.py --check-all
- Board namespace: ipfs-kit-runtime-readiness-v1
- Bundle: ipfs-kit/runtime-readiness/release/closeout
- Parallel lane: kita-closeout
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_kit_py/docs/runtime_readiness/KITA_RELEASE_RECEIPT.json, ipfs_kit_py/tests/runtime_readiness/release/test_joined_release_receipt.py
- Interfaces: KITAReleaseReceipt@1
- Allow concurrent with:
- Conflict policy: Own only the immutable joined receipt/test; do not edit plan/objectives/taskboard/scheduler/validator or production code.
- Preconditions: KITA-046 and every transitive subsystem/security/interface/backend/performance gate are current and terminal.
- Effects: One independently checked receipt binds exact repositories, packages, capabilities, validations, metrics and limitations for release.
- Evidence subset: all goals/tasks, repository forest, contracts, VFS, buckets, GraphRAG, WAL, ARC, replicas, UCAN, interfaces, backends, TPS, soak, migration
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt binds exact accelerator/kit/datasets trees and overlays, operation/support manifests, every task validation/evidence CID, environment/toolchain, zero-safety-floor counters, backend tiers, interface parity, benchmark/SLO, soak/chaos and migration/rollback; all inputs are fresh and independently verified; any missing/stale/failed/conditional evidence remains explicit and prevents a broader claim; validator proves the complete 48-task/12-goal terminal DAG and no protected control artifact changed after KITA-000.
- Embedding query: independent current tree ipfs kit runtime readiness release receipt closeout
