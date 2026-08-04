# IPFS Kit Runtime Readiness Objective Heap

This is the durable goal/subgoal hierarchy for the `KITA-` program. The
normative design is `IPFS_KIT_RUNTIME_READINESS_PLAN.md`; executable work is
projected into `ipfs_kit_runtime_readiness.todo.md`.

Program invariants:

- one canonical service defines semantics for Python, CLI, MCP, and MCP++;
- only advertised capabilities are release obligations, and unsupported
  behavior is rejected explicitly before side effects;
- committed acknowledgement is bound to the declared WAL/backend crash model;
- UCAN and Profile D admission precede every protected dispatch;
- caches and indexes are reconstructible projections, never semantic or
  authorization authorities;
- every nested-repository change lands as a reviewed nested commit followed by
  an exact serialized parent-gitlink update;
- static/analytical/proof-guided repair precedes any bounded `llm_router`
  proposal; and
- completion requires current-tree crash, security, parity, backend, and
  performance evidence.

## KITA-G000 Deliver a correct, durable, authorized, interface-equivalent, high-throughput IPFS Kit

- Status: blocked
- Review only: true
- Parent:
- Depends on:
- Fib priority: 1
- Track: runtime-readiness
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/control
- Parallel lane: control
- Resource class: cpu-large
- Goal: Make every advertised VFS, bucket, GraphRAG, WAL, ARC, replica-policy, MCP++ UCAN, interface, packaging, and storage-backend capability satisfy one versioned contract with current correctness, durability, security, recovery, parity, and performance evidence.
- Subgoals: KITA-G010, KITA-G020, KITA-G030, KITA-G040, KITA-G050, KITA-G060, KITA-G070, KITA-G080, KITA-G090, KITA-G100, KITA-G110
- Evidence: KITA-G010, KITA-G020, KITA-G030, KITA-G040, KITA-G050, KITA-G060, KITA-G070, KITA-G080, KITA-G090, KITA-G100, KITA-G110
- Evidence criteria: Every child goal has a current-tree evidence bundle and the terminal joined release receipt binds their exact revisions and validation CIDs.
- Evidence source policy: Reviewed contracts and capability manifests define claims; tests, traces, indexes, solvers, and models provide bounded evidence but cannot promote support or authorize mutation alone.
- Outputs: docs/architecture/IPFS_KIT_RUNTIME_READINESS_PLAN.md, docs/architecture/ipfs_kit_runtime_readiness.objectives.md, docs/architecture/ipfs_kit_runtime_readiness.todo.md, config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json
- Predicted files: docs/architecture/IPFS_KIT_RUNTIME_READINESS_PLAN.md, docs/architecture/ipfs_kit_runtime_readiness.objectives.md, docs/architecture/ipfs_kit_runtime_readiness.todo.md, config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json
- Interfaces: ipfs_kit_py.runtime_readiness@1, KITAReleaseReceipt@1
- Validation: python scripts/validate_ipfs_kit_runtime_readiness_board.py --check-all
- Acceptance: All 48 tasks and 11 child goals are terminal; zero acknowledged-data-loss, duplicate-non-idempotent-effect, authorization-bypass, unsafe-deserialization, path-escape, secret-leak, false-backend-support, or semantic-interface-parity failures remain; declared performance floors pass without weakened durability, consistency, or security.
- Gap task: Aggregate independently validated child evidence and decide release; do not implement subsystem behavior at the root.
- Refinement: Prefer an explicit conditional, experimental, unsupported, blocked, or approval-required disposition over an unearned production claim.
- Embedding query: ipfs kit vfs bucket graphrag wal arc replication ucan backend parity throughput release
- AST query: KITAReleaseReceipt CanonicalStorageService OperationRegistry BackendCapabilityManifest
- Conflict policy: Root is review and evidence aggregation only; child goals own implementation and the terminal task owns the immutable joined receipt.

## KITA-G010 Freeze capability truth, canonical contracts, fixtures, and baselines

- Status: active
- Parent: KITA-G000
- Depends on:
- Fib priority: 1
- Track: foundations
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/foundations
- Parallel lane: kita-foundations
- Resource class: cpu-medium
- Goal: Freeze the repository/capability/backend/interface inventory, define one operation/result/error/evidence contract, build hermetic adversarial fixtures, and establish dependency/import/performance baselines before implementation.
- Evidence: KITA-001, KITA-002, KITA-003, KITA-004
- Evidence criteria: Inventory is exhaustive under a published policy, canonical records are finite and content addressed, fixtures reproduce confirmed defects, and baselines bind environment and workload identity.
- Evidence source policy: Git trees and explicit overlays are inventory authority; current runtime probes may establish capability availability but not semantic correctness.
- Outputs: ipfs_kit_py/docs/runtime_readiness, ipfs_kit_py/ipfs_kit_py/core/operation_contracts.py, ipfs_kit_py/tests/runtime_readiness/foundations
- Predicted files: ipfs_kit_py/docs/runtime_readiness/capability_manifest.json, ipfs_kit_py/ipfs_kit_py/core/operation_contracts.py, ipfs_kit_py/tests/runtime_readiness/foundations
- Interfaces: OperationRequest@1, OperationResult@1, StorageError@1, CapabilityManifest@1, RuntimeReadinessFixture@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/foundations
- Acceptance: Exact current revisions, duplicate implementations, 23 registered backend types, public interfaces, optional dependencies, test-gate exclusions, known P0 defects, crash points, and workload profiles are recorded; records reject forged or unbounded fields; exactly KITA-001 through KITA-004 are ready after the sealed control task.
- Gap task: Establish what exists, what is promised, and how it will be measured before selecting implementation locations.
- Refinement: Import success, registry presence, mock behavior, or documentation alone does not establish a working capability.
- Embedding query: ipfs kit inventory operation contract fixtures dependency import benchmark baseline
- AST query: OperationRequest OperationResult StorageError BackendTypeRegistry
- Conflict policy: Foundation tasks own new manifests/contracts/fixtures only and do not refactor live subsystem implementations.

## KITA-G020 Make the canonical VFS transactional and recoverable

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 2
- Track: vfs-core
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/vfs
- Parallel lane: kita-vfs
- Resource class: cpu-large
- Goal: Consolidate VFS semantics behind the canonical service and make path, namespace, mount, file, directory, rename, version, snapshot, concurrency, transaction, and recovery behavior correct.
- Evidence: KITA-005, KITA-006, KITA-007, KITA-008, KITA-009
- Evidence criteria: Reference-model, differential, path-security, concurrency, and crash traces bind every supported operation and state transition.
- Evidence source policy: The admitted operation contract defines expectations; historic VFS variants are observations until mapped or retired.
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs, ipfs_kit_py/tests/runtime_readiness/vfs
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/vfs, ipfs_kit_py/tests/runtime_readiness/vfs
- Interfaces: VFSService@1, VFSPathPolicy@1, VFSTransaction@1, VFSSnapshot@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/vfs
- Acceptance: Rename/move mutate or reject explicitly; path confinement, journal names, failure/event ordering, conditional writes, atomicity, version identity, cancellation, rollback, and restart pass; wrappers contain no competing storage logic.
- Gap task: Replace overlapping optimistic wrappers with one observed-mutation and recovery-aware VFS contract.
- Refinement: A returned success without an observed admitted state change is a contract failure.
- Embedding query: virtual filesystem path mount rename snapshot transaction recovery differential
- AST query: VFSManager CanonicalVFSService VFSTransaction VFSPathPolicy
- Conflict policy: New core modules precede a serialized cutover of legacy VFS managers and adapters.

## KITA-G030 Consolidate virtual-bucket lifecycle and policy management

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 2
- Track: buckets
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/buckets
- Parallel lane: kita-buckets
- Resource class: cpu-large
- Goal: Provide one backend-scoped bucket catalog and transactional lifecycle for CRUD, objects, metadata, quota, retention, encryption, tiering, CAR/import/export, cross-bucket query, deletion, and recovery.
- Evidence: KITA-010, KITA-011, KITA-012, KITA-013
- Evidence criteria: State-machine, saga/compensation, quota, placement, import/export, query, and crash receipts cover success and partial external effects.
- Evidence source policy: The canonical bucket contract and catalog generation define state; external placement observations require integrity verification before counting.
- Outputs: ipfs_kit_py/ipfs_kit_py/core/buckets, ipfs_kit_py/tests/runtime_readiness/buckets
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/buckets, ipfs_kit_py/tests/runtime_readiness/buckets
- Interfaces: BucketCatalog@1, BucketPolicy@1, BucketTransaction@1, BucketManifest@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/buckets
- Acceptance: Equal bucket names on different backends do not collide; multi-store creation and policy update compensate or recover; deletion fences writes/replicas; quota and exactly-one-primary invariants hold; export/import and cross-bucket query bind snapshots and authorization.
- Gap task: Generalize the strongest tiering/receipt behavior and retire five competing state planes without losing compatibility.
- Refinement: Configured policy is desired state, not proof that the behavior was enforced.
- Embedding query: virtual bucket catalog policy quota tiering car import export recovery
- AST query: BucketManager BucketVFSManager UnifiedBucketInterface IrohBucketTieringManager
- Conflict policy: Catalog/policy modules are new-file work; legacy-manager cutover occurs only after differential migration evidence.

## KITA-G040 Make GraphRAG durable, safe, indexed, and interface-equivalent

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 2
- Track: graphrag
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/graphrag
- Parallel lane: kita-graphrag
- Resource class: cpu-ml-medium
- Goal: Establish one versioned GraphRAG engine with safe persistence, correct history, restart rehydration, incremental/clean equivalence, a pluggable ANN vector index, deterministic hybrid retrieval, provenance, and package/CLI/MCP parity.
- Evidence: KITA-014, KITA-015, KITA-016, KITA-017
- Evidence criteria: Safe-serialization, restart, crash, migration, recall, deterministic-query, poisoning, resource, and differential-interface receipts share the same index generation.
- Evidence source policy: Durable typed records are authoritative for index state; graph/vector projections rank admitted records and remain non-authoritative for program or access-control semantics.
- Outputs: ipfs_kit_py/ipfs_kit_py/graphrag, ipfs_kit_py/tests/runtime_readiness/graphrag
- Predicted files: ipfs_kit_py/ipfs_kit_py/graphrag, ipfs_kit_py/tests/runtime_readiness/graphrag
- Interfaces: GraphRAGService@1, GraphRAGIndexManifest@1, VectorIndex@1, GraphRAGQueryResult@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/graphrag
- Acceptance: No pickle or executable cache load remains; old versions are correct; all nodes/edges/embeddings/results survive restart or rebuild; ANN recall@10 is at least 0.95 against exact search on the pinned corpus; model/dimension mismatch fails closed; duplicate engines are adapters or retired; all interfaces return one schema.
- Gap task: Replace eager, unsafe, split-state, brute-force and competing GraphRAG paths with one reconstructible index service.
- Refinement: An ANN or hybrid score is a ranking result, not a proof or authorization decision.
- Embedding query: graphrag vector index durable restart safe serialization provenance hybrid parity
- AST query: GraphRAGSearchEngine VectorIndex GraphRAGIndexManifest GraphRAGService
- Conflict policy: The new engine and migration tools land before one serialized wrapper/tool-registry cutover.

## KITA-G050 Establish a real durable WAL and transaction protocol

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 2
- Track: wal-transactions
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/wal
- Parallel lane: kita-wal
- Resource class: io-large
- Goal: Define and implement record, acknowledgement, fsync, transaction, replay, checkpoint, compaction, archive, corruption, cancellation, and shutdown semantics across the existing WAL/journal variants.
- Evidence: KITA-018, KITA-019, KITA-020, KITA-021
- Evidence criteria: A fault-injection matrix binds every durability boundary and proves committed-only, idempotent, duplicate-safe recovery with valid-prefix preservation.
- Evidence source policy: Durable bytes plus verified backend effects determine recovery state; buffered metadata, random/mock handlers, or queued work cannot establish commit.
- Outputs: ipfs_kit_py/ipfs_kit_py/core/wal, ipfs_kit_py/tests/runtime_readiness/wal
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/wal, ipfs_kit_py/tests/runtime_readiness/wal
- Interfaces: WALRecord@1, WALTransaction@1, WALCheckpoint@1, WALRecoveryReceipt@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/wal
- Acceptance: Transaction IDs and commit/abort markers persist correctly; mutations do not precede required durable intent; file and parent-directory durability is tested; checkpoints never hide later appends; archives are confirmed before deletion; replay is idempotent; background workers stop; zero acknowledged writes are lost in the declared crash matrix.
- Gap task: Consolidate multiple incomplete journals/WALs behind one versioned protocol and bounded compatibility adapters.
- Refinement: A buffered begin marker or successful metadata write is not a committed storage transaction.
- Embedding query: write ahead log fsync transaction commit replay checkpoint compaction crash
- AST query: FilesystemJournal StorageWriteAheadLog DurableWAL WALTransaction WALRecoveryReceipt
- Conflict policy: New WAL records/reference implementation land first; legacy adapters are migrated in serialized dependency order.

## KITA-G060 Prove Adaptive Replacement Cache invariants and coherence

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 3
- Track: arc-cache
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/arc
- Parallel lane: kita-arc
- Resource class: cpu-memory-large
- Goal: Make ARC size, ghost-list, adaptation, concurrency, invalidation, serialization, persistence, metrics, and stampede behavior match a reference model under entry and byte budgets.
- Evidence: KITA-022, KITA-023, KITA-024, KITA-025
- Evidence criteria: Deterministic and randomized state-machine traces prove ARC invariants; concurrent, restart, corrupt-entry, and cross-subsystem coherence tests bind exact generations.
- Evidence source policy: Cache state is a reconstructible performance projection; every hit is revalidated against content/version, policy-sensitive scope, serializer, and generation.
- Outputs: ipfs_kit_py/ipfs_kit_py/cache/arc, ipfs_kit_py/tests/runtime_readiness/arc
- Predicted files: ipfs_kit_py/ipfs_kit_py/cache/arc, ipfs_kit_py/tests/runtime_readiness/arc
- Interfaces: AdaptiveReplacementCache@1, CacheKey@1, CacheGeneration@1, CacheMetrics@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/arc
- Acceptance: Current size equals live T1 plus T2 size and never exceeds capacity; T1/T2/B1/B2 are pairwise disjoint; ghost lists retain no values; adaptive target is bounded; updates/ghost hits account bytes; concurrent operations are linearizable or explicitly single-owner; stale/corrupt entries safely miss; default CI executes the suite.
- Gap task: Separate and correct the oversized ARC implementation, then integrate a concurrency-safe owner rather than copying an unguarded mixin.
- Refinement: Cache speed cannot conceal a stale, unauthorized, inconsistent, or corrupt value.
- Embedding query: adaptive replacement cache t1 t2 b1 b2 ghost size concurrency invalidation
- AST query: ARCCache AdaptiveReplacementCache CacheGeneration CacheMetrics
- Conflict policy: ARC core extraction owns a new package; compatibility exports and cache-manager integration are serialized later.

## KITA-G070 Make replica placement and reconciliation converge truthfully

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 3
- Track: replication
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/replication
- Parallel lane: kita-replication
- Resource class: io-large
- Goal: Define valid desired-state replica policies, deterministic placement, idempotent reconciliation, integrity verification, anti-entropy, compensation, and cross-subsystem convergence.
- Evidence: KITA-026, KITA-027, KITA-028, KITA-029
- Evidence criteria: Policy validation, placement, partition, backend-loss, corrupt/divergent replica, rebalancing, compensation, and convergence receipts distinguish every lifecycle state.
- Evidence source policy: Only verified durable replicas count toward redundancy; planned, pending, queued, copied-unverified, stale, and failed placements remain separate.
- Outputs: ipfs_kit_py/ipfs_kit_py/core/replication, ipfs_kit_py/tests/runtime_readiness/replication
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/replication, ipfs_kit_py/tests/runtime_readiness/replication
- Interfaces: ReplicaPolicy@1, PlacementPlan@1, ReplicaState@1, ReconciliationReceipt@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/replication
- Acceptance: Policy ordering/disjointness and available distinct writable backends validate before mutation; duplicate/shadowed methods are removed; pending work never counts as redundancy; placement is deterministic; integrity-verified anti-entropy repairs drift; partial external effects compensate or remain recoverable; bucket/WAL/index/cache state converges.
- Gap task: Generalize the strongest tiering reconciler and replace reporting-only or test-key-special-cased replication paths.
- Refinement: Desired count and scheduled copy count are not verified replica count.
- Embedding query: replica policy placement reconciliation anti entropy integrity failure domains convergence
- AST query: ReplicationPolicy PlacementPlan ReplicaReconciler ReplicaState
- Conflict policy: Policy/placement/reconciler modules are file-disjoint until the joined cross-subsystem integration task.

## KITA-G080 Enforce MCP++ UCAN and datasets Profile D before dispatch

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 1
- Track: mcplusplus-ucan
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/mcplusplus-ucan
- Parallel lane: kita-auth
- Resource class: cpu-security
- Goal: Repair MCP++ construction and protocol advertisement, then enforce signed attenuated UCAN and canonical datasets Profile D decisions with revocation, replay protection, audit, and transport-invariant denial before every protected handler.
- Evidence: KITA-030, KITA-031, KITA-032, KITA-033
- Evidence criteria: Startup/profile, signed-token, proof-chain, policy, revocation, replay, downgrade, confused-deputy, transport-parity, and dispatch-spy receipts are current and content bound.
- Evidence source policy: Cryptographic verification and admitted policy decisions authorize dispatch; missing validators, permissive fallbacks, mocks, or advisory calls never do.
- Outputs: ipfs_kit_py/ipfs_kit_py/mcp_server, ipfs_kit_py/tests/runtime_readiness/mcplusplus
- Predicted files: ipfs_kit_py/ipfs_kit_py/mcp_server, ipfs_kit_py/tests/runtime_readiness/mcplusplus
- Interfaces: MCPPlusPlusServer@1, UCANVerifier@1, AuthorizationDecision@1, ProfileDPolicyProvider@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/mcplusplus tests/test_profile_d_policy.py ipfs_kit_py/mcp_server/tests_e2e_interop.py
- Acceptance: MCP++ constructs on stdio/HTTP/P2P and advertises the canonical profile; every protected tool maps to exact resource/ability; missing/forged/tampered/expired/not-yet-valid/revoked/replayed/over-broad/cross-tenant/downgraded/denied requests dispatch zero handlers; validators fail closed; admitted decision CIDs and errors are transport invariant; audit contains no secrets.
- Gap task: Convert UCAN/Profile D from advisory endpoints into the single mandatory dispatcher gate.
- Refinement: Authentication, token parsing, and policy scoring are not authorization unless the exact admitted decision gates the exact effect.
- Embedding query: mcp plus plus ucan profile d delegation attenuation revocation replay dispatch authorization
- AST query: MCPServer EventDAGStore UCANValidator AuthorizationDecision ProfileDPolicyProvider
- Conflict policy: Startup/registry work precedes UCAN records and policy adapters; dispatcher cutover is serialized after both.

## KITA-G090 Unify package, Python, CLI, MCP, and installed-wheel contracts

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 2
- Track: interface-parity
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/interfaces
- Parallel lane: kita-interfaces
- Resource class: cpu-medium
- Goal: Route every public interface through one operation registry and canonical service, make sync/async/results/errors identical, and make version, exports, dependencies, extras, entry points, and lazy imports coherent.
- Evidence: KITA-034, KITA-035, KITA-036, KITA-037
- Evidence criteria: Public-contract snapshots, generated adapter checks, cold-import subprocesses, minimal/per-extra wheels, and cross-interface differential fixtures bind exact metadata.
- Evidence source policy: The reviewed registry/schema is interface authority; historical wrappers and package manifests are observations until reconciled.
- Outputs: ipfs_kit_py/ipfs_kit_py/core/operation_registry.py, ipfs_kit_py/ipfs_kit_py/cli, ipfs_kit_py/ipfs_kit_py/mcp_server/tools, ipfs_kit_py/tests/runtime_readiness/interfaces
- Predicted files: ipfs_kit_py/ipfs_kit_py/core/operation_registry.py, ipfs_kit_py/ipfs_kit_py/cli, ipfs_kit_py/ipfs_kit_py/mcp_server/tools, ipfs_kit_py/pyproject.toml, ipfs_kit_py/setup.py, ipfs_kit_py/requirements.txt
- Interfaces: OperationRegistry@1, PythonAdapter@1, CLIAdapter@1, MCPToolAdapter@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/interfaces
- Acceptance: Runtime and wheel versions match; dependency/marker projections have zero unexplained drift; minimal core and each extra install/import independently; cold root/registry imports perform no heavy optional import, network/process/model/user-state action; Python/CLI/MCP/MCP++ fixtures have 100% canonical result/error/CID/side-effect parity.
- Gap task: Replace duplicated interface logic and divergent packaging metadata with generated or mechanically checked projections.
- Refinement: Transport-specific request IDs and timings may differ; semantic payloads and effects may not.
- Embedding query: python cli mcp package import dependency version entrypoint contract parity lazy
- AST query: OperationRegistry PythonAdapter CLIAdapter MCPToolAdapter __getattr__
- Conflict policy: Registry and packaging work are isolated; CLI and MCP adapters proceed in parallel then join in one parity cutover.

## KITA-G100 Certify every registered and documented storage backend honestly

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G010
- Fib priority: 2
- Track: backends
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/backends
- Parallel lane: kita-backends
- Resource class: io-network-large
- Goal: Make registry names, aliases, schemas, runtime factories, declared capabilities, health, secrets, adapter semantics, interface exposure, and tests bijective, then certify each backend at an honest support tier.
- Evidence: KITA-038, KITA-039, KITA-040, KITA-041, KITA-042
- Evidence criteria: One BackendSpec per type/alias plus hermetic, pinned-service, or credentialed-external conformance receipts; unsupported operations reject before side effects.
- Evidence source policy: Current conformance at the declared tier establishes support; registry presence, schema-only plugins, mocks, silent omission, or fallback do not.
- Outputs: ipfs_kit_py/ipfs_kit_py/backends, ipfs_kit_py/tests/runtime_readiness/backends, ipfs_kit_py/docs/runtime_readiness/backend_support_matrix.md
- Predicted files: ipfs_kit_py/ipfs_kit_py/backends, ipfs_kit_py/tests/runtime_readiness/backends, ipfs_kit_py/docs/runtime_readiness/backend_support_matrix.md
- Interfaces: BackendSpec@1, BackendAdapter@1, BackendCapability@1, BackendCertificationReceipt@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/backends
- Acceptance: Every registered or documented backend name maps unambiguously to a schema, plugin, optional runtime factory, capability set, health behavior, secret model, and tier; all production/conditional adapters pass required operations and fault semantics; no legacy plugin is called through a missing factory; no advertised IPFS/backend is silently omitted; external evidence is current and secretless.
- Gap task: Reconcile the 23-type registry, smaller adapter registry, schema naming, standalone kits, MCP clients, and missing factories into one honest support matrix.
- Refinement: It is acceptable to classify a backend configuration-only or unsupported; it is not acceptable to claim storage behavior that cannot be invoked and verified.
- Embedding query: storage backend registry adapter factory capability schema conformance ipfs s3 iroh filecoin
- AST query: BackendTypeRegistry BackendPlugin BackendManager BackendAdapter BackendSpec
- Conflict policy: Shared BackendSpec/conformance kit land first; backend-family lanes own disjoint adapters and fixtures; the matrix join serializes registry changes.

## KITA-G110 Meet TPS/resource objectives and release through current-tree gates

- Status: active
- Parent: KITA-G000
- Depends on: KITA-G020, KITA-G030, KITA-G040, KITA-G050, KITA-G060, KITA-G070, KITA-G080, KITA-G090, KITA-G100
- Fib priority: 1
- Track: performance-release
- Priority: P0
- Bundle: ipfs-kit/runtime-readiness/release
- Parallel lane: kita-release
- Resource class: cpu-io-large
- Goal: Benchmark canonical transaction paths, optimize measured bottlenecks without weakening contracts, enforce backpressure/resource bounds, run soak/chaos/migration/rollback, and emit an independent joined release receipt.
- Evidence: KITA-043, KITA-044, KITA-045, KITA-046, KITA-047
- Evidence criteria: Reproducible throughput/latency/resource manifests, before/after profiles, soak/chaos receipts, compatibility migration, rollback, and an independent current-tree joined validator.
- Evidence source policy: Only bound benchmark and release receipts from the admitted workload/environment count; cached, partial, simulated, skipped, or stale runs are labeled and cannot satisfy production floors.
- Outputs: ipfs_kit_py/benchmarks/runtime_readiness, ipfs_kit_py/tests/runtime_readiness/release, ipfs_kit_py/docs/runtime_readiness/release
- Predicted files: ipfs_kit_py/benchmarks/runtime_readiness, ipfs_kit_py/tests/runtime_readiness/release, ipfs_kit_py/docs/runtime_readiness/release
- Interfaces: RuntimeBenchmarkManifest@1, RuntimeSLO@1, SoakReceipt@1, KITAReleaseReceipt@1
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/release && python benchmarks/runtime_readiness/run.py --profile ci-reference --check
- Acceptance: Committed reference throughput is at least 2x the bound-revision baseline or has a reviewed evidence-backed ceiling/target; accepted throughput regresses no more than 5% and p99 no more than 10%; queues, memory, descriptors, threads and tasks remain bounded; zero correctness/security/durability relaxation occurs; crash/chaos/migration/rollback and all declared backend/interface gates pass; KITA-047 binds every current evidence CID.
- Gap task: Turn correct subsystem behavior into reproducibly high-throughput, bounded, operable, releasable behavior.
- Refinement: A benchmark that omits fsync, authorization, integrity, or declared replication cannot stand in for committed transaction TPS.
- Embedding query: ipfs kit transaction tps latency p99 backpressure soak chaos release rollback
- AST query: RuntimeBenchmarkManifest RuntimeSLO BackpressureController KITAReleaseReceipt
- Conflict policy: Benchmark harness lands before optimization; hot-path edits are serialized by predicted-file conflicts; release tasks are join points and do not weaken prior gates.
