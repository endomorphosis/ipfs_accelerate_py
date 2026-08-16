# Logic-Governed Semantic Work Fabric objective heap

Machine-ingestible objective state for board namespace
`logic-governed-semantic-work-fabric-actual-v1`. The executable projection is
`docs/architecture/logic_governed_semantic_work_fabric.todo.md`.

## Goal tree

```text
LGSWF-G000  Evidence-backed semantic work fabric fixed point
|-- LGSWF-G010  A: inventory, authority and interface freeze
|-- LGSWF-G020  B: operational world-state overlay
|-- LGSWF-G030  C: semantic goal/task bindings and completion
|-- LGSWF-G040  D: composite semantic work and conflict graphs
|-- LGSWF-G050  E: deterministic safe parallel frontier
|-- LGSWF-G060  F: resource-aware multi-stage scheduling
|-- LGSWF-G070  G: multi-supervisor coordination
|-- LGSWF-G080  H: daemon packet and checkpoint protocol
|-- LGSWF-G090  I: immutable revision, refill and plan doctor
|-- LGSWF-G100  J: closed-loop semantic refresh
|-- LGSWF-G110  K: bounded fixed-point convergence
|-- LGSWF-G120  L: scheduling and coordination observability
|-- LGSWF-G130  M: fault and adversarial qualification
|-- LGSWF-G140  N: parallelism and resource benchmark
`-- LGSWF-G150  O: content-addressed qualification release
```

## LGSWF-G000 Evidence-backed semantic work fabric fixed point

- Status: active
- Parent:
- Depends on:
- Priority: P0
- Track: lgswf-root
- Goal: Compose canonical datasets semantic authorities with the accelerator-owned DuckDB + Quack operational control plane and optional non-authoritative DuckLake analytics/history in one continuous, deterministic, resource-bounded and evidence-backed supervisor loop.
- Completion contract: Every required child goal is accepted against the same release tree, canonical semantic root, accepted plan revision and policy; DuckDB transactional authority and any multi-process Quack state-owner path verify; optional DuckLake never gates scheduling or acceptance; all mandatory tasks and blocking obligations are resolved; claims and merge queue are empty; receipts and seals verify; qualification emits separate embedded, continuous-Quack and live-DuckLake go/no-go decisions.
- Evidence: lgswf/release-manifest@1, lgswf/qualification-report@1, lgswf/fixed-point-receipt@1
- Acceptance criteria: exact-revisions; canonical-authority-map; accepted-child-goals; current-semantic-root; settled-claims-and-merge; verified-release
- Outputs: docs/architecture/LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_PLAN.md, docs/architecture/logic_governed_semantic_work_fabric.todo.md, docs/architecture/logic_governed_semantic_work_fabric.objectives.md, data/agent_supervisor/logic_governed_semantic_work_fabric/release
- Validation: python3 scripts/validate_logic_governed_semantic_work_fabric_board.py --check-all
- Acceptance: No worker/model self-approval, stale accepted result, duplicate accepted effect, unresolved mandatory obligation, or weakened completion criterion remains.
- Gap task: LGSWF-001 through LGSWF-141

## LGSWF-G010 A: inventory, authority and interface freeze

- Status: active
- Parent: LGSWF-G000
- Depends on:
- Priority: P0
- Track: inventory-freeze
- Goal: Reconcile all checked-out revisions, classify every relevant implementation including the nested datasets DuckLake package, verify the package DAG, freeze cross-authority interfaces, and establish a verified current semantic root plus accepted Plan Revision R2.
- Completion contract: Full ordered revision ledgers match sealed digests; the external source recheck reproduces both HEAD/tree pairs, tracked patches and untracked content-manifest roots; C1 is bound as `301d7318868bbc238abb6862d38c77253d5653b0` / `b08b08e17c4a10c8f0627daf4e1eb4348ef8ad3f` while any later integrated checkpoint is recorded separately; the designated bootstrap populations pass 14/14 operational, 12/12 intent, 62/62 semantic/proof guard, 89/89 coordination, 7/7 live-seal gate and 34/34 seal tests with zero skips/failures; the clean final source emits a self-addressed qualification receipt and no legacy-Markdown shutdown/orphan descendant has authority; inventory and authority/DAG/interface artifacts validate; DuckDB + Quack are frozen as the operational control plane, with DuckDB owning transactional records/schema/CAS/fencing and Quack the mandatory multi-reader/multi-writer transport/exclusive state-owner boundary, while DuckLake is optional non-authoritative typed projection/query storage with no direct accelerator ATTACH/raw-SQL/catalog/credential path; the current Quack `install-required`/`import_only_insufficient` state and exact DuckDB/Quack profile gap remain control-plane no-go evidence, while DuckLake/httpfs pins and catalog/profile/binding remain a separate downstream no-go; import provenance resolves under the selected checkouts; a datasets-built semantic root is verified and persisted; R2 binds exact semantic/capsule/contract/obligation identities or a typed no-go is issued.
- Evidence: lgswf/current-state@1, lgswf/source-recheck@1, lgswf/bootstrap-qualification@1, lgswf/legacy-shutdown-quarantine@1, lgswf/authority-map@1, lgswf/package-dag@1, lgswf/interface-freeze@1, StateRepository/QuackStateRepository/Client/Server/identity/command/config signatures, datasets quack_security scoped-authorization signature, datasets/semantic-state-root@1, datasets/ducklake-capability-probe@1, plan-revision@1
- Acceptance criteria: revision-ledgers; source-content-manifests; c1-commit-tree-lineage; bootstrap-218-of-218-zero-skip-fail; clean-self-addressed-qualification; clean-descendant-shutdown; classified-inventory; state-repository-interface-freeze; server-side-scoped-authorization-freeze; duckdb-quack-control-plane-ducklake-projection-boundary; no-unresolved-authority-collision; verified-semantic-root; accepted-r2
- Outputs: config/logic_governed_semantic_work_fabric_baseline.json, config/logic_governed_semantic_work_fabric_scheduler.json, test/api/test_agent_supervisor_intent_repository.py, docs/architecture/logic_governed_semantic_work_fabric_inventory, data/agent_supervisor/logic_governed_semantic_work_fabric/evidence/semantic-baseline.json
- Validation: python3 scripts/validate_logic_governed_semantic_work_fabric_board.py --check-all
- Acceptance: Remote-only APIs are not adopted, operational fields remain outside datasets roots, stale legacy-Markdown workers and results remain quarantined, every later task is rebound or held, and continuous multi-supervisor operation remains NO-GO pending the later promotion gates.
- Gap task: LGSWF-006, LGSWF-001, LGSWF-002, LGSWF-003, LGSWF-004, LGSWF-005

## LGSWF-G020 B: operational world-state overlay

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G010
- Priority: P0
- Track: world-overlay
- Goal: Implement a content-addressed `SupervisorWorldSnapshot@1`, separately verified construction, freshness/consistency admission, mutation-free `SupervisorWorldView`, and a non-authoritative observation of optional DuckLake projection health.
- Completion contract: Snapshot identities and exclusions are deterministic; each required component records current/stale/unavailable/inconsistent/quarantined; repository/tree/plan/population/semantic-generation/policy disagreement fails scheduling; DuckLake projection health is read from a DuckDB-recorded receipt, cannot grant authority, and its absence/outage never makes an otherwise valid snapshot unschedulable; all required read queries are pure.
- Evidence: lgswf/world-snapshot@1, lgswf/world-view@1, lgswf/world-consistency-tests@1
- Acceptance criteria: schema; authority-specific-builders; fail-closed-freshness; optional-projection-observation; pure-read-model
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/world_snapshot.py, ipfs_accelerate_py/agent_supervisor/semantic_state/world_view.py
- Validation: focused world snapshot/view tests
- Acceptance: No raw source, prompt, credential, model response, mutable local path, or arbitrary provider payload is embedded.
- Gap task: LGSWF-010, LGSWF-011, LGSWF-012, LGSWF-013

## LGSWF-G030 C: semantic goal/task bindings and completion

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G020
- Priority: P0
- Track: binding-completion
- Goal: Bind goals, subgoals, tasks and attempts to canonical semantic references, scopes, effects, obligations, evidence, and explicit completion contracts.
- Completion contract: Binding validation rejects copied/reinterpreted semantic payloads and stale roots; goal completion uses observable criteria, not task-count completion; task acceptance requires worker, validation, proof, merge and canonical-refresh gates; provisional roots cannot publish as canonical.
- Evidence: lgswf/semantic-work-binding@1, lgswf/completion-contract@1, lgswf/provisional-state-gate@1
- Acceptance criteria: binding-schema; goal-contracts; task-gates; provisional-canonical-separation
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/work_binding.py, ipfs_accelerate_py/agent_supervisor/objectives/completion_contracts.py
- Validation: focused binding/completion/provisional-state tests
- Acceptance: Every executable entity has an exact current binding and only the supervisor can accept a result.
- Gap task: LGSWF-020, LGSWF-021, LGSWF-022, LGSWF-023

## LGSWF-G040 D: composite semantic work and conflict graphs

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G030
- Priority: P0
- Track: graph
- Goal: Compose all required dependency, scope, lifecycle, proof, merge, invalidation and conflict edges without collapsing their authorities or meanings.
- Completion contract: Every edge binds evidence/root/revision/invalidation; dependency and conflict are distinct; exact scopes enable concurrency while unknowns serialize conservatively; fixed-point critical-path metrics are deterministic.
- Evidence: lgswf/semantic-work-graph@1, lgswf/conflict-graph@1, lgswf/graph-metrics@1
- Acceptance criteria: all-edge-kinds; authority-provenance; conflict-safety; fixed-point-metrics
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/semantic_work_graph.py, ipfs_accelerate_py/agent_supervisor/core/conflict_graph.py
- Validation: graph composition, conflict and determinism tests
- Acceptance: Shared reads do not create false dependencies and overlapping/opaque writes cannot be concurrently admitted.
- Gap task: LGSWF-030, LGSWF-031, LGSWF-032, LGSWF-033

## LGSWF-G050 E: deterministic safe parallel frontier

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G040
- Priority: P0
- Track: frontier
- Goal: Select the largest useful deterministic conflict-free ready antichain under semantic, lifecycle, resource, provider, proof and merge constraints, with safe split/coalesce/rewire/speculation proposals.
- Completion contract: All twelve readiness predicates are enforced; identical world/policy yields identical decisions; hard constraints cannot be overridden by an LLM; every selection/rejection has a receipt; immutable lifecycle records are preserved.
- Evidence: lgswf/frontier-planner@1, lgswf/frontier-decision@1, lgswf/plan-transform-proposal@1
- Acceptance criteria: readiness; conflict-free-antichain; bounded-optimizer; deterministic-tie-break; safe-transformations
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/conflict_free_frontier.py
- Validation: optimal-small, deterministic-large, readiness and transformation tests
- Acceptance: The selected frontier contains no inadmissible pair and maximizes the documented bounded objective for its algorithm class.
- Gap task: LGSWF-040, LGSWF-041, LGSWF-042, LGSWF-043

## LGSWF-G060 F: resource-aware multi-stage scheduling

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G050
- Priority: P0
- Track: resources
- Goal: Extend the current scheduler with complete resource vectors, leased reservations, evidence-separated estimates, cache/single-flight locality, independent backpressure including the DuckDB + Quack-control-plane-outbox-to-DuckLake projection stage, and safe cancellation/preemption.
- Completion contract: Hard resources never overcommit; reservations bind task/attempt/supervisor/daemon and reclaim only after fenced expiry; predictions never overwrite observations; saturated stages do not globally stall compatible work; projection batches are bounded to 5,000 rows, 16 MiB and 10 seconds, perform no lake/network call inside a DuckDB transaction, and replay independently from the authoritative DuckDB cursor.
- Evidence: lgswf/resource-vector@1, lgswf/resource-reservation@1, lgswf/resource-estimate@1, lgswf/backpressure@1
- Acceptance criteria: full-vector; leased-reservations; hard-capacity; historical-estimates; single-flight; stage-backpressure; ducklake-projection-isolation-and-replay; safe-preemption
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/work_cache_coordinator.py
- Validation: scheduler/resource/cache/backpressure tests
- Acceptance: CPU-only analysis proceeds during provider/prover saturation and mutation dispatch responds to merge pressure.
- Gap task: LGSWF-050, LGSWF-051, LGSWF-052, LGSWF-053

## LGSWF-G070 G: multi-supervisor coordination

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G060
- Priority: P0
- Track: supervisor-fabric
- Goal: Extend the sealed multi-supervisor runner with capabilities, role-aware partitioning, a mandatory Quack multi-reader/multi-writer transport and exclusive state-owner boundary over DuckDB records/schema/CAS/fencing, epoch failover, eligible work stealing and exactly-once logical acceptance.
- Completion contract: Capability observations cannot grant authority; embedded bootstrap remains one DuckDB writer; continuous mutation requires exact pinned DuckDB/Quack 1.5.5 capability/profile evidence and Quack-backed task/coordination/attempt/provider/effect/validation/CAS repositories; configured-board and runner paths propagate the live endpoint and `StateServerIdentity`, own server start/stop, and pass an independent remote readiness probe; one fenced state owner controls each mutable shard; failover advances epoch; stale coordinators cannot commit; transfers meet all eligibility rules; one attempt alone is accepted for the idempotency tuple.
- Evidence: lgswf/supervisor-capability@1, lgswf/coordination-epoch@1, lgswf/work-partition@1, lgswf/logical-acceptance@1, sealed Quack capability/profile/repository receipts
- Acceptance criteria: capabilities; exact-1.5.5-profile; quack-state-owner; live-endpoint-and-server-identity-propagation; deterministic-start-stop; independent-remote-readiness; quack-backed-operational-repositories; roles; fenced-shards; failover; explicit-partitions; safe-steal; exact-once-acceptance
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py
- Validation: coordination, failover, stealing and duplicate-attempt tests
- Acceptance: No hidden in-memory dependency, direct multi-process DuckDB file writer, unsafe current daemon Quack fallback, or stale process identity can become accepted authority.
- Gap task: LGSWF-060, LGSWF-061, LGSWF-062

## LGSWF-G080 H: daemon packet and checkpoint protocol

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G070
- Priority: P0
- Track: daemon-protocol
- Goal: Extend one canonical daemon packet with complete semantic/operational binding, explicit lifecycle, durable checkpoints, typed stale-stop behavior, and sealed Quack transport/state-owner bindings for multi-process readers and writers.
- Completion contract: Packet admission validates every required field and Quack service/fence identity when multi-process; intent, transaction, Portal, coordination and execution paths converge on one canonical `StateRepository`/`QuackStateRepository` command gateway with datasets `quack_security` server-side scoped authorization; transitions are closed; checkpoints bind all attempt/progress/effect data but never complete a task; stale/cancelled/already-accepted daemons stop before further effects; direct DuckDB file access remains limited to the qualified one-writer embedded bootstrap.
- Evidence: lgswf/work-packet@1, lgswf/checkpoint@1, lgswf/daemon-lifecycle@1
- Acceptance criteria: canonical-packet; quack-service-binding; canonical-state-command-gateway; coordination-execution-authority-consolidation; server-side-scoped-authorization; closed-lifecycle; durable-checkpoint; stale-stop
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/work_packet.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/checkpoints.py
- Validation: packet, lifecycle, corruption, resume and stale-behavior tests
- Acceptance: No parallel incompatible packet format or checkpoint-as-completion path exists.
- Gap task: LGSWF-070, LGSWF-071, LGSWF-072

## LGSWF-G090 I: immutable revision, refill and plan doctor

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G080
- Priority: P0
- Track: plan-revision
- Goal: Use the current PlanRevisionStore/backlog refinery for evidence-backed bounded refill, immutable supersession, and deterministic plan-health proposals.
- Completion contract: All triggers produce typed deduplicated proposals; bounds prevent amplification/no-progress loops; claimed-through-accepted history is immutable; stale attempts are cancelled/fenced; Doctor diagnoses all required pathologies and never mutates directly.
- Evidence: lgswf/refill-proposal@1, lgswf/refill-bounds@1, lgswf/plan-doctor@1, plan-revision@1
- Acceptance criteria: triggers; proposal-schema; bounds; history-preservation; deterministic-doctor
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/plan_doctor.py, ipfs_accelerate_py/agent_supervisor/task_sources/semantic_refill.py
- Validation: refill dedup/bounds, revision history and doctor tests
- Acceptance: Repeated semantic-equivalent failures cannot escape bounds by changing task IDs or weakening completion.
- Gap task: LGSWF-080, LGSWF-081, LGSWF-082

## LGSWF-G100 J: closed-loop semantic refresh

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G090
- Priority: P0
- Track: semantic-refresh
- Goal: Connect task admission, provisional changes, verification/merge, canonical rescan, invalidation and goal/refill reevaluation into one datasets-governed semantic refresh loop.
- Completion contract: Pre-execution inputs are verified; provisional deltas drive scope and verification; pre-merge effects/contracts/tests/proofs/governor/assurance seal; post-merge canonical rescan alone advances semantic authority and invalidates dependents.
- Evidence: lgswf/pre-execution-seal@1, lgswf/provisional-delta@1, lgswf/incremental-seal@1, lgswf/canonical-refresh@1
- Acceptance criteria: before; during; before-merge; after-merge; predicted-observed-delta
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/work_loop.py
- Validation: end-to-end semantic refresh/invalidation tests
- Acceptance: A worker result cannot update the canonical root and accepted merge cannot bypass a fresh canonical rescan.
- Gap task: LGSWF-090, LGSWF-091, LGSWF-092

## LGSWF-G110 K: bounded fixed-point convergence

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G100
- Priority: P0
- Track: convergence
- Goal: Implement the bounded observe-plan-reserve-dispatch-verify-merge-refresh-complete-refill loop and all explicit success/non-success terminals.
- Completion contract: Success requires every documented fixed-point condition simultaneously; configured exhaustion/no-progress bounds terminate honestly; restart reconstructs from durable authority rather than process dictionaries.
- Evidence: lgswf/convergence-receipt@1, lgswf/terminal-receipt@1
- Acceptance criteria: global-loop; success-conjunction; typed-terminals; durable-restart; bounded-progress
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/logic_governed_fabric.py
- Validation: convergence, terminal and restart tests
- Acceptance: No incomplete, blocked or stale-evidence state reports successful completion.
- Gap task: LGSWF-100

## LGSWF-G120 L: scheduling and coordination observability

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G110
- Priority: P0
- Track: observability
- Goal: Emit content-addressed scheduling decisions, operational evidence and all required metrics through the existing highest-level entrypoint package, with optional DuckDB + Quack-control-plane-orchestrated projections from the authoritative DuckDB outbox to the existing datasets DuckLake APIs.
- Completion contract: Every cycle records snapshot, candidates/rejections/conflicts/frontier/resources/assignments/scores/policy/claims; all required metrics are machine-readable; the DuckDB outbox/cursor remains authoritative while typed idempotent projections expose only events, artifacts and benchmarks; DuckLake absence cannot block scheduling/acceptance; payloads exclude secrets and mutable authority.
- Evidence: lgswf/decision-receipt@1, lgswf/metrics-snapshot@1, lgswf/entrypoint-contract@1, DatabaseArtifactStore@1 export/projection receipts, datasets DuckLake typed capability/ingest/snapshot/execution/security/recovery receipts
- Acceptance criteria: full-receipt; full-metrics; content-addressed; entrypoint; duckdb-outbox-projection; 19-direct-plus-7-adjacent-sealed-cases; redaction
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py, ipfs_accelerate_py/agent_supervisor/entrypoints, ipfs_accelerate_py/agent_supervisor/integrations/ducklake_history_projection.py, test/api/test_agent_supervisor_ducklake_history_projection.py
- Validation: receipt schema, deterministic identity, metrics and entrypoint tests
- Acceptance: Operational evidence explains why work was ready, parallel, assigned, reserved, changed, verified and completed or not.
- Gap task: LGSWF-110

## LGSWF-G130 M: fault and adversarial qualification

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G120
- Priority: P0
- Track: fault-qualification
- Goal: Qualify three supervisors and ten daemons against DuckDB + Quack concurrency/state-owner failover, fencing, failure, recovery, invalidation, history, refill, DuckLake projection outage/replay/recovery/security, and all critical adversarial inputs.
- Completion contract: All 26 deterministic scenarios execute with immutable raw results; the multi-process lane uses a real exact-profile Quack service and fakes remain unit-only; independent remote readiness and server-side scoped authorization pass; Quack state-owner failover advances the epoch and stale/direct-file writers fail closed; DuckLake outage leaves control-plane scheduling/acceptance live, replay reaches 100% parity with zero duplicates/missing rows, cold recovery targets RPO 0/RTO at most 300 seconds, and forged control-plane/lake capability/profile/receipt or direct access attempts fail closed; every listed critical adversarial case fails closed; any no-go is documented without result fabrication.
- Evidence: lgswf/fault-fixture@1, lgswf/fault-results@1, lgswf/adversarial-results@1
- Acceptance criteria: 3-supervisors; 10-daemons; real-exact-profile-quack; fakes-unit-only; independent-remote-readiness; server-side-authorization; 26-scenarios; quack-state-owner-failover; direct-duckdb-bypass-rejection; projection-outage-and-replay; cold-recovery; adversarial-matrix; fail-closed-critical
- Outputs: test/api/test_agent_supervisor_logic_governed_fabric_faults.py, data/agent_supervisor/logic_governed_semantic_work_fabric/qualification
- Validation: deterministic fixture and adversarial suite
- Acceptance: Duplicate/stale/forged/overlapping/undeclared/weakened/simulated/replayed/corrupt/split-brain/impossible-capacity inputs never become accepted authority.
- Gap task: LGSWF-120, LGSWF-121, LGSWF-122

## LGSWF-G140 N: parallelism and resource benchmark

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G130
- Priority: P1
- Track: benchmark
- Goal: Compare configurations A-D on a fixed workload corpus and report actual parallelism, reuse, resources, overhead, failures, refill, recovery and cost across embedded single-writer DuckDB, sealed DuckDB + Quack multi-reader/multi-writer, and optional hermetic DuckLake projection modes.
- Completion contract: Identical corpus/policies/seeds run all four configurations and each qualified control-plane/projection mode where feasible; raw results are content addressed; calculations are reproducible; heartbeat p99 <=50 ms, commit p95/p99 regressions <=5%/10%, projection >=2x peak, 30-minute backlog drain <=30 minutes, and parity/recovery values remain targets until observed; unavailable Quack/live-DuckLake cells are typed rather than simulated as successes; environment and variance are reported.
- Evidence: lgswf/benchmark-corpus@1, lgswf/benchmark-raw@1, lgswf/performance-report@1, lgswf/resource-report@1
- Acceptance criteria: configs-a-b-c-d; embedded-duckdb-vs-duckdb-quack; duckdb-quack-vs-optional-ducklake-projection; required-workloads; required-metrics; raw-results; honest-target-comparison
- Outputs: benchmarks/logic_governed_semantic_work_fabric, docs/benchmarks/logic_governed_semantic_work_fabric_results.md
- Validation: benchmark manifest/replay/statistics validators
- Acceptance: No claim of maximum parallelism follows merely from multiple processes running.
- Gap task: LGSWF-130, LGSWF-131

## LGSWF-G150 O: content-addressed qualification release

- Status: active
- Parent: LGSWF-G000
- Depends on: LGSWF-G140
- Priority: P0
- Track: release
- Goal: Produce one signed or content-addressed qualification release with exact sources, schemas, tests, raw results, findings, guidance, rollback, level, explicit continuous-operation recommendation, and a separate live DuckLake activation decision.
- Completion contract: Manifest transitively names and verifies every required artifact in two non-conflated chains: continuous operation uses the exact DuckDB + Quack pinned profile, canonical-gateway consolidation, live endpoint/server identity/start-stop propagation, independent remote readiness, datasets-backed server authorization, seven Quack-backed operational repository seals and a clean direct-file-open audit, while live DuckLake additionally requires an admitted control plane plus exact DuckLake/httpfs pins, catalog/profile/binding, projection/security/recovery/parity/benchmark evidence, and its release receipt; security and limitations are explicit; qualification level is evidence-derived; separate go/no-go gates cover both modes for the exact environment.
- Evidence: lgswf/release-manifest@1, lgswf/qualification-report@1
- Acceptance criteria: complete-manifest; verified-cids; quack-multiwriter-capability-profile-repository-chain; ducklake-capability-profile-release-chain; security; limitations; migration; rollback; level; embedded-go-no-go; continuous-quack-go-no-go; ducklake-live-go-no-go
- Outputs: data/agent_supervisor/logic_governed_semantic_work_fabric/release/qualification-release.json, docs/architecture/LOGIC_GOVERNED_SEMANTIC_WORK_FABRIC_QUALIFICATION.md
- Validation: release verifier plus full required test/benchmark evidence
- Acceptance: A production level is not assigned merely because modules exist; recommendation and permitted summary claim are scoped to exact evidence.
- Gap task: LGSWF-140, LGSWF-141
