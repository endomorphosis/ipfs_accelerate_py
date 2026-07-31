# Agent Supervisor Proof-Carrying Test Reuse Taskboard (PTR)

Consumed by `ipfs_accelerate_py.agent_supervisor` with task prefix `PTR-`.
Companion plan:
`AGENT_SUPERVISOR_PROOF_CARRYING_TEST_REUSE_PLAN.md`. Companion goals:
`agent_supervisor_proof_carrying_test_reuse.objectives.md`.

Normative rule: this program reuses a prior authoritative test observation; it
never manufactures a fresh execution pass. Unknown, incomplete, ambiguous,
stale, tampered, or unsupported evidence executes the test.

## Parallel lanes

| Lane | Tasks |
| --- | --- |
| `ptr-contracts` | PTR-001, PTR-002 |
| `ptr-identity` | PTR-003 |
| `ptr-test-index` | PTR-004 |
| `ptr-call-closure` | PTR-005, PTR-006 |
| `ptr-receipts` | PTR-007 |
| `ptr-cache` | PTR-008 |
| `ptr-verifier` | PTR-009 |
| `ptr-zk` | PTR-010 |
| `ptr-scheduler` | PTR-011 |
| `ptr-daemon` | PTR-012 |
| `ptr-distribution` | PTR-013 |
| `ptr-adversarial` | PTR-014 |
| `ptr-shadow` | PTR-015 |
| `ptr-performance` | PTR-016 |
| `ptr-observability` | PTR-017 |
| `ptr-rollout` | PTR-018, PTR-019, PTR-020 |

## PTR-001 Seal proof-carrying test-reuse planning artifacts

- Status: completed
- Completion: manual
- Priority: P0
- Track: docs
- Depends on:
- Goal id: PTR-G000
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_TEST_REUSE_PLAN.md, docs/architecture/agent_supervisor_proof_carrying_test_reuse.objectives.md, docs/architecture/agent_supervisor_proof_carrying_test_reuse.todo.md
- Validation: test -f docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_TEST_REUSE_PLAN.md && test -f docs/architecture/agent_supervisor_proof_carrying_test_reuse.objectives.md && test -f docs/architecture/agent_supervisor_proof_carrying_test_reuse.todo.md
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/contracts
- Parallel lane: ptr-contracts
- Resource class: cpu-small
- Predicted files: docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_TEST_REUSE_PLAN.md, docs/architecture/agent_supervisor_proof_carrying_test_reuse.objectives.md, docs/architecture/agent_supervisor_proof_carrying_test_reuse.todo.md
- Conflict policy: Planning artifacts only; do not enable runtime reuse.
- Acceptance: Plan inventories current primitives, defines non-claims, identity hierarchy, algorithm, ZK role, threat model, test/load gates, rollout, goals, dependencies, and tasks.

## PTR-002 Implement reuse outcome, decision, and policy contracts

- Status: todo
- Completion: auto
- Priority: P0
- Track: contracts
- Depends on: PTR-001
- Goal id: PTR-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_contracts.py, test/api/test_agent_supervisor_validation_reuse_contracts.py, docs/architecture/agent_supervisor_proof_carrying_test_reuse_threat_model.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_contracts.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/contracts
- Parallel lane: ptr-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_contracts.py, test/api/test_agent_supervisor_validation_reuse_contracts.py
- Conflict policy: Add an orthogonal disposition; preserve `ValidationOutcome`.
- Acceptance: Canonical bounded records for policy, eligibility, proof method, disposition, reason codes, freshness, revocation, and authority. `off` default; shadow executes; environment-only enablement rejected; cache/ZK cannot upgrade trust.

## PTR-003 Add CIDv1 semantic identities for AST symbols, contexts, and edges

- Status: todo
- Completion: auto
- Priority: P0
- Track: semantic-identity
- Depends on: PTR-002
- Goal id: PTR-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/semantic_identity.py, test/api/test_agent_supervisor_validation_semantic_identity.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_semantic_identity.py test/api/test_agent_supervisor_multiformats_identity.py test/api/test_agent_supervisor_program_ast_adapters.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/identity
- Parallel lane: ptr-identity
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/semantic_identity.py, ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, test/api/test_agent_supervisor_validation_semantic_identity.py
- Conflict policy: Reuse `ASTBlobRecord` and `multiformats_identity`; sidecars only.
- Acceptance: Raw blob, module, symbol, context, and edge identities use the frozen CID profile. Defaults, decorators, globals, closures, class bases/metaclass, imports, registrations, parser/policy versions, and unknown edges are bound. Cross-machine and mutation tests pass.

## PTR-004 Build the test collection, fixture, plugin, and data dependency index

- Status: todo
- Completion: auto
- Priority: P0
- Track: test-index
- Depends on: PTR-002, PTR-003
- Goal id: PTR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/test_dependency_index.py, test/api/test_agent_supervisor_test_dependency_index.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_test_dependency_index.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/closure
- Parallel lane: ptr-test-index
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/test_dependency_index.py, test/api/test_agent_supervisor_test_dependency_index.py
- Conflict policy: Collection is observational; do not execute arbitrary plugin code during static indexing.
- Acceptance: Bind selected test nodes, `conftest.py`, fixtures/autouse fixtures, plugins/hooks, parameters, marks, setup/teardown, configs, snapshots/data, and collection identity. Unsupported/dynamic collection is explicit and ineligible.

## PTR-005 Build dependency-complete semantic slices

- Status: todo
- Completion: auto
- Priority: P0
- Track: call-closure
- Depends on: PTR-003, PTR-004
- Goal id: PTR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/semantic_slice.py, test/api/test_agent_supervisor_validation_semantic_slice.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_semantic_slice.py test/api/test_agent_supervisor_program_call_resolver.py test/api/test_agent_supervisor_validation_dag.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/closure
- Parallel lane: ptr-call-closure
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/semantic_slice.py, test/api/test_agent_supervisor_validation_semantic_slice.py
- Conflict policy: Consume `ProgramGraph`, `ProgramCallResolver`, `CodeImpactIndex`; do not infer missing direct edges.
- Acceptance: Deterministic bounded IPLD manifest contains tests, transitively reachable symbols, imports/initializers/globals, test dependencies, runtime/toolchain/config inputs, coverage, truncation, and unknown frontiers. Minimality does not remove required closure.

## PTR-006 Classify dynamic and external frontiers fail-closed

- Status: todo
- Completion: auto
- Priority: P0
- Track: call-closure
- Depends on: PTR-005
- Goal id: PTR-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/semantic_slice.py, test/api/test_agent_supervisor_validation_dynamic_frontiers.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_dynamic_frontiers.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/closure
- Parallel lane: ptr-call-closure
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/semantic_slice.py, test/api/test_agent_supervisor_validation_dynamic_frontiers.py
- Conflict policy: Adapters may close a frontier only with reviewed content-bound evidence.
- Acceptance: `eval`/`exec`, reflection, monkey patch, import hooks, callbacks/DI, RPC/MCP, subprocess, FFI/native, generated code, unpinned network/time/random/filesystem/hardware, and parser gaps force execution with stable reason codes.

## PTR-007 Define validation-input manifests and reusable prior-pass receipts

- Status: todo
- Completion: auto
- Priority: P0
- Track: receipts
- Depends on: PTR-002, PTR-004, PTR-005
- Goal id: PTR-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_receipts.py, test/api/test_agent_supervisor_validation_reuse_receipts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_receipts.py test/api/test_agent_supervisor_validation_scheduler.py -k cache
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/receipts
- Parallel lane: ptr-receipts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_receipts.py, test/api/test_agent_supervisor_validation_reuse_receipts.py
- Conflict policy: Wrap existing hermetic validation results; do not rewrite history as a new run.
- Acceptance: Manifest binds command/collection/slice/forest/overlay/runtime/environment/dependencies/toolchain/policy/capabilities/acceptance. Only stable authoritative successful executions produce reusable receipts. Stale, flaky, timeout, revoked, corrupt, or private-witness payloads reject.

## PTR-008 Add a proof-carrying reuse index to the existing cache coordinator

- Status: todo
- Completion: auto
- Priority: P0
- Track: cache
- Depends on: PTR-007
- Goal id: PTR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, ipfs_accelerate_py/agent_supervisor/validation/validation_reuse.py, test/api/test_agent_supervisor_validation_reuse_cache.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_cache.py test/api/test_agent_supervisor_program_analysis_cache.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/cache
- Parallel lane: ptr-cache
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py, ipfs_accelerate_py/agent_supervisor/validation/validation_reuse.py, test/api/test_agent_supervisor_validation_reuse_cache.py
- Conflict policy: Exact validation cache remains tier one; reuse index shares authority namespaces, CAS, quotas, GC, and single-flight.
- Acceptance: Root-to-receipt lookup, authoritative namespace isolation, transitive invalidation, negative TTL, atomic concurrent publish, corruption repair, revocation, quota/GC, and zero stale authoritative hits.

## PTR-009 Implement independent direct and IPLD Merkle reuse verification

- Status: todo
- Completion: auto
- Priority: P0
- Track: verifier
- Depends on: PTR-007, PTR-008
- Goal id: PTR-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse.py, test/api/test_agent_supervisor_validation_reuse.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/verifier
- Parallel lane: ptr-verifier
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse.py, test/api/test_agent_supervisor_validation_reuse.py
- Conflict policy: Verification derives authority; serialized decision flags are untrusted.
- Acceptance: Rebuild current manifest, verify prior receipt, compare roots, verify memberships, check policy/capability/freshness/revocation, close TOCTOU, emit accepted/rejected decision with exact reasons, and fall back to execution on any error.

## PTR-010 Add optional shadow ZK validation-input equality proofs

- Status: todo
- Completion: manual
- Priority: P1
- Track: zk-reuse
- Depends on: PTR-009
- Goal id: PTR-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_validation_reuse_zkp.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_zkp.py test/api/test_agent_supervisor_program_analysis_zkp_conformance.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/zk
- Parallel lane: ptr-zk
- Resource class: proof-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_validation_reuse_zkp.py
- Conflict policy: Extend existing public inputs/capability gates; simulated ZK cannot authorize.
- Acceptance: Prove openings, required membership/trace transitions, equal input roots, and bound prior receipt commitment. Explicitly do not claim execution honesty, call-graph completeness, Python equivalence, or broader correctness. Require approved private-witness/cross-trust use case, circuit/key/ceremony/codec pins, independent verification, no witness leakage.

## PTR-011 Integrate reuse decisions into the validation DAG scheduler

- Status: todo
- Completion: auto
- Priority: P0
- Track: scheduler
- Depends on: PTR-002, PTR-009
- Goal id: PTR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py, test/api/test_agent_supervisor_validation_reuse_scheduler.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_scheduler.py test/api/test_agent_supervisor_validation_scheduler.py test/api/test_agent_supervisor_validation_dag.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/orchestration
- Parallel lane: ptr-scheduler
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py, test/api/test_agent_supervisor_validation_reuse_scheduler.py
- Conflict policy: Preserve stage barriers, failure blocking, exact cache, and normal runner.
- Acceptance: Exact cache checked first; off/advisory/shadow/enforcement behavior exact; `proved_reuse_pass` distinct; reused nodes satisfy only reviewed policy; dependency failures block dependents; current root rechecked before completion.

## PTR-012 Wire daemon configuration, CLI, status, and audit receipts

- Status: todo
- Completion: auto
- Priority: P0
- Track: daemon
- Depends on: PTR-011
- Goal id: PTR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_validation_reuse_daemon.py, docs/operations/agent_supervisor_validation_reuse.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_daemon.py test/api/test_agent_supervisor_todo_daemon_port.py -k validation
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/orchestration
- Parallel lane: ptr-daemon
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_validation_reuse_daemon.py
- Conflict policy: Reviewed policy file required for enforcement; environment alone cannot enable.
- Acceptance: Flags/config for mode, age, rerun interval, allowed kinds, closure bounds, proof/verifier mode, and opt-out. Status records eligibility/reasons/roots/receipts/latency/saved time/TOCTOU. Merge and completion reject stale reuse.

## PTR-013 Verify IPFS/P2P replication and scoped publication

- Status: todo
- Completion: auto
- Priority: P1
- Track: distribution
- Depends on: PTR-008, PTR-009
- Goal id: PTR-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_transport.py, test/api/test_agent_supervisor_validation_reuse_transport.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_transport.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/distribution
- Parallel lane: ptr-distribution
- Resource class: io-network
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_transport.py, test/api/test_agent_supervisor_validation_reuse_transport.py
- Conflict policy: Transport and UCAN scope do not grant evidence authority.
- Acceptance: Offline/local operation; untrusted replica verification; CID/schema/authority/policy checks; bounded fetch; no witness/secret publication; optional UCAN read/publish scope; outage and poisoning fall back to execution.

## PTR-014 Build the adversarial and mutation conformance suite

- Status: todo
- Completion: auto
- Priority: P0
- Track: correctness
- Depends on: PTR-006, PTR-009, PTR-011
- Goal id: PTR-G080
- Outputs: test/api/test_agent_supervisor_validation_reuse_adversarial.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_adversarial.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/correctness
- Parallel lane: ptr-adversarial
- Resource class: cpu-large
- Predicted files: test/api/test_agent_supervisor_validation_reuse_adversarial.py
- Conflict policy: Every false reuse is a hard failure.
- Acceptance: Seed changes to body/signature/default/decorator/global/import/caller/callee/fixture/plugin/parameter/data/config/lock/submodule/env/interpreter/native/generated/dynamic/network/time/random/filesystem/collection/policy/key. Mutate graphs/manifests/receipts/cache/proofs. Every case changes root or forces execution.

## PTR-015 Run differential shadow validation across real repositories

- Status: todo
- Completion: manual
- Priority: P0
- Track: shadow
- Depends on: PTR-011, PTR-014
- Goal id: PTR-G080
- Outputs: test/integration/test_agent_supervisor_validation_reuse_shadow.py, artifacts/agent_supervisor/validation_reuse/shadow_summary.json
- Validation: python -m pytest -q test/integration/test_agent_supervisor_validation_reuse_shadow.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/correctness
- Parallel lane: ptr-shadow
- Resource class: cpu-large
- Predicted files: test/integration/test_agent_supervisor_validation_reuse_shadow.py
- Conflict policy: Shadow prediction never skips execution.
- Acceptance: Ratified accelerator/datasets/kit/lift fixtures; process restart and replica parity; deterministic roots; zero false reuse and stale hits; no mandatory DAG/acceptance loss; mismatch automatically disables affected scope.

## PTR-016 Benchmark cold/warm reuse, parallel supervisors, storage, and GC

- Status: todo
- Completion: auto
- Priority: P1
- Track: performance
- Depends on: PTR-008, PTR-009, PTR-011
- Goal id: PTR-G090
- Outputs: benchmarks/agent_supervisor/validation_reuse.py, test/load/test_agent_supervisor_validation_reuse.py
- Validation: python -m pytest -q test/load/test_agent_supervisor_validation_reuse.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/performance
- Parallel lane: ptr-performance
- Resource class: cpu-large
- Predicted files: benchmarks/agent_supervisor/validation_reuse.py, test/load/test_agent_supervisor_validation_reuse.py
- Conflict policy: Benchmark thresholds never override ineligibility.
- Acceptance: Exact/doc/unrelated/leaf/interface/fixture/config/dynamic profiles; cold/warm/local/replica; 1/4/16/64 lanes; p50/p95/p99, saved time, CPU/RSS/disk/network, single-flight, cache bounds, GC/corruption recovery, reuse precision and coverage.

## PTR-017 Add reuse metrics, diagnostics, alerting, and mismatch kill switch

- Status: todo
- Completion: auto
- Priority: P0
- Track: observability
- Depends on: PTR-011, PTR-015
- Goal id: PTR-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_rollout.py, test/api/test_agent_supervisor_validation_reuse_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_rollout.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/rollout
- Parallel lane: ptr-observability
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_rollout.py, test/api/test_agent_supervisor_validation_reuse_rollout.py
- Conflict policy: Metrics cannot self-promote enforcement.
- Acceptance: Counters/latencies/reasons by disposition and policy; shadow mismatch severity-one alert; automatic class/repository disable; redacted diagnostics; current policy/capability/receipt IDs; operator-visible rollback status.

## PTR-018 Implement off, shadow, canary, and reviewed enforcement transitions

- Status: todo
- Completion: manual
- Priority: P0
- Track: rollout
- Depends on: PTR-015, PTR-016, PTR-017
- Goal id: PTR-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_rollout.py, docs/operations/agent_supervisor_validation_reuse.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_rollout.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/rollout
- Parallel lane: ptr-rollout
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_rollout.py, docs/operations/agent_supervisor_validation_reuse.md
- Conflict policy: No autonomous promotion; operator review required.
- Acceptance: Allowlisted repositories/test classes, minimum shadow sample, zero mismatches, mandatory periodic execution, policy expiry, canary budget, promotion/demotion receipts, and direct rollback to off.

## PTR-019 Exercise corruption, capability-loss, mismatch, and emergency rollback

- Status: todo
- Completion: manual
- Priority: P0
- Track: rollout
- Depends on: PTR-018
- Goal id: PTR-G100
- Outputs: test/integration/test_agent_supervisor_validation_reuse_rollback.py, artifacts/agent_supervisor/validation_reuse/rollback_receipt.json
- Validation: python -m pytest -q test/integration/test_agent_supervisor_validation_reuse_rollback.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/rollout
- Parallel lane: ptr-rollout
- Resource class: cpu-medium
- Predicted files: test/integration/test_agent_supervisor_validation_reuse_rollback.py
- Conflict policy: Rollback uses ordinary validation and does not delete audit evidence.
- Acceptance: Cache corruption, stale receipt, verifier/key loss, capability drift, shadow mismatch, TOCTOU drift, and operator kill switch all stop reuse immediately and resume normal validation without supervisor restart.

## PTR-020 Ratify production readiness and retain ZK as optional sidecar

- Status: todo
- Completion: manual
- Priority: P0
- Track: rollout
- Depends on: PTR-014, PTR-015, PTR-016, PTR-018, PTR-019
- Goal id: PTR-G100
- Outputs: docs/operations/agent_supervisor_validation_reuse.md, artifacts/agent_supervisor/validation_reuse/release_decision.json
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_adversarial.py test/integration/test_agent_supervisor_validation_reuse_shadow.py test/load/test_agent_supervisor_validation_reuse.py test/integration/test_agent_supervisor_validation_reuse_rollback.py
- Board namespace: agent-supervisor-proof-carrying-test-reuse-v1
- Bundle: agent-supervisor/proof-test-reuse/rollout
- Parallel lane: ptr-rollout
- Resource class: cpu-large
- Predicted files: docs/operations/agent_supervisor_validation_reuse.md
- Conflict policy: Release decision is external/operator-reviewed; local task completion cannot manufacture it.
- Acceptance: Zero false reuse and stale hits; cross-machine deterministic roots; all mutation gates pass; bounded performance and storage; audit/rollback verified; local direct/Merkle reuse works without ZK; any ZK enforcement has a separately approved private-witness/cross-trust decision and production verifier conformance.
