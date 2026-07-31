# Agent Supervisor Proof-Carrying Test Reuse Objective Heap (PTR)

Machine-ingestible goals for
`AGENT_SUPERVISOR_PROOF_CARRYING_TEST_REUSE_PLAN.md`. The executable projection
is `agent_supervisor_proof_carrying_test_reuse.todo.md` with prefix `PTR-`.

## North star

Reuse a prior successful validation across repository trees only when a
current, complete, content-addressed validation-input manifest is identical to
the prior manifest and an independently verified proof binds that equality to
an authoritative prior execution receipt. Unknown means execute.

## Goal tree

```text
PTR-G000  Production-safe proof-carrying validation reuse
├── PTR-G010  Doctrine, threat model, outcome and policy contracts
├── PTR-G020  CIDv1 AST symbol/context/edge identities
├── PTR-G030  Complete semantic and test dependency closures
├── PTR-G040  Validation-input manifests and prior-pass receipts
├── PTR-G050  Direct/Merkle reuse proof and cache integration
├── PTR-G060  Optional privacy-preserving ZK reuse proof
├── PTR-G070  Scheduler, daemon, CLI, status, and completion integration
├── PTR-G080  Adversarial, mutation, differential, and conformance gates
├── PTR-G090  Performance, concurrency, IPFS replication, and GC
└── PTR-G100  Shadow, canary, enforcement, rollback, and operations
```

## Parallel waves

| Wave | Goals | Dependency |
| --- | --- | --- |
| 0 | G010 | none |
| 1 | G020, G040 | G010 |
| 2 | G030 | G020 |
| 3 | G050 | G030 + G040 |
| 4 | G060, G070 | G050; G060 may remain shadow |
| 5 | G080, G090 | G050 + G070 |
| 6 | G100 | G080 + G090 |

## PTR-G000 Production-safe proof-carrying validation reuse

- Status: active
- Parent:
- Priority: P0
- Track: proof-test-reuse
- Bundle: agent-supervisor/proof-test-reuse/root
- Goal: Deliver an opt-in, fail-closed mechanism that admits a typed prior test pass on a current tree only when all validation inputs are proven unchanged.
- Evidence: ptr/root-contract@1
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_TEST_REUSE_PLAN.md, docs/architecture/agent_supervisor_proof_carrying_test_reuse.objectives.md, docs/architecture/agent_supervisor_proof_carrying_test_reuse.todo.md
- Validation: test -f docs/architecture/agent_supervisor_proof_carrying_test_reuse.todo.md
- Acceptance: All child goals are implemented or explicitly blocked; ordinary validation remains available; no fake fresh pass; unknown always executes.
- Conflict policy: Extend existing AST, cache, validation, proof, and ZK contracts; create no second assurance lattice or cache authority.

## PTR-G010 Doctrine, threat model, outcome and policy contracts

- Status: active
- Parent: PTR-G000
- Priority: P0
- Track: contracts
- Bundle: agent-supervisor/proof-test-reuse/contracts
- Goal: Define executed, exact-cache, proved-reuse, shadow-match/mismatch, and ineligible outcomes plus reviewed reuse policy modes and non-claims.
- Evidence: ptr/reuse-contract@1, ptr/reuse-threat-model@1
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_contracts.py, docs/architecture/agent_supervisor_proof_carrying_test_reuse_threat_model.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_contracts.py
- Acceptance: Typed bounded canonical records; cache hits never upgrade trust; ZK non-claims explicit; off/shadow/advisory/enforce policies; arbitrary environment enablement rejected.
- Gap task: Implement PTR-002.
- Conflict policy: Preserve existing `ValidationOutcome`; add orthogonal disposition rather than changing historical meanings.

## PTR-G020 CIDv1 AST symbol, context, and edge identities

- Status: active
- Parent: PTR-G000, PTR-G010
- Priority: P0
- Track: semantic-identity
- Bundle: agent-supervisor/proof-test-reuse/identity
- Goal: Produce canonical CIDv1 identities for exact blobs, AST modules, symbols, semantic contexts, and resolved/unknown edges.
- Evidence: ptr/semantic-identity@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_ast_adapters.py, ipfs_accelerate_py/agent_supervisor/multiformats_identity.py, ipfs_accelerate_py/agent_supervisor/validation/semantic_identity.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_semantic_identity.py test/api/test_agent_supervisor_multiformats_identity.py
- Acceptance: Reproducible across clean machines; parser/policy versions bound; defaults/decorators/globals/class context included; double hashing and profile confusion rejected.
- Gap task: Implement PTR-003.
- Conflict policy: Keep `ASTBlobRecord` canonical and add sidecars; do not fork the AST schema.

## PTR-G030 Complete semantic and test dependency closures

- Status: active
- Parent: PTR-G000, PTR-G020
- Priority: P0
- Track: dependency-closure
- Bundle: agent-supervisor/proof-test-reuse/closure
- Goal: Build bounded dependency-complete semantic slices for selected tests, including fixtures, plugins, data, runtime, toolchain, and explicit unknown frontiers.
- Evidence: ptr/semantic-slice@1, ptr/test-dependency-closure@1
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/semantic_slice.py, ipfs_accelerate_py/agent_supervisor/validation/test_dependency_index.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_semantic_slice.py test/api/test_agent_supervisor_test_dependency_index.py
- Acceptance: Direct/transitive calls and test collection inputs covered; incomplete/ambiguous/truncated closure cannot be eligible; dynamic and external boundaries have typed fallback reasons.
- Gap task: Implement PTR-004 through PTR-006.
- Conflict policy: Reuse `ProgramGraph`, `ProgramCallResolver`, and `CodeImpactIndex`; graph retrieval cannot manufacture edges.

## PTR-G040 Validation-input manifests and prior-pass receipts

- Status: active
- Parent: PTR-G000, PTR-G010
- Priority: P0
- Track: receipt-contract
- Bundle: agent-supervisor/proof-test-reuse/receipts
- Goal: Bind command, collection, semantic slice, runtime, forest, dependencies, environment, policy, and capabilities into one validation-input root and wrap eligible prior executions.
- Evidence: ptr/validation-input-manifest@1, ptr/reusable-validation-receipt@1
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_receipts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_receipts.py
- Acceptance: Only stable successful authoritative executions can be wrapped; stale/revoked/flaky/timeout/incomplete receipts reject; private data stays outside public receipt.
- Gap task: Implement PTR-007.
- Conflict policy: Wrap existing validation receipts; do not invent a new execution result.

## PTR-G050 Direct/Merkle reuse proof and cache integration

- Status: active
- Parent: PTR-G000, PTR-G030, PTR-G040
- Priority: P0
- Track: reuse-verifier
- Bundle: agent-supervisor/proof-test-reuse/verifier
- Goal: Verify current/prior validation-input equality with direct or IPLD Merkle proofs and index receipts through existing cache/CAS authority.
- Evidence: ptr/validation-reuse-proof@1, ptr/validation-reuse-decision@1
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_reuse.py, ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse.py test/api/test_agent_supervisor_program_analysis_cache.py
- Acceptance: Current-tree rebuild and TOCTOU recheck; independent receipt verification; exact reason codes; corruption and replica poisoning reject; single-flight and transitive invalidation.
- Gap task: Implement PTR-008 and PTR-009.
- Conflict policy: Existing exact validation cache remains tier one; reuse is a separate typed lookup under the same cache coordinator.

## PTR-G060 Optional privacy-preserving ZK reuse proof

- Status: active
- Parent: PTR-G000, PTR-G050
- Priority: P1
- Track: zk-reuse
- Bundle: agent-supervisor/proof-test-reuse/zk
- Goal: Add a shadow-only ZK statement for manifest opening/membership/equality when a reviewed private-witness and cross-trust use case exists.
- Evidence: ptr/zk-reuse-statement@1, ptr/zk-reuse-verification@1
- Outputs: ipfs_accelerate_py/agent_supervisor/program_analysis_zkp.py, test/api/test_agent_supervisor_validation_reuse_zkp.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_zkp.py test/api/test_agent_supervisor_program_analysis_zkp_conformance.py
- Acceptance: Simulated proofs never authorize; independent verifier, circuit/key/ceremony/codec pins, no witness leakage, explicit non-claims, direct/Merkle mode remains sufficient.
- Gap task: Implement PTR-010.
- Conflict policy: Extend the existing program-analysis ZK public-input contract; no second ZK trust root.

## PTR-G070 Scheduler, daemon, CLI, status, and completion integration

- Status: active
- Parent: PTR-G000, PTR-G050
- Priority: P0
- Track: orchestration
- Bundle: agent-supervisor/proof-test-reuse/orchestration
- Goal: Integrate reuse decisions into selected validation DAGs while preserving stage barriers, auditability, merge freshness, and immediate disablement.
- Evidence: ptr/scheduler-integration@1
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_scheduler.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_todo_daemon_port.py -k 'reuse or validation'
- Acceptance: Distinct dispositions; off is default; shadow executes; reviewed enforcement only; merge/completion rebind current root; status/metrics/reasons emitted; rollback switch tested.
- Gap task: Implement PTR-011 through PTR-013.
- Conflict policy: Preserve ordinary runner and exact cache behavior.

## PTR-G080 Adversarial, mutation, differential, and conformance gates

- Status: active
- Parent: PTR-G000, PTR-G050, PTR-G070
- Priority: P0
- Track: correctness
- Bundle: agent-supervisor/proof-test-reuse/correctness
- Goal: Demonstrate zero false reuse under seeded dependency drift, cache/proof mutation, dynamic behavior, and shadow execution.
- Evidence: ptr/reuse-conformance@1
- Outputs: test/api/test_agent_supervisor_validation_reuse_adversarial.py, test/integration/test_agent_supervisor_validation_reuse_shadow.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_adversarial.py test/integration/test_agent_supervisor_validation_reuse_shadow.py
- Acceptance: Every authority-relevant mutation rejects; every dependency seed changes root or forces execution; zero shadow mismatches in ratified corpus; coverage cannot shrink.
- Gap task: Implement PTR-014 and PTR-015.
- Conflict policy: A false reuse prediction is a release blocker, not a tolerated flaky result.

## PTR-G090 Performance, concurrency, replication, and GC

- Status: active
- Parent: PTR-G000, PTR-G050, PTR-G070
- Priority: P1
- Track: performance
- Bundle: agent-supervisor/proof-test-reuse/performance
- Goal: Quantify saved time and bound indexing, proof, concurrency, storage, IPFS replication, and GC costs without relaxing correctness.
- Evidence: ptr/reuse-benchmark@1
- Outputs: benchmarks/agent_supervisor/validation_reuse.py, test/load/test_agent_supervisor_validation_reuse.py
- Validation: python -m pytest -q test/load/test_agent_supervisor_validation_reuse.py
- Acceptance: Zero false reuse; warm p95 target measured; unrelated-change time reduction measured; 64-lane single-flight; bounded cache and corruption/GC recovery.
- Gap task: Implement PTR-016.
- Conflict policy: Performance targets cannot convert unknown or rejected evidence into reuse.

## PTR-G100 Shadow, canary, enforcement, rollback, and operations

- Status: active
- Parent: PTR-G000, PTR-G080, PTR-G090
- Priority: P0
- Track: rollout
- Bundle: agent-supervisor/proof-test-reuse/rollout
- Goal: Operate a reversible shadow-to-enforcement rollout with class-level policy, periodic reruns, mismatch kill switch, and auditable receipts.
- Evidence: ptr/reuse-rollout@1
- Outputs: docs/operations/agent_supervisor_validation_reuse.md, ipfs_accelerate_py/agent_supervisor/validation/validation_reuse_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_validation_reuse_rollout.py
- Acceptance: Off/shadow/canary/enforce transitions; allowlist and mandatory rerun; mismatch auto-disable; status/diagnostics; rollback drill; no dependency on ZK for local operation.
- Gap task: Implement PTR-017 through PTR-020.
- Conflict policy: Operators retain explicit control; no autonomous promotion from metrics alone.
