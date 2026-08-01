# Objective Bundle: formal-verification-tactician/rocq-toolchain

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-045 Close objective gap: Install and semantically certify Rocq

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, tools/logic/certification/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_kernel_backends.py test/integration/toolchains/test_rocq_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'rocq or coq' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-045-objective-gap-de139d665eac.md
- Bundle: formal-verification-tactician/rocq-toolchain
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-rocq-toolchain.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 44
- Parallel lane: formal-verification-tactician/rocq-toolchain
- Conflict policy: Own the Rocq installer plugin, handler, and test; serialize OPAM resource use with ProVerif and never modify a global switch.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, tools/logic/certification/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, tools/logic/certification/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Interfaces: RocqToolchainCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G150
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/60651e6a3047e69167e0f39ccf0e43c7c36e39627397cc8be5f348bc31ead656
- Canonical task CID: baguqeerambsr42rqi7tjcz7a6oom6dsdy7bw4olcool4zc7f6nelympk2zla
- Semantic identity: objective-evidence-obligation/v1/aecd9b7e5259b4de83893c54d77b2b10d391c0fe9551902626bd20890676adf8
- Acceptance subset: Explicit strict installation selects Rocq 9.1.1 in an isolated pinned OPAM root, true proof, false proof, hypothesis/conclusion mutation, deterministic replay, forbidden admits/axiom escapes, malformed input, and mismatch checks pass, receipts bind imports, source, theorem, assumptions, and exact kernel identity.
- Preconditions: objective goal FVT-G150 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, satisfy evidence requirement: test/integration/toolchains/test_rocq_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Resource class: exclusive-opam-toolchain
- Token class: medium
- Estimated tokens: 0
- Resources: exclusive-opam-toolchain
- Merge fate: objective/FVT-G150
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/aecd9b7e5259b4de83893c54d77b2b10d391c0fe9551902626bd20890676adf8
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Embedding query: Complete isolated installation and real kernel certification for the locked Rocq/Coq provider.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
- Surplus group: objective/FVT-G150
- Merge key: 36eb046d624c0b2a
- Merge family: objective/FVT-G150
- Merge role: aggregate
- Work item count: 2
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: bcd154539db23360
- Acceptance: Objective scan filed this gap for FVT-G150. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-045-objective-gap-de139d665eac.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
