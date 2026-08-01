# Objective Bundle: formal-verification-tactician/atp-toolchains

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-048 Close objective gap: Install and certify Vampire and E ATP

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: external-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, tools/logic/certification/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/unit_tests/logic/CEC/provers/test_vampire_eprover.py test/integration/toolchains/test_atp_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'vampire or eprover or atp' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-048-objective-gap-29ccdb6feffc.md
- Bundle: formal-verification-tactician/atp-toolchains
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-atp-toolchains.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 47
- Parallel lane: formal-verification-tactician/atp-toolchains
- Conflict policy: Own ATP installer plugins, handler, and test; do not edit CEC semantics, shared lock, or central certificate.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, tools/logic/certification/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, tools/logic/certification/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Interfaces: ATPToolchainCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G140
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/2f1bcfb2047f4ddf009eb92490561eee87fe3dd2efba2956065fc08f969deb99
- Canonical task CID: baguqeeraf4n47mqep5g56ae6xesjavq652d74pos565csvqgl7ai7fu55omq
- Semantic identity: objective-evidence-obligation/v1/81697f5ecaaf2cab6336fdda2022c2b991018eaff190bc6314dd429c91cf370f
- Acceptance subset: Explicit strict installation selects Vampire 5.0.1 and E 3.2.5, theorem, non-theorem, premise/conclusion mutation, proof-output binding, replay, malformed output, and timeout checks pass, ATP results remain candidates unless an allowed independent kernel reconstruction validates them.
- Preconditions: objective goal FVT-G140 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, satisfy evidence requirement: test/integration/toolchains/test_atp_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G140
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/81697f5ecaaf2cab6336fdda2022c2b991018eaff190bc6314dd429c91cf370f
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Embedding query: Complete exact Vampire and E prover installation and certify theorem/non-theorem behavior for premise and proof search.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
- Surplus group: objective/FVT-G140
- Merge key: 5444bc4ce9fda2a1
- Merge family: objective/FVT-G140
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
- Todo vector key: d80ce0ee09b8e923
- Acceptance: Objective scan filed this gap for FVT-G140. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-048-objective-gap-29ccdb6feffc.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
