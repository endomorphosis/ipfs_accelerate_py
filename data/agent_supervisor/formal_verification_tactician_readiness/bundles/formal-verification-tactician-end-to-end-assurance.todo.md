# Objective Bundle: formal-verification-tactician/end-to-end-assurance

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-088 FVT:: Audit every deployment axis end to end

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: certification-integrity
- Depends on: FVT-084, FVT-087, FVT-086, FVT-085
- Outputs: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json, test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py test/integration/test_formal_verification_real_tool_matrix.py test/packaging/test_logic_verification_clean_install.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-088-objective-gap-2c7c735deb11.md
- Bundle: formal-verification-tactician/end-to-end-assurance
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-end-to-end-assurance.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 4
- Parallel lane: formal-verification-tactician/end-to-end-assurance
- Conflict policy: Own the cross-axis matrix and aggregation policy; do not hardcode green states, collapse platform exceptions into success, or let one provider stand in for another.
- Predicted files: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json, test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py
- Changed paths:
- Context paths: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json, test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py
- AST symbols: test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json
- Interfaces: FormalVerificationEndToEndAssuranceMatrix@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G220
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/227e71970b35748299bb79aed69faa1ed0f8016efb5583893c1ecafa37eb2768
- Canonical task CID: baguqeeraej7hdfylgv2ifgn3pgxnnh5kd3ipqalo7nkyhcj4d3fpun7le5ua
- Semantic identity: objective-evidence-obligation/v1/0f204668cce71fd642a326b78c546679b3d9e60dd9574cf1c99d9f86aaa90eb1
- Acceptance subset: Each provider and host tuple reports separate dependency, packaging, installer, capability, semantic, platform, authority, freshness, and public-surface states with exact evidence references and reason codes, no axis inherits success from another, supported missing dependencies, missing wheel files, placeholder dispatch, stale locks, wrong-architecture artifacts, parser fixtures, advisor-only evidence, and unsupported hosts are distinguishable, SecPAL in-process and external identities and ErgoAI advisor and independent proof authority remain distinct, an adversarial test mutates every axis and proves that the joint readiness claim fails closed.
- Preconditions: objective goal FVT-G220 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py, satisfy evidence requirement: docs/architecture/formal_verification_end_to_end_assurance_matrix.json
- Evidence subset: test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok-implement, codex-review
- Resources: cpu-validation
- Merge fate: objective/FVT-G220
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/0f204668cce71fd642a326b78c546679b3d9e60dd9574cf1c99d9f86aaa90eb1
- Missing evidence: test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json
- Embedding query: Make dependency, capability, semantic, platform-binding, authority, packaging, installer-boundary, and public-surface readiness independently visible and jointly fail closed.
- AST query: test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json
- Surplus group: objective/FVT-G220
- Merge key: 5e53086b6ab5cce4
- Merge family: objective/FVT-G220
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
- Todo vector key: 881608d4da8e1fd4
- Acceptance: Objective scan filed this gap for FVT-G220. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-088-objective-gap-2c7c735deb11.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py, docs/architecture/formal_verification_end_to_end_assurance_matrix.json), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
