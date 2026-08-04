# Objective Bundle: formal-verification-tactician/advisor-toolchains

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-050 Close objective gap: Install and role-certify SymAI, ErgoAI, Leanstral, autoencoder, and Hammer advisors

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: advisor-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py test/integration/toolchains/test_advisor_role_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'advisor or symbolicai or ergoai or leanstral or hammer or autoencoder' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-050-objective-gap-2c7c4e73710d.md
- Bundle: formal-verification-tactician/advisor-toolchains
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-advisor-toolchains.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 49
- Parallel lane: formal-verification-tactician/advisor-toolchains
- Conflict policy: Own advisor installer plugins, role handler, and test; reuse existing adapters and caches without changing model runtimes or central certificate generation.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Interfaces: AdvisorRoleCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G160
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/d0c68a7f46d9fc1d0408fd9f9065eb44e6ab417ed89c6e393c3224bcd3a7eb93
- Canonical task CID: baguqeera2ddiu72g3h6b2bai7wpzazplittkwql63cog4oj4gislzu5h5ojq
- Semantic identity: objective-evidence-obligation/v1/3947306d3d827a450874f9dc6637617b9bc5bb05c83a97f0ba94cf5e66b15717
- Acceptance subset: Explicit strict installation selects locked SymAI and ErgoAI identities where supported, SymAI, ErgoAI, Leanstral, autoencoder, and Hammer proposals are bounded, sanitized, source-bound, deterministic or replay-bound, cache-safe, and failure-explicit, no confidence, similarity, generated text, or advisor availability becomes proof without deterministic compilation and independent solver/kernel validation.
- Preconditions: objective goal FVT-G160 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, satisfy evidence requirement: test/integration/toolchains/test_advisor_role_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Resource class: cpu-medium
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-medium
- Merge fate: objective/FVT-G160
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/3947306d3d827a450874f9dc6637617b9bc5bb05c83a97f0ba94cf5e66b15717
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Embedding query: Complete missing SymAI and ErgoAI deployment support and certify every existing advisor utility as bounded candidate generation rather than semantic proof authority.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, test/integration/toolchains/test_advisor_role_certification.py
- Surplus group: objective/FVT-G160
- Merge key: 1d63430bcc1b8739
- Merge family: objective/FVT-G160
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
- Todo vector key: cd26f0071af315a3
- Acceptance: Objective scan filed this gap for FVT-G160. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-050-objective-gap-2c7c4e73710d.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, test/integration/toolchains/test_advisor_role_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
