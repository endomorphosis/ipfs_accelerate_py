# Objective Bundle: formal-verification-tactician/toolchain-governance

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-037 Close objective gap: Define role-aware toolchain authority and promotion

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: toolchain-governance
- Depends on: FVT-030
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certification/roles.py, test/api/test_formal_verification_toolchain_roles.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_verification_toolchain_security.py test/api/test_formal_verification_toolchain_roles.py test/integration/test_formal_verification_real_tool_matrix.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-037-objective-gap-0c897491b142.md
- Bundle: formal-verification-tactician/toolchain-governance
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-toolchain-governance.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 36
- Parallel lane: formal-verification-tactician/toolchain-governance
- Conflict policy: Own the canonical role schema, lane registration, and authority-boundary tests; pre-register per-lane handlers so later tasks do not concurrently edit the central certifier or generated certificate.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certification/roles.py, test/api/test_formal_verification_toolchain_roles.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certification/roles.py, test/api/test_formal_verification_toolchain_roles.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, test/api/test_formal_verification_toolchain_roles.py
- Interfaces: FormalVerificationToolRole@1, RoleAwarePromotionPolicy@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G100
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/8d426c12e58db385dc6960e37c24827fda9b9e0f584ecc4a392623fe865cbfa4
- Canonical task CID: baguqeerarvbgyexfrwzylxdjmdrxyjecp7njxhqplbhmysrzeyr75bs4x6sa
- Semantic identity: objective-evidence-obligation/v1/782ab0f8dc3b41aa838e47a000f3aff5e0a95ebc0b03548397c465a2c04326dd
- Acceptance subset: Every matrix entry has exactly one closed role and authority ceiling, Java, Maude, and OPAM are support only, Leanstral, autoencoder, SymAI, ErgoAI, and Hammer are advisor/candidate only until independent reconstruction, external Souffle/SecPAL are shadow checkers, in-process Datalog/SecPAL have authorization-only authority, Runtime MTL has finite-trace authority, state and hyperproperty tools have bounded authority, Lean/Rocq/Isabelle have kernel authority, ZKP has attestation authority only, support, advisor, or shadow presence alone can never satisfy a certified-authority requirement.
- Preconditions: objective goal FVT-G100 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, satisfy evidence requirement: test/api/test_formal_verification_toolchain_roles.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, test/api/test_formal_verification_toolchain_roles.py
- Resource class: cpu-policy
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-policy
- Merge fate: objective/FVT-G100
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/782ab0f8dc3b41aa838e47a000f3aff5e0a95ebc0b03548397c465a2c04326dd
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, test/api/test_formal_verification_toolchain_roles.py
- Embedding query: Replace availability-shaped promotion with a closed per-tool role model and split the monolithic certificate runner into independently owned semantic lanes.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, test/api/test_formal_verification_toolchain_roles.py
- Surplus group: objective/FVT-G100
- Merge key: 027da636db8828a1
- Merge family: objective/FVT-G100
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
- Todo vector key: d595c88f127213a7
- Acceptance: Objective scan filed this gap for FVT-G100. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-037-objective-gap-0c897491b142.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, test/api/test_formal_verification_toolchain_roles.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
