# Objective Bundle: formal-verification-tactician/proverif-toolchain

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-044 Close objective gap: Install and certify ProVerif in isolated OPAM

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, tools/logic/certification/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py test/integration/toolchains/test_proverif_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'proverif or opam' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-044-objective-gap-607b98f9f3ef.md
- Bundle: formal-verification-tactician/proverif-toolchain
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-proverif-toolchain.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 43
- Parallel lane: formal-verification-tactician/proverif-toolchain
- Conflict policy: Own the ProVerif installer plugin, handler, isolated root contract, and test; serialize OPAM resource use with Rocq and never modify a global OPAM switch.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, tools/logic/certification/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, tools/logic/certification/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Interfaces: ProVerifToolchainCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G131
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/00ba1a92f0bbe3f9e2f38872a4d30a15327efc99ef3d455087f019a122956f01
- Canonical task CID: baguqeeraac5bvexqxpr7tyxtrbzkjuykcuzh57ez546ukueh6am2ciuvn4aq
- Semantic identity: objective-evidence-obligation/v1/257be310d2c4edf64ed440bc9d96d6ec7d50cecd7ad7569d43995801957b48c4
- Acceptance subset: Explicit strict installation selects OPAM 2.5.2 support and ProVerif 2.05 in a repository-local isolated root, secure, attack, mutation, replay, malformed output, cancellation, and mismatch checks pass, model and claim identities bind receipts, OPAM alone has no semantic authority.
- Preconditions: objective goal FVT-G131 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, satisfy evidence requirement: test/integration/toolchains/test_proverif_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Resource class: exclusive-opam-toolchain
- Token class: medium
- Estimated tokens: 0
- Resources: exclusive-opam-toolchain
- Merge fate: objective/FVT-G131
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/257be310d2c4edf64ed440bc9d96d6ec7d50cecd7ad7569d43995801957b48c4
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Embedding query: Complete an isolated pinned OPAM/ProVerif deployment and semantic protocol certification without mutating global switches.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
- Surplus group: objective/FVT-G131
- Merge key: 4da6b70846009a8f
- Merge family: objective/FVT-G131
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
- Todo vector key: 74dc63801a784390
- Acceptance: Objective scan filed this gap for FVT-G131. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-044-objective-gap-607b98f9f3ef.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
