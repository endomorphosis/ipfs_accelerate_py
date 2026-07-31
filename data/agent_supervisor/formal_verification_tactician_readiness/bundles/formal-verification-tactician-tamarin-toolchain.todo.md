# Objective Bundle: formal-verification-tactician/tamarin-toolchain

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-043 Close objective gap: Install and certify Tamarin with Maude

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, tools/logic/certification/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_protocol_backends.py test/integration/toolchains/test_tamarin_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'tamarin or maude' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-043-objective-gap-a914bad8db21.md
- Bundle: formal-verification-tactician/tamarin-toolchain
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-tamarin-toolchain.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 42
- Parallel lane: formal-verification-tactician/tamarin-toolchain
- Conflict policy: Own the Tamarin/Maude installer plugin, handler, and test; do not edit the ProVerif lane, shared lock, or central certificate.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, tools/logic/certification/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, tools/logic/certification/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Interfaces: TamarinToolchainCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G130
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/9745047251c6b3625b93a4f02b93d2cff9b709738cd8b46937efc279d140f2e5
- Canonical task CID: baguqeeras5cqi4sry2zwew4tutycxe6sz743ocltrtmli2jx57bhtuka6lsq
- Semantic identity: objective-evidence-obligation/v1/7052cde3800c7eef47032e7c6738e2630c450823e3492870043c8117b791b6b3
- Acceptance subset: Explicit strict installation selects Tamarin 1.12.0 and Maude 3.5.1, secure, attack, mutated claim/rule, replay, malformed output, timeout, and version mismatch cases pass, theory, claims, bounds, and exact binaries are bound, Maude is support only and cannot promote a property lane by itself.
- Preconditions: objective goal FVT-G130 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, satisfy evidence requirement: test/integration/toolchains/test_tamarin_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G130
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/7052cde3800c7eef47032e7c6738e2630c450823e3492870043c8117b791b6b3
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Embedding query: Complete the exact Tamarin and compatible Maude installation and certify cryptographic-protocol claims and attacks.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
- Surplus group: objective/FVT-G130
- Merge key: 093f84818dbca8ee
- Merge family: objective/FVT-G130
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
- Todo vector key: bad8c012b0595967
- Acceptance: Objective scan filed this gap for FVT-G130. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-043-objective-gap-a914bad8db21.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
