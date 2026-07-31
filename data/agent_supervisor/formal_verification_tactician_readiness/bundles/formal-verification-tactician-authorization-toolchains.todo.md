# Objective Bundle: formal-verification-tactician/authorization-toolchains

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-051 Close objective gap: Install external Datalog and SecPAL differential shadows

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: external-capability
- Depends on: FVT-038, FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_authorization_backends.py test/integration/toolchains/test_external_authorization_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'souffle or secpal or authorization' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-051-objective-gap-bbcce6906bf1.md
- Bundle: formal-verification-tactician/authorization-toolchains
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-authorization-toolchains.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 50
- Parallel lane: formal-verification-tactician/authorization-toolchains
- Conflict policy: Own external authorization installer plugins, differential handler, and test; do not weaken or edit the in-process reference semantics.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Interfaces: ExternalAuthorizationShadowCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G180
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/ed8dccab89a9dc6ff4dd4dafeefddb6df96d6300047b88382e5a3112a5d0ae64
- Canonical task CID: baguqeera5wg4zk4jvhog75g5jwx657o3nx4w2yyaar5yqoboliyrfjoqvzsa
- Semantic identity: objective-evidence-obligation/v1/3c47e0b2ab6c844f27a22cea94a7e56befc70d484eea0286ccd169c257c7634a
- Acceptance subset: Explicit strict installation selects exact external engines, the allow/deny/unknown/conflict/delegation corpus, rule/scope mutation, replay, malformed output, timeout, and differential comparison pass, any disagreement quarantines promotion, external engines remain shadows while the certified in-process references retain authorization authority.
- Preconditions: objective goal FVT-G180 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, satisfy evidence requirement: test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-validation
- Merge fate: objective/FVT-G180
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/3c47e0b2ab6c844f27a22cea94a7e56befc70d484eea0286ccd169c257c7634a
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Embedding query: Replace the external authorization gap with pinned Souffle/SecPAL-compatible shadows and differential disagreement handling.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
- Surplus group: objective/FVT-G180
- Merge key: 0b4fced22daceb56
- Merge family: objective/FVT-G180
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
- Todo vector key: 9d43ece97c57598c
- Acceptance: Objective scan filed this gap for FVT-G180. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-051-objective-gap-bbcce6906bf1.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
