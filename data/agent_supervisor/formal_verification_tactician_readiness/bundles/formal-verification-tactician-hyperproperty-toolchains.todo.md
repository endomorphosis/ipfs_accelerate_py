# Objective Bundle: formal-verification-tactician/hyperproperty-toolchains

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-046 Close objective gap: Install and certify HyperLTL, AutoHyper, and MCHyper

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, tools/logic/certification/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_hyperproperty_backends.py test/integration/toolchains/test_hyperproperty_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'hyperltl or autohyper or mchyper or hyperproperty' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-046-objective-gap-929d4b20db70.md
- Bundle: formal-verification-tactician/hyperproperty-toolchains
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-hyperproperty-toolchains.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 45
- Parallel lane: formal-verification-tactician/hyperproperty-toolchains
- Conflict policy: Own hyperproperty installer plugins, handler, fixtures, and test; do not edit shared lock or central certificate.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, tools/logic/certification/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, tools/logic/certification/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Interfaces: HyperpropertyToolchainCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G170
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/fd2b1a4e6c3940da31b23b9039505992172a61c9527ceb72fdc76aee51f1882a
- Canonical task CID: baguqeera7uvruttmhfanumnshoidsuczsilsuyojkj6ow4x5y5vo4uprrava
- Semantic identity: objective-evidence-obligation/v1/cffee7f29c6c4f41b182a00e201098be5cfbfad39033cfeef88c06595369e2b9
- Acceptance subset: Explicit strict installation selects reviewed HyperLTL, AutoHyper, and MCHyper artifacts, quantifiers and observation projections are preserved, satisfaction, violating trace tuples, semantic mutations, replay, malformed output, disagreement, timeout, and exact bounds pass, results retain their declared bounded hyperproperty authority and cannot make universal claims beyond bounds.
- Preconditions: objective goal FVT-G170 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, satisfy evidence requirement: test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G170
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/cffee7f29c6c4f41b182a00e201098be5cfbfad39033cfeef88c06595369e2b9
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Embedding query: Replace the hyperproperty declared gap with pinned external engines and bounded information-flow semantic certification.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py
- Surplus group: objective/FVT-G170
- Merge key: 2252faa4559383bb
- Merge family: objective/FVT-G170
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
- Todo vector key: 42eeb0bc943d061a
- Acceptance: Objective scan filed this gap for FVT-G170. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-046-objective-gap-929d4b20db70.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, test/integration/toolchains/test_hyperproperty_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
