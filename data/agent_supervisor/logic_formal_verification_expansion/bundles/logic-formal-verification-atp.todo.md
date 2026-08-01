# Objective Bundle: logic-formal-verification/atp

Source todo: docs/architecture/logic_formal_verification_expansion.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## LFV-028 Close objective gap: Normalize Vampire, E, DCEC, TDFOL, and legacy prover adapters

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: prover
- Depends on: LFV-006, LFV-007, LFV-009
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py -q
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-028-objective-gap-15201f1bef99.md
- Bundle: logic-formal-verification/atp
- Bundle shard: data/agent_supervisor/logic_formal_verification_expansion/bundles/logic-formal-verification-atp.todo.md
- Bundle strategy: explicit
- Graph parents: LFV-G000
- Graph depth: 1
- Objective heap index: 27
- Parallel lane: logic-formal-verification/atp
- Conflict policy: Own the new ATP adapter package/test and minimal compatibility shims; do not refactor native engines, Hammer, public exports, or router policy.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Changed paths:
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Interfaces: ATPCompatibilityBackends@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: LFV-G042
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/944a9457375a1d509bc7d682b71b5db6cb3081c99618e870a2e79656c0c2f6d3
- Canonical task CID: baguqeerasrfjivzxliovbg6h22blog25w3ftbaojsymoq4fc46lfnqgc63jq
- Semantic identity: objective-evidence-obligation/v1/8cb256d37bf7f0b347a1f6ca8b7e5a3b9e43521d7fa3d18b99f3cae923deb70c
- Acceptance subset: Vampire/E/TPTP and native DCEC/TDFOL results are typed, bounded, and source bound, heuristic/duck-typed success is removed, unreconstructed ATP output remains candidate, reviewed legacy behavior remains compatible.
- Preconditions: objective goal LFV-G042 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-proof-solver
- Merge fate: objective/LFV-G042
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/8cb256d37bf7f0b347a1f6ca8b7e5a3b9e43521d7fa3d18b99f3cae923deb70c
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Embedding query: Wrap native and legacy ATP/legal prover stacks behind canonical requests, capabilities, candidates, proof objects, countermodels, and compatibility receipts.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py
- Surplus group: objective/LFV-G042
- Merge key: 668b802c91fadd40
- Merge family: objective/LFV-G042
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
- Todo vector key: 8e9f9b520371ade0
- Acceptance: Objective scan filed this gap for LFV-G042. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-028-objective-gap-15201f1bef99.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/atp/adapters.py, ipfs_datasets_py/tests/integration/logic/backends/test_atp_legacy_adapters.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
