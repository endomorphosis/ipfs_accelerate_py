# LFV-033 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 4d0fd4d5b8a721d382f71fe45004830189db7887
Goal id: LFV-G061
Goal title: Normalize Leanstral and SymAI as untrusted proposal providers
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P1
Track: proposal-advisors
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 32
Bundle: logic-formal-verification/advisors
Parallel lane: logic-formal-verification/advisors
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Provide strict advisor adapters for specification, lemma, tactic, premise, and repair proposals and remove legacy neural routes that infer proof from `is_valid` or confidence.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py, ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py
Conflict policy: Own the new advisor/test and narrowly repair the two identified neural proof-authority defects; do not refactor model runtimes or kernel backends.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/prover_router.py, ipfs_datasets_py/ipfs_datasets_py/logic/integration/symbolic/neurosymbolic/reasoning_coordinator.py, ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py, ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py
Interfaces: LeanstralAdvisor@1, SymAIAdvisor@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/05873f22fc778d5321f011f2adcee132d3ba6642c388a7b00f07ceab6e8e4748
Acceptance subset: Inputs/outputs are bounded and sanitized, prompts/responses are inert and source bound, generic `is_valid`, similarity, or confidence never yields proof, accepted candidates require deterministic compilation and independent solver/kernel validation.
Preconditions: objective goal LFV-G061 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py, ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py
Dependencies: LFV-G049, LFV-G060
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G061
Rejection reasons: none (accepted)

## Goal

Provide strict advisor adapters for specification, lemma, tactic, premise, and repair proposals and remove legacy neural routes that infer proof from `is_valid` or confidence.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/formalization/proposal_advisors.py
- ipfs_datasets_py/tests/integration/logic/test_proposal_advisors.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
