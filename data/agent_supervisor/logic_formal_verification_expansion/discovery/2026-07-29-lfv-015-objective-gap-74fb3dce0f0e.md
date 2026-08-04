# LFV-015 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 74fb3dce0f0e5276edea4e8c0046e7777073ae8a
Goal id: LFV-G025
Goal title: Generate weakest preconditions and verification conditions
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P0
Track: vc-generation
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 14
Bundle: logic-formal-verification/semantics-program
Parallel lane: logic-formal-verification/semantics-program
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Generate source-bound weakest-precondition and verification-condition obligations for contracts, branches, loops, exceptions, frames, and resource assertions.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py
Conflict policy: Own the VC generator and test; consume program and translation contracts without editing their definitions or any provider.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py
Interfaces: VerificationConditionGenerator@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/0ad54a9a8c74c1dab026fdfe6905a1be75c22fd3e4a0fa642f09eef4e927bd21
Acceptance subset: Each obligation binds its source construct, assumptions, generated symbols, rule, and parent contract, loop rules require invariant/variant policy, unsupported effects remain explicit, mutation tests detect dropped branches and frames.
Preconditions: objective goal LFV-G025 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py
Dependencies: LFV-G021, LFV-G024
Resource class: cpu-proof-translate
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-translate
Merge fate: objective/LFV-G025
Rejection reasons: none (accepted)

## Goal

Generate source-bound weakest-precondition and verification-condition obligations for contracts, branches, loops, exceptions, frames, and resource assertions.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/vc.py
- ipfs_datasets_py/tests/unit/logic/software_verification/test_vc.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
