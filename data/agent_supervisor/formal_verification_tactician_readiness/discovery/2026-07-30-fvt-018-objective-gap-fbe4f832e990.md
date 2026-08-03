# FVT-018 Objective Goal Gap

Date: 2026-07-30
Fingerprint: fbe4f832e9904a6715eb70e1c13bf0b68417adee
Goal id: FVT-G024
Goal title: Compile confirmed goals into shared verification semantics
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: goal-compilation
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 17
Bundle: formal-verification-tactician/goal-contracts
Parallel lane: formal-verification-tactician/goal-contracts
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Compile a confirmed EndGoalSpec into SoftwareVerificationIR properties, contracts, transition/environment models, and backend-neutral root obligations with a loss-aware translation receipt.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py
Conflict policy: Own goal-to-shared-IR composition and integration test; reuse LFV semantics and translation receipts rather than embedding provider syntax in EndGoalSpec.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py
Interfaces: FormalGoalCompiler@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/07a58fc416a80cb409f429320231fcd8fd05774a58ecd0b02e15cf486d1752a2
Acceptance subset: Exact targets and bounds reproduce from content identities, source spans and assumption classes survive, material translation loss or ambiguity fails closed, backend choice cannot raise assurance above the translation ceiling.
Preconditions: objective goal FVT-G024 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py, ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py
Dependencies: FVT-G011, FVT-G023
Resource class: cpu-proof-translate
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-translate
Merge fate: objective/FVT-G024
Rejection reasons: none (accepted)

## Goal

Compile a confirmed EndGoalSpec into SoftwareVerificationIR properties, contracts, transition/environment models, and backend-neutral root obligations with a loss-aware translation receipt.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/goal_compiler.py
- ipfs_datasets_py/tests/integration/logic/software_verification/tactician/test_goal_compiler.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
