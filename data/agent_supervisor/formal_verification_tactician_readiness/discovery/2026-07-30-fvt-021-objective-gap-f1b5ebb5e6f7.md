# FVT-021 Objective Goal Gap

Date: 2026-07-30
Fingerprint: f1b5ebb5e6f76b2b74afb0bab91af5c0c80f27f2
Goal id: FVT-G013
Goal title: Replace manifest-only examples and synthetic readiness claims
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P1
Track: runnable-examples
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 20
Bundle: formal-verification-tactician/readiness
Parallel lane: formal-verification-tactician/readiness
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Check in the referenced example sources and mutations, run them through production entrypoints, and derive outcome/security/readiness reports from actual receipts rather than manually injected witnesses or hardcoded distributions.
AST query: ipfs_datasets_py/examples/logic/software_verification/README.md, ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py
Conflict policy: Own example sources, runnable integration test, and live report; retain small deterministic fixtures but remove them from production-readiness claims.
Predicted files: ipfs_datasets_py/examples/logic/software_verification/README.md, ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py, docs/architecture/formal_verification_live_example_report.json
AST symbols: ipfs_datasets_py/examples/logic/software_verification/README.md, ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py
Interfaces: RunnableVerificationExamples@1, LiveReadinessReport@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/b6b4b29ace97e3920a7b1db049df93103d90d9bf379df612abb72fb7293e1226
Acceptance subset: Every manifest source exists and runs, negative variants generate rather than inject counterexamples, positive variants generate current receipts, reports cite run identities and clearly separate fixture, simulated, live, skipped, unsupported, and unavailable results.
Preconditions: objective goal FVT-G013 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/examples/logic/software_verification/README.md, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py
Evidence subset: ipfs_datasets_py/examples/logic/software_verification/README.md, ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py
Dependencies: FVT-G011, FVT-G012
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G013
Rejection reasons: none (accepted)

## Goal

Check in the referenced example sources and mutations, run them through production entrypoints, and derive outcome/security/readiness reports from actual receipts rather than manually injected witnesses or hardcoded distributions.

## Missing Evidence

- ipfs_datasets_py/examples/logic/software_verification/README.md
- ipfs_datasets_py/tests/integration/logic/software_verification/test_runnable_examples.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
