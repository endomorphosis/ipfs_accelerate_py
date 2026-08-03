# LFV-030 Objective Goal Gap

Date: 2026-07-29
Fingerprint: c62eeef6f9ada95351db94724983d7d4c7ac41fa
Goal id: LFV-G041
Goal title: Run shared verification conditions through Z3 and CVC5
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P0
Track: smt-execution
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 29
Bundle: logic-formal-verification/smt
Parallel lane: logic-formal-verification/smt
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Adapt the existing Z3 and CVC5 backends to the semantic compiler, typed results, models, unsat cores, exact receipts, and differential verification.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py, ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py
Conflict policy: Own the differential module/test and the Z3/CVC5 adapter integration edits; do not edit other providers, public API, or routing.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/z3/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/cvc5/compiler.py, ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py, ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py
Interfaces: Z3SoftwareVerificationBackend@1, CVC5SoftwareVerificationBackend@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/981492d12e22ccf7c223bbbe4755ce5d18fa9ecd81154903b0f1453313012555
Acceptance subset: Both adapters run identical canonical VCs when available, expose explicit unavailability otherwise, agree on reviewed fixtures, preserve disagreement evidence, reject malformed outputs, and bind versions/resources/translations.
Preconditions: objective goal LFV-G041 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py, ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py
Dependencies: LFV-G014, LFV-G040
Resource class: cpu-proof-solver
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-solver
Merge fate: objective/LFV-G041
Rejection reasons: none (accepted)

## Goal

Adapt the existing Z3 and CVC5 backends to the semantic compiler, typed results, models, unsat cores, exact receipts, and differential verification.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/smt/differential.py
- ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
