# CVESIR-008 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 9fc1e7915e0a2c877beeb425a2afb6cf4da626c0
Goal id: CVESIR-G120
Goal title: Forbidden-logic formalization views
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: formalization
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 7
Bundle: cvefixes-security-ir/security-ir
Parallel lane: cvefixes-security-ir/security-ir
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Produce typed deontic prohibitions, threat premises, transition views, claims, and proof obligations through the existing Security IR formalization adapter.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py
Conflict policy: Extend formalization_adapter views via a domain adapter; do not create another prover.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/f4ca637dbf37421c035c229448a814a735e3b2a002ec082e91255bb6b49b3558
Acceptance subset: Typed symbols bind exact scope, deny maps to prohibition, unsupported semantics emit diagnostics, formulas and obligations are non-authoritative, vulnerable/fixed controls demonstrate polarity.
Preconditions: objective goal CVESIR-G120 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py
Dependencies: CVESIR-G110
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G120
Rejection reasons: none (accepted)

## Goal

Produce typed deontic prohibitions, threat premises, transition views, claims, and proof obligations through the existing Security IR formalization adapter.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/formalize.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_formalize.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
