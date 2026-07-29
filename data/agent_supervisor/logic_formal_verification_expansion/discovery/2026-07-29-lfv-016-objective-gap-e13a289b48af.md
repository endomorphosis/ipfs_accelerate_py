# LFV-016 Objective Goal Gap

Date: 2026-07-29
Fingerprint: e13a289b48af809fbf31b7cd3faebbaa1615b181
Goal id: LFV-G026
Goal title: Add separation, heap, ownership, and resource logic
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P1
Track: separation-logic
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 15
Bundle: logic-formal-verification/semantics-program
Parallel lane: logic-formal-verification/semantics-program
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Represent heaps, points-to assertions, separating conjunction, permissions, ownership transfer, disjointness, resource algebras, and frame obligations.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py
Conflict policy: Own heap/separation modules and tests; defer provider encodings and exports.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py
Interfaces: SeparationLogicIR@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/2ecb710bcce37b10caeb7de194f2b60e9d43a8fe2bd768bc07235fd979c71539
Acceptance subset: Ownership and aliasing are typed, separating and ordinary conjunction differ, permissions are bounded and conserved, frame inference emits explicit obligations, unsupported heap theories cannot silently lower to plain FOL.
Preconditions: objective goal LFV-G026 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py, satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py
Dependencies: LFV-G024
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G026
Rejection reasons: none (accepted)

## Goal

Represent heaps, points-to assertions, separating conjunction, permissions, ownership transfer, disjointness, resource algebras, and frame obligations.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/heap.py
- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/separation.py
- ipfs_datasets_py/tests/unit/logic/software_verification/test_separation.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
