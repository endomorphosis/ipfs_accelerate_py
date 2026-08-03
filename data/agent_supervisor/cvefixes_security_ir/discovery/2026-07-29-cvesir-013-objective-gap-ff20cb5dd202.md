# CVESIR-013 Objective Goal Gap

Date: 2026-07-29
Fingerprint: ff20cb5dd202e61b42c5221153a69ea576d1fa09
Goal id: CVESIR-G110
Goal title: CVE candidates to canonical Security IR adapter
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: security-adapter
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 12
Bundle: cvefixes-security-ir/security-ir
Parallel lane: cvefixes-security-ir/security-ir
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Convert grounded candidate/reviewed records into canonical SecuritySource, Resource, Policy(DENY), ThreatAssumption, SecurityClaim, and optional StateMachine declarations without importing results into declarations.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py
Conflict policy: Extend canonical Security IR and ir_core contracts; no parallel authority/schema.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/7baa4fad92925f62fcfc379a4ecd90f087b3046cce58a1061b6d01db8097e93f
Acceptance subset: Mapping is loss-aware and round-trippable, source and review state are mandatory, candidates cannot claim authoritative result state, wildcard/generalized scopes require explicit review.
Preconditions: objective goal CVESIR-G110 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py
Dependencies: CVESIR-G020, CVESIR-G040, CVESIR-G100
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G110
Rejection reasons: none (accepted)

## Goal

Convert grounded candidate/reviewed records into canonical SecuritySource, Resource, Policy(DENY), ThreatAssumption, SecurityClaim, and optional StateMachine declarations without importing results into declarations.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/adapter.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_adapter.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
