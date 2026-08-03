# CVESIR-015 Objective Goal Gap

Date: 2026-07-29
Fingerprint: d1063da4102f2e3cb229159d18ce486c00fcdc7f
Goal id: CVESIR-G160
Goal title: Decision-runtime, permit, and merge enforcement
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: enforcement
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 14
Bundle: cvefixes-security-ir/enforcement
Parallel lane: cvefixes-security-ir/enforcement
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Wire the CVE security gate into plan admission, pre-execution permits, post-generation validation, merge admission, and merged-tree revalidation.
AST query: test/api/test_agent_supervisor_cve_security_enforcement.py
Conflict policy: Sole task allowed to edit the three shared enforcement files; preserve backward-compatible existing decisions and deny-overrides.
Predicted files: ipfs_accelerate_py/agent_supervisor/security_constraint_adapter.py, ipfs_accelerate_py/agent_supervisor/ir_constraint_compiler.py, ipfs_accelerate_py/agent_supervisor/execution_permit.py, test/api/test_agent_supervisor_cve_security_enforcement.py
AST symbols: test/api/test_agent_supervisor_cve_security_enforcement.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/add9042546b09921f838976124fccba0e29f75d8e3163d0c6e81ed682dad4f52
Acceptance subset: Deny/conflict/unknown/stale reject, allow requires existing authority, generated undeclared effects reject, root or tree drift invalidates receipts, no permit or merge path bypasses the gate.
Preconditions: objective goal CVESIR-G160 is schedulable
Effects: satisfy evidence requirement: test/api/test_agent_supervisor_cve_security_enforcement.py
Evidence subset: test/api/test_agent_supervisor_cve_security_enforcement.py
Dependencies: CVESIR-G150
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G160
Rejection reasons: none (accepted)

## Goal

Wire the CVE security gate into plan admission, pre-execution permits, post-generation validation, merge admission, and merged-tree revalidation.

## Missing Evidence

- test/api/test_agent_supervisor_cve_security_enforcement.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
