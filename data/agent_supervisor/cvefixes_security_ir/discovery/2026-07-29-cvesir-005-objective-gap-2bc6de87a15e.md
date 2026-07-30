# CVESIR-005 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 2bc6de87a15eb05e36289d8e9cd130ea28fc47a4
Goal id: CVESIR-G170
Goal title: Bounded security decision receipts and observability
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P1
Track: receipts
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 4
Bundle: cvefixes-security-ir/enforcement
Parallel lane: cvefixes-security-ir/enforcement
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Emit explainable bounded receipts linking intent/code facts, matching policies, CVE/CWE/source CIDs, roots, reason codes, counterexamples, and enforcement stage.
AST query: ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py, test/api/test_agent_supervisor_cve_security_receipts.py
Conflict policy: Own receipt module and tests; extend event fields without logging code bodies or secrets.
Predicted files: ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py, test/api/test_agent_supervisor_cve_security_receipts.py
AST symbols: ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py, test/api/test_agent_supervisor_cve_security_receipts.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/59bf8f9dc4cc504b1a56f6cff3b96d87a89e631954f174da51ea116600a26a86
Acceptance subset: Receipts are canonical, bounded, redacted, stage/tree/root-bound, and distinguish evidence from authority, cache keys invalidate on every declared dependency.
Preconditions: objective goal CVESIR-G170 is schedulable
Effects: satisfy evidence requirement: ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py, satisfy evidence requirement: test/api/test_agent_supervisor_cve_security_receipts.py
Evidence subset: ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py, test/api/test_agent_supervisor_cve_security_receipts.py
Dependencies: CVESIR-G150, CVESIR-G160
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G170
Rejection reasons: none (accepted)

## Goal

Emit explainable bounded receipts linking intent/code facts, matching policies, CVE/CWE/source CIDs, roots, reason codes, counterexamples, and enforcement stage.

## Missing Evidence

- ipfs_accelerate_py/agent_supervisor/cve_security_receipts.py
- test/api/test_agent_supervisor_cve_security_receipts.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
