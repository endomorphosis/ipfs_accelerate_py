# CVESIR-014 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 0be16a189e95c3835c903f05c436647a5769a5d9
Goal id: CVESIR-G150
Goal title: Intent/code facts to exact SecurityAuthorizationRequest
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: comparison
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 13
Bundle: cvefixes-security-ir/supervisor
Parallel lane: cvefixes-security-ir/supervisor
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Map pinned Intent IR and generated code facts independently to exact security requests and correlate undeclared, broadened, or contradictory effects.
AST query: ipfs_accelerate_py/agent_supervisor/cve_security_gate.py, test/api/test_agent_supervisor_cve_security_gate.py
Conflict policy: Own new gate module; call existing intent/security adapters rather than duplicate them.
Predicted files: ipfs_accelerate_py/agent_supervisor/cve_security_gate.py, test/api/test_agent_supervisor_cve_security_gate.py
AST symbols: ipfs_accelerate_py/agent_supervisor/cve_security_gate.py, test/api/test_agent_supervisor_cve_security_gate.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/73b8b37d87c78af8228a42b58397678ecbf2b083baffeeb1e72013ce94c5db41
Acceptance subset: Every request binds principal/action/tool/target/data_flow/effect/state/authority and Security IR root, intent pass cannot mask code fail, ambiguous mappings are unknown, exact matching follows existing adapter contracts.
Preconditions: objective goal CVESIR-G150 is schedulable
Effects: satisfy evidence requirement: ipfs_accelerate_py/agent_supervisor/cve_security_gate.py, satisfy evidence requirement: test/api/test_agent_supervisor_cve_security_gate.py
Evidence subset: ipfs_accelerate_py/agent_supervisor/cve_security_gate.py, test/api/test_agent_supervisor_cve_security_gate.py
Dependencies: CVESIR-G120, CVESIR-G130, CVESIR-G140
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G150
Rejection reasons: none (accepted)

## Goal

Map pinned Intent IR and generated code facts independently to exact security requests and correlate undeclared, broadened, or contradictory effects.

## Missing Evidence

- ipfs_accelerate_py/agent_supervisor/cve_security_gate.py
- test/api/test_agent_supervisor_cve_security_gate.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
