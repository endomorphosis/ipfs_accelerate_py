# CVESIR-018 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 8b2c34cd306fa9617bd6b099e564784cf136161a
Goal id: CVESIR-G140
Goal title: Generated-code security fact contract and extractors
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: code-facts
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 17
Bundle: cvefixes-security-ir/supervisor
Parallel lane: cvefixes-security-ir/supervisor
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Define canonical code-security facts and deterministic changed-diff extractors for actions, targets, data flow, effects, capabilities, guards, language, and source scope.
AST query: ipfs_accelerate_py/agent_supervisor/code_security_facts.py, test/api/test_agent_supervisor_code_security_facts.py
Conflict policy: Own new fact module and tests; do not edit decision runtime yet.
Predicted files: ipfs_accelerate_py/agent_supervisor/code_security_facts.py, test/api/test_agent_supervisor_code_security_facts.py
AST symbols: ipfs_accelerate_py/agent_supervisor/code_security_facts.py, test/api/test_agent_supervisor_code_security_facts.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/5523e310dc44af1887f8eb86185660904fda707653372081a53fdb699cd909ea
Acceptance subset: Facts bind tree/blob/diff/AST identities, only changed scope is attributed, unsupported/ambiguous extraction is explicit, source strings cannot inject facts, extractors never grant authority.
Preconditions: objective goal CVESIR-G140 is schedulable
Effects: satisfy evidence requirement: ipfs_accelerate_py/agent_supervisor/code_security_facts.py, satisfy evidence requirement: test/api/test_agent_supervisor_code_security_facts.py
Evidence subset: ipfs_accelerate_py/agent_supervisor/code_security_facts.py, test/api/test_agent_supervisor_code_security_facts.py
Dependencies: none
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G140
Rejection reasons: none (accepted)

## Goal

Define canonical code-security facts and deterministic changed-diff extractors for actions, targets, data flow, effects, capabilities, guards, language, and source scope.

## Missing Evidence

- ipfs_accelerate_py/agent_supervisor/code_security_facts.py
- test/api/test_agent_supervisor_code_security_facts.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
