# LFV-010 Objective Goal Gap

Date: 2026-07-29
Fingerprint: cacb4964edb73c72abcc5219c053c10ad58222d8
Goal id: LFV-G023
Goal title: Add event, trace, LTL, LTLf, MTL, CTL, and CTL-star semantics
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P0
Track: temporal-semantics
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 9
Bundle: logic-formal-verification/semantics-state
Parallel lane: logic-formal-verification/semantics-state
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Define typed events, clocks, finite and infinite traces, path quantification, temporal formulas, intervals, observation policies, and monitorability.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py
Conflict policy: Own trace/temporal leaf modules and tests; adapt existing event/temporal types by explicit conversion without modifying them.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py
Interfaces: TraceIR@1, TemporalFormula@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/534902092f941bc6e4d0c69c98729db1bcf883f0c6929ff674a992eeae20ba7c
Acceptance subset: Finite-prefix, infinite-trace, and branching-time semantics are non-interchangeable, time units and interval boundaries are canonical, monitorable fragments are declared, CTL/CTL-star remain declaration/translation-only until a conformant semantics-preserving backend exists, clean prefixes never imply global proof.
Preconditions: objective goal LFV-G023 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py, satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py, ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py
Dependencies: LFV-G020, LFV-G021
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G023
Rejection reasons: none (accepted)

## Goal

Define typed events, clocks, finite and infinite traces, path quantification, temporal formulas, intervals, observation policies, and monitorability.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/trace.py
- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/temporal.py
- ipfs_datasets_py/tests/unit/logic/software_verification/test_temporal.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
