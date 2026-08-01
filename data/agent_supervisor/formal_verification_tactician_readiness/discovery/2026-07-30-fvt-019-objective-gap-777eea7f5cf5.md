# FVT-019 Objective Goal Gap

Date: 2026-07-30
Fingerprint: 777eea7f5cf52c62257ca17c08dbadfd30f1a2bf
Goal id: FVT-G031
Goal title: Construct a bounded backward AND/OR obligation graph
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: backward-proof-search
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 18
Bundle: formal-verification-tactician/proof-search
Parallel lane: formal-verification-tactician/proof-search
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Regress formal targets through programs and transition systems using weakest preconditions, preimages, temporal regression, typed rule inversion/unification, subsumption, cycle control, and reconstructable AND/OR proof rules.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py
Conflict policy: Own the new general proof graph and tests; wrap legacy CEC/TDFOL strategies as experimental candidates unless they reconstruct through the typed rules.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py
Interfaces: BackwardProofObligationGraph@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/0ab50291c86289b345671ef58fd79f33dc59b9fc02d5b52983d01b998cefa7e0
Acceptance subset: Every edge names a checked inference/reconstruction rule, AND/OR meanings are distinct, finite budgets, SCC/cycle and subsumption controls terminate, solved leaves cite adequate evidence, legacy string-equality or forward-only “backward” paths cannot receive trusted status.
Preconditions: objective goal FVT-G031 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py
Dependencies: FVT-G024, FVT-G030
Resource class: cpu-proof-search
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-search
Merge fate: objective/FVT-G031
Rejection reasons: none (accepted)

## Goal

Regress formal targets through programs and transition systems using weakest preconditions, preimages, temporal regression, typed rule inversion/unification, subsumption, cycle control, and reconstructable AND/OR proof rules.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_graph.py
- ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_graph.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
