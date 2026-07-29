# CVESIR-006 Objective Goal Gap

Date: 2026-07-29
Fingerprint: e4dcf797d4d7794e48832bb4956ab4f3ca335614
Goal id: CVESIR-G050
Goal title: Typed GraphRAG graph builder
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: graph
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 5
Bundle: cvefixes-security-ir/graphrag
Parallel lane: cvefixes-security-ir/graphrag
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Build the reviewed node/edge ontology, integrity-bound graph tables, adjacency indexes, and deterministic graph root.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py
Conflict policy: Own graph materialization; reuse shared GraphRAG primitives where compatible.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/1049ec6b51870e9ea6bd18b54269d9923c173a37680505b907e8d8114132e172
Acceptance subset: Node and edge types/directions validate, all edges bind sources and endpoints, similarity edges are marked non-authoritative, graph rebuild is deterministic and detects tampering.
Preconditions: objective goal CVESIR-G050 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py
Dependencies: CVESIR-G040
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G050
Rejection reasons: none (accepted)

## Goal

Build the reviewed node/edge ontology, integrity-bound graph tables, adjacency indexes, and deterministic graph root.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/graph.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_graph.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
