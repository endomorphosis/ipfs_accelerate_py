# CVESIR-004 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 581f2b23a868b37df7a8c37c05f0f7f7ec4d0f6b
Goal id: CVESIR-G060
Goal title: Bounded lexical, vector, and graph retrieval
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P1
Track: retrieval
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 3
Bundle: cvefixes-security-ir/graphrag
Parallel lane: cvefixes-security-ir/graphrag
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Add bounded hybrid retrieval over CWE, language, code facts, actions, effects, policies, and graph neighborhoods with partition and authority filters.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py
Conflict policy: Own CVEfixes retrieval; embeddings route through existing accelerator ports.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/cea7ee1b98cd79455a9995841376ea97c4ebef8701da7ae839c5b0a0c3229807
Acceptance subset: Queries cap shards/nodes/results, filters cannot broaden authority, indexes bind model/config/graph roots, tampering and split crossing fail closed, retrieval never returns a grant.
Preconditions: objective goal CVESIR-G060 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py
Dependencies: CVESIR-G050
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G060
Rejection reasons: none (accepted)

## Goal

Add bounded hybrid retrieval over CWE, language, code facts, actions, effects, policies, and graph neighborhoods with partition and authority filters.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/retrieval.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_retrieval.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
