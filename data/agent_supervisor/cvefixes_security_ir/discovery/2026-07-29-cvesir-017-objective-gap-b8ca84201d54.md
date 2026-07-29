# CVESIR-017 Objective Goal Gap

Date: 2026-07-29
Fingerprint: b8ca84201d54796564674e3bf4622715c979abb5
Goal id: CVESIR-G020
Goal title: Canonical derived dataset schemas and identities
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: schema
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 16
Bundle: cvefixes-security-ir/schema
Parallel lane: cvefixes-security-ir/schema
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Define immutable schemas and canonical IDs for source records, code units, graph nodes/edges, policy candidates, formal views, evaluations, and release manifests.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
Conflict policy: Own new schema module and tests; reuse ir_core canonical identities.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/26678703f9fea694ab6720cc601665b729c139d805ee7a57b82ab90d8fc045e1
Acceptance subset: Canonical round-trip and CID stability pass, parent/source/config identities are mandatory, NaN, unknown fields, duplicate IDs, and authority broadening fail closed.
Preconditions: objective goal CVESIR-G020 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py
Dependencies: none
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G020
Rejection reasons: none (accepted)

## Goal

Define immutable schemas and canonical IDs for source records, code units, graph nodes/edges, policy candidates, formal views, evaluations, and release manifests.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/schemas.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_schemas.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
