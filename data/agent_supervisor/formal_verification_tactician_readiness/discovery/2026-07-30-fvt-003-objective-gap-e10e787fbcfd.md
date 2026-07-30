# FVT-003 Objective Goal Gap

Date: 2026-07-30
Fingerprint: e10e787fbcfdaff807cf81a185aec6405ea65721
Goal id: FVT-G006
Goal title: Make receipt verification and attestation fail closed
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: receipt-authority
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 2
Bundle: formal-verification-tactician/trust-boundary
Parallel lane: formal-verification-tactician/trust-boundary
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace permissive structural receipt handling with closed schema dispatch and exact validation of content identity, source/property/assumption/bound/tool bindings, freshness, authority, proof artifacts, and independent checker evidence.
AST query: ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py, test/api/test_logic_receipt_authority_boundary.py
Conflict policy: Own stable receipt and attestation dispatch plus adversarial tests; preserve existing typed receipt schemas and do not weaken them to accommodate legacy mappings.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py, test/api/test_logic_receipt_authority_boundary.py
AST symbols: ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py, test/api/test_logic_receipt_authority_boundary.py
Interfaces: VerifiedReceiptDispatch@2, AttestationAuthorityBoundary@2
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/3ccddcedda257673f37f0549326bba38258afe72196498d6307dfbdfbea58b7b
Acceptance subset: Empty, unknown, forged-kernel, stale, wrong-tree, wrong-property, wrong-assumption, wrong-bound, wrong-tool, and cross-authority inputs are rejected, a prepared/simulated attestation cannot report proof success, valid typed receipts round trip without authority loss.
Preconditions: objective goal FVT-G006 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py, satisfy evidence requirement: test/api/test_logic_receipt_authority_boundary.py
Evidence subset: ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py, test/api/test_logic_receipt_authority_boundary.py
Dependencies: FVT-G005
Resource class: cpu-proof-type-check
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-type-check
Merge fate: objective/FVT-G006
Rejection reasons: none (accepted)

## Goal

Replace permissive structural receipt handling with closed schema dispatch and exact validation of content identity, source/property/assumption/bound/tool bindings, freshness, authority, proof artifacts, and independent checker evidence.

## Missing Evidence

- ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py
- test/api/test_logic_receipt_authority_boundary.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
