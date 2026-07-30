# FVT-002 Objective Goal Gap

Date: 2026-07-30
Fingerprint: d3ffe0b861181f2add777b45393dead3b8f7e5dc
Goal id: FVT-G007
Goal title: Unify the secret-safe public counterexample boundary
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: counterexample-boundary
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 1
Bundle: formal-verification-tactician/trust-boundary
Parallel lane: formal-verification-tactician/trust-boundary
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Route datasets Python/CLI/MCP and supervisor/model projections through one closed, bounded, content-addressed counterexample envelope and eliminate raw payload exposure.
AST query: ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py, test/api/test_counterexample_cross_repository_contract.py
Conflict policy: Own the new datasets wire contract, verification API delegation, and cross-repository adapter tests; extend the mature supervisor normalizer without creating a second semantic identity.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py, test/api/test_counterexample_cross_repository_contract.py
AST symbols: ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py, test/api/test_counterexample_cross_repository_contract.py
Interfaces: CounterexampleEnvelope@2, PublicCounterexampleBoundary@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/354741044b7242c2b3136059a83f20492130b377b0149d4017c77056ef5db7be
Acceptance subset: Unknown fields and forged identities fail closed, hidden_witness, token, credential, raw source, stdout, and private channels never appear publicly, raw artifacts are referenced only by private digest/retention metadata, all projections preserve kind, property, source-map, tool, assumptions, bounds, and authority.
Preconditions: objective goal FVT-G007 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py, satisfy evidence requirement: test/api/test_counterexample_cross_repository_contract.py
Evidence subset: ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py, test/api/test_counterexample_cross_repository_contract.py
Dependencies: FVT-G005
Resource class: cpu-proof-sanitize
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-sanitize
Merge fate: objective/FVT-G007
Rejection reasons: none (accepted)

## Goal

Route datasets Python/CLI/MCP and supervisor/model projections through one closed, bounded, content-addressed counterexample envelope and eliminate raw payload exposure.

## Missing Evidence

- ipfs_datasets_py/tests/unit/logic/test_counterexample_public_boundary.py
- test/api/test_counterexample_cross_repository_contract.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
