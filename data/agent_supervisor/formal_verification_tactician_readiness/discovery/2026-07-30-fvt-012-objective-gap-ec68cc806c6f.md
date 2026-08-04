# FVT-012 Objective Goal Gap

Date: 2026-07-30
Fingerprint: ec68cc806c6f47f1b02c7e026bf0d8078bd4f6f5
Goal id: FVT-G030
Goal title: Emit actionable typed proof holes
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: proof-holes
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 11
Bundle: formal-verification-tactician/proof-search
Parallel lane: formal-verification-tactician/proof-search
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Make VC and model compilation return source-bound typed holes for missing invariants, variants, contracts, frames, summaries, concurrency/temporal/refinement premises, bridge lemmas, evidence, semantics, tools, and necessary implementation changes.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py
Conflict policy: Own proof-hole contracts/adapters and focused VC behavior; retain fail-closed compilation and do not invent default invariants or contracts.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py
Interfaces: TypedProofHoleEmitter@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/87de7b0e3223e0ee350cfe33fbd7b66e3dfe50c4f98b569dd12683ba3c6b0ecf
Acceptance subset: Removing a loop invariant, callee contract/frame, fairness premise, or bridge lemma yields the matching typed hole with source span, rationale, dependencies, expected authority, and validation recipe, unsupported semantics remain different from missing proof.
Preconditions: objective goal FVT-G030 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py
Dependencies: FVT-G011, FVT-G021
Resource class: cpu-proof-compile
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-compile
Merge fate: objective/FVT-G030
Rejection reasons: none (accepted)

## Goal

Make VC and model compilation return source-bound typed holes for missing invariants, variants, contracts, frames, summaries, concurrency/temporal/refinement premises, bridge lemmas, evidence, semantics, tools, and necessary implementation changes.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/proof_holes.py
- ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_proof_holes.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
