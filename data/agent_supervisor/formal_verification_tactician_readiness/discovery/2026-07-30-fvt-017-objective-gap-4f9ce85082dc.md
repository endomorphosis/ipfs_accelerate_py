# FVT-017 Objective Goal Gap

Date: 2026-07-30
Fingerprint: 4f9ce85082dc07de4455ced8a31e2365016a9c53
Goal id: FVT-G033
Goal title: Build the candidate lemma, invariant, contract, and evidence portfolio
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P1
Track: candidate-synthesis
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 16
Bundle: formal-verification-tactician/proof-search
Parallel lane: formal-verification-tactician/proof-search
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Combine exact corpus/cache/Hammer retrieval, reviewed templates, Houdini elimination, SMT cores/interpolation, CHC/PDR/IC3, SyGuS, legal evidence routing, and learned proposal/ranking providers into typed candidate sources.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py
Conflict policy: Own candidate-source composition and tests; reuse existing utilities through adapters and do not create independent caches, provider registries, or proof authority.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py
Interfaces: ProofCandidatePortfolio@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/38f5aaa0261a17c21e3a1f220595226152d7af9ff04cdb2db73a6b78e5dd95e8
Acceptance subset: Every candidate records source/provider/provenance/trust/budget and targeted holes, autoencoder, Leanstral, SymAI, embeddings, and model output remain proposal-only, legal obligations delegate evidence routing to the existing legal tactician compatibility adapter.
Preconditions: objective goal FVT-G033 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py, ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py
Dependencies: FVT-G030
Resource class: cpu-proof-portfolio
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-portfolio
Merge fate: objective/FVT-G033
Rejection reasons: none (accepted)

## Goal

Combine exact corpus/cache/Hammer retrieval, reviewed templates, Houdini elimination, SMT cores/interpolation, CHC/PDR/IC3, SyGuS, legal evidence routing, and learned proposal/ranking providers into typed candidate sources.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/tactician/candidate_synthesis.py
- ipfs_datasets_py/tests/unit/logic/software_verification/tactician/test_candidate_synthesis.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
