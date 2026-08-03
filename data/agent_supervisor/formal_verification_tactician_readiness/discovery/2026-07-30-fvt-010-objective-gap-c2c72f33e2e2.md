# FVT-010 Objective Goal Gap

Date: 2026-07-30
Fingerprint: c2c72f33e2e2f6103950c5ae27779c597e196dfc
Goal id: FVT-G011
Goal title: Build the source-to-VC-to-solver vertical slice
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: executable-pipeline
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 9
Bundle: formal-verification-tactician/vertical-slice
Parallel lane: formal-verification-tactician/vertical-slice
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Connect a source snapshot through typed program/contracts, verification-condition generation, backend-neutral SMT obligations, Z3/CVC5 execution, and source-bound proof/counterexample receipts.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py
Conflict policy: Own the new pipeline composition and integration test; reuse existing source, ProgramIR, VC, SMT compiler, runner, and receipt modules without inventing parallel semantic contracts.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py
Interfaces: SourceToVerificationPipeline@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/ba118751bde2c1724e8a30b5365ec957b76215f7bc73ec1fd658805a6b2d94ce
Acceptance subset: Checked-in buggy/fixed programs generate their own VCs and witnesses, Z3 and CVC5 agree or disagreement is quarantined, every result binds source spans/tree/property/assumptions/tool/bounds/translation, unsupported constructs fail explicitly rather than being erased.
Preconditions: objective goal FVT-G011 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py, ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py
Dependencies: FVT-G006, FVT-G007, FVT-G009
Resource class: cpu-proof-smt
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-smt
Merge fate: objective/FVT-G011
Rejection reasons: none (accepted)

## Goal

Connect a source snapshot through typed program/contracts, verification-condition generation, backend-neutral SMT obligations, Z3/CVC5 execution, and source-bound proof/counterexample receipts.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/pipeline.py
- ipfs_datasets_py/tests/integration/logic/software_verification/test_source_vc_smt_pipeline.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
