# FVT-048 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 29ccdb6feffc2bfd595e374386c8261bdcd158c7
Goal id: FVT-G140
Goal title: Install and certify Vampire and E ATP
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P1
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 47
Bundle: formal-verification-tactician/atp-toolchains
Parallel lane: formal-verification-tactician/atp-toolchains
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Complete exact Vampire and E prover installation and certify theorem/non-theorem behavior for premise and proof search.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
Conflict policy: Own ATP installer plugins, handler, and test; do not edit CEC semantics, shared lock, or central certificate.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, tools/logic/certification/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
Interfaces: ATPToolchainCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/81697f5ecaaf2cab6336fdda2022c2b991018eaff190bc6314dd429c91cf370f
Acceptance subset: Explicit strict installation selects Vampire 5.0.1 and E 3.2.5, theorem, non-theorem, premise/conclusion mutation, proof-output binding, replay, malformed output, and timeout checks pass, ATP results remain candidates unless an allowed independent kernel reconstruction validates them.
Preconditions: objective goal FVT-G140 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, satisfy evidence requirement: test/integration/toolchains/test_atp_toolchain_certification.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py, test/integration/toolchains/test_atp_toolchain_certification.py
Dependencies: FVT-G110
Resource class: cpu-proof-solver
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-solver
Merge fate: objective/FVT-G140
Rejection reasons: none (accepted)

## Goal

Complete exact Vampire and E prover installation and certify theorem/non-theorem behavior for premise and proof search.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/atp.py
- test/integration/toolchains/test_atp_toolchain_certification.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
