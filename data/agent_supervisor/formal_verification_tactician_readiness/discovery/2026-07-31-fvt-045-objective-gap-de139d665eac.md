# FVT-045 Objective Goal Gap

Date: 2026-07-31
Fingerprint: de139d665eaca78fd7966aa5368ea659ab2e5660
Goal id: FVT-G150
Goal title: Install and semantically certify Rocq
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 44
Bundle: formal-verification-tactician/rocq-toolchain
Parallel lane: formal-verification-tactician/rocq-toolchain
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Complete isolated installation and real kernel certification for the locked Rocq/Coq provider.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
Conflict policy: Own the Rocq installer plugin, handler, and test; serialize OPAM resource use with ProVerif and never modify a global switch.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, tools/logic/certification/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
Interfaces: RocqToolchainCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/aecd9b7e5259b4de83893c54d77b2b10d391c0fe9551902626bd20890676adf8
Acceptance subset: Explicit strict installation selects Rocq 9.1.1 in an isolated pinned OPAM root, true proof, false proof, hypothesis/conclusion mutation, deterministic replay, forbidden admits/axiom escapes, malformed input, and mismatch checks pass, receipts bind imports, source, theorem, assumptions, and exact kernel identity.
Preconditions: objective goal FVT-G150 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, satisfy evidence requirement: test/integration/toolchains/test_rocq_toolchain_certification.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py, test/integration/toolchains/test_rocq_toolchain_certification.py
Dependencies: FVT-G110
Resource class: exclusive-opam-toolchain
Token class: medium
Estimated tokens: 0
Resources: exclusive-opam-toolchain
Merge fate: objective/FVT-G150
Rejection reasons: none (accepted)

## Goal

Complete isolated installation and real kernel certification for the locked Rocq/Coq provider.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/rocq.py
- test/integration/toolchains/test_rocq_toolchain_certification.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
