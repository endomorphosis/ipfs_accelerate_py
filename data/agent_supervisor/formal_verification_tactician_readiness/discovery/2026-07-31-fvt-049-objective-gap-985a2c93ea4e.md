# FVT-049 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 985a2c93ea4e6d9e862406a8250190781f1b66db
Goal id: FVT-G151
Goal title: Install and semantically certify Isabelle
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P1
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 48
Bundle: formal-verification-tactician/isabelle-toolchain
Parallel lane: formal-verification-tactician/isabelle-toolchain
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Complete the pinned Isabelle installation and real session/kernel certification used for reconstruction and Hammer validation.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
Conflict policy: Own the Isabelle installer plugin, handler, and test; observe an explicit large-download/storage budget and do not edit the shared lock or central certificate.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
Interfaces: IsabelleToolchainCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/0c06bcc700647549f133f76ae385c2a83227eda44b8e647d4f6bf152268cb569
Acceptance subset: Explicit strict installation selects Isabelle2025-2, a checked theory/session passes while bad proof, mutated assumptions/conclusion, replay mismatch, malformed output, timeout, and wrong installation fail, theory heap, session, imports, source, property, and exact tool identity are bound, Hammer remains proposal-only until kernel reconstruction.
Preconditions: objective goal FVT-G151 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, satisfy evidence requirement: test/integration/toolchains/test_isabelle_toolchain_certification.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
Dependencies: FVT-G110
Resource class: large-kernel-toolchain
Token class: medium
Estimated tokens: 0
Resources: large-kernel-toolchain
Merge fate: objective/FVT-G151
Rejection reasons: none (accepted)

## Goal

Complete the pinned Isabelle installation and real session/kernel certification used for reconstruction and Hammer validation.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py
- test/integration/toolchains/test_isabelle_toolchain_certification.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
