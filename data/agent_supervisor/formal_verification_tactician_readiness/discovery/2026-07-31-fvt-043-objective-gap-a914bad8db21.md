# FVT-043 Objective Goal Gap

Date: 2026-07-31
Fingerprint: a914bad8db2135b9ecf61454b1b1f000d1b154d6
Goal id: FVT-G130
Goal title: Install and certify Tamarin with Maude
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 42
Bundle: formal-verification-tactician/tamarin-toolchain
Parallel lane: formal-verification-tactician/tamarin-toolchain
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Complete the exact Tamarin and compatible Maude installation and certify cryptographic-protocol claims and attacks.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
Conflict policy: Own the Tamarin/Maude installer plugin, handler, and test; do not edit the ProVerif lane, shared lock, or central certificate.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, tools/logic/certification/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
Interfaces: TamarinToolchainCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/7052cde3800c7eef47032e7c6738e2630c450823e3492870043c8117b791b6b3
Acceptance subset: Explicit strict installation selects Tamarin 1.12.0 and Maude 3.5.1, secure, attack, mutated claim/rule, replay, malformed output, timeout, and version mismatch cases pass, theory, claims, bounds, and exact binaries are bound, Maude is support only and cannot promote a property lane by itself.
Preconditions: objective goal FVT-G130 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, satisfy evidence requirement: test/integration/toolchains/test_tamarin_toolchain_certification.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py, test/integration/toolchains/test_tamarin_toolchain_certification.py
Dependencies: FVT-G110
Resource class: cpu-proof-solver
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-solver
Merge fate: objective/FVT-G130
Rejection reasons: none (accepted)

## Goal

Complete the exact Tamarin and compatible Maude installation and certify cryptographic-protocol claims and attacks.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/tamarin.py
- test/integration/toolchains/test_tamarin_toolchain_certification.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
