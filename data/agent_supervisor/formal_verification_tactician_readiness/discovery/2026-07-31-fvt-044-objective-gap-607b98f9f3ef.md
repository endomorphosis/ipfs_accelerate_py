# FVT-044 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 607b98f9f3efdf68b6ddb2ac64a8c5f4d57ae49a
Goal id: FVT-G131
Goal title: Install and certify ProVerif in isolated OPAM
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 43
Bundle: formal-verification-tactician/proverif-toolchain
Parallel lane: formal-verification-tactician/proverif-toolchain
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Complete an isolated pinned OPAM/ProVerif deployment and semantic protocol certification without mutating global switches.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
Conflict policy: Own the ProVerif installer plugin, handler, isolated root contract, and test; serialize OPAM resource use with Rocq and never modify a global OPAM switch.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, tools/logic/certification/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
Interfaces: ProVerifToolchainCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/257be310d2c4edf64ed440bc9d96d6ec7d50cecd7ad7569d43995801957b48c4
Acceptance subset: Explicit strict installation selects OPAM 2.5.2 support and ProVerif 2.05 in a repository-local isolated root, secure, attack, mutation, replay, malformed output, cancellation, and mismatch checks pass, model and claim identities bind receipts, OPAM alone has no semantic authority.
Preconditions: objective goal FVT-G131 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, satisfy evidence requirement: test/integration/toolchains/test_proverif_toolchain_certification.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py, test/integration/toolchains/test_proverif_toolchain_certification.py
Dependencies: FVT-G110
Resource class: exclusive-opam-toolchain
Token class: medium
Estimated tokens: 0
Resources: exclusive-opam-toolchain
Merge fate: objective/FVT-G131
Rejection reasons: none (accepted)

## Goal

Complete an isolated pinned OPAM/ProVerif deployment and semantic protocol certification without mutating global switches.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/proverif.py
- test/integration/toolchains/test_proverif_toolchain_certification.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
