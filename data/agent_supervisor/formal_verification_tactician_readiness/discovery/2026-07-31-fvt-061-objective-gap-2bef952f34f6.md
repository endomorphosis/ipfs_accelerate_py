# FVT-061 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 2bef952f34f63733c9b7a92077e2e904e543362f
Goal id: FVT-G208
Goal title: Install and live-certify supported hyperproperty engines
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 7
Bundle: formal-verification-tactician/hyperproperty-vendor-toolchains
Parallel lane: formal-verification-tactician/hyperproperty-vendor-toolchains
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Correct the upstream identities and deploy real HyperLTL satisfiability, AutoHyper, and MCHyper toolchains on every declared supported host.
AST query: test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py, docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json
Conflict policy: Own vendor acquisition, correct per-product pins, dependencies, and live adapters; preserve bounded authority and never relabel the existing hermetic engines.
Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/hyperproperty.py, tools/logic/certification/hyperproperty.py, test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py, docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json
AST symbols: test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py, docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json
Interfaces: HyperpropertyVendorToolchainCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/ed3cc7bbf9d99b3d4e443f2012e46cf0a1d21d5ed66a0d80f6fd8ab9ff35102a
Acceptance subset: AutoHyper binds its official revision, .NET runtime, Spot tools, build inputs, executable digest, and live semantic cases, MCHyper binds its official revision, ABC/AIGER dependencies, supported fragment, and live witness/counterexample cases, the selected HyperLTL satisfiability engine has its own correct upstream identity and decidable-fragment ceiling, satisfaction, violation, observation/quantifier mutation, replay, malformed output, timeout, disagreement, and exact bounds execute through real binaries, linux-aarch64 remains supported only if that complete chain is real, case-oracle, hermetic shim, fixture, parser, or canned output cannot satisfy this goal.
Preconditions: objective goal FVT-G208 is schedulable
Effects: satisfy evidence requirement: test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py, satisfy evidence requirement: docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json
Evidence subset: test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py, docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json
Dependencies: FVT-G170, FVT-G201, FVT-G202
Resource class: cpu-proof-solver
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-solver
Merge fate: objective/FVT-G208
Rejection reasons: none (accepted)

## Goal

Correct the upstream identities and deploy real HyperLTL satisfiability, AutoHyper, and MCHyper toolchains on every declared supported host.

## Missing Evidence

- test/integration/toolchains/test_hyperproperty_vendor_toolchain_certification.py
- docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
