# FVT-086 Objective Goal Gap

Date: 2026-08-03
Fingerprint: d9aca6db2edec24b1f09a1f84e67a38fa5b8ac0f
Goal id: FVT-G217
Goal title: Implement the genuine SecPAL external-toolchain path
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 2
Bundle: formal-verification-tactician/secpal-live-toolchain
Parallel lane: formal-verification-tactician/secpal-live-toolchain
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace ambiguous SecPAL acquisition and adapter behavior with an official-artifact, license-aware, host-specific lazy installer and live semantic runner.
AST query: test/integration/toolchains/test_secpal_live_toolchain_contract.py
Conflict policy: Own SecPAL artifact provenance, platform matrix, installer, and external semantics; never invent an upstream release, accept an unreviewed mirror, bypass license terms, or label the in-process engine as the external vendor binary.
Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_secpal_live_toolchain_contract.py
AST symbols: test/integration/toolchains/test_secpal_live_toolchain_contract.py
Interfaces: SecPALLiveToolchainContract@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/d0a201df52ad6c2dc470b370be114e87721612b0dd93527a071046a825aa86ff
Acceptance subset: Every supported SecPAL target binds an official publisher URL or operator-supplied reviewed artifact, immutable version and digest, redistribution and execution terms, architecture and OS, runtime dependencies, install plan, executable identity, and rollback behavior, unsupported hosts are derived from the reviewed lock and cannot install, certify, or count as complete, real allow, deny, unknown, delegation, conflict, rule/scope mutation, replay, malformed, timeout, and disagreement cases execute through the selected external engine, the in-process Datalog/SecPAL family and any hermetic adapter remain separately named and cannot impersonate the vendor tool.
Preconditions: objective goal FVT-G217 is schedulable
Effects: satisfy evidence requirement: test/integration/toolchains/test_secpal_live_toolchain_contract.py
Evidence subset: test/integration/toolchains/test_secpal_live_toolchain_contract.py
Dependencies: FVT-055, FVT-073, FVT-087
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G217
Rejection reasons: none (accepted)

## Goal

Replace ambiguous SecPAL acquisition and adapter behavior with an official-artifact, license-aware, host-specific lazy installer and live semantic runner.

## Missing Evidence

- test/integration/toolchains/test_secpal_live_toolchain_contract.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
