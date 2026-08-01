# FVT-051 Objective Goal Gap

Date: 2026-07-31
Fingerprint: bbcce6906bf194a48d06637a4989830d73f9b040
Goal id: FVT-G180
Goal title: Install external Datalog and SecPAL differential shadows
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P1
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 50
Bundle: formal-verification-tactician/authorization-toolchains
Parallel lane: formal-verification-tactician/authorization-toolchains
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace the external authorization gap with pinned Souffle/SecPAL-compatible shadows and differential disagreement handling.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
Conflict policy: Own external authorization installer plugins, differential handler, and test; do not weaken or edit the in-process reference semantics.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
Interfaces: ExternalAuthorizationShadowCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/3c47e0b2ab6c844f27a22cea94a7e56befc70d484eea0286ccd169c257c7634a
Acceptance subset: Explicit strict installation selects exact external engines, the allow/deny/unknown/conflict/delegation corpus, rule/scope mutation, replay, malformed output, timeout, and differential comparison pass, any disagreement quarantines promotion, external engines remain shadows while the certified in-process references retain authorization authority.
Preconditions: objective goal FVT-G180 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, satisfy evidence requirement: test/integration/toolchains/test_external_authorization_toolchain_certification.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, test/integration/toolchains/test_external_authorization_toolchain_certification.py
Dependencies: FVT-G102, FVT-G110
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G180
Rejection reasons: none (accepted)

## Goal

Replace the external authorization gap with pinned Souffle/SecPAL-compatible shadows and differential disagreement handling.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py
- test/integration/toolchains/test_external_authorization_toolchain_certification.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
