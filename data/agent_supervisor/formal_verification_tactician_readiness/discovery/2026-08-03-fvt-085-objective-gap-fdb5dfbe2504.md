# FVT-085 Objective Goal Gap

Date: 2026-08-03
Fingerprint: fdb5dfbe2504a19b921518c7b89ad8ae42820ca6
Goal id: FVT-G218
Goal title: Implement the genuine ErgoAI advisor-toolchain path
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 1
Bundle: formal-verification-tactician/ergoai-live-toolchain
Parallel lane: formal-verification-tactician/ergoai-live-toolchain
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace ErgoAI wrapper and proposal-only assumptions with a locked official distribution, dependency-complete lazy installer, and bounded live semantic adapter while preserving advisor authority ceilings.
AST query: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
Conflict policy: Own ErgoAI provenance, dependencies, lazy installer, wrapper, and bounded semantics; never scrape an unauthoritative artifact, download during certification, treat wrapper fixtures as live execution, or elevate an advisor verdict to theorem authority.
Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_ergoai_live_toolchain_contract.py
AST symbols: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
Interfaces: ErgoAILiveToolchainContract@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/8bea67970a56a02025411e374670d7c59ebb920ea7df71b904344f9089f1abc4
Acceptance subset: The lock binds the official ErgoAI distribution or reviewed source revision, license and acquisition conditions, archive/source digests, XSB and every runtime/build dependency, supported OS/architecture matrix, entry point, and exact identity probe, explicit lazy installation is staged, checksum-verified, atomic, relocatable, and offline after acquisition, live entailment, non-entailment, contradiction, rule/query mutation, deterministic replay, malformed input, timeout, and resource-bound cases execute through ErgoAI, results remain proposal or candidate evidence until reconstructed or checked by an independent proof authority.
Preconditions: objective goal FVT-G218 is schedulable
Effects: satisfy evidence requirement: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
Evidence subset: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
Dependencies: FVT-064, FVT-087
Resource class: cpu-proof-solver
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-solver
Merge fate: objective/FVT-G218
Rejection reasons: none (accepted)

## Goal

Replace ErgoAI wrapper and proposal-only assumptions with a locked official distribution, dependency-complete lazy installer, and bounded live semantic adapter while preserving advisor authority ceilings.

## Missing Evidence

- test/integration/toolchains/test_ergoai_live_toolchain_contract.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
