# FVT-064 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 64fd42d3cc7fe1e7f92a2614c6e63a400d4ded23
Goal id: FVT-G201
Goal title: Derive exact host support and platform exceptions from the lock
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: deployment-integrity
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 10
Bundle: formal-verification-tactician/platform-support-classifier
Parallel lane: formal-verification-tactician/platform-support-classifier
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Give every locked tool an auditable host-platform classification so missing supported capabilities can never be relabeled as exceptions.
AST query: test/integration/toolchains/test_formal_verification_platform_support.py
Conflict policy: Own platform normalization and classification only; never probe or install tools, infer support from PATH, or convert unavailability into unsupported status.
Predicted files: tools/logic/certification/platform_support.py, test/integration/toolchains/test_formal_verification_platform_support.py
AST symbols: test/integration/toolchains/test_formal_verification_platform_support.py
Interfaces: FormalVerificationPlatformSupport@1
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/7140fe479d3e8b202596942b2806965d383aa7a0ce9d22a50f075a7a3457dd10
Acceptance subset: The normalized host key is derived from the running OS and architecture, each tool reports supported_here, unsupported_here, or ambiguous from its own pins and deployment contract, `any` support is honored, absent, contradictory, or ambiguous metadata is a blocker, only an explicit host exclusion can produce a narrow platform exception, linux-aarch64 classifies HyperLTL, AutoHyper, MCHyper, Souffle, and external Runtime MTL as supported under the current lock, external SecPAL as unsupported, and ZKP as a platform-independent deployment binding, a lock mutation that adds or removes linux-aarch64 changes the classification and final digest.
Preconditions: objective goal FVT-G201 is schedulable
Effects: satisfy evidence requirement: test/integration/toolchains/test_formal_verification_platform_support.py
Evidence subset: test/integration/toolchains/test_formal_verification_platform_support.py
Dependencies: FVT-G010, FVT-G110
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G201
Rejection reasons: none (accepted)

## Goal

Give every locked tool an auditable host-platform classification so missing supported capabilities can never be relabeled as exceptions.

## Missing Evidence

- test/integration/toolchains/test_formal_verification_platform_support.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
