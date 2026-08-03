# FVT-065 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 0bbb37a543adc021294eba1aa4653dd6204d3f57
Goal id: FVT-G203
Goal title: Aggregate full specialized receipts with composite lane handlers
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: certification-integrity
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 11
Bundle: formal-verification-tactician/semantic-receipt-aggregation
Parallel lane: formal-verification-tactician/semantic-receipt-aggregation
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace first-check and one-handler-per-lane fan-in with lossless, per-tool specialized evidence aggregation.
AST query: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
Conflict policy: Own role registration and lossless aggregation; do not run installers, collapse by check kind, discard raw receipt identity, or let one tool overwrite a sibling handler.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
AST symbols: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
Interfaces: FormalVerificationSpecializedReceiptAggregation@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/8d7d04f775a3f8b4e0fdc64c6b9480ee02eb0cad857628eed078ed8b604dba24
Acceptance subset: Handlers are keyed by `(lane_id, tool_id)` or a composite lane returns distinct per-tool receipts, kernel retains Lean, Rocq, and Isabelle evidence and protocol retains Tamarin and ProVerif evidence, state, protocol, kernel, ATP, hyperproperty, advisor, in-process and external authorization, in-process and external Runtime MTL, and ZKP certifiers are all represented, every check, case, binding, executable, artifact, dependency, source, authority ceiling, and raw receipt digest participates in the top-level digest, a second failed check of an already-present kind blocks promotion, mutating any retained check or identity changes the certificate digest.
Preconditions: objective goal FVT-G203 is schedulable
Effects: satisfy evidence requirement: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
Evidence subset: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
Dependencies: FVT-G101, FVT-G102, FVT-G103, FVT-G120, FVT-G130, FVT-G131, FVT-G140, FVT-G150, FVT-G151, FVT-G160, FVT-G170, FVT-G180, FVT-G181, FVT-G190, FVT-G201, FVT-G202
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G203
Rejection reasons: none (accepted)

## Goal

Replace first-check and one-handler-per-lane fan-in with lossless, per-tool specialized evidence aggregation.

## Missing Evidence

- test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
