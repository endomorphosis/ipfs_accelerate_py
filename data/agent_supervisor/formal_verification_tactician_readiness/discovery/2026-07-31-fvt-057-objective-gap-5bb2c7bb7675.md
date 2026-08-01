# FVT-057 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 5bb2c7bb767533e47025e40c0384f53445760736
Goal id: FVT-G206
Goal title: Execute and bind Lean, Rocq, and Isabelle kernel semantics
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 3
Bundle: formal-verification-tactician/kernel-live-semantics
Parallel lane: formal-verification-tactician/kernel-live-semantics
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Require each installed proof kernel to check its own generated source and retain all assumptions, imports, theorem, and mutation evidence.
AST query: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
Conflict policy: Own kernel fan-in and live source checks; serialize expensive OPAM/Isabelle resources and preserve each kernel's separate authority.
Predicted files: tools/logic/certification/lean.py, tools/logic/certification/rocq.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
AST symbols: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
Interfaces: KernelLiveSemanticFanIn@1
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/b7f547d501ed5d6e85ed2dc95e99a9d92ba5d0da2c06278621d3521ba2aa105e
Acceptance subset: Lean, Rocq, and Isabelle independently execute a valid theorem, false theorem, hypothesis/conclusion mutation, deterministic replay, malformed source, timeout, and forbidden admit/axiom-oracle checks, Isabelle's live source/session helper is exercised rather than only offline fixtures, receipts bind exact kernel, dependency, source, imports/session, assumptions, theorem, and output digests, no advisor or sibling kernel substitutes for the selected kernel.
Preconditions: objective goal FVT-G206 is schedulable
Effects: satisfy evidence requirement: test/integration/toolchains/test_kernel_live_semantic_fanin.py, satisfy evidence requirement: docs/architecture/formal_verification_kernel_live_certificate.json
Evidence subset: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
Dependencies: FVT-G101, FVT-G150, FVT-G151, FVT-G201, FVT-G202
Resource class: large-kernel-toolchain
Token class: medium
Estimated tokens: 0
Resources: large-kernel-toolchain
Merge fate: objective/FVT-G206
Rejection reasons: none (accepted)

## Goal

Require each installed proof kernel to check its own generated source and retain all assumptions, imports, theorem, and mutation evidence.

## Missing Evidence

- test/integration/toolchains/test_kernel_live_semantic_fanin.py
- docs/architecture/formal_verification_kernel_live_certificate.json

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
