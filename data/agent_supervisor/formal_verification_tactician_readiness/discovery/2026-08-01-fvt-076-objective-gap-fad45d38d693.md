# FVT-076 Objective Goal Gap

Date: 2026-08-01
Fingerprint: fad45d38d6933f91e15cd4151c273a2eec055d7b
Goal id: FVT-G204
Goal title: Execute real TLC and Apalache state-model semantics
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 8
Bundle: formal-verification-tactician/state-model-live-semantics
Parallel lane: formal-verification-tactician/state-model-live-semantics
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: path
Embedding query: Replace classifier-backed state-model promotion with real TLC and Apalache execution against positive and adversarial models.
AST query: test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
Conflict policy: Own live state-model cases and receipts; use the installed toolchain without downloading during certification and never promote from identity or output parsing alone.
Predicted files: tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
AST symbols: test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
Interfaces: StateModelLiveSemanticCertification@1
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/8036a0f059b12dfd7ce565285db4922255d53d199005d7da835e20bd2c7b47c3
Acceptance subset: The pinned TLC jar and Apalache executable each run a valid invariant model, a violating model with concrete counterexample, specification and invariant mutations, deterministic replay, malformed input, timeout, and bounded-state/resource cases, source model, property, bound, JVM, executable, jar/archive, and output digests are exact, canned text and parser classification remain `hermetic_parser` and cannot satisfy live external semantics.
Preconditions: objective goal FVT-G204 is schedulable
Effects: satisfy evidence requirement: objective validation repair
Evidence subset: objective validation repair
Dependencies: FVT-G120, FVT-G201, FVT-G202
Resource class: exclusive-jvm-toolchain
Token class: medium
Estimated tokens: 0
Resources: exclusive-jvm-toolchain
Merge fate: objective/FVT-G204
Rejection reasons: none (accepted)

## Goal

Replace classifier-backed state-model promotion with real TLC and Apalache execution against positive and adversarial models.

## Missing Evidence

- objective validation repair

## Present Evidence

- test/integration/toolchains/test_state_model_live_semantic_certification.py: test/integration/toolchains/test_state_model_live_semantic_certification.py (path)
- docs/architecture/formal_verification_state_model_live_certificate.json: docs/architecture/formal_verification_state_model_live_certificate.json (path)

## Suggested Handling

Run and repair the objective validation command until it passes, then record the evidence.
