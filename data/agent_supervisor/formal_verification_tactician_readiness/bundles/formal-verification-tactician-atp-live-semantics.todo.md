# Objective Bundle: formal-verification-tactician/atp-live-semantics

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-054 Close objective gap: Execute real Vampire and E prover semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-048, FVT-064, FVT-062
- Outputs: tools/logic/certification/atp.py, test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Validation: python -m pytest test/integration/toolchains/test_atp_live_semantic_certification.py test/integration/toolchains/test_atp_toolchain_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-054-objective-gap-964e71995499.md
- Bundle: formal-verification-tactician/atp-live-semantics
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-atp-live-semantics.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/atp-live-semantics
- Conflict policy: Own real ATP execution and receipts; keep SZS parsing as adapter evidence and never grant kernel authority to an unreconstructed ATP result.
- Predicted files: tools/logic/certification/atp.py, test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Changed paths:
- Context paths: tools/logic/certification/atp.py, test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- AST symbols: test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Interfaces: ATPLiveSemanticCertification@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G207
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/9f0a1ef914661e612ae5b4587ea05eb4607729b623e8f8f109634d2f1cb0d7fe
- Canonical task CID: baguqeerat4fb56iumypgckxfwrmh5ic6wrqhoknwepupr4ijmngs6hfq277a
- Semantic identity: objective-evidence-obligation/v1/a533d95ef148685a75ed6eb1a8d99a66ed1db1c96da90447048e89ed8bfd0af4
- Acceptance subset: Vampire and E each execute theorem and counter-satisfiable problems, premise/conclusion mutations, replay, malformed TPTP, timeout/resource bounds, disagreement, and proof-object/reconstruction cases, receipts bind exact binary and artifact digests, TPTP source, assumptions, conclusion, limits, raw SZS output, and reconstruction status, an ATP result cannot exceed candidate/reconstruction authority until checked by a trusted kernel.
- Preconditions: objective goal FVT-G207 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_atp_live_semantic_certification.py, satisfy evidence requirement: docs/architecture/formal_verification_atp_live_certificate.json
- Evidence subset: test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G207
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/a533d95ef148685a75ed6eb1a8d99a66ed1db1c96da90447048e89ed8bfd0af4
- Missing evidence: test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Embedding query: Replace SZS parser fixtures with real pinned ATP runs while preserving reconstruction and kernel-checking ceilings.
- AST query: test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json
- Surplus group: objective/FVT-G207
- Merge key: e4888ed3c395be93
- Merge family: objective/FVT-G207
- Merge role: aggregate
- Work item count: 2
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: 4c32f8d9f14956de
- Acceptance: Objective scan filed this gap for FVT-G207. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-054-objective-gap-964e71995499.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_atp_live_semantic_certification.py, docs/architecture/formal_verification_atp_live_certificate.json), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
