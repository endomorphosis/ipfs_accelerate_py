# Objective Bundle: formal-verification-tactician/kernel-live-semantics

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-057 Close objective gap: Execute and bind Lean, Rocq, and Isabelle kernel semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-040, FVT-045, FVT-049, FVT-064, FVT-062
- Outputs: tools/logic/certification/lean.py, tools/logic/certification/rocq.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Validation: python -m pytest test/integration/toolchains/test_kernel_live_semantic_fanin.py test/integration/toolchains/test_lean_semantic_certification.py test/integration/toolchains/test_rocq_toolchain_certification.py test/integration/toolchains/test_isabelle_toolchain_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-057-objective-gap-5bb2c7bb7675.md
- Bundle: formal-verification-tactician/kernel-live-semantics
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-kernel-live-semantics.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 3
- Parallel lane: formal-verification-tactician/kernel-live-semantics
- Conflict policy: Own kernel fan-in and live source checks; serialize expensive OPAM/Isabelle resources and preserve each kernel's separate authority.
- Predicted files: tools/logic/certification/lean.py, tools/logic/certification/rocq.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Changed paths:
- Context paths: tools/logic/certification/lean.py, tools/logic/certification/rocq.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- AST symbols: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Interfaces: KernelLiveSemanticFanIn@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G206
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/b69124c60821fd445e848a4fcbfa6f6705a3eccc1dfe390f72bb1b8304f11b61
- Canonical task CID: baguqeeraw2isjrqieh6uixuerjh4x6tpm4c2h3gmdx7dsd3sxmnygbhrdnqq
- Semantic identity: objective-evidence-obligation/v1/b7f547d501ed5d6e85ed2dc95e99a9d92ba5d0da2c06278621d3521ba2aa105e
- Acceptance subset: Lean, Rocq, and Isabelle independently execute a valid theorem, false theorem, hypothesis/conclusion mutation, deterministic replay, malformed source, timeout, and forbidden admit/axiom-oracle checks, Isabelle's live source/session helper is exercised rather than only offline fixtures, receipts bind exact kernel, dependency, source, imports/session, assumptions, theorem, and output digests, no advisor or sibling kernel substitutes for the selected kernel.
- Preconditions: objective goal FVT-G206 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_kernel_live_semantic_fanin.py, satisfy evidence requirement: docs/architecture/formal_verification_kernel_live_certificate.json
- Evidence subset: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Resource class: large-kernel-toolchain
- Token class: medium
- Estimated tokens: 0
- Resources: large-kernel-toolchain
- Merge fate: objective/FVT-G206
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/b7f547d501ed5d6e85ed2dc95e99a9d92ba5d0da2c06278621d3521ba2aa105e
- Missing evidence: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Embedding query: Require each installed proof kernel to check its own generated source and retain all assumptions, imports, theorem, and mutation evidence.
- AST query: test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json
- Surplus group: objective/FVT-G206
- Merge key: 1cd34ad5a8603afb
- Merge family: objective/FVT-G206
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
- Todo vector key: e2952b5dad554d48
- Acceptance: Objective scan filed this gap for FVT-G206. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-057-objective-gap-5bb2c7bb7675.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_kernel_live_semantic_fanin.py, docs/architecture/formal_verification_kernel_live_certificate.json), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
