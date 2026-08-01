# Objective Bundle: formal-verification-tactician/state-model-live-semantics

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-060 Close objective gap: Execute real TLC and Apalache state-model semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-042, FVT-064, FVT-062
- Outputs: tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_state_model_live_semantic_certification.py test/integration/toolchains/test_state_model_toolchain_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-060-objective-gap-7220519566f7.md
- Bundle: formal-verification-tactician/state-model-live-semantics
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-state-model-live-semantics.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 6
- Parallel lane: formal-verification-tactician/state-model-live-semantics
- Conflict policy: Own live state-model cases and receipts; use the installed toolchain without downloading during certification and never promote from identity or output parsing alone.
- Predicted files: tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Changed paths:
- Context paths: tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- AST symbols: test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Interfaces: StateModelLiveSemanticCertification@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G204
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/df6514a9340af6f60abc44216fd031d4af7522e8387097f321b669a118737034
- Canonical task CID: baguqeera35srjkjubl3pmcv4iqqw7ubr2sxxkixihbyjp4zbwzu2cgdtoa2a
- Semantic identity: objective-evidence-obligation/v1/97a9abc783cec75eae0110ad32949db2ef582ec14938f20477c107d5e5271e7b
- Acceptance subset: The pinned TLC jar and Apalache executable each run a valid invariant model, a violating model with concrete counterexample, specification and invariant mutations, deterministic replay, malformed input, timeout, and bounded-state/resource cases, source model, property, bound, JVM, executable, jar/archive, and output digests are exact, canned text and parser classification remain `hermetic_parser` and cannot satisfy live external semantics.
- Preconditions: objective goal FVT-G204 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_state_model_live_semantic_certification.py, satisfy evidence requirement: docs/architecture/formal_verification_state_model_live_certificate.json
- Evidence subset: test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Resource class: exclusive-jvm-toolchain
- Token class: medium
- Estimated tokens: 0
- Resources: exclusive-jvm-toolchain
- Merge fate: objective/FVT-G204
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/97a9abc783cec75eae0110ad32949db2ef582ec14938f20477c107d5e5271e7b
- Missing evidence: test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Embedding query: Replace classifier-backed state-model promotion with real TLC and Apalache execution against positive and adversarial models.
- AST query: test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json
- Surplus group: objective/FVT-G204
- Merge key: 6dcb5b22caf970cf
- Merge family: objective/FVT-G204
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
- Todo vector key: 42018eda5021c9b8
- Acceptance: Objective scan filed this gap for FVT-G204. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-060-objective-gap-7220519566f7.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_state_model_live_semantic_certification.py, docs/architecture/formal_verification_state_model_live_certificate.json), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
