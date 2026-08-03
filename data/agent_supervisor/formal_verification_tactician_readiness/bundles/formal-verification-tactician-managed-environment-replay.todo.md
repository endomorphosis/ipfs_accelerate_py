# Objective Bundle: formal-verification-tactician/managed-environment-replay

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-094 Close objective gap: Replay managed dependency, capability, platform, and freshness bindings

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: dependency-integrity
- Depends on: FVT-064, FVT-062, FVT-084, FVT-087, FVT-090, FVT-091, FVT-088
- Outputs: tools/logic/certify_formal_verification_managed_environment.py, docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_managed_environment_replay.py test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-094-objective-gap-323f4010a6ee.md
- Bundle: formal-verification-tactician/managed-environment-replay
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-environment-replay.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 2
- Parallel lane: formal-verification-tactician/managed-environment-replay
- Conflict policy: Own the unified managed-environment replay tool and receipt; consume existing installers without weakening their explicit opt-in or offline boundaries, and never treat installation as semantic certification.
- Predicted files: tools/logic/certify_formal_verification_managed_environment.py, docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py
- Changed paths:
- Context paths: tools/logic/certify_formal_verification_managed_environment.py, docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py
- AST symbols: docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py
- Interfaces: FormalVerificationManagedEnvironmentReplay@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G226
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/762286815bcfcd54ea4242798d90f95fa132ccb8fca611119c2c29d194b214c1
- Canonical task CID: baguqeeraoyrinak3z7gvj2scij4y3ehzl6qtftfy7stbcem4fqu5dffsctaq
- Semantic identity: objective-evidence-obligation/v1/a3d34a81a930ea9b8f900664f452acfa20ed3f0ee873e7a64f93cc523875d04b
- Acceptance subset: A separately invoked, explicitly authorized acquisition phase uses only reviewed immutable URLs, versions, sizes, checksums, signatures or equivalent publisher evidence, licenses, and OS/architecture pins, publication is user-local, single-flight, symlink-safe, atomic, and rollback-preserving, certification then runs with network, download, installation, ambient PATH, user-site, source-tree, and system-package mutation disabled, every currently supported Apalache, AutoHyper, Rocq/Coq, E, HyperLTL, Isabelle, MCHyper, ProVerif, Souffle, Tamarin, TLC, Vampire, ErgoAI, external Runtime MTL, and required Maude/OPAM/Stack/Temurin dependency binds exact executable, artifact, runtime, platform, and freshness identities, missing, partial, stale, relocated-without-rebinding, wrong-architecture, byte-mutated, or dependency-mutated trees fail only their owned axes and cannot be repaired by stale receipts, support dependencies remain non-semantic and non-authoritative.
- Preconditions: objective goal FVT-G226 is schedulable
- Effects: satisfy evidence requirement: docs/architecture/formal_verification_managed_environment_replay_receipt.json, satisfy evidence requirement: test/integration/toolchains/test_managed_environment_replay.py
- Evidence subset: docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py
- Resource class: io-artifact
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: io-artifact
- Merge fate: objective/FVT-G226
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/a3d34a81a930ea9b8f900664f452acfa20ed3f0ee873e7a64f93cc523875d04b
- Missing evidence: docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py
- Embedding query: Materialize one approved managed prover environment through explicit opt-in installation, then replay every supported external dependency, capability, platform, and freshness binding offline from its immutable root.
- AST query: docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py
- Surplus group: objective/FVT-G226
- Merge key: 38c151a37455ec85
- Merge family: objective/FVT-G226
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
- Todo vector key: 2a1faa6808a26dd8
- Acceptance: Objective scan filed this gap for FVT-G226. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-094-objective-gap-323f4010a6ee.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (docs/architecture/formal_verification_managed_environment_replay_receipt.json, test/integration/toolchains/test_managed_environment_replay.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
