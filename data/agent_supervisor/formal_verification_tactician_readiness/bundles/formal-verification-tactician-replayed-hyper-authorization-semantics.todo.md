# Objective Bundle: formal-verification-tactician/replayed-hyper-authorization-semantics

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-097 Close objective gap: Certify replayed hyperproperty and external authorization semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: semantic-certification
- Depends on: FVT-061, FVT-077, FVT-055, FVT-073, FVT-094
- Outputs: tools/logic/certify_formal_verification_replayed_hyper_authorization.py, docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_replayed_hyper_authorization_semantics.py test/integration/toolchains/test_external_authorization_vendor_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-097-objective-gap-fe069251c732.md
- Bundle: formal-verification-tactician/replayed-hyper-authorization-semantics
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-replayed-hyper-authorization-semantics.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 5
- Parallel lane: formal-verification-tactician/replayed-hyper-authorization-semantics
- Conflict policy: Own the replay fan-in for hyperproperties and Souffle; do not edit legacy SecPAL artifact intake or elevate external shadows to authorization authority.
- Predicted files: tools/logic/certify_formal_verification_replayed_hyper_authorization.py, docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- Changed paths:
- Context paths: tools/logic/certify_formal_verification_replayed_hyper_authorization.py, docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- AST symbols: docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- Interfaces: ReplayedHyperAuthorizationSemantics@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G229
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/a1793131aa674bee83ac4246a1467aa4732e7040462cf47e739ee4d32f05bc77
- Canonical task CID: baguqeerauf4tcmnkm5f65a5mijdkcrt2urzs44caiywpi7ttt3snglyfxr3q
- Semantic identity: objective-evidence-obligation/v1/1369f1e51f78911144fd61650bcdbfa20ec265e770c47b9520c576db1801752d
- Acceptance subset: HyperLTL, AutoHyper, and MCHyper execute bounded satisfiable/violating information-flow hyperproperties with trace-pair witnesses, mutation, deterministic replay, malformed, timeout, and cross-engine disagreement handling, Souffle executes allow, deny, unknown, conflict, delegation, rule/scope mutation, replay, and disagreement cases through the exact managed vendor binary, each receipt binds executable, runtime, source/artifact, host, policy/formula, bounds, parser decisions, and output digests, hyperproperty authority remains bounded and Souffle remains an external authorization shadow, Microsoft SecPAL compatibility evidence is not interchangeable.
- Preconditions: objective goal FVT-G229 is schedulable
- Effects: satisfy evidence requirement: docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, satisfy evidence requirement: test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- Evidence subset: docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G229
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/1369f1e51f78911144fd61650bcdbfa20ec265e770c47b9520c576db1801752d
- Missing evidence: docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- Embedding query: Close the genuine managed semantic and freshness bindings for HyperLTL, AutoHyper, MCHyper, and Souffle without importing legacy SecPAL vendor authority.
- AST query: docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py
- Surplus group: objective/FVT-G229
- Merge key: a04839468cece86d
- Merge family: objective/FVT-G229
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
- Todo vector key: c6b645576e47d293
- Acceptance: Objective scan filed this gap for FVT-G229. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-097-objective-gap-fe069251c732.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json, test/integration/toolchains/test_replayed_hyper_authorization_semantics.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
