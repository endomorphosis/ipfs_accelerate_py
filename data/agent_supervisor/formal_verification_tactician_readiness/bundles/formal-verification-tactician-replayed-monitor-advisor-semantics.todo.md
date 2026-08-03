# Objective Bundle: formal-verification-tactician/replayed-monitor-advisor-semantics

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-098 Close objective gap: Certify replayed monitor and advisor semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: semantic-certification
- Depends on: FVT-056, FVT-072, FVT-085, FVT-091, FVT-094, FVT-095
- Outputs: tools/logic/certify_formal_verification_replayed_monitor_advisors.py, docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_replayed_monitor_advisor_semantics.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-098-objective-gap-b97274c38a81.md
- Bundle: formal-verification-tactician/replayed-monitor-advisor-semantics
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-replayed-monitor-advisor-semantics.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 6
- Parallel lane: formal-verification-tactician/replayed-monitor-advisor-semantics
- Conflict policy: Own the replay fan-in and advisor/monitor receipt; do not make core ErgoAI depend on Java, promote advice, or let a hermetic/parser fixture satisfy an external runtime lane.
- Predicted files: tools/logic/certify_formal_verification_replayed_monitor_advisors.py, docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- Changed paths:
- Context paths: tools/logic/certify_formal_verification_replayed_monitor_advisors.py, docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- AST symbols: docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- Interfaces: ReplayedMonitorAdvisorSemantics@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G230
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/bf779e2c874d2079801f4b0aa97eed1a9b0ed38ebe15173a2323e13dc1119c46
- Canonical task CID: baguqeerax53z4lehjuqhtaa7jmfks7xndknq5u4oxykroordepqt3qirtrda
- Semantic identity: objective-evidence-obligation/v1/f084e340a85536db6f210d699f14c8ac1a81d688276fd7b0e1458caf9c1f973c
- Acceptance subset: The independent Node/TypeScript Runtime MTL engine executes positive, negative, boundary, malformed, mutation, replay, timeout, and cross-runtime parity cases against the in-process monitor with disagreement quarantine, real ErgoAI and SymbolicAI execute positive/non-entailment/contradiction/mutation/replay/malformed/resource-bound advisory cases, exact package, lockfile, runtime, launcher, target, artifact, and executable identities are bound, Runtime MTL gains finite-trace authority only after parity, while advisors remain proposal-only until independent reconstruction, Stack and Temurin remain support-only and cannot satisfy public verification, semantic, or proof-authority requirements.
- Preconditions: objective goal FVT-G230 is schedulable
- Effects: satisfy evidence requirement: docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, satisfy evidence requirement: test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- Evidence subset: docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G230
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/f084e340a85536db6f210d699f14c8ac1a81d688276fd7b0e1458caf9c1f973c
- Missing evidence: docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- Embedding query: Re-execute the independent external Runtime MTL and genuine advisor lanes while preserving their finite-trace and advisory authority boundaries.
- AST query: docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py
- Surplus group: objective/FVT-G230
- Merge key: ffa06fe845dd29a1
- Merge family: objective/FVT-G230
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
- Todo vector key: f3b8cbb1e9cbcb47
- Acceptance: Objective scan filed this gap for FVT-G230. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-098-objective-gap-b97274c38a81.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json, test/integration/toolchains/test_replayed_monitor_advisor_semantics.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
