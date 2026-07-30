# Objective Bundle: formal-verification-tactician/supervisor

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-031 Close formal verification tactician readiness gap: Make proof-tactician supervisor execution restartable and fenced

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-integration
- Depends on: FVT-028, FVT-029
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py, test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Validation: python -m pytest test/api/test_goal_tactician_supervisor_lifecycle.py test/api/test_goal_tactician_supervisor_restart.py -q
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-30-fvt-031-objective-gap-9bdd9a1d85ba.md
- Bundle: formal-verification-tactician/supervisor
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-supervisor.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 30
- Parallel lane: formal-verification-tactician/supervisor
- Conflict policy: Own tactician lifecycle/restart integration and tests; reuse scheduler, proof-carrying planner, event store, leases, resources, cache, and completion authority rather than adding parallel persistence.
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py, test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Changed paths:
- AST symbols: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Interfaces: GoalTacticianSupervisorLifecycle@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G051
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/77873346df3a34d66cf70d809af5b27f55c11ab05feeda7c6762166d369e7b85
- Canonical task CID: baguqeerao6dtgrw7hi2nm3hxbwajv5nsp5k4cgvql7xnu7dhmilg2nu6pocq
- Semantic identity: objective-evidence-obligation/v1/70e1cb52c807581d28e50a4cd91293a917be38b413d0603de05d241fe38e5ead
- Acceptance subset: Restart replays identical authoritative state, stale workers/receipts cannot close or mutate a plan, cancellation/timeout/backpressure are durable, changed trees invalidate scoped work, completion requires all selected graph leaves and counterexamples to have adequate fresh receipts.
- Preconditions: objective goal FVT-G051 is schedulable
- Effects: satisfy evidence requirement: test/api/test_goal_tactician_supervisor_lifecycle.py, satisfy evidence requirement: test/api/test_goal_tactician_supervisor_restart.py
- Evidence subset: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Resource class: cpu-supervisor
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-supervisor
- Merge fate: objective/FVT-G051
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/70e1cb52c807581d28e50a4cd91293a917be38b413d0603de05d241fe38e5ead
- Missing evidence: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Embedding query: Persist end-goal, proof graph, candidate, verification, counterexample, closure, and completion transitions under content identities, leases, resource policy, retry bounds, exact cache keys, and restart-safe reconciliation.
- AST query: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
- Surplus group: objective/FVT-G051
- Merge key: 4895045f852454f4
- Merge family: objective/FVT-G051
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
- Todo vector key: 5ddfc9b2ee362b11
- Acceptance: Objective scan filed this gap for FVT-G051. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-30-fvt-031-objective-gap-9bdd9a1d85ba.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
