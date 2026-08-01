# FVT-031 Objective Goal Gap

Date: 2026-07-30
Fingerprint: 9bdd9a1d85bade9d6078ac11f4e2f652e1f92ccb
Goal id: FVT-G051
Goal title: Make proof-tactician supervisor execution restartable and fenced
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: supervisor-integration
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 30
Bundle: formal-verification-tactician/supervisor
Parallel lane: formal-verification-tactician/supervisor
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Persist end-goal, proof graph, candidate, verification, counterexample, closure, and completion transitions under content identities, leases, resource policy, retry bounds, exact cache keys, and restart-safe reconciliation.
AST query: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
Conflict policy: Own tactician lifecycle/restart integration and tests; reuse scheduler, proof-carrying planner, event store, leases, resources, cache, and completion authority rather than adding parallel persistence.
Predicted files: ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py, test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
AST symbols: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
Interfaces: GoalTacticianSupervisorLifecycle@1
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/70e1cb52c807581d28e50a4cd91293a917be38b413d0603de05d241fe38e5ead
Acceptance subset: Restart replays identical authoritative state, stale workers/receipts cannot close or mutate a plan, cancellation/timeout/backpressure are durable, changed trees invalidate scoped work, completion requires all selected graph leaves and counterexamples to have adequate fresh receipts.
Preconditions: objective goal FVT-G051 is schedulable
Effects: satisfy evidence requirement: test/api/test_goal_tactician_supervisor_lifecycle.py, satisfy evidence requirement: test/api/test_goal_tactician_supervisor_restart.py
Evidence subset: test/api/test_goal_tactician_supervisor_lifecycle.py, test/api/test_goal_tactician_supervisor_restart.py
Dependencies: FVT-G044, FVT-G050
Resource class: cpu-supervisor
Token class: medium
Estimated tokens: 0
Resources: cpu-supervisor
Merge fate: objective/FVT-G051
Rejection reasons: none (accepted)

## Goal

Persist end-goal, proof graph, candidate, verification, counterexample, closure, and completion transitions under content identities, leases, resource policy, retry bounds, exact cache keys, and restart-safe reconciliation.

## Missing Evidence

- test/api/test_goal_tactician_supervisor_lifecycle.py
- test/api/test_goal_tactician_supervisor_restart.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
