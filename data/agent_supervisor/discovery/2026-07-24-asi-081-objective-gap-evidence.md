# ASI-081 Objective Gap Evidence Packet

Date: 2026-07-24
Source fingerprint: `c28d97df1330f03cb8be6cc7e3fa8abe2f71b3bc`
Goal packet: `goal_packet/self_refill/ipfs_accelerate_py/9d87d026b79d`
Packet goals: ASI-G109, ASI-G110, ASI-G111
Status: implemented; canonical completion remains subject to fresh validation and the objective completion gate

## Resolution

ASI-081 closes the implementation-evidence gap as one self-refill lifecycle:

1. A drained, identity-bound epoch evaluates a complete benchmark population.
2. Blocker-free actionable observations may propose successor goals.
3. Quality and finite-refinement policy bounds confidence, novelty, kind, count, depth, breadth, open work, retries, and token cost.
4. Admitted goals commit through the durable objective materialization transaction with exact heap and repository-tree fencing.
5. Every committed goal is force-prioritized into exactly one supervisor task before unrelated objective gaps can consume the bounded refill capacity.
6. An exact post-refill replay returns the original receipt before benchmark or proposal callbacks and emits a zero-work replay witness.
7. A healthy no-gap epoch continues to emit the existing exhaustion/quorum witness and wait for a declared meaningful trigger.

No child-goal refinement is required. ASI-G109, ASI-G110, and ASI-G111 are already the smallest stable leaf owners under ASI-G080, and `resolve_objective_evidence_projection` now distinguishes a leaf owner from its aggregate ancestor while still rejecting incomparable or sibling ambiguity.

## Requirement Evidence Map

### `020061024173618462922348580596364003627` — ASI-G109

- Producer: `SUCCESSOR_REFILL_REQUIREMENT_ID`, `materialize_self_improvement_successors`, and `SuccessorRefillEvidence` in `ipfs_accelerate_py/agent_supervisor/self_improvement.py`.
- Admission: `preview_objective_goal_materialization` applies explicit finite limits and rejects invalid or low-quality work before a write.
- Transaction: `commit_objective_goal_materialization` binds the exact heap and stable source tree.
- Backlog alignment: `record_objective_backlog_findings(..., force_goal_ids=...)` plus forced-goal scan/ranking priority creates one task for each committed goal.
- Tests: `test_actionable_epoch_creates_bounded_successor_and_exact_replay_is_noop` and `test_successor_policy_bounds_batch_and_foreign_actionable_fails_closed`.

The typed witness binds the ASI-G109 projection, epoch and policy, observation receipts, actionable dimensions, candidate and admitted proposal identities, created goal and task identities, transaction, artifact transitions, and content identity. Proposal text, discovery prose, or an objective-node append without its matching backlog task is not completion evidence.

### `065313778069923158401871898168782520190` — ASI-G110

- Producer: `EPOCH_IDEMPOTENCY_REQUIREMENT_ID`, `run_self_improvement_epoch`, and `EpochReplayEvidence`.
- Replay boundary: strict ledger restoration and exact current artifact identities are checked before provider invocation.
- Post-successor identity: the stable external epoch binding is paired with the successor receipt's exact post-commit objective and task-board identities.
- Zero-work proof: provider, proposal, materialization, and task-board-write counts are sealed as zero.
- Tests: the actionable end-to-end test proves one provider call and one proposal call across the original invocation and exact replay, byte-identical post-state artifacts, strict witness restoration, and canonical completion-evidence projection.

A meaningful external tree, policy, capability, observation-window, operator, objective, or task-board state change does not qualify for the replay shortcut.

### `119294002389522221490347364495731444366` — ASI-G111

- Producer: `HEALTHY_EXHAUSTION_REQUIREMENT_ID` and `HealthyExhaustionEvidence`.
- Existing implementation: complete fresh healthy observations from independent same-binding channels, explicit exhaustion quorum, unchanged artifacts, zero work counters, durable meaningful-trigger wait state, exact replay, and strict restoration.
- Packet integration: the objective projection resolves ASI-G111 as the unique leaf beneath ASI-G080 even though the aggregate parent repeats the packet requirement.
- Tests: the healthy-exhaustion, replay, mutation, meaningful-trigger, nonqualifying-input, and restoration cases in `test/api/test_agent_supervisor_self_improvement_refill.py`.

ASI-081 preserves the stronger ASI-052 no-busywork proof and records it as the third member of the shared packet; it does not manufacture another G111 goal or task.

## Backlog and Stale Mapping Reconciliation

- ASI-038's stale aggregate mapping is ASI-G109.
- ASI-039's stale aggregate mapping is ASI-G110.
- ASI-052's stale aggregate mapping is ASI-G111.
- The ASI-G080 aggregate is an ancestor carrier, not a competing evidence owner.
- Sibling owners remain an error and cannot be selected implicitly.
- Forced newly materialized goals retain priority after semantic/packet ranking, so a bounded scan cannot generate an unrelated older task instead.
- Drained-scan task-count state is advanced after both generated and genuinely exhausted terminal scans, keeping supervisor refill pressure aligned with the objective heap.

## Validation

Primary command:

```text
python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
```

This discovery artifact documents the producer and test map. It is a nomination/provenance record, not standalone completion authority; fresh current-tree validation, typed evidence validation, analyzer health, descendant coverage, and the canonical objective completion transition remain required.
