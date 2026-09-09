# SAWM operator recovery work

This preserves the administrative task added after the sealed 45-task program. It is still open and is not counted as a canonical task or a completion.

## SAWM-045 Resolve dirty main checkout blocking 3 worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: 47172bf1cb5a9b1ed7c1449bb269656af02ab225
- Dedupe key: reconciliation_guardrail:main_checkout_dirty
- Depends on:
- Outputs: data/agent_supervisor/semantic_addressed_world_model/run-r2-m27/state/discovery, docs/architecture/semantic_addressed_world_model.todo.md
- Board namespace: semantic-addressed-world-model-v1
- Goal id: SAWM-G072
- Bundle: semantic-addressed-world-model/release
- Parallel lane: release
- Resource class: coordinator
- Validation: test -f /home/barberb/lift_coding/.worktrees/semantic-addressed-world-model-r2/data/agent_supervisor/semantic_addressed_world_model/run-r2-m27/state/discovery/2026-09-09-sawm-045-reconciliation-47172bf1cb5a.md
- Acceptance: Reconciliation guardrail filed this because 3 branch or worktree cleanup candidates are blocked by main_checkout_dirty. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/lift_coding/.worktrees/semantic-addressed-world-model-r2/data/agent_supervisor/semantic_addressed_world_model/run-r2-m27/state/discovery/2026-09-09-sawm-045-reconciliation-47172bf1cb5a.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.
