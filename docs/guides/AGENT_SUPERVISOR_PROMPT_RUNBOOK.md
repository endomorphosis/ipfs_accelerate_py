# Agent Supervisor Prompt-Only Runbook (v3)

## Monitor and recovery (ASE3-008)

### Policy
- Heartbeat interval: **5s**
- Stale heartbeat: **30s**
- Semantic progress budget: **300s** (configured-scheduler phase/cursor only; log noise does not count)
- Canary recoveries: **at most 3 per 30 minutes**

### RUNNING evidence
`RUNNING` requires one same-revision join of:
- lifecycle process birth, lease, fence, fresh heartbeat, event cursor
- monitor process birth, lease, fence, fresh heartbeat, event cursor
- tree reachability

Missing, stale, cross-generation, synthetic, or self-attested components deny `RUNNING`.

### Guardian ownership
Only the **ReviewedHostNamespaceReconciler** may start or adopt the durable monitor. CLI, MCP, Python clients, and the monitor itself cannot act as guardian.

### Client disconnect
Disconnecting a client session does **not** stop the durable monitor.

### Recovery
Detection never grants restart authority. An authorized recovery callback is required. Unknown external-effect outcomes are adopted as operator action and never blindly replayed.

### Terminal shutdown
After the run is terminal, shutdown stops only the exact owned monitor generation.

## Protected runtime activation (ASE3-026)

ASE3-026 is a two-evidence operator gate. Refill and the detached monitor stay
dormant until both receipts validate.

### Pre-effect authorization
Schema: `ipfs_accelerate_py.agent_supervisor.protected-runtime-activation-authorization@1`

- Binds inactive exact tree, old generation, target old+1 CAS/lease, guardian,
  bounded flags, quiescence, and expiry.
- **Must** set `authorization_effect_observed: false`.
- Must never claim birth, heartbeat, cursor, refill, reload, or completion.
- Authorization alone never proves the effect ran and cannot select public facades.

### Generation CAS
Exactly one matching old+1 CAS/lease winner may consume a validated
authorization through the `ReviewedHostNamespaceReconciler`. Retries adopt that
winner; a non-identical winner fails closed.

### Post-activation observation
Schema: `ipfs_accelerate_py.agent_supervisor.protected-runtime-post-activation-observation@1`

- Separately joins actual same-generation lifecycle and monitor births, leases,
  fences, heartbeats, and cursors plus refill DISPATCHED/ADOPTED.
- Observation cannot retroactively authorize activation.
- Both receipts must land together under strict convergence binding; ASE3-026
  remains **blocked** until an operator rehashes the completed contract freezes.

### Flags after activation (intended)
- `prompt_program_refill_enabled` / `objective_refill_enabled`: true
- `monitor_policy.enabled`: true
- `codebase_refill_enabled`: remains false
- Legacy hash sharding for active slices: remains false
