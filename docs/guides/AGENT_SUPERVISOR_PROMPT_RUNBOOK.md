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
