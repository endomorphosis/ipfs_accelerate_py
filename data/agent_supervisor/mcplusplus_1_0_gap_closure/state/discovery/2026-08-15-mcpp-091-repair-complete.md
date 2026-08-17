# MCPP-091 Repair Completion: MCPP-013 validation retry-budget

Date: 2026-08-15
Source task: MCPP-013
Follow-up task: MCPP-091
Status: **completed**
Attempt: 1

## Root cause (inherited provider failure, not a task-owned regression)

MCPP-013 never reached implementation or validation. All three attempts failed
during supervisor worktree / provider setup:

- Failed phase: `worktree_setup`
- Failure kind: `validation_failure` with reason `provider_exception`
- Exception: `RuntimeError: quota-only route requires a ready Grok primary`
- Validation attempted: **False** (no declared gate was executed)
- Changed files on source attempts: **none**

Evidence:

- Discovery finding: `data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-15-mcpp-091-mcpp-013-retry-budget.md`
- Diagnostic receipt: `state/lane-1/implementation_logs/mcpp-013-diagnostic-receipt.json`
- Retry capsules (attempts 2–3): same exception summary; validation not attempted

This is **not** a regression in production policy, validators, or tests. The
declared MCPP-013 gate (`test -s` on the ADR path) was never run because the
provider route failed before implementation. Conflict policy for MCPP-013
remains: new ADR file only; do not edit the sealed plan identity.

## Repair actions

1. Deliver the missing MCPP-013 declared output under the nested Mcp-Plus-Plus
   tree:
   - `ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0001-spec-runtime-ownership.md`
   - Status: **Accepted**
   - Records sealed plan KD-1 and §2 Repository ownership
   - Acceptance content: Mcp-Plus-Plus owns schemas, vectors, validators,
     matrices, interoperability tests, and release bundles; runtimes own
     adapters only
   - Interface checklist: **`SpecRuntimeOwnership@1`**
2. Preserve correct production policy: no protected plan/todo/config/validator
   paths modified; no assertion weakening.
3. Record this repair completion under the declared discovery directory so the
   supervisor can release MCPP-013 from strategy `blocked_tasks`.

## Declared gate proof

MCPP-013 validation command:

```bash
test -s ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0001-spec-runtime-ownership.md
```

Result: **pass** (file present and non-empty).

MCPP-091 acceptance path presence:

```bash
test -f data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-15-mcpp-091-mcpp-013-retry-budget.md
```

Result: **pass** (finding file present).

Sibling ADR set after repair:

```text
0001-spec-runtime-ownership.md  (new; this repair)
0002-crypto-canonical.md
0003-conformance-levels.md
0004-state-modes.md
0005-durable-executor.md
0006-bindings-a2a.md
```

## Supervisor release note

Completing MCPP-091 releases MCPP-013 from strategy `blocked_tasks` and resets
its validation retry budget. The MCPP-013 ADR for spec versus runtime ownership
is delivered as this repair’s primary declared output; discovery evidence under
`data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery` records the
failure diagnosis and completion.
