# ADR 0007: Agent-supervisor efficiency and state authorities

- Status: Accepted for the ASEH bootstrap inventory
- Date: 2026-08-24
- Decision scope: `agent-supervisor-efficiency-and-state-hardening-v1` at the inventory source tree recorded in `authority_inventory.json`

## Context

ASEH consolidates existing authority paths. It does not create another supervisor, task database, router, ContextPack builder, or recovery controller. The current source contains launch, state, provider, validation, merge, receipt, and recovery helpers that must be explicitly bounded before candidate work proceeds.

The accompanying inventory is the closed, machine-readable classification of the bootstrap handoff and protected authority boundary. It is source evidence, not a live-state receipt or completion claim.

## Decision

### Canonical supervisor handoff

The only canonical live handoff is:

```text
immutable reviewed objectives + task board
  -> one offline materialization
  -> IntentRepository@1 / DatabaseTaskSource@1 in DuckDB
  -> exclusive loopback QuackStateServer@1 typed owner
  -> configured_board_scheduler.py sealed execution plan
  -> existing multi_supervisor_runner.py
  -> existing implementation supervisor and worktree-bound implementation daemon
  -> admitted validators + current-tree merge receipts
  -> one terminalization and an independent promotion/non-promotion decision
```

`scripts/run_agent_supervisor_efficiency_state_hardening.py` is an operator adapter for this chain, not a new supervisor. Scheduler capacity snapshots, process health, raw logs, Markdown status, and raw test exits are observations. None can independently claim task completion, merge authority, or promotion authority.

### State-machine authority

`TypedStateOwnerGateway@1`, hosted by the exclusive `QuackStateServer@1`, is the sole production writer. `DatabaseTaskSource@1` and `IntentRepository@1` are the canonical facade and transactional substrate. Every mutation is closed-command, revision/CAS, lease/fence, and idempotency governed; a successful terminal state requires the required receipt.

Direct multi-process DuckDB file writes, raw SQL authority expansion, stale-fence completion, and automatic embedded/file fallback are forbidden. DuckLake is append-only and rebuildable analytics/history only; it is never an authority for claims, readiness, leases, completion, recovery, terminalization, or promotion.

### Routing authority

`ipfs_accelerate_py` owns executor selection and provider/model routing. The authoritative target is the requirements-defined deterministic-first ladder: exact current receipt, static/dependency analysis, schema/type/static checks, selected tests, incremental prover, small specialist, medium model, frontier model, then human decision. A model invocation requires a decision-relevant typed unresolved question. Capacity projections and provider availability cannot skip an earlier stage or become dispatch authority.

Existing daemon/provider routes and merge fallbacks are transitional implementation surfaces. They must preserve the canonical handoff, emit their bounded receipts, and may not self-authorize policy promotion.

### ContextPack and cross-repository authority

| Repository | Sole authority | Explicit limit |
|---|---|---|
| `ipfs_datasets_py` | Semantic identity, canonical IR/schema meaning, minimal ContextPack construction, obligations, lineage, and proof relationships | Does not own durable bytes or supervisor execution. |
| `ipfs_kit_py` | Canonical bytes/CIDs, immutable blocks, current-root CAS, WAL, recovery, retention, and optional replication | Does not decide semantic correctness or proof/test reuse admission. |
| `ipfs_accelerate_py` | Supervisor state, claims/leases/fences, routing, execution/validation/repair/reconciliation, terminalization/promotion, ContextPack freshness/selection, and current-tree reuse admission | Does not remint Datasets semantic identities. |

Datasets remains the sole semantic ContextPack builder. Kit durability proves stored bytes, not semantic correctness. Accelerate independently checks freshness and admits execution/proof/test reuse. Cross-repository code communicates through stable contracts and installed-package/artifact boundaries; tests do not import a sibling repository's source-tree test package.

### Classification and migration rule

The inventory uses only `canonical`, `compatibility_adapter`, `deprecated`, `test_fixture`, and `non_authoritative_input` dispositions. An adapter, fallback, fixture, or historical input may support a bounded migration or test but cannot claim authority independently. Removing a compatibility path requires a documented replacement, equivalence evidence, migrated callers, and proof that it cannot claim authority independently.

## Consequences

- ASEH work extends these existing paths and must not introduce an alternate writer, router, supervisor, store, or ContextPack identity issuer.
- Validators, provers, patchers, mergers, receipt writers, fallbacks, and recovery helpers remain subordinate to the state-owner and admitted-receipt boundaries.
- Promotion is independent, operator-authorized, CAS-protected, and cannot be self-approved by a candidate branch, model, or controller.
- Missing metrics remain unavailable. Simulated, attempted, observed, and verified states remain distinct.

## Evidence and nonclaims

The exact source commit/tree and every scoped path classification are in [authority_inventory.json](../agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json). This ADR neither changes runtime behavior nor asserts that the current tree is measured, qualified, promoted, or production-ready.
