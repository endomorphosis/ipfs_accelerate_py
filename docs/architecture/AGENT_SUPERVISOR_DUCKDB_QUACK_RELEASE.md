# Agent Supervisor DuckDB + Quack Control-Plane Release (DQP-039)

**Status:** terminal joined release gate for the DuckDB/Quack control-plane board  
**Task:** `DQP-039`  
**Goal:** `DQP-G090`  
**Board namespace:** `agent-supervisor-duckdb-quack-control-plane-v1`  
**Module:** `ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_release.py`  
**Interface:** `DuckDBControlPlaneReleaseReceipt@1`  
**Verifier:** `DuckDBControlPlaneReleaseVerifier@1` / `ReleaseEvidence`

This document describes the **joined database-control-plane release receipt**.
It is the unique board sink for program release: it depends on every prior
implementation task (`DQP-001` … `DQP-038`) and joins their current evidence
roots into one content-bound decision. A completed taskboard or a green narrow
suite is **not** sufficient.

Related operator surface: [DuckDB Quack Guide](../guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md).  
Normative program plan: [DuckDB Quack Control-Plane Plan](AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md).  
Compatibility notes: [Quack Compatibility](agent_supervisor/QUACK_COMPATIBILITY.md).

## Trust boundary

**No automatic promotion. No completion authority. Not production HA.
No inferred DuckDB 2.0 compatibility.**

| Surface | Role at release |
| --- | --- |
| Current Git tree id + database identity + schema checksum + extension fingerprint | Candidate identity |
| Independently produced, measured evidence roots | Objective coverage |
| Absolute-zero safety floors from the sealed baseline catalog | Safety authority |
| Canary proving database-only decision authority | Cutover authority |
| Rollback identity bound to tree/schema | Recovery authority |
| Quack beta limitations / experimental scope | Honest scope seal |
| Task-status counts, prose, exports, caches, vector ranks, model responses | **Not** authority |
| Production multi-node HA or DuckDB 2.0 readiness | **Out of scope** — never claimed |

Release receipts are content-addressed. Metrics and task-status counts never
become admission or completion authority. Mutation, completion, merge, and
automatic promotion remain unauthorized on this surface. The verifier never
fabricates or refreshes component evidence; producers own those roots.

## What the joined gate proves

1. **Evidence roots** — every required root is present, measured/current,
   non-synthetic, non-skipped, and bound to the same tree, schema checksum, and
   Quack profile:

   `schema`, `quack`, `import_export`, `intent`, `runtime`, `worktree`,
   `ast_mutation`, `symbolic_proof`, `context_churn`, `control`, `watchdog`,
   `backup`, `chaos`, `canary`, `shadow`, `cutover`, `rollback`.

2. **Bad evidence rejection** — missing, stale (class or age), synthetic,
   skipped, forged, failed, tree-mismatched, schema-mismatched, or
   profile-mismatched roots fail closed.

3. **Canary authority** — zero legacy-file decision reads; the database is sole
   decision authority; exports remain non-authoritative projections.

4. **Safety counters** — unauthorized SQL, stale lease writes, false
   completion, and accepted-state loss are exactly zero.

5. **Lineage / projection / rollback** — mutation lineage is complete; event
   and projection surfaces do not diverge; a rollback identity is present.

6. **Safety and quality floors** — absolute-zero floors from
   `SupervisorStateBaseline@1` / `DuckDBQuackBaseline` (duplicate non-idempotent
   effects, stale lease writes, unauthorized SQL, secret leakage, false
   completion, missing impact frontier admission, AST/mutation misbinding,
   event/projection divergence, accepted-state loss, and aggregate safety
   violations). Safety or quality regression fails the release.

7. **Cold imports** — release-critical modules import on the current tree
   without granting process authority.

8. **Experimental scope seal** — every pass records Quack beta limitations and
   explicit non-claims. A pass **does not** claim production HA or future
   DuckDB 2.0 compatibility. Compatibility with DuckDB 2.0 is unknown until
   separately tested and must not be inferred from a 1.5.x pass.

## Explicit non-claims (sealed on every receipt)

```text
not_production_ha
not_multi_failure_domain
not_duckdb_2_0_compatible_until_separately_tested
quack_remains_experimental_beta_in_1_5_x
loopback_single_owner_topology_only
```

Quack remains experimental/beta in DuckDB 1.5.x. One Quack server is one
failure domain. Loopback bind is required unless separately reviewed. Protocol
names and defaults may change before DuckDB 2.0. Server and clients must use
the identical pinned build.

## Default policy

```text
mode                         = report_only / join-only
mutation_authorized          = false
completion_authoritative     = false
promotion_allowed            = false
production_ha_claimed        = false
duckdb_2_0_compatibility     = false (until separately tested)
experimental_scope           = true
evidence_max_age_seconds     = 86400
require_zero_safety_floors   = true
require_database_sole_authority = true
require_rollback             = true
require_mutation_lineage     = true
```

## API surface

| Symbol | Role |
| --- | --- |
| `ReleaseEvidence` / `ReleaseEvidenceItem` | Joined evidence package and per-root items |
| `DuckDBControlPlaneReleasePolicy` | Immutable fail-closed release policy |
| `DuckDBControlPlaneReleaseVerifier` | Independent join-and-decide verifier |
| `DuckDBControlPlaneReleaseReceipt` | Content-addressed joined receipt |
| `issue_release_receipt` / `validate_duckdb_quack_release` | Full gate; returns sealed receipt |
| `replay_release_receipt` | Prove identity-equivalent reseal |
| `hermetic_passing_evidence` | Test fixture only (never production authority) |
| `classify_evidence_disposition` | Map one root to admissible / denial code |

### Minimal usage

```python
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_release import (
    hermetic_passing_evidence,
    issue_release_receipt,
    replay_release_receipt,
)

evidence = hermetic_passing_evidence()  # or load producer roots
receipt = issue_release_receipt(evidence)
assert receipt.passed
assert receipt.experimental_scope is True
assert receipt.production_ha_claimed is False
assert receipt.duckdb_2_0_compatibility_claimed is False
assert replay_release_receipt(receipt, evidence)["identity_ok"]
```

### Validation command

```bash
python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_release.py
```

## Fail-closed denial reasons

| Reason | Meaning |
| --- | --- |
| `missing_evidence` | Required root absent |
| `stale_evidence` | Root class is stale or age exceeds policy |
| `synthetic_evidence` | Simulated / non-measured root |
| `skipped_evidence` | Required gate was skipped |
| `forged_evidence` | Integrity seal failed |
| `legacy_file_decision_read` | Canary still decided from legacy files |
| `database_not_sole_authority` | Database is not sole decision authority |
| `unauthorized_sql` | Unauthorized SQL observed |
| `stale_lease_write` | Stale-owner write observed |
| `false_completion` | False completion observed |
| `accepted_state_loss` | Accepted state lost under declared crash model |
| `incomplete_mutation_lineage` | Mutation/AST lineage incomplete |
| `projection_divergence` | Event/projection surfaces diverge |
| `safety_regression` / `quality_regression` | Floors or quality regressed |
| `absent_rollback` | No rollback identity |
| `safety_floor_nonzero` | Absolute-zero floor violated |
| `production_ha_claim` | Evidence attempted to claim production HA |
| `duckdb_2_0_compatibility_claim` | Evidence inferred future 2.0 compatibility |
| `beta_scope_unrecorded` | Experimental/beta scope not sealed |

## Safety floors (absolute zero)

Mirrored from the sealed DuckDB/Quack baseline catalog:

- duplicate non-idempotent effects
- stale lease writes
- unauthorized SQL
- secret leakage
- false completion
- missing impact frontier admission
- AST/mutation misbinding
- event/projection divergence
- accepted-state loss
- aggregate safety floor violations

## Operator notes

1. **Join, do not summarize optimistically.** Component green lights do not
   become a release pass without independent current evidence for every root.
2. **Do not claim HA.** A single Quack state-owner is one failure domain; HA
   requires a separately reviewed multi-domain design.
3. **Do not claim DuckDB 2.0 readiness.** Pin and rehearse any future upgrade
   with restore tests before asserting compatibility.
4. **Rollback is mandatory evidence.** Cutover without a bound rollback
   identity fails the release.
5. **Exports are not authority.** Markdown/JSON/JSONL projections remain
   human/tool surfaces only after cutover.

## Related tasks

| Task | Contribution |
| --- | --- |
| DQP-001 … DQP-008 | Schema, Quack server/client, repositories |
| DQP-009 / DQP-036 | Baseline floors and quality/safety benchmark |
| DQP-010 … DQP-011 | Import / export non-authority |
| DQP-012 … DQP-019 | Intent, runtime, leases, worktrees, merge |
| DQP-020 … DQP-028 | AST/mutation lineage, symbolic/proof, context/churn |
| DQP-029 … DQP-033 | Control ops, authority modes, watchdog, backup |
| DQP-034 | Chaos / security |
| DQP-035 | Multi-lane canary |
| DQP-037 | Shadow decision parity |
| DQP-038 | Staged cutover + rollback |
| **DQP-039** | **Joined release receipt (this document)** |
