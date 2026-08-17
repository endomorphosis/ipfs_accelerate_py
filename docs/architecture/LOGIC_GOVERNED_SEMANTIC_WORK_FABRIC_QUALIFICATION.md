# LGSWF qualification report

Independent review of release tree
`agent/logic-governed-semantic-work-fabric-actual-v1` at
`085bd3bb062fd0d7cd72a9b80fdcdc4f06c5beb0`
(tree `25fa0305dc74514a77cfd1a3c75779e5b8c818a5`).
Reviewed 2026-08-17. Reviewer is distinct from the implementation daemon
session that completed LGSWF-131 and LGSWF-140.

This report is scoped to the exact environment observed on that date. It does
not authorize deployment, protected merge, or a production claim.

## 1. Exact revisions

- Branch: `agent/logic-governed-semantic-work-fabric-actual-v1`
- HEAD: `085bd3bb062fd0d7cd72a9b80fdcdc4f06c5beb0`
- Tree: `25fa0305dc74514a77cfd1a3c75779e5b8c818a5`
- Host DuckDB: `1.5.2`
- Quack extension: LOAD-able in-process; live owner served
  `quack:127.0.0.1:41307`
- Control schema revision: `datasets-authoritative-operational-v1`
- Store generation: `lgswf-actual-v6`
- Plan 1.5.5 pin: **not** installed. Observed transport is 1.5.2.

## 2. Current-state inventory

Canonical board in the Quack-attached control catalog has 47 tasks.
At review time: 46 `completed`, LGSWF-141 `todo`. No active task blocks.
LGSWF-142 exists only as a non-canonical markdown reconciliation note and is
not a control-plane task.

The LGSWF-140 release manifest is
`data/agent_supervisor/logic_governed_semantic_work_fabric/release/qualification-release.json`
and contains only `schema` plus `level=candidate`. It does not transitively
name inventory, seven Quack repository seals, canonical gateway, remote
readiness, or DuckLake receipts.

## 3. Authority map

- Semantic truth: `ipfs_datasets_py` (datasets-owned).
- Operational coordination: `ipfs_accelerate_py` agent supervisor.
- Transactional records: DuckDB.
- Multi-reader/multi-writer transport: Quack loopback ATTACH to an exclusive
  state-owner.
- DuckLake: optional non-authoritative history projection. Not started.
- Markdown todo is not completion authority.

## 4. Board / dependency graph

The remaining DAG is LGSWF-141 ← LGSWF-140. LGSWF-140 is completed, so 141 is
dependency-ready. 141 is `review_only`, `completion=manual`,
`is_schedulable=false`. The implementation daemon correctly refuses to
auto-claim it.

## 5. Reused and changed modules

Quack-default cutover reused existing `QuackStateServer`,
`DatabaseImplementationDaemon`, and `IntentRepository`. Changes required to
make the live path work:

- Recover `quack:` URIs after `Path.absolute()`.
- Keep execution and coordination on local sidecar files.
- Forward ATTACH `UPDATE`/`DELETE` through the exclusive owner mutation inbox
  because this Quack build cannot mutate attached base tables.
- Skip idempotent coordination re-registration to avoid a DuckDB 1.5 unique
  index FATAL.

## 6. World overlay

World snapshot/view code exists from earlier completed tasks. DuckLake
projection health is optional and recorded as typed unavailable. Absence of
DuckLake must not and did not unschedulable the board.

## 7. Bindings

Board namespace `logic-governed-semantic-work-fabric-actual-v1`. Live run
`run-actual-v6`. Tasks bind through the datasets-authoritative operational
control plane. The thin 140 manifest does not restate every capsule CID.

## 8. Composite / conflict analysis

No active conflicts on remaining board work. 141 is exclusive
qualification-decision review. Parallel lanes are idle because only this
serial review remains.

## 9. Frontier algorithm

Conflict-free frontier selected 131 then 140 serially after the Quack catalog
path worked. Independent earlier pairs (080/081, 090/091, 121/122) completed
before this review. Remaining work is not parallelizable.

## 10. Resource policy

Observed run used one supervisor lane and one daemon (`task-shard-index 0` of
configured 2). Resource claims for sealed A-D 12-core / 32 GiB benchmark
budgets were not consumed. Those cells are typed not-executed.

## 11. Supervisor and daemon protocols

Official path is `implementation_supervisor` + `implementation_daemon` with
`--task-source-kind duckdb --authority-mode quack`. Live PIDs at review:
owner 3589997, supervisor 3598085, daemon 3639797. Daemon completed 131 and
140 under `authority_mode=quack`.

## 12. Revision / refill

No plan refill was required for 141. Deterministic writers under
`scripts/materialize_lgswf_*.py` were used for earlier producing tasks because
model workers cannot self-seal this board.

## 13. Refresh and fixed point

Post-merge semantic refresh modules exist from completed I/J tasks. A
full fixed-point receipt for production continuous operation is **not**
accepted: claims for 141 remain open until this review lands, the 140
manifest is incomplete, and continuous-Quack production gates fail.

## 14. Fault results

Fault and adversarial suites (LGSWF-121, LGSWF-122) are marked completed on
the board. This review does not re-run those suites. It treats their
completion as board state, not as a fresh adversarial assurance package.

## 15. Parallel / resource benchmarks

`data/agent_supervisor/logic_governed_semantic_work_fabric/benchmarks/results.json`
reports schema `lgswf/benchmark-results@1`. Honest cells:

- Suite A Quack control: smoke observed (47-task count via ATTACH).
- Suites A–D embedded full repetitions: typed unavailable / not executed.
- Suites B–D Quack full repetitions: typed unavailable / not executed.
- DuckLake all suites: typed unavailable.
- Target heartbeat/commit/parity numbers were **not** substituted.

## 16. Model / proof reuse

Implementation of 131 and 140 used deterministic writers, not model
self-approval. This qualification is an independent review of those outputs
plus the live Quack probe.

## 17. Scheduling overhead

No sealed p99 heartbeat or commit-regression measurement is accepted. Those
remain targets, not observed values.

## 18. Security

Loopback-only Quack URI. Token is an opaque handle
(`handle:lgswf-actual-v6`), not logged here. Non-loopback Quack is rejected.
Provider subprocesses do not inherit the Quack token. Direct file open of a
`quack:` target is refused. The owner holds exclusive `control.duckdb`.

Gaps versus the continuous-Quack production gate: no demonstrated
server-side scoped authorization across seven Quack-backed repositories, no
canonical gateway consolidation, no independent remote-readiness receipt
beyond loopback, and ATTACH clients cannot UPDATE/DELETE base tables.

## 19. Limitations

- Host Quack is 1.5.2, not the plan’s 1.5.5 pin.
- Quack ATTACH is read/insert; mutations are owner-serialized.
- Coordination and execution metadata are local sidecars, not Quack-backed
  repositories.
- Full A–D benchmark repetitions were not run.
- DuckLake is not running.
- The 140 qualification-release manifest is a schema stub, not a transitive
  CID closure.
- Dirty untracked materialize scripts remain in the checkout (LGSWF-142
  guardrail; non-canonical).

## 20. Qualification level

**`research_demo`**

Rationale: the live loopback DuckDB + Quack control path selected, completed,
and advanced board work. That is a demonstrated research/demo of the intended
authority split. It is not `internal_pilot`, `supervised_external_pilot`, or
`production_candidate`. The 140 manifest and missing 1.5.5 / gateway /
seven-repository seals forbid those higher levels.

## 21. Continuous-operation go/no-go

**NO-GO** for continuous multi-supervisor production mutation.

A loopback research control plane is observed. That is not continuous
production operation. Plan section 9 still requires the exact pinned Quack
profile, Quack-backed task/coordination/attempt/provider/effect/validation/CAS
repositories through one canonical gateway, remote readiness, server
authorization, and a clean direct-file-open audit. Those gates are not all
green.

## 22. Embedded DuckDB versus DuckDB + Quack

- Embedded one-writer: still the legal single-process bootstrap. Not the live
  multi-process control path. Multi-process direct file opens of
  `control.duckdb` are refused while the owner lives.
- DuckDB + Quack: live control for this run. Catalog reads work. Writes to
  existing rows require the owner mutation inbox.

No comparative throughput/latency table is claimed. Full-suite cells are
typed not-executed.

## 23. Control-plane-only versus DuckLake-projected

Control-plane-only operation was observed. DuckLake projection was not
started. There is no projected-versus-control measurement. DuckLake absence
did not block scheduling.

## 24. Activation recommendations

| Mode | Decision | Binding conditions |
|---|---|---|
| Embedded one-writer | **GO** for single-process research/demo | One writer. Not multi-process file sharing. |
| Continuous DuckDB + Quack multi-supervisor mutation | **NO-GO** | Loopback research path observed; 1.5.5 pin, canonical gateway, seven Quack-backed repositories, remote readiness, and ATTACH mutation are incomplete. |
| Live DuckLake analytics/history | **NO-GO** | Not started. Optional. Do not infer from Quack. Requires control-plane admission plus DuckLake/httpfs pins, catalog/profile/binding, projection/security/recovery evidence, and a release receipt. |

Permitted summary claim for this exact environment only:

> The agent supervisor composed datasets semantic authority with
> accelerator-owned DuckDB transactional records and a loopback Quack
> state-owner. It completed 46 of 47 board tasks on that control plane.
> Continuous production Quack mutation and live DuckLake remain NO-GO.
