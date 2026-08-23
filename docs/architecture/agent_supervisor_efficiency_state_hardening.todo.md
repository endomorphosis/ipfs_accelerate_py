# Agent Supervisor Efficiency and State Hardening task board

Executable bootstrap projection for `ipfs_accelerate_py.agent_supervisor`.
Program and board namespace: `agent-supervisor-efficiency-and-state-hardening-v1`.
Task prefix: `ASEH-`. Plan revision: `ASEH-PLAN-R1`.

DuckDB `DatabaseTaskSource@1` is authoritative after one offline
materialization. One loopback `QuackStateServer@1` exclusively owns live
mutation. DuckLake is an optional append-only analytics projection and cannot
gate claims, leases, validation, completion, recovery, or promotion. Markdown
status is never operational authority.

Every leaf uses the existing submodule-aware isolated-worktree bridge. Output
paths are relative to its exact `Owning repository`. Tasks in sibling
repositories change the pinned submodule and superproject gitlink through the
canonical merge path; no sibling supervisor or direct state writer exists.

All tasks begin `todo`, automatic, schedulable, and fail closed. A dependency
that is not yet accepted means `waiting`, not blocked. The only initial-ready
tasks are `ASEH-000` and `ASEH-001`. Candidate behavior may not land before
both exact baseline tasks are accepted.

Common proof contract: focused validation must include positive, schema or
contract, and negative cases; raw logs are non-authoritative; authoritative
receipts bind current repository commit/tree, objective revision, policy,
interfaces, toolchain, environment, validator identity, and exact command.
Common hard gates are the fourteen zero-tolerance safety outcomes in the
program plan. No task may self-complete, self-promote, lower a gate, or consume
the final validation reserve.

## Parallel dependency waves

```text
W0  000 | 001
W1  010 | 020 | 030 | 040 | 050
W2+ dependency-driven fan-out within measurement, routing, ContextPack,
    state-machine, and planning/synthesis tracks
JOIN 060 -> 061 -> 062
QUAL 070 -> 071 -> 072 -> 073 -> 074 -> 075
```

The executable `Depends on` fields, not this summary, are scheduling
authority.

## ASEH-000 Inventory current authorities and bypasses

- Stable task ID: ASEH-000
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: authority-baseline
- Goal id: ASEH-G010
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G010
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Inventory every scoped mutator, launcher, validator, prover, patcher, merger, terminalizer, receipt writer, policy updater, fallback, and recovery entry point with owner/status/store/bypass/metrics/disposition, and record the canonical supervisor and cross-repository authorities in an architecture decision record.
- Depends on:
- Exact declared outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json, docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md, test/api/agent_supervisor/efficiency_state_hardening/test_authority_inventory.py
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json, docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md, test/api/agent_supervisor/efficiency_state_hardening/test_authority_inventory.py
- Owned paths: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json, docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md, test/api/agent_supervisor/efficiency_state_hardening/test_authority_inventory.py
- Predicted files: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json, docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md, test/api/agent_supervisor/efficiency_state_hardening/test_authority_inventory.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: read source and write inventory evidence only; no runtime or policy mutation
- Acceptance conditions: Every in-scope path is classified against the exact source tree, and the ADR names the canonical supervisor handoff, cross-repository boundaries, state-machine authority, routing authority, and ContextPack authorities; inventory, ADR completeness, schema, and negative tests pass.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_authority_inventory.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Every in-scope path is classified against the exact source tree, the required ADR decisions bind those classifications, and the inventory schema, ADR completeness, and negative tests pass.
- Terminal non-success criteria: Any unclassified path, unknown owner, unreadable scope, or failed completeness check yields a typed non-success receipt and no acceptance.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-001 Seal current repository and supervisor baseline

- Stable task ID: ASEH-001
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: authority-baseline
- Goal id: ASEH-G010
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G010
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Seal exact three-repository commits/trees, policy/config/provider identities, task and acceptance corpora, cost method, benchmark digest, resource limits, and environment identity before candidate behavior changes.
- Depends on:
- Exact declared outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json, benchmarks/agent_supervisor/efficiency_state_hardening/provider_model_config.json, benchmarks/agent_supervisor/efficiency_state_hardening/price_snapshot.json, benchmarks/agent_supervisor/efficiency_state_hardening/environment_identity.json, benchmarks/agent_supervisor/efficiency_state_hardening/sealed_input_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json, benchmarks/agent_supervisor/efficiency_state_hardening/provider_model_config.json, benchmarks/agent_supervisor/efficiency_state_hardening/price_snapshot.json, benchmarks/agent_supervisor/efficiency_state_hardening/environment_identity.json, benchmarks/agent_supervisor/efficiency_state_hardening/sealed_input_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py
- Owned paths: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json, benchmarks/agent_supervisor/efficiency_state_hardening/provider_model_config.json, benchmarks/agent_supervisor/efficiency_state_hardening/price_snapshot.json, benchmarks/agent_supervisor/efficiency_state_hardening/environment_identity.json, benchmarks/agent_supervisor/efficiency_state_hardening/sealed_input_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py
- Predicted files: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json, benchmarks/agent_supervisor/efficiency_state_hardening/provider_model_config.json, benchmarks/agent_supervisor/efficiency_state_hardening/price_snapshot.json, benchmarks/agent_supervisor/efficiency_state_hardening/environment_identity.json, benchmarks/agent_supervisor/efficiency_state_hardening/sealed_input_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: immutable evidence creation only; observed capability is not qualification
- Acceptance conditions: The sealed artifacts bind exact commits/trees, supervisor policy, provider/model revisions, a timestamped price snapshot, task and acceptance-input identities, cost-accounting method, benchmark-code digest, resource limits, and environment identity; unavailable fields remain unavailable and mutation-detection tests pass.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_sealed_baseline.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Every required baseline input has a canonical identity, the provider/price/environment/input manifests reconcile to the baseline, unavailable fields remain unavailable, and exact-tree and mutation-detection tests pass.
- Terminal non-success criteria: A moving branch, dirty/unbound source, synthetic identity, absent required seal, or post-seal mutation yields a typed non-success receipt.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-010 Define telemetry and efficiency receipt schemas

- Stable task ID: ASEH-010
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: measurement
- Goal id: ASEH-G020
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G020
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Extend the canonical telemetry contracts with all identity, model, compute, cost, work, quality, evidence-state, and terminal fields; missing values remain unavailable.
- Depends on: ASEH-000, ASEH-001
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/runtime/efficiency_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/task_efficiency_receipt.schema.json, ipfs_accelerate_py/agent_supervisor/runtime/schemas/paired_benchmark_manifest.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_efficiency_receipts.py
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/efficiency_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/task_efficiency_receipt.schema.json, ipfs_accelerate_py/agent_supervisor/runtime/schemas/paired_benchmark_manifest.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_efficiency_receipts.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/runtime/efficiency_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/task_efficiency_receipt.schema.json, ipfs_accelerate_py/agent_supervisor/runtime/schemas/paired_benchmark_manifest.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_efficiency_receipts.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/efficiency_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/task_efficiency_receipt.schema.json, ipfs_accelerate_py/agent_supervisor/runtime/schemas/paired_benchmark_manifest.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_efficiency_receipts.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: schema authority only; cannot attest measurements or task completion
- Acceptance conditions: Closed schemas round-trip canonical bytes, reject unknowns/bounds errors, and distinguish measured, estimated, unavailable, attempted, observed, and verified.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_efficiency_receipts.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Closed schemas round-trip canonical bytes, reject unknowns/bounds errors, and distinguish measured, estimated, unavailable, attempted, observed, and verified.
- Terminal non-success criteria: Conflated truth states, missing required identity, noncanonical numeric values, or unknown fields fail closed and prevent receipt admission.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-011 Instrument provider and model usage

- Stable task ID: ASEH-011
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: measurement
- Goal id: ASEH-G020
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G020
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Record exact provider/model revision, request IDs when safe, input/output/cached/reasoning tokens where reported, call counts/classes, reported charge, and labeled estimates.
- Depends on: ASEH-010
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_provider_usage_telemetry.py
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_provider_usage_telemetry.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_provider_usage_telemetry.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_provider_usage_telemetry.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: observe provider use; never fabricate unavailable usage or expose credentials
- Acceptance conditions: Provider responses map causally to admitted task spans and all absent usage/cost fields remain unavailable rather than zero.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_provider_usage_telemetry.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Provider responses map causally to admitted task spans and all absent usage/cost fields remain unavailable rather than zero.
- Terminal non-success criteria: Unbound usage, credential leakage, estimated-as-measured data, or missing causal identity quarantines the telemetry record.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-012 Instrument tests proofs retries merges and human intervention

- Stable task ID: ASEH-012
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: measurement
- Goal id: ASEH-G020
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G020
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Record compute/process/IO timing, selected and executed checks/proofs, reuse, retries/rescue/conflicts/recovery, human actions, outcome, validation, and patch disposition including audit overhead.
- Depends on: ASEH-010, ASEH-011
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_work_telemetry.py
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_work_telemetry.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_work_telemetry.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, test/api/agent_supervisor/efficiency_state_hardening/test_work_telemetry.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: observation only; process exit alone cannot verify external effects
- Acceptance conditions: Causal spans cover every requested work and compute field with explicit availability and admitted verifier linkage.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_work_telemetry.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Causal spans cover every requested work and compute field with explicit availability and admitted verifier linkage.
- Terminal non-success criteria: Attempted-as-observed, observed-as-verified, missing overhead, duplicate terminal accounting, or invalid bounds fail closed.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-013 Build paired benchmark harness

- Stable task ID: ASEH-013
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: measurement
- Goal id: ASEH-G020
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G020
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Build and seal at least 60 bounded hermetic fixtures, then run direct/minimal, sealed-current, and candidate arms with identical revision, task input, validators, providers, prices, limits, retries, and human policy; calculate paired statistics and audit overhead.
- Depends on: ASEH-010, ASEH-011, ASEH-012
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py, benchmarks/agent_supervisor/efficiency_state_hardening/manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py, benchmarks/agent_supervisor/efficiency_state_hardening/manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py, benchmarks/agent_supervisor/efficiency_state_hardening/manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py, benchmarks/agent_supervisor/efficiency_state_hardening/manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: hermetic harness may orchestrate isolated fixtures but cannot grant live or promotion status
- Acceptance conditions: The hermetic manifest contains at least 60 unique bounded fixtures, and pairing rejects unequal controls while computing median/mean differences, ratios, bootstrap intervals, class distributions, outliers, quality-adjusted cost, acceptance rate, and terminal time.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: At least 60 hermetic inputs are canonically sealed and pairing rejects unequal controls while computing all required statistics and audit overhead.
- Terminal non-success criteria: Unequal arms, missing paired inputs, unseeded nondeterminism, fabricated live labels, or incomplete audit cost produces an inadmissible result.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-014 Build historical replay corpus

- Stable task ID: ASEH-014
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: measurement
- Goal id: ASEH-G020
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G020
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Seal at least 20 real historical tasks at exact historical trees spanning success, failure, retry, rescue, conflict, and human escalation without changing their acceptance contracts.
- Depends on: ASEH-013
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/historical_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/historical_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_historical_corpus.py
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/historical_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/historical_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_historical_corpus.py
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/historical_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/historical_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_historical_corpus.py
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/historical_manifest.json, benchmarks/agent_supervisor/efficiency_state_hardening/historical_vectors.jsonl, test/api/agent_supervisor/efficiency_state_hardening/test_historical_corpus.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: read historical source/receipts; replay isolation only; no historical state mutation
- Acceptance conditions: At least 20 provenance-complete, deduplicated, exact-tree vectors cover every required outcome class and install without sibling test imports.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_historical_corpus.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Either at least 20 provenance-complete, deduplicated, exact-tree vectors cover every required outcome class and install without sibling test imports, or a valid typed insufficient_historical_population receipt truthfully records the bounded unavailable evidence for downstream non-promotion.
- Terminal non-success criteria: Missing source trees represented as present, inferred outcomes, altered acceptance, duplicated tasks, malformed provenance, or failure to emit either an admissible corpus or the typed insufficiency receipt rejects the task.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-015 Add live shadow and canary cohort

- Stable task ID: ASEH-015
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: measurement
- Goal id: ASEH-G020
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G020
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Define admission for at least 10 newly encountered real tasks, candidate shadow-first execution, and separately gated low-risk canary mutation with real provider and verifier evidence.
- Enrollment deadline: A sealed UTC timestamp no later than 30 calendar days after admission of the live cohort manifest; the deadline is immutable once sealed.
- Depends on: ASEH-013, ASEH-014
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json, ipfs_accelerate_py/agent_supervisor/control/live_cohort_admission.py, test/api/agent_supervisor/efficiency_state_hardening/test_live_cohort_admission.py
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json, ipfs_accelerate_py/agent_supervisor/control/live_cohort_admission.py, test/api/agent_supervisor/efficiency_state_hardening/test_live_cohort_admission.py
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json, ipfs_accelerate_py/agent_supervisor/control/live_cohort_admission.py, test/api/agent_supervisor/efficiency_state_hardening/test_live_cohort_admission.py
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json, ipfs_accelerate_py/agent_supervisor/control/live_cohort_admission.py, test/api/agent_supervisor/efficiency_state_hardening/test_live_cohort_admission.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: shadow is non-mutating; canary mutation requires current safety admission; fixtures never satisfy live
- Acceptance conditions: The admission service seals the bounded enrollment deadline, requires 10 distinct new tasks for an evidence-qualified cohort, proves live provenance, enforces shadow before canary, rejects simulation or fixture substitution, and supports a truthful typed insufficiency or non-admission disposition.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_live_cohort_admission.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: The admission service seals a bounded UTC enrollment deadline, requires 10 distinct new tasks for an evidence-qualified cohort, proves live provenance, enforces shadow before canary, rejects simulation or fixture substitution, and emits a valid typed insufficiency or non-admission receipt when the deadline or gate prevents enrollment.
- Terminal non-success criteria: A missing or mutable deadline, false live provenance, fixture substitution, shadow bypass, untyped insufficiency, or a malformed or nontruthful cohort disposition rejects the task.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-020 Consolidate deterministic-first routing

- Stable task ID: ASEH-020
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-routing
- Goal id: ASEH-G030
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G030
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Select one existing routing authority and encode the exact nine-stage receipt-to-human decision ladder without creating a router family.
- Depends on: ASEH-000, ASEH-001
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_ladder.py
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_ladder.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_ladder.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_ladder.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: routing selects executors but cannot skip validation, mutate policy, or authorize promotion
- Acceptance conditions: Every decision traverses stages in order, with typed run/skip reasons, and deterministic evidence resolves eligible cases before models.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_ladder.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Every decision traverses stages in order, with typed run/skip reasons, and deterministic evidence resolves eligible cases before models.
- Terminal non-success criteria: Two routing authorities, availability-driven skip, unordered stages, silent fallback, or policy self-mutation fails closed.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-021 Add unresolved-question contract

- Stable task ID: ASEH-021
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: deterministic-routing
- Goal id: ASEH-G030
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G030
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Add the closed question identity, exact question, prior-stage failure, present/missing evidence, decision impact, minimum capability, budgets, response schema, and deadline required for escalation.
- Depends on: ASEH-010, ASEH-020
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/unresolved_question.py, ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/unresolved_question.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unresolved_question.py
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/unresolved_question.py, ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/unresolved_question.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unresolved_question.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/semantic_state/unresolved_question.py, ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/unresolved_question.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unresolved_question.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/unresolved_question.py, ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/unresolved_question.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unresolved_question.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: contract may request an executor but grants no invocation or decision authority
- Acceptance conditions: Canonical round-trip and bounds tests pass; questions whose answers cannot change an admissible decision are rejected.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_unresolved_question.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Canonical round-trip and bounds tests pass; questions whose answers cannot change an admissible decision are rejected.
- Terminal non-success criteria: Missing decision impact, unbounded context/cost/time, unknown fields, or a no-op answer causes escalation denial.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-022 Add deterministic cache AST static test and prover route stages

- Stable task ID: ASEH-022
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-routing
- Goal id: ASEH-G030
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G030
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Connect exact receipt freshness, AST/symbol/dependency impact, schema/type/static/lint/contracts, selected tests, and incremental proof into the canonical ladder.
- Depends on: ASEH-020, ASEH-021
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/semantic_state/deterministic_stages.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_stages.py
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/semantic_state/deterministic_stages.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_stages.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/semantic_state/deterministic_stages.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_stages.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/semantic_state/deterministic_stages.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_stages.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: deterministic stages observe/admit evidence under existing authorities; they cannot mint Datasets identities or validation success
- Acceptance conditions: Known cache, stale-tree, lease/fence, policy CAS, schema, dependency, documentation-only, test-selection, and proof-identity cases resolve deterministically with exact receipts.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_stages.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Known cache, stale-tree, lease/fence, policy CAS, schema, dependency, documentation-only, test-selection, and proof-identity cases resolve deterministically with exact receipts.
- Terminal non-success criteria: Stale receipt admission, skipped required check, false selected-test negative, or reminted semantic identity is a hard safety failure.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-023 Add small medium and frontier escalation policy

- Stable task ID: ASEH-023
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deterministic-routing
- Goal id: ASEH-G030
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G030
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Select the smallest adequate local small, local/remote medium, remote frontier, or human executor under typed unresolved questions, context/cost/time ceilings, and validation reserve.
- Depends on: ASEH-022
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, ipfs_accelerate_py/agent_supervisor/autonomy/route_policy.py, test/api/agent_supervisor/efficiency_state_hardening/test_model_escalation_policy.py
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, ipfs_accelerate_py/agent_supervisor/autonomy/route_policy.py, test/api/agent_supervisor/efficiency_state_hardening/test_model_escalation_policy.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, ipfs_accelerate_py/agent_supervisor/autonomy/route_policy.py, test/api/agent_supervisor/efficiency_state_hardening/test_model_escalation_policy.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, ipfs_accelerate_py/agent_supervisor/autonomy/route_policy.py, test/api/agent_supervisor/efficiency_state_hardening/test_model_escalation_policy.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: invocation admission only; no model output is verifier, policy, completion, or promotion authority
- Acceptance conditions: Capability matching, budget reservation, escalation precision, and human boundary tests pass; large-model availability alone changes nothing.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_model_escalation_policy.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Capability matching, budget reservation, escalation precision, and human boundary tests pass; large-model availability alone changes nothing.
- Terminal non-success criteria: Missing question, exhausted validation reserve, insufficient prior-stage evidence, silent fallback, or self-authorizing output denies invocation.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-024 Add route receipts and escalation metrics

- Stable task ID: ASEH-024
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: deterministic-routing
- Goal id: ASEH-G030
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G030
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Record stages run/skipped and why, decisive evidence, escalation, executor, actual resources, outcome, deterministic/small/frontier/human shares, precision, unnecessary escalation, and question closure.
- Depends on: ASEH-011, ASEH-012, ASEH-023
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/routing_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_route_receipts.py
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/routing_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_route_receipts.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/routing_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_route_receipts.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py, ipfs_accelerate_py/agent_supervisor/runtime/schemas/routing_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_route_receipts.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: receipts report routing observations only and cannot validate patches or promote policy
- Acceptance conditions: Closed route receipts reconcile to provider/work telemetry and compute every requested rate without imputing unavailable denominators.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_route_receipts.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Closed route receipts reconcile to provider/work telemetry and compute every requested rate without imputing unavailable denominators.
- Terminal non-success criteria: Stage mismatch, unbound usage, absent decisive evidence, division by missing data, or claimed closure without evidence rejects the receipt.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-030 Define canonical ContextPack contract

- Stable task ID: ASEH-030
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: context-pack
- Goal id: ASEH-G040
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G040
- Owning repository: ipfs_datasets_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 209dbe2765593fbc6efe8e9281c34f2e8f6e37a6
- Base repository tree: 95f54df34585d0b736706fd90c83f55954489ad9
- Base plan revision: ASEH-PLAN-R1
- Objective: Extend DatasetsContextPackAuthority rather than creating a competing type; bind requested identity, scope, contracts, validation, history, questions, budgets, freshness, invalidation, parent, and delta fields.
- Depends on: ASEH-000, ASEH-001
- Exact declared outputs: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/schemas/context_pack.schema.json, tests/proof_context/test_context_pack_contract.py
- Outputs: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/schemas/context_pack.schema.json, tests/proof_context/test_context_pack_contract.py
- Owned paths: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/schemas/context_pack.schema.json, tests/proof_context/test_context_pack_contract.py
- Predicted files: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/schemas/context_pack.schema.json, tests/proof_context/test_context_pack_contract.py
- Worktree isolation: canonical isolated task worktree for ipfs_datasets_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: Datasets owns semantic meaning and CID construction only; no storage, execution, reuse, or promotion authority
- Acceptance conditions: Canonical bytes/CID and closed-schema tests pass; route tier and all view-changing inputs bind identity; fixture/live and synthetic/real identities remain distinct.
- Validation: python3 -m pytest -q tests/proof_context/test_context_pack_contract.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Canonical bytes/CID and closed-schema tests pass; route tier and all view-changing inputs bind identity; fixture/live and synthetic/real identities remain distinct.
- Terminal non-success criteria: Fixture-as-live, synthetic identity accepted as complete, omitted view input, weak freshness, unknown field, or ownership drift fails closed.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-031 Implement Datasets semantic pack builder

- Stable task ID: ASEH-031
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: context-pack
- Goal id: ASEH-G040
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G040
- Owning repository: ipfs_datasets_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 209dbe2765593fbc6efe8e9281c34f2e8f6e37a6
- Base repository tree: 95f54df34585d0b736706fd90c83f55954489ad9
- Base plan revision: ASEH-PLAN-R1
- Objective: Build minimal semantic packs with contracts, dependency meaning, obligations, lineage, missing evidence, completeness witnesses, and truthful execution mode using the sole Datasets builder.
- Depends on: ASEH-030
- Exact declared outputs: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/provider.py, tests/proof_context/test_context_pack_builder.py
- Outputs: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/provider.py, tests/proof_context/test_context_pack_builder.py
- Owned paths: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/provider.py, tests/proof_context/test_context_pack_builder.py
- Predicted files: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/provider.py, tests/proof_context/test_context_pack_builder.py
- Worktree isolation: canonical isolated task worktree for ipfs_datasets_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: semantic builder may declare meaning/completeness only; it cannot budget provider execution or admit proof reuse
- Acceptance conditions: Minimality, source lineage, deterministic CID, fixture/live separation, completeness, missing-reference, unknown-field, and numeric-bound tests pass.
- Validation: python3 -m pytest -q tests/proof_context/test_context_pack_builder.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Minimality, source lineage, deterministic CID, fixture/live separation, completeness, missing-reference, unknown-field, and numeric-bound tests pass.
- Terminal non-success criteria: Hardcoded live mode, placeholder CIDs, silent completeness, Accelerator-owned budget decisions, or noncanonical identity quarantines the pack.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-032 Implement Kit pack storage and current-root CAS

- Stable task ID: ASEH-032
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: context-pack
- Goal id: ASEH-G040
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G040
- Owning repository: ipfs_kit_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: ba5508d940fb5b23a6d0d9b2084f5195cd26a671
- Base repository tree: 7c71efa93c4e4124d12fa05515868df3a5344b2e
- Base plan revision: ASEH-PLAN-R1
- Objective: Extend Kit's canonical proof_context state, incremental-seal, and verification stores under an explicit ContextPack namespace for verified bytes, immutable retrieval, candidate storage, separate current-root CAS, WAL, and deterministic recovery.
- Depends on: ASEH-030
- Exact declared outputs: ipfs_kit_py/proof_context/state_store.py, ipfs_kit_py/proof_context/incremental_seal_store.py, ipfs_kit_py/proof_context/verification_store.py, ipfs_kit_py/proof_context/context_pack_contract_vectors.json, tests/test_context_pack_store.py
- Outputs: ipfs_kit_py/proof_context/state_store.py, ipfs_kit_py/proof_context/incremental_seal_store.py, ipfs_kit_py/proof_context/verification_store.py, ipfs_kit_py/proof_context/context_pack_contract_vectors.json, tests/test_context_pack_store.py
- Owned paths: ipfs_kit_py/proof_context/state_store.py, ipfs_kit_py/proof_context/incremental_seal_store.py, ipfs_kit_py/proof_context/verification_store.py, ipfs_kit_py/proof_context/context_pack_contract_vectors.json, tests/test_context_pack_store.py
- Predicted files: ipfs_kit_py/proof_context/state_store.py, ipfs_kit_py/proof_context/incremental_seal_store.py, ipfs_kit_py/proof_context/verification_store.py, ipfs_kit_py/proof_context/context_pack_contract_vectors.json, tests/test_context_pack_store.py
- Worktree isolation: canonical isolated task worktree for ipfs_kit_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: Kit proves exact bytes/durable root only; it cannot decide semantics, freshness, reuse, execution, or promotion
- Acceptance conditions: Real CID equality, immutable get, stale CAS rejection, WAL/crash recovery, retention, unknown-field vectors, and package-data installation tests pass.
- Validation: python3 -m pytest -q tests/test_context_pack_store.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Real CID equality, immutable get, stale CAS rejection, WAL/crash recovery, retention, unknown-field vectors, and package-data installation tests pass.
- Terminal non-success criteria: CID mismatch, proof-object masquerade, candidate auto-publication, stale CAS success, recovery divergence, or semantic admission claim fails closed.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-033 Implement Accelerate freshness and selection

- Stable task ID: ASEH-033
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: context-pack
- Goal id: ASEH-G040
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G040
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Verify Datasets semantic identity and Kit bytes/root, enforce exact tree/objective/policy/interface/toolchain/environment freshness, select the minimal adequate pack, and record reuse/invalidation.
- Depends on: ASEH-030, ASEH-031, ASEH-032
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py, ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack_selector.py, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_selector.py
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py, ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack_selector.py, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_selector.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py, ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack_selector.py, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_selector.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py, ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack_selector.py, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_selector.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: Accelerate admits freshness, reuse, and executor context but cannot remint Datasets CIDs or bypass Kit verification
- Acceptance conditions: Current minimal pack selection and exact stale-identity rejection pass against installed packages and immutable vectors.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_selector.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Current minimal pack selection and exact stale-identity rejection pass against installed packages and immutable vectors.
- Terminal non-success criteria: Any stale admission, CID remint, silent whole-repository expansion, missing dependency acceptance, or storage-as-semantic proof is a hard failure.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-034 Implement incremental pack expansion

- Stable task ID: ASEH-034
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: context-pack
- Goal id: ASEH-G040
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G040
- Owning repository: ipfs_datasets_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 209dbe2765593fbc6efe8e9281c34f2e8f6e37a6
- Base repository tree: 95f54df34585d0b736706fd90c83f55954489ad9
- Base plan revision: ASEH-PLAN-R1
- Objective: Generate safe delta or affected-suffix packs and resolve only named missing CID, symbol, contract, test, counterexample, or obligation references; require typed completeness failure for whole-repo expansion.
- Depends on: ASEH-033
- Exact declared outputs: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/incremental_context.py, tests/proof_context/test_incremental_context_pack.py
- Outputs: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/incremental_context.py, tests/proof_context/test_incremental_context_pack.py
- Owned paths: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/incremental_context.py, tests/proof_context/test_incremental_context_pack.py
- Predicted files: ipfs_datasets_py/proof_context/context_pack.py, ipfs_datasets_py/proof_context/incremental_context.py, tests/proof_context/test_incremental_context_pack.py
- Worktree isolation: canonical isolated task worktree for ipfs_datasets_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: Datasets owns semantic delta/completeness; retrieval and executor budgets remain with Kit/Accelerate
- Acceptance conditions: Parent binding, changed-tree invalidation, expansion precision/recall, missing-reference, critical omission, and deterministic CID tests pass.
- Validation: python3 -m pytest -q tests/proof_context/test_incremental_context_pack.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Parent binding, changed-tree invalidation, expansion precision/recall, missing-reference, critical omission, and deterministic CID tests pass.
- Terminal non-success criteria: Unbound parent, broad rediscovery without typed failure, omitted critical dependency, route-dependent identity drift, or silent missing reference rejects the delta.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-035 Benchmark pack reuse omissions and net savings

- Stable task ID: ASEH-035
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: context-pack
- Goal id: ASEH-G040
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G040
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Measure eligible reuse, before/after context tokens, expansion precision/recall, critical omission detection, stale rejection, build/retrieval cost, audit overhead, and net provider-cost effect.
- Depends on: ASEH-013, ASEH-024, ASEH-034
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_benchmark.py, benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_benchmark.py
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_benchmark.py, benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_benchmark.py
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_benchmark.py, benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_benchmark.py
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_benchmark.py, benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_manifest.json, test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_benchmark.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: benchmark evidence only; hermetic results cannot grant live reuse or promotion
- Acceptance conditions: Paired current-tree measures include all requested metrics and retain unavailable data; seeded critical omissions and stale packs are always rejected.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_context_pack_benchmark.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Paired current-tree measures include all requested metrics and retain unavailable data; seeded critical omissions and stale packs are always rejected.
- Terminal non-success criteria: Aggregate-only results, missing audit cost, accepted critical omission, fixture-as-live label, or stale identity admission invalidates the benchmark.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-040 Define canonical state machine

- Stable task ID: ASEH-040
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-machine
- Goal id: ASEH-G050
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G050
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Extend the existing state model with all requested states/events and machine-readable transition guards; map current legacy statuses explicitly without creating a second authority.
- Depends on: ASEH-000, ASEH-001
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_state_model.py, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition.schema.json, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition_table.json, test/api/agent_supervisor/efficiency_state_hardening/test_state_transition_table.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_state_model.py, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition.schema.json, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition_table.json, test/api/agent_supervisor/efficiency_state_hardening/test_state_transition_table.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_state_model.py, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition.schema.json, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition_table.json, test/api/agent_supervisor/efficiency_state_hardening/test_state_transition_table.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_state_model.py, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition.schema.json, ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition_table.json, test/api/agent_supervisor/efficiency_state_hardening/test_state_transition_table.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: descriptive/executable transition contract only; IntentRepository remains transactional mutation authority
- Acceptance conditions: Closed transition coverage, deterministic mapping, forbidden-edge, terminality, policy-time, unknown-outcome, and failed-validation tests pass.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_state_transition_table.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Closed transition coverage, deterministic mapping, forbidden-edge, terminality, policy-time, unknown-outcome, and failed-validation tests pass.
- Terminal non-success criteria: Unmapped production state/event, direct-write allowance, double terminal edge, retroactive policy effect, or failed-validation success path fails closed.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-041 Enforce transition authority

- Stable task ID: ASEH-041
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-machine
- Goal id: ASEH-G050
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G050
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Implement the candidate StateCommand/IntentRepository transition service with revision/CAS semantics and runtime warnings for bypass attempts, while leaving bootstrap-protected production owners unchanged until consolidation admission.
- Depends on: ASEH-040
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_transition_authority.py
- Outputs: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_transition_authority.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_transition_authority.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_transition_authority.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: sole Accelerate task-state mutation gateway; cannot bypass Quack owner or lower gates
- Acceptance conditions: The candidate service wraps the existing IntentRepository contract without direct table writes; CAS conflicts and compatibility bypasses fail closed; event and materialized revisions match; production cutover is deferred to ASEH-060/061 or an admitted current-tree plan delta.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_transition_authority.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: The candidate transition service passes the full authority contract without changing protected bootstrap/runtime files, and records the exact consolidation admission needed for production cutover.
- Terminal non-success criteria: Any direct task-table write, two writable authorities, silent legacy mutation, or missing revision is an unauthorized-mutation hard failure.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-042 Add lease fence and idempotency invariants

- Stable task ID: ASEH-042
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-machine
- Goal id: ASEH-G050
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G050
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Require one authoritative owner, current lease/fence, revision CAS, and an idempotency key for every effectful execution and completion.
- Depends on: ASEH-040, ASEH-041
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_lease_fence_idempotency.py
- Outputs: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_lease_fence_idempotency.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_lease_fence_idempotency.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/task_transition_service.py, test/api/agent_supervisor/efficiency_state_hardening/test_lease_fence_idempotency.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: authoritative execution fencing; no stale actor may effect or complete work
- Acceptance conditions: Race/property tests prove stale lease/fence completion is impossible, duplicate idempotent delivery executes once, and takeover advances fencing deterministically.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_lease_fence_idempotency.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Race/property tests prove stale lease/fence completion is impossible, duplicate idempotent delivery executes once, and takeover advances fencing deterministically.
- Terminal non-success criteria: Stale-fence completion, double effect, lost idempotency, owner ambiguity, or CAS bypass is a hard safety failure and quarantines affected tasks.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-043 Add unknown-outcome reconciliation

- Stable task ID: ASEH-043
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-machine
- Goal id: ASEH-G050
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G050
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Transition timeouts/connection loss with possible external effect to provider_outcome_unknown and reconciliation_pending; observe effect/receipt/idempotency evidence before retry, compensate, quarantine, or terminalize.
- Depends on: ASEH-041, ASEH-042
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/rescue/database_recovery.py, ipfs_accelerate_py/agent_supervisor/control/schemas/recovery_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unknown_outcome_reconciliation.py
- Outputs: ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/rescue/database_recovery.py, ipfs_accelerate_py/agent_supervisor/control/schemas/recovery_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unknown_outcome_reconciliation.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/rescue/database_recovery.py, ipfs_accelerate_py/agent_supervisor/control/schemas/recovery_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unknown_outcome_reconciliation.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/rescue/database_recovery.py, ipfs_accelerate_py/agent_supervisor/control/schemas/recovery_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_unknown_outcome_reconciliation.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: reconciliation may classify observed external effects but cannot erase them or blind-retry unknown outcomes
- Acceptance conditions: Unknown-outcome, late completion, duplicate delivery, compensation, non-idempotent effect, and unavailable-provider tests preserve truth and choose only admissible transitions.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_unknown_outcome_reconciliation.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Unknown-outcome, late completion, duplicate delivery, compensation, non-idempotent effect, and unavailable-provider tests preserve truth and choose only admissible transitions.
- Terminal non-success criteria: Blind retry, attempted-as-observed record, erased external effect, duplicate effect, or completion without admitted evidence is a hard failure.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-044 Add owner-loss and restart recovery

- Stable task ID: ASEH-044
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-machine
- Goal id: ASEH-G050
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G050
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Recover the same authoritative task/event/lease/fence/idempotency/reconciliation state after owner loss, expire or take over safely, and never require direct database edits or credential re-materialization.
- Depends on: ASEH-041, ASEH-042, ASEH-043
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py, test/api/agent_supervisor/efficiency_state_hardening/test_owner_restart_recovery.py
- Outputs: ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py, test/api/agent_supervisor/efficiency_state_hardening/test_owner_restart_recovery.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py, test/api/agent_supervisor/efficiency_state_hardening/test_owner_restart_recovery.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py, test/api/agent_supervisor/efficiency_state_hardening/test_owner_restart_recovery.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: existing Quack state owner and transition service only; no new owner/daemon/store
- Acceptance conditions: Crash-boundary and repeated restart tests reconstruct identical state/root, preserve unknown effects, advance fence on takeover, and resume authenticated operation.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_owner_restart_recovery.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Crash-boundary and repeated restart tests reconstruct identical state/root, preserve unknown effects, advance fence on takeover, and resume authenticated operation.
- Terminal non-success criteria: Divergent recovery, lost event/receipt, reused stale fence, new credential requirement, dual owner, or database repair need is a hard non-success.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-045 Add model property and crash tests

- Stable task ID: ASEH-045
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-machine
- Goal id: ASEH-G050
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G050
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Exercise transition model, invariants, crashes, restart, lease/fence races, duplicate delivery, owner loss, unknown outcomes, replay, idempotency, reconciliation, and event/materialized-state equivalence; consume Datasets formal vectors where installed.
- Depends on: ASEH-040, ASEH-041, ASEH-042, ASEH-043, ASEH-044
- Exact declared outputs: test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_properties.py, test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_crash_matrix.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/state_machine_qualification.json
- Outputs: test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_properties.py, test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_crash_matrix.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/state_machine_qualification.json
- Owned paths: test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_properties.py, test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_crash_matrix.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/state_machine_qualification.json
- Predicted files: test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_properties.py, test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_crash_matrix.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/state_machine_qualification.json
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: independent test/evidence producer only; tests cannot rewrite state or waive counterexamples
- Acceptance conditions: Bounded model/property corpus covers every state/event/invariant and all seeded violations are rejected with minimal reproducible counterexamples.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_properties.py test/api/agent_supervisor/efficiency_state_hardening/test_state_machine_crash_matrix.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Bounded model/property corpus covers every state/event/invariant and all seeded violations are rejected with minimal reproducible counterexamples.
- Terminal non-success criteria: Escaped critical seed, missing transition coverage, flaky/nonreproducible counterexample, hidden validation reduction, or direct repair invalidates qualification.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-050 Consolidate dependency and impact analysis

- Stable task ID: ASEH-050
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: planning-synthesis
- Goal id: ASEH-G060
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G060
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Run AST/symbol analysis before model analysis, calculate dependency and reverse-dependency cones, changed contracts/invariants, and semantic versus formatting/documentation changes with explicit uncertainty.
- Depends on: ASEH-000, ASEH-001
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py, test/api/agent_supervisor/efficiency_state_hardening/test_change_impact.py
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py, test/api/agent_supervisor/efficiency_state_hardening/test_change_impact.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py, test/api/agent_supervisor/efficiency_state_hardening/test_change_impact.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py, ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py, test/api/agent_supervisor/efficiency_state_hardening/test_change_impact.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: analysis supplies conservative evidence only; Datasets retains semantic identity and no change is authorized
- Acceptance conditions: Exact-file/symbol provenance, conservative dynamic uncertainty, reverse-cone, documentation-only, changed-contract, and no-model-before-AST tests pass.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_change_impact.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Exact-file/symbol provenance, conservative dynamic uncertainty, reverse-cone, documentation-only, changed-contract, and no-model-before-AST tests pass.
- Terminal non-success criteria: Under-approximated uncertain cone, heuristic authority claim, model-first analysis, or unbound source identity rejects completeness.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-051 Add assumption guarantee task contracts

- Stable task ID: ASEH-051
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: planning-synthesis
- Goal id: ASEH-G060
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G060
- Owning repository: ipfs_datasets_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 209dbe2765593fbc6efe8e9281c34f2e8f6e37a6
- Base repository tree: 95f54df34585d0b736706fd90c83f55954489ad9
- Base plan revision: ASEH-PLAN-R1
- Objective: Define semantic task-plan facts for objective revision, assumptions, guarantees, cone, interfaces, permitted paths, side effects, acceptance, tests/proofs, questions, budget, reserve, fallback, and recovery.
- Depends on: ASEH-040, ASEH-050
- Exact declared outputs: ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.py, ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.schema.json, tests/unit/logic/software_contracts/semantic_state/test_task_contract.py
- Outputs: ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.py, ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.schema.json, tests/unit/logic/software_contracts/semantic_state/test_task_contract.py
- Owned paths: ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.py, ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.schema.json, tests/unit/logic/software_contracts/semantic_state/test_task_contract.py
- Predicted files: ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.py, ipfs_datasets_py/logic/software_contracts/semantic_state/task_contract.schema.json, tests/unit/logic/software_contracts/semantic_state/test_task_contract.py
- Worktree isolation: canonical isolated task worktree for ipfs_datasets_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: Datasets defines contract meaning/identity only; Accelerate owns planning, budget enforcement, execution, and admission
- Acceptance conditions: Closed canonical contract, assumption/guarantee consistency, completeness witness, path scope, obligation, budget-reserve, and stale-identity tests pass.
- Validation: python3 -m pytest -q tests/unit/logic/software_contracts/semantic_state/test_task_contract.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Closed canonical contract, assumption/guarantee consistency, completeness witness, path scope, obligation, budget-reserve, and stale-identity tests pass.
- Terminal non-success criteria: Contradictory guarantees, open schema, missing revision, ambiguous mutation scope, consumed reserve, or semantic identity remint fails closed.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-052 Add affected-suffix replanning

- Stable task ID: ASEH-052
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: planning-synthesis
- Goal id: ASEH-G060
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G060
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Preserve accepted unaffected subplans, invalidate and regenerate only the affected suffix, deduplicate semantically equivalent tasks, coalesce identical proof/test/retrieval work, and reserve final validation.
- Depends on: ASEH-050, ASEH-051
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py, test/api/agent_supervisor/efficiency_state_hardening/test_affected_suffix_replanning.py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py, test/api/agent_supervisor/efficiency_state_hardening/test_affected_suffix_replanning.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py, test/api/agent_supervisor/efficiency_state_hardening/test_affected_suffix_replanning.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py, test/api/agent_supervisor/efficiency_state_hardening/test_affected_suffix_replanning.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: extends canonical replanner only; cannot self-accept plans or spend validation reserve
- Acceptance conditions: Plan delta identities, accepted-prefix preservation, dependency validity, duplicate coalescing, deterministic leaf acceptance, and reserve tests pass.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_affected_suffix_replanning.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Plan delta identities, accepted-prefix preservation, dependency validity, duplicate coalescing, deterministic leaf acceptance, and reserve tests pass.
- Terminal non-success criteria: Whole-plan regeneration without typed cause, changed accepted prefix, duplicate effect, cyclic delta, nondeterministic leaf, or reserve theft rejects the plan.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-053 Add counterexample and unsat-core context minimization

- Stable task ID: ASEH-053
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: planning-synthesis
- Goal id: ASEH-G060
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G060
- Owning repository: ipfs_datasets_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 209dbe2765593fbc6efe8e9281c34f2e8f6e37a6
- Base repository tree: 95f54df34585d0b736706fd90c83f55954489ad9
- Base plan revision: ASEH-PLAN-R1
- Objective: Expose minimal counterexample/unsat-core context, qualified Craig interpolation, and CEGAR refinement that expands only the affected abstraction and reports exact missing evidence.
- Depends on: ASEH-050, ASEH-051
- Exact declared outputs: ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/logic/software_contracts/semantic_state/refinement_context.py, tests/unit/logic/software_contracts/semantic_state/test_refinement_context.py
- Outputs: ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/logic/software_contracts/semantic_state/refinement_context.py, tests/unit/logic/software_contracts/semantic_state/test_refinement_context.py
- Owned paths: ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/logic/software_contracts/semantic_state/refinement_context.py, tests/unit/logic/software_contracts/semantic_state/test_refinement_context.py
- Predicted files: ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/logic/software_contracts/semantic_state/refinement_context.py, tests/unit/logic/software_contracts/semantic_state/test_refinement_context.py
- Worktree isolation: canonical isolated task worktree for ipfs_datasets_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: formal semantic evidence only; heuristics cannot claim proof and Accelerate admits reuse/execution
- Acceptance conditions: Reproducible counterexamples, minimal cores, interpolation qualification, CEGAR convergence/bounds, and incomplete-witness tests pass.
- Validation: python3 -m pytest -q tests/unit/logic/software_contracts/semantic_state/test_refinement_context.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Reproducible counterexamples, minimal cores, interpolation qualification, CEGAR convergence/bounds, and incomplete-witness tests pass.
- Terminal non-success criteria: Unqualified interpolation, heuristic-as-proof, discarded counterexample, broad unexplained refinement, or false completeness rejects the artifact.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-054 Add bounded deterministic synthesis

- Stable task ID: ASEH-054
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: planning-synthesis
- Goal id: ASEH-G060
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G060
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Implement allowlisted exact typed-error/schema/import/adapter/rename/vector/wrapper/format transforms with explicit pre/postconditions and deterministic rejection outside the grammar.
- Depends on: ASEH-020, ASEH-050, ASEH-051
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_synthesis.py
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_synthesis.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_synthesis.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py, ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py, test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_synthesis.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: bounded candidate patch generation only; no merge, completion, proof, or authority grant
- Acceptance conditions: Each allowlisted transform is deterministic, scope-bound, nonempty when applicable, idempotent, and validated against positive/negative fixtures.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_deterministic_synthesis.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Each allowlisted transform is deterministic, scope-bound, nonempty when applicable, idempotent, and validated against positive/negative fixtures.
- Terminal non-success criteria: Unknown transform, semantic ambiguity, out-of-scope file, empty patch, non-idempotence, secret/generated artifact risk, or unmet precondition rejects synthesis.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-055 Add PatchPlan validation and merge admission

- Stable task ID: ASEH-055
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: planning-synthesis
- Goal id: ASEH-G060
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G060
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Bind base tree, semantic intent, files/symbols, pre/postconditions, invariants, tests/proofs, scope, digest, secrets/generated checks, and current validation receipt before merge.
- Depends on: ASEH-041, ASEH-051, ASEH-052, ASEH-053, ASEH-054
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/merge/patch_plan.py, ipfs_accelerate_py/agent_supervisor/merge/patch_admission.py, ipfs_accelerate_py/agent_supervisor/merge/schemas/patch_plan.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_patch_plan_admission.py
- Outputs: ipfs_accelerate_py/agent_supervisor/merge/patch_plan.py, ipfs_accelerate_py/agent_supervisor/merge/patch_admission.py, ipfs_accelerate_py/agent_supervisor/merge/schemas/patch_plan.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_patch_plan_admission.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/merge/patch_plan.py, ipfs_accelerate_py/agent_supervisor/merge/patch_admission.py, ipfs_accelerate_py/agent_supervisor/merge/schemas/patch_plan.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_patch_plan_admission.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/merge/patch_plan.py, ipfs_accelerate_py/agent_supervisor/merge/patch_admission.py, ipfs_accelerate_py/agent_supervisor/merge/schemas/patch_plan.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_patch_plan_admission.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: merge admission only; a candidate/model cannot validate, terminalize, or promote itself
- Acceptance conditions: Closed PatchPlan round-trip, digest/tree/scope, nonempty, secret/generated, current-receipt, failed-validation, conflict, and stale-plan tests pass.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_patch_plan_admission.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Closed PatchPlan round-trip, digest/tree/scope, nonempty, secret/generated, current-receipt, failed-validation, conflict, and stale-plan tests pass.
- Terminal non-success criteria: Stale or missing receipt, failed validation, digest mismatch, scope escape, secret, empty patch, self-validation, or policy mutation denies merge.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-060 Inventory and retire duplicate writable authorities

- Stable task ID: ASEH-060
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: consolidation-migration
- Goal id: ASEH-G070
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G070
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Select one canonical owner for task/objective state, routing, context, reuse, repair, merge, promotion, recovery, and receipts; classify every duplicate as canonical, adapter, deprecated, fixture, or removed.
- Depends on: ASEH-024, ASEH-035, ASEH-045, ASEH-055
- Exact declared outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json, test/api/agent_supervisor/efficiency_state_hardening/test_single_authority.py
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json, test/api/agent_supervisor/efficiency_state_hardening/test_single_authority.py
- Owned paths: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json, test/api/agent_supervisor/efficiency_state_hardening/test_single_authority.py
- Predicted files: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json, test/api/agent_supervisor/efficiency_state_hardening/test_single_authority.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: classification and enforcement only; no public deletion or policy promotion
- Acceptance conditions: Every mutable fact has exactly one writable owner, all bypasses warn/fail, real-CID/capability paths are used, and migration dispositions are complete.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_single_authority.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Every mutable fact has exactly one writable owner, all bypasses warn/fail, real-CID/capability paths are used, and migration dispositions are complete.
- Terminal non-success criteria: Two writers, silent legacy fallback, pseudo-CID/mock capability contamination, unclassified duplicate, or unsupported deletion is a hard failure.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-061 Add compatibility migration adapters

- Stable task ID: ASEH-061
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: consolidation-migration
- Goal id: ASEH-G070
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G070
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Integrate the admitted transition and recovery behavior into the existing IntentRepository to typed-Quack-owner path, route supported legacy APIs through that sole authority with runtime warnings, and document replacement and rollback without creating an independent writer.
- Depends on: ASEH-045, ASEH-060
- Exact declared outputs: ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py, ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_STATE_HARDENING_MIGRATION.md, test/api/agent_supervisor/efficiency_state_hardening/test_compatibility_migration.py
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py, ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_STATE_HARDENING_MIGRATION.md, test/api/agent_supervisor/efficiency_state_hardening/test_compatibility_migration.py
- Owned paths: ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py, ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_STATE_HARDENING_MIGRATION.md, test/api/agent_supervisor/efficiency_state_hardening/test_compatibility_migration.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py, ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_STATE_HARDENING_MIGRATION.md, test/api/agent_supervisor/efficiency_state_hardening/test_compatibility_migration.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared integration outputs only in the isolated task worktree; merge is prohibited until the existing owner is paused and the canonical merge/recovery gates admit the patch; a plan delta may substitute a different exact integration path only when ASEH-060 proves it is the newer current-tree authority and cannot waive production integration
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: owner-paused staged maintenance only; all compatibility behavior delegates to the existing IntentRepository and typed Quack owner, cannot write independently, reduce validation, or delete public APIs, and must restart and reconcile the same durable authority before scheduling resumes
- Maintenance prerequisite: stop new claims, drain active mutating claims and leases, seal the current root and recovery evidence, pause the Quack owner, stage and validate the exact patch, merge atomically, restart with the same external credential handle, reconcile state and receipts, then reopen claims
- Acceptance conditions: Every production caller reaches the existing IntentRepository to typed-Quack-owner authority; supported behavior equivalence, warning, single-write, restart/reconciliation, migration, rollback, caller replacement, and fail-closed unsupported-path tests pass. A plan delta cannot waive production integration.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_compatibility_migration.py test/api/test_agent_supervisor_configured_typed_grant_handoff.py test/api/agent_supervisor/efficiency_state_hardening/test_owner_restart_recovery.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: After the owner-paused staged maintenance and authenticated restart, all production mutations traverse the existing IntentRepository and typed Quack owner, the materialized state and event/receipt history reconcile, and compatibility, rollback, and fail-closed tests pass.
- Terminal non-success criteria: A running-owner mutation, active claim or lease during cutover, dual write, disconnected wrapper, silent fallback, behavior loss, gate reduction, failed restart/reconciliation, missing rollback/replacement, or premature deletion rejects migration and keeps claims closed.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-062 Add current-head cross-repository qualification

- Stable task ID: ASEH-062
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: consolidation-migration
- Goal id: ASEH-G070
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G070
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Install exact current-head packages independently and validate canonical bytes/CIDs, unknown fields, bounds, stale identities, missing dependencies, authority boundaries, and no sibling source-tree test imports.
- Depends on: ASEH-031, ASEH-032, ASEH-033, ASEH-034, ASEH-061
- Exact declared outputs: test/api/agent_supervisor/efficiency_state_hardening/test_cross_repository_contracts.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/cross_repository_qualification.json
- Outputs: test/api/agent_supervisor/efficiency_state_hardening/test_cross_repository_contracts.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/cross_repository_qualification.json
- Owned paths: test/api/agent_supervisor/efficiency_state_hardening/test_cross_repository_contracts.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/cross_repository_qualification.json
- Predicted files: test/api/agent_supervisor/efficiency_state_hardening/test_cross_repository_contracts.py, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/cross_repository_qualification.json
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: independent contract qualification only; does not promote package or policy pointers
- Acceptance conditions: Fresh isolated installations pass all cross-repository vectors at exact commits/trees and prove each authority boundary and real CID equality.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_cross_repository_contracts.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Fresh isolated installations pass all cross-repository vectors at exact commits/trees and prove each authority boundary and real CID equality.
- Terminal non-success criteria: Editable sibling leakage, source-tree tests import, stale/missing dependency acceptance, CID inequality, mock capability, or dirty/unbound head invalidates qualification.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-070 Run hermetic development benchmark

- Stable task ID: ASEH-070
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: qualification
- Goal id: ASEH-G080
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G080
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Run at least 60 bounded fixtures across all three paired arms for rapid iteration, selected-test false negatives, safety seeds, routing, ContextPack, state, planning, and synthesis behavior.
- Depends on: ASEH-013, ASEH-024, ASEH-035, ASEH-045, ASEH-055, ASEH-062
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/hermetic.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/hermetic.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/results/hermetic.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/results/hermetic.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: hermetic evidence only; cannot be labeled live or authorize canary/promotion by itself
- Acceptance conditions: Emit an admitted hermetic qualification receipt that truthfully records exact pair count, controls, confidence intervals, quality/safety results, and escaped seeds; fewer than 60 pairs becomes insufficient_evidence and any escaped critical seed becomes safety_or_quality_failed rather than blocking the release decision.
- Validation: python3 benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py --cohort hermetic --minimum-tasks 60 --output benchmarks/agent_supervisor/efficiency_state_hardening/results/hermetic.json --qualification-output docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json --allow-honest-nonpromotion
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: A schema-valid receipt binds either a qualified 60-or-more-pair result or an exact insufficient_evidence/safety_or_quality_failed result, without representing hermetic evidence as live.
- Terminal non-success criteria: Unequal unreported controls, fabricated usage, concealed defects, malformed evidence, or a live/production claim makes the receipt inadmissible; an honestly recorded adverse result is not a task-execution failure.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-071 Run historical paired replay

- Stable task ID: ASEH-071
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: qualification
- Goal id: ASEH-G080
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G080
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Replay at least 20 sealed real tasks at exact historical trees through direct, current, and candidate arms with all original acceptance tests and outcome classes.
- Depends on: ASEH-014, ASEH-070
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/historical.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/historical.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/results/historical.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/results/historical.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R3_BOUNDED_REPOSITORY_MUTATION
- Authority requirement: isolated historical replay only; no mutation of historical branches or state and no live claim
- Acceptance conditions: Emit an admitted replay qualification receipt with exact-tree provenance, outcome coverage, and paired token/compute/cost/time/quality/safety statistics; fewer than 20 valid pairs or missing outcome coverage is recorded as insufficient_evidence rather than blocking the release decision.
- Validation: python3 benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py --cohort historical --minimum-tasks 20 --output benchmarks/agent_supervisor/efficiency_state_hardening/results/historical.json --qualification-output docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json --allow-honest-nonpromotion
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: A schema-valid receipt binds either a qualified 20-or-more-pair replay or an exact insufficient_evidence/safety_or_quality_failed replay result with every limitation explicit.
- Terminal non-success criteria: Altered acceptance presented as original, unsafe uncontained effect, aggregate-only or candidate-only evidence presented as paired, fabricated provenance, or concealed missing evidence invalidates the receipt; an honestly unavailable tree or missing class does not block ASEH-074.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-072 Run live shadow cohort

- Stable task ID: ASEH-072
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: qualification
- Goal id: ASEH-G080
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G080
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Run candidate in non-mutating shadow beside current behavior on at least 10 newly encountered real tasks with real provider usage, observed effects where applicable, and admitted validators.
- Depends on: ASEH-015, ASEH-071
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/live_shadow.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/live_shadow.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/results/live_shadow.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/results/live_shadow.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: live shadow cannot mutate, merge, promote, or self-label evidence; provider credentials remain external
- Acceptance conditions: Emit an admitted live-shadow qualification receipt with exact provenance, pair count, safety/quality observations, and measured usage/compute/cost availability; fewer than 10 pairs becomes insufficient_evidence and a gate failure becomes safety_or_quality_failed, both of which prohibit canary but remain usable by ASEH-074.
- Validation: python3 benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py --cohort live-shadow --minimum-tasks 10 --output benchmarks/agent_supervisor/efficiency_state_hardening/results/live_shadow.json --qualification-output docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json --allow-honest-nonpromotion
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: A schema-valid receipt binds either a qualified 10-or-more-pair live shadow or an exact insufficient_evidence/safety_or_quality_failed result; only the qualified status admits canary.
- Terminal non-success criteria: Simulation substituted for live evidence, attempted-as-observed effects, fabricated pairing, concealed missing validation, or an unreported hard-gate result invalidates the receipt; an honestly recorded shortfall or gate failure is terminal evidence for non-promotion.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-073 Run low-risk canary

- Stable task ID: ASEH-073
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: qualification
- Goal id: ASEH-G080
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G080
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: After shadow admission only, run low-risk candidate mutations in isolated worktrees with current validation, exact receipts, reversible merge handling, and independent safety observation.
- Depends on: ASEH-072
- Exact declared outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/canary.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json
- Outputs: benchmarks/agent_supervisor/efficiency_state_hardening/results/canary.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json
- Owned paths: benchmarks/agent_supervisor/efficiency_state_hardening/results/canary.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json
- Predicted files: benchmarks/agent_supervisor/efficiency_state_hardening/results/canary.json, docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: bounded low-risk mutation only; no protected policy promotion and high/value-laden decisions require operator
- Acceptance conditions: Emit an admitted canary qualification receipt: run mutations only after qualified shadow evidence, otherwise record not_admitted without mutation; executed canaries bind current tree/fence/validation, reversible scope, observed outcome, and every safety result.
- Validation: python3 benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py --cohort canary --require-shadow-receipt benchmarks/agent_supervisor/efficiency_state_hardening/results/live_shadow.json --output benchmarks/agent_supervisor/efficiency_state_hardening/results/canary.json --qualification-output docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json --allow-not-admitted
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: The receipt is either executed with admitted shadow prerequisite and complete current evidence, or not_admitted with zero candidate mutation and an exact prerequisite reason; an executed safety/quality failure remains truthful terminal evidence for ASEH-074.
- Terminal non-success criteria: Mutation without admitted shadow evidence, stale fence/tree used for an effect, concealed unknown outcome, scope escape, irreversible/high-risk action, fabricated observation, or missing receipt invalidates the task; safely stopping and recording not_admitted is successful evidence handling.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-074 Produce promotion or honest non-promotion receipt

- Stable task ID: ASEH-074
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: qualification
- Goal id: ASEH-G080
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G080
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Evaluate all paired populations, confidence/quality/safety evidence, thresholds, audit overhead, missing evidence, and emit only eligible-with-operator-authorization or one of three honest non-promotion dispositions.
- Depends on: ASEH-070, ASEH-071, ASEH-072, ASEH-073
- Exact declared outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json, ipfs_accelerate_py/agent_supervisor/control/schemas/promotion_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_promotion_decision.py
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json, ipfs_accelerate_py/agent_supervisor/control/schemas/promotion_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_promotion_decision.py
- Owned paths: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json, ipfs_accelerate_py/agent_supervisor/control/schemas/promotion_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_promotion_decision.py
- Predicted files: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json, ipfs_accelerate_py/agent_supervisor/control/schemas/promotion_decision.schema.json, test/api/agent_supervisor/efficiency_state_hardening/test_promotion_decision.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R4_SECURITY_OR_PROTOCOL_SENSITIVE
- Authority requirement: decision evaluation only; promotion pointer mutation remains separate operator-authorized CAS
- Acceptance conditions: Closed decision cites exact current evidence and returns the mandated disposition: unmeasured, safety_or_quality, efficiency, or eligible_operator_authorization_required.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_promotion_decision.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Closed decision cites exact current evidence and returns the mandated disposition: unmeasured, safety_or_quality, efficiency, or eligible_operator_authorization_required.
- Terminal non-success criteria: Missing measurement treated as zero, an absent live cohort not mapped to non_promoted_unmeasured, a lowered threshold, self-authorization, an ignored safety violation, or an unbound policy/tree rejects the receipt.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success

## ASEH-075 Publish final residual-gap report

- Stable task ID: ASEH-075
- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: qualification
- Goal id: ASEH-G080
- Parent goal ID: ASEH-G000
- Subgoal ID: ASEH-G080
- Owning repository: ipfs_accelerate_py
- Board namespace: agent-supervisor-efficiency-and-state-hardening-v1
- Base revision: 755f45475cc2d13dacd8b330036c1d597afeddde
- Base repository tree: 729da9f8293ecfa046a0136381a3d3808f9ed140
- Base plan revision: ASEH-PLAN-R1
- Objective: Publish exact commits changed and trees for each repository, board status, canonical architecture and deprecated paths, population sizes, baseline/current/candidate token and compute use, provider cost, model-call distribution and deterministic/small/medium/frontier/human route shares, test/proof reuse, retry/recovery rates, false-positive/false-negative, quality and safety results, disposition, limits, residual risks, and highest-return next work.
- Depends on: ASEH-074
- Exact declared outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_FINAL_REPORT.md, test/api/agent_supervisor/efficiency_state_hardening/test_release_report.py
- Outputs: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_FINAL_REPORT.md, test/api/agent_supervisor/efficiency_state_hardening/test_release_report.py
- Owned paths: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_FINAL_REPORT.md, test/api/agent_supervisor/efficiency_state_hardening/test_release_report.py
- Predicted files: docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json, docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_FINAL_REPORT.md, test/api/agent_supervisor/efficiency_state_hardening/test_release_report.py
- Worktree isolation: canonical isolated task worktree for ipfs_accelerate_py; exact pinned gitlink and current attempt tree recorded
- Write scope: exact declared outputs only; any additional path needs an admitted current-tree plan delta and supervisor re-admission
- External effect scope: local repository/worktree and declared validators only unless this task explicitly names a live cohort; no protected branch or policy-pointer mutation
- Risk class: R2_REVERSIBLE_LOCAL
- Authority requirement: reporting only; cannot infer readiness, create follow-on tasks, promote policy, or conceal missing evidence
- Acceptance conditions: Machine and human reports reconcile to admitted receipts, name every required metric or unavailable reason, preserve exact disposition, and keep deferred ideas non-executing.
- Validation: python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_release_report.py
- Proof requirements: admitted current-tree validation receipt plus negative-case evidence and exact output digest
- Required evidence: task identity, objective revision, repository commit/tree, policy identity, validator command/result, raw-log references, receipt CID, and truth-state availability
- Model-route ceiling: deterministic stages first; smallest adequate executor; typed unresolved question required before any model call
- Lease and fencing: current claim revision, lease, fence, and idempotency key required for effects and terminalization
- Terminal success criteria: Machine and human reports reconcile to admitted receipts, name every required metric or unavailable reason, preserve exact disposition, and keep deferred ideas non-executing.
- Terminal non-success criteria: Receipt mismatch, unsupported production claim, hidden limitation, absent required field, auto-created backlog task, or promotion assertion without operator CAS fails publication.
- Rollback: discard or revert only the scoped task-worktree patch through the canonical merge/recovery path; preserve observed effects and receipts
- Safety gates: all program hard gates remain zero; missing evidence never becomes zero or success
