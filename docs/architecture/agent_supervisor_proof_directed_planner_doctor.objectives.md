# Agent Supervisor Proof-Directed Planner and Doctor Objective Heap (PDR)

This is the durable, machine-ingestible goal hierarchy for the
proof-directed Planner/Doctor self-improvement program. The executable
projection is `agent_supervisor_proof_directed_planner_doctor.todo.md`
(`## PDR-` task prefix). The normative design is
`AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md`.

Audited baseline:
`f25e5719cb738a50fb96bac4bea3f66ebca9800b`.

## North star

Build a shared symbolic reasoning and mutation kernel so the supervisor can
compile prompts, steer directives, code mutations, Doctor findings, and
benchmark residuals into evidence-covered goals, subgoals, tasks, proofs,
minimal implementation context, safe parallel execution, independently
validated repairs, and bounded successor work.

## Program invariants

- Retrieval, GraphRAG, vectors, embeddings, history, model output, and
  synthetic fixtures are nomination or measurement inputs, never write,
  proof, promotion, completion, or semantic authority.
- The deterministic Doctor never calls an LLM, remote model provider, remote
  embedding provider, or network service.
- Planner/Doctor preview is read-only; apply/repair is separately authorized
  and bound to current roots, permit, lease, fence, expected effects,
  checkpoint, and rollback.
- Proof-cache hits re-derive assurance and never upgrade it.
- A caller-supplied boolean, mapping, receipt, expected outcome, candidate
  implementation, or generated test cannot certify itself.
- Automatic work cannot mutate this objective heap, the seed taskboard, the
  normative plan, scheduler/promotion policy, holdout manifest, or hidden
  oracle.
- Missing required capability, telemetry, impact closure, proof, security
  state, or current-tree evidence is unknown/blocked, never pass.
- Task completion is not objective completion; completion requires current,
  independently replayed source evidence.

## Goal tree

```text
PDR-G000  Proof-directed Planner and Doctor self-improvement loop
├── PDR-G010  Baseline, trust boundary, threat model, and benchmark anchors
├── PDR-G020  Canonical repository evidence, capability routing, and caches
├── PDR-G030  Proof-directed create/steer Planner
├── PDR-G040  Durable control-plane materialization and runtime adoption
├── PDR-G050  Production deterministic-Doctor composition and diagnosis
├── PDR-G060  Proof-gated repair, mutation, security, and fixed point
├── PDR-G070  Cryptographic lineage and optional ZKP
├── PDR-G080  Live paired benchmark and attributable telemetry
├── PDR-G090  Bounded unattended epochs, refill, and promotion
└── PDR-G100  Adversarial qualification, operations, and terminal release
```

## PDR-G000 Proof-directed Planner and Doctor self-improvement loop

- Status: active
- Parent:
- Depends on:
- Fib priority: 89
- Priority: P0
- Track: integration
- Bundle: agent-supervisor/proof-directed-planner-doctor/root
- Direct child goals: PDR-G010, PDR-G020, PDR-G030, PDR-G040, PDR-G050, PDR-G060, PDR-G070, PDR-G080, PDR-G090, PDR-G100
- Producing tasks: all tasks in the companion PDR taskboard
- Goal: Join prompt planning, formal planning, program evidence, deterministic diagnosis, proof-gated repair, parallel execution, live benchmarking, and bounded self-refill into one guarded feedback controller.
- Evidence: PDR-G010, PDR-G020, PDR-G030, PDR-G040, PDR-G050, PDR-G060, PDR-G070, PDR-G080, PDR-G090, PDR-G100
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md, docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md, docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md, config/agent_supervisor_proof_directed_planner_doctor_scheduler.json, docs/architecture/agent_supervisor/PROGRAMS.md
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_release.py -q
- Acceptance: Every child goal has current evidence or a typed blocker; live Planner and Doctor paths use the shared evidence/obligation/plan kernel; the held-out paired gate passes with zero safety-floor violations; automatic mode remains disabled until a separate current-tree promotion receipt exists.
- Conflict policy: Own PDR artifacts and semantic modules only; do not reopen or rewrite ASI, CBP, RPR, LPR, formal-verification, or foreign taskboards.

## PDR-G010 Baseline, trust boundary, threat model, and benchmark anchors

- Status: completed
- Parent: PDR-G000
- Depends on:
- Fib priority: 89
- Priority: P0
- Track: foundation
- Bundle: agent-supervisor/proof-directed-planner-doctor/foundation
- Producing tasks: PDR-001, PDR-002, PDR-003
- Goal: Freeze the exact mainline capability/gap inventory, non-compensable authority rules, immutable mutation boundary, benchmark population, metrics, oracles, and promotion policy before changing runtime behavior.
- Evidence: PDR-001, PDR-002, PDR-003
- Outputs: docs/architecture/agent_supervisor_planner_doctor_baseline.md, docs/architecture/agent_supervisor_planner_doctor_threat_model.md, config/agent_supervisor_planner_doctor_benchmark.json
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_foundation.py -q
- Acceptance: The inventory distinguishes shipped components from live wiring gaps; the threat model forbids self-certification and judge mutation; the benchmark manifest fixes paired denominators, protected holdouts, budgets, strata, concurrency sweep, safety floors, and stop conditions.
- Conflict policy: Foundation artifacts are operator-owned and protected after seal.

## PDR-G020 Canonical repository evidence, capability routing, and caches

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G010
- Fib priority: 89
- Priority: P0
- Track: evidence
- Bundle: agent-supervisor/proof-directed-planner-doctor/evidence
- Producing tasks: PDR-010, PDR-011, PDR-012, PDR-013, PDR-014, PDR-015
- Goal: Build one content-addressed, body-free view over the existing repository/AST/program/contract/value/evidence graphs; route required reasoning through certified capabilities; reuse exact caches; and make cold discovery provider-free.
- Evidence: PDR-010, PDR-011, PDR-012, PDR-013, PDR-014, PDR-015
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/repository_reasoning_snapshot.py, ipfs_accelerate_py/agent_supervisor/analysis/planning_analysis_factory.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_strategy_registry.py, ipfs_accelerate_py/agent_supervisor/analysis/planning_evidence_bundle.py, ipfs_accelerate_py/agent_supervisor/analysis/reasoning_cache.py
- Validation: python -m pytest test/api/test_agent_supervisor_repository_reasoning_snapshot.py test/api/test_agent_supervisor_planning_analysis_factory.py test/api/test_agent_supervisor_analysis_strategy_registry.py test/api/test_agent_supervisor_planning_evidence_bundle.py test/api/test_agent_supervisor_reasoning_cache.py test/api/test_agent_supervisor_doctor_cold_import.py -q
- Acceptance: Exact superproject/gitlink/dirty/task/policy/toolchain roots round-trip; incompatible Doctor records have checked bridges; required unavailable lanes abstain; optional lanes create debt; retrieval remains nomination-only; cache invalidation follows dependencies; importing discovery/service surfaces loads no network/model clients.
- Conflict policy: Reuse existing graph, cache, proof, and registry contracts; do not introduce competing truth stores.

## PDR-G030 Proof-directed create/steer Planner

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G020
- Fib priority: 89
- Priority: P0
- Track: planning
- Bundle: agent-supervisor/proof-directed-planner-doctor/planner
- Producing tasks: PDR-020, PDR-021, PDR-022, PDR-023, PDR-024, PDR-025, PDR-026, PDR-027
- Goal: Turn create/steer directives and exact evidence into append-only plan revisions, AND/OR obligations, deterministic-first candidate portfolios, bounded critique/repair, minimal residual context, independently admitted formal plans, and resource-feasible parallel execution plans.
- Evidence: PDR-020, PDR-021, PDR-022, PDR-023, PDR-024, PDR-025, PDR-026, PDR-027
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/plan_revision_contracts.py, ipfs_accelerate_py/agent_supervisor/planning/plan_analysis_query_planner.py, ipfs_accelerate_py/agent_supervisor/planning/obligation_graph_compiler.py, ipfs_accelerate_py/agent_supervisor/planning/symbolic_candidate_planner.py, ipfs_accelerate_py/agent_supervisor/planning/plan_critic.py, ipfs_accelerate_py/agent_supervisor/planning/parallel_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/planning/plan_admission_service.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_revision_contracts.py test/api/test_agent_supervisor_plan_analysis_query_planner.py test/api/test_agent_supervisor_obligation_graph_compiler.py test/api/test_agent_supervisor_symbolic_candidate_planner.py test/api/test_agent_supervisor_plan_critic.py test/api/test_agent_supervisor_parallel_plan_compiler.py test/api/test_agent_supervisor_plan_admission_service.py -q
- Acceptance: Default planning uses the analysis registry and an independent admission request; `candidate_count` drives a bounded portfolio; deterministic fallback is codebase-derived; create/steer fail stale; LLM calls are residual-only and budgeted; fake parallelism is rejected.
- Conflict policy: Extend planning and prompt contracts semantically; preserve package DAG and proposal-versus-authority boundaries.

## PDR-G040 Durable control-plane materialization and runtime adoption

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G030
- Fib priority: 55
- Priority: P0
- Track: control-runtime
- Bundle: agent-supervisor/proof-directed-planner-doctor/control-runtime
- Producing tasks: PDR-028, PDR-030, PDR-031, PDR-032, PDR-033
- Goal: Expose create/steer preview and apply through one Python/control/CLI/MCP service, persist append-only plan revisions across restart, prove Markdown/DuckDB parity, and require the compiled execution plan at dispatch.
- Evidence: PDR-030, PDR-031, PDR-032, PDR-033
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py, ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py, ipfs_accelerate_py/agent_supervisor/control/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control/control_plane.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_supervisor_service.py test/api/test_agent_supervisor_plan_revision_store.py test/api/test_agent_supervisor_plan_control_conformance.py test/api/test_agent_supervisor_parallel_plan_runtime.py -q
- Acceptance: Preview has no effects; apply is CAS/fence/permit bound and restartable; transports and projections are canonical equivalents; claimed history is immutable; runtime rejects mixed revisions, false lane labels, and stale capacity/lease state.
- Conflict policy: Control contracts remain transport-neutral; task sources remain projections, not authority.

## PDR-G050 Production deterministic-Doctor composition and diagnosis

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G020
- Fib priority: 89
- Priority: P0
- Track: doctor-runtime
- Bundle: agent-supervisor/proof-directed-planner-doctor/doctor
- Producing tasks: PDR-040, PDR-041, PDR-042, PDR-043
- Goal: Assemble the existing deterministic-Doctor stages behind a lazy production runtime, harden proof inputs, enumerate real checkouts, localize causal/contract mismatches, and translate findings into the shared obligation graph.
- Evidence: PDR-040, PDR-041, PDR-042, PDR-043
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py, ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py, ipfs_accelerate_py/agent_supervisor/analysis/doctor_causal_localization.py, ipfs_accelerate_py/agent_supervisor/planning/diagnosis_obligation_adapter.py
- Validation: python -m pytest test/api/test_agent_supervisor_deterministic_doctor_runtime.py test/api/test_agent_supervisor_deterministic_doctor_proof_authority.py test/api/test_agent_supervisor_doctor_causal_localization.py test/api/test_agent_supervisor_diagnosis_obligation_adapter.py -q
- Acceptance: `inspect --checkout-root` produces current evidence without supplied JSON; all stages are capability-probed and lazy; caller flags/prebuilt mappings cannot mint proof authority; findings bind complete/open frontiers and become typed obligations without schema ambiguity.
- Conflict policy: Keep `control` as the service contract owner; compose higher packages from `runtime` without introducing import cycles.

## PDR-G060 Proof-gated repair, mutation, security, and fixed point

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G030, PDR-G050
- Fib priority: 89
- Priority: P0
- Track: repair
- Bundle: agent-supervisor/proof-directed-planner-doctor/repair
- Producing tasks: PDR-050, PDR-051, PDR-052, PDR-053, PDR-054, PDR-055
- Goal: Generate deterministic or tightly bounded residual repairs, validate multiple candidates independently, change real bytes only in isolated overlays, transact complete impact closures, enforce IntentIR/SecurityIR, renew a live fixed point, roll back exactly, and return residuals to the Planner/refill path.
- Evidence: PDR-050, PDR-051, PDR-052, PDR-053, PDR-054, PDR-055
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/repair_operator_registry.py, ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py, ipfs_accelerate_py/agent_supervisor/runtime/doctor_worktree_adapter.py, ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_live_fixed_point.py, ipfs_accelerate_py/agent_supervisor/validation/repair_candidate_portfolio.py, ipfs_accelerate_py/agent_supervisor/objectives/doctor_plan_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_repair_operator_registry.py test/api/test_agent_supervisor_program_repair_synthesis.py test/api/test_agent_supervisor_doctor_worktree_adapter.py test/api/test_agent_supervisor_deterministic_doctor_live_fixed_point.py test/api/test_agent_supervisor_repair_candidate_portfolio.py test/api/test_agent_supervisor_doctor_plan_refill.py -q
- Acceptance: Deterministic repairs precede model calls; hybrid model output is proposal-only; actual before/after bytes and roots are reread; all-caller/SCC closure, proof, security, validation, lease/ref-CAS, rollback, and fixed-point gates hold; second-order defects iterate or abstain.
- Conflict policy: Validators stay pure and never simulate evidence production; live runners own effects under normal mutation authority.

## PDR-G070 Cryptographic lineage and optional ZKP

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G020, PDR-G060
- Fib priority: 21
- Priority: P1
- Track: attestation
- Bundle: agent-supervisor/proof-directed-planner-doctor/attestation
- Producing tasks: PDR-060
- Goal: Bind Planner, Doctor, cache, mutation, benchmark, and promotion receipt lineage with CIDs/Merkle proofs/signatures and add ZKP only for an approved privacy or fixed-computation claim.
- Evidence: PDR-060
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/planner_doctor_attestation.py, docs/architecture/agent_supervisor_planner_doctor_zkp_threat_model.md
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_attestation.py -q
- Acceptance: Invalid preimages and cross-run replay fail; private witnesses do not leak; simulated ZKP never becomes `ATTESTED`; attestation never substitutes for program semantics, inventory completeness, or translator soundness.
- Conflict policy: Reuse proof attestation and `program_analysis_zkp`; do not create another assurance lattice or proof cache.

## PDR-G080 Live paired benchmark and attributable telemetry

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G010, PDR-G040, PDR-G060
- Fib priority: 89
- Priority: P0
- Track: benchmark
- Bundle: agent-supervisor/proof-directed-planner-doctor/benchmark
- Producing tasks: PDR-070, PDR-071, PDR-072
- Goal: Replace synthetic/passive qualification with hermetic live Planner/Doctor executions, process-tree/provider telemetry, independent hidden quality oracles, cache and concurrency strata, and adversarial/ablation populations.
- Evidence: PDR-070, PDR-071, PDR-072
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_live_benchmark.py, ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py, ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_quality_oracle.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_live_benchmark.py test/api/test_agent_supervisor_benchmark_telemetry.py test/api/test_agent_supervisor_planner_doctor_quality_oracle.py -q
- Acceptance: Outcomes come from real services and independent oracles, not fixture `expected` fields; synthetic suites are labeled conformance-only; missing telemetry is unavailable; paired roots/providers/budgets/denominators match; holdouts are outside candidate scope.
- Conflict policy: Extend existing benchmark/metric contracts with live source receipts; do not narrow their closed fixture populations.

## PDR-G090 Bounded unattended epochs, refill, and promotion

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G040, PDR-G060, PDR-G080
- Fib priority: 55
- Priority: P0
- Track: self-improvement
- Bundle: agent-supervisor/proof-directed-planner-doctor/self-improvement
- Producing tasks: PDR-080, PDR-081, PDR-082
- Goal: Invoke live bounded epochs from the supervisor lifecycle, compile benchmark residuals into deduplicated successor work in a separate derived source, and compare baseline/challenger with non-compensable quality/safety gates, canary, rollback, and anti-gaming controls.
- Evidence: PDR-080, PDR-081, PDR-082
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py, ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py, ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_rollout.py
- Validation: python -m pytest test/api/test_agent_supervisor_planner_doctor_epoch.py test/api/test_agent_supervisor_planner_doctor_refill.py test/api/test_agent_supervisor_planner_doctor_rollout.py -q
- Acceptance: At most 8 goals/24 tasks per epoch; exact replay is no-op; derived work cannot mutate anchors; live receipts drive shadow/canary decisions; safety/quality cannot be compensated by speed/tokens; current-tree recheck and exact rollback are mandatory; automatic stays off pending independent qualification.
- Conflict policy: Reuse self-improvement v2 epoch/rollout/refill contracts and write generated work outside the protected seed board.

## PDR-G100 Adversarial qualification, operations, and terminal release

- Status: active
- Parent: PDR-G000
- Depends on: PDR-G070, PDR-G090
- Fib priority: 55
- Priority: P0
- Track: release
- Bundle: agent-supervisor/proof-directed-planner-doctor/release
- Producing tasks: PDR-090, PDR-091, PDR-092
- Goal: Prove transport/projection/restart equivalence, withstand adversarial/tamper/crash/resource/parallelism cases, provide protected operator launch and kill-switch controls, and issue a current-tree terminal receipt.
- Evidence: PDR-090, PDR-091, PDR-092
- Outputs: test/integration/test_agent_supervisor_planner_doctor_e2e.py, scripts/ops/agent_supervisor/proof_directed_planner_doctor.py, docs/guides/PROOF_DIRECTED_PLANNER_DOCTOR_GUIDE.md, ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_release.py
- Validation: python -m pytest test/integration/test_agent_supervisor_planner_doctor_e2e.py test/api/test_agent_supervisor_planner_doctor_release.py -q
- Acceptance: Cold/warm/delta/restart and 1/2/4/configured-maximum lane runs have zero safety-floor violations; kill switch and rollback are exact; no skip/synthetic evidence qualifies; final receipt independently reloads source evidence and proves every child goal without relying on task counts.
- Conflict policy: Release code may read protected anchors but never rewrite them; automatic promotion needs separate operator policy.
