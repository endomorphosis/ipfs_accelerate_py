# Agent Supervisor Control-Plane Planner/Doctor V2 Objective Heap (CPD)

This is the durable goal hierarchy for the prompt-to-control-plane and
Planner/Doctor integration delta. The executable projection is
`agent_supervisor_control_plane_planner_doctor_v2.todo.md` with task prefix
`## CPD-`. The normative design is
`AGENT_SUPERVISOR_CONTROL_PLANE_PLANNER_DOCTOR_V2_PLAN.md`.

Audited baseline:
`66c6fb4d46d9472e2f5bba9a4cb3e6f78d858aa5`.

## North star

Compile one prompt, through Python, CLI, or MCP, into the same independently
admitted, content-addressed goals/subgoals/tasks, scheduler policy, task-source
projections, and safe launch specification; then use the shared proof-directed
Planner and Doctor to improve the codebase in bounded, reversible, benchmarked
epochs with minimal LLM context.

## Program invariants

- CPD imports and revalidates PDR/ASI capabilities; it does not duplicate or
  trust historical completion labels.
- Preview is read-only. Apply, start, mutation, merge, benchmark activation,
  canary, and promotion have separate current-root-bound authority.
- Retrieval and model outputs nominate; independent checked evidence admits.
- The deterministic Doctor has no LLM, remote model/embedding, or network
  permission. Hybrid LLM use is residual-only and proposal-only.
- Generated work cannot mutate the seed plan, objective heap, taskboard,
  scheduler/promotion policy, authority policy, holdout, or hidden oracle.
- Grok 4.5 is primary; Codex `gpt-5.6-terra` medium is eligible only after a
  verified durable Grok hard-quota-exhaustion receipt.
- Missing required evidence, capabilities, telemetry, translation coverage,
  or impact closure is unknown/blocked, never pass.
- ZKP proves only an explicitly fixed cryptographic claim, never general
  semantic correctness.
- Task completion is not objective completion; goal closure requires current,
  independently replayed evidence.

## Goal tree

```text
CPD-G000  Prompt-to-control-plane Planner/Doctor self-improvement
├── CPD-G010  Current-tree imports, authority, and gap baseline
├── CPD-G020  Prompt artifacts and canonical control-plane contracts
├── CPD-G030  Proof-directed control-plane and launch compiler
├── CPD-G040  Python/CLI/MCP materialization and real lifecycle start
├── CPD-G050  Planner/Doctor mutation reasoning and safe correction
├── CPD-G060  Cryptographic lineage and narrow optional ZKP
├── CPD-G070  Live repository-self benchmark and quality oracle
├── CPD-G080  Bounded unattended epochs, refill, and rollout
└── CPD-G090  E2E/chaos qualification, operations, and release
```

## CPD-G000 Prompt-to-control-plane Planner/Doctor self-improvement

- Status: active
- Parent:
- Depends on:
- Fib priority: 89
- Priority: P0
- Track: integration
- Bundle: agent-supervisor/control-plane-planner-doctor/root
- Direct child goals: CPD-G010, CPD-G020, CPD-G030, CPD-G040, CPD-G050, CPD-G060, CPD-G070, CPD-G080, CPD-G090
- Producing tasks: all CPD tasks
- Goal: Join raw-prompt resolution, symbolic control-plane compilation, safe launch, deterministic diagnosis, proof-gated repair, live benchmarking, and bounded self-refill into one guarded controller.
- Evidence: every direct child goal plus the terminal current-tree release receipt
- Outputs: docs/architecture/AGENT_SUPERVISOR_CONTROL_PLANE_PLANNER_DOCTOR_V2_PLAN.md, docs/architecture/agent_supervisor_control_plane_planner_doctor_v2.objectives.md, docs/architecture/agent_supervisor_control_plane_planner_doctor_v2.todo.md, config/agent_supervisor_control_plane_planner_doctor_v2_scheduler.json
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_planner_doctor_release.py -q
- Acceptance: The same prompt artifact yields the same admitted bundle through Python/CLI/MCP; the launch spec starts a real supervised process under separate authority; Planner/Doctor gates cover every mutation; the held-out paired gate has zero safety-floor violations and quality non-inferiority; automatic rollout remains off until a separate promotion receipt is verified.
- Conflict policy: Own CPD integration contracts and adapters only; import PDR/ASI analyzers, proofs, stores, and lifecycle components through checked interfaces.

## CPD-G010 Current-tree imports, authority, and gap baseline

- Status: active
- Parent: CPD-G000
- Depends on:
- Fib priority: 89
- Priority: P0
- Track: foundation
- Bundle: agent-supervisor/control-plane-planner-doctor/foundation
- Producing tasks: CPD-001, CPD-002
- Goal: Revalidate exact PDR/ASI capabilities and failures on the current forest, and bind CPD to operator-owned authority, threat, benchmark, holdout, and provider policies without creating competing policies.
- Evidence: current artifact CIDs, behavior conformance receipts, raw-prompt sparse-path reproduction, authority-policy import receipt, benchmark-policy import receipt
- Outputs: docs/architecture/agent_supervisor_control_plane_planner_doctor_baseline.md, test/api/test_agent_supervisor_control_plane_bootstrap_baseline.py, config/agent_supervisor_control_plane_policy_imports.json
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_bootstrap_baseline.py test/api/test_agent_supervisor_control_plane_policy_imports.py -q
- Acceptance: Shipped versus live-wired capabilities are distinguished; the sparse raw-prompt result and inert start path are reproduced; every reused PDR/ASI policy or receipt is CID/current-tree checked; no candidate can change authority, holdout, oracle, provider-fallback, or promotion rules.
- Conflict policy: PDR/ASI and operator policy artifacts are read-only inputs.

## CPD-G020 Prompt artifacts and canonical control-plane contracts

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G010
- Fib priority: 89
- Priority: P0
- Track: contracts
- Bundle: agent-supervisor/control-plane-planner-doctor/contracts
- Producing tasks: CPD-010, CPD-011, CPD-012
- Goal: Resolve prompt bodies through authorized content-addressed artifacts and define one closed control-plane bundle, launch spec, revision, and receipt family over a current repository/capability snapshot.
- Evidence: contract round trips, unknown-field/bounds/path/secret/replay failures, current-root snapshot import
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/prompt_artifact_resolver.py, ipfs_accelerate_py/agent_supervisor/planning/control_plane_bundle_contracts.py, ipfs_accelerate_py/agent_supervisor/analysis/control_plane_snapshot_factory.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_artifact_resolver.py test/api/test_agent_supervisor_control_plane_bundle_contracts.py test/api/test_agent_supervisor_control_plane_snapshot_factory.py -q
- Acceptance: Prompt bodies stay outside receipts; semantic records have stable CIDs; the bundle includes goal/task/obligation/parallel/projection/scheduler/launch identities; launch argv are arrays and environment contains names/handles only; stale/tampered/cross-repository records fail closed.
- Conflict policy: Extend or adapt existing PromptWorkflow, PlanRevision, repository snapshot, lifecycle, and parallel-plan records; do not create another truth store.

## CPD-G030 Proof-directed control-plane and launch compiler

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G020
- Fib priority: 89
- Priority: P0
- Track: planner
- Bundle: agent-supervisor/control-plane-planner-doctor/compiler
- Producing tasks: CPD-020, CPD-021, CPD-022, CPD-023, CPD-024
- Goal: Compile normalized intent and exact evidence into bounded goals/subgoals/tasks, proof obligations, minimal context, a feasible parallel schedule, and a parser-validated launch spec, then admit the entire bundle independently.
- Evidence: deterministic and hybrid candidate receipts, counterexamples, evidence coverage, obligation closure, resource/conflict proof, provider-route proof, independent admission
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/prompt_intent_compiler.py, ipfs_accelerate_py/agent_supervisor/planning/control_plane_task_compiler.py, ipfs_accelerate_py/agent_supervisor/planning/control_plane_candidate_planner.py, ipfs_accelerate_py/agent_supervisor/planning/supervisor_launch_compiler.py, ipfs_accelerate_py/agent_supervisor/planning/control_plane_admission.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_intent_compiler.py test/api/test_agent_supervisor_control_plane_task_compiler.py test/api/test_agent_supervisor_control_plane_candidate_planner.py test/api/test_agent_supervisor_launch_compiler.py test/api/test_agent_supervisor_control_plane_admission.py -q
- Acceptance: Every requirement maps to acceptance/proof/evidence; fake parallelism and unsafe launch values fail; deterministic candidates precede bounded residual LLM calls; unresolved required obligations abstain; independent admission reconstructs evidence and cannot consume caller self-attestation.
- Conflict policy: Compose completed PDR query, obligation, candidate, critic, parallel, and admission services through adapters.

## CPD-G040 Python/CLI/MCP materialization and real lifecycle start

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G030
- Fib priority: 89
- Priority: P0
- Track: control-runtime
- Bundle: agent-supervisor/control-plane-planner-doctor/control-runtime
- Producing tasks: CPD-030, CPD-031, CPD-032, CPD-033, CPD-034, CPD-035
- Goal: Make one production bootstrap service resolve raw prompts, compile/admit bundles, atomically project task sources, persist resumable receipts, expose exact Python/CLI/MCP parity, and launch a real supervised process from the admitted launch spec.
- Evidence: transport-equivalent bundle CIDs, projection parity, restart replay, authorization and expected-effects receipts, real PID/process-tree and health receipts
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/control_plane_bootstrap_service.py, ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_materializer.py, ipfs_accelerate_py/agent_supervisor/control/control_plane.py, ipfs_accelerate_py/agent_supervisor/control/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, ipfs_accelerate_py/agent_supervisor/control/control_plane_lifecycle.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_bootstrap_service.py test/api/test_agent_supervisor_control_plane_materializer.py test/api/test_agent_supervisor_control_plane_public_api.py test/api/test_agent_supervisor_control_plane_cli.py test/api/test_agent_supervisor_control_plane_mcp.py test/api/test_agent_supervisor_control_plane_lifecycle.py -q
- Acceptance: Raw prompt no longer succeeds sparsely; preview has no effects; apply/start are separately permitted, fenced, idempotent and restartable; Markdown/DuckDB task CIDs match; `--start` is effective or removed; MCP cannot resolve arbitrary paths; a real child process is owned, healthy, stoppable, restartable, and rollback-safe.
- Conflict policy: Transports remain thin; one domain service owns semantics; use the existing control catalog and lifecycle runner.

## CPD-G050 Planner/Doctor mutation reasoning and safe correction

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G020, CPD-G030
- Fib priority: 89
- Priority: P0
- Track: doctor-repair
- Bundle: agent-supervisor/control-plane-planner-doctor/doctor-repair
- Producing tasks: CPD-013, CPD-014, CPD-015, CPD-016, CPD-017, CPD-040, CPD-041, CPD-042, CPD-043, CPD-044, CPD-045, CPD-046
- Goal: Correct root and capability authority, compose the real semantic graph/retrieval/prover/security/cache pipeline, prove that the formal-method portfolio is actually invoked, close the Doctor's deferred stages, plan every mutation, diagnose causal contract mismatches, synthesize deterministic-first repairs, and accept only reversible current-tree fixed points satisfying independent intent and code-security streams.
- Evidence: capability execution receipts, impact/context coverage, diagnosis obligations, repair portfolios, byte/root changes, rollback, post-change proof/security/validation closure
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/planning_analysis_factory.py, ipfs_accelerate_py/agent_supervisor/analysis/semantic_graph_pipeline.py, ipfs_accelerate_py/agent_supervisor/analysis/control_plane_evidence_retrieval.py, ipfs_accelerate_py/agent_supervisor/proof/control_plane_formal_portfolio.py, ipfs_accelerate_py/agent_supervisor/validation/independent_intent_code_security.py, ipfs_accelerate_py/agent_supervisor/analysis/control_plane_reasoning_cache.py, ipfs_accelerate_py/agent_supervisor/analysis/control_plane_reasoning_kernel.py, ipfs_accelerate_py/agent_supervisor/planning/mutation_context_planner.py, ipfs_accelerate_py/agent_supervisor/runtime/control_plane_doctor_bridge.py, ipfs_accelerate_py/agent_supervisor/runtime/control_plane_doctor_loop.py, ipfs_accelerate_py/agent_supervisor/planning/control_plane_repair_portfolio.py, ipfs_accelerate_py/agent_supervisor/validation/control_plane_mutation_fixed_point.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_root_capability_authority.py test/api/test_agent_supervisor_semantic_graph_pipeline.py test/api/test_agent_supervisor_control_plane_evidence_retrieval.py test/api/test_agent_supervisor_control_plane_formal_portfolio.py test/api/test_agent_supervisor_independent_intent_code_security.py test/api/test_agent_supervisor_control_plane_reasoning_cache.py test/api/test_agent_supervisor_control_plane_reasoning_kernel.py test/api/test_agent_supervisor_mutation_context_planner.py test/api/test_agent_supervisor_control_plane_doctor_bridge.py test/api/test_agent_supervisor_control_plane_doctor_loop.py test/api/test_agent_supervisor_control_plane_repair_portfolio.py test/api/test_agent_supervisor_control_plane_mutation_fixed_point.py -q
- Acceptance: Live roots are independently recomputed; only executed providers satisfy assurance; AST-to-semantic-graph, true BM25/hybrid retrieval, executable formal portfolio, independent IntentIR/CodeIR, and cross-stage caches are production wired; required methods execute or yield typed abstention; context is minimal and evidence-complete; Doctor findings become shared obligations and no production stage merely defers; deterministic repairs precede LLM proposals; isolated mutations cover the impacted SCC/callers/consumers and reach a live SecurityIR/IntentIR fixed point or roll back.
- Conflict policy: Do not reimplement analyzers or proof engines; register checked adapters and enforce their invocation at shared boundaries.

## CPD-G060 Cryptographic lineage and narrow optional ZKP

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G020, CPD-G040, CPD-G050
- Fib priority: 21
- Priority: P1
- Track: attestation
- Bundle: agent-supervisor/control-plane-planner-doctor/attestation
- Producing tasks: CPD-050, CPD-051
- Goal: Bind prompt, bundle, launch, mutation, benchmark, and promotion lineage with CIDs/Merkle proofs/signatures, and permit ZKP only for an operator-approved fixed claim.
- Evidence: invalid-preimage/replay failures, signature/lineage verification, approved ZK threat model and real-verifier receipt if enabled
- Outputs: ipfs_accelerate_py/agent_supervisor/proof/control_plane_attestation.py, docs/architecture/agent_supervisor_control_plane_zkp_threat_model.md
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_attestation.py test/api/test_agent_supervisor_control_plane_zkp.py -q
- Acceptance: Cross-run/root/bundle replay fails; simulated ZK never becomes attested; private witnesses do not leak; attestations never substitute for semantics, coverage, or translator soundness.
- Conflict policy: Reuse multiformats/multihash/CID, proof attestation, cache, and program-analysis ZKP interfaces.

## CPD-G070 Live repository-self benchmark and quality oracle

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G040, CPD-G050
- Fib priority: 89
- Priority: P0
- Track: benchmark
- Bundle: agent-supervisor/control-plane-planner-doctor/benchmark
- Producing tasks: CPD-060, CPD-061, CPD-062
- Goal: Run paired live baseline/challenger prompt and mutation workloads on provenance-partitioned repository cases, attribute clock/tokens/resources, and score solution quality with an independent protected oracle.
- Evidence: live service/process receipts, paired roots/budgets/providers/hardware, complete telemetry, sealed holdout oracle results, ablation and concurrency strata
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/control_plane_live_benchmark.py, ipfs_accelerate_py/agent_supervisor/runtime/control_plane_benchmark_telemetry.py, ipfs_accelerate_py/agent_supervisor/validation/control_plane_quality_oracle.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_live_benchmark.py test/api/test_agent_supervisor_control_plane_benchmark_telemetry.py test/api/test_agent_supervisor_control_plane_quality_oracle.py -q
- Acceptance: Outcomes come from real services and independently observed effects; missing telemetry is unavailable; candidate processes cannot read oracle bodies; safety/quality are non-compensable; synthetic or skipped checks cannot promote.
- Conflict policy: Import the sealed PDR corpus/partition/policy and extend its cases without changing denominators or hidden membership.

## CPD-G080 Bounded unattended epochs, refill, and rollout

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G050, CPD-G070
- Fib priority: 55
- Priority: P0
- Track: self-improvement
- Bundle: agent-supervisor/control-plane-planner-doctor/self-improvement
- Producing tasks: CPD-070, CPD-071, CPD-072
- Goal: Execute bounded baseline/propose/shadow/evaluate/retain-or-reject/canary/recheck/promote-or-rollback epochs, compile residuals into a separate derived source, and promote only quality-safe Pareto improvements.
- Evidence: epoch FSM receipts, residual/refill CIDs, budget and cooldown enforcement, anti-gaming/non-inferiority results, canary/rollback receipt
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/control_plane_epoch.py, ipfs_accelerate_py/agent_supervisor/objectives/control_plane_residual_refill.py, ipfs_accelerate_py/agent_supervisor/self_improvement/control_plane_rollout.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_epoch.py test/api/test_agent_supervisor_control_plane_residual_refill.py test/api/test_agent_supervisor_control_plane_rollout.py -q
- Acceptance: Seed inputs stay immutable; caps and stop conditions are enforced; unchanged failures cool down; no scalar reward offsets a safety/quality regression; automatic mode remains disabled until a current-tree operator promotion receipt exists.
- Conflict policy: Derived goals/tasks go only to the configured DuckDB/CAS source.

## CPD-G090 E2E/chaos qualification, operations, and release

- Status: active
- Parent: CPD-G000
- Depends on: CPD-G040, CPD-G060, CPD-G070, CPD-G080
- Fib priority: 89
- Priority: P0
- Track: release
- Bundle: agent-supervisor/control-plane-planner-doctor/release
- Producing tasks: CPD-080, CPD-081, CPD-082
- Goal: Prove the entire raw-prompt-to-real-daemon-to-repair-to-benchmark loop under restart and adversarial faults, ship operator controls and kill switch, and issue an independently replayed terminal release receipt.
- Evidence: Python/CLI/MCP E2E equivalence, chaos/adversarial results, operations drill, release replay and operator seal
- Outputs: test/e2e/test_agent_supervisor_control_plane_loop.py, scripts/ops/agent_supervisor/control_plane_planner_doctor.py, docs/guides/CONTROL_PLANE_PLANNER_DOCTOR_GUIDE.md, ipfs_accelerate_py/agent_supervisor/validation/control_plane_planner_doctor_release.py
- Validation: python -m pytest test/e2e/test_agent_supervisor_control_plane_loop.py test/api/test_agent_supervisor_control_plane_operations.py test/api/test_agent_supervisor_control_plane_planner_doctor_release.py -q
- Acceptance: Stale/tampered prompt, partial projection/start, process death, provider/solver loss, cache corruption, merge conflict, rollback, transport restart, GPU pressure, and kill switch fail safely; operations are reproducible; terminal release verifies every current artifact and independent manual authority.
- Conflict policy: Release cannot weaken or rewrite constituent evidence; it aggregates independently replayed receipts.
