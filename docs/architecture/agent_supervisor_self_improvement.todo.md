# Agent Supervisor Self-Improvement Task Board

This board implements the
[generation-1 self-improvement plan](AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md)
and the
[generation-2 integration plan](AGENT_SUPERVISOR_SELF_IMPROVEMENT_V2_PLAN.md).
The durable source of intent is
[agent_supervisor_self_improvement.objectives.md](agent_supervisor_self_improvement.objectives.md).
Task status is an execution projection; it does not replace objective or
completion evidence.

The implementation daemon must use task prefix `## ASI-`. Tasks may run in
parallel only when their dependencies are complete and the conflict/resource
scheduler admits their predicted files and resource class. New provider,
planner, and refill behavior defaults to shadow mode.

## ASI-001 Establish end-to-end supervisor efficiency baselines

- Status: completed
- Completion: manual
- Priority: P0
- Track: measurement
- Depends on:
- Goal id: ASI-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/measurement
- Parallel lane: efficiency-metrics
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Conflict policy: Keep the receipt standalone; defer package exports and runtime wiring.
- Acceptance: Define a versioned, deterministic efficiency receipt that joins stage latency, queue delay, input/output/reused token counts, cache outcomes, retries, validation and proof cost, changed scope, artifacts, and terminal acceptance. Add fixture baselines for cold, warm, failed, repaired, parallel-independent, and conflicting tasks. Store digests and bounded references instead of prompts, source bodies, decoded model output, or nested artifact graphs. Report cost per accepted task and evidence gain per thousand input tokens, and add strict bounds, round-trip, aggregation, and invalid-state tests.

## ASI-002 Define shared context, control, and operation contracts

- Status: completed
- Completion: manual
- Priority: P0
- Track: contracts
- Depends on:
- Goal id: ASI-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, test/api/test_agent_supervisor_control_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_contracts.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/contracts
- Parallel lane: control-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, test/api/test_agent_supervisor_control_contracts.py
- Conflict policy: Add standalone contracts only; do not edit package exports, CLI, MCP registries, or daemon orchestration.
- Acceptance: Add immutable versioned contracts for context budgets/capsules, operation requests/results, capability reports, lifecycle commands, dry-run previews, idempotency, authorization decisions, and typed errors. Bind requests to repository/state roots, objective and tree identities, policy, caller, bounds, and expected effects. Make read, proposal, and mutation authority explicit. Enforce canonical serialization and count/byte/depth limits, and reject unknown operations, path escapes, missing idempotency on mutations, and result claims outside the operation authority.

## ASI-003 Integrate the existing analysis cache, AST index, and retrieval layer

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: ASI-001, ASI-002
- Goal id: ASI-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/audit_scanner.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/objective_daemon.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_objective_evidence_policy.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_analysis_cache.py test/api/test_agent_supervisor_analysis_ast_index.py test/api/test_agent_supervisor_analysis_retrieval.py test/api/test_agent_supervisor_objective_evidence_policy.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/analysis
- Parallel lane: analysis-integration
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/audit_scanner.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/objective_daemon.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_objective_evidence_policy.py
- Conflict policy: This is the sole integration lane for the first analysis tranche; reuse existing analysis schemas instead of creating replacements.
- Acceptance: Compose `analysis_contracts`, `analysis_cache`, `analysis_ast_index`, and `analysis_retrieval` behind one bounded pipeline used by objective scanning and low-backlog analysis. Reuse unchanged blobs and exact-key receipts, expose explicit hit/miss/invalidation and backend-health reasons, and return ranked compact evidence references with truncation metadata. Add an evidence-source policy that excludes objective, plan, task-board, generated-discovery, and other proposal-tier prose from satisfying code, test, proof, benchmark, or runtime requirements; semantic retrieval may nominate evidence but only an exact typed receipt from an allowed source can satisfy an opaque requirement ID. Keep failed, partial, stale, or inconclusive analysis out of completion evidence. Reject partial AST substring matches and add reward-hacking, restart, corruption, optional-backend degradation, and equivalent cold/warm result tests.

## ASI-004 Add a capability-negotiated ipfs_datasets_py analysis provider

- Status: completed
- Completion: manual
- Priority: P0
- Track: datasets-offload
- Depends on: ASI-002
- Goal id: ASI-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py
- Validation: python -m pytest test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/datasets-offload
- Parallel lane: datasets-provider
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py
- Conflict policy: Keep the optional adapter standalone and lazy; do not modify sibling repositories or broaden Hammer proof authority.
- Acceptance: Define a lazy optional provider for bounded GraphRAG retrieval, dataset/provenance queries, premise selection, legal/logic analysis candidates, and related-request batching. Negotiate operations, schemas, versions, bounds, cancellation, and health before dispatch. Requests contain identities, allowlisted operation IDs, compact queries, and artifact references; responses contain bounded evidence references, provenance, truncation, resource use, and non-authority outcomes. Unavailable or incompatible capabilities must degrade to typed local fallback without eager imports, arbitrary execution, or copying large source/model/graph payloads.

## ASI-005 Build a token-budgeted evidence context compiler

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: ASI-003
- Goal id: ASI-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_context_compiler.py
- Validation: python -m pytest test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_task_proposal_router.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/context
- Parallel lane: context-compiler
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_context_compiler.py
- Conflict policy: Own prompt/context integration in this lane; preserve existing provider routing and defer retries to ASI-006.
- Acceptance: Compile immutable goal/policy/task core, ranked selected evidence, and on-demand artifact references under stage-specific input and output reserves. Prefer the effective provider tokenizer and record calibrated-estimator error when unavailable. Replace fixed character slices and ad hoc context concatenation in planning and implementation paths with deterministic capsules. Guarantee that required authority/scope/acceptance fields cannot be truncated, optional evidence is ranked with inclusion/exclusion reasons, and raw prompts, decoded output, full AST bodies, or recursive graphs do not enter receipts.

## ASI-006 Add progressive disclosure and delta retry contexts

- Status: completed
- Completion: manual
- Priority: P1
- Track: token-efficiency
- Depends on: ASI-005
- Goal id: ASI-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_context_delta.py test/api/test_agent_supervisor_formal_replanner.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/context
- Parallel lane: context-delta
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_context_delta.py
- Conflict policy: Run after ASI-005 because both tasks own context assembly.
- Acceptance: Add content-addressed on-demand expansion and retry capsules containing the prior decision identity, new failure/counterexample evidence, changed files/symbols, and unresolved requirements instead of replaying the original prompt. Identical failures must reuse the diagnostic receipt and back off or escalate. Test exact reconstruction, changed-tree invalidation, missing-reference expansion, bounded repair rounds, cancellation, and at least 35 percent lower median retry input tokens on paired fixtures without reducing required evidence coverage.

## ASI-007 Coordinate analysis, context, plan, proof, and validation caches

- Status: completed
- Completion: manual
- Priority: P0
- Track: caching
- Depends on: ASI-003, ASI-004
- Goal id: ASI-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_cache_coordinator.py test/api/test_agent_supervisor_analysis_cache.py test/api/test_agent_supervisor_formal_verification_cache.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/cache
- Parallel lane: cache-coordinator
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_cache_coordinator.py
- Conflict policy: Reuse existing cache implementations and add namespace coordination; do not collapse untrusted drafts and authoritative receipts into one namespace.
- Acceptance: Add common namespace metadata, semantic key construction, cross-process single-flight, quota/GC policy, lookup metrics, and bounded artifact references across analysis, context, planning, proof, validation, and merge classifications. Preserve namespace-specific schemas and authority. Negative and inconclusive records require TTLs and can never satisfy completion. Add exact-key reuse, every semantic invalidation dimension, concurrent miss collapse, corruption recovery, poisoned-entry rejection, bounded persistence, and zero stale authoritative-hit tests.

## ASI-008 Add cost- and evidence-aware adaptive planning

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: ASI-001, ASI-003, ASI-004, ASI-005
- Goal id: ASI-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, test/api/test_agent_supervisor_adaptive_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_plan_evaluator.py test/api/test_agent_supervisor_formal_plan_validator.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/planning
- Parallel lane: adaptive-planner
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, test/api/test_agent_supervisor_adaptive_planner.py
- Conflict policy: This lane owns plan candidate orchestration; preserve formal compiler/validator authority.
- Acceptance: Produce a deterministic baseline and optional bounded LLM, Leanstral, and `ipfs_datasets_py` candidates from the same frozen goal/context. Evaluate acceptance/evidence coverage, assumptions, semantics, dependency validity, critical path, conflict risk, validation/proof feasibility, novelty, and expected token/runtime/resource cost. Hard authority, scope, safety, and proof failures are non-compensable. Persist selected and rejected reasons, deterministic tie-breaking, fallback, and paired quality/cost metrics; test adversarial high-confidence invalid plans and provider unavailability.

## ASI-009 Make goal refinement responsive to typed runtime evidence

- Status: completed
- Completion: manual
- Priority: P0
- Track: goal-refinement
- Depends on: ASI-008, ASI-010
- Goal id: ASI-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_goal_refiner.py test/api/test_agent_supervisor_goal_refinement_verification.py test/api/test_agent_supervisor_goal_generation.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/goals
- Parallel lane: adaptive-goals
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Conflict policy: Own objective refinement policy in this lane; keep root mutation behind existing transactional admission.
- Acceptance: Add goal-quality and goal-debt records covering outcome, scope, assumptions, non-goals, acceptance, evidence producers, validation, freshness, resource envelope, unsupported semantics, and breadth. Trigger bounded replan/refinement from fresh counterexamples, stale evidence, repeated validation signatures, unavailable capability, interface change, conflict, or infeasible resources. Suppress unchanged failure churn with backoff. Freeze root and assumptions, independently verify child sufficiency, and add restart/idempotency tests.

## ASI-010 Enforce a strict implementation proposal and patch validation envelope

- Status: completed
- Completion: manual
- Priority: P0
- Track: output-validation
- Depends on: ASI-002
- Goal id: ASI-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_proposal_validation.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/validation
- Parallel lane: proposal-validation
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_proposal_validation.py
- Conflict policy: This lane owns pre-execution and patch-envelope checks; later validation stages consume its receipt.
- Acceptance: Require versioned structured proposals with exact files, operations, rationale references, validation plan, risks, and authority claims. Validate output size/depth, canonical task/tree/context binding, allowed paths, symlink/submodule boundaries, secret/binary/large-file policy, patch parse, and non-empty semantic change before expensive tests. Reject arbitrary command injection, test deletion or weakening, out-of-scope edits, forged completion/proof claims, and stale proposal replay. Persist compact failure reason codes suitable for bounded repair.

## ASI-011 Build impact-selected fail-fast validation DAGs

- Status: completed
- Completion: manual
- Priority: P0
- Track: output-validation
- Depends on: ASI-001, ASI-010
- Goal id: ASI-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_validation_dag.py
- Validation: python -m pytest test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_validation_scheduler.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/validation
- Parallel lane: validation-dag
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_validation_dag.py
- Conflict policy: Extend the existing validation scheduler; do not introduce a second subprocess scheduler.
- Acceptance: Derive mandatory syntax, type, interface, unit, integration, contract, and runtime checks from changed AST symbols, dependencies, task acceptance, and repository policy. Execute independent checks in parallel under the existing resource budget while preserving fail-fast dependencies and complete receipts. Cache only exact tree/command/environment results. Report selection reasons, skipped reasons, time to first useful failure, and uncovered impact. Seed defects outside direct file paths to prove dependency-aware selection catches them.

## ASI-012 Bind semantic, legal/logic, and proof validation to changed code

- Status: completed
- Completion: manual
- Priority: P1
- Track: output-validation
- Depends on: ASI-004, ASI-010, ASI-011
- Goal id: ASI-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_logic_provider.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_semantic_validation_pipeline.py test/api/test_agent_supervisor_code_proof_scopes.py test/api/test_agent_supervisor_formal_plan_conformance.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/semantic-validation
- Parallel lane: semantic-validation
- Resource class: cpu-proof-solver
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_logic_provider.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Conflict policy: Reuse Hammer, multi-prover, and kernel contracts; candidate solvers must not gain completion authority.
- Acceptance: Derive semantic and proof obligations from the accepted plan plus changed AST/interface/effect scope. Route supported legal/logic and premise-selection work through the existing `ipfs_datasets_py` Hammer boundary, use independent authoritative reconstruction when policy requires it, and preserve explicit unsupported/timeout results. Bind every receipt to goal, plan, tree, assumptions, toolchain, scope, and policy. Add wrong-theorem, stale-proof, candidate-as-proof, omitted-effect, and post-merge invalidation tests.

## ASI-013 Improve task sizing, quality, and semantic deduplication

- Status: completed
- Completion: manual
- Priority: P0
- Track: task-generation
- Depends on: ASI-008, ASI-009, ASI-010
- Goal id: ASI-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_task_quality.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_quality.py test/api/test_agent_supervisor_objective_graph.py test/api/test_agent_supervisor_backlog_refinery.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/task-generation
- Parallel lane: task-quality
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_task_quality.py
- Conflict policy: This lane owns generated-task admission and sizing; preserve canonical task identity compatibility.
- Acceptance: Score candidate tasks for acceptance coverage, coherent effects, predicted path/symbol breadth, context and validation cost, dependencies, conflicts, resources, and historical duplicate/failure similarity. Split over-broad candidates and coalesce tiny candidates only when they share goal, context, outputs, validation, and merge fate. Require canonical semantic identity, preconditions/effects, evidence subset, resource/token class, and rejection reasons. Separate the canonical task-ID prefix from Markdown heading rendering, normalize legacy heading-style input once at the boundary, and never generate doubled headings such as `## ## ASI-`. Test stable generation, parseable monotonic IDs, no duplicate refill, dependency preservation, and bounded open-work pressure.

## ASI-014 Optimize bundles for context reuse, conflicts, and critical path

- Status: completed
- Completion: manual
- Priority: P1
- Track: bundling
- Depends on: ASI-001, ASI-013
- Goal id: ASI-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, ipfs_accelerate_py/agent_supervisor/bundle_supervisor.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Validation: python -m pytest test/api/test_agent_supervisor_bundle_optimizer.py test/api/test_agent_supervisor_bundle_plan_cache.py test/api/test_agent_supervisor_conflict_graph.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/bundling
- Parallel lane: bundle-optimizer
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, ipfs_accelerate_py/agent_supervisor/bundle_supervisor.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Conflict policy: This lane owns bundle planning; do not edit resource admission policy handled by ASI-015.
- Acceptance: Build bundles from dependency depth, shared goal/evidence/context, file and AST conflicts, validation reuse, resource class, provider batchability, and merge locality. Explicitly model packet aggregates and exact covered siblings. Preserve independent critical-path width, serialize conflicting edits, and avoid lexical-only grouping. Compare model calls per accepted work item, context reuse, critical path, merge conflict rate, and bundle completion against the current planner.

## ASI-015 Make resource admission adaptive across supervisor stages

- Status: completed
- Completion: manual
- Priority: P0
- Track: parallelism
- Depends on: ASI-001, ASI-014
- Goal id: ASI-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/scheduler_metrics.py, ipfs_accelerate_py/agent_supervisor/bundle_supervisor.py, test/api/test_agent_supervisor_adaptive_resources.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_resources.py test/api/test_agent_supervisor_resource_scheduler.py test/api/test_agent_supervisor_scheduler_metrics.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/runtime
- Parallel lane: adaptive-resources
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/scheduler_metrics.py, ipfs_accelerate_py/agent_supervisor/bundle_supervisor.py, test/api/test_agent_supervisor_adaptive_resources.py
- Conflict policy: Extend the existing scheduler and metrics contracts; do not add an independent worker-count controller.
- Acceptance: Model deterministic analysis, inference, proof, validation, Git/merge, and persistence pools with explicit per-stage requirements. Adapt effective slots from CPU, RAM, GPU memory, provider capacity, disk pressure, queue depth, merge age, and active leases. Add hysteresis, fairness, critical-path priority, cancellation, and observable backpressure. Demonstrate no over-admission, no starvation, deterministic recovery after resource loss, and at least twice single-lane throughput on independent fixtures without duplicate execution.

## ASI-016 Add shared inference batching and single-flight provider work

- Status: completed
- Completion: manual
- Priority: P1
- Track: parallelism
- Depends on: ASI-004, ASI-007, ASI-015
- Goal id: ASI-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/leanstral_goal_development.py, test/api/test_agent_supervisor_provider_batch_scheduler.py
- Validation: python -m pytest test/api/test_agent_supervisor_provider_batch_scheduler.py test/api/test_agent_supervisor_leanstral_goal_development.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/runtime
- Parallel lane: provider-batching
- Resource class: llm-proof-draft
- Predicted files: ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/leanstral_goal_development.py, test/api/test_agent_supervisor_provider_batch_scheduler.py
- Conflict policy: Centralize compatible provider dispatch; preserve per-request context, cancellation, and receipt identity.
- Acceptance: Share model service capacity and collapse identical in-flight planning/analysis requests. Batch only compatible route, model, operation, context-limit, policy, and generation settings, while retaining independent budgets, timeouts, cancellation, provenance, and results. Enforce GPU/provider admission before loading weights, expose queue and batch metrics, and degrade to unbatched or deterministic fallback. Test fairness, partial cancellation, one failed batch member, provider limits, and lower duplicated model-load/inference cost.

## ASI-017 Parallelize validation and merge flow without weakening gates

- Status: completed
- Completion: manual
- Priority: P1
- Track: parallelism
- Depends on: ASI-011, ASI-014, ASI-015
- Goal id: ASI-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, ipfs_accelerate_py/agent_supervisor/merge_queue.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Validation: python -m pytest test/api/test_agent_supervisor_parallel_acceptance_flow.py test/api/test_agent_supervisor_validation_scheduler.py test/api/test_agent_supervisor_merge_queue.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/runtime
- Parallel lane: acceptance-throughput
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, ipfs_accelerate_py/agent_supervisor/merge_queue.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Conflict policy: Parallelize independent checks and merge preflights, but serialize target-branch mutation through the existing merge train.
- Acceptance: Run independent validation DAG nodes and merge-tree preflights concurrently, reuse exact validation receipts, and queue target-branch mutations in deterministic order with fencing. Revalidate affected dependents after each accepted merge and cancel stale work. Bound merge debt and worktree disk use. Test independent throughput, conflict serialization, stale-base repair, failed-validation quarantine, restart recovery, and proof that parallel completion cannot bypass post-merge gates.

## ASI-018 Add a shared Python supervisor control service

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: ASI-002
- Goal id: ASI-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/control_plane.py, test/api/test_agent_supervisor_control_plane.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/control
- Parallel lane: python-control
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_plane.py, test/api/test_agent_supervisor_control_plane.py
- Conflict policy: Add the service without package-root exports or CLI/MCP registration; those are separate dependent tasks.
- Acceptance: Implement a typed `SupervisorControlService` plus read-oriented client facade over capabilities, status, health, metrics, goals, tasks, bundles, lanes, events, receipts, cache inspection, preview, refine, reconcile, refill, plan, lifecycle, retry, cancel, quarantine, and validation replay. Use direct package APIs rather than shell strings. Apply repository/state allowlists, authorization, dry-run, idempotency, leases/fencing, bounded queries, stable errors, and audit receipts consistently.

## ASI-019 Add an ipfs-accelerate agent CLI group

- Status: completed
- Completion: manual
- Priority: P1
- Track: control
- Depends on: ASI-018
- Goal id: ASI-G070
- Outputs: ipfs_accelerate_py/cli.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, test/test_unified_cli_agent_supervisor.py
- Validation: python -m pytest test/test_unified_cli_agent_supervisor.py test/test_unified_cli_integration.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/control
- Parallel lane: cli-control
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/cli.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, test/test_unified_cli_agent_supervisor.py
- Conflict policy: Register one product CLI group backed exclusively by ASI-018 contracts; preserve standalone scripts as compatible low-level entry points.
- Acceptance: Add `ipfs-accelerate agent` commands for capabilities, status, health, metrics, goals, tasks, bundles, events, plan, preview, refine, reconcile, refill, start, pause, resume, drain, stop, retry, cancel, quarantine, cache, and receipts. Support stable JSON output, meaningful exit codes, explicit paths, dry-run for mutations, idempotency keys, and bounded watch/stream output. Add parity tests against direct service calls and reject ambiguous or unsafe defaults.

## ASI-020 Add policy-controlled agent-supervisor MCP tools

- Status: completed
- Completion: manual
- Priority: P1
- Track: control
- Depends on: ASI-018
- Goal id: ASI-G070
- Outputs: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/__init__.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, ipfs_accelerate_py/mcp_server/server.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/mcp_server/test_agent_supervisor_tools.py test/mcp_server/test_server.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/control
- Parallel lane: mcp-control
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/__init__.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, ipfs_accelerate_py/mcp_server/server.py, test/mcp_server/test_agent_supervisor_tools.py
- Conflict policy: Add one lazy MCP category and reuse ASI-018; do not shell out to CLI or duplicate lifecycle logic.
- Acceptance: Register bounded read tools for capabilities/status/health/goals/tasks/bundles/events/metrics/receipts and policy-gated mutation tools for preview/refine/refill/lifecycle/retry/cancel/quarantine. Require authorization, repository allowlists, idempotency, dry-run/preview, lease/fencing checks, bounded pagination, redaction, and audit receipts for mutations. Ensure tool discovery does not initialize optional providers or start a supervisor, and test Python/CLI/MCP result-schema parity.

## ASI-021 Unify lifecycle, health, events, and idempotent control

- Status: completed
- Completion: manual
- Priority: P1
- Track: control
- Depends on: ASI-019, ASI-020
- Goal id: ASI-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/supervisor_watchdog.py, ipfs_accelerate_py/agent_supervisor/runtime_temporal_monitor.py, test/api/test_agent_supervisor_control_lifecycle.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_lifecycle.py test/api/test_agent_supervisor_supervisor_watchdog.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/control
- Parallel lane: lifecycle-control
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/supervisor_watchdog.py, ipfs_accelerate_py/agent_supervisor/runtime_temporal_monitor.py, test/api/test_agent_supervisor_control_lifecycle.py
- Conflict policy: Reconcile all control surfaces through the shared state machine; retain current wrapper compatibility.
- Acceptance: Define consistent stopped, starting, healthy, degraded, paused, draining, blocked, stopping, and failed states plus legal transitions. Make repeated commands idempotent, recover interrupted transitions, and expose heartbeat, phase, active leases, refill state, backpressure, and terminal reason through one status schema. Append bounded events for every accepted/rejected mutation. Test concurrent controllers, stale PID/state, restart, pause versus drain, fenced stop, unauthorized mutation, and event replay.

## ASI-022 Implement benchmark-driven bounded self-refill

- Status: completed
- Completion: manual
- Priority: P0
- Track: self-refill
- Depends on: ASI-007, ASI-009, ASI-013, ASI-021
- Goal id: ASI-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py test/api/test_agent_supervisor_goal_generation.py test/api/test_agent_supervisor_backlog_refinery.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/refill
- Parallel lane: self-refill
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Conflict policy: Use existing objective proposal/admission and refill contracts; do not let benchmark code mutate goals directly.
- Acceptance: After a drained board, reconcile fresh goal evidence and run a content-addressed self-improvement epoch over efficiency, planning, validation, cache, throughput, control, and safety metrics. Bind every opaque requirement ID to a fresh typed receipt whose producer kind, repository tree, policy, command/toolchain, scope, result, and artifact digest satisfy the goal's evidence-source policy; textual occurrence or embedding similarity is proposal evidence only. Convert only measured regressions, uncovered criteria, stale evidence, persistent bottlenecks, or unsupported capabilities into bounded candidate successor goals. Validate goal quality/refinement, deduplicate against all lifecycle states and cooldown records, and materialize admitted goals transactionally. Identical epochs must be idempotent; healthy exhaustion records quorum and waits for a changed tree/objective/policy/capability, stale evidence, regression, operator revision, or scheduled window.

## ASI-023 Build the paired end-to-end self-improvement rollout gate

- Status: completed
- Completion: manual
- Priority: P1
- Track: rollout
- Depends on: ASI-006, ASI-007, ASI-008, ASI-012, ASI-016, ASI-017, ASI-022
- Goal id: ASI-G090
- Outputs: test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/rollout
- Parallel lane: rollout-gate
- Resource class: cpu-large
- Predicted files: test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Conflict policy: This is the integration gate; consume completed lane APIs and avoid redesigning them.
- Acceptance: Compare baseline and candidate behavior on identical cold/warm, broad-goal, contradictory, malformed-output, stale-cache, provider-unavailable, independent-parallel, conflicting-parallel, failed-validation, restart, and drained-refill fixtures. Require zero false completions and authority violations, no stale authoritative hits, bounded artifacts, stable restart, at least 35 percent lower median input tokens, at least 70 percent repeated-fixture cache reuse, and at least twice independent-lane throughput without quality or merge-conflict regression. Keep new behavior in shadow when any non-negotiable or paired gate fails.

## ASI-024 Publish stable exports, migration guidance, and operating profiles

- Status: completed
- Completion: manual
- Priority: P2
- Track: rollout
- Depends on: ASI-023
- Goal id: ASI-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md, docs/INDEX.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_control_plane.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Board namespace: agent-supervisor-self-improvement-v1
- Bundle: agent-supervisor/self-improvement/rollout
- Parallel lane: public-integration
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md, docs/INDEX.md
- Conflict policy: Perform shared exports and documentation only after every implementation and rollout gate is stable.
- Acceptance: Export the reviewed control service, contracts, and capability checks without eagerly importing optional providers. Document Python, CLI, and MCP parity; context/cache/resource profiles; shadow/assist/automatic rollout; objective and task-board commands; migration from standalone scripts; authorization; metrics; failure recovery; and self-refill epochs. Include a production profile and a smaller deterministic smoke profile, while retaining explicit capability discovery and conservative defaults.

## ASI-025 Close objective gap: Prove 208290439421789408250562066350459701853 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G010
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Changed paths: 
- AST symbols: 208290439421789408250562066350459701853
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G091
- Canonical task key: task/v1/55225f76bf99b809da8482b4f168d8868646b92b188c0f5e2e5376f7f9fe434f
- Canonical task CID: baguqeerakurf65v7tg4atwueqk2pc2gyq2denojldcga6xrokn3pp6p6inhq
- Missing evidence: 208290439421789408250562066350459701853
- Embedding query: 208290439421789408250562066350459701853
- AST query: 208290439421789408250562066350459701853
- Surplus group: objective/ASI-G091
- Merge key: cf9cce6558586133
- Merge family: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G091, ASI-G092
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: d82158879868ec2c
- Acceptance: Objective scan filed this gap for ASI-G091. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-025-objective-gap-18bc981d5df3.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (208290439421789408250562066350459701853), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb; implement a complete, cohesive change that fully advances the packet goals (ASI-G091, ASI-G092) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-026 Close objective gap: Prove 306437607356117177048620815571362227127 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G010
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Changed paths: 
- AST symbols: 306437607356117177048620815571362227127
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G092
- Canonical task key: task/v1/823c23b2c70a8f2dc2a692344940e0232c765d6b749eb743eb31fe900bf18306
- Canonical task CID: baguqeeraqi6chmwhbkhs3qvgsi2esqhaemwhmxllosploq7lgh7jac7rqmda
- Missing evidence: 306437607356117177048620815571362227127
- Embedding query: 306437607356117177048620815571362227127
- AST query: 306437607356117177048620815571362227127
- Surplus group: objective/ASI-G092
- Merge key: c7b0bcba615a78ff
- Merge family: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Goal packet role: packet_member
- Goal packet goals: ASI-G091, ASI-G092
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 91fe5602d494da2b
- Acceptance: Objective scan filed this gap for ASI-G092. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-026-objective-gap-538195101c95.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (306437607356117177048620815571362227127), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb; implement a complete, cohesive change that fully advances the packet goals (ASI-G091, ASI-G092) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-027 Close objective gap: Prove 189057730455837902155591890661235220962 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G020
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Changed paths: 
- AST symbols: 189057730455837902155591890661235220962
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G094
- Canonical task key: task/v1/bb202c8a80bd61881b6cef2059535dd908298e73a473ff8ff9ef657403d61d65
- Canonical task CID: baguqeeraxmqczcuaxvqyqg3m54qfsu253eectdtturz77d7z55sxia6wdvsq
- Missing evidence: 189057730455837902155591890661235220962
- Embedding query: 189057730455837902155591890661235220962
- AST query: 189057730455837902155591890661235220962
- Surplus group: objective/ASI-G094
- Merge key: 64503801d2488d12
- Merge family: goal_packet/analysis/ipfs_accelerate_py/2478d2e4d54c
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/analysis/ipfs_accelerate_py/2478d2e4d54c
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G094, ASI-G095
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: edc11b359256ea94
- Acceptance: Objective scan filed this gap for ASI-G094. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-027-objective-gap-90438f144aad.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (189057730455837902155591890661235220962), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/analysis/ipfs_accelerate_py/2478d2e4d54c; implement a complete, cohesive change that fully advances the packet goals (ASI-G094, ASI-G095) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-028 Close objective gap: Prove 184801846437522667882915494501685213497 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G020
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Changed paths: 
- AST symbols: 184801846437522667882915494501685213497
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G095
- Canonical task key: task/v1/b245e0e63962e3b5ee94b24817f0e159d8b6eafe93c9f6fb6ffe30b8a5532fd7
- Canonical task CID: baguqeerawjc6bzrzmlr3l3uuwjebp4hblhmln2x6spe7n63p7yylrjktf7lq
- Missing evidence: 184801846437522667882915494501685213497
- Embedding query: 184801846437522667882915494501685213497
- AST query: 184801846437522667882915494501685213497
- Surplus group: objective/ASI-G095
- Merge key: 1fd6230c14418ec1
- Merge family: goal_packet/analysis/ipfs_accelerate_py/2478d2e4d54c
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/analysis/ipfs_accelerate_py/2478d2e4d54c
- Goal packet role: packet_member
- Goal packet goals: ASI-G094, ASI-G095
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 9516b69560d1d72f
- Acceptance: Objective scan filed this gap for ASI-G095. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-028-objective-gap-e4f1f20e07e0.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (184801846437522667882915494501685213497), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/analysis/ipfs_accelerate_py/2478d2e4d54c; implement a complete, cohesive change that fully advances the packet goals (ASI-G094, ASI-G095) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-029 Close objective gap: Prove 173075880069453142914839090434430341799 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G030
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Changed paths: 
- AST symbols: 173075880069453142914839090434430341799
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G097
- Canonical task key: task/v1/24bafaaf5f769d150259dd648b3a350e9bd05ff7b50167e2121ebb7e1a6ec039
- Canonical task CID: baguqeeraes5pvl27o2orkasz3vsiworvb2n5ax7xwuawpyqsd25x4gtoya4q
- Missing evidence: 173075880069453142914839090434430341799
- Embedding query: 173075880069453142914839090434430341799
- AST query: 173075880069453142914839090434430341799
- Surplus group: objective/ASI-G097
- Merge key: 6ffe9fe3ff2766cf
- Merge family: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G097, ASI-G098
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: dfb696e9d7215f71
- Acceptance: Objective scan filed this gap for ASI-G097. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-029-objective-gap-9e1fd435ac1b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (173075880069453142914839090434430341799), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/planning/ipfs_accelerate_py/2e451c323b10; implement a complete, cohesive change that fully advances the packet goals (ASI-G097, ASI-G098) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-030 Close objective gap: Prove 003778425160038348524906247302938706902 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G030
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Changed paths: 
- AST symbols: 003778425160038348524906247302938706902
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G098
- Canonical task key: task/v1/f7c5a9e28d2e4cc80091ba52fed61f6e935e6fefecca088fbcbdd2014c5d2b5b
- Canonical task CID: baguqeera67c2tyunfzgmqaerxjjp5vq7n2jv437p5tfard54xxjactc5fnnq
- Missing evidence: 003778425160038348524906247302938706902
- Embedding query: 003778425160038348524906247302938706902
- AST query: 003778425160038348524906247302938706902
- Surplus group: objective/ASI-G098
- Merge key: 64538047183e91c1
- Merge family: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Goal packet role: packet_member
- Goal packet goals: ASI-G097, ASI-G098
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 4f2d97929bcebd8d
- Acceptance: Objective scan filed this gap for ASI-G098. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-030-objective-gap-ada9f763c05d.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (003778425160038348524906247302938706902), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/planning/ipfs_accelerate_py/2e451c323b10; implement a complete, cohesive change that fully advances the packet goals (ASI-G097, ASI-G098) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-031 Close objective gap: Prove 314133036252270790078901745919131980427 for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G040
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Changed paths: 
- AST symbols: 314133036252270790078901745919131980427
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G100
- Canonical task key: task/v1/b65f723e068a08f44913955adf73b5aa7062b419a6fa192773bfa2e282e6cfbd
- Canonical task CID: baguqeerawzpxepqgriepisitsvnn645vvjygfnazu35bsj3tx6rofaxgz66q
- Missing evidence: 314133036252270790078901745919131980427
- Embedding query: 314133036252270790078901745919131980427
- AST query: 314133036252270790078901745919131980427
- Surplus group: objective/ASI-G100
- Merge key: d713c34afa33211b
- Merge family: goal_packet/validation/ipfs_accelerate_py/c4ebb2700e38
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/validation/ipfs_accelerate_py/c4ebb2700e38
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G100, ASI-G101
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: ab28285f04221c36
- Acceptance: Objective scan filed this gap for ASI-G100. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-031-objective-gap-a77efadb10f5.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (314133036252270790078901745919131980427), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/validation/ipfs_accelerate_py/c4ebb2700e38; implement a complete, cohesive change that fully advances the packet goals (ASI-G100, ASI-G101) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-032 Close objective gap: Prove 266404049326363900535699811645710804440 for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G040
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Changed paths: 
- AST symbols: 266404049326363900535699811645710804440
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G101
- Canonical task key: task/v1/9733a3339d6204dde908411b28197225af55a373588e68799e93b03c9d8d5449
- Canonical task CID: baguqeeras4z2gm45micn32iiiensqglsewxvli3tlchgq6m6soydzhmnkreq
- Missing evidence: 266404049326363900535699811645710804440
- Embedding query: 266404049326363900535699811645710804440
- AST query: 266404049326363900535699811645710804440
- Surplus group: objective/ASI-G101
- Merge key: 636320de371530dd
- Merge family: goal_packet/validation/ipfs_accelerate_py/c4ebb2700e38
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/validation/ipfs_accelerate_py/c4ebb2700e38
- Goal packet role: packet_member
- Goal packet goals: ASI-G100, ASI-G101
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 937e1d04c579736c
- Acceptance: Objective scan filed this gap for ASI-G101. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-032-objective-gap-a68a113300c9.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (266404049326363900535699811645710804440), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/validation/ipfs_accelerate_py/c4ebb2700e38; implement a complete, cohesive change that fully advances the packet goals (ASI-G100, ASI-G101) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-033 Close objective gap: Prove 031486194157679117987393491870400400279 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G070
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Changed paths: 
- AST symbols: 031486194157679117987393491870400400279
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G103
- Canonical task key: task/v1/b7a0771a22b2a157f7141a55f80e49de95248a0d76ab323997f8a9cbd9f4a0c1
- Canonical task CID: baguqeeraw6qhogrcwkqvp5yudjk7qdsj32ksjcqno2vteomx7cu4xwpuudaq
- Missing evidence: 031486194157679117987393491870400400279
- Embedding query: 031486194157679117987393491870400400279
- AST query: 031486194157679117987393491870400400279
- Surplus group: objective/ASI-G103
- Merge key: 58ef9f3a6f4216bd
- Merge family: goal_packet/control/ipfs_accelerate_py/41f9dfafffc3
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/control/ipfs_accelerate_py/41f9dfafffc3
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G103, ASI-G104
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 3326382b6b3fd152
- Acceptance: Objective scan filed this gap for ASI-G103. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-033-objective-gap-0d78ada68b4b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (031486194157679117987393491870400400279), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/control/ipfs_accelerate_py/41f9dfafffc3; implement a complete, cohesive change that fully advances the packet goals (ASI-G103, ASI-G104) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-034 Close objective gap: Prove 127990245919649912156052660092678945998 for High-quality task generation and conflict-aware bundling

- Status: completed
- Completion: manual
- Priority: P1
- Track: task-generation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_task_quality.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_quality.py test/api/test_agent_supervisor_bundle_optimizer.py -q
- Bundle: agent-supervisor/self-improvement/task-generation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-task-generation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G050
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/task-generation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_task_quality.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Changed paths: 
- AST symbols: 127990245919649912156052660092678945998
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G106
- Canonical task key: task/v1/10db9b9dd94af56ab99b28a1578997f5aa2992e1faa2e961ca0f2953e9be013b
- Canonical task CID: baguqeeracdnzxhozjl2wvom3fcqvpcmx6wvctexb7krosyokb4uvh2n6ae5q
- Missing evidence: 127990245919649912156052660092678945998
- Embedding query: 127990245919649912156052660092678945998
- AST query: 127990245919649912156052660092678945998
- Surplus group: objective/ASI-G106
- Merge key: 9d2a9e22c09d4e31
- Merge family: goal_packet/task_generation/ipfs_accelerate_py/ce70f0ff87a8
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/task_generation/ipfs_accelerate_py/ce70f0ff87a8
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G106, ASI-G107
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 7d1a535cbd8caac6
- Acceptance: Objective scan filed this gap for ASI-G106. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-034-objective-gap-65cde082de67.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (127990245919649912156052660092678945998), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/task_generation/ipfs_accelerate_py/ce70f0ff87a8; implement a complete, cohesive change that fully advances the packet goals (ASI-G106, ASI-G107) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-035 Close objective gap: Prove 061582446926920746660485801841658333166 for High-quality task generation and conflict-aware bundling

- Status: completed
- Completion: manual
- Priority: P1
- Track: task-generation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_task_quality.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_quality.py test/api/test_agent_supervisor_bundle_optimizer.py -q
- Bundle: agent-supervisor/self-improvement/task-generation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-task-generation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G050
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/task-generation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_task_quality.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Changed paths: 
- AST symbols: 061582446926920746660485801841658333166
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G107
- Canonical task key: task/v1/2052ea9ff565e2c5faf9ebbc9a40a26f522917a86c5c9859116fa21799512de1
- Canonical task CID: baguqeeraebjovh7vmxrml6xz5o6juqfcn5jcsf5inrojqwirn6rbpgkrfxqq
- Missing evidence: 061582446926920746660485801841658333166
- Embedding query: 061582446926920746660485801841658333166
- AST query: 061582446926920746660485801841658333166
- Surplus group: objective/ASI-G107
- Merge key: 188046114fa5c607
- Merge family: goal_packet/task_generation/ipfs_accelerate_py/ce70f0ff87a8
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/task_generation/ipfs_accelerate_py/ce70f0ff87a8
- Goal packet role: packet_member
- Goal packet goals: ASI-G106, ASI-G107
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 9fe1e62b5ad643a3
- Acceptance: Objective scan filed this gap for ASI-G107. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-035-objective-gap-f7111607e003.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (061582446926920746660485801841658333166), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/task_generation/ipfs_accelerate_py/ce70f0ff87a8; implement a complete, cohesive change that fully advances the packet goals (ASI-G106, ASI-G107) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-038 Close objective gap: Prove 020061024173618462922348580596364003627 for Benchmark-driven bounded self-refill

- Status: completed
- Completion: manual
- Priority: P1
- Track: self-refill
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Bundle: agent-supervisor/self-improvement/refill
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-refill.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G080
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/refill
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Changed paths: 
- AST symbols: 020061024173618462922348580596364003627
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G112
- Canonical task key: task/v1/6f06aebf1cca38911c4bee6bae1e80274fd9956f4b004ab5a2d86406f60fd2a5
- Canonical task CID: baguqeeran4dk5py4zi4jchcl5zv24huae5h5tflpjmaevnnc3bsan5qp2ksq
- Missing evidence: 020061024173618462922348580596364003627
- Embedding query: 020061024173618462922348580596364003627
- AST query: 020061024173618462922348580596364003627
- Surplus group: objective/ASI-G112
- Merge key: 1a89496444f4ae3e
- Merge family: goal_packet/self_refill/ipfs_accelerate_py/8a96cb4debe6
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/self_refill/ipfs_accelerate_py/8a96cb4debe6
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G112, ASI-G113
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 3dad940e85a80f47
- Acceptance: Objective scan filed this gap for ASI-G112. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-038-objective-gap-a76f6a041f2a.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (020061024173618462922348580596364003627), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/self_refill/ipfs_accelerate_py/8a96cb4debe6; implement a complete, cohesive change that fully advances the packet goals (ASI-G112, ASI-G113) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-039 Close objective gap: Prove 065313778069923158401871898168782520190 for Benchmark-driven bounded self-refill

- Status: completed
- Completion: manual
- Priority: P1
- Track: self-refill
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Bundle: agent-supervisor/self-improvement/refill
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-refill.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G080
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/refill
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Changed paths: 
- AST symbols: 065313778069923158401871898168782520190
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G113
- Canonical task key: task/v1/ce2a4c8fe70c83ad64ec8c004b38bc4ef13d8f302cae61536eec6565e831338f
- Canonical task CID: baguqeerazyvezd7hbsb22zhmrqaewof4j3yt3dzqfsxgcu3o5rswl2brgohq
- Missing evidence: 065313778069923158401871898168782520190
- Embedding query: 065313778069923158401871898168782520190
- AST query: 065313778069923158401871898168782520190
- Surplus group: objective/ASI-G113
- Merge key: dec04523e504c8a4
- Merge family: goal_packet/self_refill/ipfs_accelerate_py/8a96cb4debe6
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/self_refill/ipfs_accelerate_py/8a96cb4debe6
- Goal packet role: packet_member
- Goal packet goals: ASI-G112, ASI-G113
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: a2c577fff243d1e9
- Acceptance: Objective scan filed this gap for ASI-G113. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-039-objective-gap-80145577de5a.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (065313778069923158401871898168782520190), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/self_refill/ipfs_accelerate_py/8a96cb4debe6; implement a complete, cohesive change that fully advances the packet goals (ASI-G112, ASI-G113) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-040 Close objective gap: Prove 109590900757783560279417463762322084165 for Paired rollout, stable exports, and operator adoption

- Status: completed
- Completion: manual
- Priority: P2
- Track: rollout
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
- Bundle: agent-supervisor/self-improvement/rollout
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-rollout.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G090
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/rollout
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Changed paths: 
- AST symbols: 109590900757783560279417463762322084165
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G115
- Canonical task key: task/v1/1f119403d60eaee47cdb29ada72a90ed5daec982f9b5a38677c70f05be5d6db3
- Canonical task CID: baguqeerad4izia6wb2xoi7g3fgw2okuq5vo25smc7g22hbtxy4hqlps5nwzq
- Missing evidence: 109590900757783560279417463762322084165
- Embedding query: 109590900757783560279417463762322084165
- AST query: 109590900757783560279417463762322084165
- Surplus group: objective/ASI-G115
- Merge key: 91b802c417899562
- Merge family: goal_packet/rollout/test/681e7b98bec3
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/rollout/test/681e7b98bec3
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G115, ASI-G116
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 3867a7024bb24345
- Acceptance: Objective scan filed this gap for ASI-G115. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-040-objective-gap-cd799ff05e32.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (109590900757783560279417463762322084165), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/rollout/test/681e7b98bec3; implement a complete, cohesive change that fully advances the packet goals (ASI-G115, ASI-G116) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-041 Close objective gap: Prove 146189916032404266364029134505159070240 for Paired rollout, stable exports, and operator adoption

- Status: completed
- Completion: manual
- Priority: P2
- Track: rollout
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
- Bundle: agent-supervisor/self-improvement/rollout
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-rollout.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G090
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/rollout
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Changed paths: 
- AST symbols: 146189916032404266364029134505159070240
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G116
- Canonical task key: task/v1/40d356fa3119cb536caecf97d6e9f266bb93cd618dd0b9a4543b3d90032cd8a6
- Canonical task CID: baguqeeraidjvn6rrdhfvg3foz6l5n2psm25zhtlbrxiltjcuhm6zaazm3cta
- Missing evidence: 146189916032404266364029134505159070240
- Embedding query: 146189916032404266364029134505159070240
- AST query: 146189916032404266364029134505159070240
- Surplus group: objective/ASI-G116
- Merge key: b593ee621fc1acdb
- Merge family: goal_packet/rollout/test/681e7b98bec3
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch; goal_subgoal_packet
- Goal packet: goal_packet/rollout/test/681e7b98bec3
- Goal packet role: packet_member
- Goal packet goals: ASI-G115, ASI-G116
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: aggregate
- Todo vector key: 1f9f5463d54101f9
- Acceptance: Objective scan filed this gap for ASI-G116. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-041-objective-gap-a1d0758f5e00.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (146189916032404266364029134505159070240), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/rollout/test/681e7b98bec3; implement a complete, cohesive change that fully advances the packet goals (ASI-G115, ASI-G116) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-042 Close objective gap: Adaptive parallel execution and acceptance throughput

- Status: completed
- Completion: manual
- Priority: P1
- Track: parallelism
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, ipfs_accelerate_py/agent_supervisor/merge_queue.py, test/api/test_agent_supervisor_adaptive_resources.py, test/api/test_agent_supervisor_provider_batch_scheduler.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_resources.py test/api/test_agent_supervisor_provider_batch_scheduler.py test/api/test_agent_supervisor_parallel_acceptance_flow.py -q
- Bundle: agent-supervisor/self-improvement/runtime
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-runtime.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, ipfs_accelerate_py/agent_supervisor/merge_queue.py, test/api/test_agent_supervisor_adaptive_resources.py, test/api/test_agent_supervisor_provider_batch_scheduler.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Changed paths: 
- AST symbols: ResourceScheduler BundleSupervisor ValidationScheduler MergeQueue
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G060
- Consolidated task aliases: ASI-036, ASI-037
- Canonical task key: task/v1/1b42975068842a93d2af5808f95166f2cf308c3f1edc4ff19cf00bef1778c4f1
- Canonical task CID: baguqeeradnbjoudiqqvjhuvplaepsulg6lhtbdb7d3oe74m46af66f3yytyq
- Missing evidence: 122080003600146794820964010047426915846, 124037811551945145648172208272779822741, 185033715568272291470322170325431455647
- Embedding query: adaptive parallelism CPU GPU provider batching validation workers merge queue throughput backpressure fairness
- AST query: ResourceScheduler BundleSupervisor ValidationScheduler MergeQueue
- Surplus group: objective/ASI-G060
- Merge key: 449164eeab10c013
- Merge family: objective/ASI-G060
- Merge role: aggregate
- Work item count: 3
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 5dcbcd5e3eb2335e
- Acceptance: Objective scan filed this gap for ASI-G060. This task consolidates former aliases ASI-036 and ASI-037; satisfy all three evidence obligations (122080003600146794820964010047426915846, 124037811551945145648172208272779822741, 185033715568272291470322170325431455647) in one cohesive implementation and validation pass. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-042-objective-gap-4de9b02f0a16.md, add code/tests/docs or child goals that prove the missing evidence terms are covered, and keep the supervisor-fed backlog aligned with the objective heap. Split scheduling, provider batching, and validation/merge throughput into separately benchmarked lanes.

## ASI-043 Close objective gap: Prove 248026856102230635452423769994290240744 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G010
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Changed paths: 
- AST symbols: 248026856102230635452423769994290240744
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G093
- Canonical task key: task/v1/c70cd0b5a933152e0775e5720c6be26c5c10726d8bb0d17f1a9a1672d595b6d6
- Canonical task CID: baguqeeray4gnbnnjgmks4b3v4vzay27cnroba4tnroync7y2tilhfvmvw3la
- Missing evidence: 248026856102230635452423769994290240744
- Embedding query: 248026856102230635452423769994290240744
- AST query: 248026856102230635452423769994290240744
- Surplus group: objective/ASI-G093
- Merge key: 0cc3e91ad5b303b6
- Merge family: objective/ASI-G093
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 7461411bbf6d3dc5
- Acceptance: Objective scan filed this gap for ASI-G093. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-043-objective-gap-b8e23817e136.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (248026856102230635452423769994290240744), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-044 Close objective gap: Prove 206259342916458424196977899134352826879 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G020
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Changed paths: 
- AST symbols: 206259342916458424196977899134352826879
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G096
- Canonical task key: task/v1/96e3ea0840a67006ab77cbbb29a8e263a002a54bbe61758c1d8c6939d27a1e8e
- Canonical task CID: baguqeeras3r6uccauzyank3xzo5stkhcmoqafjklxzqxlda5rruttut2d2ha
- Missing evidence: 206259342916458424196977899134352826879
- Embedding query: 206259342916458424196977899134352826879
- AST query: 206259342916458424196977899134352826879
- Surplus group: objective/ASI-G096
- Merge key: e94efed435cd071a
- Merge family: objective/ASI-G096
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 52887ef3a6297733
- Acceptance: Objective scan filed this gap for ASI-G096. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-044-objective-gap-df6e813b5a82.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (206259342916458424196977899134352826879), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-045 Close objective gap: Prove 312819945606360295782005228058369235550 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G030
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Changed paths: 
- AST symbols: 312819945606360295782005228058369235550
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G099
- Canonical task key: task/v1/ea3104920bad8db28bbb7c39891af202a0fddd970dc147475cc77d5d81c1aeec
- Canonical task CID: baguqeera5iyqjeqlvwg3fc53pq4ysgxsakqp3xmxbxauor24y56v3aobv3wa
- Missing evidence: 312819945606360295782005228058369235550
- Embedding query: 312819945606360295782005228058369235550
- AST query: 312819945606360295782005228058369235550
- Surplus group: objective/ASI-G099
- Merge key: 05e94edb3f7e80b7
- Merge family: objective/ASI-G099
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 5a79dbca2016c5eb
- Acceptance: Objective scan filed this gap for ASI-G099. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-045-objective-gap-3a1dd4ba61a6.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (312819945606360295782005228058369235550), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-046 Close objective gap: Prove 006818797857632260116084792540150258746 for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G040
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Changed paths: 
- AST symbols: 006818797857632260116084792540150258746
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G102
- Canonical task key: task/v1/9ff16a874e6d330b5f91fed5f8444933adfeac1881789e5bdd088b777438a83c
- Canonical task CID: baguqeerat7ywvb2onuzqwx4r73k7qrcjgow75layqf4j4w65bcfxo5byva6a
- Missing evidence: 006818797857632260116084792540150258746
- Embedding query: 006818797857632260116084792540150258746
- AST query: 006818797857632260116084792540150258746
- Surplus group: objective/ASI-G102
- Merge key: 75e6b8e71f056e34
- Merge family: objective/ASI-G102
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 5485dd864cd292e7
- Acceptance: Objective scan filed this gap for ASI-G102. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-046-objective-gap-9b35a8996689.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (006818797857632260116084792540150258746), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-047 Close objective gap: Prove 186773143401179107362964063059661378722 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G070
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Changed paths: 
- AST symbols: 186773143401179107362964063059661378722
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G105
- Canonical task key: task/v1/643f9c49953d62ac30d5419e15803f9815184bf569e2957715089e41a8cfe2a7
- Canonical task CID: baguqeeramq7zysmvhvrkymgvigpblab7takrqs7vnhrjk5yvbcpedkgp4ktq
- Missing evidence: 186773143401179107362964063059661378722
- Embedding query: 186773143401179107362964063059661378722
- AST query: 186773143401179107362964063059661378722
- Surplus group: objective/ASI-G105
- Merge key: c6c5499ad7d86bbe
- Merge family: objective/ASI-G105
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 863ab773f7c60c6a
- Acceptance: Objective scan filed this gap for ASI-G105. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-047-objective-gap-a69188ffd3d4.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (186773143401179107362964063059661378722), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-048 Close objective gap: Prove 184125100306462690646212311073240043804 for Goal packet aggregate for ASI-G104, ASI-G105

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G104
- Graph depth: 3
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Changed paths: 
- AST symbols: 184125100306462690646212311073240043804
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G126
- Canonical task key: task/v1/a88f368d99aa61fa80bc24787c9dde5063cb558e6b0f1da59fa786171981940b
- Canonical task CID: baguqeeravchtndmzvjq7vaf4er4hzho6kbr4wvmonmhr3jm7u6dbogmbsqfq
- Missing evidence: 184125100306462690646212311073240043804
- Embedding query: 184125100306462690646212311073240043804
- AST query: 184125100306462690646212311073240043804
- Surplus group: objective/ASI-G126
- Merge key: b578d1a541de54ff
- Merge family: objective/ASI-G126
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 2cb91fb4ff005012
- Acceptance: Objective scan filed this gap for ASI-G126. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-048-objective-gap-c740cf99d5d9.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (184125100306462690646212311073240043804), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-049 Close objective gap: Prove 186773143401179107362964063059661378722 for Prove 186773143401179107362964063059661378722 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G105
- Graph depth: 3
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Changed paths: 
- AST symbols: 186773143401179107362964063059661378722
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G128
- Canonical task key: task/v1/79325c9ca7ee97c17092258b2e8b0b89d5d5087c1b68687ff36df79a7fd77477
- Canonical task CID: baguqeerapezfzhfh52l4c4esewfs5cylrhk5kcd4dnugq77tnx3zu76xor3q
- Missing evidence: 186773143401179107362964063059661378722
- Embedding query: 186773143401179107362964063059661378722
- AST query: 186773143401179107362964063059661378722
- Surplus group: objective/ASI-G128
- Merge key: c88211ee36217155
- Merge family: objective/ASI-G128
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 35cedc6db34e16a9
- Acceptance: Objective scan filed this gap for ASI-G128. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-049-objective-gap-74155e075b42.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (186773143401179107362964063059661378722), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-050 Close objective gap: Prove 248026856102230635452423769994290240744 for Prove 248026856102230635452423769994290240744 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G093
- Graph depth: 3
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Changed paths: 
- AST symbols: 248026856102230635452423769994290240744
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G129
- Canonical task key: task/v1/73b9f0599f6b786f434e00ca501f9ad882b4efdd123e8547e441a6026b05fe75
- Canonical task CID: baguqeeraoo47awm7nn4g6q2oadffah423cblj365ci7ikr7eigtae2yf7z2q
- Missing evidence: 248026856102230635452423769994290240744
- Embedding query: 248026856102230635452423769994290240744
- AST query: 248026856102230635452423769994290240744
- Surplus group: objective/ASI-G129
- Merge key: e1c8c9c67b3aabc5
- Merge family: objective/ASI-G129
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: bb4e4883d0c177ef
- Acceptance: Objective scan filed this gap for ASI-G129. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-050-objective-gap-b2ca6ee61aa5.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (248026856102230635452423769994290240744), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-051 Close objective gap: High-quality task generation and conflict-aware bundling

- Status: completed
- Completion: manual
- Priority: P1
- Track: task-generation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_task_quality.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_quality.py test/api/test_agent_supervisor_bundle_optimizer.py -q
- Bundle: agent-supervisor/self-improvement/task-generation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-task-generation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/task-generation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_task_quality.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Changed paths: 
- AST symbols: generate_objective_todos_result build_todo_vector_index BundleSupervisor canonical_task_identity
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G050
- Canonical task key: task/v1/f9407d87ab81844617f0fd63b4707977286026a8eeb4fd84cfdc37c705fe37ff
- Canonical task CID: baguqeera7fah3b5lqgcemf7q7vr3i4dzo4ugajvi522p3bgp3q34obp6g77q
- Missing evidence: 187052702852200236079602798955260586139
- Embedding query: task generation sizing deduplication bundle optimization context reuse dependency DAG conflict graph merge locality
- AST query: generate_objective_todos_result build_todo_vector_index BundleSupervisor canonical_task_identity
- Surplus group: objective/ASI-G050
- Merge key: 4a14a4caaba4b526
- Merge family: objective/ASI-G050
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: aa83309d7f8d56f9
- Acceptance: Objective scan filed this gap for ASI-G050. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-051-objective-gap-badae2b53597.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (187052702852200236079602798955260586139), and keep the supervisor-fed backlog aligned with the objective heap.  Separate task admission from bundle optimization and preserve canonical identities through every projection.

## ASI-052 Close objective gap: Prove 119294002389522221490347364495731444366 for Benchmark-driven bounded self-refill

- Status: completed
- Completion: manual
- Priority: P1
- Track: self-refill
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Bundle: agent-supervisor/self-improvement/refill
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-refill.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G080
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/refill
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Changed paths: 
- AST symbols: 119294002389522221490347364495731444366
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G099
- Canonical task key: task/v1/2cb5f5fc0c3b8a28d9cda314ea81bf88f75cd3571c5ded835f547eda3f53877c
- Canonical task CID: baguqeerafs27l7amhofcrwonumkovan7rd3vzu2xdro63a27kr7nup2tq56a
- Missing evidence: 119294002389522221490347364495731444366
- Embedding query: 119294002389522221490347364495731444366
- AST query: 119294002389522221490347364495731444366
- Surplus group: objective/ASI-G099
- Merge key: 14a64ed96ca5f586
- Merge family: objective/ASI-G099
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: 4dff932ad6b57b55
- Acceptance: Objective scan filed this gap for ASI-G099. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-052-objective-gap-7e0bb64d4f4c.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (119294002389522221490347364495731444366), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-053 Close objective gap: Prove 300500866741873729474343907613893393545 for Paired rollout, stable exports, and operator adoption

- Status: completed
- Completion: manual
- Priority: P2
- Track: rollout
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
- Bundle: agent-supervisor/self-improvement/rollout
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-rollout.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G090
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/rollout
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Changed paths: 
- AST symbols: 300500866741873729474343907613893393545
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G100
- Canonical task key: task/v1/85d76cacfbfec3a054d8bee5553e47a2f3626fe5aae090b2f8d97741931d892e
- Canonical task CID: baguqeeraqxlwzlh373b2avgyx3svkpshulzwe37fvlqjbmxy3f3udey5rexa
- Missing evidence: 300500866741873729474343907613893393545
- Embedding query: 300500866741873729474343907613893393545
- AST query: 300500866741873729474343907613893393545
- Surplus group: objective/ASI-G100
- Merge key: 359fa1d0706a7e5e
- Merge family: objective/ASI-G100
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: aggregate
- Todo vector key: af3f8023bc6dd34d
- Acceptance: Objective scan filed this gap for ASI-G100. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-053-objective-gap-47ed9c5a270c.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (300500866741873729474343907613893393545), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-054 Close objective gap: Prove 173075880069453142914839090434430341799 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G030
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py
- Changed paths: 
- AST symbols: 173075880069453142914839090434430341799
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G097
- Canonical task key: task/v1/cad15f246fa7971f8fae2efaf1dbd66574620a9949ac627f7f259940e7d84874
- Canonical task CID: baguqeerazliv6jdpu6lr7d5of35pdw6wmv2gecuzjgwge737ewmubz6yjb2a
- Missing evidence: objective validation repair
- Embedding query: 173075880069453142914839090434430341799
- AST query: 173075880069453142914839090434430341799
- Surplus group: objective/ASI-G097
- Merge key: f0777f3dc0cad206
- Merge family: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair; goal_subgoal_packet
- Goal packet: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G097, ASI-G098
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: validation_gate
- Todo vector key: f51933e99376411f
- Acceptance: Objective scan filed this gap for ASI-G097. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-054-objective-gap-b4c8d7d78483.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/planning/ipfs_accelerate_py/2e451c323b10; implement a complete, cohesive change that fully advances the packet goals (ASI-G097, ASI-G098) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-055 Close objective gap: Prove 003778425160038348524906247302938706902 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G030
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Changed paths: 
- AST symbols: 003778425160038348524906247302938706902
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G098
- Canonical task key: task/v1/c4a99673dafa35373458d05e62ab3085abfc767bbfa6024790cc17d313b2b0e7
- Canonical task CID: baguqeeraysuzm4627i2toncy2bpgfkzqqwv7y5t3x6taer4qzql5ge5swdtq
- Missing evidence: objective validation repair
- Embedding query: 003778425160038348524906247302938706902
- AST query: 003778425160038348524906247302938706902
- Surplus group: objective/ASI-G098
- Merge key: 195d273f2f1310ce
- Merge family: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair; goal_subgoal_packet
- Goal packet: goal_packet/planning/ipfs_accelerate_py/2e451c323b10
- Goal packet role: packet_member
- Goal packet goals: ASI-G097, ASI-G098
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: validation_gate
- Todo vector key: 41f5a4b8957cd06b
- Acceptance: Objective scan filed this gap for ASI-G098. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-055-objective-gap-9db9f4739ca7.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/planning/ipfs_accelerate_py/2e451c323b10; implement a complete, cohesive change that fully advances the packet goals (ASI-G097, ASI-G098) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-056 Close objective gap: Prove 208290439421789408250562066350459701853 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G010
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Changed paths: 
- AST symbols: 208290439421789408250562066350459701853
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G091
- Canonical task key: task/v1/d6f6f776861df254c47d1473f4d6f9c0cc477953007239ad33fad034bb812735
- Canonical task CID: baguqeera233po5ugdxzfjrd5crz7jvxzydgeo6ktabzdtljt7lidjo4be42q
- Missing evidence: objective validation repair
- Embedding query: 208290439421789408250562066350459701853
- AST query: 208290439421789408250562066350459701853
- Surplus group: objective/ASI-G091
- Merge key: cc27e18ac451d476
- Merge family: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair; goal_subgoal_packet
- Goal packet: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G091, ASI-G092
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: validation_gate
- Todo vector key: 9d5b19c925e31f3c
- Acceptance: Objective scan filed this gap for ASI-G091. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-056-objective-gap-4976bc1f65de.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb; implement a complete, cohesive change that fully advances the packet goals (ASI-G091, ASI-G092) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-057 Close objective gap: Prove 306437607356117177048620815571362227127 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G010
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Changed paths: 
- AST symbols: 306437607356117177048620815571362227127
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G092
- Canonical task key: task/v1/35912d650ff6983b4763de9aaac6e385abea6ffef62c39eda005f0d48dd09966
- Canonical task CID: baguqeeragwis2zip62mdwr3d32nkvrxdqwv6u3766ywdt3naaxynjdoqtfta
- Missing evidence: objective validation repair
- Embedding query: 306437607356117177048620815571362227127
- AST query: 306437607356117177048620815571362227127
- Surplus group: objective/ASI-G092
- Merge key: 98f65c45c663dc9b
- Merge family: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair; goal_subgoal_packet
- Goal packet: goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb
- Goal packet role: packet_member
- Goal packet goals: ASI-G091, ASI-G092
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: validation_gate
- Todo vector key: 4d06f7b8f0af0d28
- Acceptance: Objective scan filed this gap for ASI-G092. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-057-objective-gap-47b60a795e58.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/token_efficiency/ipfs_accelerate_py/3841d2bd1acb; implement a complete, cohesive change that fully advances the packet goals (ASI-G091, ASI-G092) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-058 Produce completion evidence for Prove 003778425160038348524906247302938706902 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G030
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Changed paths: 
- AST symbols: 003778425160038348524906247302938706902
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G098
- Canonical task key: task/v1/9c2483cd94c42f49caaee0ba12207dc685c9b3b6fc819aa6ac504ba8700c6dce
- Canonical task CID: baguqeeratqsihtmuyqxutsvo4c5beid5y2c4tm5w7sazvjvmkbf2q4amnxha
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: A changed typed counterexample can generate and admit at most one bounded refinement in the next cycle, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 003778425160038348524906247302938706902
- AST query: 003778425160038348524906247302938706902
- Surplus group: ASI-G098
- Merge key: objective-work/v1/3e3d8913fae5a1f14a4d6f5f67e4ed03da9c353b
- Merge family: ASI-G098
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 3e3d8913fae5a1f1
- Acceptance: Objective scan filed this gap for ASI-G098. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-058-objective-gap-705cb78ae6b6.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: A changed typed counterexample can generate and admit at most one bounded refinement in the next cycle, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-059 Produce completion evidence for Prove 173075880069453142914839090434430341799 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, test/api/test_agent_supervisor_adaptive_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G030
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, test/api/test_agent_supervisor_adaptive_planner.py
- Changed paths: 
- AST symbols: 173075880069453142914839090434430341799
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G097
- Canonical task key: task/v1/37520313089e13e0d58d7b08023fbccabbeb14f23a1227b3bbd31996a1ff1326
- Canonical task CID: baguqeerag5jageyityj6bvmnpmeaep54zk56wfhshijcpm532mmznip7cmta
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Deterministic evaluation covers acceptance evidence, assumptions, semantics, dependencies, conflicts, validation and proof feasibility, novelty, and bounded resource/token cost, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 173075880069453142914839090434430341799
- AST query: 173075880069453142914839090434430341799
- Surplus group: ASI-G097
- Merge key: objective-work/v1/1fcfeaf708579f9e141b309b5ec484e0f2102e47
- Merge family: ASI-G097
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 1fcfeaf708579f9e
- Acceptance: Objective scan filed this gap for ASI-G097. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-059-objective-gap-7aebdd419a0a.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Deterministic evaluation covers acceptance evidence, assumptions, semantics, dependencies, conflicts, validation and proof feasibility, novelty, and bounded resource/token cost, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-060 Produce completion evidence for Prove 306437607356117177048620815571362227127 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G010
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Changed paths: 
- AST symbols: 306437607356117177048620815571362227127
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G092
- Canonical task key: task/v1/1b96710f6f84c2fe639f64fc6d0e70344a3ca3636e4d5a8b300ae4761c03997f
- Canonical task CID: baguqeeradolhcd3pqtbp4y47mt6g2dtqgrfdzi3dnzgvvczqblshmhadtf7q
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: A compact retry references its exact parent capsule without replaying the invariant core and transmits only deterministic changed or newly requested evidence, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 306437607356117177048620815571362227127
- AST query: 306437607356117177048620815571362227127
- Surplus group: ASI-G092
- Merge key: objective-work/v1/722bedab7449da27be39ff7838bc11bf437e395a
- Merge family: ASI-G092
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 722bedab7449da27
- Acceptance: Objective scan filed this gap for ASI-G092. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-060-objective-gap-bb0490fbb641.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: A compact retry references its exact parent capsule without replaying the invariant core and transmits only deterministic changed or newly requested evidence, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-061 Produce completion evidence for Prove 208290439421789408250562066350459701853 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G010
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Changed paths: 
- AST symbols: 208290439421789408250562066350459701853
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G091
- Canonical task key: task/v1/6d0ff7e2def3718dfdd3ca978fb48d79a912c4c907682c46909c79cbe7d78234
- Canonical task CID: baguqeeranuh7pyw66nyy37otzkly7nenpgurfrgja5ucyruqtr44xz6xqi2a
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The compiler derives the effective input limit from the supervisor ceiling, provider input/window limits, and reserved output/tool tokens, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 208290439421789408250562066350459701853
- AST query: 208290439421789408250562066350459701853
- Surplus group: ASI-G091
- Merge key: objective-work/v1/4a791f0bd99c42fa4b2e1777930a91abd7d228f1
- Merge family: ASI-G091
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 4a791f0bd99c42fa
- Acceptance: Objective scan filed this gap for ASI-G091. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-061-objective-gap-d4f0c26b57ee.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The compiler derives the effective input limit from the supervisor ceiling, provider input/window limits, and reserved output/tool tokens, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-062 Close objective gap: Prove 189057730455837902155591890661235220962 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G020
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_cache_coordinator.py
- Changed paths: 
- AST symbols: EXACT_TREE_REUSE_REQUIREMENT_ID ExactTreeReuseEvidence AnalysisPipeline AnalysisCacheKey
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G094
- Canonical task key: task/v1/df5afe2392bb4895bf395e11652ac34b014d952e57ba42766a8258c30a5c773c
- Canonical task CID: baguqeera35np4i4sxnejlpzzlyiwkkwdjmau3fjok65ee5tkqjmmgcs4o46a
- Missing evidence: objective validation repair
- Embedding query: 189057730455837902155591890661235220962
- AST query: EXACT_TREE_REUSE_REQUIREMENT_ID ExactTreeReuseEvidence AnalysisPipeline AnalysisCacheKey
- Surplus group: objective/ASI-G094
- Merge key: e4e0d4c943916e15
- Merge family: goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair; goal_subgoal_packet
- Goal packet: goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf
- Goal packet role: packet_anchor
- Goal packet goals: ASI-G094, ASI-G095
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: validation_gate
- Todo vector key: 42e82e7c5a525c25
- Acceptance: Objective scan filed this gap for ASI-G094. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-062-objective-gap-ad4026a19334.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf; implement a complete, cohesive change that fully advances the packet goals (ASI-G094, ASI-G095) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-063 Close objective gap: Prove 184801846437522667882915494501685213497 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_analysis_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G020
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_analysis_pipeline.py
- Changed paths: 
- AST symbols: IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID IpfsDatasetsProviderDegradationEvidence IpfsDatasetsAnalysisProvider
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G095
- Canonical task key: task/v1/94932abf8e726ea1389727349d44b64ee0a09339943c818892f2f31d5ac2eca0
- Canonical task CID: baguqeerassjsvp4oojxkcoexe42j2rfwj3qkbezzsq6idces6lzr2wwc5sqa
- Missing evidence: objective validation repair
- Embedding query: 184801846437522667882915494501685213497
- AST query: IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID IpfsDatasetsProviderDegradationEvidence IpfsDatasetsAnalysisProvider
- Surplus group: objective/ASI-G095
- Merge key: 64c04da6f929a58d
- Merge family: goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair; goal_subgoal_packet
- Goal packet: goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf
- Goal packet role: packet_member
- Goal packet goals: ASI-G094, ASI-G095
- Goal packet task count: 2
- Goal packet work item count: 2
- Candidate kind: validation_gate
- Todo vector key: 12f72804bb681d1d
- Acceptance: Objective scan filed this gap for ASI-G095. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-063-objective-gap-f83e780176b0.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf; implement a complete, cohesive change that fully advances the packet goals (ASI-G094, ASI-G095) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-064 Close objective gap: Prove 266404049326363900535699811645710804440 for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G040
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Changed paths: 
- AST symbols: TRANSITIVE_IMPACT_REQUIREMENT_ID TransitiveImpactValidationEvidence ImpactDependencyGraph.validation_targets ValidationDAGNodeRecord ValidationAuthorityGateRecord REQUIRED_AUTHORITY_GATES ValidationDAGReceipt.required_validation_ids ValidationDAGReceipt.selected_node_ids ValidationDAGReceipt.coverage_complete ValidationDAGReceipt.authority_gates
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G101
- Canonical task key: task/v1/82e75763f7d883d3c4c8c6ce6ae2b697c2c6376eb2d2cf22550c7d86fba6f2f4
- Canonical task CID: baguqeeraqltvoy7x3cb5hrgiy3hgvyvws7bmmn3owljm6isvbr6yn65g6l2a
- Missing evidence: objective validation repair
- Embedding query: 266404049326363900535699811645710804440
- AST query: TRANSITIVE_IMPACT_REQUIREMENT_ID TransitiveImpactValidationEvidence ImpactDependencyGraph.validation_targets ValidationDAGNodeRecord ValidationAuthorityGateRecord REQUIRED_AUTHORITY_GATES ValidationDAGReceipt.required_validation_ids ValidationDAGReceipt.selected_node_ids ValidationDAGReceipt.coverage_complete ValidationDAGReceipt.authority_gates
- Surplus group: objective/ASI-G101
- Merge key: ac052ce35a764708
- Merge family: objective/ASI-G101
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: validation_gate
- Todo vector key: 237e370a4ddea0a3
- Acceptance: Objective scan filed this gap for ASI-G101. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-064-objective-gap-d16e31809741.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-065 Produce completion evidence for Prove 189057730455837902155591890661235220962 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G020
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_cache_coordinator.py
- Changed paths: 
- AST symbols: 189057730455837902155591890661235220962
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G094
- Canonical task key: task/v1/973b06447c50c364006a101fac33674a6c3fe731a3a251bc447a31c535522056
- Canonical task CID: baguqeeras45qmrd4kdbwiadkcap2ym3hjjwd7zzruorfdpcepiy4knksebla
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The live pipeline composes the existing cache, incremental AST index, bounded multi-signal retrieval, and optional datasets adapter, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 189057730455837902155591890661235220962
- AST query: 189057730455837902155591890661235220962
- Surplus group: ASI-G094
- Merge key: objective-work/v1/f461f62e36e0f814e0265951be8eed2a9e485801
- Merge family: ASI-G094
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: f461f62e36e0f814
- Acceptance: Objective scan filed this gap for ASI-G094. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-065-objective-gap-6016ec4c40a3.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The live pipeline composes the existing cache, incremental AST index, bounded multi-signal retrieval, and optional datasets adapter, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-066 Produce completion evidence for Prove 184801846437522667882915494501685213497 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G020
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py
- Changed paths: 
- AST symbols: 184801846437522667882915494501685213497
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G095
- Canonical task key: task/v1/563518e44622c3528cb7aba6aa4a3ce77c76a2587321a30111d8e34e8555bfaf
- Canonical task CID: baguqeeraky2rrzcgelbvfdfxvotkusr4456hnisyomq2gair3dru5bkvx6xq
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Capability inspection is deterministic and side-effect free, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 184801846437522667882915494501685213497
- AST query: 184801846437522667882915494501685213497
- Surplus group: ASI-G095
- Merge key: objective-work/v1/97d81e341c97e2791e4f83c01a7107d99609b547
- Merge family: ASI-G095
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 97d81e341c97e279
- Acceptance: Objective scan filed this gap for ASI-G095. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-066-objective-gap-bb9d80e1677b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Capability inspection is deterministic and side-effect free, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-067 Close objective gap: Prove 031486194157679117987393491870400400279 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, ipfs_accelerate_py/mcp_server/server.py, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_control_lifecycle.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G070
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, ipfs_accelerate_py/mcp_server/server.py, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_control_lifecycle.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Changed paths: 
- AST symbols: CONTROL_SURFACE_PARITY_REQUIREMENT_ID ControlSurfaceParityEvidence ControlSurfaceParityCase operation_request_json_schema operation_result_json_schema COMMAND_OPERATIONS AGENT_SUPERVISOR_OPERATION_TOOLS run_agent_cli execute_agent_supervisor_operation
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G103
- Canonical task key: task/v1/47de1adbd36c138c25d8a99b2ad3242f80d09917d276acca51d91ad86cf6f1d3
- Canonical task CID: baguqeerai7pbvw6tnqjyyjoyvgnsvuzef6anbgix2j3kzssr3ennq3hw6hjq
- Missing evidence: objective validation repair
- Embedding query: 031486194157679117987393491870400400279
- AST query: CONTROL_SURFACE_PARITY_REQUIREMENT_ID ControlSurfaceParityEvidence ControlSurfaceParityCase operation_request_json_schema operation_result_json_schema COMMAND_OPERATIONS AGENT_SUPERVISOR_OPERATION_TOOLS run_agent_cli execute_agent_supervisor_operation
- Surplus group: objective/ASI-G103
- Merge key: a66a038d470f7f7d
- Merge family: objective/ASI-G103
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: validation_gate
- Todo vector key: d308edaf7575f0c9
- Acceptance: Objective scan filed this gap for ASI-G103. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-067-objective-gap-f0fe8b58bd8e.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-068 Close objective gap: Prove 248026856102230635452423769994290240744 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G010
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Changed paths: 
- AST symbols: 248026856102230635452423769994290240744
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G093
- Canonical task key: task/v1/c409109e5037a212deca7e482014fb9477b71aac49bb667da468133ef3c48f90
- Canonical task CID: baguqeerayqerbhsqg6rbfxwkpzecafh3sr33ogvmjg5wm7nenajt546er6ia
- Missing evidence: objective validation repair
- Embedding query: 248026856102230635452423769994290240744
- AST query: 248026856102230635452423769994290240744
- Surplus group: objective/ASI-G093
- Merge key: 8edce881b9c14ef7
- Merge family: objective/ASI-G093
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: validation_gate
- Todo vector key: 69018645615511a5
- Acceptance: Objective scan filed this gap for ASI-G093. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-068-objective-gap-65d7599db40b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-069 Close objective gap: Prove 206259342916458424196977899134352826879 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G020
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Changed paths: 
- AST symbols: SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID SingleFlightCollapseEvidence AnalysisCacheCoordinator._begin AnalysisCacheCoordinator.get_or_compute AnalysisCacheCoordinator.async_get_or_compute
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G096
- Canonical task key: task/v1/37ada3bc65f6335e73c29909ae5bd7c7d4389486e87f443df5db03b636dbb21b
- Canonical task CID: baguqeerag6w2hpdf6yzv446ctee24w6xy7kdrfeg5b7uippv3mb3mnw3winq
- Missing evidence: objective validation repair
- Embedding query: 206259342916458424196977899134352826879
- AST query: SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID SingleFlightCollapseEvidence AnalysisCacheCoordinator._begin AnalysisCacheCoordinator.get_or_compute AnalysisCacheCoordinator.async_get_or_compute
- Surplus group: objective/ASI-G096
- Merge key: 86e40333ee77c18b
- Merge family: objective/ASI-G096
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: validation_gate
- Todo vector key: bd187c1d5f6279c6
- Acceptance: Objective scan filed this gap for ASI-G096. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-069-objective-gap-17b2c8bb470e.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-070 Close objective gap: Prove 006818797857632260116084792540150258746 for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G040
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Changed paths: 
- AST symbols: PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID ProofCandidateNonAuthorityEvidence prove_proof_candidate_non_authority CodeProofReceiptBindingResult.result_id CompletionAdmissionGate.code_proof_result_ids CompletionAdmissionGate.proof_candidate_receipt_ids evaluate_completion_admission
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G102
- Canonical task key: task/v1/3e919b2299b719f97b14ede7b1ebabd186d8fe46c144c5f6c1e62e2f73d2d6de
- Canonical task CID: baguqeerah2izwiuzw4m7s6yu5xt3d25l2gdnr7sgyfcml5wb4yxc646s23pa
- Missing evidence: objective validation repair
- Embedding query: 006818797857632260116084792540150258746
- AST query: PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID ProofCandidateNonAuthorityEvidence prove_proof_candidate_non_authority CodeProofReceiptBindingResult.result_id CompletionAdmissionGate.code_proof_result_ids CompletionAdmissionGate.proof_candidate_receipt_ids evaluate_completion_admission
- Surplus group: objective/ASI-G102
- Merge key: c37601a178a7ebc7
- Merge family: objective/ASI-G102
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: validation_gate
- Todo vector key: e35575306c606b9e
- Acceptance: Objective scan filed this gap for ASI-G102. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-070-objective-gap-843224fc1fa9.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-071 Close objective gap: Prove 184125100306462690646212311073240043804 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_control_lifecycle.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G070
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_control_lifecycle.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Changed paths: 
- AST symbols: CONTROL_MUTATION_GUARD_REQUIREMENT_ID ControlMutationGuardEvidence MutationGuardExecutionObservation ControlMutationRuntimeState MutationGuardRejection OperationRequest._validate_mutation_bindings SupervisorControlService.mutation_runtime_state SupervisorControlService._check_authorization SupervisorControlService._check_idempotency SupervisorControlService._check_lease ControlAuditReceipt
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G104
- Canonical task key: task/v1/df18a0b1ed582d4511e8cc8c9c983b5a6c72f1322bdca6235bbef3dec6a6a214
- Canonical task CID: baguqeera34mkbmpnlawukepizsgjzgb3ljwhf4jsfpokmi23x3z55rvguika
- Missing evidence: objective validation repair
- Embedding query: 184125100306462690646212311073240043804
- AST query: CONTROL_MUTATION_GUARD_REQUIREMENT_ID ControlMutationGuardEvidence MutationGuardExecutionObservation ControlMutationRuntimeState MutationGuardRejection OperationRequest._validate_mutation_bindings SupervisorControlService.mutation_runtime_state SupervisorControlService._check_authorization SupervisorControlService._check_idempotency SupervisorControlService._check_lease ControlAuditReceipt
- Surplus group: objective/ASI-G104
- Merge key: 25082f9906213b8d
- Merge family: objective/ASI-G104
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: validation_gate
- Todo vector key: eac0eba5f164a96a
- Acceptance: Objective scan filed this gap for ASI-G104. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-071-objective-gap-127c995641c2.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-072 Close objective gap: Prove 186773143401179107362964063059661378722 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G070
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Changed paths: 
- AST symbols: CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID ControlDiscoveryManifest ControlDiscoveryRuntimeState ControlDiscoveryObservation ControlDiscoverySafetyEvidence SupervisorControlService.discovery_manifest capture_control_discovery_runtime_state agent_cli_discovery_manifest agent_supervisor_discovery_manifest agent_supervisor_service_resolution_count
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G105
- Canonical task key: task/v1/b549ea07aa1cd5d401fde6ad51f38651be6d22d6c3e2f95d7affa86964135215
- Canonical task CID: baguqeerawve6ub5kdtk5iap542wvd44gkg7g2iwwyprpsxl276ugszatkikq
- Missing evidence: objective validation repair
- Embedding query: 186773143401179107362964063059661378722
- AST query: CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID ControlDiscoveryManifest ControlDiscoveryRuntimeState ControlDiscoveryObservation ControlDiscoverySafetyEvidence SupervisorControlService.discovery_manifest capture_control_discovery_runtime_state agent_cli_discovery_manifest agent_supervisor_discovery_manifest agent_supervisor_service_resolution_count
- Surplus group: objective/ASI-G105
- Merge key: c4e1887e6d32629a
- Merge family: objective/ASI-G105
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: validation_gate
- Todo vector key: 7994c1509fc6a036
- Acceptance: Objective scan filed this gap for ASI-G105. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-072-objective-gap-5dcde08ee9be.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.

## ASI-073 Produce completion evidence for Prove 003778425160038348524906247302938706902 for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G030
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Changed paths: 
- AST symbols: 003778425160038348524906247302938706902
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G098
- Canonical task key: task/v1/9fd7f221ad93ea923f802b85bf009f64b5324ccf913484ec26cd802ee4cfe91b
- Canonical task CID: baguqeerat7l7einnspvjep4afoc36ae7ms2tetgpse2ij3bgzwac5zgp5enq
- Evidence obligation key: objective-work/v1/2a7f341e9c9dec24c6e4403a835ec127ddb90b23
- Missing evidence: completion analyzer health, completion criterion coverage, completion exhaustion quorum
- Embedding query: 003778425160038348524906247302938706902
- AST query: 003778425160038348524906247302938706902
- Surplus group: ASI-G098
- Merge key: objective-work/v1/2a7f341e9c9dec24c6e4403a835ec127ddb90b23
- Merge family: ASI-G098
- Merge role: completion_gate
- Work item count: 3
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 2a7f341e9c9dec24
- Acceptance: Objective scan filed this gap for ASI-G098. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-073-objective-gap-3fa69d827550.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (completion analyzer health, completion criterion coverage, completion exhaustion quorum), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-074 Produce completion evidence for Prove 248026856102230635452423769994290240744 for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G010
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Changed paths: 
- AST symbols: 248026856102230635452423769994290240744
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G093
- Canonical task key: task/v1/cade23456b0a03fd035d97d0ef3dad9db204fd173ef2e5d0a9f60f5178a184e0
- Canonical task CID: baguqeerazlpcgrllbib72a25s7io6pnntwzaj7ixh3zolufj6yhvc6fbqtqa
- Evidence obligation key: objective-work/v1/573cd85b5929284ecec96a6deb6dc6ac1b0e7a9b
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The exact requirement ID is emitted only by a bounded, content-addressed benchmark receipt carrying the complete typed baseline and candidate receipt populations, one frozen goal/tree/policy binding, the independently replayed paired result, source and report identities, a deterministic input digest, and a passing accounting result. A completion gate verifies the artifact against its independently enumerated benchmark cohort, so an omitted, duplicated, reordered, or substituted input is either canonicalized to the same evidence identity or fails closed. The accepted-task population must be non-empty and identical across arms, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 248026856102230635452423769994290240744
- AST query: 248026856102230635452423769994290240744
- Surplus group: ASI-G093
- Merge key: objective-work/v1/573cd85b5929284ecec96a6deb6dc6ac1b0e7a9b
- Merge family: ASI-G093
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 573cd85b5929284e
- Acceptance: Objective scan filed this gap for ASI-G093. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-074-objective-gap-4a168d424c57.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The exact requirement ID is emitted only by a bounded, content-addressed benchmark receipt carrying the complete typed baseline and candidate receipt populations, one frozen goal/tree/policy binding, the independently replayed paired result, source and report identities, a deterministic input digest, and a passing accounting result. A completion gate verifies the artifact against its independently enumerated benchmark cohort, so an omitted, duplicated, reordered, or substituted input is either canonicalized to the same evidence identity or fails closed. The accepted-task population must be non-empty and identical across arms, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-075 Produce completion evidence for Prove 266404049326363900535699811645710804440 for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py, test/api/test_agent_supervisor_validation_dag.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G040
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py, test/api/test_agent_supervisor_validation_dag.py
- Changed paths: 
- AST symbols: 266404049326363900535699811645710804440
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G101
- Canonical task key: task/v1/e61da564214aa1cde3be32c1cbe3b4c0877a58a399739d59265c5a5a7276f494
- Canonical task CID: baguqeera4yo2kzbbjkq43y56gla4xy5uycdxuwfdtfzz2wjglrnfu4tw6ska
- Evidence obligation key: objective-work/v1/8433e3f318f68068174cbfe718c3c1e6f020cfae
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The validation DAG is derived from the canonical changed-file and dependency/interface impact graph and validated declarations that explicitly name impact targets, node dependencies, and downstream authority gates. Its receipt contains the complete selected population, includes every mandatory direct and transitive validation exactly once, schedules only dependency-ready nodes under bounded parallelism, records selected, executed, succeeded, failed, blocked, and omitted dispositions deterministically, and identifies the failed prerequisite for every blocked descendant. Missing, stale, cyclic, inconsistent, or population-incomplete impact evidence fails closed before it can grant authority. A seeded upstream defect must select and execute its transitively affected consumer test, whose real failure creates explicit closed records for semantic/proof promotion, merge, freshness, and completion authority. The exact requirement ID is emitted only by a tamper-evident receipt binding the current tree, objective, policy, accepted proposal, change and graph identities, declaration set, affected closure, selected-node population, DAG dependencies, authority-gate closure, impact paths, actual defect-detecting result, fail-fast disposition, and content digest., Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 266404049326363900535699811645710804440
- AST query: 266404049326363900535699811645710804440
- Surplus group: ASI-G101
- Merge key: objective-work/v1/8433e3f318f68068174cbfe718c3c1e6f020cfae
- Merge family: ASI-G101
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 8433e3f318f68068
- Acceptance: Objective scan filed this gap for ASI-G101. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-075-objective-gap-2f6255939201.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The validation DAG is derived from the canonical changed-file and dependency/interface impact graph and validated declarations that explicitly name impact targets, node dependencies, and downstream authority gates. Its receipt contains the complete selected population, includes every mandatory direct and transitive validation exactly once, schedules only dependency-ready nodes under bounded parallelism, records selected, executed, succeeded, failed, blocked, and omitted dispositions deterministically, and identifies the failed prerequisite for every blocked descendant. Missing, stale, cyclic, inconsistent, or population-incomplete impact evidence fails closed before it can grant authority. A seeded upstream defect must select and execute its transitively affected consumer test, whose real failure creates explicit closed records for semantic/proof promotion, merge, freshness, and completion authority. The exact requirement ID is emitted only by a tamper-evident receipt binding the current tree, objective, policy, accepted proposal, change and graph identities, declaration set, affected closure, selected-node population, DAG dependencies, authority-gate closure, impact paths, actual defect-detecting result, fail-fast disposition, and content digest., Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-076 Produce completion evidence for Prove 186773143401179107362964063059661378722 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G070
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Changed paths: 
- AST symbols: 186773143401179107362964063059661378722
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G105
- Canonical task key: task/v1/b39bce9167062b0507f83b2a7307a9f85bf9a1061cdd352b6a342a7052e30475
- Canonical task CID: baguqeerawon45elhayvqkb7yhmvhgb5j7bn7tiigdtotkk3kgqvhauxdar2q
- Evidence obligation key: objective-work/v1/9de0c7b66d9f7ecd7ae5af2eeefb440e657c8d77
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Repeated Python, CLI, and MCP discovery is byte-deterministic and covers the same closed operation/schema population, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 186773143401179107362964063059661378722
- AST query: 186773143401179107362964063059661378722
- Surplus group: ASI-G105
- Merge key: objective-work/v1/9de0c7b66d9f7ecd7ae5af2eeefb440e657c8d77
- Merge family: ASI-G105
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 9de0c7b66d9f7ecd
- Acceptance: Objective scan filed this gap for ASI-G105. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-076-objective-gap-8ac999078ba3.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Repeated Python, CLI, and MCP discovery is byte-deterministic and covers the same closed operation/schema population, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-077 Produce completion evidence for Prove 184125100306462690646212311073240043804 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_lifecycle.py, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G070
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_lifecycle.py, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Changed paths: 
- AST symbols: 184125100306462690646212311073240043804
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G104
- Canonical task key: task/v1/c330fbeea16ef1d53eb4b43980b159caf3bb8b9c9d5b2032c96e87bb59f1b44a
- Canonical task CID: baguqeeraymypx3vbn3y5kpvuwq4ybmkzzlz3xc44tvnsamwjn2d3wwprwrfa
- Evidence obligation key: objective-work/v1/bcdcc08d98c2bee4c0f4829ce757072367049df5
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Unauthorized, unscoped, unfenced, stale, path-escaping, or undeclared-effect mutations fail before dispatch on every surface, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 184125100306462690646212311073240043804
- AST query: 184125100306462690646212311073240043804
- Surplus group: ASI-G104
- Merge key: objective-work/v1/bcdcc08d98c2bee4c0f4829ce757072367049df5
- Merge family: ASI-G104
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: bcdcc08d98c2bee4
- Acceptance: Objective scan filed this gap for ASI-G104. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-077-objective-gap-d94d0ddd7f02.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Unauthorized, unscoped, unfenced, stale, path-escaping, or undeclared-effect mutations fail before dispatch on every surface, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-078 Produce completion evidence for Prove 031486194157679117987393491870400400279 for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/server.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_lifecycle.py, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G070
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/server.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_lifecycle.py, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Changed paths: 
- AST symbols: 031486194157679117987393491870400400279
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G103
- Canonical task key: task/v1/826ae374810587ef1471fb3bcb4e85902d5167d63e271c35ed5aa21e5f183b24
- Canonical task CID: baguqeeraqjvog5ebawd66fdr7m54wtufsawvcz6whytrynpnlkrb4xyyhmsa
- Evidence obligation key: objective-work/v1/65258c680ddf290ac7a8af37c6cf92c2284f41ac
- Missing evidence: Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The shared schema describes all operations, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.
- Embedding query: 031486194157679117987393491870400400279
- AST query: 031486194157679117987393491870400400279
- Surplus group: ASI-G103
- Merge key: objective-work/v1/65258c680ddf290ac7a8af37c6cf92c2284f41ac
- Merge family: ASI-G103
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 65258c680ddf290a
- Acceptance: Objective scan filed this gap for ASI-G103. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-078-objective-gap-dd2576a635bc.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: The shared schema describes all operations, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree., Task completion is provisional until every criterion has valid evidence.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-079 Produce completion evidence for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_cache_coordinator.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Bundle: agent-supervisor/self-improvement/analysis
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-analysis.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_cache_coordinator.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py
- Changed paths: 
- AST symbols: 184801846437522667882915494501685213497, 189057730455837902155591890661235220962, 206259342916458424196977899134352826879
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G020
- Canonical task key: task/v1/31edd6aca7c437344e00848af35f6b8e5c4867a6e234565e04833870f827109a
- Canonical task CID: baguqeeraghw5nlfhyq3titqaqsfpgx3lrzoeqz5g4i2fmxqeqm4hb6bhccna
- Evidence obligation key: objective-work/v1/b22b3881227ebd6dc6aa41980459b59382f5561d
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Existing analysis cache, AST index, and retrieval contracts are used in the live objective/planning path, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 184801846437522667882915494501685213497 189057730455837902155591890661235220962 206259342916458424196977899134352826879
- AST query: 184801846437522667882915494501685213497, 189057730455837902155591890661235220962, 206259342916458424196977899134352826879
- Surplus group: ASI-G020
- Merge key: objective-work/v1/b22b3881227ebd6dc6aa41980459b59382f5561d
- Merge family: ASI-G020
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: b22b3881227ebd6d
- Acceptance: Objective scan filed this gap for ASI-G020. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-079-objective-gap-922f6a6ca41b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Existing analysis cache, AST index, and retrieval contracts are used in the live objective/planning path, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-080 Produce completion evidence for Evidence-aware planning and responsive goal refinement

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py, test/api/test_agent_supervisor_adaptive_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Bundle: agent-supervisor/self-improvement/planning
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-planning.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py, test/api/test_agent_supervisor_adaptive_planner.py
- Changed paths: 
- AST symbols: 003778425160038348524906247302938706902, 173075880069453142914839090434430341799, 312819945606360295782005228058369235550
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G030
- Canonical task key: task/v1/e39f135135f14e5baed958cd1e3fbcaf1c14d0bb725ba9bdf5b36a0ce43ec81a
- Canonical task CID: baguqeera4oprgujv6fhfxlwzldgr4p54v4objuf3ojn2tppvwnvazzb6zana
- Evidence obligation key: objective-work/v1/941b45b4a475a78a114da4e82944d2023025eaab
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every descendant must remain verified with all proof requirements fresh, conclusive, uncontradicted, and satisfied., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Every plan is evaluated for acceptance coverage, assumptions, semantics, dependencies, conflicts, validation/proof feasibility, novelty, and resource/token cost, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 003778425160038348524906247302938706902 173075880069453142914839090434430341799 312819945606360295782005228058369235550
- AST query: 003778425160038348524906247302938706902, 173075880069453142914839090434430341799, 312819945606360295782005228058369235550
- Surplus group: ASI-G030
- Merge key: objective-work/v1/941b45b4a475a78a114da4e82944d2023025eaab
- Merge family: ASI-G030
- Merge role: completion_gate
- Work item count: 7
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 941b45b4a475a78a
- Acceptance: Objective scan filed this gap for ASI-G030. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-080-objective-gap-95ff2e6c9e3e.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every descendant must remain verified with all proof requirements fresh, conclusive, uncontradicted, and satisfied., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Every plan is evaluated for acceptance coverage, assumptions, semantics, dependencies, conflicts, validation/proof feasibility, novelty, and resource/token cost, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-081 Close objective gap packet: ASI-G109, ASI-G110, ASI-G111

- Status: completed
- Completion: manual
- Priority: P1
- Track: self-refill
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Bundle: agent-supervisor/self-improvement/refill
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-refill.todo.md
- Bundle strategy: explicit
- Graph parents: ASI-G080
- Graph depth: 2
- Parallel lane: agent-supervisor/self-improvement/refill
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Changed paths: 
- AST symbols: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Interfaces: 
- Submodules: 
- Generated artifacts: 
- Allow concurrent with: 
- Goal id: ASI-G109
- Canonical task key: task/v1/cdab7afa46caf8b9682cf05265791788c9f595f93cc61414ca2cd675c42cf163
- Canonical task CID: baguqeerazwvxv6sgzl4ls2bm6bjgk6ixrde7lfpzhtdbifgkftlhlrbm6frq
- Evidence obligation key: objective-evidence-packet/v1/f619d2d483f1abd8e92436c545267642eb289af0b98e1352f25ba584d3b64ed1
- Missing evidence: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Embedding query: goal packet goal_packet/self_refill/ipfs_accelerate_py/9d87d026b79d; 020061024173618462922348580596364003627; 065313778069923158401871898168782520190; 119294002389522221490347364495731444366; Prove 020061024173618462922348580596364003627 for Benchmark-driven bounded self-refill; Prove 065313778069923158401871898168782520190 for Benchmark-driven bounded self-refill; Prove 119294002389522221490347364495731444366 for Benchmark-driven bounded self-refill
- AST query: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Surplus group: goal_packet/self_refill/ipfs_accelerate_py/9d87d026b79d
- Merge key: 92b2f43f6a59de24
- Merge family: goal_packet/self_refill/ipfs_accelerate_py/9d87d026b79d
- Merge role: packet_aggregate
- Work item count: 3
- Work scope: goal_subgoal_packet_aggregate; vector_ast_bundle
- Goal packet: goal_packet/self_refill/ipfs_accelerate_py/9d87d026b79d
- Goal packet role: packet_aggregate
- Goal packet goals: ASI-G109, ASI-G110, ASI-G111
- Goal packet task count: 4
- Goal packet work item count: 3
- Candidate kind: goal_packet_aggregate
- Todo vector key: b0cb67f2f9f3b9a2
- Acceptance: Objective scan filed this gap for ASI-G109. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-081-objective-gap-c28d97df1330.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366), and keep the supervisor-fed backlog aligned with the objective heap. This task is part of goal_packet/self_refill/ipfs_accelerate_py/9d87d026b79d; implement a complete, cohesive change that fully advances the packet goals (ASI-G109, ASI-G110, ASI-G111) and covers all the shared packet evidence in one comprehensive pass. Refine the objective heap if the gap needs smaller child goals.

## ASI-082 Produce completion evidence for Efficient and trustworthy supervisor control loop

- Status: completed
- Completion: manual
- Priority: P0
- Track: self-improvement
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, docs/architecture/agent_supervisor_self_improvement.todo.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py -q
- Bundle: agent-supervisor/self-improvement/root
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-root.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: none
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: docs/architecture/agent_supervisor_self_improvement.objectives.md, docs/architecture/agent_supervisor_self_improvement.todo.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Changed paths: 
- AST symbols: agent-supervisor-self-improvement-v1, agent_supervisor_self_improvement.objectives.md, AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G000
- Canonical task key: task/v1/92c6fc4c81c8fc81a52866e2edd2d5f2a119735c2da37d17523a1935bcc7122c
- Canonical task CID: baguqeerasldpytebzd6idjjim3ro3uwv6kqrs424fwrx2f2shimtlpghciwa
- Evidence obligation key: objective-work/v1/5da092dd3d872f72a99219606961bc9a105d9c32
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every descendant must remain verified with all proof requirements fresh, conclusive, uncontradicted, and satisfied., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Every child goal has fresh tree-bound evidence, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: agent-supervisor-self-improvement-v1 agent_supervisor_self_improvement.objectives.md AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- AST query: agent-supervisor-self-improvement-v1, agent_supervisor_self_improvement.objectives.md, AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Surplus group: ASI-G000
- Merge key: objective-work/v1/5da092dd3d872f72a99219606961bc9a105d9c32
- Merge family: ASI-G000
- Merge role: completion_gate
- Work item count: 7
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 5da092dd3d872f72
- Acceptance: Objective scan filed this gap for ASI-G000. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-24-asi-082-objective-gap-db30cef45181.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every descendant must remain verified with all proof requirements fresh, conclusive, uncontradicted, and satisfied., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Every child goal has fresh tree-bound evidence, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.
- Completion gate implementation: `self_improvement_completion.evaluate_self_improvement_root_completion` fixes the ASI-001..ASI-024 producer population, ASI-G010..ASI-G090 child population, four literal root criteria, and two-receipt exhaustive quorum; the E2E matrix is `test/api/test_agent_supervisor_self_improvement_e2e.py`, and the audit index is `data/agent_supervisor/discovery/2026-07-24-asi-082-completion-gate-evidence.md`.
- Lifecycle disposition: Keep this task `todo` for the implementation daemon to close after fresh validation. ASI-G000 remains `provisionally_complete` and supervisor-actionable while any original producer or descendant is not verified and until a separate fresh current-tree root evaluation passes every completion gate.

## ASI-083 Produce completion evidence for Adaptive parallel execution and acceptance throughput

- Status: completed
- Completion: manual
- Priority: P1
- Track: parallelism
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/merge_queue.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_adaptive_resources.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py, test/api/test_agent_supervisor_provider_batch_scheduler.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_resources.py test/api/test_agent_supervisor_provider_batch_scheduler.py test/api/test_agent_supervisor_parallel_acceptance_flow.py -q
- Bundle: agent-supervisor/self-improvement/runtime
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-runtime.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/merge_queue.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_adaptive_resources.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py, test/api/test_agent_supervisor_provider_batch_scheduler.py
- Changed paths: 
- AST symbols: 122080003600146794820964010047426915846, 124037811551945145648172208272779822741, 185033715568272291470322170325431455647
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G060
- Canonical task key: task/v1/108a08fa20d0cea2af8cda7b701e00769d02d4733fda28df35bb0d6fe585da5d
- Canonical task CID: baguqeeraccfar6ra2dhkfl4m3j5xahqao2oqfvdth7ncrxzvxmgw7zmf3joq
- Evidence obligation key: objective-work/v1/e0d8763c9c31e15fe8579faddc628167717f936e
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Resource pools expose backpressure and fair admission, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 122080003600146794820964010047426915846 124037811551945145648172208272779822741 185033715568272291470322170325431455647
- AST query: 122080003600146794820964010047426915846, 124037811551945145648172208272779822741, 185033715568272291470322170325431455647
- Surplus group: ASI-G060
- Merge key: objective-work/v1/e0d8763c9c31e15fe8579faddc628167717f936e
- Merge family: ASI-G060
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: e0d8763c9c31e15f
- Acceptance: Objective scan filed this gap for ASI-G060. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-083-objective-gap-df2b5b9186e1.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Resource pools expose backpressure and fair admission, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-084 Produce completion evidence for High-quality task generation and conflict-aware bundling

- Status: completed
- Completion: manual
- Priority: P1
- Track: task-generation
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, test/api/test_agent_supervisor_bundle_optimizer.py, test/api/test_agent_supervisor_task_quality.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_quality.py test/api/test_agent_supervisor_bundle_optimizer.py -q
- Bundle: agent-supervisor/self-improvement/task-generation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-task-generation.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/task-generation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, test/api/test_agent_supervisor_bundle_optimizer.py, test/api/test_agent_supervisor_task_quality.py
- Changed paths: 
- AST symbols: 061582446926920746660485801841658333166, 127990245919649912156052660092678945998, 187052702852200236079602798955260586139
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G050
- Canonical task key: task/v1/9eac2bf6f3fdc9e311c8aec0ba4ece63c81017363ba3ca33b9b3bbdcd3b4c9ae
- Canonical task CID: baguqeerat2wcx5xt7xe6geoiv3alutwompebafzwhor4um5zwo55zu5uzgxa
- Evidence obligation key: objective-work/v1/b4a99eb7f19006443473c4b528fac9a33e51dc75
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Tasks bind one coherent acceptance/effect subset with predicted scope and costs, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 061582446926920746660485801841658333166 127990245919649912156052660092678945998 187052702852200236079602798955260586139
- AST query: 061582446926920746660485801841658333166, 127990245919649912156052660092678945998, 187052702852200236079602798955260586139
- Surplus group: ASI-G050
- Merge key: objective-work/v1/b4a99eb7f19006443473c4b528fac9a33e51dc75
- Merge family: ASI-G050
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: b4a99eb7f1900644
- Acceptance: Objective scan filed this gap for ASI-G050. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-084-objective-gap-803aec4e5425.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Tasks bind one coherent acceptance/effect subset with predicted scope and costs, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-085 Produce completion evidence for Unified Python, CLI, and MCP supervisor control

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Bundle: agent-supervisor/self-improvement/control
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-control.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/mcp_server/test_agent_supervisor_tools.py, test/test_unified_cli_agent_supervisor.py
- Changed paths: 
- AST symbols: 031486194157679117987393491870400400279, 184125100306462690646212311073240043804, 186773143401179107362964063059661378722
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G070
- Canonical task key: task/v1/1cdc87b40deecbbb9d76fba50d91d37778edc7eac34b643c187a12aa4404f07b
- Canonical task CID: baguqeeradtoipnan53f3xhlw7osq3eoto54o3r7kynfwipaypijkurae6b5q
- Evidence obligation key: objective-work/v1/c52d21e406da75742aa64ccac6f570d34b854c50
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Shared operations have schema and behavior parity across Python, CLI, and MCP, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 031486194157679117987393491870400400279 184125100306462690646212311073240043804 186773143401179107362964063059661378722
- AST query: 031486194157679117987393491870400400279, 184125100306462690646212311073240043804, 186773143401179107362964063059661378722
- Surplus group: ASI-G070
- Merge key: objective-work/v1/c52d21e406da75742aa64ccac6f570d34b854c50
- Merge family: ASI-G070
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: c52d21e406da7574
- Acceptance: Objective scan filed this gap for ASI-G070. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-085-objective-gap-d45c96ecbac5.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Shared operations have schema and behavior parity across Python, CLI, and MCP, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-086 Produce completion evidence for Benchmark-driven bounded self-refill

- Status: completed
- Completion: manual
- Priority: P1
- Track: self-refill
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/self_improvement.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Bundle: agent-supervisor/self-improvement/refill
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-refill.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/refill
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/self_improvement.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Changed paths: 
- AST symbols: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G080
- Canonical task key: task/v1/5d0eef98c887770ea3b1abebec836fb1bf75ce3b086244312a47c1adfd490c0d
- Canonical task CID: baguqeeraluho7ggiq53q5i5rvpv6za3pwg7xltr3bbreimjki7a237kjbqgq
- Evidence obligation key: objective-work/v1/eb80c46cde69abbe6c94dadd5651812c9722f084
- Missing evidence: completion analyzer health, completion criterion coverage, completion exhaustion quorum, completion task closure
- Embedding query: 020061024173618462922348580596364003627 065313778069923158401871898168782520190 119294002389522221490347364495731444366
- AST query: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Surplus group: ASI-G080
- Merge key: objective-work/v1/eb80c46cde69abbe6c94dadd5651812c9722f084
- Merge family: ASI-G080
- Merge role: completion_gate
- Work item count: 4
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: eb80c46cde69abbe
- Acceptance: Objective scan filed this gap for ASI-G080. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-086-objective-gap-d839bf3e1cda.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (completion analyzer health, completion criterion coverage, completion exhaustion quorum, completion task closure), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-087 Produce completion evidence for Benchmark-driven bounded self-refill

- Status: completed
- Completion: manual
- Priority: P1
- Track: self-refill
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/self_improvement.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Bundle: agent-supervisor/self-improvement/refill
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-refill.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/refill
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/self_improvement.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Changed paths: 
- AST symbols: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G080
- Canonical task key: task/v1/96438acf9830e493c9ba19f33a03c65e6181d3b08a4ba634578160e241577f3b
- Canonical task CID: baguqeeraszbyvt4ygdsjhsn2dhztua6glzqydu5qrjf2mncxqfqoeqkxp45q
- Evidence obligation key: objective-work/v1/60ee13a4300b6f29dfd458062949af8a58479cb2
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: A drained board triggers one identity-bound evaluation epoch, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 020061024173618462922348580596364003627 065313778069923158401871898168782520190 119294002389522221490347364495731444366
- AST query: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Surplus group: ASI-G080
- Merge key: objective-work/v1/60ee13a4300b6f29dfd458062949af8a58479cb2
- Merge family: ASI-G080
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 60ee13a4300b6f29
- Acceptance: Objective scan filed this gap for ASI-G080. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-087-objective-gap-b13adfbc29b5.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: A drained board triggers one identity-bound evaluation epoch, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-088 Produce completion evidence for Token-efficient context and end-to-end measurement

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Bundle: agent-supervisor/self-improvement/context
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-context.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Changed paths: 
- AST symbols: 208290439421789408250562066350459701853, 248026856102230635452423769994290240744, 306437607356117177048620815571362227127
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G010
- Canonical task key: task/v1/3c862bd6a02b96aa019ee2db6aebcd3d7a9b9a3ea39f2695a1970db00229c890
- Canonical task CID: baguqeerahsdcxvvafolkuam64lnwv26nhv5jxgr6uopsnfnbs4g3aarjzcia
- Evidence obligation key: objective-work/v1/992e6c459e2a84ecc5369c9bc3bfe2d2a241c5a9
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Required goal, authority, scope, and acceptance context is never truncated, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 208290439421789408250562066350459701853 248026856102230635452423769994290240744 306437607356117177048620815571362227127
- AST query: 208290439421789408250562066350459701853, 248026856102230635452423769994290240744, 306437607356117177048620815571362227127
- Surplus group: ASI-G010
- Merge key: objective-work/v1/992e6c459e2a84ecc5369c9bc3bfe2d2a241c5a9
- Merge family: ASI-G010
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 992e6c459e2a84ec
- Acceptance: Objective scan filed this gap for ASI-G010. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-088-objective-gap-5d10fa4e1423.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Required goal, authority, scope, and acceptance context is never truncated, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-089 Produce completion evidence for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py, test/api/test_agent_supervisor_validation_dag.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py, test/api/test_agent_supervisor_validation_dag.py
- Changed paths: 
- AST symbols: 006818797857632260116084792540150258746, 266404049326363900535699811645710804440, 314133036252270790078901745919131980427
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G040
- Canonical task key: task/v1/30fe22d588903997f58c56b1aa45df277c36ba83b35b9a5de57b16787a33767f
- Canonical task CID: baguqeeragd7cfvmisa4zp5mmk2y2uro7e56dnoudwnnzuxpfpmlhq6rtoz7q
- Evidence obligation key: objective-work/v1/d943cb6487309dd72c6dc4230058503db4f10335
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Schema, authority, patch, path, AST/interface, impact-test, semantic/proof, merge, and freshness gates are explicit, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 006818797857632260116084792540150258746 266404049326363900535699811645710804440 314133036252270790078901745919131980427
- AST query: 006818797857632260116084792540150258746, 266404049326363900535699811645710804440, 314133036252270790078901745919131980427
- Surplus group: ASI-G040
- Merge key: objective-work/v1/d943cb6487309dd72c6dc4230058503db4f10335
- Merge family: ASI-G040
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: d943cb6487309dd7
- Acceptance: Objective scan filed this gap for ASI-G040. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-089-objective-gap-c3718670dcc6.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Schema, authority, patch, path, AST/interface, impact-test, semantic/proof, merge, and freshness gates are explicit, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-090 Produce completion evidence for Paired rollout, stable exports, and operator adoption

- Status: completed
- Completion: manual
- Priority: P2
- Track: rollout
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, test/api/test_agent_supervisor_self_improvement_e2e.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
- Bundle: agent-supervisor/self-improvement/rollout
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-rollout.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G000
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/rollout
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, test/api/test_agent_supervisor_self_improvement_e2e.py
- Changed paths: 
- AST symbols: 109590900757783560279417463762322084165, 146189916032404266364029134505159070240, 300500866741873729474343907613893393545
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G090
- Canonical task key: task/v1/480960be2eabc2ea32d65ee108de4b7150439ba6252c04c169b4e4b86f810eac
- Canonical task CID: baguqeerajaewbprovpboumwwl3qqrxslofiehg5geuwajqljwtslq34bb2wa
- Evidence obligation key: objective-work/v1/bef0ffbeecaeda580a753344fcab98755066d821
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Paired cold/warm, failure, adversarial, parallel, restart, and refill fixtures satisfy every non-negotiable safety gate and the documented token/cache/planning/throughput gates, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 109590900757783560279417463762322084165 146189916032404266364029134505159070240 300500866741873729474343907613893393545
- AST query: 109590900757783560279417463762322084165, 146189916032404266364029134505159070240, 300500866741873729474343907613893393545
- Surplus group: ASI-G090
- Merge key: objective-work/v1/bef0ffbeecaeda580a753344fcab98755066d821
- Merge family: ASI-G090
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: bef0ffbeecaeda58
- Acceptance: Objective scan filed this gap for ASI-G090. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-090-objective-gap-075ea1951c0b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Paired cold/warm, failure, adversarial, parallel, restart, and refill fixtures satisfy every non-negotiable safety gate and the documented token/cache/planning/throughput gates, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-091 Produce completion evidence for Prove 314133036252270790078901745919131980427 for Strict output, code, test, semantic, and proof validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: 
- Outputs: data/agent_supervisor/discovery, docs/architecture/agent_supervisor_self_improvement.objectives.md, ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py, test/api/test_agent_supervisor_validation_dag.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Bundle: agent-supervisor/self-improvement/validation
- Bundle shard: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/bundles/agent-supervisor-self-improvement-validation.todo.md
- Bundle strategy: bounded_objective_generation
- Graph parents: ASI-G040
- Graph depth: 1
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py, test/api/test_agent_supervisor_validation_dag.py
- Changed paths: 
- AST symbols: 314133036252270790078901745919131980427
- Interfaces: 
- Submodules: 
- Generated artifacts: data/agent_supervisor/objective_generation.json
- Allow concurrent with: 
- Goal id: ASI-G100
- Canonical task key: task/v1/3c8f5752323f41077a3289fcfa325ce4441e60c27f2268e353df27b16434cb1a
- Canonical task CID: baguqeerahshvoursh5aqo6rsrh6pums44rcb4ygcp4rgry2t34t3czbuzmna
- Evidence obligation key: objective-work/v1/30849f363ca125ed40dd2bd99a348262925df890
- Missing evidence: Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Proposal admission deterministically checks schema, authority, baseline and candidate identity, non-empty effective change, normalized path safety, and task-owned scope before any expensive validation. Empty or effectless diffs and every out-of-scope path fail closed with bounded typed diagnostics, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.
- Embedding query: 314133036252270790078901745919131980427
- AST query: 314133036252270790078901745919131980427
- Surplus group: ASI-G100
- Merge key: objective-work/v1/30849f363ca125ed40dd2bd99a348262925df890
- Merge family: ASI-G100
- Merge role: completion_gate
- Work item count: 6
- Work scope: bounded_objective_generation
- Goal packet: 
- Goal packet role: 
- Goal packet goals: 
- Goal packet task count: 0
- Goal packet work item count: 0
- Candidate kind: generated_task
- Todo vector key: 30849f363ca125ed
- Acceptance: Objective scan filed this gap for ASI-G100. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v1/discovery/2026-07-25-asi-091-objective-gap-374843191ffe.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (Complete the goal's producing tasks before requesting completion., Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one., Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree., Produce completion evidence for: Proposal admission deterministically checks schema, authority, baseline and candidate identity, non-empty effective change, normalized path safety, and task-owned scope before any expensive validation. Empty or effectless diffs and every out-of-scope path fail closed with bounded typed diagnostics, Require an explicitly healthy analyzer that is safe for completion reasoning., Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.), and keep the supervisor-fed backlog aligned with the objective heap.  Keep the parent goal actionable until fresh proof receipts satisfy its completion gate.

## ASI-092 Freeze the generation-2 benchmark and causal baseline

- Status: completed
- Completion: manual
- Priority: P0
- Track: measurement
- Depends on:
- Goal id: ASI-G200
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_v2_benchmark.py, test/api/test_agent_supervisor_v2_benchmark.py
- Validation: python -m pytest test/api/test_agent_supervisor_v2_benchmark.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/measurement
- Parallel lane: v2-benchmark
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_v2_benchmark.py, test/api/test_agent_supervisor_v2_benchmark.py
- Conflict policy: Keep the benchmark standalone and consume existing v1 receipts through adapters; do not edit package exports or rollout policy.
- Acceptance: Define a closed, versioned paired corpus covering cold, warm, broad-goal, malformed-output, contradictory-input, stale-cache, unavailable-provider, independent-lane, conflicting-lane, failed-validation, restart, drained-board, artifact-pressure, and untrusted-repository fixtures. Freeze repository, tree, objective, provider, capability, policy, fault, and observation identities; join stage latency, queue delay, provider tokens, cache reuse, retries, validation, proof, merge, persistence, idle CPU, and terminal accepted criteria; emit compact causal receipts without prompts, source bodies, decoded output, patches, or nested artifact graphs; and prove deterministic replay, population non-narrowing, non-compensable safety gates, and baseline/candidate pairing.

## ASI-093 Define generation-2 identity, receipt, policy, and promotion contracts

- Status: completed
- Completion: manual
- Priority: P0
- Track: contracts
- Depends on:
- Goal id: ASI-G200
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_v2_contracts.py, test/api/test_agent_supervisor_v2_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_v2_contracts.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/contracts
- Parallel lane: v2-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_v2_contracts.py, test/api/test_agent_supervisor_v2_contracts.py
- Conflict policy: Add provider-free immutable contracts only; defer runtime wiring, package exports, CLI, MCP, and objective lifecycle integration.
- Acceptance: Add strict versioned contracts for semantic dependency identities, stage events, evidence references, operation capabilities, uncertainty and disagreement, promotion vectors, artifact bounds, refill epochs, target descriptors, and typed failures. Bind every result to repository/tree, objective/task, policy, producer, capability, environment, and semantic dependencies; distinguish diagnostic, proposal, validation, proof, merge, mutation, and completion authority; reject unknown fields, detached references, forged summaries, path escapes, over-depth or over-byte payloads, and composite scores that attempt to compensate for a failed safety gate.

## ASI-094 Add provider-native token and accepted-criterion attribution

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: ASI-092, ASI-093
- Goal id: ASI-G210
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_token_ledger.py, test/api/test_agent_supervisor_token_ledger.py
- Validation: python -m pytest test/api/test_agent_supervisor_token_ledger.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/context
- Parallel lane: token-ledger
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_token_ledger.py, test/api/test_agent_supervisor_token_ledger.py
- Conflict policy: Keep attribution in a standalone ledger and adapt existing efficiency receipts without changing context selection yet.
- Acceptance: Attribute provider-native input, output, reused, speculative, tool, retry, and failed-attempt tokens to one stage, task, attempt, context identity, cache decision, validation result, and terminal accepted criterion. Calibrate fallback tokenizers by provider/model envelope, reconcile every lifecycle event exactly once, charge rejected and abandoned work, expose cost per accepted criterion and evidence gain per thousand tokens, and reject missing, duplicated, negative, foreign-bound, or terminally unattributed usage.

## ASI-095 Build prefix-stable context capsules and prompt-cache reuse

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: ASI-094
- Goal id: ASI-G210
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_prefix_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_prefix_context.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/context
- Parallel lane: prefix-context
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_prefix_context.py
- Conflict policy: This task owns context layout and provider-prefix reuse; run before value-of-information selection because both change capsule construction.
- Acceptance: Arrange every stage input as a canonical stable policy/objective prefix, stable task core, and volatile evidence delta while preserving required authority and acceptance fields. Bind provider prompt-cache or KV-cache identities and actual reuse when available, derive a conservative reusable-token estimate otherwise, invalidate exactly when a semantic prefix dependency changes, prohibit reuse across authority or target boundaries, and demonstrate at least 70 percent eligible stable-prefix reuse on warm fixtures without evidence loss or stale context.

## ASI-096 Add value-of-information evidence selection and bounded expansion

- Status: completed
- Completion: manual
- Priority: P0
- Track: token-efficiency
- Depends on: ASI-095
- Goal id: ASI-G210
- Outputs: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/analysis_retrieval.py, test/api/test_agent_supervisor_evidence_value.py
- Validation: python -m pytest test/api/test_agent_supervisor_evidence_value.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_analysis_retrieval.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/context
- Parallel lane: evidence-value
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/analysis_retrieval.py, test/api/test_agent_supervisor_evidence_value.py
- Conflict policy: Extend the prefix-stable compiler after ASI-095; retain deterministic retrieval as an input signal rather than replacing it.
- Acceptance: Rank optional evidence by expected decision change and uncertainty reduction divided by token, latency, invalidation, and expansion cost. Required evidence is never auctioned away; exclusions and uncertainty are explicit; on-demand expansion requires a named unresolved question and a content-addressed handle; redundant evidence is diversity-penalized; and paired fixtures show at least 40 percent lower median input tokens per accepted criterion and 60 percent lower retry-input tokens with unchanged required coverage and safety.

## ASI-097 Add an asynchronous capability-negotiated analysis transport

- Status: completed
- Completion: manual
- Priority: P0
- Track: datasets-offload
- Depends on: ASI-093
- Goal id: ASI-G220
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_transport.py, test/api/test_agent_supervisor_analysis_transport.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_transport.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/analysis
- Parallel lane: analysis-transport
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_transport.py, test/api/test_agent_supervisor_analysis_transport.py
- Conflict policy: Add a provider-independent transport and fake-provider fixtures; do not modify concrete ipfs_datasets_py adapters in this task.
- Acceptance: Define bounded asynchronous discovery and dispatch for local and optional ipfs_datasets_py reasoning operations with schema/version negotiation, deadlines, cancellation, progress, batching, health, backpressure, and deterministic fallback. Requests carry compact questions and artifact references; results carry bounded evidence references, provenance, cost, truncation, and non-authority verdicts. Import and capability discovery must remain lazy and side-effect-free, and timeout, cancellation, malformed output, capability drift, and provider loss must terminate with typed bounded outcomes.

## ASI-098 Route AST, GraphRAG, premise, logic, and proof-candidate analysis through one registry

- Status: completed
- Completion: manual
- Priority: P0
- Track: datasets-offload
- Depends on: ASI-097
- Goal id: ASI-G220
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_operation_registry.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_analysis_operation_registry.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_operation_registry.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/analysis
- Parallel lane: analysis-operations
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_operation_registry.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_analysis_operation_registry.py
- Conflict policy: This task owns concrete provider registration; preserve existing Hammer, kernel, and completion authority and retain operation-specific compatibility adapters.
- Acceptance: Register typed local and optional datasets producers for AST/symbol impact, GraphRAG retrieval, premise selection, contradiction search, legal/logic translation candidates, and proof/counterexample candidate analysis. Each operation declares cache semantics, bounds, provenance, authority, fallback, batching, and capability requirements; TDFOL, DCEC, FLogic, modal/deontic, frame, KG, and event-calculus families remain distinguishable; equivalent local and remote results normalize to one reference shape; and no operation can mutate the repository, choose validation omissions, or promote its own candidate.

## ASI-099 Normalize provenance, disagreement, and fallback receipts

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis
- Depends on: ASI-098
- Goal id: ASI-G220
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_consensus.py, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, test/api/test_agent_supervisor_analysis_consensus.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_consensus.py test/api/test_agent_supervisor_analysis_pipeline.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/analysis
- Parallel lane: analysis-consensus
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_consensus.py, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, test/api/test_agent_supervisor_analysis_consensus.py
- Conflict policy: Integrate normalized provider outcomes in the analysis pipeline; do not convert consensus into proof or completion authority.
- Acceptance: Produce one compact typed receipt for local/datasets agreement, disagreement, degraded fallback, partial result, and independent validation. Preserve source, dataset, graph, chunk, producer, model, policy, capability, and tree provenance; resolve disagreements only through deterministic policy or an independent validator; expose residual uncertainty; exclude failed, stale, inconclusive, and proposal-only outcomes from completion; and prove equivalent cold/warm behavior, bounded payloads, explicit fallback, and no confidence-based authority escalation.

## ASI-100 Build a tiered dependency-aware content-addressed runtime store

- Status: completed
- Completion: manual
- Priority: P0
- Track: caching
- Depends on: ASI-093
- Goal id: ASI-G250
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime_cas.py, ipfs_accelerate_py/agent_supervisor/artifact_store.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_runtime_cas.py, test/api/test_agent_supervisor_artifact_store.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_runtime_cas.py test/api/test_agent_supervisor_artifact_store.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/cache
- Parallel lane: runtime-cas
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime_cas.py, ipfs_accelerate_py/agent_supervisor/artifact_store.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_runtime_cas.py, test/api/test_agent_supervisor_artifact_store.py, test/api/test_agent_supervisor_cache_coordinator.py
- Conflict policy: This task owns shared CAS envelopes and dependency metadata; preserve namespace-specific schemas and authority classes.
- Acceptance: Add process-local, host-durable, optional shared immutable, and authoritative projection tiers with canonical artifact identities, dependency edges, producer/policy/capability versions, freshness, namespace authority, and invalidation traversal. Reuse existing caches through adapters, never merge drafts with authoritative receipts, reject cycles and forged dependencies, recover corrupt entries, and demonstrate exact warm reuse plus invalidation of only affected descendants after a semantic dependency change.

## ASI-101 Generalize cross-process and distributed single-flight coordination

- Status: completed
- Completion: manual
- Priority: P0
- Track: caching
- Depends on: ASI-097, ASI-100
- Goal id: ASI-G250
- Outputs: ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, ipfs_accelerate_py/agent_supervisor/lease_coordination.py, test/api/test_agent_supervisor_distributed_singleflight.py
- Validation: python -m pytest test/api/test_agent_supervisor_distributed_singleflight.py test/api/test_agent_supervisor_cache_coordinator.py test/api/test_agent_supervisor_lease_coordination.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/cache
- Parallel lane: distributed-singleflight
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, ipfs_accelerate_py/agent_supervisor/lease_coordination.py, test/api/test_agent_supervisor_distributed_singleflight.py
- Conflict policy: Generalize existing namespace implementations after the CAS contract lands; preserve proof-specific fencing and owner-attested outcomes.
- Acceptance: Collapse identical analysis, context, plan, provider, proof, validation, and merge-classification misses behind semantic keys and fenced leases across threads, processes, and optional hosts. Heartbeat and owner failure must transfer or fail deterministically; followers receive one attested bounded outcome; cancellation and deadlines remain member-specific; stale or foreign leases cannot publish; and paired fixtures achieve at least 60 percent duplicate-miss collapse with less than 5 percent duplicate compute and no stale authoritative hit.

## ASI-102 Bound persistence, retention, compaction, and payload projection

- Status: completed
- Completion: manual
- Priority: P0
- Track: persistence
- Depends on: ASI-100
- Goal id: ASI-G250
- Outputs: ipfs_accelerate_py/agent_supervisor/artifact_store.py, ipfs_accelerate_py/agent_supervisor/event_log.py, test/api/test_agent_supervisor_bounded_persistence.py
- Validation: python -m pytest test/api/test_agent_supervisor_bounded_persistence.py test/api/test_agent_supervisor_artifact_store.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/persistence
- Parallel lane: bounded-persistence
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/artifact_store.py, ipfs_accelerate_py/agent_supervisor/event_log.py, test/api/test_agent_supervisor_bounded_persistence.py
- Conflict policy: Own payload projection and storage lifecycle after ASI-100; do not edit cache lease coordination.
- Acceptance: Enforce a 256 KiB receipt bound, 1 MiB routine projection bound, configurable aggregate quotas, retention classes, incremental compaction, observable eviction, and crash-safe manifests. Store decoded model text, source bodies, proof traces, checkpoints, and nested artifact graphs once as referenced blobs; prevent recursive or duplicate embedding; give negative and inconclusive entries finite TTLs; and verify bounded shutdown, restart recovery, disk-pressure degradation, content integrity, and stable references after compaction.

## ASI-103 Add a typed goal grammar, quality linter, and uncertainty debt

- Status: completed
- Completion: manual
- Priority: P0
- Track: goal-refinement
- Depends on: ASI-093
- Goal id: ASI-G230
- Outputs: ipfs_accelerate_py/agent_supervisor/goal_quality.py, test/api/test_agent_supervisor_goal_quality.py
- Validation: python -m pytest test/api/test_agent_supervisor_goal_quality.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/goals
- Parallel lane: goal-quality
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/goal_quality.py, test/api/test_agent_supervisor_goal_quality.py
- Conflict policy: Add standalone goal contracts and linting; defer objective-heap mutation and adaptive refinement wiring.
- Acceptance: Represent outcome, scope, assumptions, non-goals, acceptance criteria, evidence producers, validation, freshness, resource envelope, uncertainty, unsupported semantics, and refinement budget. Lint circular acceptance, unbounded or conflicting scope, hidden authority, unverifiable evidence, orphan dependencies, ambiguous completion, and excessive breadth; emit repairable typed debt; preserve frozen root identity; and prove canonical serialization, deterministic scoring, adversarial rejection, and compatibility projection from current objective Markdown.

## ASI-104 Add bounded AND/OR plan search with hard-constraint pruning

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: ASI-096, ASI-099, ASI-103
- Goal id: ASI-G230
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, test/api/test_agent_supervisor_and_or_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_and_or_planner.py test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_plan_evaluator.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/planning
- Parallel lane: and-or-planning
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, test/api/test_agent_supervisor_and_or_planner.py
- Conflict policy: This task owns candidate search and evaluation; preserve formal compiler and validator authority and keep the deterministic baseline mandatory.
- Acceptance: Compile typed goals into bounded AND nodes for jointly required obligations and OR nodes for alternative producers. Include a deterministic baseline and optional LLM, Leanstral, and analysis-provider branches under one frozen context; prune authority, scope, dependency, resource, freshness, validation, and proof violations before soft scoring; compare remaining branches by evidence coverage, uncertainty reduction, critical path, conflict risk, cost, and historical failure; enforce search depth/node/token/time budgets and deterministic tie-breaking; and meet the v2 valid-first-plan or invalid-branch promotion gate without hard-constraint violations.

## ASI-105 Add counterexample-driven delta replanning and branch-failure memory

- Status: completed
- Completion: manual
- Priority: P0
- Track: planning
- Depends on: ASI-104
- Goal id: ASI-G230
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/plan_failure_memory.py, test/api/test_agent_supervisor_delta_replanning.py
- Validation: python -m pytest test/api/test_agent_supervisor_delta_replanning.py test/api/test_agent_supervisor_formal_replanner.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/planning
- Parallel lane: delta-replanning
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/formal_replanner.py, ipfs_accelerate_py/agent_supervisor/plan_failure_memory.py, test/api/test_agent_supervisor_delta_replanning.py
- Conflict policy: Run after AND/OR planning; store typed branch features and failure signatures rather than provider reasoning text.
- Acceptance: Bind counterexamples, failed constraints, validation signatures, capability loss, conflicts, and resource infeasibility to the smallest dependent plan suffix. Preserve unaffected accepted branches, invalidate stale dependencies, reuse identical diagnostics with finite backoff, and learn only typed failure features scoped by tree, policy, environment, and planner version. Changed evidence must reopen the relevant branch; unchanged delivery noise must not; and restart, tampering, poisoning, deadline, and bounded-repair tests must pass without storing chain-of-thought or full prompts.

## ASI-106 Make goal refinement event-driven and information-gain-aware

- Status: completed
- Completion: manual
- Priority: P0
- Track: goal-refinement
- Depends on: ASI-103, ASI-105
- Goal id: ASI-G230
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, test/api/test_agent_supervisor_event_goal_refinement.py
- Validation: python -m pytest test/api/test_agent_supervisor_event_goal_refinement.py test/api/test_agent_supervisor_adaptive_goal_refiner.py test/api/test_agent_supervisor_goal_refinement_verification.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/goals
- Parallel lane: event-goal-refinement
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, test/api/test_agent_supervisor_event_goal_refinement.py
- Conflict policy: This task owns objective refinement policy after plan-delta semantics are fixed; preserve transactional root revision admission.
- Acceptance: Trigger bounded refinement only from meaningful counterexample, stale evidence, uncovered criterion, capability, interface, conflict, resource, uncertainty, or operator-revision events. Estimate information gain and expected downstream cost, freeze the root and admitted assumptions, suppress unchanged event churn with persisted backoff, independently verify child sufficiency, and transactionally commit only a quality-linted delta. Polls without a changed semantic event must perform no model call or objective write.

## ASI-107 Harden the output, patch, authority, and untrusted-repository envelope

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: ASI-093
- Goal id: ASI-G240
- Outputs: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, test/api/test_agent_supervisor_untrusted_proposal.py
- Validation: python -m pytest test/api/test_agent_supervisor_untrusted_proposal.py test/api/test_agent_supervisor_proposal_validation.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/validation
- Parallel lane: untrusted-proposal
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, test/api/test_agent_supervisor_untrusted_proposal.py
- Conflict policy: Own fail-fast proposal admission only; defer command execution, semantic validation, proof, and merge assembly.
- Acceptance: Treat provider output and repository content as untrusted data. Strictly validate output schema, canonical IDs, authority, baseline/candidate identity, expected effects, non-empty semantic patch, normalized paths, symlinks, hardlinks, submodules, binaries, secrets, generated files, size/depth/count bounds, protected paths, and task-owned scope before dispatching any expensive check. Add prompt-injection, forged-receipt, path-race, encoding, archive, no-op, test-deletion, validation-weakening, and scope-confusion fixtures with bounded typed diagnostics and zero side effects.

## ASI-108 Add hermetic impact, differential, mutation, and flaky validation

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: ASI-099, ASI-107
- Goal id: ASI-G240
- Outputs: ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/validation_runtime.py, test/api/test_agent_supervisor_hermetic_validation.py
- Validation: python -m pytest test/api/test_agent_supervisor_hermetic_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_validation_scheduler.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/validation
- Parallel lane: hermetic-validation
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/validation_runtime.py, test/api/test_agent_supervisor_hermetic_validation.py
- Conflict policy: This task owns validation execution and classification; preserve task-declared semantic/proof gate selection and proposal fail-fast ordering.
- Acceptance: Execute the complete selected DAG in a hermetic resource-bounded environment with pinned command, toolchain, environment, network, filesystem, timeout, and cancellation identity. Combine dependency impact with contract, differential, metamorphic, and mutation checks; seed transitive defects; classify deterministic failure, flaky, timeout, infrastructure failure, and inconclusive separately; prevent an intermittent pass from granting authority; reuse exact diagnostics; and achieve zero escaped seeded defects with at least 30 percent lower median time to first useful failure.

## ASI-109 Assemble authoritative post-merge semantic and proof evidence

- Status: completed
- Completion: manual
- Priority: P0
- Track: validation
- Depends on: ASI-105, ASI-108
- Goal id: ASI-G240
- Outputs: ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, test/api/test_agent_supervisor_post_merge_evidence.py
- Validation: python -m pytest test/api/test_agent_supervisor_post_merge_evidence.py test/api/test_agent_supervisor_semantic_validation_pipeline.py test/api/test_agent_supervisor_proof_merge_gate.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/validation
- Parallel lane: post-merge-evidence
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, test/api/test_agent_supervisor_post_merge_evidence.py
- Conflict policy: This is the sole v2 integration lane for merged-tree evidence; do not let pre-merge candidates or provider verdicts acquire completion authority.
- Acceptance: Rebuild the evidence graph on the actual merged tree and assemble one content-addressed receipt binding proposal admission, complete executed validation, semantic and protocol checks, legal/logic and theorem obligations, accepted proof receipts, merge identity, freshness, and exact covered acceptance criteria. Re-derive every authority claim, reject missing or extra gates, stale or foreign evidence, contradictory proofs, pre-merge-only results, and changed merge trees, and close merge/completion authority on any failure.

## ASI-110 Calibrate task split and coalesce decisions from measured cost

- Status: completed
- Completion: manual
- Priority: P1
- Track: task-generation
- Depends on: ASI-092, ASI-103
- Goal id: ASI-G260
- Outputs: ipfs_accelerate_py/agent_supervisor/task_quality.py, test/api/test_agent_supervisor_task_granularity.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_granularity.py test/api/test_agent_supervisor_task_quality.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/tasks
- Parallel lane: task-granularity
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_quality.py, test/api/test_agent_supervisor_task_granularity.py
- Conflict policy: Own candidate sizing and semantic identity; defer bundle layout and runtime scheduling.
- Acceptance: Bind each task to one exact acceptance subset and predicted context, files, symbols, interfaces, validation, proof, resource cost, and merge fate. Split work exceeding measured acceptance, context, scope, validation, proof, or merge-risk bounds; coalesce only compatible tiny work; preserve complete source coverage and dependencies; use historical measurements only under matching tree/policy/toolchain features; and prove deterministic identity, zero duplicate semantic tasks, exact completion propagation, and fewer model calls per accepted criterion on paired fixtures.

## ASI-111 Build conflict, resource, context, and validation-aware bundle planning

- Status: completed
- Completion: manual
- Priority: P1
- Track: task-generation
- Depends on: ASI-100, ASI-104, ASI-110
- Goal id: ASI-G260
- Outputs: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_bundle_optimizer_v2.py
- Validation: python -m pytest test/api/test_agent_supervisor_bundle_optimizer_v2.py test/api/test_agent_supervisor_bundle_optimizer.py test/api/test_agent_supervisor_conflict_graph.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/bundling
- Parallel lane: bundle-planner-v2
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_bundle_optimizer_v2.py
- Conflict policy: This task owns pending-work bundle projection; active task identity and completion scope are immutable.
- Acceptance: Optimize bundles over prerequisite depth, path/symbol/interface conflicts, shared immutable context, provider batchability, validation reuse, artifact locality, resource class, and merge pressure. Preserve critical-path width and exact task coverage; add only necessary conflict serialization; dynamically rebundle pending work after typed changes without mutating active work; and demonstrate deterministic plans, no ambiguous completion propagation, no conflict-rate regression, and lower context/model cost than title- or goal-only grouping.

## ASI-112 Add adaptive stage scheduling, fair work stealing, batching, and backpressure

- Status: completed
- Completion: manual
- Priority: P0
- Track: parallelism
- Depends on: ASI-101, ASI-111
- Goal id: ASI-G260
- Outputs: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, test/api/test_agent_supervisor_stage_scheduler_v2.py
- Validation: python -m pytest test/api/test_agent_supervisor_stage_scheduler_v2.py test/api/test_agent_supervisor_resource_scheduler.py test/api/test_agent_supervisor_provider_batch_scheduler.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/runtime
- Parallel lane: adaptive-stage-scheduler
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, test/api/test_agent_supervisor_stage_scheduler_v2.py
- Conflict policy: Own local stage admission and batching; defer distributed lease and merge-train integration to ASI-113.
- Acceptance: Model analysis, inference, proof, validation, merge, and persistence pools independently; combine critical-path priority with starvation-bounded work stealing; batch compatible requests through shared model/prover services while preserving per-member budgets, cancellation, identities, and receipts; adapt ceilings to CPU, RAM, GPU memory, provider capacity, queue shape, disk, artifact pressure, and merge debt; apply hysteresis and task-generation backpressure; and reach at least 3x one-lane accepted throughput with less than 5 percent duplicate compute and stable resources.

## ASI-113 Add distributed lane execution and merge-train fencing

- Status: completed
- Completion: manual
- Priority: P1
- Track: parallelism
- Depends on: ASI-109, ASI-112
- Goal id: ASI-G260
- Outputs: ipfs_accelerate_py/agent_supervisor/lease_coordination.py, ipfs_accelerate_py/agent_supervisor/bundle_supervisor.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, test/api/test_agent_supervisor_distributed_lanes.py
- Validation: python -m pytest test/api/test_agent_supervisor_distributed_lanes.py test/api/test_agent_supervisor_lease_coordination.py test/api/test_agent_supervisor_merge_train.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/runtime
- Parallel lane: distributed-lanes
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/lease_coordination.py, ipfs_accelerate_py/agent_supervisor/bundle_supervisor.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, test/api/test_agent_supervisor_distributed_lanes.py
- Conflict policy: Integrate only after local scheduling and post-merge evidence contracts are fixed; retain single-host fallback.
- Acceptance: Dispatch optional remote lanes from immutable input artifacts and explicit capability/environment receipts under expiring leases and fencing epochs. Suppress duplicate work, heartbeat active ownership, reject stale publication, quarantine foreign or malformed results, preserve cancellation, and serialize accepted commits through the merge train and post-merge evidence gate. Prove partition, worker loss, lease theft, duplicate completion, capability drift, conflicting work, restart, and deterministic local-fallback behavior.

## ASI-114 Define one versioned control capability catalog and event cursor

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: ASI-093
- Goal id: ASI-G270
- Outputs: ipfs_accelerate_py/agent_supervisor/control_contracts.py, test/api/test_agent_supervisor_control_catalog.py, test/api/test_agent_supervisor_control_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_catalog.py test/api/test_agent_supervisor_control_contracts.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/control
- Parallel lane: control-catalog
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, test/api/test_agent_supervisor_control_catalog.py, test/api/test_agent_supervisor_control_contracts.py
- Conflict policy: Define provider-free catalog contracts first; defer Python service, CLI, and MCP adapter wiring.
- Acceptance: Define one immutable catalog for capabilities, health, status, metrics, goals, tasks, bundles, lanes, events, receipts, caches, objective preview/refine/reconcile, refill, plan, lifecycle, retry, cancel, quarantine, artifact query, and validation replay. Each operation declares schemas, authority, target descriptor, roots, bounds, pagination or event cursor, dry-run, idempotency, leases, fencing, backend capability, degradation, and audit receipt; discovery is lazy and side-effect-free; and version negotiation, cursor replay, unknown operation, unsupported capability, and bound tests pass.

## ASI-115 Enforce Python, CLI, and MCP operation-schema conformance

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: ASI-114
- Goal id: ASI-G270
- Outputs: ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_control_conformance_v2.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_conformance_v2.py test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/control
- Parallel lane: control-conformance
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_control_conformance_v2.py
- Conflict policy: This task owns transport adapters; every adapter must invoke the shared control service rather than shelling into another surface.
- Acceptance: Generate or conformance-test Python calls, `ipfs-accelerate agent` commands, and MCP tools against every catalog operation. Normalize target, request, result, pagination, event cursor, errors, exit status, timeout, cancellation, and capability degradation; prove equivalent canonical results and effects; keep package import and MCP tools/list provider-free and process-free; prohibit CLI-string dispatch from MCP; and fail catalog publication when any operation is missing, extra, schema-drifted, or behaviorally inconsistent.

## ASI-116 Add policy authorization, dry-run effects, idempotency, and transactions to every mutation

- Status: completed
- Completion: manual
- Priority: P0
- Track: control
- Depends on: ASI-115
- Goal id: ASI-G270
- Outputs: ipfs_accelerate_py/agent_supervisor/authorization_logic.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, test/api/test_agent_supervisor_control_transactions.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_transactions.py test/api/test_agent_supervisor_authorization_logic.py test/api/test_agent_supervisor_control_lifecycle.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/control
- Parallel lane: control-transactions
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/authorization_logic.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, test/api/test_agent_supervisor_control_transactions.py
- Conflict policy: Apply mutation policy only after surface parity; preserve read and proposal operations as non-mutating authority classes.
- Acceptance: Require every real mutation to carry an exact permit bound to operation, caller, repository/state roots, tree/objective/policy revisions, expected effects, idempotency key, live lease, and fencing epoch. Dry-run computes bounded effects with proposal authority and performs no mutation. Multi-step operations expose compare-and-swap transaction state, durable result replay, and typed compensation or repair after partial failure. Reject key reuse with changed effects, stale targets, root escapes, missing authority, lease loss, and transport-specific bypass across Python, CLI, and MCP.

## ASI-117 Replace idle polling and full-state rewrites with event-driven delta checkpoints

- Status: completed
- Completion: manual
- Priority: P0
- Track: reliability
- Depends on: ASI-100, ASI-114
- Goal id: ASI-G280
- Outputs: ipfs_accelerate_py/agent_supervisor/event_log.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_event_driven_runtime.py
- Validation: python -m pytest test/api/test_agent_supervisor_event_driven_runtime.py test/api/test_agent_supervisor_incremental_runtime.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/reliability
- Parallel lane: event-driven-runtime
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/event_log.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_event_driven_runtime.py
- Conflict policy: This task owns daemon wakeup and state projection; preserve a configurable low-frequency safety timer and current CLI compatibility.
- Acceptance: Wake from task-board, objective, repository, child-process, lease, validation, provider-capacity, policy, and observation-window events using canonical cursors, with a low-frequency safety timer for missed notifications. Read bounded metadata before expensive scans, materialize only changed projection deltas, and write nothing when state is unchanged. Prove no lost or duplicate events, cursor recovery after restart, safe fallback on unsupported filesystems, less than 2 percent of one CPU core over a 10-minute drained-board fixture, and zero unchanged state writes.

## ASI-118 Add bounded crash recovery, fault injection, and state repair evidence

- Status: completed
- Completion: manual
- Priority: P0
- Track: reliability
- Depends on: ASI-102, ASI-109, ASI-117
- Goal id: ASI-G280
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_recovery.py, ipfs_accelerate_py/agent_supervisor/event_log.py, test/api/test_agent_supervisor_fault_recovery_v2.py
- Validation: python -m pytest test/api/test_agent_supervisor_fault_recovery_v2.py test/api/test_agent_supervisor_supervisor_watchdog.py test/api/test_agent_supervisor_process_tree_fencing.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/reliability
- Parallel lane: fault-recovery
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_recovery.py, ipfs_accelerate_py/agent_supervisor/event_log.py, test/api/test_agent_supervisor_fault_recovery_v2.py
- Conflict policy: Integrate recovery after delta checkpoints and merged-tree evidence; do not infer success from process exit or partial artifacts.
- Acceptance: Fault-inject process crashes, kill escalation, partial event/checkpoint writes, stale leases, corrupt caches, duplicate events, provider loss, disk-full and slow-disk states, interrupted validation, interrupted merge, and restart during refill. Recover from the last valid content-addressed checkpoint and event cursor, fence stale actors, repair or quarantine partial state, preserve accepted merged-tree evidence, bound retries and storage, emit an exact repair receipt, and fail closed when deterministic recovery is impossible.

## ASI-119 Build reward-hacking-resistant multi-objective self-evaluation

- Status: completed
- Completion: manual
- Priority: P0
- Track: self-refill
- Depends on: ASI-092, ASI-109, ASI-113, ASI-116, ASI-118
- Goal id: ASI-G290
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement_v2.py, test/api/test_agent_supervisor_self_improvement_v2_benchmark.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_v2_benchmark.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/refill
- Parallel lane: v2-self-evaluation
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement_v2.py, test/api/test_agent_supervisor_self_improvement_v2_benchmark.py
- Conflict policy: Add a v2 evaluator beside v1 and consume owner-produced receipts; do not weaken or reinterpret v1 rollout evidence.
- Acceptance: Evaluate the complete frozen v2 population as a Pareto vector over safety, tokens, context reuse, planning, analysis, cache, validation, task quality, throughput, persistence, idle reliability, control, and refill. Recompute every component from producer receipts, run bounded ablations to identify causal contributors, detect denominator shifts, omitted hard fixtures, metric substitution, duplicated evidence, cherry-picked tasks, cache warming leakage, and work moved outside the measurement window, and force shadow on any non-compensable or population failure.

## ASI-120 Generate successor goals only from typed residuals

- Status: completed
- Completion: manual
- Priority: P0
- Track: self-refill
- Depends on: ASI-106, ASI-119
- Goal id: ASI-G290
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement_v2.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, ipfs_accelerate_py/agent_supervisor/task_quality.py, test/api/test_agent_supervisor_v2_successor_generation.py
- Validation: python -m pytest test/api/test_agent_supervisor_v2_successor_generation.py test/api/test_agent_supervisor_self_improvement_refill.py test/api/test_agent_supervisor_task_quality.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/refill
- Parallel lane: v2-successor-generation
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement_v2.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, ipfs_accelerate_py/agent_supervisor/task_quality.py, test/api/test_agent_supervisor_v2_successor_generation.py
- Conflict policy: Own proposal and admission after the v2 evaluator; defer durable materialization to ASI-121.
- Acceptance: Convert only typed benchmark residuals, regressions, stale evidence, bottlenecks, unsupported capabilities, and ablation findings into goal candidates. Apply goal-quality linting, semantic novelty distance, exact identity deduplication, historical/cooldown rejection, unsupported-dependency checks, and finite confidence, depth, breadth, open-work, token, goal, and task budgets. Generic improvement prose, completed evidence work, delivery noise, and unchanged residuals create no proposal; every rejection is bounded and typed; and one residual cannot fan out into duplicate goals or tasks.

## ASI-121 Materialize refill epochs transactionally with healthy-exhaustion quorum

- Status: completed
- Completion: manual
- Priority: P0
- Track: self-refill
- Depends on: ASI-120
- Goal id: ASI-G290
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement_v2.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, test/api/test_agent_supervisor_v2_refill_epoch.py
- Validation: python -m pytest test/api/test_agent_supervisor_v2_refill_epoch.py test/api/test_agent_supervisor_self_improvement_refill.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/refill
- Parallel lane: v2-refill-epoch
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement_v2.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, test/api/test_agent_supervisor_v2_refill_epoch.py
- Conflict policy: This task is the sole v2 objective/task-board materialization lane and must preserve compare-and-swap fencing.
- Acceptance: Bind an epoch to repository tree, objective and board revisions, benchmark policy, capabilities, operation catalog, storage policy, and observation window. Preview one exact goal/task delta, enforce maxima of 8 goals and 24 tasks, commit heap and board through compare-and-swap with a durable journal, and map every admitted goal and task exactly once. Exact replay performs zero provider, proposal, write, or task work. If no candidate survives, require independent fresh healthy exhaustive receipts, persist a wait state, and suppress another epoch for 6 hours unless a declared meaningful trigger changes.

## ASI-122 Add the generation-2 paired rollout and automatic rollback gate

- Status: completed
- Completion: manual
- Priority: P0
- Track: rollout
- Depends on: ASI-096, ASI-099, ASI-109, ASI-113, ASI-116, ASI-121
- Goal id: ASI-G290
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement_v2_rollout.py, test/api/test_agent_supervisor_self_improvement_v2_rollout.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_v2_rollout.py test/api/test_agent_supervisor_self_improvement_v2_benchmark.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/rollout
- Parallel lane: v2-rollout
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement_v2_rollout.py, test/api/test_agent_supervisor_self_improvement_v2_rollout.py
- Conflict policy: Add a provider-free rollout contract beside v1; promotion cannot mutate goals, code, or policy and all failed gates force shadow.
- Acceptance: Recompute the complete v2 report and enforce zero safety, authority, escaped-defect, stale-hit, idempotency, population, and artifact-bound failures plus the documented token, context, planning, analysis, cache, validation, throughput, persistence, idle, control, and refill thresholds. Support off, shadow, assist, and policy-approved automatic modes; require a later separate current-tree evaluation before automatic use; bind desired and effective mode to policy and capability identities; and automatically return affected behavior to shadow on any stale binding or regression.

## ASI-123 Publish stable v2 APIs, controls, operating profiles, and migration guidance

- Status: completed
- Completion: manual
- Priority: P1
- Track: rollout
- Depends on: ASI-122
- Goal id: ASI-G290
- Outputs: ipfs_accelerate_py/agent_supervisor/__init__.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, test/api/test_agent_supervisor_v2_public_api.py
- Validation: python -m pytest test/api/test_agent_supervisor_v2_public_api.py test/api/test_agent_supervisor_control_conformance_v2.py test/api/test_agent_supervisor_self_improvement_v2_rollout.py -q
- Board namespace: agent-supervisor-self-improvement-v2
- Bundle: agent-supervisor/self-improvement-v2/rollout
- Parallel lane: v2-public-integration
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/__init__.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, test/api/test_agent_supervisor_v2_public_api.py
- Conflict policy: Central exports, adapters, and documentation land last; preserve v1 compatibility and keep optional providers lazy.
- Acceptance: Export only reviewed provider-free v2 contracts and control entry points through a stable lazy manifest; expose equivalent Python, CLI, and MCP discovery and control; document smoke, production, distributed, degraded, recovery, refill, rollback, and migration profiles with measured resource ceilings rather than fixed worker folklore; retain v1 compatibility; and prove in a fresh interpreter that import and capability discovery start no process, load no optional datasets/model/prover provider, and preserve canonical object and operation identities.

## ASI-124 Define the canonical decision envelope and pinned artifact references

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-ir
- Depends on: ASI-100, ASI-114
- Goal id: ASI-G310
- Outputs: ipfs_accelerate_py/agent_supervisor/decision_contracts.py, test/api/test_agent_supervisor_decision_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_decision_contracts.py test/api/test_agent_supervisor_control_contracts.py test/api/test_agent_supervisor_artifact_store.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/ir
- Parallel lane: decision-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/decision_contracts.py, test/api/test_agent_supervisor_decision_contracts.py
- Conflict policy: Add provider-free contracts beside existing context and control contracts; do not wire optional datasets providers, daemon dispatch, package exports, or mutation behavior.
- Acceptance: Define immutable versioned bounded `PinnedArtifactRef`, `DecisionRequest`, action/effect, semantic-root, applicability-fact, budget, and authority envelopes. Bind principal, stage, objective, exact tool/action arguments, targets, expected effects, repository and dirty-worktree roots, IntentIR, LegalIR, SecurityIR, AST/program, tool-catalog and policy roots, jurisdiction/effective time when relevant, capabilities, lease, fence, and idempotency. Preserve both CIDv1 and supervisor digest only when they independently verify the same canonical bytes. Enforce canonical serialization, size/count/depth bounds, no ambient defaults for decision-changing fields, and rejection of missing roots, duplicate/conflicting references, unknown authority, root escapes, non-finite budgets, and changed round trips.

## ASI-125 Add a lazy pinned IntentIR, LegalIR, and SecurityIR registry

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-ir
- Depends on: ASI-097, ASI-099, ASI-124
- Goal id: ASI-G310
- Outputs: ipfs_accelerate_py/agent_supervisor/ir_registry.py, ipfs_accelerate_py/agent_supervisor/ir_adapters.py, test/api/test_agent_supervisor_ir_registry.py, test/api/test_agent_supervisor_ir_adapters.py
- Validation: python -m pytest test/api/test_agent_supervisor_ir_registry.py test/api/test_agent_supervisor_ir_adapters.py test/api/test_agent_supervisor_analysis_transport.py test/api/test_agent_supervisor_analysis_consensus.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/ir
- Parallel lane: pinned-ir-registry
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/ir_registry.py, ipfs_accelerate_py/agent_supervisor/ir_adapters.py, test/api/test_agent_supervisor_ir_registry.py, test/api/test_agent_supervisor_ir_adapters.py
- Conflict policy: Own only registry, verification, and normalized adapter contracts; use the existing analysis transport for optional `ipfs_datasets_py` access and do not eagerly import datasets, models, graphs, or provers.
- Acceptance: Discover supported shared IR-core, formalization, IntentIR, LegalIR, and SecurityIR schemas and operations lazily; load exact bounded local or remote artifacts by pinned reference; verify canonical bytes, CID/digest equivalence, schema/version, producer/configuration, provenance, review/trust state, declared authority, and root membership; normalize declarations, formal views, claims, assumptions, obligations, and result authority without copying source corpora into supervisor state. Return typed unsupported, unavailable, partial, stale, quarantined, ambiguous, contradiction, and bounds failures with deterministic local fixtures and fail closed for every required input. Fresh-interpreter import and capability discovery must start no process or optional provider.

## ASI-126 Compile IntentIR action contracts into supervisor constraints

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-constraints
- Depends on: ASI-125
- Goal id: ASI-G340
- Outputs: ipfs_accelerate_py/agent_supervisor/intent_constraint_adapter.py, test/api/test_agent_supervisor_intent_constraints.py
- Validation: python -m pytest test/api/test_agent_supervisor_intent_constraints.py test/api/test_agent_supervisor_ir_adapters.py test/api/test_agent_supervisor_formal_plan_compiler.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/constraints
- Parallel lane: intent-constraints
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/intent_constraint_adapter.py, test/api/test_agent_supervisor_intent_constraints.py
- Conflict policy: Keep the IntentIR adapter standalone and declaration-only; do not make retrieved SkillCenter prose, GraphRAG premises, or intent modalities grant execution authority.
- Acceptance: Compile a pinned IntentIR document and formalization artifact into exact goal, action, control-flow, precondition, guard, invariant, effect, postcondition, assumption, failure, retry, and verification constraints plus proof obligations and source bindings. Preserve grounded versus inferred nodes, review status, authority and context-only premises, action ordering/parallel joins, and undeclared or contradictory effects. Emit a canonical conformance request/result that checks an exact candidate plan and fails closed on missing required actions, unsatisfied guards/invariants, unbound inferred requirements, unsupported statements, graph truncation, changed intent roots, and attempts to treat intent or retrieval as authorization.

## ASI-127 Compile LegalIR applicability, norms, exceptions, and conflicts

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-constraints
- Depends on: ASI-125
- Goal id: ASI-G340
- Outputs: ipfs_accelerate_py/agent_supervisor/legal_constraint_adapter.py, test/api/test_agent_supervisor_legal_constraints.py
- Validation: python -m pytest test/api/test_agent_supervisor_legal_constraints.py test/api/test_agent_supervisor_ir_adapters.py test/api/test_agent_supervisor_authorization_logic.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/constraints
- Parallel lane: legal-constraints
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/legal_constraint_adapter.py, test/api/test_agent_supervisor_legal_constraints.py
- Conflict policy: Own the supervisor-facing legal applicability and constraint adapter; preserve LegalIR formalization authority and do not convert legal permission into a security grant.
- Acceptance: Select applicable pinned LegalIR declarations and formal views deterministically from exact jurisdiction, subject, principal, action, resource, effect, and effective-time facts; compile obligations, prohibitions, permissions, powers, exceptions, precedence, temporal conditions, conflicts, assumptions, and proof obligations with source/provenance bindings. Semantic retrieval may nominate candidates but cannot establish applicability or absence. Emit explicit applicable, inapplicable, unknown, conflicting, expired, superseded, and review-required outcomes; fail closed on unresolved mandatory applicability, exception, conflict, missing trusted source, changed root, or unsupported modality; and prove that similar but inapplicable provisions and a permission without SecurityIR authorization cannot admit an action.

## ASI-128 Compile SecurityIR declarations into exact authorization decisions

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-constraints
- Depends on: ASI-114, ASI-125
- Goal id: ASI-G340
- Outputs: ipfs_accelerate_py/agent_supervisor/security_constraint_adapter.py, test/api/test_agent_supervisor_security_constraints.py
- Validation: python -m pytest test/api/test_agent_supervisor_security_constraints.py test/api/test_agent_supervisor_authorization_logic.py test/api/test_agent_supervisor_ir_adapters.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/constraints
- Parallel lane: security-constraints
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/security_constraint_adapter.py, test/api/test_agent_supervisor_security_constraints.py
- Conflict policy: Adapt SecurityIR to the existing authorization engine without replacing it; keep declaration/formalization, policy evaluation, and eventual execution permits as separate authority stages.
- Acceptance: Compile pinned SecurityIR principals, assets, resources, zones, channels, policies, state machines, threat assumptions, claims, and formal obligations into exact authorization inputs and checks for principal, action, tool, target, data flow, expected effect, current state, and requested authority. Preserve deny overrides, explicit unknown/conflict, state guards/transitions, trust zones, channel constraints, assumption dependencies, and claim/result authority. Produce canonical policy and decision receipts bound to the SecurityIR root and reject wildcard broadening, unknown resources, stale state, changed effects, unsupported policy, contradiction, and every attempt to use intent, legal permission, model output, or retrieval rank as a grant.

## ASI-129 Bind dirty worktree bytes, AST behavior, tools, and proposed effects

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-graph
- Depends on: ASI-100, ASI-124
- Goal id: ASI-G320
- Outputs: ipfs_accelerate_py/agent_supervisor/program_behavior.py, test/api/test_agent_supervisor_program_behavior.py
- Validation: python -m pytest test/api/test_agent_supervisor_program_behavior.py test/api/test_agent_supervisor_analysis_ast_index.py test/api/test_agent_supervisor_artifact_store.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/dependency-graph
- Parallel lane: program-behavior-root
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/program_behavior.py, test/api/test_agent_supervisor_program_behavior.py
- Conflict policy: Add snapshot, behavior, and effect contracts without changing daemon dispatch; reuse the AST index and artifact store and never scan outside the declared repository and path budget.
- Acceptance: Compute a canonical repository/worktree behavior root covering HEAD, index, relevant tracked modifications, deletions, renames, modes/symlinks, and in-scope untracked bytes instead of treating HEAD as the executed tree. Bind incremental AST/symbol/interface/call/data-flow observations, tool catalog and versions, environment/toolchain facts that affect behavior, and a typed proposed effect manifest for file, process, network, credential, dataset, task-board, commit, and merge operations. Use bounded referenced blobs rather than source bodies; preserve exact clean equivalence and incremental reuse; and reject root escapes, symlink escapes, races, unreadable or oversized required inputs, post-hash changes, unsupported effects, and hidden/untracked changes that would otherwise leave the decision identity unchanged.

## ASI-130 Build the cross-domain semantic proof dependency graph

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-graph
- Depends on: ASI-125, ASI-129
- Goal id: ASI-G320
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_semantic_dependency_graph.py
- Validation: python -m pytest test/api/test_agent_supervisor_semantic_dependency_graph.py test/api/test_agent_supervisor_code_evidence_graph.py test/api/test_agent_supervisor_program_behavior.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/dependency-graph
- Parallel lane: semantic-proof-graph
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_dependency_graph.py, ipfs_accelerate_py/agent_supervisor/code_evidence_graph.py, test/api/test_agent_supervisor_semantic_dependency_graph.py
- Conflict policy: Extend or layer over `CodeEvidenceGraph`; preserve its authoritative proof/validation/merge edges and prohibit GraphRAG or model annotations from manufacturing authority.
- Acceptance: Add canonical typed nodes for decisions, plans, actions, effects, tools/resources, all normalized IntentIR, LegalIR and SecurityIR constraint families, worktree/AST/program elements, assumptions, premises, obligations, proofs, monitors, authorization, validation, and merge evidence. Add typed `requires`, `constrained_by`, `applies_to`, `exception_to`, `conflicts_with`, `authorizes`, `denies`, `implements`, `affects`, `depends_on`, `proven_by`, `monitored_by`, `invalidates`, and `sourced_from` edges with exact root, provenance, trust, authority, and version bindings. Compute deterministic bounded forward mandatory closure, retain proposal-only annotations outside authority closure, reject forged/cross-root edges and unsafe cycles, and prove that irrelevant graph growth does not change a decision closure.

## ASI-131 Generalize proof scope to cross-domain reverse invalidation

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-graph
- Depends on: ASI-130
- Goal id: ASI-G320
- Outputs: ipfs_accelerate_py/agent_supervisor/proof_scope_index.py, test/api/test_agent_supervisor_cross_domain_proof_scope.py
- Validation: python -m pytest test/api/test_agent_supervisor_cross_domain_proof_scope.py test/api/test_agent_supervisor_proof_scope_index.py test/api/test_agent_supervisor_semantic_dependency_graph.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/dependency-graph
- Parallel lane: cross-domain-proof-scope
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof_scope_index.py, test/api/test_agent_supervisor_cross_domain_proof_scope.py
- Conflict policy: Extend existing proof-scope kinds and indexes compatibly; retain current file, symbol, interface, premise, toolchain, policy, and contradiction queries.
- Acceptance: Add explicit scope keys for IR family/root/declaration/claim, intent action/statement, legal norm/applicability fact, security principal/resource/policy/state, program snapshot/AST edge/effect, tool operation, decision context, authorization decision, and execution permit. Index their forward obligations and reverse dependent contexts, plans, proofs, permits, validations, caches, and merges with active/stale state. A semantic input change must deterministically invalidate every transitive dependent and no independent artifact; preserve exact warm reuse; reject cycles, detached receipts, root mismatches, ambiguous aliases, forged activity, and restart-restored indexes that do not revalidate against current canonical artifacts.

## ASI-132 Add retrieval-seed receipts and authoritative proof closure

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-context
- Depends on: ASI-096, ASI-098, ASI-099, ASI-130, ASI-131
- Goal id: ASI-G330
- Outputs: ipfs_accelerate_py/agent_supervisor/proof_directed_retrieval.py, ipfs_accelerate_py/agent_supervisor/analysis_retrieval.py, test/api/test_agent_supervisor_proof_directed_retrieval.py
- Validation: python -m pytest test/api/test_agent_supervisor_proof_directed_retrieval.py test/api/test_agent_supervisor_analysis_retrieval.py test/api/test_agent_supervisor_analysis_consensus.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/context
- Parallel lane: proof-directed-retrieval
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/proof_directed_retrieval.py, ipfs_accelerate_py/agent_supervisor/analysis_retrieval.py, test/api/test_agent_supervisor_proof_directed_retrieval.py
- Conflict policy: Compose existing retrieval backends and the semantic dependency graph; do not promote BM25, vector, AST, GraphRAG, provider consensus, or embedding-guided traversal to proof or authorization authority.
- Acceptance: Derive exact seeds from the `DecisionRequest`, optionally add bounded BM25/vector/AST/GraphRAG candidates, validate candidates against exact index/model/configuration/graph roots and partitions, and then compute the complete mandatory authority/proof closure by deterministic typed edges. Emit a canonical receipt binding query, roots, model and embedding fingerprint, budgets, seeds, candidates, scores, paths, included and omitted nodes, truncation, disagreement, fallback, and closure fixed point. Approximate/truncated retrieval may affect only optional evidence; missing required indexes use deterministic exact fallback or fail closed; poisoned embeddings, cross-partition neighbors, stale roots, malformed candidates, hidden denials, and graph-budget exhaustion cannot suppress a mandatory dependency.

## ASI-133 Compile minimal decision contexts with completeness witnesses

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-context
- Depends on: ASI-096, ASI-132
- Goal id: ASI-G330
- Outputs: ipfs_accelerate_py/agent_supervisor/decision_context.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_decision_context.py
- Validation: python -m pytest test/api/test_agent_supervisor_decision_context.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_evidence_value.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/context
- Parallel lane: decision-context-compiler
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/decision_context.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_decision_context.py
- Conflict policy: Layer the decision compiler over the existing context compiler and contracts; preserve generation-1/2 capsule behavior and keep large bodies in the artifact store.
- Acceptance: Compile an immutable required core containing the exact decision and roots, selected intent action contract, applicable legal/security constraints and unknowns, authorization state, program/effect scope, assumptions, obligations, proof/monitor state, validation, acceptance, and failure behavior. Inline bounded canonical summaries, reference larger bodies by verified expansion handles, and emit a `ContextCompletenessWitness` mapping every mandatory dependency and path to an inline reference or resolvable handle. Required nodes never compete in value-of-information selection. Provider-token-remeasure the complete input and, when mandatory closure exceeds budget, deterministically split, request a named expansion, or fail closed rather than truncate. Prove that 10x irrelevant legal, skill, code, graph, and conversation growth leaves decision context unchanged except bounded index metadata.

## ASI-134 Bind progressive expansion and retries to changed dependencies

- Status: completed
- Completion: manual
- Priority: P1
- Track: proof-runtime-context
- Depends on: ASI-133
- Goal id: ASI-G330
- Outputs: ipfs_accelerate_py/agent_supervisor/decision_context.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_decision_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_decision_context_delta.py test/api/test_agent_supervisor_context_delta.py test/api/test_agent_supervisor_decision_context.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/context
- Parallel lane: decision-context-delta
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/decision_context.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_decision_context_delta.py
- Conflict policy: Own decision-context expansion and retry integration only; coordinate edits to `implementation_daemon.py` with ASI-117 and defer full live-path dispatch replacement to ASI-137.
- Acceptance: Require each expansion to name an unresolved question and a content-addressed handle admitted by the original dependency closure; reject arbitrary corpus browsing, cross-root handles, authority escalation, repeated equivalent requests, and expansion beyond count/token/byte/latency budgets. Build retry capsules from the exact parent decision/context witness plus changed diagnostics, dependencies, proofs, policies, IR roots, or explicitly expanded evidence. Reconstruct and revalidate the full mandatory closure and stable core while transmitting only the delta, invalidate on dirty-worktree and semantic-root changes, preserve omission reasons, and demonstrate lower retry tokens without required-coverage or safety loss.

## ASI-135 Integrate all IR domains into hard-constrained plan admission

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-constraints
- Depends on: ASI-104, ASI-126, ASI-127, ASI-128, ASI-130
- Goal id: ASI-G340
- Outputs: ipfs_accelerate_py/agent_supervisor/ir_constraint_compiler.py, ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, test/api/test_agent_supervisor_ir_constraint_compiler.py
- Validation: python -m pytest test/api/test_agent_supervisor_ir_constraint_compiler.py test/api/test_agent_supervisor_formal_plan_compiler.py test/api/test_agent_supervisor_and_or_planner.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/constraints
- Parallel lane: cross-domain-plan-constraints
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/ir_constraint_compiler.py, ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, test/api/test_agent_supervisor_ir_constraint_compiler.py
- Conflict policy: Integrate the three independent adapters only after their contracts land; preserve deterministic baseline planning and existing hard-constraint pruning and do not let a composite score compensate for a domain failure.
- Acceptance: Compile one canonical plan-admission request and receipt over the exact candidate action/effect graph, IntentIR conformance, LegalIR applicability and constraints, SecurityIR authorization/state checks, program dependencies, assumptions, proof obligations/results, and validation requirements. Prune every candidate with an intent violation, applicable unresolved prohibition/obligation, security deny/unknown, undeclared effect, missing proof, stale root, or authority mismatch before soft scoring. Keep permissions distinct from grants and generated formulas distinct from proofs. Preserve complete rejection reasons and counterexamples for dependency-local replanning, deterministic no-model fallback, and invariant results under candidate order or irrelevant corpus growth.

## ASI-136 Issue and verify exact short-lived execution permits

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-enforcement
- Depends on: ASI-116, ASI-129, ASI-133, ASI-135
- Goal id: ASI-G350
- Outputs: ipfs_accelerate_py/agent_supervisor/execution_permit.py, ipfs_accelerate_py/agent_supervisor/authorization_logic.py, test/api/test_agent_supervisor_execution_permit.py
- Validation: python -m pytest test/api/test_agent_supervisor_execution_permit.py test/api/test_agent_supervisor_authorization_logic.py test/api/test_agent_supervisor_control_transactions.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/enforcement
- Parallel lane: exact-execution-permit
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/execution_permit.py, ipfs_accelerate_py/agent_supervisor/authorization_logic.py, test/api/test_agent_supervisor_execution_permit.py
- Conflict policy: Extend the shared authorization and transaction boundary; do not wire individual daemon, CLI, MCP, tool, commit, or merge callers until the standalone verifier is complete.
- Acceptance: Issue an immutable permit only after exact intent conformance, legal constraint, SecurityIR authorization, mandatory proof/monitor, context completeness, and effect-scope checks pass. Bind it to the complete `DecisionRequest`, candidate action/tool arguments, targets, expected effects, repository/worktree and all semantic roots, dependency closure, context witness, domain receipts, validation plan, caller, lease, fencing epoch, expiry, idempotency key, allowed use count, and policy. Verification immediately before effect must reject replay, changed arguments/targets/effects, stale roots or receipts, expired lease, fence loss, cross-task/principal use, broadened paths, partial authority, and unknown or contradictory mandatory state; a permit grants only the exact declared operation and never completion authority.

## ASI-137 Wire the proof-directed runtime through every live supervisor path

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-enforcement
- Depends on: ASI-136
- Goal id: ASI-G350
- Outputs: ipfs_accelerate_py/agent_supervisor/decision_runtime.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, test/api/test_agent_supervisor_decision_runtime_e2e.py
- Validation: python -m pytest test/api/test_agent_supervisor_decision_runtime_e2e.py test/api/test_agent_supervisor_task_proposal_router.py test/api/test_agent_supervisor_context_delta.py test/api/test_agent_supervisor_control_transactions.py test/api/test_agent_supervisor_merge_train.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/enforcement
- Parallel lane: decision-runtime-live-path
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/decision_runtime.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, test/api/test_agent_supervisor_decision_runtime_e2e.py
- Conflict policy: This is the sole proof-runtime live-path integration lane; serialize overlapping daemon/control/merge edits with ASI-115 through ASI-118 and retain the current path behind an explicit shadow/off fallback.
- Acceptance: Route task proposal, analysis request, plan selection, implementation context, retry, expansion, validation selection/execution, file and task-board mutation, command/tool invocation, commit, merge, and completion admission through one `DecisionRuntime`. Configure exact IR roots and applicability facts through provider-free contracts and equivalent Python/CLI/MCP controls. Move generic prompt policy and edit scope inside the authoritative decision/context identity; remove post-capsule authority text as an enforcement source; check a current permit at every mutation boundary; compare observed to expected effects; and require a new merged-tree decision and evidence assembly before completion. Prove no direct-call or transport bypass, safe off/shadow behavior, deterministic degradation, cancellation, and no eager optional-provider import.

## ASI-138 Add dependency-local invalidation, re-proof, and recovery

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-enforcement
- Depends on: ASI-101, ASI-118, ASI-131, ASI-137
- Goal id: ASI-G350
- Outputs: ipfs_accelerate_py/agent_supervisor/decision_runtime.py, ipfs_accelerate_py/agent_supervisor/runtime_cas.py, ipfs_accelerate_py/agent_supervisor/event_log.py, ipfs_accelerate_py/agent_supervisor/supervisor_recovery.py, test/api/test_agent_supervisor_decision_runtime_invalidation.py
- Validation: python -m pytest test/api/test_agent_supervisor_decision_runtime_invalidation.py test/api/test_agent_supervisor_runtime_cas.py test/api/test_agent_supervisor_fault_recovery_v2.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/enforcement
- Parallel lane: decision-runtime-invalidation
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/decision_runtime.py, ipfs_accelerate_py/agent_supervisor/runtime_cas.py, ipfs_accelerate_py/agent_supervisor/event_log.py, ipfs_accelerate_py/agent_supervisor/supervisor_recovery.py, test/api/test_agent_supervisor_decision_runtime_invalidation.py
- Conflict policy: Own semantic-root event handling and proof-runtime cache/recovery integration; preserve existing CAS namespace authority, event cursors, and generation-2 recovery behavior.
- Acceptance: Convert worktree/AST/effect, IntentIR, LegalIR, SecurityIR, policy, tool catalog, capability, proof, monitor, lease, and observed-effect changes into canonical events; traverse reverse proof scope; and invalidate every and only dependent retrievals, contexts, plans, permits, proofs, validations, caches, and merge/completion receipts. Recompute the affected plan suffix and minimum authoritative proof/validation closure while retaining independent artifacts. Bind checkpoints and replay to the same roots and event cursor; fence pre-crash actors and permits; detect missed/duplicate/reordered events, corrupt indexes, partial writes, root races, and stale restored artifacts; and recover deterministically or enter bounded fail-closed quarantine with an exact repair receipt.

## ASI-139 Benchmark proof-dependency context scaling and gate rollout

- Status: completed
- Completion: manual
- Priority: P0
- Track: proof-runtime-rollout
- Depends on: ASI-119, ASI-137, ASI-138
- Goal id: ASI-G360
- Outputs: ipfs_accelerate_py/agent_supervisor/decision_runtime_benchmark.py, ipfs_accelerate_py/agent_supervisor/decision_runtime_rollout.py, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, test/api/test_agent_supervisor_decision_runtime_benchmark.py, test/api/test_agent_supervisor_decision_runtime_adversarial.py, test/api/test_agent_supervisor_decision_runtime_rollout.py, test/api/test_agent_supervisor_decision_runtime_public_api.py
- Validation: python -m pytest test/api/test_agent_supervisor_decision_runtime_benchmark.py test/api/test_agent_supervisor_decision_runtime_adversarial.py test/api/test_agent_supervisor_decision_runtime_rollout.py test/api/test_agent_supervisor_decision_runtime_public_api.py -q
- Board namespace: agent-supervisor-self-improvement-v3
- Bundle: agent-supervisor/self-improvement-v3/rollout
- Parallel lane: proof-runtime-rollout
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/decision_runtime_benchmark.py, ipfs_accelerate_py/agent_supervisor/decision_runtime_rollout.py, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, test/api/test_agent_supervisor_decision_runtime_benchmark.py, test/api/test_agent_supervisor_decision_runtime_adversarial.py, test/api/test_agent_supervisor_decision_runtime_rollout.py, test/api/test_agent_supervisor_decision_runtime_public_api.py
- Conflict policy: Keep one closed paired/adversarial population and rollout owner; integrate public exports and documentation last, preserve v1/v2 behavior, and prohibit metric or fixture narrowing during promotion.
- Acceptance: Compare the current and proof-directed live paths on the same frozen decisions while independently scaling irrelevant legal corpus, codebase, SkillCenter rows/graphs, and conversation history by at least 10x. Recompute provider tokens, mandatory closure nodes/bytes, total corpus nodes/bytes, cache reuse, invalidation precision, first-valid plans, retries, proof/validation cost, effects, and terminal results from producer receipts and causal ablations. Require zero forged-CID, canonicalization, schema, stale-root, cross-partition, prompt-injection, poisoned-embedding, inapplicable-law, legal-conflict, SecurityIR deny/unknown, intent-authority-confusion, dirty-file, changed-tool-argument, stale-lease, proof-replay, graph-truncation, recovery, path/effect escape, or mandatory-omission escapes. Context must grow with mandatory closure rather than total corpus; deterministic local degraded operation and lazy discovery must pass. Expose equivalent off, shadow, assist, policy-approved automatic, status, explanation, and rollback controls through Python/CLI/MCP, require a later separate current-root evaluation for automatic mode, and return affected behavior to shadow on any binding or safety regression.

## ASI-140 Resolve validation retry-budget failure for ASI-115

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASI-114
- Outputs: ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_control_conformance_v2.py, /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery
- Validation: test -f /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-26-asi-140-asi-115-retry-budget.md
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in ASI-115. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-26-asi-140-asi-115-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release ASI-115 from strategy blocked_tasks.

## ASI-141 Resolve validation retry-budget failure for ASI-137

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASI-136
- Outputs: ipfs_accelerate_py/agent_supervisor/decision_runtime.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, test/api/test_agent_supervisor_decision_runtime_e2e.py, /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery
- Validation: test -f /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-26-asi-141-asi-137-retry-budget.md
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in ASI-137. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-26-asi-141-asi-137-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release ASI-137 from strategy blocked_tasks.

## ASI-142 Define canonical prompt-workflow, graph, projection, run, and rescue contracts

- Status: completed
- Priority: P0
- Track: prompt-workflow-contracts
- Depends on: ASI-124
- Goal id: ASI-G410
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt_workflow.py, test/api/test_agent_supervisor_prompt_workflow_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_workflow_contracts.py test/api/test_agent_supervisor_decision_contracts.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/prompt-contracts
- Parallel lane: prompt-workflow-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt_workflow.py, test/api/test_agent_supervisor_prompt_workflow_contracts.py
- Conflict policy: Add provider-free immutable contracts only; do not edit `llm_router.py`, control adapters, storage backends, lifecycle code, package exports, or live daemons.
- Acceptance: Define strict versioned `PromptSource`, `PromptWorkflowRequest`, `DirectoryScanReceipt`, `PromptGoalGraph`, goal/task/evidence records, materialization/run references, `SupervisorIncident`, `ProgrammaticRecoveryExhaustionReceipt`, `RescuePlan`, and workflow result/receipt contracts. Bind canonical request identities to resolved repository/directory/output roots, prompt CID and redacted metadata, scan/planning/output policies, budgets, caller, and pinned IR/program/policy roots; bind immutable task CIDs to exact goal, dependencies, scope, outputs, validation, acceptance, and policy while excluding mutable status/timestamps. Enforce canonical serialization, unknown-field rejection, count/byte/depth bounds, unambiguous prompt source, no inline secrets in receipts, closed rescue operations, and stable IDs under order/status/timestamp variation. Importing the module must load no provider, DuckDB, model, graph, or supervisor process.

## ASI-143 Build a bounded content-addressed prompt directory scanner

- Status: completed
- Priority: P0
- Track: prompt-workflow-contracts
- Depends on: ASI-142
- Goal id: ASI-G410
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt_directory_scanner.py, test/api/test_agent_supervisor_prompt_directory_scanner.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_directory_scanner.py test/api/test_agent_supervisor_program_behavior.py test/api/test_agent_supervisor_analysis_pipeline.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/prompt-contracts
- Parallel lane: prompt-directory-scan
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt_directory_scanner.py, test/api/test_agent_supervisor_prompt_directory_scanner.py
- Conflict policy: Compose `program_behavior`, AST, analysis, and objective evidence through adapters; do not duplicate scanners or modify provider/router internals.
- Acceptance: Resolve the requested directory beneath an explicit repository allowlist; bind tracked, staged, modified, deleted, and policy-admitted untracked bytes to the current worktree root; return bounded language/build/interface/symbol/test/document/policy summaries and content-addressed evidence handles; and record exact include/exclude/redaction/truncation reasons, budgets, scanner/index/configuration roots, and stability checks. Exclude `.git`, supervisor state/worktrees, secrets/key material, caches, generated/vendor trees, and large binaries by default. Reject symlink/nested-repository/output-path escape, unstable root changes, secret leakage, unbounded source/log bodies, and approximate evidence promoted to authority. Equivalent scans must be stable; relevant dirty changes must invalidate the scan; unavailable optional analysis must degrade explicitly and lazily.

## ASI-144 Generate a strict goal/subgoal/task graph through llm_router with deterministic fallback

- Status: completed
- Priority: P0
- Track: prompt-goal-planning
- Depends on: ASI-103, ASI-104, ASI-142, ASI-143
- Goal id: ASI-G420
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt_goal_planner.py, test/api/test_agent_supervisor_prompt_goal_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_goal_planner.py test/api/test_agent_supervisor_task_proposal_router.py test/api/test_agent_supervisor_and_or_planner.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/planning
- Parallel lane: prompt-goal-generation
- Resource class: provider-llm
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt_goal_planner.py, test/api/test_agent_supervisor_prompt_goal_planner.py
- Conflict policy: Reuse the existing bounded `todo_daemon.llm`/task-proposal `llm_router` adapter and goal grammar; do not edit dirty router/provider files or create another routing layer.
- Acceptance: Compile only the immutable request core, bounded scan/evidence handles, schemas, capabilities, budgets, and pinned constraint summaries into the provider request. Parse strict JSON into one bounded root goal, subgoals, tasks, dependencies, acceptance, outputs, validations, predicted files, resources, risks, assumptions, uncertainty, and evidence trace links. Reject prose wrappers, unknown fields, duplicate keys, cycles, orphan nodes, invalid paths, arbitrary shell/code/policy/authority/completion instructions, missing validation/acceptance, or over-budget output. Emit a complete provider/fallback/parse receipt without raw prompts or model transcripts. Use a deterministic planner on policy-disabled, unavailable, malformed, timeout, or over-budget model paths, and prove schema-equivalent stable output plus bounded input under 10x irrelevant repository growth.

## ASI-145 Admit prompt-generated plans through quality, formal, IR, proof, and validation gates

- Status: completed
- Priority: P0
- Track: prompt-goal-planning
- Depends on: ASI-135, ASI-144
- Goal id: ASI-G420
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt_plan_admission.py, ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, test/api/test_agent_supervisor_prompt_plan_admission.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_plan_admission.py test/api/test_agent_supervisor_formal_plan_compiler.py test/api/test_agent_supervisor_ir_constraint_compiler.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/planning
- Parallel lane: prompt-plan-admission
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt_plan_admission.py, ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, test/api/test_agent_supervisor_prompt_plan_admission.py
- Conflict policy: Extend the existing formal and IR admission boundary through an adapter; do not let a model score compensate for a hard-domain failure or create a second compiler.
- Acceptance: Canonicalize and lint the complete graph; prove connectivity, acyclicity, stable topology, acceptance coverage, task granularity, conflict/resource feasibility, output/validation policy, and evidence traceability; compile it through `FormalPlanCompiler`; and independently require IntentIR conformance, LegalIR applicability/obligations/prohibitions/conflicts, SecurityIR authorization/state, declared program effects, proof obligations, and validation requirements for every exact action/effect. Unknown or contradictory mandatory state, stale roots, unbound paths, shell-bearing validation, hidden effects, and any hard-domain rejection fail closed with exact reasons/counterexamples. Assign final task/plan CIDs only after admission and prove candidate order and irrelevant corpus growth do not change the accepted result.

## ASI-146 Add a canonical Markdown task-source projection

- Status: completed
- Priority: P0
- Track: prompt-task-storage
- Depends on: ASI-145
- Goal id: ASI-G430
- Outputs: ipfs_accelerate_py/agent_supervisor/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, test/api/test_agent_supervisor_markdown_task_source.py
- Validation: python -m pytest test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_taskboard_store.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/task-storage
- Parallel lane: markdown-task-source
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, test/api/test_agent_supervisor_markdown_task_source.py
- Conflict policy: Preserve the current Markdown grammar, locks, CAS, journal, events, and recovery; add an adapter/projection without changing DuckDB or daemon code.
- Acceptance: Project every admitted goal/task/dependency/acceptance/output/validation/resource/conflict field into byte-stable supervisor-compatible Markdown with task CID, plan root, schema, and revision metadata, while excluding mutable status from the task CID. Provide bounded snapshot/query/readiness/CAS/event/watch/integrity behavior over `TaskboardStore`; make identical replay a no-op; reject duplicate aliases/CIDs, cycles, stale revision, partial render, path escape, and task-population drift; and recover an interrupted materialization through the existing journal without duplicate task acceptance.

## ASI-147 Add a transactional DuckDB task-source projection

- Status: completed
- Priority: P0
- Track: prompt-task-storage
- Depends on: ASI-145
- Goal id: ASI-G430
- Outputs: ipfs_accelerate_py/agent_supervisor/duckdb_task_source.py, test/api/test_agent_supervisor_duckdb_task_source.py
- Validation: python -m pytest test/api/test_agent_supervisor_duckdb_task_source.py test/api/test_agent_supervisor_formal_plan_compiler.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/task-storage
- Parallel lane: duckdb-task-source
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/duckdb_task_source.py, test/api/test_agent_supervisor_duckdb_task_source.py
- Conflict policy: Add a standalone lazy DuckDB backend using existing connection/lock patterns; do not edit the Markdown backend, daemon, package exports, or install/load external DuckDB extensions.
- Acceptance: Implement a versioned schema for workflow metadata, artifacts, goals, tasks, dependencies, outputs, validations, acceptance, events, and materialization receipts plus lossless `formal_plan_input_records`/metadata tables or views. Provide bounded snapshot/query/readiness/CAS/event/watch/integrity methods with one fenced writer, monotonic revisions/cursors, transactional updates, atomic initial installation, crash recovery, schema migration preview/rollback, canonical JSON, and application-checked key/edge integrity. Reject SQL identifier/value injection, external extension/network loading, corrupt/partial state, concurrent stale writers, foreign roots, and status-dependent task identities. Independently recompile the database to the original formal plan and degrade lazily when DuckDB is unavailable.

## ASI-148 Make implementation daemons consume Markdown or DuckDB through one task-source protocol

- Status: completed
- Priority: P0
- Track: prompt-task-storage
- Depends on: ASI-146, ASI-147
- Goal id: ASI-G430
- Outputs: ipfs_accelerate_py/agent_supervisor/task_source.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_task_source_e2e.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_source_e2e.py test/api/test_agent_supervisor_event_driven_runtime.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/task-storage
- Parallel lane: direct-task-source
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_source.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_task_source_e2e.py
- Conflict policy: This is the sole daemon task-source integration lane; retain the current Markdown default and event-driven behavior while replacing format-specific reads/writes with the minimal common protocol.
- Acceptance: Define `TaskSource` snapshot, bounded query, get, ready-set, CAS status, append-event, watch, and integrity semantics; adapt the daemon to accept a configured Markdown or DuckDB source directly; preserve dependency/status parsing, event-driven wakeup, leases/fences, attempts, completions, and current task prefix behavior; and expose source/root/schema identities in checkpoints and receipts. The same canonical fixture must yield the same ready ordering, claims, retries, completions, and terminal graph from either backend. Source changes, corruption, foreign roots, stale cursors/revisions, unsupported schemas, and mid-run backend swaps fail closed or use only an explicit verified fallback; DuckDB mode may not depend on generating/downloading a full Markdown board.

## ASI-149 Prove dual-projection equivalence, migration, and replay safety

- Status: completed
- Priority: P0
- Track: prompt-task-storage
- Depends on: ASI-148
- Goal id: ASI-G430
- Outputs: ipfs_accelerate_py/agent_supervisor/task_source.py, test/api/test_agent_supervisor_task_source_parity.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_source_parity.py test/api/test_agent_supervisor_task_source_e2e.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/task-storage
- Parallel lane: task-source-parity
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_source.py, test/api/test_agent_supervisor_task_source_parity.py
- Conflict policy: Integrate only after both backends and direct loading land; do not weaken either backend's independent integrity or transaction checks.
- Acceptance: Add `DualTaskSource` shadow/migration behavior that compares exact plan root, task/goal CIDs, aliases, immutable records, dependencies, ready sets, revisions, events, and terminal outcomes; applies a status mutation as one fenced logical transaction or records a recoverable partial result; and rebuilds either projection only from a verified canonical snapshot. Prove Markdown -> canonical -> DuckDB -> canonical and the reverse are identity-preserving; status changes do not alter task CIDs; identical replay performs zero duplicate claims/effects; crash/concurrent writer/corrupt projection/migration interruption deterministically resumes, rolls back, or quarantines; and parity disagreement prevents automatic promotion.

## ASI-150 Extend the shared control catalog with workflow, restart, and rescue operations

- Status: completed
- Priority: P0
- Track: prompt-control-surfaces
- Depends on: ASI-116, ASI-142
- Goal id: ASI-G440
- Outputs: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, test/api/test_agent_supervisor_prompt_control_catalog.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_control_catalog.py test/api/test_agent_supervisor_control_catalog.py test/api/test_agent_supervisor_control_transactions.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/control
- Parallel lane: prompt-control-catalog
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, test/api/test_agent_supervisor_prompt_control_catalog.py
- Conflict policy: Extend the one closed catalog/service only; do not implement CLI/MCP parsing, provider calls, process effects, or rescue execution in this task.
- Acceptance: Add exact catalog entries and shared request/result schemas for proposal `workflow_preview`/`rescue_preview` and mutation `workflow_materialize`/`restart`/`rescue`, preserving complete enum/catalog coverage. Define bounds, authority class, target kind, repository/state allowlists, dry-run, expected effects, idempotency, lease/fence, cursor, error, and receipt semantics; ensure preview operations cannot mutate and mutations cannot bypass the existing authorization/CAS/recovery/decision-runtime boundary; and provide handler interfaces without eager providers or side effects. Reject unknown operations/fields, arbitrary directory authority, stale preview/incident roots, cross-target replay, missing mutation authority, and transport-specific overrides.

## ASI-151 Build the canonical Python prompt-to-supervisor workflow service

- Status: completed
- Priority: P0
- Track: prompt-control-surfaces
- Depends on: ASI-145, ASI-148, ASI-150
- Goal id: ASI-G440
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt_workflow.py, test/api/test_agent_supervisor_prompt_workflow_service.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_workflow_service.py test/api/test_agent_supervisor_prompt_control_catalog.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/control
- Parallel lane: prompt-python-service
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt_workflow.py, test/api/test_agent_supervisor_prompt_workflow_service.py
- Conflict policy: Compose scanner, planner, admission, task source, and shared control service; do not put policy in adapters or collapse preview/materialize/start authority boundaries.
- Acceptance: Implement `PromptSupervisorService.preview` and `materialize` plus a receipt-linked `bootstrap` convenience saga that composes preview, materialize, and existing start while preserving separate authorization, idempotency, rollback, and resume boundaries. Bind every step to current request/scan/plan/IR/program/policy/catalog roots; store bounded artifact references rather than prompt/source/model bodies; expose admitted/rejected branches, expected/observed effects, partial saga continuation, event cursors, and exact task-source identity; and make exact replay reuse receipts without provider calls or duplicate writes/processes. Stale roots, changed output mode/path, missing authority, projection failure, partial start, and unavailable optional capabilities must produce deterministic resumable or fail-closed results.

## ASI-152 Expose prompt workflow and lifecycle rescue through CLI and Python entry points

- Status: completed
- Priority: P1
- Track: prompt-control-surfaces
- Depends on: ASI-151
- Goal id: ASI-G440
- Outputs: ipfs_accelerate_py/agent_supervisor/control_cli.py, scripts/ops/agent_supervisor/prompt_workflow.py, test/api/test_agent_supervisor_prompt_cli.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_cli.py test/api/test_agent_supervisor_control_conformance_v2.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/control
- Parallel lane: prompt-cli
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_cli.py, scripts/ops/agent_supervisor/prompt_workflow.py, test/api/test_agent_supervisor_prompt_cli.py
- Conflict policy: Keep CLI/module/script as thin parsers/renderers over the catalog and Python service; preserve the user's existing untracked supervisor entry script and do not add provider/policy/process logic.
- Acceptance: Add `workflow-preview`, `workflow-create`, `restart`, `rescue-preview`, and `rescue` commands plus `python -m ipfs_accelerate_py.agent_supervisor.prompt_workflow` and a thin ops wrapper. Accept exactly one inline/file/stdin prompt source; recommend file/stdin so sensitive prompts avoid process listings; support Markdown/DuckDB/both output, dry-run/materialize/start, allowlisted roots, model/fallback budgets, JSON and concise human output, event cursors, and stable error exit codes. Prove canonical equivalence with direct Python requests/results/effects, no raw prompt/secret logging, resumable partial bootstrap, no shell interpolation/path escape, no mutation without normal authority, and side-effect-free `--help`, import, and discovery.

## ASI-153 Expose exact prompt workflow and rescue parity through lazy MCP tools

- Status: completed
- Priority: P1
- Track: prompt-control-surfaces
- Depends on: ASI-151
- Goal id: ASI-G440
- Outputs: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_prompt_mcp.py, test/api/test_agent_supervisor_prompt_control_conformance.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_mcp.py test/api/test_agent_supervisor_prompt_control_conformance.py test/api/test_agent_supervisor_native_mcp_discovery.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/control
- Parallel lane: prompt-mcp
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_prompt_mcp.py, test/api/test_agent_supervisor_prompt_control_conformance.py
- Conflict policy: Generate/adapt tools from the shared catalog only; MCP descriptions, caller paths, or model tool selection must never become authorization.
- Acceptance: Lazily expose `agent_supervisor_workflow_preview`, `workflow_materialize`, `restart`, `rescue_preview`, and `rescue` with the exact catalog schemas, bounds, authority, targets, errors, cursors, receipts, and expected effects. Require server-configured repository/state allowlists and deny arbitrary caller-provided directories without matching authority. Prove Python/CLI/MCP canonical fixture equivalence for success, rejection, dry-run, idempotent replay, stale roots, partial saga, and authorization denial. Discovery/import must start no provider, DuckDB connection, model, process, or supervisor; prompt/repository injection and tool descriptions cannot widen paths, operations, policy, or completion authority.

## ASI-154 Implement fenced start, stop, and restart lifecycle orchestration

- Status: completed
- Priority: P0
- Track: prompt-lifecycle-recovery
- Depends on: ASI-118, ASI-150
- Goal id: ASI-G450
- Outputs: ipfs_accelerate_py/agent_supervisor/lifecycle_orchestrator.py, ipfs_accelerate_py/agent_supervisor/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/supervisor_watchdog.py, test/api/test_agent_supervisor_lifecycle_orchestrator.py
- Validation: python -m pytest test/api/test_agent_supervisor_lifecycle_orchestrator.py test/api/test_agent_supervisor_fault_recovery_v2.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/lifecycle
- Parallel lane: fenced-lifecycle
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/lifecycle_orchestrator.py, ipfs_accelerate_py/agent_supervisor/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/supervisor_watchdog.py, test/api/test_agent_supervisor_lifecycle_orchestrator.py
- Conflict policy: This task owns lifecycle process effects and must use the existing shared control transaction; do not add model rescue or daemon task recovery.
- Acceptance: Model start, stop, and explicit restart as bounded state transitions that persist intent before effects and post-effect receipts after verification; resolve exact process trees/configuration rather than trusting PID files; bind repository/state/run roots, old/new process identity, authorization, revision, idempotency, lease/fence, deadline, expected/observed effects, compensation, and sustained health window. Restart must prove the old tree is terminated/fenced before starting the identical validated profile and must report/resume partial failure. Reject PID reuse, orphan descendants, split brain, cross-run/root signals, overlapping transitions, stale lease/fence, changed configuration, replay, startup that only forks without health, and shutdown that leaves descendants.

## ASI-155 Unify incident diagnosis and bounded programmatic recovery

- Status: completed
- Priority: P0
- Track: prompt-lifecycle-recovery
- Depends on: ASI-154
- Goal id: ASI-G450
- Outputs: ipfs_accelerate_py/agent_supervisor/recovery_diagnostics.py, ipfs_accelerate_py/agent_supervisor/supervisor_recovery.py, test/api/test_agent_supervisor_recovery_diagnostics.py, test/api/test_agent_supervisor_programmatic_recovery.py
- Validation: python -m pytest test/api/test_agent_supervisor_recovery_diagnostics.py test/api/test_agent_supervisor_programmatic_recovery.py test/api/test_agent_supervisor_fault_recovery_v2.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/lifecycle
- Parallel lane: deterministic-recovery
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/recovery_diagnostics.py, ipfs_accelerate_py/agent_supervisor/supervisor_recovery.py, test/api/test_agent_supervisor_recovery_diagnostics.py, test/api/test_agent_supervisor_programmatic_recovery.py
- Conflict policy: Compose existing watchdog, task-attempt, worktree, validation, merge, lease, storage, and quarantine repairs behind typed diagnostics; do not invoke a model or accept arbitrary commands.
- Acceptance: Derive a semantic incident CID from bounded status/health/process/heartbeat/event/lease/lock/task/attempt/task-source/worktree/merge/provider/validation/disk evidence and prior actions; distinguish stale projections from live faults; and execute the least-invasive applicable closed recovery action: reconcile stale state, expire lease/lock/attempt, retry task, restart one lane, rescue dirty worktree, replay validation/merge, quarantine corrupt scope, reassign independent work, or objective reconcile/refill under existing policy. Bind every action to preconditions, maximum attempts, cooldown, deadline, expected effects, compensation, and post-action health. Supported injected faults must recover without a model; unchanged incidents deduplicate; bounded failure must emit a current `ProgrammaticRecoveryExhaustionReceipt` or quarantine rather than loop.

## ASI-156 Add an exhaustion-gated closed llm_router rescue planner

- Status: completed
- Priority: P0
- Track: prompt-rescue
- Depends on: ASI-143, ASI-144, ASI-155
- Goal id: ASI-G460
- Outputs: ipfs_accelerate_py/agent_supervisor/rescue_planner.py, test/api/test_agent_supervisor_rescue_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_rescue_planner.py test/api/test_agent_supervisor_task_proposal_router.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/rescue
- Parallel lane: closed-rescue-planner
- Resource class: provider-llm
- Predicted files: ipfs_accelerate_py/agent_supervisor/rescue_planner.py, test/api/test_agent_supervisor_rescue_planner.py
- Conflict policy: Reuse the existing bounded `llm_router` adapter; produce proposal-only records and do not edit router/provider, control execution, lifecycle, watchdog, or daemon code.
- Acceptance: Invoke a provider only when explicit policy, a current incident-bound programmatic exhaustion receipt, redacted bounded evidence, token/time/cost budgets, cooldown, and circuit breaker all permit it. Give the model only the incident/exhaustion references, bounded diagnostics, exact roots, and a closed catalog of typed recovery operations. Parse strict `RescuePlan/v1` with exact targets, typed parameters, preconditions, expected effects, success/stop conditions, risks, and evidence references. Reject shell commands, code patches, credentials, new paths, unknown operations, policy/authority changes, taskboard rewrites, task completion, missing stops, excess actions, stale roots, and self-authorization. Identical incidents must reuse/circuit-break prior proposals; unavailable/malformed/over-budget models return typed no-plan/quarantine guidance without effects.

## ASI-157 Validate, permit, and execute bounded rescue plans one action at a time

- Status: completed
- Priority: P0
- Track: prompt-rescue
- Depends on: ASI-136, ASI-150, ASI-155, ASI-156
- Goal id: ASI-G460
- Outputs: ipfs_accelerate_py/agent_supervisor/rescue_orchestrator.py, test/api/test_agent_supervisor_rescue_orchestrator.py
- Validation: python -m pytest test/api/test_agent_supervisor_rescue_orchestrator.py test/api/test_agent_supervisor_execution_permit.py test/api/test_agent_supervisor_control_transactions.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/rescue
- Parallel lane: permitted-rescue-executor
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/rescue_orchestrator.py, test/api/test_agent_supervisor_rescue_orchestrator.py
- Conflict policy: Execute only shared catalog operations through the existing decision/permit/control transaction; never interpret model text as a command or authorization.
- Acceptance: Rebind each proposal to the current incident/exhaustion/request/program/IR/policy/catalog roots; validate closed operation/target/parameter schemas and simulate expected effects; independently run IntentIR, LegalIR, SecurityIR, proof, and control authorization; and obtain a short-lived exact permit for one action at a time. Before every effect recheck roots, incident, lease/fence, idempotency, cooldown, and budgets; after every effect compare observed effects and require a health test; stop on health, drift, denial, unexpected effect, action/model/time budget, or quarantine. The model cannot authorize itself or claim completion. Replay, changed targets/arguments, cross-incident permits, root drift, partial effects, policy weakening, arbitrary shell, and endless action sequences fail closed with exact recovery/partial/quarantine receipts.

## ASI-158 Wire bounded autonomous unstalling into the watchdog and implementation supervisor

- Status: completed
- Priority: P0
- Track: prompt-rescue
- Depends on: ASI-157
- Goal id: ASI-G460
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, ipfs_accelerate_py/agent_supervisor/supervisor_watchdog.py, test/api/test_agent_supervisor_autonomous_unstall.py
- Validation: python -m pytest test/api/test_agent_supervisor_autonomous_unstall.py test/api/test_agent_supervisor_fault_recovery_v2.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/rescue
- Parallel lane: autonomous-unstall
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, ipfs_accelerate_py/agent_supervisor/supervisor_watchdog.py, test/api/test_agent_supervisor_autonomous_unstall.py
- Conflict policy: This is the sole live auto-unstall integration lane; retain existing retry/worktree/merge/watchdog logic and invoke model rescue only through the new exhaustion-gated orchestrator.
- Acceptance: Detect stalled/blocked lanes/tasks from semantic event and health evidence; first run the unified bounded deterministic ladder; invoke optional rescue preview/execution only after current qualifying exhaustion and explicit operating policy; and preserve independent work while quarantining the affected scope. Persist incident/action/budget/cooldown/circuit-breaker state across restart; suppress identical incidents and repeated provider calls; re-evaluate roots and health after every action; never let a model or process-liveness claim mark work complete; and surface status/events/reasons through the shared control service. Fault fixtures for stale PID/heartbeat/lease/lock/attempt, dirty worktree, corrupt board/DB, provider loss, validation/merge failure, malicious rescue, repeated unchanged failure, restart during rescue, and unexpected effects must recover or reach bounded visible quarantine without an infinite loop.

## ASI-159 Gate prompt bootstrap and rescue with paired E2E, adversarial, chaos, rollout, and documentation

- Status: completed
- Priority: P0
- Track: prompt-workflow-rollout
- Depends on: ASI-149, ASI-152, ASI-153, ASI-158
- Goal id: ASI-G470
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt_workflow_benchmark.py, ipfs_accelerate_py/agent_supervisor/prompt_workflow_rollout.py, docs/architecture/AGENT_SUPERVISOR_PROMPT_BOOTSTRAP_AND_RESCUE_PLAN.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, test/api/test_agent_supervisor_prompt_workflow_e2e.py, test/api/test_agent_supervisor_prompt_workflow_adversarial.py, test/api/test_agent_supervisor_prompt_workflow_chaos.py, test/api/test_agent_supervisor_prompt_workflow_rollout.py, test/api/test_agent_supervisor_prompt_workflow_public_api.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_workflow_e2e.py test/api/test_agent_supervisor_prompt_workflow_adversarial.py test/api/test_agent_supervisor_prompt_workflow_chaos.py test/api/test_agent_supervisor_prompt_workflow_rollout.py test/api/test_agent_supervisor_prompt_workflow_public_api.py -q
- Board namespace: agent-supervisor-self-improvement-v4
- Bundle: agent-supervisor/self-improvement-v4/rollout
- Parallel lane: prompt-workflow-rollout
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt_workflow_benchmark.py, ipfs_accelerate_py/agent_supervisor/prompt_workflow_rollout.py, docs/architecture/AGENT_SUPERVISOR_PROMPT_BOOTSTRAP_AND_RESCUE_PLAN.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, test/api/test_agent_supervisor_prompt_workflow_e2e.py, test/api/test_agent_supervisor_prompt_workflow_adversarial.py, test/api/test_agent_supervisor_prompt_workflow_chaos.py, test/api/test_agent_supervisor_prompt_workflow_rollout.py, test/api/test_agent_supervisor_prompt_workflow_public_api.py
- Conflict policy: Keep one closed paired/adversarial/chaos population and rollout owner; integrate package exports and documentation last, preserve earlier APIs, and prohibit fixture/metric narrowing during promotion.
- Acceptance: Run frozen prompt/repository fixtures through deterministic/model planning, Markdown/DuckDB/both task sources, and Python/CLI/script/MCP surfaces; require identical admitted task CIDs, ready sets, accepted effects, and terminal outcomes. Inject crashes before/after every materialization/lifecycle/rescue intent-effect-receipt boundary and adversarial prompt/repository/path/symlink/secret/CID/schema/SQL/PID/process/policy/authorization/permit/completion cases. Require zero scope, secret, identity, SQL, process, policy, authority, effect, completion, or mandatory-evidence escapes; deterministic resume/compensation/quarantine for every fault; bounded tokens/model calls/retries/storage/processes; lazy explicit degradation without optional dependencies; and off/shadow/assist/policy-approved automatic controls with immediate rollback on parity/safety/binding regression. Publish exact Python/CLI/MCP examples, DuckDB schema/migration, lifecycle/recovery runbook, threat model, receipts/metrics, and operator rescue guidance; require a later separate fresh-root evaluation before automatic promotion.

## ASI-160 Resolve validation retry-budget failure for ASI-143

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASI-142
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt_directory_scanner.py, test/api/test_agent_supervisor_prompt_directory_scanner.py, /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_directory_scanner.py test/api/test_agent_supervisor_program_behavior.py test/api/test_agent_supervisor_analysis_pipeline.py -q
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in ASI-143. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-160-asi-143-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release ASI-143 from strategy blocked_tasks.

## ASI-161 Resolve validation retry-budget failure for ASI-146

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASI-145
- Outputs: ipfs_accelerate_py/agent_supervisor/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/taskboard_store.py, test/api/test_agent_supervisor_markdown_task_source.py, /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery
- Validation: python -m pytest test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_taskboard_store.py -q
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in ASI-146. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-161-asi-146-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release ASI-146 from strategy blocked_tasks. The declared validation target paths (test/api/test_agent_supervisor_markdown_task_source.py, test/api/test_agent_supervisor_taskboard_store.py) are bounded diagnostic and repair scope: change them only when evidence proves inherited validation debt, and do not weaken correct assertions or policy.

## ASI-162 Resolve implementation retry-budget failure for ASI-152

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASI-151
- Outputs: ipfs_accelerate_py/agent_supervisor/control_cli.py, scripts/ops/agent_supervisor/prompt_workflow.py, test/api/test_agent_supervisor_prompt_cli.py, /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery
- Validation: test -f /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-162-asi-152-implementation-retry-budget.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in ASI-152. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-162-asi-152-implementation-retry-budget.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release ASI-152 from strategy blocked_tasks.

## ASI-163 Resolve implementation retry-budget failure for ASI-153

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASI-151
- Outputs: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_prompt_mcp.py, test/api/test_agent_supervisor_prompt_control_conformance.py, /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery
- Validation: test -f /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-163-asi-153-implementation-retry-budget.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in ASI-153. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-163-asi-153-implementation-retry-budget.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release ASI-153 from strategy blocked_tasks.

## ASI-164 Resolve dirty main checkout blocking 1 worktree merges

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: 8cc53e572c00d05cc3ca4fee9ec4850806c97b10
- Dedupe key: reconciliation_guardrail:main_checkout_dirty
- Depends on:
- Outputs: /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery, docs/architecture/agent_supervisor_self_improvement.todo.md
- Validation: test -f /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-164-reconciliation-8cc53e572c00.md
- Acceptance: Reconciliation guardrail filed this because 1 branch or worktree cleanup candidates are blocked by main_checkout_dirty. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/self-improvement-v2-recovered/state/discovery/2026-07-27-asi-164-reconciliation-8cc53e572c00.md, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.

## ASI-165 Define hierarchical supervisor usage envelopes and accounting bridge

- Status: blocked
- Completion: manual
- Is schedulable: true
- Review only: false
- Blocked reason: awaiting_external_usage_contracts_AICAT-027_AICAT-029
- Priority: P0
- Track: supervisor-usage-accounting
- Depends on: ASI-094, ASI-112, ASI-114
- Goal id: ASI-G510
- Outputs: ipfs_accelerate_py/agent_supervisor/provider_usage.py, ipfs_accelerate_py/agent_supervisor/supervisor_token_ledger.py, test/api/test_agent_supervisor_provider_usage.py, test/api/test_agent_supervisor_token_ledger.py
- Validation: python -m pytest test/api/test_agent_supervisor_provider_usage.py test/api/test_agent_supervisor_token_ledger.py test/api/test_agent_supervisor_efficiency_metrics.py -q
- Board namespace: agent-supervisor-self-improvement-v5
- Bundle: agent-supervisor/self-improvement-v5/usage/contracts
- Parallel lane: supervisor-usage-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/provider_usage.py, ipfs_accelerate_py/agent_supervisor/supervisor_token_ledger.py, test/api/test_agent_supervisor_provider_usage.py, test/api/test_agent_supervisor_token_ledger.py
- Conflict policy: Own provider-free supervisor envelope, lineage, and attribution adapters. Consume the canonical endpoint_usage contracts after AICAT-027 and AICAT-029 pass current-tree validation; do not fork their identities, units, ledger, coordinator, or route policy and do not edit schedulers, provider call sites, controls, MCP, or routers.
- Preconditions: Keep this task externally blocked until AICAT-025, AICAT-027, and AICAT-029 are merged and their focused suites pass on the ASI merge target. Then remove only this task's Blocked reason and set Status to todo through the protected taskboard control path. Map SupervisorEfficiencyReceipt, SupervisorTokenLedger, ResourcePolicy/LeaseBudget, provider capacity, stage/task/attempt identities, and terminal accepted-work attribution before adding fields.
- Effects: Add immutable SupervisorUsageEnvelope, SupervisorUsageScope, SupervisorUsageBudget, SupervisorUsageAttribution, and supervisor-to-endpoint request/receipt bridge contracts with nested deployment/run/goal/task/attempt/stage/lane/request lineage.
- Acceptance: A child budget can only lower its parent across every typed endpoint usage dimension/window and cost currency. Identities bind repository/state/tree/policy, supervisor run, goal/objective, task, attempt, stage, lane, request, catalog revision, usage revision, endpoint scope, caller, deadline, idempotency, lease, and fence without prompts, source, media, model output, credentials, or raw endpoints. Reject missing/foreign/stale ancestry, widened children, duplicate attempts, negative/overflowing units, mixed currency, unknown fields, and unbounded nesting. Adapt supervisor_token_ledger and efficiency metrics to consume reconciled endpoint events exactly once while preserving failed/rejected/abandoned work attribution and accepted-criterion accounting; they cannot independently authorize usage, rewrite provider settlement, or treat usage as correctness/completion evidence. Cold import and schema discovery remain provider/network/process/database/secret-store free.

## ASI-166 Add one reservation-aware supervisor provider execution gateway

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-usage-execution
- Depends on: ASI-144, ASI-156, ASI-165
- Goal id: ASI-G510
- Outputs: ipfs_accelerate_py/agent_supervisor/provider_execution.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py, test/api/test_agent_supervisor_provider_execution.py, test/api/test_agent_supervisor_todo_llm.py
- Validation: python -m pytest test/api/test_agent_supervisor_provider_execution.py test/api/test_agent_supervisor_todo_llm.py test/api/test_agent_supervisor_prompt_goal_planner.py test/api/test_agent_supervisor_rescue_planner.py -q
- Board namespace: agent-supervisor-self-improvement-v5
- Bundle: agent-supervisor/self-improvement-v5/usage/execution
- Parallel lane: supervisor-provider-execution
- Resource class: provider-simulated
- Predicted files: ipfs_accelerate_py/agent_supervisor/provider_execution.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py, test/api/test_agent_supervisor_provider_execution.py, test/api/test_agent_supervisor_todo_llm.py
- Conflict policy: Own the shared provider execution service and isolated todo_daemon LLM adapter. Do not migrate every consumer, edit endpoint_usage internals, modality routers, resource/batch schedulers, supervisor controls, or task completion logic in this lane.
- Preconditions: ASI-165 exposes stable supervisor envelopes and the merged AICAT usage coordinator/routing protocol is available. Inventory the isolated child-process request, timeout, trace, provider verification, environment, cancellation, and return paths before changing its wire envelope.
- Effects: Add ProviderExecutionRequest/Result and a gateway that derives a conservative estimate, requests an exact endpoint route/reservation, invokes through the canonical router or approved typed adapter, normalizes provider observation, settles/reconciles, and returns a redacted endpoint plus supervisor attribution receipt across process boundaries.
- Acceptance: Every request binds supervisor scope/envelope lineage, exact attempt/idempotency, catalog/usage revisions, endpoint binding, deadline, cancellation, lease/fence, and expected provider side-effect boundary. The gateway atomically reserve-invokes-settles and links each retry/fallback to a new attempt; exact replay cannot reinvoke or recharge a terminal request. Pre-dispatch cancellation releases; post-dispatch timeout/cancel conservatively settles; process crash/restart reclaims or reconciles through the store; cache/batch/single-flight outcome metadata prevents duplicate remote charge. Enforce mode fails closed on unknown/stale coordination unless a reviewed degraded budget permits local/deterministic fallback. The todo_daemon child uses a bounded versioned JSON envelope and result file/pipe, propagates receipt IDs without prompt/provider payload leakage, preserves current provider verification and timeouts, and remains behaviorally compatible in off mode.

## ASI-167 Project endpoint usage into fair resource and batch admission

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-usage-scheduling
- Depends on: ASI-112, ASI-117, ASI-165, ASI-166
- Goal id: ASI-G520
- Outputs: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, test/api/test_agent_supervisor_usage_scheduler.py, test/api/test_agent_supervisor_provider_batch_scheduler.py
- Validation: python -m pytest test/api/test_agent_supervisor_usage_scheduler.py test/api/test_agent_supervisor_resource_scheduler.py test/api/test_agent_supervisor_provider_batch_scheduler.py test/api/test_agent_supervisor_stage_scheduler_v2.py -q
- Board namespace: agent-supervisor-self-improvement-v5
- Bundle: agent-supervisor/self-improvement-v5/usage/scheduling
- Parallel lane: endpoint-usage-scheduler
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, test/api/test_agent_supervisor_usage_scheduler.py, test/api/test_agent_supervisor_provider_batch_scheduler.py
- Allow concurrent with: ASI-168, ASI-169
- Conflict policy: Own ResourceScheduler and ProviderBatchScheduler usage-aware admission and focused tests. Consume the shared gateway and endpoint snapshot through injected protocols; do not edit provider consumers, control surfaces, endpoint_usage storage/routing, modality routers, or rollout docs.
- Preconditions: The supervisor gateway is stable; map ProviderCapacity/ProviderBatchCapacity compatibility callers, stage pools, resource leases, capacity suppliers, queue priorities, fairness, batching, single-flight, cancellation, retry-after, event wakeups, and distributed lane fencing.
- Effects: Project exact endpoint scope/revision/freshness/headroom/reset/circuit state into compatibility capacity records and add hierarchical budget, atomic batch reservation, weighted fairness, deadline-aware wait/reroute/backpressure, reset event wakeup, and herd control.
- Acceptance: Effective admission is the conservative intersection of supervisor ancestor budgets, exact endpoint multi-window limits and active reservations, concurrency/context, provider health/circuit/retry-after, deadline, host CPU/RAM/GPU/disk/process constraints, and distributed lease. Unknown/stale fields follow explicit mode policy and cannot become unlimited through legacy -1 projections. Physical batches reserve once, shared overhead settles once, members receive exact attribution, and member cancellation/deadline cannot kill or charge siblings incorrectly. Weighted fair queues and per-tenant/goal/task/lane reserves prevent starvation and one scope consuming an entire shared account window. Next-eligible reset/capacity events wake bounded jittered work through existing event cursors; single-flight refresh prevents herds. The scheduler chooses an eligible policy route, bounded wait, authorized deterministic/local fallback, or typed usage_capacity_unavailable with backpressure; it never weakens authority/completion rules. Off mode preserves existing ordering, batch behavior, and capacity semantics.

## ASI-168 Migrate and prove every supervisor provider consumer

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-usage-migration
- Depends on: ASI-137, ASI-144, ASI-156, ASI-166
- Goal id: ASI-G520
- Outputs: ipfs_accelerate_py/agent_supervisor/provider_usage_migration.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/prompt_goal_planner.py, ipfs_accelerate_py/agent_supervisor/rescue_planner.py, ipfs_accelerate_py/agent_supervisor/leanstral_proof_provider.py, ipfs_accelerate_py/agent_supervisor/leanstral_goal_development.py, test/api/test_agent_supervisor_provider_usage_migration.py
- Validation: python -m pytest test/api/test_agent_supervisor_provider_usage_migration.py test/api/test_agent_supervisor_task_proposal_router.py test/api/test_agent_supervisor_prompt_goal_planner.py test/api/test_agent_supervisor_rescue_planner.py test/api/test_agent_supervisor_leanstral_proof_provider.py test/api/test_agent_supervisor_leanstral_goal_development.py -q
- Board namespace: agent-supervisor-self-improvement-v5
- Bundle: agent-supervisor/self-improvement-v5/usage/migration
- Parallel lane: supervisor-provider-migration
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/provider_usage_migration.py, ipfs_accelerate_py/agent_supervisor/task_proposal_router.py, ipfs_accelerate_py/agent_supervisor/prompt_goal_planner.py, ipfs_accelerate_py/agent_supervisor/rescue_planner.py, ipfs_accelerate_py/agent_supervisor/leanstral_proof_provider.py, ipfs_accelerate_py/agent_supervisor/leanstral_goal_development.py, test/api/test_agent_supervisor_provider_usage_migration.py
- Allow concurrent with: ASI-167, ASI-169
- Conflict policy: Own the generated callsite inventory/coverage gate and migrate provider consumers without changing their planning, proof, rescue, authority, acceptance, or output semantics. Re-read each shared file and preserve concurrent user/provider work; do not edit schedulers, controls, endpoint_usage internals, routers, taskboards, or objective heaps.
- Preconditions: The common gateway and child-process adapter are stable. Generate an AST/import/runtime inventory of direct llm_router, ipfs_datasets router, backend-manager, local model/prover, provider batch, Codex/Copilot/Grok/Goose/other CLI agent, and subprocess provider call paths, including dynamically imported and test-only compatibility surfaces.
- Effects: Route planning, proposal/refinement, prompt goal generation, rescue planning, proof/model assistance, analysis/refill, validation assistance, implementation-agent endpoints, and CLI-backed provider calls through the gateway or a typed contract-equivalent adapter, and add a CI coverage manifest that rejects new bypasses.
- Acceptance: Every in-scope provider call supplies run/goal/task/attempt/stage/lane/request envelope, deadline, budget, idempotency, and exact policy; consumes the returned selection/usage receipt; and retains current deterministic fallback and proof/authority boundaries. Child processes and CLI agents expose structured usage/reset metadata when available and otherwise return a typed non-meterable result that enforce mode admits only under a conservative reviewed ceiling. No migration gives model output completion/authorization authority, changes prompts/source/output contracts, retries side-effecting agent work, or routes data to a forbidden endpoint. Generated AST plus runtime fixtures fail for unregistered direct imports/calls, wrapper aliases, subprocess bypass, missing attribution, or receipt drops while allowlisting provider-free discovery. Off mode and existing focused suites remain compatible.

## ASI-169 Add usage-governance controls and event-derived metrics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: supervisor-usage-controls
- Depends on: ASI-114, ASI-115, ASI-123, ASI-165, ASI-166
- Goal id: ASI-G530
- Outputs: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/scheduler_metrics.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_usage_controls.py, test/api/test_agent_supervisor_usage_control_conformance.py
- Validation: python -m pytest test/api/test_agent_supervisor_usage_controls.py test/api/test_agent_supervisor_usage_control_conformance.py test/api/test_agent_supervisor_control_catalog.py test/api/test_agent_supervisor_control_conformance_v2.py test/api/test_agent_supervisor_scheduler_metrics.py -q
- Board namespace: agent-supervisor-self-improvement-v5
- Bundle: agent-supervisor/self-improvement-v5/usage/controls
- Parallel lane: supervisor-usage-controls
- Resource class: mcp-integration
- Predicted files: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/agent_supervisor/scheduler_metrics.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py, test/api/test_agent_supervisor_usage_controls.py, test/api/test_agent_supervisor_usage_control_conformance.py
- Allow concurrent with: ASI-167, ASI-168
- Conflict policy: Extend the existing operation catalog and shared control service first, then keep CLI and MCP as thin adapters. Consume ledger events through read-only projections; do not create a second counter store, edit provider consumers/schedulers/routers, or let control results affect completion evidence.
- Preconditions: Supervisor usage contracts and execution receipts are stable; map current capability/status/metrics/event/receipt/profile operations, authorization, target/root descriptors, pagination/cursors, dry-run, idempotency, expected effects, lease/fence, redaction, and lazy MCP discovery.
- Effects: Add bounded usage status/health, hierarchical budget, endpoint headroom, reservation, receipt, route-preview, blocked-work/next-eligible, and adapter capability queries plus authorized policy/budget/correction/reset operations and event-derived metrics.
- Acceptance: Python, CLI, and MCP discover and return schema/result/error-equivalent operations bound to supervisor, catalog, usage, and policy revisions. Read/query/preview is lazy, bounded, paginated/cursor-safe, redacted, and side-effect free; it cannot reserve, refresh, probe, invoke, or mutate. Budget/policy/correction/reset requires exact target, distinct authority, expected revision/effects, idempotency, lease/fence, audit, and bounds; callers cannot raise a parent budget or mutate provider truth through model/peer data. Default status aggregates credential/account/tenant detail. Metrics derive from authoritative endpoint events and expose estimate error, headroom bands, denial, wait/reroute, fairness/starvation, reset/herd, fallback, settlement/correction, and ledger health with bounded provider/deployment/stage/state/reason labels and no request, credential, tenant, prompt, media, output, model alias, or endpoint URL cardinality. Usage controls and metrics remain operational evidence only.

## ASI-170 Gate endpoint-aware supervisor rollout with paired E2E and chaos evidence

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-usage-rollout
- Depends on: ASI-167, ASI-168, ASI-169
- Goal id: ASI-G530
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_usage_rollout.py, test/api/test_agent_supervisor_usage_e2e.py, test/api/test_agent_supervisor_usage_chaos.py, test/api/test_agent_supervisor_usage_rollout.py, docs/architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md
- Validation: python -m pytest test/api/test_agent_supervisor_provider_usage.py test/api/test_agent_supervisor_provider_execution.py test/api/test_agent_supervisor_usage_scheduler.py test/api/test_agent_supervisor_provider_usage_migration.py test/api/test_agent_supervisor_usage_controls.py test/api/test_agent_supervisor_usage_control_conformance.py test/api/test_agent_supervisor_usage_e2e.py test/api/test_agent_supervisor_usage_chaos.py test/api/test_agent_supervisor_usage_rollout.py -q
- Board namespace: agent-supervisor-self-improvement-v5
- Bundle: agent-supervisor/self-improvement-v5/usage/rollout
- Parallel lane: supervisor-usage-rollout
- Resource class: test-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/supervisor_usage_rollout.py, test/api/test_agent_supervisor_usage_e2e.py, test/api/test_agent_supervisor_usage_chaos.py, test/api/test_agent_supervisor_usage_rollout.py, docs/architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md
- Conflict policy: Own one frozen paired/adversarial/chaos population, rollout decision, and final operator guidance. Do not narrow fixtures, hard gates, resource accounting, stage coverage, or failure boundaries to obtain promotion; preserve unrelated architecture/provider guidance.
- Preconditions: Scheduling, callsite adoption, and controls are complete. Freeze supervisor run/goal/task/attempt/stage/lane/request populations across planning, analysis, proof, rescue, validation assistance, implementation-agent endpoints, batch/single-flight, local fallback, multiple endpoints sharing one credential quota, and one endpoint with isolated credentials.
- Effects: Add deterministic E2E and chaos harnesses, paired legacy-versus-usage-aware reports, off/observe/shadow/assist/enforce promotion and rollback, measured operating profiles, environment-gated tiny live smoke, threat model, metrics/receipt guidance, incident runbook, and distributed coordination policy.
- Acceptance: Inject concurrent reservation races, estimate under/over actual, 429/503/billing exhaustion, malformed metadata, reset/clock skew/jitter, cache/batch/stream partials, retry/fallback, cancel/timeout before and after dispatch, child/process/supervisor crash, replay, stale lease/fence, ledger corruption/migration/outage, coordinator partition/split brain, endpoint loss/recovery, callsite bypass, unfair queue pressure, and reset herds. Require exact endpoint plus task/stage attribution, no hard-limit or ancestor-budget overshoot, no double/missing charge, no credential/account/tenant scope merge, bounded wait and no starvation/herd, no prompt/media/output/credential/private-URL leak, no authority/completion escape, and deterministic recovery or typed backpressure/quarantine. Off mode matches prior behavior; observe/shadow cannot alter execution; assist requires operator authority; enforce/automatic endpoint fallback requires a later fresh passing paired report with reviewed cost/latency/quality limits; distributed enforcement fails closed without a strong fenced coordinator; any safety, binding, parity, fairness, quality, cost, latency, or compatibility regression immediately returns the affected feature to shadow/off while retaining observed usage for diagnosis.

## ASI-171 Fence cross-lane worktree ownership before cleanup

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-worktree-fencing
- Depends on: ASI-113, ASI-118, ASI-154
- Goal id: ASI-G540
- Outputs: ipfs_accelerate_py/agent_supervisor/worktree_lifecycle.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_worktree_lifecycle.py, test/api/test_agent_supervisor_todo_daemon_port.py
- Validation: python -m pytest test/api/test_agent_supervisor_worktree_lifecycle.py test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_implementation_protected_paths.py -q
- Board namespace: agent-supervisor-self-improvement-v5
- Bundle: agent-supervisor/self-improvement-v5/worktree-fencing
- Parallel lane: worktree-lifecycle-fence
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/worktree_lifecycle.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_agent_supervisor_worktree_lifecycle.py, test/api/test_agent_supervisor_todo_daemon_port.py
- Conflict policy: This is the sole owner of task-attempt claim, managed-worktree lifecycle, and cleanup/reconciliation fencing. Preserve task parsing, routing, provider execution, merge policy, protected-path policy, and board semantics; do not weaken active-process, dirty-worktree, protected-path, or merge gates to make the race disappear.
- Preconditions: Reproduce the 2026-07-28 six-lane failure with a deterministic barrier: pause the owner after its claim and worktree creation but before child-process discovery, then run another lane's already-merged and stale-worktree cleanup against the branch-at-merge-target checkout. Inventory global claim locks, workspace creation/publication, active snapshots, PID/process-birth checks, startup reconciliation, merge cleanup, stale cleanup, retry accounting, and terminal disposal before freezing the lifecycle protocol.
- Effects: Add a durable fenced workspace lifecycle record with `preparing`, `active`, `settling`, and `terminal` states bound to canonical task CID, attempt, lane, owner PID plus process-birth identity, lease/fence, workspace, branch, merge target, and timestamps. Acquire claim and lifecycle fence before publishing a cleanup-visible worktree; require cleanup to compare-and-delete a terminal or provably stale record; let only the fenced owner advance or dispose it; and classify internal setup/reconciliation races separately from implementation failures and provider attempts.
- Acceptance: No lane may delete, prune, reuse, or unregister a worktree with a current nonterminal claim, including the interval between `git worktree add` and visible child process, even when its branch tip is an ancestor of the merge target. Deterministic multi-process tests cover simultaneous lane startup, cleanup at every lifecycle boundary, owner crash before/after publication and spawn, daemon restart, PID reuse, missing `/proc`, stale lease/fence takeover, duplicate attempt, partial worktree creation, active settlement/merge, and legitimate terminal cleanup. Exactly one attempt owns a task/workspace; stale reclamation requires expiry plus fence advancement; an internal lifecycle race makes no provider call and consumes no implementation retry; false cleanup and double execution are zero; true merged/stale cleanup remains bounded and functional; protected-path monitoring stays fail closed; and existing single-lane behavior and six-lane parallel throughput are preserved. Passing evidence unblocks `AICAT-025`.
