# Agent Supervisor Self-Improvement Task Board

This board implements the
[self-improvement plan](AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md). The durable
source of intent is
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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

- Status: todo
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
