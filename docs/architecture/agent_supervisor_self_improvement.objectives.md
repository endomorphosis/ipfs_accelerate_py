# Agent Supervisor Self-Improvement Objective Heap

This objective heap is the durable source of intent for the self-improvement
program. The companion todo board is an executable projection. A drained board
does not complete these goals without fresh, bound validation evidence.

Child-goal evidence entries are stable opaque receipt requirement IDs. The
current scanner must treat them as missing until a qualifying source emits the
exact ID; their presence in this objective heap is never evidence. ASI-003
makes that source policy explicit, and ASI-022 binds each ID to a fresh typed
receipt instead of accepting textual or embedding similarity as completion.

## ASI-G000 Efficient and trustworthy supervisor control loop

- Status: active
- Parent:
- Fib priority: 1
- Track: self-improvement
- Priority: P0
- Bundle: agent-supervisor/self-improvement/root
- Goal: Improve the agent supervisor as a bounded feedback controller that spends fewer model tokens, plans and validates better, uses local software and ipfs_datasets_py effectively, schedules independent work safely, and remains controllable through one Python, CLI, and MCP contract.
- Evidence: AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md, agent-supervisor-self-improvement-v1, agent_supervisor_self_improvement.objectives.md
- Outputs: docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md, docs/architecture/agent_supervisor_self_improvement.todo.md, docs/architecture/agent_supervisor_self_improvement.objectives.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py -q
- Acceptance: Every child goal has fresh tree-bound evidence; rollout has zero false completion or authority-boundary violations; Python, CLI, and MCP controls agree; a drained board runs bounded evidence-driven refill rather than stopping or creating duplicate busywork.
- Gap task: Implement the highest-priority uncovered child-goal evidence without weakening existing assurance boundaries.
- Refinement: Keep child goals separated by measurement, analysis, planning, validation, task generation, parallel runtime, control, refill, and rollout so independent work can be scheduled safely.
- Embedding query: agent supervisor token efficiency planning validation caching parallelism goals task bundles MCP CLI Python self refill
- AST query: SupervisorControlService SelfImprovementRefillController SupervisorEfficiencyReport

## ASI-G010 Token-efficient context and end-to-end measurement

- Status: active
- Parent: ASI-G000
- Fib priority: 2
- Track: token-efficiency
- Priority: P0
- Bundle: agent-supervisor/self-improvement/context
- Goal: Measure total cost per accepted task and compile the smallest sufficient stage context with progressive evidence disclosure and delta retries.
- Evidence: 208290439421789408250562066350459701853, 306437607356117177048620815571362227127, 248026856102230635452423769994290240744
- Evidence criteria: 208290439421789408250562066350459701853=required fields survive the effective provider budget; 306437607356117177048620815571362227127=delta retry lowers tokens without coverage loss; 248026856102230635452423769994290240744=efficiency accounting includes only terminal accepted work
- Evidence producer bindings: 208290439421789408250562066350459701853=context_compiler.REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID and a content-addressed RequiredContextBudgetEvidence witness carried by ContextCompilationReceipt record the requirement only when canonical authority-bearing provider input, including every reference descriptor regardless of its declared token hint, fits the effective provider budget; ContextCompileResult revalidates the complete capsule/receipt/decision/reference/digest binding, and supervisor_efficiency_metrics.RequiredContextPromotionReport consumes those verified results and exactly reconciles their tokens and required coverage with the same terminal accepted task population, as asserted by test/api/test_agent_supervisor_context_compiler.py and test/api/test_agent_supervisor_efficiency_metrics.py; 306437607356117177048620815571362227127=context_compiler.DELTA_RETRY_EVIDENCE_ID and a content-addressed DeltaRetryContextEvidence witness carried by ContextDeltaReceipt record the requirement only when an exact-parent-bound ContextDeltaCapsule transmits deterministic changed or requested evidence, reconstructs the invariant core and required coverage within budget, and uses fewer canonical provider input tokens than full replay; compiler-backed ContextDeltaResult revalidates the parent, delta, reconstruction, receipt, reference partitions, coverage, token counts, and joint digest, and supervisor_efficiency_metrics.DeltaRetryPromotionReport reruns its provider-token verifier before a population-complete same-task gate with complete lifecycle-token reconciliation and at least 35 percent median per-task reduction may claim promotion eligibility, as asserted by test/api/test_agent_supervisor_context_delta.py and test/api/test_agent_supervisor_efficiency_metrics.py; 248026856102230635452423769994290240744=supervisor_efficiency_metrics.TERMINAL_ACCEPTED_WORK_EVIDENCE_ID emitted through PairedEfficiencyReport.evidence_claim_references only for a non-empty, population-complete same-task comparison of terminal accepted work that charges every attempt, as asserted by test/api/test_agent_supervisor_efficiency_metrics.py
- Evidence source policy: Producer bindings are routing metadata, not completion evidence. A requirement is covered only by a fresh passing typed compiler or benchmark receipt from the bound code/test producer with the exact requirement ID, repository tree, policy, inputs, result, and content digest; this heap, todo/task prose, generated discovery reports, and an unbound occurrence of an ID remain non-qualifying sources.
- Outputs: ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_context_delta.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Acceptance: Required goal, authority, scope, and acceptance context is never truncated; optional evidence has deterministic inclusion reasons; retries use changed evidence rather than full replay; paired fixtures reduce median input tokens by at least 35 percent without lowering required evidence coverage or safety.
- Gap task: Implement or repair stage token accounting, evidence context compilation, provider-aware token budgets, on-demand expansion, and delta retries.
- Refinement: Split measurement, base context compilation, and retry deltas when they touch conflicting runtime files.
- Embedding query: supervisor prompt token budget context capsule progressive disclosure retry delta accepted task cost
- AST query: DEFAULT_TODO_VECTOR_CONTEXT_TOKEN_BUDGET build_task_proposal_prompt _build_implementation_prompt

## ASI-G020 Integrated analysis, caching, and ipfs_datasets_py offload

- Status: active
- Parent: ASI-G000
- Fib priority: 2
- Track: analysis
- Priority: P0
- Bundle: agent-supervisor/self-improvement/analysis
- Goal: Use one bounded content-addressed analysis pipeline that reuses local AST and retrieval work and optionally delegates supported reasoning to ipfs_datasets_py.
- Evidence: 189057730455837902155591890661235220962, 184801846437522667882915494501685213497, 206259342916458424196977899134352826879
- Evidence criteria: 189057730455837902155591890661235220962=exact-tree analysis reuse cannot grant stale authority; 184801846437522667882915494501685213497=the optional datasets provider degrades lazily and explicitly; 206259342916458424196977899134352826879=concurrent identical cache misses collapse to one producer
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Acceptance: Existing analysis cache, AST index, and retrieval contracts are used in the live objective/planning path; expensive identical misses collapse across lanes; stale or negative records never become completion evidence; optional datasets capabilities degrade explicitly; repeated fixtures achieve at least 70 percent cache reuse with zero stale authoritative hits.
- Gap task: Integrate existing analysis components, add the bounded datasets provider, or close a measured cache correctness, size, or reuse gap.
- Refinement: Keep local integration, optional provider, and cache coordination in separate lanes until their standalone contracts pass.
- Embedding query: agent supervisor analysis cache AST GraphRAG retrieval ipfs_datasets provider single flight invalidation
- AST query: AnalysisCache AnalysisASTIndex retrieve_analysis_evidence IPFSDatasetsLogicProvider

## ASI-G030 Evidence-aware planning and responsive goal refinement

- Status: active
- Parent: ASI-G000
- Depends on: ASI-G020
- Fib priority: 3
- Track: planning
- Priority: P0
- Bundle: agent-supervisor/self-improvement/planning
- Goal: Select feasible low-cost plans against frozen goals and refine goals promptly from typed counterexamples, stale evidence, repeated failures, and capability changes.
- Evidence: 173075880069453142914839090434430341799, 003778425160038348524906247302938706902, 312819945606360295782005228058369235550
- Evidence criteria: 173075880069453142914839090434430341799=a cheaper authority-violating plan is rejected; 003778425160038348524906247302938706902=a new counterexample triggers one bounded refinement; 312819945606360295782005228058369235550=an unchanged failure backs off without another model call
- Evidence producer bindings: 173075880069453142914839090434430341799=adaptive_planner.AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID and its content-addressed AdaptivePlanSelectionReceipt record the requirement when an authority-safe branch defeats every cheaper authority-violating branch; each independent gate receipt is bound to the canonical candidate snapshot, restored evaluations are recomputed, and the witness set must be complete, as asserted by test/api/test_agent_supervisor_adaptive_planner.py; 003778425160038348524906247302938706902=adaptive_goal_refiner.NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID and a content-addressed NewCounterexampleRefinementEvidence witness embedded in its admission receipt record the requirement only when one changed counterexample produces exactly one root-preserving bounded refinement whose exact candidate plan and repository tree were independently verified, as asserted by test/api/test_agent_supervisor_adaptive_goal_refiner.py; 312819945606360295782005228058369235550=adaptive_goal_refiner.UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID and its backoff receipt record the requirement when unchanged failure evidence produces no second model call, as asserted by test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Evidence source policy: Producer bindings are routing metadata, not completion evidence. A requirement is covered only by a fresh passing typed receipt from the bound code/test producer with exact requirement ID, repository tree, policy, inputs, result, and artifact digest; this heap, todo/task prose, and generated discovery reports remain non-qualifying sources.
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Acceptance: Every plan is evaluated for acceptance coverage, assumptions, semantics, dependencies, conflicts, validation/proof feasibility, novelty, and resource/token cost; hard safety failures cannot be traded away; unchanged failures back off; changed evidence can trigger a bounded verified refinement in the next cycle without mutating the frozen root.
- Gap task: Improve plan evaluation, goal-quality diagnostics, counterexample-driven replanning, or bounded transactional refinement.
- Refinement: Separate candidate generation from deterministic evaluation and separate root-preserving refinement from objective revision.
- Embedding query: adaptive plan selection evidence coverage goal refinement counterexample stale validation capability cost
- AST query: evaluate_plan_branches FormalReplanner LeanstralGoalDevelopmentProvider objective_goal_content_id

## ASI-G040 Strict output, code, test, semantic, and proof validation

- Status: active
- Parent: ASI-G000
- Depends on: ASI-G020
- Fib priority: 3
- Track: validation
- Priority: P0
- Bundle: agent-supervisor/self-improvement/validation
- Goal: Reject malformed, out-of-scope, unsafe, semantically insufficient, stale, or falsely authoritative implementation output with a fail-fast evidence-bound validation DAG.
- Evidence: 314133036252270790078901745919131980427, 266404049326363900535699811645710804440, 006818797857632260116084792540150258746
- Evidence criteria: 314133036252270790078901745919131980427=a no-op or out-of-scope patch cannot reach expensive validation; 266404049326363900535699811645710804440=impact-selected validation catches a seeded transitive defect; 006818797857632260116084792540150258746=a proof candidate never becomes authoritative code-completion evidence
- Outputs: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Acceptance: Schema, authority, patch, path, AST/interface, impact-test, semantic/proof, merge, and freshness gates are explicit; independent checks may run in parallel but no required gate is omitted; seeded adversarial defects do not escape; failed output yields bounded typed diagnostics and cannot claim proof or completion.
- Gap task: Close the highest-risk uncovered validation stage or add an adversarial fixture for a previously accepted invalid output.
- Refinement: Keep pre-execution proposal checks, impact-selected tests, and semantic/proof checks as separate dependency-ordered tasks.
- Embedding query: agent implementation output validation patch scope AST impact tests theorem proof merge freshness adversarial
- AST query: ValidationScheduler CodeProofObligation FormalPlanConformanceResult

## ASI-G050 High-quality task generation and conflict-aware bundling

- Status: active
- Parent: ASI-G000
- Depends on: ASI-G030, ASI-G040
- Fib priority: 5
- Track: task-generation
- Priority: P1
- Bundle: agent-supervisor/self-improvement/task-generation
- Goal: Generate coherent deduplicated tasks and optimize bundles for shared context, dependencies, conflicts, resource classes, validation reuse, and merge locality.
- Evidence: 127990245919649912156052660092678945998, 061582446926920746660485801841658333166, 187052702852200236079602798955260586139
- Evidence criteria: 127990245919649912156052660092678945998=broad work splits without duplicate refill; 061582446926920746660485801841658333166=bundling preserves independent critical-path width; 187052702852200236079602798955260586139=packet completion propagates only to explicitly bound siblings
- Outputs: ipfs_accelerate_py/agent_supervisor/task_quality.py, ipfs_accelerate_py/agent_supervisor/bundle_optimizer.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, ipfs_accelerate_py/agent_supervisor/todo_vector_index.py, ipfs_accelerate_py/agent_supervisor/conflict_graph.py, test/api/test_agent_supervisor_task_quality.py, test/api/test_agent_supervisor_bundle_optimizer.py
- Validation: python -m pytest test/api/test_agent_supervisor_task_quality.py test/api/test_agent_supervisor_bundle_optimizer.py -q
- Acceptance: Tasks bind one coherent acceptance/effect subset with predicted scope and costs; broad tasks split and compatible tiny tasks coalesce; semantic duplicates are rejected across refills; bundles preserve critical-path width and serialize conflicts; model calls per accepted work item improve without increasing merge conflicts.
- Gap task: Improve measured task sizing, semantic deduplication, dependency quality, bundle context reuse, conflict coloring, or packet completion correctness.
- Refinement: Separate task admission from bundle optimization and preserve canonical identities through every projection.
- Embedding query: task generation sizing deduplication bundle optimization context reuse dependency DAG conflict graph merge locality
- AST query: generate_objective_todos_result build_todo_vector_index BundleSupervisor canonical_task_identity

## ASI-G060 Adaptive parallel execution and acceptance throughput

- Status: active
- Parent: ASI-G000
- Depends on: ASI-G050
- Fib priority: 5
- Track: parallelism
- Priority: P1
- Bundle: agent-supervisor/self-improvement/runtime
- Goal: Increase accepted work throughput by adapting independent analysis, inference, proof, validation, merge, and persistence concurrency to live resource and conflict evidence.
- Evidence: 122080003600146794820964010047426915846, 124037811551945145648172208272779822741, 185033715568272291470322170325431455647
- Evidence criteria: 122080003600146794820964010047426915846=adaptive scheduling doubles independent-fixture accepted throughput; 124037811551945145648172208272779822741=one canceled provider-batch member does not cancel siblings; 185033715568272291470322170325431455647=parallel merge flow cannot bypass post-merge validation
- Outputs: ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/provider_batch_scheduler.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/merge_train.py, ipfs_accelerate_py/agent_supervisor/merge_queue.py, test/api/test_agent_supervisor_adaptive_resources.py, test/api/test_agent_supervisor_provider_batch_scheduler.py, test/api/test_agent_supervisor_parallel_acceptance_flow.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_resources.py test/api/test_agent_supervisor_provider_batch_scheduler.py test/api/test_agent_supervisor_parallel_acceptance_flow.py -q
- Acceptance: Resource pools expose backpressure and fair admission; compatible provider work shares model capacity; independent validation and merge preflight run concurrently; target-branch mutation remains fenced and serialized; paired independent fixtures achieve at least twice single-lane throughput without duplicate execution, stale acceptance, resource overcommit, or merge-conflict regression.
- Gap task: Remove the measured stage bottleneck with resource-aware concurrency, batching, cancellation, or backpressure while preserving gate semantics.
- Refinement: Split scheduling, provider batching, and validation/merge throughput into separately benchmarked lanes.
- Embedding query: adaptive parallelism CPU GPU provider batching validation workers merge queue throughput backpressure fairness
- AST query: ResourceScheduler BundleSupervisor ValidationScheduler MergeQueue

## ASI-G070 Unified Python, CLI, and MCP supervisor control

- Status: active
- Parent: ASI-G000
- Fib priority: 3
- Track: control
- Priority: P0
- Bundle: agent-supervisor/self-improvement/control
- Goal: Control and inspect the supervisor through one typed service used consistently by Python imports, the unified CLI, and policy-controlled MCP tools.
- Evidence: 031486194157679117987393491870400400279, 184125100306462690646212311073240043804, 186773143401179107362964063059661378722
- Evidence criteria: 031486194157679117987393491870400400279=Python CLI and MCP operations have schema and behavior parity; 184125100306462690646212311073240043804=mutations require authorization and idempotency; 186773143401179107362964063059661378722=tool discovery starts no process and loads no optional provider
- Outputs: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Acceptance: Shared operations have schema and behavior parity across Python, CLI, and MCP; read operations are bounded; mutations require authorization, explicit roots, dry-run/preview, idempotency, lease/fencing, and audit receipts; lifecycle state and errors are consistent; tool discovery has no provider or process-start side effects.
- Gap task: Add or repair the shared control operation, lifecycle transition, authorization boundary, or parity contract with the highest operator impact.
- Refinement: Implement the Python service first, then independent CLI and MCP adapters, then reconcile lifecycle/events.
- Embedding query: agent supervisor Python API CLI MCP status health goals tasks bundles lifecycle authorization idempotency
- AST query: create_server IPFSAccelerateCLI ImplementationSupervisor supervisor_watchdog

## ASI-G080 Benchmark-driven bounded self-refill

- Status: active
- Parent: ASI-G000
- Depends on: ASI-G010, ASI-G020, ASI-G030, ASI-G040, ASI-G050, ASI-G060, ASI-G070
- Fib priority: 8
- Track: self-refill
- Priority: P1
- Bundle: agent-supervisor/self-improvement/refill
- Goal: Reconcile the completed program, detect measured residual gaps or regressions, and transactionally create only bounded novel successor goals while recording healthy exhaustion when no work is justified.
- Evidence: 020061024173618462922348580596364003627, 065313778069923158401871898168782520190, 119294002389522221490347364495731444366
- Evidence criteria: 020061024173618462922348580596364003627=a drained board creates bounded novel successor goals once; 065313778069923158401871898168782520190=an identical self-improvement epoch is idempotent; 119294002389522221490347364495731444366=a healthy epoch records exhaustion and creates no busywork
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/backlog_refinery.py, test/api/test_agent_supervisor_self_improvement_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Acceptance: A drained board triggers one identity-bound evaluation epoch; measured gaps yield bounded goal proposals that pass quality, refinement, novelty, and policy checks; duplicate/cooldown work is suppressed; identical epochs are idempotent; healthy no-gap epochs persist exhaustion quorum and wait for a meaningful trigger instead of looping.
- Gap task: Implement the self-improvement epoch, successor-goal admission, deduplication, cooldown, or healthy-exhaustion behavior.
- Refinement: Keep benchmark observation, candidate generation, deterministic admission, and materialization as separately receipted stages.
- Embedding query: autonomous self improvement objective refill drained task board benchmark regression novelty cooldown exhaustion quorum
- AST query: record_objective_backlog_findings ObjectiveMaterializationTransactionResult evaluate_exhaustion_quorum

## ASI-G090 Paired rollout, stable exports, and operator adoption

- Status: active
- Parent: ASI-G000
- Depends on: ASI-G080
- Fib priority: 13
- Track: rollout
- Priority: P2
- Bundle: agent-supervisor/self-improvement/rollout
- Goal: Prove the integrated changes improve efficiency and throughput without safety or quality regression, then expose stable imports and operating guidance.
- Evidence: 109590900757783560279417463762322084165, 146189916032404266364029134505159070240, 300500866741873729474343907613893393545
- Evidence criteria: 109590900757783560279417463762322084165=shadow rollout blocks every seeded false completion; 146189916032404266364029134505159070240=paired rollout meets token cache planning and throughput gates; 300500866741873729474343907613893393545=public exports stay lazy without optional providers
- Outputs: test/api/test_agent_supervisor_self_improvement_e2e.py, test/api/test_agent_supervisor_self_improvement_benchmark.py, ipfs_accelerate_py/agent_supervisor/__init__.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_e2e.py test/api/test_agent_supervisor_self_improvement_benchmark.py -q
- Acceptance: Paired cold/warm, failure, adversarial, parallel, restart, and refill fixtures satisfy every non-negotiable safety gate and the documented token/cache/planning/throughput gates; optional integrations degrade correctly; stable exports remain lazy; operators have verified smoke and production profiles; failed gates retain shadow mode and produce bounded diagnostics.
- Gap task: Close the failing paired rollout gate or documentation/export mismatch without changing the benchmark population or weakening its policy.
- Refinement: Keep end-to-end measurement separate from public export and documentation changes until promotion passes.
- Embedding query: agent supervisor paired benchmark rollout shadow assist production smoke exports documentation safety efficiency
- AST query: evaluate_goal_rollout_promotion agent_supervisor __getattr__ register_native_agent_supervisor_tools

## ASI-G091 Prove 208290439421789408250562066350459701853 for Token-efficient context and end-to-end measurement

- Status: active
- Parent: ASI-G010
- Fib priority: 5000
- Track: token-efficiency
- Priority: P0
- Bundle: agent-supervisor/self-improvement/context
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `208290439421789408250562066350459701853`.
- Evidence: 208290439421789408250562066350459701853
- Evidence criterion: Goal, authority, scope, acceptance, and every explicitly required evidence reference survive the effective provider input budget, while a compilation that cannot fit that invariant core fails closed instead of truncating it.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/context_compiler.py REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID and the content-addressed RequiredContextBudgetEvidence witness carried by ContextCompilationReceipt; ContextCompileResult independently checks its capsule, complete decision population, required/selected references, invariant fields, effective budget, and artifact digest, and supervisor_efficiency_metrics.RequiredContextProofBinding/RequiredContextPromotionReport join only those capsule-verified results to accepted-work measurement, exercised by test/api/test_agent_supervisor_context_compiler.py and test/api/test_agent_supervisor_efficiency_metrics.py
- Proof obligation map: (1) `ContextBudget.effective_input_limit`, `ContextCompiler._provider_input_tokens`, and `ContextCompiler.compile` derive and charge the canonical effective provider input, including complete reference descriptors rather than trusting declared token hints (`test_required_fields_and_references_survive_effective_provider_budget`, `test_provider_window_subtracts_output_and_tool_reserves`, `test_canonical_provider_input_defeats_forged_reference_token_count`); (2) the compiler preserves goal, authority, scope, acceptance, and every required reference or raises `RequiredContextOverflowError`, never converting required evidence into an expansion handle (`test_required_context_fails_closed_instead_of_truncating`, `test_required_evidence_cannot_be_deferred_as_expansion_handle`); (3) `RequiredContextBudgetEvidence`, `ContextCompilationReceipt`, and `ContextCompileResult` bind and revalidate the exact tree, objective, policy, capsule, budget, decision population, required/selected references, invariant fields, token count, and artifact digest (`test_compilation_receipt_is_canonical_bounded_and_tamper_evident`, `test_compilation_result_revalidates_capsule_witness_and_decisions`); and (4) `RequiredContextProofBinding` and `RequiredContextPromotionReport` emit the requirement only for capsule-verified, population-complete terminal accepted work with exact candidate-token reconciliation, authoritative coverage equality, and the paired efficiency gate (`test_required_context_promotion_binds_capsule_to_same_task_gate`, `test_required_context_promotion_fails_closed_for_gap_or_forgery`).
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement; only a fresh passing receipt from the bound producer/test on the current repository tree and policy qualifies.
- Outputs: ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_compiler.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Acceptance: The compiler derives the effective input limit from the supervisor ceiling, provider input/window limits, and reserved output/tool tokens; counts the complete canonical provider input and never trusts a caller-declared reference token hint below the canonical descriptor cost; preserves the complete invariant core and every required reference or rejects compilation; refuses to defer required evidence as an expansion handle; orders optional material deterministically with explicit inclusion/omission reasons and bounded expansion handles; and emits the exact requirement ID only in a witness whose repository tree, objective, policy, effective budget, required and selected fields/references, capsule identity, result, and content digest are revalidated against the capsule. End-to-end promotion remains ineligible unless capsule-verified compiler results cover the complete same terminal accepted task population, exactly reconcile charged candidate input tokens, and retain the authoritative required-coverage set while the paired 35 percent gate passes.
- Objective validation repair: ASI-056/ASI-057 form one validation-gate packet. The packet validation reruns the compiler, delta, and efficiency producer suites together so canonical base-context accounting and witness hardening cannot silently regress delta reconstruction or the same-task promotion gates; the existing ASI-G091/ASI-G092 depth-2 children already separate base-compilation and retry-delta proof obligations, so no additional child goal is needed.
- Refinement depth: 2
- Embedding query: 208290439421789408250562066350459701853
- AST query: 208290439421789408250562066350459701853
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `208290439421789408250562066350459701853` with a narrow, verifiable change.

## ASI-G092 Prove 306437607356117177048620815571362227127 for Token-efficient context and end-to-end measurement

- Status: active
- Parent: ASI-G010
- Fib priority: 5001
- Track: token-efficiency
- Priority: P0
- Bundle: agent-supervisor/self-improvement/context
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `306437607356117177048620815571362227127`.
- Evidence: 306437607356117177048620815571362227127
- Evidence criterion: A retry is a parent-bound changed-evidence delta whose transmitted input is smaller than full replay and whose effective context preserves every required field and evidence requirement.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/context_compiler.py DELTA_RETRY_EVIDENCE_ID and the content-addressed DeltaRetryContextEvidence witness carried by ContextDeltaReceipt; ContextDeltaResult revalidates the exact parent, delta, reconstructed capsule, receipt, decisions, reference partitions, coverage, token counts, and joint digest, `ContextCompiler.verify_delta_result` independently remeasures canonical delta and full-replay provider input, and only this capsule-verified result is projected by supervisor_efficiency_metrics.DeltaRetryProofBinding into DeltaRetryPromotionReport, exercised by test/api/test_agent_supervisor_context_delta.py and test/api/test_agent_supervisor_efficiency_metrics.py
- Proof obligation map: (1) `ContextDeltaCapsule`, `ContextCompiler.compile_delta`, and `reconstruct_context` bind the retry to the exact parent while omitting the inherited invariant core from transmitted bytes (`test_delta_transmits_changes_and_preserves_required_coverage`, `test_reconstruction_rejects_stale_parent_and_delta_omits_invariant_core`); (2) delta selection deterministically distinguishes changed, explicitly requested-but-unchanged, retained, and deferred references and rejects an empty retry (`test_unchanged_retry_and_required_evidence_loss_fail_closed`, `test_requested_expansion_is_parent_bound_and_deterministic`, `test_requested_unchanged_reference_is_not_masqueraded_as_changed`); (3) reconstruction preserves goal, authority, scope, acceptance, requiredness, all old and newly required coverage, and deferred expansion handles, and fails closed on a downgrade, coverage loss, forged reconstructed count, stale parent, or effective-budget overflow (`test_delta_rejects_requiredness_downgrade_and_full_context_overflow`, `test_reconstruction_preserves_expansion_handles_and_rejects_token_forgery`, `test_new_required_candidate_is_included_in_witness_coverage`); (4) canonical tokenization charges the actual delta and full provider replay and requires strict per-result reduction (`test_delta_must_be_smaller_than_full_replay`, `test_top_level_delta_wrapper_binds_the_same_contract`); (5) `DeltaRetryContextEvidence`, `ContextDeltaReceipt`, and compiler-backed `ContextDeltaResult` bind and revalidate the tree, objective, policy, parent/delta/reconstruction identities, changed/requested/retained partitions, required fields and coverage, both token counts, result, and joint artifact digest (`test_delta_receipt_and_witness_round_trip_and_reject_forged_claims`); and (6) `DeltaRetryProofBinding.from_context_delta_result` reruns the result's provider-token verifier and rejects receipt-only or unverified input, persisted `@2` bindings require their receipt-keyed verifiers on re-admission, and `DeltaRetryPromotionReport` emits the requirement only for a non-empty, population-complete same-task join with matching tree/objective/policy, preserved authoritative coverage, complete per-task lifecycle-token reconciliation with no unattributed remainder, a passing paired accepted-work report, and at least 35 percent median per-task input-token reduction (`test_delta_retry_promotion_binds_typed_result_to_same_task_gate`, `test_delta_retry_promotion_fails_closed_for_missing_stale_or_unverified_proof`, `test_delta_retry_gate_accepts_requested_only_and_enforces_35_percent`, `test_delta_retry_gate_rejects_unattributed_lifecycle_input`).
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement; only a fresh passing capsule-verified `ContextDeltaResult` and its bound promotion report from the producer/tests on the current repository tree and policy qualify.
- Outputs: ipfs_accelerate_py/agent_supervisor/context_contracts.py, ipfs_accelerate_py/agent_supervisor/context_compiler.py, ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py, test/api/test_agent_supervisor_context_delta.py, test/api/test_agent_supervisor_efficiency_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q
- Acceptance: A compact retry references its exact parent capsule without replaying the invariant core and transmits only deterministic changed or newly requested evidence; applying it reconstructs goal, authority, scope, acceptance, deferred expansion handles, omission diagnostics, and all required evidence without loss or requiredness downgrade. Reconstructed input accounting includes the inherited core and every retained or replaced reference and fails closed above the effective budget; changed and requested-but-unchanged references remain distinct; stale parents and forged counts or digests fail closed; and canonical provider-tokenized delta input is smaller than canonical full replay. The exact requirement ID is emitted only in a witness binding the repository tree, policy, parent, delta, and reconstructed identities, changed/requested/retained references, required fields and coverage, token counts, result, and content digest. A population-complete same-task promotion report must consume compiler-backed `ContextDeltaResult` values rather than receipt-only claims, rerun canonical provider-token measurement, retain full required coverage, exactly reconcile every charged lifecycle input token without an unattributed remainder, and meet the 35 percent median per-task input-token reduction gate before promotion.
- Objective validation repair: ASI-056/ASI-057 form one validation-gate packet. ASI-056 owns the base-compilation witness and ASI-057 owns the retry-delta witness; both use the same terminal-accepted-work measurement boundary and jointly run `python -m pytest test/api/test_agent_supervisor_efficiency_metrics.py test/api/test_agent_supervisor_context_compiler.py test/api/test_agent_supervisor_context_delta.py -q` on the current tree. This keeps the supervisor-fed ASI-056/ASI-057 backlog projection isomorphic to the ASI-G091/ASI-G092 heap children and prevents either proof from being promoted without the shared accounting gates. The existing depth-2 children are independently testable and fully separate the base and retry obligations, so another child would duplicate rather than refine the evidence boundary.
- Refinement depth: 2
- Embedding query: 306437607356117177048620815571362227127
- AST query: 306437607356117177048620815571362227127
- Parallel lane: agent-supervisor/self-improvement/context
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `306437607356117177048620815571362227127` with a narrow, verifiable change.

## ASI-G097 Prove 173075880069453142914839090434430341799 for Evidence-aware planning and responsive goal refinement

- Status: active
- Parent: ASI-G030
- Fib priority: 5000
- Track: planning
- Priority: P0
- Bundle: agent-supervisor/self-improvement/planning
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `173075880069453142914839090434430341799`.
- Evidence: 173075880069453142914839090434430341799
- Evidence criterion: A cheaper authority-violating plan is rejected in favor of an authority-safe feasible plan, and the selection emits a typed receipt carrying the exact requirement ID.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID and content-addressed AdaptivePlanSelectionReceipt v2 with producer kind `adaptive_plan_selection`; four independent hard-gate receipts bind the full canonical candidate snapshot, and deserialization recomputes the evaluation and exact qualifying witness set, exercised by test/api/test_agent_supervisor_adaptive_planner.py
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement; only a fresh passing receipt from the bound producer/test on the current tree qualifies.
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py -q
- Acceptance: Deterministic evaluation covers acceptance evidence, assumptions, semantics, dependencies, conflicts, validation and proof feasibility, novelty, and bounded resource/token cost; authority is a non-compensable gate; an authority-safe branch defeats every cheaper authority-violating branch; hard-gate receipts cannot be replayed after any candidate, formal-plan, or repair-transition change; and the selected-plan receipt binds the exact requirement ID, frozen goal/tree/policy identities, canonical candidate snapshots, a recomputed evaluation, the complete cheaper rejection set, result, and digest. Unsupported persisted planner, evaluator, and formal-replanner versions fail closed.
- Objective validation repair: ASI-054/ASI-055 form one validation-gate packet. The packet validation reruns both bound producer suites together so G097 selection hardening cannot silently regress G098 changed-evidence refinement; no further child goal is needed because the two existing depth-2 children already separate deterministic selection from root-preserving refinement.
- Refinement depth: 2
- Embedding query: 173075880069453142914839090434430341799
- AST query: 173075880069453142914839090434430341799
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `173075880069453142914839090434430341799` with a narrow, verifiable change.

## ASI-G098 Prove 003778425160038348524906247302938706902 for Evidence-aware planning and responsive goal refinement

- Status: active
- Parent: ASI-G030
- Fib priority: 5001
- Track: planning
- Priority: P0
- Bundle: agent-supervisor/self-improvement/planning
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `003778425160038348524906247302938706902`.
- Evidence: 003778425160038348524906247302938706902
- Evidence criterion: One new typed counterexample triggers exactly one bounded verified root-preserving refinement and emits a concrete witness for the exact requirement; admissions caused by other signal kinds do not claim this evidence.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID and the content-addressed NewCounterexampleRefinementEvidence embedded in a qualifying admission receipt, exercised by test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement; only a fresh passing receipt from the bound producer/test on the current tree qualifies.
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_goal_refiner.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_goal_refiner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_goal_refiner.py -q
- Acceptance: A changed typed counterexample can generate and admit at most one bounded refinement in the next cycle; the frozen root is never mutated; the request and candidate remain on the frozen repository tree; admission is policy gated, the candidate declares the exact bounded changed-goal set, and verification binds the exact candidate plan with a boolean proof result; and the witness binds the exact requirement ID, trigger signal, request and evidence fingerprint, frozen root/tree/policy identities, previous and candidate plans, producer, verification receipt, refinement index, and content digest. Non-counterexample admissions remain non-authoritative for this requirement, and restored objective receipts reject unsupported versions, missing identities, and unknown fields.
- Objective validation repair: ASI-054/ASI-055 form one validation-gate packet. The packet validation runs `python -m pytest test/api/test_agent_supervisor_adaptive_planner.py test/api/test_agent_supervisor_adaptive_goal_refiner.py -q` on the current tree so G097 selection receipts and G098 refinement witnesses are checked together; the two existing depth-2 children fully separate these proof obligations, so no additional child goal is needed.
- Refinement depth: 2
- Embedding query: 003778425160038348524906247302938706902
- AST query: 003778425160038348524906247302938706902
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `003778425160038348524906247302938706902` with a narrow, verifiable change.
