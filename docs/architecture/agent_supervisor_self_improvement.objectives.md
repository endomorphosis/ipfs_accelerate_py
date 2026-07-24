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
- Evidence producer bindings: 173075880069453142914839090434430341799=adaptive_planner.AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID and its content-addressed AdaptivePlanSelectionReceipt record the requirement when an authority-safe branch defeats a cheaper authority-violating branch, as asserted by test/api/test_agent_supervisor_adaptive_planner.py; 003778425160038348524906247302938706902=adaptive_goal_refiner.NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID and a content-addressed NewCounterexampleRefinementEvidence witness embedded in its admission receipt record the requirement only when one changed counterexample produces exactly one root-preserving bounded refinement whose exact candidate plan and repository tree were independently verified, as asserted by test/api/test_agent_supervisor_adaptive_goal_refiner.py; 312819945606360295782005228058369235550=adaptive_goal_refiner.UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID and its backoff receipt record the requirement when unchanged failure evidence produces no second model call, as asserted by test/api/test_agent_supervisor_adaptive_goal_refiner.py
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
- Evidence producer: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID and content-addressed AdaptivePlanSelectionReceipt with producer kind `adaptive_plan_selection`, exercised by test/api/test_agent_supervisor_adaptive_planner.py
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement; only a fresh passing receipt from the bound producer/test on the current tree qualifies.
- Outputs: ipfs_accelerate_py/agent_supervisor/adaptive_planner.py, ipfs_accelerate_py/agent_supervisor/plan_evaluator.py, ipfs_accelerate_py/agent_supervisor/formal_replanner.py, test/api/test_agent_supervisor_adaptive_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_adaptive_planner.py -q
- Acceptance: Deterministic evaluation covers acceptance evidence, assumptions, semantics, dependencies, conflicts, validation and proof feasibility, novelty, and bounded resource/token cost; authority is a non-compensable gate; an authority-safe branch defeats every cheaper authority-violating branch; and the selected-plan receipt binds the exact requirement ID, frozen goal/tree/policy identities, candidate evaluations, result, and digest.
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
- Acceptance: A changed typed counterexample can generate and admit at most one bounded refinement in the next cycle; the frozen root is never mutated; the request and candidate remain on the frozen repository tree; admission is policy gated and verification binds the exact candidate plan; and the witness binds the exact requirement ID, trigger signal, request and evidence fingerprint, frozen root/tree/policy identities, previous and candidate plans, producer, verification receipt, refinement index, and content digest. Non-counterexample admissions remain non-authoritative for this requirement.
- Refinement depth: 2
- Embedding query: 003778425160038348524906247302938706902
- AST query: 003778425160038348524906247302938706902
- Parallel lane: agent-supervisor/self-improvement/planning
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `003778425160038348524906247302938706902` with a narrow, verifiable change.
