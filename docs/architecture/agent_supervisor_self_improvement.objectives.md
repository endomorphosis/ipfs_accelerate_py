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
- Evidence producer bindings: 189057730455837902155591890661235220962=analysis_pipeline.EXACT_TREE_REUSE_REQUIREMENT_ID and the content-addressed ExactTreeReuseEvidence witness emitted only after AnalysisCache returns a fresh exact seven-dimension completion hit, the caller's declared digests have been folded with the actual query, analyzer, AST/retrieval/provider inputs and pipeline policy, the referenced packet artifact passes its digest and content-identity checks, and the packet, stage receipts, typed lookup entry/digest, lookup state, and witness are rebound to the active request; invalid external artifacts re-enter keyed single-flight instead of bypassing coordination, as asserted by test/api/test_agent_supervisor_analysis_pipeline.py and test/api/test_agent_supervisor_cache_coordinator.py; 184801846437522667882915494501685213497=ipfs_datasets_analysis_provider.IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID and a content-addressed IpfsDatasetsProviderDegradationEvidence witness emitted only for a bounded offload request whose policy, lazy-load, capability, or runtime boundary enters a closed semantically valid degradation state and returns non-authoritative advisory output with the exact deterministic local-fallback directive; AnalysisProviderResult.proved_requirement_ids_for independently revalidates the exact active request and full provider policy, and AnalysisPipeline projects only those revalidated IDs into advisory_evidence_claim_references, never authoritative_evidence_claim_references, as asserted by test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py and test/api/test_agent_supervisor_analysis_pipeline.py
- Evidence source policy: Producer bindings are routing metadata, not completion evidence. A requirement is covered only by a fresh passing typed witness from the bound producer/test on the current repository tree and policy. The objective heap, todo/task prose, discovery reports, cache statistics, provider claims, and unbound occurrences of requirement IDs are non-qualifying sources. Historical successful cache entries attached to invalidated lookups remain diagnostics and never become authority.
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
- Evidence producer bindings: 314133036252270790078901745919131980427=proposal_validation.NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID and the content-addressed ProposalRejectionEvidence carried by ProposalValidationReceipt record the requirement only when a no-op or out-of-scope proposal is rejected before any expensive validation is dispatched; 266404049326363900535699811645710804440=validation_scheduler.TRANSITIVE_IMPACT_REQUIREMENT_ID and the content-addressed TransitiveImpactValidationEvidence carried by ValidationDAGReceipt record the requirement only when ImpactDependencyGraph.validation_targets produces a population-complete direct-and-transitive impact closure, the bound transitively affected validation executes and detects the seeded defect, ValidationDAGNodeRecord binds each node's validation_id, mandatory/selected decision, selection reason, and depends_on edges, and ValidationAuthorityGateRecord closes every semantic, proof, merge, freshness, and completion member of REQUIRED_AUTHORITY_GATES, as asserted by test/api/test_agent_supervisor_validation_dag.py and test/api/test_agent_supervisor_semantic_validation_pipeline.py; test/api/test_agent_supervisor_proposal_validation.py jointly proves that the same authority-bearing pipeline cannot be entered by a rejected proposal
- Evidence source policy: Producer bindings are routing metadata, not completion evidence. A requirement is covered only by a fresh passing typed receipt from the bound code/test producer with the exact requirement ID, repository tree, objective and policy identities, canonical inputs and result, validated graph declarations, complete selected-node population, dependency and authority-gate trace, actual defect-detecting result, and content digest. This heap, todo/task prose, generated discovery reports, an unbound occurrence of an ID, a proposal rejection without proof that expensive dispatch remained closed, and a static impact graph or impact-selection plan without a population-complete executed defect-detecting result and downstream authority closure are non-qualifying sources.
- Outputs: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Acceptance: Schema, authority, patch, path, AST/interface, impact-test, semantic/proof, merge, and freshness gates are explicit. Validation declarations bind canonical impact targets, DAG dependencies, and downstream authority gates; the receipt covers the complete selected population and schedules only dependency-ready checks under bounded parallelism. No required gate may be omitted, seeded adversarial defects do not escape, and failed output yields bounded typed diagnostics while closing proof, merge, freshness, and completion authority.
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
- Evidence producer bindings: 031486194157679117987393491870400400279=control_contracts.CONTROL_SURFACE_PARITY_REQUIREMENT_ID emitted only by a content-addressed ControlSurfaceParityEvidence whose ControlSurfaceParityCase records re-decode the same canonical OperationRequest and independently observed Python, CLI, and MCP OperationResult records, validate every result against the request, require byte-identical schema/status/error/effect/audit behavior, bind the full closed Operation vocabulary and shared request/result JSON Schema digests, and reject missing surfaces, operation drift, stale tree/objective/policy bindings, or altered records; the independent adapter matrix is exercised by test/api/test_agent_supervisor_control_plane.py, test/test_unified_cli_agent_supervisor.py, and test/mcp_server/test_agent_supervisor_tools.py. 184125100306462690646212311073240043804=control_contracts.CONTROL_MUTATION_GUARD_REQUIREMENT_ID emitted only by ControlMutationGuardEvidence after a real mutation request has passed exact authorization, idempotency, effect, lease, and fencing bindings, produced an applied audit-bound result, replayed to the exact prior result without a second backend call, and replayed missing-authorization, missing-idempotency, and missing-lease/fence variants through the authoritative parser to typed fail-closed errors, as exercised by test/api/test_agent_supervisor_control_lifecycle.py and test/api/test_agent_supervisor_control_plane.py
- Evidence source policy: Producer bindings route validation but do not themselves complete this goal. Only a fresh passing content-addressed evidence record from the bound producer/tests on the current repository tree, objective, and policy qualifies. CLI help text, MCP registration metadata, todo/discovery prose, an unbound requirement ID, three copies of an unverified result, a dry-run alone, or an authorization object without an applied audited idempotent replay are non-qualifying.
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

## ASI-G094 Prove 189057730455837902155591890661235220962 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: active
- Parent: ASI-G020
- Fib priority: 5000
- Track: analysis
- Priority: P0
- Bundle: agent-supervisor/self-improvement/analysis
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `189057730455837902155591890661235220962`.
- Evidence: 189057730455837902155591890661235220962
- Evidence criterion: Exact analysis reuse is authoritative only for a fresh content-addressed result matching the current repository tree, objective revision, analyzer and schema versions, configuration, query, and policy; a stale, negative, corrupt, artifact-invalid, or dimension-changed record cannot grant completion authority.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py EXACT_TREE_REUSE_REQUIREMENT_ID and the content-addressed ExactTreeReuseEvidence witness carried by AnalysisPipelineResult, emitted only after a fresh exact AnalysisCache completion hit and independent compact-receipt, packet-artifact digest/content identity, packet binding, stage binding, and completion-gate checks, exercised by test/api/test_agent_supervisor_analysis_pipeline.py
- Proof obligation map: (1) AnalysisPipelineRequest.cache_key delegates all seven authority dimensions to the established AnalysisCacheKey schema, binds repository and tree together, and derives configuration/query/policy dimensions from both declared labels and the actual authority-changing inputs (`test_every_authority_dimension_rejects_historical_success`, `test_declared_digests_cannot_hide_changed_analysis_inputs`, `test_pipeline_execution_policy_is_part_of_cache_authority`); (2) AnalysisPipeline._load_cached_packet gates on lookup.hit and lookup.is_completion_evidence before inspecting the attached entry, then validates compact summary, artifact path/digest, packet identity, request identity, every receipt dimension, and the derived packet completion gate (`test_exact_reuse_is_content_bound_and_emits_the_requirement`); (3) ExactTreeReuseEvidence restores content-addressedly and independently proves against the typed lookup, including the cache entry's recomputed integrity digest, while AnalysisPipelineResult rejects detached or forged exact-hit witnesses unless lookup, request, cache key, packet, dimensions, and execution flags agree (`test_exact_reuse_is_content_bound_and_emits_the_requirement`, `test_exact_reuse_witness_cannot_be_forged_or_detached`); (4) expired or artifact-invalid successes trigger fresh production and carry no requirement claim, and concurrent invalid-artifact repairs remain in the keyed single-flight boundary (`test_expired_success_is_attached_for_diagnostics_but_never_reused`, `test_missing_packet_artifact_forces_recompute_instead_of_cache_authority`, `test_concurrent_invalid_artifact_repairs_remain_single_flight`, `test_outer_artifact_validator_turns_compact_hit_into_keyed_miss`); (5) inconclusive records cannot bypass the producer and their no-store or bounded-TTL policy survives coordination (`test_negative_result_never_short_circuits_a_later_producer`, `test_negative_cache_policy_is_applied_inside_single_flight`, `test_negative_cache_uses_configured_bounded_ttl`); (6) the live local path builds the incremental AST index and projects its bounded evidence into retrieval before analysis (`test_incremental_ast_index_is_projected_into_live_retrieval`); and (7) ten identical fixtures produce once, reuse nine times, exceed the 70 percent target, and record zero stale authoritative hits (`test_repeated_fixture_exceeds_reuse_target_without_stale_authority`).
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement. Only a fresh passing ExactTreeReuseEvidence witness from the bound pipeline/test on the current tree qualifies; cache status counters, historical entry outcomes, todo/task prose, and a bare requirement ID do not.
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, ipfs_accelerate_py/agent_supervisor/cache_coordinator.py, test/api/test_agent_supervisor_analysis_pipeline.py, test/api/test_agent_supervisor_cache_coordinator.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Acceptance: The live pipeline composes the existing cache, incremental AST index, bounded multi-signal retrieval, and optional datasets adapter; persists packet bodies in a digest-addressed artifact store and only compact bindings in AnalysisCache; never treats an attached invalidated entry as authority; revalidates all seven key dimensions and derived completion state; collapses identical in-process lane misses without globally serializing different keys; achieves at least 70 percent reuse on repeated fixtures; and reports zero stale authoritative hits.
- Objective validation repair: ASI-062 is the anchor and ASI-063 is the member of `goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf`. ASI-062 owns typed exact-tree/cache-entry authority and artifact-aware single-flight repair; ASI-063 owns the lazy optional-provider degradation boundary. Their shared current-tree gate is `python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q`, which freshly passed on 2026-07-24 after detached-digest, concurrent artifact-repair, mixed sync/async coordination, live AST integration, active request/policy rebinding, and fabricated provider-state regressions were added. This keeps the supervisor-fed ASI-062/ASI-063 backlog projection isomorphic to G094/G095. The existing depth-2 children cleanly separate cache authority from optional-provider trust, so another child would duplicate rather than refine the evidence boundary.
- Refinement depth: 2
- Embedding query: 189057730455837902155591890661235220962
- AST query: EXACT_TREE_REUSE_REQUIREMENT_ID ExactTreeReuseEvidence AnalysisPipeline AnalysisCacheKey
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `189057730455837902155591890661235220962` with a narrow, verifiable change.

## ASI-G095 Prove 184801846437522667882915494501685213497 for Integrated analysis, caching, and ipfs_datasets_py offload

- Status: active
- Parent: ASI-G020
- Fib priority: 5001
- Track: analysis
- Priority: P0
- Bundle: agent-supervisor/self-improvement/analysis
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `184801846437522667882915494501685213497`.
- Evidence: 184801846437522667882915494501685213497
- Evidence criterion: Optional ipfs_datasets_py analysis is imported only after a policy-allowed bounded offload is requested; unavailable, disabled, unsupported, incompatible, unhealthy, timed-out, failed, malformed, or cancelled capability/runtime states are explicit and never become supervisor completion authority.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID and the content-addressed IpfsDatasetsProviderDegradationEvidence witness carried by AnalysisProviderResult for explicit disabled, unavailable, or unsupported local fallback. The witness emits the requirement only when request, repository tree, objective revision, operation, result status/reason/health/import history, and the complete content-addressed provider policy are mutually bound; `AnalysisProviderResult.proves_requirement_for` and `proved_requirement_ids_for` independently check the active request and policy. The no-context `proved_requirement_ids` surface always fails closed, while `diagnostic_requirement_ids` is explicitly non-qualifying. AnalysisPipeline uses the adapter's public `build_request` boundary and exposes a qualifying ID only through `advisory_evidence_claim_references`, separate from completion-authoritative exact-tree evidence. This is exercised by test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py and the integrated fallback cases in test/api/test_agent_supervisor_analysis_pipeline.py.
- Proof obligation map: (1) module import, provider construction, disabled execution, capabilities, and public bounded request construction perform no optional import (`test_construction_and_capability_declaration_do_not_import`, `test_disabled_provider_is_explicit_without_import`, `test_public_request_builder_applies_limits_without_loading_backend`); (2) a missing module and a policy-unsupported operation produce typed content-addressed degradation evidence with an explicit deterministic local fallback and exact requirement ID only through the active-context API (`test_missing_optional_module_degrades_explicitly_with_typed_evidence`, `test_unsupported_operation_never_loads_backend`); (3) capability negotiation and requests enforce closed operations, canonical aliases, protocol compatibility, supervisor-owned count/query/request/response/reference/time bounds, compact reference projection, deterministic identities, literal booleans, and non-expandable empty policy (`test_supported_backend_receives_bounds_and_remains_advisory`, `test_compact_success_is_deterministic_bounded_and_non_authoritative`, `test_explicit_empty_operation_policy_is_not_expanded`, `test_native_provider_operation_alias_is_validated_canonically`); (4) incompatible, unhealthy, malformed, cancelled, failed, heavy, exceptional, awaitable, and typed backend results all re-enter the same bounded fail-closed projection (`test_capability_degradation_is_typed_bound_and_non_authoritative`, `test_cancelled_and_failed_paths_are_explicit_without_false_authority`, `test_heavy_or_malformed_backend_payload_fails_closed`, `test_backend_exception_is_typed_and_async_facade_leaks_no_coroutine`, `test_typed_backend_result_reenters_bounded_projection`, `test_callable_authority_claim_and_awaitable_result_degrade_locally`); (5) restored records validate content IDs and fixed non-authority invariants, every semantic and binding dimension is rechecked against the active request and full policy, the no-context proof surface fails closed, a degradation witness cannot be detached or replayed, and matching a policy ID cannot hide changed enabled/operation semantics (`test_serialized_invariants_and_content_ids_are_fail_closed`, `test_degradation_evidence_cannot_be_detached_or_replayed`, `test_requirement_witness_requires_every_semantic_and_binding_dimension`, `test_active_policy_semantics_cannot_be_forged_with_matching_policy_id`); (6) adapter-owned import history is derived from the actual dependency path, and timed-out non-cooperative executions remain semaphore bounded with typed non-authoritative fallback (`test_capability_dependency_failure_records_actual_adapter_import_history`, `test_repeated_timeouts_have_bounded_backend_concurrency`); and (7) explicit degradation remains advisory while healthy local analysis can complete; the integrated pipeline projects the requirement only after rebuilding and revalidating the adapter's exact request/policy binding, keeps it out of authoritative claims, and converts detached, foreign, missing-identity, hostile, awaitable, or authority-claiming provider results into non-authoritative local failures (`test_optional_provider_degrades_explicitly_without_poisoning_local_result`, `test_only_exact_request_and_policy_bound_adapter_evidence_is_projected`, `test_advisory_provider_claim_cannot_be_detached_from_typed_result`, `test_advisory_provider_claim_cannot_be_replayed_from_another_query`, `test_malformed_optional_provider_identity_degrades_locally`, `test_hostile_optional_provider_inspection_never_aborts_local_analysis`, `test_optional_provider_rebinding_or_authority_claim_is_rejected_advisory`).
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement. Only a fresh passing typed degradation witness from the bound adapter/test on the current tree qualifies. Provider-supplied authority fields, successful advisory references, import availability alone, todo/task prose, and a bare requirement ID do not.
- Outputs: ipfs_accelerate_py/agent_supervisor/ipfs_datasets_analysis_provider.py, ipfs_accelerate_py/agent_supervisor/analysis_pipeline.py, test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py, test/api/test_agent_supervisor_analysis_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
- Acceptance: Capability inspection is deterministic and side-effect free; dispatch lazily imports only a supported requested operation; provider policy is a non-expandable resource envelope; capability and result payloads are canonical, bounded, compact, and identity checked; timed-out non-cooperative backends cannot create unbounded dispatch threads; unavailable, unsupported, disabled, unhealthy, timed-out, failed, malformed, and cancelled results state exact reasons and deterministic fallback; and no provider result, including a backend-supplied authority claim, is completion evidence.
- Objective validation repair: ASI-062/ASI-063 share `goal_packet/analysis/ipfs_accelerate_py/073d1a3271bf` and the current-tree validation gate recorded by G094. ASI-063 owns G095 and now requires the result-level active request/full-policy check plus a closed adapter-owned degradation-state tuple—status, reason, actual import history, backend health, and exact `local_deterministic_analysis` fallback—before the requirement ID can be projected. The pipeline carries that projection only as an advisory claim and never merges it with exact-tree completion authority. The shared gate passed on 2026-07-24. G094 and G095 remain independently testable depth-2 trust boundaries, so no additional refinement is required.
- Refinement depth: 2
- Embedding query: 184801846437522667882915494501685213497
- AST query: IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID IpfsDatasetsProviderDegradationEvidence IpfsDatasetsAnalysisProvider
- Parallel lane: agent-supervisor/self-improvement/analysis
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `184801846437522667882915494501685213497` with a narrow, verifiable change.

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

## ASI-G100 Prove 314133036252270790078901745919131980427 for Strict output, code, test, semantic, and proof validation

- Status: active
- Parent: ASI-G040
- Fib priority: 5000
- Track: validation
- Priority: P0
- Bundle: agent-supervisor/self-improvement/validation
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `314133036252270790078901745919131980427`.
- Evidence: 314133036252270790078901745919131980427
- Evidence criterion: A proposal with no effective change or any path outside its task-owned scope is rejected by deterministic pre-execution gates before a test, semantic checker, prover, or other expensive validation can be dispatched.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/proposal_validation.py NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID and the content-addressed ProposalRejectionEvidence carried by ProposalValidationReceipt; the receipt binds the exact repository tree, objective, policy, proposal and baseline identities, normalized task scope and candidate diff, bounded typed rejection, ordered gate trace, and closed expensive-dispatch state, exercised by test/api/test_agent_supervisor_proposal_validation.py and the integrated gate-order cases in test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Proof obligation map: (1) canonical proposal parsing and diff normalization distinguish a real in-scope change from empty, metadata-only, unchanged-content, malformed, unsafe-path, and out-of-scope candidates before scheduler admission; (2) task scope is an intersection boundary that proposal policy cannot expand, and normalized rename, copy, add, delete, and modified paths all participate in the check; (3) no-op and out-of-scope rejections return bounded typed diagnostics while the injected expensive-validation runner remains uncalled and downstream DAG nodes remain undispatched; (4) ProposalRejectionEvidence and ProposalValidationReceipt bind and revalidate the tree, objective, policy, proposal, baseline, scope, normalized diff, rejection reason, complete preflight gate trace, result, closed-dispatch state, and content digest; and (5) accepted proposals, unrelated rejection kinds, detached witnesses, altered scope or diff content, replay on another tree or policy, missing gates, and forged dispatch state cannot claim this requirement.
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement. Only a fresh passing ProposalValidationReceipt from the bound producer/test on the current repository tree and policy qualifies; a rejection reason alone, a scheduler report with no proposal binding, an uncalled runner without a complete gate trace, todo/task prose, and a bare requirement ID do not.
- Outputs: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Acceptance: Proposal admission deterministically checks schema, authority, baseline and candidate identity, non-empty effective change, normalized path safety, and task-owned scope before any expensive validation. Empty or effectless diffs and every out-of-scope path fail closed with bounded typed diagnostics; policy cannot widen task scope; rejected output cannot claim proof, completion, merge eligibility, or authority; and the scheduler cannot be reached through the validated pipeline after preflight rejection. The exact requirement ID is emitted only by a tamper-evident receipt that binds the current tree, objective, policy, proposal, baseline, scope, normalized diff, complete ordered gate trace, failure result, proof that expensive dispatch remained closed, and content digest.
- Objective validation repair: ASI-031/ASI-032 form one validation-gate packet. ASI-031 owns deterministic proposal admission and closed expensive dispatch, while ASI-032 owns transitive impact selection and executed defect detection; the packet reruns all three validation suites together so a proposal-gate change cannot bypass or silently narrow the downstream validation DAG. These two depth-2 children already separate pre-execution rejection from impact-selected execution, so no smaller child goal is needed.
- Refinement depth: 2
- Embedding query: 314133036252270790078901745919131980427
- AST query: NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID ProposalRejectionEvidence ProposalValidationReceipt
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `314133036252270790078901745919131980427` with a narrow, verifiable change.

## ASI-G101 Prove 266404049326363900535699811645710804440 for Strict output, code, test, semantic, and proof validation

- Status: active
- Parent: ASI-G040
- Fib priority: 5001
- Track: validation
- Priority: P0
- Bundle: agent-supervisor/self-improvement/validation
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `266404049326363900535699811645710804440`.
- Evidence: 266404049326363900535699811645710804440
- Evidence criterion: A change whose defect is observable only through a transitive dependency selects and executes the affected validation, records the seeded failure, and cannot pass through to semantic, proof, merge, freshness, or completion authority.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/validation_scheduler.py TRANSITIVE_IMPACT_REQUIREMENT_ID and the content-addressed TransitiveImpactValidationEvidence carried by ValidationDAGReceipt. Before dispatch, the producer validates the tree-bound ImpactDependencyGraph and its validation_targets declarations; it then binds the current repository tree, objective, validation policy, accepted proposal and canonical change, canonical graph payload and affected-path closure, required_validation_ids, selected_node_ids, coverage_complete, each ValidationDAGNodeRecord's validation_id, mandatory/selected decision, selection_reason, depends_on edges and disposition, the complete ValidationAuthorityGateRecord population for REQUIRED_AUTHORITY_GATES, executed result, seeded-defect identity and transitive path, fail-fast disposition, and content digest. test/api/test_agent_supervisor_validation_dag.py exercises the graph and execution producer, and test/api/test_agent_supervisor_semantic_validation_pipeline.py exercises the integrated semantic/proof/merge/freshness/completion authority boundary.
- Proof obligation map: (1) explicit graph and validation declarations are canonical, tree-bound, identity checked, and fail closed for stale bindings, unsafe or unknown targets, duplicate nodes, unknown or cyclic dependencies, or authority gates that are not downstream of their declared prerequisite; (2) impact analysis follows deterministic direct and transitive dependency edges from every changed path through affected interfaces and consumers to mandatory declared validations rather than selecting only same-file tests; (3) selection equals the full closure-derived mandatory population—every required node appears exactly once with its impact path, dependencies, authority gates, and initial disposition, while no undeclared node can create authority; (4) the scheduler admits only dependency-ready nodes under bounded parallelism and records the selected, executed, succeeded, failed, blocked, and omitted population deterministically, including executed result digests and the exact failed dependency that blocks a descendant; (5) the seeded fixture changes an upstream dependency while the observable failure lives in a transitive consumer, and the DAG includes and executes that consumer validation; (6) the detected failure closes every dependent semantic, proof, merge, freshness, and completion gate, with explicit blocked authority records that cannot be detached from the failed node, while bounded diagnostics identify the selected transitive path and actual failing result without converting an expected adversarial failure into code-completion authority; and (7) TransitiveImpactValidationEvidence and ValidationDAGReceipt reject incomplete selected populations, omitted mandatory nodes, unexecuted selection plans, passing or unrelated results, changed graph edges or validation declarations, detached seeded-defect identities, cross-tree/objective/policy replay, altered dependency, gate, or disposition records, and forged content digests.
- Evidence source policy: This child goal and its discovery report do not satisfy their own evidence requirement. Only a fresh passing ValidationDAGReceipt from the bound producer/test on the current repository tree and policy qualifies, and only when it contains the complete selected population, validated dependency and authority-gate records, an actually executed transitively selected defect-detecting result, and explicit downstream closure. A static impact graph, selected command list, partial population, direct-only failure, unexecuted or expected-failure-only validation, cached result from another tree, todo/task prose, and a bare requirement ID do not.
- Outputs: ipfs_accelerate_py/agent_supervisor/proposal_validation.py, ipfs_accelerate_py/agent_supervisor/validation_scheduler.py, ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/formal_plan_conformance.py, test/api/test_agent_supervisor_proposal_validation.py, test/api/test_agent_supervisor_validation_dag.py, test/api/test_agent_supervisor_semantic_validation_pipeline.py
- Validation: python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q
- Acceptance: The validation DAG is derived from the canonical changed-file and dependency/interface impact graph and validated declarations that explicitly name impact targets, node dependencies, and downstream authority gates. Its receipt contains the complete selected population, includes every mandatory direct and transitive validation exactly once, schedules only dependency-ready nodes under bounded parallelism, records selected, executed, succeeded, failed, blocked, and omitted dispositions deterministically, and identifies the failed prerequisite for every blocked descendant. Missing, stale, cyclic, inconsistent, or population-incomplete impact evidence fails closed before it can grant authority. A seeded upstream defect must select and execute its transitively affected consumer test, whose real failure creates explicit closed records for semantic/proof promotion, merge, freshness, and completion authority. The exact requirement ID is emitted only by a tamper-evident receipt binding the current tree, objective, policy, accepted proposal, change and graph identities, declaration set, affected closure, selected-node population, DAG dependencies, authority-gate closure, impact paths, actual defect-detecting result, fail-fast disposition, and content digest.
- Objective validation repair: ASI-031/ASI-032 share one validation-gate packet. ASI-031 proves deterministic pre-execution proposal rejection and closed dispatch; ASI-032 proves validated graph declarations, population-complete transitive impact execution, and failure propagation through the common semantic/proof/merge/freshness/completion authority boundary. The joint command runs `python -m pytest test/api/test_agent_supervisor_proposal_validation.py test/api/test_agent_supervisor_validation_dag.py test/api/test_agent_supervisor_semantic_validation_pipeline.py -q` on the same current tree, so neither preflight behavior, impact population, nor downstream closure can be promoted independently. G100 and G101 are independently testable and fully partition the pre-execution and post-admission obligations at depth 2; add a smaller child only if a future typed receipt cannot express graph validation, selected-population completeness, or authority closure without crossing that partition.
- Refinement depth: 2
- Embedding query: 266404049326363900535699811645710804440
- AST query: TRANSITIVE_IMPACT_REQUIREMENT_ID TransitiveImpactValidationEvidence ImpactDependencyGraph.validation_targets ValidationDAGNodeRecord ValidationAuthorityGateRecord REQUIRED_AUTHORITY_GATES ValidationDAGReceipt.required_validation_ids ValidationDAGReceipt.selected_node_ids ValidationDAGReceipt.coverage_complete ValidationDAGReceipt.authority_gates
- Parallel lane: agent-supervisor/self-improvement/validation
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `266404049326363900535699811645710804440` with a narrow, verifiable change.

## ASI-G103 Prove 031486194157679117987393491870400400279 for Unified Python, CLI, and MCP supervisor control

- Status: active
- Parent: ASI-G070
- Fib priority: 5000
- Track: control
- Priority: P0
- Bundle: agent-supervisor/self-improvement/control
- Goal: Prove that Python, the unified CLI, and policy-controlled MCP tools expose the same closed supervisor operation contract and canonical behavior.
- Evidence: 031486194157679117987393491870400400279
- Evidence criterion: Every Python control operation has a CLI command and lazily registered MCP tool using the same operation-specific request JSON Schema and canonical OperationResult envelope; independently invoked success, proposal/dry-run, bounded failure, and mutation/lifecycle cases agree on request/result identity, status, error, authority, effects, preview, idempotency, and audit receipt semantics.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/control_contracts.py CONTROL_SURFACE_PARITY_REQUIREMENT_ID and the content-addressed ControlSurfaceParityEvidence/ControlSurfaceParityCase contracts. The unified CLI delegates through control_cli.run_agent_cli, MCP delegates through native_agent_supervisor_tools.execute_agent_supervisor_operation, and both call SupervisorControlService.execute directly. test/api/test_agent_supervisor_control_plane.py validates the shared schema and evidence contract; test/test_unified_cli_agent_supervisor.py validates CLI parity and safety; test/mcp_server/test_agent_supervisor_tools.py independently invokes all three surfaces and constructs the qualifying receipt.
- Proof obligation map: (1) the complete closed Operation vocabulary is represented exactly once in Python, CLI COMMAND_OPERATIONS, and AGENT_SUPERVISOR_OPERATION_TOOLS; (2) operation_request_json_schema and operation_result_json_schema are the sole advertised schemas and constrain each named MCP tool to its canonical operation; (3) CLI and MCP parse OperationRequest rather than manufacturing transport-local authority and serialize exact OperationResult.to_record output; (4) read bounds, proposal authority, dry-run effects, stable errors, lifecycle bindings, idempotent replay, and audit IDs cannot drift by surface; (5) CLI real mutations require a complete canonical request and MCP execution requires a server-configured allowlist/service; (6) discovery registers only static callables/schema and does not resolve a service; and (7) ControlSurfaceParityEvidence rejects missing operations/surfaces, stale tree/objective/policy cases, forged schema digests, mismatched requests, and any changed result field or content identity.
- Evidence source policy: This child goal, its todo, and its discovery report do not satisfy their own evidence requirement. Only a fresh passing ControlSurfaceParityEvidence from the bound three-suite matrix on the current tree and policy qualifies. Matching command/tool names, shared prose, an opaque object schema, copied records that were not independently invoked, partial operation coverage, or normalized results that discard status/error/effect/audit fields do not.
- Outputs: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, ipfs_accelerate_py/mcp_server/server.py, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_control_lifecycle.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Acceptance: The shared schema describes all operations; every CLI/MCP adapter decodes and dispatches the canonical request directly; canonical records are exactly equal to Python behavior; bounded reads and watches cannot exceed contract limits; unsafe CLI defaults and unconfigured MCP mutation authority fail closed; and the exact requirement ID appears only in a tree/objective/policy-bound parity evidence record that rejects any surface, vocabulary, schema, or behavior drift.
- Objective validation repair: ASI-G103 and ASI-G104 form the control packet. G103 owns cross-surface schema/behavior equivalence; G104 owns mutation authorization/idempotency/lease/effect/audit enforcement. Their joint validation command executes the same current-tree control, lifecycle, CLI, and MCP suites so parity cannot be promoted by weakening mutation guards and mutation guards cannot be promoted through an untested transport. The obligations are independently receipted at depth 2, so no smaller child is needed.
- Refinement depth: 2
- Embedding query: 031486194157679117987393491870400400279
- AST query: CONTROL_SURFACE_PARITY_REQUIREMENT_ID ControlSurfaceParityEvidence ControlSurfaceParityCase operation_request_json_schema operation_result_json_schema COMMAND_OPERATIONS AGENT_SUPERVISOR_OPERATION_TOOLS run_agent_cli execute_agent_supervisor_operation
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `031486194157679117987393491870400400279` with a narrow, verifiable change.

## ASI-G104 Prove 184125100306462690646212311073240043804 for Unified Python, CLI, and MCP supervisor control

- Status: active
- Parent: ASI-G070
- Fib priority: 5001
- Track: control
- Priority: P0
- Bundle: agent-supervisor/self-improvement/control
- Goal: Prove that no real supervisor mutation executes without exact authorization, scoped idempotency, declared effects, and a current lease/fencing epoch, and that accepted mutations are audited and replay-safe.
- Evidence: 184125100306462690646212311073240043804
- Evidence criterion: A real mutation with all exact bindings applies only its declared effects, emits an audit receipt, and replays the exact prior result without a second backend call, while missing authorization, idempotency, or lease/fencing variants deterministically fail before dispatch.
- Evidence producer: ipfs_accelerate_py/agent_supervisor/control_contracts.py CONTROL_MUTATION_GUARD_REQUIREMENT_ID and the content-addressed ControlMutationGuardEvidence/MutationGuardRejection contracts, backed by OperationRequest cross-field validation and SupervisorControlService authorization freshness, allowlist, lease, effect, audit, and idempotency enforcement. test/api/test_agent_supervisor_control_lifecycle.py constructs the qualifying applied/replayed/rejected evidence matrix; test/api/test_agent_supervisor_control_plane.py validates authorization expiry, stale fences, allowlists, idempotency conflicts, restart replay, audit receipts, dry-run non-dispatch, and backend call counts; CLI/MCP tests prove transports cannot bypass the same parser/service.
- Proof obligation map: (1) the request binds operation, roots, repository tree, objective revision, policy revision, caller, expected effects, authorization decision, idempotency scope, lease ID, and fencing epoch; (2) permit authority, authorized effect IDs, identity fields, expiry, and external policy/lease validators are checked before dispatch; (3) dry-run is proposal-only, invokes no mutation backend, applies no effects, and requires no fabricated permit; (4) a real result cannot claim undeclared or reshaped effects and applied effects require an audit receipt; (5) exact idempotent replay returns the prior content identity without another backend call while changed payload reuse conflicts; and (6) ControlMutationGuardEvidence independently replays each missing guard through OperationRequest, binds the applied and replay result to the same current request/tree/objective/policy, and rejects incomplete guard populations, non-applied results, missing audits, or changed replay records.
- Evidence source policy: This child goal and discovery/task prose are non-qualifying. Only a fresh passing ControlMutationGuardEvidence from the bound current-tree suites qualifies. A permit alone, dry-run preview, successful handler call without applied-effect/audit binding, one-process cache entry without replay proof, rejection text not reproduced by the canonical parser, or an unbound occurrence of the requirement ID does not.
- Outputs: ipfs_accelerate_py/agent_supervisor/control_contracts.py, ipfs_accelerate_py/agent_supervisor/control_plane.py, ipfs_accelerate_py/agent_supervisor/control_cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_control_lifecycle.py, test/test_unified_cli_agent_supervisor.py, test/mcp_server/test_agent_supervisor_tools.py
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_control_lifecycle.py test/test_unified_cli_agent_supervisor.py test/mcp_server/test_agent_supervisor_tools.py -q
- Acceptance: Unauthorized, unscoped, unfenced, stale, path-escaping, or undeclared-effect mutations fail before dispatch on every surface; dry-run stays proposal-only; a permitted current mutation emits a typed applied-effect audit receipt; exact retries and restart replay do not duplicate the backend effect; conflicting reuse fails; and only the complete tamper-evident applied/replayed/rejection matrix emits the exact requirement ID.
- Objective validation repair: G103/G104 share the joint control packet and validation command but retain separate evidence records. Cross-surface parity cannot substitute for mutation authority proof, and mutation proof cannot substitute for complete transport vocabulary/schema parity. Existing depth-2 children fully partition the obligations.
- Refinement depth: 2
- Embedding query: 184125100306462690646212311073240043804
- AST query: CONTROL_MUTATION_GUARD_REQUIREMENT_ID ControlMutationGuardEvidence MutationGuardRejection OperationRequest._validate_mutation_bindings SupervisorControlService._check_authorization SupervisorControlService._check_idempotency SupervisorControlService._check_lease ControlAuditReceipt
- Parallel lane: agent-supervisor/self-improvement/control
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `184125100306462690646212311073240043804` with a narrow, verifiable change.
