# Leanstral-Assisted Goal Development Task Board

This board implements optional Leanstral-assisted goal and subgoal development
for the agent supervisor. Leanstral output remains an untrusted proposal.
Canonical objectives, assumptions, vocabulary, admission, proof authority, and
completion remain owned by deterministic supervisor components.

## LEAN-GOAL-001 Define bounded goal-development contracts

- Status: completed
- Completion: manual
- Priority: P0
- Track: leanstral-goal-development
- Depends on:
- Outputs: ipfs_accelerate_py/agent_supervisor/goal_development_contracts.py, test/api/test_agent_supervisor_goal_development_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_goal_development_contracts.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: goal-contracts
- Predicted files: ipfs_accelerate_py/agent_supervisor/goal_development_contracts.py, test/api/test_agent_supervisor_goal_development_contracts.py
- Acceptance: First inspect formal_planning_contracts.py, objective_graph.py, proof_context.py, and leanstral_proof_provider.py and reuse their canonical identity, immutable serialization, and bounds conventions. Add versioned GoalDevelopmentRequest, GoalDecompositionDraft, GoalDevelopmentPolicy, GoalDevelopmentMode, proposal receipt, and admission receipt contracts. Freeze the root goal, satisfaction formula, assumptions, evidence requirements, vocabulary profile, repository tree, scope, and policy digest. Support off, shadow, assist, auto_safe, and repair_only modes. Drafts must be explicitly unverified, content addressed, bounded by depth, breadth, count, bytes, and tokens, and unable to claim proof, admission, implementation conformance, or completion. Add round-trip, determinism, bounds, root-mutation, hidden-assumption, and invalid-authority tests. Do not expose new public package exports in this task.

## LEAN-GOAL-002 Make subgoal satisfaction first-class in the reviewed type system

- Status: completed
- Completion: manual
- Priority: P0
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-001
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_logic_vocabulary.py, ipfs_accelerate_py/agent_supervisor/formal_planning_contracts.py, test/api/test_agent_supervisor_formal_planning_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_planning_contracts.py test/api/test_agent_supervisor_formal_plan_compiler.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: typed-subgoals
- Predicted files: ipfs_accelerate_py/agent_supervisor/formal_logic_vocabulary.py, ipfs_accelerate_py/agent_supervisor/formal_planning_contracts.py, test/api/test_agent_supervisor_formal_planning_contracts.py
- Acceptance: Extend the reviewed vocabulary with a typed subgoal-satisfaction predicate and construction API without adding a natural-language parser. Preserve vocabulary versioning and deterministic identities. Make refinement mode and explicit parent linkage representable using reviewed, typed records, distinguishing sufficient refinement from equivalence without silently strengthening a root objective. Reject unknown sorts, predicates, operators, formula references, self-dependencies, and cyclic subgoal graphs. Add focused positive and adversarial tests while preserving compatibility with existing plans that contain no subgoals.

## LEAN-GOAL-003 Compile and validate canonical subgoal hierarchies

- Status: completed
- Completion: manual
- Priority: P0
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-002
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/formal_plan_validator.py, test/api/test_agent_supervisor_formal_plan_compiler.py, test/api/test_agent_supervisor_formal_plan_validator.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_plan_compiler.py test/api/test_agent_supervisor_formal_plan_validator.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: typed-subgoals
- Predicted files: ipfs_accelerate_py/agent_supervisor/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/formal_plan_validator.py, test/api/test_agent_supervisor_formal_plan_compiler.py, test/api/test_agent_supervisor_formal_plan_validator.py
- Acceptance: Replace the compiler's empty subgoal projection with deterministic compilation from reviewed objective work records. Bind every subgoal to a goal, satisfaction formula, dependencies, evidence requirements, and source IDs. Extend bounded validation to check subgoal witnesses, dependency readiness, evidence coverage, parent-refinement obligations, unsupported semantics, liveness, and circular evidence. Preserve the distinction between bounded consistency and kernel verification, and never infer generated-code assurance from plan validation. Add compatibility, countermodel, unsupported, timeout, and stale-evidence tests.

## LEAN-GOAL-004 Add a capability-isolated Leanstral goal-development provider

- Status: completed
- Completion: manual
- Priority: P0
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-001
- Outputs: ipfs_accelerate_py/agent_supervisor/leanstral_goal_development.py, test/api/test_agent_supervisor_leanstral_goal_development.py
- Validation: python -m pytest test/api/test_agent_supervisor_leanstral_goal_development.py test/api/test_agent_supervisor_leanstral_proof_provider.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: leanstral-provider
- Predicted files: ipfs_accelerate_py/agent_supervisor/leanstral_goal_development.py, test/api/test_agent_supervisor_leanstral_goal_development.py
- Acceptance: Reuse the existing llm_router transport and Leanstral resource isolation, but introduce a separate versioned goal-development operation instead of treating implementation planning as theorem proving. Build a bounded context from immutable goal, evidence-gap, AST/GraphRAG reference, capability, prior-counterexample, and reusable-receipt records. Require strict JSON and allowlisted template, evidence, assurance, resource, scope, and validation-check IDs. Reject root mutation, arbitrary formulas, arbitrary shell commands, unknown fields, cycles, excessive output, canonical-source injection, and attempts to perform a kernel check. Route unavailability, timeout, cancellation, malformed output, and overload to an explicit deterministic-fallback result without stalling the supervisor.

## LEAN-GOAL-005 Generate and independently verify refinement obligations

- Status: completed
- Completion: manual
- Priority: P0
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-003, LEAN-GOAL-004
- Outputs: ipfs_accelerate_py/agent_supervisor/goal_refinement_verification.py, ipfs_accelerate_py/agent_supervisor/multi_prover_router.py, test/api/test_agent_supervisor_goal_refinement_verification.py
- Validation: python -m pytest test/api/test_agent_supervisor_goal_refinement_verification.py test/api/test_agent_supervisor_multi_prover_router.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: refinement-verification
- Predicted files: ipfs_accelerate_py/agent_supervisor/goal_refinement_verification.py, ipfs_accelerate_py/agent_supervisor/multi_prover_router.py, test/api/test_agent_supervisor_goal_refinement_verification.py
- Acceptance: Deterministically derive obligations for child-to-parent sufficiency or declared equivalence, acceptance-criterion coverage, evidence production, task-effect sufficiency, dependency liveness, authority, resource feasibility, and deontic consistency. Classify obligations into the existing typed-planning, temporal/deontic, finite-constraint, state-machine, and first-order property families. Leanstral may supply only model-assistant candidates. Hammer, ATP, SMT, and domain reasoner results remain candidates unless the existing policy grants bounded model-checking authority; high-assurance proof requires independent Lean, Coq, or Isabelle reconstruction. Persist every attempt and bounded counterexample. Support no more than two policy-controlled Leanstral repair rounds and prove that repair cannot change the frozen root or assumptions.

## LEAN-GOAL-006 Materialize admitted goal and subgoal proposals transactionally

- Status: completed
- Completion: manual
- Priority: P1
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-005
- Outputs: ipfs_accelerate_py/agent_supervisor/objective_daemon.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, test/api/test_agent_supervisor_goal_generation.py
- Validation: python -m pytest test/api/test_agent_supervisor_goal_generation.py test/api/test_agent_supervisor_objective_graph.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: objective-materialization
- Predicted files: ipfs_accelerate_py/agent_supervisor/objective_daemon.py, ipfs_accelerate_py/agent_supervisor/objective_tracker.py, ipfs_accelerate_py/agent_supervisor/objective_graph.py, test/api/test_agent_supervisor_goal_generation.py
- Acceptance: Add preview-first, transactional admission for accepted GOAL and SUBGOAL ObjectiveWorkProposal records instead of leaving them indefinitely in the generation ledger. Preserve canonical IDs, parent hierarchy, dependencies, evidence requirements, depth and breadth limits, semantic deduplication, and lifecycle ownership. Shadow mode must never alter the objective heap. Assist mode must persist a reviewable proposal only. Auto-safe admission must require no new assumptions, no unsupported semantics, all hard policy gates, and the configured authoritative receipts. Any partial write, stale tree, lease conflict, or changed root must fail closed and remain resumable.

## LEAN-GOAL-007 Bind implementation results to fresh code-conformance obligations

- Status: completed
- Completion: manual
- Priority: P1
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-006
- Outputs: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/goal_completion.py, test/api/test_agent_supervisor_code_proof_scopes.py, test/api/test_agent_supervisor_proof_goal_completion.py
- Validation: python -m pytest test/api/test_agent_supervisor_code_proof_scopes.py test/api/test_agent_supervisor_proof_goal_completion.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: code-conformance
- Predicted files: ipfs_accelerate_py/agent_supervisor/code_proof_obligations.py, ipfs_accelerate_py/agent_supervisor/goal_completion.py, test/api/test_agent_supervisor_code_proof_scopes.py, test/api/test_agent_supervisor_proof_goal_completion.py
- Acceptance: Derive fresh implementation obligations from changed AST symbols, interfaces, effects, tests, and runtime evidence after Codex completes a task. Bind every receipt to the accepted plan, repository tree, changed scope, assumptions, and validation bounds. Preserve PlanAssurance separation so plan consistency and conformance receipts cannot be promoted to code-proof receipts. Reopen provisionally or verified complete goals when required bindings become stale or contradictory. Add adversarial tests proving a verified decomposition, model-generated proof claim, cached old receipt, or passing type check alone cannot mark generated code correct.

## LEAN-GOAL-008 Add route-aware capability, scheduling, cache, and metrics support

- Status: completed
- Completion: manual
- Priority: P1
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-004, LEAN-GOAL-005
- Outputs: ipfs_accelerate_py/agent_supervisor/formal_verification_capabilities.py, ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, ipfs_accelerate_py/agent_supervisor/proof_metrics.py, test/api/test_agent_supervisor_formal_verification_capabilities.py
- Validation: python -m pytest test/api/test_agent_supervisor_formal_verification_capabilities.py test/api/test_agent_supervisor_resource_scheduler.py test/api/test_agent_supervisor_formal_verification_cache.py test/api/test_agent_supervisor_proof_metrics.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: goal-runtime
- Predicted files: ipfs_accelerate_py/agent_supervisor/formal_verification_capabilities.py, ipfs_accelerate_py/agent_supervisor/resource_scheduler.py, ipfs_accelerate_py/agent_supervisor/formal_verification_cache.py, ipfs_accelerate_py/agent_supervisor/proof_metrics.py, test/api/test_agent_supervisor_formal_verification_capabilities.py
- Acceptance: Split Leanstral route readiness, local model execution, legal-language preprocessing, codec availability, and kernel verification into independent capabilities. Add an optional bounded inference canary and discover the effective context limit from configured route, server, model, output reserve, and safety margin. Schedule model drafting, deterministic type checking, solver portfolios, and kernel reconstruction in separate resource classes with backpressure, cancellation, and deterministic fallback. Cache untrusted drafts separately from verifier receipts using goal, tree, vocabulary, compiler, model route/version, assumptions, bounds, and policy digests. Add metrics for availability, schema acceptance, proof closure, fallback, repair convergence, latency, token use, cache hits, unsupported semantics, and false-completion prevention.

## LEAN-GOAL-009 Add end-to-end shadow-mode integration

- Status: completed
- Completion: manual
- Priority: P1
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-006, LEAN-GOAL-007, LEAN-GOAL-008
- Outputs: ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_leanstral_goal_lifecycle_e2e.py, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md
- Validation: python -m pytest test/api/test_agent_supervisor_leanstral_goal_lifecycle_e2e.py test/api/test_agent_supervisor_goal_lifecycle_e2e.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: goal-integration
- Predicted files: ipfs_accelerate_py/agent_supervisor/__init__.py, test/api/test_agent_supervisor_leanstral_goal_lifecycle_e2e.py, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md
- Acceptance: Wire the new contracts and services through reviewed package exports and an explicitly configured supervisor path whose default mode is shadow. Exercise root freezing, bounded context, multiple candidate drafts, malformed and unavailable fallback, type rejection, counterexample repair, independent proof acceptance, objective preview, implementation conformance, stale receipt reopening, and restart recovery in fixture-only end-to-end tests. Demonstrate that shadow mode produces audit artifacts and metrics while leaving the objective heap and completion state unchanged. Document trust boundaries, modes, capability requirements, receipts, and operational controls.

## LEAN-GOAL-010 Build paired benchmarks and rollout gates

- Status: todo
- Completion: manual
- Priority: P2
- Track: leanstral-goal-development
- Depends on: LEAN-GOAL-009
- Outputs: test/api/test_agent_supervisor_leanstral_goal_benchmark.py, docs/architecture/AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md
- Validation: python -m pytest test/api/test_agent_supervisor_leanstral_goal_benchmark.py -q
- Board namespace: agent-supervisor-leanstral-goal-development-v1
- Parallel lane: goal-rollout
- Predicted files: test/api/test_agent_supervisor_leanstral_goal_benchmark.py, docs/architecture/AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md
- Acceptance: Add a deterministic fixture benchmark comparing current evidence-based refinement with shadow-mode Leanstral on historical, incomplete, contradictory, adversarial, and over-broad goals. Report schema and type acceptance, evidence coverage, authoritative proof closure, unsupported semantics, duplicate/conflict rate, critical path, available parallel width, repair convergence, latency, token cost, fallback rate, and false-completion count. Define canary promotion gates for off to shadow to assist to auto-safe, but do not enable auto-safe by default or make live model/network access a test requirement. Promotion must require zero false completions, no authority-boundary violations, stable restart recovery, and a material paired improvement over the deterministic baseline.
