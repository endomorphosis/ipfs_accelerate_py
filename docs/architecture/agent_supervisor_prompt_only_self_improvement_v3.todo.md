# Agent Supervisor Prompt-Only Self-Improvement v3 Task Board

This is the executable task projection for
`agent_supervisor_prompt_only_self_improvement_v3.objectives.md`. Use task
prefix `## ASE3-` and board namespace
`agent-supervisor-prompt-only-self-improvement-v3`.

Historical ASE/ASE2 state is evidence input only. Every task starts `todo` and
requires a fresh v3 acceptance receipt on the current integration tree. Use one
DuckDB coordination shard, shared lease/fence authority, immutable event log,
isolated worktrees, predicted-file conflict admission, and serialized merge
queue for all lanes.

Initial automatic limits are two implementation attempts per task, three
refill epochs, eight new goals per epoch, twenty-four new tasks per epoch,
forty-eight open tasks, refinement depth three, and three recoveries per thirty
minutes. Missing resource or telemetry evidence grants zero additional
capacity. These are signed profile values, never prompt-controlled values.

The dependency graph admits these waves:

```text
Wave 0:            ASE3-000
Wave 1 (parallel): ASE3-001 ASE3-002 ASE3-003
Wave 2:            ASE3-004
Wave 3 (parallel): ASE3-005 ASE3-006 ASE3-007
Wave 4:            ASE3-008
Wave 5:            ASE3-009
Wave 6 (parallel): ASE3-010 ASE3-011
Wave 7:            ASE3-012
Wave 8:            ASE3-013
Wave 9:            ASE3-014
```

## ASE3-000 Establish current-main truth and a safe convergence manifest

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: convergence
- Depends on:
- Goal id: ASE3-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py, test/api/test_agent_supervisor_prompt_v3_convergence.py, config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json, .gitignore, data/agent_supervisor/prompt_only_self_improvement_v3/convergence
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_convergence.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/convergence
- Parallel lane: convergence
- Resource class: coordinator
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py, test/api/test_agent_supervisor_prompt_v3_convergence.py, config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json, .gitignore, data/agent_supervisor/prompt_only_self_improvement_v3/convergence
- Interfaces: CurrentMainBaseline, HistoricalStateContradictionReport, RescueArtifactDisposition, ConvergenceManifest, CleanIntegrationWorktreeReceipt
- Conflict policy: Bootstrap and evidence only; do not modify existing user work, activate old rollout state, restart old prompt-only processes, merge the rescue branch wholesale, or claim any historical task complete.
- Preconditions: Current checkout, Git object database, rescue branch, v1/v2 state roots, and current tests are readable; any uncommitted user changes remain untouched.
- Effects: Record exact main, checkout, submodule, dirty-path, source-board, registry/event, test, and rescue-branch identities; mark dead stale-running projections non-authoritative; classify each unique rescue file and commit as port, rewrite, superseded, or discard; create a fresh ignored state namespace and clean isolated integration worktree; seal a three-lane configured-board profile with exact source, provider, retry, timeout, protected-path, and merge-target bindings.
- Evidence subset: exact base/tree CIDs, dirty-change preservation, branch divergence, source/runtime/state contradiction matrix, per-artifact disposition, protected paths, clean worktree and state-root receipt
- Acceptance: The manifest accounts for every v2-only implementation and test, proves no user change was overwritten, binds all downstream work to one current base, prevents stale ASE/ASE2 receipts or branch-local commits from satisfying v3 completion, and the sealed scheduler profile passes current-branch preflight before launch.

## ASE3-001 Compose trusted invocation context and infer normal runtime arguments

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ambient-inference
- Depends on: ASE3-000
- Goal id: ASE3-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, test/api/test_agent_supervisor_prompt_v3_resolution.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_resolution.py test/api/test_agent_supervisor_inference_runtime.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/ambient-inference
- Parallel lane: ambient-inference
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, test/api/test_agent_supervisor_prompt_v3_resolution.py
- Interfaces: InvocationContext, TrustedEvidenceCollector, LocalInvocationContextFactory, PythonInvocationContextFactory, MCPInvocationContextFactory, MCPPlusPlusInvocationContextFactory, SupervisorResolutionService
- Conflict policy: Own evidence collection, precedence, and transport context binding only; do not own authority creation, provider attempts, plan generation, lifecycle effects, or public adapter formatting.
- Preconditions: ASE3-000 supplies an exact integration base and salvage dispositions; existing leaf target, state, objective, authority, capability, profile, resource, validation, topology, and plan resolvers import on that base.
- Effects: Replace the prototype heuristic profile/receipt-only launch path with canonical resolver composition; freeze one bounded invocation context; resolve repository, state, profile reference, run, objective, task source, resources, validation, and topology in dependency order; bind local Git root, Python allowlist, server alias, or MCP++ UCAN context; emit value/source/freshness/alternatives/confidence receipts and one typed ambiguity continuation.
- Evidence subset: frozen context CID, adapter provenance, deterministic replay, symlink/root denial matrix, field precedence, prompt non-influence matrix, ambiguity decision
- Acceptance: A normal authorized local invocation needs no low-level daemon flags; identical trusted context replays identically across transports; prompt paths, client paths, stale observations, zero/multiple targets, and unauthenticated context never launch effects.

## ASE3-002 Install bounded local authority and exact provider-attempt policy

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority-provider-policy
- Depends on: ASE3-000
- Goal id: ASE3-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, test/api/test_agent_supervisor_prompt_v3_authority.py, test/api/test_agent_supervisor_prompt_v3_provider_route.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_authority.py test/api/test_agent_supervisor_prompt_v3_provider_route.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/authority-provider-policy
- Parallel lane: authority-provider-policy
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, test/api/test_agent_supervisor_prompt_v3_authority.py, test/api/test_agent_supervisor_prompt_v3_provider_route.py
- Interfaces: LocalProfileInitializer, SignedSupervisorProfile, ProfileRotationReceipt, ProviderRoutePolicy, ProviderAttemptReceipt, QuotaExhaustionEvidence, ProviderFallbackReceipt
- Conflict policy: Own key/profile lifecycle and immutable provider/attempt policy only; private keys never enter repository files, logs, argv, prompts, or immutable public replicas; provider fallback is distinct from independent review.
- Preconditions: ASE3-000 supplies an exact repository binding; canonical DID/signature, authority, capability, and provider runner contracts are available.
- Effects: Create/import, verify, inspect, rotate, and revoke a 0600 local signing identity; sign exact repository/effect/budget/resource/provider bounds; compile the installed default provider route; admit a fallback attempt only on fresh typed policy-approved evidence before repository effects and with prompt/scope/budget/authorization equality.
- Evidence subset: key permissions and leak scan, signed profile CID, repository/effect ceiling, rotation/revocation, exact provider identities, typed failure classification, pre-effect proof, once-only attempt identity
- Acceptance: One explicit init enables later prompt-only isolated-worktree work, but never grants current-checkout rewrite, merge, push, deploy, arbitrary secrets/network, or destructive cleanup; prompt-selected providers and non-approved, repeated, stale, or post-effect fallback fail closed.

## ASE3-003 Converge durable run truth and revalidate every effect boundary

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: run-state
- Depends on: ASE3-000
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, test/api/test_agent_supervisor_prompt_v3_run_registry.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_run_registry.py test/api/test_agent_supervisor_run_registry.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/run-state
- Parallel lane: run-state
- Resource class: io-database
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, test/api/test_agent_supervisor_prompt_v3_run_registry.py
- Interfaces: DuckDBRunRegistryBackend, ImmutableRunEpoch, RunRevisionCAS, LaunchPlanGuard, EffectBoundarySnapshot, LaunchRevalidationReceipt
- Conflict policy: Sole owner of v3 mutable run heads and final effect revalidation; JSON and Markdown are import/operator projections only; Parquet/IPLD/IPFS are immutable history/read replicas and never lease authority.
- Preconditions: ASE3-000 supplies a fresh state root; current DuckDB coordination, CID, event, lease, and fence primitives pass their baseline tests.
- Effects: Persist run heads, revisions, adoption and idempotency keys, process-birth identity, event cursors, and CAS in DuckDB; export immutable epochs; migrate only verified historical facts; compare target tree, scope, authority, policy, provider, task source, run revision, lease, fence, and intended effect immediately before each effect.
- Evidence subset: schema migration, concurrent CAS/adoption, restart reconstruction, process-birth identity, immutable epoch parity, stale-field denial matrix, duplicate-effect denial
- Acceptance: Conflicting revisions cannot both win; exact replay adopts the healthy matching process; PID or projection state alone cannot report running; stale or incomplete plans fail before effects; crashes at intent/effect/receipt boundaries have one deterministic continuation.

## ASE3-004 Materialize prompt intent as canonical goals, subgoals, and tasks

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: program-materialization
- Depends on: ASE3-001, ASE3-002
- Goal id: ASE3-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, test/api/test_agent_supervisor_prompt_v3_plan_materializer.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_plan_materializer.py test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/program-materialization
- Parallel lane: program-materialization
- Resource class: io-database
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, test/api/test_agent_supervisor_prompt_v3_plan_materializer.py
- Interfaces: PromptProgramMaterializer, CanonicalGoalGraph, CanonicalTaskGraph, PlanAdmissionReceipt, TaskSourceProjectionReceipt, ProgramRevisionCAS
- Conflict policy: Compose existing prompt planner, plan admission, objective graph, Markdown task source, DuckDB task source, and IPLD adapters; do not create a second parser, scheduler, mutable authority, or prompt persistence path.
- Preconditions: Trusted resolution and policy receipts from ASE3-001/002 are available; existing planner/admission and dual-projection suites pass on the integration base.
- Effects: Derive a root goal and bounded subgoal/task DAG; attach parent/dependency lineage, evidence ownership, outputs, validation, predicted files, conflict/resource lanes, and acceptance; lint and admit; materialize DuckDB authority, bounded Markdown projection, and immutable history with one plan root and revision CAS.
- Evidence subset: redacted intent/digest, goal/task roots, lineage and cycle lint, unsafe-validation/effect denial, DuckDB/Markdown/IPLD parity, idempotent replay, concurrent CAS
- Acceptance: A single prompt yields a supervisor-readable hierarchy with no duplicate/unknown/cyclic dependency, every goal has evidence and producers, every task has required scheduler fields, unsafe proposals fail admission, and raw prompt/secrets do not enter durable or public projections.

## ASE3-005 Compose a real resumable prompt-to-run lifecycle

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-composition
- Depends on: ASE3-003, ASE3-004
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_runtime.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_runtime.py test/api/test_agent_supervisor_lifecycle_orchestrator.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/runtime-composition
- Parallel lane: runtime-composition
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_runtime.py
- Interfaces: StandardSupervisorRuntimeFactory, SupervisorIntentService, CompleteLaunchPlan, PromptToRunSaga, RunHandle
- Conflict policy: Own service composition and saga only; invoke existing domain services through typed adapters; never install a successful in-memory simulation or no-op effect in a production profile.
- Preconditions: ASE3-003 provides durable run/effect authority and ASE3-004 provides admitted materialization; lifecycle, control authorization, validation, rescue, and provider handlers are importable.
- Effects: Resolve, preview, authorize, plan, materialize, start/adopt, observe, steer, validate, and stop through one root-bound resumable saga; compile argv accepted by the actual implementation-supervisor parser; register a real START lifecycle handler; persist intent before effect and receipt after effect; refuse production construction when any required effect handler is absent.
- Evidence subset: capability/handler manifest, complete launch plan, real child birth identity, task source and process start/adopt receipts, crash matrix, typed unavailable path, no-op/simulation reachability denial
- Acceptance: A production run starts or adopts a real lifecycle and durable task source before returning accepted; missing handlers fail typed and nonzero; exact retries resume; no default path immediately fabricates completed or converts a missing effect into success.

## ASE3-006 Compile and enforce adaptive conflict-free parallel execution

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: parallel-scheduling
- Depends on: ASE3-003, ASE3-004
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py, test/api/test_agent_supervisor_prompt_v3_parallelism.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_parallelism.py test/api/test_agent_supervisor_configured_board_scheduler.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/parallel-scheduling
- Parallel lane: parallel-scheduling
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py, test/api/test_agent_supervisor_prompt_v3_parallelism.py
- Interfaces: ParallelExecutionCompiler, ConflictGraph, ResourceAdmissionSnapshot, TaskClaimFence, MergeQueuePlan, ParallelismDecisionReceipt
- Conflict policy: Own plan compilation and scheduler wiring only; reuse task claims, worktrees, leases/fences, resource scheduler, validation, and merge queue; never trust worker-count flags without current capacity evidence.
- Preconditions: Canonical task DAG/materialization and DuckDB claim authority exist; scheduler, worktree, resource, and merge services pass current baseline tests.
- Effects: Propagate `InvocationBudget.max_lanes` instead of a hard-coded ready width; compute ready closure; construct file/scope/resource/provider/validation conflict edges; select a deterministic maximal conflict-free set under inferred caps; claim by task CID; launch isolated worktrees; preserve plan revision and exact execution slice across restart; treat an empty slice as no work and require explicit full-board mode; allow fenced same-revision work stealing; recheck actual diffs; serialize accepted integration; rebase and revalidate remaining work.
- Evidence subset: dependency closure, conflict graph, capacity snapshot, concurrent claim receipts, overlap timeline, conflicting-task serialization, undeclared-overlap fence, merge/current-tree receipts
- Acceptance: At least two disjoint tasks demonstrably overlap in wall-clock time, conflicting or dependency-related tasks never overlap, capacity loss reduces lanes, an empty/restarted slice cannot select another lane's task, duplicate claims/effects are impossible, and undeclared scope overlap fences and replans both attempts.

## ASE3-007 Wire evidence-driven bounded goal and task refill

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: refill-completion
- Depends on: ASE3-004
- Goal id: ASE3-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py, test/api/test_agent_supervisor_prompt_v3_refill.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_refill.py test/api/test_agent_supervisor_backlog_refinery.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/refill-completion
- Parallel lane: refill-completion
- Resource class: coordinator
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py, test/api/test_agent_supervisor_prompt_v3_refill.py
- Interfaces: ResidualEvidenceEvaluator, RefillPolicy, RefillTrigger, RefillEpochCAS, CompletionAuthorityDecision, ProductionSelfImprovementHook
- Conflict policy: Compose existing backlog refinery, objective daemon/tracker, goal completion, Planner Doctor, self-improvement v2, Doctor findings, and task-source append/CAS paths; do not mutate an active canonical board outside revision control or let generated work widen authority/budget or its own acceptance gates.
- Preconditions: ASE3-004 supplies canonical goal/task roots and revisioned projection; current evidence and goal reconciliation services are available.
- Effects: Enable refill and the bounded Planner Doctor/self-improvement hook in the signed production profile; trigger before retry-exhausted paths return and on low-water, drained-open-goal, failed validation/review/merge, current-tree drift, retry exhaustion, or missed rollout threshold; refresh evidence; derive smallest residuals; deduplicate by canonical gap identity; append bounded child goals/tasks; evaluate baseline/candidate on isolated exact trees; force a final scan before completion; trip a circuit breaker on unchanged oscillation.
- Evidence subset: trigger matrix, current-tree residual report, gap identities, append/revision receipts, cooldown/depth/work/epoch budgets, no-refill healthy drain, oscillation and terminal decision
- Acceptance: All trigger and no-trigger cases are deterministic; initial caps are enforced; refilled work has full lineage and scheduler metadata; branch-only or stale-evidence completion reopens convergence work; repeated unchanged residuals end blocked/exhausted rather than looping.

## ASE3-008 Add a live progress watchdog, Doctor integration, and bounded recovery

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: monitoring-recovery
- Depends on: ASE3-006, ASE3-020
- Goal id: ASE3-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_monitor.py, test/api/test_agent_supervisor_prompt_v3_monitor.py, test/api/test_agent_supervisor_task_attempt_limit.py, docs/guides/AGENT_SUPERVISOR_PROMPT_RUNBOOK.md
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_monitor.py test/api/test_agent_supervisor_supervisor_watchdog.py test/api/test_agent_supervisor_task_attempt_limit.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/monitoring-recovery
- Parallel lane: monitoring-recovery
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/run_monitor.py, test/api/test_agent_supervisor_prompt_v3_monitor.py, test/api/test_agent_supervisor_task_attempt_limit.py, docs/guides/AGENT_SUPERVISOR_PROMPT_RUNBOOK.md
- Interfaces: RunHealthSnapshot, SemanticProgressClock, StallClassifier, RecoveryPolicy, RecoveryReceipt, SupervisorDoctorService
- Conflict policy: Own health aggregation and recovery orchestration only; reuse process-birth, heartbeat, temporal monitor, Doctor, rescue, lifecycle, lease/fence, and refill services; detection never implies restart authority.
- Preconditions: Real lifecycle, parallel scheduler, and refill services expose events and authorized callbacks; process and registry identities are durable.
- Effects: Run a live clock; compare heartbeat, event cursor, run revision, phase progress, task/claim/log/validation/merge ages, ready/active/blocked counts, leases/fences, tree reachability, provider/resource state, and refill outcomes; repair stale projections; adopt/restart/rescue exact processes within timestamp-based budgets; emit terminal shutdown.
- Evidence subset: 5-second heartbeat and 30-second stale control policy, 300-second bounded task-progress policy, dead/PID-reuse/frozen/false-idle/soft-complete matrix, authorized recovery callback, retry/backoff window, circuit breaker, status repair and shutdown receipts
- Acceptance: Running cannot outlive its verified process/lease/heartbeat; log noise cannot mask semantic stall; standalone detection reports its authority; at most three canary recoveries occur per thirty minutes; backoff retries later rather than becoming permanent; retry-accounting, idle-heartbeat, quota-attribution, and provider-review deferral tests use current APIs and pass without weaker behavior; every injected incident recovers once or yields typed operator action.

## ASE3-009 Export the production Python facade and stable package API

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: python-facade
- Depends on: ASE3-005, ASE3-008
- Goal id: ASE3-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, ipfs_accelerate_py/__init__.py, test/api/test_agent_supervisor_prompt_v3_python_api.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_python_api.py test/api/test_agent_supervisor_entrypoint_package.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/python-facade
- Parallel lane: python-facade
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, ipfs_accelerate_py/__init__.py, test/api/test_agent_supervisor_prompt_v3_python_api.py
- Interfaces: Supervisor, SupervisorRun, Supervisor.open, Supervisor.run, Supervisor.preview, Supervisor.steer, Supervisor.status, Supervisor.follow, Supervisor.explain, Supervisor.doctor, Supervisor.init_local
- Conflict policy: Sole owner of Python facade and package export fan-in; preserve lazy/cold imports and existing exports; no in-memory successful fallback, hidden global mutable backend, or transport-specific policy.
- Preconditions: Real intent service and monitor/Doctor service are complete; ASE3-000 protected-path manifest accounts for concurrent user edits to package initializers.
- Effects: Add injectable and configured `Supervisor.open`; expose typed run handles and lifecycle methods; infer a sole compatible run when safe; return one typed continuation on ambiguity; lazily export the same API from entrypoints, agent_supervisor, and top-level package.
- Evidence subset: export manifests, import timing/side-effect trace, canonical method request/result receipts, unavailable-backend failure, run inference ambiguity, prompt non-leak
- Acceptance: The documented `from ipfs_accelerate_py import Supervisor` path works from an installed wheel and source tree; cold import starts no integration/process; `run(prompt)` reaches the real service; absent configuration fails typed; no simulated completion path exists.

## ASE3-010 Add the prompt-first product CLI and console registration

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: cli-facade
- Depends on: ASE3-009
- Goal id: ASE3-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/cli_entry.py, pyproject.toml, setup.py, test/api/test_agent_supervisor_prompt_v3_cli.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_cli.py test/api/test_cli_agent.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/cli-facade
- Parallel lane: cli-facade
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/cli.py, ipfs_accelerate_py/cli_entry.py, pyproject.toml, setup.py, test/api/test_agent_supervisor_prompt_v3_cli.py
- Interfaces: supervisor run, supervisor preview, supervisor steer, supervisor status, supervisor follow, supervisor explain, supervisor doctor, supervisor init
- Conflict policy: Sole CLI and packaging owner; compose the Python facade; preserve existing expert `agent` commands and console scripts; help and parse paths must not import or initialize IPFS/provider runtimes.
- Preconditions: Stable Python facade exists; current CLI parser and both packaging manifests are reconciled in the integration tree.
- Effects: Register the `supervisor` group and lifecycle commands; accept positional prompt or bounded stdin; add structured JSON and event streaming; map typed errors to stable exit codes; make advanced flags optional authorized overrides; update console/package metadata and help.
- Evidence subset: cold help trace, command manifest, positional/stdin prompt cases, exit/error mapping, JSON/event parity, expert CLI regression, installed-wheel smoke
- Acceptance: `ipfs-accelerate supervisor run "prompt"` needs no daemon flags and invokes a real service; help is fast and side-effect free; ambiguous/unauthorized/unavailable cases are typed and nonzero; existing expert commands and editable/wheel installations still work.

## ASE3-011 Register equivalent MCP and MCP++ prompt lifecycle tools

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp-facade
- Depends on: ASE3-009
- Goal id: ASE3-G070
- Outputs: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/__init__.py, ipfs_accelerate_py/mcp_server/server.py, test/api/test_agent_supervisor_prompt_v3_mcp.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_mcp.py test/api/test_agent_supervisor_mcplusplus_prompt_entrypoints.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/mcp-facade
- Parallel lane: mcp-facade
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/__init__.py, ipfs_accelerate_py/mcp_server/server.py, test/api/test_agent_supervisor_prompt_v3_mcp.py
- Interfaces: agent_supervisor_run, agent_supervisor_preview, agent_supervisor_steer, agent_supervisor_status, agent_supervisor_follow, agent_supervisor_explain, agent_supervisor_doctor
- Conflict policy: Sole MCP/MCP++ registration owner; adapt authenticated server context to the Python facade; never accept raw client filesystem authority, duplicate policy, or treat transport authentication as inner effect authorization.
- Preconditions: Stable Python facade and context adapter contracts exist; server target alias and MCP++ UCAN validators are available; ASE3-000 protects concurrent server edits.
- Effects: Register versioned JSON-schema tools; bind authenticated principal, server-owned target alias, and for MCP++ verified attenuated UCAN; stream event cursors with backpressure; map typed service outcomes without changing semantics.
- Evidence subset: tool manifest/schema, server alias denial, path/symlink injection, authentication and UCAN attenuation, request/result/run-handle parity, event cursor/backpressure, prompt non-leak
- Acceptance: MCP and MCP++ normal run/preview input is a prompt, all operations reach the real common service, arbitrary paths and insufficient UCAN fail closed, and canonical outcomes match Python/CLI without weakening existing low-level tools.

## ASE3-012 Prove cross-transport, security, compatibility, and crash conformance

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: conformance
- Depends on: ASE3-010, ASE3-011
- Goal id: ASE3-G070
- Outputs: test/api/test_agent_supervisor_prompt_only_v3_conformance.py, test/api/test_agent_supervisor_prompt_only_v3_security.py, test/api/test_agent_supervisor_prompt_only_v3_compatibility.py, docs/guides/AGENT_SUPERVISOR_PROMPT_ENTRYPOINTS.md
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v3_conformance.py test/api/test_agent_supervisor_prompt_only_v3_security.py test/api/test_agent_supervisor_prompt_only_v3_compatibility.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/conformance
- Parallel lane: conformance
- Resource class: test-heavy
- Predicted files: test/api/test_agent_supervisor_prompt_only_v3_conformance.py, test/api/test_agent_supervisor_prompt_only_v3_security.py, test/api/test_agent_supervisor_prompt_only_v3_compatibility.py, docs/guides/AGENT_SUPERVISOR_PROMPT_ENTRYPOINTS.md
- Interfaces: PromptOnlyConformanceMatrix, SecurityAdversaryMatrix, ExpertCompatibilityReport, InstalledDistributionSmoke
- Conflict policy: Test and documentation fan-in only; do not modify implementation, expected capability baselines to conceal drift, safety thresholds, provider classifications, or authority fixtures to make gates green.
- Preconditions: Python, CLI, MCP, and MCP++ facades are integrated on one exact tree; their manifests and generated schemas are available.
- Effects: Execute canonical request/result parity, cold import/help, installed wheel, server alias, UCAN, prompt/secret leak, path injection, stale context, provider, crash-boundary, duplicate launch/effect, and expert compatibility matrices; document supported journey and safe bootstrap.
- Evidence subset: exact tree/environment, four-transport operation matrix, error/exit parity, security denials, prompt/secret leak scan, compatibility delta, cold-start timings, docs command smoke
- Acceptance: Every public operation has equivalent semantics, no production fallback simulates success, no unauthorized input reaches an effect, docs commands execute, cold paths have no integration startup, and capability changes are intentionally classified instead of hidden behind stale snapshot assertions.

## ASE3-013 Run and monitor a bounded self-improvement canary

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: self-host-canary
- Depends on: ASE3-008, ASE3-012
- Goal id: ASE3-G080
- Outputs: test/api/test_agent_supervisor_prompt_only_v3_e2e.py, test/api/test_agent_supervisor_prompt_only_v3_chaos.py, test/api/test_agent_supervisor_prompt_only_v3_load.py, data/agent_supervisor/prompt_only_self_improvement_v3/canary
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v3_e2e.py test/api/test_agent_supervisor_prompt_only_v3_chaos.py test/api/test_agent_supervisor_prompt_only_v3_load.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/self-host-canary
- Parallel lane: self-host-canary
- Resource class: test-heavy
- Predicted files: test/api/test_agent_supervisor_prompt_only_v3_e2e.py, test/api/test_agent_supervisor_prompt_only_v3_chaos.py, test/api/test_agent_supervisor_prompt_only_v3_load.py, data/agent_supervisor/prompt_only_self_improvement_v3/canary
- Interfaces: SelfImprovementCanary, CanaryObservationWindow, FaultInjectionMatrix, CanaryPromotionEvidence
- Conflict policy: Canary/evaluation only in a disposable isolated worktree and fresh state namespace; begin with one lane, expand to two only after conflict admission proof; never target the operator's dirty checkout or authorize merge/push/deploy.
- Preconditions: Cross-transport gates pass on the exact candidate tree; local signed profile and bounded budgets are installed; monitor has a live clock and authorized lifecycle callbacks; rollback is rehearsed.
- Effects: Invoke the new public facade with one bounded supervisor-improvement prompt; prove real child and taskboard creation; observe a disjoint two-task overlap and conflict serialization; inject stale PID, frozen worker, false idle/open goal, branch-only completion, crash boundary, lease loss, and refill residual; sustain healthy operation through the observation window; stop truthfully.
- Evidence subset: prompt digest and run/plan roots, real process births, event/revision timeline, parallel overlap, conflict serialization, forced refill, every recovery decision, budget usage, accepted commit reachability, terminal shutdown
- Acceptance: The canary performs real bounded repository work, not mocks/no-ops; every injected fault recovers once or fails typed within budget; no duplicate/unauthorized effect occurs; accepted work reaches the canary integration head; a final forced residual scan and sustained health window pass.

## ASE3-014 Materialize canonical v3 evidence and stage reversible cutover

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rollout-closeout
- Depends on: ASE3-013
- Goal id: ASE3-G080
- Outputs: data/agent_supervisor/prompt_only_self_improvement_v3/plan, data/agent_supervisor/prompt_only_self_improvement_v3/rollout, test/api/test_agent_supervisor_prompt_only_v3_release.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v3_release.py test/api/test_agent_supervisor_prompt_only_v3_e2e.py test/api/test_agent_supervisor_prompt_only_v3_chaos.py test/api/test_agent_supervisor_prompt_only_v3_load.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/rollout-closeout
- Parallel lane: closeout
- Resource class: coordinator
- Predicted files: data/agent_supervisor/prompt_only_self_improvement_v3/plan, data/agent_supervisor/prompt_only_self_improvement_v3/rollout, test/api/test_agent_supervisor_prompt_only_v3_release.py
- Interfaces: CanonicalV3Plan, V3BundleIndex, CurrentTreeEvidenceJoin, V3RolloutDecision, V3RollbackReceipt, TerminalShutdownReceipt
- Conflict policy: Final evidence/materialization fan-in only; do not rewrite the live source board, accept missing tests, weaken gates, infer promotion authority, or mark completion before accepted commits and all evidence are reachable from the exact release candidate.
- Preconditions: All producer tasks have independent current-tree acceptance receipts; canary state is terminal and healthy; no effectful old v1/v2 lease is active; rollout authority and rollback target are explicit.
- Effects: Reconcile source board, objective heap, DuckDB run state, immutable events, accepted commits, validations, and canary evidence; force one final residual scan; materialize new task/goal/plan roots and DuckDB/Markdown/Parquet/IPLD parity; run a second fresh-tree release pass; emit signed preview/assist/local-auto promotion or rollback decision; fence and shut down exact owned processes.
- Evidence subset: complete task/goal/current-tree join, final residual and completion decision, projection root parity, fresh release reports, active or rollback state, exact authority/profile/coordinator/tree identities, rollback triggers, shutdown receipt
- Acceptance: No unknown dependency, stale completion, missing command/test, branch-unreachable commit, inactive rollout claimed active, open evidence residual, live orphan, unauthorized effect, duplicate effect, projection mismatch, or transport drift is accepted; promotion is explicit, staged, reversible, and bound to the exact release tree.

## ASE3-015 Resolve implementation retry-budget failure for ASE3-001

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASE3-000
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, test/api/test_agent_supervisor_prompt_v3_resolution.py
- Validation: test -f data/agent_supervisor/prompt_only_self_improvement_v3/convergence/infrastructure_retry_credit_restoration_20260808.json
- Parallel lane: ambient-inference
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, test/api/test_agent_supervisor_prompt_v3_resolution.py
- Conflict policy: Own evidence collection, precedence, and transport context binding only; do not own authority creation, provider attempts, plan generation, lifecycle effects, or public adapter formatting.
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: ASE3-001
- Retry failure kind: implementation
- Retry repair discovery: data/agent_supervisor/prompt_only_self_improvement_v3/convergence/infrastructure_retry_credit_restoration_20260808.json
- Canonical board task: false
- Acceptance: Operator adjudication binds the charged attempt to a pre-command fallback sandbox infrastructure failure. Verify the tracked incident receipt has a 40-lowercase-hex remediation commit, a passed exact Codex gpt-5.6-terra medium workspace-write smoke receipt with exit_code 0 and infrastructure_signature_absent true, and a passed positive-count exact board pytest command from the disposable repository cwd with no missing-path, import, toolchain, or bwrap failure; then mark this repair task completed so the supervisor can release ASE3-001 from strategy blocked_tasks. Never edit attempt counters, retry receipts, or task-queue retry fields directly.

## ASE3-016 Resolve implementation retry-budget failure for ASE3-002

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASE3-000
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, test/api/test_agent_supervisor_prompt_v3_authority.py, test/api/test_agent_supervisor_prompt_v3_provider_route.py
- Validation: test -f data/agent_supervisor/prompt_only_self_improvement_v3/convergence/infrastructure_retry_credit_restoration_20260808.json
- Parallel lane: authority-provider-policy
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, test/api/test_agent_supervisor_prompt_v3_authority.py, test/api/test_agent_supervisor_prompt_v3_provider_route.py
- Conflict policy: Own key/profile lifecycle and immutable provider/attempt policy only; private keys never enter repository files, logs, argv, prompts, or immutable public replicas; provider fallback is distinct from independent review.
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: ASE3-002
- Retry failure kind: implementation
- Retry repair discovery: data/agent_supervisor/prompt_only_self_improvement_v3/convergence/infrastructure_retry_credit_restoration_20260808.json
- Canonical board task: false
- Acceptance: Operator adjudication binds the charged attempt to a pre-command fallback sandbox infrastructure failure. Verify the tracked incident receipt has a 40-lowercase-hex remediation commit, a passed exact Codex gpt-5.6-terra medium workspace-write smoke receipt with exit_code 0 and infrastructure_signature_absent true, and a passed positive-count exact board pytest command from the disposable repository cwd with no missing-path, import, toolchain, or bwrap failure; then mark this repair task completed so the supervisor can release ASE3-002 from strategy blocked_tasks. Never edit attempt counters, retry receipts, or task-queue retry fields directly.

## ASE3-017 Resolve implementation retry-budget failure for ASE3-003

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: ASE3-000
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, test/api/test_agent_supervisor_prompt_v3_run_registry.py
- Validation: test -f data/agent_supervisor/prompt_only_self_improvement_v3/convergence/infrastructure_retry_credit_restoration_20260808.json
- Parallel lane: run-state
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, test/api/test_agent_supervisor_prompt_v3_run_registry.py
- Conflict policy: Sole owner of v3 mutable run heads and final effect revalidation; JSON and Markdown are import/operator projections only; Parquet/IPLD/IPFS are immutable history/read replicas and never lease authority.
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: ASE3-003
- Retry failure kind: implementation
- Retry repair discovery: data/agent_supervisor/prompt_only_self_improvement_v3/convergence/infrastructure_retry_credit_restoration_20260808.json
- Canonical board task: false
- Acceptance: Operator adjudication binds the charged attempt to a pre-command fallback sandbox infrastructure failure. Verify the tracked incident receipt has a 40-lowercase-hex remediation commit, a passed exact Codex gpt-5.6-terra medium workspace-write smoke receipt with exit_code 0 and infrastructure_signature_absent true, and a passed positive-count exact board pytest command from the disposable repository cwd with no missing-path, import, toolchain, or bwrap failure; then mark this repair task completed so the supervisor can release ASE3-003 from strategy blocked_tasks. Never edit attempt counters, retry receipts, or task-queue retry fields directly.

## ASE3-018 Harden canonical trusted context and complete resolver composition

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ambient-inference-hardening
- Depends on: ASE3-001
- Goal id: ASE3-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py test/api/test_agent_supervisor_prompt_v3_resolution.py test/api/test_agent_supervisor_inference_runtime.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/ambient-inference-hardening
- Parallel lane: ambient-inference-hardening
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py
- Interfaces: FrozenInvocationContext, CanonicalResolutionPipeline, CanonicalResolutionCore, TransportEvidenceEnvelope, SupervisorResolutionService
- Conflict policy: Own canonical evidence collection, deep freezing, transport-neutral resolution, and leaf-resolver composition only; do not create authority, provider attempts, taskboard mutations, lifecycle effects, or transport-specific success semantics.
- Preconditions: ASE3-001 current-tree implementation and its tests are readable; canonical target, state, profile, run, objective, task-source, resource, validation, and topology resolvers import on the exact integration tree.
- Effects: Deep-freeze and canonically encode a bounded trusted-context core while keeping transport evidence in a separate authenticated envelope; verify the real Git worktree/root and installed signed profile; compose repository, state, profile, run, objective, task source, resources, validation, and topology in dependency order; require Python target allowlists, MCP server aliases, and verified attenuated MCP++ UCANs; eliminate caller-supplied authorization booleans and emit typed zero, multiple, stale, or conflicting-evidence continuations.
- Evidence subset: mutation-stable context CID, deterministic mixed-key/set canonicalization, cross-transport core parity, real-root/profile checks, complete leaf-resolution provenance, path/symlink/client-injection denials, UCAN attenuation, ambiguity matrix, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/post_wave3_residuals_20260808.json#trusted-context-canonical-composition
- Acceptance: Mutating original mappings, nested values, or a prebuilt resolution field after collection cannot change the frozen context or receipt CID; identical trusted facts have one canonical core across local, Python, MCP, and MCP++; nonexistent or symlink profiles, fake .git markers, prompt/client paths, unsigned booleans, and unverified UCANs fail closed; every required launch field is resolved by the production pipeline or launch returns one typed continuation before effects.

## ASE3-019 Seal signed provider authority, authentication lifecycle, and once-only fallback

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority-provider-hardening
- Depends on: ASE3-002
- Goal id: ASE3-G020
- Outputs: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_llm_router_agent_supervisor_fallback_route.py, test/api/test_agent_supervisor_prompt_v3_authority_hardening.py
- Validation: python -m pytest test/api/test_llm_router_agent_supervisor_fallback_route.py test/api/test_agent_supervisor_prompt_v3_authority_hardening.py test/api/test_agent_supervisor_prompt_v3_authority.py test/api/test_agent_supervisor_prompt_v3_provider_route.py test/api/test_agent_supervisor_grok_quota_terra_gate.py test/api/test_agent_supervisor_implementation_provider_receipts.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/authority-provider-hardening
- Parallel lane: authority-provider-hardening
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, test/api/test_llm_router_agent_supervisor_fallback_route.py, test/api/test_agent_supervisor_prompt_v3_authority_hardening.py
- Interfaces: AgentImplementationRoutePlan, AgentImplementationFallbackDecision, SignedSupervisorProfile, SecureLocalIdentityStore, SignedProfileLifecycleReceipt, DurableProviderAttemptCAS, AuthLifecycleFinding, QuotaExhaustionEvidence, ProviderFallbackReceipt
- Conflict policy: Own cryptographic local-profile lifecycle and provider-attempt admission while making `ipfs_accelerate_py.llm_router` the sole owner/export surface for the canonical implementation route plan and typed fallback decision; the scheduler supplies profile data only, the runner owns isolation/process effects/terminal outcome emission only, and the daemon owns task retry accounting only; never persist private key or bearer material in repository files, logs, argv, prompts, events, tests, or immutable public replicas; do not weaken independent review, retry accounting, sandbox boundaries, or provider identity pins.
- Preconditions: ASE3-002 policy objects and the exact Grok-primary, independent-verifier, pinned-Terra fallback path pass their current suites; `data/agent_supervisor/prompt_only_self_improvement_v3/convergence/provider_fallback_policy_authorization_20260808.json` validates against source HEAD `b9c1368a35cee206dff6ff34553782be851fc571`, tree `7aeb7e4d78f5b45d2213173a10deebcf6114092f`, and this namespace; a fresh state namespace is available for durable attempt CAS tests.
- Effects: Export an immutable canonical implementation route plan and typed fallback decision from `ipfs_accelerate_py.llm_router` as the sole provider-policy source; bind the exact board namespace, authorization-artifact SHA-256, authorization kind, source HEAD, source tree, nonempty reviewer identity, and reviewer provider into every route plan and terminal outcome, deny when the reviewer identity or provider matches the chosen fallback implementer, and treat the ambient six-field provider/model/trigger/effort tuple as profile input that cannot authorize fallback by itself; make the scheduler pass only the route profile, the runner apply only isolation/process effects and emit the terminal outcome, and the daemon apply only task retry accounting; remove duplicate provider/model/trigger/effort, authentication/quota classification, and fallback allow/deny logic from those layers; sign exact repository, baseline tree, effects, budgets, resources, provider route, reviewer, and fallback bounds with a verifiable Ed25519 did:key identity stored as an owned regular nonsymlink 0600 file; persist signed rotation and revocation so copied old authority cannot revive; require either runner-produced typed pre-effect Grok authentication-unavailable evidence or independently verified native signed typed hard-quota evidence, with mandatory wall-clock freshness and exact nonempty invocation/task/prompt/scope/budget/authority/provider equality; reserve exactly one Codex `gpt-5.6-terra` fallback at `high` reasoning through durable compare-and-swap before any fallback effect, execute it only inside the pinned external Docker boundary, and adopt the winning receipt after crash as the same logical attempt without counter mutation or provider-capacity restoration; deny fallback for arbitrary errors, rate limits, transport failures, invalid requests, unknown failures, a changed workspace, or post-effect evidence.
- Evidence subset: source-bound prospective operator authorization, immutable `ipfs_accelerate_py.llm_router` route-plan and typed-decision exports, scheduler/runner/daemon ownership and no-duplication matrix, signature/DID and key-path matrix, signed bounds, persistent rotation/revocation, typed authentication-unavailable and native quota/session binding, mandatory freshness, exact-field mismatch denials, concurrent/restart once-only fallback CAS, pinned Docker image/runtime/mount/environment receipt, Codex gpt-5.6-terra high command receipt, independent signed review, immutable historical 2026-08-08 expired-host-auth incident event sha256:e2dee32eb866a9a4216c809318f4066bc49bf33e1e0ef3290365cf4ccaf58f97 and redacted log SHA256 2724af1a5b52fadae7130b4a80081cf9849dabc0f0104f839033474fff332596
- Acceptance: Public immutable `AgentImplementationRoutePlan` and `AgentImplementationFallbackDecision` exports from `ipfs_accelerate_py.llm_router` are the only canonical provider-policy source; a missing or mismatched board namespace, authorization-artifact SHA-256, authorization kind, source HEAD, source tree, reviewer identity, or reviewer provider denies, a chosen fallback implementer cannot be its own reviewer, and the ambient six-field route profile alone never creates authority; the scheduler, runner, and daemon contain no independent route tuple, failure classifier, or fallback allow/deny branch, the runner executes only the router decision and emits its terminal outcome, and the daemon never reclassifies provider evidence and changes only task retry accounting; only a currently valid signed profile and the source-bound prospective authorization can authorize bounded effects; symlink, ownership, permission, substitution, copied-revoked-key, incomplete-bound, or non-descendant-tree cases fail closed; exactly one concurrent or restarted worker automatically admits a matching pre-effect Codex `gpt-5.6-terra` fallback at `high` reasoning for only typed Grok authentication-unavailable or independently verified hard-quota evidence and adopts it as the same logical attempt; arbitrary caller DTOs, optional/stale timestamps, empty equality fields, arbitrary/generic/rate-limit/transport/invalid/unknown errors, changed-workspace or post-effect evidence, self-review, and route mismatches deny; no fallback path mutates or restores attempt counters, including provider-capacity restoration, or enables legacy objective/codebase refill, and the historical `Not signed in` record remains uncharged, immutable evidence rather than being rewritten or reclassified.

## ASE3-023 Repair production plan-bound adaptive parallel dispatch

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: parallel-scheduling-repair
- Depends on: ASE3-003, ASE3-004, ASE3-005, ASE3-006
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_prompt_v3_parallelism.py, test/api/test_agent_supervisor_configured_board_scheduler.py, test/api/test_agent_supervisor_implementation_supervisor_runner.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_parallelism.py test/api/test_agent_supervisor_configured_board_scheduler.py test/api/test_agent_supervisor_implementation_supervisor_runner.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/parallel-scheduling-repair
- Parallel lane: parallel-scheduling-repair
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_prompt_v3_parallelism.py, test/api/test_agent_supervisor_configured_board_scheduler.py, test/api/test_agent_supervisor_implementation_supervisor_runner.py
- Interfaces: ProductionParallelPlanAdapter, ActivePlanBinding, PlanRevisionStore, ConfiguredBoardExecutionSlices, PlanBoundSupervisorChild, ParallelismDecisionReceipt
- Conflict policy: Repair ASE3-006 production wiring only; adapt the canonical InvocationBudget, admitted plan/revision store, task-source claim authority, worktree lifecycle, leases/fences, validation, and merge queue already present in production. Do not create a second InvocationBudget, SQLite execution ledger, task parser, claim store, worktree runner, merge path, or provider policy. Do not edit ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py or any ASE3-019-owned route/profile/provider file.
- Preconditions: ASE3-004 admitted plan and PlanRevisionStore contracts, ASE3-005 lifecycle/current-tree services, existing daemon exact task-ID/CID slice inputs, and the fenced Wave3b false-completion recovery artifact are readable on the exact integration tree.
- Effects: Replace the standalone ASE3-006 ledger/types with a thin production adapter over the canonical contracts and stores; compile the largest deterministic conflict-free ready set under canonical InvocationBudget.max_lanes and fresh host/provider capacity; durably publish one immutable ActivePlanBinding before launch; make the configured scheduler and multi-runner launch only its exact nonempty per-lane task-ID/CID slices with legacy hash sharding disabled for slice-bound children; run bounded wave passes so the coordinator replans before a next wave; persist and recheck revision, plan root, execution-plan CID, capacity snapshot, slice, process birth, lease, and fence across supervisor child restart; acquire or safely adopt the durable adaptive claim before any child/provider spawn; make an empty slice launch no implementation daemon and never mean full board; perform work stealing only through a same-revision CAS reassignment; retain existing claim/worktree/lease/fence/validation/merge services; inspect actual changed paths and record the effect before merge authority; fence and replan on overlap; serialize merge and rebase/revalidate surviving lanes on the new integration head; recompile after every board, HEAD, or capacity change.
- Evidence subset: production consumer inventory, canonical-contract/no-duplicate-store check, active-revision publication/CAS, exact launch argv per lane, empty-slice denial, restart/adoption equality, capacity shrink, two-process overlap timeline, conflict/dependency serialization, actual-diff overlap fence, merge and post-merge current-tree receipts, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/false_completion_recovery_20260808.json#false_completions/ASE3-006
- Repairs task: ASE3-006
- Canonical board task: true
- Acceptance: Configured-board production launch consumes the compiled active plan rather than merely importing a standalone module; execution_plan.py exports no duplicate InvocationBudget and opens no private SQLite claim/effect ledger; one durable canonical revision owns the full immutable plan and exact lane slices before any child starts; missing, partial, or mixed revision fields fail closed; slice-bound children use the slice as sole selection authority with legacy hash sharding disabled and execute a bounded wave before replan; a child cannot spawn until its durable claim, process birth, lease, and fence are acquired or safely adopted; at least two admitted disjoint real lane processes overlap in wall-clock time while dependency, file, resource, provider, and validation conflicts never overlap; live capacity loss narrows the set; an empty or restarted slice cannot select another lane's work; same-revision stealing has one CAS winner; actual changed paths are checked before merge and undeclared overlap fences and replans without accepting either effect; board, HEAD, or capacity drift invalidates the plan; accepted merges are serialized and every survivor is rebased and revalidated on the current head; legacy non-v3 launch behavior remains unchanged without the sealed planner profile.

## ASE3-027 Repair production canonical resolver composition and verified trust evidence

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ambient-inference-production-repair
- Depends on: ASE3-001, ASE3-018
- Goal id: ASE3-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py test/api/test_agent_supervisor_prompt_v3_resolution.py test/api/test_agent_supervisor_inference_runtime.py test/api/test_agent_supervisor_target_resolver.py test/api/test_agent_supervisor_state_resolver.py test/api/test_agent_supervisor_profile_resolver.py test/api/test_agent_supervisor_objective_resolver.py test/api/test_agent_supervisor_capability_resolver.py test/api/test_agent_supervisor_authority_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/ambient-inference-production-repair
- Parallel lane: ambient-inference-production-repair
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py
- Interfaces: FrozenInvocationContext, CanonicalResolutionCore, TransportEvidenceEnvelope, ProductionCanonicalResolverFactory, CanonicalResolutionPipeline, SupervisorResolutionService, ResolutionReceipt
- Conflict policy: Repair ASE3-018 production composition only; call the existing canonical target, state/run, profile, objective/task-source, resource/validation/topology, capability, and authority resolvers through their public verified receipts. Do not edit ASE3-019-owned local_profile/provider/router/daemon files, create authority, accept caller booleans as verification, duplicate leaf resolver policy, or add a publicly constructible verified UCAN/profile wrapper.
- Preconditions: ASE3-001 resolver surfaces and all named leaf-resolver suites import on the exact integration tree; ASE3-018 frozen-core work is present; the fenced Wave3b false-completion recovery artifact is readable. Profile and authority calls use public interfaces so ASE3-019 may rebase independently, followed by required joint current-tree revalidation.
- Effects: Make the default SupervisorResolutionService, resolve_prompt_only, orchestrate, and every launch-capable caller use one complete nine-field production pipeline for repository, state, profile, run, objective, task_source, resources, validation, and topology; treat prefilled context values as candidates that each real leaf resolver must verify, not resolved facts; verify the actual Git worktree/root and a cryptographically verified installed-profile receipt; require Python allowlist evidence, MCP server-owned alias evidence, and verifier-produced attenuated MCP++ authority evidence; remove signature-shape and caller-authenticated-boolean trust; use the bounded canonical encoder for heterogeneous mapping keys, sets, nested fields, contexts, and receipt identities; return one typed zero, multiple, stale, conflicting, or resolver-failure continuation before effects.
- Evidence subset: production call graph proving every leaf resolver is invoked, complete nine-field provenance, mutation-stable and mixed-key/set CIDs, real Git/root denial matrix, cryptographic profile verification and revoked/substituted/symlink denial, Python allowlist, MCP alias, verified UCAN attenuation and forged-wrapper denial, cross-transport core parity, ambiguity/continuation matrix, joint post-ASE3-019 current-tree validation, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/false_completion_recovery_20260808.json#false_completions/ASE3-018
- Repairs task: ASE3-018
- Canonical board task: true
- Acceptance: No production default contains required_fields=("repository",) or authorizes from arbitrary prefilled context; each launch-capable default invokes and records verified provenance for all nine real resolvers or denies before effects; non-Git or fake-.git roots, nonexistent, symlink, substituted, or shape-only profiles, profile_signed booleans, arbitrary client paths, forged or caller-constructed UCAN evidence, stale, revoked, or overbroad grants, and conflicting candidates fail closed; semantically identical heterogeneous mixed-key/set inputs produce one stable context and receipt CID without TypeError; mutation after collection cannot change identity; Python, local, MCP, and MCP++ share the same trusted core while retaining transport evidence outside it; the exact tree after ASE3-019 integration passes the declared suite.

## ASE3-022 Accept the provider-attempt daemon reload boundary

- Status: blocked
- Completion: manual
- Is schedulable: false
- Review only: true
- Priority: P0
- Track: ops
- Depends on: ASE3-006, ASE3-018, ASE3-019, ASE3-023, ASE3-027
- Outputs: data/agent_supervisor/prompt_only_self_improvement_v3/convergence/provider_attempt_daemon_reload_receipt.json
- Validation: test -f data/agent_supervisor/prompt_only_self_improvement_v3/convergence/provider_attempt_daemon_reload_receipt.json
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/provider-attempt-daemon-reload
- Parallel lane: coordinator
- Resource class: coordinator
- Predicted files: data/agent_supervisor/prompt_only_self_improvement_v3/convergence/provider_attempt_daemon_reload_receipt.json
- Conflict policy: This is an operator-owned noncanonical transition gate; it may record and authorize only the exact same-namespace fence and daemon reload after ASE3-019, and it may not implement code, dispatch a provider, reset attempts, widen policy, enable legacy refill, or satisfy any product goal.
- Preconditions: ASE3-023 and ASE3-027 have accepted current-tree repair chains and ASE3-019 attempt 2 has accepted on the exact integration tree; the declared green statuses for ASE3-006 and ASE3-018 and the failed ASE3-019 attempt-1 candidate are explicitly non-authoritative; the bootstrap master, all lane supervisors and daemons, provider runners, task worktrees, merge transactions, and scoped containers are durably identified and can be fenced without touching another run.
- Effects: While blocked, prevent ASE3-021 from becoming selectable. After all preconditions hold, fence the exact old process generation, prove zero owned descendants and effects, bind the immutable false-completion recovery artifact and the accepted implementation, merge, status, and current-tree validation chains for ASE3-019, ASE3-023, and ASE3-027, record the old and transition HEAD/tree plus implementation-daemon and configured-scheduler blob identities, then complete this gate in the same protected-board commit that adds and validates the reload receipt.
- Evidence subset: same namespace, false-completion recovery artifact SHA-256, old master and lane process-birth identities, terminal shutdown events, zero descendants/providers/containers/worktrees/merges, accepted ASE3-019/023/027 chains, old and transition HEAD/tree, implementation-daemon/configured-scheduler blob identities, joint current-tree validation, clean preflight, no legacy refill effects
- Generated by: ipfs_accelerate_py.agent_supervisor.operator-transition-gate@1
- Canonical board task: false
- Blocked reason: provider-attempt daemon reload boundary not yet accepted
- Acceptance: This task remains blocked even after its dependencies complete until an operator-owned current-tree receipt proves the exact old generation is fenced, ASE3-019 attempt 2 and ASE3-023/027 are accepted ancestors of the transition head, their declared suites pass together on that tree, and no counter restoration occurred. The old ASE3-006/018 completion projections never satisfy this gate. Completing it without that tracked fail-closed receipt is invalid. After the completion commit passes convergence validation and configured preflight, relaunch exactly one process tree in the same namespace; ASE3-021 must start from that completion commit under the reloaded daemon.

## ASE3-021 Wire durable production refill, append/adoption CAS, and completion authority

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: refill-completion-hardening
- Depends on: ASE3-004, ASE3-006, ASE3-007, ASE3-019, ASE3-022
- Goal id: ASE3-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_store.py, test/api/test_agent_supervisor_prompt_v3_refill_hardening.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_refill_hardening.py test/api/test_agent_supervisor_prompt_v3_refill.py test/api/test_agent_supervisor_backlog_refinery.py test/api/test_agent_supervisor_goal_completion.py test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/refill-completion-hardening
- Parallel lane: refill-completion-hardening
- Resource class: io-database
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_store.py, test/api/test_agent_supervisor_prompt_v3_refill_hardening.py
- Interfaces: DurableRefillState, SignedRefillPolicy, CurrentTreeResidualEvaluator, RefillAppendReceipt, RefillAdoptionReceipt, ProductionRefillRuntime
- Conflict policy: Own durable residual evaluation, refill budgets/state, canonical append/adoption CAS, and production service adapters only; do not mutate an active board outside revision control, bypass plan admission, create a second task parser, or let generated work widen authority, budgets, effects, dependencies, or acceptance gates.
- Preconditions: ASE3-004 provides admitted revisioned goal/task projections; ASE3-007 provides deterministic local evaluation primitives; backlog refinery, objective tracker, goal completion, Planner Doctor/self-improvement, Doctor findings, and canonical task-source services pass their current suites.
- Effects: Compose the actual production residual sources and completion evidence; bind evaluation to the observed current integration tree; enforce the signed caps of 3 refill epochs, 8 generated goals, 24 generated tasks, 48 open tasks, and depth 3; persist epoch, canonical seen-gap identities, cooldown, unchanged-set count, budget use, circuit state, and terminal decision across processes and restarts; append fully admitted scheduler metadata through one task-source revision CAS returning prior/new roots, revision, and generated task CIDs; on CAS loss reread and adopt an identical winner or reevaluate the new tree; wire each named trigger before retry/stall paths return and force a final root scan before completion.
- Evidence subset: production adapter manifest, signed-cap equality, current-tree join, individual trigger/no-trigger matrix, immutable append/adoption receipts, multiprocess/restart CAS, complete generated task metadata, dedup/cooldown/oscillation state, branch/stale reopening, final completion decision, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/post_wave3_residuals_20260808.json#production-durable-refill-wiring
- Acceptance: No boolean callback or caller-supplied completion booleans can authorize append or completion; two processes and crash/restart produce one revision and no duplicate gap/task; refill state and budgets never reset on restart; every generated task has lineage, dependencies, outputs, validation, predicted files, conflict/resource lane, effects, evidence, and acceptance; stale-tree or branch-only evidence appends committed convergence work; every trigger is individually wired, unchanged residuals reach the documented circuit reason and EXHAUSTED state, and a healthy evidence-complete final scan alone permits completion.

## ASE3-020 Converge transactional run truth and compose the real crash-safe prompt saga

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: transactional-runtime-hardening
- Depends on: ASE3-003, ASE3-005, ASE3-018, ASE3-019, ASE3-021
- Goal id: ASE3-G055
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_runtime_hardening.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_runtime_hardening.py test/api/test_agent_supervisor_prompt_v3_run_registry.py test/api/test_agent_supervisor_run_registry.py test/api/test_agent_supervisor_prompt_v3_runtime.py test/api/test_agent_supervisor_lifecycle_orchestrator.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/transactional-runtime-hardening
- Parallel lane: transactional-runtime-hardening
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_runtime_hardening.py
- Interfaces: AuthoritativeRunRevisionStore, ImmutableRunEpoch, EffectReservation, ProcessBirthObservation, CompleteLaunchPlanGuard, StandardSupervisorRuntimeFactory, PromptToRunSaga
- Conflict policy: Final runtime fan-in only after ASE3-018/019/021; DuckDB owns mutable run heads and effect reservations; projections, arbitrary handler mappings, PIDs, logs, and provider claims never establish success; do not weaken context, authority, fallback, refill, lifecycle, lease/fence, or validation contracts.
- Preconditions: Hardened canonical resolution, signed provider authority, durable refill, admitted materialization, lifecycle orchestration, task-source coordination, and current run-registry baselines pass on the exact integration tree.
- Effects: Make DuckDB authoritative for complete run heads, revision history, adoption/idempotency keys, effect reservations, verified process birth, event cursors, leases/fences, and persisted immutable epochs; validate stored columns against content and reconstruct exact revisions; migrate only verified JSON facts; compare every complete launch-plan field immediately before each effect; reserve/fence intent and idempotency before external effects, then after a crash observe and adopt the exact effect or complete its receipt rather than replaying blindly; compose the production handlers from ASE3-018/019/021 and ASE3-004, compile argv accepted by the real supervisor parser, and implement resolve, preview, authorize, materialize, start/adopt, observe, steer, validate, and stop against real services.
- Evidence subset: schema/migration parity, full revision history and immutable epoch, concurrent CAS/effect reservation, complete-field stale denial, real parser argv, real child birth and health observations, full operation receipts, every intent/effect/receipt crash boundary, exact restart/adoption, no-op/fabricated receipt denial, current-tree validation, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/post_wave3_residuals_20260808.json#transactional-run-truth-and-effect-recovery
- Acceptance: Concurrent revisions or effects have one winner; corrupt or column-mismatched rows and unverified migration facts deny; production construction rejects arbitrary mapping receipts and missing/fake handlers; run returns RUNNING only for a joined live process birth, lease/fence, state revision, health revision, and event cursor; all public lifecycle operations execute through the complete saga; injected crashes before and after each external effect neither duplicate nor lose the effect; an existing projection-only RUNNING row is observed/adopted or repaired, never trusted; exact retries resume the same logical run and synthetic process/health identities cannot pass.
