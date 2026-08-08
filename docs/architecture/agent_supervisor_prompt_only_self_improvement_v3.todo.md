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

- Status: todo
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

- Status: todo
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
- Depends on: ASE3-005, ASE3-006, ASE3-007
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
