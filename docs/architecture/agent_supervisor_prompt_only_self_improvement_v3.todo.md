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
Recovery (parallel): ASE3-018 ASE3-019 ASE3-027
Hermetic identity:   ASE3-030
Native dependency:   ASE3-031
DuckDB policy:       ASE3-032
Adaptive acceptance: ASE3-023
Transition gate:    ASE3-022
Contract lowering:  ASE3-029
Router policy:      ASE3-028
Planning effect:    ASE3-024
Generated runtime:  ASE3-025
Refill runtime:     ASE3-021
Transactional saga: ASE3-020
Monitor runtime:    ASE3-008
Activation gate:    ASE3-026
Python facade:      ASE3-009
Facades (parallel): ASE3-010 ASE3-011
Conformance:        ASE3-012
Canary:             ASE3-013
Closeout:           ASE3-014
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
- Depends on: ASE3-006, ASE3-020, ASE3-021
- Goal id: ASE3-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/runtime/durable_process.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_monitor.py, ipfs_accelerate_py/agent_supervisor/entrypoints/monitor_runner.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_monitor.py, test/api/test_agent_supervisor_task_attempt_limit.py, docs/guides/AGENT_SUPERVISOR_PROMPT_RUNBOOK.md
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_monitor.py test/api/test_agent_supervisor_supervisor_watchdog.py test/api/test_agent_supervisor_task_attempt_limit.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/monitoring-recovery
- Parallel lane: monitoring-recovery
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/runtime/durable_process.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_monitor.py, ipfs_accelerate_py/agent_supervisor/entrypoints/monitor_runner.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_monitor.py, test/api/test_agent_supervisor_task_attempt_limit.py, docs/guides/AGENT_SUPERVISOR_PROMPT_RUNBOOK.md
- Interfaces: RunHealthSnapshot, JoinedRunningEvidence, SemanticProgressClock, SemanticProgressCursorVector, StallClassifier, RecoveryPolicy, RecoveryReceipt, SupervisorDoctorService, DurableMonitorRunner, ReviewedHostNamespaceReconciler, MonitorAdoptionReceipt
- Conflict policy: Own monitor-specific contracts and health joins, monitor fields in the authoritative run registry, durable monitor process lifecycle, configured-scheduler semantic-progress publication, integration with the reviewed host-namespace guardian, and bounded recovery orchestration only. Reuse ASE3-020 immutable run history, cursor vectors, effect reservations, process-birth, lease/fence, and unknown-outcome adoption plus ASE3-021 refill saga cursors/deadlines; do not create another registry, process launcher, scheduler, or refill store. Detection never implies restart authority, a monitor cannot attest to its own guardian, and client sessions never own monitor liveness.
- Preconditions: Real lifecycle and parallel scheduler services expose durable events and authorized callbacks; ASE3-020 supplies monitor-ready immutable history/effect truth; ASE3-021 supplies the durable refill saga cursor and phase deadlines; process, registry, and guardian identities are durable and review-bound.
- Effects: Define the shared monitor evidence contracts; persist monitor intent before returning RUNNING; have a ReviewedHostNamespaceReconciler start or adopt one detached monitor with its own verified process birth, lease, fence, heartbeat, event cursor, and generation identity; publish configured-scheduler semantic progress as immutable phase/cursor movement; join lifecycle and monitor births, leases, fences, fresh heartbeats, and monotonic cursors with run history, task/claim/validation/merge/refill phases, tree reachability, and provider/resource state; survive client disconnect; repair stale projections; adopt/restart/rescue exact lifecycle or monitor processes within timestamp-based budgets; emit a terminal monitor shutdown receipt only after the run is terminal.
- Evidence subset: 5-second heartbeat and 30-second stale control policy, 300-second bounded semantic-progress policy, exact ReviewedHostNamespaceReconciler review and host-namespace ownership, joined lifecycle/monitor birth/lease/fence/heartbeat/cursor vector, configured-scheduler semantic-progress trace, client-disconnect survival, monitor-death adoption/restart, dead/PID-reuse/frozen/false-idle/soft-complete matrix, unknown-outcome non-replay, authorized recovery callback, retry/backoff window, circuit breaker, status repair and terminal shutdown receipts
- Acceptance: RUNNING requires one same-revision join of verified lifecycle and monitor births, leases, fences, fresh heartbeats, and monotonic event cursors; any missing, stale, cross-generation, synthetic, or self-attested component denies RUNNING. The reviewed host-namespace guardian, not a CLI, MCP, Python client, or the monitor itself, starts or adopts the durable monitor; client disconnect does not stop it and injected monitor death has exactly one restart/adoption winner. Configured-scheduler phase/cursor movement is semantic progress while log noise is not; unknown external-effect outcomes are adopted or become typed operator action and are never blindly replayed; at most three canary recoveries occur per thirty minutes; every incident recovers once or yields typed operator action, and terminal shutdown stops only the exact owned monitor generation.

## ASE3-009 Export the production Python facade and stable package API

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: python-facade
- Depends on: ASE3-005, ASE3-008, ASE3-026
- Goal id: ASE3-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/service_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, ipfs_accelerate_py/__init__.py, test/api/test_agent_supervisor_prompt_v3_python_api.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_python_api.py test/api/test_agent_supervisor_entrypoint_package.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/python-facade
- Parallel lane: python-facade
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/service_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py, ipfs_accelerate_py/__init__.py, test/api/test_agent_supervisor_prompt_v3_python_api.py
- Interfaces: Supervisor, SupervisorRun, Supervisor.open, Supervisor.run, Supervisor.preview, Supervisor.steer, Supervisor.status, Supervisor.follow, Supervisor.explain, Supervisor.doctor, Supervisor.init_local, ProductionServiceCompositionManifest
- Conflict policy: Sole owner of Python facade and package export fan-in; preserve lazy/cold imports and existing exports; no in-memory successful fallback, hidden global mutable backend, or transport-specific policy.
- Preconditions: Real intent, monitor/Doctor, and protected runtime activation services are complete; ASE3-026 binds the active generation and ASE3-000 protects concurrent user edits to package initializers.
- Effects: Add injectable and configured `Supervisor.open`; resolve the installed production registry and emit a body-free content-addressed ProductionServiceCompositionManifest binding the resolver, broker, planning, materialization, scheduler, refill, monitor, and run backends; expose typed run handles and lifecycle methods; infer a sole compatible run when safe; return one typed continuation on ambiguity; lazily export the same API from entrypoints, agent_supervisor, and top-level package.
- Evidence subset: export manifests, body-free production composition manifest CID, active-generation binding, import timing/side-effect trace, canonical method request/result receipts, unavailable-backend failure, run inference ambiguity, prompt non-leak
- Acceptance: The documented `from ipfs_accelerate_py import Supervisor` path works from an installed wheel and source tree; after one authorized local initialization, `Supervisor.open()` needs no expert constructor arguments and resolves the production service registry activated by ASE3-026; cold import starts no integration/process; `run(prompt)` reaches that real service; absent configuration fails typed; no simulated completion path exists.

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
- Interfaces: PromptOnlyConformanceMatrix, SecurityAdversaryMatrix, ExpertCompatibilityReport, InstalledDistributionSmoke, ProductionServiceCompositionManifest, PromptProductDuckDBConnectionAudit
- Conflict policy: Test and documentation fan-in only; do not modify implementation, expected capability baselines to conceal drift, safety thresholds, provider classifications, or authority fixtures to make gates green.
- Preconditions: Python, CLI, MCP, and MCP++ facades are integrated on one exact tree; their manifests and generated schemas are available.
- Effects: Build and install a wheel, then exercise Python, the console CLI, and MCP/MCP++ through black-box subprocess and JSON-RPC boundaries without injecting fake services; prove preview and run resolve the same body-free production composition manifest CID and backend identities; execute canonical request/result parity, cold import/help, server alias, UCAN, prompt/secret leak, path injection, stale context, provider, crash-boundary, duplicate launch/effect, and expert compatibility matrices. Build an exact import/call-graph plus AST inventory from every public prompt-product launch root through planning, generated-board, task-source, merge/lease, run-registry, and runtime-history consumers; reject every launch-reachable raw `duckdb.connect` call and prove each reachable connection delegates to `connect_duckdb_with_policy`. Explicitly classify each remaining raw call as non-reachable legacy or proof-only code with a current-tree reachability proof rather than claiming global source elimination; document the supported journey and safe bootstrap.
- Evidence subset: exact tree/environment and installed-wheel identity, black-box Python/CLI/MCP/MCP++ operation matrix, shared production composition manifest CID and preview/run backend identities, error/exit parity, security denials, prompt/secret leak scan, exact prompt-product import/call-graph and AST connection-birth inventory, reachable-helper delegation proof, non-reachable legacy/proof-only classification, compatibility delta, cold-start timings, docs command smoke
- Acceptance: Every public operation has equivalent semantics through black-box installed-distribution boundaries; Python, CLI, MCP, and MCP++ report one production composition CID and preview/run reach the same real common service; fake service injection, schema-only registration, and in-process adapter parity cannot satisfy the gate; no production fallback simulates success, no unauthorized input reaches an effect, and the installed-tree import/call-graph plus AST gate proves that no prompt-product launch-reachable path can call raw `duckdb.connect`. Every reachable connection delegates to `connect_duckdb_with_policy`; any surviving raw site is explicitly proven non-reachable and classified as legacy or proof-only, never silently treated as globally covered. Docs commands execute, cold paths have no integration startup, and capability changes are intentionally classified instead of hidden behind stale snapshot assertions.

## ASE3-013 Run and monitor a bounded self-improvement canary

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: self-host-canary
- Depends on: ASE3-008, ASE3-012, ASE3-026
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
- Preconditions: Cross-transport gates pass on the exact candidate tree; a fresh empty state namespace contains no objectives, taskboard, run, or task-source seed; tempting stale ASE3 state remains present only outside that namespace and must be ignored; local signed profile and bounded budgets are installed; monitor has a live clock and authorized lifecycle callbacks; rollback is rehearsed.
- Effects: Assert the generated task source is absent, then invoke the new public facade with one bounded supervisor-improvement prompt and no seed-board or objective argv; prove the prompt creates the root program, goal/subgoal/task populations, real children, and task source; trace every descendant CID from the prompt digest and program root; observe at least two generated disjoint real effects overlap and conflicts serialize; inject stale PID, frozen worker, false idle/open goal, branch-only completion, crash boundary, lease loss, client disconnect, monitor death, provider saturation, monotonic-clock rollback, merge/refill stalls, and repeated equivalent recovery/refill oscillation; require the forced refill descendant to be dispatched or durably adopted; accept a bounded non-sentinel repository change; after the last injected recovery completes, sustain continuously healthy operation for at least the signed config `monitor_policy.canary_observation_seconds: 900` window using monotonic elapsed time; stop truthfully.
- Evidence subset: fresh-state and stale-state-isolation receipts, pre-run task-source absence, prompt digest and program/goal/task/plan roots, descendant CID lineage, no-seed argv, real lifecycle and monitor births/effects, event/revision/cursor timeline, parallel overlap, conflict serialization, forced residual append and dispatch/adoption, client-disconnect/monitor-death/provider-saturation/clock-rollback/merge-stall/refill-stall/oscillation decisions, non-sentinel accepted diff reachability, every recovery decision, budget usage, signed `monitor_policy.canary_observation_seconds: 900`, monotonic post-recovery observation start/end and at-least-900-second uninterrupted healthy duration, terminal shutdown
- Acceptance: ASE3-026 activation authorization and the separate post-activation observation are accepted before this task starts. The canary begins with no preseeded objective or taskboard and stale ASE3 state cannot satisfy a dependency; every goal, task, slice, run, effect, and cursor identity descends from the one prompt/program root; no seed-board argument is present; at least two prompt-generated disjoint real effects overlap while conflicts serialize; a forced residual is appended and dispatched or durably adopted; the accepted result contains a reviewed non-sentinel repository change, not a fixture, mock, or no-op. Client disconnect, monitor death, provider saturation, clock rollback, merge/refill stalls, and oscillation each recover once or fail typed within budget without duplicate/unauthorized effects. The 900-second clock begins only after the final recovery is observed healthy; any unhealthy sample resets it. Signed policy plus monotonic start/end and continuous samples must prove at least 900 uninterrupted healthy seconds before promotion; shorter, pre-recovery, wall-clock-only, prompt-selected, or fabricated observations deny.

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

- Status: completed
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

## ASE3-030 Seal hermetic control-plane identity dependency closure

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: hermetic-control-plane-identity
- Depends on: ASE3-019
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py, ipfs_accelerate_py/utils/cid_utils.py, test/api/test_agent_supervisor_control_plane_capsule_identity.py, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_multiformats_identity.py, test/api/test_llm_router_agent_implementation_route.py, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/hermetic_control_plane_identity_acceptance_receipt.json
- Validation: python -m pytest test/api/test_agent_supervisor_control_plane_capsule_identity.py test/api/test_agent_supervisor_control_plane.py test/api/test_agent_supervisor_multiformats_identity.py test/api/test_llm_router_agent_implementation_route.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/hermetic-control-plane-identity
- Parallel lane: hermetic-control-plane-identity
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py, ipfs_accelerate_py/utils/cid_utils.py, test/api/test_agent_supervisor_control_plane_capsule_identity.py, test/api/test_agent_supervisor_control_plane.py, test/api/test_agent_supervisor_multiformats_identity.py, test/api/test_llm_router_agent_implementation_route.py, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/hermetic_control_plane_identity_acceptance_receipt.json
- Interfaces: AgentImplementationSealedControlPlane, HermeticCIDProfile, CanonicalCIDCodec, ControlPlaneDependencyClosureReceipt
- Conflict policy: Own only the Git-bound sealed control-plane dependency closure and dependency-free canonical CID implementation required inside that capsule. Preserve the ASE3-019 route/authority/fallback decisions byte-for-byte and do not widen the capsule allowlist, import mutable candidate/user-site code, use `PYTHONPATH`, weaken source/blob verification, introduce another CID profile, or grant provider/scheduler/merge authority. ASE3-023 may continue implementation, but its acceptance and ASE3-022 transition remain fail-closed until this task is accepted through the protected acceptance-prerequisite join and reserved receipt.
- Preconditions: ASE3-019's accepted sealed control-plane capsule materializer and source-generation verifier are readable on the exact accepted tree; known CIDv1/base32/raw and dag-json/sha2-256 vectors pass in the ordinary environment; a subprocess environment with `python -I`, user site disabled, and no installed `multiformats` import is available.
- Effects: Enumerate the complete transitive in-tree import closure of every allowed control-plane module; include and Git-bind the exact `agent_supervisor/core/multiformats_identity.py` and `utils/cid_utils.py` blobs plus every required package initializer in the sealed capsule manifest/archive; implement strict canonical CIDv1 varint, lowercase base32 multibase, raw/dag-json codec, and sha2-256 multihash mint/parse/validation internally so the control plane never requires user-site `multiformats`; preserve exact existing known vectors and, when the optional library is installed outside isolation, prove byte-for-byte parity; execute only from the sealed fd under `python -I`, verify every imported module origin and manifest member, import every allowed control module, and mint/validate both raw and DAG-JSON CIDs before any scheduler/provider effect.
- Evidence subset: recursive import-closure manifest, exact Git blob/archive/root binding, core identity and cid_utils inclusion, no-user-site/no-PYTHONPATH `python -I` bootstrap receipt, known raw and dag-json CID vectors, optional-library parity, every-allowed-module import/origin inventory, raw/dag-json mint/validate receipt, missing/member-substitution/extra-member/zip-shadow denial matrix, exact current-tree router/identity regression receipts, reserved `ipfs_accelerate_py.agent_supervisor.ase3-030-hermetic-identity-acceptance@1` receipt with source HEAD/tree/blob/archive/root and suite digests, strict convergence-manifest binding
- Canonical board task: true
- Acceptance: A fresh `python -I` subprocess with user site and `PYTHONPATH` unavailable loads the Git-bound capsule solely from its sealed descriptor, imports every allowed control-plane module from the verified fd, and mints and validates canonical CIDv1 lowercase-base32 raw+dag-json sha2-256 identities without importing `multiformats` or mutable repository/candidate code; exact historical known vectors and optional installed-library parity remain unchanged; the dependency-closure receipt binds every transitive in-tree module and package initializer to source HEAD/tree/blob/archive/root; omitting `utils/cid_utils.py` or any dependency, substituting a member, adding an unmanifested shadow, resolving a module outside the capsule, using a user-site/PYTHONPATH dependency, or changing a CID profile fails before scheduler/provider effects. ASE3-030 remains `todo` and its reserved receipt path remains absent until one protected acceptance commit adds a strict validator for `ipfs_accelerate_py.agent_supervisor.ase3-030-hermetic-identity-acceptance@1`, binds that receipt in the convergence manifest, and atomically records the accepted status; ASE3-023 cannot be accepted and ASE3-022 cannot transition before that Q→R→P→A fan-in.

## ASE3-031 Seal the reviewed DuckDB native extension for isolated supervisor launch

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: hermetic-native-dependency
- Depends on: ASE3-030
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/llm_router.py, test/api/test_agent_supervisor_native_dependency_pin.py
- Validation: python -m pytest test/api/test_agent_supervisor_native_dependency_pin.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/hermetic-native-dependency
- Parallel lane: hermetic-native-dependency
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/llm_router.py, test/api/test_agent_supervisor_native_dependency_pin.py
- Interfaces: AgentSupervisorNativeDependencyPin, AgentSupervisorNativeDependencyDescriptor, AgentSupervisorNativeDependencyLaunch, inspect_agent_supervisor_native_dependency_source, parse_agent_supervisor_native_dependency_pin, parse_agent_supervisor_native_dependency_launch, seal_agent_supervisor_native_dependency, verify_agent_supervisor_native_dependency_sealed_fd, preload_agent_supervisor_native_dependency, preload_agent_supervisor_native_dependency_from_bootstrap
- Reviewed implementation commit: 25fedf091dad928dad1f83c9f81a54c2d401eabe
- Conflict policy: Own only path-free inspection evidence, exact reviewed native-dependency pin validation, sealed-fd materialization, and one-process preload in `ipfs_accelerate_py.llm_router`. Inspection is evidence only and must never mint launch authority. Do not wire scheduler or daemon launch, trust an ambient `_duckdb` import, load from a repository/user-site path, relax `python -I`, grant provider/merge/refill authority, alter the ASE3-019/022/023/027 protected task blocks, or edit outside the two exact product/test outputs. ASE3-023 may continue implementation, but its acceptance and ASE3-022 transition remain fail-closed through ASE3-032.
- Preconditions: ASE3-030 is accepted at A030 on the exact committed tree; its sealed control-plane capsule and dependency-free CID implementation pass under `python -I`; the reviewed ASE3-031 implementation commit `25fedf091dad928dad1f83c9f81a54c2d401eabe` descends from and preserves the exact reviewed ASE3-030 product delta before its own two-path delta; the operator has independently reviewed an exact `_duckdb` payload, Python executable, CPython ABI, platform, ELF identity, and ordered `DT_NEEDED` closure. No accepted native launch authorization is inferred from those observations.
- Effects: Inspect a stable no-follow regular source only to produce a path-free content-addressed `AgentSupervisorNativeDependencyPin`; require that exact expected pin and a nonempty externally accepted authorization ID before copying bytes into a private sealed executable memfd; bind descriptor, device, inode, owner, mode, link count, size, payload digest, required seals, bootstrap JSON, and `pass_fds`; structurally reject `PT_INTERP`, executable GNU stack, ambient loader tags, malformed ELF ranges/tables, wrong ABI/platform/Python/payload/dependency identities, source substitution, hardlinks, FIFOs, symlinks, and unsealed or rebound descriptors; under an empty ambient loader environment, exact-match the P-phase signed authorization ID, preload `_duckdb` solely from `/proc/self/fd/<fd>` before control-plane imports, verify module origin and a real query, and make the process terminal after preload begins or fails so native unload uncertainty cannot authorize an in-process retry.
- Evidence subset: exact reviewed commit/tree and two-path diff, path-free pin and dependency CID, distribution/engine version, extension filename, CPython cache tag and SOABI, platform/machine, Python executable digest, payload size/digest, ELF class/endianness/OSABI/machine and ordered `DT_NEEDED`, stable-source denial matrix, memfd descriptor/stat/seal binding, strict launch JSON, fixed isolated bootstrap argv, empty loader environment, synthetic loader trace, real `python -I` query, failure-terminal process trace, accepted authorization ID equality, reserved signed `ipfs_accelerate_py.agent_supervisor.ase3-031-native-dependency-launch-authorization@1` P artifact, reserved `ipfs_accelerate_py.agent_supervisor.ase3-031-native-dependency-acceptance@1` A receipt, convergence-manifest bindings
- Canonical board task: true
- Acceptance: Inspection, a known payload digest, or successful local loading never creates authority. The P-phase `native_dependency_launch_authorization.json` must be signed by an accepted local profile, bind this namespace plus the exact source HEAD/tree, ASE3-030 acceptance receipt, reviewed native pin/dependency ID, Python executable/ABI/platform, allowed bootstrap purpose, expiry, nonce, and zero prior launch effects, and expose the exact accepted authorization ID consumed by sealing; it remains absent until its strict validator and P-manifest binding land. That P artifact is required before any ASE3-023 capsule/native subprocess end-to-end runtime effect used as acceptance evidence, while `authorization_may_claim_launch_effect: false` remains exact. A fresh isolated subprocess then inherits only the sealed fd and canonical launch JSON, exact-matches that accepted authorization ID before preload, imports `_duckdb` only from the descriptor, executes a real query, and denies every substitution, ambient loader surface, second preload, or retry after failure before scheduler/provider effects. ASE3-031 remains `todo` and `sealed_native_dependency_acceptance_receipt.json` remains absent until the protected A commit adds a strict validator for `ipfs_accelerate_py.agent_supervisor.ase3-031-native-dependency-acceptance@1`, binds the unchanged P authorization and all current-tree evidence in the convergence manifest, and atomically accepts this task; authorization at P cannot claim launch or acceptance, and ASE3-023/022 remain unaccepted.

## ASE3-032 Enforce one configuration-locked DuckDB connection policy across supervisor state

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: secure-duckdb-connection-policy
- Depends on: ASE3-031
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py, test/api/test_agent_supervisor_duckdb_connection_policy.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_duckdb_connection_policy.py test/api/test_agent_supervisor_duckdb_state.py test/api/test_agent_supervisor_duckdb_task_source.py test/api/test_agent_supervisor_lease_coordination.py
- Static validation commands JSON: ["python -m ruff check --select E9,F63,F7,F82,I ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py test/api/test_agent_supervisor_duckdb_connection_policy.py", "python -m py_compile ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py test/api/test_agent_supervisor_duckdb_connection_policy.py", "git diff --check"]
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/secure-duckdb-connection-policy
- Parallel lane: secure-duckdb-connection-policy
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py, test/api/test_agent_supervisor_duckdb_connection_policy.py
- Interfaces: DUCKDB_CONNECTION_POLICY_SETTINGS, DuckDBConnectionPolicyError, connect_duckdb_with_policy, DuckDBConnection
- Conflict policy: Own the single connection-birth policy and route only the named supervisor task-source and lease-coordination consumers through it. Do not create a second policy helper, accept policy overrides, install or fetch extensions, load dynamic/external extension bytes, retain an ambient direct `duckdb.connect`, use any `ATTACH` in production compaction, widen database paths or external access, grant scheduler/provider/merge authority, wire a runtime launch, alter the ASE3-019/022/023/027 protected task blocks, or edit outside the four exact product/test outputs. A statically linked reviewed module may answer `LOAD`, and DuckDB may allow `ATTACH ':memory:'`; neither observation authorizes external extension bytes or an `ATTACH`-based compactor. ASE3-020 owns run-registry/runtime-history delegation, ASE3-025 owns generated-board/planning delegation, and ASE3-012 owns the launch-reachability audit and non-reachable legacy/proof-only classification. ASE3-023 may continue implementation, but its acceptance and ASE3-022 transition remain fail-closed until this policy is accepted.
- Preconditions: ASE3-031 is accepted at A031 on the exact committed tree with the unchanged P031 native authorization and A030 receipt; the reviewed DuckDB version exposes all five required configuration settings; current DuckDBConnection, generated DuckDB task-source, MergeQueue/resolver, LeaseCoordinator, and coordination compaction suites pass before the policy join; every direct connection birth in the four-file scope is inventoried.
- Effects: Make `connect_duckdb_with_policy` the only connection-birth helper in scope; pass `autoinstall_known_extensions=false`, `autoload_known_extensions=false`, `enable_external_access=false`, and `allow_unsigned_extensions=false` atomically at connect, apply only bounded `threads` and `memory_limit` tuning, insert `lock_configuration=true` last, and verify the exact five typed boolean settings immediately after birth, closing and failing typed on any mismatch. Route DuckDBConnection and therefore MergeQueue/resolver consumers, the generated DuckDB task source, every LeaseCoordinator connection, and compaction through that helper; reject booleans or invalid tuning, caller overrides of protected settings, unsupported settings, and post-lock changes. Before compaction, seal the complete persistent catalog inventory of schemas, tables, views, sequences, macros, custom types, and indexes; rewrite production compaction with zero `ATTACH` statements while preserving the sealed catalog plus all live data/metadata atomically under the existing file lock and replacement protocol.
- Evidence subset: exact four-path diff and connection-birth inventory, one helper/import graph, fake-module exact config/order/read-only trace, real `current_setting` typed tuple, override/unknown/bool/empty tuning denial matrix, post-lock mutation denial, failed-verification close trace, extension INSTALL/fetch and dynamic/external-load denial, external-path ATTACH/access denial, statically linked LOAD and `ATTACH ':memory:'` non-authority checks, DuckDBConnection plus MergeQueue/resolver integration, generated task-source materialization/read integration, LeaseCoordinator lifecycle and restart integration, persistent-catalog seal covering schemas/tables/views/sequences/macros/custom types/indexes, zero-`ATTACH` production-compaction parity and atomic replacement, no direct ambient `duckdb.connect` AST inventory inside the exact four-file scope, explicit downstream ownership/classification for out-of-scope sites, reserved `ipfs_accelerate_py.agent_supervisor.ase3-032-duckdb-connection-policy-acceptance@1` A receipt and convergence-manifest binding
- Canonical board task: true
- Acceptance: Every scoped supervisor DuckDB connection is born with the exact four denials and `lock_configuration=true` in the atomic connect configuration, with bounded threads/memory set before the lock, and is rejected unless immediate typed verification proves all five values. Callers cannot override or unlock policy; INSTALL/fetch, dynamic or external extension bytes, external-path ATTACH/access, and any `ATTACH` in production compaction remain denied, while a successful statically linked LOAD or `ATTACH ':memory:'` never overclaims those guarantees. A verification failure closes the connection; MergeQueue/resolver access through DuckDBConnection, generated task sources, LeaseCoordinator recovery, complete persistent catalog contents, and zero-`ATTACH` compaction retain their existing semantics, and an exact source inventory finds no bypass connection in the four-file scope. This is bounded coverage, not a global claim: ASE3-020/025 must route their prompt-product-reachable consumers through the helper, and ASE3-012 must prove the final reachability/classification gate. ASE3-032 remains `todo` and `duckdb_connection_policy_acceptance_receipt.json` remains absent until the protected A commit adds a strict validator for `ipfs_accelerate_py.agent_supervisor.ase3-032-duckdb-connection-policy-acceptance@1`, binds the accepted ASE3-030/031 receipts, unchanged P native authorization, exact current-tree four-path implementation and suite evidence in the convergence manifest, and atomically accepts this task. Only after all three A receipts validate may ASE3-023 be accepted or ASE3-022 transition.

## ASE3-033 Productionize protected transition construction, replay provenance, and phase-local authority

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: protected-transition-construction
- Depends on: ASE3-000
- Goal id: ASE3-G055
- Outputs: ipfs_accelerate_py/agent_supervisor/core/protected_acceptance_contracts.py, ipfs_accelerate_py/agent_supervisor/merge/protected_acceptance_transition.py, ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition.py, ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition_cli.py, test/api/test_agent_supervisor_prompt_v3_transition.py, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/protected_acceptance_q_inventory.json, .gitignore, config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json, docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md, docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.objectives.md, docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md, ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py, test/api/test_agent_supervisor_prompt_v3_convergence.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_transition.py test/api/test_agent_supervisor_prompt_v3_convergence.py -q
- Required transition tests: test_q_inventory_rejects_every_future_pin_sentinel, test_ase3_031_and_032_require_task_correct_replay_provenance, test_every_pre_q_generation_reconstructs_exact_source_replay_integrated_commit_streams, test_multi_commit_generation_chains_parents_and_final_commit_map_are_exact, test_product_provenance_preserves_reviewed_modes_and_builder_artifacts_are_100644, test_a019_binds_actual_provider_authorization_v2_digest, test_birth_resolves_configured_target_ref_once_to_one_head, test_every_phase_requires_fresh_authority_and_witness, test_activation_is_impossible_before_valid_post_l_birth, test_p031_is_a031_only_and_failure_reauthorization_is_bounded_append_only, test_l_runtime_authorization_is_distinct_and_consumed_before_spawn, test_runtime_authority_loader_returns_strict_dto_after_full_chain_only, test_signing_adapter_uses_ascii_canonical_bytes_and_verified_ed25519_transcode, test_signing_adapter_denies_repository_authority_rotation_revocation_and_ambient_key_races, test_root_pin_builder_emits_only_exact_verified_public_artifact, test_unheld_target_publication_uses_update_ref_cas_under_canonical_lease, test_checked_out_target_publication_uses_hook_free_sign_free_ff_only_merge, test_publication_failure_recovers_exact_pre_state_and_never_rejects_real_target_as_detached_only, test_nested_agent_supervisor_core_is_not_gitignored, test_public_artifact_key_and_directory_modes_fail_closed
- Static validation commands JSON: ["python -m ruff check --select E9,F63,F7,F82,I ipfs_accelerate_py/agent_supervisor/core/protected_acceptance_contracts.py ipfs_accelerate_py/agent_supervisor/merge/protected_acceptance_transition.py ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition.py ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition_cli.py ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py test/api/test_agent_supervisor_prompt_v3_transition.py test/api/test_agent_supervisor_prompt_v3_convergence.py", "python -m py_compile ipfs_accelerate_py/agent_supervisor/core/protected_acceptance_contracts.py ipfs_accelerate_py/agent_supervisor/merge/protected_acceptance_transition.py ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition.py ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition_cli.py ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py test/api/test_agent_supervisor_prompt_v3_transition.py test/api/test_agent_supervisor_prompt_v3_convergence.py", "jq empty config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json", "git diff --check"]
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/protected-transition-construction
- Parallel lane: protected-transition-construction
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/core/protected_acceptance_contracts.py, ipfs_accelerate_py/agent_supervisor/merge/protected_acceptance_transition.py, ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition.py, ipfs_accelerate_py/agent_supervisor/entrypoints/protected_acceptance_transition_cli.py, test/api/test_agent_supervisor_prompt_v3_transition.py, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/protected_acceptance_q_inventory.json, .gitignore, config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json, docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md, docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.objectives.md, docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md, ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py, test/api/test_agent_supervisor_prompt_v3_convergence.py
- Interfaces: freeze_prompt_v3_product_provenance, build_prompt_v3_root_pin, initialize_prompt_v3_reviewer_after_root_pin, build_prompt_v3_provider_authorization, canonical_prompt_v3_review_bytes, sign_prompt_v3_operator_artifact, build_prompt_v3_phase_candidate, run_prompt_v3_phase_evidence, validate_prompt_v3_phase_candidate, publish_prompt_v3_phase_candidate, reject_prompt_v3_phase_candidate, observe_prompt_v3_quiescence, build_prompt_v3_reload_authorization, consume_prompt_v3_a031_authorization, append_prompt_v3_a031_failure_attempt, reauthorize_prompt_v3_p031_attempt, load_verified_prompt_v3_runtime_launch_authority
- Conflict policy: Own protected transition contracts in `core`, candidate-tree construction and publication in `merge`, and operator composition/CLI in `entrypoints`. `merge` receives signer, evidence, and validator protocols and never imports `entrypoints`; the entrypoint facade may compose `merge` with `entrypoints.local_profile` and re-export every sealed public API. Do not implement provider, native, DuckDB, scheduler, resolver, or reload product changes belonging to ASE3-019/030/031/032/023/027/022; do not create a protected receipt, change any existing task status or runtime flag, execute a protected transition, or alter the protected ASE3-019/022/023/027 blocks.
- Preconditions: The ordinary dependency is exactly ASE3-000. Before the Q integration freeze, the ASE3-019, ASE3-030, ASE3-031, ASE3-032, ASE3-023, and ASE3-027 product generations are each independently reviewed and integrated with valid `ipfs_accelerate_py.agent_supervisor.prompt-v3-product-generation@1` provenance. This is a review/readiness join, not an ordinary dependency or acceptance-status join: all six tasks may remain `todo`, and ASE3-033 must not depend on any of their A-phase receipts. ASE3-033 tooling is itself independently audited and replayed in Q's parent while its task remains todo there.
- Effects: Implement an explicit builder from an isolated clean checkout of one exact ref with a sanitized Git environment that rejects replacement refs, grafts, repository/config redirection, hooks, filters, ambient signing, and `AGENT_SUPERVISOR_LOCAL_PROFILE_KEY`; use an alternate index, preserve exact reviewed modes, write only builder-created Q/protected artifacts as 100644, and build exact one-parent candidates without reading or changing the working index and without `auto_commit_paths`. Run child validators and exact phase suites against the candidate tree under one canonical lease. Resolve the target once: publish an unheld ref by `git update-ref` CAS, but publish a checked-out exact target in its clean holding worktree by sanitized hook-free/sign-free `git merge --ff-only` under that same lease; never reject a real checked-out target merely as detached-only. Create a rescue ref first, bind exact pre/post ref/HEAD/tree/index/worktree state, and on failure restore by checked ref CAS plus verified checked-out-worktree recovery, retaining rescue state if exact recovery fails. Every filesystem read/write is dirfd-relative with `O_NOFOLLOW`, stable owned regular single-link checks, and group/other-write denial; public artifacts are atomically renamed at 0400 after data and directory fsync, private keys remain 0600, directories and runtime projections have group/other write removed. Correct `.gitignore` from mode 100755 to 100644 and replace its exact unanchored `core` line with `/core`, preserving the root core-dump ignore while proving the nested `agent_supervisor/core/protected_acceptance_contracts.py` output is not ignored. Q atomically adds only the Q inventory and changes only ASE3-033 from todo to completed; all other task statuses remain byte-identical. Every R-or-later candidate requires ASE3-033 completed on its parent.
- Product generation contract: Freeze source, replay, and integrated records under `ipfs_accelerate_py.agent_supervisor.prompt-v3-product-generation@1`. Source and replay commits are independently reviewed nonancestors of Q; integrated commits are ancestors of Q. Reconstruct every ordered commit for each role with the exact canonical `git diff --no-ext-diff --no-textconv --no-renames --binary --full-index` stream, retain standard-Base64 raw patch bytes plus SHA-256, and require source=replay=integrated bytes commit by commit for ASE3-019×2, ASE3-030×2, ASE3-031×1, ASE3-032×1, ASE3-023×3, and ASE3-027×2. Bind every exact ordinal/commit/parent/tree/changed-path order/per-path mode/raw/blob, strict first-base and subsequent-parent chain, and exact task→role last-commit final map. Preserve reviewed per-file modes, including `ipfs_accelerate_py/llm_router.py` 100755; cross-task/commit substitution and blanket 100644 normalization are forbidden. Portable verification survives fresh clone plus pruning without ambient local refs, worktree-only objects, keep-alive refs, or bundle artifacts: the durable carrier is the sealed `prompt-v3-product-generation@1` record; source/replay verify from sealed record fields only without requiring Git object reachability; integrated commits remain reachable Q ancestors; ASE3-033 tooling is present in Q's parent tree; git-diff reconstruction is required when objects are present and otherwise falls back to sealed canonical patch Base64 plus file raw/blob/mode comparison; ambient ref carriers are forbidden.
- Signing adapter contract: `build_prompt_v3_root_pin` safely emits a 0400 canonical-ASCII public artifact with exactly schema, root DID, stable-policy digest, verified repository identity/head/tree, and canonical digest; private key/profile/future-generation material is forbidden. `sign_prompt_v3_operator_artifact` accepts strict ASCII receipt mappings that round-trip to bytes identical to `entrypoints.local_profile.sign_profile_binding` canonical bytes, binds the expected repository and profile-generation/authority, strictly decodes the adapter's standard-Base64 signature, requires 64-byte Ed25519, transcodes to unpadded base64url, and verifies the public-key signature. Reload the active profile/generation after signing, require equality with the verified pre-sign generation, and deny rotation/revocation races and ambient signing-key authority.
- Authority contract: Q inventory contains only the pre-existing `lifecycle_root_identity_did` and stable policy; future-pin sentinels, reviewer identity, profile/profile content, lifecycle anchor, signature, time/test observations, capsule output, and generation are forbidden. R pins that root. P019 creates the reviewer profile/witness only after R and its provider authorization. Every subsequent phase and birth consumes a fresh non-reused `ipfs_accelerate_py.agent_supervisor.prompt-v3-phase-authority@1`; every `ipfs_accelerate_py.agent_supervisor.prompt-v3-phase-receipt@2` binds its own phase-local authority. A019 binds the digest of the actual canonical provider authorization@2 bytes. P031 uses `ipfs_accelerate_py.agent_supervisor.ase3-031-native-dependency-launch-authorization@2` with exact purpose, nonce, not-before, expiry, and a durable one-consumer CAS consumed terminally only by the A031 acceptance preload; ASE3-023 and post-L runtime cannot reuse it. Tooling exposes `consume_prompt_v3_a031_authorization`, `append_prompt_v3_a031_failure_attempt`, and `reauthorize_prompt_v3_p031_attempt` only for that A031 acceptance path. Once its native effect starts that authorization is terminal. A failed A031 probe appends signed failure evidence to an immutable `ipfs_accelerate_py.agent_supervisor.prompt-v3-p031-attempt-ledger@1`, keeps ASE3-031 todo, and may construct a fresh P031 nonce/expiry through a bounded re-authorization transition, at most three attempts; failed attempts remain in Git and cannot be retried, overwritten, or erased. Reload uses `ipfs_accelerate_py.agent_supervisor.provider-attempt-daemon-reload@2`, with a nullable prior authorization represented canonically rather than by a sentinel, and carries or derives a distinct fresh per-target-generation native launch authorization bound to the accepted A031 pin plus accepted A032/A023 gates; its durable one-consumer CAS is consumed by the ASE3-023 scheduler/PlanRevisionStore before spawn. `load_verified_prompt_v3_runtime_launch_authority` returns a narrow strict DTO only after verifying the full committed Q→L chain and A031/A032/A023 joins; the sealed child binds its digest and IDs without importing the convergence validator. L recovery may only boundedly adopt a dead, effectless same-generation launch under that already-consumed generation authorization and never replay a provider effect.
- Evidence subset: exact pre-Q source/replay/integrated generation graphs, all eleven ordered per-role reconstructed canonical full-index patch streams, exact parent chains/final map, and preserved per-file modes; no-future-pin Q inventory; safe exact public root-pin bytes; strict-ASCII signing-adapter compatibility, standard-Base64→verified-Ed25519→base64url trace, repository/authority binding, post-sign profile-generation reload, and rotation/revocation/ambient-key denials; fresh R/P019/A019/A030/P031/A031/A032/A023/027/L/birth authorities and witnesses; A019 actual authorization@2 digest; P031 purpose/nonce/not-before/expiry and durable one-consumer CAS; bounded append-only signed P031 failure/re-authorization ledger with fresh authority and no overwrite/erasure; phase-local receipt@2 signatures; nullable-prior reload@2 bytes and dead/effectless same-generation recovery adoption without provider replay; sanitized-Git/alternate-index/one-parent/child-suite traces; `.gitignore` exact `core`→`/core` correction, nested package non-ignore, builder-created 100644 inventory, and preserved reviewed `llm_router.py` 100755; dirfd/no-follow/link/owner/mode/fsync/rename denial matrix; unheld-ref update-ref CAS, checked-out-target hook/sign-free ff-only merge, exact pre/post ref/HEAD/tree/index/worktree state and rescue recovery; configured target ref resolved once to one post-L birth HEAD; pre-birth activation denial.
- Canonical board task: true
- Acceptance: The exact Q tree descends from each integrated generation while every source/replay remains a nonancestor; all source=replay=integrated canonical patch bytes, exact ordered parents/final map, and reviewed per-file modes match for ASE3-019/030/031/032/023/027. Q contains only its root-DID/stable-policy inventory and no future authority or observation sentinel. The public root-pin builder and signing adapter satisfy their exact repository/authority, canonical-byte, Ed25519-transcode/verification, active-generation reload, race-denial, and ambient-key-denial contracts. The builder constructs Q→R→P019→A019→A030→P031→A031→A032→A023/027→L→birth with fresh phase-local authority/witnesses, actual authorization@2 digest at A019, P031@2 consumed only by A031 preload, maximum-three append-only signed attempts, receipt@2 authority bindings, and distinct one-shot target-generation L authority. Every candidate is isolated, clean, one-parent, exact-tree/path/mode/raw/blob, validator/suite passing, and published under one lease: unheld target by update-ref CAS or checked-out exact target by sanitized hook/sign-free ff-only merge with exact pre/post ref/HEAD/tree/index/worktree proof and rescue recovery. Post-L resolves one configured target ref to one HEAD; L recovery only adopts bounded dead/effectless same-generation state without provider replay; no protected runtime activates before matching birth. Fresh clone plus pruning still verifies every nonancestor source/replay stream from the sealed product-generation record without ambient refs, while integrated commits remain reachable Q ancestors and ASE3-033 tooling remains present in Q's parent tree.

## ASE3-023 Repair production plan-bound adaptive parallel dispatch

- Status: completed
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

- Status: completed
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

- Status: completed
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
- Preconditions: ASE3-023 and ASE3-027 have accepted current-tree repair chains and the unchanged-CID ASE3-019 has been accepted through the manifest-bound operator-owned no-provider salvage after its exhausted attempt-2 self-host control-plane failure; the declared green statuses for ASE3-006 and ASE3-018, the failed ASE3-019 attempt-1 candidate, and the failed attempt-2 provider-runner invocation are explicitly non-authoritative; the bootstrap master, all lane supervisors and daemons, provider runners, task worktrees, merge transactions, and scoped containers are durably identified and fenced without touching another run.
- Effects: While blocked, prevent ASE3-021 from becoming selectable. After all preconditions hold, prove zero owned descendants and effects; bind the immutable false-completion recovery artifact, exact ASE3-019 attempt-2 event/log/incident packet, operator salvage and independent bootstrap review, mandatory accepted-control-plane provenance, and accepted implementation, merge, status, and current-tree validation chains for ASE3-019, ASE3-023, and ASE3-027; record the old and transition HEAD/tree plus implementation-daemon and configured-scheduler blob identities; then complete this gate in the same protected-board commit that adds and validates the reload receipt.
- Evidence subset: same namespace, false-completion recovery artifact SHA-256, failed_pre_dispatch_event_ase3_019_attempt_2_20260808.json, failed_pre_dispatch_log_ase3_019_attempt_2_20260808.txt, self_host_seed_failure_ase3_019_attempt_2_20260808.json, future operator_salvage_receipt_ase3_019_20260808.json with accepted_control_plane, old master and lane process-birth identities, terminal shutdown events, zero descendants/providers/containers/worktrees/merges, accepted ASE3-019/023/027 chains, old and transition HEAD/tree, implementation-daemon/configured-scheduler blob identities, joint current-tree validation, clean preflight, no counter restoration or legacy refill effects
- Generated by: ipfs_accelerate_py.agent_supervisor.operator-transition-gate@1
- Canonical board task: false
- Blocked reason: provider-attempt daemon reload boundary not yet accepted
- Acceptance: This task remains blocked even after its dependencies complete until an operator-owned current-tree receipt proves the exact old generation is fenced; ASE3-019 retains its original CID and is completed only by a no-provider salvage receipt whose accepted_control_plane binds the accepted router/runner/daemon origin and seed-shadow regression; ASE3-023/027 and that salvage are accepted ancestors of the transition head; their declared suites pass together on that tree; and no attempt counter, runtime state, or queue history was restored or rewritten. The old ASE3-006/018 completion projections and both failed ASE3-019 attempts never satisfy this gate. Completing it without the tracked strict reload receipt and convergence-manifest binding is invalid. After the completion commit passes convergence validation and configured preflight, relaunch exactly one process tree in the same namespace; ASE3-021 must start from that completion commit under the reloaded daemon.

## ASE3-029 Lower shared supervisor contracts into a neutral package

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: neutral-contract-layering
- Depends on: ASE3-022
- Goal id: ASE3-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/contracts/__init__.py, ipfs_accelerate_py/agent_supervisor/contracts/authority.py, ipfs_accelerate_py/agent_supervisor/contracts/execution.py, ipfs_accelerate_py/agent_supervisor/contracts/provider_capacity.py, ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/control/profile_authority.py, ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py, ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_provider_auto.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor_runner.py, test/api/test_agent_supervisor_contract_layering.py, test/api/test_agent_supervisor_configured_board_scheduler.py, test/api/test_agent_supervisor_implementation_daemon_runner.py, test/api/test_llm_router_agent_supervisor_fallback_route.py, test/api/test_agent_supervisor_control_plane_capsule_identity.py
- Validation: python -m pytest test/api/test_agent_supervisor_contract_layering.py test/api/test_agent_supervisor_configured_board_scheduler.py test/api/test_agent_supervisor_implementation_daemon_runner.py test/api/test_agent_supervisor_implementation_supervisor_runner.py test/api/test_agent_supervisor_implementation_provider_receipts.py test/api/test_implementation_provider_auto.py test/api/test_agent_supervisor_entrypoint_contracts.py test/api/test_agent_supervisor_prompt_v3_parallelism.py test/api/test_agent_supervisor_prompt_v3_authority_hardening.py test/api/test_agent_supervisor_profile_resolver.py test/api/test_agent_supervisor_control_plane_capsule_identity.py test/api/test_llm_router_agent_implementation_route.py test/api/test_llm_router_agent_supervisor_fallback_route.py test/api/test_agent_supervisor_grok_quota_terra_gate.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/neutral-contract-layering
- Parallel lane: neutral-contract-layering
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/contracts/__init__.py, ipfs_accelerate_py/agent_supervisor/contracts/authority.py, ipfs_accelerate_py/agent_supervisor/contracts/execution.py, ipfs_accelerate_py/agent_supervisor/contracts/provider_capacity.py, ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/control/profile_authority.py, ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py, ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py, ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_provider_auto.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor_runner.py, test/api/test_agent_supervisor_contract_layering.py, test/api/test_agent_supervisor_configured_board_scheduler.py, test/api/test_agent_supervisor_implementation_daemon_runner.py, test/api/test_llm_router_agent_supervisor_fallback_route.py, test/api/test_agent_supervisor_control_plane_capsule_identity.py
- Interfaces: VerifiedAuthorityBinding, ProviderAttemptReservation, InvocationBudget, ConfiguredBoardExecutionSlices, NonAuthoritativeProviderCapacityObservation, ProviderAttemptStoreService, ProfileAuthorityService, PlanExecutionStore, ExplicitSupervisorEffectInjection, NeutralContractCompatibilityExports, ProtectedCapsuleEffectInventory
- Conflict policy: Own only dependency inversion and behavior-preserving contract lowering. The neutral package may define immutable DTOs, codecs, validation, and verifier protocols but no key storage, lifecycle effects, provider selection, ranking, failure classification, freshness policy, authorization policy, fallback allow/deny, plan-store mutation, scheduling effects, claims, or process launch. Concrete lower `control.provider_attempt_store`, `control.profile_authority`, and `control.plan_execution_store` services alone own the displaced CAS persistence, profile key/lifecycle, and plan-store transaction effects; entrypoints remain orchestration and exact-identity compatibility/re-export APIs. Existing process roots construct and inject those services explicitly through neutral protocols; an ambient registry, service locator, import hook, environment-selected implementation, or hidden dynamic entrypoint import is forbidden. `llm_router` remains the sole provider-decision owner. Do not change the sealed ASE3-019/022/023/027 task contracts, runtime state, statuses, receipts, attempt counters, protected activation flags, `AgentImplementationRoutePlan`, `AgentImplementationFallbackDecision`, or the protected capacity projection rule `dispatch_authorized=False`.
- Preconditions: ASE3-022 has accepted and reloaded the exact ASE3-019/023/027 generation. From the final accepted ASE3-023 commit, capture a canonical AST inventory whose source HEAD, source tree, analyzer implementation digest, normalized edge records, and inventory SHA-256 are content-bound; the roadmap supplies no expected import count, and any count copied from an earlier prospective tree is stale and rejected. Before contract relocation, repair `test_agent_supervisor_configured_board_scheduler.py::_commit_v3_route_authorization` to create the canonical signed lifecycle-bound provider fallback authorization@2 plus root pin and witness, make every accepted artifact non-group/other-writable, and prove every affected baseline suite passes. Also repair `test_agent_supervisor_implementation_daemon_runner.py::test_daemon_resolves_relative_worktree_root_for_runner_workspace`: its ambient-provider-only setup must be replaced by the canonical signed accepted route authorization and binding, accepted public artifacts must be owned regular nonsymlinks at mode `0400` with private signer material kept at secure mode `0600`, and the complete daemon-runner file must pass without deselection, verifier bypass, or route weakening.
- Effects: Introduce the side-effect-free `agent_supervisor.contracts` package for shared authority DTO/verifier interfaces, invocation-budget and execution-slice contracts, provider-attempt reservation records, and non-authoritative provider-capacity observations. Move concrete CAS/profile/plan-store effects into the three lower control services and retain exact compatibility re-exports and orchestration in `entrypoints.contracts`, `entrypoints.execution_plan`, `entrypoints.provider_attempt_store`, and `entrypoints.local_profile`. Make `implementation_daemon_runner` and `implementation_supervisor_runner`, plus each already-declared runtime/todo-daemon process root that directly executes an effect, explicitly construct and inject the lower services; no process may depend on prior ambient registration. Migrate every lower-domain importer recorded by the accepted-tree inventory to neutral contracts or lower peers, then use a complete package AST inventory to reject direct, relative, aliased, constant-dynamic, and re-exported runtime/todo-daemon imports of entrypoints while preserving class identity, serialized schema/CID, persisted rows, argv, and public call behavior. Extend `llm_router`'s explicit security-critical capsule inventory to include both old compatibility wrappers and all three lower effect implementations; require protected review proving the route-plan/fallback-decision types and the `dispatch_authorized=False` capacity boundary are unchanged.
- Evidence subset: accepted-ASE3-023 source HEAD/tree and analyzer-digest-bound exact before inventory, normalized edge-list SHA-256, zero-edge after inventory, inventory-derived importer migration receipt without a roadmap-fixed count, canonical authorization@2 scheduler-fixture receipt and file-mode matrix, daemon-runner ambient-only red-baseline characterization, canonical signed accepted route authorization/binding receipt with owned regular nonsymlink `0400` public artifacts and secure private material, full non-deselected daemon-runner receipt, neutral-package side-effect/import closure, explicit constructor-injection and no-ambient-registry receipt, lower-effect ownership receipt, compatibility re-export object identity, schema/CID/round-trip and persisted-row parity, cold-import receipt, explicit old-wrapper-plus-lower-effect capsule inventory, independent protected-route semantic review, affected scheduler/runner/daemon/provider/authority/profile/capsule/parallelism regression receipts, exact current-tree DAG receipt
- Canonical board task: true
- Acceptance: The before receipt exactly reproduces the content-bound AST inventory from the final accepted ASE3-023 HEAD/tree and fails if an expected count is supplied independently of those bytes; the after gate enumerates the entire current package and reports zero imports from `agent_supervisor.runtime` or `agent_supervisor.todo_daemon` into `agent_supervisor.entrypoints`, including direct, relative, aliased, constant-dynamic, and re-export forms. Neutral authority, execution, and provider-capacity modules import neither entrypoints nor effectful control/runtime/todo-daemon code and perform no import-time I/O. Every importer discovered in the accepted-tree inventory consumes neutral contracts or lower peers; concrete effects exist only in the three declared lower control services and arrive through explicit process-root construction/injection with no ambient registry. The four entrypoint compatibility/orchestration modules preserve public object identity, schemas, CIDs, serialization, persisted rows, argv, and call behavior. The configured-scheduler canonical authorization@2 fixture passes under the ambient umask with accepted files carrying no group/other write bits. The repaired daemon-runner test exercises the production route resolver with canonical signed accepted authorization/binding and owned regular nonsymlink mode-`0400` public artifacts; the complete file passes without deselection, verifier bypass, ambient-only authorization, or route weakening. The capsule explicitly inventories old wrappers and new lower implementations, independent protected review proves `AgentImplementationRoutePlan`, `AgentImplementationFallbackDecision`, and `dispatch_authorized=False` unchanged, no provider policy moves from `llm_router`, and every declared affected suite plus the zero-upward-import DAG test passes on the exact current tree.

## ASE3-028 Restore router ownership and the package dependency direction

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-layering-repair
- Depends on: ASE3-029
- Goal id: ASE3-G020
- Outputs: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_provider_auto.py, ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py, test/api/test_implementation_provider_auto.py, test/api/test_agent_supervisor_prompt_v3_provider_route.py, test/api/test_agent_supervisor_router_owned_provider_decision.py
- Validation: python -m pytest test/api/test_implementation_provider_auto.py test/api/test_agent_supervisor_prompt_v3_provider_route.py test/api/test_agent_supervisor_router_owned_provider_decision.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/provider-layering-repair
- Parallel lane: provider-layering-repair
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_provider_auto.py, ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py, test/api/test_implementation_provider_auto.py, test/api/test_agent_supervisor_prompt_v3_provider_route.py, test/api/test_agent_supervisor_router_owned_provider_decision.py
- Interfaces: RouterOwnedProviderDecision, NonAuthoritativeProviderCapacityObservation, LegacyAutoProviderCompatibilityAdapter
- Conflict policy: ASE3-029 owns package lowering and the zero-upward-import DAG. This task owns only consolidation of duplicated provider eligibility, preference ranking, freshness, authentication/quota failure classification, authorization, and fallback allow/deny policy from `implementation_provider_auto` and `capability_resolver` into `ipfs_accelerate_py.llm_router`; both callers may normalize bounded non-authoritative inputs and execute one typed router decision, but may not retain a second decision table, reason-code mapping, route tuple, or compatibility bypass.
- Preconditions: ASE3-029 is accepted on the exact transition tree with the neutral authority, execution, and provider-capacity contracts and zero-upward-import AST receipt; ASE3-019 protected route behavior and legacy auto-provider suites are green without caller-owned provider policy.
- Effects: Define one immutable `RouterOwnedProviderDecision` and decision CID in `llm_router`; move preferred-provider eligibility filters, tie-breaking and ranking, authentication/quota/capacity classification, freshness, authorization, and final allow/deny into that router API; make `implementation_provider_auto` and the provider portion of `capability_resolver` submit the same neutral capacity observations plus verified policy context and consume the returned decision without reclassification; preserve non-v3 behavior through an explicit compatibility mode of the same router API while leaving capability resolution for resources, lanes, validation, coordination, and topology intact.
- Evidence subset: pre/post duplicate-policy AST inventory, one router decision schema/CID, observation-to-decision provenance, implementation-provider and capability-resolver decision equality, reason/freshness/authorization tamper matrix, v3 protected-route regression, legacy compatibility matrix, exact current-tree provider-route receipts
- Canonical board task: true
- Acceptance: One `RouterOwnedProviderDecision` from `ipfs_accelerate_py.llm_router` is the sole provider-policy result consumed by both `implementation_provider_auto` and `capability_resolver`; neither caller contains an independent provider/model/trigger/effort tuple, preferred-provider rank or tie-break table, authentication/quota/capacity classifier, freshness rule, authorization branch, or final allow/deny path; identical observations and policy context yield the same decision CID through both callers; the neutral capacity DTO cannot authorize dispatch; existing non-v3 auto-provider behavior and the ASE3-019 protected route contract remain green.

## ASE3-024 Make prompt intake and goal planning crash-safe and router-owned

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prompt-planning-transaction
- Depends on: ASE3-003, ASE3-004, ASE3-028
- Goal id: ASE3-G030
- Outputs: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py, ipfs_accelerate_py/agent_supervisor/entrypoints/planning_policy.py, ipfs_accelerate_py/agent_supervisor/entrypoints/planning_effect.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py, test/api/test_agent_supervisor_prompt_v3_prompt_transaction.py, test/api/test_agent_supervisor_prompt_broker.py, test/api/test_agent_supervisor_prompt_planning_policy.py, test/api/test_agent_supervisor_prompt_goal_planner.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_prompt_transaction.py test/api/test_agent_supervisor_prompt_broker.py test/api/test_agent_supervisor_prompt_planning_policy.py test/api/test_agent_supervisor_prompt_goal_planner.py test/api/test_agent_supervisor_prompt_v3_provider_route.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/prompt-planning-transaction
- Parallel lane: prompt-planning-transaction
- Resource class: provider-llm
- Predicted files: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py, ipfs_accelerate_py/agent_supervisor/entrypoints/planning_policy.py, ipfs_accelerate_py/agent_supervisor/entrypoints/planning_effect.py, ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py, test/api/test_agent_supervisor_prompt_v3_prompt_transaction.py, test/api/test_agent_supervisor_prompt_broker.py, test/api/test_agent_supervisor_prompt_planning_policy.py, test/api/test_agent_supervisor_prompt_goal_planner.py
- Interfaces: SignedPromptPlanningPolicyArtifact, DurablePromptIntent, MultiprocessPromptBrokerStore, BrokerContinuationLease, PromptPlanningRoutePlan, PlanningAttemptCAS, PlanningAttemptState, PlanningEffectAdoptionReceipt
- Conflict policy: `llm_router` owns planning-provider route and final admission, the separately versioned signed planning-policy artifact owns bounded planning route/retention/replay policy, the broker owns recoverable encrypted prompt bytes and expiring capabilities, the planner owns typed goal-graph validation, and a planning-specific durable CAS owns reservation/adoption. Do not add planning fields to or change the schema or signed bytes of `SignedSupervisorProfile`; do not reuse provider-fallback attempt CAS as planning authority; no layer may implement an independent planning retry, provider selector, replay-on-UNKNOWN rule, or prompt persistence path.
- Preconditions: ASE3-003 durable effect reservations, ASE3-004 typed planner/materializer contracts, and ASE3-028 router policy consolidation are accepted on one current tree; an operator-signed planning-policy artifact with its own schema version, CID, signer generation, bounds, expiry, and revocation state is available separately from the unchanged `SignedSupervisorProfile`.
- Effects: Verify the separate planning-policy artifact without mutating `SignedSupervisorProfile`; persist a run/context/policy-bound encrypted prompt in a multiprocess-safe durable broker with private regular-file or transactional-database invariants, bounded reads, interprocess locking, atomic commit, fencing, and crash recovery; acquire a single-use continuation lease; reserve one logical planning attempt in a dedicated CAS before any provider effect; obtain an immutable `PromptPlanningRoutePlan` from `llm_router`; advance only through `RESERVED -> EFFECT_STARTED -> TERMINAL_OBSERVED -> ADMITTED`, adopting the exact terminal output/root on restart; if an effect may have occurred but no exact terminal receipt is recoverable, durably record `UNKNOWN`, prohibit another provider call, and return `PROMPT_REPLAY_REQUIRED`; zeroize prompt bytes and continuation capability after admitted materialization or bounded expiry.
- Evidence subset: independent planning-policy signature/version/CID/rotation/revocation and unchanged SignedSupervisorProfile schema, multiprocess encrypted broker recovery/expiry and owner/mode/symlink/hardlink/swap/torn-write denials, continuation capability single-use/attenuation, planning-specific pre-effect CAS, router-owned route receipt, two-process winner/adoption, every crash boundary, terminal-output adoption, durable UNKNOWN and PROMPT_REPLAY_REQUIRED denial before a second effect, graph-root determinism, raw prompt/capability leak scan, zeroization receipt
- Canonical board task: true
- Acceptance: The signed planning-policy artifact is independently versioned, signed, content-addressed, expiring/revocable, and never changes or widens `SignedSupervisorProfile`; two real processes contending through the hardened durable broker and planning-specific CAS have one planning-effect winner and the loser adopts the same terminal output and program root; crashes before reservation, after reservation, during provider execution, after terminal output, and before graph admission yield the exact adopted result, or persist `UNKNOWN` and return `PROMPT_REPLAY_REQUIRED` before any second provider effect; restart recovers the encrypted continuation only under the exact run/context/policy capability; broker substitution, symlink, hardlink, permission, ownership, torn-write, stale-fence, and cross-run replay attacks fail closed; raw prompt bytes and bearer capabilities never enter argv, logs, taskboards, public immutable history, or receipts; prompt content cannot select a provider, model, fallback trigger, retry, budget, policy, or authority.

## ASE3-025 Prove canonical generated boards execute through the real adaptive runtime

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: generated-board-production-wiring
- Depends on: ASE3-004, ASE3-023, ASE3-024
- Goal id: ASE3-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py, ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py, ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, ipfs_accelerate_py/agent_supervisor/task_sources/generated_program_task_source.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_prompt_v3_generated_board_e2e.py, test/api/test_agent_supervisor_prompt_v3_plan_materializer.py, test/api/test_agent_supervisor_prompt_workflow_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_generated_board_e2e.py test/api/test_agent_supervisor_prompt_v3_plan_materializer.py test/api/test_agent_supervisor_prompt_workflow_contracts.py test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py test/api/test_agent_supervisor_configured_board_scheduler.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/generated-board-production-wiring
- Parallel lane: generated-board-production-wiring
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py, ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py, ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py, ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py, ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py, ipfs_accelerate_py/agent_supervisor/task_sources/generated_program_task_source.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_prompt_v3_generated_board_e2e.py, test/api/test_agent_supervisor_prompt_v3_plan_materializer.py, test/api/test_agent_supervisor_prompt_workflow_contracts.py
- Interfaces: CanonicalSupervisorProgram, AuthoritativeProgramRevision, GeneratedProgramSourceObserver, GeneratedBoardRuntimeProfile, EmbeddedTaskIdentity, GeneratedBoardExecutionReceipt, connect_duckdb_with_policy
- Conflict policy: DuckDB owns the authoritative program revision before any derived projection; Markdown and IPLD/Parquet are projections/history; the configured runtime observes the generated DuckDB source revision rather than requiring a Git-tracked board; reuse ASE3-023's production scheduler, task claims, worktrees, leases/fences, validation, merge queue, and ASE3-032's sole connection-birth policy. Do not add a second scheduler, task/claim store, connection-policy helper, generated-board-only executor, Git-track generated Markdown as an admission condition, recompute embedded task CIDs from projections, collapse subgoal ownership, retain a raw planning/generated-board `duckdb.connect`, or admit a wrapper/in-process acceptance path.
- Preconditions: ASE3-004 schema/admission, ASE3-023 active-plan dispatch, and ASE3-024 once-only prompt planning are accepted; Markdown, DuckDB, formal-plan, verified-IPLD, configured-scheduler, and implementation-supervisor baselines pass on the exact tree.
- Effects: Complete canonical goal, subgoal, task, dependency, evidence, output, validation, conflict/resource, provider, acceptance, and lineage records with embedded canonical task CIDs and explicit subgoal ownership; transactionally commit and CAS the `AuthoritativeProgramRevision` in DuckDB before emitting Markdown or immutable history; route every generated-board and planning-reachable connection birth, including `formal_plan_compiler`, through `connect_duckdb_with_policy` with no local raw-connect fallback or protected-setting override; expose a revision-fenced `GeneratedProgramSourceObserver` to the configured scheduler; derive matching optional Markdown/history projections without making Git index state authority; construct a namespace-independent per-run `GeneratedBoardRuntimeProfile` from the observed authoritative revision and signed bounds; launch the real configured scheduler and implementation supervisor as genuine subprocesses against that generated source; adopt the existing program revision on replay without invoking the planner again; repair the retired flat prompt-workflow standalone fixture to load the canonical package module.
- Evidence subset: complete generated-record schema, embedded goal/task ID-to-CID map and subgoal-owner preservation, DuckDB-first transaction/revision/CAS at every crash boundary, exact generated-board/planning connection-birth inventory and policy-helper delegation proof, concurrent materialization winner/adoption, generated-source observer revision/fence, Markdown/IPLD/history parity and partial-projection denial, untracked/missing Markdown success, planner-call count, namespace-independent runtime profile matrix, genuine configured-scheduler/supervisor/daemon subprocess births argv claims and terminal receipts, disjoint overlap timeline, conflict serialization, raw-connect/policy-override denial, restart/drift replan, retired-fixture repair
- Canonical board task: true
- Acceptance: Replay and restart never reinvoke the planner after one DuckDB-authoritative program revision exists; every generated-board/planning-reachable connection, including formal-plan compilation, delegates to `connect_duckdb_with_policy`, and raw `duckdb.connect`, a local helper clone, policy override, or insecure fallback fails closed. Every embedded task CID and subgoal owner survives DuckDB write/read, generated-source observation, scheduling, claim, and receipt without projection-based recomputation or flattening; concurrent materializers produce one authoritative revision before any projection and partial, absent, untracked, or Git-dirty Markdown/IPLD projections never gain or block execution authority; `GeneratedBoardRuntimeProfile` works for at least two arbitrary valid non-ASE3 namespaces without seed identifiers; a genuine subprocess E2E starts the real configured scheduler, implementation supervisor, and implementation daemon from the observed generated source, overlaps at least two disjoint real effects, and serializes dependency/file/resource/provider/validation conflicts; no hard-coded namespace, Git-tracked-board requirement, seed board, monkeypatched/in-process wrapper, fake claim, fabricated birth, or no-op effect can satisfy the gate; the corrected canonical prompt-workflow contract suite passes.

## ASE3-021 Wire durable production refill, append/adoption CAS, and completion authority

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: refill-completion-hardening
- Depends on: ASE3-004, ASE3-006, ASE3-007, ASE3-019, ASE3-022, ASE3-024, ASE3-025
- Goal id: ASE3-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_event_adapter.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_store.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_prompt_v3_refill_hardening.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_refill_hardening.py test/api/test_agent_supervisor_prompt_v3_refill.py test/api/test_agent_supervisor_backlog_refinery.py test/api/test_agent_supervisor_goal_completion.py test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/refill-completion-hardening
- Parallel lane: refill-completion-hardening
- Resource class: io-database
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_event_adapter.py, ipfs_accelerate_py/agent_supervisor/entrypoints/refill_store.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_prompt_v3_refill_hardening.py
- Interfaces: DurableRefillState, DurableRefillSagaCursor, RefillPhaseDeadline, SignedRefillPolicy, CurrentTreeResidualEvaluator, RefillAppendReceipt, RefillAdoptionReceipt, ProductionRefillRuntime, ProductionRefillEventAdapter, PlanInvalidationReceipt
- Conflict policy: Own durable residual evaluation, refill budgets/state, canonical append/adoption CAS, production event adapters, and plan invalidation/recompile wiring only; generated records must use the ASE3-025 canonical schema and ASE3-023 runtime. Do not mutate an active board outside revision control, bypass plan admission, create a second task parser/scheduler/claim store, activate protected config, or let generated work widen authority, budgets, effects, dependencies, or acceptance gates.
- Preconditions: ASE3-004/024 provide a once-only admitted prompt program, ASE3-025 provides the real generated-board runtime/schema, ASE3-007 provides deterministic local evaluation primitives, and ASE3-022 reloads the accepted provider generation; backlog refinery, objective tracker, goal completion, Planner Doctor/self-improvement, Doctor findings, and canonical task-source services pass their current suites.
- Effects: Compose actual production scheduler, validation, review, merge, Doctor, retry-exhaustion, drift, low-water, and open-goal events into the residual evaluator; bind evaluation to the observed current integration tree and advance one durable saga cursor through `EVALUATING -> APPEND_RESERVED -> APPENDED -> PLAN_INVALIDATED -> RECOMPILED -> DISPATCHED` or terminal `ADOPTED`. Persist every transition, predecessor, reservation, deadline, epoch, canonical seen-gap identity, cooldown, unchanged-set count, budget use, circuit state, and terminal decision across processes and restarts. Append ASE3-025-complete records through one task-source revision CAS; on CAS loss or crash, reread and adopt an identical append/plan/dispatch winner or reevaluate the new tree; never skip an intermediate phase. Publish phase-specific monitor deadlines before work, invalidate the active plan, recompile, and dispatch the admitted descendant through the configured scheduler; wire each trigger before retry/stall paths return and force a final root scan before completion. Install this path dormant; only ASE3-026 may authorize activation.
- Evidence subset: production event-adapter manifest, signed-cap equality, current-tree join, individual trigger/no-trigger matrix, exact `EVALUATING→APPEND_RESERVED→APPENDED→PLAN_INVALIDATED→RECOMPILED→DISPATCHED/ADOPTED` cursor lineage, phase-specific monitor deadlines, immutable append/adoption receipts, multiprocess/restart CAS, ASE3-025 schema parity, generated descendant claim/effect receipt, dedup/cooldown/oscillation state, branch/stale reopening, dormant-before-activation proof, final completion decision, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/post_wave3_residuals_20260808.json#production-durable-refill-wiring
- Acceptance: No boolean callback or caller-supplied completion value can authorize append or completion; two processes and crash/restart produce one revision, one monotonic durable saga cursor, and no duplicate gap/task/effect. Every transition is CAS-bound to its predecessor and current tree; missing, reordered, expired, cross-revision, or fabricated phases fail closed. Recovery adopts an identical append, compiled plan, or dispatch without replay and resumes at the first incomplete phase; monitor deadlines distinguish a live bounded phase from a refill stall. Every generated record conforms to ASE3-025 schema; every named trigger proves `EVALUATING -> APPEND_RESERVED -> APPENDED -> PLAN_INVALIDATED -> RECOMPILED -> DISPATCHED/ADOPTED`; unchanged residuals reach EXHAUSTED without oscillation; only a healthy final scan permits completion; all refill flags remain dormant until ASE3-026's validated pre-effect authorization is consumed.

## ASE3-020 Converge transactional run truth and compose the real crash-safe prompt saga

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: transactional-runtime-hardening
- Depends on: ASE3-003, ASE3-005, ASE3-018, ASE3-019, ASE3-021, ASE3-024, ASE3-025, ASE3-028
- Goal id: ASE3-G055
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/runtime/durable_process.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_runtime_hardening.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_runtime_hardening.py test/api/test_agent_supervisor_prompt_v3_run_registry.py test/api/test_agent_supervisor_run_registry.py test/api/test_agent_supervisor_prompt_v3_runtime.py test/api/test_agent_supervisor_lifecycle_orchestrator.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/transactional-runtime-hardening
- Parallel lane: transactional-runtime-hardening
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/runtime/durable_process.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_prompt_v3_runtime_hardening.py
- Interfaces: AuthoritativeRunRevisionStore, ImmutableRunEpoch, ImmutableRunHistoryVector, MonitorProgressCursorVector, MonitorReadyEffectReservation, UnknownOutcomeAdoptionReceipt, EffectReservation, ProcessBirthObservation, CompleteLaunchPlanGuard, RequiredArgumentCoverageReceipt, StandardSupervisorRuntimeFactory, PromptToRunSaga, connect_duckdb_with_policy
- Conflict policy: Final runtime fan-in only after ASE3-018/019/021/024/025/028. Own the shared monitor-ready contracts, DuckDB immutable history/cursor vectors, effect reservations, durable process observations, unknown-outcome adoption semantics, and policy-helper delegation for run-registry/runtime-history connection births; ASE3-008 alone owns monitor policy, health classification, guardian integration, and recovery decisions, while ASE3-032 alone owns `connect_duckdb_with_policy` and its settings. Projections, fixture CompleteLaunchPlan values, arbitrary handler mappings, PIDs, logs, provider claims, and a raw `duckdb.connect` never establish success; consume router decisions but keep provider layering/refactoring in ASE3-028; do not weaken context, authority, fallback, refill, lifecycle, lease/fence, database policy, or validation contracts.
- Preconditions: Hardened canonical resolution, signed provider authority, crash-safe planning reservation, authoritative generated program revision, durable refill, admitted materialization, lifecycle orchestration, task-source coordination, corrected provider layering, and current run-registry baselines pass on the exact integration tree.
- Effects: Make DuckDB authoritative for complete run heads, immutable append-only run-history vectors, monotonic lifecycle/monitor/refill cursor vectors, adoption/idempotency keys, monitor-ready effect reservations, verified process births, leases/fences, and persisted immutable epochs; validate stored columns and vector links against content, reconstruct exact revisions, and migrate only verified JSON facts. Route every run-registry and runtime-history connection birth reachable from the prompt product through ASE3-032's `connect_duckdb_with_policy`, with no local raw-connect fallback or policy override. Consume the ASE3-024 planning reservation/adoption and ASE3-025 AuthoritativeProgramRevision rather than a fixture CompleteLaunchPlan; compare every complete launch-plan field immediately before each effect; reserve and fence intent before external effects, record EFFECT_STARTED before release, then after crash observe and adopt the exact effect/receipt. If an effect may have happened but cannot be identified exactly, persist UNKNOWN, prohibit replay, and expose a typed unknown-outcome adoption continuation to ASE3-008. Introspect the actual supervisor/daemon parsers, prove every required argument has one verified resolver receipt or signed safe default, compile accepted argv, and compose every public lifecycle operation against real services.
- Evidence subset: schema/migration parity, full immutable history and predecessor/hash-chain vectors, monotonic lifecycle/monitor/refill cursor vectors, concurrent CAS and monitor-ready effect reservation, exact run-registry/runtime-history connection-birth inventory and policy-helper delegation proof, planning/program/refill joins, complete-field stale denial, actual supervisor/daemon parser manifests, required-argument resolver/signed-default coverage, real parser argv, real process birth/lease/fence/heartbeat/cursor observations, full operation receipts, every intent/effect/receipt crash boundary, exact restart/adoption, UNKNOWN non-replay, raw-connect/policy-override denial, fixture/no-op/fabricated receipt denial, current-tree validation, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/post_wave3_residuals_20260808.json#transactional-run-truth-and-effect-recovery
- Acceptance: Concurrent revisions or effects have one winner; corrupt, truncated, reordered, forked, column-mismatched, or nonmonotonic history/cursor vectors and unverified migration facts deny. Every prompt-product-reachable run-registry and runtime-history connection delegates to `connect_duckdb_with_policy`; raw `duckdb.connect`, a local helper clone, policy overrides, and insecure fallback fail the exact inventory and integration gates. Production construction rejects arbitrary mapping receipts, fixture CompleteLaunchPlan values, and missing/fake handlers; every actual parser argument has exactly one verified resolver output or signed safe default and emitted argv parses without shortcuts. RUNNING requires one same-revision join of lifecycle and monitor process births, leases, fences, fresh heartbeats, monotonic cursors, state revision, and health revision; projection-only or synthetic identities cannot pass. All public operations use the complete saga; crashes around every external effect neither duplicate nor lose known effects, while an unknown outcome is durably adopted as UNKNOWN and cannot be replayed; exact retries resume the same logical run.

## ASE3-026 Authorize, activate, and observe the durable refill and autonomous monitor runtime

- Status: blocked
- Completion: manual
- Is schedulable: false
- Review only: true
- Priority: P0
- Track: protected-runtime-activation
- Depends on: ASE3-008, ASE3-020, ASE3-021, ASE3-025
- Goal id: ASE3-G060
- Outputs: config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json, ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py, test/api/test_agent_supervisor_prompt_v3_runtime_activation.py, test/api/test_agent_supervisor_prompt_v3_convergence.py, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/protected_runtime_activation_receipt.json, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/protected_runtime_post_activation_observation_receipt.json, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/convergence_manifest.json
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_runtime_activation.py test/api/test_agent_supervisor_prompt_v3_monitor.py test/api/test_agent_supervisor_prompt_v3_refill_hardening.py test/api/test_agent_supervisor_prompt_v3_convergence.py -q
- Board namespace: agent-supervisor-prompt-only-self-improvement-v3
- Bundle: agent-supervisor/prompt-self-improvement-v3/protected-runtime-activation
- Parallel lane: protected-runtime-activation
- Resource class: coordinator
- Predicted files: config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json, ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py, test/api/test_agent_supervisor_prompt_v3_runtime_activation.py, test/api/test_agent_supervisor_prompt_v3_convergence.py, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/protected_runtime_activation_receipt.json, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/protected_runtime_post_activation_observation_receipt.json, data/agent_supervisor/prompt_only_self_improvement_v3/convergence/convergence_manifest.json
- Interfaces: ProtectedRuntimeActivationAuthorization, ProtectedRuntimePostActivationObservation, RuntimeGenerationActivationCAS, MonitorBirthReceipt, RefillActivationReceipt, RuntimeGenerationReloadReceipt
- Conflict policy: Operator-owned protected authorization, config, convergence validation, post-effect observation, and reload fan-in only; ordinary implementation workers cannot schedule this task. The signed authorization is pre-effect authority only and must never claim that a generation, lifecycle, monitor, heartbeat, cursor, or refill effect ran. A separate post-activation observation may report effects but cannot retroactively authorize them. Do not mutate attempts, queue history, canonical task identities, provider policy, accepted implementation receipts, or another namespace; do not enable broad legacy codebase refill or restore legacy hash sharding for plan-bound slices.
- Preconditions: ASE3-008/020/021/025 are accepted together on one clean exact inactive tree; refill and monitor flags are false; no owned worker/provider/merge effect is active. Both reserved receipt paths are absent until strict schemas and manifest component bindings land; the pre-effect authorization binds the candidate HEAD/tree, exact protected blobs, old generation, zero-descendant proof, target generation old+1, CAS/lease identity, guardian identity/review, bounded flags, and expiry.
- Effects: Phase one atomically validates and records only the signed pre-effect activation authorization; that receipt sets neither active flags nor process/effect observations and cannot claim success. After that authorization commit validates, exactly one CAS/lease winner adopts or launches target generation old+1 through the ReviewedHostNamespaceReconciler, fences the old generation, enables only scoped prompt-program/objective refill plus the detached monitor, keeps broad legacy codebase refill false, and disables legacy hash sharding for active slices. Phase two separately records immutable post-activation observations of actual same-generation lifecycle and monitor births, leases, fences, heartbeats, cursors, low-water/open-goal refill append -> recompile -> dispatch/adoption, worker/monitor recovery, client-disconnect survival, and durable budget/dedup state; only that observation may release ASE3-009.
- Evidence subset: exact authorization HEAD/tree/protected blobs, signed pre-effect authorization schema/digest/expiry, explicit `authorization_effect_observed: false`, zero-old-generation descendants/effects, target old+1 CAS/lease and guardian review, scoped dormant-to-active flag transition, broad legacy refill denial, active-slice sharding denial, separate post-activation observation schema/digest, actual same-generation lifecycle/monitor births/leases/fences/heartbeats/cursors, refill saga dispatch/adoption trace, worker/monitor recovery, client disconnect, durable restart budget/dedup parity, fresh convergence and configured-board preflight
- Generated by: ipfs_accelerate_py.agent_supervisor.protected-runtime-activation-gate@1
- Canonical board task: true
- Blocked reason: protected runtime activation receipt not yet accepted
- Acceptance: ASE3-026 remains blocked while either strict receipt is absent. The signed authorization must be validated on an inactive exact tree before any config/process effect and must state that no activation effect has yet been observed; any birth, heartbeat, cursor, refill, completion, or already-active claim in that authorization fails closed. Exactly one matching old+1 CAS/lease winner may consume it; retries adopt that winner. Completion and ASE3-009 readiness require a separately signed, manifest-bound post-activation observation that joins actual lifecycle and monitor process births, leases, fences, fresh heartbeats, monotonic cursors, and refill dispatch/adoption to that exact generation; authorization alone never proves the effect ran.
