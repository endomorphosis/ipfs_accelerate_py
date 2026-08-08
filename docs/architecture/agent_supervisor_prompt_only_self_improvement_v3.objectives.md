# Agent Supervisor Prompt-Only Self-Improvement v3 Objective Heap

This objective heap is the authoritative goal hierarchy for the current-main
convergence program described by
`AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md`. Its executable
projection is `agent_supervisor_prompt_only_self_improvement_v3.todo.md` with
task prefix `## ASE3-`.

ASE3 is a fresh successor namespace. Historical ASE/ASE2 task status, lane
state, PID files, branches, and receipts do not satisfy this heap unless
ASE3-000 verifies the artifact on the current base and a later ASE3 producer
emits a new exact-tree acceptance receipt.

## Goal tree

```text
ASE3-G000  Production prompt-only self-improving supervisor
|-- ASE3-G010  Current-main truth and evidence-preserving convergence
|-- ASE3-G020  Trusted prompt-only intent and argument resolution
|-- ASE3-G030  Canonical goal, subgoal, and taskboard materialization
|-- ASE3-G040  Durable real lifecycle and conflict-aware parallel execution
|-- ASE3-G050  Evidence-driven bounded goal and task refill
|-- ASE3-G060  Progress watchdog, Doctor, and deterministic recovery
|-- ASE3-G070  Python, CLI, MCP, and MCP++ product convergence
`-- ASE3-G080  Self-hosted verification, rollout, and closeout
```

## ASE3-G000 Production prompt-only self-improving supervisor

- Status: active
- Parent:
- Parent goal IDs JSON: []
- Depends on:
- Dependencies JSON: []
- Fib priority: 1
- Track: prompt-self-improvement-v3
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/root
- Parallel lane: program
- Resource class: coordinator
- Goal: Deliver one safe prompt-only product facade that creates a canonical goal/subgoal/task program, runs independent work in parallel, refills objective residuals, and remains truthfully monitored until current-tree evidence authorizes completion.
- Producing tasks: ASE3-000, ASE3-001, ASE3-002, ASE3-003, ASE3-004, ASE3-005, ASE3-006, ASE3-007, ASE3-008, ASE3-009, ASE3-010, ASE3-011, ASE3-012, ASE3-013, ASE3-014
- Evidence: prompt_self_improvement_v3.PROMPT_SELF_IMPROVEMENT_V3_REQUIREMENT_ID
- Evidence requirements JSON: ["current-main convergence receipt", "prompt resolution receipt", "canonical objective and task roots", "real lifecycle launch receipt", "parallel overlap and conflict-serialization receipt", "bounded refill receipt", "watchdog recovery receipt", "cross-transport conformance receipt", "self-hosted current-tree rollout decision"]
- Evidence criteria: One prompt plus trusted ambient context produces or adopts a durable run without expert daemon flags; prompt content never grants authority; goals, subgoals, and tasks have canonical lineage; independent work overlaps; drained-open objectives refill; false-running, stalls, and soft completion recover or fail closed; and completion is proven on the exact promoted tree.
- Evidence source policy: Prompt prose, plans, historical task status, runtime Markdown copies, PID existence, logs, provider claims, branch-local commits, stale tests, and process exit alone are non-authoritative; accept only current-tree receipts joined through the owning DuckDB run revision, leases/fences, accepted integration commits, immutable event history, validation, and rollout decision.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md, data/agent_supervisor/prompt_only_self_improvement_v3
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints", "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools", "docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md", "data/agent_supervisor/prompt_only_self_improvement_v3"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v3_e2e.py test/api/test_agent_supervisor_prompt_only_v3_conformance.py test/api/test_agent_supervisor_prompt_only_v3_chaos.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_v3_e2e.py test/api/test_agent_supervisor_prompt_only_v3_conformance.py test/api/test_agent_supervisor_prompt_only_v3_chaos.py -q"]
- Acceptance: Python, CLI, MCP, and MCP++ expose equivalent real lifecycle operations; an admitted self-improvement prompt executes a parallel taskboard, refills a forced residual, recovers a forced stall, converges accepted work onto the release candidate, and reaches completion only after a final evidence scan and active rollout receipt.
- Gap task: Execute the smallest ready ASE3 task under the shared conflict, lease, and merge coordinator.

## ASE3-G010 Current-main truth and evidence-preserving convergence

- Status: completed
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 2
- Track: convergence
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/convergence
- Parallel lane: convergence
- Resource class: coordinator
- Goal: Establish a clean current-main integration base and selectively salvage useful v2 work without trusting stale completion state or overwriting unrelated user changes.
- Producing tasks: ASE3-000
- Evidence: prompt_v3_convergence.CURRENT_MAIN_CONVERGENCE_REQUIREMENT_ID
- Evidence requirements JSON: ["exact base and tree receipt", "dirty-worktree preservation receipt", "v1/v2 state contradiction report", "per-file and per-commit port rewrite discard map", "clean isolated integration worktree receipt"]
- Evidence criteria: Every preserved rescue artifact is classified against current code and tests; current-main replacements are preferred; state projections are recorded but not promoted to authority; the integration worktree is isolated, reproducible, and has no unaccounted changes.
- Evidence source policy: Historical branch names, task status, PID files, logs, and aggregate completion indexes are hints only; Git object identity, current source, current tests, registry/event integrity, and explicit operator-owned dirty-change preservation are authoritative.
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py, test/api/test_agent_supervisor_prompt_v3_convergence.py, config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json, .gitignore, data/agent_supervisor/prompt_only_self_improvement_v3/convergence
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py", "test/api/test_agent_supervisor_prompt_v3_convergence.py", "config/agent_supervisor_prompt_only_self_improvement_v3_scheduler.json", ".gitignore", "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_convergence.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_v3_convergence.py -q"]
- Acceptance: No wholesale merge of the stale rescue branch occurs; every port is current-base revalidated; no user change is lost; and downstream tasks receive an exact base, protected paths, salvage manifest, fresh ignored runtime namespace, and a sealed configured-board profile that passes preflight on the integration branch.
- Gap task: Repair the smallest missing provenance, tree-identity, state-reconciliation, or worktree-isolation fact.

## ASE3-G020 Trusted prompt-only intent and argument resolution

- Status: active
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on: ASE3-G010
- Dependencies JSON: ["ASE3-G010"]
- Fib priority: 3
- Track: inference-policy
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/inference-policy
- Parallel lane: inference-policy
- Resource class: security-small
- Goal: Resolve every normal operational argument from one frozen trusted context while keeping intent, authority, provider policy, resource ceilings, and transport authentication separate.
- Producing tasks: ASE3-001, ASE3-002
- Evidence: prompt_v3_resolution.PROMPT_ONLY_RESOLUTION_REQUIREMENT_ID
- Evidence requirements JSON: ["frozen invocation-context CID", "field-level inference provenance", "signed local profile receipt", "transport target-binding matrix", "provider policy and attempt receipts", "prompt non-influence matrix"]
- Evidence criteria: Local CLI, Python, MCP, and MCP++ resolve deterministic target, state, profile, run, objective, task-source, provider, resources, validation, and topology values from allowed evidence; material ambiguity yields one typed continuation; inference never grants effects.
- Evidence source policy: Prompt text, repository instructions, arbitrary client paths, environment-only identity, generic provider failures, and credential presence are not authority; accept verified local profiles, server policy, UCAN, bounded repository observation, fresh provider evidence, and explicit authorized overrides.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_resolution.py test/api/test_agent_supervisor_prompt_v3_authority.py test/api/test_agent_supervisor_prompt_v3_provider_route.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_v3_resolution.py test/api/test_agent_supervisor_prompt_v3_authority.py test/api/test_agent_supervisor_prompt_v3_provider_route.py -q"]
- Acceptance: After one explicit local initialization, a normal local caller supplies only a prompt; all inferred values are explainable and replay-stable; unsafe, stale, or ambiguous inputs fail before effects; provider fallback remains exact, typed, pre-effect, and bounded.
- Gap task: Repair the smallest missing evidence adapter, precedence join, signature/UCAN check, ambiguity disposition, resource inference, or provider-attempt guard.

## ASE3-G030 Canonical goal, subgoal, and taskboard materialization

- Status: active
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on: ASE3-G020
- Dependencies JSON: ["ASE3-G020"]
- Fib priority: 5
- Track: program-materialization
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/program-materialization
- Parallel lane: program-materialization
- Resource class: io-database
- Goal: Compile prompt intent into an admitted canonical root goal, subgoal DAG, atomic task DAG, conflict/resource plan, and equivalent DuckDB, Markdown, and immutable history projections.
- Producing tasks: ASE3-004
- Evidence: prompt_v3_materializer.CANONICAL_PROMPT_PROGRAM_REQUIREMENT_ID
- Evidence requirements JSON: ["redacted prompt intent receipt", "goal population root", "task population root", "acyclic lineage and dependency report", "DuckDB Markdown IPLD parity", "plan admission receipt"]
- Evidence criteria: Generated goals own evidence and producing tasks; tasks carry all supervisor fields; duplicate IDs, cycles, unknown dependencies, unsafe validation, missing effect declarations, and unbounded scope fail admission; raw prompt bodies and secrets do not leak into durable projections.
- Evidence source policy: Model-formatted Markdown and task prose are proposals only; typed planner output, deterministic lint/admission, canonical identities, and committed projection receipts are authoritative.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, test/api/test_agent_supervisor_prompt_v3_plan_materializer.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py", "test/api/test_agent_supervisor_prompt_v3_plan_materializer.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_plan_materializer.py test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_v3_plan_materializer.py test/api/test_agent_supervisor_markdown_task_source.py test/api/test_agent_supervisor_duckdb_task_source.py -q"]
- Acceptance: One prompt deterministically produces a supervisor-readable goal/subgoal/task hierarchy and root-bound dual projection; replay is idempotent; concurrent materialization uses CAS; and every task is schedulable or has a typed admission reason.
- Gap task: Repair the smallest planner, goal-lineage, task-schema, admission, projection, identity, or prompt-redaction residual.

## ASE3-G040 Durable real lifecycle and conflict-aware parallel execution

- Status: active
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on: ASE3-G010, ASE3-G030
- Dependencies JSON: ["ASE3-G010", "ASE3-G030"]
- Fib priority: 8
- Track: runtime-parallelism
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/runtime-parallelism
- Parallel lane: runtime-parallelism
- Resource class: process-control
- Goal: Replace simulation and no-op defaults with a resumable real lifecycle and dispatch the largest deterministic conflict-free task set under shared leases, fences, resources, and merge serialization.
- Producing tasks: ASE3-003, ASE3-005, ASE3-006
- Evidence: prompt_v3_runtime.REAL_PARALLEL_RUNTIME_REQUIREMENT_ID
- Evidence requirements JSON: ["DuckDB run revision CAS", "complete launch-plan and pre-effect snapshots", "real process start or adoption receipt", "parallel overlap trace", "conflict serialization trace", "post-merge current-tree receipt"]
- Evidence criteria: Missing backend/effect handlers fail typed and closed; launch/adopt is idempotent across crashes; mutable run truth lives in DuckDB; effects revalidate tree, authority, policy, lease, and fence; admitted lane budgets propagate without a hard-coded width; empty recovered slices deny work; independent tasks overlap while conflicts serialize; only accepted work satisfies dependencies and is reachable from the integration head.
- Evidence source policy: In-memory records, no-op callbacks, JSON status, PID liveness, runtime Markdown status, worker self-report, and branch-local commits are non-authoritative without joined registry, effect, validation, merge, and current-tree receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_run_registry.py test/api/test_agent_supervisor_prompt_v3_runtime.py test/api/test_agent_supervisor_prompt_v3_parallelism.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_v3_run_registry.py test/api/test_agent_supervisor_prompt_v3_runtime.py test/api/test_agent_supervisor_prompt_v3_parallelism.py -q"]
- Acceptance: Production `run` starts or adopts a real lifecycle, never fabricates completion, survives every injected intent/effect/receipt crash boundary, overlaps at least two disjoint tasks, rejects duplicate effects, and revalidates every accepted change on the current integration tree.
- Gap task: Repair the smallest registry, launch-plan, effect-handler, lifecycle, claim, conflict, resource, worktree, merge, or current-tree validation residual.

## ASE3-G050 Evidence-driven bounded goal and task refill

- Status: active
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on: ASE3-G030, ASE3-G040
- Dependencies JSON: ["ASE3-G030", "ASE3-G040"]
- Fib priority: 13
- Track: refill-completion
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/refill-completion
- Parallel lane: refill-completion
- Resource class: coordinator
- Goal: Keep useful work available while objective evidence remains open, but bound generation, deduplicate residuals, and prohibit task-drain or branch-local work from becoming completion.
- Producing tasks: ASE3-007
- Evidence: prompt_v3_refill.BOUNDED_RESIDUAL_REFILL_REQUIREMENT_ID
- Evidence requirements JSON: ["low-water refill receipt", "drained-open-goal refill receipt", "deduplication identity", "budget and circuit-breaker receipt", "final forced residual scan", "completion authority decision"]
- Evidence criteria: Low backlog, open objectives after drain, failed validation/review/merge, evidence drift, and retry exhaustion trigger current-tree reconciliation and bounded refinement before a stuck path can return; the signed production profile registers Planner Doctor/self-improvement evaluation; healthy evidence-complete drain does not refill; equivalent residuals cannot create infinite work or relax their own gates.
- Evidence source policy: Queue emptiness, task status, model novelty claims, stale goal snapshots, and historical green tests cannot authorize refill or completion; use current evidence gaps, canonical identities, revision CAS, budgets, and goal-completion receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py, test/api/test_agent_supervisor_prompt_v3_refill.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py", "test/api/test_agent_supervisor_prompt_v3_refill.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_refill.py test/api/test_agent_supervisor_backlog_refinery.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_v3_refill.py test/api/test_agent_supervisor_backlog_refinery.py -q"]
- Acceptance: Forced tests cover all refill triggers, no-refill completion, deduplication, cooldown, depth/work/epoch budgets, oscillation, branch-only completion, and final root closure; unchanged unresolvable evidence ends blocked or exhausted without spinning.
- Gap task: Repair the smallest trigger, evidence refresh, goal reconciliation, deduplication, budget, oscillation, successor-epoch, or completion-authority residual.

## ASE3-G060 Progress watchdog, Doctor, and deterministic recovery

- Status: active
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on: ASE3-G040, ASE3-G050
- Dependencies JSON: ["ASE3-G040", "ASE3-G050"]
- Fib priority: 21
- Track: monitoring-recovery
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/monitoring-recovery
- Parallel lane: monitoring-recovery
- Resource class: process-control
- Goal: Maintain truthful lifecycle and semantic-progress health, classify stalls and false idle, and execute only bounded idempotent recovery authorized by the current launch plan.
- Producing tasks: ASE3-008
- Evidence: prompt_v3_monitor.PROGRESS_AWARE_WATCHDOG_REQUIREMENT_ID
- Evidence requirements JSON: ["process-birth and heartbeat receipt", "semantic progress deadline trace", "stale-running repair", "stall and false-idle classification matrix", "bounded restart/rescue receipt", "terminal shutdown receipt"]
- Evidence criteria: Running requires matching process identity, lease/fence, and fresh heartbeat; progress uses event/revision/phase movement rather than log noise; dead, frozen, idle-open-goal, duplicate, stale-authority, and oscillating cases recover deterministically or become typed operator action.
- Evidence source policy: PID alone, status JSON, log growth, process exit, and watchdog self-report are non-authoritative; join process birth, registry owner/revision, heartbeat, event cursor, phase deadlines, leases/fences, objective evidence, and recovery receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_monitor.py, test/api/test_agent_supervisor_prompt_v3_monitor.py, test/api/test_agent_supervisor_task_attempt_limit.py, docs/guides/AGENT_SUPERVISOR_PROMPT_RUNBOOK.md
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/run_monitor.py", "test/api/test_agent_supervisor_prompt_v3_monitor.py", "test/api/test_agent_supervisor_task_attempt_limit.py", "docs/guides/AGENT_SUPERVISOR_PROMPT_RUNBOOK.md"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_v3_monitor.py test/api/test_agent_supervisor_supervisor_watchdog.py test/api/test_agent_supervisor_task_attempt_limit.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_v3_monitor.py test/api/test_agent_supervisor_supervisor_watchdog.py test/api/test_agent_supervisor_task_attempt_limit.py -q"]
- Acceptance: Dead PID with stale running state, PID reuse, missing shutdown, ready-without-worker, frozen worker, drained-open-goal, lost lease, duplicate launch, backoff expiry, circuit-breaker, retry-accounting, idle-heartbeat, quota-attribution, and provider-review deferral tests all yield truthful health and one bounded continuation; stale attempt-limit fixtures are reconciled to current APIs without weakening behavior; the standalone monitor has a live clock and an authorized restart callback or explicitly reports detection-only mode.
- Gap task: Repair the smallest heartbeat, process-identity, phase-deadline, Doctor finding, stale-projection, recovery callback, backoff, circuit-breaker, or shutdown residual.

## ASE3-G070 Python, CLI, MCP, and MCP++ product convergence

- Status: active
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on: ASE3-G020, ASE3-G040, ASE3-G050, ASE3-G060
- Dependencies JSON: ["ASE3-G020", "ASE3-G040", "ASE3-G050", "ASE3-G060"]
- Fib priority: 34
- Track: public-facades
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/public-facades
- Parallel lane: public-facades
- Resource class: io-small
- Goal: Export one production `Supervisor` service through Python, CLI, MCP, and MCP++ with prompt-only defaults, stable typed operations, cold imports/help, and compatibility for expert surfaces.
- Producing tasks: ASE3-009, ASE3-010, ASE3-011, ASE3-012
- Evidence: prompt_v3_facade.CROSS_TRANSPORT_PRODUCT_REQUIREMENT_ID
- Evidence requirements JSON: ["package export manifest", "cold import and help receipt", "Python CLI MCP MCP++ canonical parity", "server target allowlist denial", "MCP++ UCAN attenuation", "expert compatibility report"]
- Evidence criteria: Every transport delegates to one service and returns matching handles, statuses, continuations, events, and error semantics; normal run/preview needs only a prompt; steer/observe infer a sole compatible run; absent production composition is typed unavailable, never simulated success.
- Evidence source policy: Adapter defaults, prompt paths, arbitrary client filesystem roots, transport authentication alone, and in-memory fallback results are non-authoritative; use canonical service requests/results plus transport-specific authenticated context.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py, ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, ipfs_accelerate_py/agent_supervisor/__init__.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py", "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v3_conformance.py test/api/test_agent_supervisor_prompt_only_v3_security.py test/api/test_agent_supervisor_prompt_only_v3_compatibility.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_v3_conformance.py test/api/test_agent_supervisor_prompt_only_v3_security.py test/api/test_agent_supervisor_prompt_only_v3_compatibility.py -q"]
- Acceptance: Documented Python import, CLI commands, and MCP/MCP++ schemas exist and invoke a real backend; cold help/import has no integration startup; all trust boundaries and operation results agree; existing expert entrypoints remain usable.
- Gap task: Repair the smallest export, command, console registration, MCP tool registration, UCAN, cold-start, result parity, compatibility, or documentation residual.

## ASE3-G080 Self-hosted verification, rollout, and closeout

- Status: active
- Parent: ASE3-G000
- Parent goal IDs JSON: ["ASE3-G000"]
- Depends on: ASE3-G050, ASE3-G060, ASE3-G070
- Dependencies JSON: ["ASE3-G050", "ASE3-G060", "ASE3-G070"]
- Fib priority: 55
- Track: self-host-rollout
- Priority: P0
- Bundle: agent-supervisor/prompt-self-improvement-v3/self-host-rollout
- Parallel lane: closeout
- Resource class: coordinator
- Goal: Prove the new facade by running one bounded real self-improvement canary, monitoring parallel progress/refill/recovery, and materializing an exact-tree staged rollout with reversible cutover.
- Producing tasks: ASE3-013, ASE3-014
- Evidence: prompt_v3_rollout.SELF_HOSTED_ROLLOUT_REQUIREMENT_ID
- Evidence requirements JSON: ["fresh conformance security chaos and load reports", "real child/taskboard execution proof", "parallel overlap proof", "forced refill and stall recovery proof", "current-tree reachability manifest", "signed promotion or rollback decision"]
- Evidence criteria: The canary begins at one lane, advances to at least two after conflict proof, performs actual isolated-worktree work, survives injected drain and stall cases, remains within budgets, and closes only after a second fresh-tree validation and active rollout decision.
- Evidence source policy: Mock-only E2E, immediate in-memory completion, missing test paths, staged-inactive rollout, historical branch tests, and old task receipts cannot promote; require exact release-candidate execution, immutable event lineage, current-tree validation, sustained observation, and explicit rollout authority.
- Outputs: test/api/test_agent_supervisor_prompt_only_v3_e2e.py, test/api/test_agent_supervisor_prompt_only_v3_chaos.py, test/api/test_agent_supervisor_prompt_only_v3_load.py, data/agent_supervisor/prompt_only_self_improvement_v3/rollout
- Predicted files JSON: ["test/api/test_agent_supervisor_prompt_only_v3_e2e.py", "test/api/test_agent_supervisor_prompt_only_v3_chaos.py", "test/api/test_agent_supervisor_prompt_only_v3_load.py", "data/agent_supervisor/prompt_only_self_improvement_v3/rollout"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v3_e2e.py test/api/test_agent_supervisor_prompt_only_v3_chaos.py test/api/test_agent_supervisor_prompt_only_v3_load.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_v3_e2e.py test/api/test_agent_supervisor_prompt_only_v3_chaos.py test/api/test_agent_supervisor_prompt_only_v3_load.py -q"]
- Acceptance: A real prompt-created canary has complete goal/task/run/effect/event lineage, observed concurrent disjoint workers, serialized conflicts, bounded refill, bounded recovery, reachable accepted commits, exact-tree green gates, a truthful terminal shutdown, and an active signed promotion or explicit rollback receipt.
- Gap task: Repair the smallest E2E, security, chaos, load, current-tree, canary, observation, materialization, rollout, rollback, or closeout residual.
