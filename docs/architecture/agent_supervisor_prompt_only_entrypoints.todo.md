# Agent Supervisor Prompt-Only Entrypoints Task Board

This executable board implements
`AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md`. Durable intent is in
`agent_supervisor_prompt_only_entrypoints.objectives.md`.

The implementation daemon must use task prefix `## ASE-`. A task may run in
parallel only after all `Depends on` tasks complete and the conflict/resource
scheduler admits its predicted files and resource class. Shared exports,
transport conformance, migration, documentation, and rollout are explicit
fan-in tasks.

## ASE-001 Inventory current invocation and runtime-construction friction

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: product-baseline
- Depends on:
- Goal id: ASE-G010
- Outputs: docs/architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md, test/api/test_agent_supervisor_prompt_entrypoint_baseline.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_entrypoint_baseline.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/baseline-inventory
- Parallel lane: baseline-inventory
- Resource class: cpu-small
- Predicted files: docs/architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md, test/api/test_agent_supervisor_prompt_entrypoint_baseline.py
- Interfaces: Installed CLI probe, control capability probe, prompt workflow probe
- Conflict policy: Own only the baseline document and probe test; do not change runtime or public entrypoints.
- Preconditions: Current expert CLI, prompt module and MCP discovery remain runnable.
- Effects: Add executable evidence of required flags, read-only default handlers, prompt handoff gaps, duplicated state defaults and launch surfaces.
- Evidence subset: current invocation inventory, executable capability report
- Completion evidence: 8 focused current-tree probes passed in parallel tranche 1 on 2026-08-03.
- Acceptance: Reproduce the nine required target bindings, mutation bindings, unavailable default prompt handlers, prompt-body bridge gap, start mismatch, low-level flag counts and state-root divergence using current-tree probes rather than prose alone.

## ASE-002 Freeze prompt-only journeys, fixture matrix, and quantitative gates

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: product-baseline
- Depends on:
- Goal id: ASE-G010
- Outputs: test/fixtures/agent_supervisor_prompt_entrypoints/manifest.json, test/api/test_agent_supervisor_prompt_entrypoint_acceptance.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_entrypoint_acceptance.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/baseline-fixtures
- Parallel lane: baseline-fixtures
- Resource class: cpu-small
- Predicted files: test/fixtures/agent_supervisor_prompt_entrypoints/manifest.json, test/api/test_agent_supervisor_prompt_entrypoint_acceptance.py
- Interfaces: Prompt-only journey manifest, rollout metric definitions
- Conflict policy: Own the frozen fixture/metric population; do not implement resolvers or narrow cases later to make rollout pass.
- Preconditions: None.
- Effects: Add clean, dirty, nested, worktree, submodule, ambiguous, degraded and adversarial target definitions plus run/latency/parity/safety thresholds.
- Evidence subset: frozen fixture manifest, metric schema
- Completion evidence: 15 frozen-manifest acceptance checks passed in parallel tranche 1 on 2026-08-03.
- Acceptance: Define CLI, Python, MCP and MCP++ run/steer/status/follow journeys and the full success-rate, deterministic replay, time-to-handle/event, zero-leak, zero-unexpected-effect and zero-duplicate-process gates.

## ASE-003 Define invocation, inference, profile, launch, run, and result contracts

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-contracts
- Depends on: ASE-001, ASE-002
- Goal id: ASE-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, test/api/test_agent_supervisor_entrypoint_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_entrypoint_contracts.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/contracts
- Parallel lane: entrypoint-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, test/api/test_agent_supervisor_entrypoint_contracts.py
- Interfaces: SupervisorInvocationRequest, TargetCandidate, TargetInferenceDecision, TargetResolutionReceipt, ResolvedSupervisorProfile, LaunchPlan, RunHandle, SupervisorInvocationResult
- Conflict policy: Own standalone provider-free contracts only; defer package exports, resolvers, services and transports to later tasks.
- Preconditions: Baseline field and journey populations are frozen.
- Effects: Add versioned canonical schemas, strict bounds, identity checks, prompt-body exclusion and exact serialization.
- Evidence subset: schema round trips, invalid-state tests, canonical identities
- Completion evidence: 20 focused contract tests and 12 package/provider compatibility tests passed after independent security repair; Ruff and diff checks passed on 2026-08-03.
- Acceptance: Contracts distinguish hints from authority, record every selected/defaulted/ambiguous/denied field and alternative, bind all semantic roots, reject unknown/over-bound/secret-bearing input, and contain no raw prompt body in durable records.

## ASE-004 Establish the highest-layer entrypoints package and dependency boundary

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-contracts
- Depends on: ASE-003
- Goal id: ASE-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, ipfs_accelerate_py/agent_supervisor/entrypoints/README.md, docs/architecture/agent_supervisor/PACKAGE_MAP.md, test/api/test_agent_supervisor_entrypoint_package.py
- Validation: python -m pytest test/api/test_agent_supervisor_entrypoint_package.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/package
- Parallel lane: entrypoint-package
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, ipfs_accelerate_py/agent_supervisor/entrypoints/README.md, docs/architecture/agent_supervisor/PACKAGE_MAP.md, test/api/test_agent_supervisor_entrypoint_package.py
- Interfaces: agent_supervisor.entrypoints package boundary
- Conflict policy: This is the sole package-layout owner; exports only reviewed contracts and lazy facades and does not modify resolver/service implementations.
- Preconditions: Entrypoint contracts exist.
- Effects: Add a highest-layer composition package and update the documented acyclic package DAG.
- Evidence subset: cold import, import graph, reviewed export inventory
- Completion evidence: 5 focused package/cold-import/AST-boundary tests passed with clean Ruff and diff checks on 2026-08-03.
- Acceptance: Lower domain packages never import `entrypoints`; importing the package resolves no provider/service, scans no repository, opens no database and starts no process; stable contract exports preserve object identity.

## ASE-005 Implement repository, checkout, scope, and dirty-tree resolution

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: repository-inference
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G031
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py, test/api/test_agent_supervisor_target_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_target_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/repository-resolver
- Parallel lane: repository-resolver
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py, test/api/test_agent_supervisor_target_resolver.py
- Interfaces: RepositoryTargetResolver, repository TargetInferenceDecision
- Conflict policy: Reuse repository forest, checkout authority and snapshot APIs; do not implement state/objective/provider resolution or parse authority from prompt text.
- Preconditions: Entrypoint contracts and package boundary are available.
- Effects: Select a unique allowlisted repository/scope and bind repository ID, checkout, HEAD/dirty overlay and submodule identities.
- Evidence subset: root and scope decisions, snapshot CID, topology alternatives
- Acceptance: Clean, staged, modified, deleted, admitted-untracked, worktree and submodule fixtures resolve deterministically; symlink, parent traversal and nested-repository ambiguity fail closed or return a typed preview ambiguity without widening roots.

## ASE-006 Implement platform state, namespace, and active-run resolution

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-inference
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G032
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py, test/api/test_agent_supervisor_state_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_state_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/state-resolver
- Parallel lane: state-resolver
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py, test/api/test_agent_supervisor_state_resolver.py
- Interfaces: StateRootResolver, RunCandidateResolver
- Conflict policy: Own state/run location inference only; do not implement registry persistence, process adoption or objective selection.
- Preconditions: Entrypoint contracts and package boundary are available.
- Effects: Resolve a platform state root and collision-resistant repository/run namespace and classify existing run candidates.
- Evidence subset: state decision, namespace identity, active-run alternatives
- Acceptance: State defaults outside the source checkout, remains stable for the same repository identity, separates forks/worktrees where required, adopts only one exact compatible candidate and reports multiple/incompatible/stale candidates without guessing.

## ASE-007 Implement objective, plan, task-source, and output resolution

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-inference
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G032
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py, test/api/test_agent_supervisor_objective_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_objective_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/objective-resolver
- Parallel lane: objective-resolver
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py, test/api/test_agent_supervisor_objective_resolver.py
- Interfaces: ObjectiveResolver, TaskSourceResolver, OutputPolicyResolver
- Conflict policy: Reuse objective/task-source parsers and integrity APIs; do not hand-edit projections, execute tasks or widen repository output paths.
- Preconditions: Entrypoint contracts and package boundary are available.
- Effects: Select a run-bound or unique compatible objective/task source, otherwise construct decisions for a new prompt objective and state-root projections.
- Evidence subset: objective alternatives, task-source integrity/revision, output decision
- Acceptance: Exact run bindings win; absent intent creates a content-addressed objective; multiple plausible objectives/boards are explicit; DuckDB plus Markdown mirror is selected when available with typed Markdown degradation, and outputs do not dirty the repository by default.

## ASE-008 Implement principal, policy, local authority, and effect-ceiling resolution

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority-inference
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G033
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py, test/api/test_agent_supervisor_authority_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_authority_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/authority-resolver
- Parallel lane: authority-resolver
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py, test/api/test_agent_supervisor_authority_resolver.py
- Interfaces: PrincipalBinding, PolicySelection, EffectCeiling, LocalWorktreeAuthority
- Conflict policy: Own authority resolution/adaptation; do not change the canonical control authorization verifier or UCAN transport implementation.
- Preconditions: Entrypoint contracts and package boundary are available.
- Effects: Bind authenticated transport/local principal, select trusted policies, support explicitly installed local worktree signing authority and derive exact maximum effects.
- Evidence subset: principal source, profile/policy roots, effect ceiling, authority decision reference
- Acceptance: Prompt/repository text and mere credential presence cannot create caller or authority; lower-precedence sources only narrow; local worktree mode permits isolated edits/tests after explicit setup but denies current-checkout rewrite, merge, push, deploy, secrets, arbitrary network and destructive cleanup.

## ASE-009 Implement provider, resource, lane, validation, and topology resolution

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-inference
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G034
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py, test/api/test_agent_supervisor_capability_resolver.py, test/api/test_agent_supervisor_default_provider_route.py
- Validation: python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_capability_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/capability-resolver
- Parallel lane: capability-resolver
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py, test/api/test_agent_supervisor_capability_resolver.py, test/api/test_agent_supervisor_default_provider_route.py
- Interfaces: CapabilityResolver, ProviderSelection, ProviderFallbackReceipt, ResourceEnvelope, ValidationProfile, DeploymentTopology
- Conflict policy: Reuse provider capability/usage gateways, ResourceScheduler and validation policy; do not execute providers/tests or compile final launch argv.
- Preconditions: Entrypoint contracts and package boundary are available.
- Existing compatibility evidence: The legacy `auto` command route prefers authenticated Grok, falls back to Codex during preflight or after a nonzero Grok exit, preserves forced-Grok fail-closed behavior, and passed 9 focused route/runner tests on 2026-08-03; this does not replace the typed resolver and receipt work in this task.
- Effects: Resolve Grok as the preferred healthy permitted implementation provider, Codex as its bounded typed fallback, resource/lane ceilings, structured validation candidates and local/distributed topology.
- Evidence subset: capability reports, selection reasons, resource sample, validation argv policy
- Acceptance: Selection is deterministic under frozen health/budget evidence; healthy policy-allowed Grok wins by default; Codex fallback records an unavailable/quota/capacity/pre-effect-failure reason and cannot self-satisfy an independent review; prompt text cannot choose a provider; optional degradation remains explicit; lanes come from safe ready width/resources rather than labels.

## ASE-010 Compose deterministic profile precedence and complete target resolution

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: target-inference
- Depends on: ASE-005, ASE-006, ASE-007, ASE-008, ASE-009
- Goal id: ASE-G034
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py, test/api/test_agent_supervisor_profile_resolver.py
- Validation: python -m pytest test/api/test_agent_supervisor_profile_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/profile-resolver
- Parallel lane: profile-resolver
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py, test/api/test_agent_supervisor_profile_resolver.py
- Interfaces: SupervisorProfileResolver, TargetResolutionReceipt builder
- Conflict policy: Integrate leaf resolver outputs only; do not modify their implementations, runtime services, public transports or low-level control contracts.
- Preconditions: Every leaf resolver emits canonical decisions over the same invocation context.
- Effects: Apply strict precedence and trust ceilings, detect cross-field inconsistencies and produce one complete resolved profile and receipt.
- Evidence subset: full decision population, precedence trace, alternatives, profile CID
- Acceptance: Canonical request disables inference; otherwise explicit hints, run bindings, authenticated/server policy, signed profiles, reviewed repository hints, discovery and conservative defaults merge reproducibly without lower sources widening allowlists/authority/effects; material ambiguity blocks effects but preserves safe preview.

## ASE-011 Add body-free inference explanation and reusable goal/task/profile lint

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: inference-assurance
- Depends on: ASE-010
- Goal id: ASE-G035
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_explain.py, ipfs_accelerate_py/agent_supervisor/entrypoints/plan_lint.py, test/api/test_agent_supervisor_inference_explain.py, test/api/test_agent_supervisor_plan_lint.py
- Validation: python -m pytest test/api/test_agent_supervisor_inference_explain.py test/api/test_agent_supervisor_plan_lint.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/inference-assurance
- Parallel lane: inference-assurance
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_explain.py, ipfs_accelerate_py/agent_supervisor/entrypoints/plan_lint.py, test/api/test_agent_supervisor_inference_explain.py, test/api/test_agent_supervisor_plan_lint.py
- Interfaces: render_target_resolution, lint_supervisor_plan
- Conflict policy: Consume resolver/objective/task/profile contracts read-only; do not mutate plans or duplicate canonical admission.
- Preconditions: Full target resolution exists.
- Effects: Add bounded human/JSON provenance explanations and read-only checks for goal hierarchy, task cycles/dependencies/metadata, structured validation, predicted conflicts and profile completeness.
- Evidence subset: stable explanation, lint findings, leak scan
- Acceptance: Every inferred/defaulted/ambiguous/denied field has an evidence-backed explanation; lint catches duplicate/unknown/cyclic/missing/unsafe/conflicting plan state; no prompt/source/credential body appears in output or error paths.

## ASE-012 Implement transient and capability-protected prompt-body brokering

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: run-state
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G041
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py, test/api/test_agent_supervisor_prompt_broker.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_broker.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/prompt-broker
- Parallel lane: prompt-broker
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py, test/api/test_agent_supervisor_prompt_broker.py
- Interfaces: PromptBodyBroker, PromptCapability, PromptReference
- Conflict policy: Own prompt body lifecycle only; do not change canonical prompt contracts, planners, logging frameworks or transports.
- Preconditions: Invocation contracts define body-free durable prompt references.
- Effects: Carry exact prompt text from CLI/Python/MCP intake to bounded planner use with expiry, zeroization/deletion and optional encrypted artifact continuation.
- Evidence subset: exact retrieval, capability denial, expiry, leak scan
- Acceptance: The planner receives exact bytes during the authorized window, routine requests/results/events/logs/argv/environment/state contain only CID/reference, cross-run access fails, expiry/restart behavior is explicit and secrets are absent from inspected durable surfaces.

## ASE-013 Implement the durable run registry, handle reconstruction, and CAS

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: run-state
- Depends on: ASE-003, ASE-004, ASE-006
- Goal id: ASE-G041
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, test/api/test_agent_supervisor_run_registry.py
- Validation: python -m pytest test/api/test_agent_supervisor_run_registry.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/run-registry
- Parallel lane: run-registry
- Resource class: io-artifact
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, test/api/test_agent_supervisor_run_registry.py
- Interfaces: RunRegistry, RunHandle reconstruction, run revision CAS, current-run selection
- Conflict policy: Own durable high-level run records; reuse event/artifact stores and do not duplicate task-source, process or low-level control state.
- Preconditions: Run contracts and state namespace decisions exist.
- Effects: Persist immutable run roots plus mutable CAS status/cursors, support bounded lookup/list/adopt/reconstruct and repair partial registry state.
- Evidence subset: transaction receipts, exact adoption, restart reconstruction, concurrent CAS
- Acceptance: Restart reconstructs a complete handle; unique compatible run selection is deterministic; multiple/incompatible runs are explicit; conflicting revision updates cannot both win; corruption quarantines instead of yielding a canonical-looking handle.

## ASE-014 Build the standard supervisor runtime factory with real handlers

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-composition
- Depends on: ASE-005, ASE-006, ASE-007, ASE-008, ASE-009, ASE-012, ASE-013
- Goal id: ASE-G042
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, test/api/test_agent_supervisor_runtime_factory.py
- Validation: python -m pytest test/api/test_agent_supervisor_runtime_factory.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/runtime-factory
- Parallel lane: runtime-factory
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, test/api/test_agent_supervisor_runtime_factory.py
- Interfaces: StandardSupervisorRuntimeFactory, StandardSupervisorRuntime
- Conflict policy: Compose existing domain services through handlers; do not move policy into adapters, replace `SupervisorControlService`, or reimplement scanner/planner/task-source/lifecycle/rescue logic.
- Preconditions: Leaf resolution, prompt broker and run registry contracts are available.
- Effects: Construct stores and install real prompt, objective, materialization, lifecycle, validation, retry, recovery, rescue, status and event handlers within explicit allowlists.
- Evidence subset: capability report, handler identity map, real handler fixtures
- Acceptance: A supported configured runtime advertises and executes live prompt workflow/status/lifecycle operations instead of returning unavailable; restricted read-only construction remains possible; import/discovery stays side-effect free and handlers preserve canonical control authorization.

## ASE-015 Implement the resumable resolve-to-run/adopt intent saga

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-composition
- Depends on: ASE-010, ASE-014, ASE-042
- Goal id: ASE-G042
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_intent_service.py
- Validation: python -m pytest test/api/test_agent_supervisor_intent_service.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/intent-service
- Parallel lane: intent-service
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, test/api/test_agent_supervisor_intent_service.py
- Interfaces: SupervisorIntentService, preview, run, resume
- Conflict policy: Own high-level saga transitions; call the resolver/runtime/control/prompt services directly and do not encode effects as shell commands or collapse preview/authorization/materialize/start boundaries.
- Preconditions: Complete target resolution, standard runtime factory and admitted canonical plan materializer exist.
- Effects: Persist received/resolving/previewing/authorizing/materializing/starting/adopting/running intent and effect checkpoints and return a durable handle or one continuation.
- Evidence subset: saga journal, control receipts, expected/observed effects, continuation
- Acceptance: Exact replay invokes no duplicate provider/write/process effect; crash at every boundary resumes; stale roots re-resolve or conflict; rejected/ambiguous/denied/partial states remain typed; a compatible healthy process is adopted before launch and success requires sustained health.

## ASE-042 Materialize admitted plans into canonical DuckDB and bounded Markdown projections

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-composition
- Depends on: ASE-007, ASE-010, ASE-014
- Goal id: ASE-G042
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, test/api/test_agent_supervisor_plan_materializer.py
- Validation: python -m pytest test/api/test_agent_supervisor_plan_materializer.py test/api/test_agent_supervisor_duckdb_task_source.py test/api/test_agent_supervisor_prompt_plan_admission.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/plan-materializer
- Parallel lane: plan-materializer
- Resource class: io-database
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, test/api/test_agent_supervisor_plan_materializer.py
- Interfaces: AdmittedPlanMaterializer, CanonicalProjectionReceipt, MarkdownEpochPolicy
- Conflict policy: Own the high-level admitted-plan projection adapter; call existing prompt admission and DuckDB/Markdown task-source materializers and never hand-edit canonical projections.
- Preconditions: Objective/task-source selection, complete target profile and standard runtime stores exist.
- Effects: Convert an admitted PromptGoalGraph/formal-plan input into one root-bound DuckDBTaskSource; generate canonical Markdown only within its admitted 24-task bound or split it into root-linked epochs; verify structured argv, task/goal CIDs, repository tree identity and exact replay before publication.
- Evidence subset: admission receipt, DuckDB projection/integrity receipt, Markdown epoch roots, replay/no-op receipt
- Acceptance: Prompt-only launch automatically obtains a canonical DuckDB task source without a task-source flag; projections are generated rather than hand-authored; more than 24 tasks never overflow canonical Markdown; identical replay is a no-op and root/schema/population drift fails closed.

## ASE-016 Compile resolved profiles into lifecycle and daemon/runner configuration

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: launch-coordination
- Depends on: ASE-010, ASE-014
- Goal id: ASE-G043
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/launch_profile.py, test/api/test_agent_supervisor_launch_profile.py
- Validation: python -m pytest test/api/test_agent_supervisor_launch_profile.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/launch-profile
- Parallel lane: launch-profile
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/launch_profile.py, test/api/test_agent_supervisor_launch_profile.py
- Interfaces: LaunchProfileCompiler, runner and daemon configuration adapters
- Conflict policy: Own profile projection/adapters; do not add another subprocess scheduler or parse arbitrary repository command strings.
- Preconditions: Resolved profile and runtime handler construction exist.
- Effects: Project every behavioral setting into immutable lifecycle profile and existing multi-supervisor/implementation runner configuration with bounded argv/environment and exact health paths; expose inferred `task-source-kind` plus expected plan/repository roots through the managed supervisor instead of requiring callers to bypass it for DuckDB.
- Evidence subset: field-coverage matrix, configuration/profile CIDs, argv/env bounds
- Acceptance: No normal caller needs the 50/133 daemon flags; equivalent explicit and inferred configurations produce the same profile; the managed supervisor launches verified Markdown or DuckDB task sources with expected-root checks; all current behavioral flags are covered or explicitly deprecated; credentials/unknown environment/unsafe command text cannot enter launch.

## ASE-017 Add topology-aware DuckDB coordination-shard ownership and fencing

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: launch-coordination
- Depends on: ASE-008, ASE-013, ASE-014
- Goal id: ASE-G043
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/lease_backend.py, test/api/test_agent_supervisor_lease_backends.py
- Validation: python -m pytest test/api/test_agent_supervisor_lease_backends.py test/api/test_agent_supervisor_lease_coordination.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/lease-backends
- Parallel lane: lease-backends
- Resource class: network-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/lease_backend.py, test/api/test_agent_supervisor_lease_backends.py
- Interfaces: DuckDBCoordinationShard, LeaseCoordinator protocol adapter, CoordinationShardMap, AuthenticatedShardOwner
- Conflict policy: Adapt the existing DuckDB LeaseCoordinator and distributed lane contracts; do not create a second lease database, let multiple hosts write one DuckDB file, or let IPFS/IPLD replicas grant claims.
- Preconditions: Authority, run registry and runtime topology are known.
- Effects: Assign repositories/runs/tasks deterministically to single-writer DuckDB shards; use bounded transactions and the existing process-shared lock locally; route remote MCP++ mutations to the authenticated owner; compile lease, logical epoch, fencing, takeover, degraded and owner-handoff policy.
- Evidence subset: backend conformance, lease/fence receipts, topology decision
- Acceptance: Existing DuckDB coordination compatibility remains exact; independent shards can progress concurrently; multi-host profiles cannot share-write a database file or infer an owner from IPFS/IPNS; stale owners/replicas cannot publish mutable effects; owner ambiguity fails closed; pure isolated idempotent degraded execution is explicit.

## ASE-035 Implement immutable Parquet/IPLD/IPFS coordination epochs and replicas

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: coordination-replication
- Depends on: ASE-012, ASE-013, ASE-014, ASE-017, ASE-037, ASE-038, ASE-039, ASE-040
- Goal id: ASE-G044
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_replication.py, test/api/test_agent_supervisor_coordination_replication.py
- Validation: python -m pytest test/api/test_agent_supervisor_coordination_replication.py test/api/test_agent_supervisor_coordination_epoch.py test/api/test_agent_supervisor_coordination_authority.py test/api/test_agent_supervisor_replication_policy.py test/api/test_agent_supervisor_verified_ipld_backend.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/coordination-replication
- Parallel lane: coordination-replication
- Resource class: io-network
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_replication.py, test/api/test_agent_supervisor_coordination_replication.py
- Interfaces: CoordinationEpoch, CoordinationHead, ParquetEpochExporter, IPLDCoordinationManifest, IPFSCoordinationReplica
- Conflict policy: Own immutable export/import and replication fan-in only; consume the committed-cursor, verified-backend, signing/head and disclosure-policy contracts from their producer tasks; never write the active DuckDB shard from a replica or treat IPNS/pubsub/Hugging Face as lease authority.
- Preconditions: Prompt broker, run registry, standard runtime, DuckDB owner/fence, committed epoch cursor, strict CID backend, cryptographic authority and disclosure policy are complete.
- Effects: Snapshot one committed logical epoch; export strict bounded Parquet fragments; build linked canonical DAG-JSON/IPLD manifests and capability-gated CAR bundles; publish only policy-cleared signed discovery heads; reconstruct verified read-only DuckDB projections; carry immutable CID inputs/results for remote lanes.
- Evidence subset: logical row-set parity, epoch/head CIDs, previous-epoch chain, verified backend receipt, disclosure scan, stale/tampered replica denials, partition quarantine
- Acceptance: DuckDB to Parquet to IPLD/IPFS to DuckDB logical tables round trip exactly and idempotently; missing/tampered/reordered/disallowed epochs fail closed; byte determinism is claimed only under a pinned writer profile; stale heads and replicas cannot claim or authorize effects; partitioned results remain immutable but only the current DuckDB owner may accept one under its authenticated lease and fence.

## ASE-036 Implement the typed Grok-first, Codex-implementation-fallback production route

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-runtime
- Depends on: ASE-009, ASE-014, ASE-015
- Goal id: ASE-G045
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py, test/api/test_agent_supervisor_typed_provider_fallback.py
- Validation: python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_typed_provider_fallback.py test/api/test_agent_supervisor_production_provider_route.py test/api/test_agent_supervisor_contract_packet_provider_router.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/provider-runtime
- Parallel lane: provider-runtime
- Resource class: provider-io
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py, test/api/test_agent_supervisor_typed_provider_fallback.py
- Interfaces: TypedImplementationProviderRoute, CodexImplementationFallback, ProviderFallbackReceipt, IndependentReviewContinuation
- Conflict policy: Sole owner of production implementation-provider dispatch and fallback receipt semantics; preserve the packet admission/effect boundary and do not treat Codex fallback implementation as Codex review.
- Preconditions: Provider resolution, runtime factory and intent/effect saga exist.
- Effects: Prefer admitted Grok implementation; before any accepted effect, commit a typed unavailable/quota/capacity/pre-effect-failure receipt and permit at most one budget-bounded Codex implementation fallback; carry distinct attempt/process/authorization identity into a separate review continuation.
- Evidence subset: preferred/fallback route matrix, provider attempts, fallback receipt, effect boundary, independent review identity
- Acceptance: Healthy allowed Grok always wins; every Codex implementation fallback is reproducible, pre-effect, once-only and scope/budget preserving; a Codex fallback attempt cannot self-review; post-effect replay, provider identity collision, reason forgery and prompt-selected routing fail closed.

## ASE-037 Add a transactionally committed DuckDB epoch cursor and frozen logical schema

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: coordination-epoch
- Depends on: ASE-013, ASE-017
- Goal id: ASE-G044
- Outputs: ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py, ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_epoch.py, test/api/test_agent_supervisor_coordination_epoch.py
- Validation: python -m pytest test/api/test_agent_supervisor_coordination_epoch.py test/api/test_agent_supervisor_lease_coordination.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/coordination-epoch
- Parallel lane: coordination-epoch
- Resource class: io-database
- Predicted files: ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py, ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_epoch.py, test/api/test_agent_supervisor_coordination_epoch.py
- Interfaces: CoordinationCommitSequence, CoordinationLogicalSchema, CoordinationSnapshot
- Conflict policy: Own the authoritative shard commit-sequence/schema change; preserve existing lease compatibility and serialize changes to lease_coordination.py before replication fan-in.
- Preconditions: Run registry and topology-aware DuckDB ownership/fencing exist.
- Effects: Advance a monotonic shard commit sequence in the same transaction/lock as each accepted mutation; freeze epoch tables, primary/sort keys and null/time/binary/JSON normalization; bind external broker/event/CAS cursors and roots to one exportable snapshot.
- Evidence subset: atomic mutation/cursor receipt, crash matrix, snapshot roots, row-set digests, referential checks
- Acceptance: No exporter can observe a half epoch or combine unrelated latest stores; exact committed sequences replay deterministically; aborted mutations do not advance; schema/version drift, missing roots and network/shared DuckDB paths fail closed.

## ASE-038 Harden a canonical CID/IPLD/IPFS replication adapter

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: coordination-replication
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G044
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py, ipfs_accelerate_py/ipfs_backend_router.py, test/api/test_agent_supervisor_verified_ipld_backend.py
- Validation: python -m pytest test/api/test_agent_supervisor_verified_ipld_backend.py test/test_ipfs_backend_router.py test/api/test_agent_supervisor_multiformats_identity.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/verified-ipld-backend
- Parallel lane: verified-ipld-backend
- Resource class: io-network
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py, ipfs_accelerate_py/ipfs_backend_router.py, test/api/test_agent_supervisor_verified_ipld_backend.py
- Interfaces: VerifiedIPLDBackend, BackendCapabilityReceipt, IdentityLink
- Conflict policy: Sole hardening owner for replication-facing backend semantics; retain compatibility adapters but never admit a synthetic/cache identifier as a CID or assume codec/CAR support.
- Preconditions: Entrypoint contracts and package boundary exist.
- Effects: Compute expected raw/DAG-JSON CIDv1 locally; validate backend CID/codec and rehash fetched bytes; classify Hugging Face as cache-only until conformant; capability-gate CAR; bridge runtime-CAS and MCP++ hashes through explicit identity links.
- Evidence subset: multiformats vectors, put/get byte verification, codec substitution denial, backend capability matrix
- Acceptance: Only strict CIDv1 objects enter coordination manifests; fake/truncated/mismatched CIDs and unsupported codecs/CAR fail closed; ipfs_kit_py, Kubo and cache roles are accurately reported and degradation is explicit.

## ASE-039 Add cryptographic shard-result and signed-head authority

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: coordination-authority
- Depends on: ASE-008, ASE-014, ASE-017, ASE-037
- Goal id: ASE-G044
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_authority.py, test/api/test_agent_supervisor_coordination_authority.py
- Validation: python -m pytest test/api/test_agent_supervisor_coordination_authority.py test/api/test_agent_supervisor_lease_coordination.py test/api/test_mcp_server_mcplusplus_ucan.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/coordination-authority
- Parallel lane: coordination-authority
- Resource class: crypto-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_authority.py, test/api/test_agent_supervisor_coordination_authority.py
- Interfaces: CoordinationResultEnvelope, CoordinationHead, CoordinationSigner, CoordinationVerifier
- Conflict policy: Own the transport-neutral signature/head contract; reuse authority and UCAN verification primitives without making a later transport facade a prerequisite for core coordination.
- Preconditions: Authority resolver, runtime, shard owner/fence and committed cursor exist.
- Effects: Bind issuer DID/key/algorithm, audience, resource, ability, shard/profile/epoch/fence, nonce, expiry and revocation to remote results and monotonic ancestry-linked discovery heads; detect equivocation and governed key rotation.
- Evidence subset: signature vectors, capability binding, head ancestry, revocation/rotation/equivocation receipts
- Acceptance: Content-bound claimant strings alone grant nothing; forged/expired/revoked/cross-shard/cross-profile results fail; rollback, same-sequence conflicting heads and unauthorized key rotation fail closed; signatures never convert discovery state into lease authority.

## ASE-040 Enforce replication disclosure, encryption, retention, and leak policy

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: coordination-privacy
- Depends on: ASE-008, ASE-012, ASE-013, ASE-037
- Goal id: ASE-G044
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/replication_policy.py, test/api/test_agent_supervisor_replication_policy.py
- Validation: python -m pytest test/api/test_agent_supervisor_replication_policy.py test/api/test_agent_supervisor_prompt_only_security.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/replication-policy
- Parallel lane: coordination-privacy
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/replication_policy.py, test/api/test_agent_supervisor_replication_policy.py
- Interfaces: ReplicationDisclosurePolicy, EpochFieldClassification, EncryptedEpochEnvelope, RetentionPolicy
- Conflict policy: Sole owner of export classification/redaction/encryption and publication gate; never persist bearer credentials or silently downgrade private replication to public.
- Preconditions: Authority, prompt/run stores and frozen epoch schema exist.
- Effects: Classify and allowlist export fields; redact sensitive paths/task/provider/receipt material; support recipient-bound private encrypted epochs; canary-scan encoded objects; bind retention/unpin rules and deny head publication on any degraded export.
- Evidence subset: field-classification coverage, public/private fixtures, ciphertext binding, canary leak scan, retention receipt
- Acceptance: Public epochs contain no unclassified/disallowed data or raw prompt/secrets; private epochs disclose only to authorized recipients; missing keys, fallback JSONL, exporter errors, leak findings and policy uncertainty prevent pin/head publication.

## ASE-018 Publish the lazy Python `Supervisor` convenience API

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: python-entrypoint
- Depends on: ASE-015, ASE-016, ASE-017, ASE-035, ASE-036
- Goal id: ASE-G051
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/python_api.py, test/api/test_agent_supervisor_prompt_only_python.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_python.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/python
- Parallel lane: python-surface
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/python_api.py, test/api/test_agent_supervisor_prompt_only_python.py
- Interfaces: Supervisor.open, preview, run, resume, steer, status, follow, explain, doctor
- Conflict policy: Thin typed facade over `SupervisorIntentService`; do not add transport-specific policy or mutate package exports until the conformance fan-in.
- Preconditions: Intent saga, typed Grok-first/Codex-fallback provider route, DuckDB launch/lease profile compilation and immutable coordination replication are complete.
- Effects: Add local inferred and explicit embedded Python construction plus typed synchronous/async run and event interfaces.
- Evidence subset: cold import, explicit/inferred equivalence, run-handle behavior
- Acceptance: `Supervisor.open().run(prompt)` produces a real resumable handle under a trusted local profile; explicit embedders provide allowlists/principal/stores; import and method discovery have no runtime side effects.

## ASE-019 Add the prompt-first product CLI and compatibility aliases

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: cli-entrypoint
- Depends on: ASE-015, ASE-016, ASE-017, ASE-035, ASE-036
- Goal id: ASE-G052
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/cli.py, pyproject.toml, setup.py, test/api/test_agent_supervisor_prompt_only_cli.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_cli.py test/api/test_agent_supervisor_prompt_cli.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/cli
- Parallel lane: cli-surface
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/cli.py, pyproject.toml, setup.py, test/api/test_agent_supervisor_prompt_only_cli.py
- Interfaces: `ipfs-accelerate supervisor`, `ipfs-accelerate agent` aliases, `ipfs-accelerate-supervisor`
- Conflict policy: Own CLI/parser/packaging integration; call the intent service and preserve all existing expert command schemas, names and exit codes.
- Preconditions: Intent saga, typed Grok-first/Codex-fallback provider route, DuckDB launch/lease profile compilation and immutable coordination replication are complete.
- Effects: Add run/preview/resume/steer/status/follow/explain/doctor with positional, file, stdin and reference prompt sources plus concise human/canonical JSON output.
- Evidence subset: installed subprocess results, help/discovery probe, old-command compatibility, leak scan
- Acceptance: From a supported checkout `ipfs-accelerate supervisor run "prompt"` starts useful isolated work without low-level flags; status/follow/steer infer a unique run; file/stdin protect sensitive prompts; request-json remains inference-free; help/import start nothing; expert CLI behavior remains canonical.

## ASE-020 Add lazy MCP prompt-first supervisor tools

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp-entrypoint
- Depends on: ASE-015, ASE-016, ASE-017, ASE-035, ASE-036
- Goal id: ASE-G053
- Outputs: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoint_tools.py, test/api/test_agent_supervisor_prompt_only_mcp.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_mcp.py test/api/test_agent_supervisor_native_mcp_discovery.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/mcp
- Parallel lane: mcp-surface
- Resource class: network-small
- Predicted files: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoint_tools.py, test/api/test_agent_supervisor_prompt_only_mcp.py
- Interfaces: agent_supervisor_run, preview, resume, steer, status, follow, explain, doctor
- Conflict policy: Own new high-level MCP adapters only; retain generated low-level operation tools and require server-owned service/allowlist/principal configuration.
- Preconditions: Intent saga, typed Grok-first/Codex-fallback provider route, DuckDB launch/lease profile compilation and immutable coordination replication are complete.
- Effects: Register closed minimal request/result schemas and dispatch to the shared intent service.
- Evidence subset: tool schemas, lazy registration, configured live invocation, low-level compatibility
- Acceptance: An MCP prompt can run/steer only a uniquely selected server-allowlisted target; request text cannot configure roots/caller/policy/effects; prompt bodies avoid routine logs; tools return canonical shared-service records and discovery starts no runtime.

## ASE-021 Add MCP++ supervisor IDL and UCAN invocation binding

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcplusplus-entrypoint
- Depends on: ASE-008, ASE-017, ASE-020, ASE-035
- Goal id: ASE-G054
- Outputs: ipfs_accelerate_py/mcp_server/tools/mcplusplus/agent_supervisor_tools.py, test/api/test_agent_supervisor_prompt_only_mcplusplus.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_mcplusplus.py test/api/test_mcp_server_mcplusplus_ucan.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/mcplusplus
- Parallel lane: mcplusplus-surface
- Resource class: network-small
- Predicted files: ipfs_accelerate_py/mcp_server/tools/mcplusplus/agent_supervisor_tools.py, test/api/test_agent_supervisor_prompt_only_mcplusplus.py
- Interfaces: MCP++ supervisor run/steer/status IDL, UCAN capability adapter
- Conflict policy: Own supervisor-specific MCP++ facade and UCAN mapping; do not alter generic MCP++ queue/workflow semantics or treat transport permission as inner authorization.
- Preconditions: Authority resolver, DuckDB shard owner, immutable coordination replication and MCP convenience service exist.
- Effects: Verify issuer/audience/signature/expiry/revocation/attenuation/target/operation/effects and bind a proof reference to a separate canonical mutation decision.
- Evidence subset: IDL parity, token matrix, inner/outer authorization receipts
- Acceptance: Valid attenuated UCAN permits only its target/operation/effects; forged/stale/revoked/overbroad/cross-run tokens fail; bearer material is not persisted; an allowed tool call still cannot mutate without current inner root/effect/lease/fence authorization.

## ASE-022 Prove Python, CLI, MCP, and MCP++ canonical conformance

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: transport-conformance
- Depends on: ASE-018, ASE-019, ASE-020, ASE-021
- Goal id: ASE-G055
- Outputs: test/api/test_agent_supervisor_prompt_only_conformance.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_conformance.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/conformance
- Parallel lane: transport-conformance
- Resource class: cpu-medium
- Predicted files: test/api/test_agent_supervisor_prompt_only_conformance.py
- Interfaces: CrossTransportPromptEntrypointConformance
- Conflict policy: Own the joined fixture/conformance report; do not change adapters while measuring them and do not normalize away semantic mismatches.
- Preconditions: All four public adapters are complete.
- Effects: Execute the same success/rejection/ambiguity/denial/stale/partial/cancel fixtures and compare canonical records/effects/cursors.
- Evidence subset: joined request/result bytes, discovery effects, mismatch report
- Acceptance: All transports call one service and agree canonically except declared transport metadata; no adapter adds authority or loses required identity; cold discovery parity passes; legacy low-level tools/commands remain supported.

## ASE-023 Define semantic steering contracts and closed intent classification

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: steering-contracts
- Depends on: ASE-003, ASE-004
- Goal id: ASE-G061
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py, test/api/test_agent_supervisor_steering_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_steering_contracts.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/steering-contracts
- Parallel lane: steering-contracts
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py, test/api/test_agent_supervisor_steering_contracts.py
- Interfaces: SteeringRequest, SteeringEvent, SteeringIntentKind, SteeringResult
- Conflict policy: Own provider-free steering records/classification policy; defer task-source/run mutation to ASE-024 and concurrency to ASE-025.
- Preconditions: Entrypoint contracts and package boundary exist.
- Effects: Add closed append/answer/narrow/reprioritize/replan/pause/resume/cancel/status vocabulary, bounds, run/revision binding, transient instruction reference and typed questions.
- Evidence subset: canonical classification fixtures, ambiguous material action cases
- Acceptance: Deterministic rules classify supported instructions; optional model output remains proposal tier; materially different interpretations produce one bounded question; prompt text cannot directly select authority/effects or alter state.

## ASE-024 Apply admitted steering deltas to live run and task-source revisions

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: steering-runtime
- Depends on: ASE-013, ASE-014, ASE-023
- Goal id: ASE-G062
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_runtime.py, test/api/test_agent_supervisor_steering_runtime.py
- Validation: python -m pytest test/api/test_agent_supervisor_steering_runtime.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/steering-runtime
- Parallel lane: steering-runtime
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_runtime.py, test/api/test_agent_supervisor_steering_runtime.py
- Interfaces: SteeringRuntime, PlanDelta admission/apply adapter
- Conflict policy: Reuse the existing create/steer plan design, formal admission and task-source CAS; never edit completed/accepted/claimed CID-bearing specs or interpret model text as a mutation.
- Preconditions: Run registry, standard runtime and steering contracts exist.
- Effects: Snapshot exact run/tree/plan/task-source/attempt/evidence/policy state, compile/admit a delta, publish one child revision and notify lanes.
- Evidence subset: delta/admission receipt, projection transaction, run revision, event cursor
- Acceptance: Unstarted work changes only through a valid child revision; claimed work gets deferred/successor actions; completed/accepted history is immutable; dependencies/conflicts/resources remain valid; Markdown/DuckDB populations and run registry agree exactly.

## ASE-025 Enforce concurrent steering CAS, leases, fencing, and replay

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: steering-concurrency
- Depends on: ASE-017, ASE-024
- Goal id: ASE-G063
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_concurrency.py, test/api/test_agent_supervisor_steering_concurrency.py
- Validation: python -m pytest test/api/test_agent_supervisor_steering_concurrency.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/steering-concurrency
- Parallel lane: steering-concurrency
- Resource class: cpu-medium
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_concurrency.py, test/api/test_agent_supervisor_steering_concurrency.py
- Interfaces: SteeringTransactionCoordinator, steering idempotency/fencing
- Conflict policy: Own steering transaction coordination and interleaving tests; do not weaken run/task-source CAS or use shared file databases across hosts.
- Preconditions: Lease backend and live steering runtime exist.
- Effects: Serialize conflicting run revisions, deduplicate exact semantic requests, order independent events and reject stale owners.
- Evidence subset: interleaving trace, CAS/lease/fence receipts, provider/write/event counts
- Acceptance: One conflicting revision wins; stale callers receive current context without silent rebase; exact replay performs no duplicate provider/write/event work; crash/retry preserves history; distributed fencing prevents an expired owner from publishing.

## ASE-026 Gate authority, scope, secrets, stale state, and adversarial prompts

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-security
- Depends on: ASE-008, ASE-015, ASE-021, ASE-024
- Goal id: ASE-G070
- Outputs: test/api/test_agent_supervisor_prompt_only_security.py, test/fixtures/agent_supervisor_prompt_entrypoints/adversarial/manifest.json
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_security.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/security
- Parallel lane: security-gate
- Resource class: cpu-medium
- Predicted files: test/api/test_agent_supervisor_prompt_only_security.py, test/fixtures/agent_supervisor_prompt_entrypoints/adversarial/manifest.json
- Interfaces: PromptOnlySecurityGate
- Conflict policy: Freeze and run a closed adversarial population; do not narrow fixtures or modify production validators in the gate task.
- Preconditions: Authority, intent saga, MCP++/UCAN and steering mutation boundaries exist.
- Effects: Exercise prompt/config/path/symlink/submodule/profile/provider/UCAN/authorization/CID/lease/fence/completion attacks and scan every durable/diagnostic surface for bodies/secrets.
- Evidence subset: attack-by-attack result, effect diff, leak scan
- Acceptance: Zero seeded root, caller, policy, authority, command, network, merge/push/deploy, destructive, completion, stale-state, lease/fence or secret escapes; every denial is typed and no attack is hidden by an aggregate score.

## ASE-027 Add inferred status, follow, explain, and doctor services

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: entrypoint-observability
- Depends on: ASE-010, ASE-011, ASE-013, ASE-014
- Goal id: ASE-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/status_explain.py, ipfs_accelerate_py/agent_supervisor/entrypoints/doctor.py, test/api/test_agent_supervisor_prompt_only_status.py, test/api/test_agent_supervisor_prompt_only_doctor.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_status.py test/api/test_agent_supervisor_prompt_only_doctor.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/status-doctor
- Parallel lane: status-doctor
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/status_explain.py, ipfs_accelerate_py/agent_supervisor/entrypoints/doctor.py, test/api/test_agent_supervisor_prompt_only_status.py, test/api/test_agent_supervisor_prompt_only_doctor.py
- Interfaces: status, follow, explain, doctor service methods
- Conflict policy: Read canonical resolver/run/event/capability state; do not infer authority, mutate recovery state or duplicate low-level status backends.
- Preconditions: Target/profile resolution, explanation, run registry and standard runtime exist.
- Effects: Select a unique current run, render bounded state/config provenance, stream/replay events by cursor and check required/degraded/optional dependencies with safe remedies.
- Evidence subset: status selection, cursor replay, explanation, doctor matrix, leak scan
- Acceptance: Status/follow need no IDs when exactly one compatible run exists; ambiguity is explicit; event resume is lossless/bounded; explain names every source; doctor checks Git/state/task-source/provider/authority/lease/process health and returns the smallest safe remedy without exposing bodies/secrets.

## ASE-028 Add prompt-entrypoint metrics, traces, and bounded observability

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: entrypoint-observability
- Depends on: ASE-015, ASE-024, ASE-027
- Goal id: ASE-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/metrics.py, test/api/test_agent_supervisor_prompt_only_metrics.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_metrics.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/metrics
- Parallel lane: entrypoint-metrics
- Resource class: cpu-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/metrics.py, test/api/test_agent_supervisor_prompt_only_metrics.py
- Interfaces: PromptEntrypointMetrics, stage trace projection
- Conflict policy: Consume existing events/receipts; do not create a second event log or include prompt, repository path, run ID or other high-cardinality sensitive labels in metrics.
- Preconditions: Prompt-to-run saga, steering and status services emit canonical events.
- Effects: Measure resolution dispositions, time to handle/event, saga stages, adoption/launch, provider/resource/lease degradation, steering/replay, recovery and terminal outcomes.
- Evidence subset: metric schema, bounded cardinality, receipt recomputation
- Acceptance: Metrics are recomputable from events, bounded in labels/storage, distinguish success/partial/ambiguity/denial/degradation, expose no prompt/source/credential data and support the published rollout gates.

## ASE-029 Add exhaustive resolver, profile, registry, broker, and steering unit/property tests

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-verification
- Depends on: ASE-005, ASE-006, ASE-007, ASE-008, ASE-009, ASE-010, ASE-011, ASE-012, ASE-013, ASE-023
- Goal id: ASE-G090
- Outputs: test/api/test_agent_supervisor_prompt_only_inference.py, test/api/test_agent_supervisor_prompt_only_properties.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_inference.py test/api/test_agent_supervisor_prompt_only_properties.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/unit-property
- Parallel lane: unit-property-gate
- Resource class: cpu-medium
- Predicted files: test/api/test_agent_supervisor_prompt_only_inference.py, test/api/test_agent_supervisor_prompt_only_properties.py
- Interfaces: PromptOnlyInferencePropertyGate
- Conflict policy: Test-only fan-in over leaf contracts; do not modify production implementations or discard counterexamples.
- Preconditions: All leaf inference/state/prompt/steering contract implementations exist.
- Effects: Generate precedence, topology, ambiguity, identity, bound, serialization, corruption, replay and lifecycle-state combinations.
- Evidence subset: complete property population, minimized counterexamples
- Acceptance: Unchanged evidence always resolves identically; lower trust never widens higher ceilings; every generated invalid/corrupt/stale state rejects safely; exact replay and identity round trips hold across the complete bounded population.

## ASE-030 Run real installed prompt-to-run, steering, and transport E2E

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-verification
- Depends on: ASE-018, ASE-019, ASE-020, ASE-021, ASE-022, ASE-024, ASE-025, ASE-027
- Goal id: ASE-G090
- Outputs: test/api/test_agent_supervisor_prompt_only_e2e.py, test/fixtures/agent_supervisor_prompt_entrypoints/e2e/manifest.json
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_e2e.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/e2e
- Parallel lane: e2e-gate
- Resource class: process-control
- Predicted files: test/api/test_agent_supervisor_prompt_only_e2e.py, test/fixtures/agent_supervisor_prompt_entrypoints/e2e/manifest.json
- Interfaces: InstalledPromptOnlyE2E
- Conflict policy: Use real temporary repositories, installed CLI, server tools, stores and subprocesses; do not replace missing runtime paths with synthetic handlers.
- Preconditions: Public surfaces, conformance, steering, concurrency and status services are complete.
- Effects: Exercise scan, deterministic/model plan, admission, Markdown/DuckDB/both materialization, launch/adopt, task claim, validation, steer, follow, restart, resume, completion and quarantine through every transport.
- Evidence subset: per-stage receipts, canonical transport results, process/task/effect postconditions
- Acceptance: At least 95 percent of supported single-target fixtures reach an admitted materialized healthy run from prompt plus ambient authenticated context; explicit and inferred profiles agree; real work is claimed; every failure/ambiguity is visible; no synthetic handler masks an unavailable live path.

## ASE-031 Gate crash recovery, concurrency, load, latency, resources, and rollout

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-verification
- Depends on: ASE-017, ASE-025, ASE-026, ASE-028, ASE-030, ASE-035, ASE-036
- Goal id: ASE-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/benchmark.py, ipfs_accelerate_py/agent_supervisor/entrypoints/rollout.py, test/api/test_agent_supervisor_prompt_only_chaos.py, test/api/test_agent_supervisor_prompt_only_rollout.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_chaos.py test/api/test_agent_supervisor_prompt_only_rollout.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/chaos-rollout
- Parallel lane: chaos-rollout-gate
- Resource class: cpu-large
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/benchmark.py, ipfs_accelerate_py/agent_supervisor/entrypoints/rollout.py, test/api/test_agent_supervisor_prompt_only_chaos.py, test/api/test_agent_supervisor_prompt_only_rollout.py
- Interfaces: PromptEntrypointBenchmark, PromptEntrypointRolloutDecision
- Conflict policy: Own the closed chaos/load population and promotion logic; do not narrow fixtures/thresholds or modify production runtime while evaluating it.
- Preconditions: Lease/concurrency/security/metrics and real E2E gates pass.
- Effects: Kill before/after saga boundaries; inject Grok/Codex route, DuckDB owner/cursor, signing/head, disclosure, Parquet export, IPLD/CAR, IPFS, stale replica, partition and disk faults; issue concurrent runs/steers and scale repositories/runs/shards/lanes/epochs/events while measuring bounds.
- Evidence subset: full chaos trace, resource/latency distributions, promotion/rollback decision
- Acceptance: Every fault resumes/compensates/quarantines within bounds; no duplicate process/effect or stale owner/replica wins; DuckDB-Parquet-IPLD replay remains exact; CPU/memory/descriptors/storage/provider use remain bounded; deterministic replay/parity/security gates stay perfect; published time-to-handle/event thresholds pass or rollout remains denied.

## ASE-032 Migrate legacy launchers, state layouts, and expert configuration conservatively

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: entrypoint-rollout
- Depends on: ASE-016, ASE-019, ASE-022
- Goal id: ASE-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/migration.py, test/api/test_agent_supervisor_prompt_only_compatibility.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_compatibility.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/migration
- Parallel lane: compatibility-migration
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/migration.py, test/api/test_agent_supervisor_prompt_only_compatibility.py
- Interfaces: LegacyProfileImporter, LegacyRunDiscovery, expert CLI compatibility
- Conflict policy: Preview and adapt old configuration/state without deleting or rewriting it; shared scripts/docs are changed only in ASE-033 after compatibility passes.
- Preconditions: Launch profile, CLI and transport conformance are complete.
- Effects: Discover old state roots/status/PID/taskboards/launch args, validate identity, preview profile/run import, write new registry transaction and retain rollback pointers.
- Evidence subset: legacy fixture parity, migration preview/apply/rollback receipts
- Acceptance: Existing expert requests keep canonical results/exit behavior; old runs/taskboards remain readable; import is explicit, idempotent and recoverable; no existing state is silently moved/deleted; console-script declarations remain equivalent during packaging transition.

## ASE-033 Publish guides, runbooks, threat model, rollout controls, and deprecations

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-rollout
- Depends on: ASE-022, ASE-027, ASE-031, ASE-032
- Goal id: ASE-G100
- Outputs: docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md, ipfs_accelerate_py/agent_supervisor/entrypoints/README.md, test/api/test_agent_supervisor_docs.py
- Validation: python -m pytest test/api/test_agent_supervisor_docs.py test/api/test_agent_supervisor_prompt_only_rollout.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/docs-rollout
- Parallel lane: docs-rollout
- Resource class: cpu-small
- Predicted files: docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md, ipfs_accelerate_py/agent_supervisor/entrypoints/README.md, test/api/test_agent_supervisor_docs.py
- Interfaces: Operator guide, Python/CLI/MCP examples, security/runbook, rollout profile
- Conflict policy: Sole documentation and shared rollout owner; examples must be generated/validated against current schemas and must not claim automatic authority beyond passing rollout mode.
- Preconditions: Surface conformance, status/doctor, chaos/rollout and migration gates are complete.
- Effects: Publish prompt-only quick start, advanced overrides, profile setup, inference explanation, Grok-first/Codex-fallback receipts, steering, recovery, DuckDB shard ownership, immutable Parquet/IPLD/IPFS replication, UCAN, migration, threat model, metrics, rollback and deprecation guidance.
- Evidence subset: executable documentation examples, link/schema checks, active rollout decision
- Acceptance: A new operator can install, explicitly set up bounded local authority, run/steer/follow/diagnose from prompts, understand every inference/approval boundary and recover/rollback; expert and server deployment paths are documented; automatic modes are advertised only when their current gate passes.

## ASE-034 Perform independent fresh-root closeout and objective coverage verification

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: entrypoint-rollout
- Depends on: ASE-026, ASE-029, ASE-030, ASE-031, ASE-033
- Goal id: ASE-G100
- Outputs: data/agent_supervisor/prompt_only_entrypoints/closeout/coverage.json, data/agent_supervisor/prompt_only_entrypoints/closeout/rollout.json
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_e2e.py test/api/test_agent_supervisor_prompt_only_conformance.py test/api/test_agent_supervisor_prompt_only_security.py test/api/test_agent_supervisor_prompt_only_chaos.py test/api/test_agent_supervisor_prompt_only_rollout.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v1
- Bundle: agent-supervisor/prompt-entrypoints/closeout
- Parallel lane: closeout
- Resource class: coordinator
- Predicted files: data/agent_supervisor/prompt_only_entrypoints/closeout/coverage.json, data/agent_supervisor/prompt_only_entrypoints/closeout/rollout.json
- Interfaces: GoalCoverageReport, independent PromptEntrypointRolloutDecision
- Conflict policy: Evaluation-only closeout on a later fresh tree; do not edit production code, tests, fixture population, thresholds, objectives or taskboard to make the report pass.
- Preconditions: Security, unit/property, real E2E, chaos/load and documentation gates have terminal current evidence.
- Effects: Recompute every mandatory goal criterion, producer population, validation receipt, tree/provenance/freshness binding, quantitative metric and rollback trigger.
- Evidence subset: criterion-to-receipt map, full terminal task/goal population, independent rollout result
- Acceptance: Every root and child criterion maps to fresh passing producer evidence; no stale, self-reported, prompt/model/task-status evidence is accepted; all descendants are terminal; automatic rollout is enabled only if every binding/parity/security/reliability/quantitative gate passes, otherwise the exact blockers remain open and visible.
