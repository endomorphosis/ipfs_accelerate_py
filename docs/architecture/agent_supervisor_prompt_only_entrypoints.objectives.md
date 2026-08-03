# Agent Supervisor Prompt-Only Entrypoints Objective Heap

This is the durable goal/subgoal heap for
`AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md`. The executable projection
is `agent_supervisor_prompt_only_entrypoints.todo.md` with task prefix
`## ASE-`.

## Goal tree

```text
ASE-G000  Prompt-only, steerable agent supervisor
|-- ASE-G010  Product journey and measurable baseline
|-- ASE-G020  High-level invocation contracts and entrypoint package
|-- ASE-G030  Deterministic target and profile inference
|   |-- ASE-G031  Repository, checkout, scope, and tree identity
|   |-- ASE-G032  State, run, objective, and task-source selection
|   |-- ASE-G033  Principal, policy, authority, and effect ceiling
|   |-- ASE-G034  Provider, resources, lanes, validation, and launch profile
|   `-- ASE-G035  Explainable inference and plan linting
|-- ASE-G040  Durable standard runtime and prompt-to-run saga
|   |-- ASE-G041  Prompt broker and run registry
|   |-- ASE-G042  Standard service factory and resumable bootstrap
|   |-- ASE-G043  Launch-profile and DuckDB lease/fence compilation
|   |-- ASE-G044  Parquet/IPLD/IPFS coordination replication
|   `-- ASE-G045  Typed Grok-first/Codex-fallback provider route
|-- ASE-G050  Python, CLI, MCP, and MCP++ entrypoint parity
|   |-- ASE-G051  Python convenience API
|   |-- ASE-G052  Prompt-first CLI
|   |-- ASE-G053  MCP convenience tools
|   |-- ASE-G054  MCP++ and UCAN binding
|   `-- ASE-G055  Cross-transport conformance
|-- ASE-G060  Semantic, revisioned steering
|   |-- ASE-G061  Steering contracts and intent classification
|   |-- ASE-G062  Plan-delta and live-run integration
|   `-- ASE-G063  Concurrent steering, CAS, and replay
|-- ASE-G070  Authority, scope, and adversarial safety
|-- ASE-G080  Status, follow, explain, doctor, and observability
|-- ASE-G090  E2E, chaos, load, and quantitative gates
`-- ASE-G100  Compatibility migration, documentation, rollout, and closeout
```

## ASE-G000 Prompt-only, steerable agent supervisor

- Status: active
- Parent:
- Parent goal IDs JSON: []
- Depends on:
- Dependencies JSON: []
- Fib priority: 1
- Track: prompt-entrypoints
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/root
- Parallel lane: program
- Resource class: coordinator
- Goal: Make one prompt plus ambient authenticated context sufficient to resolve, configure, launch, observe, resume, and steer useful isolated agent-supervisor work.
- Producing tasks: ASE-001, ASE-002, ASE-003, ASE-004, ASE-005, ASE-006, ASE-007, ASE-008, ASE-009, ASE-010, ASE-011, ASE-012, ASE-013, ASE-014, ASE-015, ASE-016, ASE-017, ASE-018, ASE-019, ASE-020, ASE-021, ASE-022, ASE-023, ASE-024, ASE-025, ASE-026, ASE-027, ASE-028, ASE-029, ASE-030, ASE-031, ASE-032, ASE-033, ASE-034, ASE-035, ASE-036, ASE-037, ASE-038, ASE-039, ASE-040, ASE-042
- Evidence: prompt_only_entrypoint_rollout.PROMPT_ONLY_ENTRYPOINT_ROLLOUT_REQUIREMENT_ID
- Evidence requirements JSON: ["fresh prompt-only subprocess E2E receipt", "Python CLI MCP MCP++ conformance receipt", "authority and secret non-escape receipt", "concurrent launch and steering receipt", "DuckDB Parquet IPLD IPFS replay receipt", "Grok to Codex route receipt", "fresh rollout decision"]
- Evidence criteria: Every child goal is terminal with current-tree producer evidence; one prompt reaches a healthy isolated run; every inferred field has provenance; all transports agree; restart and steering are durable; Grok is the default with typed Codex fallback; committed coordination epochs replay exactly through DuckDB, Parquet, IPLD and IPFS; and no unauthorized, out-of-scope, stale, duplicate-process, or secret-bearing behavior is accepted.
- Evidence source policy: Plans, prompts, task status, model output, inferred confidence, process liveness, and historical tests are non-authoritative. Completion requires fresh producer-owned receipts over exact repository, policy, profile, run, task-source, lease, transport, effect, and validation roots.
- Outputs: docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md, docs/architecture/agent_supervisor_prompt_only_entrypoints.objectives.md, docs/architecture/agent_supervisor_prompt_only_entrypoints.todo.md, ipfs_accelerate_py/agent_supervisor/entrypoints
- Predicted files JSON: ["docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md", "docs/architecture/agent_supervisor_prompt_only_entrypoints.objectives.md", "docs/architecture/agent_supervisor_prompt_only_entrypoints.todo.md", "ipfs_accelerate_py/agent_supervisor/entrypoints"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_e2e.py test/api/test_agent_supervisor_prompt_only_conformance.py test/api/test_agent_supervisor_prompt_only_security.py test/api/test_agent_supervisor_prompt_only_chaos.py test/api/test_agent_supervisor_prompt_only_rollout.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_e2e.py test/api/test_agent_supervisor_prompt_only_conformance.py test/api/test_agent_supervisor_prompt_only_security.py test/api/test_agent_supervisor_prompt_only_chaos.py test/api/test_agent_supervisor_prompt_only_rollout.py -q"]
- Acceptance: A supported checkout can run and steer useful isolated work from a prompt without manually supplying target, identity, task-source, provider, lane, daemon, lifecycle, or state flags; stronger effects remain separately authorized; exact expert requests remain compatible.
- Gap task: Execute the ready ASE task population by dependency, conflict, resource, and authority policy.
- Refinement: Preserve the low-level typed control plane and add one high-level inference and saga facade above it.
- Embedding query: agent supervisor prompt only entrypoint automatic inference runtime factory run handle steering CLI MCP UCAN
- AST query: PromptSupervisorService SupervisorControlService LifecycleOrchestrator PortalImplementationSupervisor ResourceScheduler

## ASE-G010 Product journey and measurable baseline

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 2
- Track: product-baseline
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/baseline
- Parallel lane: baseline
- Resource class: cpu-small
- Goal: Freeze the current flag/runtime gaps, representative target matrix, prompt-only user journeys, and measurable success/latency/safety criteria before implementation.
- Producing tasks: ASE-001, ASE-002
- Evidence: prompt_entrypoint_baseline.PROMPT_ENTRYPOINT_BASELINE_REQUIREMENT_ID
- Evidence requirements JSON: ["current CLI and MCP inventory", "frozen fixture matrix", "measurable UX acceptance contract"]
- Evidence criteria: The inventory exercises installed CLI, Python, MCP and lifecycle construction; fixtures cover clean, dirty, nested, submodule, ambiguous and degraded targets; metrics include successful run rate, time to handle/event, flags supplied, parity, leaks and unexpected effects.
- Evidence source policy: Documentation claims and parser help alone are non-authoritative; evidence is generated by executable probes and frozen fixture manifests on the current tree.
- Outputs: docs/architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md, test/fixtures/agent_supervisor_prompt_entrypoints
- Predicted files JSON: ["docs/architecture/agent_supervisor/PROMPT_ENTRYPOINT_BASELINE.md", "test/fixtures/agent_supervisor_prompt_entrypoints"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_entrypoint_baseline.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_entrypoint_baseline.py -q"]
- Acceptance: The baseline reproduces the nine-binding requirement, read-only default runtime and prompt-body handoff gap, and defines the exact zero-flag journeys and quantitative gates used by rollout.
- Gap task: Produce the smallest missing executable current-state probe or representative fixture.

## ASE-G020 High-level invocation contracts and entrypoint package

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G010
- Dependencies JSON: ["ASE-G010"]
- Fib priority: 3
- Track: entrypoint-contracts
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/contracts
- Parallel lane: contracts
- Resource class: cpu-small
- Goal: Define provider-free content-addressed invocation, target-resolution, profile, launch, run, continuation, and result contracts in a highest-layer package that composes but does not contaminate the existing domain DAG.
- Producing tasks: ASE-003, ASE-004
- Evidence: entrypoint_contracts.PROMPT_ONLY_ENTRYPOINT_CONTRACT_REQUIREMENT_ID
- Evidence requirements JSON: ["canonical contract round trips", "unknown and over-bound rejection", "cold import and package DAG receipt"]
- Evidence criteria: Contracts exclude prompt bodies and secrets, distinguish hints from authority, bind all semantic roots, record alternatives and inference provenance, and serialize identically across transports; lower packages do not import the entrypoint layer.
- Evidence source policy: Dataclass construction, import success, or a schema document is non-authoritative; evidence is strict round-trip, malformed/adversarial, size/depth, cold-import, and dependency-DAG test output.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py, docs/architecture/agent_supervisor/PACKAGE_MAP.md
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py", "docs/architecture/agent_supervisor/PACKAGE_MAP.md"]
- Validation: python -m pytest test/api/test_agent_supervisor_entrypoint_contracts.py test/api/test_agent_supervisor_entrypoint_package.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_entrypoint_contracts.py test/api/test_agent_supervisor_entrypoint_package.py -q"]
- Acceptance: All high-level surfaces can exchange one immutable request/result vocabulary while the existing `OperationRequest`, control catalog, cold-import behavior and package dependency direction remain intact.
- Gap task: Close the smallest contract, bound, identity, secret, schema, export, or package-DAG residual.

## ASE-G030 Deterministic target and profile inference

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G020
- Dependencies JSON: ["ASE-G020"]
- Fib priority: 5
- Track: target-inference
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/inference
- Parallel lane: inference-integration
- Resource class: cpu-medium
- Goal: Resolve every discoverable supervisor binding and configuration value from trusted context and current target evidence with deterministic precedence, explicit alternatives, safe defaults, and material-ambiguity handling.
- Producing tasks: ASE-005, ASE-006, ASE-007, ASE-008, ASE-009, ASE-010, ASE-011
- Evidence: target_resolver.TARGET_RESOLUTION_REQUIREMENT_ID
- Evidence requirements JSON: ["field-level inference provenance receipt", "deterministic replay receipt", "ambiguity and denial fixtures", "configuration lint result"]
- Evidence criteria: Unchanged evidence resolves identically; current dirty/submodule roots are bound; objective/task-source/run adoption is unique; lower-precedence hints cannot widen policy/authority/effects; provider/resources/lanes are capability-derived; every ambiguity is typed.
- Evidence source policy: Current directory, remote name, environment variable, repository prose, provider executable presence, or model choice alone is non-authoritative. Evidence is a canonical resolution receipt joined to exact repository, transport, profile, capability and policy observations.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py, ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py, ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py, ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py, ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py, ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_target_resolver.py test/api/test_agent_supervisor_profile_resolver.py test/api/test_agent_supervisor_inference_explain.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_target_resolver.py test/api/test_agent_supervisor_profile_resolver.py test/api/test_agent_supervisor_inference_explain.py -q"]
- Acceptance: A prompt-only local invocation resolves a complete bounded target/profile when evidence is unique, continues safely to preview on non-effect ambiguity, and never guesses identity/authority/effects across a material ambiguity.
- Gap task: Repair the smallest missing resolver, provenance edge, precedence rule, ambiguity case, replay invariant, or lint finding.

## ASE-G031 Repository, checkout, scope, and tree identity

- Status: active
- Parent: ASE-G030
- Parent goal IDs JSON: ["ASE-G030"]
- Depends on: ASE-G020
- Dependencies JSON: ["ASE-G020"]
- Fib priority: 3
- Track: repository-inference
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/repository
- Parallel lane: repository-resolution
- Resource class: io-small
- Goal: Reuse repository forest, checkout authority and repository snapshot helpers to select one allowlisted root/scope and bind the exact dirty worktree and submodule population.
- Producing tasks: ASE-005
- Evidence: target_resolver.REPOSITORY_TARGET_RESOLUTION_REQUIREMENT_ID
- Evidence requirements JSON: ["clean and dirty repository fixtures", "nested and symlink ambiguity fixtures", "submodule-aware tree identity"]
- Evidence criteria: Root and scope cannot escape configured bounds; HEAD alone is insufficient; staged, modified, deleted, admitted-untracked and submodule state affect identity; nested alternatives are explicit.
- Evidence source policy: `cwd`, `.git` existence, HEAD, or remote origin alone is non-authoritative; evidence is the canonical checkout/snapshot binding and resolution receipt.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_target_resolver.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_target_resolver.py -q"]
- Acceptance: Root, repository ID, scope and current tree resolve deterministically across ordinary repositories, worktrees and submodules and fail closed on path or authority ambiguity.
- Gap task: Add the smallest missing repository topology or dirty-state fixture.

## ASE-G032 State, run, objective, and task-source selection

- Status: active
- Parent: ASE-G030
- Parent goal IDs JSON: ["ASE-G030"]
- Depends on: ASE-G031
- Dependencies JSON: ["ASE-G031"]
- Fib priority: 5
- Track: state-inference
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/state
- Parallel lane: state-resolution
- Resource class: io-small
- Goal: Select collision-resistant platform state, create/adopt exactly one compatible run, and resolve or create objective and task-source revisions without dirtying the source checkout by default.
- Producing tasks: ASE-006, ASE-007
- Evidence: state_resolver.STATE_AND_OBJECTIVE_RESOLUTION_REQUIREMENT_ID
- Evidence requirements JSON: ["state namespace collision tests", "run adoption tests", "objective and task-source ambiguity tests"]
- Evidence criteria: State is repository-keyed and bounded; compatible runs adopt idempotently; incompatible/multiple runs do not merge; a new prompt creates content-addressed intent when no unique existing objective applies; task-source identity and revision are exact.
- Evidence source policy: Directory names, PID/status files, objective titles and taskboard filenames alone are non-authoritative; evidence is current integrity-checked registry, objective and task-source receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py, ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_state_resolver.py test/api/test_agent_supervisor_objective_resolver.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_state_resolver.py test/api/test_agent_supervisor_objective_resolver.py -q"]
- Acceptance: A new invocation receives a stable isolated namespace while status/steer/follow adopt only the exact compatible run, objective and task source.
- Gap task: Close the smallest state, adoption, collision, objective, projection, revision, or ambiguity residual.

## ASE-G033 Principal, policy, authority, and effect ceiling

- Status: active
- Parent: ASE-G030
- Parent goal IDs JSON: ["ASE-G030"]
- Depends on: ASE-G020
- Dependencies JSON: ["ASE-G020"]
- Fib priority: 2
- Track: authority-inference
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/authority
- Parallel lane: authority-resolution
- Resource class: cpu-small
- Goal: Bind authenticated local or transport identity, select only trusted policies/profiles, and distinguish inferable configuration from separately granted mutation authority and expected effects.
- Producing tasks: ASE-008
- Evidence: authority_resolver.AUTHORITY_RESOLUTION_REQUIREMENT_ID
- Evidence requirements JSON: ["local principal binding", "MCP principal binding", "effect-ceiling decision", "authority non-inference adversarial tests"]
- Evidence criteria: Prompt and repository content cannot create identity or authority; allowlists and effect ceilings only narrow; local worktree authority is explicitly installed and receipt-bound; stronger effects remain denied.
- Evidence source policy: Username text, environment claims, credentials presence, prompt instructions, repository policy or transport tool selection is non-authoritative; evidence is verified principal/profile/UCAN and inner authorization decisions.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/authority_resolver.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_authority_resolver.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_authority_resolver.py -q"]
- Acceptance: Safe local worktree effects can be authorized without repeated flags after explicit setup, while all caller, allowlist, policy and stronger-effect forgery paths fail closed.
- Gap task: Add the smallest missing principal, policy, profile, effect or denial fixture.

## ASE-G034 Provider, resources, lanes, validation, and launch profile

- Status: active
- Parent: ASE-G030
- Parent goal IDs JSON: ["ASE-G030"]
- Depends on: ASE-G031, ASE-G032, ASE-G033
- Dependencies JSON: ["ASE-G031", "ASE-G032", "ASE-G033"]
- Fib priority: 8
- Track: profile-inference
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/profile
- Parallel lane: profile-resolution
- Resource class: cpu-medium
- Goal: Compile provider capability, host resources, conflict-ready width, validation policy, worktree/merge strategy and deployment topology into one immutable conservative supervisor profile.
- Producing tasks: ASE-009, ASE-010
- Evidence: profile_resolver.RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID
- Evidence requirements JSON: ["Grok preferred and Codex fallback route matrix", "provider capability and degradation matrix", "resource/lane scheduling receipt", "profile precedence and signing tests", "structured validation policy"]
- Evidence criteria: Provider selection prefers healthy policy-allowed Grok and records a typed Codex fallback reason when needed; a fallback implementer cannot self-satisfy independent review; lane width is resource/conflict bounded; validation is structured allowlisted argv; prompt/repository content cannot inject commands or credentials; profile identity covers all behavior.
- Evidence source policy: Executable presence, advertised model, CPU count, lane labels, CI shell, or profile filename alone is non-authoritative; evidence is negotiated capability, resource scheduler, reviewed validation and verified profile receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py, ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_capability_resolver.py test/api/test_agent_supervisor_profile_resolver.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_capability_resolver.py test/api/test_agent_supervisor_profile_resolver.py -q"]
- Acceptance: The same target and trusted context compile the same complete profile; Grok is preferred and Codex fallback is typed and policy-bounded; unavailable capabilities degrade explicitly; false parallelism, self-review, shell injection and effect widening are rejected.
- Gap task: Close the smallest provider, resource, lane, validation, profile or degradation residual.

## ASE-G035 Explainable inference and plan linting

- Status: active
- Parent: ASE-G030
- Parent goal IDs JSON: ["ASE-G030"]
- Depends on: ASE-G031, ASE-G032, ASE-G033, ASE-G034
- Dependencies JSON: ["ASE-G031", "ASE-G032", "ASE-G033", "ASE-G034"]
- Fib priority: 13
- Track: inference-assurance
- Priority: P1
- Bundle: agent-supervisor/prompt-entrypoints/inference-assurance
- Parallel lane: inference-explain
- Resource class: cpu-small
- Goal: Render body-free explanations and provide a reusable plan lint that verifies objective hierarchy, task dependencies, metadata, conflicts, structured validations and inferred-profile completeness.
- Producing tasks: ASE-011
- Evidence: inference_explain.INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID
- Evidence requirements JSON: ["stable explanation fixture", "goal/task graph lint", "no-secret rendering test"]
- Evidence criteria: Every selected/defaulted/ambiguous/denied field has a source and reason; plan lint detects unknown/cyclic dependencies, missing required metadata, unsafe validation and predicted-file conflicts without mutating state.
- Evidence source policy: Human-readable output alone is non-authoritative; evidence is exact lint findings bound to parsed objective/task/profile identities plus leak scans.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_explain.py, ipfs_accelerate_py/agent_supervisor/entrypoints/plan_lint.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/inference_explain.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/plan_lint.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_inference_explain.py test/api/test_agent_supervisor_plan_lint.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_inference_explain.py test/api/test_agent_supervisor_plan_lint.py -q"]
- Acceptance: Operators and automation can reproduce why a run configuration was selected and reject malformed goal/task/profile plans before implementation.
- Gap task: Add the smallest missing explanation provenance or lint invariant.

## ASE-G040 Durable standard runtime and prompt-to-run saga

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G030
- Dependencies JSON: ["ASE-G030"]
- Fib priority: 8
- Track: invocation-runtime
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/runtime
- Parallel lane: runtime-integration
- Resource class: process-control
- Goal: Build the trusted runtime composition that carries transient prompt data, persists run/receipt state, installs real control handlers, compiles launch profiles, and idempotently resolves, plans, materializes, starts or adopts work.
- Producing tasks: ASE-012, ASE-013, ASE-014, ASE-015, ASE-016, ASE-017, ASE-035, ASE-036, ASE-037, ASE-038, ASE-039, ASE-040, ASE-042
- Evidence: intent_service.PROMPT_TO_RUN_SAGA_REQUIREMENT_ID
- Evidence requirements JSON: ["live handler capability report", "prompt body non-persistence receipt", "crash-resumable saga", "process adoption receipt", "typed Grok to Codex route receipt", "DuckDB lease and fencing receipt", "Parquet IPLD IPFS coordination epoch receipt"]
- Evidence criteria: Advertised operations have real handlers; every effect has intent and observed-effect receipts; raw prompt bodies remain transient; exact replay and restart resume; compatible processes adopt; partial failure has one continuation; Grok-first/Codex-fallback production dispatch is typed and review-separated; DuckDB shard leases/profiles are current; immutable coordination epochs replay exactly and never grant authority to a replica.
- Evidence source policy: Handler registration, process liveness, taskboard existence or successful provider response alone is non-authoritative; evidence is the complete root-bound saga and post-effect receipt population.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/lease_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_epoch.py, ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_authority.py, ipfs_accelerate_py/agent_supervisor/entrypoints/replication_policy.py, ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_replication.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/launch_profile.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/lease_backend.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_epoch.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_authority.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/replication_policy.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_replication.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_broker.py test/api/test_agent_supervisor_run_registry.py test/api/test_agent_supervisor_runtime_factory.py test/api/test_agent_supervisor_plan_materializer.py test/api/test_agent_supervisor_intent_service.py test/api/test_agent_supervisor_launch_profile.py test/api/test_agent_supervisor_lease_backends.py test/api/test_agent_supervisor_coordination_replication.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_broker.py test/api/test_agent_supervisor_run_registry.py test/api/test_agent_supervisor_runtime_factory.py test/api/test_agent_supervisor_plan_materializer.py test/api/test_agent_supervisor_intent_service.py test/api/test_agent_supervisor_launch_profile.py test/api/test_agent_supervisor_lease_backends.py test/api/test_agent_supervisor_coordination_replication.py -q"]
- Acceptance: The installed high-level service can perform real prompt-to-run work using existing domain services, survive interruption, and never duplicate a run/process/effect for exact replay.
- Gap task: Repair the smallest broker, registry, handler, saga, profile, adoption, continuation, lease or fencing residual.

## ASE-G041 Prompt broker and run registry

- Status: active
- Parent: ASE-G040
- Parent goal IDs JSON: ["ASE-G040"]
- Depends on: ASE-G020, ASE-G032
- Dependencies JSON: ["ASE-G020", "ASE-G032"]
- Fib priority: 3
- Track: run-state
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/run-state
- Parallel lane: run-state
- Resource class: io-artifact
- Goal: Deliver a bounded transient/capability-protected prompt channel and a durable content-addressed run registry with exact adoption, CAS revision and event continuation.
- Producing tasks: ASE-012, ASE-013
- Evidence: run_registry.RUN_REGISTRY_AND_PROMPT_BROKER_REQUIREMENT_ID
- Evidence requirements JSON: ["prompt leak scan", "run reconstruction and CAS tests", "adoption and ambiguity tests"]
- Evidence criteria: Planner can retrieve the exact prompt during its bounded lifetime; routine records contain only identity/reference; restart reconstructs handles; selection/adoption is exact; concurrent revision writes cannot both win.
- Evidence source policy: In-memory object presence, prompt hash alone, PID file, directory timestamp or last-run heuristic is non-authoritative; evidence is capability validation and registry transaction receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_broker.py test/api/test_agent_supervisor_run_registry.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_broker.py test/api/test_agent_supervisor_run_registry.py -q"]
- Acceptance: Prompt use and run continuation work across process boundaries without putting prompt text into ordinary state or selecting an incompatible run.
- Gap task: Close the smallest lifecycle, capability, persistence, CAS, adoption or leak residual.

## ASE-G042 Standard service factory and resumable bootstrap

- Status: active
- Parent: ASE-G040
- Parent goal IDs JSON: ["ASE-G040"]
- Depends on: ASE-G030, ASE-G041
- Dependencies JSON: ["ASE-G030", "ASE-G041"]
- Fib priority: 5
- Track: runtime-composition
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/runtime-factory
- Parallel lane: runtime-factory
- Resource class: process-control
- Goal: Install a production-shaped local runtime with real prompt, objective, materialization, lifecycle, status, validation, recovery and rescue handlers, then compose them into a resumable prompt-to-run saga.
- Producing tasks: ASE-014, ASE-015, ASE-042
- Evidence: runtime_factory.STANDARD_SUPERVISOR_RUNTIME_REQUIREMENT_ID
- Evidence requirements JSON: ["capability report with live handlers", "admitted DuckDB and bounded Markdown projection receipt", "preview materialize start/adopt saga", "intent/effect/receipt crash matrix"]
- Evidence criteria: No advertised convenience operation returns unavailable in the supported profile; handlers call existing domain services; admitted plans automatically materialize into a canonical root-bound DuckDB task source and only bounded/epoched canonical Markdown; every boundary is idempotent, root-bound, resumable and observable; partial effects do not become success.
- Evidence source policy: Static discovery, synthetic handlers, mocked process IDs, or in-memory success alone is non-authoritative; evidence uses real stores, task-source transactions and subprocess lifecycle fixtures.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py, ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_runtime_factory.py test/api/test_agent_supervisor_plan_materializer.py test/api/test_agent_supervisor_intent_service.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_runtime_factory.py test/api/test_agent_supervisor_plan_materializer.py test/api/test_agent_supervisor_intent_service.py -q"]
- Acceptance: A real supported invocation resolves, previews, materializes and starts/adopts through one service and resumes from every injected interruption boundary.
- Gap task: Close the smallest handler wiring, store, saga transition, postcondition, continuation or recovery gap.

## ASE-G043 Launch-profile and DuckDB lease/fence compilation

- Status: active
- Parent: ASE-G040
- Parent goal IDs JSON: ["ASE-G040"]
- Depends on: ASE-G034, ASE-G041
- Dependencies JSON: ["ASE-G034", "ASE-G041"]
- Fib priority: 8
- Track: launch-coordination
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/launch
- Parallel lane: launch-coordination
- Resource class: process-control
- Goal: Project one resolved profile into current runner/daemon configuration and select the existing single-writer DuckDB coordination shard, authenticated owner, lease, logical epoch and fencing policy without exposing their flag surfaces to normal users.
- Producing tasks: ASE-016, ASE-017
- Evidence: launch_profile.LAUNCH_PROFILE_AND_LEASE_BACKEND_REQUIREMENT_ID
- Evidence requirements JSON: ["profile-to-runner projection parity", "DuckDB shard lease/fence conformance", "authenticated remote owner routing", "duplicate-process prevention"]
- Evidence criteria: Projection covers every behavioral flag and stable path; argv/environment are bounded; DuckDB transaction/file-lock fencing remains exact for one owner; multi-host workers route mutations to an authenticated shard owner instead of sharing a database file; incompatible or ambiguous ownership fails closed.
- Evidence source policy: Generated argv, a database file, lease acquisition or process start alone is non-authoritative; evidence is exact profile projection, backend conformance and duplicate-launch postcondition receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/launch_profile.py, ipfs_accelerate_py/agent_supervisor/entrypoints/lease_backend.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/launch_profile.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/lease_backend.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_launch_profile.py test/api/test_agent_supervisor_lease_backends.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_launch_profile.py test/api/test_agent_supervisor_lease_backends.py -q"]
- Acceptance: Normal callers need no daemon flags; local processes and remote workers obtain the right DuckDB-owner fencing semantics, replicas cannot grant claims, and exact replay cannot create a second process tree or accepted effect.
- Gap task: Repair the smallest config projection, argv/env bound, backend selection, lease, fencing or adoption residual.

## ASE-G044 Parquet/IPLD/IPFS coordination replication

- Status: active
- Parent: ASE-G040
- Parent goal IDs JSON: ["ASE-G040"]
- Depends on: ASE-G041, ASE-G042, ASE-G043
- Dependencies JSON: ["ASE-G041", "ASE-G042", "ASE-G043"]
- Fib priority: 8
- Track: coordination-replication
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/coordination-replication
- Parallel lane: coordination-replication
- Resource class: io-network
- Goal: Export committed DuckDB coordination epochs as immutable Parquet fragments and linked DAG-JSON/IPLD manifests, distribute them through IPFS/ipfs_kit_py, and reconstruct verified read-only DuckDB query replicas without turning eventual replication into lease authority.
- Producing tasks: ASE-035, ASE-037, ASE-038, ASE-039, ASE-040
- Evidence: coordination_replication.DUCKDB_PARQUET_IPLD_COORDINATION_REQUIREMENT_ID
- Evidence requirements JSON: ["atomic DuckDB commit sequence", "DuckDB to Parquet to IPLD to DuckDB logical parity", "strict CID and previous-epoch chain verification", "IPFS and ipfs_kit_py capability conformance", "remote result and head signature verification", "replication disclosure and leak gate", "stale replica authority denial", "partitioned remote result quarantine"]
- Evidence criteria: Every committed epoch is one transactionally bound logical snapshot and binds shard, cursor, frozen schema, logical row-set digests, fence maximum, verified Parquet CIDs, previous epoch, signing authority and disclosure policy; verified import is exact and idempotent; tampered/missing/reordered/equivocating/disallowed epochs fail closed; remote workers use immutable CID inputs and submit cryptographically authenticated results to the current owner; neither IPNS nor a replica can issue a claim or accept an effect.
- Evidence source policy: A Parquet file, backend identifier, pin, signature, IPNS head, successful fetch or row-count match alone is non-authoritative; evidence is strict CID/codec and fetched-byte verification, canonical manifest ancestry, exact logical-table parity, disclosure scan, signature/capability validation and current-owner lease/fence postconditions.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_epoch.py, ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_authority.py, ipfs_accelerate_py/agent_supervisor/entrypoints/replication_policy.py, ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_replication.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_epoch.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_authority.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/replication_policy.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/coordination_replication.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_coordination_epoch.py test/api/test_agent_supervisor_verified_ipld_backend.py test/api/test_agent_supervisor_coordination_authority.py test/api/test_agent_supervisor_replication_policy.py test/api/test_agent_supervisor_coordination_replication.py test/api/test_agent_supervisor_lease_coordination.py test/test_ipfs_backend_router.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_coordination_epoch.py test/api/test_agent_supervisor_verified_ipld_backend.py test/api/test_agent_supervisor_coordination_authority.py test/api/test_agent_supervisor_replication_policy.py test/api/test_agent_supervisor_coordination_replication.py test/api/test_agent_supervisor_lease_coordination.py test/test_ipfs_backend_router.py -q"]
- Acceptance: Committed coordination history is queryable locally in DuckDB, portable and shardable as policy-cleared Parquet/IPLD and capability-gated CAR, retrievable through verified IPFS adapters, logically reconstructible, cryptographically authenticated, and incapable of authorizing a mutable action without the current DuckDB-owner lease and fence.
- Gap task: Repair the smallest commit cursor, schema, CID/codec, Parquet parity, signing/head, disclosure/encryption, IPFS routing, replica import, owner binding, fence or quarantine residual.

## ASE-G045 Typed Grok-first/Codex-fallback provider route

- Status: active
- Parent: ASE-G040
- Parent goal IDs JSON: ["ASE-G040"]
- Depends on: ASE-G034, ASE-G042
- Dependencies JSON: ["ASE-G034", "ASE-G042"]
- Fib priority: 8
- Track: provider-runtime
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/provider-runtime
- Parallel lane: provider-runtime
- Resource class: provider-io
- Goal: Extend the production packet route so healthy admitted Grok is the default implementer and Codex is a once-only, pre-effect, scope-and-budget-bounded implementation fallback with a distinct independent-review continuation.
- Producing tasks: ASE-036
- Evidence: provider_route.TYPED_GROK_CODEX_FALLBACK_REQUIREMENT_ID
- Evidence requirements JSON: ["Grok preferred route receipt", "typed Codex fallback receipt matrix", "accepted-effect boundary proof", "attempt and process identity separation", "independent review continuation"]
- Evidence criteria: Every implementation dispatch records provider executable/capability/budget and task-revision identity; Codex fallback occurs only for an allowed typed reason before any accepted effect; its attempt cannot attest its own review; post-effect, repeated, over-budget, scope-widening and prompt-selected fallback fail closed.
- Evidence source policy: CLI availability, environment selection, command construction or provider text alone is non-authoritative; evidence is the admitted packet route, committed fallback receipt, exact observed-effect boundary and distinct review authorization.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py, test/api/test_agent_supervisor_typed_provider_fallback.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py", "ipfs_accelerate_py/agent_supervisor/todo_daemon/contract_packet_provider_router.py", "test/api/test_agent_supervisor_typed_provider_fallback.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_typed_provider_fallback.py test/api/test_agent_supervisor_production_provider_route.py test/api/test_agent_supervisor_contract_packet_provider_router.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_typed_provider_fallback.py test/api/test_agent_supervisor_production_provider_route.py test/api/test_agent_supervisor_contract_packet_provider_router.py -q"]
- Acceptance: The supervisor defaults to Grok and safely falls back to Codex for implementation without losing packet admission, effect idempotency, budget/scope bounds or independent review.
- Gap task: Repair the smallest provider capability, typed reason, attempt identity, accepted-effect, fallback dispatch or review-separation residual.

## ASE-G050 Python, CLI, MCP, and MCP++ entrypoint parity

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G040
- Dependencies JSON: ["ASE-G040"]
- Fib priority: 13
- Track: public-entrypoints
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/surfaces
- Parallel lane: surface-conformance
- Resource class: cpu-medium
- Goal: Publish the same prompt-first run, preview, steer, status, follow, explain and doctor contracts through Python, product CLI, MCP and MCP++, while retaining expert operations and cold discovery.
- Producing tasks: ASE-018, ASE-019, ASE-020, ASE-021, ASE-022
- Evidence: entrypoint_conformance.ENTRYPOINT_TRANSPORT_CONFORMANCE_REQUIREMENT_ID
- Evidence requirements JSON: ["Python API tests", "installed CLI subprocess tests", "MCP schema and invocation tests", "MCP++ UCAN tests", "canonical cross-transport fixtures"]
- Evidence criteria: Every surface uses one intent service and returns equivalent canonical results/errors/effects/cursors; imports/help/tool listing are cold; low-level expert commands remain unchanged; MCP roots and identity remain server-owned.
- Evidence source policy: Matching names, schemas, HTTP success or individually passing adapters is non-authoritative; evidence is canonical fixture equivalence joined to exact request and runtime roots.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/python_api.py, ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoint_tools.py, ipfs_accelerate_py/mcp_server/tools/mcplusplus/agent_supervisor_tools.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/python_api.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py", "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoint_tools.py", "ipfs_accelerate_py/mcp_server/tools/mcplusplus/agent_supervisor_tools.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_python.py test/api/test_agent_supervisor_prompt_only_cli.py test/api/test_agent_supervisor_prompt_only_mcp.py test/api/test_agent_supervisor_prompt_only_mcplusplus.py test/api/test_agent_supervisor_prompt_only_conformance.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_python.py test/api/test_agent_supervisor_prompt_only_cli.py test/api/test_agent_supervisor_prompt_only_mcp.py test/api/test_agent_supervisor_prompt_only_mcplusplus.py test/api/test_agent_supervisor_prompt_only_conformance.py -q"]
- Acceptance: The same prompt and authenticated target context yields the same run/steer outcome through every supported surface without requiring low-level bindings.
- Gap task: Close the smallest schema, naming, parser, tool registration, identity, allowlist, result, cursor, lazy-loading or conformance residual.

## ASE-G051 Python convenience API

- Status: active
- Parent: ASE-G050
- Parent goal IDs JSON: ["ASE-G050"]
- Depends on: ASE-G042, ASE-G043
- Dependencies JSON: ["ASE-G042", "ASE-G043"]
- Fib priority: 3
- Track: python-entrypoint
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/python
- Parallel lane: python-surface
- Resource class: cpu-small
- Goal: Expose `Supervisor.open`, `run`, `preview`, `steer`, `status`, `follow`, `explain`, `doctor` and `resume` as lazy typed Python APIs.
- Producing tasks: ASE-018
- Evidence: python_api.PROMPT_ONLY_PYTHON_API_REQUIREMENT_ID
- Evidence requirements JSON: ["public import test", "embedded explicit-service test", "inferred local-service test"]
- Evidence criteria: Cold import has no side effects; embedders can supply explicit authority/stores; local open uses the same resolver/runtime; returned handles are typed and resumable.
- Evidence source policy: Symbol export or successful import is non-authoritative; evidence is direct live API behavior and canonical equality with the shared service.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/python_api.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/python_api.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_python.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_python.py -q"]
- Acceptance: Python callers can start and steer from a prompt with defaults or embed the exact same service with explicit policy and stores.
- Gap task: Repair the smallest public method, lazy import, embedding or run-handle residual.

## ASE-G052 Prompt-first CLI

- Status: active
- Parent: ASE-G050
- Parent goal IDs JSON: ["ASE-G050"]
- Depends on: ASE-G042, ASE-G043
- Dependencies JSON: ["ASE-G042", "ASE-G043"]
- Fib priority: 3
- Track: cli-entrypoint
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/cli
- Parallel lane: cli-surface
- Resource class: cpu-small
- Goal: Add `ipfs-accelerate supervisor` and compatible `agent`/console aliases with prompt positional/file/stdin input, inferred active target/run and concise human plus canonical JSON output.
- Producing tasks: ASE-019
- Evidence: entrypoint_cli.PROMPT_ONLY_CLI_REQUIREMENT_ID
- Evidence requirements JSON: ["installed console subprocess test", "help/discovery cold test", "prompt argv and log leak test", "expert command compatibility"]
- Evidence criteria: Common run needs only prompt; status/follow/steer infer a unique run; sensitive input has safe file/stdin path; canonical request mode bypasses inference; exit codes and old commands remain stable.
- Evidence source policy: Parser success and mocked service output are non-authoritative; evidence invokes the installed product command against real fixture state and subprocesses.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/cli.py, pyproject.toml, setup.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py", "ipfs_accelerate_py/cli.py", "pyproject.toml", "setup.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_cli.py test/api/test_agent_supervisor_prompt_cli.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_cli.py test/api/test_agent_supervisor_prompt_cli.py -q"]
- Acceptance: A newly installed command can start, inspect, follow and steer a local run without low-level flags while exact legacy CLI behavior remains available.
- Gap task: Close the smallest parser, dispatch, prompt source, output, exit, alias, packaging or compatibility residual.

## ASE-G053 MCP convenience tools

- Status: active
- Parent: ASE-G050
- Parent goal IDs JSON: ["ASE-G050"]
- Depends on: ASE-G042, ASE-G043
- Dependencies JSON: ["ASE-G042", "ASE-G043"]
- Fib priority: 3
- Track: mcp-entrypoint
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/mcp
- Parallel lane: mcp-surface
- Resource class: network-small
- Goal: Add lazy MCP tools accepting a minimal prompt/run request and dispatching to a server-configured standard runtime without weakening the canonical low-level tool set.
- Producing tasks: ASE-020
- Evidence: mcp_entrypoints.PROMPT_ONLY_MCP_REQUIREMENT_ID
- Evidence requirements JSON: ["closed MCP schemas", "configured-root target tests", "lazy registration test", "low-level tool compatibility"]
- Evidence criteria: Tool requests cannot configure server allowlists or caller; one target is selected only within server policy; prompt bodies do not enter routine logs; results match Python service records.
- Evidence source policy: Tool listing, schema generation or HTTP response alone is non-authoritative; evidence is live configured invocation and canonical parity.
- Outputs: ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoint_tools.py
- Predicted files JSON: ["ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoint_tools.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_mcp.py test/api/test_agent_supervisor_native_mcp_discovery.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_mcp.py test/api/test_agent_supervisor_native_mcp_discovery.py -q"]
- Acceptance: An authenticated MCP client can run and steer a server-allowlisted target with a minimal request and cannot widen roots, identity, policy or effects.
- Gap task: Repair the smallest tool, schema, service resolution, lazy import, target, prompt, or parity residual.

## ASE-G054 MCP++ and UCAN binding

- Status: active
- Parent: ASE-G050
- Parent goal IDs JSON: ["ASE-G050"]
- Depends on: ASE-G033, ASE-G043, ASE-G053
- Dependencies JSON: ["ASE-G033", "ASE-G043", "ASE-G053"]
- Fib priority: 5
- Track: mcplusplus-entrypoint
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/mcplusplus
- Parallel lane: mcplusplus-surface
- Resource class: network-small
- Goal: Publish supervisor-specific MCP++ run/steer/status contracts and validate UCAN invocation capability separately from the exact inner mutation authorization.
- Producing tasks: ASE-021
- Evidence: mcplusplus_entrypoints.PROMPT_ONLY_MCPPLUSPLUS_UCAN_REQUIREMENT_ID
- Evidence requirements JSON: ["IDL/tool schema tests", "issuer audience expiry revocation attenuation tests", "inner authorization separation test"]
- Evidence criteria: UCAN scope covers target, operation and effects; forged/stale/overbroad tokens fail; bearer material is not persisted; transport capability never substitutes for root/effect/lease/fence-bound authorization.
- Evidence source policy: Token presence, decoded claims or MCP tool permission alone is non-authoritative; evidence is cryptographic/capability verification plus a distinct canonical mutation decision and observed effect.
- Outputs: ipfs_accelerate_py/mcp_server/tools/mcplusplus/agent_supervisor_tools.py
- Predicted files JSON: ["ipfs_accelerate_py/mcp_server/tools/mcplusplus/agent_supervisor_tools.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_mcplusplus.py test/api/test_mcp_server_mcplusplus_ucan.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_mcplusplus.py test/api/test_mcp_server_mcplusplus_ucan.py -q"]
- Acceptance: MCP++ clients can invoke the same run/steer API under attenuated UCAN permissions without obtaining broader repository or mutation authority.
- Gap task: Close the smallest IDL, tool, token, scope, revocation, attenuation, inner-decision or persistence residual.

## ASE-G055 Cross-transport conformance

- Status: active
- Parent: ASE-G050
- Parent goal IDs JSON: ["ASE-G050"]
- Depends on: ASE-G051, ASE-G052, ASE-G053, ASE-G054
- Dependencies JSON: ["ASE-G051", "ASE-G052", "ASE-G053", "ASE-G054"]
- Fib priority: 8
- Track: transport-conformance
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/conformance
- Parallel lane: transport-conformance
- Resource class: cpu-medium
- Goal: Prove equivalent requests produce equivalent resolution, run, steering, error, effect and cursor records across all high-level transports.
- Producing tasks: ASE-022
- Evidence: entrypoint_conformance.ENTRYPOINT_TRANSPORT_CONFORMANCE_REQUIREMENT_ID
- Evidence requirements JSON: ["closed success and failure fixture population", "canonical byte equality", "cold discovery parity"]
- Evidence criteria: Success, rejection, ambiguity, denial, stale replay, partial saga and cancellation populations agree; differences are limited to declared transport metadata; discovery has no effects.
- Evidence source policy: Separate per-adapter tests are non-authoritative; evidence is one joined conformance report over every transport and fixture.
- Outputs: test/api/test_agent_supervisor_prompt_only_conformance.py
- Predicted files JSON: ["test/api/test_agent_supervisor_prompt_only_conformance.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_conformance.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_conformance.py -q"]
- Acceptance: No transport adds authority, changes semantics, loses required identity, or returns a canonical-looking result for another request.
- Gap task: Add the smallest missing shared fixture or repair the first canonical mismatch.

## ASE-G060 Semantic, revisioned steering

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G020, ASE-G040
- Dependencies JSON: ["ASE-G020", "ASE-G040"]
- Fib priority: 13
- Track: semantic-steering
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/steering
- Parallel lane: steering-integration
- Resource class: cpu-medium
- Goal: Convert a bounded follow-up instruction into an immutable revision-bound steering event and admitted plan delta or lifecycle operation that preserves accepted/claimed history and safely updates live work.
- Producing tasks: ASE-023, ASE-024, ASE-025
- Evidence: steering_runtime.SEMANTIC_STEERING_REQUIREMENT_ID
- Evidence requirements JSON: ["closed steering-intent parser tests", "claimed/completed immutability tests", "plan-delta projection parity", "concurrent CAS and replay tests"]
- Evidence criteria: Exact run and revisions are pinned; ambiguous material mutations ask; unstarted work may change through a child revision; claimed work receives successors; stale/concurrent conflicts fail typed; replay is idempotent.
- Evidence source policy: User prose, model classification, task status text or event delivery alone is non-authoritative; evidence is deterministic classification/admission, exact delta transaction and post-update run/task-source receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py, ipfs_accelerate_py/agent_supervisor/entrypoints/steering_runtime.py, ipfs_accelerate_py/agent_supervisor/entrypoints/steering_concurrency.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/steering_runtime.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/steering_concurrency.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_steering_contracts.py test/api/test_agent_supervisor_steering_runtime.py test/api/test_agent_supervisor_steering_concurrency.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_steering_contracts.py test/api/test_agent_supervisor_steering_runtime.py test/api/test_agent_supervisor_steering_concurrency.py -q"]
- Acceptance: A user can steer a running supervisor with one instruction while history, accepted evidence, active attempts, authority and parallel safety remain intact under replay and concurrency.
- Gap task: Close the smallest intent, ambiguity, delta, lifecycle, task immutability, CAS, event or replay residual.

## ASE-G061 Steering contracts and intent classification

- Status: active
- Parent: ASE-G060
- Parent goal IDs JSON: ["ASE-G060"]
- Depends on: ASE-G020
- Dependencies JSON: ["ASE-G020"]
- Fib priority: 3
- Track: steering-contracts
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/steering-contracts
- Parallel lane: steering-contracts
- Resource class: cpu-small
- Goal: Define a closed steering vocabulary and strict records for append, answer, narrow, reprioritize, replan, pause, resume, cancel and status intents.
- Producing tasks: ASE-023
- Evidence: steering_contracts.STEERING_CONTRACT_REQUIREMENT_ID
- Evidence requirements JSON: ["canonical records", "deterministic and model-assisted classification", "material ambiguity questions"]
- Evidence criteria: Unknown/unbounded instructions fail; model output is proposal only; lifecycle-affecting classifications require exact authority; ambiguous materially different actions are not guessed.
- Evidence source policy: Classifier confidence or plausible text is non-authoritative; evidence is strict parse, deterministic policy and admitted effect classification.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_steering_contracts.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_steering_contracts.py -q"]
- Acceptance: Every steer has one bounded closed interpretation, a typed question, or a rejection; text cannot directly mutate state.
- Gap task: Add the smallest missing intent, bound, ambiguity, schema or classification fixture.

## ASE-G062 Plan-delta and live-run integration

- Status: active
- Parent: ASE-G060
- Parent goal IDs JSON: ["ASE-G060"]
- Depends on: ASE-G041, ASE-G042, ASE-G061
- Dependencies JSON: ["ASE-G041", "ASE-G042", "ASE-G061"]
- Fib priority: 5
- Track: steering-runtime
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/steering-runtime
- Parallel lane: steering-runtime
- Resource class: cpu-medium
- Goal: Snapshot exact live state, reuse the create/steer plan-delta design, admit changes, preserve claimed/completed work, and CAS-publish child plan/task-source revisions.
- Producing tasks: ASE-024
- Evidence: steering_runtime.PLAN_DELTA_RUNTIME_REQUIREMENT_ID
- Evidence requirements JSON: ["mixed lifecycle population tests", "Markdown DuckDB parity", "event notification and run revision"]
- Evidence criteria: Completed/accepted and claimed specs never change; successors/deferred work preserve intent; dependencies remain acyclic and schedulable; dual projections and run registry publish one exact child revision.
- Evidence source policy: Changed Markdown text, DuckDB row count, model delta or lane observation is non-authoritative; evidence is canonical plan-delta admission and fenced transaction receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_runtime.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/steering_runtime.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_steering_runtime.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_steering_runtime.py -q"]
- Acceptance: Steering updates only permitted work through a traceable child revision and live lanes observe exactly the accepted new task population.
- Gap task: Repair the smallest snapshot, delta, immutability, projection, transaction, run or event residual.

## ASE-G063 Concurrent steering, CAS, and replay

- Status: active
- Parent: ASE-G060
- Parent goal IDs JSON: ["ASE-G060"]
- Depends on: ASE-G043, ASE-G062
- Dependencies JSON: ["ASE-G043", "ASE-G062"]
- Fib priority: 8
- Track: steering-concurrency
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/steering-concurrency
- Parallel lane: steering-concurrency
- Resource class: cpu-medium
- Goal: Serialize conflicting steers by run/task-source revision and lease/fence, replay identical instructions, and preserve independent non-conflicting updates.
- Producing tasks: ASE-025
- Evidence: steering_concurrency.STEERING_CONCURRENCY_REQUIREMENT_ID
- Evidence requirements JSON: ["simultaneous identical and conflicting requests", "stale lease/fence cases", "crash and replay cases"]
- Evidence criteria: At most one conflicting revision wins; stale requests return current context; identical replay has no duplicate provider/write/event effects; independent changes retain deterministic ordering.
- Evidence source policy: Lock acquisition or final row state alone is non-authoritative; evidence is joined request, lease, CAS, transaction and event receipts.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/steering_concurrency.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/steering_concurrency.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_steering_concurrency.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_steering_concurrency.py -q"]
- Acceptance: Concurrent steering cannot lose history, split the plan/task-source/run roots, duplicate effects, or let a stale owner publish.
- Gap task: Add the smallest missing interleaving, crash point, stale generation or replay fixture.

## ASE-G070 Authority, scope, and adversarial safety

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G033, ASE-G042, ASE-G054, ASE-G062
- Dependencies JSON: ["ASE-G033", "ASE-G042", "ASE-G054", "ASE-G062"]
- Fib priority: 21
- Track: entrypoint-security
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/security
- Parallel lane: security-gate
- Resource class: cpu-medium
- Goal: Prove prompts, repository state, profiles, transports, steering and degraded dependencies cannot widen roots, identity, policy, authority, commands, effects, completion, leases or secret exposure.
- Producing tasks: ASE-026
- Evidence: prompt_only_security.PROMPT_ONLY_SECURITY_REQUIREMENT_ID
- Evidence requirements JSON: ["closed adversarial population", "secret non-leak scan", "authority/effect non-escape receipt", "stale and forged identity matrix"]
- Evidence criteria: Every seeded prompt/config/path/symlink/submodule/profile/provider/UCAN/authorization/CID/lease/fence/completion attack is rejected or safely represented; no secret appears in durable inspected surfaces.
- Evidence source policy: Unit assertions over isolated validators or aggregate pass rate is non-authoritative; evidence covers the end-to-end entrypoint and effect boundary for every frozen attack case.
- Outputs: test/api/test_agent_supervisor_prompt_only_security.py, test/fixtures/agent_supervisor_prompt_entrypoints/adversarial
- Predicted files JSON: ["test/api/test_agent_supervisor_prompt_only_security.py", "test/fixtures/agent_supervisor_prompt_entrypoints/adversarial"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_security.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_security.py -q"]
- Acceptance: The complete adversarial population has zero scope, identity, policy, authority, command, effect, completion, lease/fence or secret escapes.
- Gap task: Add the smallest missing attack class or repair the first escape.

## ASE-G080 Status, follow, explain, doctor, and observability

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G030, ASE-G040, ASE-G060
- Dependencies JSON: ["ASE-G030", "ASE-G040", "ASE-G060"]
- Fib priority: 21
- Track: entrypoint-observability
- Priority: P1
- Bundle: agent-supervisor/prompt-entrypoints/observability
- Parallel lane: observability
- Resource class: cpu-small
- Goal: Make inferred configuration, current saga/run state, events, health, blockers, provider/resources, lease state, recovery and next actions understandable without low-level IDs or state paths.
- Producing tasks: ASE-027, ASE-028
- Evidence: entrypoint_observability.ENTRYPOINT_OBSERVABILITY_REQUIREMENT_ID
- Evidence requirements JSON: ["body-free explain output", "doctor capability matrix", "event follow and resume tests", "bounded metric cardinality"]
- Evidence criteria: Unique current runs need no IDs; ambiguous targets are explicit; follow resumes by cursor; explain names every source; doctor distinguishes required/degraded/optional; metrics expose latency, inference and recovery without secrets or unbounded labels.
- Evidence source policy: Human summaries and process status alone are non-authoritative; evidence is canonical run/event/capability/decision records plus leak and bounds tests.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/status_explain.py, ipfs_accelerate_py/agent_supervisor/entrypoints/doctor.py, ipfs_accelerate_py/agent_supervisor/entrypoints/metrics.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/status_explain.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/doctor.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/metrics.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_status.py test/api/test_agent_supervisor_prompt_only_doctor.py test/api/test_agent_supervisor_prompt_only_metrics.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_status.py test/api/test_agent_supervisor_prompt_only_doctor.py test/api/test_agent_supervisor_prompt_only_metrics.py -q"]
- Acceptance: A user can understand and follow a run, its inferred setup and any blocker from the high-level surfaces without reading implementation state or exposing sensitive data.
- Gap task: Close the smallest selection, cursor, explanation, doctor, metric, bound or leak residual.

## ASE-G090 E2E, chaos, load, and quantitative gates

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G050, ASE-G060, ASE-G070, ASE-G080
- Dependencies JSON: ["ASE-G050", "ASE-G060", "ASE-G070", "ASE-G080"]
- Fib priority: 34
- Track: entrypoint-verification
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/verification
- Parallel lane: verification
- Resource class: cpu-large
- Goal: Verify inference components, real installed prompt-to-run behavior, transport parity, crash recovery, concurrent load and published promotion thresholds on a closed representative population.
- Producing tasks: ASE-029, ASE-030, ASE-031
- Evidence: prompt_only_entrypoint_rollout.PROMPT_ONLY_ENTRYPOINT_ROLLOUT_REQUIREMENT_ID
- Evidence requirements JSON: ["unit/property report", "real subprocess E2E and transport report", "chaos/load benchmark and rollout decision"]
- Evidence criteria: At least 95 percent supported single-target runs reach healthy work from prompt-only input; unchanged resolution replay is exact; no unauthorized effects/leaks/duplicate trees occur; transport parity and bounded resource/latency gates pass.
- Evidence source policy: Mock-only E2E, selected successful runs, averages without full population, model judgment or historical reports are non-authoritative; evidence is a frozen complete paired/adversarial/chaos/load population with exact failures and confidence bounds.
- Outputs: test/api/test_agent_supervisor_prompt_only_inference.py, test/api/test_agent_supervisor_prompt_only_e2e.py, test/api/test_agent_supervisor_prompt_only_chaos.py, ipfs_accelerate_py/agent_supervisor/entrypoints/benchmark.py, ipfs_accelerate_py/agent_supervisor/entrypoints/rollout.py
- Predicted files JSON: ["test/api/test_agent_supervisor_prompt_only_inference.py", "test/api/test_agent_supervisor_prompt_only_e2e.py", "test/api/test_agent_supervisor_prompt_only_chaos.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/benchmark.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/rollout.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_inference.py test/api/test_agent_supervisor_prompt_only_e2e.py test/api/test_agent_supervisor_prompt_only_chaos.py test/api/test_agent_supervisor_prompt_only_rollout.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_inference.py test/api/test_agent_supervisor_prompt_only_e2e.py test/api/test_agent_supervisor_prompt_only_chaos.py test/api/test_agent_supervisor_prompt_only_rollout.py -q"]
- Acceptance: Every published quantitative gate is recomputed from the full current-root population, failures are visible, and automatic rollout is denied on any binding, authority, parity, leak, duplicate-process or resource regression.
- Gap task: Add the smallest missing topology, transport, fault, load level, metric or promotion assertion.

## ASE-G100 Compatibility migration, documentation, rollout, and closeout

- Status: active
- Parent: ASE-G000
- Parent goal IDs JSON: ["ASE-G000"]
- Depends on: ASE-G050, ASE-G070, ASE-G080, ASE-G090
- Dependencies JSON: ["ASE-G050", "ASE-G070", "ASE-G080", "ASE-G090"]
- Fib priority: 55
- Track: entrypoint-rollout
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints/rollout
- Parallel lane: rollout
- Resource class: coordinator
- Goal: Project legacy launchers and state into profiles without destructive migration, publish operator/developer/security/runbook documentation, stage rollout with automatic rollback, and close every acceptance criterion with fresh evidence.
- Producing tasks: ASE-032, ASE-033, ASE-034
- Evidence: prompt_only_entrypoint_rollout.PROMPT_ONLY_ENTRYPOINT_ROLLOUT_REQUIREMENT_ID
- Evidence requirements JSON: ["legacy compatibility and migration receipts", "operator and API documentation checks", "fresh independent closeout report"]
- Evidence criteria: Expert commands and old state remain usable; profile import is previewed and identity-checked; console registration is coherent; docs show prompt-only and override paths; rollout modes and rollback triggers are live; every root criterion maps to current evidence.
- Evidence source policy: Documentation completion, migration script exit, task status or same-run benchmark is non-authoritative; evidence is canonical compatibility fixtures, non-destructive migration receipts and a later separate current-root rollout evaluation.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/migration.py, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md, docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/migration.py", "docs/guides/AGENT_SUPERVISOR_GUIDE.md", "docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md", "docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_compatibility.py test/api/test_agent_supervisor_prompt_only_rollout.py test/api/test_agent_supervisor_docs.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_compatibility.py test/api/test_agent_supervisor_prompt_only_rollout.py test/api/test_agent_supervisor_docs.py -q"]
- Acceptance: Prompt-only entrypoints are documented and safely promoted, old expert workflows retain canonical behavior, migration is recoverable, and no mandatory goal remains incomplete or supported only by stale/self-reported evidence.
- Gap task: Close the smallest compatibility, migration, packaging, documentation, rollout, rollback or evidence-mapping residual.
