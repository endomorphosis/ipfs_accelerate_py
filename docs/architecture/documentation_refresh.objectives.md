# IPFS Accelerate documentation refresh objective heap

This is the durable goal/subgoal hierarchy for
[`DOCUMENTATION_REFRESH_PLAN_2026_08.md`](DOCUMENTATION_REFRESH_PLAN_2026_08.md).
The executable projection is
[`documentation_refresh.todo.md`](documentation_refresh.todo.md), with task
prefix `## DOC-`.

## Goal tree

```text
DOC-G000  Accurate, rationale-rich documentation
|-- DOC-G010  Evidence baseline and documentation governance
|   |-- DOC-G011  Reproducible drift inventory
|   |-- DOC-G012  Lifecycle, status and source-of-truth policy
|   `-- DOC-G013  Architecture-writing and decision-record conventions
|-- DOC-G020  Maintained product architecture guides
|   |-- DOC-G021  System context and inference runtime
|   |-- DOC-G022  Model/service and MCP control surfaces
|   `-- DOC-G023  IPFS/P2P and cross-repository boundaries
|-- DOC-G030  Agent-supervisor architecture guides
|   |-- DOC-G031  Intent, control and authority
|   |-- DOC-G032  Planning, context, analysis and proof
|   |-- DOC-G033  Scheduling, execution, merge and recovery
|   `-- DOC-G034  Prompt-first runtime, state and steering
|-- DOC-G040  Architectural decision records
|   |-- DOC-G041  Intent, trust and evidence decisions
|   |-- DOC-G042  Capability and isolation decisions
|   `-- DOC-G043  Persistence, package and compatibility decisions
|-- DOC-G050  Current user, developer and operator journeys
|   |-- DOC-G051  Installation, quickstart, Python and CLI
|   |-- DOC-G052  MCP and supervisor operation
|   `-- DOC-G053  Deployment, hardware, P2P, testing and troubleshooting
`-- DOC-G060  Navigation, verification and sustainable closeout
    |-- DOC-G061  Glossary, audience routes and archive taxonomy
    `-- DOC-G062  Link/example verification and freshness publication
```

## DOC-G000 Accurate, rationale-rich documentation

- Status: completed
- Parent:
- Parent goal IDs JSON: []
- Depends on:
- Dependencies JSON: []
- Fib priority: 1
- Track: documentation-program
- Priority: P0
- Bundle: documentation-refresh/root
- Parallel lane: program
- Resource class: coordinator
- Goal: Bring maintained documentation into agreement with the current repository and explain the system's bespoke architectural choices, boundaries, tradeoffs and failure semantics to users, developers, operators and implementation agents.
- Producing tasks: DOC-001, DOC-002, DOC-003, DOC-004, DOC-005, DOC-006, DOC-007, DOC-008, DOC-009, DOC-010, DOC-011, DOC-012, DOC-013, DOC-014, DOC-015, DOC-016, DOC-017, DOC-018, DOC-019, DOC-020, DOC-021, DOC-022, DOC-023, DOC-024, DOC-025, DOC-026, DOC-027, DOC-028
- Evidence: dated drift audit, maintained architecture set, decision records, refreshed journeys, validation report
- Evidence requirements JSON: ["current-tree documentation inventory", "source-to-document authority map", "rationale-bearing architecture guides", "current executable journey checks", "local link and path validation", "dated closeout matrix"]
- Evidence criteria: All child goals are terminal on one recorded tree; maintained guides distinguish current, planned and historical behavior; every major subsystem and bespoke decision has source anchors, trust and failure boundaries, rationale, extension constraints and a reproducible verification recipe; supported entrypoints and local links pass offline checks.
- Evidence source policy: Model prose, old completion reports, taskboard status, import success, package presence, a PID and a merged branch are non-authoritative by themselves. Evidence comes from the exact repository tree, current source and schemas, executable help, focused tests, local-link checks and post-merge documentation receipts.
- Outputs: docs/
- Predicted files JSON: ["docs/"]
- Validation: git diff --check && python scripts/docs/check_agent_supervisor_docs.py
- Validation commands JSON: ["git diff --check", "python scripts/docs/check_agent_supervisor_docs.py"]
- Acceptance: A reader can find the canonical current guide for each supported subsystem, understand why its major boundaries exist, follow verified user/operator paths, and distinguish capability, proposal, validation, proof and historical claims without reading source blindly.
- Gap task: Execute the ready DOC task population by dependency, file ownership, task shard and validation policy.
- Refinement: Add bounded follow-up work only when closeout evidence identifies a concrete current-doc gap not represented by the sealed board.
- Embedding query: ipfs accelerate documentation architecture rationale inference routing MCP IPFS P2P agent supervisor objectives evidence worktrees leases fencing prompt steering
- AST query: ipfs_accelerate_py model_catalog endpoint_usage mcp_server p2p_tasks agent_supervisor entrypoints

## DOC-G010 Evidence baseline and documentation governance

- Status: completed
- Parent: DOC-G000
- Parent goal IDs JSON: ["DOC-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 2
- Track: documentation-governance
- Priority: P0
- Bundle: documentation-refresh/governance
- Parallel lane: governance
- Resource class: io-small
- Goal: Establish reproducible current-tree evidence, document ownership, lifecycle labels and writing conventions before narrative refresh work fans out.
- Producing tasks: DOC-001, DOC-002, DOC-003, DOC-004
- Evidence: four reviewed governance artifacts bound to the baseline tree
- Evidence requirements JSON: ["drift inventory", "lifecycle and authority policy", "architecture guide contract", "ADR index and template"]
- Evidence criteria: Inventory separates maintained, reference, plan, historical and vendored content; every claim family has a source of truth and audit trigger; architecture and ADR templates require rationale, alternatives, consequences and verification.
- Evidence source policy: Filename tokens and document self-claims alone are non-authoritative; classifications are checked against navigation, source anchors, Git history and present implementation.
- Outputs: docs/development/, docs/architecture/GUIDE_CONVENTIONS.md, docs/architecture/decisions/
- Predicted files JSON: ["docs/development/", "docs/architecture/GUIDE_CONVENTIONS.md", "docs/architecture/decisions/"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: Independent writer lanes share one explicit content, status, source, diagram, link and decision-record contract.
- Gap task: Close the smallest missing inventory, ownership, lifecycle or convention requirement.

## DOC-G011 Reproducible drift inventory

- Status: completed
- Parent: DOC-G010
- Parent goal IDs JSON: ["DOC-G010"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 2
- Track: documentation-audit
- Priority: P0
- Bundle: documentation-refresh/drift-audit
- Parallel lane: drift-audit
- Resource class: io-medium
- Goal: Record what changed since the published July documentation baselines and prioritize concrete stale, missing, contradictory and broken guidance.
- Producing tasks: DOC-001
- Evidence: docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md
- Evidence requirements JSON: ["baseline and head", "changed subsystem inventory", "broken command and path findings", "known source contradictions", "prioritized ownership matrix"]
- Evidence criteria: The audit identifies prompt entrypoint, contract-assurance, acceptance, routing, CID, persistence and integration changes; records stale ASE board state without using it as completion authority; and names exact source/doc anchors for each finding.
- Evidence source policy: Git diff, current source, tests and executable help are authoritative within the recorded checkout; old board status is only a drift signal.
- Outputs: docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md
- Predicted files JSON: ["docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md"]
- Validation: test -f docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md && git diff --check
- Validation commands JSON: ["test -f docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md", "git diff --check"]
- Acceptance: Another maintainer can reproduce the baseline comparison and route every P0/P1 documentation gap to an owner without rediscovering the repository.
- Gap task: Add the smallest missing high-impact drift finding with exact evidence.

## DOC-G012 Lifecycle, status and source-of-truth policy

- Status: completed
- Parent: DOC-G010
- Parent goal IDs JSON: ["DOC-G010"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 3
- Track: documentation-governance
- Priority: P0
- Bundle: documentation-refresh/lifecycle
- Parallel lane: lifecycle
- Resource class: cpu-small
- Goal: Define maintained, reference, plan, historical, generated and vendored classifications plus ownership, freshness, supersession and archive rules.
- Producing tasks: DOC-002
- Evidence: docs/development/DOCUMENTATION_LIFECYCLE.md
- Evidence requirements JSON: ["closed status vocabulary", "source-to-doc authority matrix", "freshness triggers", "archive and supersession policy", "known code contradiction policy"]
- Evidence criteria: A document cannot declare itself current merely by filename; plans cannot masquerade as landed behavior; source inconsistencies such as version disagreement are recorded as code-owned blockers rather than papered over.
- Evidence source policy: Classification decisions cite current navigation and implementation evidence and are reviewable rather than inferred solely from age.
- Outputs: docs/development/DOCUMENTATION_LIFECYCLE.md
- Predicted files JSON: ["docs/development/DOCUMENTATION_LIFECYCLE.md"]
- Validation: test -f docs/development/DOCUMENTATION_LIFECYCLE.md && git diff --check
- Validation commands JSON: ["test -f docs/development/DOCUMENTATION_LIFECYCLE.md", "git diff --check"]
- Acceptance: Authors and agents have a deterministic rule for where information belongs, what overrides it and when it must be revalidated.
- Gap task: Close the smallest missing status, ownership, freshness or supersession rule.

## DOC-G013 Architecture-writing and decision-record conventions

- Status: completed
- Parent: DOC-G010
- Parent goal IDs JSON: ["DOC-G010"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 5
- Track: architecture-governance
- Priority: P0
- Bundle: documentation-refresh/architecture-conventions
- Parallel lane: architecture-conventions
- Resource class: cpu-small
- Goal: Give parallel writers one guide structure, vocabulary, source-anchor policy and architecture decision record format.
- Producing tasks: DOC-003, DOC-004
- Evidence: docs/architecture/GUIDE_CONVENTIONS.md and docs/architecture/decisions/
- Evidence requirements JSON: ["guide outline", "diagram vocabulary", "normative language policy", "ADR template", "decision index"]
- Evidence criteria: Conventions require status, audience, scope, source anchors, current/planned separation, rationale, alternatives, consequences, trust/failure semantics and verification; ADR filenames and statuses are collision-safe.
- Evidence source policy: Conventions preserve repository terminology and do not establish new API guarantees.
- Outputs: docs/architecture/GUIDE_CONVENTIONS.md, docs/architecture/decisions/README.md, docs/architecture/decisions/0000-template.md
- Predicted files JSON: ["docs/architecture/GUIDE_CONVENTIONS.md", "docs/architecture/decisions/README.md", "docs/architecture/decisions/0000-template.md"]
- Validation: test -f docs/architecture/GUIDE_CONVENTIONS.md && test -f docs/architecture/decisions/0000-template.md && git diff --check
- Validation commands JSON: ["test -f docs/architecture/GUIDE_CONVENTIONS.md", "test -f docs/architecture/decisions/0000-template.md", "git diff --check"]
- Acceptance: Two independent writers can produce consistent guides and ADRs without editing the same shared file.
- Gap task: Add the smallest missing guide or ADR convention.

## DOC-G020 Maintained product architecture guides

- Status: completed
- Parent: DOC-G000
- Parent goal IDs JSON: ["DOC-G000"]
- Depends on: DOC-G010
- Dependencies JSON: ["DOC-G010"]
- Fib priority: 3
- Track: product-architecture
- Priority: P0
- Bundle: documentation-refresh/product-architecture
- Parallel lane: product-architecture
- Resource class: cpu-medium
- Goal: Describe the maintained product as cooperating inference, service, MCP, storage and distributed-execution planes with explicit ownership and capability boundaries.
- Producing tasks: DOC-005, DOC-006, DOC-007, DOC-008, DOC-009, DOC-010
- Evidence: six source-anchored architecture guides
- Evidence requirements JSON: ["system context", "inference flow", "catalog and endpoint routing", "MCP runtime", "IPFS and P2P runtime", "cross-repository ownership"]
- Evidence criteria: Guides cover model_catalog, endpoint_usage, modality routers, voice jobs/providers, hf_model_server, CLI runtime, canonical MCP server, compatibility facade, IPFS backend roles, CID semantics, P2P tasks and sibling integrations without claiming optional capability from importability.
- Evidence source policy: Current package code, schemas, tests and help are authoritative; delivery plans are historical/planned context only.
- Outputs: docs/architecture/
- Predicted files JSON: ["docs/architecture/"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: A developer can trace a request and its state across every product plane and know which component owns each decision.
- Gap task: Close the smallest missing component, flow, rationale or boundary in the named guide set.

## DOC-G021 System context and inference runtime

- Status: completed
- Parent: DOC-G020
- Parent goal IDs JSON: ["DOC-G020"]
- Depends on: DOC-G011, DOC-G013
- Dependencies JSON: ["DOC-G011", "DOC-G013"]
- Fib priority: 3
- Track: runtime-architecture
- Priority: P0
- Bundle: documentation-refresh/system-inference
- Parallel lane: system-inference
- Resource class: cpu-medium
- Goal: Explain actors, containers, the inference/data plane, the separate supervisor/control plane and the end-to-end inference/router lifecycle.
- Producing tasks: DOC-005, DOC-006
- Evidence: docs/architecture/SYSTEM_CONTEXT.md, docs/architecture/overview.md, docs/architecture/INFERENCE_RUNTIME.md
- Evidence requirements JSON: ["context map", "container boundaries", "inference sequence", "optional capability checks", "failure and fallback flow"]
- Evidence criteria: Conceptual names map to live packages; current and legacy entrypoints are distinguished; service selection, model execution and result/error flow are source-anchored.
- Evidence source policy: `ipfs_accelerate.py`, routers, backends, worker and CLI/runtime code plus focused tests are primary.
- Outputs: docs/architecture/SYSTEM_CONTEXT.md, docs/architecture/overview.md, docs/architecture/INFERENCE_RUNTIME.md
- Predicted files JSON: ["docs/architecture/SYSTEM_CONTEXT.md", "docs/architecture/overview.md", "docs/architecture/INFERENCE_RUNTIME.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: The overview is accurate at one screen and the deeper guides answer both how a request moves and why inference and supervisor concerns remain separate.
- Gap task: Close the smallest system-context or inference-flow residual.

## DOC-G022 Model/service and MCP control surfaces

- Status: completed
- Parent: DOC-G020
- Parent goal IDs JSON: ["DOC-G020"]
- Depends on: DOC-G011, DOC-G013
- Dependencies JSON: ["DOC-G011", "DOC-G013"]
- Fib priority: 5
- Track: service-architecture
- Priority: P0
- Bundle: documentation-refresh/service-mcp
- Parallel lane: service-mcp
- Resource class: cpu-medium
- Goal: Document the separate catalog, usage-accounting, routing and MCP transport/policy planes and why they are not collapsed into one registry.
- Producing tasks: DOC-007, DOC-008
- Evidence: docs/architecture/MODEL_SERVICE_ROUTING.md and docs/architecture/MCP_RUNTIME.md
- Evidence requirements JSON: ["catalog snapshot and resolution flow", "endpoint reservation flow", "router fallback", "MCP registry and dispatch", "transport and policy boundaries", "compatibility status"]
- Evidence criteria: `model_catalog`, `endpoint_usage`, modality/voice routers, shared CLI runtime, `mcp_server`, `mcp`, MCP++ and UCAN/policy boundaries are accurately separated; auto-install side effects and exact facade status are disclosed.
- Evidence source policy: Package implementations, registries, schemas, descriptors and conformance tests override old plans and summary documents.
- Outputs: docs/architecture/MODEL_SERVICE_ROUTING.md, docs/architecture/MCP_RUNTIME.md
- Predicted files JSON: ["docs/architecture/MODEL_SERVICE_ROUTING.md", "docs/architecture/MCP_RUNTIME.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: Readers can identify which plane answers availability, identity, quota, invocation, transport and authorization questions and understand the compatibility boundary.
- Gap task: Close the smallest model/service/MCP ownership or flow residual.

## DOC-G023 IPFS/P2P and cross-repository boundaries

- Status: completed
- Parent: DOC-G020
- Parent goal IDs JSON: ["DOC-G020"]
- Depends on: DOC-G011, DOC-G013
- Dependencies JSON: ["DOC-G011", "DOC-G013"]
- Fib priority: 8
- Track: distributed-architecture
- Priority: P0
- Bundle: documentation-refresh/distributed-integration
- Parallel lane: distributed-integration
- Resource class: io-medium
- Goal: Explain backend roles, verified content identities, degradation, P2P task execution and the authority boundaries between ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py and MCP++.
- Producing tasks: DOC-009, DOC-010
- Evidence: docs/architecture/DISTRIBUTED_RUNTIME.md and docs/architecture/INTEGRATION_BOUNDARIES.md
- Evidence requirements JSON: ["CID versus synthetic cache key", "backend capability receipt", "P2P workflow and task flow", "gitlink and package ownership", "graceful fallback versus fail-closed assurance"]
- Evidence criteria: CIDv1/base32/dag-json-or-raw/sha2-256 expectations, CAR/pinning/replication semantics, explicit degradation receipts, optional providers and independent Git authorities are accurate and do not treat co-location as shared authority.
- Evidence source policy: Backend/router, multiformats, p2p_tasks, datasets integration, repository-forest and exact adapter tests are primary; a `bafy`-looking string is not verification.
- Outputs: docs/architecture/DISTRIBUTED_RUNTIME.md, docs/architecture/INTEGRATION_BOUNDARIES.md
- Predicted files JSON: ["docs/architecture/DISTRIBUTED_RUNTIME.md", "docs/architecture/INTEGRATION_BOUNDARIES.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: Integrators can distinguish storage, cache, coordination, immutable evidence, remote task and sibling-repository responsibilities and know how each degrades.
- Gap task: Close the smallest distributed or integration-boundary residual.

## DOC-G030 Agent-supervisor architecture guides

- Status: completed
- Parent: DOC-G000
- Parent goal IDs JSON: ["DOC-G000"]
- Depends on: DOC-G010
- Dependencies JSON: ["DOC-G010"]
- Fib priority: 5
- Track: supervisor-architecture
- Priority: P0
- Bundle: documentation-refresh/supervisor-architecture
- Parallel lane: supervisor-architecture
- Resource class: cpu-medium
- Goal: Give developers and agents a current, navigable explanation of the supervisor's intent, authority, planning, evidence, scheduling, execution, recovery, prompt-first and multi-repository assurance paths.
- Producing tasks: DOC-011, DOC-012, DOC-013, DOC-014
- Evidence: four semantic supervisor architecture guides
- Evidence requirements JSON: ["control and authority flow", "planning and assurance flow", "execution and recovery flow", "prompt-first state and steering flow"]
- Evidence criteria: Guides use domain/package vocabulary, distinguish landed primitives from planned facades, separate merge from authoritative completion, and show why evidence tiers, isolation and content identities are necessary.
- Evidence source policy: Domain packages, control contracts, entrypoint modules, daemon/runtime code and current tests are authoritative; stale ASE task status is not.
- Outputs: docs/architecture/agent_supervisor/
- Predicted files JSON: ["docs/architecture/agent_supervisor/"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: A developer or agent can place code, trace a run, assess trust and diagnose a block without relying on delivery ticket vocabulary.
- Gap task: Close the smallest missing supervisor concern in the four-guide set.

## DOC-G031 Intent, control and authority

- Status: completed
- Parent: DOC-G030
- Parent goal IDs JSON: ["DOC-G030"]
- Depends on: DOC-G011, DOC-G013
- Dependencies JSON: ["DOC-G011", "DOC-G013"]
- Fib priority: 3
- Track: supervisor-control
- Priority: P0
- Bundle: documentation-refresh/supervisor-control
- Parallel lane: supervisor-control
- Resource class: cpu-medium
- Goal: Explain objectives, task projections, canonical control contracts, discovery versus capability, principals, policies, effects, authorization and transport parity.
- Producing tasks: DOC-011
- Evidence: docs/architecture/agent_supervisor/CONTROL_PLANE.md
- Evidence requirements JSON: ["intent hierarchy", "operation contract", "authority ladder", "transport parity", "denial and audit flow"]
- Evidence criteria: Objectives remain durable intent; boards remain mutable projections; no prompt, model or transport manufactures identity or mutation authority; exact operation vocabulary is verified from source.
- Evidence source policy: control/, objectives/, task_sources/, Python/CLI/MCP adapters and conformance tests are primary.
- Outputs: docs/architecture/agent_supervisor/CONTROL_PLANE.md
- Predicted files JSON: ["docs/architecture/agent_supervisor/CONTROL_PLANE.md"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: Readers understand the entire authority path from intent through admitted mutation and why transports cannot widen it.
- Gap task: Close the smallest intent, operation, authority or transport residual.

## DOC-G032 Planning, context, analysis and proof

- Status: completed
- Parent: DOC-G030
- Parent goal IDs JSON: ["DOC-G030"]
- Depends on: DOC-G011, DOC-G013
- Dependencies JSON: ["DOC-G011", "DOC-G013"]
- Fib priority: 5
- Track: supervisor-assurance
- Priority: P0
- Bundle: documentation-refresh/supervisor-assurance
- Parallel lane: supervisor-assurance
- Resource class: cpu-medium
- Goal: Explain how goal gaps become bounded plans/context/edit packets and how lexical, AST, GraphRAG, solver, kernel and attestation evidence retain distinct trust levels.
- Producing tasks: DOC-012
- Evidence: docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md
- Evidence requirements JSON: ["plan and context flow", "repository forest and corpus index", "expected versus observed contracts", "proof routing and cache", "datasets provider limits"]
- Evidence criteria: The guide covers obligation-first context, content-addressed invalidation, multi-repository authority, contract mismatch refill, exact provider bindings and why optional datasets analysis cannot manufacture completion.
- Evidence source policy: planning/, context/, analysis/, proof/, integration adapters and authoritative evidence tests are primary.
- Outputs: docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md
- Predicted files JSON: ["docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: Readers can tell discovery, candidate evidence, deterministic validation, solver results, kernel proof and attestation apart and trace cache trust/invalidation.
- Gap task: Close the smallest planning, analysis, context or proof residual.

## DOC-G033 Scheduling, execution, merge and recovery

- Status: completed
- Parent: DOC-G030
- Parent goal IDs JSON: ["DOC-G030"]
- Depends on: DOC-G011, DOC-G013
- Dependencies JSON: ["DOC-G011", "DOC-G013"]
- Fib priority: 8
- Track: supervisor-runtime
- Priority: P0
- Bundle: documentation-refresh/supervisor-execution
- Parallel lane: supervisor-execution
- Resource class: process-control
- Goal: Explain dependency/conflict scheduling, resources, leases, fencing, worktrees, implementation providers, validation, merge, authoritative completion, retries, rescue and quarantine.
- Producing tasks: DOC-013
- Evidence: docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
- Evidence requirements JSON: ["lane lifecycle", "conflict and resource admission", "lease/fence/worktree invariants", "merge versus acceptance", "recovery decision tree"]
- Evidence criteria: Running PID, provider success and merged branch are not confused with healthy progress or task completion; retry termination and stale-evidence reopening are explicit.
- Evidence source policy: runtime/, todo_daemon/, validation/, merge/, rescue/, self_improvement/ and task-state/receipt tests are primary.
- Outputs: docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
- Predicted files JSON: ["docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: An operator can reason from a blocked task to the responsible dependency, resource, provider, validation, merge, evidence or recovery boundary.
- Gap task: Close the smallest scheduling, execution, acceptance or recovery residual.

## DOC-G034 Prompt-first runtime, state and steering

- Status: completed
- Parent: DOC-G030
- Parent goal IDs JSON: ["DOC-G030"]
- Depends on: DOC-G011, DOC-G013
- Dependencies JSON: ["DOC-G011", "DOC-G013"]
- Fib priority: 13
- Track: supervisor-entrypoints
- Priority: P0
- Bundle: documentation-refresh/supervisor-entrypoints
- Parallel lane: supervisor-entrypoints
- Resource class: cpu-medium
- Goal: Document landed target/profile resolvers, prompt broker, run registry, plan linting, steering contracts and verified IPLD backend while clearly labelling the not-yet-landed high-level facade and lifecycle.
- Producing tasks: DOC-014
- Evidence: docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md and docs/architecture/agent_supervisor/packages/entrypoints.md
- Evidence requirements JSON: ["resolver precedence", "secret-free durable state", "run CAS and reconstruction", "steering generations", "landed versus planned matrix", "entrypoints package placement"]
- Evidence criteria: The guide is derived from current modules/tests rather than stale ASE statuses; prompt bodies/credentials remain outside durable records; cold import and downward-only package direction are explicit.
- Evidence source policy: entrypoints/*.py, package exports and matching tests are primary; the prompt-only plan is intent/context, not landed-state authority.
- Outputs: docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md, docs/architecture/agent_supervisor/packages/entrypoints.md
- Predicted files JSON: ["docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md", "docs/architecture/agent_supervisor/packages/entrypoints.md"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: Readers can use the landed primitives safely, see exactly which convenience journey is still planned, and understand restart, ambiguity and steering conflict behavior.
- Gap task: Close the smallest landed/planned, resolver, registry, broker, steering or package-map residual.

## DOC-G040 Architectural decision records

- Status: completed
- Parent: DOC-G000
- Parent goal IDs JSON: ["DOC-G000"]
- Depends on: DOC-G013
- Dependencies JSON: ["DOC-G013"]
- Fib priority: 8
- Track: decisions
- Priority: P1
- Bundle: documentation-refresh/decisions
- Parallel lane: decisions
- Resource class: cpu-small
- Goal: Preserve the context, decision, alternatives and consequences behind the system boundaries most likely to be accidentally simplified by future developers or agents.
- Producing tasks: DOC-015, DOC-016, DOC-017, DOC-018, DOC-019, DOC-020
- Evidence: six accepted/proposed ADR files tied to current guides and code
- Evidence requirements JSON: ["intent projection decision", "proposal and evidence trust decision", "capability and service-plane decision", "worktree lease fencing decision", "mutable versus immutable persistence decision", "domain package and compatibility decision"]
- Evidence criteria: Each ADR follows the template, states status and scope, cites exact sources, evaluates credible alternatives, records positive/negative consequences and identifies supersession triggers.
- Evidence source policy: ADRs explain evidenced design; they do not upgrade a plan into implemented status.
- Outputs: docs/architecture/decisions/
- Predicted files JSON: ["docs/architecture/decisions/"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: The six core decisions can be reviewed independently and are linked from the architecture they constrain.
- Gap task: Close the smallest missing rationale, alternative, consequence or source binding.

## DOC-G041 Intent, trust and evidence decisions

- Status: completed
- Parent: DOC-G040
- Parent goal IDs JSON: ["DOC-G040"]
- Depends on: DOC-G031, DOC-G032
- Dependencies JSON: ["DOC-G031", "DOC-G032"]
- Fib priority: 3
- Track: trust-decisions
- Priority: P1
- Bundle: documentation-refresh/decisions-trust
- Parallel lane: decisions-trust
- Resource class: cpu-small
- Goal: Record why objectives and tasks have different mutability and why model output, validation, proof and acceptance occupy different trust tiers.
- Producing tasks: DOC-015, DOC-016
- Evidence: ADR-0001 and ADR-0002
- Evidence requirements JSON: ["context", "decision", "alternatives", "consequences", "source anchors", "supersession triggers"]
- Evidence criteria: The records explain regeneration, refinement, protected intent, proposal admission, cache re-derivation and merge-versus-acceptance implications.
- Evidence source policy: Current control/objective/task/evidence implementations and tests are primary.
- Outputs: docs/architecture/decisions/0001-objectives-and-task-projections.md, docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md
- Predicted files JSON: ["docs/architecture/decisions/0001-objectives-and-task-projections.md", "docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: Future changes cannot collapse intent/projection or proposal/acceptance without confronting recorded consequences.
- Gap task: Close the smallest trust-decision residual.

## DOC-G042 Capability and isolation decisions

- Status: completed
- Parent: DOC-G040
- Parent goal IDs JSON: ["DOC-G040"]
- Depends on: DOC-G022, DOC-G033
- Dependencies JSON: ["DOC-G022", "DOC-G033"]
- Fib priority: 5
- Track: runtime-decisions
- Priority: P1
- Bundle: documentation-refresh/decisions-runtime
- Parallel lane: decisions-runtime
- Resource class: cpu-small
- Goal: Record why discovery, capability, accounting and invocation remain separate and why concurrent implementation uses leases, fencing and worktrees.
- Producing tasks: DOC-017, DOC-018
- Evidence: ADR-0003 and ADR-0004
- Evidence requirements JSON: ["plane separation alternatives", "optional-dependency consequences", "concurrency failure model", "lease/fence/worktree consequences"]
- Evidence criteria: Import success cannot imply availability; stale/duplicate workers cannot publish effects; resource and endpoint reservations remain explicit.
- Evidence source policy: Catalog/usage/router, capability probes, scheduler, lease and worktree tests are primary.
- Outputs: docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md, docs/architecture/decisions/0004-worktrees-leases-and-fencing.md
- Predicted files JSON: ["docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md", "docs/architecture/decisions/0004-worktrees-leases-and-fencing.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: The system's capability and concurrency boundaries have explicit failure-model rationale.
- Gap task: Close the smallest capability or isolation decision residual.

## DOC-G043 Persistence, package and compatibility decisions

- Status: completed
- Parent: DOC-G040
- Parent goal IDs JSON: ["DOC-G040"]
- Depends on: DOC-G023, DOC-G034
- Dependencies JSON: ["DOC-G023", "DOC-G034"]
- Fib priority: 8
- Track: structure-decisions
- Priority: P1
- Bundle: documentation-refresh/decisions-structure
- Parallel lane: decisions-structure
- Resource class: cpu-small
- Goal: Record why mutable coordination is separated from immutable replication and why semantic domain packages coexist with bounded compatibility facades and independent repository authority.
- Producing tasks: DOC-019, DOC-020
- Evidence: ADR-0005 and ADR-0006
- Evidence requirements JSON: ["single-writer coordination decision", "immutable replication decision", "domain DAG decision", "compatibility and submodule consequences"]
- Evidence criteria: DuckDB lease authority is not assigned to Parquet/IPLD/IPFS replicas; board IDs do not define package layout; compatibility paths cannot create alternate state or authority.
- Evidence source policy: Coordination/replication, package layout/export, repository forest and compatibility conformance tests are primary.
- Outputs: docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md, docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md
- Predicted files JSON: ["docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md", "docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: Persistence, package and compatibility changes have a durable rationale and authority boundary.
- Gap task: Close the smallest persistence, layout or compatibility decision residual.

## DOC-G050 Current user, developer and operator journeys

- Status: completed
- Parent: DOC-G000
- Parent goal IDs JSON: ["DOC-G000"]
- Depends on: DOC-G020, DOC-G030
- Dependencies JSON: ["DOC-G020", "DOC-G030"]
- Fib priority: 13
- Track: current-guides
- Priority: P0
- Bundle: documentation-refresh/guides
- Parallel lane: guides
- Resource class: cpu-medium
- Goal: Align installation, quickstart, Python, CLI, MCP, supervisor, deployment, hardware, P2P, testing and troubleshooting journeys with executable current behavior.
- Producing tasks: DOC-021, DOC-022, DOC-023, DOC-024, DOC-025, DOC-026
- Evidence: refreshed canonical guide files and offline-safe checks
- Evidence requirements JSON: ["current extras and version caveat", "Python public import checks", "CLI help checks", "MCP startup checks", "supervisor operation checks", "deployment capability prerequisites", "test path checks"]
- Evidence criteria: Broken domain-layout module commands and missing test paths are fixed; case-colliding installation files are resolved without losing useful history; code-owned inconsistencies are clearly identified instead of guessed around.
- Evidence source policy: Packaging, code, installed help and focused tests are primary; network/provider/hardware-dependent examples declare prerequisites and are not required for offline acceptance.
- Outputs: docs/guides/, docs/api/, docs/development/
- Predicted files JSON: ["docs/guides/", "docs/api/", "docs/development/"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: Canonical user and operator paths are internally consistent, executable where offline-safe, and honest about optional capabilities and unresolved code-level contradictions.
- Gap task: Close the smallest broken or stale current-guide journey.

## DOC-G051 Installation, quickstart, Python and CLI

- Status: completed
- Parent: DOC-G050
- Parent goal IDs JSON: ["DOC-G050"]
- Depends on: DOC-G021, DOC-G022
- Dependencies JSON: ["DOC-G021", "DOC-G022"]
- Fib priority: 3
- Track: onboarding-api
- Priority: P0
- Bundle: documentation-refresh/onboarding-api
- Parallel lane: onboarding-api
- Resource class: cpu-medium
- Goal: Make installation, first-use, Python API and CLI guidance agree with packaging, public exports and current command groups.
- Producing tasks: DOC-021, DOC-022
- Evidence: canonical lowercase installation, quickstart, API and CLI guides
- Evidence requirements JSON: ["extras table", "case-collision disposition", "public import examples", "current CLI group table", "planned/missing command caveat", "version-source blocker"]
- Evidence criteria: Nonexistent extras and module paths are removed; version disagreement is surfaced as code-owned; examples avoid internal imports unless labelled; `agent`, `copilot` and `copilot-sdk` groups are represented accurately.
- Evidence source policy: pyproject/setup/package exports/CLI parser and help tests are primary.
- Outputs: docs/guides/getting-started/, docs/guides/QUICKSTART.md, docs/api/overview.md, docs/guides/cli/README_CLI.md
- Predicted files JSON: ["docs/guides/getting-started/", "docs/guides/QUICKSTART.md", "docs/api/overview.md", "docs/guides/cli/README_CLI.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: A clean environment can choose the right extras and discover supported Python and CLI paths without relying on a stale or case-colliding guide.
- Gap task: Close the smallest installation, import or CLI residual.

## DOC-G052 MCP and supervisor operation

- Status: completed
- Parent: DOC-G050
- Parent goal IDs JSON: ["DOC-G050"]
- Depends on: DOC-G022, DOC-G031, DOC-G032, DOC-G033, DOC-G034
- Dependencies JSON: ["DOC-G022", "DOC-G031", "DOC-G032", "DOC-G033", "DOC-G034"]
- Fib priority: 5
- Track: operator-guides
- Priority: P0
- Bundle: documentation-refresh/mcp-supervisor-guides
- Parallel lane: mcp-supervisor-guides
- Resource class: cpu-medium
- Goal: Refresh MCP setup and agent-supervisor operator/developer/agent journeys against current transports, operations, package layout and prompt-entrypoint status.
- Producing tasks: DOC-023, DOC-024
- Evidence: refreshed MCP and supervisor guides
- Evidence requirements JSON: ["canonical MCP startup", "compatibility warning", "31-operation source verification", "domain module commands", "prompt-first landed/planned matrix", "recovery flow"]
- Evidence criteria: Stale flat module execution paths and operation lists are corrected; source packages and security prerequisites are explicit; current guide examples avoid claiming the planned high-level facade exists.
- Evidence source policy: mcp_server, control operation catalog, package exports, entrypoints modules and subprocess help/tests are primary.
- Outputs: docs/MCP_SERVER.md, docs/guides/MCP_SETUP_GUIDE.md, docs/guides/QUICK_START_MCP.md, docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md, docs/architecture/agent_supervisor/FOR_AGENTS.md
- Predicted files JSON: ["docs/MCP_SERVER.md", "docs/guides/MCP_SETUP_GUIDE.md", "docs/guides/QUICK_START_MCP.md", "docs/guides/AGENT_SUPERVISOR_GUIDE.md", "docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md", "docs/architecture/agent_supervisor/FOR_AGENTS.md"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: Operators can start, inspect and recover the supported MCP/supervisor surfaces with accurate authority and capability expectations.
- Gap task: Close the smallest MCP or supervisor operator-path residual.

## DOC-G053 Deployment, hardware, P2P, testing and troubleshooting

- Status: completed
- Parent: DOC-G050
- Parent goal IDs JSON: ["DOC-G050"]
- Depends on: DOC-G021, DOC-G023
- Dependencies JSON: ["DOC-G021", "DOC-G023"]
- Fib priority: 8
- Track: operations-validation
- Priority: P1
- Bundle: documentation-refresh/operations-validation
- Parallel lane: operations-validation
- Resource class: io-medium
- Goal: Align environment-specific operation and contributor validation with capability probes, present manifests and actual test paths.
- Producing tasks: DOC-025, DOC-026
- Evidence: refreshed deployment/hardware/P2P/troubleshooting and testing/maintenance guides
- Evidence requirements JSON: ["capability prerequisites", "offline versus external validation", "correct test paths", "known failure symptoms", "documentation review checklist"]
- Evidence criteria: GPU, browser, IPFS, P2P and providers remain optional; commands reference existing paths; deploy/recovery advice names authoritative health signals; doc maintenance does not suppress failed checks.
- Evidence source policy: Installation/deployment inputs, capability probes, current tests and runtime status schemas are primary.
- Outputs: docs/guides/deployment/README.md, docs/guides/hardware/overview.md, docs/guides/p2p/README.md, docs/guides/troubleshooting/faq.md, docs/development/testing.md, docs/development/DOCUMENTATION_MAINTENANCE.md
- Predicted files JSON: ["docs/guides/deployment/README.md", "docs/guides/hardware/overview.md", "docs/guides/p2p/README.md", "docs/guides/troubleshooting/faq.md", "docs/development/testing.md", "docs/development/DOCUMENTATION_MAINTENANCE.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: Operators and contributors can select environment-appropriate checks and diagnose limitations without treating optional services as baseline requirements.
- Gap task: Close the smallest environment, test or troubleshooting residual.

## DOC-G060 Navigation, verification and sustainable closeout

- Status: completed
- Parent: DOC-G000
- Parent goal IDs JSON: ["DOC-G000"]
- Depends on: DOC-G010, DOC-G020, DOC-G030, DOC-G040, DOC-G050
- Dependencies JSON: ["DOC-G010", "DOC-G020", "DOC-G030", "DOC-G040", "DOC-G050"]
- Fib priority: 21
- Track: documentation-closeout
- Priority: P0
- Bundle: documentation-refresh/closeout
- Parallel lane: closeout
- Resource class: coordinator
- Goal: Route every audience to the maintained material, classify the remaining corpus, validate current links/examples/paths and publish a dated source-bound completion matrix.
- Producing tasks: DOC-027, DOC-028
- Evidence: glossary, architecture hub, documentation manifest, indexes, current-state page and validation report
- Evidence requirements JSON: ["canonical glossary", "audience routes", "document status manifest", "local link report", "command/path checks", "baseline and known-limitations receipt"]
- Evidence criteria: Shared navigation changes occur only after leaf guides land; current docs do not route readers into archives as normative; failures are fixed or explicitly recorded with owners.
- Evidence source policy: The final merged documentation tree and current repository are authoritative; lane-local or pre-merge success is not closeout.
- Outputs: docs/README.md, docs/INDEX.md, docs/architecture/README.md, docs/architecture/GLOSSARY.md, docs/development/DOCUMENTATION_MANIFEST.md, docs/development/DOCUMENTATION_CURRENT_STATE.md, docs/development/DOCUMENTATION_VALIDATION_2026_08.md
- Predicted files JSON: ["docs/README.md", "docs/INDEX.md", "docs/architecture/README.md", "docs/architecture/GLOSSARY.md", "docs/development/DOCUMENTATION_MANIFEST.md", "docs/development/DOCUMENTATION_CURRENT_STATE.md", "docs/development/DOCUMENTATION_VALIDATION_2026_08.md"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: The merge target contains an audience-oriented, status-labelled, locally validated documentation system with a reproducible next-audit trigger.
- Gap task: Close the smallest navigation, classification, validation or receipt residual.

## DOC-G061 Glossary, audience routes and archive taxonomy

- Status: completed
- Parent: DOC-G060
- Parent goal IDs JSON: ["DOC-G060"]
- Depends on: DOC-G012, DOC-G020, DOC-G030, DOC-G040
- Dependencies JSON: ["DOC-G012", "DOC-G020", "DOC-G030", "DOC-G040"]
- Fib priority: 5
- Track: information-architecture
- Priority: P0
- Bundle: documentation-refresh/information-architecture
- Parallel lane: information-architecture
- Resource class: io-medium
- Goal: Define one product vocabulary, architecture landing page and document-status manifest before editing shared top-level navigation.
- Producing tasks: DOC-027
- Evidence: docs/architecture/GLOSSARY.md, docs/architecture/README.md, docs/development/DOCUMENTATION_MANIFEST.md
- Evidence requirements JSON: ["semantic glossary", "audience map", "current/reference/plan/historical/vendored classification", "owner and supersession fields"]
- Evidence criteria: Terms distinguish service/catalog/router planes, MCP/MCP++, objective/task, discovery/capability/proof, CID/cache key, merge/acceptance and coordination/replication; manifests do not claim unreviewed documents are current.
- Evidence source policy: Landed leaf guides, lifecycle policy and source packages are primary.
- Outputs: docs/architecture/GLOSSARY.md, docs/architecture/README.md, docs/development/DOCUMENTATION_MANIFEST.md
- Predicted files JSON: ["docs/architecture/GLOSSARY.md", "docs/architecture/README.md", "docs/development/DOCUMENTATION_MANIFEST.md"]
- Validation: git diff --check
- Validation commands JSON: ["git diff --check"]
- Acceptance: A new reader or agent can resolve terminology and choose a maintained path without scanning hundreds of records.
- Gap task: Close the smallest glossary, audience or classification residual.

## DOC-G062 Link/example verification and freshness publication

- Status: completed
- Parent: DOC-G060
- Parent goal IDs JSON: ["DOC-G060"]
- Depends on: DOC-G050, DOC-G061
- Dependencies JSON: ["DOC-G050", "DOC-G061"]
- Fib priority: 8
- Track: closeout-validation
- Priority: P0
- Bundle: documentation-refresh/validation-closeout
- Parallel lane: validation-closeout
- Resource class: cpu-medium
- Goal: Run the offline-safe documentation gates on the integrated branch, fix current-surface link/navigation residuals and publish the final baseline and limitations.
- Producing tasks: DOC-028
- Evidence: docs/README.md, docs/INDEX.md, docs/development/DOCUMENTATION_CURRENT_STATE.md, docs/development/DOCUMENTATION_VALIDATION_2026_08.md
- Evidence requirements JSON: ["local link results", "case-fold collision results", "CLI/module/test-path checks", "supervisor docs check", "known code blockers", "next-audit triggers"]
- Evidence criteria: Current/reference surfaces pass local checks; archive/history link debt is measured separately; version and CLI-help source inconsistencies are not silently resolved in prose; final pages carry exact commit/date.
- Evidence source policy: Only checks run on the integrated merge target count; external network availability is outside this program's completion boundary.
- Outputs: docs/README.md, docs/INDEX.md, docs/development/DOCUMENTATION_CURRENT_STATE.md, docs/development/DOCUMENTATION_VALIDATION_2026_08.md
- Predicted files JSON: ["docs/README.md", "docs/INDEX.md", "docs/development/DOCUMENTATION_CURRENT_STATE.md", "docs/development/DOCUMENTATION_VALIDATION_2026_08.md"]
- Validation: python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Validation commands JSON: ["python scripts/docs/check_agent_supervisor_docs.py", "git diff --check"]
- Acceptance: The documentation entrypoints and status report are accurate for one exact tree, with reproducible validation commands and explicit remaining code-owned or historical debt.
- Gap task: Close the smallest current-surface validation or publication residual.
