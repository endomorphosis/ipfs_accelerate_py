# IPFS Accelerate documentation refresh task board

Executable projection of
[`documentation_refresh.objectives.md`](documentation_refresh.objectives.md)
for the plan in
[`DOCUMENTATION_REFRESH_PLAN_2026_08.md`](DOCUMENTATION_REFRESH_PLAN_2026_08.md).

This board is a sealed 28-task tranche. The three program inputs are
operator-owned and protected during implementation. Tasks may edit only their
declared outputs. Shared navigation is reserved for the final fan-in tasks.

## DOC-001 Freeze the current-tree documentation drift audit

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: documentation-audit
- Depends on:
- Goal id: DOC-G011
- Outputs: docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md
- Validation: test -f docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md && rg -q 'd7da3d6bf8ca2f7ec870d03742b09f26e3e16d15' docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/drift-audit
- Parallel lane: drift-audit
- Resource class: io-medium
- Predicted files: docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md
- Interfaces: DocumentationDriftAudit@1, SourceDocOwnershipFinding@1
- Source anchors: Git history since the 2026-07-24 and 2026-07-28 documentation baselines; pyproject.toml; ipfs_accelerate_py/; docs/; test/; scripts/docs/check_agent_supervisor_docs.py
- Conflict policy: Own only the dated audit. Do not edit indexes, current guides, the plan, objective heap or task board.
- Preconditions: Baseline commit d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15 is available.
- Effects: Record reproducible commands and prioritize stale, missing, contradictory and broken documentation with exact source/doc anchors and owners.
- Evidence subset: source/doc diff, broken command/path examples, case-fold collision, stale ASE status warning, version-source disagreement
- Acceptance: Cover prompt-entrypoint primitives, contract assurance, merge-versus-acceptance, model catalog, endpoint usage, CID/backend semantics, MCP compatibility and cross-repository changes; identify the incorrect module/test paths and installation case collision; never treat old board status as current behavior authority.

## DOC-002 Define documentation lifecycle, ownership and freshness policy

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: documentation-governance
- Depends on:
- Goal id: DOC-G012
- Outputs: docs/development/DOCUMENTATION_LIFECYCLE.md
- Validation: test -f docs/development/DOCUMENTATION_LIFECYCLE.md && rg -q 'Current' docs/development/DOCUMENTATION_LIFECYCLE.md && rg -q 'Historical' docs/development/DOCUMENTATION_LIFECYCLE.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/lifecycle
- Parallel lane: lifecycle
- Resource class: cpu-small
- Predicted files: docs/development/DOCUMENTATION_LIFECYCLE.md
- Interfaces: DocumentationStatus@1, DocumentationAuthorityMap@1, FreshnessTrigger@1
- Source anchors: docs/README.md; docs/INDEX.md; docs/development/DOCUMENTATION_CURRENT_STATE.md; docs/project/; docs/archive/; docs/development_history/; pyproject.toml
- Conflict policy: Own only the lifecycle policy. Do not reclassify individual files or edit shared indexes in this task.
- Preconditions: None.
- Effects: Define Current, Reference, Plan, Historical, Generated and Vendored states; owner/source/freshness/supersession metadata; archive policy; and handling of code-owned contradictions.
- Evidence subset: closed status vocabulary, source-of-truth matrix, audit triggers, exception policy
- Acceptance: The policy makes plans and completion summaries non-normative by default, forbids prose from concealing source inconsistencies, and gives agents deterministic placement and revalidation rules.

## DOC-003 Publish architecture guide conventions

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: architecture-governance
- Depends on:
- Goal id: DOC-G013
- Outputs: docs/architecture/GUIDE_CONVENTIONS.md
- Validation: test -f docs/architecture/GUIDE_CONVENTIONS.md && rg -q 'Source anchors' docs/architecture/GUIDE_CONVENTIONS.md && rg -q 'Rationale' docs/architecture/GUIDE_CONVENTIONS.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/guide-conventions
- Parallel lane: guide-conventions
- Resource class: cpu-small
- Predicted files: docs/architecture/GUIDE_CONVENTIONS.md
- Interfaces: ArchitectureGuideContract@1, DiagramVocabulary@1
- Source anchors: docs/architecture/overview.md; docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md; docs/architecture/agent_supervisor/PACKAGE_MAP.md; docs/development/DOCUMENTATION_CURRENT_STATE.md
- Conflict policy: Own only the conventions guide. Do not normalize existing guides in this task.
- Preconditions: None.
- Effects: Define required metadata, current/planned language, source-anchor style, diagram semantics, trust/failure/rationale sections, links, verification and volatile-claim policy.
- Evidence subset: guide template outline, normative vocabulary, diagram rules, review checklist
- Acceptance: Parallel writers can follow one compact contract that requires audience, scope, status, last-verified baseline, sources, flows, rationale, alternatives, consequences, failure semantics and verification without inventing API guarantees.

## DOC-004 Establish the ADR index and template

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: architecture-governance
- Depends on:
- Goal id: DOC-G013
- Outputs: docs/architecture/decisions/README.md, docs/architecture/decisions/0000-template.md
- Validation: test -f docs/architecture/decisions/README.md && test -f docs/architecture/decisions/0000-template.md && rg -q '^## Alternatives' docs/architecture/decisions/0000-template.md && rg -q '^## Consequences' docs/architecture/decisions/0000-template.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/adr-framework
- Parallel lane: adr-framework
- Resource class: cpu-small
- Predicted files: docs/architecture/decisions/README.md, docs/architecture/decisions/0000-template.md
- Interfaces: ArchitectureDecisionRecord@1, ADRStatus@1
- Source anchors: docs/architecture/; docs/development_history/ARCHIVAL_DECISIONS.md; documentation_refresh.objectives.md
- Conflict policy: Own the ADR directory index and template only. Reserve IDs 0001 through 0006 for DOC-015 through DOC-020; later ADR writers own only their numbered file.
- Preconditions: None.
- Effects: Define numbering, status, supersedes/superseded-by, context, decision, alternatives, consequences, evidence, verification and review triggers; predeclare the six program ADRs.
- Evidence subset: ADR index, collision-safe naming, complete template
- Acceptance: ADRs have a durable home and a template that separates evidenced current decisions from proposals and records negative as well as positive consequences.

## DOC-005 Refresh the system context and maintained architecture overview

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: system-architecture
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G021
- Outputs: docs/architecture/SYSTEM_CONTEXT.md, docs/architecture/overview.md
- Validation: test -f docs/architecture/SYSTEM_CONTEXT.md && rg -q 'Last verified' docs/architecture/SYSTEM_CONTEXT.md && rg -qi 'rationale\|why' docs/architecture/SYSTEM_CONTEXT.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/system-context
- Parallel lane: system-context
- Resource class: cpu-medium
- Predicted files: docs/architecture/SYSTEM_CONTEXT.md, docs/architecture/overview.md
- Interfaces: SystemContext@1, RuntimeBoundaryMap@1
- Source anchors: ipfs_accelerate_py/__init__.py; ipfs_accelerate_py/ipfs_accelerate.py; ipfs_accelerate_py/cli_entry.py; ipfs_accelerate_py/mcp_server/; ipfs_accelerate_py/agent_supervisor/; pyproject.toml
- Conflict policy: Own SYSTEM_CONTEXT.md and overview.md only. Do not edit indexes or subsystem deep dives.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Replace the stale one-screen overview and add actors, containers, entrypoints, primary flows, trust boundaries, optional capabilities and explicit inference-plane/supervisor-plane separation.
- Evidence subset: component map, container diagram, canonical entrypoints, capability boundary, change rationale
- Acceptance: Every conceptual box maps to a live package or is labelled conceptual; current and compatibility surfaces are distinct; readers understand why the inference/data plane and supervisor/control plane are coupled by adapters rather than collapsed.

## DOC-006 Document the inference runtime and router lifecycle

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: inference-architecture
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G021
- Outputs: docs/architecture/INFERENCE_RUNTIME.md
- Validation: test -f docs/architecture/INFERENCE_RUNTIME.md && rg -q 'Last verified' docs/architecture/INFERENCE_RUNTIME.md && rg -qi 'failure\|fallback' docs/architecture/INFERENCE_RUNTIME.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/inference-runtime
- Parallel lane: inference-runtime
- Resource class: cpu-medium
- Predicted files: docs/architecture/INFERENCE_RUNTIME.md
- Interfaces: InferenceRequestFlow@1, RouterFallbackFlow@1
- Source anchors: ipfs_accelerate_py/ipfs_accelerate.py; inference_backend_manager.py; unified_inference_service.py; llm_router.py; embeddings_router.py; multimodal_router.py; voice_router.py; hf_model_server/; api_backends/; worker/
- Conflict policy: Own only INFERENCE_RUNTIME.md. Link to catalog, MCP and distributed guides without editing them.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Trace endpoint/model discovery, provider selection, modality routing, execution, caching, result/error propagation and graceful degradation; explain why routers and execution adapters remain separate.
- Evidence subset: sequence diagram, ownership table, sync/async boundary, fallback and error taxonomy, extension points
- Acceptance: A developer can follow one request across public entrypoint, catalog/router, backend/worker and result without assuming optional providers or hardware are present.

## DOC-007 Document model, service and endpoint-usage routing

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: service-architecture
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G022
- Outputs: docs/architecture/MODEL_SERVICE_ROUTING.md
- Validation: test -f docs/architecture/MODEL_SERVICE_ROUTING.md && rg -q 'model_catalog' docs/architecture/MODEL_SERVICE_ROUTING.md && rg -q 'endpoint_usage' docs/architecture/MODEL_SERVICE_ROUTING.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/model-service-routing
- Parallel lane: model-service-routing
- Resource class: cpu-medium
- Predicted files: docs/architecture/MODEL_SERVICE_ROUTING.md
- Interfaces: CatalogResolutionFlow@1, EndpointReservationFlow@1, ProviderInvocationFlow@1
- Source anchors: ipfs_accelerate_py/model_catalog/; ipfs_accelerate_py/endpoint_usage/; model_manager.py; cli_runtime/; llm_router.py; embeddings_router.py; voice_jobs/; voice_providers/; hf_model_server/
- Conflict policy: Own only MODEL_SERVICE_ROUTING.md. Do not edit the delivery plans for catalog or endpoint usage.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Promote landed current behavior into a maintained guide separating catalog information, dynamic usage/accounting and router invocation planes; cover identities, snapshots, reservations, receipts, fallback, voice jobs and CLI providers.
- Evidence subset: plane map, state ownership, request flow, provider resolution, quota/concurrency failure semantics
- Acceptance: Readers know which component answers what exists, what is currently usable, what capacity is reserved and how invocation/fallback occurs, plus why one global mutable registry would violate those concerns.

## DOC-008 Document current MCP and MCP++ runtime architecture

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp-architecture
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G022
- Outputs: docs/architecture/MCP_RUNTIME.md
- Validation: test -f docs/architecture/MCP_RUNTIME.md && rg -q 'mcp_server' docs/architecture/MCP_RUNTIME.md && rg -qi 'compatib' docs/architecture/MCP_RUNTIME.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/mcp-runtime
- Parallel lane: mcp-runtime
- Resource class: cpu-medium
- Predicted files: docs/architecture/MCP_RUNTIME.md
- Interfaces: MCPDispatchFlow@1, MCPTransportBoundary@1, MCPPolicyBoundary@1
- Source anchors: ipfs_accelerate_py/mcp_server/; ipfs_accelerate_py/mcp/; ipfs_accelerate_py/mcplusplus_module/; mcpplusplus/; test/ MCP conformance suites
- Conflict policy: Own only MCP_RUNTIME.md. Do not edit MCP user guides, the unification plan or MCP++ project records.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Explain canonical server, compatibility facade, registries, hierarchical tools, dispatch, stdio/FastAPI/gRPC/P2P transports, MCP++ descriptors, UCAN/policy/audit and optional import/auto-install side effects.
- Evidence subset: component/transport map, tool registration sequence, authority flow, compatibility and migration boundary, failure semantics
- Acceptance: Integrators can select the canonical runtime, understand which compatibility imports may have side effects, and trace tool invocation through validation/policy without treating transport as authority.

## DOC-009 Document IPFS, content identity and P2P execution

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: distributed-architecture
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G023
- Outputs: docs/architecture/DISTRIBUTED_RUNTIME.md
- Validation: test -f docs/architecture/DISTRIBUTED_RUNTIME.md && rg -q 'CIDv1' docs/architecture/DISTRIBUTED_RUNTIME.md && rg -qi 'synthetic' docs/architecture/DISTRIBUTED_RUNTIME.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/distributed-runtime
- Parallel lane: distributed-runtime
- Resource class: io-medium
- Predicted files: docs/architecture/DISTRIBUTED_RUNTIME.md
- Interfaces: BackendRole@1, ContentIdentityProfile@1, DegradationReceipt@1, P2PTaskFlow@1
- Source anchors: ipfs_backend_router.py; ipfs_multiformats.py; agent_supervisor/multiformats_identity.py; entrypoints/verified_ipld_backend.py; common/*cache*.py; p2p_tasks/; p2p_workflow_discovery.py; p2p_workflow_scheduler.py
- Conflict policy: Own only DISTRIBUTED_RUNTIME.md. Do not edit older IPFS feature guides in this task.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Define backend roles and capabilities, real CID versus synthetic cache-key semantics, explicit degradation, CAR/pinning/replication, P2P discovery/scheduling/trust and fallback boundaries.
- Evidence subset: identity profile, backend selection table, P2P sequence, degradation flow, verification recipe
- Acceptance: No `bafy`-looking cache token is presented as a verified CID; immutable content/replication and mutable coordination are separate; missing IPFS/P2P degrades or fails closed according to the operation's assurance contract.

## DOC-010 Document cross-repository and nested-package boundaries

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: integration-architecture
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G023
- Outputs: docs/architecture/INTEGRATION_BOUNDARIES.md
- Validation: test -f docs/architecture/INTEGRATION_BOUNDARIES.md && rg -q 'ipfs_datasets_py' docs/architecture/INTEGRATION_BOUNDARIES.md && rg -q 'ipfs_kit_py' docs/architecture/INTEGRATION_BOUNDARIES.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/integration-boundaries
- Parallel lane: integration-boundaries
- Resource class: io-medium
- Predicted files: docs/architecture/INTEGRATION_BOUNDARIES.md
- Interfaces: RepositoryAuthorityBoundary@1, OptionalIntegrationCapability@1, GitlinkPin@1
- Source anchors: .gitmodules; router_deps.py; datasets_integration/; agent_supervisor/integrations/; repository_forest.py; repository_corpus_index.py; ipfs_kit_integration.py; MCP++ submodule descriptors
- Conflict policy: Own only INTEGRATION_BOUNDARIES.md. Do not initialize, update or commit any submodule gitlink.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Publish ownership/dependency direction for ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py and MCP++; explain pins, editable/umbrella discovery, capability probes, graceful fallback, fail-closed assurance and independent Git authority.
- Evidence subset: repository matrix, import/dependency arrows, capability and authority flow, known dirty/uninitialized checkout caveat
- Acceptance: Co-location is never equated with shared Git or mutation authority; exact integration paths and tests are current; optional providers nominate analysis/evidence but cannot manufacture completion.

## DOC-011 Document supervisor intent, control and authority

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-control
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G031
- Outputs: docs/architecture/agent_supervisor/CONTROL_PLANE.md
- Validation: test -f docs/architecture/agent_supervisor/CONTROL_PLANE.md && rg -qi 'models propose' docs/architecture/agent_supervisor/CONTROL_PLANE.md && rg -q 'Source anchors' docs/architecture/agent_supervisor/CONTROL_PLANE.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/supervisor-control
- Parallel lane: supervisor-control
- Resource class: cpu-medium
- Predicted files: docs/architecture/agent_supervisor/CONTROL_PLANE.md
- Interfaces: ObjectiveIntent@1, OperationRequest@1, AuthorizationDecision@1, ExpectedEffect@1
- Source anchors: agent_supervisor/control/; objectives/; task_sources/; control operation catalog; CLI/MCP adapters; authorization and conformance tests
- Conflict policy: Own only CONTROL_PLANE.md. Use semantic domain names and do not edit existing supervisor hubs or plans.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Explain objective hierarchy, task projections, one transport-neutral operation contract, discovery/capability distinction, target binding, principal/policy/effect/lease/fence requirements, audit and parity.
- Evidence subset: authority ladder, request/mutation flow, denial taxonomy, transport comparison, rationale
- Acceptance: Readers understand why intent is durable, taskboards are projections, prompt/model/transport cannot confer authority, and every mutation is scope/effect/identity bound.

## DOC-012 Document supervisor planning and assurance pipeline

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-assurance
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G032
- Outputs: docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md
- Validation: test -f docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md && rg -q 'Source anchors' docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md && rg -qi 'evidence tier\|trust tier' docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/supervisor-assurance
- Parallel lane: supervisor-assurance
- Resource class: cpu-medium
- Predicted files: docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md
- Interfaces: PlanBranch@1, ContextCapsule@1, ProgramContractIR@1, ProofReceipt@1
- Source anchors: agent_supervisor/planning/; context/; analysis/; proof/; program_contracts.py; program_ast_adapters.py; repository_forest.py; repository_corpus_index.py; integrations/ipfs_datasets_*; related tests
- Conflict policy: Own only PLANNING_AND_ASSURANCE.md. Do not update delivery plans, proof policies or task boards.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Trace objective gap through planning, context compilation, repository/AST/GraphRAG analysis, expected-observed contract comparison, obligations, solver/prover routing, cache invalidation, edit packets and completion evidence.
- Evidence subset: assurance pipeline, repository authority forest, evidence-tier table, provider/capability matrix, cache trust rules
- Acceptance: Lexical/AST/GraphRAG findings, LLM proposals, deterministic tests, solver candidates, kernel proofs, ZK attestations and completion receipts remain distinct; optional datasets providers cannot self-certify success.

## DOC-013 Document supervisor execution, landing and recovery

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-runtime
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G033
- Outputs: docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
- Validation: test -f docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md && rg -qi 'lease' docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md && rg -qi 'authoritative completion' docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/supervisor-execution
- Parallel lane: supervisor-execution
- Resource class: process-control
- Predicted files: docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
- Interfaces: LaneLifecycle@1, LeaseFence@1, MergeReceipt@1, AuthoritativeCompletion@1, RescueDisposition@1
- Source anchors: agent_supervisor/runtime/; todo_daemon/; validation/; merge/; rescue/; self_improvement/; core/conflict_graph.py; provider_execution.py; scheduler and authoritative-completion tests
- Conflict policy: Own only EXECUTION_AND_RECOVERY.md. Do not modify daemon/runtime code or operator guides.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Explain dependency/conflict/resource admission, task sharding, leases/fencing, worktrees, provider routing, validation, merge queue/train, separate acceptance, heartbeats, attempts/retries, reconciliation, rescue and quarantine.
- Evidence subset: lifecycle/state diagrams, scheduler admission table, merge-versus-acceptance rationale, recovery decision tree, health signals
- Acceptance: PID, provider exit and merge are not called completion; stale evidence may reopen acceptance; operators can distinguish legitimate dependency idle from provider/resource/validation/merge/recovery blocks.

## DOC-014 Document prompt-first runtime, persistence and steering status

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-entrypoints
- Depends on: DOC-001, DOC-003
- Goal id: DOC-G034
- Outputs: docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md, docs/architecture/agent_supervisor/packages/entrypoints.md
- Validation: test -f docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md && test -f docs/architecture/agent_supervisor/packages/entrypoints.md && rg -qi 'landed\|implemented' docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md && rg -qi 'planned\|not yet' docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/supervisor-entrypoints
- Parallel lane: supervisor-entrypoints
- Resource class: cpu-medium
- Predicted files: docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md, docs/architecture/agent_supervisor/packages/entrypoints.md
- Interfaces: TargetResolver@1, PromptBroker@1, RunRegistry@1, SteeringIntent@1, VerifiedIPLDBackend@1
- Source anchors: agent_supervisor/entrypoints/*.py; entrypoints/README.md; agent_supervisor/__init__.py; prompt-only tests; AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md and board as intent/history only
- Conflict policy: Own only PROMPT_FIRST_RUNTIME.md and the entrypoints package page. Do not reconcile the ASE board or edit package code.
- Preconditions: Drift audit and guide conventions are complete.
- Effects: Describe resolver precedence/provenance, ambiguity, prompt secrecy, profile/authority ceilings, run-registry CAS/reconstruction, plan lint, steering generations and verified IPLD access; publish an exact landed-versus-planned matrix.
- Evidence subset: resolver flow, durable/transient data table, restart/CAS semantics, cold-import boundary, facade status matrix
- Acceptance: Landed ASE source/tests—not stale board status—determine current behavior; high-level `Supervisor.open()`/prompt-only lifecycle remains clearly planned where absent; credentials, UCANs and prompt bodies stay out of durable records.

## DOC-015 Record why objectives and taskboards have different mutability

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: trust-decisions
- Depends on: DOC-004, DOC-011
- Goal id: DOC-G041
- Outputs: docs/architecture/decisions/0001-objectives-and-task-projections.md
- Validation: test -f docs/architecture/decisions/0001-objectives-and-task-projections.md && rg -q '^## Alternatives' docs/architecture/decisions/0001-objectives-and-task-projections.md && rg -q '^## Consequences' docs/architecture/decisions/0001-objectives-and-task-projections.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/adr-objectives
- Parallel lane: adr-objectives
- Resource class: cpu-small
- Predicted files: docs/architecture/decisions/0001-objectives-and-task-projections.md
- Interfaces: ADR-0001
- Source anchors: agent_supervisor/objectives/; task_sources/; objective graph/tracker tests; docs/architecture/agent_supervisor/CONTROL_PLANE.md
- Conflict policy: Own only ADR-0001; do not edit the ADR index or source guide.
- Preconditions: ADR template and control-plane guide are complete.
- Effects: Record immutable/durable intent versus regenerable/drainable projection decision, protected inputs, alternatives and operational consequences.
- Evidence subset: objective/task identity, reconciliation, generated projection and protected-path behavior
- Acceptance: The ADR explains why a single mutable todo list is insufficient and what rules preserve intent during refinement, execution and recovery.

## DOC-016 Record why models propose and evidence admits

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: trust-decisions
- Depends on: DOC-004, DOC-012
- Goal id: DOC-G041
- Outputs: docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md
- Validation: test -f docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md && rg -q '^## Alternatives' docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md && rg -qi 'merge' docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/adr-evidence
- Parallel lane: adr-evidence
- Resource class: cpu-small
- Predicted files: docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md
- Interfaces: ADR-0002
- Source anchors: planning/; validation/; proof/; todo_daemon/authoritative_completion.py; proposal and acceptance tests; PLANNING_AND_ASSURANCE.md
- Conflict policy: Own only ADR-0002; do not edit the ADR index or source guide.
- Preconditions: ADR template and planning/assurance guide are complete.
- Effects: Record proposal-tier model output, deterministic/typed evidence admission, cache re-derivation and merge-versus-authoritative-acceptance separation.
- Evidence subset: evidence hierarchy, proposal validation, proof/cache trust, post-merge acceptance
- Acceptance: The ADR evaluates direct model trust and merge-as-completion alternatives and records why fluent output or a landed commit cannot independently authorize completion.

## DOC-017 Record capability, catalog, usage and routing separation

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: runtime-decisions
- Depends on: DOC-004, DOC-007
- Goal id: DOC-G042
- Outputs: docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md
- Validation: test -f docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md && rg -q '^## Alternatives' docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md && rg -qi 'import' docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/adr-capabilities
- Parallel lane: adr-capabilities
- Resource class: cpu-small
- Predicted files: docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md
- Interfaces: ADR-0003
- Source anchors: model_catalog/; endpoint_usage/; routers; capability probes; MODEL_SERVICE_ROUTING.md
- Conflict policy: Own only ADR-0003; do not edit the ADR index or source guide.
- Preconditions: ADR template and model/service guide are complete.
- Effects: Record lazy optional discovery and separate information, accounting/reservation and invocation planes with their alternatives and consequences.
- Evidence subset: cold import, capability, catalog snapshot, usage receipt, router failure flow
- Acceptance: The ADR explains why importability is not availability and why one mutable registry cannot safely own service identity, live capacity and invocation side effects.

## DOC-018 Record worktree, lease and fencing isolation

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: runtime-decisions
- Depends on: DOC-004, DOC-013
- Goal id: DOC-G042
- Outputs: docs/architecture/decisions/0004-worktrees-leases-and-fencing.md
- Validation: test -f docs/architecture/decisions/0004-worktrees-leases-and-fencing.md && rg -q '^## Alternatives' docs/architecture/decisions/0004-worktrees-leases-and-fencing.md && rg -qi 'stale' docs/architecture/decisions/0004-worktrees-leases-and-fencing.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/adr-isolation
- Parallel lane: adr-isolation
- Resource class: cpu-small
- Predicted files: docs/architecture/decisions/0004-worktrees-leases-and-fencing.md
- Interfaces: ADR-0004
- Source anchors: runtime/resource_scheduler.py; merge/lease_coordination.py; worktree_lifecycle.py; todo_daemon worktree and merge logic; EXECUTION_AND_RECOVERY.md
- Conflict policy: Own only ADR-0004; do not edit the ADR index or source guide.
- Preconditions: ADR template and execution/recovery guide are complete.
- Effects: Record isolated worktrees plus leases/fencing/heartbeats as a distributed-systems safety boundary, including stale workers, dirty checkouts, retries and merge serialization.
- Evidence subset: concurrency failure model, lease epochs, path protection, worker lifecycle
- Acceptance: The ADR evaluates direct shared-checkout and PID-only alternatives and records why neither prevents stale or duplicate effects.

## DOC-019 Record mutable coordination versus immutable replication

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: structure-decisions
- Depends on: DOC-004, DOC-009, DOC-014
- Goal id: DOC-G043
- Outputs: docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md
- Validation: test -f docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md && rg -q 'DuckDB' docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md && rg -q 'IPLD' docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/adr-persistence
- Parallel lane: adr-persistence
- Resource class: cpu-small
- Predicted files: docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md
- Interfaces: ADR-0005
- Source anchors: lease_coordination.py; entrypoints run_registry and coordination modules; verified_ipld_backend.py; multiformats identity; DISTRIBUTED_RUNTIME.md; PROMPT_FIRST_RUNTIME.md
- Conflict policy: Own only ADR-0005; do not edit the ADR index or source guides.
- Preconditions: ADR template, distributed runtime and prompt-first guides are complete.
- Effects: Record single-writer mutable DuckDB coordination and immutable Parquet/IPLD/CAR/IPFS epoch replication, including authority, replay, partition and failure consequences.
- Evidence subset: claim/fence authority, epoch/head identity, replica verification, stale-owner behavior
- Acceptance: The ADR makes clear that immutable replicas cannot grant leases or mutate active state and evaluates multi-writer/shared-file and IPNS-as-authority alternatives.

## DOC-020 Record semantic domain packages and compatibility boundaries

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: structure-decisions
- Depends on: DOC-004, DOC-010, DOC-014
- Goal id: DOC-G043
- Outputs: docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md
- Validation: test -f docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md && rg -q '^## Alternatives' docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md && rg -qi 'compatib' docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/adr-packages
- Parallel lane: adr-packages
- Resource class: cpu-small
- Predicted files: docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md
- Interfaces: ADR-0006
- Source anchors: agent_supervisor domain package __init__/README files; package layout manifests/tests; mcp compatibility facade; router alias tests; INTEGRATION_BOUNDARIES.md; packages/entrypoints.md
- Conflict policy: Own only ADR-0006; do not edit the ADR index or source guides.
- Preconditions: ADR template, integration and entrypoint guides are complete.
- Effects: Record acyclic semantic domain layout, highest-layer entrypoints, independent repository authority and bounded exact compatibility facades; reject board-ID package naming and duplicate mutable state.
- Evidence subset: package DAG, public export manifest, compatibility object identity, git authority
- Acceptance: The ADR explains why delivery programs do not define code ownership and why compatibility must preserve exact behavior/state without becoming a second implementation.

## DOC-021 Refresh installation and first-use guidance

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: onboarding-api
- Depends on: DOC-005, DOC-006
- Goal id: DOC-G051
- Outputs: docs/guides/getting-started/README.md, docs/guides/getting-started/installation.md, docs/guides/getting-started/INSTALLATION.md, docs/guides/QUICKSTART.md
- Validation: test -f docs/guides/getting-started/installation.md && test ! -e docs/guides/getting-started/INSTALLATION.md && rg -q '\[project.optional-dependencies\]' pyproject.toml && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/installation-quickstart
- Parallel lane: installation-quickstart
- Resource class: cpu-medium
- Predicted files: docs/guides/getting-started/README.md, docs/guides/getting-started/installation.md, docs/guides/getting-started/INSTALLATION.md, docs/guides/QUICKSTART.md
- Interfaces: InstallationProfile@1, FirstUseJourney@1
- Source anchors: pyproject.toml; setup.py; requirements*.txt; install/; package __init__.py; cli_entry.py; architecture/SYSTEM_CONTEXT.md; architecture/INFERENCE_RUNTIME.md
- Conflict policy: Own only the four named paths. Consolidate unique current content into lowercase installation.md and remove the case-colliding uppercase tracked file; do not edit top-level indexes.
- Preconditions: System context and inference runtime guides are complete.
- Effects: Align Python range, extras, editable install, optional capabilities, first CLI/Python operation and verification; remove nonexistent extras; disclose the 0.0.45 versus 0.4.0 version-source contradiction as code-owned.
- Evidence subset: packaging table, case-fold check, offline import/help recipe, capability caveats
- Acceptance: The canonical lower-case guide is accurate, the case-insensitive collision is gone, first-use examples use supported surfaces, and no prose guesses which conflicting version declaration should win.

## DOC-022 Refresh Python API and CLI reference entrypoints

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: onboarding-api
- Depends on: DOC-005, DOC-006, DOC-007
- Goal id: DOC-G051
- Outputs: docs/api/overview.md, docs/guides/cli/README_CLI.md
- Validation: rg -q 'ipfs-accelerate agent' docs/api/overview.md && rg -q 'ipfs-accelerate agent' docs/guides/cli/README_CLI.md && rg -q 'agent_supervisor.proof.prover_matrix_registry' docs/api/overview.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/api-cli
- Parallel lane: api-cli
- Resource class: cpu-medium
- Predicted files: docs/api/overview.md, docs/guides/cli/README_CLI.md
- Interfaces: PublicPythonSurface@1, CLIGroupManifest@1
- Source anchors: ipfs_accelerate_py/__init__.py; agent_supervisor public export manifests; pyproject project.scripts; cli_entry.py; cli.py; control/control_cli.py; CLI tests; MODEL_SERVICE_ROUTING.md
- Conflict policy: Own only API overview and CLI README. Do not change CLI code to match stale docs or edit quickstart/indexes.
- Preconditions: System, inference and model/service guides are complete.
- Effects: Document intentional public versus internal imports, optional/lazy exports, console scripts and current CLI groups including agent/copilot/copilot-sdk; fix obsolete prover module invocation; flag CLI help entries that source does not implement.
- Evidence subset: export inventory, stability/availability matrix, CLI help snapshot, module execution checks
- Acceptance: The reference does not imply that all exported optional symbols are healthy; documented commands resolve; nonexistent inference/queue/network groups are not presented as usable merely because stale help mentions them.

## DOC-023 Refresh MCP setup and server reference

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operator-guides
- Depends on: DOC-008
- Goal id: DOC-G052
- Outputs: docs/MCP_SERVER.md, docs/guides/MCP_SETUP_GUIDE.md, docs/guides/QUICK_START_MCP.md
- Validation: rg -q 'mcp_server' docs/MCP_SERVER.md && rg -q 'mcp_server' docs/guides/MCP_SETUP_GUIDE.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/mcp-guides
- Parallel lane: mcp-guides
- Resource class: cpu-medium
- Predicted files: docs/MCP_SERVER.md, docs/guides/MCP_SETUP_GUIDE.md, docs/guides/QUICK_START_MCP.md
- Interfaces: MCPSetupJourney@1, MCPServerReference@1
- Source anchors: mcp_server/__main__.py; mcp_server/server.py; configs.py; tool_registry.py; transports and policy modules; mcp compatibility layer; architecture/MCP_RUNTIME.md; pyproject extras
- Conflict policy: Own only the three named MCP guides. Preserve delivery plans and MCP++ project records.
- Preconditions: MCP runtime architecture is complete.
- Effects: Align installation extras, canonical startup, configuration, transports, tool discovery, policy/security, health and compatibility migration with current code; label external-service requirements.
- Evidence subset: startup/help commands, configuration mapping, tool/transport table, security prerequisites, troubleshooting
- Acceptance: A reader can select and start the canonical server without being routed through a stale compatibility path, while legacy behavior and auto-install/optional dependency caveats are explicit.

## DOC-024 Refresh supervisor operator, developer and agent guides

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operator-guides
- Depends on: DOC-011, DOC-012, DOC-013, DOC-014
- Goal id: DOC-G052
- Outputs: docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md, docs/architecture/agent_supervisor/FOR_AGENTS.md, docs/architecture/agent_supervisor/README.md
- Validation: rg -q 'workflow_preview' docs/guides/AGENT_SUPERVISOR_GUIDE.md && rg -q 'rescue_preview' docs/guides/AGENT_SUPERVISOR_GUIDE.md && rg -q 'agent_supervisor.prompt.prompt_workflow' docs/guides/AGENT_SUPERVISOR_GUIDE.md && python scripts/docs/check_agent_supervisor_docs.py && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/supervisor-guides
- Parallel lane: supervisor-guides
- Resource class: cpu-medium
- Predicted files: docs/guides/AGENT_SUPERVISOR_GUIDE.md, docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md, docs/architecture/agent_supervisor/FOR_AGENTS.md, docs/architecture/agent_supervisor/README.md
- Interfaces: SupervisorOperatorJourney@2, DeveloperPlacementGuide@2, AgentSafetyCapsule@2
- Source anchors: control operation catalog; domain package map; runtime/todo_daemon help; prompt/prompt_workflow.py; entrypoints; new supervisor architecture guides; scripts/docs/check_agent_supervisor_docs.py
- Conflict policy: Own only the four named guides. Do not edit the deep legacy architecture monolith, package code, plans, objectives or task boards.
- Preconditions: All four new supervisor architecture guides are complete.
- Effects: Correct the 31-operation vocabulary including workflow/rescue preview/materialize/restart/rescue; replace non-executable flat module commands; route roles to the new guides and entrypoints page; fix the existing primary-doc ticket-prefix guard failure.
- Evidence subset: operation inventory, executable module paths, package placement, recovery and landed/planned prompt matrix
- Acceptance: Operators, contributors and agents have compact current guidance, the documentation guard passes, and no board ticket or stale plan is used as the primary product vocabulary.

## DOC-025 Refresh deployment, hardware, P2P and troubleshooting journeys

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: operations-validation
- Depends on: DOC-005, DOC-006, DOC-009, DOC-010
- Goal id: DOC-G053
- Outputs: docs/guides/deployment/README.md, docs/guides/hardware/overview.md, docs/guides/p2p/README.md, docs/guides/troubleshooting/faq.md
- Validation: test -f docs/guides/deployment/README.md && test -f docs/guides/hardware/overview.md && test -f docs/guides/p2p/README.md && test -f docs/guides/troubleshooting/faq.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/deployment-troubleshooting
- Parallel lane: deployment-troubleshooting
- Resource class: io-medium
- Predicted files: docs/guides/deployment/README.md, docs/guides/hardware/overview.md, docs/guides/p2p/README.md, docs/guides/troubleshooting/faq.md
- Interfaces: DeploymentCapabilityProfile@1, P2POperatorJourney@1, FailureSymptomMap@1
- Source anchors: deployments/; docker-compose*.yml; install/; hardware detection/probes; p2p_tasks/; p2p workflow modules; DISTRIBUTED_RUNTIME.md; INTEGRATION_BOUNDARIES.md
- Conflict policy: Own only the four landing/current guides. Do not rewrite the many historical infrastructure, completion or fix records.
- Preconditions: System, inference, distributed and integration guides are complete.
- Effects: Route operators to current deployment/hardware/P2P paths, state optional prerequisites and health evidence, and map common capability/CID/network/provider failures to bounded diagnostics.
- Evidence subset: environment matrix, capability probes, P2P prerequisites, health/recovery checks, historical-link routing
- Acceptance: The guides make CPU/local operation the baseline, never promise GPU/browser/IPFS/P2P/provider availability, and do not treat process liveness or synthetic identifiers as health/proof.

## DOC-026 Refresh testing and documentation maintenance guidance

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: operations-validation
- Depends on: DOC-002, DOC-005, DOC-006, DOC-008, DOC-009, DOC-011, DOC-012, DOC-013, DOC-014
- Goal id: DOC-G053
- Outputs: docs/development/testing.md, docs/development/DOCUMENTATION_MAINTENANCE.md
- Validation: rg -q 'agent_supervisor.objectives.objective_daemon' docs/development/testing.md && rg -q 'agent_supervisor.objectives.bundle_supervisor' docs/development/testing.md && rg -q 'test/test_unified_cli_integration.py' docs/development/testing.md && test -f docs/development/DOCUMENTATION_MAINTENANCE.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/testing-maintenance
- Parallel lane: testing-maintenance
- Resource class: cpu-medium
- Predicted files: docs/development/testing.md, docs/development/DOCUMENTATION_MAINTENANCE.md
- Interfaces: TestSelectionGuide@1, DocumentationReviewChecklist@1
- Source anchors: pytest.ini; pyproject.toml; test/; tests/; scripts/docs/check_agent_supervisor_docs.py; .github/workflows/README_DOCUMENTATION_MAINTENANCE.md; DOCUMENTATION_LIFECYCLE.md
- Conflict policy: Own only testing.md and DOCUMENTATION_MAINTENANCE.md. Do not add or modify CI/code in this task; record automation gaps for a separately authorized code task.
- Preconditions: Lifecycle policy and the maintained architecture guides are complete.
- Effects: Correct domain module and unified CLI test paths, distinguish offline/unit/integration/hardware/provider checks, define a non-suppressing doc review checklist, and specify desired PR gates without claiming they already run.
- Evidence subset: actual test tree, executable module paths, optional test prerequisites, current workflow limitations
- Acceptance: Contributors can run focused valid checks; missing network/hardware/provider capability is reported rather than hidden; current workflow/link/export/version drift gaps are explicit follow-up work, not fictional automation.

## DOC-027 Publish glossary, architecture hub and documentation manifest

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: information-architecture
- Depends on: DOC-002, DOC-005, DOC-006, DOC-007, DOC-008, DOC-009, DOC-010, DOC-011, DOC-012, DOC-013, DOC-014, DOC-015, DOC-016, DOC-017, DOC-018, DOC-019, DOC-020
- Goal id: DOC-G061
- Outputs: docs/architecture/GLOSSARY.md, docs/architecture/README.md, docs/development/DOCUMENTATION_MANIFEST.md
- Validation: test -f docs/architecture/GLOSSARY.md && test -f docs/architecture/README.md && test -f docs/development/DOCUMENTATION_MANIFEST.md && rg -q 'Current' docs/development/DOCUMENTATION_MANIFEST.md && rg -q 'Historical' docs/development/DOCUMENTATION_MANIFEST.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/information-architecture
- Parallel lane: information-architecture
- Resource class: io-medium
- Predicted files: docs/architecture/GLOSSARY.md, docs/architecture/README.md, docs/development/DOCUMENTATION_MANIFEST.md
- Interfaces: ProductGlossary@1, ArchitectureAudienceRouter@1, DocumentationManifest@1
- Source anchors: DOCUMENTATION_LIFECYCLE.md; all new product/supervisor architecture guides and ADRs; docs tree; current navigation
- Conflict policy: Own only the three named files. Treat every landed leaf guide/ADR as read-only; do not edit top-level README/INDEX yet.
- Preconditions: Lifecycle policy, architecture guides and ADRs are complete.
- Effects: Normalize semantic terms; route architecture readers by concern/audience; classify documents with owner, status, source paths, last-verified baseline and supersession without pretending all 464 files are maintained.
- Evidence subset: glossary, audience matrix, current/reference/plan/historical/vendored manifest, archive debt summary
- Acceptance: Terms distinguish catalog/usage/router, MCP/MCP++, objective/task, discovery/capability/proof, CID/cache key, merge/acceptance and coordination/replication; readers are never routed to a plan/archive as current without a label.

## DOC-028 Integrate navigation and publish the validation closeout

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: closeout-validation
- Depends on: DOC-021, DOC-022, DOC-023, DOC-024, DOC-025, DOC-026, DOC-027
- Goal id: DOC-G062
- Outputs: docs/README.md, docs/INDEX.md, docs/development/DOCUMENTATION_CURRENT_STATE.md, docs/development/DOCUMENTATION_VALIDATION_2026_08.md
- Validation: python scripts/docs/check_agent_supervisor_docs.py && rg -q 'Documentation baseline' docs/INDEX.md && rg -q 'Last verified' docs/development/DOCUMENTATION_CURRENT_STATE.md && test -f docs/development/DOCUMENTATION_VALIDATION_2026_08.md && git diff --check
- Board namespace: ipfs-accelerate-documentation-refresh-v1
- Bundle: documentation-refresh/validation-closeout
- Parallel lane: validation-closeout
- Resource class: coordinator
- Predicted files: docs/README.md, docs/INDEX.md, docs/development/DOCUMENTATION_CURRENT_STATE.md, docs/development/DOCUMENTATION_VALIDATION_2026_08.md
- Interfaces: DocumentationNavigation@2, DocumentationValidationReceipt@1, DocumentationBaseline@2
- Source anchors: all completed DOC outputs; docs tree; pyproject.toml; CLI and module help; scripts/docs/check_agent_supervisor_docs.py; Git merge-target identity
- Conflict policy: Sole owner of top-level docs navigation and current-state/validation closeout. Earlier leaf outputs are read-only; do not hide failed checks or rewrite source code.
- Preconditions: Every guide and information-architecture task is complete on the merge target.
- Effects: Update audience paths and baseline markers; run/record local-link, case-fold, referenced-file/module/test-path, selected help/example, public-surface and supervisor-doc checks; fix current-surface navigation residuals; separate archive/history debt; record code-owned blockers and next audit triggers.
- Evidence subset: integrated-tree commit/date, commands and return codes, link/path results, known limitations, source-owner follow-ups
- Acceptance: Canonical navigation points to maintained guides, current/reference/plan/history are explicit, all offline-safe current-surface checks pass or have a precise owned blocker, and the exact validation baseline is reproducible.

---

## Dependency DAG

```text
DOC-001 ─┬─ DOC-005 ─┬─ DOC-021 ─┐
         ├─ DOC-006 ─┤            ├─ DOC-028
         ├─ DOC-007 ─┼─ DOC-022 ──┤
         ├─ DOC-008 ─── DOC-023 ──┤
         ├─ DOC-009 ─┬─ DOC-025 ──┤
         ├─ DOC-010 ─┘            │
         ├─ DOC-011 ─┬─ DOC-024 ──┤
         ├─ DOC-012 ─┤            │
         ├─ DOC-013 ─┤            │
         └─ DOC-014 ─┘            │
                                  │
DOC-002 ─────────────── DOC-026 ──┤
DOC-003 ─┬─ DOC-005..014           │
DOC-004 ─┼─ DOC-015..020 ─ DOC-027┤
         └─────────────────────────┘
```

Initial ready tasks `DOC-001` through `DOC-004` intentionally map one per
numeric modulo-4 strict shard. Subsequent tasks retain disjoint predicted
paths; dependency fan-in, not a shared file, controls ordering.

## Operator launch contract

- Target branch: `docs/architecture-refresh-20260803`
- Repository root: `/home/barberb/lift_coding/.worktrees/ipfs-accelerate-docs-refresh`
- Runtime root: `/home/barberb/.local/state/ipfs_accelerate_py/documentation-refresh-v5`
- Lanes: 4 strict deterministic task shards
- Provider: exact `grok-4.5` in the forced-Docker quota route; exact `gpt-5.6-terra` at `medium` only after the exact durable Grok 402 balance-exhausted record is independently reproduced
- Attempts: 3 per canonical task identity
- Retry budgets: implementation 3, validation 3, merge 3
- Timeouts: 7200 seconds ordinary, 10800 seconds hard, 1200 seconds log stall
- Protected paths: this board, its objective heap and the architecture plan
- Refill/migration/janitor: disabled for the sealed tranche
- External network/provider/hardware validation: not required
