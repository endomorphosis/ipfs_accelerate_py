# Agent Supervisor Prompt-Only Entrypoints v2 Delta Task Board

This executable delta implements
`agent_supervisor_prompt_only_entrypoints_v2_delta.objectives.md` and closes
the audited residuals in
`AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md`.

Use task prefix `## ASE2-`. Tasks may run in parallel only after dependencies
complete and the shared conflict/resource scheduler admits their predicted
files. A scheduler running this delta alongside the v1 program must use the
same DuckDB lease/fence coordinator and merge queue. Do not mutate the live
v1 taskboard or reuse its canonical task CIDs; ASE2-008 owns evidence-preserving
v2 materialization and cutover.

The dependency graph admits four conflict-free waves:

```text
Wave 0 (parallel): ASE2-001 ambient inference
                   ASE2-002 provider policy/attempt evidence
                   ASE2-003 signed local bootstrap
                   ASE2-004 DuckDB run registry
Wave 1 (parallel): ASE2-005 transport context adapters
                   ASE2-006 complete LaunchPlan effect guard
Wave 2:            ASE2-007 Python/CLI/MCP/MCP++ facade convergence
Wave 3:            ASE2-008 validation, materialization, and staged cutover
```

## ASE2-001 Collect trusted ambient evidence and orchestrate prompt-only resolution

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ambient-inference
- Depends on:
- Goal id: ASE2-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, test/api/test_agent_supervisor_inference_runtime.py
- Validation: python -m pytest test/api/test_agent_supervisor_inference_runtime.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/inference-runtime
- Parallel lane: ambient-inference
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, test/api/test_agent_supervisor_inference_runtime.py
- Interfaces: InvocationContext, InferenceEvidenceCollector, SupervisorResolutionService
- Conflict policy: Compose existing leaf resolvers and verified adapters only; do not dispatch providers, mutate repositories, start processes, or own transport policy. Share the program DuckDB coordinator with v1 lanes.
- Preconditions: Entrypoint contracts, target/state/objective/authority/capability/profile resolvers, prompt broker, and run registry import successfully on the current tree.
- Effects: Gather bounded source-labelled process, CWD, repository, installed-profile, server, run, provider-capacity, host-resource, validation-policy, and topology evidence; freeze one invocation context; call leaf resolvers in dependency order; emit one complete profile, safe preview, or one typed ambiguity continuation.
- Evidence subset: adapter provenance map, freshness decisions, frozen context CID, leaf/root CID join, deterministic replay, prompt non-influence matrix
- Acceptance: Local CWD plus an installed signed profile or authenticated server context needs no low-level target/profile flags; prompt text cannot populate allowlist, caller, policy, provider, validation argv, or authority; material ambiguity never launches; unchanged evidence yields an identical receipt.

## ASE2-002 Separate exact provider policy from per-attempt and fallback evidence

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-policy
- Depends on:
- Goal id: ASE2-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, test/api/test_agent_supervisor_typed_provider_fallback.py
- Validation: python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_typed_provider_fallback.py test/api/test_agent_supervisor_production_provider_route.py test/api/test_agent_supervisor_entrypoint_contracts.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/provider-route
- Parallel lane: provider-policy
- Resource class: provider-io
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py, test/api/test_agent_supervisor_typed_provider_fallback.py
- Interfaces: ProviderRoutePolicy, ProviderAttemptReceipt, QuotaExhaustionEvidence, ProviderFallbackReceipt, IndependentReviewContinuation
- Conflict policy: Sole owner of the v2 provider-policy/attempt split and exact fallback receipt semantics; conflict with v1 ASE-009 and ASE-036 on shared files; do not conflate implementation fallback with independent review.
- Preconditions: Existing exact legacy route tests and capability resolver are available; repository-effect boundaries have stable identifiers.
- Effects: Bind the pre-launch route to `grok-4.5`; allow one exact `gpt-5.6-terra` attempt with `medium` reasoning only after fresh typed Grok quota-exhaustion evidence and before any repository effect; commit prompt/scope/worktree/budget/authorization equality and a distinct attempt identity before fallback.
- Evidence subset: policy CID, exact model identities, typed quota evidence, attempt receipt, effect-boundary proof, prompt/scope equality, once-only proof, review separation
- Acceptance: Unavailable, capacity, authentication, network, timeout, bare status, nonzero exit, unclassified, post-effect, repeated, model-drift, effort-drift, prompt-selected, scope-widening, and self-review cases fail closed; exact quota-only fallback is reproducible and idempotent.

## ASE2-003 Add one-time signed local-development profile initialization

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: local-authority
- Depends on:
- Goal id: ASE2-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, test/api/test_agent_supervisor_local_profile.py
- Validation: python -m pytest test/api/test_agent_supervisor_local_profile.py test/api/test_agent_supervisor_authority_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/local-profile
- Parallel lane: local-authority
- Resource class: security-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, test/api/test_agent_supervisor_local_profile.py
- Interfaces: LocalProfileInitializer, DevelopmentSigningAuthority, SignedSupervisorProfile, ProfileRotationReceipt, ProfileRevocationReceipt
- Conflict policy: Own local key/profile lifecycle only; reuse canonical DID/signature primitives and authority resolver; never place private keys or grants in repository files, logs, prompts, argv, or immutable public replicas.
- Preconditions: Authority resolution and canonical signature verification are available; platform config/state roots can be resolved without importing provider runtimes.
- Effects: Implement `supervisor init` support that creates or imports an Ed25519 `did:key`, writes private material with mode 0600, signs exact repository/state/effect bounds, verifies on load, and supports inspect, rotate, and revoke.
- Evidence subset: key generation/import receipt, permissions, signed profile CID, repository binding, effect ceiling, verification, rotation, revocation, leak scan
- Acceptance: One explicit setup enables subsequent prompt-only isolated-worktree edit/test runs; unsigned, tampered, permissive, wrong-repository, revoked, or prompt-derived profiles fail closed; merge, push, deploy, destructive cleanup, arbitrary secrets/network, and current-checkout rewrite remain denied.

## ASE2-004 Converge mutable run registry state on DuckDB with immutable IPLD history

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-state
- Depends on:
- Goal id: ASE2-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, test/api/test_agent_supervisor_duckdb_run_registry.py
- Validation: python -m pytest test/api/test_agent_supervisor_run_registry.py test/api/test_agent_supervisor_duckdb_run_registry.py test/api/test_agent_supervisor_coordination_epoch.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/duckdb-run-registry
- Parallel lane: run-state
- Resource class: io-database
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py, test/api/test_agent_supervisor_duckdb_run_registry.py
- Interfaces: RunRegistryBackend, DuckDBRunRegistryBackend, ImmutableRunEpoch, RunRevisionCAS, JsonRunRegistryImporter
- Conflict policy: Sole owner of v2 mutable run-head persistence; share schema/transaction primitives with lease coordination; preserve JSON as bounded import/rollback input only and never make Parquet/IPLD/IPFS a mutable lease authority.
- Preconditions: Current RunRegistry contracts, DuckDB coordination transactions, strict CID adapter, and Parquet/IPLD epoch contracts are available.
- Effects: Store mutable run heads, current revisions, adoption keys, cursors, idempotency keys, and CAS in the owning DuckDB shard; export committed immutable epochs to Parquet/IPLD/IPFS; migrate and verify legacy JSON records; reconstruct read-only query replicas.
- Evidence subset: schema migration, transaction/CAS receipts, concurrent adoption, restart reconstruction, logical epoch parity, replica authority denial, corrupt-input quarantine
- Acceptance: Conflicting updates cannot both win; restart reconstructs the same handle; one compatible healthy process is adopted; legacy JSON migration is lossless and idempotent; immutable replicas are queryable but cannot claim, fence, or accept effects.

## ASE2-005 Implement transport-specific trusted invocation-context adapters

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: public-facades
- Depends on: ASE2-001, ASE2-003
- Goal id: ASE2-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, test/api/test_agent_supervisor_context_adapters.py
- Validation: python -m pytest test/api/test_agent_supervisor_context_adapters.py test/api/test_agent_supervisor_authority_resolver.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/context-adapters
- Parallel lane: transport-context
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py, test/api/test_agent_supervisor_context_adapters.py
- Interfaces: LocalInvocationContextFactory, PythonInvocationContextFactory, MCPInvocationContextFactory, MCPPlusPlusInvocationContextFactory
- Conflict policy: Own transport-to-trusted-context adaptation only; never encode resolution policy inside an adapter or accept client/prompt filesystem paths outside its configured root aliases.
- Preconditions: Ambient resolution service and signed local profile lifecycle exist; MCP authentication and MCP++ UCAN validators are importable.
- Effects: For CLI, bind the nearest unambiguous enclosing Git root and installed signed profile; for Python, use explicit embedder allowlists or the same local default; for MCP, map server-owned target aliases; for MCP++, additionally bind verified UCAN invocation capability and attenuation.
- Evidence subset: adapter inputs, target allowlist decision, authenticated principal, signed profile or server policy, UCAN verification, canonical context CID, cross-transport denial matrix
- Acceptance: Identical authorized target/prompt inputs yield equivalent resolution while distinct trust sources remain visible; arbitrary client paths, prompt path injection, symlink escape, unauthenticated identity, absent UCAN, and transport-only authorization cannot reach mutation.

## ASE2-006 Require a complete revalidated LaunchPlan at every effect boundary

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-composition
- Depends on: ASE2-001, ASE2-002, ASE2-004
- Goal id: ASE2-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, test/api/test_agent_supervisor_launch_guard.py
- Validation: python -m pytest test/api/test_agent_supervisor_launch_guard.py test/api/test_agent_supervisor_runtime_factory.py test/api/test_agent_supervisor_lifecycle_orchestrator.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/launch-guard
- Parallel lane: runtime-composition
- Resource class: process-control
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py, ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py, test/api/test_agent_supervisor_launch_guard.py
- Interfaces: CompleteLaunchPlanGuard, EffectBoundarySnapshot, LaunchRevalidationReceipt, StandardSupervisorRuntimeFactory
- Conflict policy: Own high-level runtime composition and final effect guard; reuse control authorization, lifecycle, task-source, provider-route, DuckDB lease/fence, validation, rescue, and receipt services without creating a second mutation path.
- Preconditions: Resolution service, provider policy/attempt split, and DuckDB run registry are complete; low-level control and lifecycle contracts expose current tree/lease/fence checks.
- Effects: Compile one immutable complete LaunchPlan; reject partial plans; immediately before every write/provider/process effect re-observe and compare repository tree, scope, authority, policy, provider route, run revision, task-source root, idempotency key, DuckDB owner lease, and fence; record intended and observed effects.
- Evidence subset: complete-plan proof, pre-effect snapshots, stale-field denial matrix, intended/observed effects, retry/adoption receipt, duplicate-process denial
- Acceptance: No effectful facade can bypass the guard; stale or incomplete inputs fail before effects; exact replay adopts or returns the prior result; crashes at every intent/effect/receipt boundary have one deterministic continuation.

## ASE2-007 Converge prompt-only Python, CLI, MCP, and MCP++ facades

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: public-facades
- Depends on: ASE2-005, ASE2-006
- Goal id: ASE2-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py, test/api/test_agent_supervisor_prompt_only_v2_conformance.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_mcplusplus_prompt_entrypoints.py test/api/test_agent_supervisor_prompt_only_e2e.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/public-facades
- Parallel lane: public-facades
- Resource class: io-small
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py, test/api/test_agent_supervisor_prompt_only_v2_conformance.py
- Interfaces: Supervisor.open, Supervisor.run, Supervisor.preview, Supervisor.steer, Supervisor.status, Supervisor.follow, Supervisor.explain, Supervisor.doctor
- Conflict policy: Sole integration owner for high-level public facades and stable exports; preserve existing expert `agent` operations and canonical low-level request semantics; adapters contain no independent inference, authorization, provider, or mutation policy.
- Preconditions: Transport context factories and guarded standard runtime exist; transient prompt broker, semantic steering, event follow, and MCP++ UCAN paths are available.
- Effects: Add `ipfs-accelerate supervisor` run/preview/steer/status/follow/explain/doctor/init commands and compatibility aliases; export the same Python facade; register equivalent MCP and MCP++ tools; infer current run only when unique and ask one typed question only for material ambiguity.
- Evidence subset: command/API/tool manifests, canonical request/result parity, run-handle equality, cold help/import, prompt non-leak, UCAN attenuation, expert compatibility
- Acceptance: A normal caller supplies only a prompt for run/preview and a prompt plus optional run handle for steer; status/follow infer the sole compatible run; all transports agree on canonical outcomes and exit/error semantics; advanced flags remain explicit overrides rather than requirements.

## ASE2-008 Validate, materialize, and stage the canonical v2 cutover

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: verification-rollout
- Depends on: ASE2-002, ASE2-003, ASE2-004, ASE2-007
- Goal id: ASE2-G060
- Outputs: data/agent_supervisor/prompt_only_entrypoints_v2/plan, data/agent_supervisor/prompt_only_entrypoints_v2/rollout, test/api/test_agent_supervisor_prompt_only_v2_load.py
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v2_e2e.py test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_prompt_only_v2_load.py test/api/test_agent_supervisor_prompt_only_security.py test/api/test_agent_supervisor_prompt_only_chaos.py -q
- Board namespace: agent-supervisor-prompt-only-entrypoints-v2-delta
- Bundle: agent-supervisor/prompt-entrypoints-v2/closeout
- Parallel lane: closeout
- Resource class: coordinator
- Predicted files: data/agent_supervisor/prompt_only_entrypoints_v2/plan, data/agent_supervisor/prompt_only_entrypoints_v2/rollout, test/api/test_agent_supervisor_prompt_only_v2_load.py
- Interfaces: V1EvidenceMigrationMap, CanonicalV2Plan, V2BundleIndex, V2RolloutDecision, V2RollbackReceipt
- Conflict policy: Evaluation and materialization fan-in only; do not weaken fixtures, thresholds, authority, provider route, evidence rules, or tests to obtain promotion; never cut over while v1 holds an active effectful lease.
- Preconditions: All delta producers are landed and independently reviewed; base v1 task/goal sources and materialized CIDs are readable; exact-tree validation and a shared DuckDB coordinator are available.
- Effects: Reconcile implemented v1 outputs against current-tree evidence; compile a consolidated acyclic v2 goal/task graph with new CIDs; write DuckDB, Parquet, IPLD, bundle, conflict, and dependency projections; dry-run four lanes; run deterministic replay, cross-transport E2E, security, crash, provider, concurrent launch/steer, sharding, replication, and sustained load gates; stage observe/preview/assist/local-auto cutover with rollback.
- Evidence subset: v1-to-v2 identity/evidence map, DAG lint, task/goal/plan roots, DuckDB/Parquet/IPLD parity, four-lane dry-run, E2E/security/chaos/load reports, signed promotion or rollback decision
- Acceptance: No unknown dependency, duplicate ID, cycle, predicted-file conflict, stale completion, unauthorized effect, prompt/secret leak, duplicate process/effect, mutable replica authority, provider-route drift, or transport mismatch is accepted; cutover and rollback name exact roots, coordinator, authority, and trigger.
