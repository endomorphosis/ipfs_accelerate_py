# Agent Supervisor Prompt-Only Entrypoints v2 Delta Objective Heap

This versioned objective heap closes gaps found by the 2026-08-03 audit of
`AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md`. It supplements, rather
than rewrites, the live v1 objective heap. Its executable projection is
`agent_supervisor_prompt_only_entrypoints_v2_delta.todo.md` with task prefix
`## ASE2-`.

The v1 and delta projections may be planned together only after canonical v2
materialization assigns new task and plan CIDs. Until then the running v1
projection remains immutable. Cross-program schedulers must share the same
DuckDB lease/fence coordinator and conflict graph.

## Goal tree

```text
ASE2-G000  Audited prompt-only supervisor v2 delta
|-- ASE2-G010  Trusted ambient evidence and deterministic resolution
|-- ASE2-G020  Exact quota-only provider policy and attempt evidence
|-- ASE2-G030  Persistent signed local-development bootstrap
|-- ASE2-G040  DuckDB run state and effect-bound runtime composition
|-- ASE2-G050  Python, CLI, MCP, and MCP++ facade convergence
`-- ASE2-G060  Conformance, load, migration, and v2 cutover
```

## ASE2-G000 Audited prompt-only supervisor v2 delta

- Status: active
- Parent:
- Parent goal IDs JSON: []
- Depends on:
- Dependencies JSON: []
- Fib priority: 1
- Track: prompt-entrypoints-v2
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints-v2/root
- Parallel lane: program
- Resource class: coordinator
- Goal: Close the audited composition, authority, provider-evidence, mutable-state, and public-facade gaps so one prompt plus trusted ambient context safely creates, adopts, observes, and steers a durable run.
- Producing tasks: ASE2-001, ASE2-002, ASE2-003, ASE2-004, ASE2-005, ASE2-006, ASE2-007, ASE2-008
- Evidence: prompt_only_entrypoints_v2_delta.PROMPT_ONLY_V2_DELTA_REQUIREMENT_ID
- Evidence requirements JSON: ["ambient resolution replay receipt", "exact quota-only provider receipt", "signed local profile lifecycle receipt", "DuckDB run-state CAS receipt", "cross-transport conformance receipt", "fresh load and cutover decision"]
- Evidence criteria: Prompt-only invocation resolves from bounded current evidence; exact grok-4.5 is primary; exact gpt-5.6-terra at medium reasoning can run once only after verified pre-effect quota exhaustion; local mutation requires an installed signed profile; DuckDB owns mutable run heads and CAS; every transport compiles the same launch plan; and fresh safety/load gates pass before v2 cutover.
- Evidence source policy: Plans, prompt text, repository prose, model output, process liveness, stale capability snapshots, and self-reported completion are non-authoritative; accept only current-tree producer receipts bound to exact repository, profile, authority, provider, run, lease, task-source, effect, and validation roots.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools, docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints", "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools", "docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v2_e2e.py test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_prompt_only_v2_load.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_v2_e2e.py test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_prompt_only_v2_load.py -q"]
- Acceptance: A supported local checkout, Python embedder, MCP server, or MCP++ caller can submit one prompt and receive a resumable run handle without supplying low-level daemon flags, while every authority, target, provider, effect, lease, and completion decision remains explicit and fail-closed.
- Gap task: Execute the smallest ready ASE2 task under the shared conflict and lease coordinator.

## ASE2-G010 Trusted ambient evidence and deterministic resolution

- Status: active
- Parent: ASE2-G000
- Parent goal IDs JSON: ["ASE2-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 2
- Track: ambient-inference
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints-v2/ambient-inference
- Parallel lane: ambient-inference
- Resource class: io-small
- Goal: Gather trusted process, repository, installed-profile, server, run-registry, provider-capacity, resource, validation, and topology evidence once and compose the existing leaf resolvers in a deterministic order.
- Producing tasks: ASE2-001, ASE2-005
- Evidence: inference_runtime.AMBIENT_RESOLUTION_REQUIREMENT_ID
- Evidence requirements JSON: ["bounded evidence adapter inventory", "frozen invocation context CID", "deterministic resolution replay", "transport target-denial matrix"]
- Evidence criteria: Evidence is source-labelled, fresh, bounded, content-addressed, and immutable for one resolution; prompt content cannot populate trusted fields; material ambiguity returns preview or one typed question and never starts work.
- Evidence source policy: Ambient environment variables, repository files, symlinks, prompt path mentions, unauthenticated target hints, and stale provider snapshots are untrusted unless a named adapter validates them within an explicit allowlist.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py, ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/context_adapters.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_inference_runtime.py test/api/test_agent_supervisor_context_adapters.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_inference_runtime.py test/api/test_agent_supervisor_context_adapters.py -q"]
- Acceptance: Unchanged trusted evidence produces the same complete profile and receipt across transports; unsafe or ambiguous evidence cannot become authority or an effectful launch.
- Gap task: Repair the smallest missing evidence adapter, freshness check, precedence join, or ambiguity disposition.

## ASE2-G020 Exact quota-only provider policy and attempt evidence

- Status: active
- Parent: ASE2-G000
- Parent goal IDs JSON: ["ASE2-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 2
- Track: provider-policy
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints-v2/provider-policy
- Parallel lane: provider-policy
- Resource class: provider-io
- Goal: Separate immutable pre-launch provider policy from per-attempt evidence and enforce the exact grok-4.5 to gpt-5.6-terra-medium quota-only route.
- Producing tasks: ASE2-002
- Evidence: provider_route.EXACT_QUOTA_FALLBACK_REQUIREMENT_ID
- Evidence requirements JSON: ["exact primary model receipt", "typed quota evidence", "pre-effect boundary proof", "same prompt and scope proof", "fallback attempt receipt", "independent review separation"]
- Evidence criteria: Fallback occurs at most once, only after fresh typed Grok quota exhaustion and before any repository effect; unavailable, capacity, authentication, network, timeout, bare-status, nonzero-exit, unclassified, post-effect, prompt-selected, model-drift, and effort-drift paths fail closed.
- Evidence source policy: Provider names, CLI availability, generic error text, HTTP status alone, environment selection, and model output are non-authoritative; the admitted policy and committed attempt/effect receipts are authoritative.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py, test/api/test_agent_supervisor_typed_provider_fallback.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py", "test/api/test_agent_supervisor_typed_provider_fallback.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_typed_provider_fallback.py test/api/test_agent_supervisor_production_provider_route.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_default_provider_route.py test/api/test_agent_supervisor_typed_provider_fallback.py test/api/test_agent_supervisor_production_provider_route.py -q"]
- Acceptance: The production and compatibility routes bind exact model and reasoning identities and cannot turn any non-quota or post-effect failure into a Terra implementation attempt.
- Gap task: Repair the smallest policy/attempt split, exact-identity binding, quota classifier, effect-boundary, or review-separation residual.

## ASE2-G030 Persistent signed local-development bootstrap

- Status: active
- Parent: ASE2-G000
- Parent goal IDs JSON: ["ASE2-G000"]
- Depends on:
- Dependencies JSON: []
- Fib priority: 2
- Track: local-authority
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints-v2/local-authority
- Parallel lane: local-authority
- Resource class: security-small
- Goal: Replace repeated authority flags with a one-time, inspectable, signed local-development setup that grants only bounded isolated-worktree effects.
- Producing tasks: ASE2-003
- Evidence: local_profile.SIGNED_LOCAL_PROFILE_REQUIREMENT_ID
- Evidence requirements JSON: ["Ed25519 did:key generation receipt", "0600 key-permission check", "signed profile verification", "repository/effect allowlist", "rotation and revocation tests"]
- Evidence criteria: Setup creates or imports a development signing authority, signs exact repository and effect bounds, stores private material with restrictive permissions, and supports explain, rotate, and revoke; no prompt, repository file, or credential presence creates authority.
- Evidence source policy: Unsigned profiles, key handles without signature verification, repository prose, environment values, and filesystem existence alone are non-authoritative.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py, test/api/test_agent_supervisor_local_profile.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py", "test/api/test_agent_supervisor_local_profile.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_local_profile.py test/api/test_agent_supervisor_authority_resolver.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_local_profile.py test/api/test_agent_supervisor_authority_resolver.py -q"]
- Acceptance: `supervisor init` is sufficient for subsequent prompt-only local worktree runs, but never authorizes current-checkout rewrites, merge, push, deploy, arbitrary secrets/network, or destructive cleanup.
- Gap task: Repair the smallest key lifecycle, signature, permission, repository binding, effect ceiling, rotation, or revocation residual.

## ASE2-G040 DuckDB run state and effect-bound runtime composition

- Status: active
- Parent: ASE2-G000
- Parent goal IDs JSON: ["ASE2-G000"]
- Depends on: ASE2-G010, ASE2-G020
- Dependencies JSON: ["ASE2-G010", "ASE2-G020"]
- Fib priority: 3
- Track: runtime-state
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints-v2/runtime-state
- Parallel lane: runtime-state
- Resource class: process-control
- Goal: Converge mutable run heads and revision CAS on DuckDB and require effectful runtime paths to consume and revalidate one complete content-bound LaunchPlan.
- Producing tasks: ASE2-004, ASE2-006
- Evidence: launch_guard.EFFECT_BOUND_LAUNCH_REQUIREMENT_ID
- Evidence requirements JSON: ["DuckDB run-head CAS receipt", "restart/adoption receipt", "immutable Parquet/IPLD history receipt", "tree/authority/lease/fence revalidation", "duplicate-effect denial"]
- Evidence criteria: DuckDB is the sole mutable coordination truth per writable shard; Parquet/IPLD/IPFS carry immutable history and read replicas only; every effect rechecks current tree, authority, lease, fence, task source, provider policy, and idempotency key immediately before execution.
- Evidence source policy: JSON files, shared DuckDB files over distributed filesystems, IPFS/IPNS heads, Parquet replicas, process liveness, and stale launch plans cannot authorize mutable work.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py, ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry_backend.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_duckdb_run_registry.py test/api/test_agent_supervisor_launch_guard.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_duckdb_run_registry.py test/api/test_agent_supervisor_launch_guard.py -q"]
- Acceptance: Restart and concurrent adoption are deterministic; no stale tree, authority, lease, fence, provider policy, or run revision can pass the effect boundary; immutable replicas never become lease authority.
- Gap task: Repair the smallest mutable-head, CAS, migration, adoption, effect revalidation, lease, fence, or replica-authority residual.

## ASE2-G050 Python, CLI, MCP, and MCP++ facade convergence

- Status: active
- Parent: ASE2-G000
- Parent goal IDs JSON: ["ASE2-G000"]
- Depends on: ASE2-G010, ASE2-G030, ASE2-G040
- Dependencies JSON: ["ASE2-G010", "ASE2-G030", "ASE2-G040"]
- Fib priority: 5
- Track: public-facades
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints-v2/public-facades
- Parallel lane: public-facades
- Resource class: io-small
- Goal: Expose one high-level run/preview/steer/status/follow/explain/doctor contract through Python, CLI, MCP, and MCP++, with transport-owned target context and no normal low-level daemon arguments.
- Producing tasks: ASE2-005, ASE2-007
- Evidence: supervisor_facade.PROMPT_FACADE_CONFORMANCE_REQUIREMENT_ID
- Evidence requirements JSON: ["Python CLI MCP MCP++ canonical parity", "cold help/import receipt", "server target allowlist denial", "MCP++ UCAN proof", "transient prompt non-leak receipt"]
- Evidence criteria: Local CLI uses an enclosing Git root plus installed signed profile; Python uses explicit embedder policy or the same local bootstrap; MCP uses server-owned root aliases; MCP++ additionally verifies UCAN; all compile the same canonical invocation and launch plan.
- Evidence source policy: Prompt paths, arbitrary client filesystem paths, transport authentication without inner effect authorization, and adapter-specific defaults are non-authoritative.
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py, ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py, ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py
- Predicted files JSON: ["ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py", "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_mcplusplus_prompt_entrypoints.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_mcplusplus_prompt_entrypoints.py -q"]
- Acceptance: A normal caller supplies a prompt and, only when materially ambiguous, one typed answer; every transport returns the same canonical handle/result and preserves its distinct trust boundary.
- Gap task: Repair the smallest facade, adapter, allowlist, UCAN, cold-start, prompt-broker, or conformance residual.

## ASE2-G060 Conformance, load, migration, and v2 cutover

- Status: active
- Parent: ASE2-G000
- Parent goal IDs JSON: ["ASE2-G000"]
- Depends on: ASE2-G020, ASE2-G030, ASE2-G040, ASE2-G050
- Dependencies JSON: ["ASE2-G020", "ASE2-G030", "ASE2-G040", "ASE2-G050"]
- Fib priority: 8
- Track: verification-rollout
- Priority: P0
- Bundle: agent-supervisor/prompt-entrypoints-v2/closeout
- Parallel lane: closeout
- Resource class: coordinator
- Goal: Prove the refreshed contract under deterministic, adversarial, crash, concurrency, sharding, provider, and transport load before compiling and cutting over a new canonical v2 projection.
- Producing tasks: ASE2-008
- Evidence: prompt_only_v2_rollout.PROMPT_ONLY_V2_ROLLOUT_REQUIREMENT_ID
- Evidence requirements JSON: ["acyclic goal/task graph", "four-lane dry-run", "fresh E2E and conformance", "chaos and load report", "v1 evidence migration map", "signed v2 cutover or rollback decision"]
- Evidence criteria: All gates run on the exact landed tree; completed v1 evidence is mapped without identity forgery; new CIDs are materialized to DuckDB plus Parquet/IPLD; rollout is staged and reversible; no source board is silently rewritten under a live supervisor.
- Evidence source policy: Historical green tests, stale v1 status, task prose, model claims, or a running process alone are non-authoritative; require exact-tree test, materialization, scheduler, and migration receipts.
- Outputs: data/agent_supervisor/prompt_only_entrypoints_v2/plan, data/agent_supervisor/prompt_only_entrypoints_v2/rollout
- Predicted files JSON: ["data/agent_supervisor/prompt_only_entrypoints_v2/plan", "data/agent_supervisor/prompt_only_entrypoints_v2/rollout"]
- Validation: python -m pytest test/api/test_agent_supervisor_prompt_only_v2_e2e.py test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_prompt_only_v2_load.py -q
- Validation commands JSON: ["python -m pytest test/api/test_agent_supervisor_prompt_only_v2_e2e.py test/api/test_agent_supervisor_prompt_only_v2_conformance.py test/api/test_agent_supervisor_prompt_only_v2_load.py -q"]
- Acceptance: The canonical v2 DAG is acyclic, conflict-safe and four-lane schedulable; promotion gates pass; v1 remains recoverable; and the cutover receipt names exact old/new roots, migrated evidence, active coordinator, rollback trigger, and authority.
- Gap task: Repair the smallest conformance, load, migration, materialization, scheduler, rollout, or rollback residual.
