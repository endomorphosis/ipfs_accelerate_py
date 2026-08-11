# Semantic compression harness supervisor taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `SCH-`.

Protected companion artifacts:

- `docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md`
- `docs/architecture/semantic_compression_harness.objectives.md`
- `docs/architecture/semantic_compression_harness.todo.md`

These control documents are reviewed inputs and may be changed only by the
operator while sealing `SCH-000`; implementation workers must not edit them.
`SCH-000` is deliberately open. Do not launch implementation lanes until it
contains exact final `ipfs_datasets_py` and `ipfs_kit_py` commit pins and is
manually marked complete.

Implementation is focused in
`ipfs_accelerate_py.agent_supervisor.semantic_state`. Workers consume the pinned
semantic-index contracts rather than recreating scanner/graph/invalidation
authority. They must reuse MCP++ Profile A/B/F wire rules, lazy
`DurableCoordinationStore` plus the pinned generic root-CAS port,
`ResourceScheduler`, `ProviderExecutionGateway`, `WorktreeLifecycleStore`,
`LeaseCoordinator`, validation command contracts, and `runtime.event_log`.
They must not use `PersistentTaskQueue` as authority, the legacy mock hardware
or inference coordinator, `compute_artifact_cid` for new artifacts, or a silent
simulation fallback in production.

## Parallel waves

```text
A0  SCH-000
A1  SCH-001 | SCH-002 | SCH-003 | SCH-014
A2  SCH-004 | SCH-006 | SCH-016
A3  SCH-005 | SCH-007 | SCH-010
A4  SCH-008
A5  SCH-009
A6  SCH-011
A7  SCH-012
A8  SCH-013 | SCH-015 | SCH-017
A9  SCH-018
```

## SCH-000 Pin and validate phase-two contract authorities

- Status: todo
- Completion: manual
- Priority: P0
- Track: control
- Depends on:
- Goal id: SCH-G010
- Outputs: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md, docs/architecture/semantic_compression_harness.objectives.md, docs/architecture/semantic_compression_harness.todo.md, config/semantic_state_dependencies.seal.json
- Validation: git rev-parse HEAD
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/control
- Parallel lane: sch-control
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: operator-only
- Context budget tokens: 0
- LLM context budget bytes: 0
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 1 through 3
- Predicted files: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md, docs/architecture/semantic_compression_harness.objectives.md, docs/architecture/semantic_compression_harness.todo.md, config/semantic_state_dependencies.seal.json
- Predicted symbols: SemanticStateDependencySeal
- Interfaces: SemanticStateDependencySeal@1
- Conflict policy: Operator-only launch gate. No implementation worker may infer, update, merge, or bypass dependency pins or edit the protected control files.
- Preconditions: Accelerate baseline is `ea11293bb996f052d620eae989f5377a956764b1`; MCP++ authority is `dc3164653a48d059ae9812078359daeafb451c07`; final datasets semantic-index and kit durable-root commits have been supplied by their closeouts.
- Effects: Replaces both `UNRESOLVED` values with exact reachable 40-hex commits, records repository origins/schema versions/test commands, runs producer contract tests, and seals their outputs for every downstream worker.
- Acceptance: Seal rejects unresolved, mutable, unreachable, dirty, schema-incompatible, or failing dependencies; datasets exposes the required scan/diff/invalidate/explain/watch state contracts; kit exposes direct lazy `DurableCoordinationStore` plus generic read-root/expected-old-CAS/recover; the board remains acyclic and only then is SCH-000 marked complete.

## SCH-001 Implement MCP++ wire codec and interface descriptor

- Status: todo
- Completion: auto
- Priority: P0
- Track: wire-contracts
- Depends on: SCH-000
- Goal id: SCH-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/contracts.py, ipfs_accelerate_py/agent_supervisor/semantic_state/wire.py, ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/semantic-state-harness.interface.json, tests/agent_supervisor/semantic_state/test_wire.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_wire.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/wire-contracts
- Parallel lane: sch-wire
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 36000
- LLM context budget bytes: 294912
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 3, 6, and 13
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/contracts.py, ipfs_accelerate_py/agent_supervisor/semantic_state/wire.py, ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/semantic-state-harness.interface.json, tests/agent_supervisor/semantic_state/test_wire.py
- Predicted symbols: Availability, UnavailableResult, HarnessMode, WorkKind, SemanticCapsule, ContextPack, ModelRoute, PatchProposal, TestSelection, VerificationReceipt, HarnessResult, SemanticStateWireCodec, semantic_state_interface_descriptor
- Interfaces: SemanticStateHarnessContracts@1, SemanticStateMcpWire@1
- Conflict policy: MCP++ Profile A/B/F at the sealed commit is wire authority. Use `canonicalize_artifact` plus `kubo_cid.cid_for_bytes`; never use `compute_artifact_cid` pseudo-CIDs, copy CID code, mutate MCP server transport, or recompute datasets identities.
- Preconditions: SCH-000 sealed exact commits and MCP++ conformance inputs.
- Effects: Defines closed deterministic harness records, a Profile A descriptor, Profile B request/result envelopes, Profile F event nodes, strict decoding, bounded errors, and Kubo-compatible CIDv1 vectors.
- Acceptance: Equivalent payloads have identical canonical bytes and real CIDv1; unknown fields/enums or forged CIDs fail closed; semantic-state CIDs remain opaque verified references; descriptor/interface changes have detectable CIDs; ordinary imports perform no I/O.

## SCH-002 Implement the pinned datasets semantic-state adapter

- Status: todo
- Completion: auto
- Priority: P0
- Track: datasets-adapter
- Depends on: SCH-000
- Goal id: SCH-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py, tests/agent_supervisor/semantic_state/test_datasets_adapter.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_datasets_adapter.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/datasets-adapter
- Parallel lane: sch-datasets
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 34000
- LLM context budget bytes: 278528
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 1, 3, 7, and 9
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py, tests/agent_supervisor/semantic_state/test_datasets_adapter.py
- Predicted symbols: SemanticStateProvider, IpfsDatasetsSemanticStateProvider, SemanticStateCapability, SemanticStateUnavailable, load_semantic_state_provider
- Interfaces: SemanticStateProvider@1, IncrementalSemanticIndexModels@1
- Conflict policy: Lazy-load only the sealed public semantic-index surface. Do not parse target AST, reconstruct symbol IDs/edges/deltas, import target repositories, claim complete Python semantics, or accept an ambient incompatible package.
- Preconditions: Sealed datasets commit passes its semantic-index tests and exposes scan_repository, diff_repository_states, calculate_invalidation, explain_symbol, explain_impact, and watch_repository.
- Effects: Capability-checks schema/extractor versions, invokes canonical Git/tree scans, validates state/delta/invalidation CIDs and four confidence values, and converts missing capability into typed unavailability.
- Acceptance: Clean and incremental scans round-trip without identity translation; watcher events trigger scans but never become state; mismatched commit/schema/CID fails closed; every opaque/invalid state remains visible and requires raw source.

## SCH-003 Implement the narrow kit durable-root adapter

- Status: todo
- Completion: auto
- Priority: P0
- Track: durable-state
- Depends on: SCH-000
- Goal id: SCH-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/durable_state.py, tests/agent_supervisor/semantic_state/test_durable_state.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_durable_state.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/durable-state
- Parallel lane: sch-durable
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 36000
- LLM context budget bytes: 294912
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 3 and 13
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/durable_state.py, tests/agent_supervisor/semantic_state/test_durable_state.py
- Predicted symbols: DurableSemanticStatePort, IpfsKitDurableStateAdapter, DurableStateUnavailable, RootConflict, open_local_durable_state
- Interfaces: DurableSemanticStatePort@1, SemanticStateRootCAS@1
- Conflict policy: Directly and lazily adapt the sealed `ipfs_kit_py.mcp_server.mcplusplus.coordination_storage.DurableCoordinationStore` and sealed generic root-CAS protocol. Do not duplicate local blocks/WAL, use a facade/provisional path, VerifiedIPLDBackend, Iroh/bucket CAS, or require a daemon/network.
- Preconditions: SCH-000 pins a kit commit with `DurableCoordinationStore(storage_dir, backend=None)`, put/get/get_bytes/has/recover, and generic read_root/compare_and_swap_root behavior.
- Effects: Exposes verified put/get/has, expected-old root CAS, replay, corruption checks, and interrupted-transition recovery behind one injected protocol; remote block transport remains optional.
- Acceptance: Hermetic local tests use an explicit temporary storage directory and no daemon; supplied authoritative CID must match bytes; interrupted publication retains or completes one valid root; corrupted blocks fail closed; two distinct writers from one expected root yield at most one success.

## SCH-004 Define scheduling and execution contracts

- Status: todo
- Completion: auto
- Priority: P0
- Track: scheduling-contracts
- Depends on: SCH-001
- Goal id: SCH-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/scheduling_contracts.py, tests/agent_supervisor/semantic_state/test_scheduling_contracts.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_scheduling_contracts.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/scheduling-contracts
- Parallel lane: sch-scheduling-contracts
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 32000
- LLM context budget bytes: 262144
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 6 and 12
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/scheduling_contracts.py, tests/agent_supervisor/semantic_state/test_scheduling_contracts.py
- Predicted symbols: SemanticWorkRequest, SemanticWorkResult, SemanticWorkStatus, CancellationToken, LeaseBinding, ResourceBinding, ProviderBinding
- Interfaces: SemanticWorkScheduling@1
- Conflict policy: Define a narrow projection over existing runtime contracts; do not introduce a scheduler, queue authority, provider registry, mock coordinator, or unbounded observation payload.
- Preconditions: Closed wire/harness records exist.
- Effects: Specifies idempotent work identities for scan, capsule, selection, context, model, static, pytest, prover, and persistence stages plus typed cancellation/unavailability/fencing results.
- Acceptance: Records are deterministic, bounded, secret/source-body free in scheduler observations, distinguish unavailable/cancelled/failed/simulated, and cannot interpret scheduling success as verification success.

## SCH-005 Implement the existing-supervisor scheduling adapter

- Status: todo
- Completion: auto
- Priority: P0
- Track: scheduling
- Depends on: SCH-004
- Goal id: SCH-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/scheduling.py, tests/agent_supervisor/semantic_state/test_scheduling.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_scheduling.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/scheduling
- Parallel lane: sch-scheduling
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Provider role: codex-implement
- Context budget tokens: 40000
- LLM context budget bytes: 327680
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 3 and 12
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/scheduling.py, tests/agent_supervisor/semantic_state/test_scheduling.py
- Predicted symbols: SemanticSchedulingAdapter, ScheduledAttempt, schedule_semantic_work, replay_semantic_work
- Interfaces: SemanticSchedulingAdapter@1, ResourceScheduler@existing, ProviderExecutionGateway@existing
- Conflict policy: Compose ResourceScheduler, ProviderExecutionGateway, WorktreeLifecycleStore, LeaseCoordinator, and runtime.event_log only. `PersistentTaskQueue` is not authority. Never use legacy mock hardware/inference, bypass leases/fences, or treat a simulated provider result as available production work.
- Preconditions: SCH-004 work contracts are closed and existing runtime modules pass focused regressions.
- Effects: Performs resource admission, lease/fence acquisition, cancellation propagation, exact-attempt idempotency, bounded event journaling, and restart/replay for every harness work kind.
- Acceptance: Capacity/provider absence returns typed unavailable; cancellation reaches subprocess/provider boundary; replay does not reinvoke a terminal provider call; expired fences cannot publish; cold import starts no resources, threads, processes, databases, or network calls.

## SCH-006 Compile capsules and assurance-aware ContextPacks

- Status: todo
- Completion: auto
- Priority: P0
- Track: context
- Depends on: SCH-001, SCH-002
- Goal id: SCH-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/capsules.py, ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py, tests/agent_supervisor/semantic_state/test_capsules.py, tests/agent_supervisor/semantic_state/test_context_pack.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_capsules.py tests/agent_supervisor/semantic_state/test_context_pack.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/context
- Parallel lane: sch-context
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Provider role: codex-implement
- Context budget tokens: 40000
- LLM context budget bytes: 327680
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 6 through 11
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/capsules.py, ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py, tests/agent_supervisor/semantic_state/test_capsules.py, tests/agent_supervisor/semantic_state/test_context_pack.py
- Predicted symbols: SemanticCapsuleCompiler, CapsuleCache, CapsuleAdmission, ContextPacker, ContextCoveragePolicy, ContextTokenEstimate, compile_capsule, pack_context
- Interfaces: SemanticCapsule@1, ContextPack@1
- Conflict policy: Compile only from authoritative semantic records and exact source retrieval. Docstrings/model summaries are hints only. Exact targets/edit context/tests are never compressed; heuristic, opaque, stale, invalid, unknown, or insufficient facts force raw source rather than truncation.
- Preconditions: MCP++ records and the sealed datasets provider are available.
- Effects: Builds version/config/policy/interface-bound capsules, safely reuses unchanged dependencies, chooses source versus capsule by confidence/freshness, minimizes counterexamples, accounts tokens, and explains included/excluded context and escalation.
- Acceptance: Exact/conservative substitution rules and visible caveats pass; no LLM summary can raise confidence or satisfy proof; all required ContextPack fields are present; budget failure recommends escalation; identical inputs yield identical pack/capsule CIDs and token accounting.

## SCH-007 Implement model routing and real-provider adapters

- Status: todo
- Completion: auto
- Priority: P0
- Track: model-routing
- Depends on: SCH-001, SCH-004
- Goal id: SCH-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/semantic_state/providers.py, tests/agent_supervisor/semantic_state/test_routing.py, tests/agent_supervisor/semantic_state/test_providers.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_routing.py tests/agent_supervisor/semantic_state/test_providers.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/model-routing
- Parallel lane: sch-routing
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 36000
- LLM context budget bytes: 294912
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 6 and 12
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/routing.py, ipfs_accelerate_py/agent_supervisor/semantic_state/providers.py, tests/agent_supervisor/semantic_state/test_routing.py, tests/agent_supervisor/semantic_state/test_providers.py
- Predicted symbols: ModelRoutingPolicy, ModelProvider, ModelCapability, ProductionProviderGate, route_model, invoke_model
- Interfaces: ModelRouting@1, ModelProvider@1
- Conflict policy: Do not hardcode a provider or add mock inference. Route only on context size, confidence, risk, dependency cone, obligations, prior failures, and proof availability. Production cannot silently replay or simulate a patch.
- Preconditions: Closed wire and scheduling records exist.
- Effects: Selects deterministic_only/small_local_model/medium_model/frontier_model/human_review_required, capability-checks injected providers, and maps absence/errors to typed unavailable results through ProviderExecutionGateway.
- Acceptance: Route decision is deterministic and explained; high-risk/opaque/oversized/failed cases escalate; missing provider is unavailable and nonzero; simulated output is labeled development-only and is never eligible for production verification or root commit.

## SCH-008 Select and execute static checks, pytest tests, and provers

- Status: todo
- Completion: auto
- Priority: P0
- Track: verification
- Depends on: SCH-004, SCH-005
- Goal id: SCH-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/test_selection.py, ipfs_accelerate_py/agent_supervisor/semantic_state/verification.py, tests/agent_supervisor/semantic_state/test_test_selection.py, tests/agent_supervisor/semantic_state/test_verification.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_test_selection.py tests/agent_supervisor/semantic_state/test_verification.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/verification
- Parallel lane: sch-verification
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Provider role: codex-implement
- Context budget tokens: 40000
- LLM context budget bytes: 327680
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 9, 10, and 12
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/test_selection.py, ipfs_accelerate_py/agent_supervisor/semantic_state/verification.py, tests/agent_supervisor/semantic_state/test_test_selection.py, tests/agent_supervisor/semantic_state/test_verification.py
- Predicted symbols: TestSelector, SelectionPolicy, VerificationRunner, StaticCheckResult, PytestResult, ProverResult, FullSuiteComparison, select_tests, compare_full_suite
- Interfaces: TestSelection@1, SemanticVerification@1
- Conflict policy: Reuse validation command parsing/classification and execute bounded explicit commands only inside the fenced worktree. Do not import targets for collection, guess dynamic pytest/plugin behavior as exact, report unavailable provers as passing, or disable full-suite fallback.
- Preconditions: Scheduling adapter is cancellation/fence safe and semantic graph/test facts are supplied through SCH-002 at runtime.
- Effects: Selects tests from direct/import/caller/fixture/schema/config/generated/proof edges plus user rules, records reason paths, runs static/pytest/prover stages, and optionally/mandatorily runs the full-suite oracle by assurance policy.
- Acceptance: Direct known dependents are all selected; ambiguity and opaque/config/dependency changes trigger visible fallback; commands bind exact tree/config/toolchain; timeouts/cancellation are typed; controlled comparison metrics define false negatives correctly and support 100 percent recall.

## SCH-009 Emit MCP++ receipts and enforce freshness admission

- Status: todo
- Completion: auto
- Priority: P0
- Track: receipts
- Depends on: SCH-001, SCH-003, SCH-008
- Goal id: SCH-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/receipts.py, tests/agent_supervisor/semantic_state/test_receipts.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_receipts.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/receipts
- Parallel lane: sch-receipts
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 36000
- LLM context budget bytes: 294912
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 9 and 13
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/receipts.py, tests/agent_supervisor/semantic_state/test_receipts.py
- Predicted symbols: ReceiptCompiler, ReceiptFreshnessPolicy, ReceiptAdmission, StaleReceiptError, compile_verification_receipt, admit_receipt
- Interfaces: SemanticVerificationReceipt@1, ReceiptFreshnessAdmission@1
- Conflict policy: Receipts are MCP++ Profile B artifacts linked by Profile F events and real CIDv1. Operational/scheduler/provider receipts do not prove correctness. Never admit stale, corrupted, incomplete, unavailable, simulated, or mismatched evidence.
- Preconditions: Wire codec, durable port, and verification results exist.
- Effects: Binds receipts to exact trees/roots/delta/selection/commands/toolchain/dependency-lock/config/policy/interface/provider/proof/output identities, stores them before reference, and emits sorted stale obligations on any binding change.
- Acceptance: Rehash and closed-schema validation pass; any bound input change stales the receipt; policy/interface changes invalidate decisions/adapters; unavailable proof is explicit; a stale or simulation receipt cannot satisfy verification or state-root promotion.

## SCH-010 Implement safe fenced worktree and patch validation

- Status: todo
- Completion: auto
- Priority: P0
- Track: worktree
- Depends on: SCH-004
- Goal id: SCH-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/worktree.py, tests/agent_supervisor/semantic_state/test_worktree.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_worktree.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/worktree
- Parallel lane: sch-worktree
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 34000
- LLM context budget bytes: 278528
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 12 and 14
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/worktree.py, tests/agent_supervisor/semantic_state/test_worktree.py
- Predicted symbols: IsolatedWorktree, PatchValidator, PatchScope, PatchValidationError, create_isolated_worktree, validate_patch, apply_patch
- Interfaces: IsolatedPatchWorktree@1
- Conflict policy: Acquire WorktreeLifecycleStore/LeaseCoordinator ownership before `git worktree add`; use explicit validated paths and argv; reject control/runtime/state paths, symlink escapes, submodules, binary patches, path traversal, undeclared files, and stale fences. Never mutate the user's checkout.
- Preconditions: Scheduling lease/fence contracts exist and the target is a valid Git repository.
- Effects: Creates a disposable attempt worktree, materializes ContextPack artifacts, validates bounded unified diffs and declared file scope, applies with Git, records exact pre/post trees, and performs recoverable fenced cleanup.
- Acceptance: Malformed/out-of-scope patches cause no target/root mutation; allowed text patches apply deterministically; concurrent/stale owners cannot publish or clean a live peer worktree; interrupted prepare/apply/cleanup states recover safely.

## SCH-011 Implement the complete 14-step harness loop

- Status: todo
- Completion: auto
- Priority: P0
- Track: harness-loop
- Depends on: SCH-002, SCH-003, SCH-005, SCH-006, SCH-007, SCH-008, SCH-009, SCH-010
- Goal id: SCH-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py, ipfs_accelerate_py/agent_supervisor/semantic_state/__init__.py, tests/agent_supervisor/semantic_state/test_harness.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_harness.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/harness-loop
- Parallel lane: sch-harness
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Provider role: codex-implement
- Context budget tokens: 40000
- LLM context budget bytes: 327680
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 5 through 15
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py, ipfs_accelerate_py/agent_supervisor/semantic_state/__init__.py, tests/agent_supervisor/semantic_state/test_harness.py
- Predicted symbols: SemanticCompressionHarness, HarnessPolicy, HarnessRequest, HarnessResult, run_semantic_patch_loop
- Interfaces: SemanticCompressionHarness@1
- Conflict policy: Orchestrate existing ports only. Do not add an agent framework/server/UI, auto-rewrite dependents, bypass an obligation, scan the portfolio, run providers before admission, or move a root before all acceptance receipts are fresh.
- Preconditions: All scanner, durability, scheduling, context, routing, verification, receipt, and worktree ports pass their focused tests.
- Effects: Executes worktree creation, ContextPack materialization, model invocation, patch/scope validation, application, rescan/delta/invalidation, static checks, selected tests, provers, optional oracle, receipts, and accepted expected-old root CAS in the documented 14-step order.
- Acceptance: Rejection/unavailability/cancellation leaves the current root unchanged; acceptance requires a real production provider when a model is needed and fresh passing required receipts; actual changed symbols and obligations are returned; exact replay is idempotent; root conflict is reported rather than overwritten.

## SCH-012 Implement incremental session, watch, restart, and replay

- Status: todo
- Completion: auto
- Priority: P0
- Track: session
- Depends on: SCH-011
- Goal id: SCH-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/session.py, tests/agent_supervisor/semantic_state/test_session.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_session.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/session
- Parallel lane: sch-session
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 36000
- LLM context budget bytes: 294912
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 12, 13, and 15
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/session.py, tests/agent_supervisor/semantic_state/test_session.py
- Predicted symbols: SemanticStateSession, SessionPolicy, SessionStatus, watch_session, replay_session
- Interfaces: SemanticStateSession@1
- Conflict policy: Watch notifications only schedule a canonical scan. Reuse runtime.event_log cursors and existing fences; do not make events/mtime/queue rows authoritative, start background work on import, or allow a stale callback to commit.
- Preconditions: The end-to-end harness and durable event/root ports exist.
- Effects: Coalesces concurrent notifications by snapshot CID, serializes accepted attempts by repository/fence, checkpoints bounded event cursors, and reconciles nonterminal work plus WAL/root state on restart.
- Acceptance: Concurrent watchers do not duplicate equal work or overwrite roots; restart neither loses an accepted transition nor publishes an unverified one; corrupt/truncated events recover or fail closed; explicit shutdown cancels and joins owned work.

## SCH-013 Add semantic-state CLI and console entrypoint

- Status: todo
- Completion: auto
- Priority: P0
- Track: cli
- Depends on: SCH-011, SCH-012
- Goal id: SCH-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py, tests/agent_supervisor/semantic_state/test_cli.py, pyproject.toml, setup.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_cli.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/cli
- Parallel lane: sch-cli
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 36000
- LLM context budget bytes: 294912
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md section 16
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/cli.py, tests/agent_supervisor/semantic_state/test_cli.py, pyproject.toml, setup.py
- Predicted symbols: build_parser, main
- Interfaces: SemanticStateCLI@1
- Conflict policy: Add one dedicated `semantic-state` entrypoint mirrored in pyproject/setup. Do not expand legacy monolithic CLI, expose a network/MCP service, auto-install dependencies, or hide typed unavailable/production-simulation errors behind exit zero.
- Preconditions: Harness and restartable session APIs are stable.
- Effects: Implements scan, watch, status, graph, explain-symbol, explain-impact, invalidate, select-tests, pack-context, verify, apply-patch, compare-full-suite, and benchmark capabilities with deterministic JSON.
- Acceptance: All commands have bounded help/errors and stable exit codes; production apply-patch cannot simulate; local commands need no IPFS daemon; cold `--help` and imports cause no install/network/environment/process/database mutation.

## SCH-014 Create the controlled Python fixture repository

- Status: todo
- Completion: auto
- Priority: P0
- Track: fixtures
- Depends on: SCH-000
- Goal id: SCH-G040
- Outputs: tests/fixtures/semantic_state_harness/controlled_repo, tests/agent_supervisor/semantic_state/test_fixture_repository.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_fixture_repository.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/fixtures
- Parallel lane: sch-fixtures
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 34000
- LLM context budget bytes: 278528
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 2, 10, and 17
- Predicted files: tests/fixtures/semantic_state_harness/controlled_repo, tests/agent_supervisor/semantic_state/test_fixture_repository.py
- Predicted symbols: controlled_repository, mutation_case, fixture_oracle
- Interfaces: ControlledSemanticRepository@1
- Conflict policy: Keep the fixture small, self-contained, Python 3.12/pytest-only, deterministic, and free of network/native execution. Native/dynamic behavior is represented syntactically and not imported during scans. Do not depend on the user's portfolio checkout.
- Preconditions: Exact datasets semantic-index fixture/API contract is sealed.
- Effects: Supplies base/mutated Git trees for local body, signature, cross-module, dataclass/schema, exception, fixture/config, dynamic import, monkey patch, opaque native, formatting, delete/rename, generated file, stale receipt, failed CAS, interruption, concurrent watcher, and out-of-scope patch cases.
- Acceptance: Full fixture suite is fast and deterministic; every mutation has expected changed symbols, invalidation/test/proof oracle, and confidence/raw-source behavior; unrelated formatting and changes remain bounded; no fixture scan imports or executes fixture code.

## SCH-015 Prove the end-to-end acceptance matrix

- Status: todo
- Completion: auto
- Priority: P0
- Track: acceptance
- Depends on: SCH-011, SCH-012, SCH-014
- Goal id: SCH-G040
- Outputs: tests/agent_supervisor/semantic_state/test_acceptance.py, tests/agent_supervisor/semantic_state/test_concurrency_and_recovery.py, tests/agent_supervisor/semantic_state/test_production_gates.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_acceptance.py tests/agent_supervisor/semantic_state/test_concurrency_and_recovery.py tests/agent_supervisor/semantic_state/test_production_gates.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/acceptance
- Parallel lane: sch-acceptance
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Provider role: codex-implement
- Context budget tokens: 40000
- LLM context budget bytes: 327680
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 9 through 17
- Predicted files: tests/agent_supervisor/semantic_state/test_acceptance.py, tests/agent_supervisor/semantic_state/test_concurrency_and_recovery.py, tests/agent_supervisor/semantic_state/test_production_gates.py
- Predicted symbols: test_controlled_invalidation_matrix, test_selected_suite_recall, test_root_cas_and_recovery, test_production_rejects_simulation
- Interfaces: SemanticStateAcceptance@1
- Conflict policy: Test real local behavior and injected typed unavailable paths, not mock success claims. Do not weaken selection/oracle/freshness/scope/CAS gates or omit failing/escalated cases.
- Preconditions: Complete harness/session and controlled fixture exist.
- Effects: Runs all required mutation cases, compares selected/full suites, injects interrupted writes/concurrent writers/watchers, forges stale/corrupt receipts, and submits a simulated/out-of-scope provider patch.
- Acceptance: Unrelated changes do not invalidate the repository; known dependents are invalidated; opaque source is retrieved; stale receipts never verify; controlled selection recall is 100 percent; full fallback works; roots are deterministic; recovery is safe; CAS has one winner; simulation never reports production verification.

## SCH-016 Create the exactly-40-task benchmark corpus

- Status: todo
- Completion: auto
- Priority: P0
- Track: benchmark-corpus
- Depends on: SCH-014
- Goal id: SCH-G050
- Outputs: benchmarks/semantic_state/tasks, benchmarks/semantic_state/corpus.json, tests/agent_supervisor/semantic_state/test_benchmark_corpus.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_benchmark_corpus.py
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/benchmark-corpus
- Parallel lane: sch-benchmark-corpus
- Resource class: cpu-small
- Implementation timeout seconds: 5400
- Provider role: codex-implement
- Context budget tokens: 34000
- LLM context budget bytes: 278528
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md section 18
- Predicted files: benchmarks/semantic_state/tasks, benchmarks/semantic_state/corpus.json, tests/agent_supervisor/semantic_state/test_benchmark_corpus.py
- Predicted symbols: BenchmarkTask, BenchmarkCorpus
- Interfaces: SemanticStateBenchmarkCorpus@1
- Conflict policy: Exactly 40 checked-in deterministic tasks: 10 small bug fixes and 6 each test repair, API adapter, schema migration, multi-file/refactor, and rejection/escalation. Do not omit hard/failed tasks or required raw source to bias metrics.
- Preconditions: Controlled fixture mutation format and oracle are stable.
- Effects: Defines task objective/target/base mutation/oracle/risk/expected route and reproducible baseline-retrieval policy for every representative task.
- Acceptance: Corpus has exactly 40 unique stable task IDs and the required category counts; includes multi-file and frontier/human cases; every task is runnable offline against a pinned fixture tree and declares no expected outcome derived from benchmark implementation output.

## SCH-017 Implement benchmark runner and publish measured results

- Status: todo
- Completion: auto
- Priority: P0
- Track: benchmark
- Depends on: SCH-006, SCH-007, SCH-011, SCH-016
- Goal id: SCH-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/semantic_state/benchmark.py, benchmarks/semantic_state/run_benchmark.py, docs/benchmarks/semantic_compression_harness_results.json, docs/benchmarks/semantic_compression_harness_results.md, tests/agent_supervisor/semantic_state/test_benchmark.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state/test_benchmark.py && python benchmarks/semantic_state/run_benchmark.py --check
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/benchmark
- Parallel lane: sch-benchmark
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Provider role: codex-implement
- Context budget tokens: 40000
- LLM context budget bytes: 327680
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 11 and 18
- Predicted files: ipfs_accelerate_py/agent_supervisor/semantic_state/benchmark.py, benchmarks/semantic_state/run_benchmark.py, docs/benchmarks/semantic_compression_harness_results.json, docs/benchmarks/semantic_compression_harness_results.md, tests/agent_supervisor/semantic_state/test_benchmark.py
- Predicted symbols: BenchmarkRunner, BenchmarkResult, BenchmarkSummary, run_benchmark, compare_context_modes
- Interfaces: SemanticStateBenchmark@1
- Conflict policy: Use the same declared tokenizer/estimator and coverage assurance for raw and semantic modes. Count all 40 tasks including rejection/escalation; never manufacture model success, drop opaque code, or report unavailable/simulation as accepted.
- Preconditions: Context packer, routing, complete harness, and corpus are stable.
- Effects: Captures baseline/pack tokens, excluded raw code, capsules, invalidation cone, selected/full tests, proofs, route, outcome, timing, receipt freshness, precision/recall, fallback, and category/overall reductions into reproducible JSON/Markdown.
- Acceptance: `--check` recomputes identical deterministic fields; overall median context reduction is at least 30 percent without coverage omissions; stale/simulated admissions and controlled false negatives are zero; results report task-type reductions, precision, recall, failures, and uncertainty rather than hiding them.

## SCH-018 Complete documentation, import safety, and provider regressions

- Status: todo
- Completion: auto
- Priority: P0
- Track: release
- Depends on: SCH-013, SCH-015, SCH-017
- Goal id: SCH-G050
- Outputs: docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md, tests/agent_supervisor/semantic_state/test_import_safety.py, tests/agent_supervisor/semantic_state/test_provider_regressions.py
- Validation: python -m pytest -q tests/agent_supervisor/semantic_state && python benchmarks/semantic_state/run_benchmark.py --check
- Board namespace: semantic-compression-harness-v1
- Bundle: sch/release
- Parallel lane: sch-release
- Resource class: cpu-medium
- Implementation timeout seconds: 7200
- Provider role: codex-implement
- Context budget tokens: 40000
- LLM context budget bytes: 327680
- Plan context: docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md sections 16 through 20
- Predicted files: docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md, tests/agent_supervisor/semantic_state/test_import_safety.py, tests/agent_supervisor/semantic_state/test_provider_regressions.py
- Predicted symbols: release_report, import_safety_probe, provider_production_gate_regression
- Interfaces: SemanticStateHarnessRelease@1
- Conflict policy: Document code and measured receipts exactly. Do not claim complete Python analysis, universal verification, ZK support, provider availability, or production readiness; do not auto-install on import or weaken a failing gate to close the board.
- Preconditions: CLI, acceptance matrix, and benchmark results are committed and reproducible.
- Effects: Documents architecture/modules/commands/examples/tests/benchmark/token reductions/precision-recall/limitations/bottlenecks and exact pre-ZK/production work; tests imports with installer/network/process/environment mutation disabled and real/absent/simulated provider dispositions.
- Acceptance: Focused and relevant existing supervisor tests pass; ordinary imports are side-effect free; production unavailable/simulation is nonzero and never verified; final report is traceable to exact commits and receipts and contains every requested completion-report section.
