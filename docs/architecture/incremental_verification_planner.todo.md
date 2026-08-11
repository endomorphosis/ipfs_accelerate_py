# Incremental Verification Planner Taskboard (IVP)

Consumed by `ipfs_accelerate_py.agent_supervisor` with task prefix `IVP-`.
Companion plan: `INCREMENTAL_VERIFICATION_PLANNER_PLAN.md`. Companion goals:
`incremental_verification_planner.objectives.md`.

Normative rule: a cache hit, provider claim, timeout, unavailable tool,
simulated result, signature, CID, or historical pass never manufactures current
verification authority. Unknown or uncertain selection broadens execution.

## Advisory workstreams

These labels describe ownership only. Runtime dispatch uses the three sealed
SHA-256 shards in the scheduler config; dependency readiness and exact file/
resource claims are authoritative. `Allow concurrent with` is intentionally
empty for every task because it is not a configured-board dispatch input.

| Workstream | Tasks |
| --- | --- |
| contracts | IVP-000, IVP-001 |
| boundaries | IVP-002, IVP-003, IVP-004 |
| adapters | IVP-005, IVP-006, IVP-007 |
| cache | IVP-008 |
| selection | IVP-009, IVP-010 |
| diagnostics | IVP-011, IVP-013 |
| routing | IVP-012 |
| orchestration | IVP-014 |
| evaluation | IVP-015, IVP-016, IVP-017 |
| release | IVP-018, IVP-019 |

## IVP-000 Seal the incremental-verification planning controls

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: planning
- Depends on:
- Goal id: IVP-G000
- Outputs: .gitignore, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_PLAN.md, docs/architecture/incremental_verification_planner.objectives.md, docs/architecture/incremental_verification_planner.todo.md, config/agent_supervisor_incremental_verification_planner_scheduler.json, scripts/validate_incremental_verification_planner_board.py, scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py
- Validation: python3 scripts/validate_incremental_verification_planner_board.py --check-all
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/controls
- Parallel lane: contracts
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 0
- Implementation timeout seconds: 300
- Predicted files: .gitignore, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_PLAN.md, docs/architecture/incremental_verification_planner.objectives.md, docs/architecture/incremental_verification_planner.todo.md, config/agent_supervisor_incremental_verification_planner_scheduler.json, scripts/validate_incremental_verification_planner_board.py, scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py
- Interfaces: incremental_verification_planner_scheduler, goal_heap, markdown_taskboard
- Allow concurrent with:
- Conflict policy: Planning/control files are operator-owned and protected from implementation workers.
- Preconditions: Clean isolated branch contains the seven tracked control files and passes fail-closed preflight.
- Effects: Authorizes only the fixed IVP task DAG after explicit launch with --implement.
- Evidence subset: ivp/planning-controls@1
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Goal/subgoal lineage, task dependencies, disjoint predicted files, validation, safety invariants, benchmarks, final report, provider separation, and monitoring/recovery policy are explicit and the board validator returns valid true.
- Embedding query: incremental verification planner sealed supervisor goal subgoal task board controls

## IVP-001 Implement canonical verification contracts and exact identities

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: contracts
- Depends on: IVP-000
- Goal id: IVP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, ipfs_accelerate_py/agent_supervisor/verification/__init__.py, test/api/test_agent_supervisor_verification_contracts.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_contracts.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/contracts
- Parallel lane: contracts
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, ipfs_accelerate_py/agent_supervisor/verification/__init__.py, test/api/test_agent_supervisor_verification_contracts.py
- Interfaces: TerminalStatus, VerificationReceiptKey, VerificationIdentityCompiler, StaticAnalysisReceipt, TypeCheckReceipt, TestReceipt, ProofReceipt, CounterexampleReceipt, VerificationPlan, VerificationBundle, VerificationSummary, CacheReuseDecision, ModelRouteDecision, VerificationCommitment
- Allow concurrent with:
- Conflict policy: Reuse canonical identity helpers and bridge existing receipt payloads; do not rewrite or weaken existing proof/test receipt schemas.
- Preconditions: Existing formal_verification_contracts and multiformats identity behavior is inventoried.
- Effects: Adds one closed boundary schema family without granting cache/provider authority.
- Evidence subset: ivp/contracts@1, ivp/cache-key-vectors@1
- Symbolic first: true
- LLM context budget bytes: 180000
- Acceptance: All 12 closed statuses round-trip; exact key binds the observed executed patched tree, semantic root, affected symbol versions, effective sandbox environment, dependency lock, selector argv, obligation translation/not-applicable, exact tool name/version, configuration bytes, fixtures, network policy, receipt schema, kind, and adapter; base-tree/root/observed-inventory mismatches reject; any mutation changes the key; wrapper passed/proved projections derive from existing authoritative assurance or current direct execution, never a duplicate status field; timeout/unavailable/simulated are non-accepting; floats, secrets, witnesses, oversized values, wrong schemas, and unchecked caller-supplied identities reject.
- Embedding query: canonical receipt contracts exact verification cache key closed terminal status immutable decision plan summary commitment

## IVP-002 Add the lazy fail-closed ipfs_datasets semantic input adapter

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: datasets-adapter
- Depends on: IVP-001
- Goal id: IVP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/datasets_adapter.py, test/api/test_agent_supervisor_verification_datasets_adapter.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_datasets_adapter.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/boundaries
- Parallel lane: boundaries
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 14000
- Implementation timeout seconds: 4500
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/datasets_adapter.py, test/api/test_agent_supervisor_verification_datasets_adapter.py
- Interfaces: DatasetsVerificationInputAdapter, RepositoryStateView, InvalidationPlanView, ContextPackView, SemanticCapsuleView
- Allow concurrent with:
- Conflict policy: Do not invent or re-export absent canonical datasets classes; accept strict Mapping inputs now and explicitly registered exact leaf-module/type adapters only when upstream classes land; Protocol is typing only, arbitrary to_dict invocation is forbidden, and missing identities or uncovered impact fail closed.
- Preconditions: The current dependency gap and CodeEvidenceCorpusAdapter impact schema are recorded in tests.
- Effects: Normalizes tree/semantic/environment/lock roots, changed symbols/paths, test/proof edges, uncertainty, spans, contracts, tokens, and fixture references without eager optional imports.
- Evidence subset: ivp/datasets-adapter@1
- Symbolic first: true
- LLM context budget bytes: 140000
- Acceptance: Accelerator cold import works without ipfs_datasets; strict canonical mappings normalize deterministically and explicitly registered upstream types can be added without changing authority; exact code_evidence and BoundedToolRunner leaf-symbol probes are supported while a top-level namespace import alone is unavailable; opaque datasets repository_tree_id stays separate from receipt repository_tree_cid; validation IDs require an exact pytest-node mapping or force broader selection; missing/malformed CIDs, unknown schemas, opaque/uncovered/truncated edges, and absent leaf modules produce typed non-authoritative observations; no arbitrary attribute traversal, to_dict execution, network, or install side effect occurs.
- Embedding query: ipfs datasets RepositoryState InvalidationPlan SemanticCapsule ContextPack structural adapter impact edges fail closed

## IVP-003 Implement immutable receipt storage and generation-CAS indexes

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: receipt-store
- Depends on: IVP-001
- Goal id: IVP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/receipt_store.py, test/api/test_agent_supervisor_verification_receipt_store.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_receipt_store.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/boundaries
- Parallel lane: boundaries
- Resource class: io-medium
- Resource stage: implementation
- Estimated tokens: 20000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/receipt_store.py, test/api/test_agent_supervisor_verification_receipt_store.py
- Interfaces: VerificationReceiptStore, HermeticVerificationReceiptStore, IpfsKitVerificationReceiptStore, IndexSnapshot, CompareAndSwapResult
- Allow concurrent with:
- Conflict policy: Adapt public DurableCoordinationStore Mapping-envelope put/get/recover with an explicit storage_dir; it supplies immutable CID artifacts, not head CAS. Use the separate local generation-CAS protocol and only an injected operational async Iroh manifest bridge; do not use mock IPFS fallbacks, home defaults, private SQLite tables, or silently downgrade corruption.
- Preconditions: ipfs_kit import is lazy and its exact supported revision/API is capability-probed.
- Effects: Persists canonical immutable blocks, exact index generations, history/replay artifacts, stale tombstones, and GC metadata with atomic local CAS.
- Evidence subset: ivp/store-protocol@1, ivp/concurrent-store-cas@1
- Symbolic first: true
- LLM context budget bytes: 180000
- Acceptance: Hermetic backend survives reopen, verifies CIDs, detects corruption, replays history, fsyncs atomic updates, and prevents lost updates under concurrent writers; exact DurableCoordinationStore leaf probing, immutable Mapping envelopes, put/get/recover, explicit storage root, and cross-backend CID vectors are tested; top-level namespace-only, absent backend/revision, or unavailable CAS is typed unavailable; CAS conflicts never overwrite a peer; tombstones preserve immutable audit history.
- Embedding query: ipfs kit DurableCoordinationStore immutable block verification index root compare and swap replay corruption tombstone GC

## IVP-004 Implement the admitted explicit-argv verification process runner

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: execution-runtime
- Depends on: IVP-001
- Goal id: IVP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/process_runner.py, test/api/test_agent_supervisor_verification_process_runner.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_process_runner.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/boundaries
- Parallel lane: boundaries
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/process_runner.py, test/api/test_agent_supervisor_verification_process_runner.py
- Interfaces: VerificationProcessRunner, VerificationCommand, VerificationRunResult, VerificationCancellation
- Allow concurrent with:
- Conflict policy: Keep this a private IVP adapter over existing ResourceScheduler, HermeticValidationRuntime, BoundedToolRunner/SubprocessProofProvider lifecycle, cancellation, and artifacts; never add a second generic runner/scheduler or pass shell strings, shell true, ambient secrets, unbounded streams, or install commands.
- Preconditions: Existing formal_verification_provider and bounded tool-runner semantics are compared and process fencing tests are reusable.
- Effects: Acquires a resource lease, launches explicit argv in a deterministic allowlist environment, bounds separate stdout/stderr artifacts, and terminates the entire process tree on cancellation/timeout.
- Evidence subset: ivp/process-runner@1, ivp/process-tree-cancellation@1
- Symbolic first: true
- LLM context budget bytes: 170000
- Acceptance: shell interpolation is impossible; executable/argv/cwd/effective hermetic filesystem and sandbox identity/network policy/timeout are explicit and observed; missing executable or required sandbox is unavailable; timeout is timeout; cancellation is cancelled and fences late publication; stdout and stderr truncate deterministically with artifact digests; child, grandchild, and escaped-session processes are gone after cancellation; no mock hardware/inference or auto-install path exists.
- Embedding query: deterministic subprocess explicit argv resource admission lease cancellation process group bounded stdout stderr timeout unavailable

## IVP-005 Implement the pytest verification adapter

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: check-adapters
- Depends on: IVP-001, IVP-004
- Goal id: IVP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/adapters/pytest_adapter.py, test/api/test_agent_supervisor_pytest_verification_adapter.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_pytest_verification_adapter.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/checks
- Parallel lane: adapters
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 15000
- Implementation timeout seconds: 4500
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/adapters/pytest_adapter.py, test/api/test_agent_supervisor_pytest_verification_adapter.py
- Interfaces: PytestVerificationAdapter
- Allow concurrent with:
- Conflict policy: Use the shared runner and existing test identity contracts; do not invoke pytest through a shell or treat collection/setup/teardown skips and failures as a pass.
- Preconditions: VerificationCommand and TestReceipt contracts are stable.
- Effects: Executes exact selected node IDs or explicit full-suite oracle mode and persists bounded phase/result artifacts.
- Evidence subset: ivp/pytest-adapter@1
- Symbolic first: true
- LLM context budget bytes: 130000
- Acceptance: Explicit python -m pytest argv, selector/config/observed environment/fixture bindings, timeout/cancel/unavailable preservation, setup/call/teardown accounting, and authoritative existing-test-receipt projection are tested; empty collection/usage/malformed output is invalid, required skip/xfail is not_modeled unless predeclared advisory, and unexpected xpass or collection/setup/teardown failure cannot pass; artifact references and reproducible argv list are retained; simulated mode cannot satisfy production.
- Embedding query: pytest selected node full suite adapter test receipt setup call teardown timeout artifact

## IVP-006 Implement the mypy verification adapter

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: check-adapters
- Depends on: IVP-001, IVP-004
- Goal id: IVP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/adapters/mypy_adapter.py, test/api/test_agent_supervisor_mypy_verification_adapter.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_mypy_verification_adapter.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/checks
- Parallel lane: adapters
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 11000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/adapters/mypy_adapter.py, test/api/test_agent_supervisor_mypy_verification_adapter.py
- Interfaces: MypyVerificationAdapter
- Allow concurrent with:
- Conflict policy: Use the repository-declared mypy tool and explicit file/module/config selectors; no dependency installation or success-on-unavailable fallback.
- Preconditions: VerificationCommand and TypeCheckReceipt contracts are stable.
- Effects: Executes a reproducible bounded type-check command and records exact tool/version/config/selector identity.
- Evidence subset: ivp/mypy-adapter@1
- Symbolic first: true
- LLM context budget bytes: 100000
- Acceptance: Explicit argv and observed deterministic environment are captured; pass/fail/timeout/unavailable/cancelled map losslessly while usage/malformed output is invalid; tool executable/version and config mutations change keys; bounded diagnostics are artifacts; missing mypy remains unavailable.
- Embedding query: mypy static type checker adapter explicit selector configuration version receipt unavailable

## IVP-007 Implement Z3 and existing proof-assistant verification adapters

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prover-adapters
- Depends on: IVP-001, IVP-002, IVP-004
- Goal id: IVP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/adapters/prover_adapters.py, test/api/test_agent_supervisor_prover_verification_adapters.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_prover_verification_adapters.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/checks
- Parallel lane: adapters
- Resource class: cpu-proof
- Resource stage: implementation
- Estimated tokens: 18000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/adapters/prover_adapters.py, test/api/test_agent_supervisor_prover_verification_adapters.py
- Interfaces: Z3VerificationAdapter, ExistingProofAssistantAdapter
- Allow concurrent with:
- Conflict policy: Reuse admitted obligation/compiler and operational kernel adapters but route execution through bounded cancellation-safe infrastructure; no new prover, model draft authority, raw legacy subprocess runner, or unavailable-as-pass behavior.
- Preconditions: Capability registry and existing Z3/Lean/Rocq/Isabelle adapters are inventoried; only real executable probes may admit execution.
- Effects: Produces ProofReceipt with exact obligation/tool/environment bindings and lossless proved/disproved/unknown/timeout/unavailable/cancelled outcomes.
- Evidence subset: ivp/prover-adapter@1
- Symbolic first: true
- LLM context budget bytes: 180000
- Acceptance: Z3 sat/unsat/unknown mappings bind the exact normalized/negated obligation, translator version, solver executable/version, and existing assurance verdict; bare solver text cannot prove; absent/timeout Z3 never proves; proof-assistant route selects only an offline bounded registry-admitted kernel probe and otherwise returns unavailable; sorry/admit/unsafe or model-generated drafts cannot prove; a wrapper proved/disproved status derives from existing authoritative proof evidence or current direct execution; cancellation terminates descendants and fences late output.
- Embedding query: Z3 proof obligation Lean Rocq Isabelle existing adapter bounded cancellation unknown unavailable proof receipt

## IVP-008 Implement VerificationReceiptCache admission, lookup, and invalidation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: receipt-cache
- Depends on: IVP-001, IVP-003
- Goal id: IVP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py, test/api/test_agent_supervisor_verification_receipt_cache.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_receipt_cache.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/cache
- Parallel lane: cache
- Resource class: io-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py, test/api/test_agent_supervisor_verification_receipt_cache.py
- Interfaces: VerificationReceiptCache, lookup, admit, mark_stale, replay, gc_metadata
- Allow concurrent with:
- Conflict policy: Re-derive exact key/content/status/authority on every hit; cache/store cannot upgrade evidence or mutate immutable receipts in place.
- Preconditions: Canonical key/receipt contracts and receipt-store CAS protocol pass their focused tests.
- Effects: Adds exact-key index transactions, reuse decisions, stale tombstones, replay/corruption handling, and GC reachability/last-access metadata.
- Evidence subset: ivp/receipt-cache@1, ivp/concurrent-writer@1, ivp/replay-corruption@1
- Symbolic first: true
- LLM context budget bytes: 190000
- Acceptance: Unchanged current same-tree receipt reuses; related tree/symbol/environment/lock/selector/tool-version/config/fixture/network/schema changes reject; an unrelated edit preserves the old immutable receipt under its old key without a scoped-staleness tombstone but rejects it for the new full-tree key; stale, simulated, timeout, unavailable, invalid, malformed, kind-mismatched, key-mismatched, and corrupt candidates cannot satisfy production; two concurrent writers preserve both entries through CAS retry.
- Embedding query: VerificationReceiptCache exact lookup reuse invalidation stale simulated concurrency CAS corruption replay tombstone

## IVP-009 Implement semantic affected-check and test selection

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: selection
- Depends on: IVP-001, IVP-002
- Goal id: IVP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/selection.py, test/api/test_agent_supervisor_verification_selection.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_selection.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/planner
- Parallel lane: selection
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 17000
- Implementation timeout seconds: 4500
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/selection.py, test/api/test_agent_supervisor_verification_selection.py
- Interfaces: AffectedVerificationSelection, select_affected_verification
- Allow concurrent with:
- Conflict policy: Use supplied semantic/test/proof dependency edges and explicit conservative closure; never infer an absent edge as exact or optimize away uncertainty.
- Preconditions: Datasets adapter exposes normalized changed symbols/paths, validation targets, proof dependencies, and frontier classifications.
- Effects: Deterministically selects affected tests/static/type checks/proof obligations and records broader/full fallback reasons and dependency cone.
- Evidence subset: ivp/test-selection@1
- Symbolic first: true
- LLM context budget bytes: 160000
- Acceptance: Direct/transitive symbol/path/test/fixture/config edges select correctly; unrelated edits do not expand exact selections; unknown/dynamic/opaque/uncovered/truncated/conflicting critical edges set broader or full-suite fallback; changed obligation dependencies select proofs; deterministic order and reason chains are stable.
- Embedding query: semantic test dependency edges affected symbols selected tests static checks proof obligations uncertainty full suite fallback

## IVP-010 Implement create_verification_plan and planner policy

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: planning
- Depends on: IVP-002, IVP-008, IVP-009
- Goal id: IVP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/planner.py, test/api/test_agent_supervisor_incremental_verification_planner.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_incremental_verification_planner.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/planner
- Parallel lane: selection
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 22000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/planner.py, test/api/test_agent_supervisor_incremental_verification_planner.py
- Interfaces: IncrementalVerificationPlanner, create_verification_plan
- Allow concurrent with:
- Conflict policy: Join normalized inputs, pure selection, and exact cache decisions; do not execute commands, resolve providers, or silently fill missing identities inside planning.
- Preconditions: Datasets adapter, cache, selector, and shared decision contracts are stable and focused tests pass; model routing consumes this plan and is not a planner dependency.
- Effects: Emits a deterministic VerificationPlan containing reusable/stale decisions, affected checks, proof obligations, fallback/review flags, resource requirements, deadlines, DAG, and acceptance criteria.
- Evidence subset: ivp/verification-plan@1
- Symbolic first: true
- LLM context budget bytes: 200000
- Acceptance: The specified five-argument API works for strict mappings and explicitly registered canonical objects; patch base and RepositoryState/InvalidationPlan/ContextPack roots are cross-checked and receipt keys bind the exact target patched tree; relevant changes select/invalidate, unrelated changes avoid semantic over-selection while rejecting cross-tree admission, and environment/lock/tool observed mismatches invalidate; planning returns stale decisions without mutating tombstones; uncertainty broadens; unbound effective sandbox, policy conflict, or declared-scope crossing requires review; resource and per-step/global timeout bounds are positive and capped; acceptance requires every current production-admissible required success and no pending mandatory fallback.
- Embedding query: create_verification_plan RepositoryState InvalidationPlan ContextPack patch delta cache reuse resources timeout acceptance

## IVP-011 Implement deterministic counterexample minimization

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: diagnostics
- Depends on: IVP-005
- Goal id: IVP-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/counterexamples.py, test/api/test_agent_supervisor_verification_counterexamples.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_counterexamples.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/diagnostics
- Parallel lane: diagnostics
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 15000
- Implementation timeout seconds: 4500
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/counterexamples.py, test/api/test_agent_supervisor_verification_counterexamples.py
- Interfaces: CounterexampleMinimizer, minimize_counterexample
- Allow concurrent with:
- Conflict policy: Minimize already-bounded artifacts deterministically and rerun every candidate under a separate bounded lease; no model call and no full-log embedding unless a typed minimization failure requires a bounded reference.
- Preconditions: CounterexampleReceipt and process artifact contracts are stable.
- Effects: Slices tracebacks/assertions, prunes by relevant semantic cone, retains failing selector/input/expected/observed/spans/environment/argv/artifact references, and records minimization quality.
- Evidence subset: ivp/counterexample@1
- Symbolic first: true
- LLM context budget bytes: 130000
- Acceptance: A failing selected pytest case produces a compact CounterexampleReceipt only after the candidate argv is rerun under a separate bounded lease and preserves the same failure identity; irrelevant frames/log lines/inputs are removed; sensitive or inapplicable input/expected/observed values use typed redacted/unavailable fields; argv is a list; environment and source spans bind; output is size bounded; minimization failure is explicit and references bounded artifacts instead of embedding whole logs.
- Embedding query: counterexample minimization pytest traceback assertion input expected observed source span reproduction argv artifact

## IVP-012 Implement provider-neutral next-repair model routing

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: model-routing
- Depends on: IVP-001
- Goal id: IVP-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/test_agent_supervisor_verification_model_route.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_model_route.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/routing
- Parallel lane: routing
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 13000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/model_route.py, test/api/test_agent_supervisor_verification_model_route.py
- Interfaces: ModelRoutePlanner, choose_model_route
- Allow concurrent with:
- Conflict policy: Select capability class only; do not import/invoke llm_router, choose a provider/vendor/model ID, or let availability lower required safety.
- Preconditions: ModelRouteDecision and input policy contracts are stable.
- Effects: Applies deterministic precedence over context estimate, analysis kind, opaque dependencies, risk, cone size, unresolved obligations, failure type, prior attempts, counterexample quality, exact contracts, fallback state, reproducibility, and scope.
- Evidence subset: ivp/model-route@1
- Symbolic first: true
- LLM context budget bytes: 110000
- Acceptance: Mechanical exact formatting/import/codemod/rename selects deterministic; bounded localized exact work with a good counterexample selects small; several-file synthesis without opaque critical edges selects medium; ambiguous/broad/opaque/conflicting/overflow or failed-smaller work selects frontier; available_models is a provider-neutral capability/tier/context/locality/availability inventory and never vendor preference; an unavailable required tier cannot downgrade and a pending mandatory full suite returns human review with verification-incomplete reason; unresolved authority, unmodeled high risk, scope crossing, proof/test conflict, unsafe context, or non-reproducible environment selects human review before any model route; output contains no provider identity.
- Embedding query: choose_model_route deterministic small local medium frontier human review risk opaque context prior attempts

## IVP-013 Implement bundles, compact summaries, and verification commitments

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: diagnostics
- Depends on: IVP-001
- Goal id: IVP-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/bundle.py, test/api/test_agent_supervisor_verification_bundle.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_bundle.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/diagnostics
- Parallel lane: diagnostics
- Resource class: cpu-small
- Resource stage: implementation
- Estimated tokens: 16000
- Implementation timeout seconds: 4500
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/bundle.py, test/api/test_agent_supervisor_verification_bundle.py
- Interfaces: build_verification_bundle, build_verification_summary, build_verification_commitment
- Allow concurrent with:
- Conflict policy: Aggregate exact current leaves without upgrading them; use canonical domain-separated hashing and keep ZK/signature/structural validation non-claims explicit.
- Preconditions: Canonical receipt, bundle, summary, and commitment contracts are stable.
- Effects: Builds a bounded ContextPack-ready summary and a deterministic admitted-receipt Merkle commitment with unresolved count and aggregate terminal status.
- Evidence subset: ivp/verification-summary@1, ivp/verification-commitment@1
- Symbolic first: true
- LLM context budget bytes: 140000
- Acceptance: Bundle distinguishes required/admitted/reused/executed/unresolved receipts and rejects mixed tree/environment leaves; summary includes cone, checks, failures, compact counterexamples, fallback, timing/savings, and route within bounds; commitment binds schema/hash/codec, explicit leaf/node/empty domains, odd-node promotion, Merkle root/public statement/tree/environment/required-check-set/unresolved count/fail-closed aggregate lattice; input permutation is invariant after canonical sorting while any required membership/content change changes commitment; tests and docs state commitment is not a ZK proof, signatures require trusted issuers, and structural validation is not cryptographic validation.
- Embedding query: VerificationBundle VerificationSummary ContextPack compact Merkle receipt commitment aggregate status ZK non claim

## IVP-014 Execute verification plans and recompute acceptance

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: orchestration
- Depends on: IVP-005, IVP-006, IVP-007, IVP-008, IVP-010, IVP-011, IVP-012, IVP-013
- Goal id: IVP-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/executor.py, test/api/test_agent_supervisor_verification_executor.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_executor.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/execution
- Parallel lane: orchestration
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 24000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/executor.py, test/api/test_agent_supervisor_verification_executor.py
- Interfaces: VerificationExecutor, execute_verification_plan
- Allow concurrent with:
- Conflict policy: Execute only the sealed plan DAG under existing resource leases and adapters; do not widen selectors/network/deadlines, install tools, or accept provider text/cache presence.
- Preconditions: Planner, cache, all adapters, counterexamples, and bundle functions pass focused tests.
- Effects: Revalidates observed target-tree/sandbox/tool identities, reuses exact admitted hits, executes required misses, publishes stale tombstones only through a post-plan CAS transaction, persists bounded artifacts/receipts, minimizes failures, bundles results, and recomputes acceptance after cancellation/freshness checks.
- Evidence subset: ivp/execution-bundle@1
- Symbolic first: true
- LLM context budget bytes: 210000
- Acceptance: Dependencies and bounded parallelism hold; pre/post observed identities must match the sealed plan; resource rejection is typed; cancellation/timeout terminates process trees including grandchildren/escaped sessions and fences late receipts; unavailable tools remain unavailable; selected failures carry compact counterexamples; production acceptance is true only when every required current leaf passes/proves and no required obligation, mandatory fallback, or human review remains; explicitly advisory obligations may remain unresolved but can never be upgraded; a compact summary, provider-neutral route, and commitment are emitted.
- Embedding query: verification executor plan DAG cache reuse adapters resource lease cancellation counterexample bundle acceptance

## IVP-015 Add selected-versus-full-suite semantic fixture evaluation

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: evaluation
- Depends on: IVP-009, IVP-014
- Goal id: IVP-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/evaluation.py, test/fixtures/incremental_verification, test/api/test_agent_supervisor_verification_selection_evaluation.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_selection_evaluation.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/evaluation
- Parallel lane: evaluation
- Resource class: cpu-large
- Resource stage: evaluation
- Estimated tokens: 20000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/evaluation.py, test/fixtures/incremental_verification, test/api/test_agent_supervisor_verification_selection_evaluation.py
- Interfaces: TestSelectionEvaluation, compare_selected_with_full_suite
- Allow concurrent with:
- Conflict policy: Run real controlled pytest repositories through the shared adapter; do not label an absent SemanticCapsule fixture module as measured or suppress a full-suite discrepancy.
- Preconditions: Selection and executor are integrated; CodeEvidenceCorpusAdapter tiny fixture compatibility is available or typed absent.
- Effects: Executes selected and full suites in fresh identity-equal fixture snapshots, compares reviewed ground-truth affected sets and outcomes, and persists false-positive/false-negative plus flaky/order/inconclusive evidence for direct/transitive/unrelated/fixture/config/environment/lock/opaque changes.
- Evidence subset: ivp/test-selection-evaluation@1
- Symbolic first: true
- LLM context budget bytes: 180000
- Acceptance: Selected/full runs use fresh identical tree/environment/lock/fixture/policy snapshots; fixture ground-truth affected tests define false positives/negatives and full-suite failure comparison is a separate oracle; passing selected tests are not automatically false positives; flaky/order/outcome discrepancies are inconclusive; a full-suite timeout/unavailable or absent canonical semantic-capsule corpus is not_measured, never zero; an uncertain/uncovered selector or missing validation-ID-to-node-ID mapping requires broader/full suite before acceptance; equivalent controlled fixtures are labeled separately; evidence binds corpus/evaluated count/repository/policy/environment/selector identities and measured timing metadata without asserting target success.
- Embedding query: selected tests full suite oracle semantic fixture false negative false positive broader suite evaluation evidence

## IVP-016 Prove the required adversarial conformance matrix

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: conformance
- Depends on: IVP-012, IVP-013, IVP-014, IVP-015
- Goal id: IVP-G090
- Outputs: test/api/test_agent_supervisor_incremental_verification_conformance.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_incremental_verification_conformance.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/evaluation
- Parallel lane: evaluation
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 20000
- Implementation timeout seconds: 5400
- Predicted files: test/api/test_agent_supervisor_incremental_verification_conformance.py, ipfs_accelerate_py/agent_supervisor/verification/contracts.py, ipfs_accelerate_py/agent_supervisor/verification/datasets_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/receipt_store.py, ipfs_accelerate_py/agent_supervisor/verification/process_runner.py, ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py, ipfs_accelerate_py/agent_supervisor/verification/selection.py, ipfs_accelerate_py/agent_supervisor/verification/planner.py, ipfs_accelerate_py/agent_supervisor/verification/counterexamples.py, ipfs_accelerate_py/agent_supervisor/verification/model_route.py, ipfs_accelerate_py/agent_supervisor/verification/bundle.py, ipfs_accelerate_py/agent_supervisor/verification/executor.py, ipfs_accelerate_py/agent_supervisor/verification/evaluation.py, ipfs_accelerate_py/agent_supervisor/verification/adapters/pytest_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/adapters/mypy_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/adapters/prover_adapters.py
- Interfaces: required-conformance-matrix
- Allow concurrent with:
- Conflict policy: This sequential hardening task may make only scoped corrections required by the 18 normative tests; no feature expansion or weakened assertion is allowed.
- Preconditions: Integrated executor and selected/full fixture evidence pass their focused tests.
- Effects: Adds mutation, authority, cancellation, concurrency, routing, and commitment regression tests and fixes only demonstrated contract gaps.
- Evidence subset: ivp/conformance-matrix@1
- Symbolic first: true
- LLM context budget bytes: 230000
- Acceptance: Tests prove unchanged same-tree reuse, relevant invalidation, unrelated old-key historical preservation plus new-tree rejection, environment invalidation, lock invalidation, tool-version invalidation, stale rejection, simulated production rejection, timeout preservation, unavailable prover preservation, rerun-validated minimized selected-test counterexample, uncertain selection broadening, concurrent writer safety, grandchild/escaped-session cancellation, small localized route, frontier broad/opaque route, human review for unresolved high risk or unavailable required tier, and commitment membership/content change plus input-permutation invariance; content corruption, kind mismatch, proof/test conflict, and late cancelled success also fail closed; controlled measured fixtures have zero false negatives while missing canonical fixtures remain not_measured.
- Embedding query: adversarial verification conformance stale simulated timeout unavailable counterexample CAS cancellation model route commitment

## IVP-017 Implement and run the incremental-verification benchmark

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: benchmark
- Depends on: IVP-015
- Goal id: IVP-G090
- Outputs: benchmarks/agent_supervisor/incremental_verification.py, test/benchmarks/test_incremental_verification_planner_benchmark.py, artifacts/agent_supervisor/incremental_verification/benchmark.json
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 benchmarks/agent_supervisor/incremental_verification.py --output artifacts/agent_supervisor/incremental_verification/benchmark.json; PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/benchmarks/test_incremental_verification_planner_benchmark.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/evaluation
- Parallel lane: evaluation
- Resource class: cpu-large
- Resource stage: benchmark
- Estimated tokens: 18000
- Implementation timeout seconds: 5400
- Predicted files: benchmarks/agent_supervisor/incremental_verification.py, test/benchmarks/test_incremental_verification_planner_benchmark.py, artifacts/agent_supervisor/incremental_verification/benchmark.json
- Interfaces: run_incremental_verification_benchmark
- Allow concurrent with:
- Conflict policy: Benchmark cannot modify correctness policy, hide unavailable fixtures/provers, count skipped work as a pass, or report unobserved metrics.
- Preconditions: Selected/full evaluation is implemented; the hard conformance gate may be green or red because benchmark/report evidence must still land honestly.
- Effects: Produces a schema/order-deterministic current-tree artifact with measured samples/tolerances for timing, paired cold/hot saved time or an estimated label, tokenizer/estimator-bound token savings, cache hits, selected/full tests, selection errors, checks/proofs, routes/escalation, and counterexample context.
- Evidence subset: ivp/benchmark@1
- Symbolic first: true
- LLM context budget bytes: 170000
- Acceptance: A freshly generated artifact binds current tree, corpus CID/evaluated count, policy, effective environment, commands, measurement schema, and status; it reports cache hit rate, tests selected/full, ground-truth false negatives/positives, outcome discrepancies, static/proof execution, wall samples, paired/estimated reused time, route, frontier escalation, counterexample context, and estimator-bound token savings; zero stale/simulated acceptance is hard while all target misses are recorded rather than blocking artifact creation; deterministic commitments and old-key historical preservation hold; incompatible cross-tree unaffected reuse is explicitly unmet; small route appears in at least one and 20% of measured localized fixtures or the target is red; missing canonical fixtures or real provers are typed unavailable/not_measured.
- Embedding query: incremental verification benchmark cache hit selected full false negative wall time route frontier tokens saved honest unavailable

## IVP-018 Document operations, schemas, evidence, and limitations

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: documentation
- Depends on: IVP-017
- Goal id: IVP-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/README.md, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md, test/api/test_agent_supervisor_incremental_verification_report.py
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_incremental_verification_report.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/release
- Parallel lane: release
- Resource class: cpu-small
- Resource stage: documentation
- Estimated tokens: 12000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/README.md, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md, test/api/test_agent_supervisor_incremental_verification_report.py
- Interfaces: module documentation, final report draft
- Allow concurrent with:
- Conflict policy: Describe only landed behavior and current benchmark evidence; no future ZK/provider/platform claims.
- Preconditions: A current benchmark artifact exists; conformance evidence may be green or red so the honest report path remains runnable.
- Effects: Documents APIs, trust/status semantics, adapters, key/invalidation, store/replay, selection/fallback, routing, commitment non-claims, operations, evidence, and limitations.
- Evidence subset: ivp/documentation@1, ivp/release-report@1
- Symbolic first: true
- LLM context budget bytes: 110000
- Acceptance: A report validator binds the report and benchmark to the current tree, corpus, policy, effective environment, command identities, and measurement status and rejects stale/missing sections; report includes modules changed, adapters, schemas, exact key, invalidation, selected/full/proof results, cache hits, route distribution, counterexample examples, commitment format, limitations, every unmet target including incompatible cross-tree reuse, and exact future ZK step: freeze leaf codec/trust policy and cross-implementation Merkle vectors before an external membership/aggregation circuit; it explicitly says the commitment is not ZK, signatures need trusted issuers, and structural validation is not cryptographic validation.
- Embedding query: verification planner documentation report schemas key invalidation tests proofs hit rate routes counterexample commitment limitations ZK next step

## IVP-019 Freeze public exports and run the terminal release fan-in

- Status: todo
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release
- Depends on: IVP-016, IVP-017, IVP-018
- Goal id: IVP-G100
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/__init__.py, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md
- Validation: PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_verification_contracts.py test/api/test_agent_supervisor_verification_datasets_adapter.py test/api/test_agent_supervisor_verification_receipt_store.py test/api/test_agent_supervisor_verification_process_runner.py test/api/test_agent_supervisor_pytest_verification_adapter.py test/api/test_agent_supervisor_mypy_verification_adapter.py test/api/test_agent_supervisor_prover_verification_adapters.py test/api/test_agent_supervisor_verification_receipt_cache.py test/api/test_agent_supervisor_verification_selection.py test/api/test_agent_supervisor_incremental_verification_planner.py test/api/test_agent_supervisor_verification_counterexamples.py test/api/test_agent_supervisor_verification_model_route.py test/api/test_agent_supervisor_verification_bundle.py test/api/test_agent_supervisor_verification_executor.py test/api/test_agent_supervisor_verification_selection_evaluation.py test/api/test_agent_supervisor_incremental_verification_conformance.py test/api/test_agent_supervisor_incremental_verification_report.py test/benchmarks/test_incremental_verification_planner_benchmark.py; PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_test_execution_identity.py test/api/test_proof_reuse_invalidation_mutations.py test/api/test_proof_reuse_security_concurrency.py test/api/test_agent_supervisor_formal_verification_cache.py test/api/test_agent_supervisor_validation_scheduler.py test/api/test_agent_supervisor_process_tree_fencing.py test/api/test_agent_supervisor_resource_scheduler.py; PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m ruff check ipfs_accelerate_py/agent_supervisor/verification benchmarks/agent_supervisor/incremental_verification.py test/benchmarks/test_incremental_verification_planner_benchmark.py test/api/test_agent_supervisor_verification_*.py test/api/test_agent_supervisor_incremental_verification_*.py
- Board namespace: incremental-verification-planner-v1
- Bundle: agent-supervisor/incremental-verification/release
- Parallel lane: release
- Resource class: cpu-large
- Resource stage: release
- Estimated tokens: 18000
- Implementation timeout seconds: 5400
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/__init__.py, docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md, ipfs_accelerate_py/agent_supervisor/verification/contracts.py, ipfs_accelerate_py/agent_supervisor/verification/datasets_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/receipt_store.py, ipfs_accelerate_py/agent_supervisor/verification/process_runner.py, ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py, ipfs_accelerate_py/agent_supervisor/verification/selection.py, ipfs_accelerate_py/agent_supervisor/verification/planner.py, ipfs_accelerate_py/agent_supervisor/verification/counterexamples.py, ipfs_accelerate_py/agent_supervisor/verification/model_route.py, ipfs_accelerate_py/agent_supervisor/verification/bundle.py, ipfs_accelerate_py/agent_supervisor/verification/executor.py, ipfs_accelerate_py/agent_supervisor/verification/evaluation.py, ipfs_accelerate_py/agent_supervisor/verification/adapters/pytest_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/adapters/mypy_adapter.py, ipfs_accelerate_py/agent_supervisor/verification/adapters/prover_adapters.py
- Interfaces: create_verification_plan, choose_model_route, build_verification_commitment, VerificationReceiptCache, IncrementalVerificationPlanner, ModelRoutePlanner
- Allow concurrent with:
- Conflict policy: Terminal sequential fan-in may correct only integrated import/static/regression failures; no scope expansion, status weakening, provider hardcoding, or unsupported completion claim.
- Preconditions: Conformance, benchmark, and report tasks are complete with current-tree evidence.
- Effects: Exports the stable focused public API, runs the complete focused suite plus relevant receipt/cache/process regressions and ruff, updates report with exact final observations, and leaves a clean merge target.
- Evidence subset: ivp/public-api@1, ivp/final-validation@1, ivp/release-report@1
- Symbolic first: true
- LLM context budget bytes: 230000
- Acceptance: Required public names import lazily; focused test matrix and the declared existing identity/proof-reuse/formal-cache/validation/process-tree/resource regressions pass; ruff passes for new package/tests; target branch has no unmanaged changes; final report truthfully records measured results/limitations and no stale, simulated, timeout, unavailable, unknown, not_modeled, invalid, cancelled, or pending full-suite receipt is accepted as production success.
- Embedding query: terminal release fan in public incremental verification APIs full tests ruff report clean branch
