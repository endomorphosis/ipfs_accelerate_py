# SCG-001 Accelerate inventory

**Evidence:** `scg/accelerate-inventory@1`  
**Machine-readable twin:** `docs/architecture/semantic_compression_governor_inventory/accelerate.json`  
**Task:** SCG-001 — Inventory accelerate harness, verification, routing, execution, and benchmarks  
**Inspected tree:** `a0b825d8cfa384c284d0e77fa5341571c40adfa8`  
**Planning authority pin:** `dfd92b554e662d4312411f2e8e63a52368806f2a` (ancestor of inspected tree)  
**IVP public-API freeze:** `8c7800cedc5e1b848367db9952f912428466f8cc`

This is a read-only inventory of public current-tree surfaces under
`ipfs_accelerate_py.agent_supervisor`. It records exact exports, signatures,
schemas, status vocabularies, context/routing rules, tests, benchmark metrics,
failure cases, execution isolation, and resource/provider seams. RED, simulated,
unavailable, and known-failure evidence are recorded honestly. No production
code is changed.

Nested gitlinks observed at inventory time:

| Nested repository | Commit |
| --- | --- |
| `ipfs_datasets_py` | `1330038f626ef92993f03d46f21e1a57719e9c25` |
| `ipfs_kit_py` | `df2f9cc092456329de9724c45a50c54b410875d1` |
| `ipfs_accelerate_py/mcplusplus` | `dc3164653a48d059ae9812078359daeafb451c07` |

---

## 1. Claimed interfaces (every claim is path- and revision-bound)

| Interface | Interface ID | Source path | Source revision | Primary test path | Test revision |
| --- | --- | --- | --- | --- | --- |
| `SemanticCompressionHarness` | `SemanticCompressionHarness@1` | `ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py` | `aa0cc549dbd1e04583ca99ce0ac039760aa52b3d` | `test/api/semantic_state/test_harness.py` | `aa0cc549dbd1e04583ca99ce0ac039760aa52b3d` |
| `ContextPacker` | `ContextPack@1` | `ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py` | `69b8893087b491088a49bb53b8f256ec6fadcc3b` | `test/api/semantic_state/test_context_pack.py` | `69b8893087b491088a49bb53b8f256ec6fadcc3b` |
| `IncrementalVerificationPlanner` | `IncrementalVerificationPlanner@1` | `ipfs_accelerate_py/agent_supervisor/verification/planner.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_incremental_verification_planner.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| `VerificationReceiptCache` | `VerificationReceiptCache@1` | `ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_verification_receipt_cache.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| `ModelRoutePlanner` | `ModelRoutePlanner@1` | `ipfs_accelerate_py/agent_supervisor/verification/model_route.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` | `test/api/test_agent_supervisor_verification_model_route.py` | `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |

`source_revision` / `test_revision` are the last commits that modified those
files. The inventory itself is bound to tree revision
`a0b825d8cfa384c284d0e77fa5341571c40adfa8`.

---

## 2. Semantic compression harness

### Package surface

- Package: `ipfs_accelerate_py.agent_supervisor.semantic_state`
- Export hub: `semantic_state/__init__.py` (import is side-effect free)
- Primary symbols: `SemanticCompressionHarness`, `run_semantic_patch_loop`,
  `HarnessPolicy`, `HarnessRequest`, `HarnessLoopOutcome`
- Schema: `ipfs-accelerate.semantic-compression-harness@1`
- Evidence: `sch/harness-loop@1`
- Adapter id: `ipfs-accelerate.semantic-state.harness`

### Loop

Fourteen ordered steps in `HARNESS_STEPS`:

1. `acquire_worktree`
2. `materialize_context_pack`
3. `invoke_model`
4. `validate_proposal`
5. `enforce_scope`
6. `apply_patch`
7. `rescan_changed_symbols`
8. `recompute_delta_invalidation`
9. `run_static_checks`
10. `run_selected_tests`
11. `run_proofs`
12. `optional_oracle`
13. `store_artifacts_and_manifest`
14. `compare_and_swap_root`

Public methods: `bootstrap_scan(request)`, `run(request)`,
`run_semantic_patch_loop(request)`. Module entrypoint:
`run_semantic_patch_loop(request)`.

### Fail-closed publication rules

- Rejection / unavailability / cancellation may store immutable candidate
  blocks but **never** advance the current `RootRef`.
- Acceptance requires a real production provider when a model is needed and
  fresh, non-simulated, admission-eligible receipts.
- `human_review_required` never invokes a provider and never publishes a root.
- Exact attempt replay is idempotent and does not re-charge the provider.
- Root CAS conflicts are reported; the prior root is never overwritten.

### Status vocabularies (contracts)

| Vocabulary | Values |
| --- | --- |
| `HarnessMode` | `development`, `production` |
| `HarnessDisposition` | `accepted`, `rejected`, `unavailable` |
| `AcceptanceDisposition` | `bootstrap`, `accepted`, `candidate`, `rejected` |
| `ModelRoute` | `deterministic_only`, `small_local_model`, `medium_model`, `frontier_model`, `human_review_required` |
| Provider modes | `production`, `development`, `simulated` |

### Focused tests

- `test/api/semantic_state/test_harness.py`
- `test/api/semantic_state/test_acceptance.py`
- `test/api/semantic_state/test_production_gates.py`
- `test/api/semantic_state/test_concurrency_and_recovery.py`
- `test/api/semantic_state/test_import_safety.py`

Release narrative: `docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md`.

---

## 3. Context packing

### Surface

- Class: `ContextPacker` in `semantic_state/context_pack.py`
  (revision `69b8893087b491088a49bb53b8f256ec6fadcc3b`)
- Function: `pack_context(...)`
- Interface id: `ContextPack@1`
- Result schema: `ipfs-accelerate.context-pack-result@1`
- Coverage policy schema: `ipfs-accelerate.context-coverage-policy@1`
- Token estimator version: `context-compiler-calibrated_utf8@1`

### Signature (module entry)

```text
pack_context(*, objective, target_source_cid, surrounding_source_cid,
             test_source_cid, dependency_admissions=(), obligation_cids=(),
             counterexample_cids=(), delta_cid, interface_cids=(),
             assumptions=(), exclusions=None, raw_source_regions=(),
             budget=None, policy=None, production_slice=None,
             production_slice_builder=None,
             estimator_version=TOKEN_ESTIMATOR_VERSION) -> ContextPackResult
```

### Coverage rules

- Default required kinds: `target_source`, `surrounding_source`, `test_source`
- Those kinds are also **never compressed**
- Capsule substitution is allowed only when admission permits it; exclusion
  explanations are required when configured
- Budget failure recommends escalation; it does not silently truncate
- Capsule facts remain datasets-owned; accelerate does not reimplement the
  capsule compiler

### Tests

- `test/api/semantic_state/test_context_pack.py` (rev `69b8893087b491088a49bb53b8f256ec6fadcc3b`)
- CLI packing coverage: `test/api/semantic_state/test_cli.py`

---

## 4. Incremental verification

### Public package

`ipfs_accelerate_py.agent_supervisor.verification` freezes a lazy, side-effect-free
public API (`IncrementalVerificationPublicApi@1` / `ivp/public-api@1`) at
commit `8c7800cedc5e1b848367db9952f912428466f8cc`.

Required public names:

- `create_verification_plan` / `IncrementalVerificationPlanner`
- `choose_model_route` / `ModelRoutePlanner`
- `build_verification_commitment`
- `VerificationReceiptCache`

### Planner

- Source: `verification/planner.py` @ `52e8fe17fdc7ac63f905872b194d930ae36ab1db`
- Interface: `IncrementalVerificationPlanner@1`
- Schema: `ipfs_accelerate_py/agent-supervisor/incremental-verification-planner@1`
- Evidence: `ivp/verification-plan@1`

```text
create_verification_plan(repository_state, invalidation_plan, context_pack,
                         patch_delta, policy, *, cache=None, adapter=None)
  -> VerificationPlan
```

Properties: deterministic; side-effect free; exact-key cache lookup only; no
tombstone mutation during planning; identity cross-checks against tree and
semantic roots; uncertainty broadens selection; unbound sandbox / policy
conflict / scope crossing force human review.

Tests: `test/api/test_agent_supervisor_incremental_verification_planner.py`,
`test/api/test_agent_supervisor_incremental_verification_report.py`,
`test/api/test_agent_supervisor_incremental_verification_conformance.py`.

### Receipt cache

- Source: `verification/receipt_cache.py` @ `52e8fe17fdc7ac63f905872b194d930ae36ab1db`
- Interface: `VerificationReceiptCache@1`
- Evidence: `ivp/receipt-cache@1`
- Methods: `lookup`, `lookup_many`, `admit`, `mark_stale`, `tombstone`,
  `replay`, GC helpers; helper `production_eligible(receipt)`

Production success terminals: **only** `passed` and `proved`.  
Never production-eligible: `stale`, `simulated`, `timeout`, `unavailable`,
`invalid`, malformed / kind-mismatched / key-mismatched / corrupt candidates.

Cache-reuse dispositions: `reused`, `stale`, `missing`, `corrupt`,
`mismatched`, `simulated`, `non_authoritative`, `policy_rejected`,
`terminal_status_rejected`.

Tests: `test/api/test_agent_supervisor_verification_receipt_cache.py` (primary),
plus store/executor/planner integration tests.

### Terminal status vocabulary

Closed set from `verification/contracts.py`
(rev `0cdf81bdd283dad6c27c9d23bbb6637d7dd54cff`):

```text
passed, failed, proved, disproved, unknown, timeout, unavailable,
not_modeled, stale, invalid, cancelled, simulated
```

Receipt kinds: `static_analysis`, `type_check`, `test`, `proof`.

### Commitment boundary (non-ZK)

`VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF = False`  
Hash algorithm `sha2-256`; domains `IVP-LEAF@1` / `IVP-NODE@1` / `IVP-EMPTY@1`.

The IVP Merkle commitment is explicitly **not** a zero-knowledge proof and
**cannot** substitute for a released full-checkpoint or delta proof sealer.

### Executor, evaluation, counterexamples, bundles

| Surface | Path | Revision | Role |
| --- | --- | --- | --- |
| `VerificationExecutor` | `verification/executor.py` | `52e8fe17...` | Execute plan under resource admission; production acceptance computation |
| Selected vs full suite | `verification/evaluation.py` | `52e8fe17...` | Controlled fixture recall/precision; **not** paired semantic shadowing |
| `CounterexampleMinimizer` | `verification/counterexamples.py` | `52e8fe17...` | Bounded failure minimization for context packing |
| Bundle builders | `verification/bundle.py` | `52e8fe17...` | Bundle / summary / commitment construction |

---

## 5. Routing

### Verification model-route planner (claimed interface)

- Source: `verification/model_route.py` @ `52e8fe17fdc7ac63f905872b194d930ae36ab1db`
- Interface: `ModelRoutePlanner@1`
- Evidence: `ivp/model-route@1`

```text
choose_model_route(context_pack, verification_plan, prior_attempts,
                   available_models, policy, *, routing_hints=None)
  -> ModelRouteDecision
```

Closed routes: `deterministic_only`, `small_local_model`, `medium_model`,
`frontier_model`, `human_review_required`.

Rules:

- Selects **capability class only**; vendor/provider/model IDs are rejected
- Does **not** downgrade when the required tier is unavailable; escalates to
  `human_review_required`
- Fail-closed human review precedes any model route for unresolved authority,
  unmodeled high risk, scope crossing, proof/test conflict, unsafe context,
  non-reproducible environment, or pending mandatory full/broader suite
- Module import performs no I/O and never invokes a provider

Tests: `test/api/test_agent_supervisor_verification_model_route.py`.

### Distinct harness routing (not the claimed IVP planner)

- Interface: `ModelRouting@1`
- Source: `semantic_state/routing.py` @ `644ad9dfa26c4fb6e0fd9f19a7c64a1458cad0dc`
- Entrypoints: `route_model`, `route_requires_human_review`,
  `route_allows_provider_dispatch`
- Confidence classes: `exact`, `conservative`, `heuristic`, `opaque`
- Risk classes: `low`, `medium`, `high`, `critical`
- Tests: `test/api/semantic_state/test_routing.py`

These two routers share the closed model-route vocabulary but serve different
phases (harness pre-execution vs verification next-repair). Downstream SCG
work must not collapse them into one owner.

---

## 6. Execution isolation and resource / provider seams

### Isolated worktree

- Source: `semantic_state/worktree.py` @ `f1b5d1572be8537a138fa5ab1d69f120774664e2`
- Entrypoints: `create_isolated_worktree`, `recover_isolated_worktree`,
  `validate_patch`, `apply_patch`, `IsolatedWorktree`, `PatchValidator`
- Tests: `test/api/semantic_state/test_worktree.py`
- Properties: detached worktree at pinned base; preimage + scope validation;
  fence/journal recovery; no silent mutation of the operator checkout

### Semantic scheduling + ResourceScheduler

- Adapter: `semantic_state/scheduling.py` @ `6715ac05736c63a1bffb158d6bce47396c60a102`
  (`schedule_semantic_work`, `replay_semantic_work`)
- Host/provider admission: `runtime/resource_scheduler.py` @
  `6d2c27b91e1d726d66af9d06f70bf5646dd5559c` (`evaluate`, `acquire`, `release`,
  `cancel`, `schedule`, …)
- Also consumed by `verification/executor.py` and `verification/process_runner.py`
- Tests: `test/api/semantic_state/test_scheduling.py`,
  `test/api/test_agent_supervisor_verification_process_runner.py`

### Providers

- Source: `semantic_state/providers.py` @ `644ad9dfa26c4fb6e0fd9f19a7c64a1458cad0dc`
- Entrypoints: `invoke_model`, `select_provider_for_route`,
  `ProductionProviderGate`, `InjectedModelProvider`
- Modes: `production`, `development`, `simulated`
- Simulated / degraded / OFF paths cannot report production verification
- Tests: `test_providers.py`, `test_provider_regressions.py`,
  `test_production_gates.py`

---

## 7. Benchmarks and known failures

### Semantic compression harness benchmark (controlled oracle/replay)

| Field | Value |
| --- | --- |
| Interface | `SemanticStateBenchmark@1` |
| Runner source | `semantic_state/benchmark.py` @ `e0a22ffceb9cdbdeb537f4ea1dea809fd8c38f92` |
| Checked-in JSON | `docs/benchmarks/semantic_compression_harness_results.json` @ `e0a22ffc...` |
| Checked-in MD | `docs/benchmarks/semantic_compression_harness_results.md` @ `e0a22ffc...` |
| Tasks | **40** |
| Median reduction | **58.90%** |
| Precision / recall | **36.22%** / **100.00%** |
| Controlled false negatives | **0** |
| Outcomes | 34 pass / 4 reject / 2 escalate |
| Production-eligible rows | **0** |
| Deterministic digest | `sha256:15bddb87fcf7af223caaf43f579bbc6e38342356ec6d7ec7acc2c4f541823dd6` |

Honest limitation: offline controlled oracle/replay only; **not** live model
quality and **not** production-eligible.

### Incremental verification benchmark (explicitly RED)

| Field | Value |
| --- | --- |
| Generator | `benchmarks/agent_supervisor/incremental_verification.py` @ `eb8e2a2f4402affa2b2ee2b30ca17d94ceab88c0` |
| Report | `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` @ `8c7800cedc5e1b848367db9952f912428466f8cc` |
| Cases | **20** (15 measured / 3 inconclusive / 2 not measured) |
| Ground-truth FN / FP | **1** / **7** |
| Frontier or human escalation | **40%** (8/20) |
| Real provers | **unavailable** (`not_measured`) |
| Zero stale/simulated acceptance | **met** |
| Zero controlled false negatives | **red** |
| Measurement status | **red**, non-authoritative |

These metrics are baseline and known-failure inputs for SCG, not success claims.

---

## 8. Shadow-mode disambiguation (acceptance-critical)

### Paired semantic shadowing (SCG meaning)

Paired semantic shadowing is the planned governor behavior that:

1. Runs the **compressed** ContextPack/route path, and
2. Runs a separate **expanded** raw-cone ContextPack/route path in an isolated
   evaluation worktree, then
3. Compares them into a differential semantic report.

Planned APIs (`create_shadow_plan`, `compare_shadow_results`, shadow executor)
are **not present** on the current tree under
`ipfs_accelerate_py/agent_supervisor/semantic_governor/`.

Related but **not equivalent** current surfaces:

- `compare_selected_with_full_suite` — test-selection evaluation, not model
  compressed-vs-expanded execution
- Harness `ContextModeComparison` / benchmark packing modes — offline token and
  selection metrics under oracle/replay, not live paired shadow execution

### Rollout shadow modes (must not be aliased)

These existing “shadow” labels are rollout / canary / report-only controls:

| Surface | Path | What it is |
| --- | --- | --- |
| `AssuranceRolloutMode.SHADOW` | `control/symbolic_assurance_rollout.py` | Assurance feature rollout gate |
| `RolloutMode.SHADOW` | `validation/change_propagation_rollout.py` | Change-propagation mode that cannot authorize mutation |
| Formal planning shadow thresholds | `planning/formal_planning_rollout.py` | Planning rollout stage thresholds |
| `RepairShadowReport@1` | `evaluation/dcr_shadow.py` | DCR report-only shadow with hard no-mutation thresholds |
| Drift monitor `shadow` stage | `autonomous_repair/drift_monitor.py` | Repair canary/shadow pipeline stage |
| Planner-doctor attestation `SHADOW` | `proof/planner_doctor_attestation.py` | Backend class that cannot emit ATTESTED production receipts |

**Inventory conclusion:** rollout shadow mode is **not** paired semantic
shadowing. SCG must implement paired shadow evaluation as a new governor
surface over the existing harness/IVP primitives, without reinterpreting
rollout shadow as that capability.

---

## 9. What accelerate owns for SCG consumption

Accelerate already owns and exports the orchestration primitives SCG should
wrap rather than rebuild:

1. Harness loop, isolated worktree, context packing, provider gate, root CAS
2. Incremental verification plan, exact-key receipt cache, executor, selected
   vs full evaluation, counterexample minimization, non-ZK commitment
3. Provider-neutral model-route capability classes (verification planner) and
   separate harness pre-execution routing
4. Resource admission via `ResourceScheduler`
5. Checked-in harness and RED IVP benchmark evidence

Accelerate does **not** currently own governor-level sufficiency claims, paired
semantic shadow orchestration, omission diagnosis, calibration, or authorized
policy promotion. Those are SCG deliverables that must consume the surfaces
above.

---

## 10. Acceptance checklist

| Criterion | Result |
| --- | --- |
| Every claimed interface has exact source path + revision | **Met** (section 1) |
| Every claimed interface has exact test path + revision | **Met** (section 1) |
| Rollout shadow not mistaken for paired semantic shadowing | **Met** (section 8) |
| RED / simulated / unavailable evidence recorded honestly | **Met** (sections 4, 7) |
| IVP commitment not claimed as ZK or sealer substitute | **Met** (section 4) |
