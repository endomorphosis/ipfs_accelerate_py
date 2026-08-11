# Incremental Verification Planner — Release Report (IVP-018 / IVP-019)

**Status:** terminal release report for IVP-G100 (documentation IVP-018 + public
export fan-in IVP-019)  
**Package:** `ipfs_accelerate_py.agent_supervisor.verification`  
**Board namespace:** `incremental-verification-planner-v1`  
**Evidence:** `ivp/documentation@1`, `ivp/release-report@1`, `ivp/public-api@1`,
`ivp/final-validation@1`  
**Depends on:** IVP-017 benchmark evidence (`ivp/benchmark@1`)  
**Authority:** this report is **not** production-authoritative; it documents
landed contracts, operations, and measured evidence without upgrading any
receipt status. No stale, simulated, timeout, unavailable, unknown,
not_modeled, invalid, cancelled, or pending full-suite receipt is accepted as
production success.

This report binds the documentation and the current-tree benchmark artifact to
the repository tree, controlled corpus, policy, effective environment, command
identities, and measurement status. A machine-checkable binding block is
included below; the focused report validator rejects stale tree bindings and
missing required sections.

---

## Binding identity

The following JSON binding is the authoritative report↔benchmark identity
surface for IVP-018. Values must match a fresh benchmark run for the same
`tree_id`. IVP-019 rebinds this surface to the terminal fan-in tree and records
public-export / full-suite observations without changing the binding schema
or `task_id` (validator surface remains `IVP-018`).

```json
{
  "benchmark_content_id": "baguqeerah6577y5hjqcvubcd3c7qcwixfig3ijxorw5bwd5qqe5cvdmcbqiq",
  "benchmark_evidence": "ivp/benchmark@1",
  "benchmark_schema": "ipfs_accelerate_py/agent-supervisor/incremental-verification-benchmark@1",
  "command_identities": {
    "generate_artifact": "benchmarks/agent_supervisor/incremental_verification.py",
    "validate_benchmark": "test/benchmarks/test_incremental_verification_planner_benchmark.py",
    "validate_report": "test/api/test_agent_supervisor_incremental_verification_report.py"
  },
  "corpus": {
    "corpus_cid": "sha256:83ec610cd96baba7419f9e8219a1189d919d5f6aa6717475cb218cb154a13f8e",
    "corpus_id": "ivp-semantic-capsule-controlled-v1",
    "evaluated_count": 20,
    "measurement_status": "measured"
  },
  "cross_tree_unaffected_reuse": {
    "explicitly_unmet": true,
    "new_tree_disposition": "missing",
    "new_tree_reusable": false,
    "reason": "exact_full_tree_binding_forbids_incompatible_cross_tree_reuse",
    "status": "unmet",
    "target": "unaffected_cross_tree_reuse"
  },
  "effective_environment": {
    "machine": "aarch64",
    "platform": "Linux-6.17.0-1014-nvidia-aarch64-with-glibc2.39",
    "python_version": "3.12.3",
    "system": "Linux"
  },
  "evidence": [
    "ivp/documentation@1",
    "ivp/release-report@1"
  ],
  "goal_id": "IVP-G100",
  "interface": "IncrementalVerificationReleaseReport@1",
  "measurement_schema_version": "ivp-benchmark-measurement/v1",
  "measurement_status": "red",
  "metrics_snapshot": {
    "cache_hit_rate": 0.5,
    "false_negatives_total": 1,
    "route_counts": {
      "frontier_model": 3,
      "human_review_required": 5,
      "medium_model": 1,
      "small_local_model": 11
    },
    "static_proof_status": "not_measured",
    "tests_full_total": 108,
    "tests_selected_total": 28
  },
  "policy": {
    "policy_id": "policy:ivp-incremental-verification-benchmark@1",
    "zero_stale_simulated_acceptance_hard": true
  },
  "schema": "ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@1",
  "target_misses": [
    {
      "count": 1,
      "detail": "total_false_negatives=1",
      "status": "red",
      "target": "zero_controlled_false_negatives"
    }
  ],
  "target_statuses": {
    "deterministic_commitments": "met",
    "incompatible_cross_tree_unaffected_reuse": "unmet",
    "metrics_complete": "met",
    "old_key_historical_preservation": "met",
    "small_route_localized_distribution": "met",
    "zero_controlled_false_negatives": "red",
    "zero_stale_simulated_accepted": "met"
  },
  "task_id": "IVP-018",
  "tree_id": "8dbf80bbdaa2208e0b7b2c4b2594a29f2d56fdeb"
}
```

**Command identities (operator-facing):**

| Identity | Command |
| --- | --- |
| generate_artifact | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 benchmarks/agent_supervisor/incremental_verification.py --output artifacts/agent_supervisor/incremental_verification/benchmark.json` |
| validate_benchmark | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/benchmarks/test_incremental_verification_planner_benchmark.py` |
| validate_report | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_incremental_verification_report.py` |

---

## 1. Modules changed

The incremental-verification subsystem lives under
`ipfs_accelerate_py/agent_supervisor/verification/`:

| Module | Role |
| --- | --- |
| `contracts.py` | Closed terminal statuses, receipt kinds, `VerificationReceiptKey`, plans, bundles, summaries, commitments, identity compiler |
| `datasets_adapter.py` | Lazy, fail-closed adapter for `RepositoryState`, `InvalidationPlan`, `SemanticCapsule`, `ContextPack` |
| `receipt_store.py` | Hermetic / optional ipfs-kit durable immutable receipt store with CAS index |
| `receipt_cache.py` | Production admission layer: exact-key reuse, tombstones, stale/simulated rejection |
| `process_runner.py` | Single admitted subprocess boundary (hermetic env, cancellation, process-tree fencing) |
| `adapters/pytest_adapter.py` | Exact node-id / full-suite pytest execution |
| `adapters/mypy_adapter.py` | Explicit mypy file/module/config execution |
| `adapters/prover_adapters.py` | Z3 and registry-admitted proof-assistant probes |
| `selection.py` | Pure semantic-edge affected-check selection with conservative/full-suite fallbacks |
| `planner.py` | `create_verification_plan` / `IncrementalVerificationPlanner` |
| `counterexamples.py` | Lease-rerun minimization and compact counterexample receipts |
| `model_route.py` | Provider-neutral `choose_model_route` / `ModelRoutePlanner` |
| `bundle.py` | `build_verification_bundle`, `build_verification_summary`, `build_verification_commitment` |
| `executor.py` | `execute_verification_plan` orchestration |
| `evaluation.py` | Controlled-fixture differential selected-vs-full evaluation |
| `__init__.py` | **IVP-019 frozen public export surface** — all names in `__all__` resolve lazily via `__getattr__` |

Supporting evidence harnesses (outside the package, consumed by this report):

- `benchmarks/agent_supervisor/incremental_verification.py` (IVP-017)
- `test/fixtures/incremental_verification/` controlled semantic-capsule corpus
- conformance suite under `test/api/test_agent_supervisor_incremental_verification_conformance.py` and related verification tests

---

## 2. Adapters implemented

| Adapter | Interface / schema | Authority notes |
| --- | --- | --- |
| Datasets input adapter | `DatasetsVerificationInputAdapter` | Strict canonical mappings; optional registered upstream types; no network or install side effects |
| Pytest | `PytestVerificationAdapter@1` | Exact selected node IDs or full-suite oracle; timeout/unavailable/cancelled never pass |
| Mypy | `MypyVerificationAdapter@1` | Explicit argv only; missing mypy → `unavailable` |
| Z3 / proof assistants | prover adapters | Z3 sat/unsat/unknown and registry-admitted Lean/Coq/Isabelle probes; `sorry`/`admit` cannot prove |
| Process runner | `VerificationProcessRunner` | Shared hermetic sandbox + process-tree cancellation |
| Receipt store | `VerificationReceiptStore@1` | Local hermetic backend; optional lazy `ipfs_kit_py` byte transport |
| Receipt cache | `VerificationReceiptCache@1` | Exact-key production admission; historical preservation under old keys |

---

## 3. Receipt schemas and exact cache key

### 3.1 Closed terminal statuses

```text
passed, failed, proved, disproved, unknown, timeout, unavailable,
not_modeled, stale, invalid, cancelled, simulated
```

Non-accepting production statuses include `timeout`, `unavailable`, `unknown`,
`not_modeled`, `stale`, `invalid`, `cancelled`, and `simulated`. Wrapper
`passed`/`proved` projections derive only from authoritative assurance or
current direct execution.

### 3.2 Receipt and decision schemas

Primary wire types (package contracts):

- `StaticAnalysisReceipt`, `TypeCheckReceipt`, `TestReceipt`, `ProofReceipt`
- `CounterexampleReceipt`
- `VerificationPlan`, `VerificationBundle`, `VerificationSummary`
- `CacheReuseDecision`, `ModelRouteDecision`
- `VerificationCommitment`
- `VerificationReceiptKey@1`

Benchmark artifact schema:
`ipfs_accelerate_py/agent-supervisor/incremental-verification-benchmark@1`.

Release-report binding schema:
`ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@1`.

### 3.3 Exact key (`VerificationReceiptKey@1`)

The exact key binds every authority-relevant input:

1. repository tree CID (exact executed patched tree / dirty overlay)
2. semantic-state root CID
3. sorted affected symbol-version CIDs
4. environment CID (effective hermetic sandbox)
5. dependency-lock CID
6. check/test selector CID
7. proof-obligation CID, or canonical `not_applicable`
8. tool name
9. tool version
10. configuration CID
11. sorted fixture-data CIDs
12. network policy
13. receipt-schema version
14. receipt kind and adapter schema (prevents adapter aliasing)
15. optional proof-backend binding when applicable

Any mutation of a component yields a different key. Caller-supplied CID strings
are references only; `VerificationIdentityCompiler` re-derives and cross-checks
from observed inputs before lookup and publication. Floats, secrets, witnesses,
oversized values, wrong schemas, and unchecked identities reject fail-closed.

---

## 4. Invalidation behavior

| Event | Behavior |
| --- | --- |
| Relevant code / symbol / fixture / config change | Selects/invalidates affected checks; new tree-bound receipts required |
| Unrelated edit on same repository | Old receipt preserved under **old key**; **not** admitted for new full-tree key |
| Environment change | Invalidates (environment CID component) |
| Dependency-lock change | Invalidates |
| Tool name/version or configuration change | Invalidates |
| Stale receipt | Rejected for production acceptance |
| Simulated receipt | Rejected for production acceptance |
| Cross-tree lookup of an old receipt | Miss / non-reusable — incompatible cross-tree unaffected reuse is **unmet** by design |
| Content corruption / kind mismatch / proof–test conflict | Fail closed |
| Late success after cancellation | Publication fenced; does not become production success |

Exact full-tree binding forbids silently reusing a receipt across trees.
Historical immutability is preserved; authority is not.

---

## 5. Selected / full / proof results

From the current-tree benchmark (`measurement_status` aggregate **red**):

| Metric | Value |
| --- | --- |
| Corpus cases evaluated | 20 |
| Measured cases | 15 |
| Inconclusive cases | 3 |
| Not measured cases | 2 |
| Tests selected (total across cases) | 28 |
| Tests full suite (total across cases) | 108 |
| Ground-truth false negatives (corpus) | 1 |
| Ground-truth false positives (corpus) | 7 |
| Outcome discrepancies / inconclusive | 3 cases |
| Static checks executed | 0 (`not_measured` on controlled catalogs) |
| Type checks executed | 0 |
| Proof obligations executed | 0 |
| Real provers on PATH | `z3` and `lean` available (measured); `coqc` / `isabelle` typed `unavailable` / `not_measured` |

Proof and static execution remain typed `not_measured` on the controlled
semantic-capsule corpus when catalogs are empty; missing real provers are never
fabricated into passes. Available provers without catalog obligations still do
not mint production proof success.

The single controlled false negative is fixture `seeded-false-negative`
(ground truth / full-suite oracle:
`tests/test_mod.py::test_deliberately_fails`). It is recorded as **red**, not
suppressed.

---

## 6. Cache hits

Hermetic exact-key cache experiment (benchmark metrics):

| Metric | Value |
| --- | --- |
| Lookups | 4 |
| Hits | 2 |
| Misses | 2 |
| Cache hit rate | 0.5 (5000 bps) |
| Zero stale/simulated production acceptance | **met** (hard) |
| Old-key historical preservation | **met** (hard) |
| Cross-tree unaffected reuse | **unmet** (explicit) |

Reused-time savings use paired cold/hot cache observations where available and
estimated selected-vs-full labels otherwise; wall times are observational
samples with declared tolerance, not deterministic gates.

---

## 7. Model-route distribution

Provider-neutral routes only (no vendor selection in policy):

| Route | Count (corpus cases) |
| --- | --- |
| `small_local_model` | 11 |
| `medium_model` | 1 |
| `frontier_model` | 3 |
| `human_review_required` | 5 |
| Frontier escalation rate | 0.4 (8/20) |

Small-model routing for localized fixtures: **met** (9/9 localized measured,
fraction 1.0 ≥ 0.20 minimum). Routing remains separate from the supervisor's
implementation-provider control-plane route.

---

## 8. Counterexample examples

Compact counterexamples are bounded (default 8 KiB) projections for
ContextPack consumption — never full raw logs.

**Example A — deliberately failing selected test (fixture
`deliberately-failing-observed`):**

- Selected failure retained with minimized frames and assertion text
- Counterexample context ≈ 347 bytes / 87 estimator tokens
- Compared raw-log bound estimate ≈ 4466 bytes / 1117 tokens
- Tokens saved (estimator-bound) ≈ 1030 under tokenizer
  `ivp-estimator/utf8-bytes-div4@1` v1.0.0

**Example B — config-edge localized change (`config-edge-change`):**

- Selected: `tests/test_config.py::test_configured` (1 of 6 full suite)
- Compact context ≈ 269 bytes / 68 tokens without claiming a new failure when
  none is observed
- Route: `small_local_model` with `localized_exact_counterexample`

Corpus aggregate counterexample context on this tree: 5361 bytes / 1347
estimator tokens (bound 8192). Estimator-bound token savings total: 19834 under
`ivp-estimator/utf8-bytes-div4@1` v1.0.0 against compared artifact bounds
(raw log 262144 B / counterexample 8192 B).

Lease-rerun minimization (IVP-011) produces `CounterexampleReceipt` records
that bind selector, tree, and diagnostic digests without private witnesses.

---

## 9. Commitment format and determinism

`build_verification_commitment(verification_bundle)` produces a structural
Merkle commitment over admitted receipt leaves:

| Field | Value / rule |
| --- | --- |
| Hash | SHA-256 (`sha2-256`) |
| Leaf codec | `canonical-dag-json@1` (canonical DAG-JSON UTF-8) |
| Leaf domain | `IVP-LEAF@1` → `H("IVP-LEAF@1\0" \|\| leaf)` |
| Node domain | `IVP-NODE@1` → `H("IVP-NODE@1\0" \|\| left \|\| right)` |
| Empty domain | `IVP-EMPTY@1` |
| Odd nodes | promoted unchanged |
| Sorting | canonical by receipt key/CID before tree build |
| Outputs | Merkle root, public statement, repository tree CID, environment CID, required-check-set CID, unresolved-obligation count, fail-closed aggregate terminal status |

Aggregate status precedence is fail-closed (invalid → stale → simulated →
cancelled → timeout → unavailable → unknown → not_modeled → disproved →
failed → success). Aggregation cannot upgrade any required leaf. Changing
required membership or content changes the root; input permutation does not
after canonical sorting. Benchmark target `deterministic_commitments`: **met**.

### Commitment non-claims (mandatory)

1. **This commitment is not a ZK proof** — it is not itself a zero-knowledge
   proof of execution or correctness.
2. **Signatures need trusted issuers** — signed receipts do not prove test
   execution unless the issuer is trusted.
3. **Structural validation is not cryptographic validation** of the underlying
   execution, tool honesty, or sandbox integrity.

`VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF` is permanently `False`.

---

## 10. Limitations

- Full ZK pytest/Python execution is **out of scope**; no ZK backend is shipped.
- No new theorem prover is introduced; proof assistants require registry
  admission and real offline capabilities.
- Automatic dependency installation, mock hardware/inference, deployment, and
  provider/vendor selection inside route policy are forbidden.
- Controlled corpus static/proof catalogs are empty → static/proof execution is
  honestly `not_measured`.
- On this measurement host: `z3` and `lean` probed available; `coqc` and
  `isabelle` were **unavailable** (typed, never treated as success).
- Wall-time metrics are observational samples with tolerance; they do not
  create correctness authority.
- Token savings are estimator-bound (`utf8-bytes-div4@1`), not a production
  billing meter.
- Controlled corpus still records **one** seeded false negative
  (`seeded-false-negative`); aggregate measurement status remains **red**.
  This is measured honestly and is not promoted to production success.
- Ruff on the full verification package still reports pre-existing adapter
  lint findings (e.g. BLE001/S110 in mypy adapter) and environment `N999`
  noise from the worktree path on package `__init__.py` files; the IVP-019
  `__init__.py` itself is clean under ordinary rules when `N999` is ignored.
- Resource-scheduler leased-lane tests could not prove full process-tree
  fencing on this host (5 failures); dedicated process-tree fencing tests
  pass. Outside IVP-019 edit scope — recorded, not force-greened.

---

## 11. Unmet targets (every miss, including cross-tree)

| Target | Status | Notes |
| --- | --- | --- |
| `zero_stale_simulated_accepted` | met (hard) | Stale/simulated never production-accepted |
| `deterministic_commitments` | met (hard) | Membership/content sensitive; permutation invariant |
| `old_key_historical_preservation` | met (hard) | Old immutable receipts remain under old keys |
| `metrics_complete` | met | Typed measurements always emitted |
| `small_route_localized_distribution` | met | ≥1 and ≥20% of localized measured fixtures |
| `zero_controlled_false_negatives` | **red** | corpus total FN = 1 (`seeded-false-negative`); recorded honestly; not claimed as production success |
| `incompatible_cross_tree_unaffected_reuse` | **unmet** (explicit) | Exact full-tree binding forbids incompatible cross-tree reuse; reason `exact_full_tree_binding_forbids_incompatible_cross_tree_reuse` |

**Incompatible cross-tree reuse is an intentional unmet target**, not a silent
failure. Historical preservation under the original key holds; the new tree
cannot reuse the old receipt as production evidence.

Exact full-tree binding forbids incompatible cross-tree reuse. No timeout,
unavailable, unknown, not_modeled, invalid, cancelled, stale, simulated, or
pending full-suite receipt is accepted as production success.

---

## 12. Exact future ZK step

No ZK circuit is added by this program. The **exact next step for ZK
aggregation** is:

1. **Freeze** the admitted receipt **leaf codec** (`canonical-dag-json@1`) and
   **trust policy** (issuer trust, domain tags `IVP-LEAF@1` / `IVP-NODE@1` /
   `IVP-EMPTY@1`, fail-closed aggregate lattice).
2. **Publish deterministic cross-implementation Merkle vectors** over that
   frozen leaf codec and domain separation so independent implementations
   agree on roots for the same admitted leaves.
3. **Only then** add an **external** membership/aggregation circuit that proves
   membership or aggregation over the committed Merkle root **without changing
   ordinary verification authority** — ordinary receipts, exact keys, and
   production admission remain the source of truth.

Until those freezes and vectors exist, any ZK claim over verification outcomes
is rejected as out of scope.

---

## 13. Operations summary

| Operation | Entry point |
| --- | --- |
| Plan | `create_verification_plan(repository_state, invalidation_plan, context_pack, patch_delta, policy)` |
| Route | `choose_model_route(context_pack, verification_plan, prior_attempts, available_models, policy)` |
| Execute | `execute_verification_plan(...)` via `VerificationExecutor` |
| Commit | `build_verification_commitment(verification_bundle)` |
| Cache | `VerificationReceiptCache.lookup` / `.admit` |
| Benchmark | `benchmarks/agent_supervisor/incremental_verification.py` |
| Module ops guide | `ipfs_accelerate_py/agent_supervisor/verification/README.md` |

Trust doctrine: no cache presence, provider text, signature alone, CID string,
historical pass, or structural validation creates verification authority.

---

## 14. IVP-019 public export freeze (`ivp/public-api@1`)

Package root `ipfs_accelerate_py.agent_supervisor.verification` freezes a
lazy, side-effect-free public surface:

**Required public names (lazy via `__getattr__`):**

- `create_verification_plan`
- `choose_model_route`
- `build_verification_commitment`
- `VerificationReceiptCache`
- `IncrementalVerificationPlanner`
- `ModelRoutePlanner`

**Also frozen (lazy):** contract types (`TerminalStatus`, receipts, plan,
bundle, summary, commitment, identity compiler, …),
`execute_verification_plan` / `VerificationExecutor`,
`build_verification_bundle` / `build_verification_summary`,
`production_eligible`, and related production helpers listed in package
`__all__`.

Properties measured on this tree:

- Cold `import ipfs_accelerate_py.agent_supervisor.verification` loads only the
  package `__init__` (no planner/cache/adapter modules).
- First attribute access resolves the owning submodule and caches the object.
- Package exports preserve `is` identity with the submodule definitions.
- Evidence label: `PUBLIC_API_EVIDENCE = "ivp/public-api@1"`.
- Interface label: `IncrementalVerificationPublicApi@1`.

Example (lazy package root imports — preferred after IVP-019):

```python
from ipfs_accelerate_py.agent_supervisor.verification import (
    create_verification_plan,
    choose_model_route,
    build_verification_commitment,
    IncrementalVerificationPlanner,
    ModelRoutePlanner,
    VerificationReceiptCache,
)
```

Submodule imports remain valid and side-effect free.

---

## 15. IVP-019 terminal fan-in validation (`ivp/final-validation@1`)

Measured on tree `8dbf80bbdaa2208e0b7b2c4b2594a29f2d56fdeb` with
`PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:.` and pytest `--timeout=300`.

### 15.1 Focused verification matrix (suite 1) — **green**

Command: the full IVP-019 focused list (contracts through report + benchmark).

| Observation | Result |
| --- | --- |
| Final measured run | **496 passed**, 1 warning, 41.88s, exit 0 |
| Conformance matrix (18 required cases) | proven |
| Report validator vs fresh benchmark | green (current `tree_id` + binding) |
| Production doctrine in suite | no stale / simulated / timeout / unavailable / unknown / not_modeled / invalid / cancelled / pending full-suite receipt accepted as production success |

Preconditions applied for hermetic local measurement (not product scope changes):

- Generated gitignored corpus via `test/fixtures/incremental_verification/build_corpus.py`
- Rebound checked-in benchmark artifact to the current tree so identity tests
  bind HEAD (content id
  `baguqeerah6577y5hjqcvubcd3c7qcwixfig3ijxorw5bwd5qqe5cvdmcbqiq`)

### 15.2 Declared regression suite (suite 2) — **partial**

| File | Result |
| --- | --- |
| `test_agent_supervisor_test_execution_identity.py` | **24 passed** |
| `test_proof_reuse_invalidation_mutations.py` | **34 passed** |
| `test_proof_reuse_security_concurrency.py` | **14 passed** |
| `test_agent_supervisor_formal_verification_cache.py` | **22 passed** |
| `test_agent_supervisor_validation_scheduler.py` | **45 passed** |
| `test_agent_supervisor_process_tree_fencing.py` | **3 passed** |
| `test_agent_supervisor_resource_scheduler.py` | **43 passed, 5 failed** |

Aggregate suite 2: **185 passed, 5 failed** (exit 1) across three retries.
All five failures are the same host-side class:

```text
RuntimeError: could not prove process tree <pid> fully fenced
```

raised from `ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py` during
resource-scheduler leased-lane / dynamic-scheduler tests. Dedicated
process-tree fencing tests pass. These resource failures are **outside IVP-019
allowed edit paths** (fan-in may not expand into merge/resource runtime to
force-green them). They are recorded as an unmet host/runtime residual, not as
verification-receipt production success.

### 15.3 Ruff — **partial / pre-existing**

- `ipfs_accelerate_py/agent_supervisor/verification/__init__.py` (IVP-019):
  **All checks passed** with `--ignore N999` (worktree directory name triggers
  environment `N999` on every package `__init__.py` in this checkout).
- Whole `verification` package + declared test globs: **pre-existing** findings
  remain (adapters/tests: BLE001, S110, F401, UP037, …; ~109 on package alone
  excluding N999). Not introduced by the public-export freeze; not repaired
  here (out of allowed edit paths / no scope expansion).

### 15.4 Fan-in residual summary

| Gate | Status |
| --- | --- |
| Lazy required public names | **met** |
| Focused IVP matrix | **met** (496 passed) |
| Identity / proof-reuse / formal-cache / validation / process-tree regressions | **met** |
| Resource-scheduler regressions | **unmet** (5 fencing-proof failures on host) |
| Ruff on new `__init__.py` | **met** (ignore env N999) |
| Ruff full declared package/tests | **unmet** (pre-existing noise) |
| Benchmark measurement status | **red** (1 controlled FN; honest) |
| Production non-accepting statuses never success | **met** (doctrine + suite evidence) |

---

## 16. Evidence and honesty statement

- Benchmark evidence schema: `ivp/benchmark@1` (artifact
  `artifacts/agent_supervisor/incremental_verification/benchmark.json`
  rebound to tree `8dbf80bbdaa2208e0b7b2c4b2594a29f2d56fdeb`, content id
  `baguqeerah6577y5hjqcvubcd3c7qcwixfig3ijxorw5bwd5qqe5cvdmcbqiq`).
- Documentation evidence: `ivp/documentation@1`.
- Release report evidence: `ivp/release-report@1`.
- Public API evidence: `ivp/public-api@1`.
- Final validation evidence: `ivp/final-validation@1`.
- Aggregate measurement status on this tree: **red** (one controlled false
  negative recorded; not claimed as production success).
- Resource-scheduler residual: 5 process-tree fencing-proof failures remain on
  this host; they are not reclassified as verification success and are not
  hidden.
- This report does not assert target success from favourable performance
  metrics. Performance and route distribution are reported without changing
  status semantics.
- **Production acceptance doctrine (hard):** a cache hit, provider claim,
  signature, CID string, historical pass, structural validation, timeout,
  unavailable tool, simulated result, unknown / not_modeled / invalid /
  cancelled / stale status, or pending full-suite obligation never
  manufactures current verification authority or production success.

---

*End of IVP-018/IVP-019 release report. Public exports are frozen; terminal
fan-in records measured red measurement status and host residuals honestly
without promoting non-accepting receipts.*
