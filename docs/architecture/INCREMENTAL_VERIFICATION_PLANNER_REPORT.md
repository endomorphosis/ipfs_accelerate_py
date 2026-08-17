# Incremental Verification Planner — Release Report (IVP-018)

**Status:** honest draft report for IVP-G100 / IVP-018  
**Package:** `ipfs_accelerate_py.agent_supervisor.verification`  
**Board namespace:** `incremental-verification-planner-v1`  
**Evidence:** `ivp/documentation@1`, `ivp/release-report@2`
**Depends on:** IVP-017 benchmark evidence (`ivp/benchmark@2`)
**Authority:** this report is **not** production-authoritative; it documents
landed contracts, operations, and measured evidence without upgrading any
receipt status.

This report binds the documentation and benchmark artifact to a canonical
effective-source snapshot, controlled corpus, policy, effective environment,
command identities, and measurement status. `observed_head` records Git HEAD
for diagnostics only and grants no freshness authority. A machine-checkable
binding block is included below; the focused report validator rejects stale
source-snapshot bindings and missing required sections.

---

## Binding identity

The following JSON binding is the authoritative report↔benchmark identity
surface for IVP-018. Values must match a fresh benchmark run for the same
`source_snapshot_id`.

```json
{
  "benchmark_content_id": "baguqeeram6jtyce43lbhoazqxpw5zfltxu3qibiezbvzrsexfkxaidhksctq",
  "benchmark_evidence": "ivp/benchmark@2",
  "benchmark_schema": "ipfs_accelerate_py/agent-supervisor/incremental-verification-benchmark@2",
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
    "ivp/release-report@2"
  ],
  "goal_id": "IVP-G100",
  "interface": "IncrementalVerificationReleaseReport@2",
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
  "observed_head": "7628f9d2553b607767ce0d851949dcdaaf4ef7ea",
  "schema": "ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2",
  "source_snapshot_domain": "ivp-source-snapshot@1",
  "source_snapshot_id": "sha256:32bf46ccd07392b0c1dd4b7006f21e67fad0f5e439b034a586ebb3610065939c",
  "source_snapshot_schema": "ipfs_accelerate_py/agent-supervisor/ivp-source-snapshot@1",
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
  "task_id": "IVP-018"
}
```

**Command identities (operator-facing):**

| Identity | Command |
| --- | --- |
| generate_artifact | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 benchmarks/agent_supervisor/incremental_verification.py --output artifacts/agent_supervisor/incremental_verification/benchmark.json` |
| validate_benchmark | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/benchmarks/test_incremental_verification_planner_benchmark.py` |
| validate_report | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_incremental_verification_report.py` |

### Fixed-point source identity

The v2 evidence envelope uses schema
`ipfs_accelerate_py/agent-supervisor/ivp-source-snapshot@1` and domain
`ivp-source-snapshot@1`. Its `source_snapshot_id` hashes a sorted manifest of
the effective present tracked and nonignored-untracked paths. Regular files
bind canonical Git mode and exact bytes, symlinks bind mode `120000` and target
bytes, and gitlinks bind mode `160000` and their exact indexed object ID. The
manifest contains no tracked/untracked provenance, Git HEAD, branch, commit
metadata, timestamp, observation time, or absolute repository root.

Exactly two self-referential outputs are excluded:
`artifacts/agent_supervisor/incremental_verification/benchmark.json` and
`docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md`. There are no
wildcard or caller-supplied exclusions. In the IVP taskboard, only exact
`- Status: todo` and `- Status: completed` rows inside parsed IVP task blocks
share a lifecycle sentinel; every title, dependency, other metadata row,
status-like prose, malformed row, and other byte remains identity-bearing.
Deleted tracked paths are absent, so committing an already-observed deletion
does not alter the snapshot. The reviewed `ipfs_kit_py` and
`ipfs_datasets_py` gitlinks must be initialized, exact, and clean or snapshot
construction fails closed.

`observed_head` is diagnostic only: it is outside source identity, benchmark
content identity, and freshness validation. Production
`VerificationReceiptKey`, executed-tree admission, cache reuse, and
`VerificationCommitment` full-tree bindings remain unchanged.

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
| `source_snapshot.py` | Canonical fixed-point effective-source identity for benchmark/report evidence |
| `__init__.py` | Side-effect-free package boundary (final public export freeze is IVP-019) |

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
`ipfs_accelerate_py/agent-supervisor/incremental-verification-benchmark@2`.

Release-report binding schema:
`ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2`.

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

From the current-source benchmark (`measurement_status` aggregate **red**):

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
| Real provers on PATH (z3/lean/coqc/isabelle) | none — typed `unavailable` / `not_measured` |

Proof and static execution remain typed `not_measured` on the controlled
semantic-capsule corpus when catalogs are empty; missing real provers are never
fabricated into passes.

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
- Real provers (z3, lean, coqc, isabelle) were **unavailable** on the measurement
  host; absence is typed, never treated as success.
- Wall-time metrics are observational samples with tolerance; they do not
  create correctness authority.
- Token savings are estimator-bound (`utf8-bytes-div4@1`), not a production
  billing meter.
- Package public-export freeze and terminal fan-in remain **IVP-019**.
- Hard zero false-negative conformance is owned by IVP-016/IVP-019; this report
  records red measurements without suppressing them.

---

## 11. Unmet targets (every miss, including cross-tree)

| Target | Status | Notes |
| --- | --- | --- |
| `zero_stale_simulated_accepted` | met (hard) | Stale/simulated never production-accepted |
| `deterministic_commitments` | met (hard) | Membership/content sensitive; permutation invariant |
| `old_key_historical_preservation` | met (hard) | Old immutable receipts remain under old keys |
| `metrics_complete` | met | Typed measurements always emitted |
| `small_route_localized_distribution` | met | ≥1 and ≥20% of localized measured fixtures |
| `zero_controlled_false_negatives` | **red** | corpus total FN = 1 (`seeded-false-negative` style control); hard only in IVP-016/IVP-019 |
| `incompatible_cross_tree_unaffected_reuse` | **unmet** (explicit) | Exact full-tree binding forbids incompatible cross-tree reuse; reason `exact_full_tree_binding_forbids_incompatible_cross_tree_reuse` |

**Incompatible cross-tree reuse is an intentional unmet target**, not a silent
failure. Historical preservation under the original key holds; the new tree
cannot reuse the old receipt as production evidence.

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

## 14. Evidence and honesty statement

- Benchmark evidence schema: `ivp/benchmark@2` (artifact
  `artifacts/agent_supervisor/incremental_verification/benchmark.json` when
  regenerated for the current source snapshot).
- Documentation evidence: `ivp/documentation@1`.
- Release report evidence: `ivp/release-report@2`.
- Aggregate measurement status on this tree: **red** (one controlled false
  negative recorded; hard gate deferred to IVP-016/IVP-019).
- This report does not assert target success from favourable performance
  metrics. Performance and route distribution are reported without changing
  status semantics.

---

*End of IVP-018 release report draft. IVP-019 freezes public exports and runs
the terminal release fan-in against the full focused suite.*
