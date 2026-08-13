# AAE-002 — Datasets inventory: index, capsules, claim analysis, mutation, property, and vacuity

**Evidence id:** `aae/datasets-inventory@1`  
**Interface:** `AAEDatasetsInventory@1`  
**Task:** AAE-002  
**Schema:** `aae/datasets-inventory@1`  
**Machine-readable companion:** [`datasets.json`](datasets.json)

## Authority

| Field | Value |
| --- | --- |
| Repository | `endomorphosis/ipfs_datasets_py` |
| Gitlink path | `ipfs_datasets_py` |
| Commit | `fbd1ba9f70803de157622bb20e22595ef09d606f` |
| Inspected | 2026-08-13 |
| Status | Reusable ISI, semantic-state/capsule contracts, IR claim/property declarations, controlled mutation fixtures, and narrow tactician vacuity checks are present. No integrated AAE mutation campaign or general vacuity analyzer. |

This inventory records **existing** datasets surfaces that AAE may reuse. It does **not** invent missing campaign, detection, gap, or vacuity APIs.

## Ownership and conflict policy

Datasets owns, for AAE reuse:

- incremental semantic index (scan, delta, invalidation, explain, watch)
- storage-neutral semantic-state producer (bundle, root, Merkle, capsules)
- functional `SemanticCapsuleCompiler@1` (interface identifier, **not** a public class)
- `AnalysisConfidence` ranks `exact` / `conservative` / `heuristic` / `opaque` and durable `AnalysisLimitation` records
- IR claim / assumption / obligation declarations and software-verification property vocabulary
- controlled semantic-state mutation fixtures and ISI invalidation fixtures
- CIDv1 content identity under `logic.software_contracts.content`
- narrow heuristic vacuity / non-vacuity checks inside `ProofCandidateValidator@1`

Datasets does **not** own:

- accelerate `MutationLedger@1` (file-lineage ledger, not a campaign engine)
- accelerate context packing, verification planner/executor, or provider routing
- kit durable store / root CAS / recovery
- the planned `adversarial_assurance` domain package as an existing public export on this pin
- integrated mutation campaigns, equivalent-mutant analysis, expected-detection engines, or four-family vacuity analysis

**Conflict policy:** inventory only. Do not infer a public `SemanticCapsuleCompiler` class, a second scanner/graph/compiler/CID profile, or treat narrow `is_vacuous_statement` as `analyze_vacuity`. Missing AAE surfaces are `typed_unavailable`.

## Package map

| Package | Import | Source |
| --- | --- | --- |
| ISI | `ipfs_datasets_py.logic.software_contracts.semantic_index` | `ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_index/` |
| Semantic state | `ipfs_datasets_py.logic.software_contracts.semantic_state` | `ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state/` |
| Content identity | `ipfs_datasets_py.logic.software_contracts.content` | `.../content.py` |
| IR claims | `ipfs_datasets_py.logic.ir_core.claims` | `.../ir_core/claims.py` |
| Verification properties | `ipfs_datasets_py.logic.software_verification.properties` | `.../software_verification/properties.py` |
| Candidate validation (vacuity stage) | `ipfs_datasets_py.logic.software_verification.tactician.candidate_validation` | `.../tactician/candidate_validation.py` |

Normative docs in the pin:

- `ipfs_datasets_py/docs/software_contracts/INCREMENTAL_SEMANTIC_INDEX.md`
- `ipfs_datasets_py/docs/software_contracts/SEMANTIC_STATE_CONTRACT.md`
- `ipfs_datasets_py/docs/software_contracts/SOUNDNESS_AND_THREAT_MODEL.md`
- `ipfs_datasets_py/docs/software_contracts/CID_PROFILE_V1.md`

---

## 1. Analysis confidence: exact / conservative / heuristic / opaque

Closed enum `AnalysisConfidence` in `semantic_index/models.py`:

| Value | Rank | Meaning for AAE reuse |
| --- | --- | --- |
| `exact` | 0 | Exact under declared extractor/scope assumptions; may admit `exact_substitute` when also fresh |
| `conservative` | 1 | Sound over-approximation; may admit `conservative_substitute_with_caveats` when fresh with visible caveats |
| `heuristic` | 2 | Heuristic only; **never** raises confidence; raw source required; cannot discharge proof obligations |
| `opaque` | 3 | Opaque/unresolved; raw source required; may force full test/proof fallback; always visible |

**Promotion rule:** confidence is never raised by combining evidence; the least-confident rank wins.

Surfaces that carry confidence include symbols, artifacts, edges, invalidation obligations, capsules, symbol Merkle nodes, and `AnalysisLimitation` records. Semantic-governor coverage reuses the same four ranks as `AnalysisConfidenceRank`.

Capsule admission coupling:

- `exact_substitute` — fresh + exact
- `conservative_substitute_with_caveats` — fresh + conservative with ≥1 visible caveat
- `raw_source_required` — heuristic, opaque, stale/unknown freshness, corrupt/mismatch, or explicit raw-source obligations

---

## 2. Index asset — `IncrementalSemanticIndex`

**Kind:** public class + pure module-level functional API  
**Source:** `semantic_index/index.py`  
**Export:** `from ipfs_datasets_py.logic.software_contracts.semantic_index import IncrementalSemanticIndex, scan_repository, ...`

Importing the package does not create a store, scan, start a watch, or open a network connection.

### Construction

```text
IncrementalSemanticIndex(*, scanner: RepositoryScanner | None = None, store: SemanticIndexStore | None = None)
```

### Public operations

| Operation | Essence | Notes |
| --- | --- | --- |
| `scan_repository` / `scan` | `(repo_path, previous_state=None) -> RepositoryState` | `previous_state` is verified reuse only |
| `diff_repository_states` | `(previous, current) -> RepositoryStateDelta` | Deterministic semantic delta; `rename_candidates` are heuristic |
| `calculate_invalidation` | `(previous, current, delta) -> InvalidationPlan` | Bounded ISI obligations only |
| `explain_symbol` | `(state, symbol_id) -> SymbolExplanation` | Recorded facts + confidence limitations |
| `explain_impact` | `(state, changed_symbol_ids) -> ImpactExplanation` | Bounded reverse-dependency impact |
| `watch_repository` | `(repo_path, callback, *, debounce_ms=250)` | Only API that may create a worker thread |
| `store_state` / `publish_state` / `load_state` | store-backed | Require injected store; CAS on publish |

Cold and incremental scans of identical repository bytes produce **byte-identical** state records and root CIDs.

### Semantic-state extension (index → sealed view)

```text
build_semantic_state(...) -> SemanticStateBundle
extend_semantic_invalidation(...) -> SemanticInvalidationPlan
select_tests_and_proofs(...) -> TestSelection          # pure; does not run pytest
compare_test_selection_oracle(...) -> TestOracleComparison  # pure metrics
```

---

## 3. Capsule asset — functional `SemanticCapsuleCompiler@1`

**Kind:** interface identifier + functional API  
**Not a public class**  
**Source:** `semantic_state/capsules.py`

```text
SEMANTIC_CAPSULE_COMPILER_INTERFACE = "SemanticCapsuleCompiler@1"
SEMANTIC_CAPSULE_COMPILER_SCHEMA    = "ipfs-datasets.software-contracts.semantic-capsule-compiler@1"
CAPSULE_COMPILER_VERSION            = "1"
SEMANTIC_CAPSULE_SCHEMA             = "ipfs-datasets.software-contracts.semantic-capsule@1"
```

### API

```text
compile_semantic_capsule(semantic_index, stable_symbol_id, *, ...) -> SemanticCapsule
compile_semantic_capsules(semantic_index, *, previous_bundle=None, ...) -> CapsuleCompileResult
verify_capsule_compile_result(result) -> CapsuleCompileResult
```

`CapsuleCompileResult` fields: `capsules`, `index`, `blocks`, `reused_cids`.

### Precise boundaries

1. Sole owner of `SemanticCapsuleCompiler@1`.
2. Consumes a sealed ISI view plus a bindings-owned relevant projection.
3. Never rediscovers binding scope, never raises confidence, never references another capsule or symbol-node CID as a dependency.
4. Heuristic metadata keys (`llm_summary`, `summary`, `ai_summary`, …) are excluded from authoritative truth.
5. `previous_bundle` may accelerate only after complete current inputs reverify and stored block bytes are **byte-identical** to the cold path.

### Freshness and raw source

```text
assess_capsule_freshness(capsule, *, current_state, invalidation=None) -> CapsuleFreshness
read_required_source(semantic_index, stable_symbol_id, *, expected_producer_state_cid, ...)
```

Freshness states: `fresh` | `stale` | `unknown`.  
Admission: `exact_substitute` | `conservative_substitute_with_caveats` | `raw_source_required`.

Verified read boundary: `SemanticStateView@1` / `VerifiedSemanticStateView` with rehashing `get_block`.

Accelerate (and AAE) must hold `SemanticCapsuleRef` shapes rather than recompiling capsules.

---

## 4. Claim analysis assets

Existing claim-analysis material is **not** AAE claim/specification analysis. It is:

### 4.1 Confidence-ranked semantic graph claims

- Every durable symbol/edge/obligation carries `AnalysisConfidence`.
- Durable `AnalysisLimitation(code, message, subject_id, confidence)` remains visible on the semantic-state root.
- `explain_symbol` / `explain_impact` report recorded graph facts and limitation strings only.
- Opaque paths always surface raw-source requirements; explanations **do not** promote heuristic/opaque observations into runtime-behavior claims.

### 4.2 IR claim declarations

Source: `logic/ir_core/claims.py`

| Type | Schema | Role |
| --- | --- | --- |
| `IRClaim` | `ir-claim/v1` | Content-addressed claim with assumptions + obligations; digest independent of verification runs |
| `Assumption` | `ir-assumption/v1` | Explicit premise; declaration ≠ truth |
| `ProofObligation` | `ir-proof-obligation/v1` | Theorem-shaped target with **no** implied verification status |

An `IRClaim` requires at least one obligation; obligation `assumption_ids` must reference declared assumptions.

### Not present (do not invent)

- `analyze_claim_specification` (AAE planned)
- assurance-gap taxonomy as a datasets API
- expected-detection claim binding as a datasets API

---

## 5. Mutation assets (fixtures only)

### Controlled semantic-state mutation fixture

Path: `ipfs_datasets_py/tests/fixtures/software_contracts/semantic_state/`

| Field | Value |
| --- | --- |
| Manifest schema | `ipfs-datasets.software-contracts.semantic-state-controlled-fixture@1` |
| Interface | `SemanticStateControlledFixture@1` |
| Recipe | `recipe.py` |
| Helpers | `list_mutation_cases()`, `apply_mutation(repository, case_id)` |

Required mutation kinds (18):

`local_body`, `signature`, `cross_module`, `schema`, `exception`, `fixture`, `config`, `plugin`, `lock`, `policy`, `interface`, `generated`, `dynamic`, `monkey`, `native`, `format`, `delete`, `rename`

Constraints: no checked-in git, no state store, no generated receipt, no hand-built dependency edges, no second benchmark corpus; scanner consumes trees without importing the fixture package.

### ISI invalidation fixtures

Path: `ipfs_datasets_py/tests/fixtures/software_contracts/incremental_semantic_index/`

Cases include body/test impact, dataclass/schema, deletion/rename, dynamic import, exception recovery, fixture/config, formatting identity, git snapshot authority, lock environment, monkey patch, persistence recovery, pytest identity, relation closure, signature callers, unrelated edit.

### Out of scope / different owners

| Asset | Owner | Note |
| --- | --- | --- |
| `MutationLedger@1` | accelerate analysis | File-lineage states; reusable lineage evidence, **not** a campaign engine |
| `test/fixtures/proof_reuse_mutations.py` | accelerate / proof-reuse | Invalidation corpus for proof-backed test reuse |
| `tests/fixtures/logic/modal/leanstral_mutations.json` | datasets modal/legal-IR | Codec mutation fixture, not semantic-assurance operators |

### Not present (typed_unavailable)

`generate_mutation_candidates`, campaign `execute_mutation`, `classify_mutation_outcome`, equivalent-mutant analyzer, expected-detection engine, authorized AAE promotion flow.

---

## 6. Property assets

### Verification property vocabulary

Source: `logic/software_verification/properties.py`

- `VerificationProperty` (`verification-property/v1`) — source-mapped semantic target; no backend request or proof authority
- `VerificationAssumption` (`verification-assumption/v1`) — source-grounded premise with no implied truth

Closed `PropertyKind` includes authentication, authorization, contract, data_race_freedom, heap_safety, hyperproperty, invariant, liveness, noninterference, reachability, refinement, safety, satisfiability, secrecy, termination, theorem, trace_conformance, validity (plus namespaced custom kinds).

`AssumptionKind`: semantic, environment, modeling, platform, fairness, trust, boundedness, translation.

### Existing property / fuzz tests

| Area | Path examples |
| --- | --- |
| Semantic-governor / software contracts | `tests/unit/logic/software_contracts/semantic_governor/test_audit_contracts.py` |
| Software-verification counterexamples | `tests/integration/logic/software_verification/counterexamples/test_explanation.py` |
| Logic-parser Hypothesis/fuzz | `tests/fuzz/logic/test_parser_properties.py`, `test_wave2_parser_properties.py` |
| Tactician vacuity bounded fuzz | `tests/security/logic/test_goal_tactician_adversarial.py` |

Broader Hypothesis property tests exist under optimizers/graphrag/knowledge-graphs. That breadth is **not** integrated semantic mutation-campaign coverage.

---

## 7. Vacuity assets (narrow only)

The **only** located formal-vacuity support is a narrow heuristic path for proof candidates:

| Symbol | Kind | Role |
| --- | --- | --- |
| `is_vacuous_statement(statement) -> bool` | narrow heuristic | Recognizes tautology markers (`true`, `⊤`, `tautology`, …) after normalization |
| `is_contradiction_statement(statement) -> bool` | narrow heuristic | Recognizes explicit falsehood markers |
| `_check_non_vacuity(candidate) -> ValidationCheck` | internal stage | Pipeline stage `non_vacuity`; fails vacuous/tautological candidates |
| `validate_candidate` / `validate_candidate_set` | public entry | `ProofCandidateValidator@1` convenience API that includes the stage |

Source: `logic/software_verification/tactician/candidate_validation.py`  
(`is_vacuous_statement` also appears in `abduction.py`; the validation pipeline stage lives in candidate validation.)

This is **not**:

- `analyze_vacuity`
- formal / policy / test / ZK vacuity families
- SMT or semantic-entailment vacuity
- residual-property reporting beyond the stage detail string

Those AAE surfaces remain `typed_unavailable` on this pin.

---

## 8. Canonical identity

Authority: `ipfs_datasets_py.logic.software_contracts.content`  
Profile: `software-contract-cid-profile-v1` (CIDv1, base32, sha2-256; raw source codec; dag-json structured codec).

Public functions include `canonical_dag_json_bytes`, `cid_for_bytes`, `cid_for_structured`, `validate_cid`, `verify_source_read`, `verify_structured_read`.  
Vectors: `tests/fixtures/software_contracts/cid_vectors.json`.

---

## 9. Limitations / blind spots (must remain visible)

| Area | Typical reporting | Consequences |
| --- | --- | --- |
| Dynamic dispatch / import | conservative / heuristic / opaque | incomplete impact; raw source |
| Reflection | opaque (`reflection`) | raw source / review |
| Descriptors / unsafe decorators | conservative+ | no exact capsule substitution |
| `eval` / `exec` / runtime generation | opaque | raw source; possible full fallback |
| Metaclasses | opaque (`metaclass_mutation`) | raw source |
| Monkey patching | opaque (`monkey_patch`) | raw source |
| Pytest plugins / dynamic collection | heuristic / opaque | full pytest fallback |
| Native extensions | opaque (`native_boundary`) | full pytest/proof fallback |
| Renames | heuristic candidates only | do not preserve identity |
| Uncontrolled I/O | conservative+ | purity/security review |
| Incomplete call graphs | limitations on explain/impact | unknown reachability may force full suite |
| Docstrings / LLM summaries | non-authoritative | **cannot** raise confidence or discharge obligations |

**Contract statement:** this surface cannot make dynamic Python exact. Unknown reachability stays visible and may force raw source or `full_pytest` / `full_proofs` / `both`.

---

## 10. Missing APIs (typed_unavailable — do not invent)

| Name | Status | Existing partial |
| --- | --- | --- |
| `generate_mutation_candidates` | typed_unavailable | controlled fixture kinds only |
| expected-detection engine | typed_unavailable | — |
| mutation campaign execute/classify | typed_unavailable | — |
| equivalent-mutant / survivor diagnosis | typed_unavailable | — |
| `analyze_vacuity` and four vacuity families | typed_unavailable | narrow `is_vacuous_statement` / `_check_non_vacuity` |
| public `SemanticCapsuleCompiler` class | must_not_invent | functional `@1` interface |
| `logic.software_contracts.adversarial_assurance` package | typed_unavailable | planned AAE domain path |

---

## 11. Focused tests and validation

Recommended evidence paths:

- `ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_index`
- `ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_state`
- `ipfs_datasets_py/tests/unit/logic/software_contracts/test_content_identity.py`
- `ipfs_datasets_py/tests/unit/logic/ir_core`
- `ipfs_datasets_py/tests/security/logic/test_goal_tactician_adversarial.py`
- `ipfs_datasets_py/tests/fuzz/logic`

Task validation command:

```bash
python3 -m json.tool docs/architecture/adversarial_assurance_inventory/datasets.json >/dev/null
```

Optional focused suite (not required by AAE-002 gate):

```bash
python3.12 -m pytest -q \
  ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_state \
  ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_index \
  ipfs_datasets_py/tests/unit/logic/software_contracts/test_content_identity.py
```

---

## 12. Handoff summary for Adversarial Assurance Engine

```text
ISI scan/diff/invalidate/explain
        │
        ▼
build_semantic_state → SemanticStateBundle / SemanticStateView
        │
        ├─ compile_semantic_capsule  (SemanticCapsuleCompiler@1 functional)
        ├─ assess_capsule_freshness  (exact/conservative vs raw_source gate)
        ├─ IRClaim / VerificationProperty declarations (solver-neutral)
        ├─ controlled mutation fixture kinds + ISI invalidation cases (seed corpora)
        └─ ProofCandidateValidator non_vacuity stage (narrow hard-fail only)
                 │
                 └─ future AAE domain (typed_unavailable on this pin):
                    mutation candidates, expected detection, equivalence,
                    four vacuity families, gaps, adequacy, promotion
```

AAE **consumes** this surface. It must not reimplement the index, capsule compiler, content identity, or graph. It must not relabel fixture corpora as campaign engines or narrow vacuity markers as general vacuity analysis.

## Acceptance (AAE-002)

| Criterion | Status |
| --- | --- |
| Distinguishes exact / conservative / heuristic / opaque analysis | Yes — closed `AnalysisConfidence` ranks, promotion rule, admission coupling, surfaces |
| Functional `SemanticCapsuleCompiler@1` | Yes — constants, functions, cycle-free deps, non-class nature |
| Existing fixtures / fuzzing inventoried | Yes — controlled 18-kind mutation fixture, ISI fixtures, logic fuzz, Hypothesis breadth note |
| Narrow vacuity checks only; no invented missing APIs | Yes — `is_vacuous_statement` / `_check_non_vacuity` only; campaign and four-family vacuity marked `typed_unavailable` |
