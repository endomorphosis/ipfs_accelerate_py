# SCG-002 — Datasets inventory: semantic index, state, capsule, invalidation, and selection

**Evidence id:** `scg/datasets-inventory@1`  
**Task:** SCG-002  
**Schema:** `scg/datasets-inventory@1`  
**Machine-readable companion:** [`datasets.json`](datasets.json)

## Authority

| Field | Value |
| --- | --- |
| Repository | `endomorphosis/ipfs_datasets_py` |
| Gitlink path | `ipfs_datasets_py` |
| Commit | `1330038f626ef92993f03d46f21e1a57719e9c25` |
| Planning status | Completed incremental semantic index and semantic-state/capsule contracts |
| Inspected | 2026-08-13 |

SCG plan baseline for this pin: **67 passed** on focused index/capsule/invalidation/selection/content-identity tests. Workspace collect-only under the unit suites plus content-identity currently enumerates **503** tests; that larger number is the full suite collection, not a replacement for the plan baseline claim.

## Ownership and conflict policy

Datasets owns the incremental semantic index (ISI), storage-neutral semantic-state producer (facts, links, Merkle nodes, capsules), environment bindings, additive invalidation, pure test/proof selection, pure selected-versus-full oracle metrics, producer-bound raw-source admission, and the software-contract CIDv1 profile under `logic.software_contracts.content`.

It does **not** own a second scanner, AST frontend, symbol identity, call graph, CID profile, kit durable store/WAL/root CAS, accelerate context packing/patch loop/model routing/verification execution, or MCP++ wire types.

**Conflict policy (normative for SCG):** Do not infer a public class from an interface name or propose a second scanner, graph, compiler, or CID implementation. `SemanticCapsuleCompiler@1` is an **interface identifier** implemented by the functional capsule compiler; it is **not** a public class and must not be re-created.

## Package map

| Package | Import | Source directory |
| --- | --- | --- |
| ISI | `ipfs_datasets_py.logic.software_contracts.semantic_index` | `ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_index/` |
| Semantic state | `ipfs_datasets_py.logic.software_contracts.semantic_state` | `ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state/` |
| Content identity | `ipfs_datasets_py.logic.software_contracts.content` | `ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/content.py` |

Normative docs in the pin:

- `ipfs_datasets_py/docs/software_contracts/INCREMENTAL_SEMANTIC_INDEX.md`
- `ipfs_datasets_py/docs/software_contracts/SEMANTIC_STATE_CONTRACT.md`
- `ipfs_datasets_py/docs/software_contracts/SOUNDNESS_AND_THREAT_MODEL.md`
- `ipfs_datasets_py/docs/software_contracts/CID_PROFILE_V1.md`

---

## 1. `IncrementalSemanticIndex`

**Kind:** public class + pure module-level functional API  
**Source:** `semantic_index/index.py`  
**Export:** `from ipfs_datasets_py.logic.software_contracts.semantic_index import IncrementalSemanticIndex, scan_repository, ...`

Importing the package does not create a store, scan, start a watch, or open a network connection.

### Construction

```text
IncrementalSemanticIndex(*, scanner: RepositoryScanner | None = None, store: SemanticIndexStore | None = None)
```

- `scanner` and `store` are injected capabilities.
- No local store is inferred from a repository path.
- Persistence occurs only through `store_state` / `publish_state` when a store was supplied.

### Public operations

| Operation | Signature (essence) | Notes |
| --- | --- | --- |
| `scan_repository` / `scan` | `(repo_path, previous_state=None) -> RepositoryState` | `previous_state` is verified reuse only; cannot alter result for identical bytes |
| `diff_repository_states` | `(previous, current) -> RepositoryStateDelta` | Deterministic semantic delta |
| `calculate_invalidation` | `(previous, current, delta) -> InvalidationPlan` | Bounded ISI obligations only |
| `explain_symbol` | `(state, symbol_id) -> SymbolExplanation` | Recorded facts + confidence limits |
| `explain_impact` | `(state, changed_symbol_ids) -> ImpactExplanation` | Bounded reverse-dependency impact |
| `watch_repository` | `(repo_path, callback, *, debounce_ms=250) -> RepositoryWatch` | Only API that may create a worker thread; notification-only |
| `store_state` / `publish_state` / `load_state` | store-backed | Require injected store; CAS on publish |

Module-level functions of the same names implement the interoperable API; the class is a convenience owner for scanner, last state, and optional store.

### Scanner constants

| Constant | Value |
| --- | --- |
| `SCANNER_NAME` | `semantic-repository-scanner` |
| `SCANNER_VERSION` | `1` |
| `DEFAULT_EXTRACTOR_NAME` | `python-cpython-ast` |
| `DEFAULT_EXTRACTOR_VERSION` | `1` |
| `SEMANTIC_INDEX_SCHEMA` | `ipfs-datasets.software-contracts.semantic-index@2` |
| Python analysis schema | `ipfs-datasets.software-contracts.python-semantic-analysis@2` |

### Durable ISI records

- **`RepositoryState`:** `repository_id`, sorted `symbols` / `artifacts` / `edges`, extractor name/version, schema; identity via `state_cid`.
- **`RepositoryStateDelta`:** previous/current state CIDs; added/deleted/modified/unchanged symbol and artifact IDs; edge add/delete; `rename_candidates` (heuristic, not identity-preserving).
- **`InvalidationPlan` / `InvalidationObligation`:** previous/current state CIDs and obligations (`subject_id`, `reason_code`, `remediation_kind`, `confidence`, identity pair, supporting edges, details).
- **`SymbolRecord`:** dual identity `stable_id` + `version_cid`, source provenance, span, confidence, signature, normalized AST, etc.
- **`DependencyEdge`:** source/target, relation, extraction method, confidence, extractor version, optional span/metadata.

Cold and incremental scans of identical repository bytes produce **byte-identical** state records and root CIDs. Reuse diagnostics live on `RepositoryScanner.last_reuse_diagnostics` and never enter durable root identity.

### Scanner exclusions

Ignored directory names include `.git`, virtualenvs, caches, `node_modules`, build/dist/coverage outputs, vendored trees, and the default semantic-index store paths (`.semantic-index`, `.semantic_index`, `semantic-index-state`).

---

## 2. `SemanticCapsuleCompiler@1` (functional compiler)

**Kind:** interface identifier + functional API  
**Not a public class**  
**Source:** `semantic_state/capsules.py`  
**Constants:**

```text
SEMANTIC_CAPSULE_COMPILER_INTERFACE = "SemanticCapsuleCompiler@1"
SEMANTIC_CAPSULE_COMPILER_SCHEMA    = "ipfs-datasets.software-contracts.semantic-capsule-compiler@1"
CAPSULE_COMPILER_VERSION            = "1"
SEMANTIC_CAPSULE_SCHEMA             = "ipfs-datasets.software-contracts.semantic-capsule@1"
```

### API

```text
compile_semantic_capsule(semantic_index, stable_symbol_id, *, relevant_bindings=None, binding_set=None, environment_bindings=None, relevant_projection=None, projections=None) -> SemanticCapsule

compile_semantic_capsules(semantic_index, *, ..., previous_bundle=None) -> CapsuleCompileResult
verify_capsule_compile_result(result) -> CapsuleCompileResult
```

`CapsuleCompileResult` fields: `capsules`, `index`, `blocks`, `reused_cids`.

### Precise boundaries

1. **Sole owner** of `SemanticCapsuleCompiler@1`.
2. Consumes a sealed ISI view (`RepositoryState` or `SemanticIndexForCapsules`: `state_cid`, `symbols`, `artifacts`, `edges`, `repository_id`) plus a bindings-owned relevant projection.
3. **Never** rediscovers binding scope as a second authority, **never raises confidence**, **never** references another capsule or symbol-node CID as a dependency (cycle-free content identity).
4. Capsules reference dependency **stable IDs, versions, fact CIDs, and link IDs** only.
5. Heuristic metadata keys (`llm_summary`, `summary`, `ai_summary`, `generated_summary`, …) are excluded from authoritative truth; only closed promoted keys (`defaults`, `contracts`, `effects`, `exception_behavior`, `docstring`, `docstring_hint`) may surface as first-class fields.
6. `previous_bundle` may accelerate materialization **only after** complete current inputs reverify and stored block bytes are **byte-identical** to the cold path. Cold and verified-incremental compilation over identical inputs are always byte-identical.

### Normative producer key

```text
(stable_symbol_id, version_cid, semantic_index_schema, extractor_version)
```

### `SemanticCapsule` fields (closed)

`stable_symbol_id`, `version_cid`, `semantic_index_schema`, `extractor_version`, capsule schema/compiler version, source slice path/`source_cid`, `symbol_fact_cid`, signature/annotations/defaults/decorators/contracts/effects/exception behavior, schema/serialization relations, test/fixture/proof refs, dependency stable/version/fact/link IDs, `confidence`, `relevant_binding_projection_cid`, `docstring_hint`, `metadata`.

**Freshness is not a capsule field.** It is assessed separately as `CapsuleFreshness`.

### Accelerate reference (must not recompile)

```text
SemanticCapsuleRef(
  capsule_cid, semantic_state_root_cid, stable_symbol_id, version_cid,
  source_cid, confidence, validity_bindings, raw_source_required
)
```

---

## 3. `SemanticStateView` (verified read-only boundary)

**Kind:** protocol + verified implementation  
**Source:** `semantic_state/api.py`  
**Interface constants:**

```text
SemanticStateProducer@1
SemanticStateView@1
SemanticStateBlockReader@1
ipfs-datasets.software-contracts.semantic-state-api@1
```

### Protocols

```text
SemanticStateBlockReader:
  get_block(cid: str) -> bytes

SemanticStateView:
  root: SemanticStateRoot
  get_block(cid: str) -> bytes
  symbol_node(stable_symbol_id: str) -> SymbolMerkleNode
  capsule(stable_symbol_id: str) -> SemanticCapsule
```

**Implementation:** `VerifiedSemanticStateView` via:

| Entry | Role |
| --- | --- |
| `build_semantic_state(semantic_index, *, environment_bindings=(), previous_bundle=None) -> SemanticStateBundle` | Cold or verified-incremental assembly; no persistence side effect |
| `verify_semantic_state_bundle(bundle) -> SemanticStateRoot` | Full reverify of root-reachable blocks and indexes |
| `open_semantic_state(root_cid, get_block) -> VerifiedSemanticStateView` | Injected read-only block reader |
| `view_semantic_state_bundle(bundle) -> VerifiedSemanticStateView` | In-memory finite block map |

### Storage-neutral boundary (precise)

`get_block` on the view:

- fetches bytes through the injected reader (or finite bundle map);
- **rehashes** against the claimed CIDv1;
- raises `MissingBlockError` / `CorruptBlockError` / `UnknownSymbolError` as typed failures;
- has **no** put, CAS, WAL, provider, network, kit, scheduler, context-pack, receipt, or MCP++ envelope hasher behavior.

`SemanticStateBundle` (`root` + `blocks`) is the **only** persistence handoff object from datasets. Accelerate stores blocks through kit, re-reads through `get_block`, and publishes its **own** generation-bearing `SemanticStateRootManifest` after verification receipts. That manifest is intentionally distinct from the datasets-domain `SemanticStateRoot`.

### `SemanticStateRoot` fields and exclusions

**Included:** `repository_id`, `producer`, schema/compiler versions, sorted-pair index CIDs for symbol facts, artifact facts, semantic links, symbol nodes, capsules, environment binding set, and analysis limitations.

**Deliberately excluded** (`ROOT_EXCLUDED_FIELD_NAMES`): previous-root history, repository deltas, invalidation plans, selections, receipts, acceptance claims, timestamps/clocks, process IDs, local paths, leases/fences, CAS generations, model/provider outputs, prompts/context packs, MCP++ request/attempt/envelope identities.

---

## 4. Invalidation (`InvalidationPlan` → `SemanticInvalidationPlan`)

### ISI plan

```text
calculate_invalidation(previous_state, current_state, delta) -> InvalidationPlan
```

**ISI reason codes:** `new_capsule`, `proof_rerun`, `stale_test_receipt`, `caller_signature_mismatch`, `obsolete_schema_adapter`, `effect_assumption_stale`, `exception_recovery_stale`, `purity_security_review`, `environment_receipt_stale`, `deleted_symbol_dependency`, `raw_source_requirement`.

**ISI rules:** `body`, `signature`, `effects`, `exceptions`, `schema`, `fixture_config`, `environment`, `deletion`, `opaque`, `edge`.

**Observed ISI remediations include:** `rerun_test`, `rerun_proof`, `retire_capsule`, `retrieve_raw_source`, `review_adapter`, `review_assumption`, `review_call_site`, `review_dependent`, `review_recovery`, `review_security_purity`.

### Semantic extension

```text
extend_semantic_invalidation(
  previous_index, current_index, delta, plan,
  previous_state, current_state,
  *, previous_bindings=None, current_bindings=None, max_obligations=2000
) -> SemanticInvalidationPlan
```

- Preserves ISI obligations (`origin=isi`) and adds environment-binding obligations (`origin=environment`).
- Default obligation bound: **2000** (`MAX_SEMANTIC_INVALIDATION_OBLIGATIONS`).
- Additional semantic reasons cover lock/manifest/pytest plugin/config/proof/policy/interface/generated/toolchain/schema/compiler changes, unknown binding scope, unmapped subjects, stale bound capsule/receipt, and `full_fallback_required`.
- **Semantic remediations** include `full_pytest_fallback`, `full_proofs_fallback`, `full_fallback`, rebuild/review remediations, and `retrieve_raw_source`.

**Consumer rule:** execute only emitted obligations; do not invent dependency facts from names or rankings.

---

## 5. `TestSelection` and oracle

**Interfaces:** `TestSelection@1`, `ProofSelection@1`  
**Source:** `semantic_state/test_selection.py`, `oracle.py`, models in `models.py`  
**Schema:** `ipfs-datasets.software-contracts.semantic-test-selection@1`

### API

```text
select_tests_and_proofs(
  previous_state, current_state, invalidation, *, policy,
  explicit_rules=(), previous_index=None, current_index=None
) -> TestSelection

compare_test_selection_oracle(
  selection, *, baseline_full, selected_run, candidate_full, authored_oracle=None
) -> TestOracleComparison
```

**Purity:** selection never imports or collects target tests and never guesses pytest node IDs from names. Oracle comparison never executes pytest; accelerate supplies normalized `TestRunFacts`.

### `TestSelection` fields

`previous_root_cid`, `current_root_cid`, `selected_pytest_node_ids`, `selected_proof_ids`, `reason_paths`, covered/unresolved obligation IDs, known test universe CID/count, `fallback`, `fallback_reasons`, `policy_cid`.

### Selection policy

`SelectionPolicy(policy_id, allow_full_fallback=True, include_proofs=True, include_fixtures=True, max_selected_tests=None, metadata={})`.

**Rule kinds:** `include`, `exclude`, `force_full`, `force_full_pytest`, `force_full_proofs`.

### Fallback vocabulary

| `SelectionFallback` | Meaning |
| --- | --- |
| `none` | bounded selection only |
| `full_pytest` | force full pytest suite |
| `full_proofs` | force full proofs |
| `both` | full pytest and full proofs |

**Fallback reasons (closed set):** `dynamic_pytest_plugin`, `native_or_opaque_reachability`, `unknown_test_universe`, `insufficient_graph_evidence`, explicit force rules, full-fallback obligations, `policy_disallows_fallback`, `max_selected_tests_exceeded`.

### Oracle metrics

TP/FN/FP, new/missed regressions, fixture recall/precision (basis points), selection ratio, execution reduction, fallback rate, regression recall, selected/full counts, changed-outcome node IDs.

- Empty authored oracle → `OracleApplicability.not_applicable` (never fabricated 100%).
- Controlled acceptance requires **zero fixture false negatives** and **zero missed regressions**.
- Full-suite fallback is **measured**, not described as precise selection.

### Accelerate reference

```text
TestSelectionRef(selection_cid, previous_semantic_state_root_cid_or_null, current_semantic_state_root_cid)
```

Accelerate must not run a second graph selector.

---

## 6. Freshness and producer-bound source

### `assess_capsule_freshness`

```text
assess_capsule_freshness(capsule, *, current_state: SemanticStateView, invalidation=None) -> CapsuleFreshness
requires_raw_source(assessment) -> bool
is_safe_substitute(assessment) -> bool
```

| Taxonomy | Closed values |
| --- | --- |
| `FreshnessState` | `fresh`, `stale`, `unknown` |
| `AdmissionDecision` | `exact_substitute`, `conservative_substitute_with_caveats`, `raw_source_required` |

**Safe substitute:** only fresh `exact_substitute`, or fresh `conservative_substitute_with_caveats` with at least one visible caveat.

**Raw source required** for heuristic/opaque confidence, stale/unknown/invalid capsules, schema/compiler/producer mismatch, binding projection mismatch, raw-source obligations, and edit-context cases called out in the semantic-state contract.

### `read_required_source` (`ProducerBoundSource@1`)

```text
read_required_source(semantic_index, stable_symbol_id, *, expected_producer_state_cid, read_source_blob=None) -> VerifiedSourceMaterialization
```

Returns `evidence` (`VerifiedSourceEvidence`) + `source_bytes`. Never reads ambient filesystem paths, never imports target code, never treats capsule text as exact source. Wrong-state/TOCTOU/corrupt failures require rescan.

---

## 7. Canonical identity rules

**Sole authority:** `ipfs_datasets_py.logic.software_contracts.content`

| Profile field | Value |
| --- | --- |
| `profile_id` | `software-contract-cid-profile-v1` |
| `profile_version` | `1.0.0` |
| CID version | 1 |
| Multihash | `sha2-256` |
| Base | `base32` |
| Source codec | `raw` |
| Structured codec | `dag-json` |

**Rejected structured types:** float, bytes, set/tuple/path/host objects, NaN/infinity, `repr` fallback.

### Dual symbol identities

| Identity | Schema | Includes | Excludes |
| --- | --- | --- | --- |
| `stable_id` | `semantic-stable-symbol-id@1` | repository, language=`python`, normalized module path, qualified name, kind, namespace | source bytes, spans, comments, formatting, definition ordinal |
| `version_cid` | `semantic-symbol-version-id@2` | stable_id, index schema, extractor, normalized AST, signature, decorators/property role, annotations | raw formatting-only noise where extractor preserves semantic version |

`source_cid` is separate provenance. Semantic-state **never recalculates** stable IDs or version CIDs; it preserves final-ISI authorities.

---

## 8. Confidence, relations, statuses

**`AnalysisConfidence` (closed):** `exact` | `conservative` | `heuristic` | `opaque`. Never promoted by combining evidence; least-confident rank wins when combined.

**`RelationType` (closed):** `imports`, `calls`, `inherits`, `implements`, `reads_state`, `writes_state`, `raises`, `catches`, `serializes`, `deserializes`, `validates`, `tested_by`, `uses_fixture`, `configured_by`, `generated_from`, `proof_depends_on`.

The symbol graph is an **evidence and invalidation boundary**, not a complete runtime call graph.

---

## 9. Opaque / dynamic limitations (remain visible)

This inventory **must not** hide the following. They surface as `AnalysisConfidence`, durable `AnalysisLimitation` records, `SymbolMerkleNode.raw_source_required_reasons`, `CapsuleFreshness.caveats`, and `TestSelection.fallback_reasons`.

| Area | Typical reporting | Consequences |
| --- | --- | --- |
| Dynamic dispatch | conservative / heuristic / opaque | incomplete impact; may require raw source |
| Unsafe/unknown decorators, descriptors | conservative+ | no exact capsule substitution |
| Reflection (`getattr`/`setattr`/`inspect`/…) | opaque (`reflection`) | raw source / review |
| Import hooks, dynamic import, `eval`/`exec`/`compile` | opaque (`runtime_code_generation`, `eval_or_exec`) | raw source; possible full fallback |
| Metaclasses | opaque (`metaclass_mutation`) | raw source |
| Monkey patching | opaque (`monkey_patch`) | raw source |
| Pytest plugins / dynamic collection | heuristic / opaque | `dynamic_pytest_plugin` → full pytest fallback |
| Native extensions (`ctypes`/`cffi`/cython) | opaque (`native_boundary`) | `native_or_opaque_reachability` fallback |
| Uncontrolled I/O | conservative+ | purity/security review obligations |
| Incomplete call graphs | limitations on explain/impact | unknown reachability may force full suite |
| Generated inputs / bindings | environment obligations | rebuild remediations |
| Rename candidates | heuristic only | do not preserve identity |
| Docstrings / optional LLM summaries | non-authoritative | **cannot** raise confidence or discharge obligations |

**Contract statement:** this surface cannot make dynamic Python exact. Unknown reachability stays visible and may force raw source or `full_pytest` / `full_proofs` / `both`.

---

## 10. Fixtures

### ISI fixtures

`ipfs_datasets_py/tests/fixtures/software_contracts/incremental_semantic_index/` — cases include body/test impact, dataclass/schema, deletion/rename, dynamic import, exception recovery, fixture/config, formatting identity, git snapshot authority, lock environment, monkey patch, persistence recovery, pytest identity, relation closure, signature callers, unrelated edit.

### Controlled semantic-state fixture

`ipfs_datasets_py/tests/fixtures/software_contracts/semantic_state/`

- Manifest schema: `ipfs-datasets.software-contracts.semantic-state-controlled-fixture@1`
- Interface: `SemanticStateControlledFixture@1`
- Recipe-driven baseline + mutation kinds: local body, signature, cross-module, schema, exception, fixture, config, plugin, lock, policy, interface, generated, dynamic, monkey, native, format, delete, rename
- Constraints: no checked-in git, no state store, no generated receipt, no hand-built dependency edges, scanner consumes trees without importing the fixture package

### Content-identity vectors

`ipfs_datasets_py/tests/fixtures/software_contracts/cid_vectors.json`

---

## 11. Focused tests

### Semantic index unit

`ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_index/` — acceptance, API, delta, explain, identity contract, import safety, invalidation, models, persistence, public pipeline, pytest/python analysis (including authority adversarial), scanner, snapshot (including authority closure/adversarial), symbol graph, watch.

### Semantic state unit

`ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_state/` — API, bindings, **capsules**, dependency seal, fixture contract, **freshness**, import safety, **invalidation**, MCP payload boundary, Merkle, models, **oracle**, payload schema, public pipeline, regressions, schema packaging, **selection/oracle acceptance**, **source**, **test selection**.

### Content / frontend / CLI

- `test_content_identity.py`, `test_python_frontend.py`, `test_repository_manifest.py`, `test_resolver.py`
- `tests/cli/test_semantic_index_cli.py`

Recommended focused command (from the semantic-state contract):

```bash
python3.12 -m pytest -q \
  ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_state \
  ipfs_datasets_py/tests/unit/logic/software_contracts/semantic_index \
  ipfs_datasets_py/tests/unit/logic/software_contracts/test_content_identity.py \
  ipfs_datasets_py/tests/unit/logic/software_contracts/test_python_frontend.py \
  ipfs_datasets_py/tests/unit/logic/software_contracts/test_repository_manifest.py \
  ipfs_datasets_py/tests/unit/logic/software_contracts/test_resolver.py \
  ipfs_datasets_py/tests/cli/test_semantic_index_cli.py
```

### Import safety

Under the standard opt-outs (`IPFS_DATASETS_AUTO_INSTALL=0`, `IPFS_DATASETS_PY_MINIMAL_IMPORTS=1`, …), package import installs nothing, opens no network, starts no process/thread, and writes no filesystem path. Proven by `test_import_safety.py` under both packages.

---

## 12. Handoff summary for Semantic Compression Governor

```text
ISI scan/diff/invalidate
        │
        ▼
build_semantic_state → SemanticStateBundle
        │
        ├─ open_semantic_state / view_semantic_state_bundle → SemanticStateView
        ├─ compile_semantic_capsule (SemanticCapsuleCompiler@1 functional)
        ├─ assess_capsule_freshness → admission / raw-source gate
        ├─ read_required_source when admission forbids substitute
        ├─ extend_semantic_invalidation → SemanticInvalidationPlan
        └─ select_tests_and_proofs → TestSelection
                 │
                 └─ compare_test_selection_oracle (pure metrics)
```

SCG **consumes** this surface; it must not reimplement the index, capsule compiler, content identity, or selection graph. Opaque and dynamic limitations stay first-class evidence for governor audit, expansion, and fallback decisions.

## Acceptance (SCG-002)

| Criterion | Status |
| --- | --- |
| Functional capsule compiler described precisely | Yes — `SemanticCapsuleCompiler@1` constants, functions, cycle-free deps, cold/incremental identity, non-class nature |
| Verified state-view boundaries described precisely | Yes — protocol, rehashing `get_block`, typed errors, root exclusions, storage-neutral assembly |
| Opaque/dynamic limitations remain visible | Yes — confidence taxonomy, limitation records, fallback reasons, known weak areas table |
| No second scanner/graph/compiler/CID proposed | Yes — inventory only; conflict policy recorded |
