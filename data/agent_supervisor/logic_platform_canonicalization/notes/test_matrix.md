# LPC-140 Hermetic Required Tests and Fail-Closed Cases

**Task:** LPC-140 — Hermetic required tests and fail-closed cases  
**Goal:** LPC-G140 (`LogicConformanceMatrix@1`)  
**Depends on:** LPC-044 (slice admission), LPC-052 (typed responses), LPC-080 (cache key), LPC-110 (supervisor client)  
**Interface:** `LogicConformanceMatrix@1`  
**Board validation:** `python scripts/validate_logic_platform_canonicalization_board.py --check-test-matrix`

## Purpose

This note freezes the **mandatory hermetic conformance matrix** for the logic
platform. Every row in §1 is **hermetic required**: no network, no live prover
install, no PATH-dependent toolchain, no sibling checkout, and no mock that is
silently promoted to a real-provider gate.

Hermetic required suites must **pass** on a clean validation environment
(`PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`, Python
`/usr/bin/python3.12`, private validation `HOME`). Unavailable providers are
**not** treated as passed. Mocks cannot satisfy real-provider gates (LPC-142).

LPC-141 (direct-vs-supervisor parity) and LPC-142 (real local provider smoke)
remain part of LPC-G140 but are **not** hermetic-required lanes; they are
recorded in §3 so the full goal inventory is visible without weakening this
gate.

## Lane vocabulary

| Lane | Meaning | CI disposition |
| --- | --- | --- |
| **hermetic required** | In-process / fixture-only; must pass; failure blocks the board | required |
| **parity (LPC-141)** | Direct API vs supervisor-mediated semantic identity | required after LPC-140 |
| **real-provider smoke (LPC-142)** | At least one already-supported local provider exercised | required when provider present; mocks labeled, never substitute |
| **opt-in / heavy** | Network, install, multi-solver portfolio, live toolchain | never silent-pass |

## 1. Hermetic required matrix (LPC-140 acceptance)

The eight acceptance buckets are **listed as hermetic required** and are owned
by the modules below. Each primary module is the gate for its bucket; companion
modules share the same fail-closed invariants.

| # | Bucket | Primary suite | Companion suites | Fail-closed focus |
| --- | --- | --- | --- | --- |
| 1 | **Pure import** | `ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py` | `test_logic_api_import_quiet.py`, `integration/test_import_quiet.py` | Import contracts/catalog/syntax/formalization/provider protocol/supervisor adapter without solvers, install, network, processes, file mutation, hardware probes, or env mutation |
| 2 | **Catalog** | `ipfs_datasets_py/tests/unit/logic/families/test_catalog_drift.py` | `families/test_canonical_catalog.py` | Alias misuse, namespace coercion, profile/family drift, illegal provider ops, executable-vs-declared inflation, authority ceiling overclaim, catalog-root mutation |
| 3 | **Syntax** | `ipfs_datasets_py/tests/unit/logic/syntax_core/test_contracts.py` | `syntax_core/test_artifacts_v2.py`, `syntax_core/test_ast.py`, `syntax_core/test_publication.py` | SourceDocument/Token/CST/Parse envelopes; incomplete or free-form syntax writes fail closed before formalization |
| 4 | **Admission** | `ipfs_datasets_py/tests/unit/logic/backends/test_unadmitted_slice_rejected.py` | `formalization/test_admission.py`, `test/api/test_logic_platform_admission.py` | Unadmitted `DomainLogicSlice@2` cannot seed `BackendRequest@2`; incomplete new writes rejected; supervisor ten-point receipt admission floor |
| 5 | **Translation** | `ipfs_datasets_py/tests/unit/logic/families/test_translations_v2.py` | `translations/test_protocol_targets.py`, `translations/test_state_temporal.py`, `translations/test_policy_modal.py` | TranslationContract@2 / composition receipts; loss-receipted edges; dialect mismatch and empty axes fail closed |
| 6 | **Protocol** | `ipfs_datasets_py/tests/unit/logic/backends/test_provider_protocol_v2.py` | `backends/test_provider_response_v2.py`, `backends/test_protocol_v1_adapter.py`, `backends/test_requests_v2.py` | Typed @2 requests/responses; positive finite bounds on executable ops; v1 free-form payloads cannot bypass `BackendRequest@2`; default authority untrusted |
| 7 | **Evidence** | `ipfs_datasets_py/tests/unit/logic/backends/test_evidence_v2.py` | `backends/test_success_is_not_proof.py`, `backends/test_artifacts_v2.py`, `common/test_canonical_cache_key.py` | Execution/replay claims require real lineage; metadata/mock cannot claim execution; success ≠ proof; candidate-as-kernel rejected |
| 8 | **Adversarial** | `ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py` | `test/api/test_logic_receipt_authority_boundary.py` | Empty/unknown/forged/stale/wrong-tree/wrong-property/wrong-assumption/wrong-bound/wrong-tool/cross-authority receipts rejected; simulated attestation never reports proof success |

### 1.1 Hermetic required command (single invocation)

```bash
python -m pytest -q \
  ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py \
  ipfs_datasets_py/tests/unit/logic/families/test_catalog_drift.py \
  ipfs_datasets_py/tests/unit/logic/syntax_core/test_contracts.py \
  ipfs_datasets_py/tests/unit/logic/backends/test_unadmitted_slice_rejected.py \
  ipfs_datasets_py/tests/unit/logic/formalization/test_admission.py \
  ipfs_datasets_py/tests/unit/logic/families/test_translations_v2.py \
  ipfs_datasets_py/tests/unit/logic/backends/test_provider_protocol_v2.py \
  ipfs_datasets_py/tests/unit/logic/backends/test_provider_response_v2.py \
  ipfs_datasets_py/tests/unit/logic/backends/test_protocol_v1_adapter.py \
  ipfs_datasets_py/tests/unit/logic/backends/test_evidence_v2.py \
  ipfs_datasets_py/tests/unit/logic/backends/test_success_is_not_proof.py \
  ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py \
  test/api/test_logic_platform_admission.py \
  test/api/test_logic_receipt_authority_boundary.py
```

Board gate for this note only:

```bash
python scripts/validate_logic_platform_canonicalization_board.py --check-test-matrix
```

### 1.2 Per-bucket validation (focused)

| Bucket | Focused validation |
| --- | --- |
| Pure import | `python -m pytest ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py -q` |
| Catalog | `python -m pytest ipfs_datasets_py/tests/unit/logic/families/test_catalog_drift.py -q` |
| Syntax | `python -m pytest ipfs_datasets_py/tests/unit/logic/syntax_core/test_contracts.py -q` |
| Admission | `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_unadmitted_slice_rejected.py ipfs_datasets_py/tests/unit/logic/formalization/test_admission.py test/api/test_logic_platform_admission.py -q` |
| Translation | `python -m pytest ipfs_datasets_py/tests/unit/logic/families/test_translations_v2.py -q` |
| Protocol | `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_provider_protocol_v2.py ipfs_datasets_py/tests/unit/logic/backends/test_provider_response_v2.py ipfs_datasets_py/tests/unit/logic/backends/test_protocol_v1_adapter.py -q` |
| Evidence | `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_evidence_v2.py ipfs_datasets_py/tests/unit/logic/backends/test_success_is_not_proof.py -q` |
| Adversarial | `python -m pytest ipfs_datasets_py/tests/unit/logic/test_verification_receipt_adversarial.py test/api/test_logic_receipt_authority_boundary.py -q` |

## 2. Fail-closed case catalog (normative)

These cases must remain representable and rejected (or non-promoting). They are
covered by the hermetic required suites above; this table is the human index.

### 2.1 Pure import

| Case | Required behavior |
| --- | --- |
| Import pure-data modules | No solver load (`z3`/`cvc5`/…), no pip/ensurepip, no `socket.connect`, no `subprocess.Popen`, no write opens / path mutation, no hardware probes, `os.environ` unchanged |
| Catalog import | Presence does not imply executability or production admission |
| Supervisor adapter import | Does not import `ipfs_datasets_py` at module load |

### 2.2 Catalog

| Case | Required behavior |
| --- | --- |
| Unknown alias / wrong-namespace dual-read | Raise; no silent coerce |
| Profile family / task mismatch | Snapshot cannot seal |
| Provider ops outside family set | Reject |
| Declaration-only executable claim | Reject |
| Advisory ceiling overclaim | Reject |
| Layer content drift | Catalog root changes; integrity floor stays hard-zero for presence→executability |

### 2.3 Syntax / formalization bindings

| Case | Required behavior |
| --- | --- |
| Incomplete SourceDocument / parse envelope | Construction or admission fails closed |
| Free-form routing metadata on new write | Rejected by formalization admission |
| Cross-namespace family on artifact | Rejected |
| Content / expression digest mismatch | Rejected |

### 2.4 Admission

| Case | Required behavior |
| --- | --- |
| Missing / rejected / unsupported `DomainLogicSlice@2` | Cannot build `LogicObligation@2` or `BackendRequest@2` |
| Executable protocol op without admitted slice | `ProtocolV2AdmissionError` / admission error before provider selection |
| Supervisor receipt missing any of the ten checks | `admitted=False`; cannot affect completion or merge |
| Partial check pass | Never promotes `may_affect_completion` / `may_affect_merge` |

Supervisor ten-point floor (LPC-111 / plan §8): structural validity, content
identity, source/tree/environment/policy binding, translation chain, evidence
kind, authority ceiling, required reconstruction, freshness, non-simulation,
policy admission.

### 2.5 Translation

| Case | Required behavior |
| --- | --- |
| Dialect mismatch | Fail closed with typed error |
| Empty axes | Rejected |
| Lossy edge without loss receipt | Not admitted as exact preservation |
| Authority composition | Result authority ≤ minimum of inputs |

### 2.6 Protocol

| Case | Required behavior |
| --- | --- |
| Executable op without positive finite bounds | `MissingExecutableBoundsError` |
| Free-form v1 payload as @2 request | Rejected / advisory retention only; no `BackendRequest@2` mint |
| Response missing axis fields | Construction fails closed |
| Default evidence authority | `advisory` (untrusted) until validation/reconstruction |
| Operation status `succeeded` | Never upgrades `evidence_authority` or `semantic_verdict` |

### 2.7 Evidence / receipts

| Case | Required behavior |
| --- | --- |
| Metadata-only record claims execution | Rejected |
| Mock record claims execution or replay | Rejected |
| Replay without executable source receipt | Rejected |
| `succeeded + unknown + advisory` | Representable; cannot pass kernel-required policy |
| Candidate evidence as kernel cache key | Rejected by `CanonicalProofCacheKey@1` |

### 2.8 Adversarial receipt dispatch

| Case | Required behavior |
| --- | --- |
| Empty / unknown schema / legacy permissive mapping | Rejected |
| Forged kernel authority or content id | Rejected |
| Wrong tree / property / assumption / bound / tool | Rejected |
| Cross-authority claim | Rejected |
| Stale digest or expiry | Rejected |
| Simulated or preparation-only attestation | Cannot report proof success |
| Non-trusted receipt attestation | Rejected |

## 3. Related LPC-G140 lanes (not hermetic-required for LPC-140)

| Lane | Task | Expected artifact / suite | Notes |
| --- | --- | --- | --- |
| Direct vs supervisor parity | LPC-141 | `test/api/test_direct_vs_supervisor_logic_parity.py`, `notes/direct_supervisor_parity.md` | Request, obligation, provider request, verdict, evidence, authority, boundedness, receipt identities agree |
| Real local provider smoke | LPC-142 | `notes/real_provider_smoke.md` | Uses an already-supported local provider; mocks labeled; never satisfy the real-provider gate |
| Channel parity | LPC-130 | `test_channel_parity.py`, `test_logic_channel_parity.py` | Names, schemas, status, authority, failure codes; install ≠ verify |

## 4. Ownership and non-goals

| Owner | Responsibility |
| --- | --- |
| This note (`test_matrix.md`) | Declare hermetic required buckets, suites, fail-closed cases, and board gate |
| Datasets unit suites under `ipfs_datasets_py/tests/unit/logic/` | Semantic, catalog, syntax, protocol, evidence, adversarial hermetic coverage |
| Accelerate `test/api/test_logic_platform_admission.py` | Supervisor receipt admission floor |
| Accelerate `test/api/test_logic_receipt_authority_boundary.py` | API-level adversarial receipt dispatch |
| LPC-141 / LPC-142 | Parity and real-provider smoke (separate notes) |

This task does **not**:

* Skip or `xfail` hermetic required suites.
* Treat unavailable providers as passed.
* Allow mocks to satisfy real-provider gates.
* Add a new prover, MCP++ profile, or second semantic authority.
* Use `continue-on-error` / `|| true` on required lanes (packaging CI is LPC-151).

## 5. Dependency map

```text
LPC-044 unadmitted-slice rejection ──┐
LPC-052 typed responses ─────────────┼──► LPC-140 hermetic matrix (this note)
LPC-080 canonical cache key ─────────┤
LPC-110 supervisor client ───────────┘
         │
         ├──► LPC-141 direct vs supervisor parity
         └──► LPC-142 real local provider smoke
```

Upstream notes cited by hermetic suites:

| Note | Task |
| --- | --- |
| `notes/pure_data_imports.md` | LPC-061 |
| `notes/catalog_drift_tests.md` | LPC-021 |
| `notes/slice_admission.md` | LPC-044 |
| `notes/new_write_path.md` | LPC-040 |
| `notes/provider_protocol_migration.md` | LPC-050 |
| `notes/provider_responses.md` | LPC-052 |
| `notes/no_success_implies_proof.md` | LPC-032 |
| `notes/receipt_admission.md` | LPC-111 |
| `notes/cache_key_contract.md` | LPC-080 |

## 6. Acceptance (LPC-140)

| Criterion | Status |
| --- | --- |
| Pure import tests listed as **hermetic required** | §1 row 1 |
| Catalog tests listed as **hermetic required** | §1 row 2 |
| Syntax tests listed as **hermetic required** | §1 row 3 |
| Admission tests listed as **hermetic required** | §1 row 4 |
| Translation tests listed as **hermetic required** | §1 row 5 |
| Protocol tests listed as **hermetic required** | §1 row 6 |
| Evidence tests listed as **hermetic required** | §1 row 7 |
| Adversarial tests listed as **hermetic required** | §1 row 8 |
| Hermetic required suites pass under focused pytest | §1.1 / §1.2 |
| Board check admits this note | `--check-test-matrix` |
| Mocks cannot satisfy real-provider gates | §3 LPC-142 / §4 non-goals |
| Fail-closed cases catalogued without weakening admission | §2 |

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/test_matrix.md` | This matrix (LPC-140 sole declared output) |
| Primary suites in §1 | Hermetic required executable coverage (owned by their originating tasks) |
