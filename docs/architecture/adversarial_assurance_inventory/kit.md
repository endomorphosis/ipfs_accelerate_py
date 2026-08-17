# AAE-003 Kit inventory: durability, CAS, receipt, campaign-history, and recovery

**Schema:** `aae/kit-inventory@1`  
**Interface:** `AAEKitInventory@1`  
**Evidence:** `aae/kit-inventory@1`  
**Task:** AAE-003  
**Goal:** AAE-G010  
**Board:** `adversarial-assurance-engine-v1`  
**Date:** 2026-08-13  
**Machine-readable companion:** [`kit.json`](kit.json)

## Acceptance result

**Criterion:** Inventory identifies exact immutable-block, idempotency, CAS, corruption, recovery, history, and proof-store contracts that the AAE domain layer must reuse.

| Contract family | Exact identity to reuse | Present on pin `c7e5feeb...`? |
| --- | --- | --- |
| **Immutable blocks** | `DurableCoordinationStore.put` / `get` / `get_bytes` / `has`; sealed pattern `DurableSemanticGovernorStore.put_artifact` | **Yes** |
| **Idempotency** | `operation_id` UNIQUE per namespace on `compare_and_swap_state_root`; governor artifact ops bind table | **Yes** |
| **CAS** | `DurableCoordinationStore.compare_and_swap_state_root` + `DurableStateRootAdapter.compare_and_swap_root`; transition schema `mcp++/coordination/state-root-transition@1` | **Yes** |
| **Corruption** | `ArtifactIntegrityError` fail-closed; corrupt SQLite preserve-and-rebuild; ambiguous successor rejection | **Yes** |
| **Recovery** | `DurableCoordinationStore.recover` + `DurableStateRootAdapter.recover_roots` → `StateRootRecoveryReport` | **Yes** (governor `recover_governor_store` protocol-only) |
| **History** | `root_transitions` / `current_state_root`; pattern `DurableAuditHistoryStore@1` | **Yes** (pattern); AAE campaign histories **not** present |
| **Proof store** | `IpfsKitProofCertificateStore` / `TestCertificateStoreTransport@1` (byte transport only) | **Yes** transport; sealer APIs **typed_unavailable** |

**AAE domain package still missing:** `ipfs_kit_py.adversarial_assurance_store` (planned AAE-034–AAE-038). It must compose the primitives above rather than invent another object store, WAL, CID system, receipt hierarchy, or proof engine.

## Repository pin

| Field | Value |
| --- | --- |
| Repository | `ipfs_kit_py` (`endomorphosis/ipfs_kit_py`) |
| Planning-bound revision | `c7e5feeb24582ab68c1f5ca626366b665a82ad61` |
| Observed revision | `c7e5feeb24582ab68c1f5ca626366b665a82ad61` |
| Observed subject | `SCG-020: Implement append-only audit, calibration, and benchmark histories` |
| Match | Yes |
| Controller at inventory | `8ab0af06221701fc168a00b8276fea4fe37e42ed` |

Plan section 2 pins this kit revision as the initialized gitlink with durable roots and governor store present.

Nested gitlinks observed from the controller tree:

| Nested repository | Commit |
| --- | --- |
| `ipfs_datasets_py` | `fbd1ba9f70803de157622bb20e22595ef09d606f` |
| `ipfs_kit_py` | `c7e5feeb24582ab68c1f5ca626366b665a82ad61` |
| `ipfs_accelerate_py/mcplusplus` | `dc3164653a48d059ae9812078359daeafb451c07` |

## What AAE must reuse

### 1. `DurableCoordinationStore` (immutable blocks + root CAS + recovery)

| Item | Value |
| --- | --- |
| Kind | Class |
| Module | `ipfs_kit_py.mcp_server.mcplusplus.coordination_storage` |
| Source | `ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py` (class ~L284) |
| Canonical import | `from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import DurableCoordinationStore` |
| Package `__init__` export | **Not** re-exported by `mcplusplus`; import `coordination_storage` directly |
| DB version | 2 |
| Default dir | `$MCPPLUSPLUS_COORDINATION_DIR` or `~/.local/share/ipfs_kit_py/mcppp_coordination` |

**Durability model**

- Local `blocks/` tree is **authoritative**.
- `coordination.sqlite3` is a **rebuildable acceleration index**.
- `put()` returns only after local block fsync and index commit.
- Optional backend replication happens after local durability and never substitutes for it.
- Artifact blocks are retained forever; retention/compaction only archives and prunes derived index rows via `mcp++/coordination-index-archive@1`.

**Public methods AAE must consume**

| Method | Role for AAE |
| --- | --- |
| `put` / `put_profile_g` | Immutable content-addressed write; optional `expected_cid` fail-closed |
| `get` / `get_bytes` / `has` | CID-verified read; backend may repair missing local blocks after integrity check |
| `current_state_root` (`current_root`) | Namespace head snapshot `{namespace, root_cid, revision, transition_cid}` |
| `compare_and_swap_state_root` (`compare_and_swap_root`) | Expected-revision CAS with `operation_id` idempotency |
| `state_roots` / `root_transitions` | Enumerate heads and immutable transition history (campaign-history basis) |
| `recover` | Verify blocks; optionally rebuild all derived indexes from immutable evidence |
| `root_recovery_metrics` | Session counters: `root_index_verifications`, `root_index_rebuild_mutations` |
| `compact_indexes` / `status` | Index retention and store diagnostics |

**CID / codec**

- Transport CIDs: CIDv1, sha2-256, codecs `dag-json` (0x0129) and `raw` (0x55), lower-case base32 multibase (`b...`).
- Canonical JSON: `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`.
- Helpers: `cid_for_bytes`, `cid_for_artifact`, `validate_transport_cid`.

### 2. `DurableStateRoots` protocol and `DurableStateRootAdapter`

| Item | Value |
| --- | --- |
| Protocol | `ipfs_kit_py.mcp_server.mcplusplus.state_root_contracts.DurableStateRoots` (~L303) |
| Adapter | `ipfs_kit_py.mcp_server.mcplusplus.state_root_adapter.DurableStateRootAdapter` (~L25) |
| Export | Lazy via `ipfs_kit_py.mcp_server.mcplusplus` |

**Protocol methods**

```text
put_verified(payload, *, expected_cid, replicate=True) -> ArtifactWriteResult
get_verified(cid) -> Mapping[str, Any]
current_root(namespace) -> StateRootSnapshot
compare_and_swap_root(namespace, *, expected_revision, expected_root_cid, new_root_cid, operation_id) -> StateRootCASResult
recover_roots() -> StateRootRecoveryReport
```

**Closed status vocabularies**

| Enum | Values |
| --- | --- |
| `RootUpdateStatus` | `updated`, `unchanged`, `conflict`, `unavailable`, `corrupt` |
| `ProviderStatus` | `available`, `unavailable`, `failed`, `corrupt`, `not_requested` |

Identity rule: **`ipfs_datasets_py` owns semantic identity**. Callers supply CIDs; kit verifies and does not mint replacements. The adapter requires **dag-json** semantic CIDs for roots; raw roots are rejected at the facade.

### 3. Semantic governor store pattern (thin typed layer to mirror)

Package: `ipfs_kit_py.semantic_governor_store`  
Interface: `SemanticGovernorStore@1`  
Schema: `ipfs-kit.semantic-governor-store.contracts@1`

This is the reusable **thin domain layer pattern** over `DurableCoordinationStore`. It is **not** the AAE package; AAE owns `adversarial_assurance_store` with assurance-specific kinds and namespaces. Implementations present on this pin:

| Implementation | Module | Covers |
| --- | --- | --- |
| `DurableSemanticGovernorStore@1` | `artifacts.py` | `put_artifact`, `get_verified_artifact` |
| `DurableAuditHistoryStore@1` | `history.py` | append-only audit/calibration/benchmark histories |
| `DurableCompressionPolicyRepository@1` | `policy.py` | policy head CAS + rollback |
| `DurablePromotionStateRepository@1` | `policy.py` | promotion CAS with separate authorization CID |
| `DurablePolicyCASRepositories` | `policy.py` | combined policy + promotion facade |

**Protocol-only on this pin (no production implementer):**

- `issue_receipt` → `ReceiptIssuanceResult` (existing envelope schemas only; `run_receipt` / `promotion_receipt`)
- `recover_governor_store` → `AuditRecoveryReport` (no `recovery.py` module yet)

**Closed governor artifact kinds:** `audit`, `calibration`, `benchmark`, `policy`, `policy_candidate`, `evaluation`, `promotion`, `run_receipt`, `promotion_receipt`, `history_manifest`.

**Namespace pattern:** `semantic-governor/<workspace>/{audit,calibration,benchmark,policy,promotion,receipts}`.

AAE-034+ must define **closed adversarial-assurance namespaces** over the same segment grammar and must not silently overload `semantic-governor/*`.

### 4. Receipt surfaces

| Surface | Path | Contract |
| --- | --- | --- |
| Governor receipt issuance shape | `semantic_governor_store.contracts.ReceiptIssuanceResult` | Bind neutral payloads to **existing** envelopes; reject `semantic-governor/receipt*` hierarchy invention |
| Supervisor receipt resolver | `mcp_server.agent_supervisor_receipts.AgentSupervisorReceiptResolver` | Method `agent_supervisor.receipts.read`; loads only CID-verified blocks from `DurableCoordinationStore`; no synthetic success |

### 5. Proof-store / certificate transport

| Item | Value |
| --- | --- |
| Class | `IpfsKitProofCertificateStore` |
| Interface | `TestCertificateStoreTransport@1` |
| Schema | `ipfs_kit_py/test-certificate-store-transport@1` |
| Source | `ipfs_kit_py/ipfs_kit_py/proof_certificate_store.py` (~L332) |
| Role | Optional exact-byte CID transport for proof-backed test certificates |
| Non-claims | Does **not** decide test reuse; does **not** seal; does **not** prove ZK execution |

Default construction has no filesystem or network side effects. CID profile is CIDv1 **raw** sha2-256 with multihash verification.

**Typed unavailable (not inventable substitutes):**

- `IncrementalProofSealer`, `FullCheckpointSeal`, `DeltaSeal`
- `create_full_checkpoint`, `publish_full_checkpoint`, `build_delta_seal`, `publish_delta_seal`

Seal publication remains release-gated; AAE-004 inventories the MCP++/sealer boundary separately.

### 6. Missing AAE domain layer

| Item | Value |
| --- | --- |
| Package | `ipfs_kit_py.adversarial_assurance_store` |
| Present | **No** |
| Planned modules | `contracts`, `artifacts`, `campaigns`, `merkle`, `policy`, `recovery`, package export |
| Planned tasks | AAE-034 … AAE-038 |

Must back on `DurableCoordinationStore` + `DurableStateRootAdapter`, mirror governor thin-typed patterns, and consume datasets assurance schemas for payload identity.

## Contracts AAE must consume (normative summary)

### Immutable blocks

1. Serialize payload to canonical JSON bytes (or raw certificate bytes for the transport only).
2. Compute CID (caller / datasets authority for semantic payloads).
3. `put` / `put_verified` / `put_artifact` with `expected_cid`.
4. Local fsync block first; index second; optional replicate third.
5. Same bytes → idempotent; different bytes same CID → integrity error.

### Operation ID and expected-version CAS

| Rule | Behavior |
| --- | --- |
| `operation_id` grammar | `[a-z0-9](?:[a-z0-9._:-]{0,126}[a-z0-9])?`, length 1–128 |
| Uniqueness | `UNIQUE(namespace, operation_id)` for root transitions; separate bind for governor artifacts |
| Exact replay | `status=unchanged`, `reason_code=idempotent_replay` |
| Changed reuse | `status=conflict`, `reason_code=operation_id_reused` |
| Predecessor form | rev/gen 0 ⇒ expected head CID is `None`; > 0 ⇒ head CID required |
| Stale expectation | `status=conflict`, `reason_code=stale_expectation`, no mutation |
| Success | `status=updated`, after revision/generation = before + 1, durable transition block + index in one transaction |
| Promotion | Requires distinct `candidate_cid` and `authorization_cid`; self-promotion forbidden |

Transition wire schema: **`mcp++/coordination/state-root-transition@1`**.

Interruption points (crash-injection seam):  
`before_transaction`, `after_expectation_verification`, `after_transition_block_fsync`, `after_transition_indexing`, `before_sqlite_commit`, `after_sqlite_commit`.

### Corruption

- Local/backend CID mismatch → `ArtifactIntegrityError`; recovery fails closed.
- Non-canonical JSON → integrity failure.
- Corrupt SQLite → preserve as `coordination.sqlite3.corrupt-<ms>`, rebuild from blocks.
- Tampered root index → chain verification against immutable blocks refuses mutation.
- Ambiguous successors → reconstruction rejects without choosing a winner.
- Raw transition evidence → rejected before rebuild mutates indexes.
- Semantic facade → rejects raw CIDs for policy/history/root heads.

### Recovery

```text
DurableCoordinationStore.recover(rebuild=True|False)
  -> {verified_blocks, rebuilt, errors}

DurableStateRootAdapter.recover_roots()
  -> StateRootRecoveryReport
```

Startup rules: preserve corrupt SQLite; verify blocks on reopen; rebuild indexes when empty or inconsistent. Healthy reopen must leave `root_index_rebuild_mutations` at zero when indexes already match.

Governor `AuditRecoveryReport` / AAE `AssuranceRecoveryReport@1` project domain heads after the same primitive recovery; they must not invent a second rebuild engine.

### History (campaign-history basis)

Primitive: `current_state_root`, `root_transitions`, CAS to a successor head.

Governor pattern (`DurableAuditHistoryStore@1`):

- Roles: `audit`, `calibration`, `benchmark`
- Each append CAS-publishes a deterministic **history-manifest** head referencing an immutable entry CID
- Entry must already be durable; unavailable → `unavailable` / `entry_unavailable`; corrupt → `corrupt` / `entry_integrity_failure`
- Public projections expose portable CIDs/generations only (no local paths / private markers)
- Schemas: `ipfs-kit.semantic-governor-store.history-manifest@1`, `…history-public@1`, `…history-private@1`

Profile G claim/lease/health indexes are **not** campaign history.

### Proof-store

- Reuse `IpfsKitProofCertificateStore` only for optional certificate byte movement by verified CID.
- Do not treat certificate transport success as reuse authorization or seal publication.
- Missing sealer APIs remain `typed_unavailable`.

## What AAE must not reimplement

- Second object store, WAL, or journal engine
- New CID / content-identity authority
- New receipt envelope hierarchy or cryptography
- New daemon or provider discovery at import time
- Local `IncrementalProofSealer` or commitment masquerading as a seal
- Silent overwrite CAS without expected revision/generation and `operation_id`

## Focused tests (static inventory)

| Path | Approx. `test_*` defs | Covers |
| --- | --- | --- |
| `tests/test_coordination_storage.py` | 7 | blocks, recovery, retention, corruption |
| `tests/test_semantic_state_root_*.py` | 47 | contracts, CAS, adapter, recovery, acceptance, performance, import safety |
| `tests/semantic_governor_store/*` | 93 | contracts, artifacts, history, policy CAS |
| `tests/test_proof_certificate_store.py` | 9 | certificate transport |
| `tests/test_agent_supervisor_receipts.py` | — | verified receipt resolution |

Counts are static enumerations on pin `c7e5feeb...`, not a green acceptance run and not AAE campaign readiness.

Hermetic Profile G vectors: `ipfs_kit_py/tests/fixtures/mcp_plus_plus/profile_g_artifacts_valid.json`.

## Documentation

- `ipfs_kit_py/docs/coordination-storage.md`
- `ipfs_kit_py/docs/durable_state_roots.md`
- Plan ownership: `docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md` sections 3–4

## AAE consumption checklist

**Do**

- Compose `DurableCoordinationStore` for all durable AAE artifacts.
- Use `DurableStateRootAdapter` for semantic dag-json campaign/policy/promotion roots.
- Mirror `semantic_governor_store` thin-typed patterns under `adversarial_assurance_store`.
- Require caller-supplied verified CIDs and `operation_id`s.
- Treat CAS `conflict` / `unchanged` / `corrupt` / `unavailable` as closed outcomes.
- Recover only from verified immutable blocks.
- Bind receipts to existing envelopes; resolve only through verified reads.
- Use the certificate store only as optional byte transport.

**Do not**

- Open a second block store or WAL.
- Trust a supplied CID without recomputation/verification.
- Delete history to roll back; authorize another CAS to a prior CID.
- Treat Profile G claim/lease indexes as campaign history.
- Invent sealer APIs while typed unavailable.
- Infer `adversarial_assurance_store` APIs before AAE-034+.
