# SCG-003 Kit inventory: immutable blocks, history, recovery, and root CAS

**Schema:** `scg/kit-inventory@1`  
**Task:** SCG-003  
**Goal:** SCG-G010  
**Date:** 2026-08-13  
**Machine-readable companion:** [`kit.json`](kit.json)

## Acceptance result

**Criterion:** Inventory identifies the thin governor domain layer still missing and the primitive that must back it.

| Finding | Exact identity | Present in pinned kit tree? |
| --- | --- | --- |
| **Thin governor domain layer still missing** | `ipfs_kit_py.semantic_governor_store` / protocol `SemanticGovernorStore` (planned modules: contracts, artifacts, history, policy, recovery; tasks SCG-010, SCG-019–SCG-022) | **No** — package path does not exist |
| **Primitive that must back it** | `DurableCoordinationStore` as immutable-block + rebuildable-index authority, with typed root facade `DurableStateRootAdapter` implementing protocol `DurableStateRoots` | **Yes** — complete and tested at kit `df2f9cc0...` |

The governor must not invent another object store, WAL, CID system, or daemon. It must compose the existing coordination/root CAS primitive with thin typed manifests for audit cases, calibration/benchmark history, policy versions, promotion heads, receipts, and recovery.

## Repository pin

| Field | Value |
| --- | --- |
| Repository | `ipfs_kit_py` |
| Planning-bound revision | `df2f9cc092456329de9724c45a50c54b410875d1` |
| Observed revision | `df2f9cc092456329de9724c45a50c54b410875d1` |
| Observed subject | `fix(ksr): vendor Profile G vectors for hermetic seal tests` |
| Match | Yes |
| Controller at inventory | `a0b825d8cfa384c284d0e77fa5341571c40adfa8` |

Plan section 2 records this pin as the hermetic-vector repair used for the focused kit baseline.

## What exists (reusable storage/CAS)

### 1. `DurableCoordinationStore`

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
- `coordination.sqlite3` is a **rebuildable acceleration index** (claims, leases, daemon health, state roots, transitions).
- `put()` returns only after local block fsync and index commit.
- Optional backend replication (Helia/`store_block`, Kubo-style clients) happens after local durability and never substitutes for it.
- Artifact blocks are retained forever; retention/compaction only archives and prunes derived index rows via `mcp++/coordination-index-archive@1`.

**Public methods (storage and CAS surface)**

| Method | Role for SCG |
| --- | --- |
| `put` / `put_profile_g` | Immutable content-addressed write; optional `expected_cid` fail-closed |
| `get` / `get_bytes` / `has` | CID-verified read; backend may repair missing local blocks after integrity check |
| `current_state_root` (`current_root`) | Namespace head snapshot `{namespace, root_cid, revision, transition_cid}` |
| `compare_and_swap_state_root` (`compare_and_swap_root`) | Expected-revision CAS with `operation_id` idempotency |
| `state_roots` / `root_transitions` | Enumerate heads and immutable transition history |
| `recover` | Verify blocks; optionally rebuild all derived indexes from immutable evidence |
| `root_recovery_metrics` | Session counters: `root_index_verifications`, `root_index_rebuild_mutations` |
| `compact_indexes` / `status` | Index retention and store diagnostics |
| `claims` / `active_lease` / `daemon_health` / `record_daemon_health` | Profile G coordination indexes (not governor audit history) |

**CID / codec**

- Transport CIDs: CIDv1, sha2-256, codecs `dag-json` (0x0129) and `raw` (0x55), lower-case base32 multibase (`b...`).
- Canonical JSON: `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`.
- Helpers: `cid_for_bytes`, `cid_for_artifact`, `validate_transport_cid`.

### 2. `DurableStateRoots` (protocol) and contracts

| Item | Value |
| --- | --- |
| Kind | `typing.Protocol` |
| Module | `ipfs_kit_py.mcp_server.mcplusplus.state_root_contracts` |
| Source | `state_root_contracts.py` (~L303) |
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

**Closed value types:** `StateRootSnapshot`, `StateRootCASResult`, `ArtifactWriteResult`, `StateRootRecoveryReport` with deterministic `to_dict` / `from_dict` and fail-closed field validation.

Identity rule recorded in kit docs: **`ipfs_datasets_py` owns semantic identity**. Callers supply CIDs; kit verifies and does not mint replacements.

### 3. `DurableStateRootAdapter`

| Item | Value |
| --- | --- |
| Kind | Class implementing `DurableStateRoots` |
| Module | `ipfs_kit_py.mcp_server.mcplusplus.state_root_adapter` |
| Source | `state_root_adapter.py` (~L25) |
| Construction | `DurableStateRootAdapter(store: DurableCoordinationStore)` |
| Export | Lazy via `ipfs_kit_py.mcp_server.mcplusplus` |

Narrow facade over an injected store:

- Owns neither block directory nor database.
- Requires **dag-json** semantic CIDs (`validate_semantic_dag_json_cid`); raw roots are rejected at this facade while the generic store may still hold raw coordination artifacts.
- Projects optional replication into closed provider outcomes without hiding local durability.
- `recover_roots()` maps raw/corrupt reconstructed roots to closed `StateRootRecoveryReport` errors rather than leaking partial roots.

## Contracts the governor must consume

### Immutable blocks

1. Serialize payload to canonical JSON bytes.
2. Compute CID (caller / datasets authority for semantic payloads).
3. `put` / `put_verified` with `expected_cid`.
4. Local fsync block first; index second; optional replicate third.
5. Same bytes → idempotent; different bytes same CID → `ArtifactIntegrityError`.

### Namespace grammar

- Max 255 characters, slash-separated segments.
- Segment pattern: `[a-z0-9](?:[a-z0-9._-]{0,61}[a-z0-9])?`.
- Normalized: no whitespace edges, no empty segments.

Governor namespace layout (audit / policy / promotion heads) is **not** frozen by this primitive; **SCG-010** must define closed governor namespaces over this grammar.

### Operation ID and expected-version CAS

| Rule | Behavior |
| --- | --- |
| `operation_id` grammar | `[a-z0-9](?:[a-z0-9._:-]{0,126}[a-z0-9])?`, length 1–128 |
| Uniqueness | `UNIQUE(namespace, operation_id)` |
| Exact replay | `status=unchanged`, `reason_code=idempotent_replay` |
| Changed reuse | `status=conflict`, `reason_code=operation_id_reused` |
| Predecessor form | rev 0 ⇒ `expected_root_cid is None`; rev > 0 ⇒ root CID required |
| Stale expectation | `status=conflict`, `reason_code=stale_expectation`, no mutation |
| Success | `status=updated`, `after.revision = before.revision + 1`, durable transition block + index in one transaction |
| Writer fence | `BEGIN IMMEDIATE` + process lock; concurrent writers → one winner, typed loser |

**Immutable transition schema:** `mcp++/coordination/state-root-transition@1`  
Closed fields: `schema`, `namespace`, `operation_id`, `expected_root_cid`, `expected_revision`, `new_root_cid`, `new_revision`, `created_at_ms`.

**Crash-injection boundaries** (test seam `ROOT_CAS_INTERRUPTION_POINTS`):

```text
before_transaction
after_expectation_verification
after_transition_block_fsync
after_transition_indexing
before_sqlite_commit
after_sqlite_commit
```

Every boundary recovers to the prior root or the sole durable successor; recovery never invents promotion or completion.

### Corruption and fail-closed behavior

| Condition | Outcome |
| --- | --- |
| Local/backend CID mismatch | `ArtifactIntegrityError` / adapter `ProviderStatus.CORRUPT` |
| Non-canonical JSON | Integrity error; recovery aborts |
| Unreadable SQLite | Preserved as `*.corrupt-<ms>`, rebuilt from blocks |
| Tampered root index vs chain | Live root/CAS refuse mutation |
| Raw transition as root evidence | Rejected before rebuild mutates indexes |
| Ambiguous successors | Reconstruction fails without choosing a winner |
| Semantic facade raw root CID | Rejected; generic store raw support remains separate |

### Recovery and metrics

- Startup verifies blocks when any exist; rebuilds only when the index is empty or root indexes disagree with immutable transitions.
- Healthy reopen property: verify without `root_index_rebuild_mutations` when indexes already match.
- Orphan transition evidence (fsync before crash) reconstructs the durable successor on rebuild.
- Metrics are structural session counters, not wall-clock and not governor domain telemetry.

### What is *not* governor history

Profile G claim/lease/daemon-health indexes and coordination archives are MCP++ scheduling coordination. They prove the store can index typed artifacts and compact derived rows. They are **not** the governor’s audit, calibration, or policy history; those require the missing domain layer.

## Thin governor domain layer still missing

Plan ownership for `ipfs_kit_py`:

> Owns thin typed storage manifests over `DurableCoordinationStore`: immutable audit cases, calibration and benchmark history, policy versions, promotion state, receipts, recovery, and compare-and-swap publication.

Observed tree:

```text
ipfs_kit_py/ipfs_kit_py/semantic_governor_store/   # ABSENT
```

Planned but not present interfaces (from SCG-010 / SCG-019–022):

| Interface | Planned role |
| --- | --- |
| `SemanticGovernorStore` | Closed protocol: immutable artifacts, histories, policy/promotion heads, recovery, receipt envelope binding |
| `GovernorArtifactKind` | Closed artifact kind taxonomy for governor payloads |
| `PolicyVersionSnapshot` / `PolicyCASResult` | Versioned policy head projection and CAS result |
| `AuditRecoveryReport` | Governor-domain recovery evidence |
| `DurableSemanticGovernorStore` | Implementation composing `DurableCoordinationStore` |

**Backing primitive (must not be reimplemented):**

1. **Primary:** `DurableCoordinationStore` — immutable blocks, indexes, operation-id CAS, recovery.
2. **Typed root facade:** `DurableStateRootAdapter` / `DurableStateRoots` — dag-json semantic root publication and recovery projection.

**Ownership split to preserve**

| Owner | Responsibility |
| --- | --- |
| `ipfs_datasets_py` | Neutral receipt payload schemas and content identity |
| `ipfs_kit_py` | Durable issuance, storage/history, expected-version CAS, envelope binding |
| `ipfs_accelerate_py` | Orchestration consuming the store; never a second store |

**Follow-on tasks (not part of this inventory)**

- SCG-010 — freeze `SemanticGovernorStore` contracts  
- SCG-019 — immutable audit artifact storage  
- SCG-020 — append-only histories  
- SCG-021 — policy/promotion CAS repositories  
- SCG-022 — corruption/interruption/concurrency/privacy/recovery proofs  

## Tests and hermetic vectors

Focused collect-only on kit storage/root surfaces (current tree):

| File | Collected |
| --- | --- |
| `tests/test_coordination_storage.py` | 7 |
| `tests/test_semantic_state_root_contracts.py` | 26 |
| `tests/test_semantic_state_root_cas.py` | 13 |
| `tests/test_semantic_state_root_adapter.py` | 9 |
| `tests/test_semantic_state_root_recovery.py` | 17 |
| `tests/test_semantic_state_root_acceptance.py` | 12 |
| `tests/test_semantic_state_root_performance.py` | 2 |
| `tests/test_semantic_state_root_import_safety.py` | 1 |
| **Total** | **87** |

Plan baseline note: 72 focused tests recorded at planning after the hermetic-vector pin. Collect-only now reports 87 items on the same files (expanded contract/recovery coverage). This inventory does not claim a new green gate beyond collectability and source inspection.

**Hermetic Profile G vectors**

- Path: `ipfs_kit_py/tests/fixtures/mcp_plus_plus/profile_g_artifacts_valid.json`
- Cases: 14 kinds (Goal through TaskReceipt)
- Resolution: env override → vendored fixture → monorepo sibling MCP++ path

## Documentation

| Doc | Path |
| --- | --- |
| Coordination storage | `ipfs_kit_py/docs/coordination-storage.md` |
| Durable state roots | `ipfs_kit_py/docs/durable_state_roots.md` |
| SCG plan ownership | `docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md` §4 |

## Dependency seal selectors already recognized

`config/semantic_state_dependencies.seal.json` contract `DurableStateRoots@1` selects:

```text
DurableCoordinationStore.put
DurableCoordinationStore.get
DurableCoordinationStore.get_bytes
DurableCoordinationStore.has
DurableCoordinationStore.current_state_root
DurableCoordinationStore.compare_and_swap_state_root
DurableCoordinationStore.recover
```

## Consumption guidance for later SCG tasks

**Do**

- Compose `DurableCoordinationStore` for every durable governor artifact.
- Use `DurableStateRootAdapter` for semantic dag-json policy/promotion heads.
- Require caller-supplied verified CIDs, expected generation/root, and operation IDs.
- Treat `conflict` / `unchanged` / `corrupt` / `unavailable` as closed outcomes.
- Recover only from verified immutable blocks; roll back via authorized CAS, not history deletion.

**Do not**

- Open a second block store, WAL, or CID implementation.
- Trust a supplied CID without verification against canonical bytes.
- Treat Profile G claim/lease indexes as governor audit/calibration history.
- Infer `SemanticGovernorStore` method signatures from prose alone before SCG-010 freezes them.
- Start daemons or perform provider discovery at import time.

## Summary

Kit already provides a production durable coordination and root-CAS primitive: immutable CID blocks, rebuildable indexes, operation-id idempotent expected-revision CAS, concurrency fencing, interruption recovery, and a typed semantic root adapter. The **thin governor domain layer** (`semantic_governor_store` / `SemanticGovernorStore`) that should publish audit, history, policy, and promotion artifacts **over that primitive is still missing**. All later kit governor storage work must back onto **`DurableCoordinationStore`** (and **`DurableStateRootAdapter`** for semantic root heads), not a new storage engine.
