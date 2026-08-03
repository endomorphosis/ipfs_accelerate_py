# Distributed runtime: IPFS, content identity, and P2P execution

**Status:** Current  
**Audience:** Integrators, operators, security reviewers, and implementation
agents who need accurate backend roles, CID semantics, and optional P2P
execution boundaries  
**Scope:** IPFS/IPLD backend selection and roles; real multiformats CIDv1 versus
synthetic cache keys; verified put/get/admission; CAR/pinning/replication
capability gates; separation of immutable content from mutable coordination;
P2P TaskQueue discovery, scheduling, trust, and fallback  
**Non-goals:** Sibling-repository ownership maps (see planned
`INTEGRATION_BOUNDARIES.md` / DOC-010); operator P2P install journeys (see
planned `docs/guides/p2p/`); ADR formalization of single-writer DuckDB vs
immutable replicas (planned ADR-0005 / DOC-019); MCP transport policy as a
whole (see planned `MCP_RUNTIME.md`); inventing new public APIs  
**Last verified:** `d71cc2df31ec89716d30b153c989a8bbb557c0b2` (2026-08-03);
paths and symbols checked against `ipfs_backend_router.py`,
`multiformats_identity.py`, `verified_ipld_backend.py`, `p2p_tasks/`, and
related workflow modules on the checked-out tree  

This guide is the maintained **DistributedRuntime@1** architecture narrative.
Interfaces called out by the documentation program:
**BackendRole@1**, **ContentIdentityProfile@1**, **DegradationReceipt@1**, and
**P2PTaskFlow@1**.

---

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Backend roles and selection | `ipfs_accelerate_py/ipfs_backend_router.py` — `BackendRole`, `select_backend`, `BackendSelectionReceipt`, `describe_backend_capabilities` | Preferred kit → HF cache → Kubo; never silent degradation |
| IPFS backend protocol | `IPFSBackend` in `ipfs_backend_router.py` | `add_bytes` / `cat` / `pin` / `block_put` / `dag_export` |
| Synthetic cache keys | `HuggingFaceCacheBackend._generate_cid` | Emits `bafy…` strings that are **not** multiformats CIDs |
| Lightweight multiformats helper | `ipfs_accelerate_py/ipfs_multiformats.py` — `ipfs_multiformats_py` | CIDv1 raw/sha2-256 for file/bytes helpers |
| Frozen content-identity profile | `ipfs_accelerate_py/agent_supervisor/multiformats_identity.py` | CIDv1 / base32 / sha2-256 / raw\|dag-json; `IdentityLink` |
| Fail-closed IPLD adapter | `ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py` — `VerifiedIPLDBackend` | Rehash admission; refuses cache role by default |
| API content-addressed cache | `ipfs_accelerate_py/common/base_cache.py` — `BaseAPICache` | Optional multiformats CID keys; IPFS payload pointer when configured |
| P2P TaskQueue package | `ipfs_accelerate_py/p2p_tasks/` | libp2p service, client, trust, orchestrator |
| TaskQueue protocol | `ipfs_accelerate_py/p2p_tasks/protocol.py` — `PROTOCOL_V1` | `/ipfs-datasets/task-queue/1.0.0`; MCP++ preference |
| Peer trust tiers | `ipfs_accelerate_py/p2p_tasks/peer_trust.py` — `PeerTrustLevel` | TRUSTED / ELEVATED / BASELINE |
| Mutable queue state | `ipfs_accelerate_py/p2p_tasks/task_queue.py` — `TaskQueue` | DuckDB-backed leases, heartbeats, claims |
| Workflow discovery | `ipfs_accelerate_py/p2p_workflow_discovery.py` | Tags workflows for P2P vs GitHub |
| Workflow scheduling | `ipfs_accelerate_py/p2p_workflow_scheduler.py` — `P2PWorkflowScheduler`, `MerkleClock` | Ownership heuristics; not coordination-manifest authority |
| Mutable lease coordination (supervisor) | `ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py` | DuckDB claims/fences; distinct from IPFS replicas |

Related historical / feature material (not Current authority for this guide):
`IPFS_KIT_ARCHITECTURE.md`, `IPFS_KIT_INTEGRATION*.md`,
`CACHE_IMPLEMENTATION_SUMMARY.md`, feature IPFS guides under `docs/features/`.

---

## 1. Context and component map

Distributed capabilities sit **beside** local inference, not inside it. Import
success never implies an IPFS node, libp2p mesh, or conformant CID backend.

```text
  CLI / Python API / MCP / supervisor entrypoints
                    |
        +-----------+-----------+
        |                       |
  Inference / model plane   Distributed plane (this guide)
        |                       |
        |           +-----------+-----------+
        |           |                       |
        |    Storage / identity        P2P task execution
        |           |                       |
        |    Backend router            p2p_tasks TaskQueue
        |    VerifiedIPLDBackend       workflow discovery/scheduler
        |    multiformats_identity     peer trust / libp2p
        |           |
        |    +------+------+
        |    |             |
        |  Immutable     Mutable
        |  content       coordination
        |  (blocks,      (DuckDB leases,
        |   CAR, pins,    queue claims,
        |   CIDv1)        heartbeats)
        v
  Optional adapters: ipfs_kit_py, Kubo CLI, HF cache, memory (tests)
```

| Box | Live package / module | Plane |
| --- | --- | --- |
| Backend router | `ipfs_backend_router` | Data / storage |
| Content identity | `agent_supervisor.multiformats_identity` | Identity |
| Verified IPLD | `agent_supervisor.entrypoints.verified_ipld_backend` | Assurance + storage |
| API caches | `common.base_cache` and siblings | Cache (not authority) |
| TaskQueue P2P | `p2p_tasks` | Distributed execution |
| Workflow P2P | `p2p_workflow_*` | Optional orchestration |
| Lease DB | `agent_supervisor.merge.lease_coordination`, `p2p_tasks.task_queue` | Mutable coordination |

---

## 2. Backend roles and selection (BackendRole@1)

### 2.1 Closed role vocabulary

`BackendRole` classifies adapters structurally. Roles are **not** a ranking of
trust by themselves; `conformant_cid` and capability flags decide what may enter
coordination manifests.

| Role value | Adapter examples | `conformant_cid` | CAR | Typical use |
| --- | --- | --- | --- | --- |
| `ipfs_kit_py` | `IPFSKitBackend` | expected true (still re-verify) | no (adapter fails closed on `dag_export`) | Preferred distributed storage |
| `kubo` | `KuboCLIBackend` | true (still re-verify) | yes when CLI supports `dag export` | Local Kubo CLI |
| `cache` | `HuggingFaceCacheBackend` | **false** | no | Local/HF cache transport only |
| `memory` | `InMemoryConformantBackend` | true | yes (test envelope) | Hermetic tests / local verification |
| `unknown` | custom / unregistered | false until verified | not assumed | Fail closed for assurance paths |

### 2.2 Selection order and environment

When no explicit backend is injected, `select_backend` prefers:

1. `ipfs_kit_py` (unless disabled via `IPFS_KIT_DISABLE` / `ENABLE_IPFS_KIT`)
2. HuggingFace cache (if `ENABLE_HF_CACHE` allows)
3. Kubo CLI (`KUBO_CMD`, default `ipfs`)

| Variable | Effect |
| --- | --- |
| `IPFS_BACKEND` | Force a registered provider name |
| `ENABLE_IPFS_KIT` | Preferred kit enable (default true) |
| `IPFS_KIT_DISABLE` | Completely disable kit path |
| `ENABLE_HF_CACHE` | Allow HF cache role (default true) |
| `KUBO_CMD` | Kubo CLI binary name/path |
| `IPFS_ROUTER_CACHE` | Process-local selection cache (`0` disables) |

Selection always produces a **`BackendSelectionReceipt`**
(DegradationReceipt@1 family): selected name/role, preferred availability,
`degraded` flag, `degradation_reasons`, capability matrix, and candidate order.
Cache selection is always recorded as degraded for coordination purposes.

### 2.3 Capability matrix (static, fail-closed defaults)

`describe_backend_capabilities` does **not** probe the network. Unknown and
cache roles claim no CAR and no conformant CID. Callers that need assurance
must use `VerifiedIPLDBackend` rather than trusting role labels alone.

---

## 3. Content identity profile (ContentIdentityProfile@1)

### 3.1 Frozen supervisor multiformats profile

Coordination and verified IPLD paths use a single frozen profile from
`multiformats_identity`:

| Dimension | Required value |
| --- | --- |
| CID version | **CIDv1** (`version = 1`) |
| Multibase | lowercase **base32** |
| Multihash | **sha2-256**, full 32-byte digest |
| Codecs (coordination) | **`raw`** (exact bytes) or **`dag-json`** (canonical structured objects) |

`validate_cid` rejects empty, non-string, non-lowercase, wrong version/base/hash,
and codecs outside the allowed set. Temporal fields must not participate in
identity digests for DAG-JSON identity material.

### 3.2 Real CID versus synthetic cache key

**Critical invariant:** a string that *looks* like a CID is not a verified CID.

| Identifier kind | How it is produced | Admissible as coordination / IPLD CID? |
| --- | --- | --- |
| Multiformats CIDv1 | `cid_for_bytes` / `cid_for_dag_json` / backend that rehashes equal | Yes, after `validate_cid` + rehash when required |
| Supervisor `content_identity` | Already a profile CID; linked via `IdentityLink` | Yes when it validates under the profile |
| `runtime-artifact:sha256:…` / `sha256:…` | Local/runtime CAS digests | **No** as epoch authority; bridge with `IdentityLink` only |
| Synthetic HF / cache token | `HuggingFaceCacheBackend._generate_cid`: `bafy` + first 56 hex chars of sha256 | **Never** — not multiformats CIDv1 |

The HF cache intentionally preserves a historical `bafy…` **shape** for
compatibility. That shape is a **synthetic cache key**, not a multiformats
object. `VerifiedIPLDBackend` with `require_conformant=True` (default) refuses
cache/non-conformant backends before put/get, and `admit_cid` /
`validate_cid` reject non-profile strings even if they begin with `bafy`.

Do not present synthetic tokens as “CIDs” in logs, receipts, or docs without
labeling them cache keys. Do not pin synthetic tokens into coordination
manifests, lease grants, or immutable epoch heads.

### 3.3 Dual identity: `IdentityLink`

`IdentityLink` maps a persisted **local_id** to a multiformats **cid** without
silently rewriting either side. Kinds include `content_identity`,
`runtime_artifact`, `payload_digest`, `raw_bytes`, and `dag_json`. MCP++ /
runtime-CAS hashes are bridged (`link_runtime_artifact`,
`link_payload_digest`) and remain non-authoritative for coordination epochs.

### 3.4 API cache keys

`BaseAPICache` may compute multiformats CIDv1 raw/sha2-256 keys when the
`multiformats` library is available, otherwise a hash fallback. Optional remote
paths can store bulk payloads on IPFS and keep a CID pointer. Cache hits are
**never** proof, completion authority, or lease authority—only acceleration.

---

## 4. Verified IPLD, pinning, CAR, and replication

### 4.1 `VerifiedIPLDBackend` assurance contract

For coordination-grade operations the adapter:

1. Refuses non-conformant / cache roles when `require_conformant` is true.
2. Computes the expected CIDv1 locally (`expected_cid_for_bytes` /
   `expected_cid_for_dag_json`).
3. Stores via `block_put` (preferred) or `add_bytes`.
4. Requires the backend-returned identifier to admit as the **same** CID.
5. Re-fetches bytes and rehashes before emitting a put receipt.
6. On get: admit CID → fetch → rehash → optional DAG-JSON canonicality check.
7. `admit_for_manifest` records a coordination admission receipt only after
   strict checks (and optional payload rehash).

Unsupported codec, CID mismatch, empty CAR export, or cache role →
`VerifiedIPLDError` (**fail closed**).

### 4.2 Pinning

- Conformant backends may pin after verified put; pin failures after storage
  are best-effort if re-fetch still verifies.
- Cache “pin” is local metadata only and does not create network durability.
- Pinning does not upgrade a synthetic key into a CID.

### 4.3 CAR export

CAR is **capability-gated**. Cache and kit adapters fail closed on
`dag_export`. Kubo and memory may claim support; `export_car` still admits the
CID and rehashes fetchable bytes before returning CAR bytes. Absence of CAR
support is degradation for *export* workflows, not silent success.

### 4.4 Immutable content versus mutable coordination

| Concern | Store / medium | Authority |
| --- | --- | --- |
| Content blocks, evidence bytes, CAR snapshots | IPFS/IPLD / verified put receipts | Content-addressed CIDv1 after rehash |
| Active leases, claims, heartbeats, queue status | DuckDB (`TaskQueue`, lease coordination) | Single-writer / fenced mutable state |
| Workflow ownership heuristics | Merkle clock / scheduler state | Local P2P scheduling aid — **not** manifest authority |
| API response caches | Disk / optional IPFS pointer | Performance only |

Immutable replicas and content CIDs **must not** grant leases, mutate active
queue state, or authorize completion. Mutable DuckDB coordination **must not**
be confused with verified content identity. (ADR-0005 will record this decision
formally; this guide states the current code boundary.)

---

## 5. Flows

### 5.1 Backend selection and degradation (DegradationReceipt@1)

```text
Caller requests storage / VerifiedIPLDBackend
        |
  select_backend / get_backend_with_receipt
        |
  +-- preferred ipfs_kit available? --yes--> use kit; degraded=false
  |                                          (still re-verify CIDs)
  no
  |
  record degradation_reasons
  |
  +-- HF cache allowed? --yes--> role=cache; degraded=true
  |                               conformant_cid=false
  |                               coordination put/get refuse (default)
  no
  |
  Kubo CLI (or absolute Kubo fallback)
  degraded=true if kit was preferred
        |
  BackendSelectionReceipt + BackendCapabilityReceipt
```

**Degradation is never silent.** Preferred-path absence is listed on the
receipt. Whether the application continues depends on the operation’s
assurance contract (next section).

### 5.2 Coordination CID put (fail closed)

```text
VerifiedIPLDBackend.put_raw / put_dag_json
        |
  refuse if cache / non-conformant
        |
  compute expected CIDv1 (profile)
        |
  backend.block_put / add_bytes
        |
  admit returned string as CIDv1
        |
  returned == expected?  --no--> VerifiedIPLDError
        |
  re-fetch + rehash     --fail--> VerifiedIPLDError
        |
  VerifiedPutReceipt (rehashed=true)
```

### 5.3 P2P task flow (P2PTaskFlow@1)

```text
Submitter (client / orchestrator / workflow discovery)
        |
  optional: discover peers (mDNS / DHT / rendezvous / bootstrap)
        |
  TaskQueue RPC over libp2p
    preference: MCP++ (/mcp+p2p) then optional legacy NDJSON
    protocol id: /ipfs-datasets/task-queue/1.0.0
        |
  auth_ok (shared token / configured auth)
        |
  TaskQueue (DuckDB): enqueue with priority, attempts, lease fields
        |
  Worker claim_next
        |
  peer_trust resolve (if trust tiers enabled)
    TRUSTED  > ELEVATED > BASELINE (priority cap)
        |
  execute task locally
        |
  heartbeat while leased --> complete_task / failure + backoff
```

Workflow path (optional, separate from TaskQueue RPC):

```text
p2p_workflow_discovery
  tags: p2p-only | p2p-eligible | github-api | task-type tags
        |
p2p_workflow_scheduler
  MerkleClock ownership + Fibonacci priority
        |
  p2p-only  --> must not fall back to GitHub API
  p2p-eligible --> P2P preferred; GitHub may remain an operator path
  missing P2P stack --> degrade or fail per tag (see §6)
```

---

## 6. Trust, authorization, and failure semantics

### 6.1 Trust ladder (discovery ≠ capability ≠ proof)

| Signal | Means | Does not mean |
| --- | --- | --- |
| Module import | Vocabulary exists | IPFS daemon, libp2p, or peer mesh works |
| Backend selection receipt | A role was chosen and degradation recorded | CIDs are verified |
| `conformant_cid=true` | Adapter is expected to emit real CIDs | This particular put rehashed |
| `VerifiedPutReceipt` / admission | Rehash (and policy) passed for that payload | Future reads free of re-check |
| Shared P2P token / UCAN fields | Auth material present per config | Global multi-tenant security model |
| Peer trust tier | Claim priority envelope | Correct task result |
| Synthetic `bafy…` cache key | Local cache address | Multiformats CIDv1 or network resolvability |

### 6.2 Assurance contract: degrade vs fail closed

| Operation class | Missing IPFS / non-conformant backend | Missing P2P / libp2p |
| --- | --- | --- |
| Local inference without content replication | Proceed; IPFS optional | Proceed; P2P optional |
| HF / API cache acceleration | Use local cache keys; optional IPFS pointer skipped | Remote peer cache skipped |
| Coordination manifest CID, verified put/get, epoch heads | **Fail closed** (`VerifiedIPLDError` / identity error) | N/A (not a P2P concern) |
| CAR export when unsupported | **Fail closed** with capability reason | N/A |
| TaskQueue remote submit/claim | N/A | Service cannot start or RPC fails; no fake completion |
| `p2p-eligible` workflows | N/A | May fall back to non-P2P path when operator policy allows |
| `p2p-only` workflows | N/A | **Fail closed** / must not silently run as GitHub-only success |

Rule of thumb:

- **Immutable content / coordination identity** → fail closed without verified
  CIDv1 under the frozen profile.
- **Optional acceleration and cache** → degrade with explicit receipts or
  skipped features.
- **Mutable queue coordination** → fail closed without auth/lease validity;
  absence of peers is “no work claimed,” not “task succeeded.”

### 6.3 Peer trust and authorization

When trust tiers are enabled
(`IPFS_ACCELERATE_PY_TASK_P2P_TRUST_TIERS` / datasets compat alias):

| Level | Typical evidence | Claim priority |
| --- | --- | --- |
| TRUSTED | Matching shared token or UCAN-style fields | Uncapped by baseline |
| ELEVATED | Peer DID observed in local event DAG work history | Uncapped by baseline |
| BASELINE | Allowed peer without elevated evidence | Capped (default max priority 5) |

Default without tiers is binary allow/deny via shared token configuration
(`IPFS_ACCELERATE_PY_TASK_P2P_TOKEN` and datasets-compat aliases). Process
liveness alone is not health or proof of correct execution.

### 6.4 Recovery boundaries

- **Lease expiry / missing heartbeat:** queue may requeue for another worker;
  do not double-complete without queue state checks.
- **Backend degradation:** switch role only via router receipt; re-run verified
  admission before writing CIDs into manifests.
- **Corrupted or non-canonical DAG-JSON:** reject; do not “repair” into a
  different CID silently.
- **P2P partition:** local DuckDB remains source of mutable truth for that
  node’s queue file; remote peers are not alternate lease authorities for the
  same file without explicit multi-writer design (not claimed here).

---

## 7. Rationale

1. **Optional distributed plane.** CPU/local inference must work without Kubo,
   kit, or libp2p. Capability language and receipts keep docs and runtime
   honest.
2. **Role-separated backends.** Kit, Kubo, and HF cache solve different jobs;
   collapsing them hid synthetic identifiers inside “CID” fields.
3. **Frozen multiformats profile.** Cross-package reproducibility
   (CIDv1/base32/sha2-256/raw|dag-json) prevents coordination drift between
   supervisor, datasets helpers, and backends.
4. **Rehash admission.** Backend-returned strings can lie or use wrong codecs;
   local recompute + re-fetch is the gate for coordination.
5. **Mutable vs immutable split.** DuckDB leases need single-writer fencing;
   content-addressed replicas need different trust properties. Mixing them
   would let a pinned CAR or IPNS-like pointer appear to grant authority it
   does not have.
6. **Explicit degradation.** Silent HF fallback previously trained operators
   to treat `bafy…` cache tokens as network CIDs.

---

## 8. Alternatives

| Alternative | Why rejected / what it breaks |
| --- | --- |
| Treat any `bafy…` string as a CID | Admits synthetic HF keys into manifests; breaks multiformats validation and cross-peer fetch |
| Always require Kubo for all installs | Breaks local/CPU baseline and CI hermeticity |
| Silent kit → cache fallback without receipt | Hides non-conformant identity from coordination callers |
| Use IPNS or mutable paths as lease authority | Mutable naming is not single-writer fencing; stale publishers could steal claims |
| Collapse P2P results into completion without queue state | Remote peer output would upgrade to authority without lease/validation |
| Encode trust solely as “peer is connected” | Connectivity is not auth; shared token / UCAN / tiers exist for a reason |
| Assume codec preservation on kit `add` | Kit path does not guarantee dag-json vs raw; verified adapter must re-check |

---

## 9. Consequences

**Positive**

- Integrators can choose backend roles with a readable capability matrix.
- Coordination paths reject fake CIDs deterministically.
- Optional P2P does not block core inference.
- Degradation and fail-closed paths are distinguishable in receipts and errors.

**Negative / operational cost**

- Two identifier vocabularies (synthetic cache keys vs CIDv1) must be taught
  and labeled forever for HF compatibility.
- Verified put is more expensive (local hash + backend + re-fetch).
- CAR and dag-json support vary by role; exporters must handle capability
  errors.
- P2P env var dual naming (`IPFS_ACCELERATE_PY_*` and `IPFS_DATASETS_PY_*`)
  increases operator surface.
- Workflow Merkle-clock scheduling is a separate heuristic from TaskQueue
  leases; operators must not assume one global consensus plane.

---

## 10. Extension and compatibility

- **New backends:** implement `IPFSBackend`, register via
  `register_ipfs_backend`, set `backend_role` / `backend_name`, and extend
  `describe_backend_capabilities` honestly. Default unknown → non-conformant.
- **Do not** teach cache adapters to claim `conformant_cid=True` until they
  emit real multiformats CIDv1 and pass verified rehash tests.
- **Identity bridges:** add new `IdentityKind` values only with schema updates;
  never overwrite `local_id` with `cid` in place.
- **P2P protocols:** keep `PROTOCOL_V1` stable for interop; prefer MCP++
  transport binding; enable legacy NDJSON only via explicit env
  (`…_ALLOW_LEGACY_FALLBACK` / protocol order).
- Compatibility facades (`p2p_tasks/libp2p_runtime.py` re-exporting MCP++
  runtime) are not alternate authority.

---

## 11. Operational signals

| Signal | Where | Use |
| --- | --- | --- |
| `BackendSelectionReceipt.to_dict()` | `get_last_backend_selection()` | See selected role and degradation reasons |
| `BackendCapabilityReceipt` | `VerifiedIPLDBackend.capabilities()` | conformant_cid, CAR, pin, codec notes |
| `VerifiedPutReceipt` / `VerifiedGetReceipt` | verified adapter returns | Rehashed CID, digest, backend role |
| `CoordinationCidAdmission` | `admit_for_manifest` | Explicit purpose-bound admission |
| TaskQueue service state | `p2p_tasks.service.get_local_service_state` | peer_id, listen_port, running flag |
| Peer trust | `resolve_peer_trust_level` | Claim gating diagnostics |
| Workflow scheduler status | `P2PWorkflowScheduler.get_status` | Queue/heuristic state only |

Health for P2P means authenticated protocol progress and valid leases—not
merely a listening port. Health for content means rehashable CIDv1—not a
cache filename that starts with `bafy`.

---

## 12. Verification

Deterministic, offline-friendly checks for this guide (run from the repository
root on the verified tree):

```bash
# Guide present and vocabulary anchors
test -f docs/architecture/DISTRIBUTED_RUNTIME.md
rg -q 'CIDv1' docs/architecture/DISTRIBUTED_RUNTIME.md
rg -qi 'synthetic' docs/architecture/DISTRIBUTED_RUNTIME.md
git diff --check

# Source anchors exist
test -f ipfs_accelerate_py/ipfs_backend_router.py
test -f ipfs_accelerate_py/agent_supervisor/multiformats_identity.py
test -f ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py
test -d ipfs_accelerate_py/p2p_tasks

# Role and synthetic-key contracts still in code
rg -q 'class BackendRole' ipfs_accelerate_py/ipfs_backend_router.py
rg -q 'synthetic' ipfs_accelerate_py/ipfs_backend_router.py
rg -q 'conformant_cid' ipfs_accelerate_py/ipfs_backend_router.py
rg -q 'bafy' ipfs_accelerate_py/ipfs_backend_router.py
rg -q 'CID_VERSION' ipfs_accelerate_py/agent_supervisor/multiformats_identity.py
rg -q 'require_conformant' ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py
rg -q 'PeerTrustLevel' ipfs_accelerate_py/p2p_tasks/peer_trust.py
```

Optional focused tests when the environment has project deps:

```bash
python -m pytest test/ -q -k 'backend_router or verified_ipld or multiformats_identity or p2p_tasks' --collect-only
```

Review checklist:

- [ ] No prose treats HF `bafy…` cache tokens as verified CIDs.
- [ ] Immutable content/replication and mutable DuckDB coordination stay separate.
- [ ] Missing IPFS/P2P either degrades with receipts or fails closed per §6.2.
- [ ] CIDv1 profile (base32, sha2-256, raw|dag-json) is the only coordination identity.

---

## Related guides and decisions

| Document | Relation |
| --- | --- |
| [Architecture overview](overview.md) | System layers; optional IPFS/P2P note |
| [Guide conventions](GUIDE_CONVENTIONS.md) | Required guide contract |
| [IPFS kit architecture](IPFS_KIT_ARCHITECTURE.md) | Historical kit detail (not Current for synthetic CID policy) |
| Planned `INTEGRATION_BOUNDARIES.md` | Sibling repos (`ipfs_kit_py`, `ipfs_datasets_py`, MCP++) |
| Planned ADR-0005 | Single-writer mutable coordination vs immutable replication |
| Planned operator P2P guides | Install and troubleshooting journeys |
| [Agent supervisor architecture](AGENT_SUPERVISOR_ARCHITECTURE.md) | Control plane; uses verified identity at the edge |
