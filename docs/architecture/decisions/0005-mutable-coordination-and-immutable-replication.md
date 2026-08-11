# ADR-0005: Separate single-writer mutable coordination from immutable replication

- **Status:** Accepted
- **Date:** 2026-08-03
- **Last verified:** 2026-08-03
- **Deciders:** Agent supervisor maintainers; documentation-refresh structure-decisions track (DOC-019)
- **Scope:** How active lease, claim, fence, heartbeat, and queue authority are owned versus how committed coordination history and content are replicated. Covers single-writer DuckDB coordination shards, flock-serialized transactional CAS, immutable Parquet/IPLD/CAR/IPFS epoch distribution, and the rule that replicas never grant leases or mutate active state.
- **Non-goals:** Worktree isolation and fencing token protocol details (ADR-0004); model proposal versus evidence admission (ADR-0002); domain package DAG and compatibility facades (ADR-0006); MCP/MCP++ transport authorization; provider routing; objective or taskboard projection design (ADR-0001); full composition of the planned coordination-replication saga facades when still incomplete.
- **Supersedes:** none
- **Superseded-by:** none
- **Related guides:**
  - [`docs/architecture/DISTRIBUTED_RUNTIME.md`](../DISTRIBUTED_RUNTIME.md) — §4.4 immutable content versus mutable coordination; IPFS/IPLD admission
  - [`docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md`](../agent_supervisor/EXECUTION_AND_RECOVERY.md) — DuckDB owner authority; non-authoritative signals
  - [`docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md`](../agent_supervisor/PROMPT_FIRST_RUNTIME.md) — entrypoint storage boundary and landing status
  - [`docs/architecture/agent_supervisor/PACKAGE_MAP.md`](../agent_supervisor/PACKAGE_MAP.md) — DuckDB vs Parquet/IPLD/CAR/IPFS roles
  - [`docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md`](../AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md) — mutable coordination vs immutable distribution invariant
- **Source anchors:**
  - `ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py` — `LeaseCoordinator`, flock-serialized DuckDB claims/fences
  - `ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py` — `CoordinationShardBinding`, `ReplicationBinding` (`grants_authority` is always false)
  - `ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py` — `VerifiedIPLDBackend` rehash admission for CIDv1/IPLD/CAR
  - `ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py` — immutable run roots and CAS heads
  - `ipfs_accelerate_py/agent_supervisor/entrypoints/capability_resolver.py` — `duckdb_single_writer` / shard binding compilation
  - `ipfs_accelerate_py/agent_supervisor/multiformats_identity.py` — canonical CIDv1/base32/sha2-256/dag-json profile
  - `ipfs_accelerate_py/p2p_tasks/task_queue.py` — DuckDB-backed queue leases (related mutable plane)
  - `test/api/test_agent_supervisor_lease_coordination.py`
  - `test/api/test_agent_supervisor_entrypoint_contracts.py`
  - `test/api/test_agent_supervisor_verified_ipld_backend.py`

## Context

The agent supervisor and related distributed planes must answer two different
questions at once:

1. **Who may mutate active coordination state now?** Claims, heartbeats, fence
   advancement, lease expiry, takeover, and queue ownership require a single
   live authority per shard so stale workers cannot steal or double-claim work.
2. **How is committed history and content distributed?** Remote lanes, audit
   consumers, recovery, and offline analytics need portable, content-addressed
   snapshots of what was already committed—without giving those snapshots the
   power to issue new claims.

DuckDB is used as a local, transactional store for live coordination. Bundle
supervisors and lanes are separate processes; DuckDB permits only one external
writer process at a time, so the runtime already serializes short-lived
connections behind a process-shared file lock. Content-addressed media (Parquet
fragments, DAG-JSON/IPLD manifests, CAR bundles, IPFS pins) excel at immutable
distribution and verification but are eventually consistent, multi-reader, and
nameable through mutable pointers (including IPNS).

If those planes are collapsed, operators and agents reintroduce failure modes
this system is designed to close:

- **Split-brain leases** — two hosts both believe they own a shard because each
  holds a “latest” replica or IPNS head.
- **Stale publisher authority** — a delayed process republishes an old head or
  Parquet epoch and appears to authorize claims after takeover.
- **Fake identity as authority** — synthetic cache keys, unverified CIDs, or pin
  success are treated as lease grants.
- **Shared-file corruption / thrash** — multi-writer access to one DuckDB file
  across hosts races the single-writer assumption and loses fencing CAS.

The design must keep **mutable single-writer coordination** and **immutable
replication** distinct: different contracts, different trust properties, and a
hard rule that replicas never become active-state authorities.

## Decision

Active coordination is **single-writer mutable DuckDB**. Distribution of
committed coordination history and content is **immutable Parquet / IPLD /
CAR / IPFS** replication. Immutable replicas **cannot grant leases, cannot
mutate active coordination state, and cannot authorize effects**.

### Normative split

1. **DuckDB owns active mutable state.** Per coordination shard, one elected
   owner writes claims, fences, heartbeats, expiry, and related queue or
   lease rows through transactional compare-and-swap. The contract binding
   requires `backend == "duckdb"` and
   `write_model == "single_writer_transactional_cas"`.
2. **Local multi-process writers serialize on the owner host.**
   `LeaseCoordinator` takes a process-shared exclusive file lock, opens a
   short-lived DuckDB connection, and validates the accepted claim and fencing
   token inside one transaction. An expired worker cannot publish progress or a
   terminal receipt after takeover.
3. **Remote workers do not share the database file.** Remote access is
   `owner_rpc`: authenticated mutations route to the current shard owner. Peer
   hosts must not open the same DuckDB path for writes over a shared filesystem
   and treat that as multi-writer coordination.
4. **Parquet / IPLD / CAR / IPFS carry immutable history.** Committed logical
   epochs export as bounded Parquet partitions (repository, run, date, shard)
   and linked canonical DAG-JSON/IPLD manifests; CAR export and IPFS
   publication are capability-gated. Reconstruction yields **read-only**
   query projections or verified content—not a second live writer.
5. **Replicas never grant authority.** `ReplicationBinding.grants_authority`
   is always `false`. A successful IPFS fetch, pin, CAR export, Parquet
   row-count match, IPNS head resolution, or local replica file **must not**
   issue a lease, advance a fence, mutate the active DuckDB shard, or accept
   an effect. Only the current DuckDB-owner lease and fence authorize mutable
   actions.
6. **Content identity is fail-closed and rehashed.** Coordination-facing IPLD
   paths use `VerifiedIPLDBackend` with the frozen multiformats profile
   (CIDv1 / base32 / sha2-256 / raw|dag-json). Backend-returned strings and
   cache-role synthetic keys are not admitted as epoch or coordination CIDs.
7. **Partition and stale-owner behavior.** Under network partition, the local
   DuckDB file remains the mutable truth for that owner’s shard; remote peers
   are not alternate lease authorities for the same shard. Stale owners with
   expired or superseded fences fail closed; stale or tampered replicas may be
   quarantined or refused but never elected as writers by virtue of content
   availability.
8. **Replay and apply are distinct.** Exact logical replay of a verified epoch
   into a read-only projection is allowed. Applying a remote epoch into the
   **active** mutable store requires the current owner’s authenticated lease
   and fence (and any explicit apply policy)—replication success alone is
   insufficient.

### Ownership boundary

| Concern | Authoritative medium / symbol | Grants leases / mutates active state? |
| --- | --- | --- |
| Live claims, fences, heartbeats, expiry | DuckDB via `LeaseCoordinator` / shard owner | **Yes** (single writer) |
| Shard write policy and remote access | `CoordinationShardBinding` | Defines who may write; itself not a lease |
| Immutable epoch / checkpoint distribution | Parquet + IPLD (+ optional CAR/IPFS) via `ReplicationBinding` | **No** (`grants_authority` is false) |
| Verified content put/get / CAR | `VerifiedIPLDBackend` | **No** — content identity only |
| Immutable run identity roots | `run_registry` root + CAS head | **No** — identity and revision, not leases |
| Mutable naming (IPNS, pubsub heads) | Discovery / cache acceleration only | **No** — never lease authority |

```text
  Active path (authority)              Distribution path (no authority)
  -----------------------              --------------------------------
  Claim / heartbeat / fence            Export committed epoch
        |                                     |
        v                                     v
  DuckDB shard (single writer)         Parquet fragments + IPLD manifest
  flock + transactional CAS                   |
        |                                     v
        +-- owner_rpc for remotes      Optional CAR / IPFS pin (verified)
        |                                     |
        v                                     v
  Mutable active state only            Read-only reconstruct / audit / input CIDs
                                       (cannot grant lease or mutate active DB)
```

## Alternatives

### Alternative A: Multi-writer / shared-file DuckDB

- **Summary:** Multiple hosts or processes open the same DuckDB database file
  (NFS, shared volume, or concurrent external connections) and all write
  claims, heartbeats, and fence updates. Rely on filesystem locking, DuckDB
  multi-connection behavior, or application retries for consistency. No single
  elected owner; no `owner_rpc` requirement.
- **Expected benefits:** No shard-owner election or RPC hop; simpler mental
  model of “one database everyone updates”; easier horizontal scaling of
  writers without routing mutations.
- **Why not chosen:** Multi-writer shared-file coordination **does not**
  preserve single-writer fencing or safe reclaim.
  - DuckDB’s external process model is **one writer at a time**. Concurrent
    writers across hosts thrash locks, surface connection errors as false
    “no claim,” or corrupt the store under non-POSIX shared filesystems.
  - Without a single owner principal and transactional CAS on claim/fence
    rows, two processes can both observe “available,” both insert claims, and
    produce **duplicate** live owners for one task/shard epoch.
  - A delayed writer that still holds an old connection can commit after
    another host’s takeover—**stale mutation of active state**—because the
    file itself has no fencing generation independent of the writer’s view.
  - Remote workers sharing a path over a network filesystem couple lease
    safety to VFS semantics the runtime does not control; partitions make
    “who holds the lock?” ambiguous.
  - The entrypoint contract therefore rejects any write model other than
    `single_writer_transactional_cas` and requires remote workers to call the
    shard owner (`owner_rpc`) rather than share the DB file.

Shared-file multi-writer is rejected for active coordination. Horizontal
scale is achieved by **independent shards** (multiple DuckDB files, each with
one writer), not multi-writer access to one file.

### Alternative B: IPNS (or other mutable names) as lease / coordination authority

- **Summary:** Treat an IPNS name, mutable path, pubsub head, or “latest”
  pointer as the authority for who owns a claim or which epoch is live. Workers
  resolve the name, fetch the linked CID, and grant themselves leases or
  accept effects from that resolution. DuckDB becomes optional cache or is
  omitted.
- **Expected benefits:** Global discovery without electing a DuckDB owner;
  natural fit for IPFS-native deployments; operators can “point” IPNS at the
  current head to reassign work.
- **Why not chosen:** Mutable naming is **not** single-writer fencing and
  cannot safely authorize active state.
  - IPNS and similar pointers are **eventually consistent**. Two peers can
    observe different heads during propagation delay; both may claim the same
    work—**split-brain leases**.
  - A **stale publisher** that still holds the IPNS key (or an old publish
    delayed in the network) can re-publish a superseded head and appear to
    reclaim authority after a legitimate takeover.
  - Resolving a name and fetching bytes proves **content retrieval**, not that
    the local process holds the current fence for a shard. Pin success and
    head signature alone do not replace transactional CAS on the owner store.
  - Treating IPNS as authority would let immutable or renamed content
    **grant leases and mutate active state** by indirection—the exact
    collapse this ADR forbids.
  - IPNS may remain a **discovery or cache index** aid (for example locating a
    policy-cleared epoch CID). Discovery must never substitute for DuckDB
    owner lease and fence postconditions.

### Alternative C: Content CID or Parquet replica as active store

- **Summary:** Use the latest verified IPLD epoch CID (or a local Parquet
  dataset) as the sole coordination store. Workers import the replica, mutate
  rows in memory or in a throwaway DuckDB projection, and republish a new CID
  as the next head—no durable single-writer shard file.
- **Expected benefits:** Pure content-addressed coordination; easy audit
  chains; no DuckDB operational dependency for live claims.
- **Why not chosen:** Epoch heads answer “what was committed?” not “who may
  write now?” Concurrent publishers race to publish successor CIDs without a
  single-writer CAS register; lost updates and duplicate claims reappear.
  Reconstructed projections are valuable for **read-only** query and remote
  **immutable inputs**, but only the current DuckDB owner may accept apply
  under its authenticated lease and fence. Missing this split would again let
  replicas grant authority they do not have.

### Alternative D: Collapse replication into the mutable DB only (no immutable epochs)

- **Summary:** Keep all history only inside the live DuckDB file; ship the
  whole database file (or continuous binary replication) for remote workers
  and audit.
- **Expected benefits:** One store technology; no Parquet/IPLD export pipeline.
- **Why not chosen:** Shipping a live writable DB multiplies Alternative A’s
  shared-file risks. Content-addressed, policy-cleared, capability-gated
  distribution (verified CIDs, disclosure gates, CAR bounds) needs immutable
  fragments, not a second live writer handle. Remote lanes already consume
  **immutable** content-addressed inputs; durable epoch export matches that
  model without granting replica authority.

## Consequences

### Positive

- **Clear authority boundary:** Active leases and fences live only on the
  single-writer DuckDB shard; operators and agents cannot “promote” a pin or
  IPNS head into a claim grant.
- **Stale workers and replicas fail closed:** Superseded fences and non-owner
  replicas cannot mutate active state or authorize effects.
- **Portable history without split-brain:** Parquet/IPLD/CAR/IPFS distribute
  committed epochs for audit, analytics, and remote immutable inputs while
  remaining non-authoritative.
- **Honest multi-host scaling:** Independent shards scale writers; each shard
  keeps one owner instead of pretending multi-writer shared files are safe.
- **Verified identity for coordination content:** Rehash admission prevents
  synthetic or cache-only keys from entering epoch heads and manifests.
- **Contract-level enforcement:** `CoordinationShardBinding` and
  `ReplicationBinding` encode the split so launch plans cannot silently choose
  multi-writer or authority-granting replication modes.

### Negative

- **Owner RPC and election complexity:** Remote mutation path requires an
  authenticated owner; not every worker can open the DB path.
- **Operational dual stack:** Teams must operate DuckDB shards **and**
  Parquet/IPLD/IPFS replication tooling, including capability and disclosure
  gates.
- **Export/import cost:** Epoch export, CID verification, and optional CAR
  packaging add latency and storage versus “just use the live DB.”
- **Incomplete product facades residual:** Full coordination-replication saga
  composition may still be landing; until complete, embedders compose
  `LeaseCoordinator`, contracts, and `VerifiedIPLDBackend` explicitly—docs and
  operators must not claim end-to-end saga completeness where status tables say
  planned.
- **Shard count and ownership tuning:** Wrong shard cardinality or sticky
  owners create hot spots or slow handoff; fencing TTLs interact with export
  lag.

### Neutral / residual risks

- **Clock skew and lease TTL** still affect reclaim timing; fencing reduces but
  does not eliminate wall-clock races (see ADR-0004).
- **Local multi-process contention** on one host depends on flock + short
  transactions; pathological lock hold times delay peers.
- **IPNS/pubsub as discovery** remains available for non-authoritative indexes;
  operators must not re-label discovery as authority.
- **P2P TaskQueue DuckDB files** follow the same mutable-authority rule per
  node file; they are not a global multi-writer consensus plane.
- **Merkle-clock / workflow ownership heuristics** are scheduling aids and
  must not be confused with DuckDB lease authority or IPLD content identity.

## Evidence

| Claim in Decision | Evidence (path, test, or operational check) | Notes |
| --- | --- | --- |
| DuckDB single-writer coordination | `lease_coordination.py` module doc + `_coordinator_operation` / flock; `COORDINATION_STORE_SCHEMA` | External writer serialized |
| Contract forces duckdb + single_writer_transactional_cas | `CoordinationShardBinding.__post_init__`; `test_agent_supervisor_entrypoint_contracts.py` | Rejects other backends/write models |
| Remote must not share DB file | `remote_access == "owner_rpc"` error text in contracts | Owner RPC only |
| Replication never grants authority | `ReplicationBinding.grants_authority` → `False`; contract module docstring | Always false |
| Immutable Parquet/IPLD binding shape | `ReplicationBinding` fields; partition keys repository/run/date/shard | Epoch distribution policy |
| Verified IPLD/CAR admission | `verified_ipld_backend.py`; `test_agent_supervisor_verified_ipld_backend.py` | Fail closed on mismatch/cache role |
| Immutable lane inputs separate from leases | `ImmutableLaneInput` / distributed schemas in `lease_coordination.py` | Content-addressed inputs |
| Guide-level non-authority of replicas | `DISTRIBUTED_RUNTIME.md` §4.4; `EXECUTION_AND_RECOVERY.md` DuckDB owner authority | Operator narrative |
| IPNS not lease authority (rejected) | `DISTRIBUTED_RUNTIME.md` §8 alternatives; this ADR Alternative B | Mutable names ≠ fencing |
| Fenced claims on DuckDB path | `test_agent_supervisor_lease_coordination.py` | Complements ADR-0004 |

## Verification

From the repository root:

```text
# Contract invariants: single-writer DuckDB; replication grants no authority
python -m pytest \
  test/api/test_agent_supervisor_entrypoint_contracts.py \
  test/api/test_agent_supervisor_lease_coordination.py \
  test/api/test_agent_supervisor_verified_ipld_backend.py -q

# Symbols and hard boundaries still present
rg -n 'single_writer_transactional_cas|grants_authority|must be duckdb|owner_rpc' \
  ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py

rg -n 'VerifiedIPLDBackend|VerifiedIPLDError' \
  ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py

rg -n 'DuckDB is required for lease coordination|immutable-lane-input' \
  ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py

# Guides still separate mutable coordination from immutable replicas
rg -n 'never grant|immutable content versus mutable|IPNS or mutable paths as lease' \
  docs/architecture/DISTRIBUTED_RUNTIME.md \
  docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md
```

Pass signals: contracts still reject non-DuckDB coordination backends and
non-single-writer write models; `grants_authority` remains false; verified IPLD
paths still rehash and fail closed; lease coordination still requires DuckDB
and fencing; documentation continues to forbid replica/IPNS lease authority.

Fail signals (ADR stale): multi-writer shared-file coordination becomes the
default write model; replication bindings or IPNS resolution grant leases;
active DuckDB mutations are performed from replica import without owner fence;
synthetic/cache CIDs admitted as coordination epoch heads; guides treat pin or
IPNS success as claim authority.

## Review triggers

- [ ] Source anchors no longer match the Decision statement
- [ ] A recorded negative consequence becomes unacceptable
- [ ] A rejected alternative becomes viable without those costs (e.g. a true
      multi-writer coordination engine with fence-equivalent CAS and
      cross-host linearizability stronger than shared DuckDB files)
- [ ] Security, isolation, lease/fence, or trust-tier changes touch this scope
- [ ] Related guide or package ownership is restructured
- [ ] Superseding design is Accepted under a new ADR number
- [ ] Full coordination-replication saga modules land or change apply-authority
      rules (`coordination_replication`, epoch import postconditions)
- [ ] Shard ownership election or remote_access model changes away from
      `owner_rpc`
- [ ] IPNS or mutable naming is proposed again as active coordination authority

## Notes (optional)

- **Relation to ADR-0004.** ADR-0004 defines *how* leases, fences, and
  worktrees isolate concurrent implementation. This ADR defines *where*
  mutable authority lives versus *how* immutable history is replicated. Fence
  tokens without a single-writer store, or a single-writer store without
  fences, are each insufficient.
- **Planned saga composition.** Prompt-first objectives (ASE-G044 and related
  tasks) describe end-to-end DuckDB → Parquet → IPLD/IPFS → read-only DuckDB
  parity. This ADR is normative for the authority split whether or not every
  facade module has landed; landing work must not weaken
  `grants_authority is false` or single-writer CAS.
- **Cache IPNS indexes.** Optional cache IPNS/pubsub indexes elsewhere in the
  stack remain non-coordination; they must not be documented as lease or
  completion authorities.
- **Independent shards.** Scaling writers means more shards (more DuckDB
  files and owners), not multi-writer access to one file or IPNS-mediated
  claim grants.
