# Resource-aware sealing investigation

Investigation dates: 2026-09-08–09 UTC. Repeated hashing and verification were doing
substantial work, but the observed thread count did not establish runaway
parallel hashing: only one of the sampled process's 39 threads was busy.
Logical read counters include reads served from the page cache; they are not
evidence that the same volume reached the physical disk. Repeated verification
can therefore consume CPU and memory bandwidth even when storage traffic is low.

## Worktrees and deployment status

| Location | Role |
| --- | --- |
| `/home/barberb/lift_coding/.worktrees/agent-supervisor-efficiency-and-state-hardening-v1` | Active ASEH owner and its launcher. |
| `/home/barberb/lift_coding/.worktrees/pctdd-g9-orphan-recovery/external/ipfs_accelerate` | Separate PCTDD accelerator checkout with incremental-sealing work. |
| `/home/barberb/lift_coding/.worktrees/pctdd-g9-orphan-recovery/external/ipfs_datasets` | PCTDD's canonical semantic-state implementation. |
| `/home/barberb/lift_coding/.worktrees/resource-aware-sealing` | Repair checkout: branch `codex/resource-aware-sealing`, based on ASEH commit `73f9cbda4`. |

The active owner runs
`scripts/run_agent_supervisor_efficiency_state_hardening.py`. Its ignored,
host-local launcher is
`data/aseh/logs/cron-one-shot-launch-owner.sh` in the ASEH worktree above.
The launcher now applies nice level 10, best-effort I/O priority 7, CPU affinity
`0-3`, and one-thread defaults for OpenMP, OpenBLAS, MKL, NumExpr, BLIS, vecLib,
and Rayon. The subsequent owner, PID `1925103`, was observed starting with one
thread and the intended limits. This is a point-in-time observation, not a
permanent PID or a guarantee that every library obeys these variables. Affinity
limits where work runs; it is not a CPU-time quota. I/O priority effectiveness
depends on the storage scheduler.

Later, PID `1967757` had 20 threads after more native initialization; its CPU
affinity was still `0-3`, nice level still 10, and I/O priority still 7. The
environment variables are library-specific requests, not a universal thread
count limit. The inherited CPU affinity bounds simultaneous CPU execution for
cooperating descendants even when a library allocates extra sleeping threads.

The repeated-work sites in the active checkout are `_trusted_git_executable`
and `_trusted_receipt_validation_python` in that script, plus
`_agent_native_python_executable_sha256`,
`verify_agent_supervisor_native_dependency_sealed_fd`, and
`verify_agent_implementation_sealed_control_plane` in
`ipfs_accelerate_py/agent_implementation_route.py`. Git is about 3.9 MB, Python
7.5 MB, and the DuckDB extension 54.5 MB. The sampled owner accumulated roughly
32 GB of logical reads in a few minutes while physical reads stayed below
300 KB. Its cron relaunch history shows repeated starts about four minutes
apart, so restart costs are relevant as well as per-call costs. The reason for
those restarts was not established by this hashing investigation.

Source fixes are prepared in the repair checkout. They are **not deployed into
the accepted sealed runtime** and do not upgrade the separate PCTDD checkout.
Deployment still requires the existing source-admission and resealing workflow.
The files to review are:

- `ipfs_accelerate_py/_hash_resources.py` and
  `ipfs_accelerate_py/agent_supervisor/runtime/hash_pressure.py`: shared resource
  controls and the import facade used by hashing callers.
- `ipfs_accelerate_py/agent_implementation_route.py`: bounded reuse of digests
  for verified immutable, kernel-sealed descriptors.
- `ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py` and
  `scripts/run_agent_supervisor_efficiency_state_hardening.py`: integration
  points for resource controls and shared routine Git observations.
- `ipfs_accelerate_py/agent_supervisor/runtime/shared_hashing.py`,
  `task_sources/hash_observations.py`, `task_sources/typed_state_owner.py`, and
  `task_sources/sql/0004_hash_observations.sql` beneath that supervisor package:
  metadata-validated shared hashes, owner-side leases, narrow authenticated
  requests, and additive schema.
- `ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py`:
  interpreter verification and the isolated bootstrap share the same lock;
  the bootstrap releases it before importing the verified capsule.
- `ipfs_accelerate_py/llm_router.py`: raw CID generation now delegates to the
  existing dependency-free encoder, eliminating the non-CID `sha256_...`
  fallback when `multiformats` is absent. Existing saved entries are not migrated.

The shared gate admits one heavy hashing batch per UID across upgraded callers
sharing `/tmp`, regardless of worktree, hash kind, or XDG runtime directory.
It samples CPU affinity, cgroup v2 CPU and memory bounds, available memory, load,
and CPU/I/O/memory pressure. Its yielded worker ceiling defaults to two, caps
at four, and falls to one under pressure. Existing fingerprint streams remain
serial. `IPFS_HASH_MAX_WORKERS` configures a lower ceiling (up to four), and
`IPFS_HASH_LOCK_TIMEOUT_SECONDS` bounds waiting (default 60 seconds). The
stdlib-only pre-import bootstrap uses the same lock and a fixed 60-second wait.
Only callers using this protocol participate; older worktrees and separately
mounted container temporary directories do not automatically join the budget.

Individual file producers use `hashing_worker_slot()`: a shared batch gate plus
exclusive numbered slots permits two files by default, at most four. Pressure
can reduce admission to one; outstanding workers drain rather than being killed.
Heavy batches exclude these workers. Owner requests and waiting for another
producer happen outside the worker slot. There is no nested pool per supervisor.

Immutable verification reuse is process-local. It retains at most four duplicate
descriptors and 128 MiB of sealed objects, without storing copied payloads.
Every hit checks actual kernel seals, descriptor identity, expected digest, and
the relevant native validation policy. Another process must verify its own first
observation, with cold work serialized by the shared gate. Legacy whole-workspace
fingerprints, receipt-interpreter verification and final seal verification still
read current bytes. Routine Git checks now use the shared metadata policy below,
without skipping live ownership, permissions, capabilities or path/FD admission.

Small-file inventory remains unchanged after benchmarking the candidate
integration. A per-file database request was slower than its original bounded
hashing loop; the cache should not be pushed indiscriminately into AST scanning.
Batch owner requests or a bounded metadata-validating front cache are further
work, not implemented performance claims.

## Hash choice on this machine

The bounded benchmark used an 8 MiB in-memory buffer, approximately 0.3 seconds
per algorithm, one thread, and lowered process priority. This ARM64 machine's
Python SHA-256 implementation was `_hashlib`, linked to OpenSSL 3.0.13.

| Implementation | Measured throughput, MiB/s |
| --- | ---: |
| SHA-256 / OpenSSL | 2367.4 |
| BLAKE2b / Python standard library | 1179.6 |
| SHA3-256 / Python standard library | 1078.0 |
| BLAKE3 / installed package 1.0.9, one thread | 714.4 |

Keep SHA-256 for existing integrity identities and CIDv1 multihashes. These
short, warm-memory measurements support that choice on this host; they do not
rank all hardware, small-file workloads, or newer implementations. OpenSSL
detects supported Arm SHA-256 acceleration automatically. Python releases the
GIL for hash updates larger than 2,047 bytes, so file-level worker pools can
consume several cores despite being Python threads.
[OpenSSL Arm capabilities](https://docs.openssl.org/master/man3/OPENSSL_armcap/),
[Python hashlib](https://docs.python.org/3/library/hashlib.html).

CIDv1 describes the version, codec, and multihash; it is not a second hashing
algorithm. A raw-block CIDv1 can wrap an already computed SHA-256 digest without
rehashing the block. A chunked UnixFS file normally has a different root-block
digest from its raw-file checksum. Preserve codec, chunking, DAG layout, and
importer settings when reproducing IPFS addresses. BLAKE3 is supported by the
current Boxo allowlist, but changing the algorithm changes addresses and offers
no measured advantage here.
[IPFS content addressing](https://docs.ipfs.tech/concepts/content-addressing/),
[Boxo hash allowlist](https://github.com/ipfs/boxo/blob/main/verifcid/allowlist.go).

## Existing incremental reuse

In the PCTDD accelerator checkout, use the prefix
`ipfs_accelerate_py/agent_supervisor/` for these paths:

- `analysis/duckdb_ast_index.py`: `SourceFileSpec` verifies source bytes;
  `DuckDBASTIndex.ingest_snapshot` and `_load_parse_cache` reuse AST parsing by
  `(content_digest, parser_id)` across worktrees. Its `source_files`,
  `file_versions`, and `parse_cache` tables illustrate the existing design.
- `semantic_state/datasets_adapter.py`: the canonical interface delegates
  `scan_repository(previous_state=...)` and
  `open_semantic_state(root_cid, get_block)` to datasets. Under the
  datasets-authoritative profile, `analysis/semantic_truth_authority.py`
  explicitly disables the legacy accelerator-local AST writer.
- `proof/incremental_sealing/delta_seal.py`: reused proof units require complete,
  unchanged cache keys and fresh verification. `full_checkpoint.py` rejects
  reuse that hides missing verification. The exact-byte memo in
  `critical_path.py` is instrumentation and hashes its lookup input; it is not
  a production file-hash cache.

In the PCTDD datasets checkout,
`ipfs_datasets_py/logic/software_contracts/semantic_index/scanner.py` implements
`RepositoryScanner.scan` and `scan_snapshot`. Snapshot acquisition captures
source bytes once. The scanner reuses that captured data, verifies source CIDs,
and reuses unchanged analysis through `previous_state`; graph resolution still
runs against the current inventory. This avoids reparsing and duplicate disk
reads, while preserving verification of current inputs.

The Grok workspace fingerprint is a hash of one ordered byte stream. Replacing
file contents with cached leaf hashes would change that identity. Leaf-based
incrementality needs an explicitly versioned manifest/Merkle format, or reuse
of an already authenticated immutable snapshot; it cannot silently change the
existing fingerprint recipe.

## Shared DuckDB/Quack hash observations

Implemented in the repair sources, using the existing DuckDB/Quack owner's
authenticated typed socket and its existing exclusive connection. Workers never
open another writable DuckDB connection. The raw Quack read replica is not used
to claim work: its mutation-inbox routing cannot provide the needed atomic
claim-and-return result. No live database was migrated or otherwise changed.

The default is **24 hours**, not 30 seconds. This is a periodic recheck policy,
not a claim that the digest itself expires. Every request checks the local file's
device, inode, size, nanosecond mtime/ctime, mode, ownership and link count, plus
host boot and mount identity. A changed witness immediately misses regardless
of TTL; ordinary same-size edits with restored mtime are caught by ctime.
Atime is intentionally excluded because reads change it.

| Setting | Behavior |
| --- | --- |
| `IPFS_HASH_CACHE_TTL_SECONDS=86400` | Default maximum observation age; hits never extend it. |
| `IPFS_HASH_CACHE_TTL_SECONDS=604800` | Seven-day configurable upper bound. |
| `IPFS_HASH_CACHE_TTL_SECONDS=0` or `strict=True` | Bypass the cache and read current bytes under the worker budget. |
| `IPFS_HASH_MAX_WORKERS=2` | Default per-host/UID file concurrency, hard-capped at four and reduced under pressure. |

The existing supervisor positive environment projection carries these hash
settings; an explicit profile setting wins over an ambient default. The owner
uses a separate producer lease (30 seconds by default, at most five minutes);
the client requests a lease appropriate to its bounded operation timeout.
That lease is a crash-recovery deadline, **not the digest TTL**.

`hash.observe` is a new isolated service capability, issued through the existing
sealed, kernel-peer-bound grant broker. It does not enlarge status/event/SQL
grants, and provider children do not receive the broker credential. Lookup,
claim, complete and abort execute short owner transactions. One claim admits
one producer for an exact file identity; followers reuse the result. Lease
tokens, principal binding, fencing counters and owner generations reject stale
or foreign completions. Completed observations can survive owner restarts when
the witness and TTL still match; active old-generation producers cannot publish.
Clock rollback detected within an owner lifetime invalidates the cache; future
observations are rejected after restart. It does not detect every possible
backward wall-clock jump between owner lifetimes.

The `hash_observations` table is installed by additive full migration 0004 and
operational-profile migration 0002. The old operational profile's checksum is
unchanged. Missing schema rejects cache requests without creating tables. The
table is bounded to 4,096 observations by default; cache eviction costs a later
rehash, not correctness. Broken owner sessions are evicted without replaying an
uncertain mutation or falling back to independent hashing. Without a broker
handoff, ordinary non-authority clients perform resource-bounded fresh reads.

Sharing requires supervisors to attach to the **same owner/store**. Different
worktree databases are not silently federated. Two paths to the same filesystem
object can share a result; distinct worktree files with different inodes need an
initial read even if their content is identical. Known immutable Git objects or
content roots would support further deduplication, but are not guessed here.

This policy intentionally trusts the local filesystem's change reporting and
the admitted supervisor that publishes a digest. It is appropriate for routine
analysis and repeated root-owned Git checks. The cache is not a ZK proof, an
independent verification of its producer, or evidence against a compromised
filesystem/kernel. Retain strict reads for hostile source admission and final
sealing; use strict mode for filesystems whose attribute consistency cannot be
trusted. `_trusted_git_executable(strict=True)` also exposes a direct bypass.

A disposable real Unix-socket/DuckDB benchmark (15 iterations per size, one CPU,
nice 10, warm filesystem cache) measured the following median call times:

| File size | Resource-admitted fresh read | Warm shared observation |
| --- | ---: | ---: |
| 4 KiB | 1.247 ms | 1.104 ms |
| 1 MiB | 2.072 ms | 1.345 ms |
| 4 MiB | 3.833 ms | 1.565 ms |
| 8 MiB | 6.296 ms | 1.623 ms |

Warm hits read zero payload bytes. Initial owner-backed calculations cost
15–22 ms including durable publication. The original unguarded inventory loop
took just 0.0126 ms for 4 KiB and 0.5849 ms for 1 MiB. Thus the shared path helps
repeated larger objects and duplicate concurrent work; it is not a general
small-file latency improvement. The experimental per-file forest integration
was removed, restoring its exact original source.

Proof reuse must bind the exact content root/public inputs, proof-system and
verification profile, program/verifier identity, and applicable keys/policy.
An existing ZK proof establishes its encoded computation; it does not observe
later filesystem changes. Reuse may save proving while still requiring fresh
verification and a trustworthy connection to the current inputs.
[RISC Zero verification semantics](https://github.com/risc0/risc0#protocol-overview-and-terminology).

After source admission and resealing, confirm the new sealed capsule contains
the resource and immutable-cache changes. Measure repeated verification reads,
cache hits, CPU pressure, and owner responsiveness on a bounded workload before
increasing concurrency. Keep the host-local launcher limits in place while
admitting and deploying the durable owner-managed cache.

## Validation completed

The initial change passed 132 focused test cases: 25 resource-admission cases, 13 immutable-cache
cases, 46 existing native-dependency cases, two capsule closure/isolated-import
cases, seven hashing/CID/bootstrap entrypoint cases, four trusted executable
identity cases, and 35 existing multiformats identity cases. This includes
loading the installed ARM64 DuckDB extension from its sealed descriptor.

Regression tests cover same-size/same-mtime file changes, missing individual
kernel seals, descriptor substitution during reads, forged expected digests,
cache eviction, cross-process contention, cache/resource lock ordering, and
normal child unwinding after fork without closing unrelated descriptors.
Concurrent repeated verification reads an immutable payload once within a
process; subsequent cache hits read zero payload bytes. The legacy workspace
fingerprint still matches its fixed byte-stream vector.

Tests used lowered priority and one-thread native-library settings. The live
launcher's shell syntax and the repair diff's whitespace checks also passed.
This focused verification does not constitute a new supervisor acceptance seal
or a full end-to-end run of the implementation program.

The September 9 shared-cache extension adds owner/backend, real-socket transport,
cross-process worker slots and client regression coverage. The Git
integration test primes a hash from another client, then observes **zero payload
bytes** for the routine check and exactly one full read with `strict=True`.
Receipt-interpreter checks remain outside the metadata cache. The 41 client
tests include three spawned processes sharing the real typed owner: one payload
read, two cache hits. They also exercise the real sealed credential broker,
grant renewal, uncertain-response eviction without replay, and subsecond TTLs
encoded as integer milliseconds (the canonical wire format forbids floats).

Latest targeted validation covers 392 distinct cases across the retained source
changes, including 19 isolated capsule/migration closure checks. Two additional
existing uninitialized-gitlink forest/manifest tests fail identically with the
original HEAD source; that inventory module is unchanged in the final patch.
