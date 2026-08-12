# IPFS Kit Kernel-Mounted VFS Improvement Plan

Status: active implementation program

Date: 2026-08-09

Board namespace: `ipfs-kit-kernel-vfs-fuse-v1`
Scope: `ipfs_kit_py` canonical VFS + canonical WAL + generation-bound ARC, exposed through fusepy on Linux and through fusepy's WinFsp FUSE compatibility route on Windows, including Linux Docker operation.

Companion control artifacts:

- objective heap: `docs/architecture/ipfs_kit_fuse_vfs.objectives.md`
- executable task board: `docs/architecture/ipfs_kit_fuse_vfs.todo.md`
- scheduler: `config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json`
- validator: `scripts/validate_ipfs_kit_fuse_vfs_board.py`

## 1. Outcome

Deliver one production composition in which ordinary OS file operations reach the same semantic authority as package, CLI, and MCP callers:

```text
Linux VFS syscalls                     Windows file APIs
        |                                      |
 Linux FUSE kernel + libfuse2           WinFsp FSD + FUSE 2.8 layer
        |                                      |
        +----------- fusepy callbacks --------+
                            |
                 KernelVFSOperations
                            |
         path / errno / metadata / handle contracts
                            |
               DurableCachedVFSRuntime
                  /          |          \
       CanonicalVFSService   WAL   GenerationBoundARC
                  \          |          /
             injected ranged storage boundary
                            |
            memory / local / IPFS / Iroh adapters
```

The mount is not a second VFS implementation. fusepy is a thin kernel callback adapter; `CanonicalVFSService` remains the operation and error authority. Every acknowledged mutation is ordered through the canonical WAL, and ARC contains only version/generation-bound committed data.

## 2. Evidence-backed baseline

The bound `ipfs_kit_py` revision already contains strong components:

- `ipfs_kit_py/core/vfs/contracts.py`, `service.py`, `transactions.py`, and `snapshots.py` define the canonical VFS state machine, path policy, version preconditions, and deterministic locking.
- `ipfs_kit_py/core/wal/` defines bounded records, segments, writer, transaction coordinator, recovery, and checkpoints.
- `ipfs_kit_py/arc_cache.py` and `ipfs_kit_py/cache/arc/` define generation-bound ARC, persistence, metrics, reference behavior, and single-flight helpers.
- 258 focused runtime-readiness VFS/WAL/ARC tests and 36 legacy VFS/MCP contract tests passed during the 2026-08-09 audit.

Confirmed gaps:

1. There is no active fusepy/WinFsp operations adapter, mount CLI, platform doctor, live mount test, FUSE image, or FUSE-specific CI.
2. Runtime authority is split: `ipfs_fsspec.VFSCore` reaches real backends but has a simple unbounded dictionary cache and weak whole-file semantics; `CanonicalVFSService` has stronger contracts but only an in-memory boundary. The kernel adapter must bind the canonical service, not expose the old cache directly.
3. The VFS, WAL, and ARC components are manually joined in tests but are not composed by one production runtime.
4. The canonical VFS lacks kernel-shaped handle, partial-write, truncate, append, fsync/flush/release, stable inode, statfs, unlink-while-open, and rename-while-open semantics.
5. WAL recovery currently needs an audit of the marker/sidecar crash gap; canonical WAL records must carry the recoverable transaction source of truth.
6. ARC invalidation is manual and whole-object oriented; large FUSE reads need generation-bound range keys and generation-aware single-flight.
7. Windows needs an explicit collision-safe case/name policy, drive/directory mount behavior, and delete/share/rename compatibility contract.
8. Docker images do not install libfuse/fusermount and Compose does not expose `/dev/fuse` or `SYS_ADMIN`.

## 3. Architectural decisions

### 3.1 One semantic authority

`CanonicalVFSService` owns path, operation, result, error, version, transaction, and observed-state-transition semantics. A new ranged `VFSStorageBoundary` family connects that authority to memory, local, IPFS, and Iroh storage. `VFSCore` and legacy managers become compatibility callers or are explicitly unsupported; they are not a bypass around durability or cache coherence.

### 3.2 A closed callback contract

The minimum production callback set is:

- metadata: `getattr`, `readdir`, `access`, `statfs`, `utimens`;
- file lifecycle: `open`, `create`, offset `read`, offset `write`, `truncate`, `flush`, `fsync`, idempotent `release`;
- namespace: `mkdir`, `rmdir`, `unlink`, atomic same-mount `rename`;
- lifecycle: `init` and `destroy`.

`readlink`, `symlink`, `link`, `mknod`, `chmod`, `chown`, and xattrs must either have canonical semantics and tests or return a stable `ENOSYS`/`EOPNOTSUPP`. Unknown callbacks and false success are forbidden.

### 3.3 Handle and data visibility rules

- File handles, not paths, identify open instances. A rename or unlink does not invalidate an already-open handle.
- Handles are bounded, generation-tagged, lease-aware, and reclaimed after crash.
- Dirty extents live in per-handle staging, never in the shared ARC.
- Reads observe their own staged writes; cross-handle visibility follows the declared mount consistency mode.
- `O_CREAT`, `O_EXCL`, `O_TRUNC`, `O_APPEND`, sparse/partial writes, and case-only rename have explicit state-machine traces.

### 3.4 WAL acknowledgement pipeline

Default mutation ordering is:

```text
validate + authorize
  -> acquire deterministic path/handle locks
  -> append recoverable WAL intent and payload/reference
  -> meet configured intent durability boundary
  -> apply canonical VFS transaction/backend effect
  -> append decision/effect identity
  -> invalidate/advance ARC generation
  -> return callback result
```

`fsync` succeeds only after WAL and selected backend durability receipts are current. `flush` may be called repeatedly and must return prior deferred write errors consistently. `release` is idempotent and cannot manufacture durability. Recovery finishes before the mount readiness handshake.

### 3.5 ARC coherence

- Keys bind namespace, inode/content/version CID, generation, serializer, offset, and length.
- Only committed bytes enter the shared cache.
- Concurrent misses single-flight under the same generation.
- Write, truncate, unlink, rename, recovery replay, and backend reconciliation invalidate or advance exactly affected bindings.
- Persisted ARC is admitted only after WAL recovery; corrupt or stale state is a safe miss.

### 3.6 Linux fusepy profile

fusepy is loaded lazily because importing it loads a native library. The supported Linux profile uses fusepy's high-level FUSE 2.x ABI over the normal Linux FUSE kernel device. The doctor checks the Python binding, libfuse ABI, `/dev/fuse`, fusermount helper, mountpoint, permissions, and state-directory separation without mounting.

The default security profile enables kernel permission checking, keeps `allow_other` off, rejects arbitrary mount options, and runs with the mounting user's authority. `allow_other` requires explicit configuration and an operator-visible warning.

### 3.7 Windows WinFsp profile

The same operations object routes through fusepy's Windows loader to the WinFsp FUSE compatibility DLL. Loader resolution is deterministic: explicit `FUSE_LIBRARY_PATH`, then validated WinFsp registry installation, with Python/DLL architecture agreement. A missing driver, service, DLL, or incompatible ABI is a typed capability error within five seconds.

The Windows namespace contract preserves display spelling while using a collision-safe lookup identity when the selected WinFsp volume is case-insensitive; ambiguous case-fold collisions fail closed. Reserved device names, trailing dots/spaces, UTF-8/UTF-16 conversion, drive-letter versus directory mounts, open-delete sharing, ACL/ADS/reparse limitations, and case-only rename are executable tests. Windows support is advertised only after a live WinFsp receipt.

### 3.8 Containers

The normal image stays unprivileged. A dedicated Linux FUSE image/profile installs fusepy, libfuse2 compatibility, and fusermount, runs the mount in foreground/PID 1 aware mode, and persists WAL/cache/state separately. Docker's supported FUSE profile requires both `--device /dev/fuse` and `--cap-add SYS_ADMIN`; blanket `--privileged` is forbidden.

In-container access and host-visible propagation are separate claims. Host-visible native-Linux mounts require an explicitly shared bind mount; Docker Desktop propagation is not claimed. Windows containers remain conditional/experimental: process isolation plus a host-started WinFsp driver is required.

## 4. Supervisor goal tree

```text
KVFS-G000  Production kernel-mounted durable cached VFS
|-- KVFS-G100  Authority, contracts, fixtures, capability baseline
|-- KVFS-G200  Ranged storage, paths, metadata, handles, operations runtime
|-- KVFS-G300  Canonical WAL source, durability, replay, maintenance
|-- KVFS-G400  Generation-bound range ARC and coherence
|-- KVFS-G500  Linux fusepy loader, lifecycle, live conformance, soak
|-- KVFS-G600  Windows WinFsp loader, semantics, lifecycle, conformance
|-- KVFS-G700  Packaging, Docker profiles, CLI, observability
`-- KVFS-G800  Security, differential tests, CI, performance, release
```

The detailed machine-ingestible records are in the objective heap. The task board contains 40 fixed tasks, one completed control seal and 39 implementation/release tasks.

## 5. Parallel execution waves

Tasks are sharded by the supervisor's actual `sha256(task_id)[:8] % 4` rule. The four initial tasks cover all four shards.

```text
W0  KVFS-000
W1  KVFS-103 | KVFS-101 | KVFS-108 | KVFS-100
W2  KVFS-200 | KVFS-202 | KVFS-201 | KVFS-204
W3  KVFS-203 | KVFS-205 | KVFS-208 | KVFS-303 | KVFS-400 | KVFS-503
W4  KVFS-210 | KVFS-309 | KVFS-401 | KVFS-608
W5  KVFS-206 | KVFS-300 | KVFS-600
W6  KVFS-301 | KVFS-404 | KVFS-500
W7  KVFS-304 | KVFS-403 | KVFS-601 | KVFS-703
W8  KVFS-506 | KVFS-603 | KVFS-701 | KVFS-702
W9  KVFS-501 | KVFS-700 | KVFS-808 | KVFS-800
W10 KVFS-802 | KVFS-801
W11 KVFS-811
```

The dependency DAG, rather than wave prose, is authoritative. Independent files and platforms may run earlier when their declared prerequisites are satisfied.

## 6. Test and certification matrix

| Lane | Platforms | Frequency | Required evidence |
| --- | --- | --- | --- |
| Hermetic callbacks | Linux + Windows, Python 3.12/3.13 | every PR | callback/result/errno, flags, handles, paths, concurrency; no driver |
| Core VFS/WAL/ARC | Linux, Python 3.12/3.13 | every PR | existing readiness suites plus composed runtime crash matrix |
| Linux live mount | Ubuntu 22.04/24.04 AMD64 | every PR on capable runner | kernel CRUD, offset I/O, truncate, rename, fsync, kill/replay, unmount |
| Linux ARM64 | native ARM64 | nightly/release | ABI, concurrency, 100-cycle mount soak, bounded resources |
| Windows hermetic | hosted Windows x64 | every PR | loader, path/name policy, callback and error projection |
| Windows live | labeled self-hosted Windows x64 + pinned WinFsp | nightly/release | drive and directory mounts, PowerShell/Explorer CRUD, kill cleanup |
| Linux Docker | native Docker AMD64; ARM64 nightly | every PR/nightly | minimal capability positive/negative tests, restart, propagation profile |
| Packaging | Linux/macOS/Windows, Python 3.12/3.13 | release | default wheel inert; `[fuse]` extra; driver-present/absent doctor |
| Performance/chaos | Linux and Windows live runners | nightly/release | cold/warm reads, random/sequential I/O, metadata, WAL/ARC/handle bounds |

Real-mount tests use unique temporary mountpoints or leased drive letters, a 15-second readiness handshake, a 60-second per-case timeout, cleanup in `finally`, and an independent watchdog. Missing platform capability emits a bounded `capability_unavailable` receipt; it never leaves a task running.

## 7. SLO and safety floors

Release floors:

- acknowledged committed data loss: 0;
- duplicate non-idempotent replay effects: 0;
- stale ARC read after committed mutation/replay: 0;
- path traversal, symlink escape, reserved-name alias escape: 0;
- false-success errno translation: 0;
- leaked mount, drive letter, child process, handle, or state lease after test: 0;
- unbounded startup/doctor/mount/unmount operation: 0;
- core import requiring fusepy/libfuse/WinFsp: 0;
- blanket privileged container profile: 0.

Initial performance budgets are captured, not guessed, by KVFS-108. KVFS-801 freezes reviewed floors for metadata operations, sequential/random reads and writes, p95/p99 latency, ARC hit/miss behavior, WAL queue depth, memory, open handles, and mount cycles. Performance cannot be improved by weakening correctness, durability, permissions, or cache coherence.

## 8. Rollout and rollback

1. **Hermetic:** common callback/runtime tests, no native mount.
2. **Linux shadow:** read-only fixture mount, then writable temporary mount.
3. **Linux beta:** explicit CLI opt-in and container profile.
4. **Windows shadow:** hermetic contract on hosted runners; live WinFsp on labeled self-hosted runner.
5. **Windows beta:** explicit opt-in after current live receipt.
6. **Production:** terminal receipt binds source revisions, dependency/ABI versions, live receipts, benchmark floors, migration compatibility, and support matrix.

Rollback unmounts first, prevents new handles, drains or aborts bounded callbacks, fsyncs/records the current WAL position, preserves WAL/state for the prior compatible runtime, invalidates nonportable ARC state, and never deletes recovery data automatically.

## 9. Anti-stall supervisor policy

- The initial ready set contains one non-native task per strict shard: authority inventory, callback contract, fixtures, and bounded doctor/baseline.
- No initial task imports fusepy or requires `/dev/fuse`, WinFsp, Docker, credentials, or a network backend.
- Native tests first run a <=5 second capability probe. Missing capability produces a typed terminal receipt for that run and leaves other lanes runnable.
- Mount daemons are child processes with readiness/heartbeat files; workers never call a foreground blocking FUSE loop in-process.
- Provider, resource, task, mountpoint, state-directory, and drive-letter leases are bounded and visible in state.
- Objective/codebase refill is disabled for the sealed first projection, preventing generated work from rewriting protected plan/control artifacts.
- Four strict shards, four provider slots, bounded retries, log-stall recycling, and current task-state/heartbeat checks are encoded in the scheduler.

## 10. Upstream compatibility references

- Linux kernel FUSE overview and control filesystem: https://www.kernel.org/doc/html/latest/filesystems/fuse/fuse.html
- libfuse high-level callback model: https://github.com/libfuse/libfuse
- fusepy ctypes binding and WinFsp loader source: https://github.com/fusepy/fusepy
- WinFsp FUSE versions and mount differences: https://winfsp.dev/doc/Frequently-Asked-Questions/
- WinFsp native API versus FUSE semantics: https://winfsp.dev/doc/Native-API-vs-FUSE/
- WinFsp Windows-container constraints: https://winfsp.dev/doc/WinFsp-Container-Support/
- Docker FUSE device/capability requirements: https://docs.docker.com/engine/containers/run/

These references constrain the adapter and test plan; repository contracts and passing current-tree evidence remain completion authority.
