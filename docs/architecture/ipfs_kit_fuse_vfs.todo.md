# IPFS Kit Kernel VFS Taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `KVFS-`.
The plan, objective heap, scheduler configuration, and validator are protected
control inputs. This sealed projection contains 40 tasks over nine goals.
`KVFS-000` is complete; exactly `KVFS-100`, `KVFS-101`, `KVFS-103`, and
`KVFS-108` are initially ready and cover all four strict SHA-256 shards.

Implementation tasks modify the nested `ipfs_kit_py` repository from isolated
worktrees. Each must commit the nested change before the parent gitlink update.
Native probes and mounts are bounded child processes; capability absence is a
typed receipt, never an indefinitely running worker.

## KVFS-000 Seal the architecture, objective heap, task DAG, and scheduler controls

- Status: completed
- Completion: manual
- Is schedulable: false
- Review only: true
- Priority: P0
- Track: control
- Depends on:
- Goal id: KVFS-G000
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Outputs: docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md, docs/architecture/ipfs_kit_fuse_vfs.objectives.md, docs/architecture/ipfs_kit_fuse_vfs.todo.md, config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json, scripts/validate_ipfs_kit_fuse_vfs_board.py
- Validation: python scripts/validate_ipfs_kit_fuse_vfs_board.py --check-all
- Scope paths: docs/architecture, config, scripts/validate_ipfs_kit_fuse_vfs_board.py
- Conflict policy: Protected control artifacts are immutable to implementation lanes.
- Acceptance: The tracked plan is complete; all references resolve; the task DAG is acyclic; exact initial readiness and four-shard coverage validate; source revisions and provider route are sealed.

## KVFS-100 Select canonical VFS authority and compatibility disposition

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundation-authority
- Depends on: KVFS-000
- Goal id: KVFS-G100
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/foundation/authority
- Parallel lane: authority
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/docs/kernel_vfs/authority.md, ipfs_kit_py/tests/kernel_vfs/contracts/test_authority.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/contracts/test_authority.py
- Scope paths: ipfs_kit_py/docs/kernel_vfs, ipfs_kit_py/tests/kernel_vfs/contracts
- Conflict policy: Own the ADR and authority test only; inspect legacy services read-only.
- Acceptance: The ADR selects CanonicalVFSService as semantics authority; dispositions VFSCore, VFSManager, legacy journals, Python, CLI, MCP, and future FUSE callers; names the storage, WAL, and cache cutover; and tests that no advertised mutation bypasses it.

## KVFS-101 Define HostFilesystemAdapter callback, error, and lifecycle contracts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundation-contracts
- Depends on: KVFS-000
- Goal id: KVFS-G100
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/foundation/contracts
- Parallel lane: contracts
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/host_contracts.py, ipfs_kit_py/tests/kernel_vfs/contracts/test_host_contracts.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/contracts/test_host_contracts.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/host_contracts.py, ipfs_kit_py/tests/kernel_vfs/contracts
- Conflict policy: Own the inert versioned contract and tests; do not import fusepy or mutate service behavior.
- Acceptance: Finite records cover callback inputs/results, exact errno, flags, metadata, handles, durability modes, cache consistency, mount lifecycle, cancellation/deadline, Linux/Windows differences, and explicit ENOSYS/EOPNOTSUPP without false success.

## KVFS-103 Build hermetic callback, path, handle, and fault fixtures

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundation-fixtures
- Depends on: KVFS-000
- Goal id: KVFS-G100
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/foundation/fixtures
- Parallel lane: fixtures
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/tests/kernel_vfs/fixtures/manifest.json, ipfs_kit_py/tests/kernel_vfs/fixtures/test_fixture_manifest.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/fixtures/test_fixture_manifest.py
- Scope paths: ipfs_kit_py/tests/kernel_vfs/fixtures
- Conflict policy: Own inert bounded fixture data only; no native driver, credential, network, user path, or executable payload.
- Acceptance: Content-identified fixtures cover every callback, flag combination, traversal and Unicode/case edge, partial/sparse I/O, rename/unlink while open, concurrent faults, WAL crash points, corrupt ARC, WinFsp names, Docker capability failures, and exact expected traces.

## KVFS-108 Add bounded platform doctor and capture performance/resource baselines

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: foundation-baseline
- Depends on: KVFS-000
- Goal id: KVFS-G100
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/foundation/baseline
- Parallel lane: baseline
- Resource class: cpu-io-large
- Outputs: ipfs_kit_py/benchmarks/kernel_vfs/baseline.py, ipfs_kit_py/benchmarks/kernel_vfs/workloads.json, ipfs_kit_py/tests/kernel_vfs/platform/test_doctor_baseline.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/platform/test_doctor_baseline.py && python benchmarks/kernel_vfs/baseline.py --check-schema
- Scope paths: ipfs_kit_py/benchmarks/kernel_vfs, ipfs_kit_py/tests/kernel_vfs/platform
- Conflict policy: Measure and probe only; do not install drivers, mount, optimize, or change production imports.
- Acceptance: Doctor finishes within five seconds and records OS/architecture, Python binding, native ABI, device/driver/service, helper, mountpoint/state permissions, Docker capability, and actionable absence; baseline binds environment/workload/seed and cold/warm I/O, metadata, memory, handles, WAL, and ARC observations.

## KVFS-200 Implement persistent ranged VFS storage boundaries and backend adapters

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-storage
- Depends on: KVFS-100, KVFS-101
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/storage
- Parallel lane: common-storage
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/storage.py, ipfs_kit_py/tests/kernel_vfs/common/test_ranged_storage.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_ranged_storage.py tests/runtime_readiness/vfs
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/storage.py, ipfs_kit_py/tests/kernel_vfs/common/test_ranged_storage.py
- Conflict policy: Own the new boundary/adapters; do not cut over callers until KVFS-203.
- Acceptance: Memory, local, IPFS, and Iroh adapters expose confined stat/list/range-read/staged-write/delete/rename; files exceed 1 MiB without whole-object loading; immutable or unavailable backend operations reject explicitly; effects and versions are observable.

## KVFS-202 Implement namespace routing, mount table, stable inode, and path policy

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-namespace
- Depends on: KVFS-100, KVFS-101
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/namespace
- Parallel lane: common-namespace
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/namespace.py, ipfs_kit_py/tests/kernel_vfs/common/test_namespace.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_namespace.py tests/runtime_readiness/vfs
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/namespace.py, ipfs_kit_py/tests/kernel_vfs/common/test_namespace.py
- Conflict policy: Own namespace and mount routing only; metadata projection is KVFS-201.
- Acceptance: Longest-prefix mount resolution is deterministic; unknown or cross-mount mutation rejects; stable inode identity survives restart and same-mount rename; root confinement, Unicode normalization, symlink policy, pagination, and case policy have executable traces.

## KVFS-201 Add kernel metadata, stat, access, statfs, time, and unsupported-operation semantics

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-metadata
- Depends on: KVFS-101, KVFS-103
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/metadata
- Parallel lane: common-metadata
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/metadata.py, ipfs_kit_py/tests/kernel_vfs/common/test_metadata.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_metadata.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/metadata.py, ipfs_kit_py/tests/kernel_vfs/common/test_metadata.py
- Conflict policy: Own metadata types/projection; do not implement mount-specific callbacks.
- Acceptance: File type, mode, nlink, size, uid/gid policy, inode, atime/mtime/ctime, access, statfs, utimens and exact errors are deterministic; chmod/chown/xattr/link/symlink/mknod either have reviewed semantics or stable unsupported results.

## KVFS-204 Implement bounded file handles and per-handle staged extents

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-handles
- Depends on: KVFS-101, KVFS-103
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/handles
- Parallel lane: common-handles
- Resource class: cpu-memory-large
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/handles.py, ipfs_kit_py/tests/kernel_vfs/common/test_handles.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_handles.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/handles.py, ipfs_kit_py/tests/kernel_vfs/common/test_handles.py
- Conflict policy: Own handles and staging; shared cache admission and WAL effects remain out of scope.
- Acceptance: Generation-tagged bounded handles implement O_CREAT/O_EXCL/O_TRUNC/O_APPEND, random/sparse writes, read-own-writes, deferred errors, idempotent flush/release, stale-handle rejection, rename/unlink while open, orphan reclamation, and explicit pressure behavior.

## KVFS-203 Extend CanonicalVFSService over storage, namespace, metadata, and handles

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-service
- Depends on: KVFS-200, KVFS-201, KVFS-202, KVFS-204
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/service
- Parallel lane: common-service
- Resource class: cpu-io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/host_service.py, ipfs_kit_py/tests/kernel_vfs/common/test_host_service.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_host_service.py tests/runtime_readiness/vfs
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/host_service.py, ipfs_kit_py/tests/kernel_vfs/common/test_host_service.py
- Conflict policy: Integrate through a new host façade; preserve existing public contracts until differential cutover.
- Acceptance: Every supported host operation reaches CanonicalVFSService contracts with one path/result/error/effect authority; real storage is injected; create/read/write/truncate/list/mkdir/rmdir/unlink/rename/metadata work without a driver; legacy paths cannot bypass admitted mutations.

## KVFS-205 Implement offset I/O, sparse/partial write assembly, truncate, and append

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-io
- Depends on: KVFS-200, KVFS-204
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/io
- Parallel lane: common-io
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/io_runtime.py, ipfs_kit_py/tests/kernel_vfs/common/test_offset_io.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_offset_io.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/io_runtime.py, ipfs_kit_py/tests/kernel_vfs/common/test_offset_io.py
- Conflict policy: Own data-plane assembly; durability and cache admission are later tasks.
- Acceptance: Offset reads, overlapping/random/short writes, holes, append serialization, grow/shrink truncate, zero length, EOF, large files, and partial backend failures match the reference trace without loading unrelated ranges or leaking dirty bytes.

## KVFS-208 Add callback concurrency, lock ordering, open-unlink, and open-rename semantics

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-concurrency
- Depends on: KVFS-202, KVFS-204
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/concurrency
- Parallel lane: common-concurrency
- Resource class: cpu-large
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/host_concurrency.py, ipfs_kit_py/tests/kernel_vfs/common/test_concurrency.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_concurrency.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/vfs/host_concurrency.py, ipfs_kit_py/tests/kernel_vfs/common/test_concurrency.py
- Conflict policy: Own lock/lease primitives; mutations integrate them in KVFS-206 and KVFS-309.
- Acceptance: Deterministic inode/path/handle lock ordering prevents deadlock; callbacks are linearizable or return typed conflict; open handles survive same-mount rename/unlink per policy; tables, queues, waits, cancellation, and shutdown are bounded under randomized concurrency.

## KVFS-210 Add a bounded async bridge for synchronous fusepy callbacks

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: common-async
- Depends on: KVFS-203
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/async
- Parallel lane: common-async
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/async_bridge.py, ipfs_kit_py/tests/kernel_vfs/common/test_async_bridge.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_async_bridge.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/async_bridge.py, ipfs_kit_py/tests/kernel_vfs/common/test_async_bridge.py
- Conflict policy: Own only the sync/async bridge and lifecycle tests.
- Acceptance: One bounded owner loop executes async services from concurrent synchronous callbacks with deadlines, cancellation, context/error preservation, backpressure, reentrant-call rejection, deterministic close, no per-call loop creation, and no orphan tasks or threads.

## KVFS-206 Implement the platform-neutral KernelVFSOperations and composed request runtime

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: common-operations
- Depends on: KVFS-203, KVFS-205, KVFS-208, KVFS-210
- Goal id: KVFS-G200
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/common/operations
- Parallel lane: common-operations
- Resource class: cpu-io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/operations.py, ipfs_kit_py/tests/kernel_vfs/common/test_operations.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/common/test_operations.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/operations.py, ipfs_kit_py/tests/kernel_vfs/common/test_operations.py
- Conflict policy: Own the driver-independent operations class; native loaders are separate.
- Acceptance: Direct tests exercise getattr/readdir/access/statfs/utimens/open/create/read/write/truncate/flush/fsync/release/mkdir/rmdir/unlink/rename/init/destroy; every result and errno matches the contract; unsupported callbacks reject; no fuse import or native side effect occurs.

## KVFS-303 Make canonical WAL records the recoverable transaction source of truth

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-records
- Depends on: KVFS-100, KVFS-101
- Goal id: KVFS-G300
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/wal/records
- Parallel lane: wal-records
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/core/wal/vfs_records.py, ipfs_kit_py/tests/kernel_vfs/wal/test_vfs_records.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/wal/test_vfs_records.py tests/runtime_readiness/wal
- Scope paths: ipfs_kit_py/ipfs_kit_py/core/wal/vfs_records.py, ipfs_kit_py/tests/kernel_vfs/wal/test_vfs_records.py
- Conflict policy: Own new WAL record/payload schemas and compatibility reader; do not bind live mutations.
- Acceptance: Canonical durable data contains transaction/operation/effect IDs, intent, bounded inline payload or staged content reference, checksum, preconditions, decision, and acknowledgement; marker-to-sidecar crash gaps are covered; corrupt tail preserves valid prefix; secrets and unbounded data reject.

## KVFS-309 Bind staged VFS mutation effects to the WAL coordinator

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-runtime
- Depends on: KVFS-203, KVFS-205, KVFS-303
- Goal id: KVFS-G300
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/wal/runtime
- Parallel lane: wal-runtime
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/durable_mutation.py, ipfs_kit_py/tests/kernel_vfs/wal/test_durable_mutation.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/wal/test_durable_mutation.py tests/runtime_readiness/wal
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/durable_mutation.py, ipfs_kit_py/tests/kernel_vfs/wal/test_durable_mutation.py
- Conflict policy: Own mutation ordering façade; ARC events are consumed later by KVFS-404.
- Acceptance: Validate/authorize/lock precedes durable intent; effect follows required intent durability; decision and effect identity are durable before committed acknowledgement; create/write/truncate/unlink/rename have idempotent apply/compensate behavior and exact partial-effect receipts.

## KVFS-300 Map flush, fsync, release, and deferred errors to durability receipts

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-fsync
- Depends on: KVFS-309
- Goal id: KVFS-G300
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/wal/fsync
- Parallel lane: wal-fsync
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/durability.py, ipfs_kit_py/tests/kernel_vfs/wal/test_durability_callbacks.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/wal/test_durability_callbacks.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/durability.py, ipfs_kit_py/tests/kernel_vfs/wal/test_durability_callbacks.py
- Conflict policy: Own callback durability modes and receipts; do not change native lifecycle.
- Acceptance: fsync waits for configured WAL and backend file/parent-directory durability; flush is repeatable and reports deferred errors consistently; release is idempotent and creates no false durability; timeout/cancel/ENOSPC/EIO traces never acknowledge lost data.

## KVFS-301 Implement pre-ready recovery, idempotent replay, orphan-stage reclamation, and leases

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: wal-recovery
- Depends on: KVFS-208, KVFS-300, KVFS-309
- Goal id: KVFS-G300
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/wal/recovery
- Parallel lane: wal-recovery
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/wal_recovery.py, ipfs_kit_py/tests/kernel_vfs/wal/test_mount_recovery.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/wal/test_mount_recovery.py tests/runtime_readiness/wal
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/wal_recovery.py, ipfs_kit_py/tests/kernel_vfs/wal/test_mount_recovery.py
- Conflict policy: Own recovery/startup and state leases; native launchers call the API later.
- Acceptance: A single-writer state lease fences concurrent mounts; recovery completes before ready; repeated restart applies committed effects exactly once, resolves incomplete transactions per policy, reclaims only provably orphaned stages/handles, preserves evidence on error, and terminates within a declared bound.

## KVFS-304 Add checkpoint, compaction, archive, and bounded maintenance lifecycle

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: wal-maintenance
- Depends on: KVFS-301
- Goal id: KVFS-G300
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/wal/maintenance
- Parallel lane: wal-maintenance
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/wal_maintenance.py, ipfs_kit_py/tests/kernel_vfs/wal/test_maintenance.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/wal/test_maintenance.py tests/runtime_readiness/wal
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/wal_maintenance.py, ipfs_kit_py/tests/kernel_vfs/wal/test_maintenance.py
- Conflict policy: Own mount-scoped maintenance orchestration; canonical WAL primitives remain compatible.
- Acceptance: Checkpoints cannot hide later appends; compaction retains recovery closure; archive is verified before source deletion; disk pressure applies explicit backpressure; workers heartbeat and stop; mount shutdown preserves the latest durable recovery position.

## KVFS-400 Add generation-bound range/chunk ARC keys and generation-aware single-flight

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: arc-bindings
- Depends on: KVFS-101, KVFS-103, KVFS-200
- Goal id: KVFS-G400
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/arc/bindings
- Parallel lane: arc-bindings
- Resource class: cpu-memory-large
- Outputs: ipfs_kit_py/ipfs_kit_py/cache/arc/range_bindings.py, ipfs_kit_py/tests/kernel_vfs/arc/test_range_bindings.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/arc/test_range_bindings.py tests/runtime_readiness/arc
- Scope paths: ipfs_kit_py/ipfs_kit_py/cache/arc/range_bindings.py, ipfs_kit_py/tests/kernel_vfs/arc/test_range_bindings.py
- Conflict policy: Own new range binding and single-flight types; no VFS cutover.
- Acceptance: Keys bind namespace, inode/content/version, generation, serializer, offset, and length; overlapping and exact-range policy is deterministic; concurrent misses single-flight only under equal generation; cancellation/error fan-out is bounded; ARC byte and ghost invariants remain valid.

## KVFS-401 Implement committed read-through and bounded range admission

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: arc-readthrough
- Depends on: KVFS-203, KVFS-400
- Goal id: KVFS-G400
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/arc/readthrough
- Parallel lane: arc-readthrough
- Resource class: cpu-memory-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/cached_storage.py, ipfs_kit_py/tests/kernel_vfs/arc/test_cached_storage.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/arc/test_cached_storage.py tests/runtime_readiness/arc
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/cached_storage.py, ipfs_kit_py/tests/kernel_vfs/arc/test_cached_storage.py
- Conflict policy: Own read path/admission; mutation invalidation belongs to KVFS-404.
- Acceptance: Cache hits revalidate exact bindings; misses fetch only requested bounded ranges; dirty staged bytes never enter shared ARC; policy/authorization-sensitive scopes cannot alias; oversized ranges bypass or segment predictably; errors and corrupt entries become safe misses.

## KVFS-404 Drive exact ARC invalidation/generation advance from mutation and replay

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: arc-coherence
- Depends on: KVFS-301, KVFS-309, KVFS-401
- Goal id: KVFS-G400
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/arc/coherence
- Parallel lane: arc-coherence
- Resource class: cpu-memory-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/cache_coherence.py, ipfs_kit_py/tests/kernel_vfs/arc/test_coherence.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/arc/test_coherence.py tests/runtime_readiness/arc
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/cache_coherence.py, ipfs_kit_py/tests/kernel_vfs/arc/test_coherence.py
- Conflict policy: Own durable event-to-cache projection; do not modify core mutation ordering.
- Acceptance: Committed create/replace/write/truncate/unlink/rename and recovery replay advance or invalidate exactly affected bindings before new admission; unrelated data remains; aborted/failed effects do not publish; randomized interleavings return no stale committed byte.

## KVFS-403 Define post-recovery ARC persistence, corruption policy, and metrics

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: arc-persistence
- Depends on: KVFS-304, KVFS-400, KVFS-404
- Goal id: KVFS-G400
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/arc/persistence
- Parallel lane: arc-persistence
- Resource class: cpu-memory-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/cache_state.py, ipfs_kit_py/tests/kernel_vfs/arc/test_cache_state.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/arc/test_cache_state.py tests/runtime_readiness/arc
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/cache_state.py, ipfs_kit_py/tests/kernel_vfs/arc/test_cache_state.py
- Conflict policy: Own cache startup/persistence/metrics integration only.
- Acceptance: WAL recovery precedes cache admission; persisted entries require compatible schema, revision, namespace, generation and checksums; stale/corrupt state safely misses; atomic persistence and bounded startup/shutdown work; hits/misses/evictions/bytes/single-flight/invalidation expose low-cardinality metrics.

## KVFS-503 Add lazy fusepy/libfuse loading and bounded Linux capability doctor

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: linux-loader
- Depends on: KVFS-100, KVFS-108
- Goal id: KVFS-G500
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/linux/loader
- Parallel lane: linux-loader
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/platform.py, ipfs_kit_py/tests/kernel_vfs/linux/test_loader_doctor.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/linux/test_loader_doctor.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/platform.py, ipfs_kit_py/tests/kernel_vfs/linux/test_loader_doctor.py
- Conflict policy: Own native-loading seam and Linux probe; packaging is KVFS-703.
- Acceptance: Core import is inert; binding/native library load is explicit and architecture-aware; doctor checks fusepy, libfuse2 ABI, /dev/fuse, fusermount helper, permissions, mountpoint and state separation within five seconds; absence raises typed actionable capability error without mounting.

## KVFS-500 Implement Linux mount lifecycle, readiness, heartbeat, signal, and unmount

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: linux-lifecycle
- Depends on: KVFS-206, KVFS-300, KVFS-404, KVFS-503
- Goal id: KVFS-G500
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/linux/lifecycle
- Parallel lane: linux-lifecycle
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/linux.py, ipfs_kit_py/tests/kernel_vfs/linux/test_lifecycle.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/linux/test_lifecycle.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/linux.py, ipfs_kit_py/tests/kernel_vfs/linux/test_lifecycle.py
- Conflict policy: Own Linux launcher/process lifecycle; live mountpoints require exclusive leases.
- Acceptance: Foreground child recovery precedes ready; readiness arrives within 15 seconds or exits nonzero; heartbeat/status bind PID/mount/state/WAL/cache; SIGINT/SIGTERM and repeated unmount drain bounded callbacks, stop workers, release mount/lease, preserve recovery state, and report stale mounts without blocking.

## KVFS-506 Build bounded real Linux kernel-mount conformance and crash harness

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: linux-conformance
- Depends on: KVFS-301, KVFS-304, KVFS-403, KVFS-500
- Goal id: KVFS-G500
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/linux/conformance
- Parallel lane: linux-conformance
- Resource class: io-large
- Outputs: ipfs_kit_py/tests/kernel_vfs/linux/test_live_mount.py, ipfs_kit_py/tests/kernel_vfs/linux/live_harness.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/linux/test_live_mount.py
- Scope paths: ipfs_kit_py/tests/kernel_vfs/linux/test_live_mount.py, ipfs_kit_py/tests/kernel_vfs/linux/live_harness.py
- Conflict policy: Own tests/harness only; use unique mount/state leases and cleanup watchdog.
- Acceptance: On a capable runner, kernel CRUD, flags, offset/sparse I/O, truncate, metadata, concurrent handles, unlink/rename, fsync, forced kill, replay, ARC coherence and unmount pass; readiness is 15 seconds, each case 60 seconds, cleanup is finally plus watchdog; absent capability emits bounded capability_unavailable evidence.

## KVFS-501 Certify Linux ARM64 and repeated mount/resource soak

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: linux-soak
- Depends on: KVFS-506
- Goal id: KVFS-G500
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/linux/soak
- Parallel lane: linux-soak
- Resource class: io-large
- Outputs: ipfs_kit_py/tests/kernel_vfs/linux/test_arm64_soak.py, ipfs_kit_py/benchmarks/kernel_vfs/linux_soak.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/linux/test_arm64_soak.py
- Scope paths: ipfs_kit_py/tests/kernel_vfs/linux/test_arm64_soak.py, ipfs_kit_py/benchmarks/kernel_vfs/linux_soak.py
- Conflict policy: Own soak evidence only; require labeled native capability and exclusive mount lease.
- Acceptance: Native ARM64 ABI and concurrency pass; 100 mount/unmount and crash/recover cycles show zero leaked process/mount/handle/lease, bounded WAL/cache/memory/descriptors, no stale read or lost acknowledgement; capability absence is a finite nonpromotion receipt.

## KVFS-608 Add deterministic WinFsp/fusepy loader and bounded Windows doctor

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: windows-loader
- Depends on: KVFS-100, KVFS-108, KVFS-503
- Goal id: KVFS-G600
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/windows/loader
- Parallel lane: windows-loader
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/winfsp_loader.py, ipfs_kit_py/tests/kernel_vfs/windows/test_loader_doctor.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/windows/test_loader_doctor.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/winfsp_loader.py, ipfs_kit_py/tests/kernel_vfs/windows/test_loader_doctor.py
- Conflict policy: Own Windows loader/doctor; do not install or start WinFsp.
- Acceptance: Explicit FUSE_LIBRARY_PATH then validated WinFsp registry lookup resolves the matching x86/x64 DLL; service/driver/DLL/version/architecture and drive/directory prerequisites are diagnosed within five seconds; missing or incompatible native support is typed and core imports remain inert.

## KVFS-600 Define Windows namespace, case/name, permission, and open/delete semantics

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: windows-semantics
- Depends on: KVFS-201, KVFS-202, KVFS-608
- Goal id: KVFS-G600
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/windows/semantics
- Parallel lane: windows-semantics
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/windows_semantics.py, ipfs_kit_py/tests/kernel_vfs/windows/test_semantics.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/windows/test_semantics.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/windows_semantics.py, ipfs_kit_py/tests/kernel_vfs/windows/test_semantics.py
- Conflict policy: Own pure Windows policy/projection; lifecycle is KVFS-601.
- Acceptance: Collision-safe lookup preserves display spelling; ambiguous folds, reserved device names, trailing dots/spaces, invalid UTF conversion and traversal reject; case-only rename, drive/directory roots, delete/share/rename while open, uid/gid/mode projection, ACL/ADS/reparse/symlink limits and errno behavior are executable.

## KVFS-601 Implement WinFsp drive/directory mount lifecycle and cleanup

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: windows-lifecycle
- Depends on: KVFS-206, KVFS-300, KVFS-301, KVFS-404, KVFS-600
- Goal id: KVFS-G600
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/windows/lifecycle
- Parallel lane: windows-lifecycle
- Resource class: io-large
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/windows.py, ipfs_kit_py/tests/kernel_vfs/windows/test_lifecycle.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/windows/test_lifecycle.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/kernel_vfs/windows.py, ipfs_kit_py/tests/kernel_vfs/windows/test_lifecycle.py
- Conflict policy: Own Windows launcher/process lifecycle; use exclusive drive/directory and state leases.
- Acceptance: Same operations object mounts through WinFsp FUSE compatibility; recovery precedes 15-second readiness; drive-letter and directory forms validate; status/heartbeat bind resources; stop/crash/repeated unmount release drive/directory/process/lease and preserve WAL state without a foreground worker hang.

## KVFS-603 Build real WinFsp conformance with PowerShell/Explorer-compatible operations

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: windows-conformance
- Depends on: KVFS-403, KVFS-601
- Goal id: KVFS-G600
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/windows/conformance
- Parallel lane: windows-conformance
- Resource class: io-large
- Outputs: ipfs_kit_py/tests/kernel_vfs/windows/test_live_winfsp.py, ipfs_kit_py/tests/kernel_vfs/windows/live_harness.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/windows/test_live_winfsp.py
- Scope paths: ipfs_kit_py/tests/kernel_vfs/windows/test_live_winfsp.py, ipfs_kit_py/tests/kernel_vfs/windows/live_harness.py
- Conflict policy: Own tests/harness; require labeled WinFsp runner and exclusive drive/directory lease.
- Acceptance: Pinned WinFsp x64 live receipts cover PowerShell and Explorer-compatible CRUD, random I/O, metadata, Unicode/case, concurrent open/delete/rename, fsync, forced crash/recovery, ARC coherence, drive/directory cleanup; each case is bounded and absent capability cannot promote support.

## KVFS-703 Add optional FUSE packaging, guarded imports, classifiers, and wheel probes

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: packaging
- Depends on: KVFS-503, KVFS-608
- Goal id: KVFS-G700
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/delivery/packaging
- Parallel lane: packaging
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/pyproject.toml, ipfs_kit_py/tests/kernel_vfs/packaging/test_wheels.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/packaging/test_wheels.py
- Scope paths: ipfs_kit_py/pyproject.toml, ipfs_kit_py/tests/kernel_vfs/packaging
- Conflict policy: Own dependency/entry-point/classifier projections; native support tier remains receipt-gated.
- Acceptance: Default wheel imports without fusepy/libfuse/WinFsp and no native side effect; a pinned [fuse] extra installs the binding; mount CLI is discoverable; missing driver is diagnostic; Python 3.12/3.13 Linux/Windows wheel probes pass; Windows classifier is conditional on live gate policy.

## KVFS-701 Build a dedicated minimally privileged Linux FUSE image and Compose profile

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: container-image
- Depends on: KVFS-500, KVFS-703
- Goal id: KVFS-G700
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/delivery/container
- Parallel lane: container-image
- Resource class: docker-large
- Outputs: ipfs_kit_py/docker/kernel-vfs.Dockerfile, ipfs_kit_py/docker-compose.kernel-vfs.yml, ipfs_kit_py/tests/kernel_vfs/container/test_image_profile.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/container/test_image_profile.py
- Scope paths: ipfs_kit_py/docker/kernel-vfs.Dockerfile, ipfs_kit_py/docker-compose.kernel-vfs.yml, ipfs_kit_py/tests/kernel_vfs/container/test_image_profile.py
- Conflict policy: Own dedicated image/profile only; keep normal image and service unprivileged.
- Acceptance: Python floor, fuse extra, libfuse2 compatibility and fusermount are reproducible; mount runs foreground with readiness; WAL/cache/state volumes are separate; profile requires /dev/fuse and SYS_ADMIN but never privileged; either missing input fails within five seconds.

## KVFS-700 Add positive/negative Docker mount, restart, and propagation conformance

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: container-conformance
- Depends on: KVFS-506, KVFS-701
- Goal id: KVFS-G700
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/delivery/container-tests
- Parallel lane: container-tests
- Resource class: docker-large
- Outputs: ipfs_kit_py/tests/kernel_vfs/container/test_live_container.py, ipfs_kit_py/tests/kernel_vfs/container/test_propagation.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/container/test_live_container.py tests/kernel_vfs/container/test_propagation.py
- Scope paths: ipfs_kit_py/tests/kernel_vfs/container/test_live_container.py, ipfs_kit_py/tests/kernel_vfs/container/test_propagation.py
- Conflict policy: Own container harness; exclusive container/mount/state leases and bounded cleanup required.
- Acceptance: Minimal-capability mount passes in-container CRUD/fsync/restart/recovery; absent device and absent capability each fail promptly; native Linux rshared host propagation is a distinct tested profile; Docker Desktop propagation is not claimed; no container, process, mount, volume lease, or privileged profile leaks.

## KVFS-702 Add mount/doctor/status/unmount CLI, status schema, and observability

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: operations-cli
- Depends on: KVFS-500, KVFS-601, KVFS-703
- Goal id: KVFS-G700
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/delivery/cli
- Parallel lane: operations-cli
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/ipfs_kit_py/cli/kernel_vfs.py, ipfs_kit_py/ipfs_kit_py/kernel_vfs/status.py, ipfs_kit_py/tests/kernel_vfs/cli/test_cli.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/cli/test_cli.py
- Scope paths: ipfs_kit_py/ipfs_kit_py/cli/kernel_vfs.py, ipfs_kit_py/ipfs_kit_py/kernel_vfs/status.py, ipfs_kit_py/tests/kernel_vfs/cli/test_cli.py
- Conflict policy: Own CLI/status surfaces; invoke platform lifecycle APIs without duplicating semantics.
- Acceptance: doctor/mount/unmount/status have machine JSON and human output, foreground mode, explicit safe options, bounded readiness/stop timeouts, PID/lease validation and idempotent cleanup; status exposes platform, mount, recovery/WAL, ARC, handles, errors and heartbeat without secrets or high-cardinality paths.

## KVFS-808 Close mount option, permission, path, state, symlink, and resource security boundaries

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-security
- Depends on: KVFS-500, KVFS-601
- Goal id: KVFS-G800
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/release/security
- Parallel lane: release-security
- Resource class: cpu-large
- Outputs: ipfs_kit_py/tests/kernel_vfs/security/test_security_boundaries.py, ipfs_kit_py/docs/kernel_vfs/security.md
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/security/test_security_boundaries.py
- Scope paths: ipfs_kit_py/tests/kernel_vfs/security, ipfs_kit_py/docs/kernel_vfs/security.md
- Conflict policy: Own adversarial tests/security policy; production fixes must respect owning subsystem contracts.
- Acceptance: Traversal, symlink escape, Unicode/case/reserved aliases, mount-option injection, unsafe allow_other, state/mount overlap, stale PID/lease, permission confusion, oversized request, handle/WAL/ARC exhaustion, malformed native error, secret/log leakage and cleanup attacks fail closed with zero side effect outside admitted roots.

## KVFS-800 Add model-based, differential, and property tests across service and host operations

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-differential
- Depends on: KVFS-206, KVFS-301, KVFS-403, KVFS-600
- Goal id: KVFS-G800
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/release/differential
- Parallel lane: release-differential
- Resource class: cpu-large
- Outputs: ipfs_kit_py/tests/kernel_vfs/model.py, ipfs_kit_py/tests/kernel_vfs/test_differential.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/test_differential.py
- Scope paths: ipfs_kit_py/tests/kernel_vfs/model.py, ipfs_kit_py/tests/kernel_vfs/test_differential.py
- Conflict policy: Own independent model/generators; do not encode production implementation as oracle.
- Acceptance: Generated sequential/concurrent traces compare CanonicalVFSService, KernelVFSOperations and platform projections for state/result/errno/effect identity; flags, ranges, metadata, rename/unlink, crash/replay, ARC and Windows names shrink reproducibly; legacy compatibility has explicit differential dispositions.

## KVFS-802 Add mandatory hermetic, Linux, Windows, Docker, packaging, and path-trigger CI gates

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: release-ci
- Depends on: KVFS-506, KVFS-603, KVFS-700, KVFS-800
- Goal id: KVFS-G800
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/release/ci
- Parallel lane: release-ci
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/.github/workflows/kernel-vfs.yml, ipfs_kit_py/tests/kernel_vfs/test_ci_contract.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/test_ci_contract.py
- Scope paths: ipfs_kit_py/.github/workflows/kernel-vfs.yml, ipfs_kit_py/tests/kernel_vfs/test_ci_contract.py
- Conflict policy: Own the new workflow/gate contract; native jobs use explicit labeled capabilities.
- Acceptance: Core/VFS/WAL/ARC/FUSE/packaging/Docker paths trigger; Python 3.12/3.13 Ubuntu and Windows hermetic tests are mandatory; capable Linux, self-hosted WinFsp and Docker lanes emit receipts; zero collection, skip-only, stale receipt, permissive continue-on-error or `|| true` cannot pass a support gate.

## KVFS-801 Establish performance, chaos, saturation, and resource-leak release floors

- Status: completed
- Completion: auto
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: release-performance
- Depends on: KVFS-403, KVFS-506, KVFS-603, KVFS-700
- Goal id: KVFS-G800
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/release/performance
- Parallel lane: release-performance
- Resource class: cpu-io-large
- Outputs: ipfs_kit_py/benchmarks/kernel_vfs/run.py, ipfs_kit_py/benchmarks/kernel_vfs/reviewed_floors.json, ipfs_kit_py/tests/kernel_vfs/test_chaos_floors.py
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/test_chaos_floors.py && python benchmarks/kernel_vfs/run.py --check-reviewed-floors
- Scope paths: ipfs_kit_py/benchmarks/kernel_vfs/run.py, ipfs_kit_py/benchmarks/kernel_vfs/reviewed_floors.json, ipfs_kit_py/tests/kernel_vfs/test_chaos_floors.py
- Conflict policy: Own workloads/floors; a failing change cannot lower its own floor or weaken correctness.
- Acceptance: Reviewed environments and workloads bind cold/warm metadata, sequential/random read/write, p95/p99, committed throughput, ARC ratios, WAL queue, memory, descriptors, handles and mount cycles; kill/torn/corrupt/ENOSPC/backpressure chaos meets zero safety floors and bounded degradation.

## KVFS-811 Publish migration, support matrix, operations/rollback guide, and joined release receipt

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: release-terminal
- Depends on: KVFS-501, KVFS-603, KVFS-700, KVFS-702, KVFS-808, KVFS-802, KVFS-800, KVFS-801
- Goal id: KVFS-G800
- Board namespace: ipfs-kit-kernel-vfs-fuse-v1
- Bundle: ipfs-kit/kernel-vfs/release/terminal
- Parallel lane: release-terminal
- Resource class: cpu-medium
- Outputs: ipfs_kit_py/docs/kernel_vfs/operations.md, ipfs_kit_py/docs/kernel_vfs/migration.md, ipfs_kit_py/docs/kernel_vfs/support_matrix.json, ipfs_kit_py/docs/kernel_vfs/release_receipt.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs && python benchmarks/kernel_vfs/run.py --check-reviewed-floors
- Scope paths: ipfs_kit_py/docs/kernel_vfs
- Conflict policy: Aggregate current immutable evidence only; do not implement or weaken a gate at release.
- Acceptance: Guide covers install, doctor, Linux/Windows/container mount/unmount/status, options, limitations, monitoring, backup/recovery, VFSCore migration, downgrade and rollback; support matrix separates hermetic/conditional/live claims; reviewed receipt binds exact source/dependency/ABI/driver/environment/test/benchmark evidence and all safety floors before promotion.

## KVFS-812 Resolve validation retry-budget failure for KVFS-101

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: KVFS-000
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/host_contracts.py, ipfs_kit_py/tests/kernel_vfs/contracts/test_host_contracts.py
- Validation: test -f /home/barberb/lift_coding/.worktrees/ipfs-kit-fuse-vfs-supervisor/data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery/2026-08-09-kvfs-812-kvfs-101-retry-budget.md
- Parallel lane: contracts
- Conflict policy: Own the inert versioned contract and tests; do not import fusepy or mutate service behavior.
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: KVFS-101
- Retry failure kind: validation
- Retry repair discovery: /home/barberb/lift_coding/.worktrees/ipfs-kit-fuse-vfs-supervisor/data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery/2026-08-09-kvfs-812-kvfs-101-retry-budget.md
- Canonical board task: false

- Acceptance: Retry-budget guardrail filed this from repeated validation failures in KVFS-101. Use evidence in /home/barberb/lift_coding/.worktrees/ipfs-kit-fuse-vfs-supervisor/data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery/2026-08-09-kvfs-812-kvfs-101-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release KVFS-101 from strategy blocked_tasks.

## KVFS-813 Resolve validation retry-budget failure for KVFS-103

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: KVFS-000
- Outputs: ipfs_kit_py/tests/kernel_vfs/fixtures/manifest.json, ipfs_kit_py/tests/kernel_vfs/fixtures/test_fixture_manifest.py
- Validation: PYTHONPATH=ipfs_kit_py cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/fixtures/test_fixture_manifest.py
- Parallel lane: fixtures
- Conflict policy: Own inert bounded fixture data only; no native driver, credential, network, user path, or executable payload.
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: KVFS-103
- Retry failure kind: validation
- Retry repair discovery: /home/barberb/lift_coding/.worktrees/ipfs-kit-fuse-vfs-supervisor/data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery/2026-08-09-kvfs-813-kvfs-103-retry-budget.md
- Canonical board task: false
- Validation failure paths: tests/kernel_vfs/fixtures/test_fixture_manifest.py, ipfs_kit_py/tests/kernel_vfs/fixtures/test_fixture_manifest.py
- Validation failure path authority: diagnostic-read-only
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in KVFS-103. Use evidence in /home/barberb/lift_coding/.worktrees/ipfs-kit-fuse-vfs-supervisor/data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery/2026-08-09-kvfs-813-kvfs-103-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release KVFS-103 from strategy blocked_tasks. The declared validation failure paths (tests/kernel_vfs/fixtures/test_fixture_manifest.py, ipfs_kit_py/tests/kernel_vfs/fixtures/test_fixture_manifest.py) are bounded diagnostic/read-only metadata: they may be inspected and used to focus validation, but do not grant write authority. Repair edits remain limited to the source task Outputs; do not weaken correct assertions or policy.
