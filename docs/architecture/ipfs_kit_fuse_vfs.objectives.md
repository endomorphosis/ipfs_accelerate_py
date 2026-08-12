# IPFS Kit Kernel VFS Objective Heap

This is the durable goal hierarchy for the `KVFS-` supervisor program. The
normative architecture is `IPFS_KIT_FUSE_VFS_PLAN.md`; executable work is in
`ipfs_kit_fuse_vfs.todo.md`.

Program invariants:

- `CanonicalVFSService` is the sole semantic authority exposed to the host;
- every acknowledged mutation crosses one canonical WAL durability boundary;
- shared ARC entries bind committed content/version and generation;
- fusepy is a lazy, thin callback adapter, never a second VFS;
- unavailable native capability fails or defers within a bounded probe; and
- production support requires current Linux, Windows, or container receipts.

## KVFS-G000 Deliver a production kernel-mounted durable cached IPFS Kit VFS

- Status: active
- Parent:
- Depends on:
- Fib priority: 1
- Track: kernel-vfs
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/control
- Goal: Expose the canonical VFS, WAL, and generation-bound ARC through fusepy on Linux and WinFsp on Windows, with a minimally privileged Linux container profile and evidence-bound support claims.
- Subgoals: KVFS-G100, KVFS-G200, KVFS-G300, KVFS-G400, KVFS-G500, KVFS-G600, KVFS-G700, KVFS-G800
- Evidence: KVFS-G100, KVFS-G200, KVFS-G300, KVFS-G400, KVFS-G500, KVFS-G600, KVFS-G700, KVFS-G800
- Outputs: docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md, docs/architecture/ipfs_kit_fuse_vfs.objectives.md, docs/architecture/ipfs_kit_fuse_vfs.todo.md, config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json
- Validation: python scripts/validate_ipfs_kit_fuse_vfs_board.py --check-all
- Acceptance: All 40 tasks and eight child goals are terminal; acknowledged data loss, duplicate effects, stale committed cache reads, path escape, false success, leaked native resources, and unbounded lifecycle operations remain at zero; support claims bind current platform receipts.
- Gap task: Aggregate child evidence and make the release decision without implementing subsystem behavior at the root.
- Refinement: Conditional, experimental, unsupported, or capability-unavailable is preferred to an unearned support claim.
- Conflict policy: Root owns planning and terminal aggregation only; implementation is owned by child goals.

## KVFS-G100 Freeze authority, host contracts, fixtures, and capability baseline

- Status: active
- Parent: KVFS-G000
- Depends on:
- Fib priority: 1
- Track: foundations
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/foundations
- Goal: Resolve VFS authority, define the versioned host-filesystem callback/error/lifecycle contract, build hermetic fixtures, and record bounded platform and performance baselines.
- Evidence: KVFS-100, KVFS-101, KVFS-103, KVFS-108
- Outputs: ipfs_kit_py/docs/kernel_vfs, ipfs_kit_py/tests/kernel_vfs/fixtures, ipfs_kit_py/benchmarks/kernel_vfs
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/contracts tests/kernel_vfs/platform
- Acceptance: One authority and compatibility disposition is reviewed; every callback is supported or explicitly rejected; fixtures cover flags, errno, paths, handles, faults, and platform differences; doctor and baseline finish without native mounting.
- Gap task: Freeze the semantic and evidence boundary before production implementation.
- Refinement: Import, registry, documentation, or mock success does not establish native capability.
- Conflict policy: Foundation tasks own new ADR, contract, fixture, and probe files and do not refactor runtime code.

## KVFS-G200 Build the common ranged, handle-aware host VFS runtime

- Status: active
- Parent: KVFS-G000
- Depends on: KVFS-G100
- Fib priority: 2
- Track: common-runtime
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/common
- Goal: Extend the canonical VFS with persistent ranged storage, mount routing, stable metadata, file handles, partial I/O, concurrency, and a platform-neutral operations adapter.
- Evidence: KVFS-200, KVFS-202, KVFS-201, KVFS-204, KVFS-203, KVFS-205, KVFS-208, KVFS-210, KVFS-206
- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs, ipfs_kit_py/ipfs_kit_py/kernel_vfs/common, ipfs_kit_py/tests/kernel_vfs/common
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/vfs tests/kernel_vfs/common
- Acceptance: Large and ranged I/O, metadata, flags, handles, namespace transitions, concurrency, and callback/errno projection pass deterministic and model-based tests without a native driver.
- Gap task: Make the canonical service kernel-shaped while retaining one operation/error authority.
- Refinement: A callback result without the admitted observed state transition is failure.
- Conflict policy: Common contracts precede service/operations integration; compatibility cutover is serialized.

## KVFS-G300 Bind mutations to canonical WAL durability and recovery

- Status: active
- Parent: KVFS-G000
- Depends on: KVFS-G100
- Fib priority: 1
- Track: wal-durability
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/wal
- Goal: Make canonical WAL data the transaction source of truth and bind staged mutation, flush, fsync, replay, checkpoint, and maintenance semantics to host callbacks.
- Evidence: KVFS-303, KVFS-309, KVFS-300, KVFS-301, KVFS-304
- Outputs: ipfs_kit_py/ipfs_kit_py/core/wal, ipfs_kit_py/ipfs_kit_py/kernel_vfs/runtime.py, ipfs_kit_py/tests/kernel_vfs/wal
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/wal tests/kernel_vfs/wal
- Acceptance: Fault injection at every WAL/sidecar/stage/effect/decision/fsync boundary loses no acknowledged writes, duplicates no non-idempotent effect, preserves the valid prefix, and completes recovery before ready.
- Gap task: Remove marker/sidecar ambiguity and compose VFS mutations with real durability.
- Refinement: Buffered intent, queued work, flush, and release are not durable commit evidence.
- Conflict policy: WAL record/source changes precede runtime mutation binding; recovery and maintenance follow.

## KVFS-G400 Integrate generation-bound ranged ARC coherently

- Status: active
- Parent: KVFS-G000
- Depends on: KVFS-G100
- Fib priority: 2
- Track: arc-coherence
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/arc
- Goal: Add generation-bound range keys, generation-aware single-flight, committed read-through admission, precise mutation/replay invalidation, safe persistence, and bounded metrics.
- Evidence: KVFS-400, KVFS-401, KVFS-404, KVFS-403
- Outputs: ipfs_kit_py/ipfs_kit_py/cache/arc, ipfs_kit_py/ipfs_kit_py/arc_cache.py, ipfs_kit_py/tests/kernel_vfs/arc
- Validation: cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/arc tests/kernel_vfs/arc
- Acceptance: Only committed bytes enter ARC; every hit matches namespace, inode/content/version, generation, serializer, and range; mutations and replay cannot return stale bytes; capacity, persistence, and stampede bounds pass.
- Gap task: Connect the mature ARC implementation to the durable VFS runtime without weakening coherence.
- Refinement: Cache state is a reconstructible projection and never semantic, durability, or authorization authority.
- Conflict policy: Range binding and read-through integration precede invalidation and persistence cutover.

## KVFS-G500 Deliver and certify Linux fusepy mounts

- Status: active
- Parent: KVFS-G000
- Depends on: KVFS-G200, KVFS-G300, KVFS-G400
- Fib priority: 2
- Track: linux-fuse
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/linux
- Goal: Load fusepy/libfuse lazily, provide bounded doctor and lifecycle behavior, certify real Linux kernel I/O, and prove ARM64 and repeated mount stability.
- Evidence: KVFS-503, KVFS-500, KVFS-506, KVFS-501
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/linux.py, ipfs_kit_py/tests/kernel_vfs/linux
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/linux
- Acceptance: Driver-free tests always run; a capable runner mounts within 15 seconds, passes kernel CRUD/ranged I/O/fsync/crash-replay/unmount, and leaves no resource after 100 native cycles; unavailable capability is bounded evidence.
- Gap task: Add the standard Linux FUSE boundary and lifecycle without importing native libraries in core paths.
- Refinement: A hermetic callback test cannot promote Linux native support.
- Conflict policy: Loader/doctor precedes lifecycle; live and soak suites own isolated mountpoint leases.

## KVFS-G600 Deliver and certify Windows WinFsp compatibility mounts

- Status: active
- Parent: KVFS-G000
- Depends on: KVFS-G200, KVFS-G300, KVFS-G400
- Fib priority: 2
- Track: windows-winfsp
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/windows
- Goal: Route the same fusepy operations object through WinFsp, define Windows namespace and open/delete semantics, implement drive/directory lifecycle, and certify a pinned real WinFsp environment.
- Evidence: KVFS-608, KVFS-600, KVFS-601, KVFS-603
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/windows.py, ipfs_kit_py/tests/kernel_vfs/windows
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/windows
- Acceptance: DLL/driver/service/architecture detection is bounded; collision-safe case/name policy and Windows differences are executable; drive and directory mounts clean up after success or crash; support requires current live receipt.
- Gap task: Add a deliberate Windows compatibility policy instead of assuming POSIX semantics.
- Refinement: Loader success or hosted hermetic tests cannot promote live WinFsp support.
- Conflict policy: Loader and namespace contracts precede lifecycle; live tests require exclusive drive or directory leases.

## KVFS-G700 Package, operate, and containerize the mount service

- Status: active
- Parent: KVFS-G000
- Depends on: KVFS-G500, KVFS-G600
- Fib priority: 3
- Track: delivery
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/delivery
- Goal: Add optional dependencies, guarded imports, mount/doctor/status/unmount CLI, observability, a dedicated minimally privileged Linux FUSE image/profile, and propagation tests.
- Evidence: KVFS-703, KVFS-701, KVFS-700, KVFS-702
- Outputs: ipfs_kit_py/pyproject.toml, ipfs_kit_py/Dockerfile, ipfs_kit_py/docker-compose.yml, ipfs_kit_py/ipfs_kit_py/cli, ipfs_kit_py/tests/kernel_vfs/container
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs/packaging tests/kernel_vfs/container tests/kernel_vfs/cli
- Acceptance: Default installs remain inert; the FUSE extra is reproducible; CLI lifecycle is bounded and observable; the dedicated container uses `/dev/fuse` plus only required capability, persists recovery state, and distinguishes in-container from host propagation.
- Gap task: Deliver native support as explicit opt-in packaging and operations profiles.
- Refinement: Blanket privileged mode and silent native fallback are forbidden.
- Conflict policy: Packaging precedes image build; CLI and container tests use isolated state and process leases.

## KVFS-G800 Prove security, quality, performance, CI, and release readiness

- Status: active
- Parent: KVFS-G000
- Depends on: KVFS-G500, KVFS-G600, KVFS-G700
- Fib priority: 3
- Track: release
- Priority: P0
- Bundle: ipfs-kit/kernel-vfs/release
- Goal: Close security boundaries, differential/property tests, cross-platform CI, chaos/performance floors, migration and rollback documentation, and an immutable joined release receipt.
- Evidence: KVFS-808, KVFS-800, KVFS-802, KVFS-801, KVFS-811
- Outputs: ipfs_kit_py/tests/kernel_vfs/security, ipfs_kit_py/.github/workflows, ipfs_kit_py/benchmarks/kernel_vfs, ipfs_kit_py/docs/kernel_vfs
- Validation: cd ipfs_kit_py && python -m pytest -q tests/kernel_vfs && python benchmarks/kernel_vfs/run.py --check-reviewed-floors
- Acceptance: Path, option, permission, symlink, state, and resource attacks fail closed; differential/crash/chaos tests pass; workflow paths trigger mandatory hermetic and capable native lanes; reviewed floors and current receipts bind the terminal release decision.
- Gap task: Turn implementation evidence into a truthful, reversible support matrix and release decision.
- Refinement: Skipped, stale, print-only, driver-absent, or simulated native evidence does not satisfy a production claim.
- Conflict policy: Security and differential work fan out; CI/performance integrate later; KVFS-811 alone aggregates terminal evidence.
