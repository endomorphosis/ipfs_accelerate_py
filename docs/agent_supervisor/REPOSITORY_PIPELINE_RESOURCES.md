# Managed repository resource scopes

`runtime/repository_pipeline_resources.py` is an opt-in composition of the existing
`RepositoryResourceBridge`, native datasets scheduler, and
`DaemonResourceReservation`. Existing bridge callers and defaults are unchanged.

The authority chain is one datasets host root, a bridge phase, the durable daemon
owner, then the existing consumer's child lease. CPU, RAM and process reservations
are charged once at the root. There is no new host allocator or disk ledger. The
daemon's public `native_lease` supplies downstream authority; passing the bridge
phase lease instead would incorrectly make the daemon and consumer siblings.

The adapter partitions the parent's CPU, RAM, process slots, disk declarations,
queue slots, phase history and retained payload bytes. Ordinary phases cannot use
the protected validation/cleanup compartment. Validation and cleanup share that
compartment; they can queue behind each other. Ordinary and protected queues are
FIFO independently. Native pressure sampling, queue admission and backoff remain
authoritative after local admission.

The policy is a required argument. For the finite supervisor's 1 GiB validation
workload, use an explicit policy such as `protected_memory_mb=1024` under a parent
of at least 3 GiB, with suitable CPU/process and disk reservations. The small test
profile uses 128 MiB validation demands explicitly. A default 512 MiB
`RepositoryPhaseDemand('validation')` is refused by a 128 MiB protected policy.

```python
resources = RepositoryPipelineResources(supervisor_resource_scheduler)
with resources.reserve(
    repository_id=repository_id,
    workspace=workspace,
    budget=RepositoryResourceBudget(
        cpu_slots=3, memory_mb=3072, process_slots=3,
        disk_bytes=64 * MIB, wall_time_ms=600000),
    policy=PipelineResourcePolicy(
        protected_memory_mb=1024, protected_disk_bytes=16 * MIB),
    ledger_path=existing_disk_ledger,
    roots=existing_named_storage_roots,
) as owner:
    with owner.phase(
        RepositoryPhaseDemand('validation', memory_mb=1024, disk_bytes=8 * MIB),
        payload=b'bounded input',
        attempt_directory=existing_unique_attempt_directory,
    ) as phase:
        # Native APIs accept these options and must finish/reap synchronously.
        options = phase.native_options()
        # Or phase.run(argv) supplies this payload as stdin to the bounded runner.
        # Persist and fsync owned outputs before asserting their durability.
        phase.finalize(artifacts_durable=True)
```

Payloads must be immutable `bytes`. Their actual lengths count while queued and
active; every phase also reserves a bounded two-stream capture allowance. Mutable
inputs, queue overflow and retained-byte overflow are refused before native phase
admission. Payload and process-output references are cleared when the phase ends.
Caller-retained copies and returned output objects are outside this inventory.
These are serialized payload byte bounds, not exact Python object RSS accounting.

Each phase has a unique disjoint attempt directory. The daemon records the named
roots, full outstanding disk claim and sampled group RSS. `charge_external(path,
bytes)` records a conservative charge before a CAS/journal output is written in
those roots. Identical charges are idempotent; changing an amount is refused.
Successful finalization replaces the claim with final charged bytes, which still
count against that parent's compartment. Failed attempts retain their full claims.
The disk partition protects against this parent's ordinary phases; it cannot
guarantee disk availability against other repositories or writes outside the
ledger. Named-root/free-space admission can still refuse protected work.

`phase.run` reuses the existing bounded process runner and process-group cleanup.
It registers the actual child before work proceeds, samples disk/RSS/process count,
applies explicit thread environment settings and requires synchronous ownership.
It does not claim a hard aggregate CPU/RSS or thread ceiling. Existing consumer APIs
must honor their native lease, deadline, cancellation and thread contracts and reap
their own subprocesses before returning.

The adapter refuses finalization while native consumer leases remain. If a scope
exits with a registered process or unresolved native consumer, it retains the
bridge phase as well as the disk claim. A protected cleanup phase can call
`recover_retained(reservation_id, artifacts_durable=True)` only after consumers
have released and registered processes are dead. Durability remains an explicit
caller assertion, as in the existing daemon owner.

If the whole parent exits before reaping, the canonical bridge refuses to report
clean shutdown. Pending owner handles remain strongly reachable so garbage
collection cannot release generator-owned host capacity. Inspect
`pending_pipeline_recoveries()`; a newly admitted cleanup phase with the exact same
native authority, ledger and roots can recover with `owner=old_handle`. Cross-process
crash recovery stays with the existing native scheduler and durable disk owner.

Qualification distinguishes injected host telemetry from actual-host acceptance.
The controlled suite uses real file-backed owners and real subprocesses, including
overshoot, cancellation, unsafe scope exit and subsequent cleanup. The actual-host
attempt was refused by `host_disk_high_watermark` before work. This adapter has not
qualified full production pipeline execution. GPU/per-device admission, hard
enforcement, host-wide retained payload accounting and gradual recovery remain
outside the implemented profile.
