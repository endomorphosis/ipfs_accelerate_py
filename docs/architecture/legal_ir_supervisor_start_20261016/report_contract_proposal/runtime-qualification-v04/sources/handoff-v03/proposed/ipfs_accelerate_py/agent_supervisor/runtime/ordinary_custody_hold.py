"""Trusted host binder for genuine pre-effect native capacity consumers.

This module never dispatches, creates a grant, releases a native lease, closes
Docker, accepts a cleanup receipt, calls a callback, or grants acceptance. The
caller supplies actual preacquired public ResourceLease children. Strong
same-process references prevent accidental acknowledgement by scope cleanup;
original owner/ancestor cancellation still revokes productive authority.
"""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import re
import stat
import sys

NATIVE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer."
DATASETS = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
NATIVE_NAMES = ("parented_phase_resources", "autoencoder_daemon_resources", "existing_resource_scheduler", "resource_scheduler")
NATIVE_PATHS = tuple(DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / (name + ".py") for name in NATIVE_NAMES)
SCHEMA = "legal-ir-ordinary-custody-capacity-observation@1"
NO_AUTHORITY = {name: False for name in (
    "productive_authority_granted", "cleanup_verified", "Docker_quiescence_verified",
    "provider_execution_qualified", "acceptance_authority", "completion_authority", "dispatch_authority")}
SHA = re.compile(r"[0-9a-f]{64}\Z")
ENVELOPE = ("lease_id", "parent_lease_id", "owner_pid", "lane", "cpu_slots", "memory_mb",
            "child_process_slots", "requires_gpu", "gpu_memory_mb", "unified_memory_mb")


class CustodyHoldRefusal(RuntimeError):
    pass


def need(condition, reason):
    if not condition:
        raise CustodyHoldRefusal(reason)


def required_source_paths():
    return (Path(__file__).absolute(), *NATIVE_PATHS)


def read_pinned_source(path, expected):
    """Stable bounded owned trusted source; group write requires exact SHA."""
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and ".." not in path.parts
         and not any(item.is_symlink() for item in (path, *path.parents)), "canonical_unaliased_source_required")
    need(type(expected) is str and SHA.fullmatch(expected), "exact_source_digest_required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 < before.st_size <= 4 * 1024 * 1024,
             "bounded_owned_trusted_source_required")
        raw = stream.read(4 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
    witness = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid,
                           row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    need(witness(before) == witness(after) == witness(path.lstat()) and len(raw) == before.st_size,
         "source_changed_during_read")
    need(hashlib.sha256(raw).hexdigest() == expected, "actual_custody_source_changed")
    return raw


def verify_source_pins(pins):
    need(type(pins) is dict and set(pins) == {str(path) for path in required_source_paths()},
         "exact_custody_implementation_pins_required")
    for path, expected in pins.items():
        read_pinned_source(Path(path), expected)


def check_binding_projection(primary, watchdog, primary_consumer, watchdog_consumer, active, *, owner_pid):
    """Pure necessary correspondence; supplied mappings are never admission."""
    need(type(active) is list and all(type(row) is dict and type(row.get("lease_id")) is str for row in active),
         "redacted_native_inventory_required")
    by_id = {row["lease_id"]: row for row in active}
    need(len(by_id) == len(active), "duplicate_native_lease_inventory")
    parents, consumers, reservation_ids, phase_ids, lease_ids = [], [], [], [], []
    for public, child, memory, process_slots in ((primary, primary_consumer, 512, 8), (watchdog, watchdog_consumer, 256, 4)):
        need(type(public) is dict and public.get("status") == "active"
             and public.get("scheduler_lane") == "orchestration"
             and public.get("workload") == "canonical_parented_phase"
             and public.get("cleanup_error") is None, "actual_active_parented_owner_required")
        record, parent = public.get("record"), public.get("resource_lease")
        need(type(record) is dict and record.get("child") is None
             and record.get("prior_children", []) == [], "fresh_pre_effect_owner_required")
        need(type(parent) is dict and parent.get("owner_pid") == owner_pid and parent.get("lane") == "orchestration"
             and parent.get("released") is False and parent.get("cancelled") is False
             and type(parent.get("cpu_slots")) is int and parent["cpu_slots"] >= 1
             and type(parent.get("memory_mb")) is int and parent["memory_mb"] >= memory
             and type(parent.get("child_process_slots")) is int and parent["child_process_slots"] >= process_slots,
             "actual_parent_native_envelope_required")
        need(type(child) is dict and child.get("owner_pid") == owner_pid and child.get("lane") == "orchestration"
             and child.get("released") is False and child.get("cancelled") is False
             and child.get("parent_lease_id") == parent.get("lease_id")
             and type(child.get("cpu_slots")) is int and child["cpu_slots"] == 1
             and type(child.get("memory_mb")) is int and child["memory_mb"] == 1
             and type(child.get("child_process_slots")) is int and child["child_process_slots"] == 0,
             "actual_direct_bounded_consumer_required")
        for row in (parent, child):
            need(row.get("requires_gpu") is False and row.get("gpu_memory_mb") == 0
                 and row.get("unified_memory_mb") == 0, "CPU_only_custody_required")
            identifier = row.get("lease_id")
            need(type(identifier) is str and identifier in by_id
                 and all(by_id[identifier].get(name) == row.get(name) for name in ENVELOPE)
                 and by_id[identifier].get("cancelled") is False
                 and by_id[identifier].get("release_requested", False) is False,
                 "actual_native_inventory_correspondence_failed")
            lease_ids.append(identifier)
        need(record.get("native_lease_id") == parent["lease_id"]
             and record.get("owner_pid") == owner_pid, "actual_disk_native_identity_differs")
        reservation_ids.append(public.get("reservation_id"))
        phase_ids.append(parent.get("parent_lease_id"))
        parents.append(parent)
        consumers.append(child)
    need(all(type(value) is str and value for value in reservation_ids + phase_ids)
         and len(set(reservation_ids)) == 2 and len(set(lease_ids)) == 4 and phase_ids[0] == phase_ids[1],
         "distinct_sibling_owners_and_consumers_required")
    for parent, child in zip(parents, consumers):
        need({row["lease_id"] for row in active if row.get("parent_lease_id") == parent["lease_id"]}
             == {child["lease_id"]}, "unexpected_direct_native_consumer")
    return {"owner_pid": owner_pid, "parent_native_lease_ids": [row["lease_id"] for row in parents],
        "consumer_native_lease_ids": [row["lease_id"] for row in consumers], "phase_lease_id": phase_ids[0],
        "disk_reservation_ids": reservation_ids, "consumer_cpu_slots_each": 1,
        "consumer_memory_mb_each": 1, "consumer_child_process_slots_each": 0, **NO_AUTHORITY}


class OrdinaryCustodyHold:
    """Opaque strong-reference holder. No context exit, finalizer or release."""
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise CustodyHoldRefusal("genuine_native_binding_only_hold_constructor")

    def __reduce__(self):
        raise CustodyHoldRefusal("custody_hold_not_serializable")

    def __copy__(self):
        raise CustodyHoldRefusal("custody_hold_not_copyable")

    def __deepcopy__(self, memo):
        raise CustodyHoldRefusal("custody_hold_not_copyable")


_HOLDS = {}


def _native_modules():
    modules = tuple(importlib.import_module(NATIVE + name) for name in NATIVE_NAMES)
    need(all(Path(module.__file__).absolute() == path for module, path in zip(modules, NATIVE_PATHS)),
         "actual_pinned_native_source_modules_required")
    return modules


def bind_ordinary_custody_hold(*, primary_owner, watchdog_owner, primary_consumer, watchdog_consumer,
                              protected_source_sha256):
    """Bind actual preacquired consumer objects BEFORE any Docker/provider effect.

    Caller uses each genuine owner's public native_lease.acquire_child with
    cpu_slots=1,memory_mb=1,child_process_slots=0 and noGPU. No native capacity
    is acquired or released here. If this binder refuses, no productive effect
    has been authorized; caller owns its actual unstarted child cleanup.
    """
    need(sys.dont_write_bytecode is True, "native_host_no_bytecode_required")
    need(all(type(owner).__module__ == NATIVE + "parented_phase_resources"
             and type(owner).__name__ == "ParentedPhaseResourceReservation"
             for owner in (primary_owner, watchdog_owner)), "actual_parented_owner_metadata_type_required")
    need(all(type(consumer).__module__ == NATIVE + "resource_scheduler"
             and type(consumer).__name__ == "ResourceLease"
             for consumer in (primary_consumer, watchdog_consumer)), "actual_native_consumer_metadata_type_required")
    verify_source_pins(protected_source_sha256)
    parented, _daemon, existing, scheduler = _native_modules()
    need(type(primary_owner) is parented.ParentedPhaseResourceReservation
         and type(watchdog_owner) is parented.ParentedPhaseResourceReservation
         and primary_owner is not watchdog_owner, "actual_distinct_parented_owner_objects_required")
    need(type(primary_consumer) is scheduler.ResourceLease and type(watchdog_consumer) is scheduler.ResourceLease
         and primary_consumer is not watchdog_consumer, "actual_distinct_native_consumer_objects_required")
    need(len(_HOLDS) < 128 and not any(
        consumer is existing_consumer
        for entry in _HOLDS.values() for existing_consumer in entry["consumers"]
        for consumer in (primary_consumer, watchdog_consumer)), "native_consumer_already_bound_or_registry_full")
    native_parents = []
    for owner in (primary_owner, watchdog_owner):
        need(owner.heartbeat(), "actual_owner_heartbeat_failed")
        owner.check_usage(owner.directory, full=True)
        native_parents.append(owner.native_lease)
    public = (primary_owner.to_dict(), watchdog_owner.to_dict())
    records = tuple(value.get("record") for value in public)
    need(all(type(value) is dict for value in records), "actual_native_disk_records_required")
    state, policy = records[0].get("scheduler_state_path"), records[0].get("scheduler_policy_sha256")
    need(type(state) is str and Path(state).is_absolute() and type(policy) is str and SHA.fullmatch(policy)
         and records[1].get("scheduler_state_path") == state
         and records[1].get("scheduler_policy_sha256") == policy, "actual_same_scheduler_policy_required")
    opened = existing.open_existing_scheduler(Path(state), policy)
    for consumer in (primary_consumer, watchdog_consumer):
        # The public renew authenticates its own genuine key inside the SDK;
        # this helper never reads, copies, reconstructs or exports that key.
        need(Path(consumer.token.state_path) == Path(state) and consumer.renew(),
             "actual_same_scheduler_consumer_authority_required")
    observation = check_binding_projection(*public, primary_consumer.to_dict(), watchdog_consumer.to_dict(),
        opened.active_leases(), owner_pid=os.getpid())
    verify_source_pins(protected_source_sha256)
    handle = object.__new__(OrdinaryCustodyHold)
    _HOLDS[handle] = {"owner_pid": os.getpid(), "owners": (primary_owner, watchdog_owner),
        "native_parents": tuple(native_parents), "consumers": (primary_consumer, watchdog_consumer),
        "scheduler": opened, "source_sha256": dict(protected_source_sha256), "binding": observation}
    return handle


def _entry(handle):
    need(type(handle) is OrdinaryCustodyHold and handle in _HOLDS, "genuine_bound_custody_hold_required")
    entry = _HOLDS[handle]
    need(entry["owner_pid"] == os.getpid(), "custody_hold_owner_process_changed")
    verify_source_pins(entry["source_sha256"])
    _native_modules()
    return entry


def held_native_consumers(handle):
    """Return the SAME actual public objects for trusted stopped closeout only.

    Their public release must be called only after independent actual Docker,
    kernel, private-path and both-group closure. No cleanup proof is accepted
    or inferred here. This generation implements the capacity barrier only.
    """
    return _entry(handle)["consumers"]


def observe_ordinary_custody_hold(handle):
    """Report actual record retention separately from revoked ancestry."""
    entry = _entry(handle)
    active = entry["scheduler"].active_leases()
    by_id = {row["lease_id"]: row for row in active}
    consumers = entry["consumers"]
    present = [consumer.lease_id in by_id for consumer in consumers]
    cancelled = [consumer.cancelled for consumer in consumers]
    return {"schema": SCHEMA, "observation_only": True, **NO_AUTHORITY,
        "owner_pid": os.getpid(), "consumer_native_lease_ids": [consumer.lease_id for consumer in consumers],
        "consumer_records_present": present, "consumer_ancestry_cancelled": cancelled,
        "consumer_objects_released": [consumer.released for consumer in consumers],
        "capacity_records_retained": all(present), "productive_ancestry_retained": all(present) and not any(cancelled),
        "no_extra_global_root_allocation": True, "host_process_death_survival_guaranteed": False,
        "independent_stopped_cleanup_required_before_acknowledgement": True}


def forget_acknowledged_custody_hold(handle):
    """Remove references only AFTER actual consumers were explicitly released.

    This does not release anything or authenticate Docker/kernel cleanup.
    Trusted fixture/controller performs independent stopped proof before any
    public consumer.release; this function cannot substitute for that proof.
    """
    entry = _entry(handle)
    need(all(consumer.released for consumer in entry["consumers"]), "unacknowledged_native_consumers_still_held")
    active = {row["lease_id"] for row in entry["scheduler"].active_leases()}
    need(not any(consumer.lease_id in active for consumer in entry["consumers"]),
         "acknowledged_consumer_rows_not_drained")
    observation = {"schema": SCHEMA, "observation_only": True, **NO_AUTHORITY,
        "hold_references_forgotten": True, "native_consumers_acknowledged_and_absent": True,
        "native_leases_released_by_this_helper": False, "independent_cleanup_not_authenticated_by_this_helper": True}
    del _HOLDS[handle]
    return observation


def require_bound_ordinary_custody_hold(handle, *, primary_owner, watchdog_owner):
    """Verify actual same-process owner identity and original charged consumers.

    This grants no admission and accepts no portable owner/proof mapping.
    Cancellation observation is separate: the caller checks productive owner
    authority before dispatch; stopped cleanup may inspect cancelled records.
    """
    entry = _entry(handle)
    need(entry["owners"][0] is primary_owner and entry["owners"][1] is watchdog_owner,
         "actual_bound_custody_owner_identity_differs")
    observed = observe_ordinary_custody_hold(handle)
    need(observed["capacity_records_retained"] and observed["consumer_objects_released"] == [False, False],
         "original_unacknowledged_custody_consumers_required")


def retain_ordinary_custody_references(handle, references):
    """Keep original trusted caller objects alive; no cleanup/grant inferred.

    The native binder already authenticates its actual hold. References are
    retained only, never executed, inspected as authority, copied or serialized.
    This prevents loss of original TemporaryDirectory/watchdog/process objects
    when an unknown-cleanup exception crosses a caller scope.
    """
    entry = _entry(handle)
    need(type(references) is tuple and 1 <= len(references) <= 16,
         "bounded_original_custody_references_required")
    previous = entry.get("custody_references")
    need(previous is None or len(previous) == len(references) and all(
        old is new for old, new in zip(previous, references)), "original_custody_references_cannot_be_replaced")
    entry["custody_references"] = references
