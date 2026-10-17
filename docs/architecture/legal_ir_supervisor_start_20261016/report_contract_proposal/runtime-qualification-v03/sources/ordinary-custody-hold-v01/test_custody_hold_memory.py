"""Pure correspondence/refusal controls; never native custody qualification.

Normative dictionaries are necessary-condition unit inputs only. No owner,
consumer, scheduler or grant is fabricated; the actual public binder is never
allowed past its early fake-owner refusal. No temp files, native admission,
Docker, process execution, provider, cleanup proof or acceptance is performed.
"""
import copy
import importlib.util
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("custody_hold_memory_controls", Path(__file__).with_name("ordinary_custody_hold.py"))
hold = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hold
spec.loader.exec_module(hold)
PID = 12345


def normative():
    def parent(name, memory, slots):
        return {"lease_id": name, "parent_lease_id": "phase", "owner_pid": PID,
            "lane": "orchestration", "cpu_slots": 1, "memory_mb": memory,
            "child_process_slots": slots, "requires_gpu": False, "gpu_memory_mb": 0,
            "unified_memory_mb": 0, "released": False, "cancelled": False}
    def owner(name, lease):
        return {"reservation_id": name, "status": "active", "scheduler_lane": "orchestration",
            "workload": "canonical_parented_phase", "cleanup_error": None,
            "resource_lease": lease, "record": {"child": None, "prior_children": [],
                "native_lease_id": lease["lease_id"], "owner_pid": PID}}
    def consumer(name, ancestor):
        return {"lease_id": name, "parent_lease_id": ancestor, "owner_pid": PID,
            "lane": "orchestration", "cpu_slots": 1, "memory_mb": 1,
            "child_process_slots": 0, "requires_gpu": False, "gpu_memory_mb": 0,
            "unified_memory_mb": 0, "released": False, "cancelled": False}
    first = owner("primary-disk", parent("primary-native", 512, 8))
    second = owner("watchdog-disk", parent("watchdog-native", 256, 4))
    return [first, second, consumer("primary-consumer", "primary-native"),
            consumer("watchdog-consumer", "watchdog-native")]


def inventory(values):
    rows = [values[0]["resource_lease"], values[1]["resource_lease"], values[2], values[3]]
    # Actual GlobalResourceScheduler.acquire omits release_requested until an
    # explicit release/dead-owner recovery sets it true. Do not invent a field.
    return [dict(row) for row in rows]


class CustodyHoldMemoryControls(unittest.TestCase):
    def refuse(self, reason, function, *args, **kwargs):
        with self.assertRaisesRegex(hold.CustodyHoldRefusal, "^" + reason + "$", msg="Exact normative refusal required"):
            function(*args, **kwargs)

    def check(self, values, rows=None):
        return hold.check_binding_projection(*values, inventory(values) if rows is None else rows, owner_pid=PID)

    def changed(self, index, key, value, reason):
        values = normative()
        target = values[index]["resource_lease"] if index < 2 else values[index]
        target[key] = value
        self.refuse(reason, self.check, values)

    def test_complete_normative_join_has_no_authority(self):
        result = self.check(normative())
        self.assertEqual(result["consumer_cpu_slots_each"], 1)
        self.assertEqual(result["consumer_memory_mb_each"], 1)
        self.assertTrue(all(result[name] is False for name in hold.NO_AUTHORITY))

    def test_pure_join_never_mints_a_hold_or_invokes_io(self):
        values = normative()
        before = copy.deepcopy(values)
        with patch.object(hold.os, "open", side_effect=AssertionError("No source/native I/O")), patch.object(hold.importlib, "import_module", side_effect=AssertionError("No native import")):
            self.check(values)
        self.assertEqual(values, before)
        self.assertEqual(hold._HOLDS, {})

    def test_duplicate_active_inventory_refuses(self):
        values = normative()
        rows = inventory(values)
        rows.append(dict(rows[0]))
        self.refuse("duplicate_native_lease_inventory", self.check, values, rows)

    def test_primary_foreign_owner_refuses(self):
        self.changed(0, "owner_pid", PID + 1, "actual_parent_native_envelope_required")

    def test_primary_other_lane_refuses(self):
        self.changed(0, "lane", "validation", "actual_parent_native_envelope_required")

    def test_retained_owner_cannot_bind_new_effect_scope(self):
        values = normative()
        values[0]["status"] = "retained"
        self.refuse("actual_active_parented_owner_required", self.check, values)

    def test_primary_memory_envelope_too_small(self):
        self.changed(0, "memory_mb", 511, "actual_parent_native_envelope_required")

    def test_watchdog_memory_envelope_too_small(self):
        self.changed(1, "memory_mb", 255, "actual_parent_native_envelope_required")

    def test_primary_process_envelope_too_small(self):
        self.changed(0, "child_process_slots", 7, "actual_parent_native_envelope_required")

    def test_watchdog_process_envelope_too_small(self):
        self.changed(1, "child_process_slots", 3, "actual_parent_native_envelope_required")

    def test_parent_cancelled_refuses_new_binding(self):
        self.changed(0, "cancelled", True, "actual_parent_native_envelope_required")

    def test_parent_released_refuses_new_binding(self):
        self.changed(0, "released", True, "actual_parent_native_envelope_required")

    def test_zero_CPU_consumer_forbidden(self):
        self.changed(2, "cpu_slots", 0, "actual_direct_bounded_consumer_required")

    def test_zero_memory_consumer_forbidden(self):
        self.changed(2, "memory_mb", 0, "actual_direct_bounded_consumer_required")

    def test_boolean_CPU_consumer_forbidden(self):
        self.changed(2, "cpu_slots", True, "actual_direct_bounded_consumer_required")

    def test_boolean_memory_consumer_forbidden(self):
        self.changed(2, "memory_mb", True, "actual_direct_bounded_consumer_required")

    def test_consumer_cannot_add_process_allocation(self):
        self.changed(2, "child_process_slots", 1, "actual_direct_bounded_consumer_required")

    def test_foreign_consumer_owner_refuses(self):
        self.changed(2, "owner_pid", PID + 1, "actual_direct_bounded_consumer_required")

    def test_consumer_must_be_actual_direct_child(self):
        self.changed(2, "parent_lease_id", "unrelated", "actual_direct_bounded_consumer_required")

    def test_cancelled_consumer_cannot_bind(self):
        self.changed(2, "cancelled", True, "actual_direct_bounded_consumer_required")

    def test_different_native_phase_refuses(self):
        self.changed(1, "parent_lease_id", "other-phase", "distinct_sibling_owners_and_consumers_required")

    def test_same_disk_reservation_refuses(self):
        values = normative()
        values[1]["reservation_id"] = values[0]["reservation_id"]
        self.refuse("distinct_sibling_owners_and_consumers_required", self.check, values)

    def test_preexisting_primary_child_refuses(self):
        values = normative()
        values[0]["record"]["child"] = {"pid": 1}
        self.refuse("fresh_pre_effect_owner_required", self.check, values)

    def test_watchdog_child_history_refuses(self):
        values = normative()
        values[1]["record"]["prior_children"] = [{"pid": 1}]
        self.refuse("fresh_pre_effect_owner_required", self.check, values)

    def test_unbound_direct_consumer_refuses(self):
        values = normative()
        rows = inventory(values)
        rows.append(dict(rows[-1], lease_id="unbound-consumer"))
        self.refuse("unexpected_direct_native_consumer", self.check, values, rows)

    def test_missing_actual_consumer_row_refuses(self):
        values = normative()
        self.refuse("actual_native_inventory_correspondence_failed", self.check, values, inventory(values)[:-1])

    def test_disk_native_record_mismatch_refuses(self):
        values = normative()
        values[0]["record"]["native_lease_id"] = "unrelated"
        self.refuse("actual_disk_native_identity_differs", self.check, values)

    def test_GPU_consumer_refuses(self):
        self.changed(2, "requires_gpu", True, "CPU_only_custody_required")

    def test_actual_row_release_requested_refuses(self):
        values = normative()
        rows = inventory(values)
        rows[-1]["release_requested"] = True
        self.refuse("actual_native_inventory_correspondence_failed", self.check, values, rows)

    def test_incomplete_source_pin_set_refuses_before_reads(self):
        with patch.object(hold.os, "open", side_effect=AssertionError("Missing pins must fail first")):
            self.refuse("exact_custody_implementation_pins_required", hold.verify_source_pins, {})

    def test_fake_owner_refuses_before_source_or_native_import(self):
        with patch.object(hold.importlib, "import_module", side_effect=AssertionError("No native import")), patch.object(hold.os, "open", side_effect=AssertionError("No source I/O")):
            self.refuse("actual_parented_owner_metadata_type_required", hold.bind_ordinary_custody_hold,
                primary_owner=object(), watchdog_owner=object(), primary_consumer=object(),
                watchdog_consumer=object(), protected_source_sha256={})

    def test_hold_constructor_cannot_mint_a_capability(self):
        self.refuse("genuine_native_binding_only_hold_constructor", hold.OrdinaryCustodyHold)

    def test_forged_unregistered_handle_refuses_observation_and_refs(self):
        forged = object.__new__(hold.OrdinaryCustodyHold)
        self.refuse("genuine_bound_custody_hold_required", hold.observe_ordinary_custody_hold, forged)
        self.refuse("genuine_bound_custody_hold_required", hold.held_native_consumers, forged)
        self.refuse("genuine_bound_custody_hold_required", hold.forget_acknowledged_custody_hold, forged)

    def test_no_context_exit_or_finalizer_can_acknowledge(self):
        self.assertFalse(hasattr(hold.OrdinaryCustodyHold, "__enter__"))
        self.assertFalse(hasattr(hold.OrdinaryCustodyHold, "__exit__"))
        self.assertFalse(hasattr(hold.OrdinaryCustodyHold, "__del__"))
        self.assertFalse(hasattr(hold, "release"))

    def test_inert_handle_copy_and_serialization_refuse(self):
        forged = object.__new__(hold.OrdinaryCustodyHold)
        self.refuse("custody_hold_not_copyable", copy.copy, forged)
        self.refuse("custody_hold_not_copyable", copy.deepcopy, forged)
        self.refuse("custody_hold_not_serializable", pickle.dumps, forged)


if __name__ == "__main__":
    unittest.main()
