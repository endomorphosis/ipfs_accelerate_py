"""Cold source-boundary controls; never establish a native or active context.

No owner, process, hold, lifecycle, positive provider result or native grant is
fabricated. These controls only exercise early refusals and legacy no-op hooks.
The real same-object binding and stopped acknowledgement need native fixtures.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
RUNTIME = Path(__file__).parent / "proposed/ipfs_accelerate_py/agent_supervisor/runtime"

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, RUNTIME / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

H = load("callable_boundary_handoff", "ordinary_success_handoff.py")
C = load("callable_boundary_custody", "ordinary_custody_hold.py")

def host_callback(_outcome=None):
    raise AssertionError("This control must verify source without calling it")

class CallableHandoffRefusals(unittest.TestCase):
    def refusal(self, reason, function, *args, **kwargs):
        with self.assertRaisesRegex(H.OrdinaryHandoffRefusal, "^" + reason + "$"):
            function(*args, **kwargs)

    def test_missing_callback_with_invocation_refuses(self):
        self.refusal("explicit_callable_handoff_requires_success_callback", H.run_ordinary_with_success_handoff,
                     None, -1, invocation=object())

    def test_missing_callback_with_hold_refuses(self):
        self.refusal("explicit_callable_handoff_requires_success_callback", H.run_ordinary_with_success_handoff,
                     None, -1, custody_hold=object())

    def test_missing_callback_with_custody_pins_refuses(self):
        self.refusal("explicit_callable_handoff_requires_success_callback", H.run_ordinary_with_success_handoff,
                     None, -1, custody_source_sha256={})

    def test_missing_callback_with_before_run_refuses(self):
        self.refusal("explicit_callable_handoff_requires_success_callback", H.run_ordinary_with_success_handoff,
                     None, -1, before_run=host_callback)

    def test_noncallable_callback_refuses(self):
        self.refusal("explicit_required_success_callback", H.run_ordinary_with_success_handoff,
                     None, -1, on_success=1)

    def test_empty_source_closure_refuses(self):
        self.refusal("finite_exact_callable_source_closure_required", H.run_ordinary_with_success_handoff,
                     None, -1, on_success=host_callback, source_sha256={})

    def test_callable_without_context_refuses(self):
        self.refusal("actual_admitted_callable_context_required", H.ordinary_callable_context_required, object())

    def test_provider_guard_without_context_refuses(self):
        self.refusal("actual_callable_provider_guard_required", H.ordinary_provider_guard)

    def test_success_authenticator_without_callback_refuses(self):
        self.refusal("actual_active_original_success_callback_required", H.require_active_ordinary_success, object())

    def test_callback_source_requires_plain_function(self):
        self.refusal("source_pinned_plain_host_callback_required", H._verify_callback_source, print, {})

    def test_callback_source_requires_actual_pin(self):
        self.refusal("actual_host_callback_source_pin_required", H._verify_callback_source, host_callback, {})

    def test_actual_callback_source_verification_is_not_invocation(self):
        path = Path(__file__).absolute()
        H._verify_callback_source(host_callback, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()})

    def test_unbound_hold_cannot_be_checked(self):
        with self.assertRaisesRegex(C.CustodyHoldRefusal, "^genuine_bound_custody_hold_required$"):
            C.require_bound_ordinary_custody_hold(object(), primary_owner=object(), watchdog_owner=object())
        self.assertEqual(C._HOLDS, {})

    def test_unbound_hold_cannot_retain_references(self):
        with self.assertRaisesRegex(C.CustodyHoldRefusal, "^genuine_bound_custody_hold_required$"):
            C.retain_ordinary_custody_references(object(), (object(),))
        self.assertEqual(C._HOLDS, {})

    def test_legacy_allocation_hooks_have_no_active_effect(self):
        H.ordinary_private_allocations_created(None, None)
        H.ordinary_container_creation_started(None, None)
        H.ordinary_container_created(None, None)
        self.assertTrue(H.ordinary_allocation_close_permitted(None))
        self.assertIsNone(H._current())

if __name__ == "__main__":
    unittest.main()
