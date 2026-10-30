"""Pure/normative lifecycle controls; no provider/native/process qualification.

Only cold stdlib module import, pure necessary-condition checks, inert outcome
copy refusal, default no-op helpers and required fake-owner refusal execute.
Normative projection/order strings are not host observation or acceptance.
"""
from copy import copy, deepcopy
import importlib.util
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
import argparse
import hashlib
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).parent.parent / "handoff-v02/proposed/ipfs_accelerate_py/agent_supervisor/runtime/ordinary_success_handoff.py"
spec = importlib.util.spec_from_file_location("ordinary_success_handoff_control", SOURCE)
hook = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hook
spec.loader.exec_module(hook)
ORDER = ("invocation_started", "route_selected", "task_started", "watchdog_native_attached", "lease_created", "transport_started",
         "native_child_attached", "container_scope_captured", "provider_start_released",
         "provider_reaped", "lease_closed", "kernel_scope_quiescent", "native_owner_checked", "callback_ready")


def projection():
    return dict(execution_id="a" * 32, provider_calls=1, returncode=0,
        process_reaped=True, lease_closed=True, watchdog_reaped=True, watchdog_owner_group_quiescent=True, kernel_scope_quiescent=True,
        exact_container_absent=True, lease_root_absent=True, provider_home_absent=True, prompt_path_absent=True, daemon_unchanged=True, runtime_unchanged=True,
        provider_id="grok_cli", model_id="grok-4.7", reasoning_effort="", provider_trigger="primary",
        model_host_pid=300, owner_pid=100, transport_pid=200, model_start_time_ticks=400)


class MemoryHandoffControls(unittest.TestCase):
    def refusal(self, reason, function, *args, **kwargs):
        with self.assertRaisesRegex(hook.OrdinaryHandoffRefusal, "^" + reason + "$", msg="Exact normative refusal required"):
            function(*args, **kwargs)

    def test_complete_normative_grok_projection_is_necessary_only(self):
        self.assertIsNone(hook.check_lifecycle_projection(projection()))

    def test_quota_high_projection_refuses_in_primary_only_generation(self):
        value = projection()
        value.update(provider_id="codex", model_id="gpt-6.1-sol", reasoning_effort="high", provider_trigger="primary_quota_exhausted")
        self.refusal("actual_supported_ordinary_route_required", hook.check_lifecycle_projection, value)

    def test_complete_normative_order_is_necessary_only(self):
        self.assertIsNone(hook.check_success_order(ORDER))

    def test_callback_before_cleanup_refuses(self):
        wrong = list(ORDER)
        wrong.remove("callback_ready")
        wrong.insert(wrong.index("lease_closed"), "callback_ready")
        self.refusal("successful_handoff_must_follow_actual_closeout", hook.check_success_order, wrong)

    def test_callback_before_native_owner_check_refuses(self):
        wrong = list(ORDER)
        wrong[-1], wrong[-2] = wrong[-2], wrong[-1]
        self.refusal("successful_handoff_must_follow_actual_closeout", hook.check_success_order, wrong)

    def test_duplicate_callback_ready_refuses(self):
        self.refusal("successful_handoff_must_follow_actual_closeout", hook.check_success_order, ORDER + ("callback_ready",))

    def test_default_hooks_are_noop_even_with_untrusted_inputs(self):
        # No active context: original CLI helper behavior, no native imports.
        hook.select_ordinary_route(None, None)
        hook.begin_ordinary_task()
        hook.select_ordinary_fallback(None, None)
        hook.ordinary_lease_created(None, None)
        hook.ordinary_provider_started(None, None)
        hook.ordinary_provider_returned(None, None)
        hook.ordinary_lease_closed(None)
        self.assertIsNone(hook._current())

    def test_none_callback_delegates_original_result_without_handoff(self):
        sentinel_args = object()
        calls = []
        normative_runner = SimpleNamespace(_run=lambda args, descriptor: calls.append((args, descriptor)) or 7)
        with patch.object(hook.importlib, "import_module", return_value=normative_runner):
            result = hook.run_ordinary_with_success_handoff(sentinel_args, 13, on_success=None)
        self.assertEqual(result, 7)
        self.assertEqual(calls, [(sentinel_args, 13)])
        self.assertIsNone(hook._current())

    def test_fake_owner_never_invokes_callback_or_imports_product(self):
        callbacks = []
        def actual_source_pinned_callback(outcome):
            callbacks.append(outcome)
        path = Path(__file__).absolute()
        pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
        with patch.object(hook.importlib, "import_module", side_effect=AssertionError("fake owner must fail before product imports")):
            self.refusal("genuine_parented_owner_required", hook.run_ordinary_with_success_handoff,
                         SimpleNamespace(), 1, owner=object(), source_sha256=pins,
                         on_success=actual_source_pinned_callback)
        self.assertEqual(callbacks, [])

    def test_noncallable_callback_never_imports_product(self):
        with patch.object(hook.importlib, "import_module", side_effect=AssertionError("bad callback must fail before imports")):
            self.refusal("explicit_required_success_callback", hook.run_ordinary_with_success_handoff,
                         SimpleNamespace(), 1, owner=object(), source_sha256={}, on_success=True)

    def test_nested_context_refuses_before_owner_or_product(self):
        token = hook._ACTIVE.set(object())
        try:
            with patch.object(hook.importlib, "import_module", side_effect=AssertionError("nested context must not import")):
                self.refusal("nested_ordinary_handoff_forbidden", hook.run_ordinary_with_success_handoff,
                             SimpleNamespace(), 1, owner=object(), source_sha256={}, on_success=lambda _: None)
        finally:
            hook._ACTIVE.reset(token)

    def test_inert_outcome_pickle_refuses(self):
        value = hook.OrdinaryProviderSuccess("a" * 32, None, {}, None, "grok_cli", "grok-4.7", "", "primary", "", None,
                                            300, 400, {}, "", "", "", 0, None)
        self.assertFalse(value.acceptance_authority)
        self.refusal("same_interpreter_outcome_not_serializable", pickle.dumps, value)

    def test_inert_outcome_copy_refuses(self):
        value = hook.OrdinaryProviderSuccess("a" * 32, None, {}, None, "grok_cli", "grok-4.7", "", "primary", "", None,
                                            300, 400, {}, "", "", "", 0, None)
        self.refusal("actual_outcome_not_copyable", copy, value)
        self.refusal("actual_outcome_not_copyable", deepcopy, value)

    def test_primary_options_require_actual_namespace(self):
        self.refusal("actual_parsed_runner_namespace_required", lambda args: hook.check_primary_only_options(args, environment={}), SimpleNamespace())

    def test_primary_options_reject_fallback_before_invocation(self):
        self.refusal("primary_only_handoff_forbids_ungated_preflight_or_fallback", lambda args: hook.check_primary_only_options(args, environment={}),
                     argparse.Namespace(codex_fallback_command_json='["codex","exec"]'))

    def test_primary_options_reject_internal_preflight_before_invocation(self):
        self.refusal("primary_only_handoff_forbids_ungated_preflight_or_fallback", lambda args: hook.check_primary_only_options(args, environment={}),
                     argparse.Namespace(canonical_legacy_preflight_route=True))

    def test_primary_options_reject_nonce_before_invocation(self):
        self.refusal("primary_only_handoff_forbids_ungated_preflight_or_fallback", lambda args: hook.check_primary_only_options(args, environment={}),
                     argparse.Namespace(grok_failure_receipt_nonce="preflight-identity"))

    def test_primary_options_reject_bound_route_before_invocation(self):
        self.refusal("primary_only_handoff_forbids_ungated_preflight_or_fallback", lambda args: hook.check_primary_only_options(args, environment={}),
                     argparse.Namespace(agent_implementation_route_json="{}"))

    def test_primary_options_reject_recovery_before_invocation(self):
        self.refusal("ordinary_handoff_forbids_protected_recovery", lambda args: hook.check_primary_only_options(args, environment={}),
                     argparse.Namespace(agent_implementation_recovery_json="{}"))

    def test_primary_options_without_probes_are_necessary_only(self):
        self.assertIsNone(hook.check_primary_only_options(argparse.Namespace(), environment={}))

    def test_actual_receipt_fd_refuses_before_invocation(self):
        self.refusal("primary_only_handoff_forbids_supervised_receipt_dispatch", hook.check_primary_only_options,
                     argparse.Namespace(), 7, {})

    def test_declared_receipt_fd_refuses_before_invocation(self):
        self.refusal("primary_only_handoff_forbids_supervised_receipt_dispatch", hook.check_primary_only_options,
                     argparse.Namespace(receipt_fd_declared=True), -1, {})

    def test_ambient_terminal_receipt_channel_refuses(self):
        self.refusal("primary_only_handoff_forbids_supervised_receipt_dispatch", hook.check_primary_only_options,
                     argparse.Namespace(), -1, {"IPFS_ACCELERATE_GROK_TERMINAL_RECEIPT_FD": "7"})

    def test_ambient_trusted_failure_receipt_channel_refuses(self):
        self.refusal("primary_only_handoff_forbids_supervised_receipt_dispatch", hook.check_primary_only_options,
                     argparse.Namespace(), -1, {"IPFS_ACCELERATE_AGENT_TRUSTED_FAILURE_RECEIPT_FD": "7"})

    def storage_control(self):
        path = "/tmp/pytest-of-barberb/ordinary-handoff-control"
        key = hashlib.sha256(path.encode()).hexdigest()
        public = {"record": {"external_charges": {key: 1048576}}, "roots": [{"path": "/tmp/pytest-of-barberb"}]}
        return public, [{"path": path, "bytes": 1048576}]

    def test_normative_sdk_hashed_label_joins_explicit_path_declaration(self):
        public, declared = self.storage_control()
        self.assertEqual(hook.check_external_storage_declarations(public, declared), ((declared[0]["path"], 1048576),))

    def test_physical_path_as_charge_key_is_not_the_sdk_convention(self):
        public, declared = self.storage_control()
        public["record"]["external_charges"] = {declared[0]["path"]: 1048576}
        self.refusal("actual_hashed_external_charge_differs", hook.check_external_storage_declarations, public, declared)

    def test_hashed_charge_requires_exact_actual_budget(self):
        public, declared = self.storage_control()
        declared[0]["bytes"] = 2097152
        self.refusal("actual_hashed_external_charge_differs", hook.check_external_storage_declarations, public, declared)

    def test_hashed_charge_does_not_infer_an_undeclared_root(self):
        public, _declared = self.storage_control()
        self.refusal("declared_external_storage_required", hook.check_external_storage_declarations, public, None)

    def test_explicit_storage_must_be_below_actual_named_root(self):
        public, declared = self.storage_control()
        public["roots"] = [{"path": "/different/native/root"}]
        self.refusal("declared_external_storage_outside_named_roots", hook.check_external_storage_declarations, public, declared)

    def test_duplicate_storage_declaration_refuses(self):
        public, declared = self.storage_control()
        declared.append(dict(declared[0]))
        self.refusal("canonical_unique_positive_external_storage_required", hook.check_external_storage_declarations, public, declared)

    def test_zero_storage_declaration_refuses(self):
        public, declared = self.storage_control()
        declared[0]["bytes"] = 0
        self.refusal("canonical_unique_positive_external_storage_required", hook.check_external_storage_declarations, public, declared)

    def owner_pair_control(self):
        # Necessary public shape only; neither object is a native owner/grant.
        first = {"reservation_id": "normative-primary", "record": {"child": None},
                 "resource_lease": {"lease_id": "normative-1", "parent_lease_id": "normative-parent"}}
        second = {"reservation_id": "normative-watchdog", "record": {"child": None},
                  "resource_lease": {"lease_id": "normative-2", "parent_lease_id": "normative-parent"}}
        return first, second

    def test_two_normative_fresh_sibling_shapes_are_necessary_only(self):
        self.assertIsNone(hook.check_owner_pair_projection(*self.owner_pair_control()))
        self.assertIsNone(hook._current())

    def test_same_normative_reservation_refuses(self):
        first, second = self.owner_pair_control()
        second["reservation_id"] = first["reservation_id"]
        self.refusal("distinct_sibling_native_owners_required", hook.check_owner_pair_projection, first, second)

    def test_same_normative_native_lease_refuses(self):
        first, second = self.owner_pair_control()
        second["resource_lease"]["lease_id"] = first["resource_lease"]["lease_id"]
        self.refusal("distinct_sibling_native_owners_required", hook.check_owner_pair_projection, first, second)

    def test_different_normative_ancestor_refuses(self):
        first, second = self.owner_pair_control()
        second["resource_lease"]["parent_lease_id"] = "other-normative-parent"
        self.refusal("distinct_sibling_native_owners_required", hook.check_owner_pair_projection, first, second)

    def test_preexisting_normative_primary_child_refuses(self):
        first, second = self.owner_pair_control()
        first["record"]["child"] = {"pid": 12345}
        self.refusal("fresh_native_owner_child_inventory_required", hook.check_owner_pair_projection, first, second)

    def test_preexisting_normative_watchdog_history_refuses(self):
        first, second = self.owner_pair_control()
        second["record"]["prior_children"] = [{"pid": 12345}]
        self.refusal("fresh_native_owner_child_inventory_required", hook.check_owner_pair_projection, first, second)

    def test_fake_watchdog_owner_is_not_a_native_grant(self):
        with patch.object(hook.importlib, "import_module", side_effect=AssertionError("fake owner must fail before import")):
            self.refusal("genuine_parented_owner_required", hook._guard_owner, object(), watchdog=True)

    def test_projection_is_not_a_private_actual_outcome(self):
        self.assertIsNone(hook.check_lifecycle_projection(projection()))
        self.assertIsNone(hook._current())
        self.assertFalse(hasattr(hook, "issue_native_acceptance"))


def missing_closeout_test(key):
    def control(self):
        value = projection()
        value[key] = False
        self.refusal("actual_closeout_missing:" + key, hook.check_lifecycle_projection, value)
    return control


for key in ("process_reaped", "lease_closed", "watchdog_reaped", "watchdog_owner_group_quiescent", "kernel_scope_quiescent",
            "exact_container_absent", "lease_root_absent", "provider_home_absent", "prompt_path_absent", "daemon_unchanged", "runtime_unchanged"):
    setattr(MemoryHandoffControls, "test_missing_" + key, missing_closeout_test(key))


def missing_order_test(key):
    def control(self):
        self.refusal("successful_handoff_must_follow_actual_closeout", hook.check_success_order,
                     tuple(item for item in ORDER if item != key))
    return control


for key in ORDER:
    setattr(MemoryHandoffControls, "test_order_requires_" + key, missing_order_test(key))


def invalid_projection_test(key, value, reason):
    def control(self):
        observed = projection()
        observed[key] = value
        self.refusal(reason, hook.check_lifecycle_projection, observed)
    return control


for name, key, value, reason in (
    ("boolean_returncode", "returncode", False, "successful_provider_return_required"),
    ("failed_returncode", "returncode", 1, "successful_provider_return_required"),
    ("unknown_returncode", "returncode", None, "successful_provider_return_required"),
    ("zero_provider", "provider_calls", 0, "exactly_one_task_provider_required"),
    ("two_providers", "provider_calls", 2, "exactly_one_task_provider_required"),
    ("boolean_count", "provider_calls", True, "exactly_one_task_provider_required"),
    ("missing_uuid", "execution_id", "", "fresh_execution_identity_required"),
    ("pid_as_uuid", "execution_id", "300", "fresh_execution_identity_required"),
    ("model_is_host", "model_host_pid", 100, "actual_model_process_role_required"),
    ("model_is_docker_client", "model_host_pid", 200, "actual_model_process_role_required"),
    ("model_boolean_pid", "model_host_pid", True, "actual_model_process_role_required"),
    ("missing_model_birth", "model_start_time_ticks", 0, "actual_model_birth_required"),
    ("boolean_model_birth", "model_start_time_ticks", True, "actual_model_birth_required"),
    ("model_drift", "model_id", "grok-4", "actual_supported_ordinary_route_required"),
    ("auth_fallback", "provider_trigger", "primary_quota_or_auth_unavailable", "actual_supported_ordinary_route_required"),
    ("medium_fallback", "reasoning_effort", "medium", "actual_supported_ordinary_route_required"),
):
    setattr(MemoryHandoffControls, "test_refuse_" + name, invalid_projection_test(key, value, reason))


if __name__ == "__main__":
    unittest.main()
