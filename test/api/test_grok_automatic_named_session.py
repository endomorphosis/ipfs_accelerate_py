"""Private source fixtures only: no native/provider/network qualification.

The protocol fixtures use real private pipes/files and a controlled transport,
never a CLI process or a replacement project module. Normal candidate package
origins and the genuine pytest/import seal belong to the external QA owner.
"""
from __future__ import annotations

import hashlib
import contextlib
import ctypes.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import uuid

from ipfs_accelerate_py.cli_runtime import grok_named_session as helper

KEY = "unit-fixture-not-an-actual-credential"
FINGERPRINT = hashlib.sha256(KEY.encode()).hexdigest()[:16]
SESSION = "b051c978-432f-4c42-8557-2539d50952d6"
BASELINE = {"process_group": 1234, "members_pid_start_ticks": {"1234": 5678}}
CONFIG = ('[cli]\nuse_leader = false\n[features]\ntitle_refresh = false\n'
          '[models]\ntemperature = 0.0\nsession_summary = "grok-4.7"\n'
          '[model."grok-4.7"]\ntemperature = 0.0\n')
for _vendor in ("claude", "cursor", "codex"):
    CONFIG += f"[compat.{_vendor}]\n" + "".join(
        f"{_name} = false\n" for _name in ("skills", "rules", "agents", "mcps", "hooks", "sessions"))


def owner_guard(_stage):
    """A pure fixture witness; not a genuine native-resource grant."""
    return None


def _pin(path, maximum=2*1024*1024):
    return helper.SourcePin(path, hashlib.sha256(path.read_bytes()).hexdigest(), maximum)


class InputPipe:
    def __init__(self, process):
        self.process = process
        self.closed = False
        self.frames = []

    def write(self, raw):
        self.frames.append(json.loads(raw))
        if self.process.after_write:
            self.process.after_write(len(self.frames))
        return len(raw)

    def close(self):
        self.closed = True


class ProtocolTransport:
    """Controlled pipe source; no actual subprocess, fabricated result only in tests."""
    def __init__(self, frames, after_write=None, exit_code=0):
        self.after_write = after_write
        self.exit_code = exit_code
        self.pid = 99999999
        self.returncode = None
        self.stdin = InputPipe(self)
        read, write = os.pipe()
        raw = b"".join(json.dumps(frame).encode()+b"\n" for frame in frames)
        os.write(write, raw)
        os.close(write)
        self.stdout = os.fdopen(read, "rb", buffering=0)
        read, write = os.pipe()
        os.close(write)
        self.stderr = os.fdopen(read, "rb", buffering=0)

    def poll(self):
        if self.stdin.closed:
            self.returncode = self.exit_code
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.exit_code
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def frames():
    results = [{"protocolVersion": 1}, {"sessionId": SESSION}, {"success": True},
               {"error": None, "result": {"success": True, "outcome": "closed"}}]
    return [{"jsonrpc": "2.0", "id": i, "result": value}
            for i, value in enumerate(results, 1)]


class AutomaticNamedSessionSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for name in ("workspace", "native", "audit", "initialization", "inputs"):
            (self.root/name).mkdir(mode=0o700)
        self.binary = self.root / "grok-unit-not-executed"
        self.binary.write_bytes(b"unit fixture, never executed\n")
        self.binary.chmod(0o700)
        self.configuration = self.root/"inputs/config-seed.toml"
        self.configuration.write_text(CONFIG)
        self.configuration.chmod(0o600)
        self.requirements = self.root/"inputs/sampling-policy.toml"
        self.requirements.write_text("[features]\ntitle_refresh = false\n")
        self.requirements.chmod(0o600)

    def request(self, **changes):
        values = dict(workspace=self.root/"workspace", native_home=self.root/"native",
            audit_directory=self.root/"audit", binary=_pin(self.binary),
            configuration=_pin(self.configuration, helper.CONFIG_LIMIT),
            owner_sampling_policy=_pin(self.requirements, helper.CONFIG_LIMIT),
            preparation=helper.NativeHomePreparation(self.root/"initialization", _pin(self.configuration, helper.CONFIG_LIMIT)),
            helper_source=_pin(Path(helper.__file__).resolve()),
            owner_source=_pin(Path(__file__).resolve()), owner_guard=owner_guard,
            source_generation="grok-automatic-named-session/source-fixture-v1",
            deadline_monotonic=time.monotonic()+60,
            allocation_session_id="unit-allocation",
            credential_slot="unit-slot", credential_fingerprint=FINGERPRINT)
        values.update(changes)
        return helper.NamedSessionRequest(**values)

    def lifetime(self, request=None, **changes):
        values = dict(model=helper.MODEL, base_command=[str(self.binary)],
            selected_api_key=KEY, selected_slot="unit-slot", selected_fingerprint=FINGERPRINT)
        values.update(changes)
        return helper.NamedSessionLifetime(request or self.request(), {"HOME": str(self.root)}, **values)

    def transport(self, frame_values=None, **changes):
        return ProtocolTransport(frames() if frame_values is None else frame_values, **changes)

    def prepared_home_fixture(self, active):
        """Private unit stand-in; native preparation has separate transport tests."""
        request = active.request
        request._target_configuration = helper._durable_configuration(
            request.native_home/"config.toml", request.configuration.read()[0])
        request._preparation_receipt = {"initialization_successful": True,
            "only_configuration_copied": True, "native_state_copied": False,
            "source_fixture_only": True, "production_configuration_qualified": False}
        request._phase = "home_prepared"
        return request._target_configuration

    def bootstrap(self, transport, lifetime=None):
        with patch.object(helper, "_group_births", return_value=BASELINE), \
             patch.object(helper.uuid, "uuid4", return_value=uuid.UUID(SESSION)), \
             patch.object(helper.subprocess, "Popen", return_value=transport) as native, \
             patch.object(helper, "prepare_native_home", side_effect=self.prepared_home_fixture):
            active = (lifetime or self.lifetime()).__enter__()
        return active, native

    def test_exact_supported_acp_wire_and_no_prompt(self):
        transport = self.transport()
        active, native = self.bootstrap(transport)
        self.assertEqual([f["method"] for f in transport.stdin.frames], list(helper.METHODS))
        self.assertEqual(transport.stdin.frames[1]["params"]["mcpServers"], [])
        self.assertEqual(active.session_id, SESSION)
        self.assertEqual(native.call_args.args[0], [str(self.binary), "agent", "--no-leader", "--model", "grok-4.7", "stdio"])
        self.assertNotIn("start_new_session", native.call_args.kwargs)
        self.assertIsNone(active.request.receipt()["zero_inference_posts"])
        self.assertFalse(active.request.receipt()["production_configuration_qualified"])

    def test_exact_uuid_mismatch_blocks_headless(self):
        rows = frames(); rows[1]["result"]["sessionId"] = str(uuid.uuid4())
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "uuid_mismatch"):
            self.bootstrap(self.transport(rows))

    def test_rename_failure_blocks_headless(self):
        rows = frames(); rows[2]["result"]["success"] = False
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "rename_not_acknowledged"):
            self.bootstrap(self.transport(rows))

    def test_partial_close_blocks_headless(self):
        rows = frames(); rows[3]["result"] = {"success": True}
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "close_not_acknowledged"):
            self.bootstrap(self.transport(rows))

    def test_rpc_error_blocks_headless(self):
        rows = frames(); rows[1] = {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601}}
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "rpc_error_or_shape"):
            self.bootstrap(self.transport(rows))

    def test_incoming_request_blocks_headless(self):
        rows = frames(); rows[0]["method"] = "client/tool"
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "rpc_error_or_shape"):
            self.bootstrap(self.transport(rows))

    def test_deadline_after_rename_blocks_close_and_headless(self):
        request = self.request()
        def expire(sequence):
            if sequence == 3:
                object.__setattr__(request, "deadline_monotonic", time.monotonic()+.01)
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "absolute_deadline"):
            self.bootstrap(self.transport(after_write=expire), self.lifetime(request))

    def test_nonzero_bootstrap_process_blocks_headless(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "process_nonzero"):
            self.bootstrap(self.transport(exit_code=4))

    def test_duplicate_json_rejected(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "duplicate"):
            helper._object(b'{"id":1,"id":2}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "nonfinite"):
            helper._object(b'{"x":NaN}')

    def test_oversized_frame_rejected_before_parse(self):
        with patch.object(helper.json, "loads", side_effect=AssertionError("must not parse")):
            with self.assertRaisesRegex(helper.NamedSessionRefusal, "frame_byte_bound"):
                helper._object(b" "*(helper.FRAME_LIMIT+1))

    def test_boolean_rpc_id_rejected(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "rpc_error_or_shape"):
            helper._ack({"jsonrpc": "2.0", "id": True, "result": {}}, 1)

    def test_boolean_protocol_version_rejected(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "version_mismatch"):
            helper._validate_result(1, {"protocolVersion": True}, SESSION)

    def test_expired_owner_before_bootstrap(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "absolute_deadline"):
            self.request(deadline_monotonic=time.monotonic()+1)

    def test_wrong_owner_guard_origin(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "guard_source_origin"):
            self.request(owner_guard=lambda _stage: None, owner_source=_pin(self.binary))

    def test_changed_config_rejected_before_bootstrap(self):
        request = self.request()
        self.configuration.write_text(CONFIG+"# changed\n")
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "source_pin_changed"):
            self.lifetime(request)

    def test_native_state_must_be_fresh(self):
        (self.root/"native/sessions").mkdir()
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "native_home_not_fresh"):
            self.request()

    def test_request_is_not_serializable(self):
        with self.assertRaisesRegex(TypeError, "cannot be serialized"):
            pickle.dumps(self.request())

    def test_owner_deadline_cannot_be_extended_after_preparation(self):
        request = self.request()
        with self.assertRaisesRegex(AttributeError, "immutable"):
            request.deadline_monotonic = time.monotonic()+600

    def test_credential_slot_mismatch_blocks_bootstrap(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "credential_binding_mismatch"):
            self.lifetime(selected_slot="different-slot")

    def test_selected_key_mismatch_blocks_bootstrap(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "credential_binding_mismatch"):
            self.lifetime(selected_api_key="another-unit-fixture")

    def test_environment_and_repr_never_expose_key(self):
        environment = dict(os.environ)
        active = self.lifetime()
        self.assertEqual(active.environment["XAI_API_KEY"], KEY)
        self.assertNotIn(KEY, repr(active)+repr(active.request)+json.dumps(active.request.receipt()))
        self.assertEqual(environment, dict(os.environ))

    def test_extra_base_arguments_rejected(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "base_command"):
            self.lifetime(base_command=[str(self.binary), "--resume", SESSION])

    def test_native_selectors_refused_even_zero(self):
        request = self.request()
        for key, value in (("resume_session_id", SESSION), ("chat_session_id", 0),
                           ("session_id", False), ("continue_session", True)):
            with self.subTest(key=key), self.assertRaisesRegex(helper.NamedSessionRefusal, "conflicting_native_selector"):
                helper.validate_preparation_option("grok_cli", request, {key: value})

    def test_foreign_provider_refused(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "exact_grok_owner_request"):
            helper.validate_preparation_option("codex_cli", self.request(), {})

    def test_plain_dictionary_cannot_grant_preparation(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "exact_grok_owner_request"):
            helper.validate_preparation_option("grok_cli", {}, {})

    def test_private_environment_keeps_state_and_deadline(self):
        with patch.object(helper, "_group_births", return_value=BASELINE):
            active = self.lifetime()
            home = active.environment["GROK_HOME"]
            active.request._phase = "headless_ready"
            object.__setattr__(active.request, "deadline_monotonic", time.monotonic()+12)
            timeout = active.headless_timeout(90)
            self.assertLess(timeout, 10)
            self.assertEqual(active.environment["GROK_HOME"], home)
            self.assertEqual(Path(home), self.root/"native")

    def test_surviving_descendant_blocks_headless(self):
        active = self.lifetime()
        active.request._phase = "headless_ready"
        with patch.object(helper, "_group_births", return_value={"foreign": "not baseline"}):
            with self.assertRaisesRegex(helper.NamedSessionRefusal, "group_not_restored"):
                active.headless_timeout(90)

    def test_primary_exception_survives_failed_closeout(self):
        active = self.lifetime()
        error = ValueError("primary-fixture-failure")
        with patch.object(helper, "_group_births", return_value={"survivor": 1}):
            self.assertFalse(active.__exit__(ValueError, error, None))
        self.assertEqual(active.request._phase, "failed")
        self.assertEqual(active.environment, {})

    def test_session_not_captured_before_usable_response(self):
        request = self.request()
        request._session = SESSION
        request._phase = "headless_closed"
        self.assertEqual(request.completed_session_id, "")
        request.accept_response("unit candidate")
        self.assertEqual(request.completed_session_id, SESSION)

    def test_inline_key_refused(self):
        self.configuration.write_text(CONFIG.replace('temperature = 0.0\n[compat.claude]',
            'temperature = 0.0\napi_key = "unit-only"\n[compat.claude]'))
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "inline_credential"):
            self.lifetime()

    def test_overlay_refused_before_bootstrap(self):
        request = self.request()
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "overlay_not_qualified"):
            helper.NamedSessionLifetime(request, {"GROK_CONFIG": "{}"}, model=helper.MODEL,
                base_command=[str(self.binary)], selected_api_key=KEY,
                selected_slot="unit-slot", selected_fingerprint=FINGERPRINT)

    def test_summary_model_mismatch_refused(self):
        self.configuration.write_text(CONFIG.replace('session_summary = "grok-4.7"', 'session_summary = "other"'))
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "sampling_profile_missing"):
            self.lifetime()

    def test_requirements_title_conflict_refused(self):
        self.requirements.write_text("[features]\ntitle_refresh = true\n")
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "sampling_profile_missing"):
            self.lifetime()

    def provider_call(self, *, bootstrap_failure=None, native_failure=None,
                      allocation_context=("unit-allocation", "cli")):
        from ipfs_accelerate_py import llm_router as router
        request = self.request()
        observed = []
        def bootstrap(active):
            if bootstrap_failure:
                raise bootstrap_failure
            request._receipt = {"bootstrap_successful": True, "lifetime_closed": False,
                "response_accepted": False, "production_configuration_qualified": False}
            return SESSION
        def headless(command, **kwargs):
            observed.append({"command": list(command), "environment": dict(kwargs["env"]),
                             "timeout": kwargs["timeout"]})
            if native_failure:
                raise native_failure
            return subprocess.CompletedProcess(command, 0,
                stdout=json.dumps({"text": "unit planning response", "stopReason": "end_turn"}), stderr="")
        with contextlib.chdir(request.workspace), \
             patch.object(router, "_grok_cli_command", return_value=str(self.binary)), \
             patch.object(router, "_cli_available", return_value=True), \
             patch.object(router, "_grok_requirements_allow_zero"), \
             patch.object(router, "_allocation_context", return_value=allocation_context), \
             patch.object(router._API_KEY_CONTEXT, "slot", "unit-slot", create=True), \
             patch.object(router._API_KEY_CONTEXT, "fingerprint", FINGERPRINT, create=True), \
             patch.object(helper, "_group_births", return_value=BASELINE), \
             patch.object(helper, "_bootstrap", side_effect=bootstrap), \
             patch.object(helper, "prepare_native_home", side_effect=self.prepared_home_fixture) as prepare, \
             patch.object(router.subprocess, "run", side_effect=headless) as launch:
            provider = router._get_grok_cli_provider()
            try:
                output = provider.generate("unit only", model_name="grok-4.7", temperature=0,
                    timeout=90, api_key=KEY, grok_named_session_request=request,
                    grok_cli_cmd=[str(self.binary)],
                    grok_tools="read_file", grok_disallowed_tools="read_file,search_tool,use_tool",
                    grok_permission_mode="dontAsk", grok_max_turns=2)
            finally:
                self.last_headless_count = launch.call_count
                self.last_preparation_count = prepare.call_count
                self.last_request = request
        return output, request, observed

    def test_provider_handoff_uses_one_resume_selected_key_and_remaining_budget(self):
        output, request, observed = self.provider_call()
        self.assertEqual(output, "unit planning response")
        self.assertEqual(len(observed), 1)
        command = observed[0]["command"]
        self.assertEqual(command.count("--resume"), 1)
        self.assertEqual(command[command.index("--resume")+1], SESSION)
        self.assertNotIn("--session-id", command)
        self.assertNotIn("--continue", command)
        self.assertEqual(observed[0]["environment"]["XAI_API_KEY"], KEY)
        self.assertEqual(observed[0]["environment"]["GROK_HOME"], str(request.native_home))
        self.assertLess(observed[0]["timeout"], 60)
        self.assertEqual(request.completed_session_id, SESSION)
        self.assertTrue(request.receipt()["response_accepted"])
        self.assertFalse(request.receipt()["production_configuration_qualified"])

    def test_bootstrap_failure_launches_no_headless_process(self):
        failure = ValueError("unit bootstrap primary")
        with self.assertRaises(ValueError) as observed:
            self.provider_call(bootstrap_failure=failure)
        self.assertIs(observed.exception, failure)
        self.assertEqual(self.last_headless_count, 0)
        self.assertEqual(self.last_request.completed_session_id, "")

    def test_headless_primary_exception_kept_and_session_not_captured(self):
        failure = RuntimeError("unit headless primary")
        with self.assertRaises(RuntimeError) as observed:
            self.provider_call(native_failure=failure)
        self.assertIs(observed.exception, failure)
        self.assertEqual(self.last_request.completed_session_id, "")

    def test_allocation_suppresses_create_on_first_only_for_typed_request(self):
        from ipfs_accelerate_py import llm_router as router
        from ipfs_accelerate_py import llm_allocation
        request = self.request()
        store = SimpleNamespace(get_cli_session=lambda *_args: None)
        with patch.object(router, "_allocation_context", return_value=("unit-allocation", None)), \
             patch.object(llm_allocation, "get_allocation_store", return_value=store):
            options = {"grok_named_session_request": request}
            self.assertIs(router._bind_cli_session_kwargs("grok_cli", options), options)
            self.assertNotIn("chat_session_id", options)
            default = router._bind_cli_session_kwargs("grok_cli", {})
            self.assertEqual(default["chat_session_id"], "unit-allocation")

    def test_allocation_cannot_replace_an_existing_native_session(self):
        from ipfs_accelerate_py import llm_router as router
        from ipfs_accelerate_py import llm_allocation
        request = self.request()
        store = SimpleNamespace(get_cli_session=lambda *_args: {"native_session_id": SESSION})
        with patch.object(router, "_allocation_context", return_value=("unit-allocation", None)), \
             patch.object(llm_allocation, "get_allocation_store", return_value=store):
            with self.assertRaisesRegex(ValueError, "fresh allocation"):
                router._bind_cli_session_kwargs("grok_cli", {"grok_named_session_request": request})

    def test_allocation_capture_waits_for_validated_response(self):
        from ipfs_accelerate_py import llm_router as router
        from ipfs_accelerate_py import llm_allocation
        request = self.request()
        persisted = []
        store = SimpleNamespace(upsert_cli_session=lambda **row: persisted.append(row),
            get_cli_session=lambda *_args: ({"allocation_session_id": persisted[-1]["session_id"],
                "provider": persisted[-1]["provider"], "native_session_id": persisted[-1]["native_session_id"],
                "workspace": persisted[-1]["workspace"]} if persisted else {}))
        with patch.object(router, "_allocation_context", return_value=("unit-allocation", None)), \
             patch.object(llm_allocation, "get_allocation_store", return_value=store):
            router._capture_cli_session("grok_cli", {"grok_named_session_request": request})
            self.assertEqual(persisted, [])
            request._session, request._phase = SESSION, "headless_closed"
            request.accept_response("unit accepted response")
            router._capture_cli_session("grok_cli", {"grok_named_session_request": request})
            self.assertEqual(persisted[0]["native_session_id"], SESSION)

    def runtime_call(self, *, observed_session=SESSION, provider="grok_cli", resume=None,
                     response_lifetime_closed=True, failure=None, allocation_id="unit-allocation"):
        from ipfs_accelerate_py import llm_router as router
        from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
        from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting, multi_supervisor_runner
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        from ipfs_accelerate_py.llm_allocation import intelligence_index
        from test.api.test_grok_planning_temperature_zero import _fixture_library_resolver
        request = self.request(allocation_session_id=allocation_id)
        calls = []
        def generate(_prompt, **kwargs):
            calls.append(kwargs)
            self.assertIs(kwargs["grok_named_session_request"], request)
            self.assertEqual(kwargs["allocation_session_id"], request.allocation_session_id)
            self.assertEqual(kwargs["grok_cli_cmd"], [str(request.binary.path)])
            self.assertEqual(kwargs["temperature"], 0)
            self.assertEqual(kwargs["max_tokens"], 4096)
            self.assertEqual(kwargs["timeout"], 90)
            self.assertEqual(kwargs["grok_tools"], "read_file")
            self.assertEqual(kwargs["grok_disallowed_tools"], "read_file,search_tool,use_tool")
            self.assertFalse(kwargs["allow_cross_provider_fallback"])
            self.assertFalse(kwargs["allow_local_fallback"])
            if failure:
                raise failure
            request._session, request._phase = SESSION, "response_accepted"
            request._receipt = {"bootstrap_successful": True,
                "lifetime_closed": response_lifetime_closed, "response_accepted": True,
                "production_configuration_qualified": False}
            set_last_cli_observation("grok_cli", {"exit_code": 0, "session_id": observed_session})
            return "unit planning response"
        with patch.dict(os.environ), \
             patch.object(ctypes.util, "find_library", side_effect=_fixture_library_resolver()), \
             patch.object(subprocess, "check_output", return_value=str(request.workspace)), \
             contextlib.chdir(request.workspace), \
             patch.object(intelligence_index, "discover_available_providers", return_value=[provider]), \
             patch.object(intelligence_index, "select_efficient_route", return_value=SimpleNamespace(
                 provider=provider, model_name="grok-4.7", catalog_revision="unit-only")), \
             patch.object(router, "get_llm_provider", return_value=object()), \
             patch.object(router, "generate_text", side_effect=generate), \
             patch.object(context_delivery_accounting, "observe_invocation", return_value={"status": "unit-only"}), \
             contextlib.redirect_stdout(io.StringIO()):
            for name in (*multi_supervisor_runner.DATABASE_PROGRAM_ENV_NAMES,
                         *multi_supervisor_runner.STATE_CREDENTIAL_ENV_NAMES,
                         "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN"):
                os.environ.pop(name, None)
            try:
                output, receipt = runner.run(prompt="unit planning source fixture", provider=provider,
                    model="grok-4.7", purpose="planning", timeout=90, max_output_tokens=4096,
                    resume_session_id=resume, grok_named_session_request=request)
            finally:
                self.last_runtime_call_count = len(calls)
        self.assertEqual(output, "unit planning response")
        self.assertNotEqual(receipt["invocation_id"], request.allocation_session_id)
        self.assertEqual(receipt["provider_invocation_policy"]["allocation_session_id"], request.allocation_session_id)
        return request, receipt

    def test_main_runtime_accepts_integer_timeout_and_exact_automatic_result(self):
        request, receipt = self.runtime_call()
        self.assertEqual(self.last_runtime_call_count, 1)
        policy = receipt["provider_invocation_policy"]
        self.assertTrue(policy["owned_named_session_requested"])
        self.assertEqual(policy["tools_profile"], "none")
        self.assertEqual(policy["native_resume_session_id"], SESSION)
        self.assertEqual(policy["owned_named_session"], request.receipt())
        self.assertFalse(policy["production_configuration_qualified"])
        self.assertFalse(receipt["completion_authority"])

    def test_main_runtime_explicit_resume_conflict_prevents_transport(self):
        with self.assertRaisesRegex(ValueError, "exclusive Grok planning route"):
            self.runtime_call(resume=SESSION)
        self.assertEqual(self.last_runtime_call_count, 0)

    def test_main_runtime_foreign_provider_prevents_transport(self):
        with self.assertRaisesRegex(ValueError, "exclusive Grok planning route"):
            self.runtime_call(provider="codex_cli")
        self.assertEqual(self.last_runtime_call_count, 0)

    def test_main_runtime_native_observation_mismatch_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "lifecycle or native observation differs"):
            self.runtime_call(observed_session="d051c978-432f-4c42-8557-2539d50952d6")

    def test_main_runtime_missing_lifetime_close_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "lifecycle or native observation differs"):
            self.runtime_call(response_lifetime_closed=False)

    def test_main_runtime_preserves_primary_provider_failure(self):
        failure = RuntimeError("unit automatic provider primary")
        with self.assertRaises(RuntimeError) as observed:
            self.runtime_call(failure=failure)
        self.assertIs(observed.exception, failure)

    def public_options(self, request):
        return dict(provider="grok_cli", model_name="grok-4.7", temperature=0,
            side_effecting=True, allow_local_fallback=False,
            allow_cross_provider_fallback=False, grok_named_session_request=request,
            allocation_session_id=getattr(request, "allocation_session_id", "unit-allocation"))

    def public_refusal(self, options, reason="exact_public_dispatch_policy"):
        from ipfs_accelerate_py import llm_router as router
        with patch.object(router, "get_default_router_deps", side_effect=AssertionError("dependency work forbidden")) as deps:
            with self.assertRaisesRegex(helper.NamedSessionRefusal, reason):
                router.generate_text("unit public entry", **options)
            deps.assert_not_called()

    def test_public_entry_default_fallbacks_refuse_before_dependencies(self):
        options = self.public_options(self.request())
        options.pop("allow_cross_provider_fallback")
        options.pop("allow_local_fallback")
        self.public_refusal(options)

    def test_public_entry_foreign_provider_or_model_refuse_before_dependencies(self):
        request = self.request()
        for key, value in (("provider", "codex_cli"), ("model_name", "other-model")):
            with self.subTest(key=key):
                options = self.public_options(request); options[key] = value
                self.public_refusal(options)

    def test_public_entry_nonzero_nonfinite_boolean_or_missing_temperature_refuse(self):
        request = self.request()
        for value in (True, 1, float("nan"), None):
            with self.subTest(value=value):
                options = self.public_options(request)
                if value is None:
                    options.pop("temperature")
                else:
                    options["temperature"] = value
                self.public_refusal(options)

    def test_public_entry_missing_side_effects_refuses_before_dependencies(self):
        options = self.public_options(self.request())
        options.pop("side_effecting")
        self.public_refusal(options)

    def test_public_entry_dictionary_cannot_select_preparation(self):
        self.public_refusal(self.public_options({}), "exact_grok_owner_request")

    def test_public_entry_valid_policy_reaches_dependencies_without_dispatch(self):
        from ipfs_accelerate_py import llm_router as router
        sentinel = RuntimeError("unit dependency boundary, no transport")
        options = self.public_options(self.request())
        with patch.object(router, "get_default_router_deps", side_effect=sentinel) as deps:
            with self.assertRaises(RuntimeError) as observed:
                router.generate_text("unit public entry", **options)
            self.assertIs(observed.exception, sentinel)
            deps.assert_called_once_with()


class NativeHomePreparationSourceTests(unittest.TestCase):
    """Controlled private transport; no installed CLI or native qualification."""
    setUp = AutomaticNamedSessionSourceTests.setUp
    request = AutomaticNamedSessionSourceTests.request
    lifetime = AutomaticNamedSessionSourceTests.lifetime

    def prepare(self, *, after_write=None, exit_code=0):
        transport = ProtocolTransport(frames()[:1], after_write=after_write, exit_code=exit_code)
        with patch.object(helper, "_group_births", return_value=BASELINE), \
             patch.object(helper.subprocess, "Popen", return_value=transport) as launched:
            active = self.lifetime()
            self.last_active = active
            self.last_transport = transport
            result = helper.prepare_native_home(active)
        return active, result, transport, launched

    def test_initialize_only_then_copy_configuration_without_native_state(self):
        def initialized(_count):
            home = self.root/"initialization"
            (home/"config.toml").write_text(CONFIG+"\n[marketplace]\ndefault_skills_installs_purged = true\n")
            (home/"agent_id").write_text("controlled fixture metadata, never copied")
        active, selected, transport, launched = self.prepare(after_write=initialized)
        self.assertEqual([row["method"] for row in transport.stdin.frames], ["initialize"])
        self.assertEqual(launched.call_args.kwargs["env"]["GROK_HOME"], str(self.root/"initialization"))
        self.assertEqual({p.name for p in active.request.native_home.iterdir()}, {"config.toml"})
        self.assertEqual(selected.path, self.root/"native/config.toml")
        self.assertEqual(active.request._phase, "home_prepared")
        report = active.request._preparation_receipt
        self.assertTrue(report["initialization_successful"])
        self.assertTrue(report["only_configuration_copied"])
        self.assertFalse(report["native_state_copied"])
        self.assertFalse(report["session_new_attempted"])
        self.assertIsNone(report["zero_inference_posts"])
        self.assertFalse(report["production_configuration_qualified"])

    def test_initializer_can_leave_configuration_unchanged(self):
        active, selected, _, _ = self.prepare()
        self.assertEqual(selected.read()[0], CONFIG.encode())
        self.assertTrue(active.request._preparation_receipt["process_group_baseline_restored"])

    def test_initialization_does_not_repin_a_changed_target(self):
        active, selected, _, _ = self.prepare()
        selected.path.write_text(CONFIG+"# changed after preparation\n")
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "source_pin_changed"):
            active.request.assert_current("bootstrap_protocol")

    def test_native_policy_cache_is_not_owner_policy(self):
        def created_cache(_count):
            (self.root/"initialization/requirements.toml").write_text("[features]\ntitle_refresh=false\n")
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "native_policy_output_not_qualified"):
            self.prepare(after_write=created_cache)
        self.assertEqual(self.last_active.request._phase, "failed")
        self.assertEqual(self.last_active.environment, {})
        self.assertEqual(list((self.root/"native").iterdir()), [])

    def test_native_temperature_change_blocks_configuration_copy(self):
        def changed(_count):
            (self.root/"initialization/config.toml").write_text(CONFIG.replace("temperature = 0.0", "temperature = 0.7"))
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "initialization_configuration_changed"):
            self.prepare(after_write=changed)
        self.assertEqual(list((self.root/"native").iterdir()), [])

    def test_only_the_observed_marker_is_permitted(self):
        candidates = [CONFIG+"\n[marketplace]\ndefault_skills_installs_purged = false\n",
                      CONFIG+"\n[other]\nextra = true\n",
                      CONFIG.replace('session_summary = "grok-4.7"', 'session_summary = "other"'),
                      CONFIG.replace("title_refresh = false", "title_refresh = true")]
        for changed in candidates:
            with self.subTest(changed=changed), self.assertRaisesRegex(
                    helper.NamedSessionRefusal, "initialization_configuration_changed"):
                helper._normalized_configuration(CONFIG.encode(), changed.encode())

    def test_nonzero_initializer_never_prepares_a_target(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "process_nonzero"):
            self.prepare(exit_code=1)
        self.assertEqual(self.last_active.request._phase, "failed")
        self.assertEqual(list((self.root/"native").iterdir()), [])

    def test_preparation_cannot_be_reused_as_automatic_retry(self):
        active, _, _, _ = self.prepare()
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "already_used"):
            helper.prepare_native_home(active)

    def test_preparation_requires_the_typed_lifetime(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "exact_lifetime"):
            helper.prepare_native_home({"native_grant": True})

    def test_preparation_requires_the_typed_plan(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "source_pin_types_or_bounds"):
            self.request(preparation={"home": str(self.root/"initialization")})

    def test_initialization_home_must_be_fresh(self):
        (self.root/"initialization/agent_id").write_text("old private fixture")
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "native_home_not_fresh"):
            self.request()

    def test_owner_policy_must_live_outside_native_cache(self):
        path = self.root/"native/requirements.toml"
        path.write_text("[features]\ntitle_refresh=false\n");path.chmod(0o600)
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "private_configuration_origin"):
            self.request(owner_sampling_policy=_pin(path, helper.CONFIG_LIMIT))

    def test_public_preparation_then_target_entry_does_not_initialize_again(self):
        active, _, _, _ = self.prepare()
        transport = ProtocolTransport(frames())
        with patch.object(helper, "_group_births", return_value=BASELINE), \
             patch.object(helper.uuid, "uuid4", return_value=uuid.UUID(SESSION)), \
             patch.object(helper.subprocess, "Popen", return_value=transport) as launched:
            self.assertIs(active.__enter__(), active)
        self.assertEqual(launched.call_count, 1)
        self.assertEqual([row["method"] for row in transport.stdin.frames], list(helper.METHODS))
        self.assertEqual(active.session_id, SESSION)

if __name__ == "__main__":
    unittest.main()


class AllocationMetadataFixture:
    """Public API argument fixture; no allocation database or actual secret."""
    def __init__(self):
        other = "second-unit-fixture-key-not-an-actual-credential"
        self.rows = [dict(backend="grok_cli", slot_id="unit-slot", key_fingerprint=FINGERPRINT,
                          enabled=True, remaining_requests=1, label="owner"),
                     dict(backend="grok_cli", slot_id="other-slot", enabled=True,
                          key_fingerprint=hashlib.sha256(other.encode()).hexdigest()[:16],
                          remaining_requests=10000, label="healthier competitor")]
        self.bound = {}
        self.writes = []
        self.reads = 0
        self.native_rows = {}
        self.credentials = {"api_key/grok_cli/unit-slot": KEY,
                            "api_key/grok_cli/other-slot": other}

    def bind_session_api_key(self, session_id, backend, slot_id):
        self.writes.append((session_id, backend, slot_id))
        self.bound = dict(session_id=session_id, backend=backend, slot_id=slot_id)

    def get_session_api_key(self, session_id, backend):
        self.reads += 1
        return dict(self.bound) if self.bound.get("session_id") == session_id and self.bound.get("backend") == backend else {}

    def list_api_key_slots(self, backend):
        self.reads += 1
        return [dict(row) for row in self.rows if row["backend"] == backend]

    def resolve_session(self, value):
        return {"allocation_session_id": value}

    def get_session(self, _session_id):
        return {}


class SlotSecretFixture:
    """A fake selected-slot manager supplied through the real public API."""
    def __init__(self, store, hook=None):
        self.store, self.hook, self.calls = store, hook, []

    def get_credential(self, name):
        self.calls.append(name)
        if self.hook is not None:
            self.hook()
        return self.store.credentials.get(name)


class StrictAllocationBindingSourceTests(unittest.TestCase):
    setUp = AutomaticNamedSessionSourceTests.setUp
    request = AutomaticNamedSessionSourceTests.request
    lifetime = AutomaticNamedSessionSourceTests.lifetime
    public_options = AutomaticNamedSessionSourceTests.public_options
    public_refusal = AutomaticNamedSessionSourceTests.public_refusal
    provider_call = AutomaticNamedSessionSourceTests.provider_call
    prepared_home_fixture = AutomaticNamedSessionSourceTests.prepared_home_fixture
    runtime_call = AutomaticNamedSessionSourceTests.runtime_call

    def selection(self, *, hook=None):
        from ipfs_accelerate_py.llm_allocation import api_key_slots as slots
        store = AllocationMetadataFixture()
        # Genuine public owner prebinding. Strict selection below must not write.
        slots.bind_session_api_key("unit-allocation", "grok_cli", "unit-slot", store=store)
        self.assertEqual(store.writes, [("unit-allocation", "grok_cli", "unit-slot")])
        store.writes.clear()
        required = slots.RequiredApiKeyBinding("grok_cli", "unit-allocation", "unit-slot", FINGERPRINT)
        manager = SlotSecretFixture(store, hook)
        return slots, store, manager, required

    def select(self, slots, store, manager, required):
        return slots.select_api_key("grok_cli", session_id="unit-allocation", store=store,
                                    secrets=manager, required_binding=required)

    def test_request_allocation_identity_is_bounded_and_immutable(self):
        for value in (None, True, 1, "", " unit-allocation", "unit-allocation ", "a"*129, "bad/id", "a\n"):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(helper.NamedSessionRefusal, "owner_allocation_session_invalid"):
                    self.request(allocation_session_id=value)
        request = self.request()
        with self.assertRaisesRegex(AttributeError, "immutable"):
            request.allocation_session_id = "replacement"

    def test_public_missing_or_conflicting_allocation_refuses_before_dependencies(self):
        request = self.request()
        for value in (None, "other-allocation", " unit-allocation", "unit-allocation ", True):
            with self.subTest(value_type=type(value).__name__):
                options = self.public_options(request)
                if value is None:
                    options.pop("allocation_session_id")
                else:
                    options["allocation_session_id"] = value
                self.public_refusal(options, "allocation_identity_mismatch")

    def test_public_inline_key_or_slot_override_refuses_before_dependencies(self):
        request = self.request()
        for field in ("api_key", "_api_key_slot", "_api_key_fingerprint"):
            with self.subTest(field=field):
                options = self.public_options(request); options[field] = "unbound-unit-override"
                self.public_refusal(options, "inline_credential_override")

    def test_alias_resolution_cannot_redirect_owner_allocation(self):
        from ipfs_accelerate_py import llm_router as router, llm_allocation
        store = SimpleNamespace(resolve_session=lambda _value: {"allocation_session_id": "different-allocation"})
        with patch.object(llm_allocation, "get_allocation_store", return_value=store):
            with self.assertRaisesRegex(helper.NamedSessionRefusal, "allocation_identity_mismatch"):
                router.generate_text("private alias refusal fixture", deps=object(), **self.public_options(self.request()))

    def test_cli_binding_refuses_actual_allocation_context_drift(self):
        from ipfs_accelerate_py import llm_router as router
        with patch.object(router, "_allocation_context", return_value=("different-allocation", "cli")):
            with self.assertRaisesRegex(helper.NamedSessionRefusal, "allocation_identity_mismatch"):
                router._bind_cli_session_kwargs("grok_cli", {"grok_named_session_request": self.request()})

    def test_native_session_lookup_failure_is_strict_only_for_automatic(self):
        from ipfs_accelerate_py import llm_router as router, llm_allocation
        failure = RuntimeError("unit lookup unavailable")
        def lookup(*_args):
            raise failure
        with patch.object(router, "_allocation_context", return_value=("unit-allocation", "cli")), \
             patch.object(llm_allocation, "get_allocation_store", return_value=SimpleNamespace(get_cli_session=lookup)):
            with self.assertRaises(RuntimeError) as observed:
                router._bind_cli_session_kwargs("grok_cli", {"grok_named_session_request": self.request()})
            self.assertIs(observed.exception, failure)
            self.assertEqual(router._bind_cli_session_kwargs("grok_cli", {})["chat_session_id"], "unit-allocation")

    def test_provider_context_drift_blocks_both_preparation_and_headless(self):
        with self.assertRaisesRegex(helper.NamedSessionRefusal, "allocation_identity_mismatch"):
            self.provider_call(allocation_context=("different-allocation", "cli"))
        self.assertEqual(self.last_preparation_count, 0)
        self.assertEqual(self.last_headless_count, 0)

    def test_runtime_keeps_owner_allocation_distinct_from_invocation(self):
        request, receipt = self.runtime_call()
        self.assertEqual(receipt["provider_invocation_policy"]["allocation_session_id"], request.allocation_session_id)
        self.assertNotEqual(receipt["invocation_id"], request.allocation_session_id)

    def test_runtime_invocation_collision_refuses_before_provider(self):
        with patch.object(helper.uuid, "uuid4", return_value=uuid.UUID(SESSION)):
            with self.assertRaisesRegex(ValueError, "identities must differ"):
                self.runtime_call(allocation_id=uuid.UUID(SESSION).hex)
        self.assertEqual(self.last_runtime_call_count, 0)

    def test_strict_selection_uses_bound_slot_over_healthier_competitor(self):
        slots, store, manager, required = self.selection()
        selected = self.select(slots, store, manager, required)
        self.assertEqual((selected["slot_id"], selected["key_fingerprint"], selected["secret"]),
                         ("unit-slot", FINGERPRINT, KEY))
        self.assertTrue(selected["bound"])
        self.assertEqual(manager.calls, ["api_key/grok_cli/unit-slot"])
        self.assertEqual(store.writes, [])

    def test_strict_missing_binding_never_auto_binds_or_resolves_secret(self):
        slots, store, manager, required = self.selection(); store.bound.clear()
        with self.assertRaisesRegex(ValueError, "session_binding_mismatch"):
            self.select(slots, store, manager, required)
        self.assertEqual((store.writes, manager.calls), ([], []))

    def test_strict_deleted_disabled_or_ambiguous_slot_never_rebinds(self):
        for mode in ("deleted", "disabled", "ambiguous"):
            with self.subTest(mode=mode):
                slots, store, manager, required = self.selection()
                if mode == "deleted":
                    store.rows.pop(0)
                elif mode == "disabled":
                    store.rows[0]["enabled"] = False
                else:
                    store.rows.append(dict(store.rows[0]))
                with self.assertRaises(ValueError):
                    self.select(slots, store, manager, required)
                self.assertEqual((store.writes, manager.calls), ([], []))

    def test_strict_metadata_fingerprint_drift_precedes_secret_resolution(self):
        slots, store, manager, required = self.selection()
        store.rows[0]["key_fingerprint"] = "0"*16
        with self.assertRaisesRegex(ValueError, "slot_metadata_mismatch"):
            self.select(slots, store, manager, required)
        self.assertEqual((store.writes, manager.calls), ([], []))

    def test_strict_actual_secret_missing_or_wrong_never_uses_other_slot(self):
        for value in (None, "unit-secret-with-wrong-fingerprint"):
            with self.subTest(value_type=type(value).__name__):
                slots, store, manager, required = self.selection()
                store.credentials["api_key/grok_cli/unit-slot"] = value
                with self.assertRaisesRegex(ValueError, "secret_mismatch"):
                    self.select(slots, store, manager, required)
                self.assertEqual(manager.calls, ["api_key/grok_cli/unit-slot"])
                self.assertEqual(store.writes, [])

    def test_strict_selection_rereads_binding_and_stable_slot_metadata(self):
        for mode in ("binding", "disabled", "fingerprint"):
            with self.subTest(mode=mode):
                slots, store, manager, required = self.selection()
                def drift():
                    if mode == "binding":
                        store.bound["slot_id"] = "other-slot"
                    elif mode == "disabled":
                        store.rows[0]["enabled"] = False
                    else:
                        store.rows[0]["key_fingerprint"] = "0"*16
                manager.hook = drift
                with self.assertRaises(ValueError):
                    self.select(slots, store, manager, required)
                self.assertEqual(store.writes, [])
                self.assertEqual(manager.calls, ["api_key/grok_cli/unit-slot"])

    def test_strict_security_readback_ignores_volatile_stats_and_label(self):
        slots, store, manager, required = self.selection()
        def progress():
            store.rows[0].update(label="updated label", total_cost_usd=15, remaining_requests=0)
        manager.hook = progress
        self.assertEqual(self.select(slots, store, manager, required)["slot_id"], "unit-slot")
        self.assertEqual(store.writes, [])

    def test_strict_selection_requires_exact_typed_identity_before_store_work(self):
        slots, store, manager, required = self.selection()
        class InvalidStrictBackend:
            def __str__(self):
                raise AssertionError("strict selection must not normalize an untyped backend")
        for backend, binding, session in (("grok_cli", {}, "unit-allocation"),
                ("grok_cli", required, "other-allocation"),
                (InvalidStrictBackend(), required, "unit-allocation")):
            with self.subTest(binding_type=type(binding).__name__, backend_type=type(backend).__name__):
                with self.assertRaisesRegex(ValueError, "selection_identity_mismatch"):
                    slots.select_api_key(backend, session_id=session, store=store,
                                         secrets=manager, required_binding=binding)
        self.assertEqual((store.reads, store.writes, manager.calls), (0, [], []))

    def test_strict_apply_uses_normal_selected_slot_api(self):
        slots, store, manager, required = self.selection()
        kwargs = slots.apply_api_key_to_kwargs("grok_cli", {}, session_id="unit-allocation", store=store,
                                               secrets=manager, required_binding=required)
        self.assertEqual((kwargs["api_key"], kwargs["_api_key_slot"], kwargs["_api_key_fingerprint"]),
                         (KEY, "unit-slot", FINGERPRINT))
        self.assertEqual((store.writes, manager.calls), ([], ["api_key/grok_cli/unit-slot"]))

    def test_strict_apply_rejects_inline_key_and_metadata_bypasses(self):
        slots, store, manager, required = self.selection()
        for field in ("api_key", "_api_key_slot", "_api_key_fingerprint"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "inline_override"):
                    slots.apply_api_key_to_kwargs("grok_cli", {field: "unit-bypass"}, session_id="unit-allocation",
                        store=store, secrets=manager, required_binding=required)
        self.assertEqual((store.reads, store.writes, manager.calls), (0, [], []))

    def test_legacy_missing_or_disabled_binding_still_ranks_and_rebinds(self):
        for mode in ("missing", "disabled"):
            with self.subTest(mode=mode):
                slots, store, manager, _required = self.selection()
                if mode == "missing":
                    store.bound.clear()
                else:
                    store.rows[0]["enabled"] = False
                selected = slots.select_api_key("grok_cli", session_id="unit-allocation", store=store, secrets=manager)
                self.assertEqual(selected["slot_id"], "other-slot")
                self.assertEqual(store.writes, [("unit-allocation", "grok_cli", "other-slot")])

    def test_legacy_explicit_key_bypass_remains_unchanged(self):
        slots, store, manager, _required = self.selection()
        kwargs = {"api_key": "legacy-explicit-unit-only"}
        self.assertIs(slots.apply_api_key_to_kwargs("grok_cli", kwargs, store=store, secrets=manager), kwargs)
        self.assertEqual((store.reads, store.writes, manager.calls), (0, [], []))

    def test_public_strict_selection_error_is_not_swallowed_or_retried(self):
        from ipfs_accelerate_py import llm_router as router, llm_allocation
        from ipfs_accelerate_py.llm_allocation import api_key_slots as slots
        request = self.request(); store = AllocationMetadataFixture()
        failure = ValueError("unit strict selection refusal")
        with patch.object(llm_allocation, "get_allocation_store", return_value=store), \
             patch.object(slots, "apply_api_key_to_kwargs", side_effect=failure) as selected, \
             patch.object(router, "get_llm_provider", side_effect=AssertionError("provider work forbidden")) as provider, \
             patch.object(router._ALLOCATION_CONTEXT, "session_id", "unit-allocation", create=True), \
             patch.object(router._ALLOCATION_CONTEXT, "path", "cli", create=True), \
             patch.object(router._API_KEY_CONTEXT, "slot", "stale-unit-slot", create=True), \
             patch.object(router._API_KEY_CONTEXT, "fingerprint", "stale-unit-fingerprint", create=True):
            with self.assertRaises(ValueError) as observed:
                router.generate_text("private selection refusal fixture", deps=object(), **self.public_options(request))
            self.assertIs(observed.exception, failure)
            selected.assert_called_once()
            self.assertEqual(selected.call_args.kwargs["required_binding"].session_id, request.allocation_session_id)
            provider.assert_not_called()
            self.assertEqual((router._API_KEY_CONTEXT.slot, router._API_KEY_CONTEXT.fingerprint), ("", ""))

    def test_capture_context_drift_cannot_persist_a_response(self):
        from ipfs_accelerate_py import llm_router as router
        request = self.request(); request._session, request._phase = SESSION, "response_accepted"
        with patch.object(router, "_allocation_context", return_value=("different-allocation", "cli")):
            with self.assertRaisesRegex(helper.NamedSessionCaptureRefusal, "capture_unconfirmed"):
                router._capture_cli_session("grok_cli", {"grok_named_session_request": request})
        self.assertEqual(request.completed_session_id, SESSION)

    def test_unconfirmed_capture_preserves_accepted_response_and_blocks_retry(self):
        from ipfs_accelerate_py import llm_router as router, llm_allocation
        for mode in ("write_exception", "noop", "wrong_uuid", "readback_exception"):
            with self.subTest(mode=mode):
                request = self.request(); request._session, request._phase = SESSION, "response_accepted"
                persisted = []
                def write(**row):
                    if mode == "write_exception":
                        raise RuntimeError("unit capture write unavailable")
                    if mode != "noop":
                        persisted.append(dict(allocation_session_id=row["session_id"], provider=row["provider"],
                            native_session_id=("different-native-id" if mode == "wrong_uuid" else row["native_session_id"]),
                            workspace=row["workspace"]))
                def read(*_args):
                    if mode == "readback_exception":
                        raise RuntimeError("unit capture readback unavailable")
                    return dict(persisted[-1]) if persisted else {}
                store = SimpleNamespace(upsert_cli_session=write, get_cli_session=read)
                with patch.object(router, "_allocation_context", return_value=("unit-allocation", "cli")), \
                     patch.object(llm_allocation, "get_allocation_store", return_value=store):
                    with self.assertRaisesRegex(helper.NamedSessionCaptureRefusal, "capture_unconfirmed") as observed:
                        router._capture_cli_session("grok_cli", {"grok_named_session_request": request})
                self.assertTrue(observed.exception.side_effects_started)
                self.assertFalse(observed.exception.retryable)
                self.assertEqual((request._phase, request.completed_session_id), ("response_accepted", SESSION))


class AutomaticNamedSessionOutputBudgetRefusalTests(unittest.TestCase):
    """Append to the existing named-session test module; source proposal only.

    The inherited fixture methods create synthetic source-bound owner inputs.
    No lower TOML scalar is presented as qualified native effective policy.
    """
    setUp = AutomaticNamedSessionSourceTests.setUp
    request = AutomaticNamedSessionSourceTests.request
    provider_call = AutomaticNamedSessionSourceTests.provider_call
    prepared_home_fixture = AutomaticNamedSessionSourceTests.prepared_home_fixture

    def refused_call(self, options, expected):
        from ipfs_accelerate_py import llm_router as router
        request = self.request()
        original = {path: (path.read_bytes(), path.stat().st_ino,
                           path.stat().st_mode, path.stat().st_mtime_ns)
                    for path in (self.configuration, self.requirements, self.binary)}
        environment = dict(os.environ)
        key_context = tuple(getattr(router._API_KEY_CONTEXT, field, None)
                            for field in ("slot", "fingerprint"))
        with contextlib.chdir(request.workspace), \
             patch.object(router, "_grok_cli_command", return_value=str(self.binary)), \
             patch.object(router, "_cli_available", return_value=True), \
             patch.object(helper, "validate_preparation_option", side_effect=AssertionError("owner precheck forbidden")) as precheck, \
             patch.object(helper, "NamedSessionLifetime", side_effect=AssertionError("lifetime forbidden")) as lifetime, \
             patch.object(helper, "prepare_native_home", side_effect=AssertionError("preparation forbidden")) as prepare, \
             patch.object(helper, "_bootstrap", side_effect=AssertionError("bootstrap forbidden")) as bootstrap, \
             patch.object(router.subprocess, "run", side_effect=AssertionError("headless forbidden")) as run, \
             patch.object(router.subprocess, "Popen", side_effect=AssertionError("process forbidden")) as popen:
            provider = router._get_grok_cli_provider()
            with self.assertRaisesRegex(ValueError, expected):
                provider.generate("unit only", model_name="grok-4.7", temperature=0,
                    grok_named_session_request=request, grok_cli_cmd=[str(self.binary)], **options)
        for forbidden in (precheck, lifetime, prepare, bootstrap, run, popen):
            forbidden.assert_not_called()
        self.assertEqual(request._phase, "prepared")
        self.assertEqual(original, {path: (path.read_bytes(), path.stat().st_ino,
                           path.stat().st_mode, path.stat().st_mtime_ns)
                    for path in original})
        self.assertEqual(environment, dict(os.environ))
        self.assertEqual(key_context, tuple(getattr(router._API_KEY_CONTEXT, field, None)
                                            for field in ("slot", "fingerprint")))
        self.assertEqual(list(request.native_home.iterdir()), [])
        self.assertEqual(list(request.preparation.home.iterdir()), [])
        self.assertEqual(list(request.audit_directory.iterdir()), [])

    def test_each_budget_alias_and_equal_aliases_refuse_before_native_work(self):
        aliases = ("max_tokens", "max_new_tokens", "max_completion_tokens")
        for options in (*({name: 4096} for name in aliases), dict.fromkeys(aliases, 4096)):
            with self.subTest(options=options):
                self.refused_call(options, "qualified pinned owner cap")

    def test_even_pinned_lower_or_equal_caps_need_effective_native_qualification(self):
        # A scalar is not a qualified effective-cap receipt for initialization.
        marker = '[model."grok-4.7"]\ntemperature = 0.0\n'
        for cap in (None, 2048, 4096, 8192):
            config = CONFIG if cap is None else CONFIG.replace(
                marker, marker + "max_completion_tokens = " + str(cap) + "\n")
            self.assertEqual(config.count("max_completion_tokens"), int(cap is not None))
            self.configuration.write_text(config)
            with self.subTest(pinned_selected_cap=cap):
                self.refused_call({"max_tokens": 4096}, "qualified pinned owner cap")

    def test_invalid_or_conflicting_budget_aliases_keep_shared_validation(self):
        for value in (True, None, 0, -1, 4096.0, "4096"):
            with self.subTest(value=value):
                self.refused_call({"max_tokens": value}, "positive TOML integer")
        self.refused_call({"max_tokens": 4096, "max_completion_tokens": 2048}, "aliases conflict")

    def test_named_route_without_caller_budget_keeps_existing_owner_handoff(self):
        output, request, observed = self.provider_call()
        self.assertEqual(output, "unit planning response")
        self.assertEqual(self.last_preparation_count, 1)
        self.assertEqual(self.last_headless_count, 1)
        self.assertEqual(observed[0]["environment"]["GROK_HOME"], str(request.native_home))
        self.assertEqual(observed[0]["environment"]["XAI_API_KEY"], KEY)
        self.assertTrue(request.receipt()["response_accepted"])

    def refused_public_call(self, options, expected):
        from ipfs_accelerate_py import llm_router as router
        from ipfs_accelerate_py.llm_allocation import api_key_slots
        request = self.request()
        before_environment = dict(os.environ)
        before_allocation = router._allocation_context()
        before_keys = tuple(getattr(router._API_KEY_CONTEXT, field, None)
                            for field in ("slot", "fingerprint"))
        before_inputs = {path: (path.read_bytes(), path.stat().st_ino,
                               path.stat().st_mode, path.stat().st_mtime_ns)
                         for path in (self.configuration, self.requirements, self.binary)}
        with contextlib.ExitStack() as stack:
            forbidden = []
            for module, name in (
                (helper, "validate_public_dispatch"),
                (router, "get_default_router_deps"),
                (router, "_resolve_allocation_session_id"),
                (router, "get_allocation_session"),
                (router, "_set_allocation_context"),
                (api_key_slots, "apply_api_key_to_kwargs"),
                (router, "_clear_last_generation_trace"),
                (router, "_set_last_usage_admission"),
                (router, "get_llm_provider"),
                (helper, "NamedSessionLifetime"),
                (helper, "prepare_native_home"),
                (helper, "_bootstrap"),
                (router.subprocess, "Popen"),
                (router.subprocess, "run"),
            ):
                forbidden.append(stack.enter_context(patch.object(module, name,
                    side_effect=AssertionError("public refusal reached " + name))))
            with self.assertRaisesRegex(ValueError, expected):
                router.generate_text("unit only", provider="grok_cli", model_name="grok-4.7",
                    temperature=0, side_effecting=True, allow_local_fallback=False,
                    allow_cross_provider_fallback=False, allocation_session_id="unit-allocation",
                    grok_named_session_request=request, **options)
            for check in forbidden:
                check.assert_not_called()
        self.assertEqual(before_environment, dict(os.environ))
        self.assertEqual(before_allocation, router._allocation_context())
        self.assertEqual(before_keys, tuple(getattr(router._API_KEY_CONTEXT, field, None)
                                            for field in ("slot", "fingerprint")))
        self.assertEqual(before_inputs, {path: (path.read_bytes(), path.stat().st_ino,
                               path.stat().st_mode, path.stat().st_mtime_ns)
                         for path in before_inputs})
        self.assertEqual(request._phase, "prepared")
        for directory in (request.native_home, request.preparation.home, request.audit_directory):
            self.assertEqual(list(directory.iterdir()), [])

    def test_public_budget_aliases_refuse_before_owner_credentials_or_allocation(self):
        aliases = ("max_tokens", "max_new_tokens", "max_completion_tokens")
        for options in (*({name: 4096} for name in aliases), dict.fromkeys(aliases, 4096)):
            with self.subTest(options=options):
                self.refused_public_call(options, "qualified pinned owner cap")

    def test_public_invalid_or_conflicting_aliases_fail_without_dispatch_side_effects(self):
        for value in (True, None, 0, -1, 4096.0, "4096"):
            with self.subTest(value=value):
                self.refused_public_call({"max_tokens": value}, "positive TOML integer")
        self.refused_public_call({"max_tokens": 4096, "max_new_tokens": 2048}, "aliases conflict")
