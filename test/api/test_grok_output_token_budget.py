"""Public Grok provider budget policy with synthetic files and mocked transport.

These tests inspect the private configuration supplied to the transport. They
do not qualify native configuration merging, real generation or billing caps.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import toml


class GrokOutputTokenBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ipfs_accelerate_py import llm_router
        cls.router = llm_router

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.home = self.root / "operator"
        self.home.mkdir(mode=0o700)
        self.grok_home = self.home / ".grok"
        self.grok_home.mkdir(mode=0o700)
        self.temporary = self.root / "temporary"
        self.temporary.mkdir(mode=0o700)
        self.stack.enter_context(patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
            "HOME": str(self.home), "GROK_HOME": str(self.grok_home),
            "XAI_API_KEY": "synthetic-budget-unit-key",
            "TMPDIR": str(self.temporary), "TMP": str(self.temporary),
            "TEMP": str(self.temporary),
            "IPFS_ACCELERATE_LLM_ALLOCATION_DB": str(self.root / "allocation.duckdb"),
            "IPFS_ACCELERATE_META_INDEX_DUCKDB": str(self.root / "meta-index.duckdb"),
        }, clear=True))
        self.stack.enter_context(patch.object(subprocess, "Popen", side_effect=AssertionError("process forbidden")))
        self.run_mock = self.stack.enter_context(patch.object(subprocess, "run", side_effect=AssertionError("process forbidden")))
        self.stack.enter_context(patch.object(self.router, "_grok_cli_command", return_value="/fixture/grok"))
        self.stack.enter_context(patch.object(self.router, "_cli_available", return_value=True))
        self.system_policies = {}
        original_read = self.router._grok_sampling_config_text

        def synthetic_policy(path):
            path = Path(path)
            if path.parent == Path("/etc/grok"):
                return self.system_policies.get(path.name, "")
            return original_read(path)

        self.stack.enter_context(patch.object(self.router, "_grok_sampling_config_text", side_effect=synthetic_policy))

    def write_file(self, name, content):
        path = self.grok_home / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
        return path

    def config(self, value):
        return self.write_file("config.toml", toml.dumps(value))

    def clear_policies(self):
        for name in ("config.toml", "managed_config.toml", "requirements.toml"):
            (self.grok_home / name).unlink(missing_ok=True)
        self.system_policies.clear()

    def generate(self, **kwargs):
        return self.router._get_grok_cli_provider().generate(
            "synthetic budget prompt", model_name="grok-4.7", **kwargs)

    def reply(self, _command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "text": "unit budget response", "stopReason": "end_turn"}), stderr="")

    def capture(self, inspect):
        roots = []

        def execute(command, **kwargs):
            root = Path(kwargs["env"]["GROK_HOME"])
            roots.append(root)
            self.assertNotEqual(root, self.grok_home)
            config = toml.loads((root / "config.toml").read_text())
            inspect(command, kwargs, root, config)
            return self.reply(command, **kwargs)

        self.run_mock.side_effect = execute
        return roots

    def test_positive_aliases_and_equal_aliases_bind_one_budget(self):
        wanted = 4096
        roots = self.capture(lambda _cmd, _kw, _root, cfg:
            self.assertEqual(cfg["model"]["grok-4.7"]["max_completion_tokens"], wanted))
        aliases = ("max_tokens", "max_new_tokens", "max_completion_tokens")
        for name in aliases:
            with self.subTest(alias=name):
                self.assertEqual(self.generate(**{name: wanted}), "unit budget response")
        self.assertEqual(self.generate(**dict.fromkeys(aliases, wanted)), "unit budget response")
        wanted = 1
        self.assertEqual(self.generate(max_tokens=wanted), "unit budget response")
        self.assertEqual(self.run_mock.call_count, 5)
        self.assertTrue(all(not root.exists() for root in roots))

    def test_invalid_alias_values_refuse_before_transport(self):
        for name in ("max_tokens", "max_new_tokens", "max_completion_tokens"):
            for value in (True, False, None, 0, -1, 4096.0, "4096", float("nan"), float("inf")):
                with self.subTest(alias=name, value=repr(value)):
                    with self.assertRaises(ValueError):
                        self.generate(**{name: value})
        self.run_mock.assert_not_called()

    def test_conflicting_aliases_refuse_before_transport(self):
        for first, second in (("max_tokens", "max_new_tokens"),
                              ("max_tokens", "max_completion_tokens"),
                              ("max_new_tokens", "max_completion_tokens")):
            with self.subTest(first=first, second=second):
                with self.assertRaises(ValueError):
                    self.generate(**{first: 4096, second: 2048})
        self.run_mock.assert_not_called()

    def test_custom_wrappers_config_options_and_terminators_refuse(self):
        commands = (
            "/fixture/custom-wrapper --custom",
            ["/fixture/grok", "--config", "/fixture/unread-config.toml"],
            ["/fixture/grok", "--config=/fixture/unread-config.toml"],
            ["/fixture/grok", "--"],
        )
        with patch.object(self.router, "_run_cli_command", side_effect=AssertionError("wrapper forbidden")) as fallback:
            for command in commands:
                with self.subTest(command=command), self.assertRaises(ValueError):
                    self.generate(max_tokens=4096, grok_cli_cmd=command)
        fallback.assert_not_called()
        self.run_mock.assert_not_called()

    def test_budget_only_preserves_temperature_auth_context_and_environment(self):
        original = {
            "models": {"temperature": 0.8, "max_completion_tokens": 7000},
            "model": {"grok-4.7": {"temperature": 1.25, "context_window": 8192,
                "max_completion_tokens": 6000, "env_key": "XAI_API_KEY"},
                "other": {"temperature": 0.6, "max_completion_tokens": 9000}},
            "cli": {"auto_update": False, "use_leader": True},
        }
        source = self.config(original)
        before = source.read_bytes()
        auth = self.write_file("auth.json", "synthetic budget auth backing")
        context = self.write_file("AGENTS.md", "synthetic context marker\n")
        cache = self.grok_home / "cache"
        cache.mkdir()
        identity = lambda path: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
        originals = {path: identity(path) for path in (auth, context)}
        expected = deepcopy(original)
        expected["model"]["grok-4.7"]["max_completion_tokens"] = 4096
        expected["cli"]["use_leader"] = False

        def inspect(_cmd, kwargs, root, config):
            self.assertEqual(config, expected)
            self.assertEqual(kwargs["env"]["HOME"], str(self.home))
            self.assertEqual(kwargs["env"]["XAI_API_KEY"], "synthetic-budget-unit-key")
            self.assertEqual(kwargs["env"]["BUDGET_UNIT_MARKER"], "preserved")
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((root / "config.toml").stat().st_mode), 0o600)
            for path in (auth, context, cache):
                self.assertEqual((root / path.name).readlink(), path)

        roots = self.capture(inspect)
        real_open = os.open

        def no_backing_payload(path, *args, **kwargs):
            if Path(path) in (auth, context):
                raise AssertionError("auth/context payload must not be read")
            return real_open(path, *args, **kwargs)

        with patch.dict(os.environ, {"BUDGET_UNIT_MARKER": "preserved"}), patch.object(os, "open", side_effect=no_backing_payload):
            self.assertEqual(self.generate(max_tokens=4096), "unit budget response")
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual({path: identity(path) for path in originals}, originals)
        self.assertTrue(all(not root.exists() for root in roots))

    def test_tighter_selected_and_global_config_limits_are_retained(self):
        for selected, global_limit, wanted in ((1024, None, 1024), (None, 512, 512),
                                                (1024, 512, 512), (8192, 9000, 4096)):
            with self.subTest(selected=selected, global_limit=global_limit):
                value = {"model": {"grok-4.7": {"temperature": 1.1}}}
                if selected is not None:
                    value["model"]["grok-4.7"]["max_completion_tokens"] = selected
                if global_limit is not None:
                    value["models"] = {"max_completion_tokens": global_limit}
                source = self.config(value)
                before = source.read_bytes()
                roots = self.capture(lambda _cmd, _kw, _root, cfg:
                    self.assertEqual(cfg["model"]["grok-4.7"]["max_completion_tokens"], wanted))
                self.generate(max_tokens=4096)
                self.assertEqual(source.read_bytes(), before)
                self.assertTrue(all(not root.exists() for root in roots))

    def test_user_and_system_managed_caps_are_minimum_and_backing_is_preserved(self):
        source = self.config({"model": {"grok-4.7": {"temperature": 1.4, "max_completion_tokens": 8192}}})
        managed = self.write_file("managed_config.toml", '[models]\nmax_completion_tokens=2048\n')
        self.system_policies["managed_config.toml"] = '[model."grok-4.7"]\nmax_completion_tokens=1024\n'
        originals = {path: path.read_bytes() for path in (source, managed)}

        def inspect(_cmd, _kw, root, cfg):
            self.assertEqual(cfg["model"]["grok-4.7"]["max_completion_tokens"], 1024)
            self.assertEqual(cfg["model"]["grok-4.7"]["temperature"], 1.4)
            self.assertEqual((root / "managed_config.toml").readlink(), managed)

        roots = self.capture(inspect)
        self.generate(max_new_tokens=4096)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        self.assertTrue(all(not root.exists() for root in roots))

    def test_zero_temperature_and_budget_share_one_original_backed_private_home(self):
        self.config({"model": {"grok-4.7": {"temperature": 1.5}}, "cli": {"use_leader": True}})
        context = self.write_file("AGENTS.md", "synthetic combined policy context\n")
        auth = self.write_file("auth.json", "synthetic combined policy auth")

        def inspect(_cmd, _kw, root, cfg):
            self.assertEqual(cfg["model"]["grok-4.7"]["temperature"], 0.0)
            self.assertEqual(cfg["model"]["grok-4.7"]["max_completion_tokens"], 2048)
            self.assertIs(cfg["cli"]["use_leader"], False)
            self.assertEqual((root / "AGENTS.md").readlink(), context)
            self.assertEqual((root / "auth.json").readlink(), auth)

        roots = self.capture(inspect)
        self.generate(temperature=0, max_tokens=2048)
        self.assertEqual(len(roots), 1)
        self.assertTrue(all(not root.exists() for root in roots))

    def test_invalid_config_managed_and_requirement_limits_refuse(self):
        for origin in ("config", "user-managed", "system-managed", "user-requirements", "system-requirements"):
            for value in (True, 0, -1, 1.5, "4096"):
                with self.subTest(origin=origin, value=value):
                    self.clear_policies()
                    policy = toml.dumps({"model": {"grok-4.7": {"max_completion_tokens": value}}})
                    if origin == "config":
                        self.write_file("config.toml", policy)
                    elif origin.startswith("user-"):
                        self.write_file("managed_config.toml" if origin.endswith("managed") else "requirements.toml", policy)
                    else:
                        self.system_policies["managed_config.toml" if origin.endswith("managed") else "requirements.toml"] = policy
                    with self.assertRaises(ValueError):
                        self.generate(max_completion_tokens=4096)
        self.run_mock.assert_not_called()

    def test_selected_requirements_cannot_override_a_stricter_chosen_limit(self):
        for origin in ("user", "system"):
            for configured, required in ((None, 8192), (512, 1024)):
                with self.subTest(origin=origin, configured=configured, required=required):
                    self.clear_policies()
                    if configured is not None:
                        self.config({"model": {"grok-4.7": {"max_completion_tokens": configured}}})
                    policy = '[model."grok-4.7"]\nmax_completion_tokens=' + str(required) + '\n'
                    if origin == "user":
                        path = self.write_file("requirements.toml", policy)
                        before = path.read_bytes()
                    else:
                        self.system_policies["requirements.toml"] = policy
                    with self.assertRaises(ValueError):
                        self.generate(max_tokens=4096)
                    if origin == "user":
                        self.assertEqual(path.read_bytes(), before)
        self.run_mock.assert_not_called()

    def test_equal_or_lower_user_and_system_requirements_are_preserved(self):
        for origin in ("user", "system"):
            for selected, cap in ((True, 4096), (True, 512), (False, 256)):
                with self.subTest(origin=origin, selected=selected, cap=cap):
                    self.clear_policies()
                    self.config({"model": {"grok-4.7": {"temperature": 1.3}}})
                    policy = toml.dumps({"model": {"grok-4.7": {"max_completion_tokens": cap}}}
                                        if selected else {"models": {"max_completion_tokens": cap}})
                    requirements = self.write_file("requirements.toml", policy) if origin == "user" else None
                    if origin == "system":
                        self.system_policies["requirements.toml"] = policy

                    def inspect(_cmd, _kw, root, cfg):
                        self.assertEqual(cfg["model"]["grok-4.7"]["max_completion_tokens"], cap)
                        self.assertEqual(cfg["model"]["grok-4.7"]["temperature"], 1.3)
                        if requirements is not None:
                            self.assertEqual((root / "requirements.toml").readlink(), requirements)

                    roots = self.capture(inspect)
                    self.generate(max_tokens=4096)
                    if requirements is not None:
                        self.assertEqual(requirements.read_text(), policy)
                    self.assertTrue(all(not root.exists() for root in roots))

    def test_unqualified_user_system_and_managed_patches_refuse(self):
        for origin in ("config", "user-managed", "system-managed", "user-requirements", "system-requirements"):
            for kind in ("experiments", "version_overrides", "campaigns"):
                with self.subTest(origin=origin, kind=kind):
                    self.clear_policies()
                    policy = ('[[' + kind + ']]\nid="synthetic-budget-patch"\n['
                              + kind + '.model."grok-4.7"]\nmax_completion_tokens=8192\n')
                    filename = "config.toml" if origin == "config" else "managed_config.toml" if origin.endswith("managed") else "requirements.toml"
                    if origin.startswith("system-"):
                        self.system_policies[filename] = policy
                    else:
                        self.write_file(filename, policy)
                    with self.assertRaises(ValueError):
                        self.generate(max_tokens=4096)
        self.run_mock.assert_not_called()

    def test_requirements_default_model_must_match_the_selected_model(self):
        for origin in ("user", "system"):
            for selected in ("other-model", "grok-4.7"):
                with self.subTest(origin=origin, default_model=selected):
                    self.clear_policies()
                    policy = toml.dumps({"models": {"default": selected, "max_completion_tokens": 1024}})
                    if origin == "user":
                        self.write_file("requirements.toml", policy)
                    else:
                        self.system_policies["requirements.toml"] = policy
                    calls_before = self.run_mock.call_count
                    if selected != "grok-4.7":
                        with self.assertRaises(ValueError):
                            self.generate(max_tokens=4096)
                        self.assertEqual(self.run_mock.call_count, calls_before)
                    else:
                        roots = self.capture(lambda _cmd, _kw, _root, cfg:
                            self.assertEqual(cfg["model"]["grok-4.7"]["max_completion_tokens"], 1024))
                        self.assertEqual(self.generate(max_tokens=4096), "unit budget response")
                        self.assertTrue(all(not root.exists() for root in roots))

    def test_active_environment_config_overlays_refuse_budget_dispatch(self):
        for name, value in (("GROK_CONFIG", '{"model":{"grok-4.7":{"max_completion_tokens":8192}}}'),
                            ("GROK_CONFIG_PATH", "/fixture/never-open-overlay.toml")):
            for zero in (False, True):
                with self.subTest(overlay=name, zero_temperature=zero):
                    with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                        self.generate(max_tokens=4096, **({"temperature": 0} if zero else {}))
        self.run_mock.assert_not_called()

    def test_transport_exception_cleans_budget_home_and_prompt(self):
        roots, prompts = [], []

        def execute(command, **kwargs):
            root = Path(kwargs["env"]["GROK_HOME"])
            roots.append(root)
            prompts.append(Path(command[command.index("--prompt-file") + 1]))
            self.assertEqual(toml.loads((root / "config.toml").read_text())["model"]["grok-4.7"]["max_completion_tokens"], 4096)
            raise subprocess.TimeoutExpired(command, 1, output="", stderr="")

        self.run_mock.side_effect = execute
        with self.assertRaises(subprocess.TimeoutExpired):
            self.generate(max_tokens=4096, timeout=1)
        self.assertEqual(self.run_mock.call_count, 1)
        self.assertTrue(all(not path.exists() for path in [*roots, *prompts]))

    def test_streaming_uses_the_same_budget_context_and_cleans_it(self):
        roots, prompts = [], []

        def capture(command, **kwargs):
            root = Path(kwargs["env"]["GROK_HOME"])
            roots.append(root)
            prompts.append(Path(command[command.index("--prompt-file") + 1]))
            cfg = toml.loads((root / "config.toml").read_text())
            self.assertEqual(cfg["model"]["grok-4.7"]["max_completion_tokens"], 1536)
            self.assertIs(cfg["cli"]["use_leader"], False)
            payload = {"text": "unit streamed budget response", "stopReason": "end_turn"}
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="",
                timed_out=False, final_payload=payload, response_text=payload["text"],
                progress={"response_usable": True})

        with patch("ipfs_accelerate_py.cli_runtime.grok_stream_capture.capture_grok_stream", side_effect=capture) as stream:
            result = self.generate(max_new_tokens=1536, grok_stream_progress=True)
        self.assertEqual(result, "unit streamed budget response")
        self.assertEqual(stream.call_count, 1)
        self.run_mock.assert_not_called()
        self.assertTrue(all(not path.exists() for path in [*roots, *prompts]))


if __name__ == "__main__":
    unittest.main()
