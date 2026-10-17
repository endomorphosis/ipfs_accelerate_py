"""Actual Grok provider sampling and planning dispatch, with mocked transport.

No provider, credential file, Docker engine, database or native claim is used.
Private files are bounded synthetic fixtures; effective CLI/server sampling
still requires separate native qualification.
"""
from __future__ import annotations

import contextlib
import ctypes.util
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sysconfig
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import toml


def _fixture_library_paths():
    """Resolve just libc/pthread from root-owned local ELF files, no commands."""
    multiarch = sysconfig.get_config_var("MULTIARCH")
    directories = []
    if isinstance(multiarch, str) and multiarch and "/" not in multiarch:
        directories.extend((Path("/usr/lib") / multiarch, Path("/lib") / multiarch))
    directories.extend((Path("/usr/lib"), Path("/lib")))
    libraries = {}
    for name, filename in (("c", "libc.so.6"), ("pthread", "libpthread.so.0")):
        for directory in directories:
            try:
                path = (directory / filename).resolve(strict=True)
                facts = path.stat(follow_symlinks=False)
                if (not stat.S_ISREG(facts.st_mode) or facts.st_uid != 0
                        or facts.st_mode & 0o022):
                    continue
                if any(parent.stat().st_uid != 0 or parent.stat().st_mode & 0o022
                       for parent in path.parents):
                    continue
                with path.open("rb") as stream:
                    if stream.read(4) != b"\x7fELF":
                        continue
                libraries[name] = path
                break
            except OSError:
                continue
        if name not in libraries:
            raise AssertionError("root-owned local fixture library is unavailable: " + name)
    return libraries


def _fixture_library_resolver():
    paths = _fixture_library_paths()
    identities = {name: (path.stat().st_dev, path.stat().st_ino,
        path.stat().st_size, path.stat().st_mtime_ns) for name, path in paths.items()}
    def resolve(name):
        if name not in paths:
            raise AssertionError("unqualified library lookup in source fixture")
        path = paths[name]
        facts = path.stat(follow_symlinks=False)
        if ((facts.st_dev, facts.st_ino, facts.st_size, facts.st_mtime_ns) != identities[name]
                or facts.st_uid != 0 or facts.st_mode & 0o022
                or not stat.S_ISREG(facts.st_mode)):
            raise AssertionError("fixture system library identity changed")
        return str(path)
    return resolve


def _load_native_modules():
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner
    return llm_router, router_implementation_runner, llm_router


class GrokPlanningTemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router, cls.runner, cls.canonical_router = _load_native_modules()

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        root = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(root).resolve()
        self.home = self.root / "operator"
        self.home.mkdir(mode=0o700)
        self.grok_home = self.home / ".grok"
        self.grok_home.mkdir(mode=0o700)
        self.stack.enter_context(patch.dict(os.environ, {
            "HOME": str(self.home), "GROK_HOME": str(self.grok_home),
            "XAI_API_KEY": "synthetic-unit-key",
        }))
        self.stack.enter_context(patch.object(subprocess, "Popen", side_effect=AssertionError("process forbidden")))
        self.run_mock = self.stack.enter_context(patch.object(subprocess, "run", side_effect=AssertionError("process forbidden")))
        self.stack.enter_context(patch.object(self.router, "_grok_cli_command", return_value="/fixture/grok"))
        self.stack.enter_context(patch.object(self.router, "_cli_available", return_value=True))
        self.read_config = self.router._grok_sampling_config_text
        self.stack.enter_context(patch.object(self.router, "_grok_sampling_config_text",
            side_effect=lambda path: "" if path == Path("/etc/grok/requirements.toml") else self.read_config(path)))

    def write_config(self, content):
        path = self.grok_home / "config.toml"
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
        return path

    def provider(self):
        return self.router._get_grok_cli_provider()

    def reply(self, command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "text": "unit response", "stopReason": "end_turn"}), stderr="")

    def test_private_config_changes_sampling_and_disables_leader_only(self):
        original = {
            "models": {"default": "grok-4.7", "temperature": 0.8},
            "model": {"grok-4.7": {"model": "grok-4.7", "context_window": 8192,
                "max_completion_tokens": 4096, "temperature": 1.7,
                "env_key": "XAI_API_KEY"}, "other": {"temperature": 0.6}},
            "cli": {"auto_update": False},
        }
        expected = deepcopy(original)
        expected["model"]["grok-4.7"]["temperature"] = 0.0
        expected["cli"]["use_leader"] = False
        observed = toml.loads(self.router._grok_zero_temperature_config(toml.dumps(original), "grok-4.7"))
        self.assertEqual(observed, expected)

    def test_actual_provider_uses_private_zero_config_and_preserves_auth_and_context(self):
        config = self.write_config('[model."grok-4.7"]\nmodel="grok-4.7"\ncontext_window=8192\ntemperature=1.2\n')
        original = config.read_bytes()
        auth = self.grok_home / "auth.json"
        auth.write_bytes(b"synthetic credential fixture")
        auth.chmod(0o600)
        auth_before = auth.stat()
        private_roots = []

        def execute(command, **kwargs):
            env = kwargs["env"]
            root = Path(env["GROK_HOME"])
            private_roots.append(root)
            self.assertNotEqual(root, self.grok_home)
            self.assertEqual(env["HOME"], str(self.home))
            self.assertEqual(env["XAI_API_KEY"], "synthetic-unit-key")
            self.assertEqual(env["GROK_CONFIG"], '{"models":{"temperature":1.4}}')
            self.assertEqual((root / "auth.json").readlink(), auth)
            self.assertEqual(toml.loads((root / "config.toml").read_text())["model"]["grok-4.7"],
                {"model": "grok-4.7", "context_window": 8192, "temperature": 0.0})
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((root / "config.toml").stat().st_mode), 0o600)
            self.assertEqual(command[command.index("--model") + 1], "grok-4.7")
            self.assertEqual(kwargs["timeout"], 90.0)
            return self.reply(command, **kwargs)

        self.run_mock.side_effect = execute
        # The native CLI documents this as a global default, subordinate to
        # the explicit per-model value. Unit evidence only: do not claim the
        # mocked process has verified the native CLI's merged sampling view.
        with patch.dict(os.environ, {"GROK_CONFIG": '{"models":{"temperature":1.4}}'}):
            self.assertEqual(self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0, timeout=90), "unit response")
        self.assertEqual(self.run_mock.call_count, 1)
        self.assertEqual(config.read_bytes(), original)
        self.assertEqual(auth.stat().st_ino, auth_before.st_ino)
        self.assertEqual(auth.stat().st_mtime_ns, auth_before.st_mtime_ns)
        self.assertTrue(all(not root.exists() for root in private_roots))

    def test_omitted_temperature_retains_existing_provider_environment(self):
        def execute(command, **kwargs):
            self.assertEqual(kwargs["env"]["GROK_HOME"], str(self.grok_home))
            return self.reply(command, **kwargs)
        self.run_mock.side_effect = execute
        self.assertEqual(self.provider().generate("unit prompt", model_name="grok-4.7"), "unit response")
        self.assertEqual(self.run_mock.call_count, 1)

    def test_nonzero_or_invalid_temperature_refuses_before_transport(self):
        for value in (0.2, -1, float("nan"), float("inf"), True, None, "0"):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "temperature must be zero"):
                    self.provider().generate("unit prompt", temperature=value)
        self.run_mock.assert_not_called()

    def test_custom_non_native_wrapper_cannot_claim_zero_temperature(self):
        with self.assertRaisesRegex(ValueError, "structured native CLI"):
            self.provider().generate("unit prompt", temperature=0, grok_cli_cmd="/fixture/wrapper --custom")
        self.run_mock.assert_not_called()

    def test_conflicting_model_command_refuses_before_transport(self):
        with self.assertRaisesRegex(ValueError, "sampling model differs"):
            self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0,
                grok_cli_cmd=["/fixture/grok", "--model", "other-model"])
        self.run_mock.assert_not_called()

    def test_provider_exception_cleans_private_sampling_and_prompt_files(self):
        roots = []
        prompts = []
        def execute(command, **kwargs):
            roots.append(Path(kwargs["env"]["GROK_HOME"]))
            prompts.append(Path(command[command.index("--prompt-file") + 1]))
            raise subprocess.TimeoutExpired(command, 1, output="", stderr="")
        self.run_mock.side_effect = execute
        with self.assertRaises(subprocess.TimeoutExpired):
            self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0, timeout=1)
        self.assertEqual(self.run_mock.call_count, 1)
        self.assertTrue(all(not path.exists() for path in [*roots, *prompts]))

    def test_conflicting_requirements_are_retained_and_refused(self):
        requirements = self.grok_home / "requirements.toml"
        requirements.write_text('[model."grok-4.7"]\ntemperature=0.6\n')
        requirements.chmod(0o600)
        original = requirements.read_bytes()
        with self.assertRaisesRegex(ValueError, "temperature must be zero"):
            self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0)
        self.run_mock.assert_not_called()
        self.assertEqual(requirements.read_bytes(), original)

    def test_malformed_original_config_is_not_silently_discarded(self):
        config = self.write_config('[model."grok-4.7"\ncontext_window=8192\n')
        original = config.read_bytes()
        with self.assertRaises(ValueError):
            self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0)
        self.run_mock.assert_not_called()
        self.assertEqual(config.read_bytes(), original)

    def test_config_size_limit_refuses_before_transport(self):
        self.write_config('large="' + 'x' * 300 + '"\n')
        with patch.object(self.router, "_GROK_SAMPLING_MAX_CONFIG_BYTES", 128):
            with self.assertRaisesRegex(ValueError, "bounded trusted regular"):
                self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0)
        self.run_mock.assert_not_called()

    def test_user_config_patches_cannot_replace_inserted_sampling_policy(self):
        for kind in ("experiments", "version_overrides", "campaigns"):
            with self.subTest(kind=kind):
                config = self.write_config(
                    '[model."grok-4.7"]\ntemperature=0.0\n'
                    + '[[' + kind + ']]\nid="unit-sampling-patch"\n'
                    + '[' + kind + '.model."grok-4.7"]\ntemperature=1.0\n'
                    + '[' + kind + '.cli]\nuse_leader=true\n')
                original = config.read_bytes()
                with self.assertRaisesRegex(ValueError, "user config patches"):
                    self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0)
                self.assertEqual(config.read_bytes(), original)
        self.run_mock.assert_not_called()

    def test_system_requirements_patches_are_not_silently_overridden(self):
        for kind in ("experiments", "version_overrides", "campaigns"):
            with self.subTest(kind=kind):
                policy = '[[' + kind + ']]\nid="unit-required-patch"\n'
                policy += '[' + kind + '.model."grok-4.7"]\ntemperature=1.0\n'
                with patch.object(self.router, "_grok_sampling_config_text", side_effect=lambda path:
                        policy if path == Path("/etc/grok/requirements.toml") else self.read_config(path)):
                    with self.assertRaisesRegex(ValueError, "requirements config patches"):
                        self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0)
        self.run_mock.assert_not_called()

    def test_home_metadata_bound_stops_the_streaming_scan(self):
        real_scandir = os.scandir
        observed = []
        class BoundedListing:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def __iter__(self):
                return self
            def __next__(self):
                if len(observed) == 3:
                    raise AssertionError("metadata scan continued beyond its refusal bound")
                name = "unit-entry-" + str(len(observed))
                observed.append(name)
                return SimpleNamespace(name=name)
        with (patch.object(self.router, "_GROK_SAMPLING_MAX_HOME_ENTRIES", 2),
              patch.object(os, "scandir", side_effect=lambda path:
                  BoundedListing() if Path(path) == self.grok_home else real_scandir(path))):
            with self.assertRaisesRegex(ValueError, "metadata bound"):
                self.provider().generate("unit prompt", model_name="grok-4.7", temperature=0)
        self.assertEqual(len(observed), 3)
        self.run_mock.assert_not_called()

    def test_auth_and_other_home_entries_are_referenced_without_content_reads(self):
        auth = self.grok_home / "auth.json"
        auth.write_bytes(b"synthetic fixture")
        requirements = self.grok_home / "requirements.toml"
        requirements.write_text('[model."grok-4.7"]\ntemperature=0.0\n')
        requirements.chmod(0o600)
        cache = self.grok_home / "cache"
        cache.mkdir()
        source_open = os.open
        opened = []
        def bounded_open(path, *args, **kwargs):
            opened.append(Path(path))
            if Path(path) == auth:
                raise AssertionError("credential bytes must not be opened")
            return source_open(path, *args, **kwargs)
        environment = {"HOME": str(self.home), "GROK_HOME": str(self.grok_home), "SAFE_UNIT_MARKER": "kept"}
        with patch.object(os, "open", side_effect=bounded_open):
            with self.router._grok_zero_temperature_environment(environment,
                    model="grok-4.7", command=["/fixture/grok", "--model", "grok-4.7"]) as env:
                root = Path(env["GROK_HOME"])
                self.assertEqual((root / "auth.json").readlink(), auth)
                self.assertEqual((root / "requirements.toml").readlink(), requirements)
                self.assertEqual((root / "cache").readlink(), cache)
                self.assertEqual(env["SAFE_UNIT_MARKER"], "kept")
        self.assertNotIn(auth, opened)
        self.assertEqual(environment["GROK_HOME"], str(self.grok_home))

    def test_planning_runner_passes_zero_once_with_fallback_disabled(self):
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        from ipfs_accelerate_py.llm_allocation import intelligence_index
        from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner
        from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting

        self.stack.enter_context(patch.dict(os.environ))
        for name in (*multi_supervisor_runner.DATABASE_PROGRAM_ENV_NAMES,
                     *multi_supervisor_runner.STATE_CREDENTIAL_ENV_NAMES,
                     "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN"):
            os.environ.pop(name, None)
        observed = []
        def generate(prompt, **kwargs):
            observed.append(kwargs)
            self.assertEqual(kwargs["temperature"], 0)
            self.assertEqual(kwargs["model_name"], "grok-4.7")
            self.assertFalse(kwargs["allow_local_fallback"])
            self.assertFalse(kwargs["allow_cross_provider_fallback"])
            set_last_cli_observation("grok_cli", {"exit_code": 0})
            return "unit planning response"
        resolve = _fixture_library_resolver()
        for name in ("c", "pthread"):
            path = Path(resolve(name))
            self.assertTrue(path.is_absolute())
            self.assertEqual(path.stat().st_uid, 0)
            with path.open("rb") as stream:
                self.assertEqual(stream.read(4), b"\x7fELF")
        with self.assertRaisesRegex(AssertionError, "unqualified library"):
            resolve("arbitrary-library")
        with (patch.object(ctypes.util, "find_library", side_effect=resolve),
              patch.object(subprocess, "check_output", return_value=str(self.root)),
              contextlib.chdir(self.root),
              patch.object(intelligence_index, "discover_available_providers", return_value=["grok_cli"]),
              patch.object(intelligence_index, "select_efficient_route", return_value=SimpleNamespace(
                  provider="grok_cli", model_name="grok-4.7", catalog_revision="unit-only")),
              patch.object(self.canonical_router, "get_llm_provider", return_value=object()),
              patch.object(self.canonical_router, "generate_text", side_effect=generate),
              patch.object(context_delivery_accounting, "observe_invocation", return_value={"status": "unit-only"}),
              contextlib.redirect_stdout(io.StringIO())):
            output, receipt = self.runner.run(prompt="unit planning instruction", provider="grok_cli",
                model="grok-4.7", purpose="planning", timeout=90, max_output_tokens=4096)
        self.assertEqual(output, "unit planning response")
        self.assertEqual(len(observed), 1)
        self.assertFalse(receipt["completion_authority"])


if __name__ == "__main__":
    unittest.main()
