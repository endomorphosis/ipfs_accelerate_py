"""Private-config writer tests; no live Grok or native-route qualification.

Extract only the actual reviewed function, with explicit fixture dependencies.
This avoids importing a provider module or inspecting operator credentials.
The full supervisor must separately qualify effective Grok sampling settings.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import tomllib
from types import SimpleNamespace

import pytest


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    """Create ordinary directories without pytest's numbered-current symlink."""
    return tmp_path_factory.mktemp(request.node.name, numbered=False)


def _source():
    configured = os.environ.get("GROK_TEMPERATURE_ZERO_CANDIDATE", "")
    expected = os.environ.get("GROK_TEMPERATURE_ZERO_CANDIDATE_SHA256", "")
    if bool(configured) != bool(expected):
        raise ValueError("candidate path and digest must be supplied together")
    path = (Path(configured) if configured else Path(__file__).resolve().parents[2]
            / "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py")
    raw = path.read_bytes()
    if configured and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("reviewed candidate source changed")
    return path, ast.parse(raw, filename=str(path))


def _fixture_writer(tmp_path, deny=None):
    path, tree = _source()
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_isolated_grok_home"]
    assert len(functions) == 1
    profiles = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == "GROK_PRIMARY_SANDBOX_PROFILE"
                        for target in node.targets)
                and isinstance(node.value, ast.Constant)]
    assert len(profiles) == 1 and isinstance(profiles[0], str)
    operator = tmp_path / "fixture-operator"
    operator.mkdir()
    workspace = tmp_path / "fixture-workspace"
    workspace.mkdir()
    made = []
    denied = []

    def allocate(**kwargs):
        result = tempfile.TemporaryDirectory(dir=tmp_path, **kwargs)
        made.append(Path(result.name))
        return result

    def fixture_denials(**kwargs):
        values = (kwargs["grok_home"], kwargs["sentinel_path"])
        denied.extend(values)
        return values if deny is None else deny(kwargs)

    namespace = {"Path": Path, "json": json,
                 "tempfile": SimpleNamespace(TemporaryDirectory=allocate),
                 "GROK_PRIMARY_SANDBOX_PROFILE": profiles[0],
                 "_grok_isolation_deny_paths": fixture_denials,
                 "user_home_from_env": lambda env: Path(env["HOME"])}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), functions[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_isolated_grok_home"], operator, workspace, made, denied, profiles[0]


def _open_fixture(tmp_path, *, operator_config=None):
    writer, operator, workspace, made, denied, profile = _fixture_writer(tmp_path)
    if operator_config is not None:
        (operator / "config.toml").write_text(operator_config, encoding="utf-8")
    # This synthetic home contains no auth.json. No operator account is read.
    child = {"SAFE_FIXTURE_MARKER": "preserved", "HOME": "unused-parent-home"}
    result = writer(base_env={"HOME": str(operator), "GROK_HOME": str(operator)},
                    child_env=child, codex_fallback_command=(), workspace=workspace)
    return result, child, made, denied, profile


def test_exact_quoted_model_key_and_zero_temperature(tmp_path):
    (temporary, _env, _policy, _denied), _child, _made, _values, _profile = _open_fixture(tmp_path)
    try:
        text = (Path(temporary.name) / "config.toml").read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert '[model."grok-4.7"]' in text
        assert parsed["model"] == {"grok-4.7": {"temperature": 0.0}}
        assert type(parsed["model"]["grok-4.7"]["temperature"]) is float
    finally:
        temporary.cleanup()


def test_existing_compatibility_and_leader_settings_preserved(tmp_path):
    (temporary, _env, _policy, _denied), _child, _made, _values, _profile = _open_fixture(tmp_path)
    try:
        parsed = tomllib.loads((Path(temporary.name) / "config.toml").read_text())
        assert set(parsed) == {"compat", "cli", "model"}
        assert parsed["compat"] == {
            "cursor": {key: False for key in ("skills", "rules", "agents", "mcps", "hooks", "sessions")},
            "claude": {key: False for key in ("skills", "rules", "agents", "mcps", "hooks", "sessions")},
            "codex": {"sessions": False},
        }
        assert parsed["cli"] == {"use_leader": False}
    finally:
        temporary.cleanup()


def test_no_context_or_model_identity_override_or_operator_config_copy(tmp_path):
    operator_config = ('[model."grok-4.7"]\nmodel="fixture-other-model"\n'
                       'temperature=1.7\ncontext_window=999999\n'
                       '[models]\ndefault="fixture-other-model"\n')
    (temporary, _env, _policy, _denied), _child, _made, _values, _profile = _open_fixture(
        tmp_path, operator_config=operator_config)
    try:
        text = (Path(temporary.name) / "config.toml").read_text()
        parsed = tomllib.loads(text)
        assert parsed["model"]["grok-4.7"] == {"temperature": 0.0}
        assert "models" not in parsed
        assert "context_window" not in text and "max_completion_tokens" not in text
        assert "fixture-other-model" not in text and "999999" not in text
    finally:
        temporary.cleanup()


def test_private_modes_and_original_environment_isolation_preserved(tmp_path):
    (temporary, env, _policy, _denied), child, made, _values, _profile = _open_fixture(tmp_path)
    root = Path(temporary.name)
    try:
        assert made == [root] and stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE((root / "config.toml").stat().st_mode) == 0o600
        assert env["HOME"] == str(root) and env["GROK_HOME"] == str(root)
        for key, suffix in (("XDG_CONFIG_HOME", "xdg-config"), ("XDG_DATA_HOME", "xdg-data"),
                            ("XDG_STATE_HOME", "xdg-state")):
            assert env[key] == str(root / suffix)
        assert env["SAFE_FIXTURE_MARKER"] == "preserved"
        assert child == {"SAFE_FIXTURE_MARKER": "preserved", "HOME": "unused-parent-home"}
        assert not (root / "auth.json").exists()
    finally:
        temporary.cleanup()
    assert not root.exists()


def test_original_sandbox_policy_is_preserved(tmp_path):
    (temporary, _env, policy, denied), _child, _made, values, profile = _open_fixture(tmp_path)
    try:
        root = Path(temporary.name)
        parsed = tomllib.loads(policy.read_text())
        assert denied == tuple(values) and root in denied
        assert parsed == {"profiles": {profile: {"extends": "workspace",
            "restrict_network": True, "deny": [str(path) for path in values]}}}
        assert stat.S_IMODE(policy.stat().st_mode) == 0o600
        assert (root / "alternate-provider-deny-sentinel").read_text() == "provider isolation sentinel\n"
    finally:
        temporary.cleanup()


def test_missing_original_home_deny_refuses_and_cleans_fixture(tmp_path):
    writer, operator, workspace, made, _values, _profile = _fixture_writer(tmp_path,
        deny=lambda kwargs: (kwargs["sentinel_path"],))
    with pytest.raises(ValueError, match="state-directory deny was not resolved"):
        writer(base_env={"HOME": str(operator)}, child_env={},
               codex_fallback_command=(), workspace=workspace)
    assert len(made) == 1 and not made[0].exists()
