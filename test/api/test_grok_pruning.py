"""Authored isolated config checks; no native provider or host config access."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tomllib

import pytest

from ipfs_accelerate_py.cli_runtime import grok_pruning as policy


OFF = "grok-pruning-off@1"
ON = "grok-pruning-on@1"


def _installed(tmp_path, profile=OFF):
    path = tmp_path / policy.CONFIG_RELATIVE_PATH
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_bytes(policy.render_grok_pruning_config(profile))
    path.chmod(0o444)
    return path


def _observe(root, profile=OFF):
    return policy.observe_deployed_grok_pruning(profile, isolated_root=root,
        expected_uid=os.getuid(), expected_gid=os.getgid())


def test_legacy_is_no_io_and_preserves_omitted_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy performed I/O")
    monkeypatch.setattr(policy.os, "open", forbidden)
    monkeypatch.setattr(policy.Path, "read_bytes", forbidden)
    monkeypatch.setattr(policy.subprocess, "Popen", forbidden)
    assert policy.validate_grok_pruning_profile("legacy", provider="other", purpose="planning") == "legacy"
    assert policy.grok_pruning_selection("legacy") == {}
    assert policy.render_grok_pruning_config("legacy") is None
    assert policy.grok_pruning_declaration("legacy") == {}
    assert policy.validate_grok_pruning_declaration({}) == {}
    assert policy.observe_deployed_grok_pruning("legacy", isolated_root=object()) == {}
    assert policy.require_grok_pruning_archive(object(), "legacy") is None
    assert policy.run_grok_pruning_preflight("legacy", command=object(), env=object(), timeout=object()) == {}


@pytest.mark.parametrize("value", [None, False, 1, [], {}, "", "on", "grok-pruning-on@2"])
def test_profiles_are_strict_and_versioned(value):
    with pytest.raises(ValueError):
        policy.validate_grok_pruning_profile(value)


@pytest.mark.parametrize("provider,purpose", [("codex_cli", "coding"), ("grok_cli", "planning"),
    (False, "coding"), ("grok_cli", None)])
def test_nonlegacy_scope_is_only_grok_coding(provider, purpose):
    with pytest.raises(ValueError):
        policy.validate_grok_pruning_profile(ON, provider=provider, purpose=purpose)


def test_pruning_configs_differ_only_by_enabled():
    configs = [policy.render_grok_pruning_config(profile) for profile in (OFF, ON)]
    parsed = [tomllib.loads(value.decode("ascii")) for value in configs]
    assert configs[0].replace(b"enabled = false", b"enabled = true") == configs[1]
    for profile, value, enabled in zip((OFF, ON), parsed, (False, True)):
        assert value == {"compaction": {"pruning": {"enabled": enabled,
            "keep_last_n_turns": 3, "soft_trim_threshold": 4000, "soft_trim_head": 1500,
            "soft_trim_tail": 1500, "hard_clear_age_turns": 10}}}
        assert policy.grok_pruning_selection(profile) == {"grok_pruning_profile": profile}


@pytest.mark.parametrize("profile", [OFF, ON])
def test_declaration_is_hash_bound_and_has_no_effectiveness_or_savings_claim(profile):
    value = policy.grok_pruning_declaration(profile)
    raw = policy.render_grok_pruning_config(profile)
    assert value["config_bytes"] == len(raw)
    assert value["config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert value["effective_settings_verified"] is value["token_savings_established"] is False
    assert policy.validate_grok_pruning_declaration(value, profile=profile) == value
    with pytest.raises(ValueError):
        policy.validate_grok_pruning_declaration(value, profile=ON if profile == OFF else OFF)


@pytest.mark.parametrize("change", [dict(config_bytes=True), dict(config_bytes=151.0),
    dict(config_sha256="0"*64), dict(effective_settings_verified=True), dict(proof_authority=0),
    dict(profile=ON), dict(extra="PRIVATE"), dict(schema="native-grok-pruning-declaration@2")])
def test_declaration_rejects_tampering_and_type_confusion(change):
    value = policy.grok_pruning_declaration(OFF)
    value.update(change)
    with pytest.raises(ValueError):
        policy.validate_grok_pruning_declaration(value, profile=OFF)


@pytest.mark.parametrize("profile", [OFF, ON])
def test_exact_deployed_file_observation_is_preinvoke_only(tmp_path, profile):
    _installed(tmp_path, profile)
    value = _observe(tmp_path, profile)
    assert value["exact_bytes_verified"] is True
    assert value["immutable_during_invocation"] is value["effective_settings_verified"] is False
    assert value["mode"] == 0o444
    assert "config.toml" not in json.dumps(value)
    assert str(tmp_path) not in json.dumps(value)
    assert policy.validate_grok_pruning_file_observation(value, profile=profile,
        expected_uid=os.getuid(), expected_gid=os.getgid()) == value


@pytest.mark.parametrize("change", ["wrong_profile", "trailing_byte", "truncated", "writable", "execute", "hardlink",
    "owner", "group", "directory", "fifo", "missing"])
def test_deployed_file_refuses_unsafe_or_mismatched_inputs(tmp_path, change):
    path = _installed(tmp_path)
    kwargs = dict(expected_uid=os.getuid(), expected_gid=os.getgid())
    if change in ("wrong_profile", "trailing_byte", "truncated"):
        path.chmod(0o644)
        raw = policy.render_grok_pruning_config(ON) if change == "wrong_profile" else path.read_bytes()
        raw = raw + b"\n" if change == "trailing_byte" else raw[:-1] if change == "truncated" else raw
        path.write_bytes(raw)
        path.chmod(0o444)
    elif change == "writable": path.chmod(0o644)
    elif change == "execute": path.chmod(0o544)
    elif change == "hardlink": os.link(path, tmp_path / "alias")
    elif change == "owner": kwargs["expected_uid"] += 1
    elif change == "group": kwargs["expected_gid"] += 1
    else:
        path.unlink()
        if change == "directory": path.mkdir()
        elif change == "fifo": os.mkfifo(path, 0o444)
    with pytest.raises(ValueError, match="deployed Grok pruning"):
        policy.observe_deployed_grok_pruning(OFF, isolated_root=tmp_path, **kwargs)


@pytest.mark.parametrize("component", ["file", "parent", "root"])
def test_symlinks_are_rejected_at_every_boundary(tmp_path, component):
    root = tmp_path / "root"
    path = _installed(root)
    if component == "file":
        target = tmp_path / "target"
        path.rename(target)
        path.symlink_to(target)
    elif component == "parent":
        target = tmp_path / "grok"
        path.parent.rename(target)
        path.parent.symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "target"
        root.rename(target)
        root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="deployed Grok pruning"):
        _observe(root)


def test_replacement_during_read_is_rejected(tmp_path, monkeypatch):
    path = _installed(tmp_path)
    real_read = policy.os.read
    replaced = False
    def replace_then_read(fd, size):
        nonlocal replaced
        if not replaced:
            replaced = True
            path.unlink()
            path.write_bytes(policy.render_grok_pruning_config(OFF))
            path.chmod(0o444)
        return real_read(fd, size)
    monkeypatch.setattr(policy.os, "read", replace_then_read)
    with pytest.raises(ValueError, match="deployed Grok pruning"):
        _observe(tmp_path)


@pytest.mark.parametrize("change", [dict(mode=0o644), dict(owner_uid=True), dict(single_link=1),
    dict(effective_settings_verified=True), dict(immutable_during_invocation=True), dict(extra="PRIVATE")])
def test_file_receipt_cannot_claim_stronger_evidence(tmp_path, change):
    _installed(tmp_path)
    value = _observe(tmp_path)
    value.update(change)
    with pytest.raises(ValueError):
        policy.validate_grok_pruning_file_observation(value, expected_uid=os.getuid(), expected_gid=os.getgid())


def test_archive_requires_exact_current_inventory(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    relative = "ipfs_accelerate_py/cli_runtime/grok_pruning.py"
    path = checkout / relative
    path.parent.mkdir(parents=True)
    raw = b"# authored current source\n"
    path.write_bytes(raw)
    monkeypatch.setattr(policy, "__file__", str(path))
    monkeypatch.setattr(policy, "SOURCE_PATHS", (relative,))
    row = {"path": "source/" + relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    manifest = {"files": [row]}
    assert policy.require_grok_pruning_archive(manifest, ON) is None
    for invalid in ({}, {"files": []}, {"files": [row, row]},
            {"files": [dict(row, bytes=True)]}, {"files": [dict(row, sha256="0"*64)]}):
        with pytest.raises(ValueError): policy.require_grok_pruning_archive(invalid, ON)
    path.write_bytes(b"# changed\n")
    with pytest.raises(ValueError): policy.require_grok_pruning_archive(manifest, ON)


def _native_report():
    value = {key: [] for key in ("agents", "hooks", "lspServers", "marketplaces", "mcpServers", "plugins",
                                  "projectInstructions", "skills")}
    value.update(configSources={"layers": []}, externalCompat={}, loginPolicy={}, permissions={},
        channel="authored-channel", cwd="/PRIVATE-WORKSPACE", grokVersion="authored-format",
        projectRoot=None, projectTrusted=False)
    return value


def _inspection(profile=OFF, report=None, stderr=b""):
    raw = json.dumps(_native_report() if report is None else report).encode()
    return policy.observe_grok_pruning_inspection(profile, stdout=raw, stderr=stderr, returncode=0)


def _preflight(profile=OFF):
    return policy.build_grok_pruning_preflight(profile,
        file_observation=policy._expected_file_observation(profile, 0, 0),
        effective_observation=_inspection(profile))


def test_inspection_preserves_unreported_effectiveness_and_does_not_echo_native_fields():
    value = _inspection(stderr=b"PRIVATE-STDERR")
    assert value["status"] == "unavailable"
    assert value["native_inspect_commands"] == 1 and value["native_exit_code"] == 0
    assert value["pruning_values_reported"] is value["config_source_verified"] is False
    assert value["effective_settings_verified"] is value["native_version_scalar_verified"] is False
    assert value["version_basis"] == "pinned_binary_identity"
    assert value["binary_sha256"] == policy.GROK_BINARY_SHA256
    assert value["stderr_sha256"] == hashlib.sha256(b"PRIVATE-STDERR").hexdigest()
    assert "PRIVATE" not in json.dumps(value) and "authored-format" not in json.dumps(value)
    assert policy.validate_grok_pruning_effective_observation(value) == value


@pytest.mark.parametrize("key,value", [("grokVersion", None), ("grokVersion", "x"*129),
    ("grokVersion", "secret\n"), ("channel", "é"), ("projectTrusted", 1), ("agents", {}),
    ("configSources", {"layers": [{"path": "/PRIVATE"}]}), ("configSources", {}),
    ("cwd", ""), ("permissions", []), ("projectRoot", 1), ("unqualified", {})])
def test_inspection_requires_qualified_bounded_schema(key, value):
    report = _native_report()
    report[key] = value
    with pytest.raises(ValueError, match="unsupported bounded native Grok inspection"):
        _inspection(report=report)


@pytest.mark.parametrize("key", ["pruning", "compaction", "keep_last_n_turns", "softTrimThreshold",
    "soft_trim_head", "soft_trim_tail", "hardClearAgeTurns", "compaction.pruning.enabled"])
def test_new_pruning_fields_cannot_silently_become_unreported(key):
    report = _native_report()
    report["permissions"] = {key: False}
    with pytest.raises(ValueError, match="unsupported bounded native Grok inspection"):
        _inspection(report=report)


@pytest.mark.parametrize("stdout,stderr,code", [(b"", b"", 0), (b"PRIVATE invalid", b"", 0),
    (b"{}", b"", 0), (b"{}", b"", 1), (b"{}", b"", False), (b"{}", "PRIVATE", 0),
    (b"x"*(policy.MAX_INSPECT_STDOUT_BYTES+1), b"", 0),
    (b"{}", b"x"*(policy.MAX_INSPECT_STDERR_BYTES+1), 0),
    (b'{"x":1,"x":2}', b"", 0), (b"[NaN]", b"", 0)])
def test_bad_native_inspection_has_closed_safe_error(stdout, stderr, code):
    with pytest.raises(ValueError) as error:
        policy.observe_grok_pruning_inspection(OFF, stdout=stdout, stderr=stderr, returncode=code)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("profile", [OFF, ON])
def test_combined_preflight_binds_all_profiles_but_not_native_behavior(profile):
    value = _preflight(profile)
    assert policy.validate_grok_pruning_preflight(value, profile=profile) == value
    assert value["preflight_completed"] is value["requested_profile_deployed"] is value["native_inspect_completed"] is True
    assert value["effective_settings_verified"] is value["config_file_consumption_verified"] is False
    assert value["behavior_verified"] is value["token_savings_established"] is False
    for lane in ("declaration", "file_observation", "effective_observation"):
        assert value[lane]["profile"] == profile
    assert policy.validate_grok_pruning_preflight(value) == value


@pytest.mark.parametrize("path,replacement", [
    (("profile",), ON), (("declaration", "profile"), ON), (("file_observation", "profile"), ON),
    (("effective_observation", "profile"), ON), (("effective_observation", "status"), "verified"),
    (("effective_observation", "native_exit_code"), False),
    (("effective_observation", "stdout_bytes"), True),
    (("effective_observation", "stderr_sha256"), "0"*64),
    (("effective_observation", "native_version_scalar_verified"), True),
    (("file_observation", "owner_uid"), 1001), (("file_observation", "mode"), 0o644),
    (("effective_settings_verified",), True), (("config_file_consumption_verified",), True),
    (("behavior_verified",), True), (("token_savings_established",), True),
    (("preflight_completed",), 1), (("extra",), "PRIVATE")])
def test_combined_receipt_rejects_tampering_and_unearned_authority(path, replacement):
    value = _preflight()
    target = value
    for key in path[:-1]: target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError): policy.validate_grok_pruning_preflight(value, profile=OFF)


def _coding(receipt=None, provider="grok_cli"):
    return {"purpose": "coding", "phase": "coding", "provider": provider,
        "provider_invocation_policy": {} if receipt is None else {"grok_pruning": receipt}}


def test_runtime_match_means_observed_deployment_only_and_legacy_remains_omitted():
    row = _coding(_preflight())
    report = {"provider_invocations": [{"purpose": "planning", "provider": "other"}, row]}
    assert policy.grok_pruning_runtime_matches(report, OFF) is True
    assert row["provider_invocation_policy"]["grok_pruning"]["effective_settings_verified"] is False
    assert policy.grok_pruning_runtime_matches(report, ON) is False
    assert policy.grok_pruning_runtime_matches({"provider_invocations": [_coding()]}, "legacy") is True
    assert policy.grok_pruning_runtime_matches({"provider_invocations": [_coding()]}, OFF) is False


@pytest.mark.parametrize("report", [None, {}, {"provider_invocations": []},
    {"provider_invocations": [{"purpose": "planning", "provider": "other"}]}])
def test_runtime_match_preserves_missing_evidence(report):
    assert policy.grok_pruning_runtime_matches(report, OFF) is None


@pytest.mark.parametrize("mutation", ["wrong_provider", "phase", "unreceipted", "malformed_row", "extra_coding"])
def test_runtime_match_requires_every_coding_invocation(mutation):
    report = {"provider_invocations": [_coding(_preflight())]}
    row = report["provider_invocations"][0]
    if mutation == "wrong_provider": row["provider"] = "codex_cli"
    elif mutation == "phase": row["phase"] = "planning"
    elif mutation == "unreceipted": report["unreceipted_provider_attempt"] = True
    elif mutation == "malformed_row": report["provider_invocations"].append(None)
    else: report["provider_invocations"].append(_coding())
    assert policy.grok_pruning_runtime_matches(report, OFF) is False


def _env():
    home = str(policy.ISOLATED_ROOT / "worker-home")
    return {"HOME": home, "GROK_HOME": home + "/.grok"}


@pytest.mark.parametrize("command", ["grok", ["grok"], [policy.GROK_BINARY_PATH, "--config=PRIVATE"],
    [policy.GROK_BINARY_PATH, "inspect"], [], None])
def test_preflight_rejects_arbitrary_commands_before_any_io(monkeypatch, command):
    monkeypatch.setattr(policy, "observe_deployed_grok_pruning", lambda *a, **k: pytest.fail("unexpected I/O"))
    with pytest.raises(ValueError):
        policy.run_grok_pruning_preflight(OFF, command=command, env=_env(), timeout=1)


@pytest.mark.parametrize("changes", [{"HOME": "/PRIVATE"}, {"GROK_HOME": "/PRIVATE"},
    {"GROK_CONFIG": ""}, {"GROK_CONFIG_PATH": "/PRIVATE"}, {"PRIVATE": None}, {"X\0": "bad"}])
def test_preflight_rejects_config_environment_ambiguity_before_io(monkeypatch, changes):
    monkeypatch.setattr(policy, "observe_deployed_grok_pruning", lambda *a, **k: pytest.fail("unexpected I/O"))
    environment = _env()
    environment.update(changes)
    with pytest.raises(ValueError):
        policy.run_grok_pruning_preflight(OFF, command=[policy.GROK_BINARY_PATH], env=environment, timeout=1)


@pytest.mark.parametrize("timeout", [False, 0, -1, float("inf"), float("nan"), 31, "1",
    pytest.param(1 << 100_000, id="huge-integer")])
def test_preflight_timeout_is_bounded_before_io(monkeypatch, timeout):
    monkeypatch.setattr(policy, "observe_deployed_grok_pruning", lambda *a, **k: pytest.fail("unexpected I/O"))
    with pytest.raises(ValueError):
        policy.run_grok_pruning_preflight(OFF, command=[policy.GROK_BINARY_PATH], env=_env(), timeout=timeout)


def test_preflight_rechecks_config_after_inspection_and_freezes_environment(monkeypatch):
    observations = []
    file_receipt = policy._expected_file_observation(OFF, 0, 0)
    def observe(profile):
        observations.append(profile)
        return deepcopy(file_receipt)
    monkeypatch.setattr(policy, "observe_deployed_grok_pruning", observe)
    monkeypatch.setattr(policy, "_verify_native_binary", lambda deadline: None)
    environment = _env()
    def inspect(command, env, timeout):
        assert env == environment and env is not environment
        assert command == [policy.GROK_BINARY_PATH] and 0 < timeout <= 1
        return json.dumps(_native_report()).encode(), b"", 0
    monkeypatch.setattr(policy, "_run_inspect", inspect)
    result = policy.run_grok_pruning_preflight(OFF, command=[policy.GROK_BINARY_PATH], env=environment, timeout=1)
    assert observations == [OFF, OFF] and result == _preflight()
    def changed(profile):
        observations.append(profile)
        if len(observations) > 3: raise ValueError("authored config drift")
        return deepcopy(file_receipt)
    monkeypatch.setattr(policy, "observe_deployed_grok_pruning", changed)
    with pytest.raises(ValueError, match="authored config drift"):
        policy.run_grok_pruning_preflight(OFF, command=[policy.GROK_BINARY_PATH], env=environment, timeout=1)


def test_cli_observation_preserves_valid_closed_pruning_receipt_without_aliasing_input():
    from ipfs_accelerate_py.cli_runtime import cli_metadata
    value = _preflight()
    expected = deepcopy(value)
    cli_metadata.set_last_cli_observation("grok_cli", {"exit_code": 0, "native_grok_pruning": value})
    value["effective_observation"]["status"] = "PRIVATE"
    observed = cli_metadata.get_last_cli_observation("grok_cli")
    assert observed["native_grok_pruning"] == expected
    assert type(observed["native_grok_pruning"]) is dict
    assert "PRIVATE" not in json.dumps(observed)
    cli_metadata.set_last_cli_observation("grok_cli", {})


@pytest.mark.parametrize("value", [None, {}, "PRIVATE", ["PRIVATE"], {"text": "PRIVATE"},
    {"schema": "native-grok-pruning-preflight@2", "secret": "PRIVATE"}])
def test_cli_observation_drops_invalid_or_empty_receipts_everywhere(value):
    from ipfs_accelerate_py.cli_runtime import cli_metadata
    payload = {"native_grok_pruning": value, "exit_code": 0}
    cli_metadata.set_last_cli_observation("grok_cli", payload)
    observed = cli_metadata.get_last_cli_observation("grok_cli")
    extracted = cli_metadata.extract_cli_metadata(provider="grok_cli", extra=payload)
    remembered = cli_metadata.remember_cli_run("grok_cli", extra=payload)
    for candidate in (observed, extracted, remembered, cli_metadata.get_last_cli_observation("grok_cli")):
        assert "native_grok_pruning" not in candidate
        assert "PRIVATE" not in json.dumps(candidate)
    cli_metadata.set_last_cli_observation("grok_cli", {})


@pytest.mark.parametrize("provider", ["codex_cli", "", "unknown"])
def test_pruning_receipt_is_only_preserved_for_grok(provider):
    from ipfs_accelerate_py.cli_runtime import cli_metadata
    payload = {"native_grok_pruning": _preflight()}
    assert "native_grok_pruning" not in cli_metadata.extract_cli_metadata(provider=provider, extra=payload)
    if provider:
        cli_metadata.set_last_cli_observation(provider, payload)
        assert "native_grok_pruning" not in cli_metadata.get_last_cli_observation(provider)
        cli_metadata.set_last_cli_observation(provider, {})


def test_cli_explicit_extra_preserves_valid_receipt_but_native_stdout_cannot_forge_it():
    from ipfs_accelerate_py.cli_runtime import cli_metadata
    value = _preflight()
    extracted = cli_metadata.extract_cli_metadata(provider="grok_cli", extra={"native_grok_pruning": value})
    assert extracted["native_grok_pruning"] == value
    remembered = cli_metadata.remember_cli_run("grok_cli", extra={"native_grok_pruning": value})
    assert remembered["native_grok_pruning"] == value
    assert cli_metadata.get_last_cli_observation("grok_cli")["native_grok_pruning"] == value
    for candidate in (value, {"model": "PRIVATE", "input_tokens": 777}):
        forged = cli_metadata.extract_cli_metadata(json.dumps({"native_grok_pruning": candidate}), provider="grok_cli")
        assert "native_grok_pruning" not in forged and "prompt_tokens" not in forged
        assert "PRIVATE" not in json.dumps(forged)
    cli_metadata.set_last_cli_observation("grok_cli", {})


def test_cli_receipt_drops_tampering_without_stringifying_private_extensions():
    from ipfs_accelerate_py.cli_runtime import cli_metadata
    value = _preflight()
    value["effective_observation"]["PRIVATE"] = "PRIVATE"
    cli_metadata.set_last_cli_observation("grok_cli", {"native_grok_pruning": value})
    assert "native_grok_pruning" not in cli_metadata.get_last_cli_observation("grok_cli")
    assert "PRIVATE" not in json.dumps(cli_metadata.extract_cli_metadata(provider="grok_cli",
        extra={"native_grok_pruning": value}))
    cli_metadata.set_last_cli_observation("grok_cli", {})
