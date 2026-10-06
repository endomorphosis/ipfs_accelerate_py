"""Selected setup-cache transport controls; no Docker, model or provider calls."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_advice as policy
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_codex as native
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_files as archive_files
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_libraries as libraries


@pytest.fixture
def selected_cache(tmp_path, monkeypatch):
    # Source384's existing transport suite owns model validation. This fixture
    # exercises the additive cache binding without loading a checkpoint.
    monkeypatch.setattr(policy.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(deployment, "validate_source384_binding", lambda m: m.get("source384"))
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    monkeypatch.setattr(adapter, "validate_source384_binding", lambda m: m.get("source384"))
    rows = []
    for name in policy.HELPERS:
        raw = Path(policy.__file__).with_name(name).read_bytes()
        rows.append(dict(path=policy.PREFIX + name, bytes=len(raw), mode=0o644,
                         sha256=hashlib.sha256(raw).hexdigest()))
    rows.extend(dict(path="extensions/" + row["name"], bytes=row["expected_bytes"],
        mode=row["mode"], sha256=row["sha256"]) for row in libraries.ROWS[:3])
    wheel_path = "runtime-wheels/torch-cpu/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"
    rows.append(dict(path=wheel_path, bytes=155005253, mode=0o644, sha256=libraries.WHEEL_SHA256))
    manifest = dict(schema="terminal-supervisor-runtime-archive@1", archive_sha256="a" * 64,
                    source384={"authored_fixture": True, "config": {}}, codex_version="0.158.0",
                    learned_requirements=[], files=rows, torch_cpu_requirement=libraries.TORCH_REQUIREMENT,
                    torch_cpu_wheel=dict(schema="terminal-torch-cpu-wheel@1", path=wheel_path,
                        bytes=155005253, sha256=libraries.WHEEL_SHA256, requirement=libraries.TORCH_REQUIREMENT))
    manifest["setup_cache"] = policy.binding_for_manifest(manifest, policy.POLICY)
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    selection = policy.select_setup_cache(root, policy.POLICY)
    auth = tmp_path / "auth.json"
    auth.write_text("private authored fixture: never read")
    auth.chmod(0o600)
    return root, manifest, selection, auth


def test_absent_policy_preserves_default_without_prerequisites(tmp_path):
    manifest = {"schema": "legacy authored fixture", "files": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert policy.select_setup_cache(tmp_path, None) is None
    assert policy.validate_setup_cache_selection(tmp_path, None) == (manifest, None)
    policy.validate_setup_cache_prerequisites(None, install_codex=False, auth_json=None,
                                              arm="no-index", resource_profile=None)
    assert asyncio.run(policy.apply_setup_cache_advice(None, archive_dir=tmp_path,
        expected=None, boundary_output=tmp_path / "absent", output=tmp_path / "unused")) is None
    assert not (tmp_path / "unused").exists()


def test_historical_cache_policy_is_not_reinterpreted_for_current_cli(selected_cache):
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
    _, manifest, _, _ = selected_cache
    assert manifest["codex_version"] == "0.158.0"
    assert policy.validate_manifest_binding(manifest)["codex_version"] == "0.158.0"
    manifest["codex_version"] = CLI_VERSION
    with pytest.raises(ValueError, match="pinned Source384 and Codex profile"):
        policy.validate_manifest_binding(manifest)


@pytest.fixture
def selected_cache_v2(selected_cache):
    root, manifest, _, auth = selected_cache
    manifest["codex_version"] = "0.160.0"
    manifest["setup_cache"] = policy.binding_for_manifest(manifest, policy.POLICY_V2)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest, policy.select_setup_cache(root, policy.POLICY_V2), auth


def _native_receipt(version, pins):
    return dict(schema="native-codex-runtime-bundle@1", codex_version=version, provider_calls=0,
        files=[dict(name=name, source_sha256=pin, sha256=pin, uid=0, mode=0o755)
               for name, pin in pins.items()],
        executable_checks={name: dict(returncode=0, stdout_sha256="a" * 64) for name in pins})


def test_v2_current_cli_binds_independent_npm_hashes_and_preserves_populations(selected_cache_v2):
    root, manifest, selection, auth = selected_cache_v2
    expected = {
        "codex": "50b06603bdcdac39b714f5c3e68583c002b8ad8779ebfdaaf4932ff016b379c0",
        "codex-code-mode-host": "7e0004bd8b37936753981729365c448bdc67f0173d9bc6453e40e3ad28774b6c",
    }
    binding = policy.validate_manifest_binding(manifest)
    assert binding["policy"] == policy.POLICY_V2 and binding["codex_version"] == "0.160.0"
    assert binding["codex_sha256"] == expected == native.PINS_V2
    assert policy.validate_setup_cache_selection(root, selection) == (manifest, binding)
    assert len(binding["native_libraries"]["rows"]) == 131
    assert binding["native_libraries"] == policy.library_binding_for_manifest(manifest)
    policy.validate_setup_cache_prerequisites(selection, install_codex=True, auth_json=auth)
    native.require_receipt(_native_receipt("0.160.0", expected), codex_version="0.160.0")


@pytest.mark.parametrize("mutation", ["old_policy", "old_version", "old_pins", "mixed_pins", "unknown_policy"])
def test_v2_policy_rejects_cross_version_or_unknown_substitution(selected_cache_v2, mutation):
    root, manifest, _, _ = selected_cache_v2
    if mutation == "old_policy":
        manifest["setup_cache"]["policy"] = policy.POLICY
    elif mutation == "old_version":
        manifest["codex_version"] = manifest["setup_cache"]["codex_version"] = "0.158.0"
    elif mutation == "old_pins":
        manifest["setup_cache"]["codex_sha256"] = dict(policy.CODEX_PINS)
    elif mutation == "mixed_pins":
        manifest["setup_cache"]["codex_sha256"]["codex"] = policy.CODEX_PINS["codex"]
    else:
        manifest["setup_cache"]["policy"] = "source384-native-aarch64-dontneed@3"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        policy.select_setup_cache(root, policy.POLICY_V2)


@pytest.mark.parametrize("selected_version,receipt_version", [
    ("0.160.0", "0.158.0"), ("0.158.0", "0.160.0"), ("0.161.0", "0.160.0"),
])
def test_native_receipt_version_cannot_select_a_different_cache_policy(selected_version, receipt_version):
    receipt = _native_receipt(receipt_version, native.PINS_BY_VERSION[receipt_version])
    with pytest.raises(ValueError):
        native.require_receipt(receipt, codex_version=selected_version)


def test_v2_native_advice_selects_four_copies_of_independent_current_binary_pins(tmp_path, monkeypatch):
    vendor = tmp_path / "home/.nvm/versions/node/v24/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-arm64/vendor/aarch64-unknown-linux-musl/bin"
    vendor.mkdir(parents=True)
    (vendor / "codex").write_text("public location fixture")
    calls = []
    def advise_selected(root, rows, *, expected_count):
        calls.append(rows)
        assert root == tmp_path and expected_count == 4
        assert {row["role"] for row in rows} == {"vendor", "exposed"}
        assert {row["uid"] for row in rows} == {0, 1000}
        assert all(row["sha256"] == native.PINS_V2[row["name"]] for row in rows)
        return {"selected_files": 4}
    monkeypatch.setattr(native, "advise_selected", advise_selected)
    receipt = _native_receipt("0.160.0", native.PINS_V2)
    assert native.advise(tmp_path, receipt, codex_version="0.160.0")["codex_version"] == "0.160.0"
    assert len(calls) == 1
    receipt["files"][-1]["sha256"] = native.PINS[receipt["files"][-1]["name"]]
    with pytest.raises(ValueError):
        native.advise(tmp_path, receipt, codex_version="0.160.0")
    assert len(calls) == 1


def test_v2_selection_refuses_old_boundary_receipt_before_upload(selected_cache_v2, tmp_path):
    root, _, selection, _ = selected_cache_v2
    boundary = tmp_path / "boundary/installation"
    boundary.mkdir(parents=True)
    (boundary / "native-codex-binary.log").write_text(json.dumps(_native_receipt("0.158.0", native.PINS)))
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock())
    output = tmp_path / "advice"
    with pytest.raises(ValueError):
        asyncio.run(policy.apply_setup_cache_advice(environment, archive_dir=root, expected=selection,
            boundary_output=boundary.parent, output=output))
    environment.upload_file.assert_not_called()
    environment.exec.assert_not_called()
    assert not output.exists()


def test_v2_transport_rejects_old_native_observation(selected_cache_v2, tmp_path):
    root, manifest, selection, _ = selected_cache_v2
    boundary = tmp_path / "boundary/installation"
    boundary.mkdir(parents=True)
    receipt = json.dumps(_native_receipt("0.160.0", native.PINS_V2)).encode()
    (boundary / "native-codex-binary.log").write_bytes(receipt)
    archive_result = dict(schema="manifest-cache-advice@1", manifest_sha256=selection["manifest_sha256"],
        archive_sha256=manifest["archive_sha256"], selected_files=len(manifest["files"]),
        body_reads=0, metadata_unchanged=True, freed_bytes_claimed=False)
    old_native_result = dict(schema="pinned-native-codex-cache-advice@1", codex_version="0.158.0",
        post_boundary_receipt_sha256=hashlib.sha256(receipt).hexdigest(), selected_files=4, hashed_files=4,
        body_reads_after_advice=0, body_read_bytes=100, selected_bytes=100,
        metadata_unchanged=True, freed_bytes_claimed=False)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(side_effect=[
        SimpleNamespace(return_code=0, stdout=json.dumps(row), stderr="")
        for row in (archive_result, old_native_result)]))
    with pytest.raises(ValueError, match="native advice receipt differs"):
        asyncio.run(policy.apply_setup_cache_advice(environment, archive_dir=root, expected=selection,
            boundary_output=boundary.parent, output=tmp_path / "advice"))
    assert environment.exec.await_count == 2


@pytest.mark.parametrize("field,value", [
    ("policy", "other"), ("architecture", "x86_64"), ("codex_version", "0.159.0"),
    ("codex_sha256", {"codex": "f" * 64}), ("helpers", {}), ("extra", True),
])
def test_manifest_policy_pin_substitution_refused_before_container(selected_cache, field, value):
    root, manifest, selection, auth = selected_cache
    manifest["setup_cache"][field] = value
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        policy.select_setup_cache(root, policy.POLICY)
    # No environment object can be invoked when the prepared manifest differs.
    with pytest.raises(ValueError):
        asyncio.run(deployment.deploy_supervisor(None, archive_dir=root,
            output=root / "deployment", auth_json=auth, setup_cache_selection=selection))
    assert not (root / "deployment").exists()


def test_policy_cannot_be_silently_enabled_or_dropped(selected_cache):
    root, manifest, selection, _ = selected_cache
    with pytest.raises(ValueError):
        policy.validate_setup_cache_selection(root, None)
    manifest.pop("setup_cache")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        policy.validate_setup_cache_selection(root, selection)
    with pytest.raises(ValueError):
        policy.select_setup_cache(root, policy.POLICY)


@pytest.mark.parametrize("mutation", ["legacy_embedding", "unknown_model", "no_source384", "wrong_helper"])
def test_incompatible_asset_layout_rejected(selected_cache, mutation):
    _, manifest, _, _ = selected_cache
    if mutation == "legacy_embedding":
        manifest["learned_requirements"] = ["legacy"]
    elif mutation == "unknown_model":
        manifest["files"].append(dict(path="models/legal/weights.bin", bytes=1,
            mode=0o644, sha256="b" * 64))
    elif mutation == "no_source384":
        manifest.pop("source384")
    else:
        manifest["files"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        policy.binding_for_manifest(manifest, policy.POLICY)


@pytest.mark.parametrize("overrides", [dict(install_codex=False), dict(auth_json=None),
    dict(arm="no-index"), dict(resource_profile=None), dict(resource_profile="source384-5cpu-8gib@1")])
def test_route_prerequisites_refuse_before_any_container_work(selected_cache, overrides):
    _, _, selection, auth = selected_cache
    kwargs = dict(install_codex=True, auth_json=auth, arm="full", resource_profile=policy.PROFILE)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        policy.validate_setup_cache_prerequisites(selection, **kwargs)


def test_auth_prerequisites_never_read_hash_or_log_credentials(selected_cache, monkeypatch):
    _, _, selection, auth = selected_cache
    def denied(*args, **kwargs):
        raise AssertionError("credential body read")
    monkeypatch.setattr(Path, "read_bytes", denied)
    monkeypatch.setattr(Path, "read_text", denied)
    policy.validate_setup_cache_prerequisites(selection, install_codex=True, auth_json=auth)
    alias = auth.with_name("alias")
    alias.symlink_to(auth)
    with pytest.raises(ValueError):
        policy.validate_setup_cache_prerequisites(selection, install_codex=True, auth_json=alias)
    monkeypatch.setattr(policy.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError):
        policy.validate_setup_cache_prerequisites(selection, install_codex=True, auth_json=auth)


def test_isolated_loader_binds_dependencies_and_protects_receipt_before_main(tmp_path):
    base = Path(archive_files.__file__)
    fixture = tmp_path / "native_fixture.py"
    fixture.write_text("from terminal_setup_cache_files import identity\n"
        "ready=False\ndef protect_receipt(root,size):\n global ready\n assert size==19\n ready=True\n"
        "def main():\n assert ready\n print('protected-before-main')\n")
    modules = [("terminal_setup_cache_files", str(base), hashlib.sha256(base.read_bytes()).hexdigest()),
               ("native_fixture", str(fixture), hashlib.sha256(fixture.read_bytes()).hexdigest())]
    def execute(rows):
        code = policy.isolated_loader(rows, run="native_fixture", argv=[str(tmp_path / "receipt"), "b" * 64], receipt_bytes=19)
        return subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code],
                              capture_output=True, text=True, timeout=5)
    result = execute(modules)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "protected-before-main"
    for index in range(2):
        bad = list(modules)
        name, path, _ = bad[index]
        bad[index] = (name, path, "0" * 64)
        assert execute(bad).returncode != 0


def test_public_receipt_protection_keeps_strict_reader(selected_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(native, "RECEIPT_UID", os.getuid())
    receipt = tmp_path / "codex-cache-exposure.json"
    raw = b'{"public":true}'
    receipt.write_bytes(raw)
    receipt.chmod(0o664)
    before = receipt.stat()
    pin = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        native.read_receipt(tmp_path, pin)
    native.protect_receipt(tmp_path, len(raw))
    assert native.read_receipt(tmp_path, pin) == {"public": True}
    assert receipt.stat().st_ino == before.st_ino
    assert receipt.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError):
        native.read_receipt(tmp_path, "0" * 64)


def test_harbor_selected_order_is_deploy_boundary_then_advice(selected_cache, tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
    from benchmarks.agent_supervisor.container_coding import benchmark_provider_profile
    # Exercise the retained historical policy with its historical configured
    # CLI; current-profile refusal is covered separately before any deployment.
    monkeypatch.setattr(benchmark_provider_profile, "CLI_VERSION", "0.158.0")
    root, manifest, _, auth = selected_cache
    from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
    # Deployment is mocked in this ordering test; keep its cache binding current
    # after supplying the explicit current worker capability metadata.
    (root / "manifest.json").write_text(json.dumps(_supervisor_manifest(manifest)))
    selection = policy.select_setup_cache(root, policy.POLICY)
    events = []
    async def deployed(*args, **kwargs):
        assert kwargs["setup_cache_selection"] == selection
        events.append("deploy")
    async def boundary(*args, **kwargs):
        events.append("boundary")
    async def advised(*args, **kwargs):
        assert events == ["deploy", "boundary"]
        assert kwargs["expected"] == selection
        assert kwargs["boundary_output"] == tmp_path / "logs/worker-boundary"
        events.append("advice")
        return {"completed": True, "admission_authority": False}
    monkeypatch.setattr(harbor, "deploy_supervisor", deployed)
    monkeypatch.setattr(harbor, "deploy_worker_boundary", boundary)
    monkeypatch.setattr(policy, "apply_setup_cache_advice", advised)
    agent = SimpleNamespace(runtime_archive=root, auth_json=auth, arm="full",
        resource_profile=policy.PROFILE, setup_cache_selection=selection, logs_dir=tmp_path / "logs")
    asyncio.run(harbor.FullSupervisorAgent.setup(agent, object()))
    assert events == ["deploy", "boundary", "advice"]
    assert agent.setup_cache_receipt["admission_authority"] is False


def test_prepared_harbor_kwargs_preserve_exact_selection_and_budget(selected_cache, tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import config_for
    root, _, selection, _ = selected_cache
    config = config_for(tmp_path, tmp_path / "trial", root, "full",
                        resource_profile=policy.PROFILE, setup_cache_selection=selection)
    agent = config["agents"][0]
    assert agent["kwargs"]["setup_cache_selection"] == selection
    assert agent["override_timeout_sec"] == 300 and agent["override_setup_timeout_sec"] == 1800
    selection["manifest_sha256"] = "f" * 64
    assert agent["kwargs"]["setup_cache_selection"] != selection


@pytest.mark.parametrize("uid,gid", [(1000, 0), (0, 1000), (1000, 1000)])
def test_archive_owner_must_match_root_owned_tar_population(uid, gid):
    info = SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_nlink=1, st_size=17, st_uid=0, st_gid=0)
    row = dict(bytes=17, mode=0o644)
    archive_files.require_file(info, row)
    info.st_uid, info.st_gid = uid, gid
    with pytest.raises(ValueError):
        archive_files.require_file(info, row)


@pytest.mark.parametrize("mutation", ["extension", "wheel_pin", "wheel_version", "wheel_architecture", "library_pin", "library_path", "extra_library"])
def test_native_library_policy_refuses_source_or_population_drift(selected_cache, mutation):
    _, manifest, _, _ = selected_cache
    if mutation == "extension":
        next(row for row in manifest["files"] if row["path"].startswith("extensions/"))["sha256"] = "f" * 64
    elif mutation == "wheel_pin":
        manifest["torch_cpu_wheel"]["sha256"] = "f" * 64
        next(row for row in manifest["files"] if row["path"].startswith("runtime-wheels/"))["sha256"] = "f" * 64
    elif mutation == "wheel_version":
        manifest["torch_cpu_wheel"]["requirement"] = "torch==2.14.0+cpu"
    elif mutation == "wheel_architecture":
        manifest["torch_cpu_wheel"]["path"] = manifest["torch_cpu_wheel"]["path"].replace("aarch64", "x86_64")
        row = next(row for row in manifest["files"] if row["path"].startswith("runtime-wheels/"))
        row["path"] = row["path"].replace("aarch64", "x86_64")
    elif mutation == "library_pin":
        manifest["setup_cache"]["native_libraries"]["rows"][3]["sha256"] = "f" * 64
    elif mutation == "library_path":
        manifest["setup_cache"]["native_libraries"]["rows"][3]["path"] = "home/.codex/auth.json"
    else:
        manifest["setup_cache"]["native_libraries"]["rows"].append(manifest["setup_cache"]["native_libraries"]["rows"][0])
    with pytest.raises(ValueError):
        policy.validate_manifest_binding(manifest)


def test_library_binding_preserves_exact_rows_and_origin(selected_cache):
    _, manifest, _, _ = selected_cache
    binding = policy.library_binding_for_manifest(manifest)
    assert len(binding["rows"]) == 131
    assert sum(row["expected_bytes"] for row in binding["rows"]) == 565702269
    assert [row["source_kind"] for row in binding["rows"]] == ["archive_member"] * 3 + ["wheel_member"] * 128
    assert binding["wheel_sha256"] == libraries.WHEEL_SHA256
    binding["rows"][0]["sha256"] = "f" * 64
    assert libraries.ROWS[0]["sha256"] != "f" * 64


def test_fixed_payload_identity_and_nonexecutable_mode_population(selected_cache):
    # Golden identity comes from the separately reviewed wheel extraction and
    # actual 131-body Docker check; no installed tree is discovered here.
    encoded = json.dumps(libraries.ROWS, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == "3cfe981126d86bd3f3e7730e6125d9aeb9c7b7a891a2e1edc4d641e5b15171c7"
    payloads = libraries.ROWS[3:]
    assert payloads == sorted(payloads, key=lambda row: (-row["expected_bytes"], row["path"]))
    assert {row["mode"] for row in payloads} == {0o644, 0o755}
    assert all(row["uid"] == 0 and row["role"] == "installed_torch_wheel_payload" for row in payloads)
    assert sum(row["expected_bytes"] for row in payloads) == 483988539
    binding = policy.library_binding_for_manifest(selected_cache[1])
    assert binding["source_pins_sha256"] == libraries.SOURCE_PINS_SHA256
    assert binding["source_selection"] == libraries.SOURCE_SELECTION
    assert binding["installer_mode_policy"] == libraries.INSTALLER_MODE_POLICY


@pytest.mark.parametrize("field,value", [
    ("installer_mode_policy", "trust_zip_mode"), ("source_selection", "scan_installed_tree"),
    ("source_pins_sha256", "0" * 64),
])
def test_payload_provenance_substitution_refused(selected_cache, field, value):
    root, manifest, _, _ = selected_cache
    manifest["setup_cache"]["native_libraries"][field] = value
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        policy.select_setup_cache(root, policy.POLICY)


@pytest.fixture
def tiny_payload_population(tmp_path):
    root = tmp_path / "payloads"
    root.mkdir()
    rows = []
    for number in range(131):
        raw = ("authored public payload %d" % number).encode()
        name = "%03d.bin" % number
        mode = 0o644 if number % 2 else 0o755
        path = root / name
        path.write_bytes(raw)
        path.chmod(mode)
        rows.append(dict(path=name, name=name, role="authored_payload", uid=os.getuid(),
            mode=mode, expected_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return root, rows


def test_131_finite_population_hashes_every_body_before_first_hint(tiny_payload_population, monkeypatch):
    root, rows = tiny_payload_population
    original_read = native.os.read
    reads, hints = [], []
    def read(fd, amount):
        assert not hints, "body read after first advice"
        raw = original_read(fd, amount)
        if raw:
            reads.append(raw)
        return raw
    def advise(fd, *args):
        assert len(reads) == 131
        hints.append(fd)
    monkeypatch.setattr(native.os, "read", read)
    monkeypatch.setattr(native.os, "posix_fadvise", advise)
    result = native.advise_selected(root, rows, expected_count=131)
    assert len(hints) == 131
    assert result["body_read_bytes"] == sum(row["expected_bytes"] for row in rows)
    assert result["body_reads_after_advice"] == 0
    assert result["metadata_unchanged"] is True
    assert result["freed_bytes_claimed"] is False


@pytest.mark.parametrize("scope", ["archive", "archive_v2", "codex", "codex_v2", "libraries"])
@pytest.mark.parametrize("syscall", ["fdatasync", "posix_fadvise"])
def test_registered_advice_alarm_propagates_and_cleans_up(
        tiny_payload_population, monkeypatch, capsys, scope, syscall):
    """Invoke each real main's registered handler at the syscall boundary.

    The clock is injected; no real process-wide alarm or 60 second wait is
    required. Selection uses tiny public fixtures, while the advice loops,
    handler, exception identity, descriptor cleanup and timer cleanup are real.
    """
    current_policy = scope.endswith("_v2")
    scope = scope.removesuffix("_v2")
    root, rows = tiny_payload_population
    module = {"archive": archive_files, "codex": native, "libraries": libraries}[scope]
    registered, timers, calls, expired_errors = [], [], [], []
    opened = set()
    monkeypatch.setattr(module, "ROOT", str(root))
    monkeypatch.setattr(module.signal, "signal", lambda sig, handler: registered.append((sig, handler)))
    monkeypatch.setattr(module.signal, "setitimer", lambda timer, seconds: timers.append((timer, seconds)))
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(policy.platform, "machine", lambda: "aarch64")
    original_root_fd = archive_files.root_fd
    def tracked_root(path):
        fd = original_root_fd(path)
        opened.add(fd)
        return fd
    monkeypatch.setattr(archive_files if scope == "archive" else native, "root_fd", tracked_root)
    open_owner = archive_files if scope == "archive" else native
    open_name = "open_file" if scope == "archive" else "open_binary"
    original_open = getattr(open_owner, open_name)
    def tracked_open(*args):
        fd = original_open(*args)
        opened.add(fd)
        return fd
    monkeypatch.setattr(open_owner, open_name, tracked_open)
    if scope == "archive":
        (root / "source").mkdir()
        inventory = []
        for row in rows[:2]:
            (root / row["path"]).rename(root / "source" / row["path"])
            inventory.append(dict(path="source/" + row["path"], bytes=row["expected_bytes"],
                                  mode=row["mode"], sha256=row["sha256"]))
        manifest = dict(archive_sha256="a" * 64,
            setup_cache={"policy": policy.POLICY_V2 if current_policy else policy.POLICY}, files=inventory)
        raw = json.dumps(manifest).encode()
        manifest_path = root / "setup-cache-manifest.json"
        manifest_path.write_bytes(raw)
        monkeypatch.setattr(sys, "argv", ["archive", str(manifest_path), hashlib.sha256(raw).hexdigest(), "a" * 64])
        original_require = archive_files.require_file
        def fixture_owner(info, row):
            # The production archive is root-owned; these temporary test files
            # belong to the test runner. Preserve all other metadata checks.
            return original_require(SimpleNamespace(st_mode=info.st_mode, st_nlink=info.st_nlink,
                st_size=info.st_size, st_uid=0, st_gid=0), row)
        monkeypatch.setattr(archive_files, "require_file", fixture_owner)
    elif scope == "codex":
        monkeypatch.setattr(sys, "argv", ["codex", str(root / "codex-cache-exposure.json"), "a" * 64]
            + (["0.160.0"] if current_policy else []))
        monkeypatch.setattr(native, "read_receipt", lambda *args: {})
        def advise_codex(path, receipt, **kwargs):
            assert kwargs == ({"codex_version": "0.160.0"} if current_policy else {})
            return native.advise_selected(path, rows[:4], expected_count=4)
        monkeypatch.setattr(native, "advise", advise_codex)
    else:
        monkeypatch.setattr(libraries, "ROWS", rows)
    def invoke(name, fd):
        calls.append((name, fd))
        if name == syscall:
            assert len(registered) == 1 and registered[0][0] == module.signal.SIGALRM
            try:
                registered[0][1](module.signal.SIGALRM, None)
            except TimeoutError as error:
                expired_errors.append(error)
                raise
    monkeypatch.setattr(os, "fdatasync", lambda fd: invoke("fdatasync", fd))
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, *args: invoke("posix_fadvise", fd))
    with pytest.raises(TimeoutError) as raised:
        module.main()
    assert raised.value is expired_errors[0]
    assert len(expired_errors) == 1
    assert [name for name, _ in calls] == (["fdatasync"] if syscall == "fdatasync"
                                         else ["fdatasync", "posix_fadvise"])
    assert timers == [(module.signal.ITIMER_REAL, 60), (module.signal.ITIMER_REAL, 0)]
    assert capsys.readouterr().out == ""
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("mutation", ["last_digest", "last_mode", "last_size", "duplicate", "extra", "symlink", "hardlink", "population_bound"])
def test_131_population_late_fault_never_advises(tiny_payload_population, monkeypatch, mutation):
    root, rows = tiny_payload_population
    hints = []
    monkeypatch.setattr(native.os, "posix_fadvise", lambda *args: hints.append(args))
    if mutation == "last_digest":
        rows[-1]["sha256"] = "0" * 64
    elif mutation == "last_mode":
        (root / rows[-1]["path"]).chmod(0o664)
    elif mutation == "last_size":
        rows[-1]["expected_bytes"] += 1
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    elif mutation == "extra":
        rows.append(dict(rows[0]))
    elif mutation == "population_bound":
        monkeypatch.setattr(native, "MAX_TOTAL", sum(row["expected_bytes"] for row in rows) - 1)
    else:
        last = root / rows[-1]["path"]
        last.unlink()
        if mutation == "symlink":
            last.symlink_to(root / rows[-2]["path"])
        else:
            os.link(root / rows[-2]["path"], last)
    with pytest.raises((ValueError, OSError)):
        native.advise_selected(root, rows, expected_count=131)
    assert not hints


@pytest.mark.parametrize("primary_failure", [False, "value", "timeout"])
@pytest.mark.parametrize("cache_policy", [policy.POLICY, policy.POLICY_V2])
def test_receipt_write_failure_preserves_primary_or_refuses_success(
        selected_cache, tmp_path, monkeypatch, primary_failure, cache_policy):
    from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualify
    root, manifest, selection, _ = selected_cache
    version = policy.CODEX_VERSIONS[cache_policy]
    if cache_policy == policy.POLICY_V2:
        manifest["codex_version"] = version
        manifest["setup_cache"] = policy.binding_for_manifest(manifest, cache_policy)
        (root / "manifest.json").write_text(json.dumps(manifest))
        selection = policy.select_setup_cache(root, cache_policy)
    receipt = _native_receipt(version, native.PINS_BY_VERSION[version])
    boundary = tmp_path / "boundary"
    (boundary / "installation").mkdir(parents=True)
    raw = json.dumps(receipt).encode()
    (boundary / "installation/native-codex-binary.log").write_bytes(raw)
    archive_result = dict(schema="manifest-cache-advice@1", manifest_sha256=selection["manifest_sha256"],
        archive_sha256=manifest["archive_sha256"], selected_files=len(manifest["files"]),
        body_reads=0, metadata_unchanged=True, freed_bytes_claimed=False)
    codex_result = dict(schema="pinned-native-codex-cache-advice@1", codex_version=version,
        post_boundary_receipt_sha256=hashlib.sha256(raw).hexdigest(), selected_files=4, hashed_files=4,
        body_reads_after_advice=0, body_read_bytes=100, selected_bytes=100,
        metadata_unchanged=True, freed_bytes_claimed=False)
    count = sum(row["expected_bytes"] for row in libraries.ROWS)
    library_result = dict(schema="pinned-native-libraries-cache-advice@1", wheel_sha256=libraries.WHEEL_SHA256,
        torch_requirement=libraries.TORCH_REQUIREMENT, source_pins_sha256=libraries.SOURCE_PINS_SHA256,
        source_selection=libraries.SOURCE_SELECTION, installer_mode_policy=libraries.INSTALLER_MODE_POLICY, selected_files=131, hashed_files=131,
        selected_bytes=count, body_read_bytes=count, body_reads_after_advice=0,
        metadata_unchanged=True, freed_bytes_claimed=False, files=deepcopy(libraries.ROWS))
    primary = (TimeoutError("authored primary advice deadline") if primary_failure == "timeout"
               else ValueError("authored primary advice failure"))
    responses = [SimpleNamespace(return_code=0, stdout=json.dumps(value), stderr="")
                 for value in (archive_result, codex_result, library_result)]
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(
        side_effect=primary if primary_failure else responses))
    monkeypatch.setattr(qualify, "observe_resources", AsyncMock(return_value={"actual_profile": policy.PROFILE}))
    original = Path.write_text
    def failing_write(path, *args, **kwargs):
        if path.name == "setup-cache-advice.json":
            raise OSError("authored receipt storage failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", failing_write)
    with pytest.raises(type(primary) if primary_failure else OSError) as raised:
        asyncio.run(policy.apply_setup_cache_advice(env, archive_dir=root, expected=selection,
            boundary_output=boundary, output=tmp_path / "advice"))
    if primary_failure:
        assert raised.value is primary
        qualify.observe_resources.assert_not_called()
    else:
        assert "receipt storage failure" in str(raised.value)
        assert env.exec.await_count == 3
        native_command = env.exec.await_args_list[1].kwargs["command"]
        assert ("0.160.0" in native_command) == (cache_policy == policy.POLICY_V2)
        qualify.observe_resources.assert_awaited_once()


@pytest.mark.parametrize("advice_failure", [False, True])
def test_qualifier_runs_same_boundary_advice_order_and_preserves_cleanup(
        selected_cache, tmp_path, monkeypatch, advice_failure):
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as worker
    from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualify
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import SOURCE384_ENVIRONMENT
    import harbor.environments.docker.docker as docker
    root, manifest, _, auth = selected_cache
    with tarfile.open(root / "runtime.tar.gz", "w:gz"):
        pass
    manifest["archive_sha256"] = hashlib.sha256((root / "runtime.tar.gz").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    selection = policy.select_setup_cache(root, policy.POLICY)
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "instruction.md").write_text("Inspect public source.")
    (task / "task.toml").write_text("[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=10240\n")
    events, constructors = [], []
    async def event(name, value=None, **kwargs):
        events.append(name)
        return value
    env = SimpleNamespace(start=lambda **kw: event("start"), stop=lambda **kw: event("stop"))
    def constructor(**kwargs):
        constructors.append(kwargs)
        return env
    async def deployed(*args, **kwargs):
        assert kwargs["setup_cache_selection"] == selection
        return await event("deploy", dict(original_inputs=dict(files={"bottle.py": dict(sha256="b" * 64)})))
    async def boundary(*args, **kwargs):
        await event("boundary")
    async def advice(*args, **kwargs):
        assert kwargs["expected"] == selection
        assert events == ["start", "deploy", "boundary"]
        await event("advice")
        if advice_failure:
            raise ValueError("authored cache metadata refusal")
        return dict(completed=True, admission_authority=False)
    monkeypatch.setattr(docker, "DockerEnvironment", constructor)
    # This control exercises route ordering; the neighboring Torch suite owns
    # archive payload verification. The fixture declares pins without large bytes.
    monkeypatch.setattr(deployment, "verify_torch_cpu_wheel_archive",
                        lambda archive, value: deployment.validate_torch_cpu_wheel_binding(value))
    monkeypatch.setattr(deployment, "deploy_supervisor", deployed)
    monkeypatch.setattr(worker, "deploy_worker_boundary", boundary)
    monkeypatch.setattr(policy, "apply_setup_cache_advice", advice)
    monkeypatch.setattr(qualify, "observe_resources", lambda *a, **kw: event("resources", {}))
    monkeypatch.setattr(qualify, "qualify_context", lambda *a, **kw: event("context", dict(signed_source_hashes={"bottle.py": "b" * 64})))
    run = deployment.qualify_original_container(task_dir=task, archive_dir=root,
        output=tmp_path / "qualification", auth_json=auth, install_codex=True,
        resource_profile=policy.PROFILE, source384_context=True, setup_cache_selection=selection)
    if advice_failure:
        with pytest.raises(ValueError, match="authored cache metadata refusal"):
            asyncio.run(run)
        assert events == ["start", "deploy", "boundary", "advice", "stop"]
    else:
        assert asyncio.run(run)["qualified"] is True
        assert events == ["start", "deploy", "boundary", "advice", "resources", "context", "resources", "stop"]
    assert {k: constructors[0][k] for k in SOURCE384_ENVIRONMENT} == SOURCE384_ENVIRONMENT
