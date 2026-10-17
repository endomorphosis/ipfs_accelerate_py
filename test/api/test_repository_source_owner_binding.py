"""Public owner binding contracts; descriptor authority is an authored seam.

These cases create no native context, contact no owner/provider, and do not
qualify a cross-UID deployment. Native CLI execution is mocked where exercised.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import tomllib
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as binding_api


@pytest.fixture
def case(tmp_path, monkeypatch):
    implementation = tmp_path / "implementation"
    implementation.mkdir()
    original = Path(binding_api.__file__).parent
    for name in (*binding_api._SOURCES, "repository_source_owner_binding.py"):
        (implementation / name).write_bytes((original / name).read_bytes())
    monkeypatch.setattr(binding_api, "__file__", str(implementation / "repository_source_owner_binding.py"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor_path = tmp_path / "public-owner" / "descriptor.json"
    now = time.monotonic_ns()
    descriptor = dict(schema=binding_api.protocol.DESCRIPTOR_SCHEMA, session_id="1" * 32,
        owner_uid=1001, worker_uid=1002, owner_pid=1234, owner_start_time_ticks=100,
        workspace=str(workspace), boundary_manifest_sha256="a" * 64,
        socket_path=str(descriptor_path.with_name("source.sock")), socket_device=1, socket_inode=2,
        issued_monotonic_ns=now, expires_monotonic_ns=now + 60_000_000_000,
        max_requests=64, max_total_response_bytes=524288,
        proof_authority=False, execution_authority=False, completion_authority=False,
        representation="compact", config_sha256="b" * 64, helper_sha256="c" * 64, read_only=True)
    calls = []
    def load(path, digest, *, workspace):
        calls.append((path, digest, workspace))
        assert Path(path) == descriptor_path and Path(workspace) == Path(descriptor["workspace"])
        assert digest == hashlib.sha256(binding_api.protocol.canonical_json(descriptor)).hexdigest()
        return copy.deepcopy(descriptor)
    monkeypatch.setattr(binding_api.protocol, "load_descriptor", load)
    def create(policy="repository-source-compact@1"):
        return binding_api.create_binding(workspace=workspace, requested_policy=policy,
            descriptor_path=descriptor_path,
            descriptor_sha256=hashlib.sha256(binding_api.protocol.canonical_json(descriptor)).hexdigest())
    return SimpleNamespace(workspace=workspace, descriptor_path=descriptor_path, descriptor=descriptor,
        calls=calls, create=create, implementation=implementation)


def test_binding_is_public_and_creation_has_no_store(case):
    before = sorted(str(path) for path in case.workspace.parent.rglob("*"))
    value = case.create()
    after = sorted(str(path) for path in case.workspace.parent.rglob("*"))
    assert before == after
    assert not case.descriptor_path.exists()
    assert len(case.calls) == 2
    assert value["schema"] == binding_api.SCHEMA and value["read_only"] is True
    assert all(value[key] is False for key in binding_api._FALSE_FIELDS)
    assert not {"state_dir", "audit_store", "index_path", "resource_grant", "parent_lease",
                "invocation_dir", "config_path", "tokenizer"} & set(value)
    assert value["descriptor"] == case.descriptor
    assert value["config_sha256"] != value["descriptor_sha256"] != value["owner_config_sha256"]
    assert value["helper_sha256"] == value["helper_sources"][binding_api._SOURCES[0]]
    assert value["python_identity"]["resolved_path"] == str(Path(sys.executable).resolve())


def _forwarding_target(case):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_worker_receipts as receipts
    name = "repository_source_worker_receipts.py"
    (case.implementation / name).write_bytes(Path(receipts.__file__).read_bytes())
    return dict(schema=receipts.TARGET_SCHEMA, directory="/authored/worker-outbox", device=1, inode=2,
        uid=case.descriptor["worker_uid"], session_id=case.descriptor["session_id"],
        descriptor_sha256=hashlib.sha256(binding_api.protocol.canonical_json(case.descriptor)).hexdigest(),
        invocation_id="a" * 32)


def test_optional_forwarding_rebinds_only_launch_metadata_not_prompt(case):
    original = case.create()
    before = copy.deepcopy(original)
    target = _forwarding_target(case)
    rebound = binding_api.with_worker_forwarding(original, target)
    assert original == before and "worker_forwarding" not in original
    assert rebound["instruction_sha256"] == original["instruction_sha256"]
    assert rebound["instruction_bytes"] == original["instruction_bytes"]
    assert rebound["config_sha256"] != original["config_sha256"]
    assert rebound["worker_forwarding"] == rebound["config"]["worker_forwarding"] == target
    assert set(rebound["helper_sources"]) == set(binding_api._SOURCES) | {"repository_source_worker_receipts.py"}
    settings = tomllib.loads(binding_api.native_config_arguments(rebound)[1])["mcp_servers"]["repository_source"]
    assert settings["args"][-2:] == ["--worker-forwarding-target",
        binding_api.protocol.canonical_json(target).decode()]
    assert binding_api.validate_binding(rebound, materialized=True)
    target["directory"] = "/changed"
    assert rebound["worker_forwarding"]["directory"] == "/authored/worker-outbox"


@pytest.mark.parametrize("field,value", [("uid", 999999), ("session_id", "f" * 32),
    ("descriptor_sha256", "f" * 64), ("directory", "relative"), ("extra", "PRIVATE")])
def test_optional_forwarding_identity_and_closed_target_are_required(case, field, value):
    target = _forwarding_target(case)
    target[field] = value
    with pytest.raises(ValueError):
        binding_api.with_worker_forwarding(case.create(), target)


def test_forwarding_target_inside_checkout_is_rejected(case):
    target = _forwarding_target(case)
    target["directory"] = str(case.workspace / "outbox")
    with pytest.raises(ValueError):
        binding_api.with_worker_forwarding(case.create(), target)


def test_opt_in_forwarding_helper_source_is_pinned_without_opening_outbox(case):
    rebound = binding_api.with_worker_forwarding(case.create(), _forwarding_target(case))
    assert binding_api.validate_binding(rebound, materialized=True)
    (case.implementation / "repository_source_worker_receipts.py").write_text("changed helper")
    assert binding_api.validate_binding(rebound)  # Historical reconstruction.
    with pytest.raises(ValueError, match="implementation differs"):
        binding_api.validate_binding(rebound, materialized=True)


@pytest.mark.parametrize("policy,representation", [
    ("repository-source-raw@1", "original"), ("repository-source-compact@1", "compact")])
def test_fixed_read_only_isolated_command(case, policy, representation):
    case.descriptor["representation"] = representation
    value = case.create(policy)
    argv = binding_api.native_config_arguments(value)
    settings = tomllib.loads(argv[1])["mcp_servers"]["repository_source"]
    assert argv[::2] == ["-c", "-c"] and argv[3] == "mcp_optional_startup_grace_ms=0"
    assert settings["command"] == str(Path(sys.executable).absolute())
    assert settings["args"] == ["-I", "-B", value["helper_script"],
        binding_api.protocol.DESCRIPTOR_FLAG, str(case.descriptor_path),
        binding_api.protocol.DIGEST_FLAG, value["descriptor_sha256"], "--workspace", str(case.workspace)]
    assert settings["enabled_tools"] == sorted(binding_api.protocol.READ_ONLY_TOOL_NAMES)
    assert "repository_replace_symbol" not in settings["enabled_tools"]
    assert settings["env_vars"] == [] and settings["required"] is True
    assert settings["startup_timeout_sec"] == settings["tool_timeout_sec"] == 30
    assert len(case.calls) == 3
    instruction = binding_api.validate_binding(value)
    assert "repository_replace_symbol" not in instruction
    assert "tokenizer" not in instruction and "read-only" in instruction.lower()
    assert value["instruction_bytes"] == len(instruction.encode())
    assert value["instruction_sha256"] == hashlib.sha256(instruction.encode()).hexdigest()


@pytest.mark.parametrize("bad", ["legacy", "foreign", None, True, {}])
def test_unsupported_policy_refused_without_descriptor_access(case, bad):
    with pytest.raises(ValueError):
        case.create(bad)
    assert case.calls == []


def test_representation_mismatch_is_refused(case):
    with pytest.raises(ValueError, match="descriptor binding differs"):
        case.create("repository-source-raw@1")


@pytest.mark.parametrize("key,value", [
    ("schema", "other"), ("read_only", 1), ("completion_authority", True),
    ("proof_authority", True), ("execution_authority", True),
    ("requested_policy", "legacy"), ("representation", "original"),
    ("descriptor_sha256", "0" * 64), ("session_id", "0" * 32),
    ("owner_config_sha256", "0" * 64), ("owner_helper_sha256", "0" * 64),
    ("helper_sha256", "0" * 64), ("config_sha256", "0" * 64),
    ("instruction_sha256", "0" * 64), ("instruction_bytes", True),
    ("descriptor_path", "/tmp/../owner"), ("python_executable", "python"),
    ("helper_script", "/tmp/foreign.py"), ("state_dir", "/tmp/private"),
    ("resource_grant", {}), ("tokenizer", {}), ("extra", "injected"),
])
def test_binding_tampering_refused(case, key, value):
    altered = case.create()
    altered[key] = value
    with pytest.raises(ValueError):
        binding_api.validate_binding(altered)


@pytest.mark.parametrize("target", ["descriptor", "config", "python_identity", "helper_sources"])
def test_nested_closed_schemas(case, target):
    value = case.create()
    value[target]["extra"] = "injected"
    with pytest.raises(ValueError):
        binding_api.validate_binding(value)


def test_retired_metadata_reconstructs_without_claiming_live_authority(case, monkeypatch):
    value = case.create()
    instruction = binding_api.validate_binding(value)
    def retired(*args, **kwargs):
        raise ValueError("owner retired")
    monkeypatch.setattr(binding_api.protocol, "load_descriptor", retired)
    assert binding_api.validate_binding(value) == instruction
    with pytest.raises(ValueError, match="owner retired"):
        binding_api.validate_binding(value, materialized=True)
    with pytest.raises(ValueError, match="owner retired"):
        binding_api.native_config_arguments(value)


def test_expired_snapshot_is_historical_metadata_only(case, monkeypatch):
    case.descriptor.update(issued_monotonic_ns=1, expires_monotonic_ns=2)
    value = case.create()  # Authored loader seam; production loader checks time.
    monkeypatch.setattr(binding_api.protocol, "load_descriptor",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("session expired")))
    assert binding_api.validate_binding(value)
    with pytest.raises(ValueError, match="session expired"):
        binding_api.native_config_arguments(value)


@pytest.mark.parametrize("key", ["session_id", "boundary_manifest_sha256", "config_sha256",
                                  "representation", "expires_monotonic_ns"])
def test_materialized_use_requires_exact_live_snapshot(case, monkeypatch, key):
    value = case.create()
    changed = copy.deepcopy(case.descriptor)
    changed[key] = changed[key] + 1 if type(changed[key]) is int else "changed"
    monkeypatch.setattr(binding_api.protocol, "load_descriptor", lambda *a, **k: changed)
    with pytest.raises(ValueError, match="live owner descriptor differs"):
        binding_api.native_config_arguments(value)


@pytest.mark.parametrize("name", binding_api._SOURCES)
def test_every_fixed_sibling_is_pinned(case, name):
    value = case.create()
    assert binding_api.helper_sources_match(value)
    source = case.implementation / name
    source.write_bytes(source.read_bytes() + b"\n# changed fixed source\n")
    assert not binding_api.helper_sources_match(value)
    assert binding_api.validate_binding(value)  # Historical prompt survives source retirement.
    with pytest.raises(ValueError, match="runtime implementation differs"):
        binding_api.native_config_arguments(value)


def test_helper_symlink_is_not_followed(case):
    value = case.create()
    source = case.implementation / binding_api._SOURCES[0]
    original = source.with_suffix(".original")
    source.rename(original)
    source.symlink_to(original)
    assert not binding_api.helper_sources_match(value)
    with pytest.raises(ValueError, match="runtime implementation differs"):
        binding_api.native_config_arguments(value)


@pytest.mark.parametrize("key", ["python_executable", "helper_script", "python_identity"])
def test_rehashed_arbitrary_runtime_cannot_be_launched(case, key):
    value = case.create()
    if key == "python_identity":
        value[key]["sha256"] = "0" * 64
    else:
        value[key] = "/foreign/" + Path(value[key]).name
    value["config"] = binding_api._config(value)
    value["config_sha256"] = hashlib.sha256(binding_api.protocol.canonical_json(value["config"])).hexdigest()
    assert binding_api.validate_binding(value)  # Offline identity, never execution authority.
    with pytest.raises(ValueError, match="runtime implementation differs"):
        binding_api.native_config_arguments(value)


def test_workspace_mismatch_refused_before_live_access(case):
    value = case.create()
    calls = len(case.calls)
    with pytest.raises(ValueError, match="workspace differs"):
        binding_api.validate_binding(value, workspace=case.workspace.parent, materialized=True)
    assert len(case.calls) == calls


def test_relocated_historical_helper_hashes_match_without_launch_authority(case):
    value = case.create()
    instruction = binding_api.validate_binding(value)
    value["helper_script"] = "/retired-container/runtime/" + binding_api._SOURCES[0]
    value["config"] = binding_api._config(value)
    value["config_sha256"] = hashlib.sha256(binding_api.protocol.canonical_json(value["config"])).hexdigest()
    calls = len(case.calls)
    assert binding_api.validate_binding(value) == instruction
    assert binding_api.helper_sources_match(value)
    with pytest.raises(ValueError, match="runtime implementation differs"):
        binding_api.native_config_arguments(value)
    assert len(case.calls) == calls


def test_owner_usage_remains_unknown_and_cannot_claim_zeros(case):
    value = case.create()
    usage = binding_api.unavailable_usage()
    assert usage["status"] == "unavailable" and usage["reason"] == "owner_audit_required"
    assert all(item is None for key, item in usage.items() if key not in {"status", "reason"})
    value["usage"] = usage
    assert binding_api.validate_binding(value)
    usage["tool_calls"] = 0
    with pytest.raises(ValueError, match="separate owner audit"):
        binding_api.validate_binding(value)
    assert binding_api.unavailable_usage()["tool_calls"] is None


def test_existing_command_builder_accepts_owner_binding_once(case):
    from ipfs_accelerate_py import llm_router
    value = case.create()
    command = llm_router.build_codex_cli_command(codex_repository_source_binding=value)
    assert command[-1] == "-"
    assert sum(arg.startswith("mcp_servers.repository_source=") for arg in command) == 1
    table = next(arg for arg in command if arg.startswith("mcp_servers.repository_source="))
    assert tomllib.loads(table)["mcp_servers"]["repository_source"]["args"][:2] == ["-I", "-B"]


def test_mock_provider_observes_launch_identity_separately_from_owner_config(case, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    value = case.create()
    monkeypatch.chdir(case.workspace)
    monkeypatch.setattr(llm_router.shutil, "which", lambda name: "/test/codex")
    seen = []
    def execute(command, **kwargs):
        seen.append(command)
        Path(command[command.index("--output-last-message") + 1]).write_text("done")
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed"}', stderr="")
    monkeypatch.setattr(llm_router, "_run_codex_repository_command", execute)
    assert llm_router._get_codex_cli_provider().generate("literal task", codex_repository_source_binding=value) == "done"
    assert len(seen) == 1
    observation = get_last_cli_observation("codex_cli")
    assert observation["codex_repository_source_configured"] is True
    assert observation["codex_repository_source_config_sha256"] == value["config_sha256"]
    assert observation["codex_repository_source_config_sha256"] != value["owner_config_sha256"]
    assert "compact_reads" not in observation and "token_savings_measured" not in observation
