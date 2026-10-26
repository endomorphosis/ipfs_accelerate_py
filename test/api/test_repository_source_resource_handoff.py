"""Private MCP resource grants, with native cases borrowing real admission.

No case creates a scheduler, fakes native resource telemetry, or calls a model.
Authored router callbacks exercise transport only. The optional injected real
parent fixture is owned and released by the enclosing qualification wrapper.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import tomllib

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_resource_handoff as handoff
from test.api.test_repository_source_parent_admission import native_parent
from test.api.test_repository_source_router import route, _run


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    binding = policy.create_binding(workspace=root, requested_policy=policy.POLICIES[1])
    identity = policy.create_runtime_store(binding)
    try:
        yield binding
    finally:
        usage = policy.finish_runtime_store(binding, expected_identity=identity)
        assert usage["store_cleanup_complete"]


def _policy_digest(parent):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import policy_sha256
    return policy_sha256(parent._scheduler.config.persisted_dict())


def _install(binding, parent, *, deadline=None):
    return handoff.install_repository_source_resources(binding, parent_lease=parent,
        expected_policy_sha256=_policy_digest(parent),
        deadline_monotonic=time.monotonic() + 60 if deadline is None else deadline)


def _delegate(binding):
    return handoff.delegated_repository_source_resources(
        artifact=binding["resource_grant"]["path"], expected_sha256=binding["resource_grant"]["sha256"],
        config_sha256=binding["config_sha256"], workspace=Path(binding["workspace"]))


def test_invalid_owner_never_writes_capability(runtime):
    with pytest.raises(ValueError, match="invalid_repository_source_resource_grant"):
        handoff.install_repository_source_resources(runtime, parent_lease=object(),
            expected_policy_sha256="a" * 64, deadline_monotonic=time.monotonic() + 30)
    assert "resource_grant" not in runtime
    assert not (Path(runtime["invocation_dir"]) / handoff.NAME).exists()


@pytest.mark.parametrize("reference", [None, {}, {"path": "/tmp/elsewhere", "sha256": "a" * 64},
    {"path": "/tmp/anything", "sha256": True}, {"path": "/tmp/anything", "sha256": "a" * 63},
    {"path": "/tmp/anything", "sha256": "a" * 64, "parent_token": {}}])
def test_public_reference_is_closed_and_invocation_bound(runtime, reference):
    altered = copy.deepcopy(runtime)
    altered["resource_grant"] = reference
    with pytest.raises(ValueError):
        policy.validate_binding(altered)


def test_public_reference_hash_and_storage_checked_before_command(runtime):
    path = Path(runtime["invocation_dir"]) / handoff.NAME
    path.write_bytes(b"authored non-capability bytes")
    path.chmod(0o400)
    runtime["resource_grant"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Public path/digest validation does not assert native capability validity.
    args = policy.native_config_arguments(runtime)
    table = tomllib.loads(args[1])["mcp_servers"]["repository_source"]
    assert table["args"][-4:] == ["--resource-grant", str(path), "--resource-grant-sha256",
                                  runtime["resource_grant"]["sha256"]]
    path.chmod(0o600)
    with pytest.raises(ValueError):
        policy.native_config_arguments(runtime)


@pytest.mark.parametrize("change", ["symlink", "hardlink", "digest", "mode"])
def test_bad_private_storage_never_reaches_scheduler(runtime, monkeypatch, change):
    path = Path(runtime["invocation_dir"]) / handoff.NAME
    raw = b"{}"
    path.write_bytes(raw)
    path.chmod(0o400)
    runtime["resource_grant"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if change == "symlink":
        target = path.with_name("aliased.json")
        path.rename(target)
        path.symlink_to(target)
    elif change == "hardlink":
        os.link(path, path.with_name("alias.json"))
    elif change == "mode":
        path.chmod(0o600)
    else:
        runtime["resource_grant"]["sha256"] = "0" * 64
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import existing_resource_scheduler
    def forbidden(*args, **kwargs):
        pytest.fail("bad private storage reached the real scheduler")
    monkeypatch.setattr(existing_resource_scheduler, "open_existing_scheduler", forbidden)
    with pytest.raises((OSError, ValueError)):
        with _delegate(runtime):
            pytest.fail("bad grant admitted")


@pytest.mark.parametrize("controls", [
    {"repository_source_parent_lease": object()},
    {"repository_source_scheduler_policy_sha256": "a" * 64},
    {"repository_source_deadline_monotonic": 1.0},
    {"repository_source_parent_lease": object(), "repository_source_scheduler_policy_sha256": "a" * 64,
     "repository_source_deadline_monotonic": 1.0},
])
def test_incomplete_or_untyped_router_controls_never_dispatch(route, controls):
    with pytest.raises(ValueError):
        _run(repository_source_policy=policy.POLICIES[1], **controls)
    assert not route[1]["seen"]


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), -1.0,
    pytest.param(10**400, id="positive-huge-int"), pytest.param(-(10**400), id="negative-huge-int")])
def test_invalid_deadline_never_writes(runtime, native_parent, deadline):
    with pytest.raises(ValueError):
        _install(runtime, native_parent, deadline=deadline)
    assert "resource_grant" not in runtime
    assert not (Path(runtime["invocation_dir"]) / handoff.NAME).exists()


def test_native_private_grant_preserves_borrowed_parent_and_never_exports_token(runtime, native_parent):
    instruction = policy.validate_binding(runtime)
    reference = _install(runtime, native_parent)
    path = Path(reference["path"])
    assert path.stat().st_mode & 0o777 == 0o400
    assert set(reference) == {"path", "sha256"}
    assert policy.validate_binding(runtime, materialized=True) == instruction
    public = json.dumps(runtime)
    assert "parent_token" not in public and "lease_key" not in public
    assert "resource-grant" not in instruction
    with _delegate(runtime) as phase:
        borrowed = phase.native
        assert borrowed.parent_lease_id == native_parent.lease_id
        assert borrowed.owner_pid == os.getpid()
        assert borrowed.memory_mb == 512 and borrowed.child_process_slots == 1
        controls = phase.native_options()
        assert controls["parent_lease"] is borrowed
        assert not controls["cancel_event"].is_set()
        assert controls["deadline_monotonic"] > time.monotonic()
    assert borrowed.released
    assert not native_parent.released and not native_parent.cancelled


def test_native_wrong_existing_policy_refuses_without_new_state(runtime, native_parent):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import ExistingResourceSchedulerError
    with pytest.raises(ExistingResourceSchedulerError):
        handoff.install_repository_source_resources(runtime, parent_lease=native_parent,
            expected_policy_sha256="0" * 64, deadline_monotonic=time.monotonic() + 30)
    assert "resource_grant" not in runtime
    assert not (Path(runtime["invocation_dir"]) / handoff.NAME).exists()
    assert not native_parent.released and not native_parent.cancelled


@pytest.mark.parametrize("field,value", [("config_sha256", "f" * 64), ("workspace", "/unrelated"),
    ("owner_pid", True), ("owner_birth_marker", "0"), ("boot_id", "foreign-boot"),
    ("demand", {"cpu_slots": True, "memory_mb": 512, "child_process_slots": 1}),
    ("authority", {"resource_capacity_only": True, "execution_authority": True,
                   "proof_authority": False, "completion_authority": False}),
    ("deadline_monotonic", True), ("helper_sha256", "0" * 64),
    pytest.param("deadline_monotonic", 10**400, id="positive-huge-int"),
    pytest.param("deadline_monotonic", -(10**400), id="negative-huge-int")])
def test_native_tampered_private_binding_refuses_before_delegation(runtime, native_parent, field, value):
    _install(runtime, native_parent)
    path = Path(runtime["resource_grant"]["path"])
    private = json.loads(path.read_bytes())
    private[field] = value
    raw = handoff._wire(private)
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o400)
    runtime["resource_grant"]["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises((OSError, ValueError)):
        with _delegate(runtime):
            pytest.fail("tampered grant admitted")
    assert not native_parent.released


def test_native_deadline_signal_does_not_release_live_helper(runtime, native_parent, monkeypatch):
    from types import SimpleNamespace
    _install(runtime, native_parent)
    with _delegate(runtime) as phase:
        signal = phase.native_options()["cancel_event"]
        monkeypatch.setattr(handoff, "time", SimpleNamespace(monotonic=lambda: phase.deadline_monotonic + 1))
        assert signal.is_set()
        assert not phase.native.released, "polling must not release capacity during a synchronous handler"
    assert not native_parent.released


def test_native_router_public_binding_private_grant_cleanup(route, native_parent):
    _workspace, state = route
    references = []
    def provider_boundary(binding):
        references.append(dict(binding["resource_grant"]))
        with _delegate(binding) as phase:
            assert phase.native.parent_lease_id == native_parent.lease_id
        assert not native_parent.released
    state["callback"] = provider_boundary
    output, receipt = _run(repository_source_policy=policy.POLICIES[1],
        repository_source_parent_lease=native_parent,
        repository_source_scheduler_policy_sha256=_policy_digest(native_parent),
        repository_source_deadline_monotonic=time.monotonic() + 60)
    assert output == "completed authored provider result"
    assert len(references) == 1 and not Path(references[0]["path"]).exists()
    public = json.dumps(receipt)
    assert "parent_token" not in public and "lease_key" not in public
    assert "resource-grant.json" not in state["seen"][0][0]
    assert receipt["repository_source_policy"]["usage"]["store_cleanup_complete"]
    assert not native_parent.released
