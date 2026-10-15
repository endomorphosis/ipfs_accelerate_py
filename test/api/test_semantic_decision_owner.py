"""Owner policy materialization and immutable publication without providers."""

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_owner as owner
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_worker import load_decision_launch_policy

pytest_plugins = ("test.api.test_semantic_router_integration",)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _nomination(path, policy):
    path.write_bytes(_encoded(owner.build_decision_nomination(policy)))
    return path, _sha(path.read_bytes())


def _arguments(repository, nomination, digest, task="task"):
    return ["--model", "pinned", "--semantic-repository", str(repository),
            "--semantic-decision-nomination-artifact", str(nomination),
            "--semantic-decision-nomination-sha256", digest,
            "--semantic-decision-task-id", task]


@pytest.fixture
def light(tmp_path, monkeypatch):
    repository, workspace = tmp_path / "source", tmp_path / "candidate"
    repository.mkdir(mode=0o755)
    workspace.mkdir()
    policy = SemanticDecisionPolicy(required_capsule_ids=("required",), optional_capsule_ids=("optional",))
    nomination, digest = _nomination(tmp_path / "nomination.json", policy)
    prompt = b'{"objective_id":"task"}\nLITERAL native suffix \xe2\x9c\x93\n'
    calls = []

    def parts(text):
        wire, end = json.JSONDecoder().raw_decode(text)
        return wire, text[end:], (), (), "retained parent", ".runtime/semantic/worker-context.json"

    def build(**kwargs):
        calls.append(kwargs)
        return {"schema": "supervisor-semantic-decision-launch-policy@1", "task_id": "task",
                "parent_artifact": ".runtime/semantic/worker-context.json",
                "parent_sha256": _sha(b"retained parent"), "policy": kwargs["policy"].to_dict()}

    monkeypatch.setattr(owner.codec, "_prompt_parts", parts)
    monkeypatch.setattr(owner, "build_decision_launch_policy", build)
    return repository, workspace, policy, nomination, digest, prompt, calls


def _prepare(light, arguments=None, prompt=None):
    repository, workspace, _, nomination, digest, original, _ = light
    return owner.prepare_decision_owner_dispatch(
        arguments=arguments if arguments is not None else _arguments(repository, nomination, digest),
        prompt_bytes=original if prompt is None else prompt, workspace=workspace,
    )


def test_owner_materializes_exact_prompt_binding_and_preserves_unowned_arguments(light):
    repository, workspace, policy, nomination, digest, prompt, calls = light
    arguments = _arguments(repository, nomination, digest)
    extras = ["--purpose", "coding", "--reasoning-effort", "high", "--timeout", "90",
              "--public-instruction-artifact", "/literal/public requirement.json",
              "--doctor-residual-task-cid", "literal-task", "--semantic-decision-max-calls", "2"]
    arguments[2:2] = extras
    result = _prepare(light, arguments=arguments)
    expected = ["--model", "pinned", *extras, "--semantic-repository", str(repository)]
    assert list(result.worker_arguments[:-4]) == expected
    assert result.worker_arguments[-4::2] == ("--semantic-decision-policy-artifact", "--semantic-decision-policy-sha256")
    assert result.prompt_bytes == prompt
    assert calls == [{"native_prompt": prompt.decode(), "policy": policy, "repository": repository}]
    receipt = result.receipt
    published = Path(receipt["launch_policy_artifact"])
    assert not published.is_relative_to(workspace)
    assert published.parent == repository / ".runtime/semantic-decision-policies"
    assert published.name == receipt["launch_policy_sha256"] + ".json"
    assert _sha(published.read_bytes()) == receipt["launch_policy_sha256"]
    assert stat.S_IMODE(published.stat().st_mode) == 0o444 and published.stat().st_nlink == 1
    assert stat.S_IMODE(published.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(published.parent.parent.stat().st_mode) == 0o755
    assert receipt["nomination_sha256"] == digest and receipt["nomination_artifact"] == str(nomination)
    assert receipt["native_prompt_sha256"] == _sha(prompt) and receipt["native_prompt_bytes"] == len(prompt)
    assert receipt["policy_sha256"] == _sha(_encoded(policy.to_dict()))
    assert receipt["provider_called"] is False and receipt["completion_authority"] is False
    receipt["task_id"] = "changed copy"
    assert result.receipt["task_id"] == "task"
    with pytest.raises(FrozenInstanceError):
        result.prompt_bytes = b"changed"


def test_owner_reuses_only_exact_immutable_descriptor_with_fresh_binding_receipt(light):
    first = _prepare(light)
    artifact = Path(first.receipt["launch_policy_artifact"])
    identity = artifact.stat().st_ino, artifact.stat().st_mtime_ns
    second = _prepare(light)
    assert second.worker_arguments == first.worker_arguments
    assert (artifact.stat().st_ino, artifact.stat().st_mtime_ns) == identity
    assert second.receipt["binding_id"] != first.receipt["binding_id"]
    assert list(artifact.parent.iterdir()) == [artifact]


@pytest.mark.parametrize("alteration", ["bytes", "mode", "hardlink", "symlink"])
def test_owner_refuses_mutated_or_aliased_immutable_descriptor(light, alteration):
    result = _prepare(light)
    artifact = Path(result.receipt["launch_policy_artifact"])
    if alteration == "bytes":
        artifact.chmod(0o644)
        artifact.write_bytes(b"x" * artifact.stat().st_size)
        artifact.chmod(0o444)
    elif alteration == "mode":
        artifact.chmod(0o644)
    elif alteration == "hardlink":
        os.link(artifact, artifact.with_name("alias.json"))
    else:
        original = artifact.with_name("original.json")
        artifact.rename(original)
        artifact.symlink_to(original)
    with pytest.raises((ValueError, OSError)):
        _prepare(light)


@pytest.mark.parametrize("flag", sorted(owner._OWNER_FLAGS))
@pytest.mark.parametrize("alteration", ["equals", "duplicate", "missing", "no_value"])
def test_owner_flags_are_complete_unique_literal_two_token_bindings(light, flag, alteration):
    repository, _, _, nomination, digest, _, calls = light
    arguments = _arguments(repository, nomination, digest)
    index = arguments.index(flag)
    if alteration == "equals":
        arguments[index:index + 2] = [flag + "=" + arguments[index + 1]]
    elif alteration == "duplicate":
        arguments += arguments[index:index + 2]
    elif alteration == "missing":
        del arguments[index:index + 2]
    else:
        arguments = arguments[:index + 1]
    with pytest.raises(ValueError):
        _prepare(light, arguments=arguments)
    assert calls == []


@pytest.mark.parametrize("flags", [
    ["--preflight"], ["--doctor-candidate-artifact", "/candidate"],
    ["--doctor-contract-sha256=hash"], ["--finite-repository-task-cid", "task"],
    ["--finite-proof-query-context", "/context"], ["--semantic-context-projection"],
    ["--semantic-slot-policy-revision", "legacy@1"],
    ["--semantic-decision-policy-artifact", "/prebound"],
    ["--semantic-decision-policy-sha256=hash"], ["--purpose", "planning"],
    ["--purpose=coding"], ["--purpose", "coding", "--purpose", "coding"],
])
def test_owner_refuses_other_modes_or_prebound_policy(light, flags):
    repository, _, _, nomination, digest, _, calls = light
    with pytest.raises(ValueError):
        _prepare(light, arguments=_arguments(repository, nomination, digest) + flags)
    assert calls == []


@pytest.mark.parametrize("task", ["foreign", " ", "x" * 513])
def test_owner_requires_exact_bounded_signed_task_key(light, task):
    repository, _, _, nomination, digest, _, calls = light
    with pytest.raises(ValueError):
        _prepare(light, arguments=_arguments(repository, nomination, digest, task=task))
    assert calls == []


@pytest.mark.parametrize("prompt", [b"", b" \n", b"\xff", b"x" * 256001, "not bytes"],
                         ids=["empty", "whitespace", "invalid-utf8", "over-byte-bound", "wrong-type"])
def test_owner_refuses_invalid_native_stdin_before_builder_or_publication(light, prompt):
    with pytest.raises((ValueError, UnicodeError)):
        _prepare(light, prompt=prompt)
    assert light[-1] == []
    assert not (light[0] / ".runtime/semantic-decision-policies").exists()


@pytest.mark.parametrize("alteration", ["schema", "extra", "policy_extra", "duplicate", "nested_duplicate", "over_bound", "digest"])
def test_nomination_is_closed_bounded_and_independently_hash_bound(light, alteration):
    _, workspace, policy, artifact, digest, _, calls = light
    value = owner.build_decision_nomination(policy)
    if alteration == "schema":
        value["schema"] = "unknown@1"
    elif alteration == "extra":
        value["task_id"] = "static-task"
    elif alteration == "policy_extra":
        value["policy"]["infer_optional"] = True
    raw = _encoded(value)
    if alteration == "duplicate":
        raw = b'{"schema":"duplicate",' + raw[1:]
    elif alteration == "nested_duplicate":
        raw = raw.replace(b'"policy":{', b'"policy":{"optional_capsule_ids":[],', 1)
    elif alteration == "over_bound":
        raw = b" " * (owner.MAX_NOMINATION_BYTES + 1)
    artifact.write_bytes(raw)
    if alteration != "digest":
        digest = _sha(raw)
    else:
        artifact.write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        owner.load_decision_nomination(artifact=artifact, expected_sha256=digest, workspace=workspace)
    assert calls == []


@pytest.mark.parametrize("location", ["candidate", "symlink", "directory"])
def test_nomination_path_is_canonical_regular_and_outside_candidate(light, location):
    _, workspace, _, original, digest, _, calls = light
    if location == "candidate":
        artifact = workspace / "nomination.json"
        artifact.write_bytes(original.read_bytes())
    elif location == "symlink":
        artifact = original.with_name("nomination-link.json")
        artifact.symlink_to(original)
    else:
        artifact = original.with_name("nomination-directory")
        artifact.mkdir()
    with pytest.raises(ValueError):
        owner.load_decision_nomination(artifact=artifact, expected_sha256=digest, workspace=workspace)
    assert calls == []


@pytest.mark.parametrize("alteration", ["repository_writable", "runtime_symlink", "runtime_writable", "policy_symlink", "policy_writable", "runtime_private"])
def test_policy_publication_refuses_unsafe_existing_directories(light, alteration):
    repository = light[0]
    runtime = repository / ".runtime"
    other = repository.parent / "foreign"
    other.mkdir(mode=0o755)
    if alteration == "repository_writable":
        repository.chmod(0o775)
    elif alteration == "runtime_symlink":
        runtime.symlink_to(other, target_is_directory=True)
    else:
        runtime.mkdir(mode=0o755)
        policy_directory = runtime / "semantic-decision-policies"
        if alteration == "runtime_writable":
            runtime.chmod(0o777)
        elif alteration == "runtime_private":
            runtime.chmod(0o700)
        elif alteration == "policy_symlink":
            policy_directory.symlink_to(other, target_is_directory=True)
        else:
            policy_directory.mkdir(mode=0o755)
            policy_directory.chmod(0o777)
    with pytest.raises((ValueError, OSError)):
        _prepare(light)
    assert list(other.iterdir()) == []


def test_owner_fsync_failure_leaves_no_published_or_pending_policy(light, monkeypatch):
    repository = light[0]
    directory = repository / ".runtime/semantic-decision-policies"
    directory.parent.mkdir(mode=0o755)
    directory.mkdir(mode=0o755)

    def fail(descriptor):
        raise OSError("authored sync refusal")

    monkeypatch.setattr(owner.os, "fsync", fail)
    with pytest.raises(OSError, match="sync refusal"):
        _prepare(light)
    assert list(directory.iterdir()) == []


def test_owner_unicode_policy_commitment_matches_shared_codec_canonical_bytes(light):
    policy = SemanticDecisionPolicy(required_capsule_ids=("required-\u03bb",), optional_capsule_ids=("optional-\u96ea",))
    nomination, digest = _nomination(light[3], policy)
    result = _prepare(light, arguments=_arguments(light[0], nomination, digest))
    assert result.receipt["policy_sha256"] == _sha(owner.codec._json(policy.to_dict()).encode())
    assert result.receipt["policy_sha256"] == _sha(_encoded(policy.to_dict()))


def _real_arguments(allocated):
    repository, workspace, prompt, _ = allocated
    # These directories belong to this authored fixture, whose creation may
    # otherwise inherit the host's group-writable umask. Production checks
    # remain strict and never change existing repository permissions.
    repository.chmod(0o755)
    (repository / ".runtime").chmod(0o755)
    parent = json.loads((repository / ".runtime/semantic/worker-context.json").read_text())
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids[1:], optional_capsule_ids=ids[:1])
    nomination, digest = _nomination(repository.parent / "real-nomination.json", policy)
    task = owner.codec._prompt_parts(prompt)[0]["objective_id"]
    return repository, workspace, prompt, policy, _arguments(repository, nomination, digest, task)


def test_owner_real_native_producer_policy_reaches_existing_worker_loader(allocated):
    repository, workspace, prompt, policy, arguments = _real_arguments(allocated)
    result = owner.prepare_decision_owner_dispatch(arguments=arguments, prompt_bytes=prompt.encode(), workspace=workspace)
    receipt = result.receipt
    loaded = load_decision_launch_policy(
        artifact=Path(receipt["launch_policy_artifact"]), expected_sha256=receipt["launch_policy_sha256"],
        native_prompt=prompt, repository=repository, workspace=workspace,
    )
    assert loaded == policy and result.prompt_bytes == prompt.encode()
    assert receipt["parent_sha256"] == _sha((repository / receipt["parent_artifact"]).read_bytes())
    assert receipt["task_id"] == owner.codec._prompt_parts(prompt)[0]["objective_id"]


def test_owner_real_native_producer_refuses_stale_source_before_publication(allocated):
    repository, workspace, prompt, _, arguments = _real_arguments(allocated)
    (repository / "mod.py").write_text(allocated[3] + "\n# changed canonical source\n")
    with pytest.raises(ValueError, match="stale"):
        owner.prepare_decision_owner_dispatch(arguments=arguments, prompt_bytes=prompt.encode(), workspace=workspace)
    assert not (repository / ".runtime/semantic-decision-policies").exists()


def test_concurrent_same_policy_publication_waits_for_immutable_single_link(light, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    linked, release_first = threading.Event(), threading.Event()
    second_lock_requested, second_lock_acquired = threading.Event(), threading.Event()
    role = threading.local()
    actual_link, actual_flock = owner.os.link, owner.fcntl.flock

    def link(*args, **kwargs):
        result = actual_link(*args, **kwargs)
        if getattr(role, "name", None) == "first":
            linked.set()
            if not release_first.wait(10):
                raise AssertionError("first publisher was not released")
        return result

    def flock(*args, **kwargs):
        lock_info = os.fstat(args[0])
        assert stat.S_ISREG(lock_info.st_mode) and lock_info.st_uid == os.geteuid()
        assert stat.S_IMODE(lock_info.st_mode) == 0o600
        assert lock_info.st_nlink == 1 and lock_info.st_size == 0
        second = getattr(role, "name", None) == "second"
        if second:
            second_lock_requested.set()
        result = actual_flock(*args, **kwargs)
        if second:
            second_lock_acquired.set()
        return result

    def publish(name):
        role.name = name
        return _prepare(light)

    monkeypatch.setattr(owner.os, "link", link)
    monkeypatch.setattr(owner.fcntl, "flock", flock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(publish, "first")
        try:
            assert linked.wait(5), "first publisher never created its final hardlink"
            directory = light[0] / ".runtime/semantic-decision-policies"
            final = next(directory.glob("*.json"))
            assert final.stat().st_nlink == 2
            second = pool.submit(publish, "second")
            assert second_lock_requested.wait(5), "second publisher never requested serialization"
            assert not second_lock_acquired.is_set() and not second.done()
        finally:
            release_first.set()
        results = first.result(timeout=10), second.result(timeout=10)
    assert second_lock_acquired.is_set()
    first_receipt, second_receipt = (result.receipt for result in results)
    assert first_receipt["launch_policy_artifact"] == second_receipt["launch_policy_artifact"]
    assert first_receipt["launch_policy_sha256"] == second_receipt["launch_policy_sha256"]
    assert results[0].worker_arguments == results[1].worker_arguments
    assert results[0].prompt_bytes == results[1].prompt_bytes == light[5]
    final = Path(first_receipt["launch_policy_artifact"])
    assert _sha(final.read_bytes()) == first_receipt["launch_policy_sha256"]
    assert stat.S_IMODE(final.stat().st_mode) == 0o444 and final.stat().st_nlink == 1
    assert not list(final.parent.glob(".pending-*"))
    lock = final.parent.parent / ".semantic-decision-publish.lock"
    lock_info = lock.stat()
    assert stat.S_ISREG(lock_info.st_mode) and lock_info.st_uid == os.geteuid()
    assert stat.S_IMODE(lock_info.st_mode) == 0o600
    assert lock_info.st_nlink == 1 and lock_info.st_size == 0


def test_owner_pending_cleanup_failure_always_releases_lock_and_directory_fds(light, monkeypatch):
    import errno

    captured = {}
    actual_directory, actual_child_directory = owner._directory, owner._child_directory
    actual_open, actual_unlink, actual_close = owner.os.open, owner.os.unlink, owner.os.close

    def directory(path):
        descriptor = actual_directory(path)
        captured["root"] = descriptor
        return descriptor

    def child_directory(parent, name):
        descriptor = actual_child_directory(parent, name)
        captured["runtime" if name == ".runtime" else "policy"] = descriptor
        return descriptor

    def open_file(path, *args, **kwargs):
        descriptor = actual_open(path, *args, **kwargs)
        if path == ".semantic-decision-publish.lock":
            captured["lock"] = descriptor
        return descriptor

    def link(*args, **kwargs):
        raise OSError("authored primary publication failure")

    def unlink(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(".pending-"):
            raise OSError("authored pending cleanup failure")
        return actual_unlink(path, *args, **kwargs)

    monkeypatch.setattr(owner, "_directory", directory)
    monkeypatch.setattr(owner, "_child_directory", child_directory)
    monkeypatch.setattr(owner.os, "open", open_file)
    monkeypatch.setattr(owner.os, "link", link)
    monkeypatch.setattr(owner.os, "unlink", unlink)
    try:
        with pytest.raises(OSError, match="pending cleanup failure"):
            _prepare(light)
        assert set(captured) == {"root", "runtime", "policy", "lock"}
        for descriptor in captured.values():
            with pytest.raises(OSError) as closed:
                os.fstat(descriptor)
            assert closed.value.errno == errno.EBADF
        lock = light[0] / ".runtime/.semantic-decision-publish.lock"
        probe = actual_open(lock, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            owner.fcntl.flock(probe, owner.fcntl.LOCK_EX | owner.fcntl.LOCK_NB)
        finally:
            actual_close(probe)
    finally:
        # A failing implementation must not retain descriptors or locks for
        # subsequent tests in the same sealed qualification process.
        for descriptor in captured.values():
            try:
                actual_close(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
