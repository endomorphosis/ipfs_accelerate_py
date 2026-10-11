"""Dispatch guard contracts; owner inference/replay is exercised separately."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    build_text_context_references, context_provider_input_payload,
)
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(payload):
    text = guard._json(payload)
    references = build_text_context_references(text, reference_prefix="source-formalization",
        kind=guard.KIND, path=payload["context_bundle"]["artifact"],
        repository_id="repository:authored", tree_id="tree:authored", required=True, chunk_bytes=6144)
    return context_provider_input_payload(repository_id="repository:authored", tree_id="tree:authored",
        objective_id=payload["task_id"], objective_revision="revision:1", policy_id="policy:authored",
        policy_revision="revision:1", caller="coding-worker", stage="implementation",
        goal="Change the authored function", authority={}, scope={}, acceptance={}, evidence=references)


@pytest.fixture
def authored(tmp_path, monkeypatch):
    # Real isolated Git repositories establish the common-directory contract.
    # Only the already-validated owner selection is controlled in these unit tests.
    root, workspace = tmp_path / "repository", tmp_path / "allocated"
    root.mkdir()
    source = "def authored(value):\n    return value + 1\n".encode()
    (root / "source.py").write_bytes(source)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Authored Test",
        "-c", "user.email=authored@example.invalid", "commit", "-qm", "authored source"], check=True)
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    hashes = {"source.py": _sha(source)}
    contexts = []
    for kind in ("source384-candidate-context", "numerical-formalization-context"):
        summary = {"schema": "authored-owner-summary@1", "proof_authority": False,
                   "text": "λ" * 1700}
        contexts.append(dict(kind=kind,
            receipt_reference=dict(schema="authored-owner-reference@1", output=str(tmp_path / kind),
                                   receipt_sha256="a" * 64, receipt_bytes=200),
            source_hashes_sha256=_sha(guard._json(hashes).encode()),
            summary_sha256=_sha(guard._json(summary).encode()), summary=summary))
    payload = dict(schema=guard.WORKER_EVIDENCE_SCHEMA, repository=str(root),
        task_cid="task:authored", task_id="authored-task",
        context_bundle={"artifact": ".runtime/context.json", "sha256": "b" * 64}, contexts=contexts,
        proof_authority=False, formalization_authority=False, execution_authority=False,
        completion_authority=False, planning_authority=False, dispatch_authority=False,
        source_semantics_verified=False, whole_program_semantics_verified=False, source_obligations_discharged=0)
    selected = dict(metadata={}, worker_evidence=deepcopy(payload),
        source384_context={"source_hashes": dict(hashes)}, numerical_formalization={"source_hashes": dict(hashes)})
    calls = []
    def selection(**kwargs):
        calls.append(kwargs)
        bounded = {k: v for k, v in kwargs.items() if k != "source384_timeout_seconds"}
        if "source384_timeout_seconds" in kwargs:
            assert 0 < kwargs["source384_timeout_seconds"] <= 90
        assert bounded == dict(repository=root, artifact=payload["context_bundle"]["artifact"],
            expected_sha256=payload["context_bundle"]["sha256"], task_cid=payload["task_cid"], task_id=payload["task_id"])
        return deepcopy(selected)
    monkeypatch.setattr(guard, "load_task_context_selection", selection)
    return dict(root=root, workspace=workspace, payload=payload, selected=selected, calls=calls)


def _invoke(case, wire=None, **kwargs):
    return guard.validate_source_formalization_context(prompt=guard._json(wire or _wire(case["payload"])),
        repository=kwargs.get("repository", case["root"]), workspace=kwargs.get("workspace", case["workspace"]))


def test_real_allocated_git_relation_and_exact_multichunk_owner_evidence(authored):
    result = _invoke(authored)
    assert result["chunks"] >= 2
    assert result["source_files"] == 1
    assert result["context_kinds"] == ["source384-candidate-context", "numerical-formalization-context"]
    assert result["shared_git_repository_verified"] and result["workspace_source_current"]
    assert result["worker_evidence_sha256"] == _sha(guard._json(authored["payload"]).encode())
    assert result["source_obligations_discharged"] == 0 and result["proof_authority"] is False
    assert result["neural_inference_replayed"] is False and len(authored["calls"]) == 1


@pytest.mark.parametrize("change", ["required", "priority", "chunk_index", "chunk_count",
    "artifact_content_id", "coverage_ids", "byte_count", "referenced_content_id", "path", "order", "omitted", "extra"])
def test_rehashed_chunk_metadata_or_missing_chunks_cannot_bypass_exact_native_binding(authored, change):
    wire = json.loads(guard._json(_wire(authored["payload"])))
    row = wire["evidence"][0]
    if change in {"required", "priority", "chunk_index", "chunk_count", "artifact_content_id", "coverage_ids"}:
        row["metadata"][change] = {"required": False, "priority": 1, "chunk_index": 1,
            "chunk_count": 1, "artifact_content_id": "sha256:" + "0" * 64, "coverage_ids": ["foreign"]}[change]
    elif change == "order":
        wire["evidence"].reverse()
    elif change == "omitted":
        wire["evidence"].pop()
    elif change == "extra":
        row["metadata"]["unknown"] = "ignored?"
    else:
        row[change] = {"byte_count": 1, "referenced_content_id": "sha256:" + "0" * 64,
                       "path": ".runtime/foreign.json"}[change]
    with pytest.raises(ValueError):
        _invoke(authored, wire)
    assert not authored["calls"]


@pytest.mark.parametrize("change", ["foreign_objective", "noncanonical", "duplicate", "escaped_duplicate", "missing_repository"])
def test_native_prompt_and_objective_binding(authored, change):
    wire = json.loads(guard._json(_wire(authored["payload"])))
    if change == "foreign_objective":
        wire["objective_id"] = "another-task"
    prompt = guard._json(wire)
    if change == "noncanonical":
        prompt = json.dumps(wire, indent=2)
    elif change in {"duplicate", "escaped_duplicate"}:
        prompt = '{"objective_id":"another-task",' + prompt[1:]
        if change == "escaped_duplicate":
            prompt = prompt.replace(guard.KIND, "source-\\u0066ormalization-context")
    with pytest.raises(ValueError):
        guard.validate_source_formalization_context(prompt=prompt,
            repository=None if change == "missing_repository" else authored["root"], workspace=authored["workspace"])
    assert not authored["calls"]


@pytest.mark.parametrize("change", ["candidate", "authority", "task_cid", "bundle_sha", "bundle_path"])
def test_reconstructed_but_foreign_payload_is_not_current_owner_evidence(authored, change):
    payload = deepcopy(authored["payload"])
    if change == "candidate":
        payload["contexts"][0]["summary"]["candidate_ir_json"] = '{"operator":"forged"}'
    elif change == "authority":
        payload["proof_authority"] = True
    elif change == "task_cid":
        payload["task_cid"] = "foreign:task"
    elif change == "bundle_sha":
        payload["context_bundle"]["sha256"] = "c" * 64
    else:
        payload["context_bundle"]["artifact"] = "../outside.json"
    # A real owner would refuse task/digest rebinding. Keep that boundary
    # controlled here and exercise the guard's independent payload equality.
    if change in {"task_cid", "bundle_sha"}:
        def refuse(**kwargs):
            raise ValueError("owner task/bundle mismatch")
        authored["calls"].clear()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(guard, "load_task_context_selection", refuse)
    else:
        monkeypatch = None
    try:
        with pytest.raises(ValueError):
            _invoke(authored, _wire(payload))
    finally:
        if monkeypatch:
            monkeypatch.undo()


@pytest.mark.parametrize("change", ["stale", "symlink", "parent_symlink", "foreign_git", "same_root"])
def test_allocated_workspace_source_and_git_relation(authored, tmp_path, change):
    workspace = authored["workspace"]
    if change == "stale":
        (workspace / "source.py").write_text("different source\n")
    elif change == "symlink":
        (workspace / "source.py").unlink()
        (workspace / "source.py").symlink_to(authored["root"] / "source.py")
    elif change == "parent_symlink":
        (workspace / "alias").symlink_to(authored["root"], target_is_directory=True)
        for key in ("source384_context", "numerical_formalization"):
            authored["selected"][key]["source_hashes"] = {"alias/source.py": _sha((authored["root"] / "source.py").read_bytes())}
    elif change == "foreign_git":
        workspace = tmp_path / "foreign"
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    else:
        workspace = authored["root"]
    with pytest.raises((ValueError, OSError)):
        _invoke(authored, workspace=workspace)


def test_retry_trailer_is_preserved_and_validation_is_pre_dispatch_only(authored):
    prompt = guard._json(_wire(authored["payload"])) + "\nRetry after an earlier attempt."
    result = guard.validate_source_formalization_context(prompt=prompt,
        repository=authored["root"], workspace=authored["workspace"])
    (authored["workspace"] / "source.py").write_text("expected provider edit\n")
    assert result["validation_phase"] == "before_provider"
    with pytest.raises(ValueError, match="stale"):
        guard.validate_source_formalization_context(prompt=prompt,
            repository=authored["root"], workspace=authored["workspace"])


@pytest.mark.parametrize("prompt", ["legacy coding prompt", '{"objective_id":"legacy","evidence":[]}'])
def test_absent_advice_has_no_repository_or_owner_activity(prompt, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("absent advice must not inspect owner or Git")
    monkeypatch.setattr(guard, "_workspace", unexpected)
    monkeypatch.setattr(guard, "load_task_context_selection", unexpected)
    assert guard.validate_source_formalization_context(prompt=prompt, repository=None, workspace=None) is None


@pytest.fixture
def sealed(authored, tmp_path, monkeypatch):
    # Unit tests run as one UID. Real UID1000/1001 denial and protected /opt
    # ancestors are qualified separately in the deployment container.
    deployment = tmp_path / "deployment"
    parent = deployment / "worker-observations"
    parent.mkdir(parents=True, mode=0o750)
    parent.chmod(0o750)
    boundary = deployment / "container-boundary.json"
    boundary.write_text("authored boundary placeholder")
    digest = _sha(boundary.read_bytes())
    identity = {"owner_uid": os.getuid(), "worker_uid": os.getuid()+1}
    monkeypatch.setattr(guard, "_owner_boundary", lambda path: (identity, digest))
    monkeypatch.setattr(guard, "_worker_boundary", lambda *args: identity)
    def controlled_directory(path, *, owner, group, worker=False):
        assert path == parent and owner == group == os.getuid()
        assert path.stat().st_mode & 0o777 == 0o750
    monkeypatch.setattr(guard, "_protected_directory", controlled_directory)
    prompt = guard._json(_wire(authored["payload"]))
    original = os.memfd_create("authored-input")
    os.write(original, prompt.encode())
    os.lseek(original, 0, os.SEEK_SET)
    try:
        prepared = guard.prepare_container_source_formalization_dispatch(
            argv=["--semantic-repository", str(authored["root"])], boundary=boundary,
            workspace=authored["workspace"], input_fd=original)
    finally:
        os.close(original)
    args = prepared["argv"]
    request = dict(prompt=prompt, repository=authored["root"], workspace=authored["workspace"],
        boundary=boundary, boundary_sha256=digest,
        handoff=Path(args[args.index(guard.HANDOFF_FLAGS[0])+1]),
        handoff_sha256=args[args.index(guard.HANDOFF_FLAGS[1])+1])
    try:
        yield authored, prepared, request
    finally:
        os.close(prepared["stdin_fd"])


def test_owner_seals_exact_stdin_and_worker_never_reads_private_owner_state(sealed, monkeypatch):
    import fcntl
    case, prepared, request = sealed
    assert len(case["calls"]) == 1
    assert os.read(prepared["stdin_fd"], 2*1024*1024) == request["prompt"].encode()
    seals = fcntl.fcntl(prepared["stdin_fd"], fcntl.F_GET_SEALS)
    assert seals & fcntl.F_SEAL_WRITE and seals & fcntl.F_SEAL_GROW and seals & fcntl.F_SEAL_SHRINK
    with pytest.raises(OSError):
        os.write(prepared["stdin_fd"], b"tamper")
    def private_owner(*args, **kwargs):
        raise AssertionError("unprivileged worker must not read owner indexes")
    monkeypatch.setattr(guard, "load_task_context_selection", private_owner)
    result = guard.validate_source_formalization_context(**request)
    assert result["owner_validation_inherited"] is True
    assert result["owner_private_state_replayed_here"] is False
    assert result["canonical_source_current"] and result["workspace_source_current"]
    assert result["proof_authority"] is False
    assert request["handoff"].stat().st_mode & 0o777 == 0o440


@pytest.mark.parametrize("change", ["stripped_prompt", "retry_prompt", "missing_handoff", "wrong_digest",
    "writable_file", "symlink", "canonical_source", "workspace_source", "expired", "future", "foreign_workspace"])
def test_worker_seal_refuses_rebinding_and_source_drift(sealed, change):
    case, _, original = sealed
    request = dict(original)
    if change == "stripped_prompt":
        request["prompt"] = "legacy prompt without required advice"
    elif change == "retry_prompt":
        request["prompt"] += "\nadditional retry text"
    elif change == "missing_handoff":
        request["handoff"] = request["handoff_sha256"] = None
    elif change == "wrong_digest":
        request["handoff_sha256"] = "0" * 64
    elif change == "writable_file":
        request["handoff"].chmod(0o640)
    elif change == "symlink":
        alias = request["handoff"].with_name("formalization-" + "a"*32 + ".json")
        alias.symlink_to(request["handoff"])
        request["handoff"] = alias
    elif change in {"canonical_source", "workspace_source"}:
        root = case["root"] if change == "canonical_source" else case["workspace"]
        (root / "source.py").write_text("changed after owner validation\n")
    else:
        payload = json.loads(request["handoff"].read_bytes())
        if change in {"expired", "future"}:
            delta = -2*guard.HANDOFF_LIFETIME_NS if change == "expired" else 2*guard.HANDOFF_LIFETIME_NS
            payload["issued_monotonic_ns"] += delta
            payload["expires_monotonic_ns"] += delta
        else:
            payload["workspace"] = str(case["root"])
        request["handoff"].chmod(0o640)
        raw = guard._json(payload).encode()
        request["handoff"].write_bytes(raw)
        request["handoff"].chmod(0o440)
        request["handoff_sha256"] = _sha(raw)
    with pytest.raises((ValueError, OSError)):
        guard.validate_source_formalization_context(**request)


def test_no_handoff_without_boundary_and_caller_cannot_supply_owner_arguments(authored):
    with pytest.raises(ValueError, match="container boundary"):
        guard.validate_source_formalization_context(prompt=guard._json(_wire(authored["payload"])),
            repository=authored["root"], workspace=authored["workspace"], handoff="/forged")
    for flag in guard.HANDOFF_FLAGS:
        with pytest.raises(ValueError, match="independently"):
            guard.prepare_container_source_formalization_dispatch(argv=[flag, "forged"],
                boundary=None, workspace=None, input_fd=-1)


def test_legacy_owner_capture_preserves_stdin_without_private_state(monkeypatch):
    raw = b"ordinary no-index coding request\n"
    fd = os.memfd_create("authored-legacy")
    os.write(fd, raw)
    os.lseek(fd, 0, os.SEEK_SET)
    def unexpected(*args, **kwargs):
        raise AssertionError("legacy capture must not inspect private state")
    monkeypatch.setattr(guard, "_owner_boundary", unexpected)
    monkeypatch.setattr(guard, "load_task_context_selection", unexpected)
    try:
        result = guard.prepare_container_source_formalization_dispatch(argv=["--provider", "grok_cli"],
            boundary=None, workspace=None, input_fd=fd)
    finally:
        os.close(fd)
    try:
        assert result["argv"] == ["--provider", "grok_cli"]
        assert os.read(result["stdin_fd"], len(raw)+1) == raw
    finally:
        os.close(result["stdin_fd"])


def test_owner_capture_rejects_oversize_and_wait_timeout(monkeypatch):
    import select
    fd = os.memfd_create("authored-oversize")
    os.write(fd, b"x" * (2*1024*1024+1))
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        with pytest.raises(ValueError, match="owner prompt exceeds"):
            guard.prepare_container_source_formalization_dispatch(argv=[], boundary=None, workspace=None, input_fd=fd)
        monkeypatch.setattr(select, "select", lambda *args: ([], [], []))
        with pytest.raises(TimeoutError, match="capture deadline"):
            guard.prepare_container_source_formalization_dispatch(argv=[], boundary=None, workspace=None, input_fd=fd)
    finally:
        os.close(fd)


def test_seal_must_remain_unexpired_through_source_validation(sealed, monkeypatch):
    _, _, request = sealed
    body = json.loads(request["handoff"].read_bytes())
    times = iter((body["issued_monotonic_ns"], body["expires_monotonic_ns"] + 1))
    monkeypatch.setattr(guard.time, "monotonic_ns", lambda: next(times))
    with pytest.raises(ValueError, match="expired during"):
        guard.validate_source_formalization_context(**request)
