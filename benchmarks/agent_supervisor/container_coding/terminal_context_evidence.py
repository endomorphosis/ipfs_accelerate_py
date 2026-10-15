"""Closed historical evidence capture; source identities stay in their namespace.

This exports public admitted inputs, not a private state directory. The replay
container must mount the verified repository/state trees read-only at their
original names. No loader identity check or current-source check is relaxed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zlib

SCHEMA = "terminal-context-evidence-export@1"
RECEIPT_SCHEMA = "terminal-context-evidence-export-receipt@1"
PROFILE = "public-bottle-owner-packet-replay@1"
MAX_FILES = 20_000
MAX_CONTENT_BYTES = 256 * 1024 * 1024
MAX_FILE_BYTES = 2_000_000
MAX_SNAPSHOT_BYTES = 64_000_000
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 65_536
MAX_ARCHIVE_BYTES = MAX_CONTENT_BYTES + MAX_FILES * 1024 + MAX_MANIFEST_BYTES + 10_240
MAX_PUBLIC_SOURCES = 512
MAX_CAPSULES = 32
MAX_RECEIPTS = 16
_COLLECTOR_GROUP_MARKER = "IPFS_SUPERVISOR_CONTEXT_EVIDENCE_GROUP"
_IN_OWNED_COLLECTOR_GROUP = False
_GIT_TIMEOUT_SECONDS = 10.0
_CHILD_CLEANUP_SECONDS = .2


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(items):
        value = {}
        for key, child in items:
            _require(key not in value, "duplicate_json_key")
            value[key] = child
        return value
    def constant(value):
        raise ValueError("nonfinite_json")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _require(depth <= 40 and count <= max(1, len(raw)), "json_structure_bound")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            _require(math.isfinite(item), "nonfinite_json")
    return value


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _relative(value):
    _require(type(value) is str and 0 < len(value.encode()) <= 4096,
             "bounded_relative_path_required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and str(path) == value
             and all(part not in {"", ".", ".."} for part in value.split("/"))
             and not any(c in value for c in ("\0", "\n", "\r", "\\", ":")),
             "canonical_relative_path_required")
    return value


def _absolute(value):
    value = str(value)
    path = Path(value)
    _require(path.is_absolute() and str(path) == value and value != "/"
             and all(part not in {".", ".."} for part in path.parts)
             and not any(c in value for c in ("\0", "\n", "\r", "\\")),
             "canonical_absolute_path_required")
    return path


def _read(path, limit=MAX_FILE_BYTES):
    """Bounded, single-link read through no-follow directory handles."""
    path = _absolute(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    file_fd = None
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        before = os.fstat(file_fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                 and 0 <= before.st_size <= limit, "regular_single_link_input_required")
        with os.fdopen(file_fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(file_fd)
        current = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        def identity(row):
            return tuple(getattr(row, name) for name in
                         ("st_dev", "st_ino", "st_size", "st_mode", "st_nlink", "st_mtime_ns", "st_ctime_ns"))
        _require(len(raw) == before.st_size and identity(before) == identity(after) == identity(current),
                 "input_changed_during_read")
        return raw
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(fd)


def _artifact(repository, value):
    """Only independently held runtime artifacts may nominate repository reads."""
    path = Path(value)
    if path.is_absolute():
        _require(path.is_relative_to(repository), "artifact_outside_repository")
        value = path.relative_to(repository).as_posix()
    value = _relative(str(value))
    _require(PurePosixPath(value).parts[0] == ".runtime", "runtime_artifact_required")
    return repository / value, "repository/" + value


def _cleanup_child(process, *, owns_group):
    """Signal only our new group, or the exact direct child, then reap it."""
    def send(number):
        try:
            if owns_group:
                os.killpg(process.pid, number)
            elif process.poll() is None:
                process.send_signal(number)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    try:
        process.wait(timeout=_CHILD_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # A descendant may retain stdout after the direct child exits. The group
    # belongs to this launch only; never signal an inherited caller group.
    send(signal.SIGKILL)
    process.wait(timeout=_CHILD_CLEANUP_SECONDS)


def _bounded_child(command, *, timeout, limit, environment, input_bytes=None,
                   owns_group=True):
    """Bound pipe input/output and elapsed time without a blocking read/write."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, start_new_session=owns_group)
    deadline = time.monotonic() + timeout
    result = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            if input_bytes is not None:
                os.set_blocking(process.stdin.fileno(), False)
                if input_bytes:
                    selector.register(process.stdin, selectors.EVENT_WRITE)
                else:
                    process.stdin.close()
            offset = 0
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(remaining):
                    if key.fileobj is process.stdout:
                        block = os.read(process.stdout.fileno(), min(65_536, limit - len(result) + 1))
                        if not block:
                            selector.unregister(process.stdout)
                        else:
                            result.extend(block)
                            _require(len(result) <= limit, "child_output_bound")
                    else:
                        try:
                            offset += os.write(process.stdin.fileno(), input_bytes[offset:offset + 65_536])
                        except BrokenPipeError:
                            offset = len(input_bytes)
                        if offset == len(input_bytes):
                            selector.unregister(process.stdin)
                            process.stdin.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            returncode = process.wait(timeout=remaining)
            return returncode, bytes(result)
    finally:
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()
        _cleanup_child(process, owns_group=owns_group)


def _git(root, *args, limit=MAX_FILE_BYTES):
    """Read exact public Git objects; no hooks, replacements, or lazy fetching."""
    environment = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
                   "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
                   "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1",
                   "GIT_OPTIONAL_LOCKS": "0"}
    command = ["git", "--no-optional-locks", "--literal-pathspecs", "-c", "credential.helper=",
               "-c", "protocol.allow=never", "-c", "core.hooksPath=/dev/null",
               "-c", "core.fsmonitor=false", "-C", str(root), *args]
    try:
        returncode, raw = _bounded_child(command, timeout=_GIT_TIMEOUT_SECONDS, limit=limit,
            environment=environment, owns_group=not _IN_OWNED_COLLECTOR_GROUP)
    except ValueError as error:
        if str(error) == "child_output_bound":
            raise ValueError("git_output_bound") from None
        raise
    _require(returncode == 0, "git_object_unavailable")
    return raw


def _git_object(root, oid, expected_type, held):
    _require(type(oid) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid),
             "git_identity_required")
    member = "repository/.git/objects/" + oid[:2] + "/" + oid[2:]
    if member in held:
        envelope = zlib.decompress(held[member])
        header, raw = envelope.split(b"\0", 1)
        _require(header.split(b" ", 1)[0].decode() == expected_type, "git_object_type_differs")
        return raw
    kind = _git(root, "cat-file", "-t", oid, limit=32).strip().decode()
    _require(kind == expected_type, "git_object_type_differs")
    size = int(_git(root, "cat-file", "-s", oid, limit=32).strip())
    _require(0 <= size <= MAX_FILE_BYTES, "git_object_bound")
    raw = _git(root, "cat-file", kind, oid, limit=MAX_FILE_BYTES)
    _require(len(raw) == size, "git_object_size_differs")
    envelope = kind.encode() + b" " + str(size).encode() + b"\0" + raw
    digest = hashlib.sha1(envelope).hexdigest() if len(oid) == 40 else hashlib.sha256(envelope).hexdigest()
    _require(digest == oid, "git_object_identity_differs")
    held[member] = zlib.compress(envelope)
    return raw


def _git_public_closure(root, baseline, sources):
    """Commit and only path-traversal trees/blobs for the signed public inventory."""
    _require(type(sources) is dict and 1 <= len(sources) <= MAX_PUBLIC_SOURCES,
             "public_source_population_bound")
    held = {}
    commit = _git_object(root, baseline, "commit", held)
    tree = commit.split(b"\n", 1)[0]
    _require(tree.startswith(b"tree "), "git_commit_tree_required")
    root_tree = tree[5:].decode()
    trees = {"": root_tree}
    for path, binding in sorted(sources.items()):
        _relative(path)
        _require(PurePosixPath(path).parts[0] not in {".git", ".runtime", ".codex"}
                 and type(binding) is dict and re.fullmatch(r"[0-9a-f]{64}", binding.get("sha256", "")),
                 "signed_public_source_required")
        parts = PurePosixPath(path).parts
        for count in range(len(parts)):
            parent = "/".join(parts[:count])
            if parent not in trees:
                trees[parent] = _git(root, "rev-parse", baseline + ":" + parent, limit=128).strip().decode()
            _git_object(root, trees[parent], "tree", held)
        entry = _git(root, "ls-tree", "-z", baseline, "--", path, limit=8192)
        header, actual_path = entry.rstrip(b"\0").split(b"\t", 1)
        mode, kind, oid = header.decode().split(" ")
        _require(actual_path.decode() == path and kind == "blob" and mode in {"100644", "100755"},
                 "public_git_source_must_be_regular")
        _require("executable" not in binding or (type(binding["executable"]) is bool
                 and binding["executable"] == (mode == "100755")), "public_git_mode_differs")
        raw = _git_object(root, oid, "blob", held)
        _require(_sha(raw) == binding["sha256"], "signed_public_git_source_differs")
    held["repository/.git/HEAD"] = baseline.encode() + b"\n"
    return held


def _request(receipts, decision_report):
    from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
    _require(type(receipts) is list and len(receipts) <= MAX_RECEIPTS
             and type(decision_report) is dict, "bounded_export_request_required")
    _require(not decision_report.get("semantic_decision_typed_nomination_expected", False),
             "typed_lineage_export_not_supported")
    _require(not any(type(row) is dict and "world_task_packet" in row for row in receipts),
             "world_packet_export_not_supported")
    _require(not any(type(row) is dict and row.get("source_formalization_context") is not None
                     for row in receipts), "formalization_export_not_supported")
    rows = [{key: row[key] for key in (*audit.RECEIPT_FIELDS, *audit.TRANSLATION_FIELDS, "provider", "model")
             if key in row} for row in receipts if type(row) is dict and row.get("phase") == "coding"]
    selected = {key: decision_report[key] for key in audit.DECISION_REPORT_FIELDS if key in decision_report}
    value = {"receipts": rows, "decision_report": selected}
    audit._bounded_request(value)
    owners = selected.get("semantic_decision_owner_bindings", [])
    _require(type(owners) is list, "bounded_owner_binding_population_required")
    _require(not any(type(owner) is dict and "typed_nomination" in owner for owner in owners),
             "typed_lineage_export_not_supported")
    _require(len(_encode(value)) <= MAX_REQUEST_BYTES, "export_request_bound")
    # The existing bounded owner record contains hashes/counters, never reply text.
    def no_content(item):
        if type(item) is dict:
            _require(not (set(item) & {"prompt", "response", "response_text", "stdout", "stderr",
                                      "text", "content", "messages", "transcript", "auth_json",
                                      "api_key", "access_token", "password", "credentials"}),
                     "private_or_provider_content_in_request")
            for child in item.values():
                no_content(child)
        elif type(item) is list:
            for child in item:
                no_content(child)
    no_content(value)
    return value


def export_terminal_context_evidence(*, state: Path, receipts: list, workspace_root: Path,
                                     decision_report: dict) -> dict:
    """Create a fresh immutable sibling archive; do not copy the private state tree."""
    from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
    from benchmarks.agent_supervisor.container_coding import semantic_decision_input_audit as policy_audit
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
    from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import SemanticProjectionBindings
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_retrieval_query_reference import (
        SemanticRetrievalQueryBinding, _historical_material, validate_native_retrieval_references,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_nomination
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider

    state, allocation = _absolute(state), _absolute(workspace_root)
    _require(state.resolve(strict=True) == state and state.is_dir(), "canonical_state_required")
    _require(allocation == Path("/opt/ipfs-supervisor/worktrees"), "original_workspace_namespace_required")
    _require(state.is_relative_to(Path("/opt/ipfs-supervisor/state")), "owner_state_namespace_required")
    request = _request(receipts, decision_report)
    held, originals, roles = {}, {}, {}
    def capture(path, member, limit=MAX_FILE_BYTES, expected=None, role="semantic-cas"):
        _relative(member)
        raw = _read(path, limit)
        _require(expected is None or _sha(raw) == expected, "held_artifact_digest_differs")
        _require(member not in held or held[member] == raw, "export_path_binding_conflict")
        held[member] = raw
        _require(member not in roles or roles[member] == role, "export_role_conflict")
        roles[member] = role
        originals[str(path)] = (raw, limit)
        _require(len(held) <= MAX_FILES and sum(map(len, held.values())) <= MAX_CONTENT_BYTES,
                 "export_content_bound")
        return raw

    prepared = _json(capture(state / "prepared.json", "state/prepared.json", role="prepared-state"))
    expected = _json(capture(state / "context-result.json", "state/context-result.json", role="context-result"))
    repository = _absolute(prepared.get("repository", ""))
    _require(repository == Path("/app") and repository.resolve(strict=True) == repository,
             "original_repository_namespace_required")
    _require(set(prepared.get("worker_inputs", [])) ==
             {".supervisor-instruction.md", ".supervisor-public-smoke.py", "bottle.py"},
             "public_worker_source_scope_required")
    _require(prepared.get("official_verifier_in_context") is False, "verifier_context_refused")
    bundle = expected.get("context_bundle")
    _require(type(bundle) is dict and set(bundle) == {"artifact", "sha256"},
             "independently_pinned_task_bundle_required")
    bundle_path, bundle_member = _artifact(repository, bundle["artifact"])
    capture(bundle_path, bundle_member, expected=bundle["sha256"], role="task-context-bundle")
    nomination = load_task_context_nomination(repository=repository, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=expected["task_cid"],
        task_id=prepared["spec"]["task_key"])
    fd = audit._open_directory(state)
    try:
        paths = audit._capsule_paths(fd)
    finally:
        os.close(fd)
    _require(len(paths) <= MAX_CAPSULES, "capsule_population_bound")
    dialogues = request["decision_report"].get("semantic_decision_dialogues", [])
    _require(type(dialogues) is list and 0 < len(dialogues) <= MAX_RECEIPTS,
             "owner_packet_dialogue_required")
    wanted = {(d.get("worker_launch") or {}).get("original_native_prompt_sha256") for d in dialogues}
    native = {}
    for relative in paths:
        raw = _read(state / relative)
        observed, rendered, formalization = audit._capsule(raw, expected=expected, prepared=prepared)
        if observed["native_prompt_sha256"] not in wanted:
            continue
        _require(formalization is None and "source_formalization_context" not in observed,
                 "formalization_export_not_supported")
        _require(observed["all_context_checks_passed"], "native_capsule_binding_differs")
        capture(state / relative, "state/" + relative, expected=_sha(raw), role="native-capsule")
        native[observed["native_prompt_sha256"]] = rendered
    _require(set(native) == wanted, "matching_native_capsule_unavailable")
    public_manifest = None
    for dialogue in dialogues:
        launch = dialogue.get("worker_launch") or {}
        _require(launch.get("doctor_residual_context") is None
                 and dialogue.get("selected_representation") in {"native", "decision_packet"},
                 "unsupported_replay_profile")
        prompt = native[launch["original_native_prompt_sha256"]]
        policy_audit._policy(launch=launch, workspace=Path(dialogue["workspace"]),
                             native_prompt=prompt, repository=repository)
        policy_path, policy_member = _artifact(repository, launch["launch_policy_artifact"])
        descriptor = _json(capture(policy_path, policy_member, expected=launch["launch_policy_sha256"],
                                   role="launch-policy"))
        _require(descriptor.get("schema") in {"supervisor-semantic-decision-launch-policy@2",
                                              "supervisor-semantic-decision-launch-policy@3"},
                 "task_text_packet_launch_required")
        wire, _, _, _, text, artifact = codec._prompt_parts(prompt)
        parent_path, parent_member = _artifact(repository, artifact)
        parent = _json(capture(parent_path, parent_member, expected=descriptor["parent_sha256"],
                               role="semantic-parent"))
        _require(text.encode() == held[parent_member] and parent.get("task_id") == descriptor["task_id"],
                 "native_semantic_parent_differs")
        _require(set(parent.get("manifest", {})) == set(prepared["worker_inputs"]),
                 "semantic_parent_source_scope_differs")
        _require(nomination.get("semantic context artifact") == artifact
                 and nomination.get("semantic context sha256") == descriptor["parent_sha256"],
                 "task_bundle_semantic_parent_differs")
        blocks = PurePosixPath(artifact).parent / "blocks"
        def reader(cid):
            _require(type(cid) is str and PurePosixPath(cid).name == cid and cid not in {".", ".."},
                     "producer_block_identity_required")
            path, member = _artifact(repository, (blocks / cid).as_posix())
            return capture(path, member)
        view = IpfsDatasetsSemanticStateProvider().open_verified_view(parent["semantic_root_cid"], reader)
        view.artifact_fact("artifact:snapshot-evidence")
        for capsule in parent["capsules"]:
            _require(view.capsule(capsule["stable_symbol_id"]).to_dict() == capsule,
                     "producer_capsule_differs")
        for binding in parent["manifest"].values():
            _require(_sha(reader(binding["source_cid"])) == binding["sha256"],
                     "retained_source_digest_differs")
        binding = SemanticRetrievalQueryBinding(artifact=nomination.get("code retrieval artifact"),
            expected_sha256=nomination.get("code retrieval sha256"), task_id=descriptor["task_id"])
        if descriptor["schema"].endswith("@3"):
            _require(binding.to_dict() == descriptor["retrieval_query_binding"],
                     "task_bundle_retrieval_parent_differs")
        path, member = _artifact(repository, binding.artifact)
        raw = capture(path, member, expected=binding.expected_sha256, role="raw-retrieval")
        raw_payload = _json(raw)
        if "snapshot_ref" in raw_payload:
            _artifact(repository, raw_payload["snapshot_ref"]["path"])
        material = {**parent, "_bound_artifact": artifact}
        native_retrieval, dependencies = _historical_material(repository, binding, material, raw)
        validate_native_retrieval_references(wire=wire, binding=binding, native_text=native_retrieval)
        for relative, original, maximum in dependencies:
            path, member = _artifact(repository, relative)
            capture(path, member, maximum, expected=_sha(original),
                    role="retrieval-snapshot" if relative != binding.artifact else "raw-retrieval")
        retained = launch.get("public_instruction")
        _require(type(retained) is dict, "bound_public_instruction_required")
        path, member = _artifact(repository, retained["artifact"])
        payload = _json(capture(path, member, expected=retained["artifact_sha256"], role="public-instruction"))
        selected = instruction._selection(payload)
        _require(payload["repository"] == str(repository) and payload["manifest"] == prepared["manifest"],
                 "public_manifest_or_origin_differs")
        _require(public_manifest is None or public_manifest == selected, "public_baseline_conflict")
        public_manifest = selected
        instruction.load_public_instruction(artifact=path, expected_sha256=retained["artifact_sha256"],
            repository=repository, task_cid=retained["task_cid"], prompt=prompt, require_current_source=False)
    _require(public_manifest is not None, "authenticated_public_manifest_required")
    git = _git_public_closure(repository, public_manifest["baseline_commit"], public_manifest["sources"])
    held.update(git)
    roles.update({name: "synthetic-git-head" if name.endswith("/HEAD") else "public-git-object"
                  for name in git})
    held["replay-request.json"] = _encode(request)
    roles["replay-request.json"] = "replay-request"
    _require(len(held) <= MAX_FILES and sum(map(len, held.values())) <= MAX_CONTENT_BYTES,
             "export_content_bound")
    for path, (raw, limit) in originals.items():
        _require(_read(path, limit) == raw, "export_origin_changed")
    manifest = {"schema": SCHEMA, "profile": PROFILE,
                "repository_origin": str(repository), "state_origin": str(state),
                "workspace_root_origin": str(allocation),
                "native_prompt_bindings": sorted(wanted),
                "public_baseline_commit": public_manifest["baseline_commit"],
                "public_source_count": len(public_manifest["sources"]),
                "files": [{"path": name, "role": roles[name], "bytes": len(raw), "sha256": _sha(raw)}
                          for name, raw in sorted(held.items())],
                "content_bytes": sum(map(len, held.values())),
                "provider_calls": 0, "current_source_freshness_claimed": False,
                "private_state_tree_copied": False, "provider_transcripts_included": False,
                "execution_authority": False, "completion_authority": False}
    encoded = _encode(manifest)
    _require(len(encoded) <= MAX_MANIFEST_BYTES, "export_manifest_bound")
    archive = Path(str(state) + "-context-evidence.tar")
    _require(not archive.exists(), "fresh_evidence_archive_required")
    with tempfile.TemporaryDirectory(prefix="context-evidence-", dir=state.parent) as temp:
        staging = Path(temp)
        for name, raw in {**held, "manifest.json": encoded}.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
            path.chmod(0o444)
        with archive.open("xb") as stream:
            with tarfile.open(fileobj=stream, mode="w") as tar:
                for name in sorted((*held, "manifest.json")):
                    info = tar.gettarinfo(str(staging / name), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with (staging / name).open("rb") as source:
                        tar.addfile(info, source)
    archive.chmod(0o444)
    _require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, "export_archive_bound")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"schema": RECEIPT_SCHEMA, "status": "exported", "artifact": str(archive),
            "bytes": archive.stat().st_size, "sha256": digest,
            "manifest_sha256": _sha(encoded), "manifest_bytes": len(encoded),
            "file_count": len(held), "content_bytes": manifest["content_bytes"],
            "repository_origin": str(repository), "state_origin": str(state),
            "workspace_root_origin": str(allocation), "profile": PROFILE,
            "provider_calls": 0, "execution_authority": False, "completion_authority": False}


def collect_terminal_context_evidence(*, state: Path, receipts: list, workspace_root: Path,
                                      decision_report: dict, timeout_seconds: float = 30.0) -> dict:
    """Bound capture in a trusted child; failures never change completion/usage."""
    failure = {"schema": RECEIPT_SCHEMA, "status": "unavailable", "reason": "export_unavailable",
               "provider_calls": 0, "execution_authority": False, "completion_authority": False}
    if timeout_seconds < .1:
        return {**failure, "reason": "export_budget_unavailable"}
    try:
        request = _encode(_request(receipts, decision_report))
        returncode, output = _bounded_child([sys.executable, "-B", "-P", str(Path(__file__).resolve()),
            "export", "--state", str(state), "--workspace-root", str(workspace_root)],
            input_bytes=request, timeout=min(30.0, timeout_seconds), limit=MAX_REQUEST_BYTES,
            environment={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", _COLLECTOR_GROUP_MARKER: "owned"})
        _require(returncode == 0,
                 "export_process_unavailable")
        result = _json(output)
        _require(type(result) is dict and result.get("schema") == RECEIPT_SCHEMA,
                 "export_result_invalid")
        return result
    except subprocess.TimeoutExpired:
        return {**failure, "reason": "export_timeout"}
    except Exception as error:
        return {**failure, "error_type": type(error).__name__}


def validate_evidence_receipt(receipt: dict, *, state: Path) -> dict:
    """Lightweight closed contract for the Harbor download boundary."""
    fields = {"schema", "status", "artifact", "bytes", "sha256", "manifest_sha256",
              "manifest_bytes", "file_count", "content_bytes", "repository_origin",
              "state_origin", "workspace_root_origin", "profile", "provider_calls",
              "execution_authority", "completion_authority"}
    _require(type(receipt) is dict and set(receipt) == fields
             and receipt["schema"] == RECEIPT_SCHEMA and receipt["status"] == "exported"
             and receipt["profile"] == PROFILE, "closed_export_receipt_required")
    _require(receipt["state_origin"] == str(state)
             and receipt["artifact"] == str(state) + "-context-evidence.tar"
             and receipt["repository_origin"] == "/app"
             and receipt["workspace_root_origin"] == "/opt/ipfs-supervisor/worktrees"
             and receipt["provider_calls"] == 0
             and receipt["execution_authority"] is False and receipt["completion_authority"] is False,
             "export_receipt_origin_or_authority_differs")
    for key in ("sha256", "manifest_sha256"):
        _require(type(receipt[key]) is str and re.fullmatch(r"[0-9a-f]{64}", receipt[key]),
                 "export_receipt_digest_required")
    for key, maximum in (("bytes", MAX_ARCHIVE_BYTES), ("manifest_bytes", MAX_MANIFEST_BYTES),
                         ("file_count", MAX_FILES), ("content_bytes", MAX_CONTENT_BYTES)):
        _require(type(receipt[key]) is int and 1 <= receipt[key] <= maximum,
                 "export_receipt_bound")
    return receipt


def _allowed_member(name, role, size):
    _relative(name)
    _require(type(size) is int and 0 <= size <= (MAX_SNAPSHOT_BYTES if role == "retrieval-snapshot"
                                               else MAX_FILE_BYTES), "export_leaf_bound")
    valid = ((role == "replay-request" and name == "replay-request.json" and size <= MAX_REQUEST_BYTES)
             or (role == "prepared-state" and name == "state/prepared.json")
             or (role == "context-result" and name == "state/context-result.json")
             or (role == "native-capsule" and name.startswith("state/")
                 and name.endswith("-base-context-capsule.json"))
             or (role == "synthetic-git-head" and name == "repository/.git/HEAD")
             or (role == "public-git-object" and re.fullmatch(
                 r"repository/\.git/objects/[0-9a-f]{2}/(?:[0-9a-f]{38}|[0-9a-f]{62})", name))
             or (role in {"launch-policy", "semantic-parent", "semantic-cas", "raw-retrieval",
                          "retrieval-snapshot", "public-instruction", "task-context-bundle"}
                 and name.startswith("repository/.runtime/")))
    _require(bool(valid), "export_member_scope_refused")


def _verified_archive(*, archive: Path, expected_sha256: str, manifest_sha256: str):
    raw = _read(archive, MAX_ARCHIVE_BYTES)
    _require(_sha(raw) == expected_sha256, "evidence_archive_digest_differs")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        members = []
        for member in tar:
            _require(member.isfile() and not member.linkname and _relative(member.name)
                     and 0 <= member.size <= MAX_SNAPSHOT_BYTES, "evidence_archive_member_refused")
            members.append(member)
            _require(len(members) <= MAX_FILES + 1, "evidence_archive_population_bound")
        by_name = {m.name: m for m in members}
        _require(len(by_name) == len(members) and "manifest.json" in by_name,
                 "duplicate_or_missing_archive_member")
        _require(by_name["manifest.json"].size <= MAX_MANIFEST_BYTES, "manifest_bound")
        encoded = tar.extractfile(by_name["manifest.json"]).read(MAX_MANIFEST_BYTES + 1)
        _require(_sha(encoded) == manifest_sha256, "manifest_digest_differs")
        manifest = _json(encoded)
        _require(set(manifest) == {"schema", "profile", "repository_origin", "state_origin",
            "workspace_root_origin", "native_prompt_bindings", "public_baseline_commit",
            "public_source_count", "files", "content_bytes", "provider_calls",
            "current_source_freshness_claimed", "private_state_tree_copied", "provider_transcripts_included",
            "execution_authority", "completion_authority"}
                 and manifest.get("schema") == SCHEMA and manifest.get("profile") == PROFILE
                 and manifest.get("repository_origin") == "/app"
                 and manifest.get("workspace_root_origin") == "/opt/ipfs-supervisor/worktrees"
                 and _absolute(manifest["state_origin"]).is_relative_to(Path("/opt/ipfs-supervisor/state"))
                 and all(manifest[key] is False for key in ("current_source_freshness_claimed",
                     "private_state_tree_copied", "provider_transcripts_included",
                     "execution_authority", "completion_authority")) and manifest["provider_calls"] == 0,
                 "evidence_namespace_or_schema_differs")
        rows = manifest.get("files")
        _require(type(rows) is list and len(rows) <= MAX_FILES, "manifest_population_bound")
        expected = {row["path"]: row for row in rows}
        _require(len(expected) == len(rows) and set(by_name) == set(expected) | {"manifest.json"},
                 "archive_manifest_population_differs")
        _require(sum(row["bytes"] for row in rows) == manifest.get("content_bytes")
                 and manifest["content_bytes"] <= MAX_CONTENT_BYTES, "manifest_content_bound")
        captured = {"manifest.json": encoded}
        for name, row in expected.items():
            _allowed_member(name, row.get("role"), row.get("bytes"))
            _require(set(row) == {"path", "role", "bytes", "sha256"} and by_name[name].size == row["bytes"],
                     "manifest_file_binding_differs")
            content = tar.extractfile(by_name[name]).read(row["bytes"] + 1)
            _require(len(content) == row["bytes"] and _sha(content) == row["sha256"],
                     "exported_content_digest_differs")
            captured[name] = content
    return manifest, captured


def verify_evidence_archive(*, archive: Path, expected_sha256: str, manifest_sha256: str) -> dict:
    """Verify transferred bytes and allowlist without invoking an auditor or unpacking."""
    return _verified_archive(archive=archive, expected_sha256=expected_sha256,
                             manifest_sha256=manifest_sha256)[0]


def verify_and_unpack(*, archive: Path, expected_sha256: str, manifest_sha256: str,
                      destination: Path) -> dict:
    """Extract only exact manifest-listed regular files into a fresh directory."""
    _require(not destination.exists() and destination.parent.resolve(strict=True) == destination.parent,
             "fresh_canonical_unpack_destination_required")
    manifest, captured = _verified_archive(archive=archive, expected_sha256=expected_sha256,
                                          manifest_sha256=manifest_sha256)
    destination.mkdir(mode=0o700)
    for name, content in captured.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        path.chmod(0o444)
    (destination / "repository/.git/refs").mkdir(exist_ok=True)
    return manifest


def replay_docker_argv(*, export: Path, image: str, auditor_path: str, output: Path) -> list[str]:
    """Explicit provider-free namespace command; caller supplies an existing trusted image."""
    export, output = _absolute(export), _absolute(output)
    _require(export.resolve(strict=True) == export and export.is_dir()
             and output.resolve(strict=True) == output and output.is_dir()
             and not output.is_relative_to(export) and not export.is_relative_to(output)
             and not any(output.iterdir()), "canonical_separate_fresh_output_required")
    manifest = _json(_read(export / "manifest.json", MAX_MANIFEST_BYTES))
    _require(manifest.get("schema") == SCHEMA and manifest.get("profile") == PROFILE,
             "evidence_manifest_required")
    state = _absolute(manifest["state_origin"])
    _require(state.is_relative_to(Path("/opt/ipfs-supervisor/state")), "owner_state_namespace_required")
    _require(re.fullmatch(r"sha256:[0-9a-f]{64}", image), "pinned_existing_audit_image_required")
    _require(auditor_path == "/opt/ipfs-supervisor/source/benchmarks/agent_supervisor/container_coding/terminal_context_evidence.py",
             "trusted_replay_auditor_required")
    return ["docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--mount", f"type=bind,src={export / 'repository'},dst=/app,readonly",
            "--mount", f"type=bind,src={export / 'state'},dst={state},readonly",
            "--mount", f"type=bind,src={export},dst=/evidence,readonly",
            "--mount", f"type=bind,src={output},dst=/audit-output",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=HF_HUB_OFFLINE=1",
            "--env=TRANSFORMERS_OFFLINE=1", "--entrypoint=python3", image,
            "-B", "-P", auditor_path, "replay-inner", "--evidence", "/evidence",
            "--output", "/audit-output/replay.json"]


def _owner_imports():
    from pathlib import Path
    # Direct script invocation must retain the sibling trusted benchmark package.
    source = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(source))
    # A packed owner may expose benchmarks as a namespace while datasets has
    # a regular package with the same name. Load the exact sibling first; its
    # genuine owner bootstrap pins all benchmark namespaces before discovery.
    name = "benchmarks.agent_supervisor.container_coding.terminal_context_audit"
    path = Path(__file__).resolve().with_name("terminal_context_audit.py")
    owner = sys.modules.get(name)
    if owner is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("evidence audit owner unavailable")
        owner = importlib.util.module_from_spec(spec)
        sys.modules[name] = owner
        spec.loader.exec_module(owner)
    if Path(getattr(owner, "__file__", "")).resolve() != path:
        raise ValueError("evidence audit owner differs")
    owner._pin_owner_runtime()


def main():
    global _IN_OWNED_COLLECTOR_GROUP
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    capture = sub.add_parser("export")
    capture.add_argument("--state", type=Path, required=True)
    capture.add_argument("--workspace-root", type=Path, required=True)
    replay = sub.add_parser("replay-inner")
    replay.add_argument("--evidence", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get(_COLLECTOR_GROUP_MARKER) == "owned":
        _require(os.getpid() == os.getpgrp() == os.getsid(0), "owned_collector_group_required")
        _IN_OWNED_COLLECTOR_GROUP = True
    _owner_imports()
    if args.operation == "export":
        try:
            request = _json(sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1))
            _require(len(_encode(request)) <= MAX_REQUEST_BYTES
                     and set(request) == {"receipts", "decision_report"}, "export_request_bound")
            result = export_terminal_context_evidence(state=args.state, workspace_root=args.workspace_root,
                receipts=request["receipts"], decision_report=request["decision_report"])
        except Exception as error:
            result = {"schema": RECEIPT_SCHEMA, "status": "unavailable", "reason": "export_refused",
                      "error_type": type(error).__name__, "provider_calls": 0,
                      "execution_authority": False, "completion_authority": False}
        print(json.dumps(result, sort_keys=True))
    else:
        from benchmarks.agent_supervisor.container_coding.terminal_context_audit import audit_terminal_context
        manifest = _json(_read(args.evidence / "manifest.json", MAX_MANIFEST_BYTES))
        request = _json(_read(args.evidence / "replay-request.json", MAX_REQUEST_BYTES))
        for row in manifest["files"]:
            relative = _relative(row["path"])
            _require(_sha(_read(args.evidence / relative, max(MAX_FILE_BYTES, row["bytes"]))) == row["sha256"],
                     "sealed_replay_input_differs")
        result = audit_terminal_context(state=Path(manifest["state_origin"]),
            workspace_root=Path(manifest["workspace_root_origin"]), receipts=request["receipts"],
            decision_report=request["decision_report"])
        with args.output.open("x") as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False)
        print(json.dumps({"schema": "terminal-context-evidence-replay-result@1", "status": result["status"],
                          "provider_calls": 0, "raw_prompts_exported": False}, sort_keys=True))


if __name__ == "__main__":
    main()
