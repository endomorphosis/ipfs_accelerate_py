"""Revalidate task-bound source advice in the allocated coding workspace.

This gate runs immediately before provider dispatch. It verifies source bytes
and retained owner evidence, without replaying neural inference or granting
proof, execution or completion authority. Expected provider edits mean its
workspace check must not be reused as a post-provider check.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

from ..context.context_compiler import build_text_context_references, CONTEXT_COMPILER_VERSION
from ..context.context_contracts import canonical_context_json_bytes, ABSOLUTE_MAX_CONTEXT_BYTES
from .doctor_candidate_runner import _git
from .security_autoencoder_advisor import _read
from .task_context_bundle import load_task_context_selection, MAX_WORKER_EVIDENCE_BYTES, WORKER_EVIDENCE_SCHEMA

KIND = "source-formalization-context"
SCHEMA = "source-formalization-dispatch-observation@1"
HANDOFF_SCHEMA = "source-formalization-owner-handoff@1"
MAX_HANDOFF_BYTES = 131072
HANDOFF_LIFETIME_NS = 90 * 1_000_000_000
HANDOFF_FLAGS = ("--source-formalization-handoff", "--source-formalization-handoff-sha256")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source formalization field")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite source formalization field")


def _json(value):
    return canonical_context_json_bytes(value).decode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative(value):
    if (type(value) is not str or not value or len(value.encode()) > 4096
            or any(c in value for c in "\\\n\r\0") or Path(value).is_absolute()
            or Path(value).as_posix() != value or ".." in Path(value).parts
            or value == "."):
        raise ValueError("canonical source formalization relative path required")
    return value


def _parts(prompt):
    if type(prompt) is not str or len(prompt.encode()) > ABSOLUTE_MAX_CONTEXT_BYTES:
        raise ValueError("bounded native source formalization prompt required")
    try:
        wire, end = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_nonfinite).raw_decode(prompt)
    except (ValueError, TypeError, RecursionError) as exc:
        # A duplicate-key envelope can spell its kind with Unicode escapes.
        # Inspect only for presence in this fallback; it never admits bytes.
        detected = KIND in prompt
        if not detected:
            try:
                value, _ = json.JSONDecoder().raw_decode(prompt)
                detected = (type(value) is dict and type(value.get("evidence")) is list
                    and any(type(row) is dict and row.get("kind") == KIND for row in value["evidence"]))
            except (ValueError, TypeError, RecursionError):
                pass
        if detected:
            raise ValueError("source formalization requires canonical compiled context") from exc
        return None
    evidence = wire.get("evidence") if type(wire) is dict else None
    if type(evidence) is not list:
        if KIND in prompt[:end]:
            raise ValueError("source formalization requires native evidence references")
        return None
    rows = [row for row in evidence if type(row) is dict and row.get("kind") == KIND]
    if not rows:
        return None
    if (_json(wire) != prompt[:end] or type(wire.get("contract_version")) is not int
            or wire["contract_version"] != CONTEXT_COMPILER_VERSION
            or not 1 <= len(rows) <= 6):
        raise ValueError("source formalization requires canonical compiled context")
    try:
        text = "".join(row["summary"] for row in rows)
        if len(text.encode()) > MAX_WORKER_EVIDENCE_BYTES:
            raise ValueError("source formalization worker evidence exceeds bound")
        payload = json.loads(text, object_pairs_hook=_unique, parse_constant=_nonfinite)
        if type(payload) is not dict or _json(payload) != text:
            raise ValueError("canonical source formalization evidence required")
        binding = payload["context_bundle"]
        if (type(binding) is not dict or set(binding) != {"artifact", "sha256"}
                or not _digest(binding["sha256"])):
            raise ValueError("exact source formalization bundle binding required")
        artifact = _relative(binding["artifact"])
        expected = build_text_context_references(text, reference_prefix="source-formalization",
            kind=KIND, path=artifact, repository_id=wire["repository_id"], tree_id=wire["tree_id"],
            required=True, chunk_bytes=6144)
        if _json([row.to_record() for row in expected]) != _json(rows):
            raise ValueError("source formalization chunk binding differs")
        if (payload.get("schema") != WORKER_EVIDENCE_SCHEMA
                or type(payload.get("task_id")) is not str or not payload["task_id"]
                or payload["task_id"] != wire.get("objective_id")
                or type(payload.get("task_cid")) is not str or not 0 < len(payload["task_cid"]) <= 4096):
            raise ValueError("source formalization belongs to another native task")
    except (KeyError, TypeError, RecursionError) as exc:
        raise ValueError("invalid source formalization evidence envelope") from exc
    return payload, text, len(rows)


def _workspace(repository, workspace):
    root, allocated = Path(repository).absolute(), Path(workspace).absolute()
    if (root.resolve(strict=True) != root or allocated.resolve(strict=True) != allocated
            or root == allocated):
        raise ValueError("source formalization requires a separate canonical allocated worktree")
    for path in (root, allocated):
        if _git(path, "rev-parse", "--show-toplevel").decode().strip() != str(path):
            raise ValueError("source formalization requires exact Git checkout roots")
    common = lambda path: Path(_git(path, "rev-parse", "--path-format=absolute",
                                   "--git-common-dir").decode().strip()).resolve(strict=True)
    if common(root) != common(allocated):
        raise ValueError("source formalization repository differs from allocated worktree")
    return root, allocated


def _live_selection(payload, text, root, *, timeout_seconds=None):
    binding = payload["context_bundle"]
    selected = load_task_context_selection(repository=root, artifact=binding["artifact"],
        expected_sha256=binding["sha256"], task_cid=payload["task_cid"], task_id=payload["task_id"],
        **({"source384_timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}))
    if _json(selected.get("worker_evidence")) != text:
        raise ValueError("source formalization differs from current task evidence")
    sources = {}
    for key in ("source384_context", "numerical_formalization"):
        if key not in selected:
            continue
        population = selected[key]["source_hashes"]
        if type(population) is not dict or len(population) > 256:
            raise ValueError("bounded source formalization source population required")
        for name, digest in population.items():
            _relative(name)
            if not _digest(digest) or name in sources and sources[name] != digest:
                raise ValueError("source formalization source bindings disagree")
            sources[name] = digest
    if len(sources) > 320:
        raise ValueError("source formalization source population exceeds bound")
    return sources


def _current_sources(root, sources):
    for name, digest in sorted(sources.items()):
        _relative(name)
        if not _digest(digest) or _sha(_read(root / name, 1024 * 1024)) != digest:
            raise ValueError("allocated source formalization source is stale")


def _observation(payload, text, chunks, sources, *, inherited=False):
    return dict(schema=SCHEMA, task_id=payload["task_id"], task_cid=payload["task_cid"],
        worker_evidence_sha256=_sha(text.encode()), context_bundle_sha256=payload["context_bundle"]["sha256"],
        context_kinds=[row["kind"] for row in payload["contexts"]], chunks=chunks,
        source_files=len(sources), shared_git_repository_verified=True,
        canonical_source_current=True, workspace_source_current=True,
        validation_phase="before_provider", neural_inference_replayed=False,
        owner_validation_inherited=inherited, owner_private_state_replayed_here=not inherited,
        proof_authority=False, formalization_authority=False, execution_authority=False,
        completion_authority=False, source_semantics_verified=False,
        whole_program_semantics_verified=False, source_obligations_discharged=0)


def _protected_directory(path, *, owner, group, worker=False):
    if path.resolve(strict=True) != path:
        raise ValueError("canonical owner handoff directory required")
    for parent in (path, *path.parents):
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, owner)
                or info.st_mode & 0o022):
            raise ValueError("owner handoff ancestor is not protected")
        if worker and os.access(parent, os.W_OK):
            raise ValueError("worker can write an owner handoff ancestor")
    info = path.lstat()
    if (info.st_uid != owner or info.st_gid != group or stat.S_IMODE(info.st_mode) != 0o750):
        raise ValueError("exact owner handoff directory permissions required")


def _owner_boundary(path):
    path = Path(path).absolute()
    raw = _read(path, 32768)
    info = path.lstat()
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
    if (info.st_uid != 0 or info.st_mode & 0o022
            or value.get("schema") != "supervisor-container-worker-boundary@1"
            or value.get("owner_uid") != os.getuid() or os.geteuid() != os.getuid()
            or type(value.get("worker_uid")) is not int or value["worker_uid"] == os.getuid()
            or value.get("namespaces") != {k: os.readlink("/proc/self/ns/" + k) for k in ("pid", "mnt", "net")}):
        raise ValueError("trusted container owner boundary required")
    return value, _sha(raw)


def _worker_boundary(path, digest, workspace):
    from .container_worker_boundary import verify_container_worker_boundary
    result = verify_container_worker_boundary(artifact=path, expected_sha256=digest,
                                               workspace=workspace, purpose="coding")
    return result


def _sealed_handoff(*, path, expected_sha256, boundary, boundary_sha256, prompt, payload, text, root, allocated):
    boundary = Path(boundary).absolute()
    verified = _worker_boundary(boundary, boundary_sha256, allocated)
    parent = boundary.parent / "worker-observations"
    # The production boundary uses the supervisor's shared read-only group.
    _protected_directory(parent, owner=verified["owner_uid"], group=verified["owner_uid"], worker=True)
    path = Path(path).absolute()
    if (path.parent != parent or re.fullmatch(r"formalization-[0-9a-f]{32}\.json", path.name) is None
            or not _digest(expected_sha256)):
        raise ValueError("exact owner handoff nomination required")
    raw = _read(path, MAX_HANDOFF_BYTES)
    info = path.lstat()
    if (info.st_uid != verified["owner_uid"] or info.st_gid != verified["owner_uid"]
            or stat.S_IMODE(info.st_mode) != 0o440 or os.access(path, os.W_OK)
            or _sha(raw) != expected_sha256):
        raise ValueError("owner handoff file ownership, permissions or digest differs")
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
    fields = {"schema", "repository", "workspace", "prompt_sha256", "worker_evidence_sha256", "task_id",
        "task_cid", "context_bundle", "source_hashes", "boundary_sha256", "issued_monotonic_ns",
        "expires_monotonic_ns", "owner_observation", "proof_authority", "execution_authority", "completion_authority"}
    if (type(value) is not dict or set(value) != fields or value["schema"] != HANDOFF_SCHEMA
            or _json(value).encode() != raw or value["repository"] != str(root)
            or value["workspace"] != str(allocated) or value["prompt_sha256"] != _sha(prompt.encode())
            or value["worker_evidence_sha256"] != _sha(text.encode())
            or value["task_id"] != payload["task_id"] or value["task_cid"] != payload["task_cid"]
            or value["context_bundle"] != payload["context_bundle"] or value["boundary_sha256"] != boundary_sha256
            or any(value[key] is not False for key in ("proof_authority", "execution_authority", "completion_authority"))):
        raise ValueError("owner handoff differs from exact worker invocation")
    issued, expires = value["issued_monotonic_ns"], value["expires_monotonic_ns"]
    if (type(issued) is not int or type(expires) is not int or expires-issued != HANDOFF_LIFETIME_NS
            or not issued <= time.monotonic_ns() <= expires):
        raise ValueError("owner handoff is expired or from a future invocation")
    sources = value["source_hashes"]
    if type(sources) is not dict or len(sources) > 320:
        raise ValueError("bounded sealed owner source population required")
    expected = _observation(payload, text, len(build_text_context_references(text,
        reference_prefix="source-formalization", kind=KIND, required=True, chunk_bytes=6144)), sources)
    if value["owner_observation"] != expected:
        raise ValueError("sealed owner observation differs")
    return sources, expires


def validate_source_formalization_context(*, prompt, repository, workspace, boundary=None,
        boundary_sha256=None, handoff=None, handoff_sha256=None):
    """Validate owner state locally, or a sealed owner projection in a worker."""
    parsed = _parts(prompt)
    if parsed is None:
        if handoff is not None or handoff_sha256 is not None:
            raise ValueError("selected owner handoff requires exact prompt evidence")
        return None
    if repository is None:
        raise ValueError("source formalization requires the selected semantic repository")
    payload, text, chunks = parsed
    root, allocated = _workspace(repository, workspace)
    if payload.get("repository") != str(root):
        raise ValueError("source formalization belongs to another repository")
    inherited = boundary is not None
    if inherited:
        if handoff is None or not _digest(handoff_sha256) or not _digest(boundary_sha256):
            raise ValueError("container source formalization requires a sealed owner handoff")
        sources, expires = _sealed_handoff(path=handoff, expected_sha256=handoff_sha256,
            boundary=boundary, boundary_sha256=boundary_sha256, prompt=prompt, payload=payload,
            text=text, root=root, allocated=allocated)
        _current_sources(root, sources)
    else:
        if handoff is not None or handoff_sha256 is not None or boundary_sha256 is not None:
            raise ValueError("owner handoff requires an exact container boundary")
        sources = _live_selection(payload, text, root)
    _current_sources(allocated, sources)
    if inherited and time.monotonic_ns() > expires:
        raise ValueError("owner handoff expired during source validation")
    return _observation(payload, text, chunks, sources, inherited=inherited)


def prepare_container_source_formalization_dispatch(*, argv, boundary, workspace, input_fd=0):
    """Owner-only stdin capture and optional handoff; caller closes returned fd.

    The returned sealed memfd must replace child stdin. No prompt is logged or
    written to a normal file. Selected private state is replayed only as owner.
    """
    import fcntl
    import select
    if (type(argv) is not list or any(type(arg) is not str for arg in argv)
            or any(arg == flag or arg.startswith(flag + "=") for arg in argv for flag in HANDOFF_FLAGS)):
        raise ValueError("owner creates formalization handoff arguments independently")
    deadline = time.monotonic() + 90.
    data = bytearray()
    while True:
        remaining = deadline-time.monotonic()
        if remaining <= 0 or not select.select([input_fd], [], [], remaining)[0]:
            raise TimeoutError("owner prompt capture deadline expired")
        part = os.read(input_fd, min(65536, 2*1024*1024+1-len(data)))
        if not part:
            break
        data.extend(part)
        if len(data) > 2*1024*1024:
            raise ValueError("owner prompt exceeds bound")
    raw = bytes(data)
    prompt = raw.decode("utf-8")
    parsed = _parts(prompt)
    command = list(argv)
    if parsed is not None:
        selected = []
        for index, arg in enumerate(argv):
            if arg == "--semantic-repository" and index + 1 < len(argv):
                selected.append(argv[index+1])
            elif arg.startswith("--semantic-repository="):
                selected.append(arg.split("=", 1)[1])
        if len(selected) != 1:
            raise ValueError("owner formalization needs exact semantic repository selection")
        value, boundary_sha256 = _owner_boundary(boundary)
        root, allocated = _workspace(selected[0], workspace)
        payload, text, chunks = parsed
        if payload.get("repository") != str(root):
            raise ValueError("owner formalization repository differs")
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("owner formalization validation deadline expired")
        sources = _live_selection(payload, text, root, timeout_seconds=min(90., remaining))
        _current_sources(root, sources)
        _current_sources(allocated, sources)
        if time.monotonic() >= deadline:
            raise TimeoutError("owner formalization validation deadline expired")
        parent = Path(boundary).absolute().parent / "worker-observations"
        _protected_directory(parent, owner=value["owner_uid"], group=value["owner_uid"])
        issued = time.monotonic_ns()
        receipt = dict(schema=HANDOFF_SCHEMA, repository=str(root), workspace=str(allocated),
            prompt_sha256=_sha(raw), worker_evidence_sha256=_sha(text.encode()),
            task_id=payload["task_id"], task_cid=payload["task_cid"], context_bundle=payload["context_bundle"],
            source_hashes=sources, boundary_sha256=boundary_sha256, issued_monotonic_ns=issued,
            expires_monotonic_ns=issued+HANDOFF_LIFETIME_NS,
            owner_observation=_observation(payload, text, chunks, sources),
            proof_authority=False, execution_authority=False, completion_authority=False)
        body = _json(receipt).encode()
        if len(body) > MAX_HANDOFF_BYTES:
            raise ValueError("owner formalization handoff exceeds bound")
        path = parent / ("formalization-" + uuid.uuid4().hex + ".json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
        command.extend([HANDOFF_FLAGS[0], str(path), HANDOFF_FLAGS[1], _sha(body)])
    fd = os.memfd_create("supervisor-owner-prompt", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(raw)
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
    except BaseException:
        os.close(fd)
        raise
    return dict(argv=command, stdin_fd=fd)


__all__ = ["validate_source_formalization_context", "prepare_container_source_formalization_dispatch"]
