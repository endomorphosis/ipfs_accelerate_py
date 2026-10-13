"""Prepare a compact span, write locally, and confirm its exact candidate.

This standard-library client also runs with Python ``-I``. The owner maps and
checks source; the worker uses only its existing allocated file permissions.
Original file bytes and literal edit inputs never appear in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import socket
import stat
import sys
import time
import uuid

if __package__:
    from . import worker_source_client as source
else:
    _spec = importlib.util.spec_from_file_location("_edit_source_transport",
        Path(__file__).resolve().with_name("worker_source_client.py"))
    source = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(source)

REQUEST_SCHEMA = "supervisor-worker-source-edit-request@1"
RESPONSE_SCHEMA = source.RESPONSE_SCHEMA
PREPARED_SCHEMA = "supervisor-worker-source-edit-prepared@1"
CONFIRMED_SCHEMA = "supervisor-worker-source-edit-confirmed@1"
DESCRIPTOR_SCHEMA = "supervisor-worker-source-descriptor@3"
MODE = "mapped-span@1"
REPEATED_MODE = "mapped-span@2"
MODES = (MODE, REPEATED_MODE)
MAX_FRAME_BYTES = source.MAX_FRAME_BYTES
MAX_SOURCE_BYTES = 1024 * 1024
MAX_EDIT_BYTES = 4096
MIN_RESPONSE_BYTES = 4096
canonical_json = source.canonical_json
strict_json_loads = source.strict_json_loads
_need, _hex, _integer = source._need, source._hex, source._integer

EDIT_FIELDS = frozenset({"path", "expected_workspace_generation", "expected_source_sha256",
    "expected_view_sha256", "expected_manifest_sha256", "view_start", "view_end",
    "expected_view_preimage", "replacement"})
_PREPARED_FIELDS = frozenset({"schema", "status", "proposal_sha256", "prepare_request_sha256",
    "path", "source_generation", "before_source_sha256", "before_view_sha256", "before_manifest_sha256",
    "original_start", "original_end", "preimage_sha256", "preimage_bytes", "replacement_sha256",
    "replacement_bytes", "candidate_sha256", "candidate_bytes", "candidate_view_sha256",
    "candidate_view_bytes", "candidate_manifest_sha256", "view_policy_binding_sha256",
    "validation_pending", "owner_writes_performed", "source_postimage_verified",
    "proof_authority", "execution_authority", "publication_authority", "completion_authority",
    "mutation_authority", "semantic_equivalence_claimed", "source_obligations_discharged",
    "provider_calls", "prover_calls"})
_CONFIRMED_FIELDS = _PREPARED_FIELDS | {"workspace_generation", "source_population_sha256",
    "baseline_evidence_applicable"}
_FALSE_FLAGS = ("owner_writes_performed", "proof_authority", "execution_authority",
    "publication_authority", "completion_authority", "mutation_authority", "semantic_equivalence_claimed")
_ZERO_COUNTS = ("source_obligations_discharged", "provider_calls", "prover_calls")
REASON_CODES = source.REASON_CODES | {"edit_unavailable", "proposal_consumed", "mapping_refused",
    "confirmation_refused"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _literal(value):
    _need(type(value) is str, "literal UTF-8 edit text required")
    try:
        raw = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("literal UTF-8 edit text required") from exc
    _need(len(raw) <= MAX_EDIT_BYTES and b"\0" not in raw, "edit text exceeds supported bounds")
    return raw


def validate_edit(value):
    """Validate caller input only; source admission remains with the owner."""
    _need(type(value) is dict and set(value) == EDIT_FIELDS, "closed compact edit required")
    source._validate_source_path(value["path"])
    for key in ("expected_workspace_generation", "expected_source_sha256", "expected_view_sha256",
                "expected_manifest_sha256"):
        _need(_hex(value[key]), "exact source/view/generation/manifest pins required")
    start, end = value["view_start"], value["view_end"]
    _need(_integer(start, 0, MAX_SOURCE_BYTES) and _integer(end, start, MAX_SOURCE_BYTES),
        "bounded compact byte interval required")
    preimage, replacement = _literal(value["expected_view_preimage"]), _literal(value["replacement"])
    _need(end - start == len(preimage) and preimage != replacement, "compact edit preimage extent or no-op differs")
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "compact edit input exceeds bound")
    return value


def validate_request(value):
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "query"}
        and value["schema"] == REQUEST_SCHEMA and _hex(value["session_id"], 32)
        and _hex(value["request_id"], 32), "invalid compact edit request")
    query = value["query"]
    _need(type(query) is dict and type(query.get("operation")) is str
        and query["operation"] in {"prepare", "confirm"}, "unsupported compact edit operation")
    common = {"operation", "view_policy_binding_sha256", "max_bytes"}
    if query["operation"] == "prepare":
        _need(set(query) == common | EDIT_FIELDS, "invalid compact prepare fields")
        validate_edit({key: query[key] for key in EDIT_FIELDS})
    else:
        _need(set(query) == common | {"proposal_sha256"} and _hex(query["proposal_sha256"]),
            "invalid compact confirm fields")
    _need(query["view_policy_binding_sha256"] is None or _hex(query["view_policy_binding_sha256"]),
        "invalid compact edit policy binding")
    _need(_integer(query["max_bytes"], MIN_RESPONSE_BYTES, MAX_FRAME_BYTES)
        and len(canonical_json(value)) <= MAX_FRAME_BYTES, "compact edit request exceeds bound")
    return value


def validate_descriptor(value):
    source.validate_descriptor(value)
    _need(value["schema"] == DESCRIPTOR_SCHEMA and value.get("source_edit_mode") in MODES,
        "source owner did not enable compact edits")
    return value


def _validate_page(page, *, confirmed):
    _need(type(page) is dict and set(page) == (_CONFIRMED_FIELDS if confirmed else _PREPARED_FIELDS)
        and page["schema"] == (CONFIRMED_SCHEMA if confirmed else PREPARED_SCHEMA)
        and page["status"] == ("applied_validation_pending" if confirmed else "prepared_validation_pending"),
        "invalid compact edit page")
    source._validate_source_path(page["path"])
    for key in ("proposal_sha256", "prepare_request_sha256", "source_generation", "before_source_sha256",
                "before_view_sha256", "before_manifest_sha256", "preimage_sha256", "replacement_sha256",
                "candidate_sha256", "candidate_view_sha256", "candidate_manifest_sha256"):
        _need(_hex(page[key]), "invalid compact edit page digest")
    _need(page["view_policy_binding_sha256"] is None or _hex(page["view_policy_binding_sha256"]),
        "invalid compact edit page policy")
    for key in ("preimage_bytes", "replacement_bytes"):
        _need(_integer(page[key], 0, MAX_EDIT_BYTES), "invalid compact edit text extent")
    _need(_integer(page["candidate_bytes"], 0, MAX_SOURCE_BYTES)
        and _integer(page["candidate_view_bytes"], 0, page["candidate_bytes"]),
        "invalid compact candidate extent")
    original_bytes = page["candidate_bytes"] - page["replacement_bytes"] + page["preimage_bytes"]
    _need(_integer(original_bytes, 0, MAX_SOURCE_BYTES)
        and _integer(page["original_start"], 0, original_bytes)
        and _integer(page["original_end"], page["original_start"], original_bytes)
        and page["original_end"] - page["original_start"] == page["preimage_bytes"],
        "invalid original edit interval")
    _need(page["preimage_sha256"] != page["replacement_sha256"]
        and page["before_source_sha256"] != page["candidate_sha256"]
        and page["before_view_sha256"] != page["candidate_view_sha256"], "compact edit is a no-op")
    _need(all(page[key] is False for key in _FALSE_FLAGS)
        and page["validation_pending"] is True and page["source_postimage_verified"] is confirmed
        and all(type(page[key]) is int and page[key] == 0 for key in _ZERO_COUNTS),
        "compact edit page grants unsupported authority")
    if confirmed:
        _need(_hex(page["workspace_generation"]) and _hex(page["source_population_sha256"])
            and page["workspace_generation"] != page["source_generation"]
            and page["baseline_evidence_applicable"] is False, "invalid confirmed source generation")
    return page


def validate_response(value, *, request, descriptor=None):
    validate_request(request)
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "ok", "page", "reason_code"}
        and value["schema"] == RESPONSE_SCHEMA and type(value["ok"]) is bool
        and value["session_id"] == request["session_id"] and value["request_id"] == request["request_id"],
        "compact edit response differs from request")
    _need(len(canonical_json(value)) <= request["query"]["max_bytes"], "compact edit response exceeds bound")
    if descriptor is not None:
        validate_descriptor(descriptor)
        _need(descriptor["session_id"] == request["session_id"]
            and descriptor["view_policy_binding_sha256"] == request["query"]["view_policy_binding_sha256"],
            "compact edit request differs from owner session")
    if not value["ok"]:
        _need(value["page"] is None and type(value["reason_code"]) is str
            and value["reason_code"] in REASON_CODES, "invalid compact edit refusal")
        return value
    query = request["query"]
    confirmed = query["operation"] == "confirm"
    page = _validate_page(value["page"], confirmed=confirmed)
    _need(value["reason_code"] is None and page["view_policy_binding_sha256"] == query["view_policy_binding_sha256"],
        "compact edit response policy differs")
    if confirmed:
        _need(page["proposal_sha256"] == query["proposal_sha256"], "compact proposal differs")
    else:
        _need(page["prepare_request_sha256"] == _sha(canonical_json(request))
            and page["path"] == query["path"] and page["source_generation"] == query["expected_workspace_generation"],
            "compact preparation differs from request")
        for page_key, query_key in (("before_source_sha256", "expected_source_sha256"),
                ("before_view_sha256", "expected_view_sha256"), ("before_manifest_sha256", "expected_manifest_sha256")):
            _need(page[page_key] == query[query_key], "compact preparation pins differ")
        for label, key in (("preimage", "expected_view_preimage"), ("replacement", "replacement")):
            raw = _literal(query[key])
            _need(page[label + "_sha256"] == _sha(raw) and page[label + "_bytes"] == len(raw),
                "compact preparation literal binding differs")
    return value


def _exchange(descriptor_path, descriptor, request):
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "compact edit session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30.0, remaining))
        channel.connect(descriptor["socket_path"])
        source.transport._verify_peer(channel, descriptor)
        source.transport._socket_identity(descriptor, Path(descriptor_path).parent, Path(descriptor_path).parent.stat().st_gid)
        source.send_frame(channel, request)
        response = source.receive_frame(channel, max_bytes=request["query"]["max_bytes"])
        source.transport._verify_peer(channel, descriptor)
        _need(time.monotonic_ns() <= descriptor["expires_monotonic_ns"], "compact edit session expired")
    return validate_response(response, request=request, descriptor=descriptor)


def _request(descriptor, **query):
    return validate_request(dict(schema=REQUEST_SCHEMA, session_id=descriptor["session_id"],
        request_id=uuid.uuid4().hex, query=dict(view_policy_binding_sha256=descriptor["view_policy_binding_sha256"],
            max_bytes=MAX_FRAME_BYTES, **query)))


def _snapshot(info):
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
        "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _directory(path):
    path = Path(path)
    _need(path.is_absolute() and path.resolve(strict=True) == path, "canonical allocated directory required")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(parent, name, *, maximum=MAX_SOURCE_BYTES):
    initial = os.stat(name, dir_fd=parent, follow_symlinks=False)
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and 0 <= before.st_size <= maximum, "bounded single-link regular file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        _need(len(raw) <= maximum and len(raw) == before.st_size
            and _snapshot(initial) == _snapshot(before) == _snapshot(os.fstat(fd))
            == _snapshot(os.stat(name, dir_fd=parent, follow_symlinks=False)), "allocated source changed during read")
        return raw, before
    finally:
        os.close(fd)


def _write_candidate(*, descriptor, edit, prepared, progress):
    """Use existing worker file rights; sequential checks are not a writer lock."""
    validate_edit(edit)
    _validate_page(prepared, confirmed=False)
    _need(prepared["path"] == edit["path"] and prepared["source_generation"] == edit["expected_workspace_generation"]
        and prepared["before_view_sha256"] == edit["expected_view_sha256"]
        and prepared["before_manifest_sha256"] == edit["expected_manifest_sha256"],
        "prepared edit identity differs")
    root = Path(descriptor["workspace"])
    _need(Path.cwd() == root and root.resolve(strict=True) == root, "allocated workspace differs")
    path = Path(edit["path"])
    parent_path = root / path.parent
    parent = _directory(parent_path)
    temporary = ".compact-edit-" + uuid.uuid4().hex
    temp_created = False
    try:
        parent_info = os.fstat(parent)
        original, info = _read(parent, path.name)
        _need(info.st_uid in (descriptor["owner_uid"], descriptor["worker_uid"])
            and not stat.S_IMODE(info.st_mode) & 0o7000, "unsupported allocated source ownership or mode")
        expected_size = prepared["candidate_bytes"] - prepared["replacement_bytes"] + prepared["preimage_bytes"]
        _need(len(original) == expected_size and _sha(original) == edit["expected_source_sha256"]
            == prepared["before_source_sha256"], "allocated source preimage differs")
        before, after = prepared["original_start"], prepared["original_end"]
        preimage, replacement = _literal(edit["expected_view_preimage"]), _literal(edit["replacement"])
        _need(original[before:after] == preimage, "allocated source span differs")
        candidate = original[:before] + replacement + original[after:]
        _need(len(candidate) == prepared["candidate_bytes"] and _sha(candidate) == prepared["candidate_sha256"],
            "allocated candidate differs from prepared map")
        def unchanged():
            current_parent = _directory(parent_path)
            try:
                now_parent = os.fstat(current_parent)
                _need((now_parent.st_dev, now_parent.st_ino) == (parent_info.st_dev, parent_info.st_ino),
                    "allocated parent changed")
                current, current_info = _read(parent, path.name)
                _need(current == original and _snapshot(current_info) == _snapshot(info),
                    "allocated source changed before write")
            finally:
                os.close(current_parent)
        if info.st_uid != os.geteuid():
            # Native sticky allocation directories protect owner-owned inodes;
            # use the same preserved-inode policy as the Doctor worker.
            unchanged()
            fd = os.open(path.name, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
            try:
                _need(_snapshot(os.fstat(fd)) == _snapshot(info), "allocated source inode changed")
                progress["write_attempted"] = True
                offset = 0
                while offset < len(candidate):
                    count = os.write(fd, candidate[offset:])
                    _need(count > 0, "candidate write made no progress")
                    offset += count
                os.ftruncate(fd, len(candidate))
                os.fsync(fd)
            finally:
                os.close(fd)
            mode = "preserved_owner_inode"
        else:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600, dir_fd=parent)
            temp_created = True
            with os.fdopen(fd, "wb") as stream:
                stream.write(candidate)
                stream.flush()
                os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode))
                _need(os.fstat(stream.fileno()).st_gid == info.st_gid, "candidate file group differs")
                os.fsync(stream.fileno())
            unchanged()
            progress["write_attempted"] = True
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
            temp_created = False
            mode = "atomic_replace"
        os.fsync(parent)
        actual, written = _read(parent, path.name)
        _need(actual == candidate and stat.S_IMODE(written.st_mode) == stat.S_IMODE(info.st_mode)
            and written.st_uid == info.st_uid and written.st_gid == info.st_gid
            and (mode != "preserved_owner_inode" or (written.st_dev, written.st_ino) == (info.st_dev, info.st_ino)),
            "allocated candidate differs after write")
        current_parent = _directory(parent_path)
        try:
            current_info = os.fstat(current_parent)
            _need((current_info.st_dev, current_info.st_ino) == (parent_info.st_dev, parent_info.st_ino),
                "allocated parent changed after write")
        finally:
            os.close(current_parent)
        progress["local_postimage_verified"] = True
        return mode
    finally:
        if temp_created:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
        os.close(parent)


def edit_source(descriptor_path, descriptor_sha256, *, edit, workspace=None):
    """One attempt only; uncertainty after a write never triggers replay."""
    progress = {"write_attempted": False, "local_postimage_verified": False}
    result = dict(schema="supervisor-worker-source-edit-result@1", ok=False, status="not_written",
        reason_code="transport_error", validation_pending=True, atomic_application_claimed=False,
        proof_authority=False, execution_authority=False, publication_authority=False,
        completion_authority=False, source_obligations_discharged=0, automatic_retry_performed=False)
    try:
        edit = strict_json_loads(canonical_json(validate_edit(edit)))
        descriptor = source.load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
        validate_descriptor(descriptor)
        request = _request(descriptor, operation="prepare", **edit)
        response = _exchange(descriptor_path, descriptor, request)
        if not response["ok"]:
            result["reason_code"] = response["reason_code"]
            return result | progress
        prepared = response["page"]
        # Reread descriptor custody/liveness before using the prepared mapping.
        _need(source.load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace) == descriptor,
            "compact edit descriptor changed before write")
        write_mode = _write_candidate(descriptor=descriptor, edit=edit, prepared=prepared, progress=progress)
        result.update(write_mode=write_mode, proposal_sha256=prepared["proposal_sha256"])
        confirm_request = _request(descriptor, operation="confirm", proposal_sha256=prepared["proposal_sha256"])
        confirmed = _exchange(descriptor_path, descriptor, confirm_request)
        if not confirmed["ok"]:
            result.update(status="written_confirmation_unknown", reason_code=confirmed["reason_code"])
            return result | progress
        page = confirmed["page"]
        _need(all(page[key] == value for key, value in prepared.items()
            if key not in {"schema", "status", "source_postimage_verified"}), "confirmed proposal differs from preparation")
        result.update(ok=True, status="applied_validation_pending", reason_code=None,
            workspace_generation=page["workspace_generation"], source_sha256=page["candidate_sha256"],
            view_sha256=page["candidate_view_sha256"], manifest_sha256=page["candidate_manifest_sha256"],
            source_population_sha256=page["source_population_sha256"], baseline_evidence_applicable=False,
            owner_postimage_verified=True)
    except (OSError, ValueError, TypeError, RuntimeError):
        if progress["write_attempted"]:
            result["status"] = "written_confirmation_unknown"
    return result | progress


def guidance(profile="python-whitespace@1", *, mode=MODE):
    _need(profile in source.PROFILES, "unsupported compact edit profile")
    _need(mode in MODES, "unsupported compact edit mode")
    introduction = ("\nOne mapped compact span edit is available in this source session. Read a current candidate with "
        if mode == MODE else
        "\nSequential mapped compact span edits are available in this source session, with one outstanding proposal "
        "and at most 16 successful preparations. Before each edit, read a fresh current candidate with ")
    fallback = ("Unsupported spans, consumed edit slots or stale pins require a fresh read and ordinary original-source editing; "
        if mode == MODE else
        "After each confirmed edit, reread with --working --edit-ready and use the fresh generation/source/view/manifest pins "
        "for the next edit. Unsupported spans, stale pins or the session preparation limit require a fresh read and ordinary original-source editing; ")
    uncertainty = ("If status is written_confirmation_unknown, reread the working candidate before any further edit; "
        if mode == MODE else
        "If status is written_confirmation_unknown, mapped editing stops for this session; reread the working candidate "
        "and inspect the result before ordinary source editing; ")
    return (introduction +
        "/opt/ipfs-supervisor/venv/bin/python -I "
        "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py "
        "--path <path> --view compact --working --edit-ready (use --discover to list admitted paths). "
        "Run /opt/ipfs-supervisor/venv/bin/python -I "
        "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_edit_client.py "
        "--edit-file <JSON-file> or --edit-file - for stdin. Keep request JSON files outside the checkout "
        "so they cannot enter the final patch, or use stdin. Supply exactly path, "
        "expected_workspace_generation, expected_source_sha256, expected_view_sha256, "
        "expected_manifest_sha256 from a fresh --working compact source page, view_start/view_end "
        "as UTF-8 byte offsets, and literal expected_view_preimage/replacement strings. "
        "The owner maps one retained span; the worker privately checks and edits original source "
        "without displaying it, then the owner verifies the complete admitted postimage. "
        "Each literal is at most 4096 UTF-8 bytes and the request at most 8192 bytes. "
        + fallback +
        "the source reader accepts --view original --working with the current generation/source/view pins. "
        + uncertainty +
        "never replay the old edit automatically. Successful edits remain validation-pending, "
        "invalidate baseline evidence, and establish no behavior, proof or completion. "
        "To rebuild advisory candidate context, run /opt/ipfs-supervisor/venv/bin/python -I "
        "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_context_client.py "
        "--expected-workspace-generation <generation> --select --artifact catalog. "
        "Selected source profile: " + profile + ".\n")


def _edit_input(name):
    if name == "-":
        raw = sys.stdin.buffer.read(MAX_FRAME_BYTES + 1)
    else:
        path = Path(name).absolute()
        parent = _directory(path.parent)
        try:
            raw, _ = _read(parent, path.name, maximum=MAX_FRAME_BYTES)
        finally:
            os.close(parent)
    _need(len(raw) <= MAX_FRAME_BYTES, "compact edit input exceeds bound")
    return validate_edit(strict_json_loads(raw))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(source.DESCRIPTOR_FLAG, default=os.environ.get(source.DESCRIPTOR_ENV))
    parser.add_argument(source.DIGEST_FLAG, default=os.environ.get(source.DIGEST_ENV))
    parser.add_argument("--edit-file", required=True, help="Bounded literal edit JSON file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        _need(args.source_view_descriptor is not None and args.source_view_descriptor_sha256 is not None,
            "source descriptor required")
        result = edit_source(args.source_view_descriptor, args.source_view_descriptor_sha256,
            edit=_edit_input(args.edit_file))
    except (OSError, ValueError, TypeError, RuntimeError):
        result = dict(schema="supervisor-worker-source-edit-result@1", ok=False,
            status="not_written", reason_code="invalid_input", automatic_retry_performed=False)
    print(canonical_json(result).decode("utf-8"))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
