"""Read an exact selected baseline capsule through the existing source socket.

This client uses only the standard library and runs with Python ``-I``. A
descriptor advertises the capability; the existing owner admits readers and
checks current source bytes. Capsule identity conveys no proof authority.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time
import uuid

if __package__:
    from . import worker_source_client as source
else:
    _spec = importlib.util.spec_from_file_location("_semantic_source_transport",
        Path(__file__).resolve().with_name("worker_source_client.py"))
    source = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(source)

MODE = "selected-capsules@1"
REQUEST_SCHEMA = "supervisor-worker-semantic-evidence-request@1"
PAGE_SCHEMA = "supervisor-worker-semantic-evidence-page@1"
CURSOR_SCHEMA = "supervisor-worker-semantic-evidence-cursor@1"
MAX_CAPSULES = 256
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_SELECTED_CAPSULES = 16
MAX_SELECTED_CAPSULE_BYTES = 32768
MAX_CONTINUATIONS = 256
MAX_FRAME_BYTES = source.MAX_FRAME_BYTES
AUTHORITY_FLAGS = ("proof_authority", "execution_authority", "completion_authority",
                   "omission_authority", "mutation_authority", "semantic_equivalence_claimed")
canonical_json = source.canonical_json
strict_json_loads = source.strict_json_loads
_need, _hex, _integer, _text = source._need, source._hex, source._integer, source._text


def validate_capability(descriptor):
    source.validate_descriptor(descriptor)
    _need(descriptor["schema"] == source.DESCRIPTOR_SCHEMA_V4
        and descriptor["semantic_evidence_mode"] == MODE,
        "selected baseline capsule retrieval is not advertised")
    return descriptor


def validate_binding(binding):
    _need(type(binding) is dict and set(binding) == set(source.BINDING_FIELDS),
        "exact semantic evidence source binding required")
    _need(all(_text(binding[key]) for key in ("task_id", "scope_cid", "semantic_root_cid"))
        and all(_hex(binding[key]) for key in (
            "semantic_sha256", "native_prompt_sha256", "source_manifest_sha256"))
        and type(binding["profile"]) is str and binding["profile"] in source.PROFILES,
        "invalid semantic evidence source binding")
    return binding


def validate_cursor(value):
    _need(type(value) is dict and set(value) == {"schema", "request_sha256", "offset", "token"}
        and value["schema"] == CURSOR_SCHEMA and _hex(value["request_sha256"])
        and _hex(value["token"], 32) and _integer(value["offset"], 1, MAX_ARTIFACT_BYTES),
        "invalid semantic evidence continuation")
    return value


def validate_request(value):
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "query"}
        and value["schema"] == REQUEST_SCHEMA and _hex(value["session_id"], 32)
        and _hex(value["request_id"], 32), "invalid semantic evidence request")
    query = value["query"]
    _need(type(query) is dict and set(query) == {"capsule_index", "cursor", "max_bytes"}
        and _integer(query["capsule_index"], 0, MAX_CAPSULES - 1)
        and _integer(query["max_bytes"], source.MIN_PAGE_BYTES, MAX_FRAME_BYTES),
        "invalid semantic evidence query")
    if query["cursor"] is not None:
        validate_cursor(query["cursor"])
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "semantic evidence request exceeds bound")
    return value


def validate_response(value, *, request, descriptor=None):
    validate_request(request)
    _need(type(value) is dict and set(value) == {
        "schema", "session_id", "request_id", "ok", "page", "reason_code"}
        and value["schema"] == source.RESPONSE_SCHEMA and type(value["ok"]) is bool
        and value["session_id"] == request["session_id"] and value["request_id"] == request["request_id"],
        "semantic evidence response differs from request")
    _need(len(canonical_json(value)) <= request["query"]["max_bytes"],
        "semantic evidence response exceeds bound")
    if descriptor is not None:
        validate_capability(descriptor)
        _need(descriptor["session_id"] == request["session_id"], "semantic evidence session differs")
    if not value["ok"]:
        _need(value["page"] is None and type(value["reason_code"]) is str
            and value["reason_code"] in source.REASON_CODES, "invalid semantic evidence refusal")
        return value
    page = value["page"]
    fields = {"schema", "binding", "artifact", "capsule_index", "artifact_sha256", "artifact_bytes",
        "source_state", "canonical_source_current", "workspace_source_current", "read_only",
        "source_obligations_discharged", "offset", "end_offset", "encoding", "content",
        "slice_sha256", "exhausted", "next_cursor", *AUTHORITY_FLAGS}
    _need(type(page) is dict and set(page) == fields and page["schema"] == PAGE_SCHEMA
        and value["reason_code"] is None, "invalid semantic evidence page")
    query, cursor = request["query"], request["query"]["cursor"]
    _need(page["artifact"] == "selected-capsules" and type(page["capsule_index"]) is int
        and page["capsule_index"] == query["capsule_index"], "semantic evidence selection differs")
    _need(page["source_state"] == "baseline"
        and all(page[key] is True for key in ("canonical_source_current", "workspace_source_current", "read_only"))
        and all(page[key] is False for key in AUTHORITY_FLAGS)
        and type(page["source_obligations_discharged"]) is int and page["source_obligations_discharged"] == 0,
        "semantic evidence freshness or authority differs")
    validate_binding(page["binding"])
    if descriptor is not None:
        _need(all(page["binding"][key] == descriptor[key] for key in source.BINDING_FIELDS),
            "semantic evidence owner binding differs")
    _need(_hex(page["artifact_sha256"]) and _hex(page["slice_sha256"]),
        "invalid semantic evidence digest")
    _need(_integer(page["artifact_bytes"], 1, MAX_ARTIFACT_BYTES)
        and all(_integer(page[key], 0, MAX_ARTIFACT_BYTES) for key in ("offset", "end_offset"))
        and page["offset"] < page["end_offset"] <= page["artifact_bytes"]
        and page["offset"] == (0 if cursor is None else cursor["offset"]),
        "invalid semantic evidence byte accounting")
    _need(type(page["content"]) is str and page["encoding"] in ("utf-8", "base64"),
        "invalid semantic evidence content")
    raw = (page["content"].encode("utf-8") if page["encoding"] == "utf-8"
        else base64.b64decode(page["content"], validate=True))
    if page["encoding"] == "base64":
        _need(base64.b64encode(raw).decode("ascii") == page["content"], "noncanonical evidence base64")
    _need(len(raw) == page["end_offset"] - page["offset"]
        and hashlib.sha256(raw).hexdigest() == page["slice_sha256"], "semantic evidence slice differs")
    if page["offset"] == 0 and page["end_offset"] == page["artifact_bytes"]:
        _need(hashlib.sha256(raw).hexdigest() == page["artifact_sha256"], "semantic evidence artifact differs")
    _need(type(page["exhausted"]) is bool
        and page["exhausted"] == (page["end_offset"] == page["artifact_bytes"]),
        "semantic evidence completion differs")
    if page["exhausted"]:
        _need(page["next_cursor"] is None, "exhausted semantic evidence has continuation")
    else:
        validate_cursor(page["next_cursor"])
        _need(page["next_cursor"]["offset"] == page["end_offset"], "semantic evidence continuation differs")
        if cursor is not None:
            _need(page["next_cursor"]["request_sha256"] == cursor["request_sha256"],
                "semantic evidence continuation query differs")
    return value


def request_page(descriptor_path, descriptor_sha256, *, capsule_index, cursor=None,
                 max_bytes=8192, workspace=None, expected_capsule_sha256=None):
    _need(expected_capsule_sha256 is None or _hex(expected_capsule_sha256),
        "invalid expected capsule digest")
    descriptor = validate_capability(source.load_descriptor(
        descriptor_path, descriptor_sha256, workspace=workspace))
    request = validate_request(dict(schema=REQUEST_SCHEMA, session_id=descriptor["session_id"],
        request_id=uuid.uuid4().hex, query=dict(capsule_index=capsule_index, cursor=cursor, max_bytes=max_bytes)))
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "semantic evidence session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30., remaining))
        channel.connect(descriptor["socket_path"])
        source.transport._verify_peer(channel, descriptor)
        source.transport._socket_identity(descriptor, Path(descriptor_path).parent,
            Path(descriptor_path).parent.stat().st_gid)
        source.send_frame(channel, request)
        response = source.receive_frame(channel, max_bytes=max_bytes)
        source.transport._verify_peer(channel, descriptor)
        _need(time.monotonic_ns() <= descriptor["expires_monotonic_ns"], "semantic evidence session expired")
    result = validate_response(response, request=request, descriptor=descriptor)
    if result["ok"] and expected_capsule_sha256 is not None:
        _need(result["page"]["artifact_sha256"] == expected_capsule_sha256,
            "retrieved capsule differs from expected reference")
    return result


def render_page_for_model(response):
    if not response["ok"]:
        return dict(ok=False, reason_code=response["reason_code"])
    page = response["page"]
    return dict(ok=True, **{key: page[key] for key in (
        "artifact", "capsule_index", "artifact_sha256", "artifact_bytes", "binding", "source_state",
        "canonical_source_current", "workspace_source_current", "read_only", *AUTHORITY_FLAGS,
        "source_obligations_discharged", "offset", "end_offset", "encoding", "content", "slice_sha256",
        "exhausted", "next_cursor")})


def guidance():
    return ("\nExact selected baseline capsules are available from the allocated workspace with "
        "/opt/ipfs-supervisor/venv/bin/python -I /opt/ipfs-supervisor/source/ipfs_accelerate_py/"
        "agent_supervisor/runtime/worker_semantic_evidence_client.py --capsule-index <capsule_index> "
        "--expected-capsule-sha256 <capsule_sha256>. Retain the index, digest and page "
        "budget and pass next_cursor JSON as --cursor for further pages; decode base64 slices before "
        "concatenating bytes. These are the exact selected native capsule bytes, not new inference "
        "or a safety proof. Any admitted source edit invalidates this baseline endpoint. Use fresh "
        "working-candidate retrieval after edits; it does not restore baseline evidence. "
        "The existing source session shares 64 requests and 524288 response bytes across all "
        "operations; other reads can exhaust that budget.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(source.DESCRIPTOR_FLAG, default=os.environ.get(source.DESCRIPTOR_ENV))
    parser.add_argument(source.DIGEST_FLAG, default=os.environ.get(source.DIGEST_ENV))
    parser.add_argument("--capsule-index", required=True, type=int)
    parser.add_argument("--expected-capsule-sha256")
    parser.add_argument("--cursor")
    parser.add_argument("--max-bytes", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        result = request_page(args.source_view_descriptor, args.source_view_descriptor_sha256,
            capsule_index=args.capsule_index,
            cursor=None if args.cursor is None else strict_json_loads(args.cursor),
            max_bytes=args.max_bytes, workspace=Path.cwd(), expected_capsule_sha256=args.expected_capsule_sha256)
        print(canonical_json(render_page_for_model(result)).decode())
        return 0 if result["ok"] else 2
    except (ValueError, OSError, RuntimeError, TypeError, KeyError):
        # Do not leak source, descriptor locators or provider text in diagnostics.
        print(canonical_json(dict(ok=False, reason_code="client_validation_failed")).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
