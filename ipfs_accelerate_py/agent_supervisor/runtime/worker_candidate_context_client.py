"""Bounded candidate-context reads over the isolated worker's source socket.

This standard-library client also runs directly with Python ``-I``. The source
descriptor authenticates routing and baseline admission; a required working
generation selects advisory candidate artifacts, never accepted source or proof.
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
    _spec = importlib.util.spec_from_file_location("_candidate_source_transport",
        Path(__file__).resolve().with_name("worker_source_client.py"))
    source = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(source)

REQUEST_SCHEMA = "supervisor-worker-candidate-context-request@1"
REQUEST_SCHEMA_V2 = "supervisor-worker-candidate-context-request@2"
RESPONSE_SCHEMA = source.RESPONSE_SCHEMA
PAGE_SCHEMA = "supervisor-worker-candidate-context-page@1"
PAGE_SCHEMA_V2 = "supervisor-worker-candidate-context-page@2"
CURSOR_SCHEMA = "supervisor-worker-candidate-context-cursor@1"
ARTIFACTS = ("summary", "capsules", "formal")
SELECTED_ARTIFACTS = ("catalog", "capsules", "obligations")
SELECTION_FIELDS = ("path", "symbol_id", "function_id", "kind")
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_FRAME_BYTES = source.MAX_FRAME_BYTES
MIN_PAGE_BYTES = source.MIN_PAGE_BYTES
canonical_json = source.canonical_json
strict_json_loads = source.strict_json_loads
_need, _hex, _integer, _text = source._need, source._hex, source._integer, source._text


def _validate_cursor(value):
    _need(type(value) is dict and set(value) == {"schema", "request_sha256", "offset", "token"}
        and value["schema"] == CURSOR_SCHEMA and _hex(value["request_sha256"])
        and _hex(value["token"], 32) and _integer(value["offset"], 1, MAX_ARTIFACT_BYTES),
        "invalid candidate-context cursor")


def validate_selection(artifact, selection):
    """Validate exact filters; a selection never asserts proof coverage."""
    _need(type(artifact) is str and artifact in SELECTED_ARTIFACTS,
        "unsupported selected candidate-context artifact")
    _need(type(selection) is dict and set(selection) == set(SELECTION_FIELDS),
        "invalid candidate-context selection fields")
    _need(all(value is None or (type(value) is str
        and not any(ord(char) < 32 or ord(char) == 127 for char in value))
        for value in selection.values()), "candidate-context filters contain control characters")
    path = selection["path"]
    _need(path is None or (_text(path, 4096) and not Path(path).is_absolute()
        and str(Path(path)) == path and path != "."
        and not any(part in ("..", ".git") for part in Path(path).parts)
        and "\\" not in path), "canonical selected candidate-context path required")
    _need(all(selection[key] is None or _text(selection[key], 512)
        for key in SELECTION_FIELDS if key != "path"),
        "invalid candidate-context filter")
    if artifact == "catalog":
        _need(all(selection[key] is None for key in SELECTION_FIELDS if key != "path"),
            "catalog accepts only a path filter")
    elif artifact == "capsules":
        _need(path is not None and selection["function_id"] is None and selection["kind"] is None,
            "capsules require a path and accept only a symbol filter")
    else:
        _need(path is not None and selection["symbol_id"] is None,
            "obligations require a path and accept only function and kind filters")
    return selection


def validate_request(value):
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "query"}
        and value["schema"] in (REQUEST_SCHEMA, REQUEST_SCHEMA_V2) and _hex(value["session_id"], 32)
        and _hex(value["request_id"], 32), "invalid candidate-context request")
    query = value["query"]
    selected = value["schema"] == REQUEST_SCHEMA_V2
    fields = {"artifact", "expected_workspace_generation", "cursor", "max_bytes"}
    if selected:
        fields.add("selection")
    _need(type(query) is dict and set(query) == fields,
        "invalid candidate-context query")
    if selected:
        validate_selection(query["artifact"], query["selection"])
    else:
        _need(type(query["artifact"]) is str and query["artifact"] in ARTIFACTS,
            "unsupported candidate-context artifact")
    _need(_hex(query["expected_workspace_generation"]), "candidate-context generation required")
    if query["cursor"] is not None:
        _validate_cursor(query["cursor"])
    _need(_integer(query["max_bytes"], MIN_PAGE_BYTES, MAX_FRAME_BYTES),
        "invalid candidate-context page budget")
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "candidate-context request exceeds bound")
    return value


def validate_response(value, *, request, descriptor=None):
    validate_request(request)
    _need(type(value) is dict and set(value) == {
        "schema", "session_id", "request_id", "ok", "page", "reason_code"}
        and value["schema"] == RESPONSE_SCHEMA and type(value["ok"]) is bool
        and value["session_id"] == request["session_id"] and value["request_id"] == request["request_id"],
        "candidate-context response differs from request")
    _need(len(canonical_json(value)) <= request["query"]["max_bytes"],
        "candidate-context response exceeds budget")
    if descriptor is not None:
        source.validate_descriptor(descriptor)
        _need(descriptor["session_id"] == request["session_id"],
            "candidate-context response belongs to another session")
    if not value["ok"]:
        _need(value["page"] is None and type(value["reason_code"]) is str
            and value["reason_code"] in source.REASON_CODES, "invalid candidate-context refusal")
        return value
    page = value["page"]
    selected = request["schema"] == REQUEST_SCHEMA_V2
    fields = {"schema", "binding", "workspace_generation",
        "source_state", "baseline_evidence_applicable", "proof_authority", "completion_authority",
        "source_obligations_discharged", "artifact", "artifact_sha256", "artifact_bytes", "offset",
        "end_offset", "encoding", "content", "slice_sha256", "exhausted", "next_cursor"}
    if selected:
        fields.add("selection")
    _need(type(page) is dict and set(page) == fields
        and page["schema"] == (PAGE_SCHEMA_V2 if selected else PAGE_SCHEMA)
        and value["reason_code"] is None,
        "invalid candidate-context page")
    query = request["query"]
    if selected:
        validate_selection(page["artifact"], page["selection"])
        _need(page["selection"] == query["selection"], "candidate-context filters differ")
    _need(page["artifact"] == query["artifact"]
        and page["workspace_generation"] == query["expected_workspace_generation"],
        "candidate-context selection differs")
    _need(page["source_state"] == "working-candidate"
        and all(page[key] is False for key in (
            "baseline_evidence_applicable", "proof_authority", "completion_authority"))
        and type(page["source_obligations_discharged"]) is int
        and page["source_obligations_discharged"] == 0,
        "candidate-context grants no proof or completion authority")
    binding = page["binding"]
    _need(type(binding) is dict and set(binding) == set(source.BINDING_FIELDS),
        "invalid candidate-context owner binding")
    _need(all(_text(binding[key]) for key in ("task_id", "scope_cid", "semantic_root_cid"))
        and all(_hex(binding[key]) for key in (
            "semantic_sha256", "native_prompt_sha256", "source_manifest_sha256"))
        and type(binding["profile"]) is str and binding["profile"] in source.PROFILES,
        "invalid candidate-context binding values")
    if descriptor is not None:
        _need(all(binding[key] == descriptor[key] for key in source.BINDING_FIELDS),
            "candidate-context owner context differs")
    _need(_hex(page["artifact_sha256"]) and _hex(page["slice_sha256"]),
        "invalid candidate-context digest")
    _need(all(_integer(page[key], 0, MAX_ARTIFACT_BYTES)
        for key in ("artifact_bytes", "offset", "end_offset"))
        and page["offset"] <= page["end_offset"] <= page["artifact_bytes"],
        "invalid candidate-context byte accounting")
    cursor = query["cursor"]
    _need(page["offset"] == (0 if cursor is None else cursor["offset"]),
        "candidate-context page offset differs")
    _need(type(page["content"]) is str and page["encoding"] in ("utf-8", "base64"),
        "invalid candidate-context content")
    raw = (page["content"].encode("utf-8") if page["encoding"] == "utf-8"
        else base64.b64decode(page["content"], validate=True))
    if page["encoding"] == "base64":
        _need(base64.b64encode(raw).decode("ascii") == page["content"],
            "noncanonical candidate-context base64")
    _need(len(raw) == page["end_offset"] - page["offset"]
        and hashlib.sha256(raw).hexdigest() == page["slice_sha256"],
        "candidate-context slice differs")
    if page["offset"] == 0 and page["end_offset"] == page["artifact_bytes"]:
        _need(hashlib.sha256(raw).hexdigest() == page["artifact_sha256"],
            "complete candidate-context digest differs")
    _need(type(page["exhausted"]) is bool
        and page["exhausted"] == (page["end_offset"] == page["artifact_bytes"]),
        "candidate-context completion differs")
    if page["exhausted"]:
        _need(page["next_cursor"] is None, "exhausted candidate-context has continuation")
    else:
        _validate_cursor(page["next_cursor"])
        _need(page["end_offset"] > page["offset"]
            and page["next_cursor"]["offset"] == page["end_offset"],
            "candidate-context continuation makes no progress")
        if cursor is not None:
            _need(page["next_cursor"]["request_sha256"] == cursor["request_sha256"],
                "candidate-context continuation query differs")
    return value


def request_page(descriptor_path, descriptor_sha256, *, expected_workspace_generation,
                 artifact="summary", cursor=None, max_bytes=8192, workspace=None, selection=None):
    descriptor = source.load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    query = dict(artifact=artifact, expected_workspace_generation=expected_workspace_generation,
        cursor=cursor, max_bytes=max_bytes)
    if selection is not None:
        query["selection"] = selection
    request = validate_request(dict(schema=REQUEST_SCHEMA if selection is None else REQUEST_SCHEMA_V2,
        session_id=descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "candidate-context session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30.0, remaining))
        channel.connect(descriptor["socket_path"])
        source.transport._verify_peer(channel, descriptor)
        source.transport._socket_identity(descriptor, Path(descriptor_path).parent,
            Path(descriptor_path).parent.stat().st_gid)
        source.send_frame(channel, request)
        response = source.receive_frame(channel, max_bytes=max_bytes)
        source.transport._verify_peer(channel, descriptor)
        _need(time.monotonic_ns() <= descriptor["expires_monotonic_ns"], "candidate-context session expired")
    return validate_response(response, request=request, descriptor=descriptor)


def render_page_for_model(response):
    """Keep exact artifact slices and candidate warnings after wire validation."""
    _need(type(response) is dict and type(response.get("ok")) is bool,
        "validated candidate-context response required")
    if not response["ok"]:
        _need(response.get("page") is None and response.get("reason_code") in source.REASON_CODES,
            "validated candidate-context refusal required")
        return dict(ok=False, reason_code=response["reason_code"])
    page = response["page"]
    result = dict(ok=True, **{key: page[key] for key in (
        "artifact", "artifact_sha256", "artifact_bytes", "workspace_generation", "source_state",
        "baseline_evidence_applicable", "proof_authority", "completion_authority", "offset",
        "end_offset", "encoding", "content", "exhausted", "next_cursor")})
    if page["schema"] == PAGE_SCHEMA_V2:
        result["selection"] = page["selection"]
        result["source_obligations_discharged"] = page["source_obligations_discharged"]
    return result


def guidance():
    return ("\nCandidate semantic context can be rebuilt for a captured working source generation. "
        "First obtain --working source views, then run from the allocated workspace: "
        "/opt/ipfs-supervisor/venv/bin/python -I "
        "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/"
        "worker_candidate_context_client.py --expected-workspace-generation <workspace_generation> "
        "--select --artifact catalog. Narrow the catalog with --path <path>; select capsules with "
        "--select --artifact capsules --path <path> [--symbol-id <symbol_id>], or pending obligations "
        "with --select --artifact obligations --path <path> [--function-id <function_id>] [--kind <kind>]. "
        "Filters are exact identifiers from the catalog, not proof coverage. "
        "Legacy --artifact summary, --artifact capsules, or --artifact formal without --select "
        "still fetch complete artifacts. For further slices retain the generation, artifact, and "
        "all selection filters and pass next_cursor JSON as --cursor. "
        "Decode base64 slices before concatenating bytes and parsing the artifact JSON. "
        "Every candidate artifact is advisory: baseline evidence is inapplicable, obligations remain "
        "pending, and no proof or task completion is granted. Any admitted-file edit requires a new "
        "working generation and invalidates old candidate continuations.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(source.DESCRIPTOR_FLAG, default=os.environ.get(source.DESCRIPTOR_ENV))
    parser.add_argument(source.DIGEST_FLAG, default=os.environ.get(source.DIGEST_ENV))
    parser.add_argument("--artifact", choices=tuple(dict.fromkeys(ARTIFACTS + SELECTED_ARTIFACTS)))
    parser.add_argument("--select", action="store_true",
        help="Read a selected catalog, capsules, or obligations (default: catalog)")
    parser.add_argument("--path")
    parser.add_argument("--symbol-id")
    parser.add_argument("--function-id")
    parser.add_argument("--kind")
    parser.add_argument("--expected-workspace-generation", required=True)
    parser.add_argument("--cursor")
    parser.add_argument("--max-bytes", type=int, default=8192)
    parser.add_argument("--raw-response", action="store_true",
        help="Emit the complete validated response for qualification/debugging")
    args = parser.parse_args(argv)
    try:
        selection = {key: getattr(args, key) for key in SELECTION_FIELDS}
        _need(args.select or all(value is None for value in selection.values()),
            "candidate-context filters require --select")
        artifact = args.artifact if args.artifact is not None else ("catalog" if args.select else "summary")
        if args.select:
            validate_selection(artifact, selection)
        else:
            _need(artifact in ARTIFACTS, "selected candidate-context artifact requires --select")
        _need(args.source_view_descriptor is not None and args.source_view_descriptor_sha256 is not None,
            "source descriptor required")
        result = request_page(args.source_view_descriptor, args.source_view_descriptor_sha256,
            expected_workspace_generation=args.expected_workspace_generation, artifact=artifact,
            cursor=strict_json_loads(args.cursor) if args.cursor else None, max_bytes=args.max_bytes,
            **({"selection": selection} if args.select else {}))
        print(canonical_json(result if args.raw_response else render_page_for_model(result)).decode("utf-8"))
        return 0 if result["ok"] else 2
    except (OSError, ValueError, TypeError, RuntimeError):
        print('{"reason_code":"transport_error"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
