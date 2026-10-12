"""Read-only source views for an isolated coding child, using Python ``-I``.

The fixed sibling protocol contains only standard-library imports. Descriptor
paths are public routing data; the owner admits the actual spawned descendant.
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
    from . import numerical_obligation_client as transport
else:
    _spec = importlib.util.spec_from_file_location("_source_view_transport",
        Path(__file__).resolve().with_name("numerical_obligation_client.py"))
    transport = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(transport)

DESCRIPTOR_FLAG = "--source-view-descriptor"
DIGEST_FLAG = "--source-view-descriptor-sha256"
DESCRIPTOR_ENV = "IPFS_SUPERVISOR_SOURCE_VIEW_DESCRIPTOR"
DIGEST_ENV = "IPFS_SUPERVISOR_SOURCE_VIEW_DESCRIPTOR_SHA256"
DESCRIPTOR_SCHEMA = "supervisor-worker-source-descriptor@1"
REQUEST_SCHEMA = "supervisor-worker-source-request@1"
REQUEST_SCHEMA_V2 = "supervisor-worker-source-request@2"
RESPONSE_SCHEMA = "supervisor-worker-source-response@1"
PROFILES = ("identity@1", "python-whitespace@1", "python-comments@1")
MAX_FRAME_BYTES = transport.MAX_FRAME_BYTES
MIN_PAGE_BYTES = 2048
MAX_SESSION_SECONDS = transport.MAX_SESSION_SECONDS
REASON_CODES = transport.REASON_CODES
canonical_json = transport.canonical_json
strict_json_loads = transport.strict_json_loads
send_frame = transport.send_frame
receive_frame = transport.receive_frame
process_identity = transport.process_identity
_need = transport._need
_hex = transport._hex
_integer = transport._integer
_text = transport._text
_DESCRIPTOR_FIELDS = (transport._DESCRIPTOR_FIELDS - {
    "task_cid", "context_bundle_sha256", "context_receipt_sha256", "source_tree_id"}) | {
    "scope_cid", "semantic_root_cid", "semantic_sha256", "native_prompt_sha256",
    "source_manifest_sha256", "profile"}
BINDING_FIELDS = ("task_id", "scope_cid", "semantic_root_cid", "semantic_sha256",
                  "native_prompt_sha256", "source_manifest_sha256", "profile")


def semantic_binding(encoded, profile):
    """Project only identities from the already verified semantic encoder."""
    _need(profile in PROFILES, "unsupported source view profile")
    table = encoded.table.to_dict()
    _need(encoded.freshness_checked is True, "current semantic encoder required")
    return {**{key: table[key] for key in BINDING_FIELDS if key not in {
        "profile", "source_manifest_sha256"}}, "profile": profile,
        "source_manifest_sha256": hashlib.sha256(canonical_json(table["source_manifest"])).hexdigest()}


def validate_descriptor(value):
    transport._validate_descriptor(value, schema=DESCRIPTOR_SCHEMA, fields=_DESCRIPTOR_FIELDS,
        digest_fields=("semantic_sha256", "native_prompt_sha256", "source_manifest_sha256",
                       "boundary_manifest_sha256"),
        identity_fields=("task_id", "scope_cid", "semantic_root_cid"))
    _need(type(value["profile"]) is str and value["profile"] in PROFILES,
          "unsupported source view profile")
    return value


def load_descriptor(path, digest, *, workspace=None):
    return transport.load_descriptor(path, digest, workspace=workspace, validator=validate_descriptor)


def validate_request(value):
    if type(value) is dict and value.get("schema") == "supervisor-worker-candidate-context-request@1":
        from .worker_candidate_context_client import validate_request as validate_candidate_request
        return validate_candidate_request(value)
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "query"}
        and value["schema"] in (REQUEST_SCHEMA, REQUEST_SCHEMA_V2) and _hex(value["session_id"], 32)
        and _hex(value["request_id"], 32), "invalid source request envelope")
    query = value["query"]
    working = value["schema"] == REQUEST_SCHEMA_V2
    fields = {"path", "view", "cursor", "max_bytes", "expected_source_sha256", "expected_view_sha256"}
    if working:
        fields |= {"source_state", "expected_workspace_generation"}
    _need(type(query) is dict and set(query) == fields, "invalid source query fields")
    path = query["path"]
    _need(_text(path) and not Path(path).is_absolute() and str(Path(path)) == path
          and path != "." and not any(part in ("..", ".git") for part in Path(path).parts)
          and "\\" not in path, "canonical selected source path required")
    _need(type(query["view"]) is str and query["view"] in {"compact", "original"},
          "unsupported source view")
    _need(query["cursor"] is None or type(query["cursor"]) is dict, "invalid source cursor")
    if query["cursor"] is not None:
        _validate_cursor(query["cursor"])
    pins = query["expected_source_sha256"], query["expected_view_sha256"]
    _need(all(pin is None for pin in pins) or all(_hex(pin) for pin in pins),
          "both source and compact view pins required")
    if working:
        generation = query["expected_workspace_generation"]
        _need(query["source_state"] == "working-candidate"
              and (generation is None or _hex(generation)), "invalid working source selection")
        _need((query["cursor"] is None and all(pin is None for pin in pins)) or _hex(generation),
              "working source continuation and expansion require a workspace generation")
    _need(_integer(query["max_bytes"], MIN_PAGE_BYTES, MAX_FRAME_BYTES), "invalid source page budget")
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "source request exceeds bound")
    return value


def _validate_cursor(cursor):
    _need(type(cursor) is dict and set(cursor) == {"schema", "request_sha256", "offset", "token"}
        and cursor["schema"] == "supervisor-worker-source-view-cursor@1"
        and _hex(cursor["request_sha256"]) and _hex(cursor["token"], 32)
        and _integer(cursor["offset"], 1, 1024 * 1024), "invalid source cursor fields")


def validate_response(value, *, request, descriptor=None):
    if type(request) is dict and request.get("schema") == "supervisor-worker-candidate-context-request@1":
        from .worker_candidate_context_client import validate_response as validate_candidate_response
        return validate_candidate_response(value, request=request, descriptor=descriptor)
    validate_request(request)
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "ok", "page", "reason_code"}
        and value["schema"] == RESPONSE_SCHEMA and type(value["ok"]) is bool
        and value["session_id"] == request["session_id"] and value["request_id"] == request["request_id"],
        "source response differs from request")
    _need(len(canonical_json(value)) <= request["query"]["max_bytes"], "source response exceeds budget")
    if descriptor is not None:
        validate_descriptor(descriptor)
        _need(descriptor["session_id"] == request["session_id"], "source response belongs to another session")
    if not value["ok"]:
        _need(value["page"] is None and type(value["reason_code"]) is str
              and value["reason_code"] in REASON_CODES, "invalid source refusal")
        return value
    page = value["page"]
    working = request["schema"] == REQUEST_SCHEMA_V2
    fields = {"schema", "binding", "scope_sha256", "path", "view", "source_sha256", "view_sha256",
        "manifest_sha256", "source_bytes", "view_bytes", "requested_profile", "effective_profile",
        "fallback_reason", "representation_sha256", "slice_sha256", "encoding", "content", "offset",
        "end_offset", "representation_bytes", "exhausted", "next_cursor", "canonical_source_current",
        "workspace_source_current", "source_check_profile", "read_only", "proof_authority",
        "completion_authority", "execution_authority", "mutation_authority", "omission_authority",
        "semantic_equivalence_claimed", "source_obligations_discharged"}
    if working:
        fields |= {"source_state", "workspace_generation", "baseline_source_sha256", "baseline_evidence_applicable"}
    _need(type(page) is dict and set(page) == fields and value["reason_code"] is None
        and page["schema"] == ("supervisor-worker-source-view-page@2" if working
                               else "supervisor-worker-source-view-page@1"),
          "invalid source page")
    _need(page.get("path") == request["query"]["path"] and page.get("view") == request["query"]["view"],
          "source page selection differs")
    _need(all(page[key] is False for key in ("proof_authority", "mutation_authority", "completion_authority",
        "execution_authority", "omission_authority", "semantic_equivalence_claimed"))
        and type(page["source_obligations_discharged"]) is int and page["source_obligations_discharged"] == 0
        and page["read_only"] is True and page["canonical_source_current"] is True
        and page["workspace_source_current"] is True
        and page["source_check_profile"] == ("owner-working-source-candidate@1" if working
                                              else "owner-admitted-exact-source@1"),
          "source view grants no authority")
    if working:
        _need(page["source_state"] == "working-candidate" and _hex(page["workspace_generation"])
              and _hex(page["baseline_source_sha256"]) and page["baseline_evidence_applicable"] is False,
              "invalid working source page identity")
        generation = request["query"]["expected_workspace_generation"]
        _need(generation is None or generation == page["workspace_generation"],
              "working source generation differs")
    _need(type(page["binding"]) is dict and set(page["binding"]) == set(BINDING_FIELDS),
          "invalid source page binding")
    if descriptor is not None:
        _need(all(page["binding"][key] == descriptor[key] for key in BINDING_FIELDS),
              "source page owner context differs")
    _need(page["requested_profile"] == page["binding"]["profile"]
          and page["requested_profile"] in PROFILES and page["effective_profile"] in PROFILES
          and (page["fallback_reason"] is None or _text(page["fallback_reason"], 256)),
          "source page profile differs")
    for key in ("scope_sha256", "source_sha256", "view_sha256", "manifest_sha256",
                "representation_sha256", "slice_sha256"):
        _need(_hex(page[key]), "invalid source page digest")
    for key in ("source_bytes", "view_bytes", "representation_bytes", "offset", "end_offset"):
        _need(_integer(page[key], 0, 1024 * 1024), "invalid source page size")
    kind = "source" if page["view"] == "original" else "view"
    _need(page["representation_sha256"] == page[kind + "_sha256"]
        and page["representation_bytes"] == page[kind + "_bytes"]
        and page["offset"] <= page["end_offset"] <= page["representation_bytes"],
        "source representation accounting differs")
    cursor = request["query"]["cursor"]
    _need(page["offset"] == (0 if cursor is None else cursor["offset"]), "source page offset differs")
    _need(type(page["content"]) is str and page["encoding"] in ("utf-8", "base64"), "invalid source content")
    raw = (page["content"].encode("utf-8") if page["encoding"] == "utf-8"
           else base64.b64decode(page["content"], validate=True))
    if page["encoding"] == "base64":
        _need(base64.b64encode(raw).decode("ascii") == page["content"], "noncanonical source base64")
    _need(len(raw) == page["end_offset"] - page["offset"]
          and hashlib.sha256(raw).hexdigest() == page["slice_sha256"], "source slice digest differs")
    if page["offset"] == 0 and page["end_offset"] == page["representation_bytes"]:
        _need(hashlib.sha256(raw).hexdigest() == page["representation_sha256"], "complete source digest differs")
    _need(type(page["exhausted"]) is bool and page["exhausted"] == (page["end_offset"] == page["representation_bytes"]),
          "source page completion differs")
    if page["exhausted"]:
        _need(page["next_cursor"] is None, "exhausted source has continuation")
    else:
        _validate_cursor(page["next_cursor"])
        _need(page["end_offset"] > page["offset"] and page["next_cursor"]["offset"] == page["end_offset"],
              "source continuation makes no progress")
    for key in ("source_sha256", "view_sha256"):
        pin = request["query"]["expected_" + key]
        _need(pin is None or pin == page[key], "source expansion pin differs")
    return value


def request_page(descriptor_path, descriptor_sha256, *, path, view="compact", cursor=None,
                 max_bytes=8192, expected_source_sha256=None, expected_view_sha256=None, workspace=None,
                 working=False, expected_workspace_generation=None):
    _need(type(working) is bool and (working or expected_workspace_generation is None),
          "workspace generation requires working source selection")
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    query = dict(path=path, view=view, cursor=cursor, max_bytes=max_bytes,
        expected_source_sha256=expected_source_sha256, expected_view_sha256=expected_view_sha256)
    if working:
        query.update(source_state="working-candidate", expected_workspace_generation=expected_workspace_generation)
    request = validate_request(dict(schema=REQUEST_SCHEMA_V2 if working else REQUEST_SCHEMA,
        session_id=descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "source session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30.0, remaining))
        channel.connect(descriptor["socket_path"])
        transport._verify_peer(channel, descriptor)
        transport._socket_identity(descriptor, Path(descriptor_path).parent, Path(descriptor_path).parent.stat().st_gid)
        send_frame(channel, request)
        response = receive_frame(channel, max_bytes=max_bytes)
        transport._verify_peer(channel, descriptor)
        _need(time.monotonic_ns() <= descriptor["expires_monotonic_ns"], "source session expired")
    return validate_response(response, request=request, descriptor=descriptor)


def render_page_for_model(response):
    """Project an already validated response into bounded tool-visible JSON.

    ``request_page`` verifies the complete private transport envelope first.
    Expansion pins, exact slice bytes and continuation remain visible; repeated
    session/authority metadata stays in the owner audit and optional raw reply.
    """
    _need(type(response) is dict and type(response.get("ok")) is bool,
          "validated source response required")
    if not response["ok"]:
        _need(response.get("page") is None and response.get("reason_code") in REASON_CODES,
              "validated source refusal required")
        return dict(ok=False, reason_code=response["reason_code"])
    page = response["page"]
    fields = ("path", "view", "source_sha256", "view_sha256", "offset", "end_offset",
              "exhausted", "next_cursor", "encoding", "content")
    if page["schema"] == "supervisor-worker-source-view-page@2":
        fields += ("source_state", "workspace_generation", "baseline_source_sha256", "baseline_evidence_applicable")
    return dict(ok=True, **{key: page[key] for key in fields})


def guidance(profile):
    _need(profile in PROFILES, "unsupported source view profile")
    return ("\nRead-only indexed source views are available for the paths in the semantic manifest. "
        "From the allocated workspace run: /opt/ipfs-supervisor/venv/bin/python -I "
        "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py "
        "--path <path> --view compact. Use --view original to read exact source before editing; "
        "pass --expected-source-sha256 and --expected-view-sha256 from the compact page to bind expansion. "
        "For another page pass its next_cursor JSON as --cursor. Edit original workspace files, never "
        "the compact response. After edits, use --working with no cursor or pins to request a fresh "
        "working-candidate first page. For its continuation or exact-source expansion, keep --working "
        "and pass --expected-workspace-generation plus both source/view pins from that candidate page. "
        "A changed workspace generation refuses old continuations and pins; request a fresh first page. "
        "Working snapshots do not refresh or prove the accepted semantic index; treat prior capsules/proofs as stale. "
        "To rebuild advisory capsules and pending formulas for that generation, run the sibling "
        "worker_candidate_context_client.py with --expected-workspace-generation <generation> "
        "--artifact summary (or capsules/formal for exact JSON artifact pages). Keep that generation "
        "and pass next_cursor as --cursor for continuation. Candidate context is deterministic, "
        "may have unsupported coverage, and runs no solver or Lake build. It grants no proof authority. "
        "Profile: " + profile + ". "
        "Views confer no proof or completion.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(DESCRIPTOR_FLAG, default=os.environ.get(DESCRIPTOR_ENV))
    parser.add_argument(DIGEST_FLAG, default=os.environ.get(DIGEST_ENV))
    parser.add_argument("--path", required=True)
    parser.add_argument("--view", choices=("compact", "original"), default="compact")
    parser.add_argument("--cursor")
    parser.add_argument("--max-bytes", type=int, default=8192)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-view-sha256")
    parser.add_argument("--working", action="store_true",
        help="Read a current working candidate without refreshing semantic evidence")
    parser.add_argument("--expected-workspace-generation")
    parser.add_argument("--raw-response", action="store_true",
        help="Emit the complete validated transport envelope for qualification/debugging")
    args = parser.parse_args(argv)
    try:
        _need(args.source_view_descriptor is not None and args.source_view_descriptor_sha256 is not None,
              "source descriptor required")
        result = request_page(args.source_view_descriptor, args.source_view_descriptor_sha256,
            path=args.path, view=args.view, cursor=strict_json_loads(args.cursor) if args.cursor else None,
            max_bytes=args.max_bytes, expected_source_sha256=args.expected_source_sha256,
            expected_view_sha256=args.expected_view_sha256, working=args.working,
            expected_workspace_generation=args.expected_workspace_generation)
        print(canonical_json(result if args.raw_response else render_page_for_model(result)).decode("utf-8"))
        return 0 if result["ok"] else 2
    except (OSError, ValueError, TypeError, RuntimeError):
        print('{"reason_code":"transport_error"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
