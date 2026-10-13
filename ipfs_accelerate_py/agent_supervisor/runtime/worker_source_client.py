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
DESCRIPTOR_SCHEMA_V2 = "supervisor-worker-source-descriptor@2"
REQUEST_SCHEMA = "supervisor-worker-source-request@1"
REQUEST_SCHEMA_V2 = "supervisor-worker-source-request@2"
REQUEST_SCHEMA_V3 = "supervisor-worker-source-request@3"
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
_DESCRIPTOR_FIELDS_V2 = _DESCRIPTOR_FIELDS | {"view_policy_binding_sha256", "source_scope_sha256"}
BINDING_FIELDS = ("task_id", "scope_cid", "semantic_root_cid", "semantic_sha256",
                  "native_prompt_sha256", "source_manifest_sha256", "profile")
_READ_QUERY_FIELDS = {"path", "view", "cursor", "max_bytes", "expected_source_sha256",
                      "expected_view_sha256"}
_AUTHORITY_FLAGS = ("proof_authority", "mutation_authority", "completion_authority",
                    "execution_authority", "omission_authority", "semantic_equivalence_claimed")


def semantic_binding(encoded, profile):
    """Project only identities from the already verified semantic encoder."""
    _need(profile in PROFILES, "unsupported source view profile")
    table = encoded.table.to_dict()
    _need(encoded.freshness_checked is True, "current semantic encoder required")
    return {**{key: table[key] for key in BINDING_FIELDS if key not in {
        "profile", "source_manifest_sha256"}}, "profile": profile,
        "source_manifest_sha256": hashlib.sha256(canonical_json(table["source_manifest"])).hexdigest()}


def validate_descriptor(value):
    shared = type(value) is dict and value.get("schema") == DESCRIPTOR_SCHEMA_V2
    digests = ("semantic_sha256", "native_prompt_sha256", "source_manifest_sha256",
               "boundary_manifest_sha256")
    if shared:
        digests += ("view_policy_binding_sha256", "source_scope_sha256")
    transport._validate_descriptor(value,
        schema=DESCRIPTOR_SCHEMA_V2 if shared else DESCRIPTOR_SCHEMA,
        fields=_DESCRIPTOR_FIELDS_V2 if shared else _DESCRIPTOR_FIELDS,
        digest_fields=digests,
        identity_fields=("task_id", "scope_cid", "semantic_root_cid"))
    _need(type(value["profile"]) is str and value["profile"] in PROFILES,
          "unsupported source view profile")
    return value


def load_descriptor(path, digest, *, workspace=None):
    return transport.load_descriptor(path, digest, workspace=workspace, validator=validate_descriptor)


def validate_request(value):
    if type(value) is dict and value.get("schema") in {
            "supervisor-worker-candidate-context-request@1", "supervisor-worker-candidate-context-request@2"}:
        from .worker_candidate_context_client import validate_request as validate_candidate_request
        return validate_candidate_request(value)
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "query"}
        and value["schema"] in (REQUEST_SCHEMA, REQUEST_SCHEMA_V2, REQUEST_SCHEMA_V3)
        and _hex(value["session_id"], 32)
        and _hex(value["request_id"], 32), "invalid source request envelope")
    query = value["query"]
    shared = value["schema"] == REQUEST_SCHEMA_V3
    _need(type(query) is dict, "invalid source query fields")
    if shared:
        _need(type(query.get("operation")) is str and query["operation"] in {"discover", "read"},
              "invalid source operation")
        if query["operation"] == "discover":
            _need(set(query) == {"operation", "view_policy_binding_sha256", "cursor", "max_bytes"}
                  and (query["view_policy_binding_sha256"] is None or _hex(query["view_policy_binding_sha256"])),
                  "invalid source discovery fields")
            _validate_query_cursor(query)
            _need(query["cursor"] is None or query["cursor"]["offset"] < 64,
                  "discovery continuation exceeds source population")
            return _validate_request_budget(value)
    working = value["schema"] == REQUEST_SCHEMA_V2 or (shared and query.get("source_state") == "working-candidate")
    fields = set(_READ_QUERY_FIELDS)
    if working or shared:
        fields |= {"source_state", "expected_workspace_generation"}
    if shared:
        fields |= {"operation", "view_policy_binding_sha256"}
    _need(set(query) == fields, "invalid source query fields")
    if shared:
        _need(_hex(query["view_policy_binding_sha256"])
              and type(query["source_state"]) is str
              and query["source_state"] in {"baseline", "working-candidate"},
              "invalid shared source selection")
        if not working:
            _need(query["expected_workspace_generation"] is None,
                  "baseline source selection has no working generation")
    _validate_source_path(query["path"])
    _need(type(query["view"]) is str and query["view"] in {"compact", "original"},
          "unsupported source view")
    _validate_query_cursor(query)
    pins = query["expected_source_sha256"], query["expected_view_sha256"]
    _need(all(pin is None for pin in pins) or all(_hex(pin) for pin in pins),
          "both source and compact view pins required")
    if working:
        generation = query["expected_workspace_generation"]
        _need(query["source_state"] == "working-candidate"
              and (generation is None or _hex(generation)), "invalid working source selection")
        _need((query["cursor"] is None and all(pin is None for pin in pins)) or _hex(generation),
              "working source continuation and expansion require a workspace generation")
    return _validate_request_budget(value)


def _validate_source_path(path):
    _need(_text(path) and not Path(path).is_absolute() and str(Path(path)) == path
          and path != "." and not any(part in ("..", ".git") for part in Path(path).parts)
          and "\\" not in path, "canonical selected source path required")


def _validate_query_cursor(query):
    _need(query["cursor"] is None or type(query["cursor"]) is dict, "invalid source cursor")
    if query["cursor"] is not None:
        _validate_cursor(query["cursor"])


def _validate_request_budget(value):
    _need(_integer(value["query"]["max_bytes"], MIN_PAGE_BYTES, MAX_FRAME_BYTES), "invalid source page budget")
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "source request exceeds bound")
    return value


def _validate_cursor(cursor):
    _need(type(cursor) is dict and set(cursor) == {"schema", "request_sha256", "offset", "token"}
        and cursor["schema"] == "supervisor-worker-source-view-cursor@1"
        and _hex(cursor["request_sha256"]) and _hex(cursor["token"], 32)
        and _integer(cursor["offset"], 1, 1024 * 1024), "invalid source cursor fields")


def validate_response(value, *, request, descriptor=None):
    if type(request) is dict and request.get("schema") in {
            "supervisor-worker-candidate-context-request@1", "supervisor-worker-candidate-context-request@2"}:
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
        policy = request["query"].get("view_policy_binding_sha256")
        if descriptor["schema"] == DESCRIPTOR_SCHEMA_V2:
            _need(request["schema"] == REQUEST_SCHEMA_V3
                  and policy == descriptor["view_policy_binding_sha256"],
                  "source request downgrades or differs from shared descriptor policy")
        else:
            _need(policy is None, "legacy source descriptor cannot select shared policy")
    if not value["ok"]:
        _need(value["page"] is None and type(value["reason_code"]) is str
              and value["reason_code"] in REASON_CODES, "invalid source refusal")
        return value
    page = value["page"]
    shared = request["schema"] == REQUEST_SCHEMA_V3 and request["query"]["view_policy_binding_sha256"] is not None
    if request["schema"] == REQUEST_SCHEMA_V3 and request["query"]["operation"] == "discover":
        return _validate_discovery_response(value, request=request, descriptor=descriptor)
    working = request["schema"] == REQUEST_SCHEMA_V2 or (
        request["schema"] == REQUEST_SCHEMA_V3 and request["query"]["source_state"] == "working-candidate")
    fields = {"schema", "binding", "scope_sha256", "path", "view", "source_sha256", "view_sha256",
        "manifest_sha256", "source_bytes", "view_bytes", "requested_profile", "effective_profile",
        "fallback_reason", "representation_sha256", "slice_sha256", "encoding", "content", "offset",
        "end_offset", "representation_bytes", "exhausted", "next_cursor", "canonical_source_current",
        "workspace_source_current", "source_check_profile", "read_only", "proof_authority",
        "completion_authority", "execution_authority", "mutation_authority", "omission_authority",
        "semantic_equivalence_claimed", "source_obligations_discharged"}
    if working:
        fields |= {"source_state", "workspace_generation", "baseline_source_sha256", "baseline_evidence_applicable"}
    if shared:
        fields.add("view_policy_binding_sha256")
    schema = ("supervisor-worker-source-view-page@4" if working else "supervisor-worker-source-view-page@3") if shared else (
        "supervisor-worker-source-view-page@2" if working else "supervisor-worker-source-view-page@1")
    _need(type(page) is dict and set(page) == fields and value["reason_code"] is None
        and page["schema"] == schema,
          "invalid source page")
    if shared:
        _need(_hex(page["view_policy_binding_sha256"])
              and page["view_policy_binding_sha256"] == request["query"]["view_policy_binding_sha256"],
              "source page policy binding differs")
        if descriptor is not None and not working:
            _need(page["scope_sha256"] == descriptor["source_scope_sha256"],
                  "baseline source scope differs from descriptor")
    _need(page.get("path") == request["query"]["path"] and page.get("view") == request["query"]["view"],
          "source page selection differs")
    _need(all(page[key] is False for key in _AUTHORITY_FLAGS)
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


def _validate_discovery_response(value, *, request, descriptor):
    page, query = value["page"], request["query"]
    shared = query["view_policy_binding_sha256"] is not None
    fields = {"schema", "scope_sha256", "total_count", "returned_count", "offset", "end_offset",
        "rows", "exhausted", "next_cursor", "read_only", "source_obligations_discharged", *_AUTHORITY_FLAGS}
    if shared:
        fields.add("view_policy_binding_sha256")
    _need(type(page) is dict and set(page) == fields and value["reason_code"] is None
          and page["schema"] == ("supervisor-worker-source-view-discovery@2" if shared
                                  else "supervisor-worker-source-view-discovery@1"),
          "invalid source discovery page")
    _need(_hex(page["scope_sha256"]) and page["read_only"] is True
          and all(page[key] is False for key in _AUTHORITY_FLAGS)
          and type(page["source_obligations_discharged"]) is int
          and page["source_obligations_discharged"] == 0,
          "source discovery grants no authority")
    if shared:
        _need(_hex(page["view_policy_binding_sha256"])
              and page["view_policy_binding_sha256"] == query["view_policy_binding_sha256"],
              "source discovery policy differs")
        if descriptor is not None:
            _need(page["scope_sha256"] == descriptor["source_scope_sha256"],
                  "source discovery scope differs from descriptor")
    _need(_integer(page["total_count"], 1, 64) and _integer(page["returned_count"], 1, 64)
          and _integer(page["offset"], 0, 63) and _integer(page["end_offset"], 1, 64)
          and page["end_offset"] == page["offset"] + page["returned_count"]
          and page["end_offset"] <= page["total_count"]
          and type(page["rows"]) is list and len(page["rows"]) == page["returned_count"],
          "source discovery population differs")
    cursor = query["cursor"]
    _need(page["offset"] == (0 if cursor is None else cursor["offset"]),
          "source discovery offset differs")
    paths, total_bytes = [], 0
    for row in page["rows"]:
        _need(type(row) is dict and set(row) == {"path", "source_sha256", "view_sha256", "manifest_sha256",
              "source_bytes", "view_bytes", "effective_profile"}, "invalid source discovery row")
        _validate_source_path(row["path"])
        paths.append(row["path"])
        _need(all(_hex(row[key]) for key in ("source_sha256", "view_sha256", "manifest_sha256"))
              and _integer(row["source_bytes"], 0, 1024 * 1024)
              and _integer(row["view_bytes"], 0, row["source_bytes"])
              and type(row["effective_profile"]) is str and row["effective_profile"] in PROFILES,
              "invalid source discovery row identity")
        total_bytes += row["source_bytes"]
    _need(paths == sorted(set(paths)) and total_bytes <= 1024 * 1024,
          "source discovery rows are noncanonical or exceed source bounds")
    _need(type(page["exhausted"]) is bool
          and page["exhausted"] == (page["end_offset"] == page["total_count"]),
          "source discovery completion differs")
    if page["exhausted"]:
        _need(page["next_cursor"] is None, "exhausted source discovery has continuation")
    else:
        _validate_cursor(page["next_cursor"])
        _need(page["next_cursor"]["offset"] == page["end_offset"],
              "source discovery continuation makes no progress")
    return value


def request_page(descriptor_path, descriptor_sha256, *, path, view="compact", cursor=None,
                 max_bytes=8192, expected_source_sha256=None, expected_view_sha256=None, workspace=None,
                 working=False, expected_workspace_generation=None):
    _need(type(working) is bool and (working or expected_workspace_generation is None),
          "workspace generation requires working source selection")
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    query = dict(path=path, view=view, cursor=cursor, max_bytes=max_bytes,
        expected_source_sha256=expected_source_sha256, expected_view_sha256=expected_view_sha256)
    shared = descriptor["schema"] == DESCRIPTOR_SCHEMA_V2
    if shared:
        query.update(operation="read", view_policy_binding_sha256=descriptor["view_policy_binding_sha256"],
            source_state="working-candidate" if working else "baseline",
            expected_workspace_generation=expected_workspace_generation)
    elif working:
        query.update(source_state="working-candidate", expected_workspace_generation=expected_workspace_generation)
    request = validate_request(dict(schema=REQUEST_SCHEMA_V3 if shared else REQUEST_SCHEMA_V2 if working else REQUEST_SCHEMA,
        session_id=descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))
    return _exchange(descriptor_path, descriptor, request)


def request_discovery(descriptor_path, descriptor_sha256, *, cursor=None, max_bytes=8192, workspace=None):
    """Discover only the already admitted baseline paths and their representation pins."""
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    request = validate_request(dict(schema=REQUEST_SCHEMA_V3,
        session_id=descriptor["session_id"], request_id=uuid.uuid4().hex,
        query=dict(operation="discover", view_policy_binding_sha256=descriptor.get("view_policy_binding_sha256"),
                   cursor=cursor, max_bytes=max_bytes)))
    return _exchange(descriptor_path, descriptor, request)


def _exchange(descriptor_path, descriptor, request):
    """Use the existing authenticated, deadline-bound source socket lifecycle."""
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "source session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30.0, remaining))
        channel.connect(descriptor["socket_path"])
        transport._verify_peer(channel, descriptor)
        transport._socket_identity(descriptor, Path(descriptor_path).parent, Path(descriptor_path).parent.stat().st_gid)
        send_frame(channel, request)
        response = receive_frame(channel, max_bytes=request["query"]["max_bytes"])
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
    if page["schema"] in {"supervisor-worker-source-view-discovery@1", "supervisor-worker-source-view-discovery@2"}:
        fields = ("total_count", "returned_count", "offset", "end_offset", "rows", "exhausted", "next_cursor")
        if page["schema"] == "supervisor-worker-source-view-discovery@2":
            fields += ("view_policy_binding_sha256",)
        return dict(ok=True, **{key: page[key] for key in fields})
    fields = ("path", "view", "source_sha256", "view_sha256", "offset", "end_offset",
              "exhausted", "next_cursor", "encoding", "content")
    if page["schema"] in {"supervisor-worker-source-view-page@2", "supervisor-worker-source-view-page@4"}:
        fields += ("source_state", "workspace_generation", "baseline_source_sha256", "baseline_evidence_applicable")
    if page["schema"] in {"supervisor-worker-source-view-page@3", "supervisor-worker-source-view-page@4"}:
        fields += ("view_policy_binding_sha256",)
    return dict(ok=True, **{key: page[key] for key in fields})


def guidance(profile):
    _need(profile in PROFILES, "unsupported source view profile")
    return ("\nRead-only indexed source views are available for the paths in the semantic manifest. "
        "From the allocated workspace run: /opt/ipfs-supervisor/venv/bin/python -I "
        "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py "
        "--discover to list admitted paths and exact source/view pins. Read a selected path with "
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
        "--select --artifact catalog for compact discovery. Then use --select --artifact capsules "
        "--path <path> [--symbol-id <id>], or --select --artifact obligations --path <path> "
        "[--function-id <id>] [--kind <kind>] for exact native subsets. Filters do not establish "
        "task relevance or complete proof coverage. Omit --select and use --artifact summary, "
        "capsules or formal for full JSON artifacts. Keep generation, filters and page budget fixed "
        "and pass next_cursor as --cursor for continuation. Candidate context is deterministic, "
        "may have unsupported coverage, and runs no solver or Lake build. It grants no proof authority. "
        "Profile: " + profile + ". "
        "Views confer no proof or completion.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(DESCRIPTOR_FLAG, default=os.environ.get(DESCRIPTOR_ENV))
    parser.add_argument(DIGEST_FLAG, default=os.environ.get(DIGEST_ENV))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--path")
    selection.add_argument("--discover", action="store_true",
        help="List only the owner-admitted baseline paths and source/view pins")
    parser.add_argument("--view", choices=("compact", "original"))
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
        cursor = strict_json_loads(args.cursor) if args.cursor else None
        if args.discover:
            _need(args.view is None and args.expected_source_sha256 is None
                  and args.expected_view_sha256 is None and not args.working
                  and args.expected_workspace_generation is None,
                  "source discovery cannot select source representation, pins or working generation")
            result = request_discovery(args.source_view_descriptor, args.source_view_descriptor_sha256,
                cursor=cursor, max_bytes=args.max_bytes)
        else:
            result = request_page(args.source_view_descriptor, args.source_view_descriptor_sha256,
                path=args.path, view=args.view or "compact", cursor=cursor,
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
