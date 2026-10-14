"""Read-only, generation-pinned retrieval for a separate candidate-round session.

This standard-library client runs directly with Python ``-I``. Its descriptor
retains baseline identity fields separately from candidate snapshot identities.
It cannot discover baseline sources, edit files, refresh a generation, or admit
a coding child. Ordinary source clients deliberately reject this descriptor.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time
import uuid

if __package__:
    from . import worker_source_client as source
    from . import worker_candidate_context_client as candidate
else:
    def _sibling(name):
        spec = importlib.util.spec_from_file_location("_candidate_round_" + name,
            Path(__file__).resolve().with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    source = _sibling("worker_source_client")
    candidate = _sibling("worker_candidate_context_client")

transport = source.transport
DESCRIPTOR_SCHEMA = "supervisor-worker-candidate-round-descriptor@1"
DESCRIPTOR_FLAG = "--candidate-round-descriptor"
DIGEST_FLAG = "--candidate-round-descriptor-sha256"
DESCRIPTOR_ENV = "IPFS_SUPERVISOR_CANDIDATE_ROUND_DESCRIPTOR"
DIGEST_ENV = "IPFS_SUPERVISOR_CANDIDATE_ROUND_DESCRIPTOR_SHA256"
REQUEST_SCHEMA_V2, REQUEST_SCHEMA_V3 = source.REQUEST_SCHEMA_V2, source.REQUEST_SCHEMA_V3
RESPONSE_SCHEMA = source.RESPONSE_SCHEMA
BINDING_FIELDS = source.BINDING_FIELDS
PROFILES = source.PROFILES
MAX_FRAME_BYTES, MIN_PAGE_BYTES = source.MAX_FRAME_BYTES, source.MIN_PAGE_BYTES
MAX_SESSION_SECONDS, REASON_CODES = source.MAX_SESSION_SECONDS, source.REASON_CODES
canonical_json, strict_json_loads = source.canonical_json, source.strict_json_loads
send_frame, receive_frame, process_identity = source.send_frame, source.receive_frame, source.process_identity
_need, _hex, _integer, _text = source._need, source._hex, source._integer, source._text
_DESCRIPTOR_FIELDS = source._DESCRIPTOR_FIELDS | {
    "view_policy_binding_sha256", "source_scope_sha256", "workspace_generation",
    "round_context_sha256", "prior_session_id", "prior_descriptor_sha256", "round_number",
    "candidate_scope_cid", "candidate_semantic_root_cid"}
_CANDIDATE_REQUESTS = (candidate.REQUEST_SCHEMA, candidate.REQUEST_SCHEMA_V2)


def validate_descriptor(value):
    transport._validate_descriptor(value, schema=DESCRIPTOR_SCHEMA, fields=_DESCRIPTOR_FIELDS,
        digest_fields=("semantic_sha256", "native_prompt_sha256", "source_manifest_sha256",
            "boundary_manifest_sha256", "source_scope_sha256", "workspace_generation",
            "round_context_sha256", "prior_descriptor_sha256"),
        identity_fields=("task_id", "scope_cid", "semantic_root_cid",
            "candidate_scope_cid", "candidate_semantic_root_cid"))
    _need(type(value["profile"]) is str and value["profile"] in PROFILES,
        "unsupported candidate-round source profile")
    _need(value["view_policy_binding_sha256"] is None or _hex(value["view_policy_binding_sha256"]),
        "invalid candidate-round source policy")
    _need(_hex(value["prior_session_id"], 32) and value["prior_session_id"] != value["session_id"]
        and type(value["round_number"]) is int and value["round_number"] == 2,
        "fresh second-round session required")
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "candidate-round descriptor exceeds bound")
    return value


def load_descriptor(path, digest, *, workspace=None):
    return transport.load_descriptor(path, digest, workspace=workspace, validator=validate_descriptor)


def validate_request(value):
    _need(type(value) is dict and type(value.get("schema")) is str,
        "candidate-round request required")
    if value["schema"] in _CANDIDATE_REQUESTS:
        return candidate.validate_request(value)
    _need(value["schema"] in (REQUEST_SCHEMA_V2, REQUEST_SCHEMA_V3),
        "candidate round supports only working-source and candidate-context reads")
    source.validate_request(value)
    query = value["query"]
    _need(query.get("source_state") == "working-candidate"
        and _hex(query.get("expected_workspace_generation"))
        and (value["schema"] != REQUEST_SCHEMA_V3 or query["operation"] == "read"),
        "candidate-round reads require the exact working generation")
    return value


def _request_binding(request, descriptor):
    validate_descriptor(descriptor)
    validate_request(request)
    _need(request["session_id"] == descriptor["session_id"]
        and request["query"]["expected_workspace_generation"] == descriptor["workspace_generation"],
        "candidate-round request differs from admitted session or generation")
    if request["schema"] not in _CANDIDATE_REQUESTS:
        policy = descriptor["view_policy_binding_sha256"]
        if policy is None:
            _need(request["schema"] == REQUEST_SCHEMA_V2,
                "candidate-round source request changes the legacy policy")
        else:
            _need(request["schema"] == REQUEST_SCHEMA_V3
                and request["query"]["view_policy_binding_sha256"] == policy,
                "candidate-round source request downgrades or changes its policy")


def validate_response(value, *, request, descriptor=None):
    validate_request(request)
    if descriptor is not None:
        _request_binding(request, descriptor)
    # Reuse the unchanged wire grammar, without pretending this new descriptor
    # is an ordinary baseline source descriptor. Candidate imports also work -I.
    validator = candidate.validate_response if request["schema"] in _CANDIDATE_REQUESTS else source.validate_response
    validator(value, request=request, descriptor=None)
    if value["ok"] and descriptor is not None:
        page = value["page"]
        _need(all(page["binding"][key] == descriptor[key] for key in BINDING_FIELDS)
            and page["workspace_generation"] == descriptor["workspace_generation"],
            "candidate-round response differs from baseline identity or candidate generation")
        if request["schema"] not in _CANDIDATE_REQUESTS:
            _need(page.get("view_policy_binding_sha256") == descriptor["view_policy_binding_sha256"],
                "candidate-round response policy differs")
    if value["ok"]:
        cursor, following = request["query"]["cursor"], value["page"]["next_cursor"]
        if cursor is not None and following is not None:
            _need(cursor["request_sha256"] == following["request_sha256"],
                "candidate-round continuation changed its query")
    return value


def _exchange(descriptor_path, descriptor, request):
    _request_binding(request, descriptor)
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "candidate-round session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30., remaining))
        channel.connect(descriptor["socket_path"])
        transport._verify_peer(channel, descriptor)
        parent = Path(descriptor_path).parent
        transport._socket_identity(descriptor, parent, parent.stat().st_gid)
        send_frame(channel, request)
        response = receive_frame(channel, max_bytes=request["query"]["max_bytes"])
        transport._verify_peer(channel, descriptor)
        _need(time.monotonic_ns() <= descriptor["expires_monotonic_ns"], "candidate-round session expired")
    return validate_response(response, request=request, descriptor=descriptor)


def request_page(descriptor_path, descriptor_sha256, *, path, expected_workspace_generation,
                 view="compact", cursor=None, max_bytes=8192, expected_source_sha256=None,
                 expected_view_sha256=None, workspace=None):
    """Read only this session's pinned candidate; no implicit recapture is allowed."""
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    query = dict(path=path, view=view, cursor=cursor, max_bytes=max_bytes,
        expected_source_sha256=expected_source_sha256, expected_view_sha256=expected_view_sha256,
        source_state="working-candidate", expected_workspace_generation=expected_workspace_generation)
    shared = descriptor["view_policy_binding_sha256"] is not None
    if shared:
        query.update(operation="read", view_policy_binding_sha256=descriptor["view_policy_binding_sha256"])
    request = dict(schema=REQUEST_SCHEMA_V3 if shared else REQUEST_SCHEMA_V2,
        session_id=descriptor["session_id"], request_id=uuid.uuid4().hex, query=query)
    return _exchange(descriptor_path, descriptor, request)


def request_context_page(descriptor_path, descriptor_sha256, *, expected_workspace_generation,
                         artifact="summary", cursor=None, max_bytes=8192, workspace=None, selection=None):
    """Read full or selected advisory candidate artifacts for the pinned snapshot."""
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    query = dict(artifact=artifact, expected_workspace_generation=expected_workspace_generation,
        cursor=cursor, max_bytes=max_bytes)
    if selection is not None:
        query["selection"] = selection
    request = dict(schema=candidate.REQUEST_SCHEMA if selection is None else candidate.REQUEST_SCHEMA_V2,
        session_id=descriptor["session_id"], request_id=uuid.uuid4().hex, query=query)
    return _exchange(descriptor_path, descriptor, request)


request_source_page = request_page
request_candidate_page = request_context_page


def render_page_for_model(response):
    if response.get("ok") is True and response["page"]["schema"] in (candidate.PAGE_SCHEMA, candidate.PAGE_SCHEMA_V2):
        return candidate.render_page_for_model(response)
    return source.render_page_for_model(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(DESCRIPTOR_FLAG, default=os.environ.get(DESCRIPTOR_ENV))
    parser.add_argument(DIGEST_FLAG, default=os.environ.get(DIGEST_ENV))
    parser.add_argument("--expected-workspace-generation", required=True)
    parser.add_argument("--path")
    parser.add_argument("--view", choices=("original", "compact"))
    parser.add_argument("--artifact", choices=tuple(dict.fromkeys(candidate.ARTIFACTS + candidate.SELECTED_ARTIFACTS)))
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--symbol-id")
    parser.add_argument("--function-id")
    parser.add_argument("--kind")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-view-sha256")
    parser.add_argument("--cursor")
    parser.add_argument("--max-bytes", type=int, default=8192)
    parser.add_argument("--raw-response", action="store_true")
    args = parser.parse_args(argv)
    try:
        _need(args.candidate_round_descriptor is not None and args.candidate_round_descriptor_sha256 is not None,
            "candidate-round descriptor required")
        common = dict(expected_workspace_generation=args.expected_workspace_generation,
            cursor=strict_json_loads(args.cursor) if args.cursor else None, max_bytes=args.max_bytes)
        filters = {key: getattr(args, key) for key in candidate.SELECTION_FIELDS}
        if args.artifact is not None or args.select:
            _need(args.view is None and args.expected_source_sha256 is None and args.expected_view_sha256 is None,
                "candidate-context reads cannot select source representations")
            _need(args.select or all(value is None for value in filters.values()),
                "candidate-context filters require --select")
            result = request_context_page(args.candidate_round_descriptor, args.candidate_round_descriptor_sha256,
                artifact=args.artifact or "catalog", **common,
                **({"selection": filters} if args.select else {}))
        else:
            _need(args.path is not None and all(filters[key] is None for key in filters if key != "path"),
                "source reads require a path and no candidate-context filters")
            result = request_page(args.candidate_round_descriptor, args.candidate_round_descriptor_sha256,
                path=args.path, view=args.view or "compact", expected_source_sha256=args.expected_source_sha256,
                expected_view_sha256=args.expected_view_sha256, **common)
        print(canonical_json(result if args.raw_response else render_page_for_model(result)).decode("utf-8"))
        return 0 if result["ok"] else 2
    except (OSError, ValueError, TypeError, RuntimeError):
        print('{"reason_code":"transport_error"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
