"""Authenticated host-owner candidate dispatch and advisory reads, usable ``-I``.

The descriptor is routing data. Only an exchange with its protected, live owner
can issue a local typed input; neither JSON nor a retained checkpoint restores
native authority. The endpoint consumes one provider-dispatch nomination, not a
proof, completion, retry, or production-container grant.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import socket
import sys
import threading
import time
import uuid
import weakref

if __package__:
    from . import worker_candidate_round_client as round_client
else:
    _spec = importlib.util.spec_from_file_location("_dispatch_round_client",
        Path(__file__).resolve().with_name("worker_candidate_round_client.py"))
    round_client = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(round_client)

transport = round_client.transport
DESCRIPTOR_SCHEMA = "supervisor-worker-candidate-round-descriptor@2"
DESCRIPTOR_FLAG = "--candidate-dispatch-descriptor"
DIGEST_FLAG = "--candidate-dispatch-descriptor-sha256"
DESCRIPTOR_ENV, DIGEST_ENV = round_client.DESCRIPTOR_ENV, round_client.DIGEST_ENV
REQUEST_SCHEMA = "supervisor-worker-candidate-dispatch-request@1"
RESPONSE_SCHEMA = round_client.RESPONSE_SCHEMA
PAGE_SCHEMA = "supervisor-worker-candidate-dispatch-observation@1"
MAX_FRAME_BYTES, MIN_PAGE_BYTES = round_client.MAX_FRAME_BYTES, round_client.MIN_PAGE_BYTES
MAX_SESSION_SECONDS, REASON_CODES = round_client.MAX_SESSION_SECONDS, round_client.REASON_CODES
MAX_PROMPT_BYTES = 256_000
MAX_TIMEOUT_SECONDS = 600
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
canonical_json, strict_json_loads = round_client.canonical_json, round_client.strict_json_loads
send_frame, receive_frame, process_identity = round_client.send_frame, round_client.receive_frame, round_client.process_identity
_need, _hex, _integer, _text = round_client._need, round_client._hex, round_client._integer, round_client._text

DISPATCH_BINDING_FIELDS = (
    "admission_id", "admission_binding_sha256", "worker_attempt_id",
    "candidate_prompt_sha256", "candidate_prompt_bytes", "original_requirements_sha256",
    "original_native_prompt_sha256", "public_instruction_block_sha256",
    "absolute_deadline_ns", "provider", "model", "timeout", "max_output_tokens", "reasoning_effort")
_DESCRIPTOR_FIELDS = round_client._DESCRIPTOR_FIELDS | set(DISPATCH_BINDING_FIELDS)
_REQUEST_FIELDS = {"schema", "session_id", "request_id", "query"}
_QUERY_FIELDS = {"operation", "invocation_id", "admission_id", "worker_attempt_id",
    "candidate_prompt_sha256", "expected_workspace_generation", "max_bytes"}
_PAGE_FIELDS = set(DISPATCH_BINDING_FIELDS) | {
    "schema", "session_id", "invocation_id", "workspace_generation", "boundary_manifest_sha256",
    "operation", "dispatch_consumed", "timeout_seconds", "host_native_admission_only",
    "serialized_observation_grants_authority", "provider_dispatch_observed",
    "provider_attempted", "proof_authority", "completion_authority", "retry_authority",
    "production_container_dispatch_qualified"}
_SEAL = object()
_ISSUED = weakref.WeakKeyDictionary()
_REGISTRY_LOCK = threading.Lock()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _round_descriptor(value):
    """Project only after V2 shape checks; the V1 validator remains unchanged."""
    projected = {name: value[name] for name in round_client._DESCRIPTOR_FIELDS}
    projected["schema"] = round_client.DESCRIPTOR_SCHEMA
    return projected


def validate_policy(*, provider, model, timeout, max_output_tokens, reasoning_effort):
    _need(provider == "codex_cli" and type(provider) is str, "candidate dispatch requires codex_cli")
    _need(_text(model, 256), "bounded explicit candidate model required")
    _need(_integer(timeout, 1, MAX_TIMEOUT_SECONDS), "bounded candidate timeout required")
    _need(_integer(max_output_tokens, 1, 16384), "bounded candidate output limit required")
    _need(type(reasoning_effort) is str and reasoning_effort in REASONING_EFFORTS,
        "explicit candidate reasoning effort required")
    return dict(provider=provider, model=model, timeout=timeout,
        max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort)


def validate_descriptor(value):
    _need(type(value) is dict and set(value) == _DESCRIPTOR_FIELDS
        and value.get("schema") == DESCRIPTOR_SCHEMA, "native candidate dispatch descriptor required")
    round_client.validate_descriptor(_round_descriptor(value))
    for name in ("admission_id", "worker_attempt_id"):
        _need(_hex(value[name], 32), "invalid candidate admission identity")
    for name in ("admission_binding_sha256", "candidate_prompt_sha256", "original_requirements_sha256",
            "original_native_prompt_sha256", "public_instruction_block_sha256"):
        _need(_hex(value[name]), "invalid candidate dispatch digest")
    _need(_integer(value["candidate_prompt_bytes"], 1, MAX_PROMPT_BYTES), "bounded full candidate prompt required")
    _need(_integer(value["absolute_deadline_ns"], 1, (1 << 63) - 1)
        and value["expires_monotonic_ns"] <= value["absolute_deadline_ns"]
        and value["original_native_prompt_sha256"] == value["native_prompt_sha256"],
        "candidate dispatch changed the original prompt or deadline")
    validate_policy(**{name: value[name] for name in
        ("provider", "model", "timeout", "max_output_tokens", "reasoning_effort")})
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "candidate dispatch descriptor exceeds bound")
    return value


def load_descriptor(path, digest, *, workspace=None):
    return transport.load_descriptor(path, digest, workspace=workspace, validator=validate_descriptor)


def validate_request(value):
    if type(value) is dict and value.get("schema") != REQUEST_SCHEMA:
        return round_client.validate_request(value)
    _need(type(value) is dict and set(value) == _REQUEST_FIELDS
        and value.get("schema") == REQUEST_SCHEMA, "closed candidate dispatch request required")
    _need(_hex(value["session_id"], 32) and _hex(value["request_id"], 32), "invalid dispatch request identity")
    query = value["query"]
    _need(type(query) is dict and set(query) == _QUERY_FIELDS
        and query["operation"] in ("check", "consume"), "closed dispatch operation required")
    _need(all(_hex(query[name], 32) for name in ("invocation_id", "admission_id", "worker_attempt_id"))
        and all(_hex(query[name]) for name in ("candidate_prompt_sha256", "expected_workspace_generation"))
        and query["max_bytes"] == MAX_FRAME_BYTES and type(query["max_bytes"]) is int,
        "invalid candidate dispatch request binding")
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "dispatch request exceeds bound")
    return value


def _request_binding(request, descriptor):
    validate_descriptor(descriptor)
    validate_request(request)
    if request["schema"] != REQUEST_SCHEMA:
        return round_client._request_binding(request, _round_descriptor(descriptor))
    query = request["query"]
    _need(request["session_id"] == descriptor["session_id"]
        and query["expected_workspace_generation"] == descriptor["workspace_generation"]
        and all(query[name] == descriptor[name] for name in
            ("admission_id", "worker_attempt_id", "candidate_prompt_sha256")),
        "dispatch request differs from retained admission")


def dispatch_request(descriptor, *, operation, invocation_id):
    validate_descriptor(descriptor)
    request = dict(schema=REQUEST_SCHEMA, session_id=descriptor["session_id"], request_id=uuid.uuid4().hex,
        query=dict(operation=operation, invocation_id=invocation_id, max_bytes=MAX_FRAME_BYTES,
            expected_workspace_generation=descriptor["workspace_generation"],
            **{name: descriptor[name] for name in ("admission_id", "worker_attempt_id", "candidate_prompt_sha256")}))
    _request_binding(request, descriptor)
    return request


def validate_response(value, *, request, descriptor=None):
    validate_request(request)
    if request["schema"] != REQUEST_SCHEMA:
        if descriptor is not None:
            validate_descriptor(descriptor)
        return round_client.validate_response(value, request=request,
            descriptor=None if descriptor is None else _round_descriptor(descriptor))
    if descriptor is not None:
        _request_binding(request, descriptor)
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "ok", "page", "reason_code"}
        and value["schema"] == RESPONSE_SCHEMA and value["session_id"] == request["session_id"]
        and value["request_id"] == request["request_id"] and type(value["ok"]) is bool,
        "closed dispatch response required")
    if value["ok"] is False:
        _need(value["page"] is None and value["reason_code"] in REASON_CODES, "invalid dispatch refusal")
        return value
    page = value["page"]
    _need(value["reason_code"] is None and type(page) is dict and set(page) == _PAGE_FIELDS
        and page["schema"] == PAGE_SCHEMA and page["session_id"] == request["session_id"]
        and page["invocation_id"] == request["query"]["invocation_id"]
        and page["operation"] == request["query"]["operation"]
        and page["dispatch_consumed"] is (request["query"]["operation"] == "consume"),
        "dispatch response changed its operation or identity")
    _need(page["host_native_admission_only"] is True and page["provider_attempted"] is None
        and all(page[name] is False for name in ("serialized_observation_grants_authority",
            "provider_dispatch_observed", "proof_authority", "completion_authority", "retry_authority",
            "production_container_dispatch_qualified")), "dispatch observation grants no other authority")
    validate_policy(**{name: page[name] for name in
        ("provider", "model", "timeout", "max_output_tokens", "reasoning_effort")})
    _need(_integer(page["timeout_seconds"], 1, page["timeout"]), "invalid remaining dispatch timeout")
    if descriptor is not None:
        _need(all(page[name] == descriptor[name] for name in DISPATCH_BINDING_FIELDS)
            and all(page[name] == descriptor[name] for name in
                ("workspace_generation", "boundary_manifest_sha256")), "dispatch response binding differs")
    _need(len(canonical_json(value)) <= request["query"]["max_bytes"], "dispatch response exceeds bound")
    return value


def _remaining(descriptor):
    remaining = (min(descriptor["expires_monotonic_ns"], descriptor["absolute_deadline_ns"])
        - time.monotonic_ns()) // 1_000_000_000
    _need(remaining >= 1, "candidate original deadline has less than one second remaining")
    return min(descriptor["timeout"], remaining)


def _exchange(descriptor_path, descriptor, request):
    _request_binding(request, descriptor)
    remaining = _remaining(descriptor)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30., remaining))
        channel.connect(descriptor["socket_path"])
        transport._verify_peer(channel, descriptor)
        parent = Path(descriptor_path).parent
        transport._socket_identity(descriptor, parent, parent.stat().st_gid)
        send_frame(channel, request)
        response = receive_frame(channel, max_bytes=request["query"]["max_bytes"])
        transport._verify_peer(channel, descriptor)
        _remaining(descriptor)
    return validate_response(response, request=request, descriptor=descriptor)


class AdmittedCandidateRouterInput:
    """Immutable local issuance from the authenticated live owner, never JSON."""
    __slots__ = ("__weakref__", "_seal", "_descriptor_path", "_descriptor_sha256", "_descriptor_bytes", "_prompt_bytes")

    def __init__(self, seal, *, descriptor_path, descriptor_sha256, descriptor, prompt_bytes):
        _need(seal is _SEAL and type(self) is AdmittedCandidateRouterInput, "authenticated candidate input factory required")
        for name, item in (("_seal", seal), ("_descriptor_path", str(descriptor_path)),
                ("_descriptor_sha256", descriptor_sha256), ("_descriptor_bytes", canonical_json(descriptor)),
                ("_prompt_bytes", prompt_bytes)):
            object.__setattr__(self, name, item)

    def __setattr__(self, name, value):
        raise AttributeError("admitted candidate input is immutable")

    def _issued(self, *, allow_spent=False):
        _need(type(self) is AdmittedCandidateRouterInput and self._seal is _SEAL,
            "exact issued candidate input required")
        with _REGISTRY_LOCK:
            state = _ISSUED.get(self)
            _need(type(state) is dict and state["literal"] == (self._descriptor_path,
                self._descriptor_sha256, self._descriptor_bytes, self._prompt_bytes),
                "candidate input differs from authenticated issuance")
            _need(allow_spent or not state["spent"], "candidate dispatch input already spent")
        _need(process_identity(os.getpid()) == state["process"],
            "candidate input belongs to another process birth")
        return state

    @property
    def descriptor_path(self):
        AdmittedCandidateRouterInput._issued(self, allow_spent=True)
        return Path(self._descriptor_path)

    @property
    def descriptor_sha256(self):
        AdmittedCandidateRouterInput._issued(self, allow_spent=True)
        return self._descriptor_sha256

    @property
    def metadata(self):
        state = AdmittedCandidateRouterInput._issued(self, allow_spent=True)
        return strict_json_loads(state["observation"])

    def _descriptor(self):
        AdmittedCandidateRouterInput._issued(self)
        descriptor = load_descriptor(self._descriptor_path, self._descriptor_sha256, workspace=Path.cwd())
        _need(canonical_json(descriptor) == self._descriptor_bytes, "candidate input descriptor changed")
        _remaining(descriptor)
        return descriptor

    def validate_for_run(self, *, prompt, provider, model, timeout, max_output_tokens, reasoning_effort,
            workspace, boundary_sha256, worker_attempt_id):
        descriptor = AdmittedCandidateRouterInput._descriptor(self)
        _need(type(prompt) is str and prompt.encode("utf-8") == self._prompt_bytes,
            "candidate input requires exact complete original prompt bytes")
        policy = validate_policy(provider=provider, model=model, timeout=timeout,
            max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort)
        _need(all(descriptor[name] == value for name, value in policy.items())
            and str(Path(workspace).resolve(strict=True)) == descriptor["workspace"] == str(Path.cwd())
            and descriptor["boundary_manifest_sha256"] == boundary_sha256
            and descriptor["worker_attempt_id"] == worker_attempt_id,
            "candidate router policy, workspace, boundary or attempt differs")
        response = _exchange(self._descriptor_path, descriptor,
            dispatch_request(descriptor, operation="check", invocation_id=uuid.uuid4().hex))
        _need(response["ok"], "candidate native dispatch check refused: " + str(response["reason_code"]))
        page = dict(response["page"])
        page["timeout_seconds"] = min(page["timeout_seconds"], _remaining(descriptor))
        return page

    def authorize_dispatch(self, *, invocation_id):
        _need(_hex(invocation_id, 32), "exact router invocation identity required")
        descriptor = AdmittedCandidateRouterInput._descriptor(self)
        # Spend locally before any final socket observation; an uncertain reply
        # cannot be retried through this handle or an independently issued one.
        with _REGISTRY_LOCK:
            state = _ISSUED.get(self)
            _need(state is not None and not state["spent"], "candidate dispatch input already spent")
            state["spent"] = True
        response = _exchange(self._descriptor_path, descriptor,
            dispatch_request(descriptor, operation="consume", invocation_id=invocation_id))
        _need(response["ok"], "candidate native dispatch consume refused: " + str(response["reason_code"]))
        page = dict(response["page"])
        page["timeout_seconds"] = min(page["timeout_seconds"], _remaining(descriptor))
        return page

    def guidance(self):
        AdmittedCandidateRouterInput._issued(self, allow_spent=True)
        descriptor = strict_json_loads(self._descriptor_bytes)
        command = "python -I " + shlex.quote(str(Path(__file__).resolve()))
        return ("Candidate source and structural artifacts are advisory; original baseline evidence remains historical. "
            "For bounded reads of this exact candidate generation, run " + command
            + " --expected-workspace-generation " + descriptor["workspace_generation"]
            + " --path <declared-path> --view original, or replace the path/view arguments with --artifact summary. "
            "The owner-bound candidate descriptor and digest are supplied in the candidate-round environment. "
            "Reads cannot refresh the generation, discharge proofs, expand scope or authorize another dispatch.")


def prepare_candidate_router_input(*, descriptor_path, descriptor_sha256, prompt, workspace=None):
    _need(type(prompt) is str, "complete candidate prompt string required")
    raw = prompt.encode("utf-8")
    _need(1 <= len(raw) <= MAX_PROMPT_BYTES, "candidate input exceeds complete prompt bound")
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    _need(_sha(raw) == descriptor["candidate_prompt_sha256"]
        and len(raw) == descriptor["candidate_prompt_bytes"], "candidate input prompt differs from owner admission")
    _remaining(descriptor)
    response = _exchange(descriptor_path, descriptor,
        dispatch_request(descriptor, operation="check", invocation_id=uuid.uuid4().hex))
    _need(response["ok"], "candidate native dispatch check refused: " + str(response["reason_code"]))
    _remaining(descriptor)
    result = AdmittedCandidateRouterInput(_SEAL, descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256, descriptor=descriptor, prompt_bytes=raw)
    with _REGISTRY_LOCK:
        _ISSUED[result] = dict(literal=(result._descriptor_path, result._descriptor_sha256,
            result._descriptor_bytes, result._prompt_bytes), spent=False,
            observation=canonical_json(response["page"]), process=process_identity(os.getpid()))
    return result


@contextmanager
def candidate_descriptor_environment(candidate_input):
    _need(type(candidate_input) is AdmittedCandidateRouterInput, "issued candidate input required")
    selected = {DESCRIPTOR_ENV: str(candidate_input.descriptor_path), DIGEST_ENV: candidate_input.descriptor_sha256}
    previous = {name: os.environ.get(name) for name in selected}
    try:
        os.environ.update(selected)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def request_page(descriptor_path, descriptor_sha256, *, path, expected_workspace_generation,
        view="compact", cursor=None, max_bytes=8192, expected_source_sha256=None,
        expected_view_sha256=None, workspace=None):
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    query = dict(path=path, view=view, cursor=cursor, max_bytes=max_bytes,
        expected_source_sha256=expected_source_sha256, expected_view_sha256=expected_view_sha256,
        source_state="working-candidate", expected_workspace_generation=expected_workspace_generation)
    shared = descriptor["view_policy_binding_sha256"] is not None
    if shared:
        query.update(operation="read", view_policy_binding_sha256=descriptor["view_policy_binding_sha256"])
    return _exchange(descriptor_path, descriptor, dict(schema=round_client.REQUEST_SCHEMA_V3 if shared
        else round_client.REQUEST_SCHEMA_V2, session_id=descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))


def request_context_page(descriptor_path, descriptor_sha256, *, expected_workspace_generation,
        artifact="summary", cursor=None, max_bytes=8192, workspace=None, selection=None):
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    query = dict(artifact=artifact, expected_workspace_generation=expected_workspace_generation,
        cursor=cursor, max_bytes=max_bytes)
    if selection is not None:
        query["selection"] = selection
    return _exchange(descriptor_path, descriptor, dict(schema=round_client.candidate.REQUEST_SCHEMA if selection is None
        else round_client.candidate.REQUEST_SCHEMA_V2, session_id=descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))


request_source_page, request_candidate_page = request_page, request_context_page
render_page_for_model = round_client.render_page_for_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(DESCRIPTOR_FLAG, default=os.environ.get(DESCRIPTOR_ENV))
    parser.add_argument(DIGEST_FLAG, default=os.environ.get(DIGEST_ENV))
    parser.add_argument("--expected-workspace-generation", required=True)
    parser.add_argument("--path")
    parser.add_argument("--view", choices=("original", "compact"), default="compact")
    parser.add_argument("--artifact", choices=round_client.candidate.ARTIFACTS)
    parser.add_argument("--cursor")
    parser.add_argument("--max-bytes", type=int, default=8192)
    parser.add_argument("--raw-response", action="store_true")
    args = parser.parse_args(argv)
    try:
        _need(args.candidate_dispatch_descriptor is not None and args.candidate_dispatch_descriptor_sha256 is not None,
            "candidate dispatch descriptor pair required")
        common = dict(expected_workspace_generation=args.expected_workspace_generation,
            cursor=strict_json_loads(args.cursor) if args.cursor else None, max_bytes=args.max_bytes)
        if args.artifact is not None:
            _need(args.path is None, "select exactly one candidate read")
            result = request_context_page(args.candidate_dispatch_descriptor, args.candidate_dispatch_descriptor_sha256,
                artifact=args.artifact, **common)
        else:
            _need(args.path is not None, "candidate source read requires a declared path")
            result = request_page(args.candidate_dispatch_descriptor, args.candidate_dispatch_descriptor_sha256,
                path=args.path, view=args.view, **common)
        print(canonical_json(result if args.raw_response else render_page_for_model(result)).decode("utf-8"))
        return 0 if result["ok"] else 2
    except (OSError, ValueError, TypeError, RuntimeError):
        print('{"reason_code":"transport_error"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
