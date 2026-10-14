"""Bounded native implementation command dispatched through the shared router.

The supervisor owns worktree allocation, credentials isolation, validation and
completion. This runner performs one pinned coding-provider invocation in the
current worktree and retains native usage; it grants no task completion.
The explicit census mode stops before provider discovery and emits only closed
input metadata after the same current source and workspace checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .semantic_metadata_view import SemanticMetadataTokenCounter

MAX_IMPLEMENTATION_TIMEOUT_SECONDS = 600
GROK_PROGRESS_PREFIX = "IPFS_GROK_PROGRESS "
NUMERICAL_DESCRIPTOR_ENV = "IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR"
NUMERICAL_DESCRIPTOR_DIGEST_ENV = "IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR_SHA256"
SOURCE_LOOKUP_DESCRIPTOR_ENV = "IPFS_SUPERVISOR_SOURCE_LOOKUP_DESCRIPTOR"
SOURCE_LOOKUP_DESCRIPTOR_DIGEST_ENV = "IPFS_SUPERVISOR_SOURCE_LOOKUP_DESCRIPTOR_SHA256"
SOURCE_DESCRIPTOR_ENV = "IPFS_SUPERVISOR_SOURCE_VIEW_DESCRIPTOR"
SOURCE_DESCRIPTOR_DIGEST_ENV = "IPFS_SUPERVISOR_SOURCE_VIEW_DESCRIPTOR_SHA256"


@contextmanager
def _numerical_descriptor_environment(path, digest):
    with _descriptor_environment(path, digest, NUMERICAL_DESCRIPTOR_ENV, NUMERICAL_DESCRIPTOR_DIGEST_ENV):
        yield


@contextmanager
def _source_descriptor_environment(path, digest):
    with _descriptor_environment(path, digest, SOURCE_DESCRIPTOR_ENV, SOURCE_DESCRIPTOR_DIGEST_ENV):
        yield


@contextmanager
def _descriptor_environment(path, digest, path_key, digest_key):
    """Expose only the validated invocation's public endpoint during the call."""
    if path is None:
        yield
        return
    selected = {path_key: str(path), digest_key: digest}
    previous = {key: os.environ.get(key) for key in selected}
    try:
        os.environ.update(selected)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _numerical_broker_observation(*, path, digest, workspace, prompt, boundary_sha256,
                                  formalization_receipt):
    from .numerical_obligation_client import load_descriptor
    from .source_formalization_context import _parts

    descriptor = load_descriptor(path, digest, workspace=workspace)
    parsed = _parts(prompt)
    if parsed is None or formalization_receipt is None:
        raise ValueError("numerical descriptor requires current task formalization evidence")
    payload = parsed[0]
    contexts = [row for row in payload["contexts"] if row["kind"] == "numerical-formalization-context"]
    if len(contexts) != 1:
        raise ValueError("numerical descriptor requires one selected numerical context")
    context = contexts[0]
    expected = dict(task_id=payload["task_id"], task_cid=payload["task_cid"],
        context_bundle_sha256=payload["context_bundle"]["sha256"],
        context_receipt_sha256=context["receipt_reference"]["receipt_sha256"],
        source_tree_id=context["summary"]["source_tree_id"],
        boundary_manifest_sha256=boundary_sha256)
    if (any(descriptor.get(key) != value for key, value in expected.items())
            or any(formalization_receipt.get(key) != expected[key]
                   for key in ("task_id", "task_cid", "context_bundle_sha256"))):
        raise ValueError("numerical descriptor differs from validated task evidence")
    return dict(schema="numerical-obligation-dispatch-observation@1",
        descriptor_sha256=digest, session_id=descriptor["session_id"], **expected,
        requests_observed=False, proof_authority=False, execution_authority=False,
        completion_authority=False)


@contextmanager
def _source_lookup_descriptor_environment(path, digest):
    """Expose only this invocation's public source endpoint, then restore it."""
    if path is None:
        yield
        return
    selected = {SOURCE_LOOKUP_DESCRIPTOR_ENV: str(path), SOURCE_LOOKUP_DESCRIPTOR_DIGEST_ENV: digest}
    previous = {key: os.environ.get(key) for key in selected}
    try:
        os.environ.update(selected)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _source_lookup_broker_observation(*, path, digest, workspace, prompt, boundary_sha256,
                                      formalization_receipt):
    from .source_lookup_worker_client import load_descriptor
    from .source_formalization_context import _parts

    descriptor = load_descriptor(path, digest, workspace=workspace)
    expected = dict(prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    workspace=str(workspace), boundary_manifest_sha256=boundary_sha256)
    if any(descriptor[key] != value for key, value in expected.items()):
        raise ValueError("source descriptor differs from original prompt or worker boundary")
    parsed = _parts(prompt)
    if parsed is not None:
        if (formalization_receipt is None or descriptor["task_cid"] != parsed[0]["task_cid"]
                or formalization_receipt.get("task_cid") != descriptor["task_cid"]):
            raise ValueError("source descriptor differs from validated task formalization")
    helper_path, helper_sha256 = _source_lookup_executable(descriptor)
    # Context CID and repository ID are the native owner's admitted identities.
    # A context bundle digest or filesystem path is not an alias for either.
    return dict(schema="source-lookup-dispatch-observation@1", descriptor_sha256=digest,
        session_id=descriptor["session_id"], **expected,
        task_cid=descriptor["task_cid"], task_revision=descriptor["task_revision"],
        repository_id=descriptor["repository_id"], context_cid=descriptor["context_cid"],
        source_binding_cid=descriptor["source_binding_cid"],
        helper_path=str(helper_path), helper_sha256=helper_sha256,
        source_ref_count=len(descriptor["source_refs"]),
        source_refs_sha256=hashlib.sha256(json.dumps(descriptor["source_refs"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
        max_requests=descriptor["max_requests"], max_response_bytes=descriptor["max_response_bytes"],
        max_slice_bytes=descriptor["max_slice_bytes"],
        max_total_response_bytes=descriptor["max_total_response_bytes"],
        requests_observed=False, actual_tool_usage_unknown=True, source_check_profile="clean-committed-regular@1",
        proof_authority=False, execution_authority=False, completion_authority=False,
        required_fact_omission_authority=False, total_token_savings_measured=False)


def _source_lookup_executable(descriptor):
    """Observe the deployer's protected helper, without owner or provider work."""
    root = Path(descriptor["socket_path"]).parent.parent
    helper = root / "bin/source-lookup"
    if helper.resolve(strict=True) != helper:
        raise ValueError("source helper must be an exact deployed path")
    for parent in helper.parents:
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022
                or os.access(parent, os.W_OK)):
            raise ValueError("source helper ancestors must be deployer protected")
    def facts(info):
        return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_mode", "st_uid",
            "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))
    before = helper.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_gid != 0
            or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o555
            or not 1 <= before.st_size <= 8192 or not os.access(helper, os.X_OK)):
        raise ValueError("source helper must be a bounded root-owned executable")
    fd = os.open(helper, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        raw = os.read(fd, 8193)
        if facts(before) != facts(os.fstat(fd)) or facts(before) != facts(helper.lstat()):
            raise ValueError("source helper changed during admission")
    finally:
        os.close(fd)
    if len(raw) != before.st_size:
        raise ValueError("source helper byte read differs")
    return helper, hashlib.sha256(raw).hexdigest()


def _source_lookup_guidance(receipt):
    return ("\n\nRead-only exact-source lookup is explicitly available for this coding session. "
        "Use the pinned shell helper: " + receipt["helper_path"] + ". "
        "Read source_refs in the public JSON descriptor at "
        "$IPFS_SUPERVISOR_SOURCE_LOOKUP_DESCRIPTOR for the admitted raw_path_hex/source_cid pairs. "
        "Send one JSON object on stdin with exactly these fields: "
        '{"schema":"supervisor-current-source-request@1","binding_cid":"'
        + receipt["source_binding_cid"] + '","raw_path_hex":"<admitted raw path bytes in lowercase hex>",'
        '"source_cid":"<host-admitted raw source CID>","offset":0,"max_bytes":'
        + str(min(4096, receipt["max_slice_bytes"])) + ',"continuation":null}. '
        'source_scope={"binding_cid":"' + receipt["source_binding_cid"] + '"}. '
        "Use only admitted path and source CID pairs. For later slices retain max_bytes and use the returned "
        "end_offset and next_continuation. Only clean committed regular files are supported; reads refuse "
        "after source edits. Required facts remain inline. Retrieval provides no proof, edit, omission or "
        "completion authority.\n")

def _source_broker_observation(*, path, digest, workspace, encoded, boundary_sha256, profile, source_edit_mode="disabled", semantic_working_set="legacy"):
    from .worker_source_client import load_descriptor, semantic_binding, guidance

    descriptor = load_descriptor(path, digest, workspace=workspace)
    expected = dict(**semantic_binding(encoded, profile), boundary_manifest_sha256=boundary_sha256)
    if any(descriptor.get(key) != value for key, value in expected.items()):
        raise ValueError("source descriptor differs from validated semantic evidence")
    if descriptor.get("source_edit_mode", "disabled") != source_edit_mode:
        raise ValueError("source edit mode differs from owner descriptor")
    expected_evidence = "selected-capsules@1" if semantic_working_set == "capsule-references@1" else "disabled"
    if descriptor.get("semantic_evidence_mode", "disabled") != expected_evidence:
        raise ValueError("semantic working set differs from owner evidence service")
    advice = guidance(profile, **({"source_edit_mode": source_edit_mode} if source_edit_mode != "disabled" else {})).encode()
    observation = dict(schema="worker-source-dispatch-observation@1", descriptor_sha256=digest,
        session_id=descriptor["session_id"], **expected,
        guidance_sha256=hashlib.sha256(advice).hexdigest(), guidance_bytes=len(advice),
        requests_observed=False, proof_authority=False, execution_authority=False,
        completion_authority=False)
    if descriptor["schema"] == "supervisor-worker-source-descriptor@2":
        observation.update(schema="worker-source-dispatch-observation@2",
            **{key: descriptor[key] for key in ("view_policy_binding_sha256", "source_scope_sha256")})
    if descriptor["schema"] == "supervisor-worker-source-descriptor@3":
        observation.update(schema="worker-source-dispatch-observation@3",
            **{key: descriptor[key] for key in ("source_edit_mode", "view_policy_binding_sha256", "source_scope_sha256")})
    if descriptor["schema"] == "supervisor-worker-source-descriptor@4":
        observation.update(schema="worker-source-dispatch-observation@4",
            **{key: descriptor[key] for key in ("source_edit_mode", "view_policy_binding_sha256",
                "source_scope_sha256", "semantic_evidence_mode")})
    return observation


def _grok_progress_emitter(invocation_id: str):
    """Keep advisory progress outside native JSON receipt and custody parsers."""
    def emit(value):
        try:
            from ipfs_accelerate_py.cli_runtime.grok_stream_capture import grok_stream_progress_summary
            progress = grok_stream_progress_summary(value)
            receipt = {
                "schema": "router-implementation-progress@1", "invocation_id": invocation_id,
                "provider": "grok_cli", "purpose": "coding", "native_grok_progress": progress,
                "observation_only": True, "completion_authority": False,
                "settlement_authority": False, "automatic_retry_admitted": False,
                "provider_dispatch_observed": None,
            }
            print(GROK_PROGRESS_PREFIX + json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        except Exception:
            # A malformed observation or unavailable sink cannot change the
            # provider's return, failure, or existing bounded cleanup path.
            pass
    return emit


def implementation_timeout_limit(budget_profile=None):
    from .benchmark_budget_policy import validate_budget_profile
    validate_budget_profile(budget_profile)
    return 1800 if budget_profile is not None else MAX_IMPLEMENTATION_TIMEOUT_SECONDS

# Diagnostic vocabulary is intentionally closed: exception messages, dynamic
# type names, caller paths and locals never enter the child-reported record.
RUNNER_FAILURE_PHASES = frozenset({
    "runner_initialization", "argument_validation", "planning_contract",
    "container_boundary", "workspace_identity", "semantic_context", "source_formalization_context",
    "numerical_obligation_descriptor", "source_lookup_descriptor", "source_view_descriptor",
    "doctor_residual", "public_instruction", "model_prompt", "coding_reply_contract",
    "semantic_metadata_view", "semantic_working_set",
    "semantic_metadata_census",
    "credential_isolation", "provider_discovery", "provider_allocation",
    "provider_initialization", "provider_invocation", "provider_result_validation",
    "semantic_response_decode", "coding_reply_validation", "unclassified",
})
RUNNER_FAILURE_TYPES = frozenset({
    "ValueError", "TypeError", "RuntimeError", "PermissionError", "FileNotFoundError",
    "CalledProcessError", "TimeoutExpired", "TimeoutError", "JSONDecodeError",
    "UnicodeDecodeError", "ImportError", "ModuleNotFoundError", "OSError",
    "SemanticTranslationError", "other",
})
RUNNER_FAILURE_FILES = frozenset({
    "router_implementation_runner.py", "container_worker_boundary.py",
    "semantic_router_translation.py", "doctor_residual_context.py",
    "router_public_instruction.py", "local_planning_admission.py",
    "terminal_planner_contract.py", "doctor_candidate_runner.py", "coding_reply_contract.py",
    "semantic_metadata_view.py", "semantic_instruction_view.py",
    "semantic_metadata_census.py", "semantic_working_set.py",
    "source_formalization_context.py",
    "numerical_obligation_client.py", "source_lookup_worker_client.py", "worker_source_client.py",
})


def validate_runner_failure_diagnostic(value: object) -> dict:
    """Validate child-reported metadata, never evidence of dispatch or custody."""
    if (type(value) is not dict or set(value) != {
            "schema", "phase", "exceptions", "chain_truncated", "completion_authority",
            "automatic_retry_admitted", "settlement_authority", "provider_dispatch_observed"}
            or value["schema"] != "router-implementation-failure-diagnostic@1"
            or type(value["phase"]) is not str or value["phase"] not in RUNNER_FAILURE_PHASES
            or type(value["chain_truncated"]) is not bool
            or any(value[key] is not False for key in (
                "completion_authority", "automatic_retry_admitted", "settlement_authority"))
            or value["provider_dispatch_observed"] is not None
            or type(value["exceptions"]) is not list or not 1 <= len(value["exceptions"]) <= 4):
        raise ValueError("invalid closed router failure diagnostic")
    for item in value["exceptions"]:
        if (type(item) is not dict or set(item) != {"exception_type", "frames"}
                or type(item["exception_type"]) is not str or item["exception_type"] not in RUNNER_FAILURE_TYPES
                or type(item["frames"]) is not list or len(item["frames"]) > 12):
            raise ValueError("invalid closed router exception diagnostic")
        for frame in item["frames"]:
            if (type(frame) is not dict or set(frame) != {"file", "line"}
                    or type(frame["file"]) is not str or frame["file"] not in RUNNER_FAILURE_FILES
                    or type(frame["line"]) is not int or not 1 <= frame["line"] <= 1_000_000):
                raise ValueError("invalid closed router failure frame")
    return json.loads(json.dumps(value, allow_nan=False))


def _runner_failure_diagnostic(error: BaseException) -> dict:
    """Walk bounded code metadata directly; do not load traceback source text."""
    phase = "unclassified"
    exceptions = []
    seen = set()
    walked = 0
    truncated = False
    runtime_dir = os.path.dirname(os.path.abspath(__file__))
    current = error
    while current is not None and len(exceptions) < 4 and id(current) not in seen:
        seen.add(id(current))
        kind = type(current).__name__
        kind = kind if kind in RUNNER_FAILURE_TYPES else "other"
        frames = []
        trace = current.__traceback__
        while trace is not None and walked < 256:
            walked += 1
            code = trace.tb_frame.f_code
            if code is run.__code__:
                selected = trace.tb_frame.f_locals.get("failure_phase")
                if phase == "unclassified" and type(selected) is str and selected in RUNNER_FAILURE_PHASES:
                    phase = selected
            filename = os.path.abspath(code.co_filename)
            name = os.path.basename(filename)
            if (os.path.dirname(filename) == runtime_dir and name in RUNNER_FAILURE_FILES
                    and 1 <= trace.tb_lineno <= 1_000_000):
                frames.append({"file": name, "line": trace.tb_lineno})
                frames = frames[-12:]
            trace = trace.tb_next
        truncated = truncated or trace is not None
        exceptions.append({"exception_type": kind, "frames": frames})
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return validate_runner_failure_diagnostic({
        "schema": "router-implementation-failure-diagnostic@1", "phase": phase,
        "exceptions": exceptions, "chain_truncated": truncated or current is not None,
        "completion_authority": False, "automatic_retry_admitted": False,
        "settlement_authority": False, "provider_dispatch_observed": None,
    })


def validate_runner_error_envelope(value: object) -> dict:
    """Require an exact wrapper error, without assigning trust to its author."""
    if (type(value) is not dict or set(value) != {"schema", "error_type", "diagnostic"}
            or value["schema"] != "router-implementation-error@2"):
        raise ValueError("invalid closed router error envelope")
    diagnostic = validate_runner_failure_diagnostic(value["diagnostic"])
    if (type(value["error_type"]) is not str
            or value["error_type"] != diagnostic["exceptions"][0]["exception_type"]):
        raise ValueError("router error type differs from its diagnostic")
    return {"schema": value["schema"], "error_type": value["error_type"], "diagnostic": diagnostic}


def _provider_failure_diagnostic(error: BaseException, *, provider: str) -> dict:
    """Project the shared provider classifier without retaining its message."""
    from ipfs_accelerate_py.llm_allocation.observations import CallErrorKind, classify_provider_failure

    reason = CallErrorKind.UNKNOWN
    typed_kind = getattr(error, "codex_error_kind", None) if provider == "codex_cli" else None
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        reason = CallErrorKind.TIMEOUT
    elif isinstance(typed_kind, CallErrorKind) and typed_kind is not CallErrorKind.SUCCESS:
        reason = typed_kind
    else:
        try:
            failure = classify_provider_failure(provider, str(error)[:16384], exc=error)
            if isinstance(failure.kind, CallErrorKind) and failure.kind is not CallErrorKind.SUCCESS:
                reason = failure.kind
        except Exception:
            # Diagnostics must not replace the original provider failure.
            pass
    result = {"phase": "provider_invocation", "reason_code": reason.value}
    status = getattr(error, "codex_error_status", None) if provider == "codex_cli" else None
    if type(status) is int and 100 <= status <= 599:
        result["http_status"] = status
    return result


def render_model_prompt(*, prompt: str, purpose: str, workspace: Path,
                        semantic_transport: bool = False) -> tuple[str, str]:
    """Preserve native task bytes while making the allocated coding path explicit."""
    if purpose not in {"planning", "coding"}:
        raise ValueError("explicit planning or coding purpose required")
    if purpose == "planning":
        return prompt, ""
    location = json.dumps(str(workspace), ensure_ascii=True)
    advisory = (
        "Supervisor coding workspace contract:\n"
        f"Your allocated checkout and current working directory is {location}.\n"
        "When the task names /app/<relative-path>, use the corresponding file under "
        "this allocated checkout. Edit the declared outputs there, including any "
        "declared new output. Do not write to canonical /app.\n"
        "This mapping changes tool filesystem locations only. In report contents, "
        "keep the task's requested logical path names, including original /app paths; "
        "do not substitute allocated checkout paths.\n"
        "The supervisor owner validates and publishes the candidate. Do not stage "
        "or commit Git changes, and do not modify Git control files. This path "
        "mapping does not expand the task's allowed files or grant completion authority.\n\n"
        + ("Task context follows using a reversible semantic symbol table; source code and authority remain canonical:\n"
           if semantic_transport else "Original native task prompt follows without modification:\n")
    )
    if len(advisory.encode()) > 8192:
        raise ValueError("coding workspace advisory exceeds its byte bound")
    return advisory + prompt, advisory


def run(*, prompt: str, provider: str, model: str, timeout: int, max_output_tokens: int,
        reasoning_effort: str = "high", container_boundary: Path | None = None,
        container_boundary_sha256: str = "", purpose: str = "coding",
        semantic_repository: Path | None = None,
        source_formalization_handoff: Path | None = None,
        source_formalization_handoff_sha256: str | None = None,
        numerical_obligation_descriptor: Path | None = None,
        numerical_obligation_descriptor_sha256: str | None = None,
        source_lookup_descriptor: Path | None = None,
        source_lookup_descriptor_sha256: str | None = None,
        source_view_profile: str = "disabled",
        source_edit_mode: str = "disabled",
        source_view_descriptor: Path | None = None,
        source_view_descriptor_sha256: str | None = None,
        semantic_transport_schema: str = "supervisor-semantic-router-input@1",
        semantic_metadata_view: str = "legacy",
        semantic_working_set: str = "legacy",
        grok_coding_max_turns: int = 128,
        semantic_metadata_token_counter: SemanticMetadataTokenCounter | None = None,
        census_only: bool = False,
        coding_reply_mode: str = "legacy",
        doctor_residual_artifact: Path | None = None,
        doctor_residual_sha256: str = "", doctor_residual_task_cid: str = "",
        public_instruction_artifact: Path | None = None, public_instruction_sha256: str = "",
        public_instruction_task_cid: str = "", budget_profile: str | None = None) -> tuple[str, dict]:
    failure_phase = "runner_initialization"
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation.intelligence_index import discover_available_providers, select_efficient_route
    from ipfs_accelerate_py.llm_router import generate_text, get_llm_provider
    from ipfs_accelerate_py.router_deps import RouterDeps

    failure_phase = "argument_validation"
    from .worker_source_client import PROFILES
    if (type(source_edit_mode) is not str or source_edit_mode not in {"disabled", "mapped-span@1", "mapped-span@2"}
            or source_edit_mode != "disabled" and (source_view_profile == "disabled" or census_only)):
        raise ValueError("source editing requires an enabled coding source session")
    source_args = (source_view_descriptor, source_view_descriptor_sha256)
    source_selected = any(value is not None for value in source_args)
    if type(source_view_profile) is not str or source_view_profile not in ("disabled", *PROFILES):
        raise ValueError("unsupported source view profile")
    if source_selected or source_view_profile != "disabled":
        if (not all(source_args) or source_view_profile == "disabled" or purpose != "coding"
                or semantic_repository is None or container_boundary is None or not container_boundary_sha256):
            raise ValueError("exact source descriptor requires isolated semantic coding")
    elif any(key in os.environ for key in (SOURCE_DESCRIPTOR_ENV, SOURCE_DESCRIPTOR_DIGEST_ENV)):
        raise ValueError("source descriptor environment requires explicit owner selection")
    numerical_args = (numerical_obligation_descriptor, numerical_obligation_descriptor_sha256)
    numerical_selected = any(value is not None for value in numerical_args)
    if numerical_selected:
        if (not all(numerical_args) or purpose != "coding" or semantic_repository is None
                or container_boundary is None or not container_boundary_sha256
                or source_formalization_handoff is None or not source_formalization_handoff_sha256):
            raise ValueError("exact numerical descriptor requires isolated owner formalization coding")
    elif any(key in os.environ for key in (NUMERICAL_DESCRIPTOR_ENV, NUMERICAL_DESCRIPTOR_DIGEST_ENV)):
        raise ValueError("numerical descriptor environment requires explicit owner selection")
    source_lookup_args = (source_lookup_descriptor, source_lookup_descriptor_sha256)
    source_lookup_selected = any(value is not None for value in source_lookup_args)
    if source_lookup_selected:
        if (not all(source_lookup_args) or purpose != "coding" or container_boundary is None
                or not container_boundary_sha256 or census_only):
            raise ValueError("exact source descriptor requires isolated coding dispatch")
    elif any(key in os.environ for key in (SOURCE_LOOKUP_DESCRIPTOR_ENV, SOURCE_LOOKUP_DESCRIPTOR_DIGEST_ENV)):
        raise ValueError("source descriptor environment requires explicit owner selection")
    if provider not in {"codex_cli", "grok_cli"}:
        raise ValueError("explicit supported shared-router coding provider required")
    if provider == "grok_cli" and model != "grok-4.7":
        raise ValueError("explicit pinned Grok benchmark model required")
    maximum_timeout = implementation_timeout_limit(budget_profile)
    if (not model or type(timeout) is not int or not 1 <= timeout <= maximum_timeout
            or not 1 <= max_output_tokens <= 16_384):
        raise ValueError("explicit model and bounded invocation settings required")
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("explicit supported reasoning effort required")
    from .coding_reply_contract import validate_coding_reply_mode
    validate_coding_reply_mode(mode=coding_reply_mode, purpose=purpose, provider=provider)
    if semantic_transport_schema not in {"supervisor-semantic-router-input@1",
                                         "supervisor-semantic-router-input@2"}:
        raise ValueError("unsupported semantic transport schema")
    if (semantic_transport_schema != "supervisor-semantic-router-input@1"
            and (purpose != "coding" or semantic_repository is None)):
        raise ValueError("compact semantic transport requires a semantic coding route")
    if type(semantic_metadata_view) is not str or semantic_metadata_view not in {"legacy", "common-bindings@1", "common-bindings-instruction@1"}:
        raise ValueError("unsupported semantic metadata view")
    if semantic_metadata_view != "legacy" and (purpose != "coding" or semantic_repository is None
            or semantic_transport_schema != "supervisor-semantic-router-input@1"):
        raise ValueError("metadata view requires the original semantic coding transport")
    if semantic_metadata_token_counter is not None:
        from .semantic_metadata_view import SemanticMetadataTokenCounter
        if semantic_metadata_view not in {"common-bindings@1", "common-bindings-instruction@1"}:
            raise ValueError("metadata token counter requires the explicit metadata view")
        if (type(semantic_metadata_token_counter) is not SemanticMetadataTokenCounter
                or semantic_metadata_token_counter.provider != provider
                or semantic_metadata_token_counter.model != model):
            raise ValueError("metadata token counter must match the pinned provider and model")
    if type(census_only) is not bool:
        raise ValueError("explicit Boolean census selection required")
    from .grok_coding_budget import validate_grok_coding_max_turns
    validate_grok_coding_max_turns(grok_coding_max_turns, provider=provider, purpose=purpose)
    if grok_coding_max_turns != 128 and census_only:
        raise ValueError("nondefault coding turns require generation, not metadata census")
    if type(semantic_working_set) is not str or semantic_working_set not in {"legacy", "capsule-references@1"}:
        raise ValueError("unsupported semantic working set")
    if semantic_working_set != "legacy" and (purpose != "coding" or not source_selected
            or semantic_repository is None or semantic_transport_schema != "supervisor-semantic-router-input@1"
            or semantic_metadata_view != "legacy" or census_only):
        raise ValueError("semantic working set requires isolated original semantic coding with source retrieval")
    if census_only and semantic_metadata_view not in {"common-bindings@1", "common-bindings-instruction@1"}:
        raise ValueError("metadata census requires the explicit common-bindings coding view")
    if not prompt.strip() or len(prompt.encode()) > 256_000:
        raise ValueError("implementation prompt exceeds its byte bound")
    failure_phase = "planning_contract"
    response_format = None
    task_contract = None
    codex_planning_schema = None
    codex_schema_projection = None
    codex_planning_validator = None
    if provider == "codex_cli" and purpose == "planning":
        from .codex_planning_schema import planning_schema
        selection = planning_schema(prompt)
        if selection is not None:
            codex_planning_schema, codex_schema_projection, codex_planning_validator = selection
    if provider == "grok_cli" and purpose == "planning":
        from ipfs_accelerate_py.cli_runtime.grok_structured_output import planning_response_format
        response_format = planning_response_format(prompt)
        from .terminal_planner_contract import contract_from_prompt
        task_contract = contract_from_prompt(prompt)
        if task_contract is not None and response_format is None:
            raise ValueError("terminal task constraints require canonical structured planning")
    failure_phase = "container_boundary"
    root = Path.cwd().resolve()
    boundary = None
    if container_boundary is not None or container_boundary_sha256:
        from .container_worker_boundary import verify_container_worker_boundary
        if container_boundary is None:
            raise ValueError("container boundary artifact required")
        boundary = verify_container_worker_boundary(
            artifact=container_boundary, expected_sha256=container_boundary_sha256, workspace=root,
        )
    if provider == "grok_cli" and purpose == "coding" and boundary is None:
        raise ValueError("Grok coding requires the verified isolated container worker boundary")
    failure_phase = "workspace_identity"
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != root:
        raise ValueError("implementation must start at its allocated Git worktree root")
    failure_phase = "semantic_context"
    encoded = None
    semantic_component_census = None
    residual_receipt = None
    residual_advisory = ""
    router_prompt = prompt
    if semantic_repository is not None:
        semantic_repository = Path(semantic_repository).resolve(strict=True)
        if semantic_repository == root:
            raise ValueError("semantic coding requires a separate allocated worktree")
        # The trusted launch chooses the canonical source root. A model cannot
        # nominate a foreign catalog or substitute a different checkout.
        def common_dir(repository):
            value = subprocess.check_output(
                ["/usr/bin/git", "-c", "safe.directory=" + str(repository), "-C", str(repository),
                 "rev-parse", "--path-format=absolute", "--git-common-dir"],
                text=True, timeout=10,
            ).strip()
            return Path(value).resolve(strict=True)
        if common_dir(semantic_repository) != common_dir(root):
            raise ValueError("semantic source repository differs from allocated worktree")
        from .semantic_router_translation import encode_semantic_router_prompt
        encoded = encode_semantic_router_prompt(prompt=prompt, repository=semantic_repository,
                                               transport_schema=semantic_transport_schema)
        router_prompt = encoded.provider_prompt
        # Retain size diagnostics outside the model input. This observes the
        # verified transport before optional metadata projection and wrappers;
        # the existing model_prompt fields bind the actual final input.
        try:
            from .semantic_router_translation import census_semantic_router_prompt
            semantic_component_census = census_semantic_router_prompt(encoded=encoded)
        except Exception:
            # Diagnostic failure cannot change dispatch or expose source text
            # through an exception. The encoder's validation above still rules.
            semantic_component_census = {
                "schema": "supervisor-semantic-prompt-component-census@1",
                "status": "unavailable",
                "raw_content_exported": False,
            }
    failure_phase = "source_formalization_context"
    from .source_formalization_context import validate_source_formalization_context
    formalization_receipt = validate_source_formalization_context(
        prompt=prompt, repository=semantic_repository, workspace=root,
        boundary=container_boundary, boundary_sha256=container_boundary_sha256 or None,
        handoff=source_formalization_handoff, handoff_sha256=source_formalization_handoff_sha256)
    numerical_receipt = None
    if numerical_selected:
        failure_phase = "numerical_obligation_descriptor"
        numerical_receipt = _numerical_broker_observation(path=numerical_obligation_descriptor,
            digest=numerical_obligation_descriptor_sha256, workspace=root, prompt=prompt,
            boundary_sha256=container_boundary_sha256, formalization_receipt=formalization_receipt)
    source_lookup_receipt = None
    if source_lookup_selected:
        failure_phase = "source_lookup_descriptor"
        source_lookup_receipt = _source_lookup_broker_observation(path=source_lookup_descriptor,
            digest=source_lookup_descriptor_sha256, workspace=root, prompt=prompt,
            boundary_sha256=container_boundary_sha256, formalization_receipt=formalization_receipt)
    source_receipt = None
    source_guidance = None
    source_edit_options = {"source_edit_mode": source_edit_mode} if source_edit_mode != "disabled" else {}
    if source_selected:
        failure_phase = "source_view_descriptor"
        source_receipt = _source_broker_observation(path=source_view_descriptor,
            digest=source_view_descriptor_sha256, workspace=root, encoded=encoded,
            boundary_sha256=container_boundary_sha256, profile=source_view_profile,
            **({"semantic_working_set": semantic_working_set} if semantic_working_set != "legacy" else {}),
            **source_edit_options)
        from .worker_source_client import guidance
        source_guidance = guidance(source_view_profile, **source_edit_options)
        router_prompt += source_guidance
    failure_phase = "doctor_residual"
    residual_args = (doctor_residual_artifact, doctor_residual_sha256, doctor_residual_task_cid)
    if any(residual_args):
        if not all(residual_args) or semantic_repository is None or purpose != "coding":
            raise ValueError("exact residual binding requires the full semantic coding route")
        from .doctor_residual_context import load_doctor_residual_advisory
        residual_advisory, residual_receipt = load_doctor_residual_advisory(
            artifact=doctor_residual_artifact, expected_sha256=doctor_residual_sha256,
            repository=semantic_repository, task_cid=doctor_residual_task_cid, prompt=prompt, workspace=root)
        router_prompt += residual_advisory
    failure_phase = "public_instruction"
    instruction_receipt = None
    instruction = None
    instruction_args = (public_instruction_artifact, public_instruction_sha256, public_instruction_task_cid)
    if any(instruction_args):
        if not all(instruction_args) or purpose != "coding":
            raise ValueError("exact public instruction binding requires a coding route")
        from .router_public_instruction import load_public_instruction
        instruction, instruction_receipt = load_public_instruction(
            artifact=public_instruction_artifact, expected_sha256=public_instruction_sha256,
            task_cid=public_instruction_task_cid, prompt=prompt, workspace=root,
            repository=semantic_repository)
        # Preserve the literal requirements after semantic encoding. Native
        # capsule identity remains the stdin identity, not this projection.
        router_prompt += instruction
    if source_lookup_receipt is not None:
        router_prompt += _source_lookup_guidance(source_lookup_receipt)
    failure_phase = "model_prompt"
    model_prompt, advisory = render_model_prompt(prompt=router_prompt, purpose=purpose, workspace=root,
                                                semantic_transport=encoded is not None)
    coding_reply_receipt = None
    if coding_reply_mode != "legacy":
        failure_phase = "coding_reply_contract"
        from .coding_reply_contract import apply_coding_reply_contract
        model_prompt, coding_reply_receipt = apply_coding_reply_contract(
            model_prompt=model_prompt, mode=coding_reply_mode, purpose=purpose, provider=provider)
    metadata_receipt = None
    metadata_native_reply = None
    metadata_census = None
    displacement_receipt = None
    if semantic_metadata_view != "legacy":
        failure_phase = "semantic_metadata_view"
        from .semantic_metadata_view import project_semantic_metadata_view, select_semantic_metadata_view
        selection_options = {}
        selector = select_semantic_metadata_view
        if semantic_metadata_view == "common-bindings-instruction@1":
            if instruction is None or instruction_receipt is None:
                raise ValueError("instruction metadata view requires admitted public requirements")
            from .semantic_instruction_view import project_instruction_view, select_instruction_view
            view = project_instruction_view(encoded.provider_prompt,
                instruction_block=instruction, instruction_receipt=instruction_receipt)
            selector = select_instruction_view
            selection_options["instruction_block"] = instruction
        else:
            view = project_semantic_metadata_view(encoded.provider_prompt)
        if not router_prompt.startswith(encoded.provider_prompt):
            raise ValueError("metadata view differs from the verified semantic transport")
        candidate_router = view.provider_prompt + router_prompt[len(encoded.provider_prompt):]
        candidate_model, candidate_advisory = render_model_prompt(prompt=candidate_router,
            purpose=purpose, workspace=root, semantic_transport=True)
        if candidate_advisory != advisory:
            raise ValueError("metadata view changed the workspace advisory")
        if coding_reply_mode != "legacy":
            candidate_model, candidate_reply = apply_coding_reply_contract(model_prompt=candidate_model,
                mode=coding_reply_mode, purpose=purpose, provider=provider)
            prompt_fields = {"model_prompt_before_sha256", "model_prompt_before_bytes",
                             "model_prompt_after_sha256", "model_prompt_after_bytes"}
            if ({key: value for key, value in candidate_reply.items() if key not in prompt_fields}
                    != {key: value for key, value in coding_reply_receipt.items() if key not in prompt_fields}):
                raise ValueError("metadata view changed the coding response contract")
            metadata_native_reply = dict(coding_reply_receipt)
        if census_only:
            # Count each complete input once. Census repeats all representation
            # checks and selects through the same existing selector.
            from .semantic_metadata_census import census_semantic_metadata_view
            metadata_census = census_semantic_metadata_view(view=view,
                native_complete_prompt=model_prompt, candidate_complete_prompt=candidate_model,
                token_counter=semantic_metadata_token_counter, **selection_options,
                **({"instruction_receipt": instruction_receipt} if selection_options else {}))
            if metadata_census["selected_mode"] != "legacy":
                model_prompt = candidate_model
        else:
            selection = selector(view=view, native_complete_prompt=model_prompt,
                candidate_complete_prompt=candidate_model, token_counter=semantic_metadata_token_counter,
                **selection_options)
            from .representation_displacement import observe_representation_displacement
            displacement_receipt = observe_representation_displacement(
                view=view, selection=selection, native_complete_prompt=model_prompt,
                candidate_complete_prompt=candidate_model, provider=provider, model=model,
                instruction_block=instruction if selection_options else None,
                instruction_receipt=instruction_receipt if selection_options else None,
                source_guidance=source_guidance)
            model_prompt, metadata_receipt = selection.selected_prompt, selection.receipt
        if coding_reply_mode != "legacy" and model_prompt == candidate_model:
            coding_reply_receipt = candidate_reply
    working_set_receipt = None
    if semantic_working_set != "legacy":
        failure_phase = "semantic_working_set"
        from .semantic_working_set import project_semantic_working_set, select_working_set
        from .worker_source_client import load_descriptor
        descriptor = load_descriptor(source_view_descriptor, source_view_descriptor_sha256, workspace=root)
        view = project_semantic_working_set(encoded, descriptor)
        if not router_prompt.startswith(encoded.provider_prompt):
            raise ValueError("working set differs from validated semantic transport")
        candidate_router = view.provider_prompt + router_prompt[len(encoded.provider_prompt):]
        candidate_model, candidate_advisory = render_model_prompt(prompt=candidate_router,
            purpose=purpose, workspace=root, semantic_transport=True)
        if candidate_advisory != advisory:
            raise ValueError("working set changed the workspace advisory")
        candidate_reply = None
        if coding_reply_mode != "legacy":
            candidate_model, candidate_reply = apply_coding_reply_contract(model_prompt=candidate_model,
                mode=coding_reply_mode, purpose=purpose, provider=provider)
        selection = select_working_set(view, native_complete_prompt=model_prompt,
            candidate_complete_prompt=candidate_model)
        working_set_receipt = selection.receipt
        model_prompt = selection.provider_prompt
        if selection.selected_mode != "legacy":
            router_prompt = candidate_router
            coding_reply_receipt = candidate_reply
    if len(model_prompt.encode()) > 256_000:
        raise ValueError("model prompt with workspace contract exceeds its byte bound")
    # Native provider children must never inherit a typed owner token, private
    # bootstrap channel, or database program. The daemon supplies this scrub.
    failure_phase = "credential_isolation"
    from .multi_supervisor_runner import DATABASE_PROGRAM_ENV_NAMES, STATE_CREDENTIAL_ENV_NAMES
    if any(name in os.environ for name in (*DATABASE_PROGRAM_ENV_NAMES, *STATE_CREDENTIAL_ENV_NAMES,
                                           "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN")):
        raise ValueError("implementation inherited database authority")
    if census_only:
        failure_phase = "semantic_metadata_census"
        census = metadata_census
        def digest_json(value):
            if value is None:
                return None
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
            return hashlib.sha256(raw).hexdigest()
        receipt = {
            "schema": ("router-semantic-metadata-census@2" if semantic_metadata_token_counter is not None
                       else "router-semantic-metadata-census@1"), "purpose": "coding",
            "observation_only": True, "provider": provider,
            "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
            "reasoning_effort": reasoning_effort,
            "native_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "native_prompt_bytes": len(prompt.encode("utf-8")),
            "workspace_identity_sha256": hashlib.sha256(str(root).encode("utf-8")).hexdigest(),
            "repository_common_dir_checked": True,
            "current_semantic_encoder_used": True, "source_freshness_authority": False,
            "credential_isolation_checked": True,
            "container_boundary_checked": boundary is not None,
            "container_boundary_artifact_sha256": container_boundary_sha256 if boundary else None,
            "doctor_residual_included": residual_receipt is not None,
            "doctor_residual_receipt_sha256": digest_json(residual_receipt),
            "public_instruction_included": instruction_receipt is not None,
            "public_instruction_receipt_sha256": digest_json(instruction_receipt),
            "workspace_advisory_sha256": hashlib.sha256(advisory.encode("utf-8")).hexdigest(),
            "workspace_advisory_bytes": len(advisory.encode("utf-8")),
            "coding_reply_mode": coding_reply_mode,
            "selected_reply_contract_sha256": digest_json(coding_reply_receipt),
            "native_reply_contract_sha256": digest_json(metadata_native_reply),
            "native_output_schema_sha256": (coding_reply_receipt or {}).get("native_output_schema_sha256"),
            "native_output_schema_bytes": (coding_reply_receipt or {}).get("native_output_schema_bytes"),
            "input_measurement_scope": "router-model-prompt-with-literal-wrappers-and-reply-instruction@1",
            "cli_system_prompt_measured": False, "native_output_schema_tokens_measured": False,
            "census": census, "provider_calls": 0, "router_calls": 0,
            "provider_dispatch_observed": False, "planning_performed_by_census": False,
            "provider_token_usage": None, "total_token_savings_measured": False,
            "proof_authority": False, "execution_authority": False,
            "completion_authority": False, "settlement_authority": False,
            "publication_authority": False, "required_fact_omission_authority": False,
        }
        if numerical_receipt is not None:
            receipt["numerical_obligation_retrieval"] = numerical_receipt
        if source_lookup_receipt is not None:
            receipt["source_lookup_retrieval"] = source_lookup_receipt
        if source_receipt is not None:
            receipt["source_view_retrieval"] = source_receipt
        if semantic_component_census is not None:
            receipt["semantic_prompt_components"] = semantic_component_census
        return json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False), receipt
    failure_phase = "provider_discovery"
    available = discover_available_providers()
    if provider not in available:
        raise ValueError("configured shared-router provider is unavailable")
    failure_phase = "provider_allocation"
    route = select_efficient_route(model_name=model, provider=provider, task_kind="coding", available_providers=available)
    if route.provider != provider or route.model_name != model:
        raise ValueError("model manager changed the explicit benchmark route")
    failure_phase = "provider_initialization"
    deps = RouterDeps()
    instance = get_llm_provider(provider, deps=deps, use_cache=False)
    set_last_cli_observation(provider, {})
    invocation = uuid.uuid4().hex
    started = time.monotonic()
    previous = os.environ.get("ipfs_accelerate_py_CODEX_SANDBOX")
    previous_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    # External mode is available only inside the explicitly deployed Docker
    # worker identity. The ordinary host route keeps its workspace sandbox.
    os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] = "danger-full-access" if boundary else "workspace-write"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    receipt = {
        "schema": "router-implementation-invocation@1", "invocation_id": invocation,
        "provider": provider, "model": model, "allocation_catalog": route.catalog_revision,
        "reasoning_effort": reasoning_effort, "purpose": purpose,
        # Keep the original fields bound to native stdin for capsule comparison.
        # The distinct model fields attest the complete actual provider input.
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_bytes": len(prompt.encode()),
        "native_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "native_prompt_bytes": len(prompt.encode()),
        "model_prompt_sha256": hashlib.sha256(model_prompt.encode()).hexdigest(),
        "model_prompt_bytes": len(model_prompt.encode()),
        "workspace_advisory_sha256": hashlib.sha256(advisory.encode()).hexdigest() if advisory else None,
        "workspace_advisory_bytes": len(advisory.encode()),
        "workspace": str(root),
        "router_prompt_sha256": hashlib.sha256(router_prompt.encode()).hexdigest(),
        "router_prompt_bytes": len(router_prompt.encode()),
        "semantic_translation": dict(encoded.receipt) if encoded is not None else None,
        "doctor_residual_context": residual_receipt,
        "public_instruction": instruction_receipt,
        "timeout_seconds": timeout, "requested_output_tokens": max_output_tokens,
        "provider_output_token_cap_enforced": False, "router_calls": 0,
        "usage": None, "completion_authority": False,
        "external_container_boundary": boundary,
    }
    if coding_reply_receipt is not None:
        receipt["coding_reply_contract"] = coding_reply_receipt
    if semantic_component_census is not None:
        receipt["semantic_prompt_components"] = semantic_component_census
    if metadata_receipt is not None:
        receipt["semantic_metadata_view"] = metadata_receipt
    if working_set_receipt is not None:
        receipt["semantic_working_set"] = working_set_receipt
    if displacement_receipt is not None:
        receipt["representation_displacement"] = displacement_receipt
    if metadata_native_reply is not None:
        receipt["semantic_metadata_native_coding_reply_contract"] = metadata_native_reply
    failure_phase = "provider_invocation"
    provider_options = {}
    if codex_planning_schema is not None:
        provider_options["codex_output_schema"] = codex_planning_schema
        receipt["provider_invocation_policy"] = {"structured_output": {
            "schema": "codex-native-planning-json-schema@1",
            "native_schema_projection": codex_schema_projection,
            "native_schema_requested": True,
            "response_schema_validated": False,
            "plan_admitted": False,
        }}
    if coding_reply_receipt is not None:
        from .coding_reply_contract import coding_completion_schema
        provider_options["codex_output_schema"] = coding_completion_schema()
        receipt["provider_invocation_policy"] = {"structured_output": {
            "schema": "codex-native-coding-completion-json-schema@1",
            "native_schema_requested": True,
            "response_schema_validated": False,
            "execution_authority": False,
            "completion_authority": False,
            "settlement_authority": False,
        }}
    if formalization_receipt is not None:
        receipt["source_formalization_context"] = formalization_receipt
    if numerical_receipt is not None:
        receipt["numerical_obligation_retrieval"] = numerical_receipt
    if source_lookup_receipt is not None:
        receipt["source_lookup_retrieval"] = source_lookup_receipt
    if source_receipt is not None:
        receipt["source_view_retrieval"] = source_receipt
    if provider == "grok_cli":
        # Grok treats an empty allowlist as its default toolset. A nonempty
        # singleton followed by its explicit denial requests zero planning
        # tools; MCP dispatch is denied separately for both purposes. Actual
        # native tool exposure still requires independent qualification.
        provider_options = {"grok_tools": "read_file" if purpose == "planning" else
            "read_file,search_replace,grep,list_dir,todo_write,run_terminal_cmd",
            "grok_disallowed_tools": "read_file,search_tool,use_tool" if purpose == "planning" else
            "search_tool,use_tool",
            "grok_permission_mode": "dontAsk" if purpose == "planning" else "bypassPermissions",
            "grok_max_turns": 2 if purpose == "planning" else grok_coding_max_turns}
        receipt["provider_invocation_policy"] = {
            "max_turns": provider_options["grok_max_turns"],
            "tools_profile": "none" if purpose == "planning" else "isolated_coding",
            "permission_mode": provider_options["grok_permission_mode"],
            "native_tool_allowlist": provider_options["grok_tools"].split(","),
            "native_tool_denylist": provider_options["grok_disallowed_tools"].split(","),
            "disallowed_tools": provider_options["grok_disallowed_tools"].split(","),
            "effective_toolset_verified": False,
        }
        if purpose == "coding":
            provider_options.update(grok_stream_progress=True,
                                    grok_progress_callback=_grok_progress_emitter(invocation))
        if response_format is not None:
            provider_options["response_format"] = response_format
            from ipfs_accelerate_py.cli_runtime.grok_structured_output import native_schema_projection
            _wire_schema, _wire_argument, projection = native_schema_projection(
                response_format["json_schema"]["schema"], task_contract=task_contract)
            if task_contract is not None:
                provider_options["grok_task_contract"] = task_contract
            schema_json = json.dumps(response_format["json_schema"]["schema"],
                ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            receipt["provider_invocation_policy"]["structured_output"] = {
                "schema": "grok-native-json-schema@1",
                "response_schema_sha256": hashlib.sha256(schema_json.encode()).hexdigest(),
                "response_schema_bytes": len(schema_json.encode()),
                "native_schema_projection": projection,
                "native_schema_requested": True,
                "response_schema_validated": False,
                "plan_admitted": False,
            }
    try:
        if source_receipt is not None:
            failure_phase = "source_view_descriptor"
            current = _source_broker_observation(path=source_view_descriptor,
                digest=source_view_descriptor_sha256, workspace=root, encoded=encoded,
                boundary_sha256=container_boundary_sha256, profile=source_view_profile,
                **({"semantic_working_set": semantic_working_set} if semantic_working_set != "legacy" else {}),
                **source_edit_options)
            if current != source_receipt:
                raise ValueError("source descriptor changed before provider dispatch")
        if numerical_receipt is not None:
            failure_phase = "numerical_obligation_descriptor"
            current = _numerical_broker_observation(path=numerical_obligation_descriptor,
                digest=numerical_obligation_descriptor_sha256, workspace=root, prompt=prompt,
                boundary_sha256=container_boundary_sha256, formalization_receipt=formalization_receipt)
            if current != numerical_receipt:
                raise ValueError("numerical descriptor changed before provider dispatch")
        if source_lookup_receipt is not None:
            failure_phase = "source_lookup_descriptor"
            current = _source_lookup_broker_observation(path=source_lookup_descriptor,
                digest=source_lookup_descriptor_sha256, workspace=root, prompt=prompt,
                boundary_sha256=container_boundary_sha256, formalization_receipt=formalization_receipt)
            if current != source_lookup_receipt:
                raise ValueError("source descriptor changed before provider dispatch")
        if working_set_receipt is not None and working_set_receipt["selected_mode"] != "legacy":
            failure_phase = "semantic_working_set"
            from .worker_semantic_evidence_client import request_page
            references = json.loads(view.provider_prompt)["working_set"]["capsule_references"]
            if not references:
                raise ValueError("selected working set has no retrievable capsule")
            reference = references[0]
            probe = request_page(source_view_descriptor, source_view_descriptor_sha256,
                capsule_index=reference["capsule_index"], max_bytes=4096, workspace=root)
            if (probe.get("ok") is not True
                    or probe["page"]["artifact_sha256"] != reference["capsule_sha256"]):
                raise ValueError("selected working set evidence is unavailable")
            from .worker_semantic_evidence_client import canonical_json as evidence_json
            probe_bytes = evidence_json(probe)
            receipt["semantic_working_set_preflight"] = dict(
                schema="supervisor-semantic-working-set-preflight@1", requests=1,
                session_id=source_receipt["session_id"], capsule_index=reference["capsule_index"],
                capsule_sha256=reference["capsule_sha256"],
                response_bytes=len(probe_bytes), response_sha256=hashlib.sha256(probe_bytes).hexdigest(),
                provider_input_bytes=0, source_freshness_checked=True, proof_authority=False)
        failure_phase = "provider_invocation"
        with _numerical_descriptor_environment(numerical_obligation_descriptor,
                                               numerical_obligation_descriptor_sha256), \
                _source_descriptor_environment(source_view_descriptor, source_view_descriptor_sha256), \
                _source_lookup_descriptor_environment(source_lookup_descriptor,
                                                      source_lookup_descriptor_sha256):
            receipt["router_calls"] = 1
            output = generate_text(
                model_prompt, provider=provider, model_name=model, provider_instance=instance, deps=deps,
                allow_local_fallback=False, allow_cross_provider_fallback=False,
                timeout=timeout, max_tokens=max_output_tokens, max_new_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                # Both purposes launch a workspace-capable agent and require a fresh
                # native receipt. Cached response text cannot replay those effects.
                side_effecting=True,
                task_kind="coding", allocation_path="cli", allocation_session_id=invocation,
                **provider_options,
            )
        failure_phase = "provider_result_validation"
        observation = get_last_cli_observation(provider)
        if observation.get("exit_code") != 0:
            raise RuntimeError("coding CLI did not return an observed successful exit")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("coding router returned an empty response")
        if codex_planning_validator is not None:
            if (observation.get("codex_output_schema_sha256") != codex_schema_projection["native_wire_schema_sha256"]
                    or observation.get("codex_output_schema_bytes") != codex_schema_projection["native_wire_schema_bytes"]):
                raise RuntimeError("Codex planning schema invocation metadata is missing or differs")
            from .codex_planning_schema import validate_planning_response
            output = validate_planning_response(output, codex_planning_validator)
            receipt["provider_invocation_policy"]["structured_output"]["response_schema_validated"] = True
        if response_format is not None:
            # The shared adapter enforces this too. Keep the runner boundary
            # explicit in case an injected provider returns unchecked text.
            from ipfs_accelerate_py.cli_runtime.grok_structured_output import validate_response_format, decode_response
            _schema, _argument, validator = validate_response_format(response_format)
            wire_validator = None
            if task_contract is not None:
                wire_format = {**response_format, "json_schema": {**response_format["json_schema"], "schema": _wire_schema}}
                _wire, _argument, wire_validator = validate_response_format(wire_format)
            output = decode_response(json.dumps({"text": output}), validator, additional_validator=wire_validator)
            receipt["provider_invocation_policy"]["structured_output"]["response_schema_validated"] = True
        if encoded is not None:
            failure_phase = "semantic_response_decode"
            from .semantic_router_translation import decode_semantic_router_response
            decoded = decode_semantic_router_response(response=output, encoded=encoded,
                                                      repository=semantic_repository)
            receipt["semantic_response_translation"] = dict(decoded.receipt)
            output = decoded.text
        if coding_reply_receipt is not None:
            # Reserved semantic envelopes always pass through the unchanged
            # strict decoder above; the ordinary contract cannot relabel them.
            failure_phase = "coding_reply_validation"
            if (observation.get("codex_output_schema_sha256") != coding_reply_receipt["native_output_schema_sha256"]
                    or observation.get("codex_output_schema_bytes") != coding_reply_receipt["native_output_schema_bytes"]):
                raise RuntimeError("Codex coding reply schema invocation metadata is missing or differs")
            from .coding_reply_contract import validate_coding_completion_reply
            output = validate_coding_completion_reply(output)
            coding_reply_receipt["response_validated"] = True
            receipt["provider_invocation_policy"]["structured_output"]["response_schema_validated"] = True
        receipt["status"] = "provider_returned"
        return output, receipt
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, failure_phase=failure_phase)
        if failure_phase == "provider_invocation":
            receipt["provider_failure"] = _provider_failure_diagnostic(error, provider=provider)
        if failure_phase == "semantic_response_decode":
            from .semantic_router_translation import SemanticTranslationError
            # The closed projection never serializes model text or arbitrary
            # exception messages. Candidate rejection remains unchanged.
            if type(error) is SemanticTranslationError:
                receipt["semantic_response_failure"] = {
                    "phase": failure_phase, "reason_code": error.reason_code,
                }
        raise
    finally:
        observation = get_last_cli_observation(provider)
        # Keep native categories intact; do not add cached tokens to input or
        # infer a missing total. Provider failures remain measured attempts.
        receipt["usage"] = {key: observation[key] for key in (
            "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
            "total_cost_usd", "model_id", "session_id", "thread_id", "num_turns", "exit_code", "timed_out",
            "codex_output_schema_sha256", "codex_output_schema_bytes",
        ) if key in observation}
        if provider == "codex_cli":
            from .codex_usage_receipt import recover_codex_usage
            receipt["native_rollout_usage"] = recover_codex_usage(
                home=Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex"),
                thread_id=str(observation.get("thread_id") or observation.get("session_id") or ""),
                workspace=root)
        else:
            from ipfs_accelerate_py.cli_runtime.grok_native_usage import grok_outcome_receipt, grok_usage_receipt
            receipt["native_rollout_usage"] = grok_usage_receipt(observation)
            receipt["native_provider_outcome"] = grok_outcome_receipt(observation)
            if purpose == "coding" and observation.get("native_grok_progress") is not None:
                try:
                    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import validate_grok_stream_progress
                    receipt["native_grok_progress"] = validate_grok_stream_progress(observation["native_grok_progress"])
                except Exception:
                    receipt["native_grok_progress_error"] = "invalid_observation"
            if receipt["native_provider_outcome"]["stop_reason"] != "unknown":
                receipt["usage"]["stop_reason"] = receipt["native_provider_outcome"]["stop_reason"]
        receipt["seconds"] = time.monotonic() - started
        print(json.dumps(receipt, sort_keys=True), flush=True)
        if previous is None:
            os.environ.pop("ipfs_accelerate_py_CODEX_SANDBOX", None)
        else:
            os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] = previous
        if previous_bytecode is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous_bytecode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="codex_cli", choices=["codex_cli", "grok_cli"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--budget-profile")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--grok-coding-max-turns", type=int, default=128)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--purpose", default="coding", choices=["planning", "coding"])
    parser.add_argument("--container-boundary", type=Path)
    parser.add_argument("--container-boundary-sha256", default="")
    parser.add_argument("--semantic-repository", type=Path)
    parser.add_argument("--source-formalization-handoff", type=Path)
    parser.add_argument("--source-formalization-handoff-sha256")
    parser.add_argument("--numerical-obligation-descriptor", type=Path)
    parser.add_argument("--numerical-obligation-descriptor-sha256")
    parser.add_argument("--source-lookup-descriptor", type=Path)
    parser.add_argument("--source-lookup-descriptor-sha256")
    parser.add_argument("--source-view-profile", default="disabled",
                        choices=("disabled", "identity@1", "python-whitespace@1", "python-comments@1"))
    parser.add_argument("--source-edit-mode", default="disabled", choices=("disabled", "mapped-span@1", "mapped-span@2"))
    parser.add_argument("--source-view-descriptor", type=Path)
    parser.add_argument("--source-view-descriptor-sha256")
    parser.add_argument("--semantic-transport-schema", default="supervisor-semantic-router-input@1",
                        choices=["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
    parser.add_argument("--semantic-metadata-view", default="legacy", choices=["legacy", "common-bindings@1", "common-bindings-instruction@1"])
    parser.add_argument("--semantic-working-set", default="legacy", choices=["legacy", "capsule-references@1"])
    parser.add_argument("--semantic-metadata-census-only", action="store_true",
                        help="Check the complete metadata view and emit only census metadata before provider dispatch")
    parser.add_argument("--coding-reply-mode", default="legacy",
                        choices=["legacy", "ordinary-completion@1"])
    parser.add_argument("--doctor-residual-artifact", type=Path)
    parser.add_argument("--doctor-residual-sha256", default="")
    parser.add_argument("--doctor-residual-task-cid", default="")
    parser.add_argument("--public-instruction-artifact", type=Path)
    parser.add_argument("--public-instruction-sha256", default="")
    parser.add_argument("--public-instruction-task-cid", default="")
    args = parser.parse_args()
    try:
        prompt = sys.stdin.buffer.read(256_001).decode("utf-8")
        output, _ = run(prompt=prompt, provider=args.provider, model=args.model,
                        timeout=args.timeout, max_output_tokens=args.max_output_tokens,
                        budget_profile=args.budget_profile,
                        reasoning_effort=args.reasoning_effort,
                        container_boundary=args.container_boundary,
                        container_boundary_sha256=args.container_boundary_sha256,
                        purpose=args.purpose, semantic_repository=args.semantic_repository,
                        source_formalization_handoff=args.source_formalization_handoff,
                        source_formalization_handoff_sha256=args.source_formalization_handoff_sha256,
                        numerical_obligation_descriptor=args.numerical_obligation_descriptor,
                        numerical_obligation_descriptor_sha256=args.numerical_obligation_descriptor_sha256,
                        source_lookup_descriptor=args.source_lookup_descriptor,
                        source_lookup_descriptor_sha256=args.source_lookup_descriptor_sha256,
                        source_view_profile=args.source_view_profile,
                        source_edit_mode=args.source_edit_mode,
                        source_view_descriptor=args.source_view_descriptor,
                        source_view_descriptor_sha256=args.source_view_descriptor_sha256,
                        semantic_transport_schema=args.semantic_transport_schema,
                        semantic_metadata_view=args.semantic_metadata_view,
                        semantic_working_set=args.semantic_working_set,
                        grok_coding_max_turns=args.grok_coding_max_turns,
                        census_only=args.semantic_metadata_census_only,
                        coding_reply_mode=args.coding_reply_mode,
                        doctor_residual_artifact=args.doctor_residual_artifact,
                        doctor_residual_sha256=args.doctor_residual_sha256,
                        doctor_residual_task_cid=args.doctor_residual_task_cid,
                        public_instruction_artifact=args.public_instruction_artifact,
                        public_instruction_sha256=args.public_instruction_sha256,
                        public_instruction_task_cid=args.public_instruction_task_cid)
    except Exception as error:
        try:
            diagnostic = _runner_failure_diagnostic(error)
            failure = {"schema": "router-implementation-error@2",
                       "error_type": diagnostic["exceptions"][0]["exception_type"],
                       "diagnostic": diagnostic}
        except Exception:
            # Preserve the failure exit even if observation itself fails.
            failure = {"schema": "router-implementation-error@1", "error_type": "other"}
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
