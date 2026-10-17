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
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .repository_world_packet import PreparedRepositoryWorldPacket
    from .semantic_metadata_view import SemanticMetadataTokenCounter

MAX_IMPLEMENTATION_TIMEOUT_SECONDS = 600
MAX_MODEL_PROMPT_BYTES = 256_000
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


@dataclass(frozen=True)
class PromptTokenCounter:
    """Owner-bound offline tokenizer; remote tokenizer equivalence is not implied."""

    provider: str
    model: str
    tokenizer_id: str
    tokenizer_revision: str
    special_token_policy: str
    count_tokens: Callable[[str], int] = field(repr=False, compare=False)
    count_kind: str = "tokenizer"
    compatibility_basis: str = "caller_declaration"

    def __post_init__(self):
        for name, value in self.binding().items():
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError(f"explicit {name} token-counter binding required")
        if not callable(self.count_tokens):
            raise ValueError("token counter requires a callable")
        if type(self.count_kind) is not str or self.count_kind not in {"tokenizer", "estimate"}:
            raise ValueError("token counter requires a tokenizer or estimate count kind")
        if (type(self.compatibility_basis) is not str or not self.compatibility_basis.strip()
                or self.compatibility_basis != self.compatibility_basis.strip()):
            raise ValueError("explicit tokenizer compatibility basis required")

    @property
    def count_field(self) -> str:
        return "estimated_input_tokens" if self.count_kind == "estimate" else "input_tokens"

    def binding(self) -> dict:
        return {name: getattr(self, name) for name in (
            "provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy")}

    def count(self, prompt: str) -> int:
        count = self.count_tokens(prompt)
        if type(count) is not int or count < 1:
            raise ValueError("token counter must return a positive untruncated token count")
        return count


def _select_complete_model_prompt(*, native_complete_prompt: str, candidate_complete_prompt: str | None,
                        provider: str, model: str, token_counter: PromptTokenCounter | None,
                        max_input_tokens: int | None = None,
                        native_output_schema: dict | None = None,
                        semantic_complete_prompt: str | None = None,
                        candidate_representation: str = "semantic") -> tuple[str, dict]:
    """Compare complete rendered alternatives without dropping required text.

    Callers verify candidate restoration/freshness and render all workspace,
    task, dictionary and reply instructions before selection. The declared
    tokenizer counts each complete user prompt and the separate native output
    schema. Hidden provider instructions, tool definitions, chat framing and
    subsequent turns remain outside this local budget and its measurements.
    """
    if max_input_tokens is not None and (type(max_input_tokens) is not int or max_input_tokens < 1):
        raise ValueError("complete prompt token budget must be a positive integer")
    if token_counter is not None and (not isinstance(token_counter, PromptTokenCounter)
            or token_counter.provider != provider or token_counter.model != model):
        raise ValueError("token counter differs from the pinned provider/model")
    if candidate_representation not in {"semantic", "semantic_metadata"}:
        raise ValueError("unsupported prompt candidate representation")
    rendered = {"native": native_complete_prompt}
    if semantic_complete_prompt is not None:
        if candidate_representation != "semantic_metadata":
            raise ValueError("a separate semantic baseline requires a metadata candidate")
        rendered["semantic"] = semantic_complete_prompt
    if candidate_complete_prompt is not None:
        rendered[candidate_representation] = candidate_complete_prompt
    candidates = {}
    for name, text in rendered.items():
        if type(text) is not str or not text.strip():
            raise ValueError("complete prompt must be nonempty literal text")
        raw = text.encode("utf-8")
        candidates[name] = {"model_prompt_sha256": hashlib.sha256(raw).hexdigest(),
                            "model_prompt_bytes": len(raw), "content_tokens": None,
                            "input_tokens": None}
    schema_text = (json.dumps(native_output_schema, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False)
                   if native_output_schema is not None else None)
    if schema_text is not None and len(schema_text.encode()) > MAX_MODEL_PROMPT_BYTES:
        raise ValueError("native output schema exceeds its byte bound")
    schema_tokens = None
    status, error_type = "unavailable", None
    count_field = token_counter.count_field if token_counter is not None else "input_tokens"
    if token_counter is not None:
        try:
            counts = {name: token_counter.count(text) for name, text in rendered.items()}
            schema_tokens = token_counter.count(schema_text) if schema_text is not None else 0
        except Exception as error:
            status = "failed"
            error_type = type(error).__name__ if type(error).__name__ in RUNNER_FAILURE_TYPES else "other"
        else:
            status = "estimated" if token_counter.count_kind == "estimate" else "measured"
            for name, count in counts.items():
                if status == "estimated":
                    candidates[name].update(estimated_content_tokens=count,
                                            estimated_input_tokens=count + schema_tokens)
                else:
                    candidates[name].update(content_tokens=count, input_tokens=count + schema_tokens)
    count_available = status in {"measured", "estimated"}
    if max_input_tokens is not None and not count_available:
        raise ValueError("complete prompt token budget requires a working pinned local counter")
    selected, reason = "native", "semantic_not_requested"
    if candidate_complete_prompt is not None:
        reason = "native_not_larger" if count_available else "tokenizer_" + status
        if count_available:
            for name, candidate in candidates.items():
                if (candidate[count_field] < candidates[selected][count_field]
                        and candidate["model_prompt_bytes"] <= MAX_MODEL_PROMPT_BYTES):
                    selected, reason = name, ("candidate_strictly_fewer_estimated_content_tokens"
                        if status == "estimated" else "candidate_strictly_fewer_content_tokens")
    chosen = candidates[selected]
    if chosen["model_prompt_bytes"] > MAX_MODEL_PROMPT_BYTES:
        raise ValueError("model prompt with all required instructions exceeds its byte bound")
    if max_input_tokens is not None and chosen[count_field] > max_input_tokens:
        raise ValueError("model prompt with all required instructions exceeds its token budget")
    receipt = {
        "schema": "supervisor-model-prompt-selection@1", "selected_representation": selected,
        "selection_reason": reason, "candidates": candidates, "measurement_status": status,
        "measurement_error_type": error_type,
        "tokenizer_binding": token_counter.binding() if token_counter is not None else None,
        "count_scope": "rendered_user_prompt_and_separate_native_output_schema",
        "native_output_schema_tokens": schema_tokens if status != "estimated" else None,
        "native_output_schema_sha256": hashlib.sha256(schema_text.encode()).hexdigest() if schema_text else None,
        "selected_content_tokens": chosen["content_tokens"], "selected_input_tokens": chosen["input_tokens"],
        "token_count_kind": token_counter.count_kind if token_counter is not None else None,
        "tokenizer_compatibility_basis": token_counter.compatibility_basis if token_counter is not None else None,
        "selected_estimated_content_tokens": chosen.get("estimated_content_tokens"),
        "token_budget_basis": ("estimated_content_tokens" if status == "estimated"
                               else "local_tokenizer_content_tokens") if max_input_tokens is not None else None,
        "token_budget_guaranteed": False,
        "max_input_tokens": max_input_tokens, "max_prompt_bytes": MAX_MODEL_PROMPT_BYTES,
        "provider_framing_measured": False, "provider_tokenizer_equivalence_verified": False,
        "total_session_token_savings": None, "completion_authority": False,
    }
    if status == "estimated":
        receipt.update(token_count_kind="estimate",
            tokenizer_compatibility_basis=token_counter.compatibility_basis,
            estimated_native_output_schema_tokens=schema_tokens,
            selected_estimated_content_tokens=chosen["estimated_content_tokens"],
            selected_estimated_input_tokens=chosen["estimated_input_tokens"],
            token_budget_basis="estimated_content_tokens" if max_input_tokens is not None else None,
            token_budget_guaranteed=False)
    return rendered[selected], receipt


def _select_legacy_model_prompt(*, native_prompt: str, semantic_prompt: str | None,
                        suffix: str, purpose: str, workspace: Path,
                        provider: str, model: str,
                        token_counter: PromptTokenCounter | None = None,
                        max_input_tokens: int | None = None,
                        max_prompt_bytes: int = MAX_MODEL_PROMPT_BYTES) -> tuple[str, str, str, dict]:
    """Measure both full wires and select a strictly smaller supported transport.

    Semantic identity/currentness and exact restoration must already have been
    verified by the encoder. Required residual/instruction text is appended
    verbatim to both alternatives before measurement. Unknown counts retain
    native context. Estimated counts may guide selection and a labelled budget;
    they cannot guarantee the provider's token ceiling.
    """
    if (isinstance(max_prompt_bytes, bool) or not isinstance(max_prompt_bytes, int)
            or not 1 <= max_prompt_bytes <= MAX_MODEL_PROMPT_BYTES):
        raise ValueError("model prompt byte bound cannot exceed its existing ceiling")
    if max_input_tokens is not None and (
            isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int)
            or max_input_tokens < 1):
        raise ValueError("complete prompt token budget must be a positive integer")
    if token_counter is not None and (
            not isinstance(token_counter, PromptTokenCounter)
            or token_counter.provider != provider or token_counter.model != model):
        raise ValueError("token counter differs from the pinned provider/model")

    rendered = {}
    candidates = {}
    count_field = token_counter.count_field if token_counter is not None else "input_tokens"
    for name, prompt in (("native", native_prompt), ("semantic", semantic_prompt)):
        if prompt is None:
            continue
        router_prompt = prompt + suffix
        model_prompt, advisory = render_model_prompt(
            prompt=router_prompt, purpose=purpose, workspace=workspace,
            semantic_transport=name == "semantic")
        rendered[name] = (model_prompt, router_prompt, advisory)
        candidates[name] = {
            "model_prompt_sha256": hashlib.sha256(model_prompt.encode()).hexdigest(),
            "model_prompt_bytes": len(model_prompt.encode()), "input_tokens": None,
        }
        if count_field == "estimated_input_tokens":
            candidates[name][count_field] = None

    measurement_status = "unavailable"
    measurement_error_type = None
    if token_counter is not None:
        try:
            counts = {name: token_counter.count(values[0]) for name, values in rendered.items()}
        except Exception as error:
            measurement_status = "failed"
            measurement_error_type = type(error).__name__
        else:
            measurement_status = "estimated" if count_field == "estimated_input_tokens" else "measured"
            for name, count in counts.items():
                candidates[name][count_field] = count
    count_available = measurement_status in {"measured", "estimated"}
    if max_input_tokens is not None and not count_available:
        raise ValueError("complete prompt token budget requires a working pinned local counter")

    selected = "native"
    reason = "semantic_not_requested"
    if "semantic" in candidates:
        reason = "tokenizer_" + measurement_status if not count_available else "native_not_larger"
        if count_available:
            semantic = candidates["semantic"]
            if (semantic[count_field] < candidates["native"][count_field]
                    and semantic["model_prompt_bytes"] <= max_prompt_bytes):
                selected = "semantic"
                reason = ("semantic_strictly_fewer_estimated_content_tokens" if measurement_status == "estimated"
                          else "semantic_strictly_fewer_content_tokens")
    chosen = candidates[selected]
    if chosen["model_prompt_bytes"] > max_prompt_bytes:
        raise ValueError("model prompt with all required instructions exceeds its byte bound")
    if max_input_tokens is not None and chosen[count_field] > max_input_tokens:
        raise ValueError("model prompt with all required instructions exceeds its token budget")
    receipt = {
        "schema": "supervisor-model-prompt-selection@2",
        "selected_representation": selected, "selection_reason": reason,
        "candidates": candidates, "measurement_status": measurement_status,
        "measurement_error_type": measurement_error_type,
        "tokenizer_binding": token_counter.binding() if token_counter is not None else None,
        "token_count_kind": token_counter.count_kind if token_counter is not None else None,
        "tokenizer_compatibility_basis": token_counter.compatibility_basis if token_counter is not None else None,
        "count_scope": "rendered_user_prompt_only",
        "selected_content_tokens": chosen["input_tokens"],
        "selected_estimated_content_tokens": chosen.get("estimated_input_tokens"),
        "token_budget_basis": ("estimated_content_tokens" if count_field == "estimated_input_tokens"
                               else "local_tokenizer_content_tokens") if max_input_tokens is not None else None,
        "token_budget_guaranteed": False,
        "max_input_tokens": max_input_tokens, "max_prompt_bytes": max_prompt_bytes,
        "total_session_token_savings": None, "completion_authority": False,
    }
    return (*rendered[selected], receipt)


_PROMPT_ARGUMENT_UNSET = object()


def select_model_prompt(*,
                        native_complete_prompt=_PROMPT_ARGUMENT_UNSET,
                        candidate_complete_prompt=_PROMPT_ARGUMENT_UNSET,
                        native_output_schema=_PROMPT_ARGUMENT_UNSET,
                        semantic_complete_prompt=_PROMPT_ARGUMENT_UNSET,
                        candidate_representation=_PROMPT_ARGUMENT_UNSET,
                        native_prompt=_PROMPT_ARGUMENT_UNSET,
                        semantic_prompt=_PROMPT_ARGUMENT_UNSET,
                        suffix=_PROMPT_ARGUMENT_UNSET,
                        purpose=_PROMPT_ARGUMENT_UNSET,
                        workspace=_PROMPT_ARGUMENT_UNSET,
                        max_prompt_bytes=_PROMPT_ARGUMENT_UNSET,
                        provider: str, model: str,
                        token_counter=_PROMPT_ARGUMENT_UNSET,
                        max_input_tokens: int | None = None):
    """Admit exactly one of the complete-input and legacy rendered-input APIs.

    Explicit None is supplied, rather than omitted. Complete-input callers keep
    their two-value result; legacy callers retain the existing four-value result.
    Unknown keywords are rejected by this signature, and mixed forms are refused.
    """
    complete_supplied = any(value is not _PROMPT_ARGUMENT_UNSET for value in (
        native_complete_prompt, candidate_complete_prompt, native_output_schema,
        semantic_complete_prompt, candidate_representation))
    legacy_supplied = any(value is not _PROMPT_ARGUMENT_UNSET for value in (
        native_prompt, semantic_prompt, suffix, purpose, workspace, max_prompt_bytes))
    if complete_supplied and legacy_supplied:
        raise TypeError("mixed complete and legacy prompt-selection forms")
    if complete_supplied:
        if any(value is _PROMPT_ARGUMENT_UNSET for value in (
                native_complete_prompt, candidate_complete_prompt, token_counter)):
            raise TypeError("complete prompt selection requires native, candidate and token counter")
        return _select_complete_model_prompt(
            native_complete_prompt=native_complete_prompt,
            candidate_complete_prompt=candidate_complete_prompt,
            provider=provider, model=model, token_counter=token_counter,
            max_input_tokens=max_input_tokens,
            native_output_schema=(None if native_output_schema is _PROMPT_ARGUMENT_UNSET
                                  else native_output_schema),
            semantic_complete_prompt=(None if semantic_complete_prompt is _PROMPT_ARGUMENT_UNSET
                                      else semantic_complete_prompt),
            candidate_representation=("semantic" if candidate_representation is _PROMPT_ARGUMENT_UNSET
                                      else candidate_representation))
    if not legacy_supplied or any(value is _PROMPT_ARGUMENT_UNSET for value in (
            native_prompt, semantic_prompt, suffix, purpose, workspace)):
        raise TypeError("legacy prompt selection requires native, semantic, suffix, purpose and workspace")
    return _select_legacy_model_prompt(
        native_prompt=native_prompt, semantic_prompt=semantic_prompt,
        suffix=suffix, purpose=purpose, workspace=workspace,
        provider=provider, model=model,
        token_counter=None if token_counter is _PROMPT_ARGUMENT_UNSET else token_counter,
        max_input_tokens=max_input_tokens,
        max_prompt_bytes=(MAX_MODEL_PROMPT_BYTES if max_prompt_bytes is _PROMPT_ARGUMENT_UNSET
                          else max_prompt_bytes))

def _measure_world_packet_prompt(*, model_prompt, provider, model, token_counter,
                                 max_input_tokens, native_output_schema):
    """Budget the selected current-state input without comparing unlike tasks."""
    _selected, measured = select_model_prompt(
        native_complete_prompt=model_prompt, candidate_complete_prompt=None,
        provider=provider, model=model, token_counter=token_counter,
        max_input_tokens=max_input_tokens, native_output_schema=native_output_schema,
    )
    receipt = {
        "schema": "supervisor-world-task-prompt-measurement@1",
        "measurement_kind": ("declared_local_estimate" if measured["measurement_status"] == "estimated"
                             else "declared_local_tokenizer"),
        "model_prompt_sha256": hashlib.sha256(model_prompt.encode()).hexdigest(),
        "model_prompt_bytes": len(model_prompt.encode()),
        **{name: measured[name] for name in (
            "measurement_status", "measurement_error_type", "tokenizer_binding", "count_scope",
            "native_output_schema_tokens", "native_output_schema_sha256", "selected_content_tokens",
            "selected_input_tokens", "max_input_tokens", "max_prompt_bytes",
            "provider_framing_measured", "provider_tokenizer_equivalence_verified",
            "total_session_token_savings", "completion_authority",
        )},
        "provider_system_prompt_measured": False,
        "provider_tool_schemas_measured": False,
        "provider_internal_history_measured": False,
    }
    if measured["measurement_status"] == "estimated":
        receipt.update({name: measured[name] for name in (
            "token_count_kind", "tokenizer_compatibility_basis", "estimated_native_output_schema_tokens",
            "selected_estimated_content_tokens", "selected_estimated_input_tokens",
            "token_budget_basis", "token_budget_guaranteed")})
    return receipt


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
    "terminal_semantic_scope_currentness",
    "world_task_packet", "world_task_packet_currentness", "world_packet_audit_admission",
    "semantic_metadata_view", "semantic_working_set",
    "semantic_metadata_census",
    "prompt_tokenizer", "prompt_selection",
    "source_input_view", "source_input_freshness", "query_reference_view", "query_reference_freshness",
    "repository_source_store", "source_view_store", "tool_output_store",
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
    "local_prompt_tokenizer.py",
    "repository_world_packet.py", "repository_world_artifact_store.py",
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
    if reason is CallErrorKind.RATE_LIMIT:
        from ipfs_accelerate_py.llm_router import _bounded_provider_retry_hint
        prefix = "codex_" if provider == "codex_cli" else ""
        try:
            result.update(_bounded_provider_retry_hint(
                getattr(error, prefix + "retry_after_seconds", None),
                unbounded=getattr(error, prefix + "retry_after_unbounded", False),
            ))
        except Exception:
            # Optional timing diagnostics cannot mask the native failure.
            pass
    return result


def render_model_prompt(*, prompt: str, purpose: str, workspace: Path,
                        semantic_transport: bool = False,
                        world_packet: bool = False,
                        tool_output_binding: dict | None = None,
                        source_view_binding: dict | None = None,
                        repository_source_binding: dict | None = None) -> tuple[str, str]:
    """Preserve native task bytes while making the allocated coding path explicit."""
    if purpose not in {"planning", "coding"}:
        raise ValueError("explicit planning or coding purpose required")
    if type(world_packet) is not bool or (world_packet and semantic_transport):
        raise ValueError("explicit compatible world-packet prompt selection required")
    if purpose == "planning":
        if (tool_output_binding is not None or source_view_binding is not None
                or repository_source_binding is not None or world_packet):
            raise ValueError("helper guidance is coding-only")
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
        + ("Current task requirements and source-bound world context follow. This is a self-contained "
           "decision packet; no retained prior provider context is assumed. Observations and retrieval "
           "references do not grant proof or completion authority:\n" if world_packet else
           "Task context follows using a reversible semantic symbol table; source code and authority remain canonical:\n"
           if semantic_transport else "Original native task prompt follows without modification:\n")
    )
    if tool_output_binding is not None:
        from .tool_output_policy import validate_binding
        advisory = validate_binding(tool_output_binding, workspace=workspace) + advisory
    if source_view_binding is not None:
        from .source_view_policy import validate_binding as validate_source_binding
        advisory = validate_source_binding(source_view_binding, workspace=workspace) + advisory
    if repository_source_binding is not None:
        from .repository_source_policy import validate_binding as validate_repository_binding
        advisory = validate_repository_binding(repository_source_binding, workspace=workspace) + advisory
    if len(advisory.encode()) > 8192:
        raise ValueError("coding workspace advisory exceeds its byte bound")
    return advisory + prompt, advisory


def _refuse_candidate_preparation(prompt):
    """Prepared candidate envelopes need their own admission before dispatch.

    An owner-local preparation must not silently become an ordinary untyped
    prompt merely because its caller omits the baseline semantic route.
    """
    if type(prompt) is not str:
        return
    try:
        envelope, _ = json.JSONDecoder().raw_decode(prompt.lstrip())
    except (ValueError, RecursionError):
        return
    if (type(envelope) is dict and type(envelope.get("schema")) is str
            and envelope["schema"].startswith("supervisor-owner-candidate-router-input@")):
        raise ValueError("candidate round prompt requires independent dispatch admission")


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
        tool_output_policy: str = "legacy",
        source_view_advisory_policy: str = "legacy",
        repository_source_policy: str = "legacy",
        repository_source_parent_lease=None,
        repository_source_scheduler_policy_sha256: str | None = None,
        repository_source_deadline_monotonic: float | None = None,
        repository_source_owner_descriptor: Path | None = None,
        repository_source_owner_descriptor_sha256: str | None = None,
        source_input_policy: str = "legacy",
        query_reference_policy: str = "legacy",
        census_only: bool = False,
        coding_reply_mode: str = "legacy",
        doctor_residual_artifact: Path | None = None,
        doctor_residual_sha256: str = "", doctor_residual_task_cid: str = "",
        public_instruction_artifact: Path | None = None, public_instruction_sha256: str = "",
        public_instruction_task_cid: str = "", budget_profile: str | None = None,
        prompt_token_counter: PromptTokenCounter | None = None,
        prompt_tokenizer_artifact: Path | None = None, prompt_tokenizer_sha256: str = "",
        max_input_tokens: int | None = None,
        world_task_packet: PreparedRepositoryWorldPacket | None = None,
        world_packet_audit_directory: Path | None = None,
        codex_cli_literal_response: bool = False,
        receipt_observer: Callable[[dict], None] | None = None) -> tuple[str, dict]:
    """Dispatch once; an optional owner-prepared packet replaces task context."""
    failure_phase = "runner_initialization"
    _refuse_candidate_preparation(prompt)
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation.intelligence_index import discover_available_providers, select_efficient_route
    from ipfs_accelerate_py.llm_router import generate_text, get_llm_provider
    from ipfs_accelerate_py.router_deps import RouterDeps

    failure_phase = "argument_validation"
    repository_resources = (repository_source_parent_lease,
        repository_source_scheduler_policy_sha256, repository_source_deadline_monotonic)
    repository_owner_selected = any(value is not None for value in (
        repository_source_owner_descriptor, repository_source_owner_descriptor_sha256))
    if repository_owner_selected:
        if (repository_source_owner_descriptor is None or not repository_source_owner_descriptor_sha256
                or repository_source_policy == "legacy" or purpose != "coding" or provider != "codex_cli"
                or census_only or any(value is not None for value in repository_resources)):
            raise ValueError("exact repository owner descriptor requires its exclusive Codex coding route")
    if any(value is not None for value in repository_resources):
        if (any(value is None for value in repository_resources) or repository_source_policy == "legacy"
                or purpose != "coding" or provider != "codex_cli"):
            raise ValueError("explicit repository source parent, policy and deadline require its Codex coding route")
        from .repository_source_resource_handoff import validate_owner_resources
        validate_owner_resources(parent_lease=repository_source_parent_lease,
            expected_policy_sha256=repository_source_scheduler_policy_sha256,
            deadline_monotonic=repository_source_deadline_monotonic)
    if type(codex_cli_literal_response) is not bool:
        raise ValueError("Codex literal response must be an explicit Boolean")
    if codex_cli_literal_response and provider != "codex_cli":
        raise ValueError("Codex literal response requires the Codex CLI provider")
    if receipt_observer is not None and not callable(receipt_observer):
        raise ValueError("explicit receipt observer callback required")
    if world_packet_audit_directory is not None and world_task_packet is None:
        raise ValueError("world packet audit capture requires a prepared world packet")
    if world_task_packet is not None:
        from .repository_world_packet import PreparedRepositoryWorldPacket
        if type(world_task_packet) is not PreparedRepositoryWorldPacket:
            raise ValueError("world task dispatch requires an exact prepared repository packet")
        if (purpose != "coding" or provider != "codex_cli" or semantic_repository is not None
                or semantic_transport_schema != "supervisor-semantic-router-input@1"
                or semantic_metadata_view != "legacy" or semantic_metadata_token_counter is not None
                or source_input_policy != "legacy" or query_reference_policy != "legacy"
                or source_view_advisory_policy != "legacy" or source_view_profile != "disabled"
                or source_edit_mode != "disabled" or census_only
                or any(value is not None for value in (
                    source_formalization_handoff, source_formalization_handoff_sha256,
                    numerical_obligation_descriptor, numerical_obligation_descriptor_sha256,
                    source_lookup_descriptor, source_lookup_descriptor_sha256,
                    source_view_descriptor, source_view_descriptor_sha256,
                    doctor_residual_artifact, public_instruction_artifact))
                or any((doctor_residual_sha256, doctor_residual_task_cid,
                        public_instruction_sha256, public_instruction_task_cid))):
            raise ValueError("world task packet requires an independently selected Codex coding route")
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
    if source_view_profile != "disabled" and any(policy != "legacy" for policy in (
            source_view_advisory_policy, repository_source_policy, source_input_policy, query_reference_policy)):
        raise ValueError("signed source views require independently qualified projection policies")
    from .coding_reply_contract import validate_coding_reply_mode
    validate_coding_reply_mode(mode=coding_reply_mode, purpose=purpose, provider=provider)
    from .tool_output_policy import validate_policy
    validate_policy(tool_output_policy, purpose=purpose, provider=provider)
    from .query_reference_policy import validate_policy as validate_query_policy
    validate_query_policy(query_reference_policy, purpose=purpose, provider=provider,
        semantic_repository=semantic_repository, semantic_metadata_view=semantic_metadata_view,
        source_input_policy=source_input_policy, census_only=census_only)
    from .source_input_policy import validate_policy as validate_input_policy, projection_options
    validate_input_policy(source_input_policy, purpose=purpose, provider=provider,
        semantic_repository=semantic_repository, semantic_metadata_view=semantic_metadata_view,
        census_only=census_only, prompt_token_counter=prompt_token_counter,
        prompt_tokenizer_artifact=prompt_tokenizer_artifact, max_input_tokens=max_input_tokens)
    from .source_view_policy import validate_policy as validate_source_policy
    validate_source_policy(source_view_advisory_policy, purpose=purpose, provider=provider)
    from .repository_source_policy import validate_policy as validate_repository_policy
    validate_repository_policy(repository_source_policy, purpose=purpose, provider=provider,
        source_view_policy=source_view_advisory_policy, census_only=census_only)
    if source_view_advisory_policy != "legacy" and (census_only or semantic_repository is None):
        raise ValueError("source view requires an actual semantic coding invocation")
    if tool_output_policy != "legacy" and census_only:
        raise ValueError("tool output policy requires an actual coding invocation")
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
    failure_phase = "prompt_tokenizer"
    tokenizer_source = None
    if type(prompt_tokenizer_sha256) is not str:
        raise ValueError("exact local tokenizer artifact binding required")
    if prompt_tokenizer_artifact is not None or prompt_tokenizer_sha256:
        if prompt_tokenizer_artifact is None or not prompt_tokenizer_sha256:
            raise ValueError("exact local tokenizer artifact binding required")
        if prompt_token_counter is not None:
            raise ValueError("choose one owner token-counter binding")
        from .local_prompt_tokenizer import load_owner_prompt_tokenizer
        local = load_owner_prompt_tokenizer(artifact=prompt_tokenizer_artifact,
            expected_sha256=prompt_tokenizer_sha256, provider=provider, model=model)
        prompt_token_counter = PromptTokenCounter(**local.binding(), count_tokens=local.count,
                                                 compatibility_basis="owner_declaration")
        tokenizer_source = local.provenance()
    elif (prompt_token_counter is None and purpose == "coding"
            and semantic_repository is None and world_task_packet is None and not census_only
            and not source_lookup_selected
            and all(policy == "legacy" for policy in (
                semantic_metadata_view, semantic_working_set, tool_output_policy,
                source_view_advisory_policy, repository_source_policy, source_input_policy,
                query_reference_policy, coding_reply_mode))):
        # Default calibration is confined to the ordinary coding route. Prepared
        # world/semantic and explicitly selected source/tool routes retain their
        # existing owner-controlled measurement and selection contracts.
        from .automatic_prompt_tokenizer import resolve_prompt_tokenizer
        prompt_token_counter, tokenizer_source = resolve_prompt_tokenizer(provider=provider, model=model)
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
    required_suffix = ""
    world_packet_receipt = None
    world_packet_artifact = None
    world_packet_artifact_sha256 = None
    world_packet_store_identity = None
    if world_task_packet is not None:
        failure_phase = "world_task_packet"
        world_packet_receipt = world_task_packet.validate_current(workspace=root, task_text=prompt)
        router_prompt = world_task_packet.prompt
        if world_packet_audit_directory is not None:
            failure_phase = "world_packet_audit_admission"
            from .repository_world_packet_replay import freeze_repository_world_packet
            from .repository_world_artifact_store import prepare_repository_world_artifact_directory
            # Capture before provider discovery/initialization can mutate source.
            # Historical bytes never substitute for the late live freshness gate.
            world_packet_artifact = freeze_repository_world_packet(world_task_packet)
            world_packet_artifact_sha256 = hashlib.sha256(world_packet_artifact).hexdigest()
            world_packet_store_identity = prepare_repository_world_artifact_directory(
                world_packet_audit_directory, workspace=root)
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
    query_reference_view = None
    query_reference_encoded = encoded
    query_reference_selected = False
    if query_reference_policy != "legacy":
        failure_phase = "query_reference_view"
        from .semantic_query_reference import project_semantic_query_reference
        from .query_reference_policy import validate_bound_receipt as validate_query_receipt
        query_reference_view = project_semantic_query_reference(provider_prompt=encoded.provider_prompt)
        validate_query_receipt(query_reference_policy, query_reference_view.receipt)
        router_prompt = query_reference_view.provider_prompt
        query_reference_selected = True
    source_input_view = None
    source_input_encoded = encoded
    if source_input_policy != "legacy":
        failure_phase = "source_input_view"
        from .source_input_view import project_source_input_view
        source_input_view = project_source_input_view(encoded=encoded, repository=semantic_repository,
            workspace=root, require_current=True, **projection_options(source_input_policy))
        from .source_input_policy import validate_bound_receipt
        validate_bound_receipt(source_input_policy, source_input_view.receipt,
            semantic_metadata_view=semantic_metadata_view)
        router_prompt = source_input_view.provider_prompt
        required_suffix += source_input_view.block
    source_view_encoded = encoded
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
        required_suffix += source_guidance
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
        required_suffix += residual_advisory
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
        if (instruction_receipt.get("schema") == "supervisor-public-instruction-inclusion@5"
                and encoded is None):
            raise ValueError("signed Terminal scope requires the verified semantic coding route")
        # Preserve the literal requirements after semantic encoding. Native
        # capsule identity remains the stdin identity, not this projection.
        router_prompt += instruction
        required_suffix += instruction
    if source_lookup_receipt is not None:
        source_lookup_guidance = _source_lookup_guidance(source_lookup_receipt)
        router_prompt += source_lookup_guidance
        required_suffix += source_lookup_guidance
    failure_phase = "model_prompt"
    repository_source_binding = None
    if repository_owner_selected:
        from .repository_source_owner_binding import create_binding as create_owner_binding
        repository_source_binding = create_owner_binding(workspace=root,
            requested_policy=repository_source_policy, descriptor_path=repository_source_owner_descriptor,
            descriptor_sha256=repository_source_owner_descriptor_sha256)
        if container_boundary is not None:
            descriptor = repository_source_binding["descriptor"]
            # A public manifest digest alone does not identify its nominated
            # owner. Bind the live descriptor to the identities returned by
            # the existing boundary verifier, without rereading its artifact.
            if (type(boundary) is not dict
                    or descriptor["boundary_manifest_sha256"] != container_boundary_sha256
                    or any(type(boundary.get(key)) is not int or descriptor[key] != boundary[key]
                           for key in ("owner_uid", "worker_uid"))):
                raise ValueError("repository owner differs from the selected container boundary")
    elif repository_source_policy != "legacy":
        from .repository_source_policy import create_binding as create_repository_binding
        repository_source_binding = create_repository_binding(workspace=root, requested_policy=repository_source_policy,
            **({"tokenizer": {"artifact": tokenizer_source["descriptor_path"],
                "expected_sha256": tokenizer_source["descriptor_sha256"], "provider": provider, "model": model}}
               if tokenizer_source is not None else {}))
    source_view_binding = None
    if source_view_advisory_policy != "legacy":
        from .source_view_policy import create_binding as create_source_binding
        source_view_binding = create_source_binding(workspace=root, requested_policy=source_view_advisory_policy)
    tool_output_binding = None
    if tool_output_policy != "legacy":
        from .tool_output_policy import create_binding
        tool_output_binding = create_binding(workspace=root, requested_policy=tool_output_policy)
    model_prompt, advisory = render_model_prompt(prompt=router_prompt, purpose=purpose, workspace=root,
                                                semantic_transport=encoded is not None,
                                                world_packet=world_task_packet is not None,
                                                tool_output_binding=tool_output_binding, source_view_binding=source_view_binding,
                                                repository_source_binding=repository_source_binding)
    coding_reply_receipt = None
    if coding_reply_mode != "legacy":
        failure_phase = "coding_reply_contract"
        from .coding_reply_contract import apply_coding_reply_contract
        model_prompt, coding_reply_receipt = apply_coding_reply_contract(
            model_prompt=model_prompt, mode=coding_reply_mode, purpose=purpose, provider=provider)
    semantic_model_prompt = model_prompt
    semantic_reply_receipt = coding_reply_receipt
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
            purpose=purpose, workspace=root, semantic_transport=True,
            tool_output_binding=tool_output_binding, source_view_binding=source_view_binding,
            repository_source_binding=repository_source_binding)
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
            purpose=purpose, workspace=root, semantic_transport=True,
            tool_output_binding=tool_output_binding, source_view_binding=source_view_binding,
            repository_source_binding=repository_source_binding)
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
    prompt_selection = None
    world_prompt_measurement = None
    metadata_candidate_receipt = None
    if world_task_packet is not None and (prompt_token_counter is not None or max_input_tokens is not None):
        failure_phase = "prompt_selection"
        from .coding_reply_contract import coding_completion_schema
        world_prompt_measurement = _measure_world_packet_prompt(
            model_prompt=model_prompt, provider=provider, model=model,
            token_counter=prompt_token_counter, max_input_tokens=max_input_tokens,
            native_output_schema=coding_completion_schema() if coding_reply_receipt is not None else None,
        )
    elif prompt_token_counter is not None or max_input_tokens is not None:
        failure_phase = "prompt_selection"
        native_router_prompt = prompt + required_suffix
        native_model_prompt, native_advisory = render_model_prompt(
            prompt=native_router_prompt, purpose=purpose, workspace=root,
            tool_output_binding=tool_output_binding, source_view_binding=source_view_binding,
            repository_source_binding=repository_source_binding)
        from .coding_reply_contract import apply_coding_reply_contract, coding_completion_schema
        native_model_prompt, native_reply_receipt = apply_coding_reply_contract(
            model_prompt=native_model_prompt, mode=coding_reply_mode, purpose=purpose, provider=provider)
        output_schema = (coding_completion_schema() if native_reply_receipt is not None
                         else codex_planning_schema)
        if response_format is not None:
            from ipfs_accelerate_py.cli_runtime.grok_structured_output import native_schema_projection
            output_schema, _, _ = native_schema_projection(
                response_format["json_schema"]["schema"], task_contract=task_contract)
        representation = ("semantic_metadata" if metadata_receipt is not None
                          and metadata_receipt["selected_mode"] in {"common-bindings@1", "common-bindings-instruction@1"} else "semantic")
        model_prompt, prompt_selection = select_model_prompt(
            native_complete_prompt=native_model_prompt,
            candidate_complete_prompt=model_prompt if encoded is not None else None,
            provider=provider, model=model, token_counter=prompt_token_counter,
            max_input_tokens=max_input_tokens, native_output_schema=output_schema,
            semantic_complete_prompt=semantic_model_prompt if representation == "semantic_metadata" else None,
            candidate_representation=representation)
        if prompt_selection["selected_representation"] == "native":
            router_prompt, advisory = native_router_prompt, native_advisory
            coding_reply_receipt = native_reply_receipt
            encoded = None
            # This census describes the translated candidate. A native input
            # must not publish it as evidence of the selected representation.
            # The token-selection receipt retains the comparison itself.
            semantic_component_census = None
            query_reference_selected = False
            # The earlier proxy receipt describes an evaluated candidate; it
            # must not attest metadata transport that the tokenizer rejected.
            metadata_candidate_receipt = metadata_receipt
            metadata_receipt = None
            metadata_native_reply = None
            displacement_receipt = None
        elif prompt_selection["selected_representation"] == "semantic" and representation == "semantic_metadata":
            coding_reply_receipt = semantic_reply_receipt
            metadata_candidate_receipt = metadata_receipt
            metadata_receipt = None
            metadata_native_reply = None
            # The complete-input selector rejected the metadata view; its
            # displacement observation no longer describes the selected input.
            displacement_receipt = None
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
        if prompt_selection is not None:
            receipt["prompt_selection"] = prompt_selection
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
    if world_packet_receipt is not None:
        receipt["world_task_packet"] = {
            **world_packet_receipt,
            "dispatch_rechecked": False,
            "provider_attempted": False,
            "provider_retained_parent": False,
            "prompt_measurement": world_prompt_measurement,
        }
    if semantic_component_census is not None:
        receipt["semantic_prompt_components"] = semantic_component_census
    if tool_output_binding is not None:
        receipt["tool_output_policy"] = tool_output_binding
    if source_view_binding is not None:
        receipt["source_view_advisory_policy"] = source_view_binding
    if repository_source_binding is not None:
        receipt["repository_source_policy"] = repository_source_binding
    if source_input_view is not None:
        receipt["source_input_policy"] = source_input_policy
        receipt["source_input_view"] = source_input_view.receipt
        receipt["source_input_dispatched"] = False
    if query_reference_view is not None:
        receipt["query_reference_policy"] = query_reference_policy
        receipt["query_reference_view"] = query_reference_view.receipt
        receipt["query_reference_selected"] = query_reference_selected
        receipt["query_reference_dispatched"] = False
        receipt["query_reference_provider_attempted"] = False
    if metadata_receipt is not None:
        receipt["semantic_metadata_view"] = metadata_receipt
    if working_set_receipt is not None:
        receipt["semantic_working_set"] = working_set_receipt
    if displacement_receipt is not None:
        receipt["representation_displacement"] = displacement_receipt
    if metadata_native_reply is not None:
        receipt["semantic_metadata_native_coding_reply_contract"] = metadata_native_reply
    if metadata_candidate_receipt is not None:
        receipt["semantic_metadata_candidate"] = metadata_candidate_receipt
    if prompt_selection is not None:
        receipt["prompt_selection"] = prompt_selection
    if tokenizer_source is not None:
        receipt["prompt_tokenizer_source"] = tokenizer_source
    failure_phase = "provider_invocation"
    provider_options = {}
    if repository_source_binding is not None:
        provider_options["codex_repository_source_binding"] = repository_source_binding
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
    scoped_instruction = (instruction_receipt is not None
        and instruction_receipt.get("schema") == "supervisor-public-instruction-inclusion@5")
    if scoped_instruction:
        receipt["terminal_semantic_scope_dispatch_rechecked"] = False
    tool_output_store_identity = None
    source_view_store_identity = None
    repository_store_identity = None
    worker_forwarding_outbox = None
    worker_forwarding_fallback = None
    worker_forwarding_interruption = None
    pending_provider_error = None
    try:
        if repository_owner_selected:
            failure_phase = "repository_source_worker_forwarding"
            try:
                from . import repository_source_worker_receipts as worker_receipts
                from .repository_source_owner_binding import with_worker_forwarding
                worker_forwarding_fallback = worker_receipts.unavailable_capture(
                    session_id=repository_source_binding["session_id"],
                    descriptor_sha256=repository_source_binding["descriptor_sha256"],
                    invocation_id=invocation, reason="capture_unavailable")
                receipt["repository_source_worker_forwarding"] = {
                    **worker_forwarding_fallback, "receipts": []}
                worker_forwarding_outbox = worker_receipts.create_outbox(workspace=root,
                    session_id=repository_source_binding["session_id"],
                    descriptor_sha256=repository_source_binding["descriptor_sha256"],
                    invocation_id=invocation)
                observed_binding = with_worker_forwarding(
                    repository_source_binding, worker_forwarding_outbox.target)
            except Exception:
                # Optional worker accounting cannot prevent a source/provider
                # invocation. Any created outbox still belongs to this finally.
                pass
            else:
                repository_source_binding = observed_binding
                receipt["repository_source_policy"] = observed_binding
                provider_options["codex_repository_source_binding"] = observed_binding
            failure_phase = "provider_invocation"
        if repository_source_binding is not None and not repository_owner_selected:
            failure_phase = "repository_source_store"
            from .repository_source_policy import create_runtime_store
            repository_store_identity = create_runtime_store(repository_source_binding)
            if repository_source_parent_lease is not None:
                from .repository_source_resource_handoff import install_repository_source_resources
                install_repository_source_resources(repository_source_binding,
                    parent_lease=repository_source_parent_lease,
                    expected_policy_sha256=repository_source_scheduler_policy_sha256,
                    deadline_monotonic=repository_source_deadline_monotonic)
            failure_phase = "provider_invocation"
        if source_view_binding is not None:
            failure_phase = "source_view_store"
            from .source_view_policy import create_audit_store
            source_view_store_identity = create_audit_store(source_view_binding)
            failure_phase = "provider_invocation"
        if tool_output_binding is not None:
            failure_phase = "tool_output_store"
            from .tool_output_artifact import create_store
            created_store = create_store(Path(tool_output_binding["store_path"]))
            tool_output_store_identity = created_store["store_identity"]
            failure_phase = "provider_invocation"
        if source_input_view is not None:
            failure_phase = "source_input_freshness"
            from .source_input_view import assert_current_source_input_view
            assert_current_source_input_view(encoded=source_input_encoded, repository=semantic_repository,
                workspace=root, receipt=source_input_view.receipt)
            failure_phase = "provider_invocation"
        if query_reference_view is not None:
            failure_phase = "query_reference_freshness"
            from .source_input_view import recheck_source_input_freshness
            recheck_source_input_freshness(encoded=query_reference_encoded, repository=semantic_repository,
                workspace=root)
            failure_phase = "provider_invocation"
        if source_receipt is not None:
            failure_phase = "source_view_descriptor"
            current = _source_broker_observation(path=source_view_descriptor,
                digest=source_view_descriptor_sha256, workspace=root, encoded=source_view_encoded,
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
        if world_task_packet is not None:
            failure_phase = "world_task_packet_currentness"
            current_packet = world_task_packet.validate_current(workspace=root, task_text=prompt)
            if current_packet != world_packet_receipt:
                raise ValueError("world task packet changed before provider dispatch")
            receipt["world_task_packet"]["dispatch_rechecked"] = True
        failure_phase = "provider_invocation"
        with _numerical_descriptor_environment(numerical_obligation_descriptor,
                                               numerical_obligation_descriptor_sha256), \
                _source_descriptor_environment(source_view_descriptor, source_view_descriptor_sha256), \
                _source_lookup_descriptor_environment(source_lookup_descriptor,
                                                      source_lookup_descriptor_sha256):
            if repository_source_parent_lease is not None:
                failure_phase = "repository_source_resources"
                validate_owner_resources(parent_lease=repository_source_parent_lease,
                    expected_policy_sha256=repository_source_scheduler_policy_sha256,
                    deadline_monotonic=repository_source_deadline_monotonic)
            if repository_owner_selected:
                failure_phase = "repository_source_owner_currentness"
                from .repository_source_owner_binding import validate_binding as validate_owner_binding
                validate_owner_binding(repository_source_binding, workspace=root, materialized=True)
            if scoped_instruction:
                # Initialization, descriptor checks and working-set preflight
                # may run caller code. Close the complete signed source replay
                # after those callbacks and descriptor-context entry, without
                # native owner keys or trusting an RPC freshness claim.
                failure_phase = "terminal_semantic_scope_currentness"
                current_instruction, current_receipt = load_public_instruction(
                    artifact=public_instruction_artifact, expected_sha256=public_instruction_sha256,
                    task_cid=public_instruction_task_cid, prompt=prompt, workspace=root,
                    repository=semantic_repository, require_current_source=True)
                if current_instruction != instruction or current_receipt != instruction_receipt:
                    raise ValueError("signed Terminal scope changed before provider dispatch")
                if Path.cwd().resolve(strict=True) != root:
                    raise ValueError("signed Terminal scope requires the current allocated worktree")
                receipt["terminal_semantic_scope_dispatch_rechecked"] = True
            failure_phase = "provider_invocation"
            if source_input_view is not None:
                receipt["source_input_dispatched"] = True
            if query_reference_view is not None:
                receipt["query_reference_dispatched"] = query_reference_selected
                receipt["query_reference_provider_attempted"] = True
            receipt["router_calls"] = 1
            if world_task_packet is not None:
                receipt["world_task_packet"]["provider_attempted"] = True
            output = generate_text(
                model_prompt, provider=provider, model_name=model, provider_instance=instance, deps=deps,
                allow_local_fallback=False, allow_cross_provider_fallback=False,
                timeout=timeout, max_tokens=max_output_tokens, max_new_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                # Both purposes launch a workspace-capable agent and require a fresh
                # native receipt. Cached response text cannot replay those effects.
                side_effecting=True,
                task_kind="coding", allocation_path="cli", allocation_session_id=invocation,
                codex_cli_literal_response=codex_cli_literal_response,
                temperature=0,
                **provider_options,
            )
        failure_phase = "provider_result_validation"
        observation = get_last_cli_observation(provider)
        if repository_source_binding is not None and (
                observation.get("codex_repository_source_config_sha256") != repository_source_binding["config_sha256"]
                or observation.get("codex_repository_source_configured") is not True):
            raise RuntimeError("repository source native configuration observation is missing or differs")
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
        pending_provider_error = error
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
        try:
            if worker_forwarding_outbox is not None:
                try:
                    receipt["repository_source_worker_forwarding"] = worker_forwarding_outbox.collect_and_close()
                except BaseException as capture_error:
                    # Setup established this closed snapshot before allocating
                    # the outbox. Reuse it without another fallible metadata
                    # callback or mutable binding lookup during finalization.
                    receipt["repository_source_worker_forwarding"] = {
                        **worker_forwarding_fallback, "reason": "capture_failed", "receipts": []}
                    # Preserve an existing provider failure. A fresh control-flow
                    # interruption propagates after ordinary receipt publication.
                    if not isinstance(capture_error, Exception) and pending_provider_error is None:
                        worker_forwarding_interruption = capture_error
            if repository_owner_selected and repository_source_binding is not None:
                from .repository_source_owner_binding import unavailable_usage
                repository_source_binding["usage"] = unavailable_usage()
            elif repository_source_binding is not None:
                from .repository_source_policy import finish_runtime_store
                repository_source_binding["usage"] = finish_runtime_store(repository_source_binding,
                    expected_identity=repository_store_identity)
            if source_view_binding is not None:
                from .source_view_policy import finish_audit_store
                source_view_binding["usage"] = finish_audit_store(source_view_binding, workspace=root,
                    expected_identity=source_view_store_identity)
            if tool_output_binding is not None:
                try:
                    from .tool_output_artifact import summarize_store
                    tool_output_binding["usage"] = summarize_store(Path(tool_output_binding["store_path"]))
                except Exception:
                    # Worker-writable diagnostics cannot mask a provider failure
                    # or grant task/completion authority.
                    tool_output_binding["usage"] = {"status": "unavailable", "helper_operations_observed": None}
                usage = tool_output_binding["usage"]
                if "helper_operations_observed" not in usage:
                    usage["helper_operations_observed"] = bool(usage.get("capture_count", 0) or usage.get("read_count", 0))
                usage["store_cleanup_complete"] = False
                usage["artifacts_retained_after_invocation"] = None
                if tool_output_store_identity is not None:
                    try:
                        from .tool_output_artifact import remove_store
                        cleanup = remove_store(Path(tool_output_binding["store_path"]),
                            expected_identity=tool_output_store_identity)
                        usage["store_cleanup_complete"] = cleanup["removed"] is True
                        if cleanup["removed"] is True:
                            usage["artifacts_retained_after_invocation"] = False
                    except Exception:
                        usage["store_cleanup_error"] = "cleanup_unavailable"
            observation = get_last_cli_observation(provider)
            # Keep native categories intact; do not add cached tokens to input or
            # infer a missing total. Provider failures remain measured attempts.
            receipt["usage"] = {key: observation[key] for key in (
                "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
                "total_cost_usd", "model_id", "session_id", "thread_id", "num_turns", "exit_code", "timed_out",
                "codex_output_schema_sha256", "codex_output_schema_bytes",
                "codex_repository_source_config_sha256", "codex_repository_source_configured",
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
            if world_packet_artifact is not None:
                try:
                    from .repository_world_artifact_store import store_repository_world_invocation_artifact
                    receipt["world_packet_replay_artifact"] = store_repository_world_invocation_artifact(
                        directory=world_packet_audit_directory, expected_identity=world_packet_store_identity,
                        packet_artifact=world_packet_artifact,
                        expected_packet_sha256=world_packet_artifact_sha256, invocation_receipt=receipt)
                except Exception:
                    # Optional retained observation cannot mask a provider success,
                    # failure, or rate limit. Never print artifact bodies or paths.
                    receipt["world_packet_replay_capture"] = {
                        "schema": "repository-world-invocation-capture@1", "status": "unavailable",
                        "reason": "artifact_capture_failed", "raw_content_exported": False,
                        "completion_authority": False,
                    }
            receipt["seconds"] = time.monotonic() - started
            try:
                from .context_delivery_accounting import observe_invocation
                receipt["context_delivery_observation"] = observe_invocation(receipt)
            except Exception:
                # Accounting is a derived observation. It must not replace a
                # provider failure, change its counters, or prevent publication.
                receipt["context_delivery_observation"] = {
                    "schema": "supervisor-context-delivery-invocation@1",
                    "status": "unavailable", "reason": "observer_unavailable",
                    "observation_only": True, "completion_authority": False,
                    "provider_ingestion_verified": False,
                    "total_token_savings_measured": False,
                    "raw_content_exported": False,
                }
        finally:
            if previous is None:
                os.environ.pop("ipfs_accelerate_py_CODEX_SANDBOX", None)
            else:
                os.environ["ipfs_accelerate_py_CODEX_SANDBOX"] = previous
            if previous_bytecode is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = previous_bytecode
        if receipt_observer is not None:
            try:
                receipt_observer(json.loads(json.dumps(receipt)))
            except BaseException as observer_error:
                receipt["receipt_observer_error_type"] = type(observer_error).__name__
                # Diagnostic callbacks cannot replace a pending provider failure
                # or the first deferred capture interruption.
                # A fresh interruption still propagates after environment cleanup.
                if (not isinstance(observer_error, Exception) and pending_provider_error is None
                        and worker_forwarding_interruption is None):
                    raise
        print(json.dumps(receipt, sort_keys=True), flush=True)
        if worker_forwarding_interruption is not None:
            raise worker_forwarding_interruption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="codex_cli", choices=["codex_cli", "grok_cli"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--budget-profile")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int,
                        help="Local budget for rendered user prompt and native output schema; requires a bound tokenizer")
    parser.add_argument("--prompt-tokenizer-artifact", type=Path)
    parser.add_argument("--prompt-tokenizer-sha256")
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
    from .tool_output_policy import POLICIES as TOOL_OUTPUT_POLICIES
    parser.add_argument("--tool-output-policy", default="legacy", choices=TOOL_OUTPUT_POLICIES)
    from .source_view_policy import POLICIES as SOURCE_VIEW_POLICIES
    parser.add_argument("--source-view-advisory-policy", default="legacy", choices=SOURCE_VIEW_POLICIES)
    from .repository_source_policy import POLICIES as REPOSITORY_SOURCE_POLICIES
    parser.add_argument("--repository-source-policy", default="legacy", choices=REPOSITORY_SOURCE_POLICIES)
    parser.add_argument("--repository-source-owner-descriptor", type=Path)
    parser.add_argument("--repository-source-owner-descriptor-sha256")
    from .source_input_policy import POLICIES as SOURCE_INPUT_POLICIES
    parser.add_argument("--source-input-policy", default="legacy", choices=SOURCE_INPUT_POLICIES)
    from .query_reference_policy import POLICIES as QUERY_REFERENCE_POLICIES
    parser.add_argument("--query-reference-policy", default="legacy", choices=QUERY_REFERENCE_POLICIES)
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
    if ((args.prompt_tokenizer_artifact is None) != (args.prompt_tokenizer_sha256 is None)
            or args.prompt_tokenizer_sha256 == ""):
        parser.error("both tokenizer binding arguments require nonempty values")
    if args.max_input_tokens is not None and args.max_input_tokens < 1:
        parser.error("max-input-tokens must be positive")
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
                        tool_output_policy=args.tool_output_policy,
                        source_view_advisory_policy=args.source_view_advisory_policy,
                        repository_source_policy=args.repository_source_policy,
                        repository_source_owner_descriptor=args.repository_source_owner_descriptor,
                        repository_source_owner_descriptor_sha256=args.repository_source_owner_descriptor_sha256,
                        source_input_policy=args.source_input_policy,
                        query_reference_policy=args.query_reference_policy,
                        census_only=args.semantic_metadata_census_only,
                        coding_reply_mode=args.coding_reply_mode,
                        doctor_residual_artifact=args.doctor_residual_artifact,
                        doctor_residual_sha256=args.doctor_residual_sha256,
                        doctor_residual_task_cid=args.doctor_residual_task_cid,
                        public_instruction_artifact=args.public_instruction_artifact,
                        public_instruction_sha256=args.public_instruction_sha256,
                        public_instruction_task_cid=args.public_instruction_task_cid,
                        prompt_tokenizer_artifact=args.prompt_tokenizer_artifact,
                        prompt_tokenizer_sha256=args.prompt_tokenizer_sha256 or "",
                        max_input_tokens=args.max_input_tokens)
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
