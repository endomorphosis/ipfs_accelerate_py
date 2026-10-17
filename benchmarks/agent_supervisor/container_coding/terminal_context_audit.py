"""Bounded historical context/input audit, without exporting prompt contents.

The supervisor calls this after collecting router receipts and before Harbor
removes the container. Receipt workspaces remain usable as input identities
after workers exit or their worktrees are removed. This is observation only:
neither successful reconstruction nor a provider receipt grants authority or
establishes task correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.machinery import ModuleSpec
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time


SCHEMA = "terminal-final-context-audit@1"
KINDS = ("semantic-context", "code-retrieval-context", "intent-world-context")
MAX_FILE_BYTES = 2_000_000
MAX_INVENTORY_ENTRIES = 20_000
MAX_CAPSULES = 32
MAX_RECEIPTS = 16
MAX_DEPTH = 32
MAX_REQUEST_BYTES = 65_536
MAX_RESULT_BYTES = 262_144
MAX_CENSUS_BYTES = 131_072
MAX_AUDIT_SECONDS = 30.0
RECEIPT_FIELDS = (
    "schema", "phase", "invocation_id", "purpose", "workspace",
    "prompt_sha256", "prompt_bytes", "native_prompt_sha256", "native_prompt_bytes",
    "model_prompt_sha256", "model_prompt_bytes", "workspace_advisory_sha256",
    "workspace_advisory_bytes",
)
TRANSLATION_FIELDS = ('router_prompt_sha256', 'router_prompt_bytes', 'semantic_translation', 'doctor_residual_context', 'public_instruction', 'coding_reply_contract', 'semantic_metadata_view', 'semantic_metadata_native_coding_reply_contract', 'source_formalization_context', 'source_view_retrieval', 'representation_displacement', 'terminal_semantic_scope_dispatch_rechecked', 'semantic_prompt_components', 'tool_output_policy', 'source_view_advisory_policy', 'repository_source_policy', 'source_input_policy', 'source_input_view', 'source_input_dispatched', 'query_reference_policy', 'query_reference_view', 'query_reference_selected', 'query_reference_dispatched', 'query_reference_provider_attempted', 'semantic_working_set', 'semantic_working_set_preflight')

DECISION_REPORT_FIELDS = (
    "semantic_decision_dialogues", "semantic_decision_dialogue_expected",
    "semantic_decision_owner_bindings", "semantic_decision_owner_binding_expected",
    "semantic_decision_expected_task_id", "semantic_decision_nomination_sha256",
    "semantic_decision_typed_nomination_expected",
)


def _semantic_prompt_components(value):
    """Accept only the closed, bounded diagnostic; replay checks its numbers."""
    schema = "supervisor-semantic-prompt-component-census@1"
    if _same_json(value, {"schema": schema, "status": "unavailable", "raw_content_exported": False}):
        return False
    fixed = dict(schema=schema,
        measurement_scope="canonical-serialized-prompt-utf8-bytes@1",
        partition_rule="member_bytes_plus_outer_punctuation_equals_object_bytes",
        nested_partitions_overlap=True, token_estimate=False, provider_tokens=None,
        model_prompt_wrappers_included=False, provider_calls=0,
        raw_content_exported=False, dynamic_labels_exported=False,
        source_freshness_verified=False, proof_authority=False, omission_authority=False,
        completion_authority=False, execution_authority=False,
        exact_native_round_trip_verified=True)
    partitions = {
        "native_object": ("evidence", "goal", "scope", "acceptance", "authority"),
        "provider_object": ("native_context", "translated_semantic", "translation_table", "instructions", "native_suffix"),
        "remaining_native_context": ("evidence", "goal", "scope", "acceptance", "authority"),
        "translated_semantic": ("capsules", "admissions", "raw_sources", "manifest"),
    }
    evidence = ("native_evidence", "remaining_native_evidence")
    digests = ("native_prompt_sha256", "provider_prompt_sha256")
    counts = ("native_prompt_bytes", "native_suffix_bytes", "provider_prompt_bytes")
    if (type(value) is not dict or set(value) != {*fixed, *partitions, *evidence, *digests, *counts}
            or any(not _same_json(value[key], expected) for key, expected in fixed.items())
            or len(json.dumps(value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode()) > 16_384):
        raise ValueError("invalid semantic prompt component census")

    def integer(number):
        if type(number) is not int or not 0 <= number <= 2_000_000:
            raise ValueError("invalid semantic prompt component count")

    for key in digests:
        _digest(value[key])
    for key in counts:
        integer(value[key])
    for key, labels in partitions.items():
        part = value[key]
        if (type(part) is not dict or set(part) != {"object_bytes", "outer_punctuation_bytes", "members"}
                or type(part["members"]) is not dict or set(part["members"]) != {*labels, "other_fields"}):
            raise ValueError("invalid semantic prompt component partition")
        integer(part["object_bytes"])
        integer(part["outer_punctuation_bytes"])
        for row in part["members"].values():
            if type(row) is not dict or set(row) != {"count", "value_bytes", "member_bytes"}:
                raise ValueError("invalid semantic prompt component member")
            for number in row.values():
                integer(number)
    for key in evidence:
        part = value[key]
        if (type(part) is not dict or set(part) != {"array_bytes", "outer_punctuation_bytes", "groups"}
                or type(part["groups"]) is not dict or set(part["groups"]) != {
                    "semantic_context", "source_formalization_context", "other_evidence"}):
            raise ValueError("invalid semantic prompt evidence census")
        integer(part["array_bytes"])
        integer(part["outer_punctuation_bytes"])
        for row in part["groups"].values():
            if type(row) is not dict or set(row) != {"count", "row_bytes", "summary_utf8_bytes",
                    "summary_json_string_bytes", "summary_json_string_overhead_bytes", "row_envelope_bytes"}:
                raise ValueError("invalid semantic prompt evidence group")
            for number in row.values():
                integer(number)
    return True


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same_json(left, right) -> bool:
    """Compare typed canonical values: false is distinct from zero."""
    options = dict(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return json.dumps(left, **options) == json.dumps(right, **options)


def _terminal_scope_inclusion(value):
    """Validate bounded signed-selection observations, never native grants."""
    identities = ("manifest_cid", "declaration_cid", "frontier_cid", "scope_observation_cid",
                  "scope_cid", "semantic_root_cid")
    digests = ("profile_sha256", "semantic_payload_sha256")
    counts = ("selected_count", "full_count", "frontier_count", "selected_source_bytes",
              "max_files", "max_source_bytes", "required_count", "program_count")
    flags = ("native_claim_verified", "native_allocation_verified", "process_identity_verified",
             "proof_authority", "execution_authority", "completion_authority", "omission_authority")
    if (type(value) is not dict or set(value) != {"schema", *identities, *digests, *counts, *flags}
            or value["schema"] != "terminal-signed-semantic-worker-inclusion@1"
            or any(value[key] is not False for key in flags)
            or any(type(value[key]) is not int for key in counts)
            or value["max_files"] != 64 or value["max_source_bytes"] != 1_000_000
            or not 4 <= value["selected_count"] <= 64
            or not value["selected_count"] <= value["full_count"] <= 255
            or value["frontier_count"] != value["full_count"] - value["selected_count"]
            or not 0 <= value["selected_source_bytes"] <= 1_000_000
            or not 3 <= value["required_count"] <= value["selected_count"]
            or not 1 <= value["program_count"] <= value["selected_count"] - 3
            or value["required_count"] + value["program_count"] < value["selected_count"]):
        raise ValueError("invalid signed Terminal semantic scope inclusion")
    for key in identities:
        _identity(value[key])
    for key in digests:
        _digest(value[key])


def source_input_audit_binding_sha256(item: dict) -> str:
    """Bind a successful audit to its complete bounded coding receipt projection.

    This digest is an audit join key, not evidence of successful reconstruction
    by itself. Keep it outside the boolean model-input checks.
    """
    if (type(item) is not dict or item.get("schema") != "router-implementation-invocation@1"
            or item.get("phase") != "coding" or item.get("purpose") != "coding"
            or item.get("provider") != "codex_cli" or item.get("source_input_view") is None
            or type(item.get("source_input_dispatched")) is not bool):
        raise ValueError("source input audit binding requires an explicit coding receipt")
    projection = {key: item[key] for key in RECEIPT_FIELDS}
    projection.update({key: item[key] for key in TRANSLATION_FIELDS if key in item})
    projection["provider"] = item["provider"]
    raw = json.dumps({"schema": "terminal-source-input-audit-binding@1", "receipt": projection},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("source input audit binding exceeds receipt bound")
    return _sha(raw)


def query_reference_audit_binding_sha256(item: dict) -> str:
    """Bind a successful audit to its complete bounded coding receipt projection.

    This digest is an audit join key, not evidence of successful reconstruction
    by itself. Keep it outside the boolean model-input checks.
    """
    if (type(item) is not dict or item.get("schema") != "router-implementation-invocation@1"
            or item.get("phase") != "coding" or item.get("purpose") != "coding"
            or item.get("provider") != "codex_cli" or item.get("query_reference_view") is None
            or type(item.get("query_reference_dispatched")) is not bool
            or type(item.get("query_reference_provider_attempted")) is not bool
            or type(item.get("query_reference_selected")) is not bool):
        raise ValueError("query reference audit binding requires an explicit coding receipt")
    projection = {key: item[key] for key in RECEIPT_FIELDS}
    projection.update({key: item[key] for key in TRANSLATION_FIELDS if key in item})
    projection["provider"] = item["provider"]
    raw = json.dumps({"schema": "terminal-query-reference-audit-binding@1", "receipt": projection},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("query reference audit binding exceeds receipt bound")
    return _sha(raw)


def _base(status="unknown", reason="observation_unavailable") -> dict:
    return {"schema": SCHEMA, "status": status, "reason": reason,
        "any_native_input_verified": None, "any_model_input_verified": None,
        "all_observed_coding_inputs_verified": None,
        "capsules": [], "invocations": [], "matches": [], "read_errors": [],
        "rendered_input_census": {"schema": "terminal-rendered-input-census@1",
            "status": "unavailable", "reason": "no_exact_model_input",
            "observations": [], "unavailable": [], "provider_calls": 0,
            "verified_invocation_count": 0, "observed_invocation_count": 0,
            "raw_content_exported": False, "token_savings_established": False},
        "world_scope": "sealed historical capture; claims may supersede its intent revision",
        "input_scope": "router task prompt; provider harness/system context is outside this digest",
        "raw_prompts_exported": False, "provider_calls": 0,
        "proof_authority": False, "formalization_authority": False,
        "source_obligations_discharged": 0,
        "execution_authority": False, "completion_authority": False,
        "task_correctness_established": False}


def _add_rendered_input_census(result, *, model_prompt, workspace_advisory, invocation_id,
                              input_representation=None):
    """Append bounded optional metadata after the caller's exact input checks."""
    census = result["rendered_input_census"]
    identity = _sha(invocation_id.encode("utf-8"))
    if any(row["invocation_id_sha256"] == identity
           for row in (*census["observations"], *census["unavailable"])):
        return
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.rendered_input_census import SCHEMA, census_rendered_input
        options = {} if input_representation is None else {"input_representation": input_representation}
        observation = census_rendered_input(model_prompt=model_prompt,
            workspace_advisory=workspace_advisory, invocation_id=invocation_id, **options)
        if (type(observation) is not dict or observation.get("schema") != SCHEMA
                or observation.get("invocation_id_sha256") != identity
                or observation.get("model_prompt_sha256") != _sha(model_prompt.encode("utf-8"))
                or type(observation.get("model_prompt_bytes")) is not int
                or observation["model_prompt_bytes"] != len(model_prompt.encode("utf-8"))
                or observation.get("input_representation") != input_representation):
            raise ValueError("census result binding unavailable")
        rows = [*census["observations"], observation]
        if (len(rows) > MAX_RECEIPTS
                or sum(len(json.dumps(row, sort_keys=True, allow_nan=False).encode())
                       for row in rows) > MAX_CENSUS_BYTES):
            raise ValueError("census result budget unavailable")
    except Exception:
        census["unavailable"].append({"invocation_id_sha256": identity,
            "reason": "census_bound_or_input_unavailable"})
    else:
        census["observations"].append(observation)


def _finish_rendered_input_census(result):
    """Bound optional census output across routes without changing audit verdicts."""
    census = result.get("rendered_input_census")
    if census is None:
        return result
    verified = {_sha(row["invocation_id"].encode("utf-8")) for row in result["invocations"]
                if row.get("model_input_verified") is True}
    census["verified_invocation_count"] = len(verified)
    known = {row["invocation_id_sha256"] for row in (*census["observations"], *census["unavailable"])}
    census["unavailable"].extend({"invocation_id_sha256": identity,
        "reason": "census_bound_or_input_unavailable"} for identity in sorted(verified - known))
    while True:
        census["observed_invocation_count"] = len(census["observations"])
        if census["observations"]:
            census["status"] = ("complete_for_verified_inputs"
                if len(census["observations"]) == len(verified) else "partial_for_verified_inputs")
            census["reason"] = "one_observation_per_exact_verified_router_invocation"
        elif verified:
            census["status"], census["reason"] = "unavailable", "census_bound_or_input_unavailable"
        if (not census["observations"] or (
                sum(len(json.dumps(row, sort_keys=True).encode()) for row in census["observations"])
                    <= MAX_CENSUS_BYTES
                and len(json.dumps(result, sort_keys=True).encode()) <= MAX_RESULT_BYTES)):
            break
        removed = census["observations"].pop()
        census["unavailable"].append({"invocation_id_sha256": removed["invocation_id_sha256"],
            "reason": "audit_result_budget_unavailable"})
    if len(json.dumps(result, sort_keys=True).encode()) > MAX_RESULT_BYTES:
        result.pop("rendered_input_census")
    return result


def _open_directory(path: Path) -> int:
    """Open every absolute directory component without following a symlink."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("noncanonical directory")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_at(root_fd: int, relative: str, limit=MAX_FILE_BYTES) -> bytes:
    """Race-safe, bounded regular-file read under the already opened state."""
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(p in {".", ".."} for p in parts):
        raise ValueError("invalid state-relative file")
    parent = os.dup(root_fd)
    fd = None
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=parent)
            os.close(parent)
            parent = next_fd
        # NONBLOCK prevents a substituted FIFO from blocking before fstat.
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
            raise ValueError("nonregular or oversized state file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(raw) > limit or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError("state file changed during observation")
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def _json(raw: bytes):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique)


def _identity(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_:./-]{1,512}", value) is None:
        raise ValueError("invalid context identity")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid digest")
    return value


def _capsule_paths(root_fd: int) -> list[str]:
    result, entries = [], 0
    def failed(error):
        raise error
    for directory, dirs, files, _ in os.fwalk(".", topdown=True, follow_symlinks=False,
                                             dir_fd=root_fd, onerror=failed):
        entries += len(dirs) + len(files)
        if entries > MAX_INVENTORY_ENTRIES or len(PurePosixPath(directory).parts) > MAX_DEPTH:
            raise ValueError("state inventory exceeds audit bound")
        dirs.sort()
        for name in sorted(files):
            if name.endswith("-base-context-capsule.json"):
                result.append((PurePosixPath(directory) / name).as_posix())
                if len(result) > MAX_CAPSULES:
                    raise ValueError("capsule inventory exceeds audit bound")
    return result


def _formalization_receipt(value: dict) -> dict:
    """Check the closed dispatch observation without asserting freshness now."""
    from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _observation
    if type(value) is not dict:
        raise ValueError("invalid source formalization dispatch observation")
    kinds = value.get("context_kinds")
    allowed = ("source384-candidate-context", "numerical-formalization-context", "learned-numerical-selection-context")
    if (type(kinds) is not list or not kinds or kinds != [kind for kind in allowed if kind in kinds]
            or type(value.get("owner_validation_inherited")) is not bool
            or type(value.get("chunks")) is not int or not 1 <= value["chunks"] <= 6
            or type(value.get("source_files")) is not int or not 0 <= value["source_files"] <= 768):
        raise ValueError("invalid source formalization dispatch population")
    payload = dict(task_id=_identity(value.get("task_id")), task_cid=_identity(value.get("task_cid")),
        context_bundle={"sha256": _digest(value.get("context_bundle_sha256"))},
        contexts=[{"kind": kind} for kind in kinds])
    expected = _observation(payload, "", value["chunks"], range(value["source_files"]),
                            inherited=value["owner_validation_inherited"])
    expected["worker_evidence_sha256"] = _digest(value.get("worker_evidence_sha256"))
    if not _same_json(value, expected):
        raise ValueError("invalid source formalization dispatch contract")
    return value


def _token_selection_receipt(metadata: dict, item: dict) -> None:
    """Validate recorded @2 arithmetic; a receipt cannot authenticate its counter."""
    base_fields = {"schema", "view_receipt_sha256", "selected_mode", "fallback_reason",
        "token_proxy", "candidate_only", "native_complete_sha256", "native_complete_bytes",
        "native_complete_proxy_tokens", "candidate_complete_sha256", "candidate_complete_bytes",
        "candidate_complete_proxy_tokens", "selected_complete_sha256", "selected_complete_bytes",
        "selected_complete_proxy_tokens", "source_freshness_verified", "program_semantics_proved",
        "omission_authority", "proof_authority", "execution_authority", "completion_authority",
        "publication_authority"}
    token_fields = {"token_counter_status", "token_counter_identity", "token_counter_identity_sha256",
        "native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens",
        "selection_basis", "actual_provider_usage_verified", "provider_usage_reported",
        "cli_scaffolding_measured", "complete_run_token_savings_claimed"}
    if "requested_mode" in metadata:
        base_fields.add("requested_mode")
    requested = metadata.get("requested_mode", "common-bindings@1")
    if type(requested) is not str or requested not in {"common-bindings@1", "common-bindings-instruction@1"}:
        raise ValueError("invalid requested metadata representation")
    if (set(metadata) != base_fields | token_fields
            or metadata["selection_basis"] != "token_counter"
            or metadata["token_proxy"] != "utf8-bytes-ceil-div4@1"
            or any(metadata[key] is not False for key in (
                "actual_provider_usage_verified", "provider_usage_reported",
                "cli_scaffolding_measured", "complete_run_token_savings_claimed"))):
        raise ValueError("invalid token-counted metadata selection schema or usage claim")
    status = metadata["token_counter_status"]
    identity = metadata["token_counter_identity"]
    if type(status) is not str or status not in {"counted", "failed", "invalid"}:
        raise ValueError("invalid metadata token counter status")
    for key in ("provider", "model"):
        value = item.get(key)
        if type(value) is not str or not value.strip() or len(value.encode()) > 4096:
            raise ValueError("token-counted metadata requires caller provider/model identity")
    if status == "invalid":
        if identity is not None or metadata["token_counter_identity_sha256"] is not None:
            raise ValueError("invalid counter cannot carry a qualified identity")
    else:
        identity_fields = {"schema", "provider", "model", "tokenizer", "tokenizer_revision",
            "qualification", "measurement_scope", "identity_caller_supplied", "provider_usage_verified"}
        if (type(identity) is not dict or set(identity) != identity_fields
                or identity["schema"] != "supervisor-semantic-metadata-token-counter@1"
                or identity["measurement_scope"] != "complete-rendered-prompt-text@1"
                or type(identity["qualification"]) is not str
                or identity["qualification"] not in {"provider_tokenizer", "estimate"}
                or identity["identity_caller_supplied"] is not True
                or identity["provider_usage_verified"] is not False):
            raise ValueError("invalid metadata token counter identity or measurement scope")
        for key in ("provider", "model", "tokenizer", "tokenizer_revision", "qualification"):
            value = identity[key]
            if type(value) is not str or not value.strip() or len(value.encode()) > 4096:
                raise ValueError("invalid metadata token counter identity field")
        if any(identity[key] != item[key] for key in ("provider", "model")):
            raise ValueError("metadata token counter differs from caller provider/model")
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
        if _digest(metadata["token_counter_identity_sha256"]) != _sha(raw):
            raise ValueError("metadata token counter identity hash differs")
    token_keys = ("native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens")
    if status == "counted":
        if any(type(metadata[key]) is not int or not 1 <= metadata[key] <= 2**63 - 1
               for key in token_keys):
            raise ValueError("invalid complete input token count")
        selected = (metadata["candidate_complete_bytes"] < metadata["native_complete_bytes"]
                    and metadata["candidate_complete_tokens"] < metadata["native_complete_tokens"])
        fallback = None if selected else "complete_input_not_smaller_under_bytes_and_token_counter"
    else:
        if any(metadata[key] is not None for key in token_keys):
            raise ValueError("unavailable counter cannot report token measurements")
        selected = False
        fallback = "complete_input_token_counter_" + status
    chosen = "candidate" if selected else "native"
    if (metadata["selected_mode"] != (requested if selected else "legacy")
            or metadata["fallback_reason"] != fallback
            or any(metadata["selected_complete_" + field] != metadata[chosen + "_complete_" + field]
                   for field in ("sha256", "bytes", "proxy_tokens", "tokens"))
            or any(metadata[arm + "_complete_proxy_tokens"] != (metadata[arm + "_complete_bytes"] + 3) // 4
                   for arm in ("native", "candidate", "selected"))):
        raise ValueError("token-counted metadata selection arithmetic differs")


def _receipt(item: dict, workspace_root: Path) -> dict:
    if not isinstance(item, dict) or item.get("schema") != "router-implementation-invocation@1":
        raise ValueError("invalid router receipt")
    if "world_task_packet" in item:
        # World input has a dedicated frozen artifact route. Never drop its
        # marker and audit it as the legacy rendered-capsule prompt, even if
        # a caller supplies matching legacy hashes.
        raise ValueError("world task packet requires qualified Terminal Bench replay")
    if item.get("phase") != "coding" or item.get("purpose") != "coding":
        raise ValueError("non-coding router receipt")
    result = {key: item[key] for key in RECEIPT_FIELDS}
    result.update({key: item[key] for key in TRANSLATION_FIELDS if key in item})
    # Historical dialogue joins require the independently recorded route. A
    # selected policy cannot supply a missing provider or model identity.
    for field in ("provider", "model"):
        if field in item:
            result[field] = _identity(item[field])
    if "source_formalization_context" in result:
        _formalization_receipt(result["source_formalization_context"])
    _identity(result["invocation_id"])
    location = result["workspace"]
    if not isinstance(location, str) or len(location.encode()) > 4096:
        raise ValueError("invalid workspace identity")
    workspace, root = Path(location), Path(workspace_root)
    if (not workspace.is_absolute() or ".." in workspace.parts
            or str(workspace) != location or not root.is_absolute()
            or workspace == root or not workspace.is_relative_to(root)):
        raise ValueError("workspace outside configured allocation root")
    for field in ("prompt_sha256", "native_prompt_sha256", "model_prompt_sha256", "workspace_advisory_sha256"):
        _digest(result[field])
    for field in ("prompt_bytes", "native_prompt_bytes", "model_prompt_bytes", "workspace_advisory_bytes"):
        if type(result[field]) is not int or not 0 < result[field] <= 262_144:
            raise ValueError("invalid input byte count")
    translated = result.get("semantic_translation") is not None
    if "semantic_prompt_components" in result:
        if not translated:
            raise ValueError("semantic component census requires its translated input")
        _semantic_prompt_components(result["semantic_prompt_components"])
    instruction = result.get("public_instruction")
    source_input = result.get("source_input_view")
    query_reference = result.get("query_reference_view")
    if query_reference is not None or result.get("query_reference_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.query_reference_policy import validate_bound_receipt
        validate_bound_receipt(result.get("query_reference_policy"), query_reference,
            semantic_metadata_view="legacy" if result.get("semantic_metadata_view") is None else "common-bindings@1",
            source_input_policy=result.get("source_input_policy", "legacy"))
        if (item.get("provider") != "codex_cli" or type(result.get("query_reference_selected")) is not bool
                or type(result.get("query_reference_dispatched")) is not bool
                or type(result.get("query_reference_provider_attempted")) is not bool
                or result["query_reference_selected"] is not translated
                or result["query_reference_dispatched"] and (not result["query_reference_selected"]
                    or not result["query_reference_provider_attempted"])):
            raise ValueError("invalid query reference coding selection")
        result["provider"] = item["provider"]
    if source_input is not None or result.get("source_input_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.source_input_policy import validate_bound_receipt
        validate_bound_receipt(result.get("source_input_policy"), source_input,
            semantic_metadata_view="legacy" if result.get("semantic_metadata_view") is None else "common-bindings@1")
        if item.get("provider") != "codex_cli":
            raise ValueError("source input requires the Codex coding route")
        result["provider"] = item["provider"]
        if type(result.get("source_input_dispatched")) is not bool:
            raise ValueError("source input requires explicit provider boundary observation")
        if source_input["factor_ast"] and not translated:
            raise ValueError("source input AST references require semantic transport")
    router_bytes = result.get("router_prompt_bytes", result["native_prompt_bytes"])
    if translated or instruction is not None or source_input is not None or query_reference is not None:
        _digest(result["router_prompt_sha256"])
        if type(router_bytes) is not int or not 0 < router_bytes <= 262_144:
            raise ValueError("invalid translated input byte count")
    if translated:
        translation = result["semantic_translation"]
        if (not isinstance(translation, dict)
                or translation.get("schema") not in {"supervisor-semantic-router-encoding@1",
                                                     "supervisor-semantic-router-encoding@2"}
                or translation.get("freshness_checked") is not True):
            raise ValueError("invalid semantic translation receipt")
        if (translation["schema"] == "supervisor-semantic-router-encoding@2"
                and translation.get("transport_schema") != "supervisor-semantic-router-input@2"):
            raise ValueError("invalid semantic transport version")
        if (translation["schema"] == "supervisor-semantic-router-encoding@1"
                and translation.get("transport_schema", "supervisor-semantic-router-input@1")
                != "supervisor-semantic-router-input@1"):
            raise ValueError("invalid semantic transport version")
        _identity(translation["translation_cid"])
    elif instruction is None and source_input is None and query_reference is None and (router_bytes != result["native_prompt_bytes"]
          or result.get("router_prompt_sha256", result["native_prompt_sha256"]) != result["native_prompt_sha256"]):
        raise ValueError("unexplained router projection")
    residual = result.get("doctor_residual_context")
    if residual is not None:
        if ((not translated and source_input is None and query_reference is None) or not isinstance(residual, dict)
                or residual.get("schema") != "doctor-residual-router-context@1"
                or residual.get("candidate_only") is not True):
            raise ValueError("invalid Doctor residual receipt")
        for key in ("artifact_sha256", "advisory_sha256"):
            _digest(residual[key])
        _identity(residual["task_cid"])
    if instruction is not None:
        if (not isinstance(instruction, dict)
                or instruction.get("schema") not in {"supervisor-public-instruction-inclusion@1", "supervisor-public-instruction-inclusion@2", "supervisor-public-instruction-inclusion@5"}
                or any(instruction.get(k) is not True for k in
                       ("manifest_signature_verified", "verbatim_utf8", "source_freshness_verified"))
                or any(instruction.get(k) is not False for k in
                       ("semantic_minification_applied", "historical_replay", "completion_authority", "scope_expansion_authority"))):
            raise ValueError("invalid public instruction inclusion receipt")
        for key in ("artifact_sha256", "source_sha256", "block_sha256"):
            _digest(instruction[key])
        _identity(instruction["task_cid"])
        if instruction["schema"] == "supervisor-public-instruction-inclusion@5":
            if not translated or result.get("terminal_semantic_scope_dispatch_rechecked") is not True:
                raise ValueError("signed Terminal scope lacks its compiled dispatch replay")
            _terminal_scope_inclusion(instruction.get("terminal_semantic_scope"))
            if instruction["terminal_semantic_scope"]["manifest_cid"] != instruction["manifest_cid"]:
                raise ValueError("signed Terminal scope belongs to another manifest")
        elif "terminal_semantic_scope" in instruction:
            raise ValueError("signed Terminal scope requires its versioned instruction receipt")
        if instruction["schema"] == "supervisor-public-instruction-inclusion@2":
            requirements = instruction.get("intent_requirements")
            if (not isinstance(requirements, dict) or requirements.get("admission_graph_verified") is not True
                    or any(requirements.get(key) is not False for key in ("native_persistence_verified_here",
                        "source_semantics_verified", "semantic_alignment_verified", "proof_authority",
                        "execution_authority", "completion_authority"))):
                raise ValueError("invalid task-specific intent requirement receipt")
            for key in ("requirements_context_cid", "contract_cid", "graph_cid", "coverage_cid", "planning_receipt_cid"):
                _identity(requirements[key])
            _digest(requirements["ledger_sha256"])
            for key in ("requirement_ids", "dependency_requirement_ids", "prohibition_requirement_ids"):
                if (not isinstance(requirements.get(key), list) or len(requirements[key]) > 512
                        or any(not isinstance(item, str) for item in requirements[key])
                        or len(set(requirements[key])) != len(requirements[key])):
                    raise ValueError("invalid intent requirement identity population")
                for requirement_id in requirements[key]:
                    _identity(requirement_id)
    if ("terminal_semantic_scope_dispatch_rechecked" in result
            and (instruction is None or instruction.get("schema") != "supervisor-public-instruction-inclusion@5")):
        raise ValueError("signed Terminal dispatch replay requires its versioned instruction receipt")
    reply = result.get("coding_reply_contract")
    if result.get("repository_source_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_policy import validate_binding as validate_repository_binding
        if item.get("provider") != "codex_cli" or result.get("source_view_advisory_policy") is not None:
            raise ValueError("repository source tools require an exclusive Codex coding route")
        validate_repository_binding(result["repository_source_policy"], workspace=workspace)
    if result.get("source_view_advisory_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import validate_binding as validate_source_binding
        if item.get("provider") != "codex_cli":
            raise ValueError("source view guidance requires the Codex coding route")
        validate_source_binding(result["source_view_advisory_policy"], workspace=workspace)
    if result.get("tool_output_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.tool_output_policy import validate_binding
        if item.get("provider") != "codex_cli":
            raise ValueError("tool output guidance requires the Codex coding route")
        validate_binding(result["tool_output_policy"], workspace=workspace)
    metadata = result.get("semantic_metadata_view")
    working_set = result.get("semantic_working_set")
    preflight = result.get("semantic_working_set_preflight")
    if preflight is not None and working_set is None:
        raise ValueError("working set preflight requires its projection")
    if working_set is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_working_set import validate_working_set_selection_receipt
        validate_working_set_selection_receipt(working_set)
        if (not translated or metadata is not None or result.get("source_view_retrieval", {}).get("schema")
                != "worker-source-dispatch-observation@4"
                or working_set["selected_complete_bytes"] != result["model_prompt_bytes"]
                or working_set["selected_complete_sha256"] != result["model_prompt_sha256"]):
            raise ValueError("working set requires its exact semantic retrieval and provider input")
        binding = working_set["view_receipt"]["binding"]
        source = result["source_view_retrieval"]
        if (binding["descriptor_sha256"] != source.get("descriptor_sha256")
                or binding["descriptor_session_id"] != source.get("session_id")
                or binding["semantic_evidence_mode"] != source.get("semantic_evidence_mode")
                or any(value != source.get(key) for key, value in binding["semantic_binding"].items())):
            raise ValueError("working set differs from its source session")
        if working_set["selected_mode"] != "legacy":
            if (type(preflight) is not dict or set(preflight) != {"schema", "requests", "session_id",
                    "capsule_index", "capsule_sha256", "response_bytes", "response_sha256",
                    "provider_input_bytes", "source_freshness_checked", "proof_authority"}
                    or preflight["schema"] != "supervisor-semantic-working-set-preflight@1"
                    or type(preflight["requests"]) is not int or preflight["requests"] != 1
                    or type(preflight["capsule_index"]) is not int or preflight["capsule_index"] != 0
                    or preflight["session_id"] != source["session_id"]
                    or type(preflight["response_bytes"]) is not int or not 1 <= preflight["response_bytes"] <= 4096
                    or type(preflight["provider_input_bytes"]) is not int or preflight["provider_input_bytes"] != 0
                    or preflight["source_freshness_checked"] is not True or preflight["proof_authority"] is not False):
                raise ValueError("selected working set lacks its bounded endpoint preflight")
            _digest(preflight["capsule_sha256"])
            _digest(preflight["response_sha256"])
        elif preflight is not None:
            raise ValueError("unchanged prompt has an unexpected working set preflight")
    if metadata is not None:
        if (not translated or result["semantic_translation"]["schema"] != "supervisor-semantic-router-encoding@1"
                or type(metadata) is not dict
                or type(metadata.get("schema")) is not str
                or metadata.get("schema") not in {"supervisor-semantic-metadata-selection@1",
                                                   "supervisor-semantic-metadata-selection@2"}
                or type(metadata.get("selected_mode")) is not str
                or metadata.get("selected_mode") not in {"legacy", "common-bindings@1", "common-bindings-instruction@1"}
                or metadata.get("candidate_only") is not True
                or any(metadata.get(key) is not False for key in (
                    "source_freshness_verified", "program_semantics_proved", "omission_authority",
                    "proof_authority", "execution_authority", "completion_authority", "publication_authority"))):
            raise ValueError("metadata coding view requires a verified original semantic transport")
        requested = metadata.get("requested_mode", "common-bindings@1")
        if (type(requested) is not str or requested not in {"common-bindings@1", "common-bindings-instruction@1"}
                or metadata["selected_mode"] not in {"legacy", requested}
                or (requested == "common-bindings-instruction@1" and instruction is None)):
            raise ValueError("metadata selection differs from requested representation")
        for key in ("view_receipt_sha256", "native_complete_sha256", "candidate_complete_sha256", "selected_complete_sha256"):
            _digest(metadata.get(key))
        for key in ("native_complete_bytes", "candidate_complete_bytes", "selected_complete_bytes",
                    "native_complete_proxy_tokens", "candidate_complete_proxy_tokens", "selected_complete_proxy_tokens"):
            if type(metadata.get(key)) is not int or not 0 < metadata[key] <= 262_144:
                raise ValueError("invalid metadata input count")
        if metadata["schema"] == "supervisor-semantic-metadata-selection@2":
            _token_selection_receipt(metadata, item)
            result.update(provider=item["provider"], model=item["model"])
        if (metadata["selected_complete_bytes"] != result["model_prompt_bytes"]
                or metadata["selected_complete_sha256"] != result["model_prompt_sha256"]):
            raise ValueError("metadata selection differs from actual provider input")
    if "representation_displacement" in result:
        if metadata is None:
            raise ValueError("displacement observation requires retained metadata selection")
        for key in ("provider", "model"):
            value = item.get(key)
            if type(value) is not str or not value.strip() or len(value.encode()) > 4096:
                raise ValueError("displacement observation requires recorded provider/model identity")
            result[key] = value
        from ipfs_accelerate_py.agent_supervisor.runtime.representation_displacement import (
            validate_representation_displacement,
        )
        result["representation_displacement"] = validate_representation_displacement(
            result["representation_displacement"], selected_input_sha256=result["model_prompt_sha256"],
            selected_input_bytes=result["model_prompt_bytes"])
    reply_bytes = 0
    if reply is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
        if type(reply) is not dict or reply.get("mode") != "ordinary-completion@1" or item.get("provider") != "codex_cli":
            raise ValueError("invalid coding reply contract receipt")
        _, expected = apply_coding_reply_contract(model_prompt="", mode=reply["mode"],
            purpose="coding", provider="codex_cli")
        dynamic = {"model_prompt_before_sha256", "model_prompt_before_bytes",
                   "model_prompt_after_sha256", "model_prompt_after_bytes", "response_validated"}
        if (set(reply) != set(expected) or type(reply["response_validated"]) is not bool
                or any(type(reply[key]) is not type(expected[key]) or reply[key] != expected[key]
                       for key in expected if key not in dynamic)):
            raise ValueError("invalid coding reply contract receipt")
        for field in ("model_prompt_before_sha256", "model_prompt_after_sha256"):
            _digest(reply[field])
        for field in ("model_prompt_before_bytes", "model_prompt_after_bytes"):
            if type(reply[field]) is not int or not 0 < reply[field] <= 256_000:
                raise ValueError("invalid coding reply contract byte count")
        reply_bytes = reply["instruction_bytes"]
        expected_before = (router_bytes + result["workspace_advisory_bytes"] if metadata is None
                           else metadata["selected_complete_bytes"] - reply_bytes)
        if (reply["model_prompt_before_bytes"] != expected_before
                or reply["model_prompt_after_bytes"] != result["model_prompt_bytes"]
                or reply["model_prompt_after_sha256"] != result["model_prompt_sha256"]):
            raise ValueError("inconsistent coding reply contract input identities")
    expected_complete_bytes = result["model_prompt_bytes"] if metadata is None else metadata["native_complete_bytes"]
    if (result["prompt_sha256"] != result["native_prompt_sha256"]
            or result["prompt_bytes"] != result["native_prompt_bytes"]
            or router_bytes + result["workspace_advisory_bytes"] + reply_bytes != expected_complete_bytes
            or metadata is None and result.get("semantic_metadata_native_coding_reply_contract") is not None
            or reply is None and result.get("semantic_metadata_native_coding_reply_contract") is not None):
        raise ValueError("inconsistent router input identities")
    return result


def _model_projection(*, rendered: str, receipt: dict, repository: Path | None,
                      formalization: dict | None = None, token_counter=None):
    """Reconstruct historical provider bytes without granting freshness."""
    if "world_task_packet" in receipt:
        raise ValueError("world task packet requires qualified Terminal Bench replay")
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    translated = receipt.get("semantic_translation") is not None
    router_prompt, checks = rendered, {}
    recorded = receipt.get("source_formalization_context")
    if formalization is not None or recorded is not None:
        expected_dispatch = dict(formalization) if formalization is not None else None
        if expected_dispatch is not None and recorded is not None:
            expected_dispatch.update(owner_validation_inherited=recorded["owner_validation_inherited"],
                owner_private_state_replayed_here=not recorded["owner_validation_inherited"])
        checks["source_formalization_dispatch"] = (
            expected_dispatch is not None and recorded is not None
            and _same_json(recorded, expected_dispatch))
    if translated:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import replay_semantic_router_prompt_for_audit
        if repository is None:
            raise ValueError("translated receipt requires canonical artifact repository")
        encoded = replay_semantic_router_prompt_for_audit(prompt=rendered, repository=repository,
            transport_schema=receipt["semantic_translation"].get("transport_schema",
                                                               "supervisor-semantic-router-input@1"))
        router_prompt = encoded.provider_prompt
        checks.update({
            "semantic_translation_cid": encoded.table.translation_cid == receipt["semantic_translation"]["translation_cid"],
            "semantic_translation_schema": encoded.receipt["schema"] == receipt["semantic_translation"]["schema"],
        })
        if "semantic_prompt_components" in receipt:
            from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import census_semantic_router_prompt
            census = receipt["semantic_prompt_components"]
            if _semantic_prompt_components(census):
                checks["semantic_prompt_components"] = _same_json(census,
                    census_semantic_router_prompt(encoded=encoded))
    source_input = receipt.get("source_input_view")
    query_reference = receipt.get("query_reference_view")
    if query_reference is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.query_reference_policy import validate_bound_receipt
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_query_reference import project_semantic_query_reference
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import replay_semantic_router_prompt_for_audit
        validate_bound_receipt(receipt.get("query_reference_policy"), query_reference,
            semantic_metadata_view="legacy" if receipt.get("semantic_metadata_view") is None else "common-bindings@1",
            source_input_policy=receipt.get("source_input_policy", "legacy"))
        if repository is None:
            raise ValueError("query reference requires retained canonical source blocks")
        reference_encoded = encoded if translated else replay_semantic_router_prompt_for_audit(
            prompt=rendered, repository=repository, transport_schema=query_reference["source_transport_schema"])
        projected = project_semantic_query_reference(provider_prompt=reference_encoded.provider_prompt)
        checks["query_reference_projection"] = _same_json(projected.receipt, query_reference)
        checks["query_reference_selection"] = receipt.get("query_reference_selected") is translated
        checks["query_reference_dispatch"] = receipt.get("query_reference_dispatched") is translated
        checks["query_reference_provider_attempted"] = receipt.get("query_reference_provider_attempted") is True
        if translated:
            router_prompt = projected.provider_prompt
    if source_input is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.source_input_policy import validate_bound_receipt
        from ipfs_accelerate_py.agent_supervisor.runtime.source_input_view import replay_source_input_view_for_audit
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import replay_semantic_router_prompt_for_audit
        validate_bound_receipt(receipt.get("source_input_policy"), source_input,
            semantic_metadata_view="legacy" if receipt.get("semantic_metadata_view") is None else "common-bindings@1")
        if repository is None:
            raise ValueError("source input requires retained canonical source blocks")
        source_encoded = encoded if translated else replay_semantic_router_prompt_for_audit(
            prompt=rendered, repository=repository, transport_schema=source_input["source_transport_schema"])
        projected = replay_source_input_view_for_audit(encoded=source_encoded, repository=repository, receipt=source_input)
        router_prompt = projected.provider_prompt if translated else rendered + projected.block
        checks["source_input_projection"] = True
        checks["source_input_positive_inclusion"] = True
        checks["source_input_dispatched"] = receipt.get("source_input_dispatched") is True
    source_views = receipt.get("source_view_retrieval")
    source_guidance = None
    if source_views is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_client import BINDING_FIELDS, guidance, canonical_json
        if not translated:
            raise ValueError("source view guidance requires retained semantic input")
        table = encoded.table.to_dict()
        expected = {**{key: table[key] for key in BINDING_FIELDS if key not in {
            "profile", "source_manifest_sha256"}}, "profile": source_views["profile"],
            "source_manifest_sha256": _sha(canonical_json(table["source_manifest"]))}
        # Historical read-only observations retain their original wire shape.
        # The edit guidance is selected only by the explicit current dispatch
        # schema; a mode field cannot upgrade a historical observation.
        edit_options = {}
        if "source_edit_mode" in source_views or source_views.get("schema") == "worker-source-dispatch-observation@3":
            from .terminal_source_edit_policy import DEFAULT_SOURCE_EDIT_MODE, validate_source_edit_mode
            mode = validate_source_edit_mode(source_views.get("source_edit_mode", "disabled"),
                source_view_profile=source_views["profile"])
            if (source_views.get("schema") not in {"worker-source-dispatch-observation@3", "worker-source-dispatch-observation@4"}
                    or source_views.get("schema") == "worker-source-dispatch-observation@3" and mode == DEFAULT_SOURCE_EDIT_MODE):
                raise ValueError("mapped source guidance requires its explicit dispatch schema and mode")
            if mode != DEFAULT_SOURCE_EDIT_MODE:
                edit_options["source_edit_mode"] = mode
        advice = guidance(source_views["profile"], **edit_options)
        source_guidance = advice
        checks["source_view_semantic_binding"] = all(source_views.get(key) == value for key, value in expected.items())
        checks["source_view_guidance"] = (_sha(advice.encode()) == source_views.get("guidance_sha256")
            and len(advice.encode()) == source_views.get("guidance_bytes"))
        router_prompt += advice
    residual = receipt.get("doctor_residual_context")
    if residual is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import load_doctor_residual_advisory
        if (not translated and source_input is None and query_reference is None) or repository is None:
            raise ValueError("Doctor residual requires a canonical semantic repository")
        advisory, observed = load_doctor_residual_advisory(artifact=Path(residual["artifact"]),
            expected_sha256=residual["artifact_sha256"], repository=repository,
            task_cid=residual["task_cid"], prompt=rendered, require_current_source=False)
        router_prompt += advisory
        for field in ("context_cid", "native_capsule_id", "advisory_sha256", "advisory_bytes"):
            checks["doctor_residual_" + field] = observed[field] == residual[field]
    instruction = receipt.get("public_instruction")
    block = observed_instruction = None
    if instruction is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction
        if repository is None:
            raise ValueError("public instruction requires a canonical artifact repository")
        block, observed = load_public_instruction(artifact=Path(instruction["artifact"]),
            expected_sha256=instruction["artifact_sha256"], repository=repository,
            task_cid=instruction["task_cid"], prompt=rendered, require_current_source=False)
        observed_instruction = observed
        router_prompt += block
        for field in ("schema", "context_cid", "manifest_cid", "task_id", "source_path", "source_sha256",
                      "source_bytes", "block_sha256", "block_bytes"):
            checks["public_instruction_" + field] = observed[field] == instruction[field]
        if instruction["schema"] == "supervisor-public-instruction-inclusion@2":
            checks["public_instruction_intent_requirements"] = observed.get("intent_requirements") == instruction["intent_requirements"]
        if instruction["schema"] == "supervisor-public-instruction-inclusion@5":
            checks["public_instruction_terminal_semantic_scope"] = _same_json(
                observed.get("terminal_semantic_scope"), instruction.get("terminal_semantic_scope"))
            checks["terminal_semantic_scope_dispatch_observed"] = receipt.get("terminal_semantic_scope_dispatch_rechecked") is True
    if (translated or instruction is not None or source_input is not None or query_reference is not None) and receipt.get("semantic_working_set") is None:
        checks.update(router_prompt_sha256=_sha(router_prompt.encode()) == receipt["router_prompt_sha256"],
            router_prompt_bytes=len(router_prompt.encode()) == receipt["router_prompt_bytes"])
    model, advisory = render_model_prompt(prompt=router_prompt, purpose="coding",
        workspace=Path(receipt["workspace"]), semantic_transport=translated,
        tool_output_binding=receipt.get("tool_output_policy"),
        source_view_binding=receipt.get("source_view_advisory_policy"),
        repository_source_binding=receipt.get("repository_source_policy"))
    if receipt.get("repository_source_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_policy import helper_sources_match
        checks["repository_source_helper_source"] = helper_sources_match(receipt["repository_source_policy"])
    if receipt.get("source_view_advisory_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime import code_source_view
        source = Path(code_source_view.__file__).read_bytes()
        checks["source_view_helper_source"] = _sha(source) == receipt["source_view_advisory_policy"]["helper_sha256"]
    if receipt.get("tool_output_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime import tool_output_artifact
        source = Path(tool_output_artifact.__file__).read_bytes()
        checks["tool_output_helper_source"] = _sha(source) == receipt["tool_output_policy"]["helper_sha256"]
    reply = receipt.get("coding_reply_contract")
    if reply is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
        model, expected = apply_coding_reply_contract(model_prompt=model, mode=reply["mode"],
            purpose="coding", provider="codex_cli")
        baseline_reply = expected
        if receipt.get("semantic_metadata_view") is None and receipt.get("semantic_working_set") is None:
            for field in expected:
                if field != "response_validated":
                    checks["coding_reply_" + field] = reply[field] == expected[field]
    metadata = receipt.get("semantic_metadata_view")
    if metadata is not None:
        displacement_native_model = model
        if not translated:
            raise ValueError("metadata view requires retained semantic input")
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_view import (
            project_semantic_metadata_view, select_semantic_metadata_view,
        )
        selection_options = {}
        selector = select_semantic_metadata_view
        requested = metadata.get("requested_mode", "common-bindings@1")
        if requested == "common-bindings-instruction@1":
            if block is None or observed_instruction is None:
                raise ValueError("instruction metadata view lacks admitted public requirements")
            from ipfs_accelerate_py.agent_supervisor.runtime.semantic_instruction_view import (
                project_instruction_view, select_instruction_view,
            )
            view = project_instruction_view(encoded.provider_prompt,
                instruction_block=block, instruction_receipt=observed_instruction)
            selector = select_instruction_view
            selection_options["instruction_block"] = block
        else:
            view = project_semantic_metadata_view(encoded.provider_prompt)
        if not router_prompt.startswith(encoded.provider_prompt):
            raise ValueError("metadata audit differs from reconstructed native transport")
        candidate_router = view.provider_prompt + router_prompt[len(encoded.provider_prompt):]
        candidate, candidate_advisory = render_model_prompt(prompt=candidate_router, purpose="coding",
            workspace=Path(receipt["workspace"]), semantic_transport=True,
            tool_output_binding=receipt.get("tool_output_policy"),
            source_view_binding=receipt.get("source_view_advisory_policy"),
            repository_source_binding=receipt.get("repository_source_policy"))
        if candidate_advisory != advisory:
            raise ValueError("metadata audit changed workspace advisory")
        if reply is not None:
            candidate, candidate_reply = apply_coding_reply_contract(model_prompt=candidate,
                mode=reply["mode"], purpose="coding", provider="codex_cli")
        if metadata["schema"] == "supervisor-semantic-metadata-selection@2":
            # Reconstruct recorded delivery even when the public audit child
            # lacks the trusted callable. Never substitute the @1 byte proxy
            # for an unavailable @2 counter or claim to have replayed it.
            _token_selection_receipt(metadata, receipt)
            native = model
            chosen = candidate if metadata["selected_mode"] != "legacy" else native
            checks["semantic_metadata_recorded_prompt_identities"] = (
                metadata["view_receipt_sha256"] == _sha(view.receipt_json.encode())
                and all(metadata[arm + "_complete_sha256"] == _sha(text.encode())
                        and metadata[arm + "_complete_bytes"] == len(text.encode())
                        for arm, text in (("native", native), ("candidate", candidate), ("selected", chosen))))
            checks["semantic_metadata_token_counter_verified"] = False
            checks["semantic_metadata_selection_receipt"] = False
            if token_counter is not None:
                from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_view import (
                    SemanticMetadataTokenCounter,
                )
                caller_matches = (type(token_counter) is SemanticMetadataTokenCounter
                    and token_counter.provider == receipt["provider"]
                    and token_counter.model == receipt["model"])
                if caller_matches:
                    selection = selector(view=view, native_complete_prompt=native,
                        candidate_complete_prompt=candidate, token_counter=token_counter, **selection_options)
                    matched = _same_json(metadata, selection.receipt)
                    checks["semantic_metadata_selection_receipt"] = matched
                    checks["semantic_metadata_token_counter_verified"] = (
                        matched and metadata["token_counter_status"] == "counted")
            model = chosen
        else:
            selection = selector(view=view, native_complete_prompt=model,
                candidate_complete_prompt=candidate, **selection_options)
            checks["semantic_metadata_selection_receipt"] = _same_json(metadata, selection.receipt)
            model = selection.selected_prompt
        if "representation_displacement" in receipt:
            from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_view import (
                SemanticMetadataSelection, _json as metadata_json,
            )
            from ipfs_accelerate_py.agent_supervisor.runtime.representation_displacement import (
                observe_representation_displacement,
            )
            # This replays recorded representation identities, not a trusted
            # counter callback or provider delivery. Existing @2 verification
            # remains unavailable unless that caller supplies its real counter.
            recorded_selection = SemanticMetadataSelection(model, metadata["selected_mode"], metadata_json(metadata))
            observed_displacement = observe_representation_displacement(
                view=view, selection=recorded_selection, native_complete_prompt=displacement_native_model,
                candidate_complete_prompt=candidate, provider=receipt["provider"], model=receipt["model"],
                instruction_block=block if selection_options else None,
                # Historical loading changes freshness/replay flags. Bind the
                # original retained inclusion receipt after checking its exact
                # historical contents above; the rebuilt view uses observed.
                instruction_receipt=instruction if selection_options else None,
                source_guidance=source_guidance)
            checks["representation_displacement_receipt"] = _same_json(
                receipt["representation_displacement"], observed_displacement)
        if reply is not None:
            checks["semantic_metadata_native_reply_contract"] = (
                _same_json(receipt.get("semantic_metadata_native_coding_reply_contract"), baseline_reply))
            selected_reply = candidate_reply if model == candidate else baseline_reply
            for field in selected_reply:
                if field != "response_validated":
                    checks["metadata_selected_coding_reply_" + field] = reply[field] == selected_reply[field]
    working_set = receipt.get("semantic_working_set")
    if working_set is not None:
        if not translated or metadata is not None or source_views is None:
            raise ValueError("working set audit requires its original semantic source route")
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_working_set import (
            replay_semantic_working_set_for_audit, select_working_set,
        )
        view = replay_semantic_working_set_for_audit(encoded, working_set["view_receipt"])
        if working_set["selected_mode"] != "legacy":
            reference = json.loads(view.provider_prompt)["working_set"]["capsule_references"][0]
            checks["semantic_working_set_preflight_capsule"] = all(
                receipt["semantic_working_set_preflight"][key] == reference[key]
                for key in ("capsule_index", "capsule_sha256"))
        if not router_prompt.startswith(encoded.provider_prompt):
            raise ValueError("working set audit differs from reconstructed transport")
        candidate_router = view.provider_prompt + router_prompt[len(encoded.provider_prompt):]
        candidate, candidate_advisory = render_model_prompt(prompt=candidate_router, purpose="coding",
            workspace=Path(receipt["workspace"]), semantic_transport=True)
        if candidate_advisory != advisory:
            raise ValueError("working set audit changed workspace advisory")
        if reply is not None:
            candidate, candidate_reply = apply_coding_reply_contract(model_prompt=candidate,
                mode=reply["mode"], purpose="coding", provider="codex_cli")
        selection = select_working_set(view, native_complete_prompt=model, candidate_complete_prompt=candidate)
        checks["semantic_working_set_selection_receipt"] = _same_json(working_set, selection.receipt)
        model = selection.provider_prompt
        if selection.selected_mode != "legacy":
            router_prompt = candidate_router
        checks.update(router_prompt_sha256=_sha(router_prompt.encode()) == receipt["router_prompt_sha256"],
            router_prompt_bytes=len(router_prompt.encode()) == receipt["router_prompt_bytes"])
        if reply is not None:
            selected_reply = candidate_reply if selection.selected_mode != "legacy" else baseline_reply
            for field in selected_reply:
                if field != "response_validated":
                    checks["working_set_coding_reply_" + field] = reply[field] == selected_reply[field]
    checks.update(model_prompt_sha256=_sha(model.encode()) == receipt["model_prompt_sha256"],
        model_prompt_bytes=len(model.encode()) == receipt["model_prompt_bytes"],
        workspace_advisory_sha256=_sha(advisory.encode()) == receipt["workspace_advisory_sha256"],
        workspace_advisory_bytes=len(advisory.encode()) == receipt["workspace_advisory_bytes"])
    return model, advisory, checks


def _historical_formalization(*, rendered: str, expected: dict, prepared: dict):
    """Join selected receipt bytes, required prompt evidence and signed inputs.

    The independent preparation result establishes whether advice was selected.
    A prompt cannot nominate itself as the expected population. A rebound
    bundle may retain the same task/receipts under another artifact identity;
    both exact bundle digests are checked, without changing world-state checks.
    No source file, index owner or neural inference is replayed here.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import (
        _observation, _parts, _relative,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import (
        _worker_evidence, read_task_context_historical_selection,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import canonical_context_json_bytes

    keys = ("source384_context", "numerical_formalization", "numerical_selection_384")
    nominated = {key: expected[key] for key in keys if key in expected}
    parsed = _parts(rendered)
    binding = expected.get("context_bundle")
    if binding is None:
        if nominated or parsed is not None:
            raise ValueError("formalization requires an independently selected task bundle")
        return None, None
    if type(binding) is not dict:
        raise ValueError("invalid expected context bundle")
    root = Path(prepared["repository"])
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("canonical historical formalization repository required")
    task_id, task_cid = prepared["spec"]["task_key"], expected["task_cid"]

    def selected(reference):
        value = read_task_context_historical_selection(repository=root,
            artifact=reference["artifact"], expected_sha256=_digest(reference["sha256"]),
            task_cid=task_cid, task_id=task_id)
        return {key: value[key] for key in keys if key in value}

    original = selected(binding)
    if not _same_json(original, nominated):
        raise ValueError("retained formalization selection differs from preparation")
    if not original:
        if parsed is not None:
            raise ValueError("native prompt contains unselected formalization evidence")
        return None, None
    if parsed is None:
        raise ValueError("native prompt omitted selected formalization evidence")
    payload, text, chunks = parsed
    dispatched = payload["context_bundle"]
    if not _same_json(selected(dispatched), original):
        raise ValueError("dispatched formalization bundle changed selected task receipts")
    evidence = _worker_evidence(original, repository=root, artifact=dispatched["artifact"],
        expected_sha256=dispatched["sha256"], task_cid=task_cid, task_id=task_id)
    if canonical_context_json_bytes(evidence).decode() != text:
        raise ValueError("native formalization differs from retained selected receipts")
    sources = {}
    signed_sources = prepared["manifest"]["payload"]["sources"]
    for receipt in original.values():
        population = receipt.get("source_hashes")
        if type(population) is not dict or len(population) > 256:
            raise ValueError("invalid historical formalization source population")
        for name, digest in population.items():
            _relative(name)
            if (_digest(digest) != signed_sources[name]["sha256"]
                    or name in sources and sources[name] != digest):
                raise ValueError("formalization source differs from signed captured input")
            sources[name] = digest
    if len(sources) > 768:
        raise ValueError("historical formalization source population exceeds bound")
    dispatch = _observation(evidence, text, chunks, sources)
    observed = dict(schema="terminal-historical-source-formalization-observation@1",
        task_id=task_id, task_cid=task_cid,
        expected_context_bundle_sha256=binding["sha256"],
        dispatched_context_bundle_sha256=dispatched["sha256"],
        worker_evidence_sha256=_sha(text.encode()), chunks=chunks, source_files=len(sources),
        contexts=[dict(kind=row["kind"], receipt_sha256=row["receipt_reference"]["receipt_sha256"],
            summary_sha256=row["summary_sha256"], source_hashes_sha256=row["source_hashes_sha256"])
            for row in evidence["contexts"]],
        historical_receipt_bindings_verified=True, current_sources_checked=False,
        neural_inference_replayed=False, proof_authority=False, formalization_authority=False,
        execution_authority=False, completion_authority=False, source_semantics_verified=False,
        whole_program_semantics_verified=False, source_obligations_discharged=0)
    return observed, dispatch


def _capsule(raw: bytes, *, expected: dict, prepared: dict) -> tuple[dict, str, dict | None]:
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
        ContextCapsule, canonical_context_json_bytes,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule

    capsule = ContextCapsule.from_dict(_json(raw))
    rendered = render_context_capsule(capsule)
    formalization, dispatch = _historical_formalization(rendered=rendered, expected=expected, prepared=prepared)
    wire = _json(rendered.encode())
    contexts, chunks = {}, {}
    for kind in KINDS:
        refs = sorted((row for row in wire["evidence"] if row["kind"] == kind),
                      key=lambda row: row["reference_id"])
        if not refs or not all(row["metadata"].get("required") is True for row in refs):
            raise ValueError("required context missing")
        text = "".join(row["summary"] for row in refs)
        artifact = "sha256:" + _sha(canonical_context_json_bytes({"text": text}))
        for index, row in enumerate(refs):
            if (row["metadata"].get("chunk_index") != index
                    or row["metadata"].get("chunk_count") != len(refs)
                    or row["metadata"].get("artifact_content_id") != artifact
                    or row["referenced_content_id"] != "sha256:" + _sha(row["summary"].encode())):
                raise ValueError("context chunk identity mismatch")
        contexts[kind] = _json(text.encode())
        chunks[kind] = {"count": len(refs), "bytes": len(text.encode()), "sha256": _sha(text.encode())}
    semantic, retrieval, world = (contexts[kind] for kind in KINDS)
    if semantic.get("schema") == "supervisor-semantic-worker-context@2":
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import _validate_program_payload
        _validate_program_payload(semantic)
    query = prepared["query"]
    if not isinstance(query, str) or len(query.encode()) > 32768:
        raise ValueError("invalid public query")
    sources = prepared["manifest"]["payload"]["sources"]
    source_hashes = retrieval["source_sha256"]
    from .terminal_task_profile import PROFILE
    if (PROFILE in sources) != (type(prepared.get("task_profile")) is dict):
        raise ValueError("historical signed public profile cannot be omitted or invented")
    expected_paths = {"bottle.py"}
    semantic_scope_paths = set(prepared["worker_inputs"])
    scoped_receipt = None
    if prepared.get("task_profile") is not None:
        from .terminal_task_profile import (PROFILE, validate_task_profile,
            task_profile_bytes, task_profile_index_paths)
        profile = validate_task_profile(prepared["task_profile"], instruction=query)
        if sources.get(PROFILE, {}).get("sha256") != _sha(task_profile_bytes(profile)):
            raise ValueError("retrieval scope profile differs from signed source")
        expected_paths = set(task_profile_index_paths(profile))
        from .terminal_task_profile import SCOPED_SCHEMA
        if profile["schema"] == SCOPED_SCHEMA:
            from ipfs_accelerate_py.agent_supervisor.runtime.terminal_semantic_scope import terminal_semantic_scope_observation
            captured_scope = expected.get("terminal_semantic_scope")
            if type(captured_scope) is not dict:
                raise ValueError("historical selected context lacks its signed scope observation")
            scoped_receipt = terminal_semantic_scope_observation(profile, prepared["manifest"],
                selected_sources=captured_scope.get("selected_sources"))
            if captured_scope != scoped_receipt:
                raise ValueError("historical semantic selection differs from its signed declaration")
            semantic_scope_paths = set(scoped_receipt["selected_paths"])
            if (semantic.get("required_raw_paths") != scoped_receipt["required_raw_paths"]
                    or semantic.get("program_paths") != scoped_receipt["program_paths"]
                    or any(semantic["manifest"].get(name) != {
                        "sha256": row["sha256"], "source_cid": row["source_cid"]}
                        for name, row in scoped_receipt["selected_sources"].items())
                    or any(len(text.encode("utf-8")) != scoped_receipt["selected_sources"][name]["source_bytes"]
                        for name, text in semantic["raw_sources"].items())):
                raise ValueError("historical semantic payload differs from selected source roles or bytes")
    if scoped_receipt is None and expected.get("terminal_semantic_scope") is not None:
        raise ValueError("historical selected receipt requires its signed profile")
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_paths:
        raise ValueError("unexpected retrieval source scope")
    empty = retrieval.get("retrieval_schema") == "supervisor-empty-code-retrieval@1"
    if empty:
        from ipfs_accelerate_py.agent_supervisor.runtime.empty_code_retrieval import validate_empty_worker_context
        from .terminal_task_profile import INSTRUCTION, PROFILE, SMOKE
        validate_empty_worker_context(retrieval)
        support = {name: {"role": role, "sha256": sources[name]["sha256"]} for name, role in (
            (INSTRUCTION, "instruction"), (PROFILE, "task_profile"), (SMOKE, "structural_smoke"))}
        if (prepared.get("task_profile") is None
                or retrieval["program_paths"] != sorted(expected_paths)
                or retrieval["original_support_hashes"] != support
                or semantic.get("schema") != "supervisor-semantic-worker-context@2"
                or semantic.get("program_paths") != sorted(expected_paths)
                or set(sources) != expected_paths | set(support)
                or any(retrieval[key] != expected.get(key) for key in (
                    "retrieval_schema", "source_population_cid", "disposition", "embedding_calls"))):
            raise ValueError("historical empty population differs from signed task source roles")
    elif retrieval["index_id"] is None or expected.get("retrieval_schema") == "supervisor-empty-code-retrieval@1":
        raise ValueError("historical retrieval lane differs from captured source population")
    tasks = [row for row in world.get("tasks", []) if row.get("task_cid") == expected["task_cid"]]
    checks = {
        "task_alias": semantic["task_id"] == retrieval["task_id"] == wire["objective_id"] == prepared["spec"]["task_key"],
        "semantic_root": semantic["semantic_root_cid"] == expected["semantic_root_cid"] == world["semantic_root_cid"],
        "vector_root": retrieval["index_id"] == expected["index_id"],
        "world_root": world["world_snapshot_cid"] == expected["world_snapshot_cid"],
        "public_query": retrieval["query_text"] == query,
        "source_hashes": all(_digest(digest) == sources[path]["sha256"] for path, digest in source_hashes.items()),
        "semantic_source_hashes": all(semantic["manifest"][path]["sha256"] == digest for path, digest in source_hashes.items()),
        "semantic_input_scope": set(semantic["manifest"]) == semantic_scope_paths,
        "semantic_captured_hashes": all(binding["sha256"] == sources[path]["sha256"]
            for path, binding in semantic["manifest"].items()),
        "semantic_program_scope": (semantic.get("program_paths") == sorted(expected_paths)
            if semantic.get("schema") == "supervisor-semantic-worker-context@2" else "program_paths" not in semantic),
        "current_retrieval_at_capture": retrieval["status"] == "current",
        "nomination_only": retrieval.get("nomination_only") is True,
        "no_retrieval_authority": all(retrieval.get(k) is False for k in ("semantic_authority", "execution_authority", "completion_authority")),
        "no_world_authority": all(world.get(k) is False for k in ("execution_authority", "completion_authority")),
        "no_semantic_completion": semantic.get("completion_authority") is False,
        "canonical_task_contract": len(tasks) == 1 and bool(tasks[0].get("body", {}).get("local_planning_contract_cid")),
    }
    result = {"capsule_id": _identity(capsule.capsule_id), "persisted_capsule_sha256": _sha(raw),
        "native_prompt_sha256": _sha(rendered.encode()), "native_prompt_bytes": len(rendered.encode()),
        "context_chunks": chunks, "checks": checks, "all_context_checks_passed": all(checks.values()),
        "semantic_root_cid": _identity(semantic["semantic_root_cid"]),
        "index_id": None if empty else _identity(retrieval["index_id"]), "retrieval_result_id": _identity(retrieval["result_id"]),
        "retrieval_query_id": _identity(retrieval["query_id"]),
        "world_snapshot_cid": _identity(world["world_snapshot_cid"]),
        "plan_projection_cid": _identity(world["plan_projection_cid"]),
        "task_cid": _identity(expected["task_cid"]), "public_query_sha256": _sha(query.encode()),
        "source_sha256": {key: _digest(value) for key, value in source_hashes.items()},
        "intent_freshness_checked": world.get("intent_freshness_checked") is True}
    if empty:
        result.update(retrieval_schema=retrieval["retrieval_schema"],
            source_population_cid=_identity(retrieval["source_population_cid"]),
            disposition=retrieval["disposition"], embedding_calls=0)
    if len(tasks) == 1:
        result["local_contract_cid"] = _identity(tasks[0].get("body", {}).get("local_planning_contract_cid"))
    if formalization is not None:
        result["source_formalization_context"] = formalization
    if scoped_receipt is not None:
        result["terminal_semantic_scope"] = scoped_receipt
        result["scope_observation_profile_bound"] = True
        result["all_selected_lengths_independently_replayed"] = False
        result["scope_observation_is_historical_only"] = True
    return result, rendered, dispatch


def _decision_auditor():
    """Load the sibling observer from the trusted owner's actual directory."""
    spec = importlib.util.spec_from_file_location("_terminal_owner_decision_input_audit",
        Path(__file__).resolve().with_name("semantic_decision_input_audit.py"))
    if spec is None or spec.loader is None:
        raise ValueError("dialogue input audit owner unavailable")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner.audit_semantic_decision_dialogue

def _typed_nomination_auditor():
    """Load the sibling lineage observer from the trusted owner directory."""
    spec = importlib.util.spec_from_file_location("_terminal_owner_typed_nomination_audit",
        Path(__file__).resolve().with_name("semantic_decision_input_audit.py"))
    if spec is None or spec.loader is None:
        raise ValueError("typed nomination input audit owner unavailable")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner.audit_semantic_typed_nomination_lineage

def _typed_nomination_population(*, report, originals, repository, workspace_root):
    """Replay sidecars outside the bounded IPC; return only computed summaries."""
    verifications, issues = [], []
    if report.get("semantic_decision_typed_nomination_expected") is not True:
        return verifications, issues

    def issue(reason, **ordinals):
        if len(issues) < MAX_RECEIPTS:
            issues.append({"reason": reason, **ordinals})

    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_usage import (
        _dialogue_owner_key, _owner_binding_key, _valid_owner_binding,
    )
    owners = report.get("semantic_decision_owner_bindings", [])
    dialogues = report.get("semantic_decision_dialogues", [])
    if (type(owners) is not list or type(dialogues) is not list
            or len(owners) > MAX_RECEIPTS or len(dialogues) > MAX_RECEIPTS):
        issue("invalid_typed_nomination_owner_or_dialogue_population")
        return verifications, issues
    expected_task = report.get("semantic_decision_expected_task_id")
    origin_pin = report.get("semantic_decision_nomination_sha256")
    seen = []
    auditor = None
    for ordinal, owner in enumerate(owners):
        if owner in seen:
            continue
        seen.append(owner)
        if not _valid_owner_binding(owner) or "typed_nomination" not in owner:
            issue("invalid_or_missing_typed_nomination_binding", owner_binding_ordinal=ordinal)
            continue
        # A copied receipt hash is not an input anchor. The complete literal
        # prompt must have been independently reconstructed by _capsule first.
        captured = [(row, prompt) for row, prompt in originals
                    if row["native_prompt_sha256"] == owner["native_prompt_sha256"]
                    and row["native_prompt_bytes"] == owner["native_prompt_bytes"]]
        prompts = {prompt for _, prompt in captured}
        if len(prompts) != 1 or repository is None:
            issue("typed_nomination_original_native_context_unavailable", owner_binding_ordinal=ordinal)
            continue
        matching = []
        for dialogue in dialogues:
            if (type(dialogue) is dict and dialogue.get("schema") == "supervisor-semantic-decision-dialogue@1"
                    and _dialogue_owner_key(dialogue) == _owner_binding_key(owner)
                    and dialogue not in matching):
                matching.append(dialogue)
        if len(matching) != 1:
            issue("typed_nomination_dialogue_owner_join_unavailable", owner_binding_ordinal=ordinal)
            continue
        try:
            if auditor is None:
                auditor = _typed_nomination_auditor()
            verification = auditor(
                native_prompt=next(iter(prompts)), owner_binding=owner, dialogue=matching[0],
                repository=repository, workspace_root=workspace_root,
                expected_origin_nomination_sha256=origin_pin, expected_task_id=expected_task,
            )
        except Exception as error:
            issue("typed_nomination_historical_verification_unavailable",
                  owner_binding_ordinal=ordinal, error_type=type(error).__name__)
        else:
            verifications.append(verification)
    return verifications, issues

def _bounded_request(value):
    """Bound new dialogue observations before allocating their JSON envelope."""
    pending, cells, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        cells += 1
        if cells > 8192 or depth > MAX_DEPTH:
            raise ValueError("audit request structure exceeds bound")
        if type(item) is str:
            if len(item) > MAX_REQUEST_BYTES:
                raise ValueError("audit request text exceeds bound")
            text_bytes += len(item.encode())
            if text_bytes > MAX_REQUEST_BYTES:
                raise ValueError("audit request text exceeds bound")
        elif type(item) is dict:
            if len(item) > 8192:
                raise ValueError("audit request object exceeds bound")
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("audit request keys must be strings")
                pending.extend(((key, depth + 1), (child, depth + 1)))
        elif type(item) is list:
            if len(item) > 8192:
                raise ValueError("audit request list exceeds bound")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is int:
            if item.bit_length() > 64:
                raise ValueError("audit request integer exceeds bound")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("audit request number must be finite")
        elif item is not None and type(item) not in {bool, float}:
            raise ValueError("unsupported audit request value")

def _decision_population(*, report: dict, receipts: list[dict], originals: list,
        repository: Path | None, workspace_root: Path) -> tuple[dict, list]:
    """Join historical round replay to independently collected coding receipts."""
    value = {"schema": "terminal-semantic-decision-input-population-audit@1",
        "status": "unverified", "dialogue_expected": report.get("semantic_decision_dialogue_expected") is True,
        "dialogues": [], "issues": [], "observed_dialogue_count": 0,
        "all_recorded_dialogue_inputs_verified": False,
        "all_observed_dialogue_inputs_verified": False,
        "all_native_invocations_matched": False,
        "provider_calls": 0, "raw_prompts_exported": False,
        "source_freshness_revalidated": False, "token_savings_established": False,
        "completion_authority": False}
    matches = []
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_usage import semantic_decision_accounting_inputs
        if "semantic_decision_typed_nomination_expected" in report:
            verifications, typed_issues = _typed_nomination_population(
                report=report, originals=originals, repository=repository, workspace_root=workspace_root,
            )
            value["typed_nomination_verifications"] = verifications
            value["issues"].extend(typed_issues)
        # The observer's computed summaries are intentionally not accepted in
        # the caller's request. Feed the fresh result to the same pure coverage
        # join used later by final-report/Harbor accounting.
        audit_view = {"schema": SCHEMA, "provider_calls": 0, "raw_prompts_exported": False,
                      "completion_authority": False, "semantic_decision_inputs": value}
        _, coverage = semantic_decision_accounting_inputs({
            **report, "provider_invocations": receipts, "context_input_audit": audit_view,
        })
        value["receipt_binding_coverage"] = coverage
        value["receipt_binding_coverage_complete"] = (
            coverage["all_attempts_receipted"] and coverage["provider_call_count_complete"]
            and coverage.get("owner_binding_coverage_complete", True) and not coverage["issues"])
        dialogues = report.get("semantic_decision_dialogues", [])
        if type(dialogues) is not list or len(dialogues) > MAX_RECEIPTS:
            raise ValueError("bounded dialogue population required")
        if not dialogues:
            value["issues"].append({"reason": "expected_dialogue_missing" if value["dialogue_expected"]
                                    else "no_dialogue_observation"})
            return value, matches
        audit_dialogue = _decision_auditor()
        captured = {row["native_prompt_sha256"]: (row, prompt) for row, prompt in originals}
        seen = set()
        candidates = []
        for ordinal, dialogue in enumerate(dialogues):
            if type(dialogue) is not dict:
                value["issues"].append({"dialogue_ordinal": ordinal, "reason": "invalid_dialogue_observation"})
                continue
            fingerprint = _sha(json.dumps(dialogue, sort_keys=True, allow_nan=False).encode())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            value["observed_dialogue_count"] += 1
            launch = dialogue.get("worker_launch")
            original = captured.get(launch.get("original_native_prompt_sha256")) if type(launch) is dict else None
            if original is None or repository is None:
                value["issues"].append({"dialogue_ordinal": ordinal, "reason": "original_native_context_unavailable"})
                continue
            observed, prompt = original
            replay = audit_dialogue(native_prompt=prompt, dialogue=dialogue,
                repository=repository, workspace_root=workspace_root)
            value["dialogues"].append(replay)
            for attempt in replay.get("attempts", []):
                invocation_hash = attempt.get("native_invocation_id_sha256")
                if (attempt.get("model_input_verified") is not True
                        or attempt.get("provider_receipt_binding_validated") is not True
                        or attempt.get("provider_dispatch_reported") is not True
                        or not isinstance(invocation_hash, str)):
                    continue
                rows = [row for row in receipts if _sha(row["invocation_id"].encode()) == invocation_hash]
                if len(rows) != 1:
                    continue
                row = rows[0]
                checks = {
                    "historical_round_reconstructed": True,
                    "provider_model": row.get("provider") == dialogue.get("provider") and row.get("model") == dialogue.get("model"),
                    "workspace": row["workspace"] == dialogue.get("workspace"),
                    "prompt_sha256": row["native_prompt_sha256"] == attempt.get("prompt_sha256"),
                    "prompt_bytes": row["native_prompt_bytes"] == attempt.get("prompt_bytes"),
                    "model_prompt_sha256": row["model_prompt_sha256"] == attempt.get("model_prompt_sha256"),
                    "model_prompt_bytes": row["model_prompt_bytes"] == attempt.get("model_prompt_bytes"),
                }
                if all(checks.values()):
                    candidates.append({"capsule_id": observed["capsule_id"],
                        "invocation_id": row["invocation_id"], "workspace_sha256": _sha(row["workspace"].encode()),
                        "native_prompt_sha256": row["native_prompt_sha256"], "native_prompt_bytes": row["native_prompt_bytes"],
                        "original_native_prompt_sha256": observed["native_prompt_sha256"],
                        "model_prompt_sha256": row["model_prompt_sha256"], "model_prompt_bytes": row["model_prompt_bytes"],
                        "model_input_checks": checks, "model_input_verified": True,
                        "historical_dialogue_replay": True})
        for match in candidates:
            if sum(row["invocation_id"] == match["invocation_id"] for row in candidates) == 1:
                matches.append(match)
            else:
                value["issues"].append({"reason": "ambiguous_native_invocation_join"})
        value["all_recorded_dialogue_inputs_verified"] = (
            len(value["dialogues"]) == value["observed_dialogue_count"] and bool(value["dialogues"])
            and all(row.get("all_recorded_round_inputs_verified") is True for row in value["dialogues"]))
        value["all_native_invocations_matched"] = (bool(receipts)
            and {row["invocation_id"] for row in matches} == {row["invocation_id"] for row in receipts})
        value["all_observed_dialogue_inputs_verified"] = (
            value["all_recorded_dialogue_inputs_verified"] and value["all_native_invocations_matched"]
            and value["receipt_binding_coverage_complete"] and not value["issues"])
        value["status"] = ("verified_all_observed_dialogue_inputs" if value["all_observed_dialogue_inputs_verified"]
            else "verified_some_recorded_dialogue_inputs" if any(row.get("verified_round_count", 0) for row in value["dialogues"])
            else "unverified")
    except Exception as error:
        value["issues"].append({"reason": "dialogue_input_audit_unavailable", "error_type": type(error).__name__})
    return value, matches

def _audit_legacy_terminal_context(*, state: Path, receipts: list[dict], workspace_root: Path,
        decision_report: dict | None = None) -> dict:
    """Read historical evidence and compare exact native and model input bytes."""
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt

    result = _base()
    if decision_report is not None:
        _validate_decision_report(decision_report)
        _bounded_request({"receipts": receipts, "decision_report": decision_report})
        result["semantic_decision_inputs"] = {
            "schema": "terminal-semantic-decision-input-population-audit@1",
            "status": "unavailable", "reason": "native_capsule_unavailable",
            "dialogue_expected": decision_report.get("semantic_decision_dialogue_expected") is True,
            "all_observed_dialogue_inputs_verified": False, "provider_calls": 0,
            "raw_prompts_exported": False, "completion_authority": False}
    if len(receipts) > MAX_RECEIPTS:
        raise ValueError("receipt inventory exceeds audit bound")
    parsed, seen = [], set()
    result["coding_receipts_seen"] = len(receipts)
    for item in receipts:
        try:
            value = _receipt(item, workspace_root)
            if value["invocation_id"] in seen:
                raise ValueError("duplicate router invocation")
            seen.add(value["invocation_id"])
            parsed.append(value)
        except (ValueError, KeyError, TypeError):
            result["read_errors"].append({"kind": "receipt", "error_type": "InvalidReceipt"})
    result["usable_coding_receipts"] = len(parsed)
    if not parsed and decision_report is None:
        result["reason"] = "no_usable_coding_receipt"
        return result
    result["invocations"] = [{"invocation_id": row["invocation_id"],
        "native_input_verified": None, "model_input_verified": None,
        "status": "unverified", "reason": "native_capsule_unavailable",
        **({"semantic_prompt_components": row["semantic_prompt_components"],
            "semantic_prompt_components_verified": None}
            if "semantic_prompt_components" in row else {})} for row in parsed]
    root_fd = _open_directory(state)
    originals = []
    census_seen = set()
    try:
        expected_raw = _read_at(root_fd, "context-result.json")
        prepared_raw = _read_at(root_fd, "prepared.json")
        expected, prepared = _json(expected_raw), _json(prepared_raw)
        result.update(context_result_sha256=_sha(expected_raw), prepared_state_sha256=_sha(prepared_raw))
        paths = _capsule_paths(root_fd)
        if not paths:
            result["reason"] = "native_capsule_unavailable"
            return result
        for relative in paths:
            try:
                observed, rendered, formalization = _capsule(_read_at(root_fd, relative), expected=expected, prepared=prepared)
                result["capsules"].append(observed)
                if not observed["all_context_checks_passed"]:
                    continue
                # Keep authenticated native bytes independently of any router
                # receipt. Decision packets need their original parent anchor.
                originals.append((observed, rendered))
                for receipt in parsed:
                    if (observed["native_prompt_sha256"] != receipt["native_prompt_sha256"]
                            or observed["native_prompt_bytes"] != receipt["native_prompt_bytes"]):
                        continue
                    model, advisory, checks = _model_projection(rendered=rendered, receipt=receipt,
                        repository=Path(prepared["repository"]) if prepared.get("repository") else None,
                        formalization=formalization)
                    match = {"capsule_id": observed["capsule_id"],
                        "invocation_id": receipt["invocation_id"],
                        "workspace_sha256": _sha(receipt["workspace"].encode()),
                        "native_prompt_sha256": observed["native_prompt_sha256"],
                        "native_prompt_bytes": observed["native_prompt_bytes"],
                        "model_prompt_sha256": _sha(model.encode()), "model_prompt_bytes": len(model.encode()),
                        "workspace_advisory_sha256": _sha(advisory.encode()),
                        "workspace_advisory_bytes": len(advisory.encode()),
                        "model_input_checks": checks, "model_input_verified": all(checks.values())}
                    if match["model_input_verified"] and receipt.get("source_input_view") is not None:
                        match["source_input_audit_binding_sha256"] = source_input_audit_binding_sha256(receipt)
                    if match["model_input_verified"] and receipt.get("query_reference_view") is not None:
                        match["query_reference_audit_binding_sha256"] = query_reference_audit_binding_sha256(receipt)
                    result["matches"].append(match)
                    # Census only the exact selected message. Retried invocations
                    # are distinct; duplicate stored capsules are not deliveries.
                    if match["model_input_verified"] and receipt["invocation_id"] not in census_seen:
                        census_seen.add(receipt["invocation_id"])
                        _add_rendered_input_census(result, model_prompt=model,
                            workspace_advisory=advisory, invocation_id=receipt["invocation_id"])
            except Exception as error:
                result["read_errors"].append({"kind": "capsule", "error_type": type(error).__name__})
    finally:
        os.close(root_fd)
    if decision_report is not None:
        population, round_matches = _decision_population(report=decision_report, receipts=parsed,
            originals=originals, repository=Path(prepared["repository"]) if prepared.get("repository") else None,
            workspace_root=workspace_root)
        result["semantic_decision_inputs"] = population
        for match in round_matches:
            if not any(row["invocation_id"] == match["invocation_id"] and row["model_input_verified"]
                       for row in result["matches"]):
                result["matches"].append(match)
            if match["invocation_id"] not in census_seen:
                census_seen.add(match["invocation_id"])
                result["rendered_input_census"]["unavailable"].append({
                    "invocation_id_sha256": _sha(match["invocation_id"].encode()),
                    "reason": "dialogue_packet_census_not_supported"})
    for invocation in result["invocations"]:
        matching = [row for row in result["matches"] if row["invocation_id"] == invocation["invocation_id"]]
        if "semantic_prompt_components" in invocation:
            observations = [row["model_input_checks"].get("semantic_prompt_components") for row in matching]
            invocation["semantic_prompt_components_verified"] = (
                True if any(value is True for value in observations)
                else False if any(value is False for value in observations) else None)
        invocation["native_input_verified"] = bool(matching)
        invocation["model_input_verified"] = any(row["model_input_verified"] for row in matching)
        invocation["status"] = ("verified_exact_model_input" if invocation["model_input_verified"]
            else "verified_native_input_only" if matching else "unverified")
        invocation["reason"] = ("exact_context_and_input_digests_match" if invocation["model_input_verified"]
            else "model_projection_mismatch" if matching else "no_exact_stored_capsule_match")
    result["any_native_input_verified"] = any(row["native_input_verified"] for row in result["invocations"])
    result["any_model_input_verified"] = any(row["model_input_verified"] for row in result["invocations"])
    result["all_observed_coding_inputs_verified"] = (bool(parsed) and len(parsed) == len(receipts)
        and all(row["model_input_verified"] for row in result["invocations"])
        and (decision_report is None or result["semantic_decision_inputs"]["all_observed_dialogue_inputs_verified"]))
    result["status"] = ("verified_all_observed_model_inputs" if result["all_observed_coding_inputs_verified"]
        else "verified_some_observed_model_inputs" if result["any_model_input_verified"]
        else "verified_native_input_only" if result["any_native_input_verified"] else "unverified")
    result["reason"] = "per_invocation_context_and_input_comparison"
    return _finish_rendered_input_census(result)


def _validate_decision_report(value):
    if type(value) is not dict or not set(value) <= set(DECISION_REPORT_FIELDS):
        raise ValueError("closed dialogue observation report required")
    _bounded_request(value)


def audit_terminal_context(*, state: Path, receipts: list[dict], workspace_root: Path,
        decision_report: dict | None = None) -> dict:
    """Audit each explicit input route while retaining exact receipt population."""
    if len(receipts) > MAX_RECEIPTS:
        raise ValueError("receipt inventory exceeds audit bound")
    world = [row for row in receipts if isinstance(row, dict) and "world_task_packet" in row]
    if decision_report is not None:
        _validate_decision_report(decision_report)
    if not world:
        return _audit_legacy_terminal_context(state=state, receipts=receipts, workspace_root=workspace_root,
            decision_report=decision_report)
    from .terminal_world_packet_audit import audit_world_terminal_context

    identities = [row.get("invocation_id") for row in receipts if isinstance(row, dict)]
    strings = [value for value in identities if type(value) is str]
    if len(strings) != len(set(strings)):
        result = _base(status="unverified", reason="duplicate_router_invocation")
        result.update(coding_receipts_seen=len(receipts), usable_coding_receipts=0,
            all_observed_coding_inputs_verified=False,
            read_errors=[{"kind": "receipt", "error_type": "DuplicateInvocation"}])
        return result
    legacy = [row for row in receipts if not isinstance(row, dict) or "world_task_packet" not in row]
    world_result = audit_world_terminal_context(state=state, receipts=world, workspace_root=workspace_root)
    if not legacy and decision_report is None:
        return world_result
    # A world marker never becomes a legacy receipt. An expected dialogue with
    # no legacy invocation remains independently unverified.
    legacy_result = _audit_legacy_terminal_context(state=state, receipts=legacy, workspace_root=workspace_root,
        decision_report=decision_report)
    result = _base()
    for key in ("capsules", "invocations", "matches", "read_errors"):
        result[key] = legacy_result[key] + world_result[key]
    result["world_packet_replays"] = world_result["world_packet_replays"]
    if "semantic_decision_inputs" in legacy_result:
        result["semantic_decision_inputs"] = legacy_result["semantic_decision_inputs"]
    for route in (legacy_result, world_result):
        census = route.get("rendered_input_census")
        if census is not None:
            for field in ("observations", "unavailable"):
                result["rendered_input_census"][field].extend(census[field])
    result["coding_receipts_seen"] = len(receipts)
    result["usable_coding_receipts"] = sum(row.get("usable_coding_receipts", 0)
        for row in (legacy_result, world_result))
    for key in ("context_result_sha256", "prepared_state_sha256"):
        values = [row[key] for row in (legacy_result, world_result) if key in row]
        if len(set(values)) > 1:
            result["read_errors"].append({"kind": "state", "error_type": "ChangedBetweenRoutes"})
        elif values:
            result[key] = values[0]
    result["any_native_input_verified"] = any(row.get("native_input_verified") is True
        for row in result["invocations"])
    result["any_model_input_verified"] = any(row.get("model_input_verified") is True
        for row in result["invocations"])
    result["all_observed_coding_inputs_verified"] = (
        result["usable_coding_receipts"] == len(receipts)
        and legacy_result["all_observed_coding_inputs_verified"] is True
        and world_result["all_observed_coding_inputs_verified"] is True
        and not any(row["error_type"] == "ChangedBetweenRoutes" for row in result["read_errors"]))
    result["status"] = ("verified_all_observed_model_inputs" if result["all_observed_coding_inputs_verified"]
        else "verified_some_observed_model_inputs" if result["any_model_input_verified"]
        else "verified_native_input_only" if result["any_native_input_verified"] else "unverified")
    result["reason"] = "per_invocation_context_and_input_comparison"
    return _finish_rendered_input_census(result)


def collect_terminal_context_audit(*, state: Path, receipts: list[dict],
        workspace_root: Path, timeout_seconds: float = MAX_AUDIT_SECONDS,
        decision_report: dict | None = None) -> dict:
    """Run in a bounded child, with the caller's remaining cleanup allowance.

    The default is a ceiling rather than a reserved allocation. Owners with an
    enclosing deadline must pass their remaining time after cleanup reserves.
    Invalid or exhausted budgets never launch a child or affect completion.
    """
    started = time.monotonic()
    if (type(timeout_seconds) not in (int, float)
            or (type(timeout_seconds) is float and not math.isfinite(timeout_seconds))):
        return _base(reason="audit_budget_invalid")
    if timeout_seconds < .1:
        return _base(reason="audit_budget_unavailable")
    try:
        if type(receipts) is not list or len(receipts) > MAX_RECEIPTS:
            return _base(reason="audit_request_exceeds_bound")
        if decision_report is not None and type(decision_report) is not dict:
            raise ValueError("dialogue observation report must be an object")
        selected = ({key: decision_report[key] for key in DECISION_REPORT_FIELDS if key in decision_report}
                    if decision_report is not None else None)
        rows = []
        for row in receipts:
            if not isinstance(row, dict) or row.get("phase") != "coding":
                continue
            if "world_task_packet" in row:
                from .terminal_world_packet_audit import project_world_receipt
                try:
                    rows.append(project_world_receipt(row, workspace_root=workspace_root))
                except (ValueError, KeyError, TypeError):
                    return _base(reason="world_task_packet_replay_unqualified")
                continue
            fields = (*RECEIPT_FIELDS, *TRANSLATION_FIELDS, "provider", "model")
            # The child must verify ordinary completion against the recorded
            # provider. Never infer a missing provider from the selected mode.
            if (row.get("coding_reply_contract") is not None or row.get("tool_output_policy") is not None
                    or row.get("source_view_advisory_policy") is not None or row.get("source_input_view") is not None
                    or row.get("repository_source_policy") is not None
                    or row.get("query_reference_view") is not None):
                fields += ("provider",)
            metadata = row.get("semantic_metadata_view")
            if (type(metadata) is dict
                    and metadata.get("schema") == "supervisor-semantic-metadata-selection@2"):
                # Preserve the caller's exact identities for counter binding.
                # Neither the selection receipt nor the child creates a trusted
                # counter, and missing identities must remain missing.
                fields += ("provider", "model")
            if "representation_displacement" in row:
                fields += ("provider", "model")
            rows.append({key: row[key] for key in fields if key in row})
        envelope = rows if selected is None else {"receipts": rows, "decision_report": selected}
        try:
            _bounded_request(envelope)
        except ValueError:
            return _base(reason="audit_request_exceeds_bound")
        request = json.dumps(envelope, allow_nan=False).encode()
        if len(rows) > MAX_RECEIPTS or len(request) > MAX_REQUEST_BYTES:
            return _base(reason="audit_request_exceeds_bound")
        if not rows and selected is None:
            return _base(reason="no_coding_receipt")
        # The owner may still have the task checkout as cwd. Never prepend
        # that worker-controlled directory to the trusted module search path.
        remaining = min(MAX_AUDIT_SECONDS, timeout_seconds) - (time.monotonic() - started)
        if remaining < .1:
            return _base(reason="audit_budget_unavailable")
        returncode, output = _bounded_audit_child(
            [sys.executable, "-B", "-P", str(Path(__file__).resolve()), "--state", str(state),
             "--workspace-root", str(workspace_root)], input_bytes=request, timeout=remaining,
            limit=MAX_RESULT_BYTES, environment={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if returncode:
            return _base(reason="audit_process_unavailable")
        result = _json(output)
        if not isinstance(result, dict) or result.get("schema") != SCHEMA:
            return _base(reason="audit_result_invalid")
        result["seconds"] = time.monotonic() - started
        return result
    except subprocess.TimeoutExpired:
        return _base(reason="audit_timeout")
    except Exception as error:
        result = _base(reason="audit_unavailable")
        result["error_type"] = type(error).__name__
        return result


def _bounded_audit_child(command, *, input_bytes, timeout, limit, environment):
    """Use the genuine bounded pipe/group primitive from our trusted sibling."""
    path = Path(__file__).resolve().with_name("terminal_context_evidence.py")
    spec = importlib.util.spec_from_file_location("_terminal_owner_evidence_transport", path)
    if spec is None or spec.loader is None:
        raise ValueError("audit bounded owner transport unavailable")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    if Path(owner.__file__).resolve() != path:
        raise ValueError("audit bounded owner transport differs")
    return owner._bounded_child(command, timeout=timeout, limit=limit,
        environment=environment, input_bytes=input_bytes)


def _pin_owner_runtime():
    """Load the actual supervisor package for this trusted standalone child.

    The datasets tree is first for datasets imports, but it also vendors another
    supervisor checkout. Neither that runtime nor task cwd owns this audit.
    Existing foreign imports are refused rather than replaced in a live process.
    """
    package = Path(__file__).resolve().parents[3] / "ipfs_accelerate_py"
    initializer = package / "__init__.py"
    loaded = sys.modules.get("ipfs_accelerate_py")
    if loaded is None:
        spec = importlib.util.spec_from_file_location("ipfs_accelerate_py", initializer,
            submodule_search_locations=[str(package)])
        if spec is None or spec.loader is None:
            raise ValueError("audit owner runtime unavailable")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules["ipfs_accelerate_py"] = loaded
        spec.loader.exec_module(loaded)
    if (Path(getattr(loaded, "__file__", "")).resolve() != initializer
            or [Path(path).resolve() for path in getattr(loaded, "__path__", ())] != [package]):
        raise ValueError("audit owner runtime differs")

    # These are genuine owner package/namespace locations, not a worker cwd or
    # datasets' other benchmarks package. Refuse a previously foreign import.
    source = Path(__file__).resolve().parents[3]
    for name, directory in (("benchmarks", source/"benchmarks"),
            ("benchmarks.agent_supervisor", source/"benchmarks/agent_supervisor"),
            ("benchmarks.agent_supervisor.container_coding", Path(__file__).resolve().parent)):
        initializer = directory/"__init__.py"
        module = sys.modules.get(name)
        if module is None:
            if initializer.is_file():
                spec = importlib.util.spec_from_file_location(name, initializer,
                    submodule_search_locations=[str(directory)])
            else:
                spec = ModuleSpec(name, loader=None, is_package=True)
                spec.submodule_search_locations = [str(directory)]
            if spec is None or (initializer.is_file() and spec.loader is None):
                raise ValueError("audit owner benchmark namespace unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            if spec.loader is not None:
                spec.loader.exec_module(module)
        if ([Path(path).resolve() for path in getattr(module, "__path__", ())] != [directory]
                or initializer.is_file() and Path(getattr(module, "__file__", "")).resolve() != initializer):
            raise ValueError("audit owner benchmark namespace differs")
    if __name__ == "__main__":
        global __package__
        __package__ = "benchmarks.agent_supervisor.container_coding"
        canonical = __package__ + ".terminal_context_audit"
        existing = sys.modules.get(canonical)
        if existing is not None and existing is not sys.modules[__name__]:
            raise ValueError("audit owner module already differs")
        sys.modules[canonical] = sys.modules[__name__]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        _pin_owner_runtime()
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("audit request exceeds bound")
        supplied = _json(raw)
        decision_report = None
        if type(supplied) is dict and set(supplied) == {"receipts", "decision_report"}:
            receipts, decision_report = supplied["receipts"], supplied["decision_report"]
            _validate_decision_report(decision_report)
        else:
            receipts = supplied
        if type(receipts) is not list:
            raise ValueError("audit receipts must be a list")
        _bounded_request(supplied)
        result = audit_terminal_context(state=args.state, receipts=receipts, workspace_root=args.workspace_root,
            decision_report=decision_report)
    except Exception as error:
        result = _base(reason="audit_unavailable")
        result["error_type"] = type(error).__name__
    encoded = json.dumps(result, sort_keys=True)
    if len(encoded.encode()) > MAX_RESULT_BYTES:
        encoded = json.dumps(_base(reason="audit_result_exceeds_bound"))
    print(encoded)


if __name__ == "__main__":
    main()
