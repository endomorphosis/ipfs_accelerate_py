"""Compile a bounded coding read set from one current native source session.

Native query results nominate optional paths, never obligations or authority.
Every delivered file is whole original UTF-8 source. The caller supplies every
required instruction/path; omission of optional evidence proves no relevance,
task completeness, successful delivery or provider-token savings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

from .repository_source_session import RepositorySourceSession
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter, render_model_prompt

SCHEMA = "repository-context-read-set@1"
MAX_TOKEN_COUNT = (1 << 63) - 1
MAX_REQUIRED_PATHS = 8
MAX_REQUIRED_TEXTS = 16
MAX_REQUIRED_TEXT_BYTES = 16384
GUIDANCE = (
    "This coding read set is self-contained; no prior provider context is assumed. "
    "The JSON required_texts are literal caller-supplied obligations and remain in force. "
    "Sources contain whole original UTF-8 files, including comments and formatting. "
    "Lexical path nominations and optional omissions do not prove relevance, source coverage, "
    "semantic equivalence or task completeness. Canonical source remains authoritative. "
    "Obtain exact current source when additional context matters; this packet grants no "
    "proof, execution, editing or completion authority.\n"
)
_SESSION_FIELDS = ("schema", "session_id", "generation", "package_manifest_sha256",
    "previous_package_manifest_sha256", "source_tree_sha256", "datasets_head",
    "semantic_state_cid", "index_coverage", "representation", "proof_authority",
    "execution_authority", "completion_authority")
_QUERY_FIELDS = ("generation", "source_tree_sha256", "semantic_state_cid",
    "semantic_root_cid", "capsule_index_cid", "authority", "semantic_equivalence_claimed",
    "proof_authority", "execution_authority", "completion_authority")
_QUERY_COUNTS = ("population_count", "matching_count", "selected_count", "omitted_count", "truncated")


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("read_set_invalid_json") from error


def _sha(raw):
    return sha256(raw).hexdigest()


def _text(value, maximum, reason, *, nonempty=True):
    _need(type(value) is str, reason)
    try:
        raw = value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(reason) from error
    _need(len(raw) <= maximum and (not nonempty or bool(value.strip())), reason)
    return raw


def _path(value):
    _text(value, 1024, "read_set_invalid_path")
    _need("\\" not in value and all(char.isprintable() for char in value)
        and not PurePosixPath(value).is_absolute()
        and all(part not in {"", ".", "..", ".git"} for part in value.split("/")), "read_set_invalid_path")
    return value


def _integer(value, low, high, reason):
    _need(type(value) is int and low <= value <= high, reason)


def _binding(status, generation):
    _need(type(status) is dict and status.get("source_current") is True
        and status.get("refresh_required") is False and status.get("generation") == generation,
        "read_set_source_not_current")
    _need(all(key in status for key in _SESSION_FIELDS), "read_set_session_binding_missing")
    return {key: status[key] for key in _SESSION_FIELDS}


@dataclass(frozen=True)
class RepositoryReadSet:
    selected_prompt: str = field(repr=False)
    receipt_json: str = field(repr=False)

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def compile_repository_read_set(*, session: RepositorySourceSession, generation: int, query: str,
        required_paths, required_texts, task_id: str, provider: str, model: str, workspace: Path,
        token_counter: PromptTokenCounter, max_input_tokens: int, limit: int = 8,
        max_prompt_bytes: int = MAX_MODEL_PROMPT_BYTES) -> RepositoryReadSet:
    """Greedily admit native-ranked whole optional files after the required core.

    Each eligible complete rendering is counted twice under the same declared
    counter. No fragment subtraction, native schema/framing/history accounting,
    automatic refresh, source execution or provider dispatch is performed.
    """
    _need(type(session) is RepositorySourceSession, "read_set_exact_session_required")
    _integer(generation, 1, 8, "read_set_invalid_generation")
    _integer(limit, 1, 32, "read_set_invalid_limit")
    _integer(max_input_tokens, 1, MAX_TOKEN_COUNT, "read_set_invalid_token_budget")
    _integer(max_prompt_bytes, 1, MAX_MODEL_PROMPT_BYTES, "read_set_invalid_byte_budget")
    for value in (task_id, provider, model):
        _text(value, 256, "read_set_invalid_identity")
        _need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}", value) is not None,
            "read_set_invalid_identity")
    _text(query, 4096, "read_set_invalid_query")
    _need(isinstance(workspace, Path) and workspace.is_absolute() and ".." not in workspace.parts
        and workspace == session.workspace, "read_set_workspace_differs")
    _text(str(workspace), 4096, "read_set_invalid_workspace")
    _need(type(token_counter) is PromptTokenCounter
        and (token_counter.provider, token_counter.model) == (provider, model), "read_set_counter_binding_differs")
    _need(type(required_paths) in (list, tuple) and len(required_paths) <= MAX_REQUIRED_PATHS,
        "read_set_required_paths_bound")
    required = [_path(path) for path in required_paths]
    _need(len(set(required)) == len(required), "read_set_duplicate_required_path")
    _need(type(required_texts) in (list, tuple) and 1 <= len(required_texts) <= MAX_REQUIRED_TEXTS,
        "read_set_required_texts_bound")
    texts = list(required_texts)
    text_bytes = [_text(text, MAX_REQUIRED_TEXT_BYTES, "read_set_required_texts_bound") for text in texts]
    _need(sum(map(len, text_bytes)) <= MAX_REQUIRED_TEXT_BYTES, "read_set_required_texts_bound")
    metadata = (token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis)
    _need(type(metadata[1]) is str and metadata[1] in {"tokenizer", "estimate"}, "read_set_counter_quality_differs")
    for value in [*metadata[0].values(), metadata[2]]:
        _text(value, 4096, "read_set_counter_metadata_invalid")
        _need(value == value.strip() and all(char.isprintable() for char in value),
            "read_set_counter_metadata_invalid")
    status = session.status_summary()
    binding = _binding(status, generation)
    native = session.context_query(generation=generation, query=query, limit=limit)
    _need(type(native) is dict and native.get("schema") == "repository-source-context@1"
        and all(key in native for key in (*_QUERY_FIELDS, *_QUERY_COUNTS))
        and native.get("query") == query
        and all(native[key] == binding[key] for key in ("generation", "source_tree_sha256", "semantic_state_cid")),
        "read_set_native_binding_differs")
    candidates = native.get("candidates")
    _need(type(candidates) is list and len(candidates) <= limit, "read_set_native_population_invalid")
    native_json = _json(native)
    query_binding = {key: native[key] for key in _QUERY_FIELDS}
    _need(_binding(session.status_summary(), generation) == binding, "read_set_session_changed")
    paths, nominations = [], {}
    for candidate in candidates:
        _need(type(candidate) is dict, "read_set_native_candidate_invalid")
        path = _path(candidate.get("path"))
        digest = candidate.get("source_sha256")
        _need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "read_set_native_candidate_invalid")
        _need(path not in nominations or nominations[path] == digest, "read_set_nomination_hash_conflict")
        if path not in nominations:
            paths.append(path)
            nominations[path] = digest
    files, sources, selected, measurements, observations = [], {}, [], [], {}
    evaluations = 0

    def read(path, mandatory):
        row = session.read(generation=generation, path=path, representation="original")
        _need(all(row.get(key) == value for key, value in binding.items() if key != "representation")
            and row.get("representation") == "original" and row.get("path") == path
            and row.get("view_sha256") == row.get("original_sha256")
            and row.get("view_bytes") == row.get("original_bytes")
            and (path not in nominations or row["original_sha256"] == nominations[path]),
            "read_set_source_binding_differs")
        receipt = dict(path=path, original_sha256=row["original_sha256"], original_bytes=row["original_bytes"],
            required=mandatory, selected=False, omission_reason=None)
        files.append(receipt)
        if row.get("encoding") != "utf-8":
            _need(not mandatory, "read_set_required_source_not_utf8")
            receipt["omission_reason"] = "non_utf8_source"
            return None
        raw = _text(row.get("content"), session.limits["max_file_bytes"], "read_set_source_bytes_invalid", nonempty=False)
        _need(_sha(raw) == row["original_sha256"] and len(raw) == row["original_bytes"],
            "read_set_source_bytes_differ")
        sources[path] = dict(path=path, source_sha256=row["original_sha256"], content=row["content"])
        return receipt

    def render(paths):
        body = dict(task_id=task_id, required_texts=texts, sources=[sources[path] for path in paths])
        return render_model_prompt(prompt=GUIDANCE + _json(body), purpose="coding", workspace=workspace)[0]

    def count(prompt):
        nonlocal evaluations
        values = []
        for _ in range(2):
            _need((token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis) == metadata,
                "read_set_counter_changed")
            evaluations += 1
            try:
                value = token_counter.count(prompt)
            except Exception:
                raise ValueError("read_set_counter_failed") from None
            _need((token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis) == metadata,
                "read_set_counter_changed")
            _integer(value, 1, MAX_TOKEN_COUNT, "read_set_counter_magnitude_exceeded")
            values.append(value)
        digest = _sha(prompt.encode())
        _need(values[0] == values[1] and (digest not in observations or observations[digest] == values[0]),
            "read_set_counter_not_repeatable")
        observations[digest] = values[0]
        return values[0]

    for path in required:
        read(path, True)
        selected.append(path)
        files[-1]["selected"] = True
    prompt = render(selected)
    _need(len(prompt.encode()) <= max_prompt_bytes, "read_set_required_byte_budget_exceeded")
    total = count(prompt)
    _need(total <= max_input_tokens, "read_set_required_token_budget_exceeded")
    measurements.append(dict(model_prompt_sha256=_sha(prompt.encode()), model_prompt_bytes=len(prompt.encode()),
        input_count=total, candidate_path=None, selected=True))
    for path in paths:
        if path in required:
            continue
        row = read(path, False)
        if row is None:
            continue
        candidate = render([*selected, path])
        size = len(candidate.encode())
        amount = None if size > max_prompt_bytes else count(candidate)
        accepted = amount is not None and amount <= max_input_tokens
        row.update(selected=accepted, omission_reason=None if accepted else
            "complete_input_byte_budget" if amount is None else "complete_input_token_budget")
        measurements.append(dict(model_prompt_sha256=_sha(candidate.encode()), model_prompt_bytes=size,
            input_count=amount, candidate_path=path, selected=accepted))
        if accepted:
            selected.append(path)
            prompt, total = candidate, amount
    _need(count(prompt) == total, "read_set_counter_not_repeatable")
    closing = session.context_query(generation=generation, query=query, limit=limit)
    _need(_json(closing) == native_json
        and _binding(session.status_summary(), generation) == binding, "read_set_session_changed")
    estimate = metadata[1] == "estimate"
    receipt = dict(schema=SCHEMA, task_id=task_id, provider=provider, model=model,
        workspace_binding_sha256=_sha(str(workspace).encode()), session_binding=binding,
        native_query_binding=query_binding, native_query_receipt_sha256=_sha(native_json.encode()),
        native_query_counts={key: native[key] for key in _QUERY_COUNTS},
        query_sha256=_sha(query.encode()), query_bytes=len(query.encode()), required_paths=required,
        required_text_bindings=[dict(sha256=_sha(raw), bytes=len(raw)) for raw in text_bytes],
        selected_paths=selected, nominated_paths=paths, files=files,
        model_prompt_sha256=_sha(prompt.encode()), model_prompt_bytes=len(prompt.encode()),
        guidance_sha256=_sha(GUIDANCE.encode()), measurements=measurements,
        measurement_status="estimated" if estimate else "measured",
        tokenizer_binding=metadata[0], token_count_kind=metadata[1], tokenizer_compatibility_basis=metadata[2],
        selected_input_tokens=None if estimate else total, selected_estimated_input_tokens=total if estimate else None,
        counter_evaluation_count=evaluations, max_input_tokens=max_input_tokens, max_prompt_bytes=max_prompt_bytes,
        token_budget_basis="declared_local_estimate" if estimate else "declared_local_tokenizer",
        complete_prompt_repeatability_observed=True, source_current_checked=True, session_refreshed=False,
        full_original_sources_only=True, required_obligations_preserved=True,
        count_scope="complete_rendered_user_input_only", native_schema_and_framing_measured=False,
        native_query_relevance_proved=False, source_coverage_complete=False,
        provider_invoked=False, provider_delivery_verified=False, task_acceptance_verified=False,
        provider_token_usage=None, total_session_token_savings=None, token_budget_guaranteed=False,
        proof_authority=False, execution_authority=False, completion_authority=False)
    return RepositoryReadSet(prompt, _json(receipt))
