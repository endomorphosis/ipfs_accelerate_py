"""Bounded observations of a caller-trusted native producer's final selection.

This module performs no I/O, loading, inference, source authentication or new
admission. It preserves opaque producer identities and never changes its input.
Prepared selection is distinct from actual provider delivery or token usage.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath

SCHEMA = "supervisor-context-selection-observation@1"
MAX_PAYLOAD_BYTES = 2_000_000
MAX_OBSERVATION_BYTES = 262_144
MAX_CAPSULES = 1024
MAX_SOURCES = 64
MAX_DEPTH = 32
MAX_NODES = 100_000
_CAPSULE_SCHEMA = "ipfs-datasets.software-contracts.semantic-capsule@1"
_CAPSULE_FIELDS = set(("schema stable_symbol_id version_cid semantic_index_schema extractor_version "
    "capsule_schema capsule_compiler_version source_slice_path source_cid symbol_fact_cid "
    "signature annotations defaults decorators contracts effects exception_behavior schema_relations "
    "serialization_relations test_refs fixture_refs proof_obligation_refs dependency_stable_ids "
    "dependency_version_cids dependency_fact_cids dependency_link_ids confidence "
    "relevant_binding_projection_cid docstring_hint metadata capsule_cid").split())
_AUTHORITY_FIELDS = ("omission_authority", "proof_authority", "completion_authority",
    "execution_authority", "mutation_authority", "provider_delivery_verified",
    "source_freshness_verified", "semantic_completeness_claimed", "full_dependency_graph_claimed")
_CAVEATS = (
    "Caller supplies an already trusted native producer payload; this observation does not authenticate sources or admissions.",
    "Dependency references are producer declarations; not selected does not mean unavailable, resolved, or semantically complete.",
    "Role is owner declared; preparation does not verify provider delivery or whole-session token use.",
)


class ContextSelectionObservationError(ValueError):
    """The supplied observation or native payload contract differs."""


class ContextSelectionObservationLimit(ContextSelectionObservationError):
    """Observation resource bounds exceeded; callers may retain native success."""


def _need(condition, reason):
    if not condition:
        raise ContextSelectionObservationError(reason)


def _limit(condition, reason):
    if not condition:
        raise ContextSelectionObservationLimit(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _tree(value, limit, *, allow_tuples=False):
    stack, nodes, text_bytes = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _limit(nodes <= MAX_NODES and depth <= MAX_DEPTH, "JSON structure bound exceeded")
        kind = type(item)
        if kind is dict:
            _limit(nodes + len(stack) + 2 * len(item) <= MAX_NODES, "JSON object bound exceeded")
            for key, child in item.items():
                _need(type(key) is str, "JSON object keys must be exact strings")
                stack.extend(((key, depth + 1), (child, depth + 1)))
        elif kind is list or (allow_tuples and kind is tuple):
            _limit(nodes + len(stack) + len(item) <= MAX_NODES, "JSON array bound exceeded")
            stack.extend((child, depth + 1) for child in item)
        elif kind is str:
            _limit(len(item) <= limit, "JSON string bound exceeded")
            try:
                text_bytes += len(item.encode("utf-8"))
            except UnicodeError as exc:
                raise ContextSelectionObservationError("JSON strings must be UTF-8") from exc
            _limit(text_bytes <= limit, "JSON string population bound exceeded")
        elif kind is float:
            _need(math.isfinite(item), "nonfinite JSON numbers are unsupported")
        else:
            _need(kind in (int, bool, type(None)), "exact JSON types required")


def _encode(value, limit, *, allow_tuples=False):
    _tree(value, limit, allow_tuples=allow_tuples)
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ContextSelectionObservationError("canonical bounded JSON required") from exc
    _limit(len(raw) <= limit, "encoded JSON byte bound exceeded")
    return raw


def _decode(raw, limit):
    _need(type(raw) is bytes, "exact JSON bytes required")
    _limit(len(raw) <= limit, "JSON input byte bound exceeded")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def constant(_):
        raise ContextSelectionObservationError("nonfinite JSON numbers are unsupported")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ContextSelectionObservationError):
            raise
        raise ContextSelectionObservationError("valid bounded JSON required") from exc
    _need(_encode(value, limit) == raw, "exact canonical JSON bytes required")
    return value


def _text(value, *, optional=False):
    if optional and value is None:
        return None
    _need(type(value) is str and bool(value), "nonempty producer identity required")
    _limit(len(value.encode()) <= 1024, "producer identity byte bound exceeded")
    return value


def _digest(value):
    _need(type(value) is str and len(value) == 64
        and all(char in "0123456789abcdef" for char in value), "lowercase SHA256 required")
    return value


def _path(value):
    _text(value)
    path = PurePosixPath(value)
    _need(value not in ("", ".") and not path.is_absolute() and ".." not in path.parts
        and path.as_posix() == value and "\x00" not in value, "canonical relative source handle required")
    return value


def _refs(value):
    _need(type(value) is list, "producer reference list required")
    _limit(len(value) <= MAX_CAPSULES, "reference population bound exceeded")
    result = [_text(item) for item in value]
    _need(result == sorted(set(result)), "canonical unique sorted producer references required")
    return result


def _closed(value, fields):
    _need(type(value) is dict and set(value) == set(fields), "closed native producer object required")


def _count(value, upper=MAX_CAPSULES):
    _need(type(value) is int and 0 <= value <= upper, "bounded exact integer required")
    return value


def _projection(payload, capsules, sources):
    value = payload.get("worker_projection")
    count = len(capsules)
    result = dict(projection_schema=None, selector="full_producer_population", query_sha256=None,
        max_capsules=None, max_bytes=None, full_capsule_count=count, selected_capsule_count=count,
        omitted_capsule_count=0, exact_byte_budget_repacked=False, retrieval_sha256=None,
        ranked_symbol_ids=[])
    fetch = set()
    if value is None:
        return result, fetch
    _need(type(value) is dict, "native worker projection required")
    fields = set(("schema selector query max_capsules max_bytes full_capsule_count selected_capsule_count "
        "omitted_capsule_count selected_symbols capsule_index_cid raw_source_fetch_required omission_reason "
        "raw_required_caveat semantic_equivalence_claimed completion_authority").split())
    schema = value.get("schema")
    _need(type(schema) is str and schema in {
        "supervisor-semantic-worker-projection@1", "supervisor-semantic-worker-projection@2"},
        "unsupported native worker projection schema")
    if schema.endswith("@2"):
        fields.update(("retrieval", "ranked_symbol_ids", "unranked_policy"))
    indexed = (schema == "supervisor-semantic-worker-projection@1"
        and value.get("selector") == "duckdb-query-symbol-name@1")
    if indexed:
        # The native indexed selector carries its complete candidate snapshot
        # identity. Observe that identity without treating SQL state or this
        # sidecar as source/admission authority; the producer replays sources.
        _need(payload.get("schema") == "supervisor-semantic-worker-context@2",
            "indexed selection requires an explicit native program scope")
        fields.add("candidate_snapshot_cid")
    if "exact_byte_budget_repacked" in value:
        fields.add("exact_byte_budget_repacked")
    _closed(value, fields)
    expected_selector = ("verified-retrieval-symbol-rank@1" if schema.endswith("@2")
        else "duckdb-query-symbol-name@1" if indexed else "lexical-query-symbol-name@1")
    _need(value["selector"] == expected_selector and value["semantic_equivalence_claimed"] is False
        and value["completion_authority"] is False, "native projection policy or authority differs")
    query = value["query"]
    _need(type(query) is str and 0 < len(query.encode()) <= 8192, "bounded projection query required")
    full = _count(value["full_capsule_count"])
    _need(_count(value["selected_capsule_count"]) == count
        and _count(value["omitted_capsule_count"]) == full - count
        and count <= _count(value["max_capsules"], 32)
        and type(value["max_bytes"]) is int and 8192 <= value["max_bytes"] <= 65536,
        "final projection counts or bounds differ")
    symbols = value["selected_symbols"]
    _need(type(symbols) is list and len(symbols) == count, "ordered selected symbols differ")
    for symbol, capsule in zip(symbols, capsules):
        _closed(symbol, ("path", "qualified_name", "stable_symbol_id"))
        _text(symbol["qualified_name"])
        _need(symbol["path"] == capsule["source_path"]
            and symbol["stable_symbol_id"] == capsule["stable_symbol_id"], "selected symbol identity differs")
    _text(value["capsule_index_cid"])
    fetched = value["raw_source_fetch_required"]
    _need(type(fetched) is dict and set(fetched) <= set(sources), "fetch source handles differ")
    for name, binding in fetched.items():
        _need(binding == sources[name], "fetch source binding differs")
        fetch.add(name)
    repacked = value.get("exact_byte_budget_repacked", False)
    _need(type(repacked) is bool, "exact repacking flag required")
    result.update(projection_schema=schema, selector=expected_selector, query_sha256=_sha(query.encode()),
        max_capsules=value["max_capsules"], max_bytes=value["max_bytes"], full_capsule_count=full,
        selected_capsule_count=count, omitted_capsule_count=full - count, exact_byte_budget_repacked=repacked)
    if indexed:
        result["candidate_snapshot_cid"] = _text(value["candidate_snapshot_cid"])
    if schema.endswith("@2"):
        retrieval = value["retrieval"]
        _closed(retrieval, ("artifact", "sha256"))
        _path(retrieval["artifact"])
        _digest(retrieval["sha256"])
        ranked = value["ranked_symbol_ids"]
        _need(type(ranked) is list and len(ranked) <= 20
            and all(type(item) is str for item in ranked) and len(set(ranked)) == len(ranked),
            "bounded unique ranked identities required")
        _need(value["unranked_policy"] == "lexical-query-symbol-name@1", "unranked policy differs")
        result.update(retrieval_sha256=_sha(_encode(retrieval, MAX_PAYLOAD_BYTES)),
            ranked_symbol_ids=[_text(item) for item in ranked])
    return result, fetch


def _capsule_rows(payload, sources):
    capsules, admissions = payload["capsules"], payload["admissions"]
    _need(type(capsules) is list and type(admissions) is list and len(capsules) == len(admissions),
        "paired ordered native capsules and admissions required")
    _limit(len(capsules) <= MAX_CAPSULES, "capsule population bound exceeded")
    result, seen = [], set()
    for ordinal, (capsule, admission) in enumerate(zip(capsules, admissions)):
        _closed(capsule, _CAPSULE_FIELDS)
        _need(capsule["schema"] == capsule["capsule_schema"] == _CAPSULE_SCHEMA,
            "unsupported native capsule schema")
        path = _path(capsule["source_slice_path"])
        _need(path in sources and capsule["source_cid"] == sources[path]["source_cid"],
            "capsule source differs from captured handles")
        stable = _text(capsule["stable_symbol_id"])
        _need(stable not in seen, "duplicate selected capsule symbol")
        seen.add(stable)
        _closed(admission, ("schema", "interface", "ref", "admission", "freshness", "caveats", "assessment_cid"))
        _need(admission["schema"] == "ipfs-accelerate.semantic-capsule-admission@1"
            and admission["interface"] == "SemanticCapsuleAdmission@1", "unsupported capsule admission schema")
        ref = admission["ref"]
        _closed(ref, ("capsule_cid", "semantic_state_root_cid", "stable_symbol_id", "version_cid",
            "source_cid", "confidence", "validity_bindings", "raw_source_required"))
        for field in ("capsule_cid", "stable_symbol_id", "version_cid", "source_cid", "confidence"):
            _need(ref[field] == capsule[field], "admission and capsule identity differ")
        _need(ref["semantic_state_root_cid"] == payload["semantic_root_cid"], "admission root differs")
        decision, confidence, raw = admission["admission"], capsule["confidence"], ref["raw_source_required"]
        _need(type(confidence) is str and confidence in {"exact", "conservative", "heuristic", "opaque"}
            and type(raw) is bool and type(admission["freshness"]) is str
            and admission["freshness"] in {"fresh", "stale", "unknown"}
            and type(decision) is str and decision in {
                "exact_substitute", "conservative_substitute_with_caveats", "raw_source_required"}
            and raw == (decision == "raw_source_required"), "native admission flags differ")
        _need(decision != "exact_substitute" or confidence == "exact", "exact admission confidence differs")
        _need(decision != "conservative_substitute_with_caveats" or confidence == "conservative",
            "conservative admission confidence differs")
        _refs(admission["caveats"])
        result.append(dict(ordinal=ordinal, capsule_cid=_text(capsule["capsule_cid"]), stable_symbol_id=stable,
            version_cid=_text(capsule["version_cid"]), source_cid=_text(capsule["source_cid"]),
            symbol_fact_cid=_text(capsule["symbol_fact_cid"], optional=True), source_path=path, confidence=confidence,
            admission=dict(sha256=_sha(_encode(admission, MAX_PAYLOAD_BYTES)), decision=decision,
                freshness=admission["freshness"], assessment_cid=_text(admission["assessment_cid"], optional=True),
                raw_source_required=raw, validity_bindings=_refs(ref["validity_bindings"])),
            proof_obligation_refs=_refs(capsule["proof_obligation_refs"]),
            dependency_refs={label: _refs(capsule[field]) for label, field in (
                ("stable_ids", "dependency_stable_ids"), ("version_cids", "dependency_version_cids"),
                ("fact_cids", "dependency_fact_cids"), ("link_ids", "dependency_link_ids"))}))
    for row in result:
        refs = row["dependency_refs"]["stable_ids"]
        row["dependency_selection"] = dict(selected_stable_ids=[item for item in refs if item in seen],
            not_selected_or_unresolved_stable_ids=[item for item in refs if item not in seen])
    return result


def build_context_selection_observation(*, payload: dict, payload_bytes: bytes, role: str = "coding") -> dict:
    """Observe final native selection; the caller owns trust and freshness."""
    _need(type(payload) is dict and type(role) is str and role in {"coding", "planning", "review"},
        "native payload and explicit supported role required")
    _need(type(payload_bytes) is bytes, "exact native payload bytes required")
    _limit(len(payload_bytes) <= MAX_PAYLOAD_BYTES, "native payload byte bound exceeded")
    # Native producer objects can retain tuples before json.dumps turns them
    # into arrays. Accept only that standard conversion for this comparison;
    # the decoded snapshot and all observation bytes remain strict JSON.
    _need(_encode(payload, MAX_PAYLOAD_BYTES, allow_tuples=True) == payload_bytes,
        "native payload bytes differ")
    native = _decode(payload_bytes, MAX_PAYLOAD_BYTES)
    schema = native.get("schema")
    _need(type(schema) is str and schema in {
        "supervisor-semantic-worker-context@1", "supervisor-semantic-worker-context@2"},
        "unsupported native payload schema")
    fields = set(("schema task_id objective scope_cid semantic_root_cid manifest required_raw_paths "
        "preparation_bounds doctor_snapshot_id doctor_manifest_cid reconstruction refresh_lineage "
        "completion_authority capsules admissions raw_sources pack").split())
    if schema.endswith("@2"):
        fields.add("program_paths")
    fields.update(key for key in ("worker_projection", "task_data_projection") if key in native)
    _closed(native, fields)
    _need(native["completion_authority"] is False, "native completion authority differs")
    _need(type(native["objective"]) is str and bool(native["objective"].strip()), "native objective required")
    bounds = native["preparation_bounds"]
    _closed(bounds, ("max_symbols", "context_input_tokens"))
    _need(type(bounds["max_symbols"]) is int and 1 <= bounds["max_symbols"] <= MAX_CAPSULES
        and type(bounds["context_input_tokens"]) is int and 1 <= bounds["context_input_tokens"] <= 128000,
        "native preparation bounds differ")
    reconstruction = native["reconstruction"]
    _need(type(reconstruction) is dict and reconstruction.get("schema") ==
        ("supervisor-scoped-semantic-reconstruction@2" if schema.endswith("@2")
         else "supervisor-scoped-semantic-reconstruction@1"), "unsupported reconstruction schema")
    sources, inline = native["manifest"], native["raw_sources"]
    _need(type(sources) is dict and bool(sources) and type(inline) is dict, "native source objects required")
    _limit(len(sources) <= MAX_SOURCES, "source population bound exceeded")
    _need(set(inline) <= set(sources), "inline sources escape captured handles")
    required = _refs(native["required_raw_paths"])
    _need(bool(required) and set(required) <= set(inline), "required raw source not preserved inline")
    for name, binding in sources.items():
        _path(name)
        _closed(binding, ("sha256", "source_cid"))
        _digest(binding["sha256"])
        _text(binding["source_cid"])
        if name in inline:
            _need(type(inline[name]) is str and _sha(inline[name].encode()) == binding["sha256"],
                "inline raw source hash differs")
    program = _refs(native["program_paths"]) if schema.endswith("@2") else None
    _need(program is None or set(program) <= set(sources), "program handles escape captured scope")
    pack = native["pack"]
    _closed(pack, ("schema", "interface", "pack", "pack_cid", "references", "token_estimate",
        "coverage_satisfied", "production_slice_cid", "budget_exceeded", "decisions"))
    _need(pack["schema"] == "ipfs-accelerate.context-pack-result@1" and pack["interface"] == "ContextPack@1"
        and type(pack["budget_exceeded"]) is bool and type(pack["coverage_satisfied"]) is bool,
        "unsupported native context pack schema or flags")
    capsules = _capsule_rows(native, sources)
    selection, fetch = _projection(native, capsules, sources)
    data = native.get("task_data_projection")
    if data is not None:
        _closed(data, ("schema", "profile_sha256", "immutable_support", "raw_source_fetch_required",
            "fetch_policy", "semantic_equivalence_claimed", "proof_authority", "completion_authority"))
        _need(schema.endswith("@2") and data["schema"] == "supervisor-task-data-projection@1"
            and all(data[key] is False for key in ("semantic_equivalence_claimed", "proof_authority", "completion_authority")),
            "unsupported task data projection or authority")
        _need(type(data["raw_source_fetch_required"]) is dict
            and set(data["raw_source_fetch_required"]) <= set(sources), "task data handles differ")
        for name, binding in data["raw_source_fetch_required"].items():
            _closed(binding, ("sha256", "source_cid", "media_type", "role"))
            _need(all(binding[key] == sources[name][key] for key in ("sha256", "source_cid")), "task data binding differs")
            fetch.add(name)
    raw_paths = {row["source_path"] for row in capsules if row["admission"]["raw_source_required"]}
    result = dict(schema=SCHEMA, observation_stage="native_semantic_context_preparation", role=role,
        role_basis="owner_declared", payload=dict(schema=schema, sha256=_sha(payload_bytes), bytes=len(payload_bytes)),
        task_id=_text(native["task_id"]), objective_sha256=_sha(native["objective"].encode()),
        bindings={key: _text(pack["pack_cid"] if key == "pack_cid" else native[key]) for key in
            ("scope_cid", "semantic_root_cid", "pack_cid", "doctor_snapshot_id", "doctor_manifest_cid")},
        selection=selection, capsules=capsules,
        sources=[dict(path=name, **sources[name], inline_raw=name in inline, required_raw=name in required,
            fetch_required=name in fetch, program_source=None if program is None else name in program,
            selected_capsule_requires_raw=name in raw_paths) for name in sorted(sources)],
        modality_evidence=dict(codebase=dict(status="structural_producer_observed", semantic_root_cid=native["semantic_root_cid"]),
            **{key: dict(status="not_observed_in_this_payload") for key in ("intent", "security", "ui_ux")}),
        scope_caveats=list(_CAVEATS), authority={key: False for key in _AUTHORITY_FIELDS},
        provider_token_usage=None, token_savings=None)
    _encode(result, MAX_OBSERVATION_BYTES)
    return result


def encode_context_selection_observation(observation) -> bytes:
    """Encode bounded metadata; verification additionally replays native input."""
    _need(type(observation) is dict and observation.get("schema") == SCHEMA, "observation schema required")
    return _encode(observation, MAX_OBSERVATION_BYTES)


def verify_context_selection_observation(*, raw: bytes, expected_sha256: str,
        payload: dict, payload_bytes: bytes, role: str = "coding") -> dict:
    """Recompute against caller-trusted native bytes; rehashing cannot bless edits."""
    _digest(expected_sha256)
    observed = _decode(raw, MAX_OBSERVATION_BYTES)
    _need(_sha(raw) == expected_sha256, "observation digest differs")
    expected = build_context_selection_observation(payload=payload, payload_bytes=payload_bytes, role=role)
    _need(raw == encode_context_selection_observation(expected), "observation differs from native selection replay")
    return observed
