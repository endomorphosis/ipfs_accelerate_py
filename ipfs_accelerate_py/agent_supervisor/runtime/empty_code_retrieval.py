"""Authenticated absence of retrievable symbols in an explicit source scope.

This artifact is separate from a vector snapshot. Support roles are caller
declarations whose bytes are checked here; their signed authorization belongs
to the task owner. No observation covers undeclared repository files.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from ..analysis.program_ast_adapters import build_program_evidence_index
from ..proof.formal_verification_contracts import content_identity

SCHEMA = "supervisor-empty-code-retrieval@1"
WORKER_SCHEMA = "supervisor-code-retrieval-context@1"
MAX_BYTES = 2_000_000
MAX_WORKER_BYTES = 65_536
SCOPE = "explicit program inputs and declared support; no whole-repository claim"
FALSE = dict(semantic_authority=False, proof_authority=False, execution_authority=False,
    completion_authority=False, omission_authority=False, equivalence_authority=False,
    model_inference_used=False, whole_repository_verified=False)
FIELDS = {"schema", "task_id", "query_text", "program_paths", "source_sha256", "support_hashes",
    "disposition", "native_scan", "index_id", "source_population_cid", "query_id", "result_id",
    "hits", "embedding_calls", "nomination_only", "scope", *FALSE}
WORKER_FIELDS = FIELDS | {"retrieval_schema", "status", "stale_paths",
    "original_source_sha256", "original_support_hashes"}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _name(name):
    _require(type(name) is str and 0 < len(name.encode()) <= 4096,
        "bounded canonical empty-retrieval source path required")
    path = Path(name)
    _require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name
        and name not in {".", ".."} and "\\" not in name,
        "canonical relative empty-retrieval source path required")
    return name


def _paths(paths):
    _require(type(paths) in (list, tuple) and len(paths) <= 64,
        "empty retrieval program scope must contain zero to 64 paths")
    names = [_name(name) for name in paths]
    _require(len(set(names)) == len(names), "duplicate empty-retrieval program path")
    return sorted(names)


def _support(support, program_paths):
    _require(type(support) is dict and len(support) <= 64,
        "bounded explicit support bindings required")
    result = {}
    for name, binding in support.items():
        _name(name)
        _require(type(binding) is dict and set(binding) == {"role", "sha256"}
            and type(binding["role"]) is str
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", binding["role"]) is not None
            and _digest(binding["sha256"]), "closed support role/hash binding required")
        result[name] = dict(binding)
    _require(not set(result).intersection(program_paths),
        "empty-retrieval program and support scopes overlap")
    return {name: result[name] for name in sorted(result)}


def _location(root, name):
    name = _name(name)
    path = root / name
    _require(path.resolve() == path and path.is_relative_to(root) and not path.is_symlink(),
        "empty-retrieval path escapes canonical repository scope")
    return path


def _read(root, name, *, allow_missing=False):
    path = _location(root, name)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES,
            "bounded regular empty-retrieval input required")
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    _require(len(raw) <= MAX_BYTES and identity(before) == identity(after)
        and identity(path.lstat()) == identity(before) and path.resolve() == path,
        "empty-retrieval input changed during capture")
    return raw


def _capture(root, program_paths, support_hashes, *, allow_missing=False, decode_sources=True):
    raw = {name: _read(root, name, allow_missing=allow_missing)
        for name in [*program_paths, *support_hashes]}
    hashes = {name: hashlib.sha256(value).hexdigest() if value is not None else None
        for name, value in raw.items()}
    sources = {name: raw[name].decode("utf-8") for name in program_paths if raw[name] is not None} if decode_sources else {}
    return raw, sources, {name: hashes[name] for name in program_paths}, {
        name: {"role": binding["role"], "sha256": hashes[name]}
        for name, binding in support_hashes.items()}


def _native_scan(sources, paths):
    evidence = build_program_evidence_index(sources)
    ast = evidence.ast_index
    _require(evidence.exhaustive and len(evidence.results) == len(paths)
        and {item.path for item in evidence.results} == set(paths)
        and all(item.status == "success" for item in evidence.results)
        and set(ast.paths) == set(paths),
        "empty retrieval requires complete successful native program coverage")
    symbols = sum(len(item.ast_record.qualified_symbols) for item in ast.path_records)
    return dict(ast_index_id=ast.index_id, path_count=len(paths),
        qualified_symbol_count=symbols, coverage_complete=True)


def _population(program_paths, source_sha256, support_hashes, native_scan, disposition):
    return content_identity(dict(schema="supervisor-empty-code-source-population@1",
        program_paths=program_paths, source_sha256=source_sha256, support_hashes=support_hashes,
        native_scan=native_scan, disposition=disposition))


def observe_empty_program_population(*, repository, program_paths, support_hashes):
    """Return a current scoped absence observation, or None for genuine symbols.

    Unsupported or incomplete scans raise; they cannot be relabeled empty.
    No embedding provider, model asset or vector builder is invoked.
    """
    root = Path(repository).resolve(strict=True)
    paths, support = _paths(program_paths), _support(support_hashes, program_paths)
    captured, sources, hashes, observed_support = _capture(root, paths, support)
    _require(observed_support == support, "empty-retrieval support bytes differ from binding")
    scan = _native_scan(sources, paths)
    _require(_capture(root, paths, support)[0] == captured,
        "empty-retrieval source/support changed during native observation")
    if scan["qualified_symbol_count"]:
        return None
    disposition = "zero_qualified_symbols" if paths else "no_program_inputs"
    return dict(program_paths=paths, source_sha256=hashes, support_hashes=support,
        native_scan=scan, disposition=disposition,
        source_population_cid=_population(paths, hashes, support, scan, disposition))


def _task_query(task_id, query_text):
    _require(type(task_id) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", task_id),
        "empty-retrieval task identity is invalid")
    _require(type(query_text) is str and query_text.strip() and len(query_text.encode()) <= 8192,
        "empty-retrieval query text is outside bounds")


def _query_id(payload):
    return content_identity(dict(schema="supervisor-empty-code-query@1", task_id=payload["task_id"],
        query_text=payload["query_text"], source_population_cid=payload["source_population_cid"]))


def _result_id(payload):
    return content_identity({name: value for name, value in payload.items() if name != "result_id"})


def prepare_empty_code_retrieval_context(*, repository, task_id, query_text,
        program_paths, support_hashes, output):
    root = Path(repository).resolve(strict=True)
    _task_query(task_id, query_text)
    observation = observe_empty_program_population(repository=root,
        program_paths=program_paths, support_hashes=support_hashes)
    _require(observation is not None, "empty retrieval cannot contain qualified symbols")
    output = Path(output).absolute()
    _require(output.is_relative_to(root) and output.resolve() == output and not output.exists()
        and output.relative_to(root).as_posix() not in set(observation["program_paths"]) | set(observation["support_hashes"]),
        "empty-retrieval output must be a new separate repository-contained file")
    payload = dict(schema=SCHEMA, task_id=task_id, query_text=query_text, **observation,
        index_id=None, hits=[], embedding_calls=0, nomination_only=True, scope=SCOPE, **FALSE)
    payload["query_id"] = _query_id(payload)
    payload["result_id"] = _result_id(payload)
    raw = _wire(payload)
    _require(len(raw) <= MAX_BYTES, "empty-retrieval artifact exceeds byte bound")
    context = validate_empty_code_retrieval_artifact(payload, repository=root, task_id=task_id)
    _require(context["status"] == "current", "empty-retrieval source/support changed before persistence")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    return dict(schema=WORKER_SCHEMA, retrieval_schema=SCHEMA,
        sha256=hashlib.sha256(raw).hexdigest(), index_id=None,
        query_id=payload["query_id"], result_id=payload["result_id"],
        source_population_cid=payload["source_population_cid"], disposition=payload["disposition"],
        metadata={"Code retrieval artifact": output.relative_to(root).as_posix(),
            "Code retrieval sha256": hashlib.sha256(raw).hexdigest()},
        embedding_calls=0, execution_authority=False, completion_authority=False)


def _validate_declared_payload(payload, task_id):
    """Validate inert identities only; this does not prove native source absence."""
    _require(type(payload) is dict and set(payload) == FIELDS and len(_wire(payload)) <= MAX_BYTES,
        "closed bounded empty-retrieval artifact required")
    _require(payload["schema"] == SCHEMA and payload["task_id"] == task_id,
        "empty-retrieval task or schema differs")
    _task_query(payload["task_id"], payload["query_text"])
    paths = _paths(payload["program_paths"])
    _require(type(payload["program_paths"]) is list and payload["program_paths"] == paths,
        "empty-retrieval program paths are not canonical")
    support = _support(payload["support_hashes"], paths)
    hashes = payload["source_sha256"]
    _require(type(hashes) is dict and set(hashes) == set(paths)
        and all(_digest(value) for value in hashes.values()), "empty-retrieval source bindings differ")
    disposition = "zero_qualified_symbols" if paths else "no_program_inputs"
    scan = payload["native_scan"]
    _require(payload["disposition"] == disposition and type(scan) is dict
        and set(scan) == {"ast_index_id", "path_count", "qualified_symbol_count", "coverage_complete"}
        and type(scan["ast_index_id"]) is str and scan["ast_index_id"]
        and type(scan["path_count"]) is int and scan["path_count"] == len(paths)
        and type(scan["qualified_symbol_count"]) is int and scan["qualified_symbol_count"] == 0
        and scan["coverage_complete"] is True, "empty-retrieval native scan disposition differs")
    _require(payload["index_id"] is None and type(payload["hits"]) is list and not payload["hits"]
        and type(payload["embedding_calls"]) is int and payload["embedding_calls"] == 0
        and payload["nomination_only"] is True and payload["scope"] == SCOPE
        and all(payload[key] is False for key in FALSE), "empty retrieval cannot invent vectors or authority")
    _require(payload["source_population_cid"] == _population(paths, hashes, support, scan, disposition)
        and payload["query_id"] == _query_id(payload) and payload["result_id"] == _result_id(payload),
        "empty-retrieval population/query/result identity differs")
    return paths, support


def validate_empty_worker_context(context):
    """Check a sealed worker projection's structure/hashes, without live source replay.

    Historical audit must independently bind its population and support roles
    to the original signed declaration. This helper makes no currentness or
    cold native scan claim; it authenticates the declared historical identities.
    """
    _require(type(context) is dict and set(context) == WORKER_FIELDS
        and len(_wire(context)) <= MAX_WORKER_BYTES and context["schema"] == WORKER_SCHEMA
        and context["retrieval_schema"] == SCHEMA, "closed empty-retrieval worker context required")
    payload = {key: deepcopy(context[key]) for key in FIELDS}
    payload.update(schema=SCHEMA, source_sha256=deepcopy(context["original_source_sha256"]),
        support_hashes=deepcopy(context["original_support_hashes"]))
    paths, support = _validate_declared_payload(payload, payload["task_id"])
    observed = context["source_sha256"]
    _require(type(observed) is dict and set(observed) == set(paths)
        and all(value is None or _digest(value) for value in observed.values()),
        "empty-retrieval observed program hashes differ")
    observed_support = context["support_hashes"]
    _require(type(observed_support) is dict and set(observed_support) == set(support),
        "empty-retrieval observed support scope differs")
    for name, binding in observed_support.items():
        _require(type(binding) is dict and set(binding) == {"role", "sha256"}
            and binding["role"] == support[name]["role"]
            and (binding["sha256"] is None or _digest(binding["sha256"])),
            "empty-retrieval observed support role/hash differs")
    stale = sorted([name for name in paths if observed[name] != payload["source_sha256"][name]] + [
        name for name in support if observed_support[name] != support[name]])
    _require(type(context["stale_paths"]) is list and context["stale_paths"] == stale
        and context["status"] == ("stale" if stale else "current"),
        "empty-retrieval worker stale disposition differs")


def validate_empty_code_retrieval_artifact(payload, *, repository, task_id):
    """Authenticate closed history and observe current scoped source/support bytes."""
    _require(type(payload) is dict, "closed bounded empty-retrieval artifact required")
    caller, original_wire = payload, _wire(payload)
    payload = deepcopy(payload)
    paths, support = _validate_declared_payload(payload, task_id)
    hashes, scan = payload["source_sha256"], payload["native_scan"]
    root = Path(repository).resolve(strict=True)
    captured, _, observed_hashes, observed_support = _capture(root, paths, support,
        allow_missing=True, decode_sources=False)
    stale = sorted([name for name in paths if observed_hashes[name] != hashes[name]] + [
        name for name in support if observed_support[name] != support[name]])
    if not stale:
        sources = {name: captured[name].decode("utf-8") for name in paths}
        actual_scan = _native_scan(sources, paths)
        _require(actual_scan == scan, "empty-retrieval native absence does not replay")
    _require(_capture(root, paths, support, allow_missing=True, decode_sources=False)[0] == captured
        and _wire(caller) == original_wire, "empty-retrieval inputs changed during dispatch observation")
    context = dict(payload, schema=WORKER_SCHEMA, retrieval_schema=SCHEMA,
        status="stale" if stale else "current", stale_paths=stale,
        source_sha256=observed_hashes, support_hashes=observed_support,
        original_source_sha256=hashes, original_support_hashes=support)
    validate_empty_worker_context(context)
    return context


def read_empty_code_retrieval_artifact(*, repository, artifact, expected_sha256, task_id):
    root = Path(repository).resolve(strict=True)
    _require(_digest(expected_sha256), "independent empty-retrieval artifact digest required")
    raw = _read(root, artifact)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "empty-retrieval artifact digest differs")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate empty-retrieval artifact key")
            result[key] = value
        return result
    payload = json.loads(raw, object_pairs_hook=unique)
    context = validate_empty_code_retrieval_artifact(payload, repository=root, task_id=task_id)
    _require(_read(root, artifact) == raw, "empty-retrieval artifact changed during replay")
    return payload, context


__all__ = ["SCHEMA", "observe_empty_program_population", "prepare_empty_code_retrieval_context",
    "validate_empty_code_retrieval_artifact", "read_empty_code_retrieval_artifact", "validate_empty_worker_context"]
