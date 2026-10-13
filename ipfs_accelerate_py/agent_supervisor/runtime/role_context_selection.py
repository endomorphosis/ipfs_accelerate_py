"""Explicit host role profiles over the existing native context preparation.

Profiles choose existing query, retrieval and projection controls; they are not
learned rankers or evidence that a role received useful or complete context.
The manifest adds no worker text, database owner, source-omission permission or
provider integration. Loading always replays the native current-source loader.
Stale artifacts refuse: prepare a fresh role artifact rather than implicitly
refreshing a role through the legacy coding-default refresh path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

PROFILE_SCHEMA = "supervisor-role-context-profile@1"
MANIFEST_SCHEMA = "supervisor-role-context-selection@1"
REFERENCE_SCHEMA = "supervisor-role-context-selection-ref@1"
MAX_MANIFEST_BYTES = 131072
MAX_WORKER_BYTES = 2_000_000
_ROLES = frozenset({"coding", "planning", "review"})
_PROFILE_FIELDS = frozenset(("schema role query policy_id policy_revision capsule_limit max_bytes "
    "retrieval_artifact retrieval_sha256").split())
_AUTHORITY_FIELDS = frozenset(("omission_authority proof_authority execution_authority "
    "completion_authority mutation_authority provider_delivery_verified source_freshness_verified "
    "semantic_completeness_claimed").split())


class RoleContextSelectionError(ValueError):
    """A role profile, host artifact or its native binding differs."""


def _need(value, reason):
    if not value:
        raise RoleContextSelectionError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _text(value, maximum=1024):
    _need(type(value) is str and bool(value.strip()), "bounded exact text required")
    try:
        _need(len(value.encode("utf-8")) <= maximum, "text byte bound exceeded")
    except UnicodeError as exc:
        raise RoleContextSelectionError("UTF-8 text required") from exc
    return value


def _digest(value):
    _need(type(value) is str and len(value) == 64
        and all(char in "0123456789abcdef" for char in value), "lowercase SHA256 required")
    return value


def _path(value):
    _text(value)
    path = PurePosixPath(value)
    _need(value != "." and not path.is_absolute() and ".." not in path.parts
        and path.as_posix() == value and "\x00" not in value, "canonical relative artifact path required")
    return value


def _integer(value, low, high):
    _need(type(value) is int and low <= value <= high, "bounded exact integer required")
    return value


def _closed(value, fields):
    _need(type(value) is dict and all(type(key) is str for key in value)
        and set(value) == set(fields), "closed role context object required")


def _encode(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise RoleContextSelectionError("canonical JSON required") from exc


def _decode(raw, maximum, *, max_depth=32):
    _need(type(raw) is bytes and len(raw) <= maximum, "role context byte bound exceeded")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate role context field")
            result[key] = value
        return result
    def constant(_):
        raise RoleContextSelectionError("nonfinite JSON is unsupported")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RoleContextSelectionError("bounded JSON required") from exc
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _need(nodes <= 100000 and (max_depth is None or depth <= max_depth),
            "role context structure bound exceeded")
        if type(item) is dict:
            _need(nodes + len(stack) + len(item) <= 100000, "role context node bound exceeded")
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _need(nodes + len(stack) + len(item) <= 100000, "role context node bound exceeded")
            stack.extend((child, depth + 1) for child in item)
    _need(_encode(value) == raw, "exact canonical JSON bytes required")
    return value


def _located(root, artifact):
    path = root / _path(artifact)
    current = root
    for part in PurePosixPath(artifact).parts:
        current = current / part
        _need(not current.is_symlink(), "symlink role context path refused")
    _need(path.resolve(strict=False) == path and path.is_relative_to(root), "role context path escapes repository")
    return path


def _read(root, artifact, maximum):
    path = _located(root, artifact)
    _need(path.is_file() and path.stat().st_size <= maximum, "bounded regular role context artifact required")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    _need(len(raw) <= maximum, "role context artifact byte bound exceeded")
    return raw


@dataclass(frozen=True, slots=True)
class RoleContextProfile:
    """Immutable owner-declared role and existing native projection controls."""

    role: str
    query: str
    policy_id: str = "role-context@1"
    policy_revision: int = 1
    capsule_limit: int = 8
    max_bytes: int = 32768
    retrieval_artifact: str | None = None
    retrieval_sha256: str | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        _need(type(self) is RoleContextProfile, "exact role context profile required")
        _need(type(self.role) is str and self.role in _ROLES, "explicit supported role required")
        _text(self.query, 8192)
        _text(self.policy_id, 256)
        _integer(self.policy_revision, 1, 2147483647)
        _integer(self.capsule_limit, 0, 32)
        _integer(self.max_bytes, 8192, 65536)
        _need((self.retrieval_artifact is None) == (self.retrieval_sha256 is None), "complete retrieval pair required")
        if self.retrieval_artifact is not None:
            _path(self.retrieval_artifact)
            _digest(self.retrieval_sha256)

    def to_dict(self):
        self._validate()
        return dict(schema=PROFILE_SCHEMA, role=self.role, query=self.query, policy_id=self.policy_id,
            policy_revision=self.policy_revision, capsule_limit=self.capsule_limit, max_bytes=self.max_bytes,
            retrieval_artifact=self.retrieval_artifact, retrieval_sha256=self.retrieval_sha256)

    @classmethod
    def from_dict(cls, value):
        _need(cls is RoleContextProfile, "exact role profile class required")
        _closed(value, _PROFILE_FIELDS)
        _need(type(value["schema"]) is str and value["schema"] == PROFILE_SCHEMA, "unsupported role profile schema")
        return cls(**{key: value[key] for key in _PROFILE_FIELDS if key != "schema"})

    def fingerprint(self):
        return _sha(_encode(self.to_dict()))

    def native_policy(self):
        self._validate()
        policy = dict(worker_query=self.query, worker_capsule_limit=self.capsule_limit, worker_max_bytes=self.max_bytes)
        if self.retrieval_artifact is not None:
            policy["worker_retrieval"] = dict(artifact=self.retrieval_artifact, sha256=self.retrieval_sha256)
        return policy


def _snapshot(profile):
    _need(type(profile) is RoleContextProfile, "exact role context profile required")
    return RoleContextProfile.from_dict(profile.to_dict())


def _native_bindings(payload, profile, task_id):
    _need(type(payload) is dict and payload.get("schema") == "supervisor-semantic-worker-context@2"
        and payload.get("task_id") == task_id and payload.get("completion_authority") is False,
        "explicit native task context required")
    projection = payload.get("worker_projection")
    _need(type(projection) is dict and projection.get("query") == profile.query
        and type(projection.get("max_capsules")) is int and projection["max_capsules"] == profile.capsule_limit
        and type(projection.get("max_bytes")) is int and projection["max_bytes"] == profile.max_bytes,
        "native role projection differs")
    expected = profile.native_policy().get("worker_retrieval")
    _need(_encode(projection.get("retrieval")) == _encode(expected), "native role retrieval differs")
    _need(projection.get("schema") == ("supervisor-semantic-worker-projection@1" if expected is None
        else "supervisor-semantic-worker-projection@2"), "native role projection schema differs")
    return {key: _text(payload["pack"]["pack_cid"] if key == "pack_cid" else payload[key])
        for key in ("scope_cid", "semantic_root_cid", "pack_cid")}


def _selection(root, directory, reference, payload, worker, role):
    from .context_selection_observation import (
        MAX_OBSERVATION_BYTES, ContextSelectionObservationLimit,
        build_context_selection_observation, verify_context_selection_observation,
    )
    from ...mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    unavailable = dict(schema="supervisor-context-selection-observation-unavailable@1",
        status="unavailable", reason="observation_resource_bound", role=role)
    artifact = (directory / "context-selection.json").relative_to(root).as_posix()
    if _encode(reference) == _encode(unavailable):
        path = _located(root, artifact)
        _need(not path.exists(), "unavailable observation has a sidecar")
        try:
            build_context_selection_observation(payload=payload, payload_bytes=worker, role=role)
        except ContextSelectionObservationLimit:
            return
        raise RoleContextSelectionError("unavailable observation does not reproduce its resource bound")
    _closed(reference, ("schema", "artifact", "cid", "sha256", "bytes", "role"))
    _need(reference["schema"] == "supervisor-context-selection-observation-ref@1"
        and reference["artifact"] == "context-selection.json" and reference["role"] == role,
        "role observation reference differs")
    _digest(reference["sha256"])
    _text(reference["cid"])
    _integer(reference["bytes"], 1, MAX_OBSERVATION_BYTES)
    raw = _read(root, artifact, MAX_OBSERVATION_BYTES)
    _need(len(raw) == reference["bytes"] and cid_for_bytes(raw) == reference["cid"], "role observation CID or size differs")
    verify_context_selection_observation(raw=raw, expected_sha256=reference["sha256"],
        payload=payload, payload_bytes=worker, role=role)


def prepare_role_semantic_context(*, profile, repository, paths, required_raw_paths, program_paths,
        objective, task_id, output, max_source_bytes=1_000_000, max_files=64, max_symbols=256,
        context_input_tokens=8192, defer_task_data=False):
    """Prepare a fresh explicit role artifact; never overwrite an earlier view."""
    from .semantic_context_runtime import prepare_semantic_context

    profile = _snapshot(profile)
    _text(objective, MAX_MANIFEST_BYTES)
    _text(task_id)
    _integer(max_source_bytes, 1, 1_000_000)
    _integer(max_files, 1, 64)
    _integer(max_symbols, 1, 1024)
    _integer(context_input_tokens, 1, 128000)
    _need(type(defer_task_data) is bool, "exact deferred-data flag required")
    def captured(values):
        _need(type(values) in (list, tuple) and len(values) <= max_files, "explicit bounded source paths required")
        result = tuple(_path(value) for value in values)
        _need(len(set(result)) == len(result), "duplicate role source paths")
        return tuple(sorted(result))
    paths, required, program = captured(paths), captured(required_raw_paths), captured(program_paths)
    _need(bool(paths) and bool(required) and set(required) <= set(paths) and set(program) <= set(paths),
        "required and explicit program paths must remain in captured scope")
    root = Path(repository).resolve(strict=True)
    _need(root.is_dir(), "canonical repository directory required")
    if profile.retrieval_artifact is not None:
        _located(root, profile.retrieval_artifact)
    destination = Path(output).absolute()
    _need(destination != root and destination.is_relative_to(root), "fresh output must be inside repository")
    directory = _located(root, destination.relative_to(root).as_posix())
    _need(not directory.exists(), "role context output already exists")
    _need(not any((root / name).is_relative_to(directory) for name in paths),
        "role output must be separate from captured sources")
    result = prepare_semantic_context(repository=root, paths=paths, required_raw_paths=required,
        program_paths=program, objective=objective, task_id=task_id, output=directory,
        max_source_bytes=max_source_bytes, max_files=max_files, max_symbols=max_symbols,
        context_input_tokens=context_input_tokens, defer_task_data=defer_task_data,
        observation_role=profile.role, **profile.native_policy())
    worker_artifact = (directory / "worker-context.json").relative_to(root).as_posix()
    worker = _read(root, worker_artifact, profile.max_bytes)
    # Native worker bytes have their own small projection bound. Do not turn
    # the observer's depth limit into a new native acceptance requirement.
    payload = _decode(worker, profile.max_bytes, max_depth=None)
    bindings = _native_bindings(payload, profile, task_id)
    _need(result["worker_payload_sha256"] == _sha(worker) and result["compact_bytes"] == len(worker),
        "native prepared worker identity differs")
    selection = result["context_selection_observation"]
    _selection(root, directory, selection, payload, worker, profile.role)
    manifest = dict(schema=MANIFEST_SCHEMA, profile=profile.to_dict(), profile_sha256=profile.fingerprint(),
        task_id=task_id, worker=dict(artifact=worker_artifact, sha256=_sha(worker), bytes=len(worker)),
        bindings=bindings, context_selection_observation=selection,
        authority={key: False for key in _AUTHORITY_FIELDS}, provider_token_usage=None, token_savings=None)
    raw = _encode(manifest)
    _need(len(raw) <= MAX_MANIFEST_BYTES, "role manifest byte bound exceeded")
    artifact = (directory / "role-context.json").relative_to(root).as_posix()
    with _located(root, artifact).open("xb") as stream:
        stream.write(raw)
    return dict(schema=REFERENCE_SCHEMA, artifact=artifact, sha256=_sha(raw), role=profile.role,
        profile_sha256=profile.fingerprint(), worker_payload_sha256=_sha(worker),
        completion_authority=False, provider_delivery_verified=False)


def load_role_semantic_context(*, repository, artifact, expected_sha256, task_id, profile) -> bytes:
    """Replay native freshness and the expected role; stale views must be rebuilt."""
    from .semantic_context_runtime import load_semantic_worker_context

    profile = _snapshot(profile)
    _text(task_id)
    _digest(expected_sha256)
    root = Path(repository).resolve(strict=True)
    _need(root.is_dir(), "canonical repository directory required")
    if profile.retrieval_artifact is not None:
        _located(root, profile.retrieval_artifact)
    path = _located(root, artifact)
    _need(path.name == "role-context.json", "role manifest filename differs")
    raw = _read(root, artifact, MAX_MANIFEST_BYTES)
    _need(_sha(raw) == expected_sha256, "role manifest digest differs")
    manifest = _decode(raw, MAX_MANIFEST_BYTES)
    _closed(manifest, ("schema", "profile", "profile_sha256", "task_id", "worker", "bindings",
        "context_selection_observation", "authority", "provider_token_usage", "token_savings"))
    _need(manifest["schema"] == MANIFEST_SCHEMA and manifest["task_id"] == task_id
        and _encode(manifest["profile"]) == _encode(profile.to_dict())
        and manifest["profile_sha256"] == profile.fingerprint(), "expected role profile or task differs")
    _closed(manifest["authority"], _AUTHORITY_FIELDS)
    _need(all(value is False for value in manifest["authority"].values())
        and manifest["provider_token_usage"] is None and manifest["token_savings"] is None,
        "role context authority or usage claim refused")
    binding = manifest["worker"]
    _closed(binding, ("artifact", "sha256", "bytes"))
    worker_artifact = (path.parent / "worker-context.json").relative_to(root).as_posix()
    _need(binding["artifact"] == worker_artifact, "role worker must be the manifest sibling")
    _digest(binding["sha256"])
    _integer(binding["bytes"], 1, profile.max_bytes)
    _located(root, worker_artifact)
    worker = load_semantic_worker_context(repository=root, artifact=worker_artifact,
        expected_sha256=binding["sha256"], task_id=task_id, max_payload_bytes=profile.max_bytes).encode("utf-8")
    _need(len(worker) == binding["bytes"], "role worker size differs")
    payload = _decode(worker, profile.max_bytes, max_depth=None)
    _need(_encode(manifest["bindings"]) == _encode(_native_bindings(payload, profile, task_id)),
        "native role bindings differ")
    _selection(root, path.parent, manifest["context_selection_observation"], payload, worker, profile.role)
    return worker
