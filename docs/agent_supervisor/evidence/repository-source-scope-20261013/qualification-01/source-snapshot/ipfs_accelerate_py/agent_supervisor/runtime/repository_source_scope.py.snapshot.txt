"""Explicit bounded source nominations from an existing historical CAS inventory.

Cold replay verifies the entire committed inventory, including unselected blob
bodies. Output pagination does not reduce that setup work. This module creates
no store/session, reads no source checkout, and grants no current-source or task
admission. Callers must independently admit current originals and required facts
before using the returned path arguments with a semantic producer.
The request's task_id remains the historical capture label, not a current
native target task or a task-admission identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_structured,
)
from .repository_source_inventory import (
    SourceInventoryLimits, SourceInventoryRequest, _cid, _input_guard,
    load_source_inventory,
)

SCHEMA = "supervisor-repository-source-scope-nomination@1"
FRONTIER_SCHEMA = "supervisor-repository-source-scope-frontier@1"
PAGE_SCHEMA = "supervisor-repository-source-scope-frontier-page@1"
MAX_FILES = 64
MAX_SOURCE_BYTES = 1_000_000
MAX_PATH_BYTES = 4096
FRONTIER_STATUSES = (
    "not_selected_eligible", "symlink_requires_separate_scope",
    "gitlink_requires_separate_repository", "unsupported_path",
    "unsupported_source_encoding", "source_exceeds_task_byte_bound",
)
_FLAGS = dict(historical_only=True, current_source_verified=False,
    task_admission_verified=False, proof_authority=False, execution_authority=False,
    completion_authority=False, omission_authority=False)
_ROW_FIELDS = {"raw_path_hex", "path", "git_mode", "git_object_oid", "source_cid",
               "source_sha256", "source_bytes"}
_OBSERVATION_FIELDS = {"schema", "request", "source_limits", "source_inventory_cid",
    "native_snapshot_cid", "population_cid", "git_commit", "git_tree", "inventory_entry_count",
    "selected", "selected_count", "selected_bytes", "required_paths", "program_paths",
    "max_files", "max_source_bytes", "frontier_count", "frontier_status_counts",
    "frontier_cid", "scope_cid", *_FLAGS}


class RepositorySourceScopeError(ValueError):
    """A complete explicit historical nomination cannot be returned."""


def _need(condition, message):
    if not condition:
        raise RepositorySourceScopeError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _raw_path(value):
    _need(type(value) is str and value, "exact raw path hex required")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise RepositorySourceScopeError("canonical raw path hex required") from error
    _need(raw.hex() == value, "canonical raw path hex required")
    return raw


def _paths(value, label, *, allow_empty=False):
    _need(type(value) is tuple and (0 if allow_empty else 1) <= len(value) <= MAX_FILES,
          "bounded explicit " + label + " tuple required")
    for item in value:
        _need(type(item) is str and 0 < len(item) <= MAX_PATH_BYTES * 2,
              "selected raw path exceeds task path bound")
        _need(len(_raw_path(item)) <= MAX_PATH_BYTES, "selected raw path exceeds task path bound")
    _need(value == tuple(sorted(set(value))), label + " must be sorted and unique")
    return value


def _path(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        return None
    parts = text.split("/")
    if (not text or len(raw) > MAX_PATH_BYTES or not text.isprintable() or "\\" in text
            or PurePosixPath(text).is_absolute() or str(PurePosixPath(text)) != text
            or any(part in {"", ".", "..", ".git"} for part in parts)):
        return None
    return text


def _metadata(item):
    entry, data = item.entry, item.data
    return dict(raw_path_hex=entry.raw_path_hex, path=_path(_raw_path(entry.raw_path_hex)),
        git_mode=entry.git_mode, git_object_oid=entry.git_object_oid, source_cid=item.source_cid,
        source_sha256=None if data is None else _sha(data),
        source_bytes=None if data is None else len(data))


def _status(row, data):
    if row["git_mode"] == "160000":
        return "gitlink_requires_separate_repository"
    if row["git_mode"] == "120000":
        return "symlink_requires_separate_scope"
    if row["path"] is None:
        return "unsupported_path"
    if type(data) is not bytes or b"\0" in data:
        return "unsupported_source_encoding"
    try:
        data.decode("utf-8")
    except UnicodeError:
        return "unsupported_source_encoding"
    if len(data) > MAX_SOURCE_BYTES:
        return "source_exceeds_task_byte_bound"
    return "not_selected_eligible"


def _parse(raw):
    _need(type(raw) is bytes, "immutable canonical scope metadata required")
    try:
        value = json.loads(raw)
        _need(canonical_dag_json_bytes(value) == raw, "noncanonical scope metadata")
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise RepositorySourceScopeError("invalid canonical scope metadata") from error


def _validate_row(row, *, selected):
    fields = _ROW_FIELDS | ({"role", "required"} if selected else {"status"})
    _need(type(row) is dict and set(row) == fields, "closed source scope row required")
    raw = _raw_path(row["raw_path_hex"])
    _need(row["path"] == _path(raw), "source row path differs from exact raw path")
    _need(type(row["git_mode"]) is str and row["git_mode"] in {"100644", "100755", "120000", "160000"},
          "unsupported committed source mode")
    _need(type(row["git_object_oid"]) is str
          and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", row["git_object_oid"]) is not None,
          "full committed source object identity required")
    if row["git_mode"] == "160000":
        _need(row["source_cid"] is row["source_sha256"] is row["source_bytes"] is None,
              "gitlink cannot claim retained source body")
    else:
        _cid(row["source_cid"], "raw")
        _need(type(row["source_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]) is not None
              and type(row["source_bytes"]) is int and row["source_bytes"] >= 0,
              "source row digest or original byte length differs")
    if selected:
        _need(row["git_mode"] in {"100644", "100755"} and row["path"] is not None
              and row["source_bytes"] <= MAX_SOURCE_BYTES
              and type(row["required"]) is bool and type(row["role"]) is str
              and row["role"] in {"program", "support"}, "invalid selected source row")
    else:
        _need(type(row["status"]) is str and row["status"] in FRONTIER_STATUSES,
              "unsupported source frontier status")
        expected = ("gitlink_requires_separate_repository" if row["git_mode"] == "160000" else
            "symlink_requires_separate_scope" if row["git_mode"] == "120000" else
            "unsupported_path" if row["path"] is None else None)
        # Encoding is not recoverable from body-free metadata. Its status is
        # assigned only after verified replay, never inferred from the suffix.
        if expected is not None:
            _need(row["status"] == expected, "source frontier classification differs")
        else:
            eligible = ("source_exceeds_task_byte_bound" if row["source_bytes"] > MAX_SOURCE_BYTES
                        else "not_selected_eligible")
            _need(row["status"] in {eligible, "unsupported_source_encoding"},
                  "source frontier classification differs")
    return row


def _frontier_payload(observation, rows):
    return dict(schema=FRONTIER_SCHEMA, source_inventory_cid=observation["source_inventory_cid"],
        request=observation["request"], entries=rows, **_FLAGS)


@dataclass(frozen=True)
class RepositorySourceScope:
    """Immutable body-free historical nomination; exports never admit source."""

    _observation_bytes: bytes
    _frontier_rows: tuple[bytes, ...]

    def __post_init__(self):
        self._validated()

    def _validated(self):
        _need(type(self) is RepositorySourceScope and type(self._frontier_rows) is tuple,
              "exact immutable source scope required")
        value = _parse(self._observation_bytes)
        _need(type(value) is dict and set(value) == _OBSERVATION_FIELDS and value["schema"] == SCHEMA,
              "closed source scope nomination required")
        _need(all(type(value[key]) is type(expected) and value[key] == expected for key, expected in _FLAGS.items()),
              "historical scope grants no admission or authority")
        _need(type(value["request"]) is dict and type(value["source_limits"]) is dict,
              "typed historical source identities required")
        try:
            request = SourceInventoryRequest(**value["request"])
            limits = SourceInventoryLimits(**value["source_limits"])
        except TypeError as error:
            raise RepositorySourceScopeError("closed historical request and source limits required") from error
        _input_guard(request, limits)()
        for key in ("source_inventory_cid", "native_snapshot_cid", "population_cid", "frontier_cid", "scope_cid"):
            _cid(value[key])
        _need(value["population_cid"] == request.expected_population_cid
              and value["git_commit"] == request.expected_commit and value["git_tree"] == request.expected_tree,
              "scope differs from independent historical request")
        _need(type(value["max_files"]) is int and value["max_files"] == MAX_FILES
              and type(value["max_source_bytes"]) is int and value["max_source_bytes"] == MAX_SOURCE_BYTES,
              "task source bounds changed")
        selected = value["selected"]
        _need(type(selected) is list and 1 <= len(selected) <= MAX_FILES, "bounded selected source rows required")
        for row in selected:
            _validate_row(row, selected=True)
        names = tuple(row["raw_path_hex"] for row in selected)
        _paths(names, "selected paths")
        required = _paths(tuple(value["required_paths"]) if type(value["required_paths"]) is list else None,
                          "required paths")
        program = _paths(tuple(value["program_paths"]) if type(value["program_paths"]) is list else None,
                         "program paths", allow_empty=True)
        _need(set(required) <= set(names) and set(program) <= set(names)
              and required == tuple(row["raw_path_hex"] for row in selected if row["required"])
              and program == tuple(row["raw_path_hex"] for row in selected if row["role"] == "program"),
              "explicit source roles or required population changed")
        _need(len(self._frontier_rows) <= limits.max_entries,
              "source frontier exceeds inventory entry bound")
        frontier = [_validate_row(_parse(raw), selected=False) for raw in self._frontier_rows]
        frontier_names = tuple(row["raw_path_hex"] for row in frontier)
        _need(frontier_names == tuple(sorted(set(frontier_names))) and not set(names) & set(frontier_names),
              "source population is duplicated or overlaps its frontier")
        for field, expected in (("selected_count", len(selected)),
            ("selected_bytes", sum(row["source_bytes"] for row in selected)),
            ("frontier_count", len(frontier)), ("inventory_entry_count", len(selected)+len(frontier))):
            _need(type(value[field]) is int and value[field] == expected, "source population accounting differs")
        _need(value["selected_bytes"] <= MAX_SOURCE_BYTES and value["inventory_entry_count"] <= limits.max_entries,
              "source scope exceeds its original-byte or inventory bound")
        statuses = {status: sum(row["status"] == status for row in frontier) for status in FRONTIER_STATUSES}
        _need(type(value["frontier_status_counts"]) is dict
              and all(type(count) is int for count in value["frontier_status_counts"].values())
              and value["frontier_status_counts"] == statuses, "source frontier counts differ")
        _need(cid_for_structured(_frontier_payload(value, frontier)) == value["frontier_cid"]
              and cid_for_structured({key: item for key, item in value.items() if key != "scope_cid"}) == value["scope_cid"],
              "source scope or frontier commitment differs")
        return value, frontier

    def observation(self):
        return self._validated()[0]

    def semantic_inputs(self):
        value, _ = self._validated()
        selected = value["selected"]
        return dict(paths=sorted(row["path"] for row in selected),
            required_raw_paths=sorted(row["path"] for row in selected if row["required"]),
            program_paths=sorted(row["path"] for row in selected if row["role"] == "program"),
            max_files=MAX_FILES, max_source_bytes=MAX_SOURCE_BYTES)

    def frontier_page(self, *, offset=0, limit=64, max_bytes=16384):
        value, frontier = self._validated()
        _need(type(offset) is int and 0 <= offset <= len(frontier), "invalid source frontier offset")
        _need(type(limit) is int and 1 <= limit <= MAX_FILES, "invalid source frontier row limit")
        _need(type(max_bytes) is int and 2048 <= max_bytes <= 65536, "invalid source frontier byte limit")
        end = min(offset + limit, len(frontier))
        while True:
            page = dict(schema=PAGE_SCHEMA, source_inventory_cid=value["source_inventory_cid"],
                native_snapshot_cid=value["native_snapshot_cid"], population_cid=value["population_cid"],
                scope_cid=value["scope_cid"], frontier_cid=value["frontier_cid"],
                total_count=len(frontier), offset=offset, end_offset=end, returned_count=end-offset,
                entries=frontier[offset:end], next_offset=None if end == len(frontier) else end,
                exhausted=end == len(frontier), **_FLAGS)
            if len(canonical_dag_json_bytes(page)) <= max_bytes:
                _need(end > offset or offset == len(frontier), "source frontier cannot fit one complete entry")
                return page
            _need(end > offset, "source frontier metadata exceeds byte bound")
            end -= 1


def nominate_source_scope(*, cas, expected_inventory_cid, request,
                          limits=SourceInventoryLimits(), selected_paths: tuple[str, ...],
                          required_paths: tuple[str, ...], program_paths: tuple[str, ...]) -> RepositorySourceScope:
    """Nominate the exact reviewed subset; no auto-append, truncation or fallback.

    Paths are canonical lowercase raw-path hex, with required/program paths as
    explicit subsets; required paths are nonempty and program paths may be empty.
    Every unselected committed entry remains in the
    historical frontier, including symlinks, gitlinks and unsupported bytes.
    Mandatory support must be selected explicitly and consumes the same bounds.
    """
    guard = _input_guard(request, limits)
    selected = _paths(selected_paths, "selected paths")
    required = _paths(required_paths, "required paths")
    program = _paths(program_paths, "program paths", allow_empty=True)
    _need(set(required) <= set(selected) and set(program) <= set(selected),
          "required and program paths must be explicit selected subsets")
    cid = _cid(expected_inventory_cid)
    independent = SourceInventoryRequest(**asdict(request))
    copied_limits = SourceInventoryLimits(**asdict(limits))
    guard()
    restored = load_source_inventory(cas=cas, expected_inventory_cid=cid,
        request=independent, limits=copied_limits)
    guard()
    selected_set, required_set, program_set = set(selected), set(required), set(program)
    chosen, frontier, population = [], [], set()
    for item in restored.entries:
        row = _metadata(item)
        raw_path = row["raw_path_hex"]
        _need(raw_path not in population, "committed source population is duplicated")
        population.add(raw_path)
        status = _status(row, item.data)
        if raw_path in selected_set:
            _need(status == "not_selected_eligible", "selected source is outside supported task scope: " + status)
            chosen.append(dict(row, role="program" if raw_path in program_set else "support",
                               required=raw_path in required_set))
        else:
            frontier.append(dict(row, status=status))
    _need(selected_set <= population, "selected source path is absent from retained inventory")
    chosen.sort(key=lambda row: row["raw_path_hex"])
    frontier.sort(key=lambda row: row["raw_path_hex"])
    selected_bytes = sum(row["source_bytes"] for row in chosen)
    _need(selected_bytes <= MAX_SOURCE_BYTES, "selected originals exceed task source byte bound")
    snapshot = restored.native_snapshot
    value = dict(schema=SCHEMA, request=asdict(independent), source_limits=asdict(copied_limits),
        source_inventory_cid=cid, native_snapshot_cid=snapshot.snapshot_cid,
        population_cid=snapshot.population_cid, git_commit=snapshot.git_commit, git_tree=snapshot.git_tree,
        inventory_entry_count=len(population), selected=chosen, selected_count=len(chosen),
        selected_bytes=selected_bytes, required_paths=list(required), program_paths=list(program),
        max_files=MAX_FILES, max_source_bytes=MAX_SOURCE_BYTES, frontier_count=len(frontier),
        frontier_status_counts={status: sum(row["status"] == status for row in frontier) for status in FRONTIER_STATUSES},
        **_FLAGS)
    value["frontier_cid"] = cid_for_structured(_frontier_payload(value, frontier))
    value["scope_cid"] = cid_for_structured(value)
    result = RepositorySourceScope(canonical_dag_json_bytes(value),
        tuple(canonical_dag_json_bytes(row) for row in frontier))
    guard()
    return result


__all__ = ["RepositorySourceScope", "RepositorySourceScopeError", "nominate_source_scope",
           "SCHEMA", "PAGE_SCHEMA", "FRONTIER_STATUSES", "MAX_FILES", "MAX_SOURCE_BYTES"]
