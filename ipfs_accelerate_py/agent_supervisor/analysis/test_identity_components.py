"""Fail-closed identity components for proof-backed pytest reuse.

This module implements the component side of ``TestExecutionKey@1``.  It does
not decide whether a cached result may skip a test.  Instead, it produces
content identities for behavior-affecting pytest inputs and reports inputs
that cannot be represented safely as explicit non-reusable reasons.

Privacy is part of the identity contract:

* environment variable names come from a fixed, reviewed allowlist;
* raw environment and fixture values are reduced to content identities;
* absolute host paths, distribution locations, host names, device serials,
  and network identifiers are never retained;
* callers must explicitly allow every capability identifier.

Only finite, reviewed parameter types are canonical.  There is deliberately no
``repr``/``str``/pickle fallback: such fallbacks are unstable and can execute
user code or disclose private state.
"""

from __future__ import annotations

import os
import platform as _platform
import re
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)

TEST_IDENTITY_COMPONENTS_INTERFACE: Final = "TestIdentityComponents@1"
TEST_IDENTITY_COMPONENTS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-identity-components@1"
)
PARAMETER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/pytest-parameter@1"
)
FIXTURE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/pytest-fixture@1"
HOOK_PLUGIN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/pytest-hook-plugin@1"
)
DEPENDENCY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-dependency-identity@1"
)
ENVIRONMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-environment-identity@1"
)

MAX_PARAMETER_DEPTH: Final = 16
MAX_PARAMETER_ITEMS: Final = 1_024
MAX_PARAMETER_TEXT_CHARS: Final = 16_384
MAX_PARAMETER_BYTES: Final = 1_048_576
MAX_RECORDS: Final = 512
MAX_SOURCE_BYTES: Final = 4 * 1_048_576
MAX_LOCK_BYTES: Final = 16 * 1_048_576
MAX_ENV_VALUE_CHARS: Final = 256

_CID_RE: Final = re.compile(r"^b[a-z2-7]{20,}$")
_SAFE_NAME_RE: Final = re.compile(r"^[A-Za-z0-9_.:@/+ -]{1,256}$")
_CAPABILITY_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._:/+-]{0,127}$")
_DISTRIBUTION_NAME_RE: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_INTEGER_TEXT_RE: Final = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_NONNEGATIVE_INTEGER_TEXT_RE: Final = re.compile(r"^(?:0|[1-9][0-9]*)$")
_LOCALE_RE: Final = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
_DEVICE_SELECTION_RE: Final = re.compile(r"^(?:-1|none|void|[0-9]+(?:,[0-9]+)*)$", re.I)
_VERSION_RE: Final = re.compile(r"^[A-Za-z0-9_.+!~-]{1,128}$")

_MISSING: Final = object()
_NO_PARAMETER: Final = object()

FIXTURE_SCOPES: Final = frozenset(
    {"function", "class", "module", "package", "session"}
)
HOOK_KINDS: Final = frozenset({"hook", "plugin"})

# Values are validation policies, not merely documented suggestions.  A caller
# cannot add an arbitrary name to this set.
ENVIRONMENT_VALUE_POLICIES: Final = MappingProxyType(
    {
        "CI": "boolean",
        "CUDA_VISIBLE_DEVICES": "device_selection",
        "HIP_VISIBLE_DEVICES": "device_selection",
        "IPFS_TEST_PROOF_REUSE_MODE": "reuse_mode",
        "LANG": "locale",
        "LC_ALL": "locale",
        "MKL_NUM_THREADS": "positive_integer",
        "OMP_NUM_THREADS": "positive_integer",
        "PYTHONHASHSEED": "hash_seed",
        "PYTHONUTF8": "boolean",
        "TZ": "timezone",
    }
)
DEFAULT_ENVIRONMENT_ALLOWLIST: Final = tuple(sorted(ENVIRONMENT_VALUE_POLICIES))

INTERPRETER_FACT_ALLOWLIST: Final = frozenset(
    {
        "implementation",
        "version",
        "cache_tag",
        "abi_flags",
        "byteorder",
        "pointer_bits",
    }
)
PLATFORM_FACT_ALLOWLIST: Final = frozenset(
    {"system", "release", "machine", "python_compiler", "libc"}
)
HARDWARE_FACT_ALLOWLIST: Final = frozenset(
    {
        "architecture",
        "cpu_count",
        "accelerator_backend",
        "accelerator_count",
        "accelerator_architectures",
    }
)
LOCK_FILE_NAMES: Final = frozenset(
    {
        "cargo.lock",
        "composer.lock",
        "conda-lock.yml",
        "gemfile.lock",
        "package-lock.json",
        "pipfile.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "requirements.lock",
        "uv.lock",
        "yarn.lock",
    }
)
LOCK_FILE_SUFFIXES: Final = (".lock", ".lock.json")


class TestIdentityComponentError(ValueError):
    """Base error for malformed or unsafe identity-component input."""

    __test__ = False


class UnsupportedPytestParameter(TestIdentityComponentError):
    """A pytest parameter cannot be represented by the reviewed finite profile."""


def _reason(code: str) -> str:
    """Return one of the bounded public reason codes used by this module."""

    if not isinstance(code, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]{0,95}", code
    ):
        return "identity_component_rejected"
    return code


def _safe_name(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME_RE.fullmatch(value):
        raise TestIdentityComponentError(
            f"{field_name} must be a bounded, privacy-safe name"
        )
    return value


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TestIdentityComponentError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TestIdentityComponentError(f"{field_name} keys must be strings")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: Iterable[str], *, field_name: str
) -> None:
    if set(value).difference(allowed):
        raise TestIdentityComponentError(
            f"{field_name} contains unsupported fields"
        )


def _require_cid(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _CID_RE.fullmatch(value):
        raise TestIdentityComponentError(
            f"{field_name} must be a lowercase base32 CIDv1"
        )
    return value


def _canonical_parameter(
    value: Any,
    *,
    depth: int,
    budget: list[int],
    active_ids: set[int],
) -> dict[str, Any]:
    if depth > MAX_PARAMETER_DEPTH:
        raise UnsupportedPytestParameter("pytest_parameter_too_deep")
    budget[0] += 1
    if budget[0] > MAX_PARAMETER_ITEMS:
        raise UnsupportedPytestParameter("pytest_parameter_too_many_items")

    value_type = type(value)
    if value is None:
        return {"type": "none"}
    if value_type is bool:
        return {"type": "bool", "value": value}
    if value_type is int:
        if value.bit_length() > 256:
            raise UnsupportedPytestParameter("pytest_parameter_integer_too_large")
        return {"type": "int", "value": value}
    if value_type is str:
        if len(value) > MAX_PARAMETER_TEXT_CHARS:
            raise UnsupportedPytestParameter("pytest_parameter_text_too_large")
        return {"type": "str", "value": value}
    if value_type is bytes:
        if len(value) > MAX_PARAMETER_BYTES:
            raise UnsupportedPytestParameter("pytest_parameter_bytes_too_large")
        return {"type": "bytes", "hex": value.hex()}
    if value_type is float:
        # Even finite IEEE values have several non-portable edge cases (-0,
        # NaN payloads, JSON formatting).  The v1 profile rejects all floats.
        raise UnsupportedPytestParameter("unsupported_pytest_parameter_float")

    container_types = {list, tuple, dict, set, frozenset}
    if value_type not in container_types:
        raise UnsupportedPytestParameter("unsupported_pytest_parameter_type")

    object_id = id(value)
    if object_id in active_ids:
        raise UnsupportedPytestParameter("cyclic_pytest_parameter")
    active_ids.add(object_id)
    try:
        if value_type in (list, tuple):
            if len(value) > MAX_PARAMETER_ITEMS:
                raise UnsupportedPytestParameter(
                    "pytest_parameter_too_many_items"
                )
            return {
                "type": "tuple" if value_type is tuple else "list",
                "items": [
                    _canonical_parameter(
                        item,
                        depth=depth + 1,
                        budget=budget,
                        active_ids=active_ids,
                    )
                    for item in value
                ],
            }
        if value_type is dict:
            if len(value) > MAX_PARAMETER_ITEMS:
                raise UnsupportedPytestParameter(
                    "pytest_parameter_too_many_items"
                )
            if not all(type(key) is str for key in value):
                raise UnsupportedPytestParameter(
                    "pytest_parameter_mapping_key_not_string"
                )
            entries = []
            for key in sorted(value):
                if len(key) > MAX_PARAMETER_TEXT_CHARS:
                    raise UnsupportedPytestParameter(
                        "pytest_parameter_text_too_large"
                    )
                entries.append(
                    {
                        "key": key,
                        "value": _canonical_parameter(
                            value[key],
                            depth=depth + 1,
                            budget=budget,
                            active_ids=active_ids,
                        ),
                    }
                )
            return {"type": "mapping", "items": entries}

        encoded_items = [
            _canonical_parameter(
                item,
                depth=depth + 1,
                budget=budget,
                active_ids=active_ids,
            )
            for item in value
        ]
        encoded_items.sort(key=canonical_json_bytes)
        for previous, current in zip(
            encoded_items, encoded_items[1:], strict=False
        ):
            if canonical_json_bytes(previous) == canonical_json_bytes(current):
                raise UnsupportedPytestParameter(
                    "pytest_parameter_set_has_canonical_collision"
                )
        return {
            "type": "frozenset" if value_type is frozenset else "set",
            "items": encoded_items,
        }
    finally:
        active_ids.remove(object_id)


def canonicalize_pytest_parameter(value: Any) -> dict[str, Any]:
    """Return a type-preserving canonical parameter or raise explicitly.

    Supported values are ``None``, booleans, bounded integers, strings, bytes,
    lists, tuples, string-keyed dictionaries, sets, and frozensets.  Container
    type is retained, mapping keys and set members are deterministically
    ordered, and cycles/canonical set collisions are rejected.
    """

    return _canonical_parameter(value, depth=0, budget=[0], active_ids=set())


def _private_value_cid(value: Any, *, domain: str) -> str:
    """Content-address a value without returning the value in public records."""

    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/private-value@1",
            "domain": domain,
            "value": canonicalize_pytest_parameter(value),
        }
    )


def _source_cid(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        raw = value.encode("utf-8")
        media_type = "text/x-python"
    elif isinstance(value, bytes):
        raw = value
        media_type = "application/octet-stream"
    else:
        raise TestIdentityComponentError(
            f"{field_name} must be source text, bytes, or a CID"
        )
    if len(raw) > MAX_SOURCE_BYTES:
        raise TestIdentityComponentError(f"{field_name} exceeds the source bound")
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/source-blob@1",
            "media_type": media_type,
            "bytes": raw.hex(),
        }
    )


def _record_content_cid(
    record: Mapping[str, Any],
    *,
    cid_field: str,
    content_field: str,
    field_name: str,
) -> str:
    cid = record.get(cid_field)
    content = record.get(content_field, _MISSING)
    if (cid in (None, "")) == (content is _MISSING):
        raise TestIdentityComponentError(
            f"{field_name} requires exactly one of {cid_field} or {content_field}"
        )
    return (
        _require_cid(cid, field_name=cid_field)
        if content is _MISSING
        else _source_cid(content, field_name=content_field)
    )


@dataclass(frozen=True)
class FixtureHookIdentity:
    """Content roots for fixtures, conftests, and hook/plugin code."""

    fixture_cids: tuple[str, ...]
    conftest_closure_cid: str
    hook_plugin_cids: tuple[str, ...]
    non_reusable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_cids", tuple(sorted(set(self.fixture_cids))))
        object.__setattr__(
            self, "hook_plugin_cids", tuple(sorted(set(self.hook_plugin_cids)))
        )
        for name, values in (
            ("fixture_cids", self.fixture_cids),
            ("hook_plugin_cids", self.hook_plugin_cids),
        ):
            if len(values) > MAX_RECORDS:
                raise TestIdentityComponentError(f"{name} exceeds its record bound")
            for value in values:
                _require_cid(value, field_name=name)
        _require_cid(self.conftest_closure_cid, field_name="conftest_closure_cid")
        object.__setattr__(
            self,
            "non_reusable_reasons",
            tuple(
                sorted(
                    {
                        _reason(reason)
                        for reason in self.non_reusable_reasons
                    }
                )
            ),
        )


def _fixture_cid(raw: Any) -> tuple[str, str, tuple[str, ...]]:
    record = _require_mapping(raw, field_name="fixture")
    allowed = {
        "name",
        "scope",
        "definition",
        "definition_cid",
        "value",
        "value_cid",
        "value_adapter_cid",
        "dependencies",
        "autouse",
        "parameter_values",
    }
    _reject_unknown_fields(record, allowed, field_name="fixture")
    name = _safe_name(record.get("name"), field_name="fixture.name")
    scope = record.get("scope")
    if scope not in FIXTURE_SCOPES:
        raise TestIdentityComponentError("fixture.scope is not a supported pytest scope")
    definition_cid = _record_content_cid(
        record,
        cid_field="definition_cid",
        content_field="definition",
        field_name="fixture definition",
    )
    dependencies = record.get("dependencies", ())
    if isinstance(dependencies, str) or not isinstance(dependencies, Sequence):
        raise TestIdentityComponentError("fixture.dependencies must be a sequence")
    dependency_names = tuple(
        sorted({_safe_name(item, field_name="fixture dependency") for item in dependencies})
    )
    if len(dependency_names) > MAX_RECORDS:
        raise TestIdentityComponentError("fixture dependencies exceed their bound")
    autouse = record.get("autouse", False)
    if not isinstance(autouse, bool):
        raise TestIdentityComponentError("fixture.autouse must be boolean")

    value_modes = sum(
        (
            "value" in record,
            record.get("value_cid") not in (None, ""),
            record.get("value_adapter_cid") not in (None, ""),
        )
    )
    if value_modes > 1:
        raise TestIdentityComponentError(
            "fixture value must use one value, value_cid, or value_adapter_cid"
        )
    reasons: list[str] = []
    if "value" in record:
        try:
            value_cid = _private_value_cid(
                record["value"], domain=f"fixture:{name}"
            )
        except UnsupportedPytestParameter:
            reasons.append("unsupported_fixture_value")
            value_identity = {"kind": "unsupported"}
        else:
            value_identity = {"kind": "canonical_value", "cid": value_cid}
    elif record.get("value_cid") not in (None, ""):
        value_identity = {
            "kind": "content",
            "cid": _require_cid(record["value_cid"], field_name="fixture.value_cid"),
        }
    elif record.get("value_adapter_cid") not in (None, ""):
        value_identity = {
            "kind": "adapter",
            "cid": _require_cid(
                record["value_adapter_cid"], field_name="fixture.value_adapter_cid"
            ),
        }
    else:
        # A fixture definition alone does not control the object supplied to a
        # test.  Preserve a deterministic identity for diagnostics but prevent
        # it from authorizing reuse.
        value_identity = {"kind": "uncontrolled"}
        reasons.append("uncontrolled_fixture_value")

    parameter_values = record.get("parameter_values", ())
    if isinstance(parameter_values, (str, bytes)) or not isinstance(
        parameter_values, Sequence
    ):
        raise TestIdentityComponentError("fixture.parameter_values must be a sequence")
    if len(parameter_values) > MAX_RECORDS:
        raise TestIdentityComponentError("fixture.parameter_values exceeds its bound")
    parameter_cids = []
    for item in parameter_values:
        try:
            parameter_cids.append(
                _private_value_cid(item, domain=f"fixture-parameter:{name}")
            )
        except UnsupportedPytestParameter:
            reasons.append("unsupported_fixture_parameter")
            parameter_cids.append(
                content_identity(
                    {
                        "schema": (
                            "ipfs_accelerate_py/agent-supervisor/"
                            "unsupported-fixture-parameter@1"
                        ),
                        "fixture_name": name,
                    }
                )
            )
    fixture_cid = content_identity(
        {
            "schema": FIXTURE_SCHEMA,
            "name": name,
            "scope": scope,
            "definition_cid": definition_cid,
            "value_identity": value_identity,
            "dependencies": list(dependency_names),
            "autouse": autouse,
            "parameter_cids": list(parameter_cids),
        }
    )
    return name, fixture_cid, tuple(sorted(set(reasons)))


def _normalized_relative_path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise TestIdentityComponentError(f"{field_name} must be a relative path")
    raw = os.fspath(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or re.match(r"^[A-Za-z]:/", raw)
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise TestIdentityComponentError(
            f"{field_name} must not contain an absolute path or traversal"
        )
    normalized = path.as_posix()
    if len(normalized) > 1_024:
        raise TestIdentityComponentError(f"{field_name} is too long")
    return normalized


def _conftest_cid(raw: Any) -> tuple[str, str]:
    record = _require_mapping(raw, field_name="conftest")
    _reject_unknown_fields(
        record, {"path", "content", "content_cid"}, field_name="conftest"
    )
    path = _normalized_relative_path(record.get("path"), field_name="conftest.path")
    if PurePosixPath(path).name != "conftest.py":
        raise TestIdentityComponentError("conftest.path must name conftest.py")
    source_cid = _record_content_cid(
        record,
        cid_field="content_cid",
        content_field="content",
        field_name="conftest content",
    )
    return path, content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/conftest@1",
            "path": path,
            "content_cid": source_cid,
        }
    )


def _hook_plugin_cid(raw: Any) -> str:
    record = _require_mapping(raw, field_name="hook/plugin")
    allowed = {
        "kind",
        "name",
        "implementation",
        "implementation_cid",
        "distribution",
        "version",
        "registered",
        "order",
    }
    _reject_unknown_fields(record, allowed, field_name="hook/plugin")
    kind = record.get("kind")
    if kind not in HOOK_KINDS:
        raise TestIdentityComponentError("hook/plugin.kind must be hook or plugin")
    name = _safe_name(record.get("name"), field_name="hook/plugin.name")
    implementation_cid = _record_content_cid(
        record,
        cid_field="implementation_cid",
        content_field="implementation",
        field_name="hook/plugin implementation",
    )
    distribution = record.get("distribution", "")
    if distribution:
        distribution = _normalize_distribution_name(distribution)
    version = record.get("version", "")
    if version and (not isinstance(version, str) or not _VERSION_RE.fullmatch(version)):
        raise TestIdentityComponentError("hook/plugin.version is invalid")
    registered = record.get("registered", True)
    if not isinstance(registered, bool):
        raise TestIdentityComponentError("hook/plugin.registered must be boolean")
    order = record.get("order", 0)
    if isinstance(order, bool) or not isinstance(order, int) or abs(order) > 1_000_000:
        raise TestIdentityComponentError("hook/plugin.order must be a bounded integer")
    return content_identity(
        {
            "schema": HOOK_PLUGIN_SCHEMA,
            "kind": kind,
            "name": name,
            "implementation_cid": implementation_cid,
            "distribution": distribution,
            "version": version,
            "registered": registered,
            "order": order,
        }
    )


def collect_fixture_hook_identity(
    *,
    fixtures: Iterable[Mapping[str, Any]] = (),
    conftests: Iterable[Mapping[str, Any]] = (),
    hooks: Iterable[Mapping[str, Any]] = (),
    plugins: Iterable[Mapping[str, Any]] = (),
) -> FixtureHookIdentity:
    """Compile fixture definitions/values and the active hook/plugin closure."""

    fixture_records = tuple(fixtures)
    conftest_records = tuple(conftests)
    hook_records = tuple(hooks)
    plugin_records = tuple(plugins)
    if any(
        len(records) > MAX_RECORDS
        for records in (fixture_records, conftest_records, hook_records, plugin_records)
    ):
        raise TestIdentityComponentError("fixture/hook input exceeds its record bound")
    compiled_fixtures = [_fixture_cid(item) for item in fixture_records]
    fixture_names = [name for name, _, _ in compiled_fixtures]
    if len(fixture_names) != len(set(fixture_names)):
        raise TestIdentityComponentError("duplicate fixture name")
    fixture_cids = tuple(sorted(cid for _, cid, _ in compiled_fixtures))
    fixture_reasons = tuple(
        sorted(
            {
                reason
                for _, _, reasons in compiled_fixtures
                for reason in reasons
            }
        )
    )
    normalized_conftests = sorted(_conftest_cid(item) for item in conftest_records)
    conftest_paths = [path for path, _ in normalized_conftests]
    if len(conftest_paths) != len(set(conftest_paths)):
        raise TestIdentityComponentError("duplicate conftest path")
    conftest_closure_cid = content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/conftest-closure@1",
            "entries": [
                {"path": path, "cid": cid} for path, cid in normalized_conftests
            ],
        }
    )
    normalized_hook_records = []
    for kind, records in (("hook", hook_records), ("plugin", plugin_records)):
        for item in records:
            record = _require_mapping(item, field_name=kind)
            normalized_hook_records.append(
                _hook_plugin_cid(
                    {**record, "kind": record.get("kind", kind)}
                )
            )
    if len(normalized_hook_records) != len(set(normalized_hook_records)):
        raise TestIdentityComponentError("duplicate hook/plugin identity")
    hook_plugin_cids = tuple(sorted(normalized_hook_records))
    return FixtureHookIdentity(
        fixture_cids=fixture_cids,
        conftest_closure_cid=conftest_closure_cid,
        hook_plugin_cids=hook_plugin_cids,
        non_reusable_reasons=fixture_reasons,
    )


@dataclass(frozen=True)
class DependencyIdentity:
    """Lock-file and installed-distribution content roots."""

    dependency_lock_cid: str
    installed_distributions_cid: str

    def __post_init__(self) -> None:
        _require_cid(self.dependency_lock_cid, field_name="dependency_lock_cid")
        _require_cid(
            self.installed_distributions_cid,
            field_name="installed_distributions_cid",
        )


def _is_lock_file(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return name in LOCK_FILE_NAMES or any(name.endswith(suffix) for suffix in LOCK_FILE_SUFFIXES)


def _lock_records(
    lock_files: Mapping[Any, Any] | Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(lock_files, Mapping):
        raw_records = [
            {"path": path, "content": content} for path, content in lock_files.items()
        ]
    else:
        raw_records = list(lock_files)
    if len(raw_records) > MAX_RECORDS:
        raise TestIdentityComponentError("lock_files exceeds its record bound")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_records:
        record = _require_mapping(raw, field_name="lock file")
        _reject_unknown_fields(
            record, {"path", "content", "content_cid"}, field_name="lock file"
        )
        path = _normalized_relative_path(record.get("path"), field_name="lock file.path")
        if not _is_lock_file(path):
            raise TestIdentityComponentError("lock file path is not allowlisted")
        if path in seen:
            raise TestIdentityComponentError("duplicate normalized lock file path")
        seen.add(path)
        content = record.get("content", _MISSING)
        if content is not _MISSING:
            if isinstance(content, str):
                raw_content = content.encode("utf-8")
            elif isinstance(content, bytes):
                raw_content = content
            else:
                raise TestIdentityComponentError("lock file content must be text or bytes")
            if len(raw_content) > MAX_LOCK_BYTES:
                raise TestIdentityComponentError("lock file exceeds its byte bound")
            content_cid = content_identity(
                {
                    "schema": "ipfs_accelerate_py/agent-supervisor/lock-bytes@1",
                    "bytes": raw_content.hex(),
                }
            )
            if record.get("content_cid") not in (None, ""):
                raise TestIdentityComponentError(
                    "lock file must not provide both content and content_cid"
                )
        else:
            content_cid = _require_cid(
                record.get("content_cid"), field_name="lock file.content_cid"
            )
        records.append({"path": path, "content_cid": content_cid})
    return sorted(records, key=lambda item: item["path"])


def _normalize_distribution_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TestIdentityComponentError("distribution name must be a string")
    normalized = re.sub(r"[-_.]+", "-", value.strip().lower())
    if not _DISTRIBUTION_NAME_RE.fullmatch(normalized):
        raise TestIdentityComponentError("distribution name is invalid")
    return normalized


def _distribution_records(
    installed_distributions: Mapping[str, str] | Iterable[tuple[str, str]],
) -> list[dict[str, str]]:
    items = (
        list(installed_distributions.items())
        if isinstance(installed_distributions, Mapping)
        else list(installed_distributions)
    )
    if len(items) > MAX_RECORDS:
        raise TestIdentityComponentError(
            "installed_distributions exceeds its record bound"
        )
    normalized: dict[str, str] = {}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TestIdentityComponentError(
                "installed distributions must contain name/version pairs"
            )
        name = _normalize_distribution_name(item[0])
        version = item[1]
        if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
            raise TestIdentityComponentError("distribution version is invalid")
        previous = normalized.get(name)
        if previous is not None and previous != version:
            raise TestIdentityComponentError(
                "normalized distribution names collide"
            )
        normalized[name] = version
    return [
        {"name": name, "version": normalized[name]} for name in sorted(normalized)
    ]


def collect_dependency_identity(
    *,
    lock_files: Mapping[Any, Any] | Iterable[Mapping[str, Any]] = (),
    installed_distributions: (
        Mapping[str, str] | Iterable[tuple[str, str]]
    ) = (),
) -> DependencyIdentity:
    """Compile normalized lock bytes and distribution name/version pairs.

    Distribution installation locations and direct-url metadata are
    intentionally excluded because they can contain user names and credentials.
    """

    locks = _lock_records(lock_files)
    distributions = _distribution_records(installed_distributions)
    return DependencyIdentity(
        dependency_lock_cid=content_identity(
            {
                "schema": DEPENDENCY_SCHEMA,
                "kind": "lock_files",
                "entries": locks,
            }
        ),
        installed_distributions_cid=content_identity(
            {
                "schema": DEPENDENCY_SCHEMA,
                "kind": "installed_distributions",
                "entries": distributions,
            }
        ),
    )


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Privacy-safe execution environment roots."""

    environment_cid: str
    platform_cid: str
    interpreter_abi_cid: str
    hardware_capability_cid: str

    def __post_init__(self) -> None:
        for name in (
            "environment_cid",
            "platform_cid",
            "interpreter_abi_cid",
            "hardware_capability_cid",
        ):
            _require_cid(getattr(self, name), field_name=name)


def _validate_environment_value(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) > MAX_ENV_VALUE_CHARS:
        raise TestIdentityComponentError(
            f"allowlisted environment value for {name} is invalid"
        )
    policy = ENVIRONMENT_VALUE_POLICIES[name]
    stripped = value.strip()
    if policy == "boolean" and stripped.lower() not in {
        "",
        "0",
        "1",
        "false",
        "true",
        "no",
        "yes",
    }:
        raise TestIdentityComponentError(f"environment value for {name} is invalid")
    if (
        policy == "device_selection"
        and stripped
        and not _DEVICE_SELECTION_RE.fullmatch(stripped)
    ):
        raise TestIdentityComponentError(f"environment value for {name} is invalid")
    if policy == "reuse_mode" and stripped.lower() not in {
        "off",
        "shadow",
        "read",
        "write",
        "readwrite",
    }:
        raise TestIdentityComponentError(f"environment value for {name} is invalid")
    if policy == "locale" and not _LOCALE_RE.fullmatch(stripped):
        raise TestIdentityComponentError(f"environment value for {name} is invalid")
    if policy == "positive_integer" and (
        not _NONNEGATIVE_INTEGER_TEXT_RE.fullmatch(stripped) or int(stripped) < 1
    ):
        raise TestIdentityComponentError(f"environment value for {name} is invalid")
    if policy == "hash_seed" and (
        stripped.lower() != "random"
        and not _NONNEGATIVE_INTEGER_TEXT_RE.fullmatch(stripped)
    ):
        raise TestIdentityComponentError(f"environment value for {name} is invalid")
    if policy == "timezone" and (
        len(stripped) > 64
        or not re.fullmatch(r"[A-Za-z0-9_+./:-]{1,64}", stripped)
        or ".." in stripped
    ):
        raise TestIdentityComponentError(f"environment value for {name} is invalid")
    return stripped


def _current_interpreter_facts() -> dict[str, Any]:
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:5]),
        "cache_tag": sys.implementation.cache_tag or "",
        "abi_flags": getattr(sys, "abiflags", ""),
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
    }


def _current_platform_facts() -> dict[str, Any]:
    libc_name, libc_version = _platform.libc_ver()
    return {
        "system": _platform.system().lower(),
        "release": _platform.release(),
        "machine": _platform.machine().lower(),
        "python_compiler": _platform.python_compiler(),
        "libc": [libc_name, libc_version],
    }


def _current_hardware_facts() -> dict[str, Any]:
    return {
        "architecture": _platform.machine().lower(),
        "cpu_count": os.cpu_count() or 0,
        "accelerator_backend": "none",
        "accelerator_count": 0,
        "accelerator_architectures": [],
    }


def _allowlisted_fact_map(
    value: Mapping[str, Any],
    *,
    allowlist: frozenset[str],
    field_name: str,
) -> dict[str, Any]:
    facts = _require_mapping(value, field_name=field_name)
    fact_names = set(facts)
    if fact_names.difference(allowlist):
        raise TestIdentityComponentError(
            f"{field_name} contains a non-allowlisted fact"
        )
    if fact_names != allowlist:
        raise TestIdentityComponentError(
            f"{field_name} is missing a required allowlisted fact"
        )
    # The tagged parameter profile supplies finite recursion and type bounds.
    return {
        key: canonicalize_pytest_parameter(facts[key]) for key in sorted(facts)
    }


def collect_environment_identity(
    *,
    environment: Mapping[str, str] | None = None,
    environment_allowlist: Iterable[str] = DEFAULT_ENVIRONMENT_ALLOWLIST,
    interpreter_facts: Mapping[str, Any] | None = None,
    platform_facts: Mapping[str, Any] | None = None,
    hardware_facts: Mapping[str, Any] | None = None,
    capability_facts: Mapping[str, Any] | None = None,
    capability_allowlist: Iterable[str] = (),
) -> EnvironmentIdentity:
    """Collect privacy-safe environment, runtime, and capability roots.

    Raw environment values are not present in the resulting artifacts.  Each
    value is represented by a domain-separated CID, including an explicit
    marker for an allowlisted-but-unset variable.  Extra keys in the supplied
    environment mapping are ignored, so passing ``os.environ`` does not ingest
    secrets accidentally.
    """

    source_environment = os.environ if environment is None else environment
    source_environment = _require_mapping(
        source_environment, field_name="environment"
    )
    try:
        requested_environment_names_raw = tuple(environment_allowlist)
    except TypeError as exc:
        raise TestIdentityComponentError(
            "environment allowlist must be an iterable of names"
        ) from exc
    if any(not isinstance(name, str) for name in requested_environment_names_raw):
        raise TestIdentityComponentError(
            "environment allowlist contains a non-reviewed variable"
        )
    requested_environment_names = tuple(
        sorted(set(requested_environment_names_raw))
    )
    if len(requested_environment_names) > len(ENVIRONMENT_VALUE_POLICIES):
        raise TestIdentityComponentError("environment allowlist exceeds its bound")
    if any(name not in ENVIRONMENT_VALUE_POLICIES for name in requested_environment_names):
        raise TestIdentityComponentError(
            "environment allowlist contains a non-reviewed variable"
        )
    environment_entries = []
    for name in requested_environment_names:
        if name in source_environment:
            normalized = _validate_environment_value(name, source_environment[name])
            value_cid = content_identity(
                {
                    "schema": "ipfs_accelerate_py/agent-supervisor/environment-value@1",
                    "name": name,
                    "state": "set",
                    "value": normalized,
                }
            )
            state = "set"
        else:
            value_cid = content_identity(
                {
                    "schema": "ipfs_accelerate_py/agent-supervisor/environment-value@1",
                    "name": name,
                    "state": "unset",
                }
            )
            state = "unset"
        environment_entries.append(
            {"name": name, "state": state, "value_cid": value_cid}
        )

    interpreter = _allowlisted_fact_map(
        _current_interpreter_facts()
        if interpreter_facts is None
        else interpreter_facts,
        allowlist=INTERPRETER_FACT_ALLOWLIST,
        field_name="interpreter_facts",
    )
    platform = _allowlisted_fact_map(
        _current_platform_facts() if platform_facts is None else platform_facts,
        allowlist=PLATFORM_FACT_ALLOWLIST,
        field_name="platform_facts",
    )
    hardware = _allowlisted_fact_map(
        _current_hardware_facts() if hardware_facts is None else hardware_facts,
        allowlist=HARDWARE_FACT_ALLOWLIST,
        field_name="hardware_facts",
    )

    capabilities = {} if capability_facts is None else _require_mapping(
        capability_facts, field_name="capability_facts"
    )
    try:
        allowed_capabilities_raw = tuple(capability_allowlist)
    except TypeError as exc:
        raise TestIdentityComponentError(
            "capability allowlist must be an iterable of IDs"
        ) from exc
    if any(not isinstance(item, str) for item in allowed_capabilities_raw):
        raise TestIdentityComponentError("capability allowlist ID is invalid")
    allowed_capabilities = tuple(sorted(set(allowed_capabilities_raw)))
    if len(allowed_capabilities) > MAX_RECORDS:
        raise TestIdentityComponentError("capability allowlist exceeds its bound")
    for capability_id in allowed_capabilities:
        if not _CAPABILITY_ID_RE.fullmatch(capability_id):
            raise TestIdentityComponentError("capability allowlist ID is invalid")
    if set(capabilities).difference(allowed_capabilities):
        raise TestIdentityComponentError(
            "capability facts contain a non-allowlisted capability"
        )
    capability_entries = []
    for capability_id in allowed_capabilities:
        state = "present" if capability_id in capabilities else "absent"
        payload: dict[str, Any] = {
            "schema": "ipfs_accelerate_py/agent-supervisor/capability-value@1",
            "capability_id": capability_id,
            "state": state,
        }
        if capability_id in capabilities:
            payload["facts"] = canonicalize_pytest_parameter(
                capabilities[capability_id]
            )
        capability_entries.append(
            {
                "capability_id": capability_id,
                "state": state,
                "facts_cid": content_identity(payload),
            }
        )

    return EnvironmentIdentity(
        environment_cid=content_identity(
            {
                "schema": ENVIRONMENT_SCHEMA,
                "kind": "environment",
                "entries": environment_entries,
            }
        ),
        interpreter_abi_cid=content_identity(
            {
                "schema": ENVIRONMENT_SCHEMA,
                "kind": "interpreter_abi",
                "facts": interpreter,
            }
        ),
        platform_cid=content_identity(
            {
                "schema": ENVIRONMENT_SCHEMA,
                "kind": "platform",
                "facts": platform,
            }
        ),
        hardware_capability_cid=content_identity(
            {
                "schema": ENVIRONMENT_SCHEMA,
                "kind": "hardware_capabilities",
                "hardware": hardware,
                "capabilities": capability_entries,
            }
        ),
    )


@dataclass(frozen=True)
class TestIdentityComponents:
    """Versioned component bundle consumed by ``TestExecutionKey@1``."""

    __test__: ClassVar[bool] = False

    parameter_cid: str
    fixture_cids: tuple[str, ...]
    conftest_closure_cid: str
    hook_plugin_cids: tuple[str, ...]
    dependency_lock_cid: str
    installed_distributions_cid: str
    environment_cid: str
    platform_cid: str
    interpreter_abi_cid: str
    hardware_capability_cid: str
    non_reusable_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "parameter_cid",
            "conftest_closure_cid",
            "dependency_lock_cid",
            "installed_distributions_cid",
            "environment_cid",
            "platform_cid",
            "interpreter_abi_cid",
            "hardware_capability_cid",
        ):
            _require_cid(getattr(self, name), field_name=name)
        for name in ("fixture_cids", "hook_plugin_cids"):
            values = tuple(sorted(set(getattr(self, name))))
            if len(values) > MAX_RECORDS:
                raise TestIdentityComponentError(f"{name} exceeds its record bound")
            for value in values:
                _require_cid(value, field_name=name)
            object.__setattr__(self, name, values)
        reasons = tuple(sorted({_reason(item) for item in self.non_reusable_reasons}))
        object.__setattr__(self, "non_reusable_reasons", reasons)

    @property
    def reusable(self) -> bool:
        return not self.non_reusable_reasons

    @property
    def component_root_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TEST_IDENTITY_COMPONENTS_SCHEMA,
            "interface": TEST_IDENTITY_COMPONENTS_INTERFACE,
            "parameter_cid": self.parameter_cid,
            "fixture_cids": list(self.fixture_cids),
            "conftest_closure_cid": self.conftest_closure_cid,
            "hook_plugin_cids": list(self.hook_plugin_cids),
            "dependency_lock_cid": self.dependency_lock_cid,
            "installed_distributions_cid": self.installed_distributions_cid,
            "environment_cid": self.environment_cid,
            "platform_cid": self.platform_cid,
            "interpreter_abi_cid": self.interpreter_abi_cid,
            "hardware_capability_cid": self.hardware_capability_cid,
            "reusable": self.reusable,
            "non_reusable_reasons": list(self.non_reusable_reasons),
        }

    def execution_key_fields(self) -> dict[str, Any]:
        """Return exactly the fields supplied to ``TestExecutionKey@1``."""

        return {
            "parameter_source_cid": self.parameter_cid,
            "fixture_cids": self.fixture_cids,
            "conftest_closure_cid": self.conftest_closure_cid,
            "hook_plugin_cids": self.hook_plugin_cids,
            "dependency_lock_cid": self.dependency_lock_cid,
            "installed_distributions_cid": self.installed_distributions_cid,
            "environment_cid": self.environment_cid,
            "platform_cid": self.platform_cid,
            "interpreter_abi_cid": self.interpreter_abi_cid,
            "hardware_capability_cid": self.hardware_capability_cid,
            "components": {
                "identity_components": self.component_root_cid,
                "parameter": self.parameter_cid,
            },
        }

    @classmethod
    def compile(
        cls,
        *,
        parameter_value: Any = _NO_PARAMETER,
        parameter_id: str = "",
        fixtures: Iterable[Mapping[str, Any]] = (),
        conftests: Iterable[Mapping[str, Any]] = (),
        hooks: Iterable[Mapping[str, Any]] = (),
        plugins: Iterable[Mapping[str, Any]] = (),
        lock_files: Mapping[Any, Any] | Iterable[Mapping[str, Any]] = (),
        installed_distributions: (
            Mapping[str, str] | Iterable[tuple[str, str]]
        ) = (),
        environment: Mapping[str, str] | None = None,
        environment_allowlist: Iterable[str] = DEFAULT_ENVIRONMENT_ALLOWLIST,
        interpreter_facts: Mapping[str, Any] | None = None,
        platform_facts: Mapping[str, Any] | None = None,
        hardware_facts: Mapping[str, Any] | None = None,
        capability_facts: Mapping[str, Any] | None = None,
        capability_allowlist: Iterable[str] = (),
    ) -> TestIdentityComponents:
        """Compile every PTR-011 root, explicitly marking unsafe parameters."""

        if not isinstance(parameter_id, str) or len(parameter_id) > 1_024:
            raise TestIdentityComponentError("parameter_id must be bounded text")
        reasons: tuple[str, ...] = ()
        if parameter_value is _NO_PARAMETER:
            parameter_payload: dict[str, Any] = {
                "schema": PARAMETER_SCHEMA,
                "parameter_id": parameter_id,
                "value": {"type": "not_parameterized"},
            }
        else:
            try:
                canonical_parameter = canonicalize_pytest_parameter(parameter_value)
            except UnsupportedPytestParameter as exc:
                # Do not reflect repr(value) or exception details.  The exception
                # itself contains only one of this module's bounded reason codes.
                reasons = (_reason(str(exc)),)
                canonical_parameter = {
                    "type": "unsupported",
                    "reason": reasons[0],
                }
            parameter_payload = {
                "schema": PARAMETER_SCHEMA,
                "parameter_id": parameter_id,
                "value": canonical_parameter,
            }

        fixture_hook = collect_fixture_hook_identity(
            fixtures=fixtures,
            conftests=conftests,
            hooks=hooks,
            plugins=plugins,
        )
        dependency = collect_dependency_identity(
            lock_files=lock_files,
            installed_distributions=installed_distributions,
        )
        environment_identity = collect_environment_identity(
            environment=environment,
            environment_allowlist=environment_allowlist,
            interpreter_facts=interpreter_facts,
            platform_facts=platform_facts,
            hardware_facts=hardware_facts,
            capability_facts=capability_facts,
            capability_allowlist=capability_allowlist,
        )
        reasons = tuple(
            sorted(set(reasons).union(fixture_hook.non_reusable_reasons))
        )
        return cls(
            parameter_cid=content_identity(parameter_payload),
            fixture_cids=fixture_hook.fixture_cids,
            conftest_closure_cid=fixture_hook.conftest_closure_cid,
            hook_plugin_cids=fixture_hook.hook_plugin_cids,
            dependency_lock_cid=dependency.dependency_lock_cid,
            installed_distributions_cid=dependency.installed_distributions_cid,
            environment_cid=environment_identity.environment_cid,
            platform_cid=environment_identity.platform_cid,
            interpreter_abi_cid=environment_identity.interpreter_abi_cid,
            hardware_capability_cid=environment_identity.hardware_capability_cid,
            non_reusable_reasons=reasons,
        )


def compile_test_identity_components(**kwargs: Any) -> TestIdentityComponents:
    """Functional spelling for :meth:`TestIdentityComponents.compile`."""

    return TestIdentityComponents.compile(**kwargs)


__all__ = [
    "DEFAULT_ENVIRONMENT_ALLOWLIST",
    "DEPENDENCY_SCHEMA",
    "ENVIRONMENT_SCHEMA",
    "ENVIRONMENT_VALUE_POLICIES",
    "FIXTURE_SCHEMA",
    "HOOK_PLUGIN_SCHEMA",
    "PARAMETER_SCHEMA",
    "TEST_IDENTITY_COMPONENTS_INTERFACE",
    "TEST_IDENTITY_COMPONENTS_SCHEMA",
    "DependencyIdentity",
    "EnvironmentIdentity",
    "FixtureHookIdentity",
    "TestIdentityComponentError",
    "TestIdentityComponents",
    "UnsupportedPytestParameter",
    "canonicalize_pytest_parameter",
    "collect_dependency_identity",
    "collect_environment_identity",
    "collect_fixture_hook_identity",
    "compile_test_identity_components",
]
