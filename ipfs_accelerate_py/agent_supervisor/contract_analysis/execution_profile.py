"""Hermetic execution contracts for datasets symbolic contract analysis.

This module is deliberately a policy and validation layer.  It never installs a
missing tool, invokes a package manager, reads credential values, or enables the
network.  A launcher supplies a :class:`CapabilitySnapshot`; this module checks
that fact against a reviewed :class:`AnalysisExecutionProfile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


PROFILE_SCHEMA = "datasets_contract_analysis/analyzer-profile@1"
RESOURCE_BOUNDS_SCHEMA = "datasets_contract_analysis/resource-bounds@1"
IDENTITY_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_CREDENTIAL_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_CLIENT_SECRET",
        "GITHUB_TOKEN",
        "GITLAB_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NPM_TOKEN",
        "OPENAI_API_KEY",
        "SSH_AUTH_SOCK",
    }
)
_CREDENTIAL_SUFFIXES = (
    "_API_KEY",
    "_ACCESS_KEY",
    "_AUTH_TOKEN",
    "_CLIENT_SECRET",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)
_HOME_CACHE_NAMES = frozenset(
    {
        "CARGO_HOME",
        "GRADLE_USER_HOME",
        "HF_HOME",
        "HOME",
        "MYPY_CACHE_DIR",
        "NPM_CONFIG_CACHE",
        "PIP_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "RUSTUP_HOME",
        "XDG_CACHE_HOME",
        "YARN_CACHE_FOLDER",
    }
)


class ExecutionProfileError(ValueError):
    """Raised when an execution profile is malformed or ambiguous."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExecutionProfileError("profile must contain canonical JSON values") from exc


def _content_identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _closed_mapping(
    value: Any,
    *,
    name: str,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionProfileError(f"{name} must be an object")
    keys = {str(key) for key in value}
    unsupported = sorted(keys - set(allowed))
    missing = sorted(set(required) - keys)
    if unsupported:
        raise ExecutionProfileError(f"{name} has unsupported fields: {unsupported}")
    if missing:
        raise ExecutionProfileError(f"{name} is missing fields: {missing}")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExecutionProfileError(f"{name} must be a positive integer")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ExecutionProfileError(f"{name} must be an array")
    normalized = tuple(str(item).strip() for item in value)
    if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
        raise ExecutionProfileError(f"{name} must contain unique non-empty strings")
    return normalized


def _identity(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not IDENTITY_PATTERN.fullmatch(result):
        raise ExecutionProfileError(f"{name} must be a lowercase sha256 identity")
    return result


def _real_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.realpath(os.fspath(value)))


def _is_within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _is_credential_name(name: str) -> bool:
    upper = name.upper()
    return upper in _CREDENTIAL_NAMES or upper.endswith(_CREDENTIAL_SUFFIXES)


@dataclass(frozen=True)
class ResourceBudget:
    """Hard analysis limits.

    Every dimension is required and positive so a missing or zero-valued limit
    cannot silently mean "unbounded".
    """

    max_blob_bytes: int
    max_files: int
    max_ast_nodes: int
    max_edges: int
    max_scc_nodes: int
    max_recursion_depth: int
    max_timeout_ms: int
    max_memory_bytes: int
    max_proof_bytes: int
    max_receipt_bytes: int
    max_findings: int
    max_tasks: int
    max_prompt_bytes: int
    max_prompt_tokens: int

    _FIELDS = (
        "max_blob_bytes",
        "max_files",
        "max_ast_nodes",
        "max_edges",
        "max_scc_nodes",
        "max_recursion_depth",
        "max_timeout_ms",
        "max_memory_bytes",
        "max_proof_bytes",
        "max_receipt_bytes",
        "max_findings",
        "max_tasks",
        "max_prompt_bytes",
        "max_prompt_tokens",
    )
    _USAGE_ALIASES = MappingProxyType(
        {
            "blob_bytes": "max_blob_bytes",
            "files": "max_files",
            "file_count": "max_files",
            "ast_nodes": "max_ast_nodes",
            "edges": "max_edges",
            "graph_edges": "max_edges",
            "scc_nodes": "max_scc_nodes",
            "recursion_depth": "max_recursion_depth",
            "timeout_ms": "max_timeout_ms",
            "wall_time_ms": "max_timeout_ms",
            "memory_bytes": "max_memory_bytes",
            "proof_bytes": "max_proof_bytes",
            "receipt_bytes": "max_receipt_bytes",
            "findings": "max_findings",
            "tasks": "max_tasks",
            "prompt_bytes": "max_prompt_bytes",
            "prompt_tokens": "max_prompt_tokens",
        }
    )

    def __post_init__(self) -> None:
        for name in self._FIELDS:
            _positive_integer(getattr(self, name), name)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceBudget":
        payload = _closed_mapping(
            value,
            name="resource_bounds",
            allowed={"schema", "class", *cls._FIELDS},
            required={"schema", "class", *cls._FIELDS},
        )
        if payload["schema"] != RESOURCE_BOUNDS_SCHEMA:
            raise ExecutionProfileError("unsupported resource-bounds schema")
        if not str(payload["class"]).strip():
            raise ExecutionProfileError("resource_bounds.class must be non-empty")
        return cls(
            **{
                name: _positive_integer(payload[name], f"resource_bounds.{name}")
                for name in cls._FIELDS
            }
        )

    def to_dict(self, *, resource_class: str) -> dict[str, Any]:
        if not resource_class.strip():
            raise ExecutionProfileError("resource_class must be non-empty")
        return {
            "schema": RESOURCE_BOUNDS_SCHEMA,
            "class": resource_class,
            **{name: getattr(self, name) for name in self._FIELDS},
        }

    @property
    def max_file_count(self) -> int:
        """Compatibility spelling for callers that count files explicitly."""

        return self.max_files

    @property
    def max_graph_edges(self) -> int:
        """Compatibility spelling for callers that count graph edges."""

        return self.max_edges

    def exhausted(self, usage: Mapping[str, Any]) -> tuple[str, ...]:
        """Return every exceeded dimension, rejecting ambiguous usage records."""

        exceeded: list[str] = []
        seen: set[str] = set()
        for supplied_name, supplied_value in usage.items():
            field_name = self._USAGE_ALIASES.get(str(supplied_name))
            if field_name is None:
                raise ExecutionProfileError(
                    f"usage has unsupported resource dimension: {supplied_name}"
                )
            if field_name in seen:
                raise ExecutionProfileError(
                    f"usage specifies resource dimension twice: {field_name}"
                )
            seen.add(field_name)
            if (
                isinstance(supplied_value, bool)
                or not isinstance(supplied_value, int)
                or supplied_value < 0
            ):
                raise ExecutionProfileError(
                    f"usage.{supplied_name} must be a non-negative integer"
                )
            if supplied_value > getattr(self, field_name):
                exceeded.append(field_name)
        return tuple(sorted(exceeded))

    def validate_usage(
        self, usage: Mapping[str, Any], *, proof_required: bool = False
    ) -> "HermeticValidation":
        exhausted = self.exhausted(usage)
        return HermeticValidation(
            safe=True,
            complete=not exhausted,
            disposition=(
                "pass" if not exhausted else ("unknown" if proof_required else "incomplete")
            ),
            exhausted_resources=exhausted,
        )


@dataclass(frozen=True)
class ToolIdentity:
    """Reviewed identity for one executable, module, parser, or checker."""

    name: str
    kind: str
    locator: str
    version: str
    identity: str
    roles: tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        for name in ("name", "kind", "locator", "version"):
            if not str(getattr(self, name)).strip():
                raise ExecutionProfileError(f"tool.{name} must be non-empty")
        _identity(self.identity, f"tool.{self.name}.identity")
        if not self.roles:
            raise ExecutionProfileError(f"tool.{self.name}.roles must be non-empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolIdentity":
        payload = _closed_mapping(
            value,
            name="tool",
            allowed={"name", "kind", "locator", "version", "identity", "roles", "required"},
            required={"name", "kind", "locator", "version", "identity", "roles", "required"},
        )
        if not isinstance(payload["required"], bool):
            raise ExecutionProfileError("tool.required must be a boolean")
        return cls(
            name=str(payload["name"]).strip(),
            kind=str(payload["kind"]).strip(),
            locator=str(payload["locator"]).strip(),
            version=str(payload["version"]).strip(),
            identity=_identity(payload["identity"], "tool.identity"),
            roles=_strings(payload["roles"], "tool.roles"),
            required=payload["required"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "locator": self.locator,
            "version": self.version,
            "identity": self.identity,
            "roles": list(self.roles),
            "required": self.required,
        }


@dataclass(frozen=True)
class LockIdentity:
    """Content identity for a dependency lock or reviewed dependency input."""

    path: str
    identity: str

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ExecutionProfileError("lock.path must be a normalized repository-relative path")
        _identity(self.identity, f"lock.{self.path}.identity")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LockIdentity":
        payload = _closed_mapping(
            value,
            name="lock",
            allowed={"path", "identity"},
            required={"path", "identity"},
        )
        return cls(
            path=str(payload["path"]),
            identity=_identity(payload["identity"], "lock.identity"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "identity": self.identity}


@dataclass(frozen=True)
class SandboxPolicy:
    """Declarative fail-closed process sandbox requirements."""

    network: str
    auto_install: str
    home_cache: str
    credentials: str
    read_roots: tuple[str, ...]
    write_roots: tuple[str, ...]
    environment_allowlist: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("network", "auto_install", "home_cache", "credentials"):
            if getattr(self, name) != "deny":
                raise ExecutionProfileError(f"sandbox.{name} must be 'deny'")
        if not self.read_roots or not self.write_roots:
            raise ExecutionProfileError("sandbox read_roots and write_roots must be non-empty")
        for name, roots in (("read_roots", self.read_roots), ("write_roots", self.write_roots)):
            for root in roots:
                path = Path(root)
                if path.is_absolute() or ".." in path.parts:
                    raise ExecutionProfileError(
                        f"sandbox.{name} entries must be repository-relative"
                    )
        if any(_is_credential_name(name) for name in self.environment_allowlist):
            raise ExecutionProfileError("sandbox environment allowlist contains a credential")
        if any(name.upper() in _HOME_CACHE_NAMES for name in self.environment_allowlist):
            raise ExecutionProfileError("sandbox environment allowlist contains a home cache")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SandboxPolicy":
        fields = {
            "network",
            "auto_install",
            "home_cache",
            "credentials",
            "read_roots",
            "write_roots",
            "environment_allowlist",
        }
        payload = _closed_mapping(
            value, name="sandbox", allowed=fields, required=fields
        )
        return cls(
            network=str(payload["network"]),
            auto_install=str(payload["auto_install"]),
            home_cache=str(payload["home_cache"]),
            credentials=str(payload["credentials"]),
            read_roots=_strings(payload["read_roots"], "sandbox.read_roots"),
            write_roots=_strings(payload["write_roots"], "sandbox.write_roots"),
            environment_allowlist=_strings(
                payload["environment_allowlist"], "sandbox.environment_allowlist"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "network": self.network,
            "auto_install": self.auto_install,
            "home_cache": self.home_cache,
            "credentials": self.credentials,
            "read_roots": list(self.read_roots),
            "write_roots": list(self.write_roots),
            "environment_allowlist": list(self.environment_allowlist),
        }


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Facts observed by a launcher without installing or mutating anything."""

    tool_identities: Mapping[str, str] = field(default_factory=dict)
    lock_identities: Mapping[str, str] = field(default_factory=dict)
    unavailable_tools: tuple[str, ...] = ()
    network_enabled: bool = False
    auto_install_enabled: bool = False
    home_cache_enabled: bool = False
    credential_names: tuple[str, ...] = ()
    environment_names: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tool_identities = dict(self.tool_identities)
        lock_identities = dict(self.lock_identities)
        for name, identity in tool_identities.items():
            if not str(name).strip():
                raise ExecutionProfileError("capability tool name must be non-empty")
            _identity(identity, f"capability tool {name}")
        for path, identity in lock_identities.items():
            if not str(path).strip():
                raise ExecutionProfileError("capability lock path must be non-empty")
            _identity(identity, f"capability lock {path}")
        object.__setattr__(self, "tool_identities", MappingProxyType(tool_identities))
        object.__setattr__(self, "lock_identities", MappingProxyType(lock_identities))
        for name in (
            "network_enabled",
            "auto_install_enabled",
            "home_cache_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ExecutionProfileError(f"capability {name} must be a boolean")
        for name in (
            "unavailable_tools",
            "credential_names",
            "environment_names",
            "read_paths",
            "write_paths",
        ):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))

    @classmethod
    def observe(
        cls,
        profile: "AnalysisExecutionProfile",
        *,
        repository_root: str | os.PathLike[str],
        environment: Mapping[str, str] | None = None,
        network_enabled: bool = False,
        auto_install_enabled: bool = False,
        home_cache_enabled: bool = False,
        read_paths: Sequence[str] = (),
        write_paths: Sequence[str] = (),
    ) -> "CapabilitySnapshot":
        """Hash present executables and locks without invoking or installing tools."""

        repository = _real_path(repository_root)
        tool_identities: dict[str, str] = {}
        unavailable: list[str] = []
        for tool in profile.tools:
            if tool.kind != "executable":
                # Non-executable capabilities require an attested launcher snapshot.
                unavailable.append(tool.name)
                continue
            executable = shutil.which(tool.locator)
            if executable is None:
                unavailable.append(tool.name)
                continue
            try:
                tool_identities[tool.name] = _file_identity(Path(executable))
            except OSError:
                unavailable.append(tool.name)

        lock_identities: dict[str, str] = {}
        for lock in profile.locks:
            candidate = repository / lock.path
            if candidate.is_file() and _is_within(_real_path(candidate), (repository,)):
                try:
                    lock_identities[lock.path] = _file_identity(candidate)
                except OSError:
                    continue

        env = environment or {}
        credential_names = tuple(sorted(name for name in env if _is_credential_name(name)))
        return cls(
            tool_identities=tool_identities,
            lock_identities=lock_identities,
            unavailable_tools=tuple(sorted(unavailable)),
            network_enabled=network_enabled,
            auto_install_enabled=auto_install_enabled,
            home_cache_enabled=home_cache_enabled,
            credential_names=credential_names,
            environment_names=tuple(sorted(env)),
            read_paths=tuple(read_paths),
            write_paths=tuple(write_paths),
        )


@dataclass(frozen=True)
class HermeticValidation:
    """Fail-closed validation result for policy, capabilities, or resource use."""

    safe: bool
    complete: bool
    disposition: str
    violations: tuple[str, ...] = ()
    unavailable_capabilities: tuple[str, ...] = ()
    exhausted_resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {"pass", "incomplete", "unknown", "rejected"}
        if self.disposition not in allowed:
            raise ExecutionProfileError("invalid hermetic validation disposition")
        if self.disposition == "pass" and (not self.safe or not self.complete):
            raise ExecutionProfileError("pass requires safe and complete validation")
        if (self.violations or not self.safe) and self.disposition != "rejected":
            raise ExecutionProfileError("unsafe validation must be rejected")
        if (self.unavailable_capabilities or self.exhausted_resources) and self.complete:
            raise ExecutionProfileError("missing capability or exhaustion cannot be complete")

    @property
    def ok(self) -> bool:
        return self.safe and self.complete and self.disposition == "pass"

    @property
    def valid(self) -> bool:
        """Whether execution is both safe and ready to produce evidence."""

        return self.ok


@dataclass(frozen=True)
class AnalysisExecutionProfile:
    """Reviewed toolchain, dependency, resource, and sandbox policy."""

    profile_id: str
    goal_id: str
    resource_class: str
    resource_bounds_evidence: str
    tools: tuple[ToolIdentity, ...]
    locks: tuple[LockIdentity, ...]
    resources: ResourceBudget
    sandbox: SandboxPolicy

    def __post_init__(self) -> None:
        for name in ("profile_id", "goal_id", "resource_class"):
            if not str(getattr(self, name)).strip():
                raise ExecutionProfileError(f"{name} must be non-empty")
        evidence = Path(self.resource_bounds_evidence)
        if (
            not self.resource_bounds_evidence.strip()
            or evidence == Path(".")
            or evidence.is_absolute()
            or ".." in evidence.parts
        ):
            raise ExecutionProfileError(
                "resource_bounds_evidence must be a repository-relative file"
            )
        tool_names = [tool.name for tool in self.tools]
        lock_paths = [lock.path for lock in self.locks]
        if not self.tools or len(tool_names) != len(set(tool_names)):
            raise ExecutionProfileError("tools must have unique names")
        if not self.locks or len(lock_paths) != len(set(lock_paths)):
            raise ExecutionProfileError("locks must have unique paths")
        required_roles = {
            "python",
            "node",
            "parser",
            "typescript",
            "solver",
            "proof",
        }
        present_roles = {role for tool in self.tools for role in tool.roles}
        missing_roles = sorted(required_roles - present_roles)
        if missing_roles:
            raise ExecutionProfileError(f"profile is missing tool roles: {missing_roles}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisExecutionProfile":
        fields = {
            "schema",
            "profile_id",
            "goal_id",
            "resource_class",
            "resource_bounds_evidence",
            "tools",
            "locks",
            "resource_bounds",
            "sandbox",
        }
        payload = _closed_mapping(
            value, name="profile", allowed=fields, required=fields
        )
        if payload["schema"] != PROFILE_SCHEMA:
            raise ExecutionProfileError("unsupported analyzer-profile schema")
        if not isinstance(payload["tools"], Sequence) or isinstance(
            payload["tools"], (str, bytes, bytearray)
        ):
            raise ExecutionProfileError("profile.tools must be an array")
        if not isinstance(payload["locks"], Sequence) or isinstance(
            payload["locks"], (str, bytes, bytearray)
        ):
            raise ExecutionProfileError("profile.locks must be an array")
        resource_class = str(payload["resource_class"]).strip()
        resources = ResourceBudget.from_dict(payload["resource_bounds"])
        if str(payload["resource_bounds"]["class"]) != resource_class:
            raise ExecutionProfileError(
                "resource_bounds.class must match profile.resource_class"
            )
        return cls(
            profile_id=str(payload["profile_id"]).strip(),
            goal_id=str(payload["goal_id"]).strip(),
            resource_class=resource_class,
            resource_bounds_evidence=str(payload["resource_bounds_evidence"]),
            tools=tuple(ToolIdentity.from_dict(item) for item in payload["tools"]),
            locks=tuple(LockIdentity.from_dict(item) for item in payload["locks"]),
            resources=resources,
            sandbox=SandboxPolicy.from_dict(payload["sandbox"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "AnalysisExecutionProfile":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ExecutionProfileError("profile is not valid JSON") from exc
        return cls.from_dict(payload)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        repository_root: str | os.PathLike[str] | None = None,
    ) -> "AnalysisExecutionProfile":
        profile = cls.from_json(Path(path).read_text(encoding="utf-8"))
        if repository_root is not None:
            profile.validate_resource_bounds_evidence(
                repository_root=repository_root
            )
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "goal_id": self.goal_id,
            "resource_class": self.resource_class,
            "resource_bounds_evidence": self.resource_bounds_evidence,
            "tools": [tool.to_dict() for tool in self.tools],
            "locks": [lock.to_dict() for lock in self.locks],
            "resource_bounds": self.resources.to_dict(
                resource_class=self.resource_class
            ),
            "sandbox": self.sandbox.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @property
    def content_identity(self) -> str:
        return _content_identity(self.to_dict())

    def validate_resource_bounds_evidence(
        self, *, repository_root: str | os.PathLike[str]
    ) -> ResourceBudget:
        """Load and verify the reviewed standalone resource policy.

        The analyzer profile embeds the limits used at runtime and names the
        objective evidence file.  Requiring the sidecar to decode to the exact
        same closed-schema object prevents either copy from drifting silently.
        """

        repository = _real_path(repository_root)
        evidence = _real_path(repository / self.resource_bounds_evidence)
        if not _is_within(evidence, (repository,)):
            raise ExecutionProfileError(
                "resource_bounds_evidence escapes the repository root"
            )
        try:
            payload = json.loads(evidence.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ExecutionProfileError(
                "resource_bounds_evidence is unavailable"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ExecutionProfileError(
                "resource_bounds_evidence is not valid JSON"
            ) from exc

        budget = ResourceBudget.from_dict(payload)
        if str(payload["class"]) != self.resource_class:
            raise ExecutionProfileError(
                "resource_bounds_evidence.class must match profile.resource_class"
            )
        expected = self.resources.to_dict(resource_class=self.resource_class)
        if payload != expected or budget != self.resources:
            raise ExecutionProfileError(
                "resource_bounds_evidence must exactly match profile.resource_bounds"
            )
        return budget

    def validate(
        self,
        snapshot: CapabilitySnapshot,
        *,
        repository_root: str | os.PathLike[str],
        proof_required: bool = False,
    ) -> HermeticValidation:
        """Validate runtime facts without attempting to repair the environment."""

        violations: list[str] = []
        unavailable: list[str] = []

        if snapshot.network_enabled:
            violations.append("network_enabled")
        if snapshot.auto_install_enabled:
            violations.append("auto_install_enabled")
        if snapshot.home_cache_enabled:
            violations.append("home_cache_enabled")
        if snapshot.credential_names:
            # Names are sufficient evidence; values must never enter diagnostics.
            violations.append("credentials_present")
        if any(_is_credential_name(name) for name in snapshot.environment_names):
            violations.append("credential_environment_present")
        if any(name.upper() in _HOME_CACHE_NAMES for name in snapshot.environment_names):
            violations.append("home_cache_environment_present")
        disallowed_environment = sorted(
            set(snapshot.environment_names) - set(self.sandbox.environment_allowlist)
        )
        if disallowed_environment:
            violations.append("ambient_environment_present")

        declared_unavailable = set(snapshot.unavailable_tools)
        for tool in self.tools:
            observed = snapshot.tool_identities.get(tool.name)
            if observed is None or tool.name in declared_unavailable:
                if tool.required:
                    unavailable.append(f"tool:{tool.name}")
            elif observed != tool.identity:
                violations.append(f"tool_identity_mismatch:{tool.name}")
        for lock in self.locks:
            observed = snapshot.lock_identities.get(lock.path)
            if observed is None:
                unavailable.append(f"lock:{lock.path}")
            elif observed != lock.identity:
                violations.append(f"lock_identity_mismatch:{lock.path}")

        repository = _real_path(repository_root)
        read_roots = tuple(_real_path(repository / root) for root in self.sandbox.read_roots)
        write_roots = tuple(
            _real_path(repository / root) for root in self.sandbox.write_roots
        )
        if any(not _is_within(root, (repository,)) for root in (*read_roots, *write_roots)):
            violations.append("sandbox_root_escape")
        for supplied in snapshot.read_paths:
            if not _is_within(_real_path(supplied), read_roots):
                violations.append("read_root_escape")
                break
        for supplied in snapshot.write_paths:
            if not _is_within(_real_path(supplied), write_roots):
                violations.append("write_root_escape")
                break

        violations_tuple = tuple(sorted(set(violations)))
        unavailable_tuple = tuple(sorted(set(unavailable)))
        if violations_tuple:
            disposition = "rejected"
        elif unavailable_tuple:
            disposition = "unknown" if proof_required else "incomplete"
        else:
            disposition = "pass"
        return HermeticValidation(
            safe=not violations_tuple,
            complete=not violations_tuple and not unavailable_tuple,
            disposition=disposition,
            violations=violations_tuple,
            unavailable_capabilities=unavailable_tuple,
        )

    def validate_usage(
        self, usage: Mapping[str, Any], *, proof_required: bool = False
    ) -> HermeticValidation:
        return self.resources.validate_usage(usage, proof_required=proof_required)


def load_execution_profile(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str] | None = None,
) -> AnalysisExecutionProfile:
    """Load and strictly validate a reviewed analyzer execution profile."""

    return AnalysisExecutionProfile.load(path, repository_root=repository_root)


__all__ = [
    "AnalysisExecutionProfile",
    "CapabilitySnapshot",
    "ExecutionProfileError",
    "HermeticValidation",
    "LockIdentity",
    "PROFILE_SCHEMA",
    "RESOURCE_BOUNDS_SCHEMA",
    "ResourceBudget",
    "SandboxPolicy",
    "ToolIdentity",
    "load_execution_profile",
]
