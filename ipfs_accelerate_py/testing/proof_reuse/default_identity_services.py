"""Lazy session-scoped default identity services for proof-backed test reuse.

This module supplies the production default for automatic item-identity
assembly when proof reuse is enabled (``read``, ``write``, or ``readwrite``).
It amortizes expensive stable inputs—repository-forest identity, per-root AST
indexes, distribution/lock inventories, environment facts, and policy
snapshots—once per session while still invalidating identities when dirty
overlays or source files change.

Import is intentionally inert:

* no optional providers (multiformats, datasets verifier, certificate store);
* no repository walks, network I/O, package installers, or cache writes;
* no pytest dependency at module import time.

Heavy agent-supervisor collectors are imported only after a non-off mode
requests service construction.  Explicit injected providers always win over
defaults.  Every incomplete, unavailable, or exceptional component surfaces as
typed non-reusable diagnostics rather than aborting pytest collection.
"""

from __future__ import annotations

import hashlib
import os
import platform
import struct
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Optional

from .config import ProofReuseMode
from .item_identity import (
    CurrentInputCompleteness,
    CurrentItemComponentInputs,
    CurrentItemPolicyInputs,
    ItemIdentityAssemblyServices,
)

DEFAULT_IDENTITY_SERVICE_FACTORY_INTERFACE: Final = (
    "DefaultIdentityServiceFactory@1"
)
PROOF_REUSE_SESSION_IDENTITY_INTERFACE: Final = "ProofReuseSessionIdentity@1"
ANALYSIS_AST_INDEX_PROVIDER_INTERFACE: Final = "AnalysisASTIndexProvider@1"
DEFAULT_ITEM_STATIC_IDENTITY_INTERFACE: Final = "DefaultItemStaticIdentity@1"
DEFAULT_ITEM_STATIC_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/default-item-static-identity@1"
)

_MAX_SOURCE_BYTES: Final = 8 * 1_048_576
_MAX_LOCK_BYTES: Final = 16 * 1_048_576
_MAX_FIXTURES: Final = 512
_MAX_PLUGINS: Final = 512
_MAX_PYTHON_FILES: Final = 8_192
_SAFE_VERSION_CHARS: Final = 128

_LOCK_FILE_NAMES: Final = frozenset(
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
_LOCK_FILE_SUFFIXES: Final = (".lock", ".lock.json")

_TRUE_IDENTITY_MODES: Final = frozenset(
    {
        ProofReuseMode.READ,
        ProofReuseMode.WRITE,
        ProofReuseMode.READWRITE,
        ProofReuseMode.SHADOW,
    }
)


class DefaultIdentityReason(str, Enum):
    """Closed reason codes for default static-identity construction."""

    ADMITTED = "admitted"
    MODE_OFF = "mode_off"
    ITEM_PATH_UNAVAILABLE = "item_path_unavailable"
    REPOSITORY_ROOT_UNAVAILABLE = "repository_root_unavailable"
    REPOSITORY_FOREST_UNAVAILABLE = "repository_forest_unavailable"
    REPOSITORY_FOREST_INCOMPLETE = "repository_forest_incomplete"
    AST_INDEX_UNAVAILABLE = "ast_index_unavailable"
    STATIC_TRACE_INCOMPLETE = "static_trace_incomplete"
    COMPONENT_INPUT_UNAVAILABLE = "component_input_unavailable"
    COMPONENTS_NON_REUSABLE = "components_non_reusable"
    LOCATOR_REJECTED = "locator_rejected"
    INTERNAL_ERROR_FAIL_OPEN = "internal_error_fail_open"


@dataclass(frozen=True, slots=True)
class DefaultItemStaticIdentity:
    """Forest, locator, and current static components for one collected item.

    This result never authorizes ``SKIP``.  Incomplete construction yields
    ``reusable=False`` with a closed reason code so pytest continues.
    """

    __test__: ClassVar[bool] = False

    reason: DefaultIdentityReason
    stage: str
    forest: Any = None
    forest_id: str = ""
    locator_artifact: Any = None
    component_inputs: Optional[CurrentItemComponentInputs] = None
    components: Any = None
    static_trace: Any = None
    descriptor: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def interface(self) -> str:
        return DEFAULT_ITEM_STATIC_IDENTITY_INTERFACE

    @property
    def reusable(self) -> bool:
        return (
            self.reason is DefaultIdentityReason.ADMITTED
            and self.forest is not None
            and self.locator_artifact is not None
            and getattr(self.locator_artifact, "reusable", False) is True
            and self.components is not None
            and self.static_trace is not None
        )

    @property
    def action(self) -> str:
        return "RUN"

    @property
    def authorizes_skip(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEFAULT_ITEM_STATIC_IDENTITY_SCHEMA,
            "interface": self.interface,
            "reason": self.reason.value,
            "stage": self.stage,
            "reusable": self.reusable,
            "action": self.action,
            "authorizes_skip": False,
            "forest_id": self.forest_id,
            "has_locator": self.locator_artifact is not None,
            "has_components": self.components is not None,
            "has_static_trace": self.static_trace is not None,
            "diagnostics": dict(self.diagnostics),
        }


def _failure(
    reason: DefaultIdentityReason,
    stage: str,
    **diagnostics: Any,
) -> DefaultItemStaticIdentity:
    bounded: dict[str, Any] = {}
    for key, value in list(diagnostics.items())[:16]:
        name = str(key)[:64]
        if value is None or isinstance(value, (bool, int)):
            bounded[name] = value
        elif isinstance(value, str):
            bounded[name] = value[:128]
        else:
            bounded[name] = type(value).__name__[:64]
    return DefaultItemStaticIdentity(
        reason=reason,
        stage=str(stage)[:64],
        diagnostics=MappingProxyType(bounded),
    )


def _parse_mode(value: Any) -> ProofReuseMode:
    if isinstance(value, ProofReuseMode):
        return value
    try:
        return ProofReuseMode.parse(value)
    except Exception:
        return ProofReuseMode.OFF


def _item_path(item: Any) -> Path:
    raw = getattr(item, "path", None)
    if raw is None:
        raw = getattr(item, "fspath", None)
    if raw is None:
        raise ValueError("collected item has no path")
    path = Path(os.fspath(raw)).resolve(strict=True)
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("collected item path is not a Python file")
    return path


def _discover_git_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        git_dir = candidate / ".git"
        try:
            if git_dir.exists():
                return candidate.resolve(strict=True)
        except OSError:
            continue
    raise ValueError("no git repository root found for item")


def _path_under(path: Path, root: Path) -> Optional[str]:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _is_lock_file(name: str) -> bool:
    lowered = name.lower()
    return lowered in _LOCK_FILE_NAMES or any(
        lowered.endswith(suffix) for suffix in _LOCK_FILE_SUFFIXES
    )


def _source_fingerprint(path: Path) -> tuple[str, int, int, str]:
    try:
        stat = path.stat()
        size = int(stat.st_size)
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
        if size > _MAX_SOURCE_BYTES:
            digest = "oversized"
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return (path.as_posix(), size, mtime_ns, digest)
    except OSError as exc:
        raise ValueError("source fingerprint unavailable") from exc


def _python_file_fingerprints(root: Path) -> tuple[tuple[str, int, int, str], ...]:
    records: list[tuple[str, int, int, str]] = []
    try:
        paths = sorted(root.rglob("*.py"))
    except OSError as exc:
        raise ValueError("repository walk failed") from exc
    count = 0
    for path in paths:
        if count >= _MAX_PYTHON_FILES:
            break
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        try:
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            size = int(stat.st_size)
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
            if size > _MAX_SOURCE_BYTES:
                digest = "oversized"
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append((relative, size, mtime_ns, digest))
            count += 1
        except OSError:
            continue
    return tuple(records)


def _interpreter_facts() -> dict[str, Any]:
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:5]),
        "cache_tag": sys.implementation.cache_tag or "",
        "abi_flags": getattr(sys, "abiflags", ""),
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
    }


def _platform_facts() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "system": platform.system().lower(),
        "release": platform.release(),
        "machine": platform.machine().lower(),
        "python_compiler": platform.python_compiler(),
        "libc": [libc_name, libc_version],
    }


def _hardware_facts() -> dict[str, Any]:
    return {
        "architecture": platform.machine().lower(),
        "cpu_count": os.cpu_count() or 0,
        "accelerator_backend": "none",
        "accelerator_count": 0,
        "accelerator_architectures": [],
    }


class AnalysisASTIndexProvider:
    """Session-memoized AST index provider with source-change invalidation.

    Implements ``AnalysisASTIndexProvider@1``.  Construction does not index a
    repository; the first call for a descriptor root builds the index, and
    later calls reuse it until the Python source fingerprint set changes.
    """

    __test__: ClassVar[bool] = False

    def __init__(self, session: "ProofReuseSessionIdentity") -> None:
        if not isinstance(session, ProofReuseSessionIdentity):
            raise TypeError("session must be ProofReuseSessionIdentity")
        self._session = session

    @property
    def interface(self) -> str:
        return ANALYSIS_AST_INDEX_PROVIDER_INTERFACE

    @property
    def build_count(self) -> int:
        return self._session.ast_index_build_count

    def provide(self, descriptor: Any) -> Any:
        """Return the current AST index for one repository descriptor root."""

        return self._session.ast_index_for(descriptor)

    def __call__(self, item: Any, descriptor: Any) -> Any:
        del item
        return self.provide(descriptor)


class ProofReuseSessionIdentity:
    """Session-scoped memoization of expensive identity inputs.

    Built only when a non-off mode requests identity services.  Dirty overlay
    digests and source fingerprints invalidate cached forest and AST indexes
    without requiring a process restart.  All mutators are thread-safe for
    concurrent collection workers that share one session object.
    """

    __test__: ClassVar[bool] = False

    def __init__(
        self,
        *,
        mode: ProofReuseMode,
        root_path: Path | None = None,
        config: Any = None,
        forest_roots: Sequence[Any] | None = None,
        sole_write_alias: str = "repo",
        identity_compiler: Any = None,
    ) -> None:
        self.mode = mode
        self.root_path = (
            Path(root_path).resolve() if root_path is not None else None
        )
        self.config = config
        self.forest_roots = tuple(forest_roots or ())
        self.sole_write_alias = str(sole_write_alias or "repo")
        self._identity_compiler = identity_compiler
        self._lock = threading.RLock()
        self._forest: Any = None
        self._forest_key: tuple[Any, ...] | None = None
        self._forest_roots_key: tuple[Any, ...] | None = None
        self._forest_build_count = 0
        self._ast_indexes: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self._ast_index_build_count = 0
        self._lock_files: tuple[Mapping[str, Any], ...] | None = None
        self._installed_distributions: tuple[tuple[str, str], ...] | None = None
        self._environment_snapshot: Mapping[str, str] | None = None
        self._environment_allowlist: tuple[str, ...] | None = None
        self._interpreter_facts: Mapping[str, Any] | None = None
        self._platform_facts: Mapping[str, Any] | None = None
        self._hardware_facts: Mapping[str, Any] | None = None
        self._policy_inputs: CurrentItemPolicyInputs | None = None
        self._policy_build_count = 0
        self._dependency_build_count = 0

    @property
    def interface(self) -> str:
        return PROOF_REUSE_SESSION_IDENTITY_INTERFACE

    @property
    def forest_build_count(self) -> int:
        return self._forest_build_count

    @property
    def ast_index_build_count(self) -> int:
        return self._ast_index_build_count

    @property
    def policy_build_count(self) -> int:
        return self._policy_build_count

    @property
    def dependency_build_count(self) -> int:
        return self._dependency_build_count

    def identity_compiler(self) -> Any:
        if self._identity_compiler is not None:
            return self._identity_compiler
        from ...agent_supervisor.analysis.test_execution_identity import (
            TestExecutionIdentityCompiler,
        )

        self._identity_compiler = TestExecutionIdentityCompiler()
        return self._identity_compiler

    def forest(self, *, seed_path: Path | None = None) -> Any:
        """Return the admitted repository forest, rebuilding on dirty change."""

        with self._lock:
            return self._forest_unlocked(seed_path=seed_path)

    def _forest_unlocked(self, *, seed_path: Path | None = None) -> Any:
        from ...agent_supervisor.repository_forest import (
            ForestPolicy,
            ForestRootSpec,
            RepositoryAuthority,
            build_repository_forest,
            descriptor_satisfies_repository_descriptor,
        )

        roots = self._resolve_forest_roots(seed_path=seed_path)
        roots_key = self._forest_cache_key(roots)
        if (
            self._forest is not None
            and self._forest_key is not None
            and self._forest_roots_key == roots_key
            and self._forest_still_valid(self._forest)
        ):
            return self._forest

        normalized: list[Any] = []
        for index, root in enumerate(roots):
            if isinstance(root, ForestRootSpec):
                normalized.append(root)
                continue
            if isinstance(root, Mapping):
                alias = str(root.get("alias") or f"repo{index}")
                path = root.get("root_path") or root.get("path")
                mode = str(root.get("mode") or "read_write")
            elif isinstance(root, (tuple, list)) and len(root) >= 2:
                alias = str(root[0])
                path = root[1]
                mode = "read_write"
            else:
                alias = self.sole_write_alias if index == 0 else f"repo{index}"
                path = root
                mode = "read_write"
            normalized.append(
                ForestRootSpec(
                    alias=alias,
                    root_path=path,
                    authority=RepositoryAuthority(mode=mode),
                )
            )
        if not normalized:
            raise ValueError("repository forest roots are empty")
        write_alias = normalized[0].alias
        policy = ForestPolicy(
            roots=tuple(normalized),
            sole_write_alias=write_alias,
        )

        forest = build_repository_forest(policy)
        if forest.reason_codes or not forest.descriptors:
            raise ValueError("repository forest is incomplete")
        if any(
            not descriptor_satisfies_repository_descriptor(descriptor)
            for descriptor in forest.descriptors
        ):
            raise ValueError("repository forest descriptors are not admitted")
        self._forest = forest
        self._forest_key = self._forest_identity_key(forest)
        self._forest_roots_key = roots_key
        self._forest_build_count += 1
        return forest

    def _resolve_forest_roots(
        self, *, seed_path: Path | None = None
    ) -> tuple[Any, ...]:
        from ...agent_supervisor.repository_forest import ForestRootSpec, RepositoryAuthority

        if self.forest_roots:
            return tuple(self.forest_roots)
        root: Path | None = self.root_path
        if root is None and self.config is not None:
            raw = getattr(self.config, "rootpath", None)
            if raw is not None:
                try:
                    root = Path(os.fspath(raw)).resolve(strict=True)
                except (OSError, TypeError, ValueError):
                    root = None
        if root is None and seed_path is not None:
            root = _discover_git_root(seed_path)
        if root is None:
            raise ValueError("repository root is unavailable")
        # Prefer the git root when the configured rootpath is a subdirectory.
        try:
            git_root = _discover_git_root(root)
        except ValueError:
            git_root = root
        return (
            ForestRootSpec(
                alias=self.sole_write_alias,
                root_path=git_root,
                authority=RepositoryAuthority(mode="read_write"),
            ),
        )

    def _forest_cache_key(self, roots: Sequence[Any]) -> tuple[Any, ...]:
        entries: list[tuple[str, str]] = []
        for root in roots:
            if hasattr(root, "alias") and hasattr(root, "root_path"):
                alias = str(root.alias)
                path = str(Path(root.root_path).resolve())
            elif isinstance(root, Mapping):
                alias = str(root.get("alias") or "")
                path = str(Path(str(root.get("root_path") or root.get("path") or "")).resolve())
            elif isinstance(root, (tuple, list)) and len(root) >= 2:
                alias = str(root[0])
                path = str(Path(root[1]).resolve())
            else:
                alias = self.sole_write_alias
                path = str(Path(root).resolve())
            entries.append((alias, path))
        return tuple(sorted(entries))

    def _forest_identity_key(self, forest: Any) -> tuple[Any, ...]:
        return tuple(
            sorted(
                (
                    str(descriptor.alias),
                    str(descriptor.commit),
                    str(descriptor.tree),
                    str(descriptor.dirty_overlay_digest),
                    bool(descriptor.dirty),
                )
                for descriptor in forest.descriptors
            )
        )

    def _forest_still_valid(self, forest: Any) -> bool:
        try:
            from ...agent_supervisor.repository_forest import (
                compute_dirty_overlay_digest,
            )

            for descriptor in forest.descriptors:
                dirty, overlay, _reasons = compute_dirty_overlay_digest(
                    descriptor.root_path,
                    ignore_policy=descriptor.ignore_policy,
                )
                if (
                    bool(dirty) != bool(descriptor.dirty)
                    or str(overlay) != str(descriptor.dirty_overlay_digest)
                ):
                    return False
            return self._forest_identity_key(forest) == self._forest_key
        except Exception:
            return False

    def invalidate_forest(self) -> None:
        with self._lock:
            self._forest = None
            self._forest_key = None
            self._forest_roots_key = None

    def ast_index_for(self, descriptor: Any) -> Any:
        """Return a memoized AST index, rebuilding when sources change."""

        with self._lock:
            root = Path(descriptor.root_path).resolve(strict=True)
            cache_key = root.as_posix()
            fingerprints = _python_file_fingerprints(root)
            cached = self._ast_indexes.get(cache_key)
            if cached is not None and cached[0] == fingerprints:
                return cached[1]
            index = self._build_ast_index(root, fingerprints)
            self._ast_indexes[cache_key] = (fingerprints, index)
            self._ast_index_build_count += 1
            return index

    def _build_ast_index(
        self,
        root: Path,
        fingerprints: tuple[tuple[str, int, int, str], ...],
    ) -> Any:
        from ...agent_supervisor.analysis.analysis_ast_index import (
            build_analysis_ast_index,
        )
        from ...agent_supervisor.core.conflict_graph import (
            build_python_ast_blob_record,
        )

        records: list[tuple[str, Any]] = []
        for relative, size, _mtime_ns, digest in fingerprints:
            if digest == "oversized" or size > _MAX_SOURCE_BYTES:
                continue
            path = root / relative
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            records.append(
                (
                    relative,
                    build_python_ast_blob_record(
                        source,
                        source_sha256=f"sha256:{digest}",
                    ),
                )
            )
        return build_analysis_ast_index(records)

    def invalidate_ast_indexes(self) -> None:
        with self._lock:
            self._ast_indexes.clear()

    def lock_files_for(self, root: Path) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            if self._lock_files is not None:
                return self._lock_files
            collected: list[dict[str, Any]] = []
            try:
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    if ".git" in path.parts:
                        continue
                    if not _is_lock_file(path.name):
                        continue
                    try:
                        if path.stat().st_size > _MAX_LOCK_BYTES:
                            continue
                        content = path.read_bytes()
                        relative = path.relative_to(root).as_posix()
                    except OSError:
                        continue
                    collected.append({"path": relative, "content": content})
                    if len(collected) >= _MAX_FIXTURES:
                        break
            except OSError:
                collected = []
            self._lock_files = tuple(collected)
            self._dependency_build_count += 1
            return self._lock_files

    def installed_distributions(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            if self._installed_distributions is not None:
                return self._installed_distributions
            distributions: list[tuple[str, str]] = []
            try:
                from importlib import metadata as importlib_metadata

                for dist in importlib_metadata.distributions():
                    try:
                        name = dist.metadata["Name"]
                        version = dist.version
                    except Exception:
                        continue
                    if not name or not version:
                        continue
                    distributions.append((str(name), str(version)[:_SAFE_VERSION_CHARS]))
                    if len(distributions) >= _MAX_PLUGINS:
                        break
            except Exception:
                distributions = []
            if not any(
                str(name).strip().lower().replace("_", "-") == "pytest"
                for name, _version in distributions
            ):
                try:
                    import pytest as pytest_module

                    distributions.append(
                        ("pytest", str(pytest_module.__version__)[:_SAFE_VERSION_CHARS])
                    )
                except Exception:
                    distributions.append(("pytest", "0"))
            # Stable ordering.
            normalized: dict[str, str] = {}
            for name, version in distributions:
                key = str(name).strip().lower().replace("_", "-")
                normalized[key] = version
            self._installed_distributions = tuple(
                sorted(normalized.items(), key=lambda pair: pair[0])
            )
            self._dependency_build_count += 1
            return self._installed_distributions

    def environment_snapshot(self) -> tuple[Mapping[str, str], tuple[str, ...]]:
        with self._lock:
            if (
                self._environment_snapshot is not None
                and self._environment_allowlist is not None
            ):
                return self._environment_snapshot, self._environment_allowlist
            from ...agent_supervisor.analysis.test_identity_components import (
                DEFAULT_ENVIRONMENT_ALLOWLIST,
            )

            allowlist = tuple(DEFAULT_ENVIRONMENT_ALLOWLIST)
            snapshot = {
                name: str(os.environ[name])
                for name in allowlist
                if name in os.environ
            }
            self._environment_snapshot = MappingProxyType(snapshot)
            self._environment_allowlist = allowlist
            return self._environment_snapshot, self._environment_allowlist

    def runtime_facts(
        self,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        with self._lock:
            if self._interpreter_facts is None:
                self._interpreter_facts = MappingProxyType(_interpreter_facts())
            if self._platform_facts is None:
                self._platform_facts = MappingProxyType(_platform_facts())
            if self._hardware_facts is None:
                self._hardware_facts = MappingProxyType(_hardware_facts())
            return (
                self._interpreter_facts,
                self._platform_facts,
                self._hardware_facts,
            )

    def policy_inputs(self) -> CurrentItemPolicyInputs:
        with self._lock:
            if self._policy_inputs is not None:
                return self._policy_inputs
            from ...agent_supervisor.analysis.test_execution_identity import (
                mint_content_identity,
            )
            from ...agent_supervisor.analysis.test_reuse_eligibility import (
                TestReuseEligibilityPolicy,
            )

            def _cid(label: str) -> str:
                return mint_content_identity(
                    {
                        "schema": (
                            "ipfs_accelerate_py/testing/proof-reuse/"
                            "default-identity-policy@1"
                        ),
                        "label": label,
                    }
                ).cid

            policy_identity = mint_content_identity(
                {
                    "schema": (
                        "ipfs_accelerate_py/testing/proof-reuse/"
                        "default-reuse-policy@1"
                    ),
                    "revision": 1,
                }
            )
            try:
                import pytest as pytest_module

                pytest_version = str(pytest_module.__version__)[:_SAFE_VERSION_CHARS]
            except Exception:
                pytest_version = "0"
            self._policy_inputs = CurrentItemPolicyInputs(
                completeness=CurrentInputCompleteness.EXACT_CURRENT,
                policy_identity=policy_identity,
                verification_policy={
                    "policy_cid": policy_identity.cid,
                    "statement_cid": _cid("statement"),
                    "circuit_cid": _cid("circuit"),
                    "verifying_key_cid": _cid("verifying-key"),
                    "proof_system_id": "groth16",
                    "trusted_issuer_ids": ("issuer:default",),
                    "allowed_epochs": ("epoch:1",),
                },
                reuse_policy=TestReuseEligibilityPolicy(),
                command_semantics={
                    "schema": "ipfs_accelerate_py/testing/pytest-command@1",
                    "selection": "exact-node",
                },
                pytest_config={
                    "schema": "ipfs_accelerate_py/testing/pytest-config@1",
                    "root": "repository",
                },
                plugin_versions={
                    "schema": "ipfs_accelerate_py/testing/pytest-plugins@1",
                    "pytest": pytest_version,
                },
                runtime_completeness_policy={
                    "schema": (
                        "ipfs_accelerate_py/testing/runtime-completeness@1"
                    ),
                    "require_complete": True,
                },
                canonicalization_schema={
                    "schema": (
                        "ipfs_accelerate_py/testing/canonicalization@1"
                    ),
                    "profile": "dag-json",
                },
                tracer_schema={
                    "schema": "ipfs_accelerate_py/testing/tracer-schema@1",
                    "static": 1,
                    "runtime": 1,
                },
                certificate_schema={
                    "schema": (
                        "ipfs_accelerate_py/testing/certificate-schema@1"
                    ),
                    "version": 1,
                },
            )
            self._policy_build_count += 1
            return self._policy_inputs


class DefaultIdentityServiceFactory:
    """Build session-scoped default :class:`ItemIdentityAssemblyServices`.

    Implements ``DefaultIdentityServiceFactory@1``.  Explicit provider
    callables supplied to the constructor always override the session defaults.
    Off mode returns an empty service bundle without loading optional providers
    or repository collectors.
    """

    __test__: ClassVar[bool] = False

    def __init__(
        self,
        *,
        mode: Any = ProofReuseMode.OFF,
        root_path: str | Path | None = None,
        config: Any = None,
        forest_roots: Sequence[Any] | None = None,
        sole_write_alias: str = "repo",
        session_identity: ProofReuseSessionIdentity | None = None,
        repository_forest_provider: Optional[Callable[[Any], Any]] = None,
        analysis_index_provider: Optional[Callable[[Any, Any], Any]] = None,
        component_inputs_provider: Optional[
            Callable[[Any, Any, Any, Any], Any]
        ] = None,
        policy_inputs_provider: Optional[
            Callable[[Any, Any, Any, Any, Any], Any]
        ] = None,
        runtime_evidence_provider: Optional[
            Callable[[Any, Any, Any, Any, Any, Any], Any]
        ] = None,
        identity_compiler: Any = None,
    ) -> None:
        self.mode = _parse_mode(mode)
        self.root_path = (
            Path(root_path).resolve() if root_path is not None else None
        )
        self.config = config
        self.forest_roots = tuple(forest_roots or ())
        self.sole_write_alias = str(sole_write_alias or "repo")
        self._session = session_identity
        self._override_forest = repository_forest_provider
        self._override_index = analysis_index_provider
        self._override_components = component_inputs_provider
        self._override_policy = policy_inputs_provider
        self._override_runtime = runtime_evidence_provider
        self._override_compiler = identity_compiler
        self._services: ItemIdentityAssemblyServices | None = None
        self._ast_provider: AnalysisASTIndexProvider | None = None

    @property
    def interface(self) -> str:
        return DEFAULT_IDENTITY_SERVICE_FACTORY_INTERFACE

    @property
    def enabled(self) -> bool:
        return self.mode in _TRUE_IDENTITY_MODES

    def session_identity(self) -> ProofReuseSessionIdentity:
        """Return (and lazily create) the memoized session identity object."""

        if self._session is not None:
            return self._session
        if not self.enabled:
            # Off mode still exposes an inert session object for diagnostics,
            # but never builds expensive inputs until an enabled factory is used.
            self._session = ProofReuseSessionIdentity(
                mode=self.mode,
                root_path=self.root_path,
                config=self.config,
                forest_roots=self.forest_roots,
                sole_write_alias=self.sole_write_alias,
                identity_compiler=self._override_compiler,
            )
            return self._session
        self._session = ProofReuseSessionIdentity(
            mode=self.mode,
            root_path=self.root_path,
            config=self.config,
            forest_roots=self.forest_roots,
            sole_write_alias=self.sole_write_alias,
            identity_compiler=self._override_compiler,
        )
        return self._session

    def analysis_ast_index_provider(self) -> AnalysisASTIndexProvider:
        if self._ast_provider is None:
            self._ast_provider = AnalysisASTIndexProvider(self.session_identity())
        return self._ast_provider

    def build_services(self) -> ItemIdentityAssemblyServices:
        """Return the session-scoped DI bundle for item-identity assembly.

        Off mode returns empty providers so assembly fails open to ``RUN``
        without importing optional collectors.  Explicit overrides always win.
        """

        if self._services is not None:
            return self._services
        if not self.enabled:
            self._services = ItemIdentityAssemblyServices(
                repository_forest_provider=self._override_forest,
                analysis_index_provider=self._override_index,
                component_inputs_provider=self._override_components,
                policy_inputs_provider=self._override_policy,
                runtime_evidence_provider=self._override_runtime,
                identity_compiler=self._override_compiler,
            )
            return self._services

        session = self.session_identity()
        ast_provider = self.analysis_ast_index_provider()

        def forest_provider(item: Any) -> Any:
            if self._override_forest is not None:
                return self._override_forest(item)
            try:
                path = _item_path(item)
            except Exception:
                path = None
            return session.forest(seed_path=path)

        def index_provider(item: Any, descriptor: Any) -> Any:
            if self._override_index is not None:
                return self._override_index(item, descriptor)
            return ast_provider(item, descriptor)

        def component_provider(
            item: Any, facts: Any, descriptor: Any, static_trace: Any
        ) -> Any:
            if self._override_components is not None:
                return self._override_components(
                    item, facts, descriptor, static_trace
                )
            return self._collect_component_inputs(
                item, facts, descriptor, static_trace
            )

        def policy_provider(
            item: Any,
            facts: Any,
            descriptor: Any,
            static_trace: Any,
            components: Any,
        ) -> Any:
            if self._override_policy is not None:
                return self._override_policy(
                    item, facts, descriptor, static_trace, components
                )
            return session.policy_inputs()

        def runtime_provider(
            item: Any,
            facts: Any,
            descriptor: Any,
            static_trace: Any,
            components: Any,
            policy_inputs: Any,
        ) -> Any:
            if self._override_runtime is not None:
                return self._override_runtime(
                    item,
                    facts,
                    descriptor,
                    static_trace,
                    components,
                    policy_inputs,
                )
            # Warm runtime evidence requires a controlled preflight that is
            # out of scope for default static identity.  Absence fails open to
            # RUN in the assembler without aborting collection.
            raise LookupError("default runtime evidence is not available")

        self._services = ItemIdentityAssemblyServices(
            repository_forest_provider=forest_provider,
            analysis_index_provider=index_provider,
            component_inputs_provider=component_provider,
            policy_inputs_provider=policy_provider,
            runtime_evidence_provider=runtime_provider,
            identity_compiler=(
                self._override_compiler
                if self._override_compiler is not None
                else session.identity_compiler()
            ),
        )
        return self._services

    def obtain_static_identity(self, item: Any) -> DefaultItemStaticIdentity:
        """Obtain forest, locator, and current static components for one item.

        Does not require conftest service attributes or a per-test registry.
        Failures return typed non-reusable results.
        """

        if not self.enabled:
            return _failure(DefaultIdentityReason.MODE_OFF, "mode")
        try:
            return self._obtain_static_identity(item)
        except BaseException as exc:
            return _failure(
                DefaultIdentityReason.INTERNAL_ERROR_FAIL_OPEN,
                "factory",
                exception_type=type(exc).__name__,
            )

    def _obtain_static_identity(self, item: Any) -> DefaultItemStaticIdentity:
        try:
            path = _item_path(item)
        except (OSError, TypeError, ValueError) as exc:
            return _failure(
                DefaultIdentityReason.ITEM_PATH_UNAVAILABLE,
                "item",
                exception_type=type(exc).__name__,
            )

        session = self.session_identity()
        try:
            if self._override_forest is not None:
                forest = self._override_forest(item)
            else:
                forest = session.forest(seed_path=path)
        except BaseException as exc:
            return _failure(
                DefaultIdentityReason.REPOSITORY_FOREST_UNAVAILABLE,
                "repository_forest",
                exception_type=type(exc).__name__,
            )

        from ...agent_supervisor.repository_forest import (
            RepositoryForest,
            descriptor_satisfies_repository_descriptor,
        )

        if not isinstance(forest, RepositoryForest):
            return _failure(
                DefaultIdentityReason.REPOSITORY_FOREST_UNAVAILABLE,
                "repository_forest",
            )
        if forest.reason_codes or not forest.descriptors or any(
            not descriptor_satisfies_repository_descriptor(descriptor)
            for descriptor in forest.descriptors
        ):
            return _failure(
                DefaultIdentityReason.REPOSITORY_FOREST_INCOMPLETE,
                "repository_forest",
                reason_count=len(getattr(forest, "reason_codes", ()) or ()),
            )

        # Local imports keep collection helpers out of the cold import surface.
        from .item_identity import (
            _item_facts,
            _select_descriptor,
        )

        try:
            descriptor = _select_descriptor(forest, path)
            facts = _item_facts(item, descriptor)
        except (OSError, TypeError, ValueError) as exc:
            return _failure(
                DefaultIdentityReason.ITEM_PATH_UNAVAILABLE,
                "item",
                exception_type=type(exc).__name__,
            )

        try:
            if self._override_index is not None:
                index = self._override_index(item, descriptor)
            else:
                index = self.analysis_ast_index_provider().provide(descriptor)
        except BaseException as exc:
            return _failure(
                DefaultIdentityReason.AST_INDEX_UNAVAILABLE,
                "ast_index",
                exception_type=type(exc).__name__,
            )

        from ...agent_supervisor.analysis.analysis_ast_index import AnalysisASTIndex
        from ...agent_supervisor.analysis.test_static_dependency_trace import (
            StaticTestDependencyTracer,
        )

        if not isinstance(index, AnalysisASTIndex):
            return _failure(
                DefaultIdentityReason.AST_INDEX_UNAVAILABLE, "ast_index"
            )
        try:
            static_trace = StaticTestDependencyTracer(
                index, descriptor.root_path
            ).trace(
                facts.relative_path,
                test_symbol=facts.function_name,
                node_id=facts.node_id,
            )
            static_trace.verify()
        except BaseException as exc:
            return _failure(
                DefaultIdentityReason.STATIC_TRACE_INCOMPLETE,
                "static_trace",
                exception_type=type(exc).__name__,
            )

        try:
            if self._override_components is not None:
                component_inputs = self._override_components(
                    item, facts, descriptor, static_trace
                )
            else:
                component_inputs = self._collect_component_inputs(
                    item, facts, descriptor, static_trace
                )
        except BaseException as exc:
            return _failure(
                DefaultIdentityReason.COMPONENT_INPUT_UNAVAILABLE,
                "components",
                exception_type=type(exc).__name__,
            )
        if not isinstance(component_inputs, CurrentItemComponentInputs):
            return _failure(
                DefaultIdentityReason.COMPONENT_INPUT_UNAVAILABLE,
                "components",
            )

        from ...agent_supervisor.analysis.test_identity_components import (
            TestIdentityComponents,
        )

        try:
            values = {
                "parameter_id": facts.parameter_id,
                "fixtures": component_inputs.fixtures,
                "conftests": component_inputs.conftests,
                "hooks": component_inputs.hooks,
                "plugins": component_inputs.plugins,
                "lock_files": component_inputs.lock_files,
                "installed_distributions": (
                    component_inputs.installed_distributions
                ),
                "environment": component_inputs.environment,
                "environment_allowlist": component_inputs.environment_allowlist,
                "interpreter_facts": component_inputs.interpreter_facts,
                "platform_facts": component_inputs.platform_facts,
                "hardware_facts": component_inputs.hardware_facts,
                "capability_facts": component_inputs.capability_facts,
                "capability_allowlist": component_inputs.capability_allowlist,
            }
            if facts.parameterized:
                values["parameter_value"] = facts.parameter_value
            components = TestIdentityComponents.compile(**values)
        except BaseException as exc:
            return _failure(
                DefaultIdentityReason.COMPONENT_INPUT_UNAVAILABLE,
                "components",
                exception_type=type(exc).__name__,
            )
        if not components.reusable:
            # Still return the partial artifacts for diagnostics; locator may
            # still be compiled for stable node addressing.  Parameterized
            # nodes bind the exact parameter-value CID even when other
            # components are non-reusable.
            locator_partial = self._compile_locator(
                facts, descriptor, components=components
            )
            return DefaultItemStaticIdentity(
                reason=DefaultIdentityReason.COMPONENTS_NON_REUSABLE,
                stage="components",
                forest=forest,
                forest_id=str(getattr(forest, "forest_id", "") or ""),
                locator_artifact=locator_partial,
                component_inputs=component_inputs,
                components=components,
                static_trace=static_trace,
                descriptor=descriptor,
                diagnostics={
                    "reason_count": len(components.non_reusable_reasons),
                },
            )

        locator_artifact = self._compile_locator(
            facts, descriptor, components=components
        )
        if (
            locator_artifact is None
            or not getattr(locator_artifact, "reusable", False)
            or getattr(locator_artifact, "locator", None) is None
        ):
            return DefaultItemStaticIdentity(
                reason=DefaultIdentityReason.LOCATOR_REJECTED,
                stage="locator",
                forest=forest,
                forest_id=str(getattr(forest, "forest_id", "") or ""),
                locator_artifact=locator_artifact,
                component_inputs=component_inputs,
                components=components,
                static_trace=static_trace,
                descriptor=descriptor,
                diagnostics={
                    "compiler_reason": str(
                        getattr(locator_artifact, "reason_code", "") or ""
                    )[:128],
                },
            )

        return DefaultItemStaticIdentity(
            reason=DefaultIdentityReason.ADMITTED,
            stage="complete",
            forest=forest,
            forest_id=str(getattr(forest, "forest_id", "") or ""),
            locator_artifact=locator_artifact,
            component_inputs=component_inputs,
            components=components,
            static_trace=static_trace,
            descriptor=descriptor,
            diagnostics={},
        )

    def _compile_locator(
        self,
        facts: Any,
        descriptor: Any,
        *,
        components: Any = None,
    ) -> Any:
        """Compile a stable locator, binding the exact parameter-value CID.

        Parameterized nodes require ``parameter_values_cid`` (or an explicit
        non-reusable reason) on :class:`TestLocatorKey`.  The CID is taken
        from the already-compiled identity components so collection seeds and
        later warm lookup share the same parameter binding.
        """

        compiler = (
            self._override_compiler
            if self._override_compiler is not None
            else self.session_identity().identity_compiler()
        )
        from ...agent_supervisor.analysis.test_execution_identity import (
            TestExecutionIdentityCompiler,
        )

        if compiler is None:
            compiler = TestExecutionIdentityCompiler()
        if not isinstance(compiler, TestExecutionIdentityCompiler):
            return None
        policy = self.session_identity().policy_inputs()

        parameter_values_cid = ""
        non_reusable_reason = ""
        if getattr(facts, "parameterized", False):
            if components is not None:
                parameter_values_cid = str(
                    getattr(components, "parameter_cid", "") or ""
                )
            if not parameter_values_cid:
                reasons = tuple(
                    getattr(components, "non_reusable_reasons", ()) or ()
                )
                if reasons:
                    non_reusable_reason = str(reasons[0])[:256]
                else:
                    non_reusable_reason = "parameter_values_cid_unavailable"

        return compiler.compile_locator(
            repository_id=descriptor.repository_id,
            package_identity=descriptor.descriptor_cid,
            root_identity=descriptor.descriptor_cid,
            node_id=facts.node_id,
            collection_schema_version=policy.collection_schema_version,
            parameter_id=facts.parameter_id,
            parameter_values_cid=parameter_values_cid,
            non_reusable_reason=non_reusable_reason,
            selection_semantics="exact_node",
            metadata={
                "factory_interface": DEFAULT_IDENTITY_SERVICE_FACTORY_INTERFACE,
                "repository_alias": descriptor.alias,
            },
        )

    def _collect_component_inputs(
        self,
        item: Any,
        facts: Any,
        descriptor: Any,
        static_trace: Any,
    ) -> CurrentItemComponentInputs:
        del static_trace
        session = self.session_identity()
        root = Path(descriptor.root_path).resolve(strict=True)

        fixtures = self._fixture_records(item, facts, root)
        conftests = self._conftest_records(facts, root)
        plugins = self._plugin_records(item)
        hooks: tuple[Mapping[str, Any], ...] = ()
        lock_files = session.lock_files_for(root)
        installed = session.installed_distributions()
        environment, allowlist = session.environment_snapshot()
        interpreter, platform_facts, hardware = session.runtime_facts()

        return CurrentItemComponentInputs(
            completeness=CurrentInputCompleteness.EXACT_CURRENT,
            fixtures=fixtures,
            conftests=conftests,
            hooks=hooks,
            plugins=plugins,
            lock_files=lock_files,
            installed_distributions=installed,
            environment=dict(environment),
            environment_allowlist=allowlist,
            interpreter_facts=dict(interpreter),
            platform_facts=dict(platform_facts),
            hardware_facts=dict(hardware),
            capability_facts={},
            capability_allowlist=(),
        )

    def _fixture_records(
        self,
        item: Any,
        facts: Any,
        root: Path,
    ) -> tuple[Mapping[str, Any], ...]:
        del root
        records: list[dict[str, Any]] = []
        names = tuple(getattr(facts, "fixture_names", ()) or ())
        if len(names) > _MAX_FIXTURES:
            raise ValueError("fixture inventory exceeds bound")
        for name in names:
            definition = self._fixture_definition(item, name)
            records.append(
                {
                    "name": str(name),
                    "scope": definition.get("scope", "function"),
                    "definition": definition.get(
                        "definition",
                        f"def {name}():\n    raise NotImplementedError\n",
                    ),
                    "dependencies": definition.get("dependencies", ()),
                    "autouse": bool(definition.get("autouse", False)),
                }
            )
        return tuple(records)

    def _fixture_definition(self, item: Any, name: str) -> dict[str, Any]:
        """Best-effort fixture definition extraction; never raises."""

        try:
            session = getattr(item, "session", None)
            fixturemanager = getattr(session, "_fixturemanager", None)
            if fixturemanager is None:
                return {}
            get_defs = getattr(fixturemanager, "getfixturedefs", None)
            if not callable(get_defs):
                return {}
            defs = get_defs(name, getattr(item, "nodeid", "")) or ()
            if not defs:
                return {}
            fixture_def = defs[-1]
            scope = str(getattr(fixture_def, "scope", "function") or "function")
            func = getattr(fixture_def, "func", None)
            definition = ""
            if func is not None:
                try:
                    import inspect

                    definition = inspect.getsource(func)
                except Exception:
                    definition = f"def {name}():\n    pass\n"
            argnames = tuple(getattr(fixture_def, "argnames", ()) or ())
            autouse = bool(getattr(fixture_def, "autouse", False))
            return {
                "scope": scope if scope in {
                    "function", "class", "module", "package", "session"
                } else "function",
                "definition": definition or f"def {name}():\n    pass\n",
                "dependencies": argnames,
                "autouse": autouse,
            }
        except Exception:
            return {}

    def _conftest_records(
        self, facts: Any, root: Path
    ) -> tuple[Mapping[str, Any], ...]:
        records: list[dict[str, Any]] = []
        test_path = Path(facts.path)
        current = test_path.parent
        while True:
            candidate = current / "conftest.py"
            try:
                if candidate.is_file():
                    resolved = candidate.resolve(strict=True)
                    relative = _path_under(resolved, root)
                    if relative is None:
                        raise ValueError("conftest escapes repository root")
                    content = resolved.read_text(encoding="utf-8")
                    if len(content.encode("utf-8")) <= _MAX_SOURCE_BYTES:
                        records.append({"path": relative, "content": content})
            except (OSError, UnicodeError, ValueError):
                pass
            if current == root:
                break
            if current.parent == current or _path_under(current.parent, root) is None:
                break
            current = current.parent
        return tuple(sorted(records, key=lambda item: str(item["path"])))

    def _plugin_records(self, item: Any) -> tuple[Mapping[str, Any], ...]:
        from ...agent_supervisor.analysis.test_execution_identity import (
            mint_content_identity,
        )

        records: list[dict[str, Any]] = []
        try:
            import pytest as pytest_module

            pytest_version = str(pytest_module.__version__)[:_SAFE_VERSION_CHARS]
        except Exception:
            pytest_version = "0"

        implementation = mint_content_identity(
            {
                "schema": "ipfs_accelerate_py/testing/pytest-plugin@1",
                "name": "pytest",
                "version": pytest_version,
            }
        )
        records.append(
            {
                "name": "pytest",
                "implementation_cid": implementation.cid,
                "distribution": "pytest",
                "version": pytest_version,
                "registered": True,
                "order": 0,
            }
        )

        session = getattr(item, "session", None)
        config = getattr(session, "config", None) if session is not None else None
        if config is None:
            config = self.config
        pluginmanager = getattr(config, "pluginmanager", None) if config else None
        if pluginmanager is not None:
            try:
                names = list(pluginmanager.list_name_plugin())
            except Exception:
                names = []
            order = 1
            for plugin_name, plugin in names:
                if plugin is None:
                    continue
                name = str(plugin_name or "")[:128]
                if not name or name == "pytest":
                    continue
                try:
                    plugin_impl = mint_content_identity(
                        {
                            "schema": (
                                "ipfs_accelerate_py/testing/pytest-plugin@1"
                            ),
                            "name": name,
                        }
                    )
                    records.append(
                        {
                            "name": name,
                            "implementation_cid": plugin_impl.cid,
                            "distribution": name,
                            "version": "0",
                            "registered": True,
                            "order": order,
                        }
                    )
                    order += 1
                    if order >= _MAX_PLUGINS:
                        break
                except Exception:
                    continue
        return tuple(records)


def build_default_identity_services(
    *,
    mode: Any = ProofReuseMode.OFF,
    root_path: str | Path | None = None,
    config: Any = None,
    forest_roots: Sequence[Any] | None = None,
    sole_write_alias: str = "repo",
    session_identity: ProofReuseSessionIdentity | None = None,
    repository_forest_provider: Optional[Callable[[Any], Any]] = None,
    analysis_index_provider: Optional[Callable[[Any, Any], Any]] = None,
    component_inputs_provider: Optional[
        Callable[[Any, Any, Any, Any], Any]
    ] = None,
    policy_inputs_provider: Optional[
        Callable[[Any, Any, Any, Any, Any], Any]
    ] = None,
    runtime_evidence_provider: Optional[
        Callable[[Any, Any, Any, Any, Any, Any], Any]
    ] = None,
    identity_compiler: Any = None,
) -> ItemIdentityAssemblyServices:
    """Construct default identity services for the requested mode.

    Off mode returns an empty (or override-only) bundle without loading
    optional providers.  Enabled modes compose session-memoized collectors.
    Explicit provider arguments always override session defaults.
    """

    factory = DefaultIdentityServiceFactory(
        mode=mode,
        root_path=root_path,
        config=config,
        forest_roots=forest_roots,
        sole_write_alias=sole_write_alias,
        session_identity=session_identity,
        repository_forest_provider=repository_forest_provider,
        analysis_index_provider=analysis_index_provider,
        component_inputs_provider=component_inputs_provider,
        policy_inputs_provider=policy_inputs_provider,
        runtime_evidence_provider=runtime_evidence_provider,
        identity_compiler=identity_compiler,
    )
    return factory.build_services()


__all__ = (
    "ANALYSIS_AST_INDEX_PROVIDER_INTERFACE",
    "DEFAULT_IDENTITY_SERVICE_FACTORY_INTERFACE",
    "DEFAULT_ITEM_STATIC_IDENTITY_INTERFACE",
    "DEFAULT_ITEM_STATIC_IDENTITY_SCHEMA",
    "PROOF_REUSE_SESSION_IDENTITY_INTERFACE",
    "AnalysisASTIndexProvider",
    "DefaultIdentityReason",
    "DefaultIdentityServiceFactory",
    "DefaultItemStaticIdentity",
    "ProofReuseSessionIdentity",
    "build_default_identity_services",
)
