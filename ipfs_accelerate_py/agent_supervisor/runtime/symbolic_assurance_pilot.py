"""Generic frozen multi-repository symbolic assurance pilot orchestration.

Evidence schema and product/domain vocabulary are supplied exclusively by an
injected :class:`PilotProgramProfile`.  The orchestrator freezes repository
descriptors via an injected forest builder, admits paths via
:class:`RepositoryAdmissionPolicy`, executes injectable stage runners, and
publishes content-addressed artifacts with atomic bounded writes.

Safety invariants (non-waivable):

* dry-run and verify never call a model provider;
* neither mode mutates source trees;
* every file and finding is provenance-bound to forest / inventory CIDs;
* inconclusive, ambiguous, or partial findings remain non-executable;
* verification fails closed on changed trees, incomplete inventory, stale
  evidence, non-canonical artifacts, duplicate stages, unsafe output paths,
  provider access, or source mutation.

This module deliberately contains no product-specific path regexes, fixed
repository aliases, environment-variable names, fixture construction, or CLI
entry points.  Domain job adapters own those surfaces.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

from ipfs_accelerate_py.agent_supervisor.program_ast_adapters import adapt_program_source
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    ProgramGraph,
    ProgramGraphRoots,
    ProgramGraphSnapshot,
    ProgramNode,
    ProgramNodeKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json,
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.repository_corpus_index import (
    InclusionDecision,
    InventoryLimits,
    RepositoryCorpusIndex,
    build_repository_corpus_index,
)
from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AuthorityMode,
    ForestPolicy,
    ForestRootSpec,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForest,
    build_repository_forest,
)

# ---------------------------------------------------------------------------
# Bounds (generic; not product-specific)
# ---------------------------------------------------------------------------

MAX_ADMITTED_PARSE: Final[int] = 4_096
MAX_GRAPH_NODES: Final[int] = 16_384
MAX_BOARD_TASKS: Final[int] = 4_096
MAX_REPORT_BYTES: Final[int] = 8 * 1024 * 1024
MAX_FINDINGS_BOARD_BYTES: Final[int] = 1_000_000
MAX_STAGE_REASON_CODES: Final[int] = 128
MAX_ARTIFACT_PATH_BYTES: Final[int] = 4_096
MAX_PROFILE_TEXT_BYTES: Final[int] = 512

_DEFAULT_PARSER_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".md",
        ".markdown",
    }
)

_PROVIDER_MODULE_NAMES: Final[tuple[str, ...]] = (
    "openai",
    "anthropic",
    "groq",
    "litellm",
    "google.generativeai",
)

# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class SymbolicAssurancePilotError(ValueError):
    """Pilot input, pipeline, or verification failure."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "pilot_error").strip()
        detail = str(message or "").strip()
        super().__init__(detail or self.reason_code)


class PilotVerificationError(SymbolicAssurancePilotError):
    """Verify mode rejected the pilot report or live forest."""


class PilotMode(str, Enum):
    """Operator mode for the pilot pipeline."""

    DRY_RUN = "dry_run"
    VERIFY = "verify"
    RUN = "run"


class PilotStage(str, Enum):
    """Closed pipeline stages executed by the pilot."""

    FREEZE = "freeze"
    INVENTORY = "inventory"
    SCAN = "scan"
    GRAPH = "graph"
    CONTRACT = "contract"
    CACHE = "cache"
    PROOF = "proof"
    ZK_SHADOW = "zk_shadow"
    FINDINGS = "findings"
    TASKBOARD = "taskboard"
    PUBLISH = "publish"


class PilotConclusion(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_json_bytes(_plain(value))
    except (TypeError, ValueError) as exc:
        raise SymbolicAssurancePilotError(
            "noncanonical_artifact",
            "pilot data must be canonical JSON",
        ) from exc


def _identity(value: Any) -> str:
    return content_identity(_plain(value))


def _text(value: Any, name: str, *, maximum: int = 4_096) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value or value != value.strip():
        raise SymbolicAssurancePilotError(
            "invalid_text",
            f"{name} must be non-empty canonical text",
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise SymbolicAssurancePilotError(
            "invalid_text",
            f"{name} is unsafe or exceeds {maximum} bytes",
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SymbolicAssurancePilotError("invalid_boolean", f"{name} must be boolean")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SymbolicAssurancePilotError(
            "invalid_count",
            f"{name} must be a non-negative integer",
        )
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    body = canonical_json(_plain(payload))
    if not body.endswith("\n"):
        body = body + "\n"
    _atomic_write_text(path, body)


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SymbolicAssurancePilotError(
            "missing_artifact",
            f"cannot read {path}: {exc}",
        ) from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise SymbolicAssurancePilotError(
                    "noncanonical_artifact",
                    f"{path} contains duplicate JSON key {key!r}",
                )
            result[key] = item
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise SymbolicAssurancePilotError(
            "noncanonical_artifact",
            f"{path} is not valid JSON",
        ) from exc


def _assert_no_provider_surface() -> None:
    """Fail closed if provider SDKs are loaded during pilot execution."""

    for module_name in _PROVIDER_MODULE_NAMES:
        if module_name in sys.modules:
            raise SymbolicAssurancePilotError(
                "provider_call_forbidden",
                f"provider SDK {module_name!r} must not be loaded during pilot",
            )


def _resolve_under_allowed(path: Path, allowed_roots: Sequence[Path]) -> Path:
    """Reject output paths that escape every allowed root."""

    resolved = path.expanduser().resolve(strict=False)
    if not allowed_roots:
        return resolved
    for root in allowed_roots:
        try:
            root_resolved = root.expanduser().resolve(strict=False)
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue
    raise SymbolicAssurancePilotError(
        "unsafe_output_path",
        f"output path {resolved} is outside every allowed root",
    )


# ---------------------------------------------------------------------------
# Profile / admission / config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotProgramProfile:
    """Immutable, content-identified pilot identity and schema bindings.

    All product/domain strings live here.  The generic orchestrator never
    invents goal IDs, board namespaces, or schema identities.
    """

    schema: str
    version: int = 1
    objective_id: str = ""
    task_id: str = ""
    requirement_id: str = ""
    producer: str = "symbolic-assurance-pilot@1"
    board_namespace: str = ""
    policy_revision: str = ""
    evidence: str = ""
    manifest_schema: str = ""
    coverage_schema: str = ""
    stage_receipt_schema: str = ""
    artifact_set_schema: str = ""
    contract_schema: str = ""
    cache_schema: str = ""
    proof_schema: str = ""
    zk_shadow_schema: str = ""
    findings_schema: str = ""
    primary_repository_aliases: tuple[str, ...] = ()
    broken_contract_marker: str = "PILOT_CONTRACT_BROKEN"
    inconclusive_marker: str = "PILOT_INCONCLUSIVE"
    board_title: str = "Symbolic Assurance Findings Board"
    parser_suffixes: tuple[str, ...] = tuple(sorted(_DEFAULT_PARSER_SUFFIXES))
    max_graph_nodes: int = MAX_GRAPH_NODES
    max_board_tasks: int = MAX_BOARD_TASKS

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema", maximum=MAX_PROFILE_TEXT_BYTES))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise SymbolicAssurancePilotError("invalid_bound", "version must be a positive integer")
        for name in (
            "objective_id",
            "task_id",
            "requirement_id",
            "producer",
            "board_namespace",
            "policy_revision",
        ):
            raw = getattr(self, name)
            if raw:
                object.__setattr__(
                    self,
                    name,
                    _text(raw, name, maximum=MAX_PROFILE_TEXT_BYTES),
                )
        evidence = self.evidence or self.schema
        object.__setattr__(
            self, "evidence", _text(evidence, "evidence", maximum=MAX_PROFILE_TEXT_BYTES)
        )
        for name, default_suffix in (
            ("manifest_schema", "-manifest@1"),
            ("coverage_schema", "-coverage@1"),
            ("stage_receipt_schema", "-stage@1"),
            ("artifact_set_schema", "-artifacts@1"),
            ("contract_schema", "-contract@1"),
            ("cache_schema", "-cache@1"),
            ("proof_schema", "-proof@1"),
            ("zk_shadow_schema", "-zk-shadow@1"),
            ("findings_schema", "-findings@1"),
        ):
            raw = getattr(self, name)
            if not raw:
                base = self.schema.rsplit("@", 1)[0]
                raw = f"{base}{default_suffix}"
            object.__setattr__(
                self, name, _text(raw, name, maximum=MAX_PROFILE_TEXT_BYTES)
            )
        aliases = tuple(
            dict.fromkeys(
                _text(item, "primary_repository_alias", maximum=128)
                for item in self.primary_repository_aliases
                if str(item).strip()
            )
        )
        object.__setattr__(self, "primary_repository_aliases", aliases)
        object.__setattr__(
            self,
            "broken_contract_marker",
            _text(self.broken_contract_marker, "broken_contract_marker", maximum=256),
        )
        object.__setattr__(
            self,
            "inconclusive_marker",
            _text(self.inconclusive_marker, "inconclusive_marker", maximum=256),
        )
        object.__setattr__(
            self,
            "board_title",
            _text(self.board_title, "board_title", maximum=512),
        )
        suffixes = tuple(
            sorted(
                {
                    item if str(item).startswith(".") else f".{item}"
                    for item in self.parser_suffixes
                    if str(item).strip()
                }
            )
        )
        if not suffixes:
            raise SymbolicAssurancePilotError("invalid_bound", "parser_suffixes must be non-empty")
        object.__setattr__(self, "parser_suffixes", suffixes)
        for name in ("max_graph_nodes", "max_board_tasks"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SymbolicAssurancePilotError(
                    "invalid_bound",
                    f"{name} must be a positive integer",
                )

    @property
    def profile_cid(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "objective_id": self.objective_id,
            "task_id": self.task_id,
            "requirement_id": self.requirement_id,
            "producer": self.producer,
            "board_namespace": self.board_namespace,
            "policy_revision": self.policy_revision,
            "evidence": self.evidence,
            "manifest_schema": self.manifest_schema,
            "coverage_schema": self.coverage_schema,
            "stage_receipt_schema": self.stage_receipt_schema,
            "artifact_set_schema": self.artifact_set_schema,
            "contract_schema": self.contract_schema,
            "cache_schema": self.cache_schema,
            "proof_schema": self.proof_schema,
            "zk_shadow_schema": self.zk_shadow_schema,
            "findings_schema": self.findings_schema,
            "primary_repository_aliases": list(self.primary_repository_aliases),
            "broken_contract_marker": self.broken_contract_marker,
            "inconclusive_marker": self.inconclusive_marker,
            "board_title": self.board_title,
            "parser_suffixes": list(self.parser_suffixes),
            "max_graph_nodes": self.max_graph_nodes,
            "max_board_tasks": self.max_board_tasks,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PilotProgramProfile":
        if not isinstance(payload, Mapping):
            raise SymbolicAssurancePilotError("invalid_profile", "profile must be an object")
        return cls(
            schema=str(payload.get("schema") or ""),
            version=int(payload.get("version") or 1),
            objective_id=str(payload.get("objective_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            requirement_id=str(payload.get("requirement_id") or ""),
            producer=str(payload.get("producer") or "symbolic-assurance-pilot@1"),
            board_namespace=str(payload.get("board_namespace") or ""),
            policy_revision=str(payload.get("policy_revision") or ""),
            evidence=str(payload.get("evidence") or ""),
            manifest_schema=str(payload.get("manifest_schema") or ""),
            coverage_schema=str(payload.get("coverage_schema") or ""),
            stage_receipt_schema=str(payload.get("stage_receipt_schema") or ""),
            artifact_set_schema=str(payload.get("artifact_set_schema") or ""),
            contract_schema=str(payload.get("contract_schema") or ""),
            cache_schema=str(payload.get("cache_schema") or ""),
            proof_schema=str(payload.get("proof_schema") or ""),
            zk_shadow_schema=str(payload.get("zk_shadow_schema") or ""),
            findings_schema=str(payload.get("findings_schema") or ""),
            primary_repository_aliases=tuple(payload.get("primary_repository_aliases") or ()),
            broken_contract_marker=str(
                payload.get("broken_contract_marker") or "PILOT_CONTRACT_BROKEN"
            ),
            inconclusive_marker=str(
                payload.get("inconclusive_marker") or "PILOT_INCONCLUSIVE"
            ),
            board_title=str(
                payload.get("board_title") or "Symbolic Assurance Findings Board"
            ),
            parser_suffixes=tuple(
                payload.get("parser_suffixes") or sorted(_DEFAULT_PARSER_SUFFIXES)
            ),
            max_graph_nodes=int(payload.get("max_graph_nodes") or MAX_GRAPH_NODES),
            max_board_tasks=int(payload.get("max_board_tasks") or MAX_BOARD_TASKS),
        )


@dataclass(frozen=True)
class RepositoryAdmissionPolicy:
    """Path admission policy for pilot scan selection.

    Domain path filters are policy data.  Generic code never hard-codes product
    path regexes or repository alias tables.
    """

    admit_all_included: bool = False
    admit_all_aliases: tuple[str, ...] = ()
    path_patterns: tuple[str, ...] = ()
    alias_path_patterns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    predicate: Callable[[str, str], bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.admit_all_included, bool):
            raise SymbolicAssurancePilotError(
                "invalid_boolean", "admit_all_included must be boolean"
            )
        aliases = tuple(
            dict.fromkeys(
                _text(item, "admit_alias", maximum=128)
                for item in self.admit_all_aliases
                if str(item).strip()
            )
        )
        object.__setattr__(self, "admit_all_aliases", aliases)
        patterns = tuple(
            _text(item, "path_pattern", maximum=1_024)
            for item in self.path_patterns
            if str(item).strip()
        )
        object.__setattr__(self, "path_patterns", patterns)
        compiled_alias: dict[str, tuple[str, ...]] = {}
        for key, values in dict(self.alias_path_patterns or {}).items():
            alias = _text(key, "alias_path_pattern_key", maximum=128)
            compiled_alias[alias] = tuple(
                _text(item, "alias_path_pattern", maximum=1_024)
                for item in values
                if str(item).strip()
            )
        object.__setattr__(
            self, "alias_path_patterns", dict(sorted(compiled_alias.items()))
        )
        if self.predicate is not None and not callable(self.predicate):
            raise SymbolicAssurancePilotError(
                "invalid_admission_predicate",
                "predicate must be callable or None",
            )
        if (
            not self.admit_all_included
            and not self.admit_all_aliases
            and not self.path_patterns
            and not self.alias_path_patterns
            and self.predicate is None
        ):
            raise SymbolicAssurancePilotError(
                "empty_admission_policy",
                "admission policy must admit something",
            )

    def admits(self, relative_path: str, *, repository_alias: str) -> bool:
        """Return True when an included parser-eligible path is admitted."""

        alias = str(repository_alias or "").strip()
        path = str(relative_path or "").replace("\\", "/").strip()
        if not path or not alias:
            return False
        if self.predicate is not None:
            try:
                if bool(self.predicate(path, alias)):
                    return True
            except Exception as exc:
                raise SymbolicAssurancePilotError(
                    "admission_predicate_failed",
                    f"admission predicate raised: {exc}",
                ) from exc
        if self.admit_all_included:
            return True
        if alias in self.admit_all_aliases:
            return True
        for pattern in self.path_patterns:
            if re.search(pattern, path):
                return True
        for pattern in self.alias_path_patterns.get(alias, ()):
            if re.search(pattern, path):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "admit_all_included": self.admit_all_included,
            "admit_all_aliases": list(self.admit_all_aliases),
            "path_patterns": list(self.path_patterns),
            "alias_path_patterns": {
                key: list(values) for key, values in self.alias_path_patterns.items()
            },
            "has_predicate": self.predicate is not None,
        }


ForestBuilder = Callable[["PilotConfig"], RepositoryForest]
StageRunner = Callable[["PilotStageContext"], "StageRunnerResult"]


@dataclass(frozen=True)
class PilotConfig:
    """Tuple/profile-driven pilot configuration.

    Repository roots are a tuple of :class:`ForestRootSpec` (or a prebuilt
    :class:`ForestPolicy`).  Forest construction, path admission, stage runners,
    schema/goal/task/board identities, and artifact destinations are all
    injected — never fixed product fields.
    """

    profile: PilotProgramProfile
    admission_policy: RepositoryAdmissionPolicy
    repositories: tuple[ForestRootSpec, ...] = ()
    forest_policy: ForestPolicy | None = None
    forest_builder: ForestBuilder | None = None
    stage_runners: Mapping[str, StageRunner] | None = None
    artifact_dir: Path | None = None
    findings_board_path: Path | None = None
    inventory_limits: InventoryLimits | None = None
    max_admitted_parse: int = MAX_ADMITTED_PARSE
    write_artifacts: bool = True
    write_findings_board: bool = True
    require_exhaustive_aliases: tuple[str, ...] = ()
    allowed_output_roots: tuple[Path, ...] = ()
    sole_write_alias: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.profile, PilotProgramProfile):
            if isinstance(self.profile, Mapping):
                object.__setattr__(
                    self, "profile", PilotProgramProfile.from_dict(self.profile)
                )
            else:
                raise SymbolicAssurancePilotError(
                    "invalid_profile", "profile is required"
                )
        if not isinstance(self.admission_policy, RepositoryAdmissionPolicy):
            if isinstance(self.admission_policy, Mapping):
                object.__setattr__(
                    self,
                    "admission_policy",
                    RepositoryAdmissionPolicy(**dict(self.admission_policy)),
                )
            else:
                raise SymbolicAssurancePilotError(
                    "invalid_admission_policy",
                    "admission_policy is required",
                )
        roots: list[ForestRootSpec] = []
        for item in self.repositories or ():
            if isinstance(item, ForestRootSpec):
                roots.append(item)
            elif isinstance(item, Mapping):
                roots.append(
                    ForestRootSpec(
                        alias=str(item.get("alias") or ""),
                        root_path=item.get("root_path") or item.get("path") or "",
                        authority=item.get("authority"),
                        required=bool(item.get("required", True)),
                        logical_name=str(item.get("logical_name") or ""),
                        remote_url=str(item.get("remote_url") or ""),
                    )
                )
            else:
                raise SymbolicAssurancePilotError(
                    "invalid_repository_spec",
                    "repositories must be ForestRootSpec values",
                )
        object.__setattr__(self, "repositories", tuple(roots))
        if self.forest_policy is not None and not isinstance(
            self.forest_policy, ForestPolicy
        ):
            raise SymbolicAssurancePilotError("invalid_forest_policy")
        if not self.repositories and self.forest_policy is None and self.forest_builder is None:
            raise SymbolicAssurancePilotError(
                "missing_repositories",
                "PilotConfig requires repositories, forest_policy, or forest_builder",
            )
        if self.forest_builder is not None and not callable(self.forest_builder):
            raise SymbolicAssurancePilotError(
                "invalid_forest_builder",
                "forest_builder must be callable or None",
            )
        if self.stage_runners is not None:
            runners = {
                str(key): value for key, value in dict(self.stage_runners).items()
            }
            for key, value in runners.items():
                if not callable(value):
                    raise SymbolicAssurancePilotError(
                        "invalid_stage_runner",
                        f"stage runner {key!r} must be callable",
                    )
            object.__setattr__(self, "stage_runners", dict(sorted(runners.items())))
        if self.artifact_dir is not None:
            object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        if self.findings_board_path is not None:
            object.__setattr__(
                self, "findings_board_path", Path(self.findings_board_path)
            )
        if not isinstance(self.max_admitted_parse, int) or self.max_admitted_parse < 1:
            raise SymbolicAssurancePilotError(
                "invalid_bound",
                "max_admitted_parse must be a positive integer",
            )
        for flag_name in ("write_artifacts", "write_findings_board"):
            if not isinstance(getattr(self, flag_name), bool):
                raise SymbolicAssurancePilotError(
                    "invalid_boolean",
                    f"{flag_name} must be boolean",
                )
        exhaustive = tuple(
            dict.fromkeys(
                _text(item, "require_exhaustive_alias", maximum=128)
                for item in self.require_exhaustive_aliases
                if str(item).strip()
            )
        )
        object.__setattr__(self, "require_exhaustive_aliases", exhaustive)
        allowed = tuple(Path(item) for item in self.allowed_output_roots or ())
        object.__setattr__(self, "allowed_output_roots", allowed)
        write_alias = str(self.sole_write_alias or "").strip()
        if not write_alias:
            if self.forest_policy is not None:
                write_alias = self.forest_policy.sole_write_alias
            elif self.repositories:
                # Prefer the first writable root; otherwise the first root.
                for root in self.repositories:
                    auth = root.authority
                    mode = ""
                    if isinstance(auth, RepositoryAuthority):
                        mode = auth.mode
                    elif isinstance(auth, Mapping):
                        mode = str(auth.get("mode") or "")
                    if mode == AuthorityMode.READ_WRITE.value:
                        write_alias = root.alias
                        break
                if not write_alias:
                    write_alias = self.repositories[0].alias
        if write_alias:
            object.__setattr__(
                self,
                "sole_write_alias",
                _text(write_alias, "sole_write_alias", maximum=128),
            )

    def resolved_artifact_dir(self) -> Path:
        if self.artifact_dir is None:
            raise SymbolicAssurancePilotError(
                "missing_artifact_dir",
                "artifact_dir must be provided when writing artifacts",
            )
        path = Path(self.artifact_dir)
        if self.allowed_output_roots:
            return _resolve_under_allowed(path, self.allowed_output_roots)
        return path.expanduser().resolve(strict=False)

    def resolved_findings_board_path(self) -> Path:
        if self.findings_board_path is None:
            raise SymbolicAssurancePilotError(
                "missing_findings_board_path",
                "findings_board_path must be provided when writing the board",
            )
        path = Path(self.findings_board_path)
        if self.allowed_output_roots:
            return _resolve_under_allowed(path, self.allowed_output_roots)
        return path.expanduser().resolve(strict=False)

    def without_writes(self) -> "PilotConfig":
        """Return a config that never writes artifacts or boards."""

        return PilotConfig(
            profile=self.profile,
            admission_policy=self.admission_policy,
            repositories=self.repositories,
            forest_policy=self.forest_policy,
            forest_builder=self.forest_builder,
            stage_runners=self.stage_runners,
            artifact_dir=None,
            findings_board_path=None,
            inventory_limits=self.inventory_limits,
            max_admitted_parse=self.max_admitted_parse,
            write_artifacts=False,
            write_findings_board=False,
            require_exhaustive_aliases=self.require_exhaustive_aliases,
            allowed_output_roots=self.allowed_output_roots,
            sole_write_alias=self.sole_write_alias,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "admission_policy": self.admission_policy.to_dict(),
            "repositories": [
                {
                    "alias": root.alias,
                    "root_path": str(root.root_path),
                    "required": root.required,
                    "logical_name": root.logical_name,
                    "remote_url": root.remote_url,
                }
                for root in self.repositories
            ],
            "artifact_dir": (
                str(self.artifact_dir) if self.artifact_dir is not None else None
            ),
            "findings_board_path": (
                str(self.findings_board_path)
                if self.findings_board_path is not None
                else None
            ),
            "max_admitted_parse": self.max_admitted_parse,
            "write_artifacts": self.write_artifacts,
            "write_findings_board": self.write_findings_board,
            "require_exhaustive_aliases": list(self.require_exhaustive_aliases),
            "sole_write_alias": self.sole_write_alias,
            "has_forest_builder": self.forest_builder is not None,
            "has_stage_runners": bool(self.stage_runners),
        }


# ---------------------------------------------------------------------------
# Stage / artifact / report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageReceipt:
    """One deterministic pipeline stage receipt."""

    stage: PilotStage
    status: PilotConclusion
    artifact_cid: str
    input_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    schema: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage",
            self.stage if isinstance(self.stage, PilotStage) else PilotStage(self.stage),
        )
        object.__setattr__(
            self,
            "status",
            self.status
            if isinstance(self.status, PilotConclusion)
            else PilotConclusion(self.status),
        )
        object.__setattr__(
            self, "artifact_cid", _text(self.artifact_cid, "artifact_cid", maximum=128)
        )
        object.__setattr__(
            self,
            "input_cids",
            tuple(_text(item, "input_cid", maximum=128) for item in self.input_cids),
        )
        reasons = tuple(
            dict.fromkeys(
                _text(item, "reason_code", maximum=192)
                for item in self.reason_codes
                if str(item).strip()
            )
        )
        if len(reasons) > MAX_STAGE_REASON_CODES:
            raise SymbolicAssurancePilotError("stage_reason_bound_exceeded")
        object.__setattr__(self, "reason_codes", reasons)
        metrics: dict[str, int] = {}
        for key, value in dict(self.metrics or {}).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SymbolicAssurancePilotError(
                    "invalid_metric",
                    f"metric {key!r} must be a non-negative int",
                )
            metrics[str(key)] = value
        object.__setattr__(self, "metrics", dict(sorted(metrics.items())))
        if self.schema:
            object.__setattr__(
                self, "schema", _text(self.schema, "stage_schema", maximum=256)
            )

    @property
    def receipt_cid(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "stage": self.stage.value,
            "status": self.status.value,
            "artifact_cid": self.artifact_cid,
            "input_cids": list(self.input_cids),
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
        }
        if self.schema:
            payload["schema"] = self.schema
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageReceipt":
        return cls(
            stage=str(payload.get("stage") or ""),
            status=str(payload.get("status") or ""),
            artifact_cid=str(payload.get("artifact_cid") or ""),
            input_cids=tuple(payload.get("input_cids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            metrics=dict(payload.get("metrics") or {}),
            schema=str(payload.get("schema") or ""),
        )


@dataclass(frozen=True)
class PilotArtifactSet:
    """Published content identities for every pilot product."""

    forest_cid: str
    manifest_cid: str
    coverage_cid: str
    inventory_cid: str
    graph_cid: str
    cache_cid: str
    proof_cid: str
    zk_shadow_cid: str
    finding_ledger_cid: str
    taskboard_cid: str
    report_cid: str = ""
    schema: str = ""

    def __post_init__(self) -> None:
        for name in (
            "forest_cid",
            "manifest_cid",
            "coverage_cid",
            "inventory_cid",
            "graph_cid",
            "cache_cid",
            "proof_cid",
            "zk_shadow_cid",
            "finding_ledger_cid",
            "taskboard_cid",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=128)
            )
        if self.report_cid:
            object.__setattr__(
                self, "report_cid", _text(self.report_cid, "report_cid", maximum=128)
            )
        if self.schema:
            object.__setattr__(
                self, "schema", _text(self.schema, "artifact_schema", maximum=256)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "forest_cid": self.forest_cid,
            "manifest_cid": self.manifest_cid,
            "coverage_cid": self.coverage_cid,
            "inventory_cid": self.inventory_cid,
            "graph_cid": self.graph_cid,
            "cache_cid": self.cache_cid,
            "proof_cid": self.proof_cid,
            "zk_shadow_cid": self.zk_shadow_cid,
            "finding_ledger_cid": self.finding_ledger_cid,
            "taskboard_cid": self.taskboard_cid,
        }
        if self.schema:
            payload["schema"] = self.schema
        if self.report_cid:
            payload["report_cid"] = self.report_cid
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PilotArtifactSet":
        return cls(
            forest_cid=str(payload.get("forest_cid") or ""),
            manifest_cid=str(payload.get("manifest_cid") or ""),
            coverage_cid=str(payload.get("coverage_cid") or ""),
            inventory_cid=str(payload.get("inventory_cid") or ""),
            graph_cid=str(payload.get("graph_cid") or ""),
            cache_cid=str(payload.get("cache_cid") or ""),
            proof_cid=str(payload.get("proof_cid") or ""),
            zk_shadow_cid=str(payload.get("zk_shadow_cid") or ""),
            finding_ledger_cid=str(payload.get("finding_ledger_cid") or ""),
            taskboard_cid=str(payload.get("taskboard_cid") or ""),
            report_cid=str(payload.get("report_cid") or ""),
            schema=str(payload.get("schema") or ""),
        )


@dataclass(frozen=True)
class SymbolicAssurancePilotReport:
    """Authoritative, content-addressed pilot receipt."""

    schema: str = ""
    version: int = 1
    objective_id: str = ""
    task_id: str = ""
    requirement_id: str = ""
    mode: PilotMode = PilotMode.DRY_RUN
    conclusion: PilotConclusion = PilotConclusion.PASSED
    forest_id: str = ""
    tree_bindings: Mapping[str, str] = field(default_factory=dict)
    commit_bindings: Mapping[str, str] = field(default_factory=dict)
    stages: tuple[StageReceipt, ...] = ()
    artifacts: PilotArtifactSet | None = None
    admitted_file_count: int = 0
    primary_file_count: int = 0
    closure_file_count: int = 0
    finding_count: int = 0
    executable_task_count: int = 0
    review_count: int = 0
    inconclusive_count: int = 0
    provider_calls: int = 0
    source_mutations: int = 0
    reason_codes: tuple[str, ...] = ()
    board_markdown_cid: str = ""
    board_namespace: str = ""
    policy_revision: str = ""
    evidence: str = ""
    profile_cid: str = ""
    authorizes_repair: bool = False
    is_completion_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise SymbolicAssurancePilotError("unsupported_pilot_version")
        object.__setattr__(
            self,
            "mode",
            self.mode if isinstance(self.mode, PilotMode) else PilotMode(self.mode),
        )
        object.__setattr__(
            self,
            "conclusion",
            self.conclusion
            if isinstance(self.conclusion, PilotConclusion)
            else PilotConclusion(self.conclusion),
        )
        object.__setattr__(self, "forest_id", _text(self.forest_id, "forest_id"))
        trees = {
            _text(key, "tree_alias"): _text(value, "tree_id", maximum=128)
            for key, value in dict(self.tree_bindings or {}).items()
        }
        commits = {
            _text(key, "commit_alias"): _text(value, "commit_id", maximum=128)
            for key, value in dict(self.commit_bindings or {}).items()
        }
        object.__setattr__(self, "tree_bindings", dict(sorted(trees.items())))
        object.__setattr__(self, "commit_bindings", dict(sorted(commits.items())))
        stages = tuple(
            item if isinstance(item, StageReceipt) else StageReceipt.from_dict(item)
            for item in self.stages
        )
        seen_stages: set[PilotStage] = set()
        for stage in stages:
            if stage.stage in seen_stages:
                raise SymbolicAssurancePilotError(
                    "duplicate_stage",
                    f"duplicate stage {stage.stage.value}",
                )
            seen_stages.add(stage.stage)
        object.__setattr__(self, "stages", stages)
        artifacts = self.artifacts
        if artifacts is not None and not isinstance(artifacts, PilotArtifactSet):
            artifacts = PilotArtifactSet.from_dict(artifacts)
        object.__setattr__(self, "artifacts", artifacts)
        for name in (
            "admitted_file_count",
            "primary_file_count",
            "closure_file_count",
            "finding_count",
            "executable_task_count",
            "review_count",
            "inconclusive_count",
            "provider_calls",
            "source_mutations",
        ):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        if self.provider_calls != 0:
            raise SymbolicAssurancePilotError(
                "provider_call_forbidden",
                "pilot reports must record zero provider calls",
            )
        if self.source_mutations != 0:
            raise SymbolicAssurancePilotError(
                "source_mutation_forbidden",
                "pilot reports must record zero source mutations",
            )
        reasons = tuple(
            dict.fromkeys(
                _text(item, "reason_code", maximum=192)
                for item in self.reason_codes
                if str(item).strip()
            )
        )
        object.__setattr__(self, "reason_codes", reasons)
        if self.board_markdown_cid:
            object.__setattr__(
                self,
                "board_markdown_cid",
                _text(self.board_markdown_cid, "board_markdown_cid", maximum=128),
            )
        if self.board_namespace:
            object.__setattr__(
                self,
                "board_namespace",
                _text(self.board_namespace, "board_namespace", maximum=256),
            )
        if self.policy_revision:
            object.__setattr__(
                self,
                "policy_revision",
                _text(self.policy_revision, "policy_revision", maximum=256),
            )
        object.__setattr__(
            self, "evidence", _text(self.evidence or self.schema, "evidence", maximum=256)
        )
        if self.profile_cid:
            object.__setattr__(
                self, "profile_cid", _text(self.profile_cid, "profile_cid", maximum=128)
            )
        if self.authorizes_repair:
            raise SymbolicAssurancePilotError(
                "authority_drift",
                "pilot report must never authorize repair",
            )
        if self.is_completion_evidence:
            raise SymbolicAssurancePilotError(
                "authority_drift",
                "pilot report is not completion evidence",
            )
        body = _canonical_bytes(self._core_payload())
        if len(body) > MAX_REPORT_BYTES:
            raise SymbolicAssurancePilotError(
                "report_bound_exceeded",
                f"pilot report exceeds {MAX_REPORT_BYTES} bytes",
            )

    def _core_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "objective_id": self.objective_id,
            "task_id": self.task_id,
            "requirement_id": self.requirement_id,
            "mode": self.mode.value,
            "conclusion": self.conclusion.value,
            "forest_id": self.forest_id,
            "tree_bindings": dict(self.tree_bindings),
            "commit_bindings": dict(self.commit_bindings),
            "stages": [item.to_dict() for item in self.stages],
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "admitted_file_count": self.admitted_file_count,
            "primary_file_count": self.primary_file_count,
            "closure_file_count": self.closure_file_count,
            "finding_count": self.finding_count,
            "executable_task_count": self.executable_task_count,
            "review_count": self.review_count,
            "inconclusive_count": self.inconclusive_count,
            "provider_calls": self.provider_calls,
            "source_mutations": self.source_mutations,
            "reason_codes": list(self.reason_codes),
            "board_markdown_cid": self.board_markdown_cid,
            "board_namespace": self.board_namespace,
            "policy_revision": self.policy_revision,
            "evidence": self.evidence,
            "profile_cid": self.profile_cid,
            "authorizes_repair": False,
            "is_completion_evidence": False,
        }

    @property
    def report_cid(self) -> str:
        return _identity(self._core_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._core_payload()
        payload["report_cid"] = self.report_cid
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicAssurancePilotReport":
        if not isinstance(payload, Mapping):
            raise SymbolicAssurancePilotError("invalid_report", "report must be an object")
        data = dict(payload)
        claimed = str(data.pop("report_cid", "") or "")
        report = cls(
            schema=str(data.get("schema") or ""),
            version=int(data.get("version") or 1),
            objective_id=str(data.get("objective_id") or ""),
            task_id=str(data.get("task_id") or ""),
            requirement_id=str(data.get("requirement_id") or ""),
            mode=str(data.get("mode") or PilotMode.DRY_RUN.value),
            conclusion=str(data.get("conclusion") or PilotConclusion.PASSED.value),
            forest_id=str(data.get("forest_id") or ""),
            tree_bindings=dict(data.get("tree_bindings") or {}),
            commit_bindings=dict(data.get("commit_bindings") or {}),
            stages=tuple(data.get("stages") or ()),
            artifacts=data.get("artifacts"),
            admitted_file_count=int(data.get("admitted_file_count") or 0),
            primary_file_count=int(data.get("primary_file_count") or 0),
            closure_file_count=int(data.get("closure_file_count") or 0),
            finding_count=int(data.get("finding_count") or 0),
            executable_task_count=int(data.get("executable_task_count") or 0),
            review_count=int(data.get("review_count") or 0),
            inconclusive_count=int(data.get("inconclusive_count") or 0),
            provider_calls=int(data.get("provider_calls") or 0),
            source_mutations=int(data.get("source_mutations") or 0),
            reason_codes=tuple(data.get("reason_codes") or ()),
            board_markdown_cid=str(data.get("board_markdown_cid") or ""),
            board_namespace=str(data.get("board_namespace") or ""),
            policy_revision=str(data.get("policy_revision") or ""),
            evidence=str(data.get("evidence") or data.get("schema") or ""),
            profile_cid=str(data.get("profile_cid") or ""),
            authorizes_repair=bool(data.get("authorizes_repair", False)),
            is_completion_evidence=bool(data.get("is_completion_evidence", False)),
        )
        if claimed and claimed != report.report_cid:
            raise PilotVerificationError(
                "stale_evidence",
                "report_cid does not match canonical body",
            )
        return report


# ---------------------------------------------------------------------------
# Stage context / runner results
# ---------------------------------------------------------------------------


@dataclass
class PilotFinding:
    """Lightweight deterministic finding used by default stage runners."""

    finding_cid: str
    status: str
    severity: str
    summary: str
    repository_alias: str
    relative_path: str
    repository_id: str = ""
    executable: bool = False
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    expected_contract_cid: str = ""
    observed_contract_cid: str = ""
    root_cause_family: str = ""
    remediation_scope: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_cid": self.finding_cid,
            "status": self.status,
            "severity": self.severity,
            "summary": self.summary,
            "repository_alias": self.repository_alias,
            "relative_path": self.relative_path,
            "repository_id": self.repository_id,
            "executable": self.executable,
            "symbols": list(self.symbols),
            "interfaces": list(self.interfaces),
            "expected_contract_cid": self.expected_contract_cid,
            "observed_contract_cid": self.observed_contract_cid,
            "root_cause_family": self.root_cause_family,
            "remediation_scope": list(self.remediation_scope),
        }


@dataclass
class StageRunnerResult:
    """Result produced by one injectable stage runner."""

    artifact_cid: str
    status: PilotConclusion = PilotConclusion.PASSED
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    payload: Mapping[str, Any] | None = None
    write_name: str = ""
    findings: tuple[PilotFinding, ...] = ()
    board_json: Mapping[str, Any] | None = None
    board_markdown: str = ""
    board_markdown_cid: str = ""
    executable_task_count: int = 0
    review_count: int = 0
    finding_count: int = 0
    admitted_count: int = 0
    inconclusive_count: int = 0
    repair_packets: tuple[Mapping[str, Any], ...] = ()
    graph: ProgramGraph | None = None
    # Optional multi-artifact write map: relative name -> payload
    extra_writes: Mapping[str, Any] = field(default_factory=dict)
    # Side channels used by later default stages
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class PilotStageContext:
    """Mutable pipeline context shared across stage runners."""

    config: PilotConfig
    mode: PilotMode
    profile: PilotProgramProfile
    forest: RepositoryForest | None = None
    index: RepositoryCorpusIndex | None = None
    admitted: tuple[Any, ...] = ()
    stages: list[StageReceipt] = field(default_factory=list)
    artifacts_partial: dict[str, str] = field(default_factory=dict)
    findings: list[PilotFinding] = field(default_factory=list)
    graph: ProgramGraph | None = None
    coverage_bundle: dict[str, Any] = field(default_factory=dict)
    contract_cid: str = ""
    cache_cid: str = ""
    proof_cid: str = ""
    zk_shadow_cid: str = ""
    finding_ledger_cid: str = ""
    taskboard_cid: str = ""
    board_json: dict[str, Any] = field(default_factory=dict)
    board_markdown: str = ""
    board_markdown_cid: str = ""
    executable_task_count: int = 0
    review_count: int = 0
    inconclusive_count: int = 0
    repair_packets: list[dict[str, Any]] = field(default_factory=list)
    incomplete_inventory: bool = False
    reason_codes: list[str] = field(default_factory=list)
    artifact_dir: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class StageRunnerProtocol(Protocol):
    def __call__(self, context: PilotStageContext) -> StageRunnerResult: ...


# ---------------------------------------------------------------------------
# Admission / freeze / inventory helpers
# ---------------------------------------------------------------------------


def admitted_entries_for_pilot(
    index: RepositoryCorpusIndex,
    admission_policy: RepositoryAdmissionPolicy,
) -> tuple[Any, ...]:
    """Return included corpus entries admitted by the policy."""

    selected = []
    for entry in index.entries:
        if entry.inclusion != InclusionDecision.INCLUDED.value:
            continue
        if not entry.parser_eligible:
            continue
        if not admission_policy.admits(
            entry.relative_path, repository_alias=entry.repository_alias
        ):
            continue
        selected.append(entry)
    selected.sort(
        key=lambda item: (
            item.repository_alias,
            item.relative_path,
            item.content_sha256,
        )
    )
    return tuple(selected)


def _descriptor_map(forest: RepositoryForest) -> dict[str, RepositoryDescriptor]:
    return {descriptor.alias: descriptor for descriptor in forest.descriptors}


def freeze_repository_descriptors(config: PilotConfig) -> RepositoryForest:
    """Freeze fresh repository descriptors using the injected forest builder."""

    if config.forest_builder is not None:
        forest = config.forest_builder(config)
    elif config.forest_policy is not None:
        forest = build_repository_forest(config.forest_policy)
    else:
        policy = ForestPolicy(
            roots=config.repositories,
            sole_write_alias=config.sole_write_alias,
        )
        forest = build_repository_forest(policy)
    if not isinstance(forest, RepositoryForest):
        raise SymbolicAssurancePilotError(
            "invalid_forest",
            "forest_builder must return a RepositoryForest",
        )
    if not forest.descriptors:
        raise SymbolicAssurancePilotError("empty_forest", "forest has no descriptors")
    return forest


def scan_inventory(
    forest: RepositoryForest,
    *,
    limits: InventoryLimits | None = None,
) -> RepositoryCorpusIndex:
    """Exhaustively inventory every forest descriptor."""

    return build_repository_corpus_index(forest, limits=limits)


def build_coverage_manifest(
    *,
    forest: RepositoryForest,
    index: RepositoryCorpusIndex,
    admitted: Sequence[Any],
    profile: PilotProgramProfile,
) -> dict[str, Any]:
    """Build the portable coverage + manifest projection."""

    by_alias: dict[str, list[dict[str, Any]]] = {}
    for entry in admitted:
        by_alias.setdefault(entry.repository_alias, []).append(
            {
                "relative_path": entry.relative_path,
                "canonical_path": entry.canonical_path,
                "blob_oid": entry.blob_oid,
                "content_sha256": entry.content_sha256,
                "entry_cid": entry.entry_cid,
                "size": entry.size,
                "classifications": list(entry.classifications),
            }
        )
    for alias in by_alias:
        by_alias[alias].sort(key=lambda item: item["relative_path"])

    primary_aliases = set(profile.primary_repository_aliases)
    primary_count = sum(
        1 for entry in admitted if entry.repository_alias in primary_aliases
    )
    closure_count = len(admitted) - primary_count

    repo_coverage = []
    for inventory in index.repositories:
        repo_coverage.append(
            {
                "repository_alias": inventory.repository_alias,
                "repository_id": inventory.repository_id,
                "descriptor_cid": inventory.descriptor_cid,
                "exhaustive": inventory.exhaustive,
                "observed_entry_count": inventory.observed_entry_count,
                "included_entry_count": inventory.included_entry_count,
                "excluded_entry_count": inventory.excluded_entry_count,
                "omitted_entry_count": inventory.omitted_entry_count,
                "reason_codes": list(inventory.reason_codes),
                "admitted_for_pilot": len(by_alias.get(inventory.repository_alias, [])),
            }
        )
    repo_coverage.sort(key=lambda item: item["repository_alias"])

    manifest = {
        "schema": profile.manifest_schema,
        "forest_id": forest.forest_id,
        "inventory_cid": index.inventory_cid,
        "policy_cid": forest.policy_cid,
        "admitted_file_count": len(admitted),
        "primary_file_count": primary_count,
        "closure_file_count": closure_count,
        "repositories": [
            {
                "alias": descriptor.alias,
                "repository_id": descriptor.repository_id,
                "descriptor_cid": descriptor.descriptor_cid,
                "commit": descriptor.commit,
                "tree": descriptor.tree,
                "authority": descriptor.authority.mode,
            }
            for descriptor in sorted(forest.descriptors, key=lambda item: item.alias)
        ],
        "admitted_by_alias": {
            alias: files for alias, files in sorted(by_alias.items())
        },
    }
    coverage = {
        "schema": profile.coverage_schema,
        "forest_id": forest.forest_id,
        "inventory_cid": index.inventory_cid,
        "repository_coverage": repo_coverage,
        "admitted_file_count": len(admitted),
        "primary_file_count": primary_count,
        "closure_file_count": closure_count,
        "complete": all(item["exhaustive"] for item in repo_coverage)
        and all(item["omitted_entry_count"] == 0 for item in repo_coverage),
    }
    return {
        "manifest": manifest,
        "coverage": coverage,
        "manifest_cid": _identity(manifest),
        "coverage_cid": _identity(coverage),
        "primary_file_count": primary_count,
        "closure_file_count": closure_count,
    }


def _read_entry_text(
    entry: Any,
    descriptors: Mapping[str, RepositoryDescriptor],
    *,
    parser_suffixes: Sequence[str],
) -> str | None:
    descriptor = descriptors.get(entry.repository_alias)
    if descriptor is None:
        return None
    path = descriptor.root_path / entry.relative_path
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix not in set(parser_suffixes):
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None


def build_pilot_program_graph(
    *,
    forest: RepositoryForest,
    admitted: Sequence[Any],
    profile: PilotProgramProfile,
    max_parse: int,
) -> tuple[ProgramGraph, dict[str, Any]]:
    """Parse admitted sources and emit a content-addressed program graph."""

    descriptors = _descriptor_map(forest)
    roots = ProgramGraphRoots(
        forest_id=forest.forest_id,
        tree_id=forest.forest_id,
        coverage_id=_identity(
            {
                "admitted": [
                    entry.entry_cid for entry in admitted[:max_parse]
                ]
            }
        ),
        included_roots=tuple(
            sorted({f"{entry.repository_alias}:{entry.relative_path}" for entry in admitted})
        ),
        extractor_id=profile.producer,
        config_id=profile.profile_cid,
        toolchain_id=profile.policy_revision or profile.producer,
    )
    nodes: list[ProgramNode] = []
    parse_metrics = {
        "parsed": 0,
        "unsupported": 0,
        "malformed": 0,
        "unreadable": 0,
        "skipped_bound": 0,
    }
    reasons: list[str] = []

    for index, entry in enumerate(admitted):
        if index >= max_parse:
            parse_metrics["skipped_bound"] += 1
            reasons.append("parse_bound_reached")
            continue
        text = _read_entry_text(
            entry, descriptors, parser_suffixes=profile.parser_suffixes
        )
        if text is None:
            parse_metrics["unreadable"] += 1
            continue
        result = adapt_program_source(
            text,
            path=entry.relative_path,
            blob_identity=entry.content_sha256 or entry.blob_oid or entry.entry_cid,
        )
        if result.status == "success":
            parse_metrics["parsed"] += 1
        elif result.status == "malformed":
            parse_metrics["malformed"] += 1
        else:
            parse_metrics["unsupported"] += 1

        blob_cid = entry.content_sha256 or entry.blob_oid or entry.entry_cid
        node_id = f"module:{entry.repository_alias}:{entry.relative_path}"
        nodes.append(
            ProgramNode(
                node_id=node_id,
                kind=ProgramNodeKind.MODULE,
                name=PurePosixPath(entry.relative_path).name or entry.relative_path,
                roots=roots,
                path=entry.relative_path,
                qualified_name=f"{entry.repository_alias}:{entry.relative_path}",
                language=str(getattr(result, "language", "") or ""),
                blob_identity=blob_cid,
                source_sha256=entry.content_sha256 or "",
                extractor_id=profile.producer,
                attributes={
                    "repository_alias": entry.repository_alias,
                    "entry_cid": entry.entry_cid,
                    "status": result.status,
                    "fact_count": len(result.facts),
                },
            )
        )
        for fact in result.facts[:64]:
            kind_name = str(fact.kind or "").lower()
            if "import" in kind_name:
                node_kind = ProgramNodeKind.IMPORT
            elif "export" in kind_name:
                node_kind = ProgramNodeKind.EXPORT
            elif "call" in kind_name:
                node_kind = ProgramNodeKind.FUNCTION
            elif "function" in kind_name or "class" in kind_name or "def" in kind_name:
                node_kind = ProgramNodeKind.FUNCTION
            elif "schema" in kind_name:
                node_kind = ProgramNodeKind.SCHEMA
            else:
                node_kind = ProgramNodeKind.SYMBOL
            fact_id = str(getattr(fact, "fact_id", "") or fact.name or kind_name)
            nodes.append(
                ProgramNode(
                    node_id=f"fact:{entry.repository_alias}:{fact_id}:{len(nodes)}",
                    kind=node_kind,
                    name=str(fact.name or fact_id),
                    roots=roots,
                    path=entry.relative_path,
                    qualified_name=str(fact.name or fact_id),
                    language=str(getattr(result, "language", "") or ""),
                    blob_identity=blob_cid,
                    extractor_id=profile.producer,
                    attributes={
                        "kind": str(fact.kind or ""),
                        "name": str(fact.name or ""),
                        "owner": str(getattr(fact, "owner", "") or ""),
                        "target": str(getattr(fact, "target", "") or ""),
                        "ambiguous": bool(getattr(fact, "ambiguous", False)),
                        "fact_id": fact_id,
                    },
                )
            )
            if len(nodes) >= profile.max_graph_nodes:
                reasons.append("graph_node_bound_reached")
                break
        if len(nodes) >= profile.max_graph_nodes:
            break

    complete = not reasons and parse_metrics["unreadable"] == 0
    snapshot = ProgramGraphSnapshot(
        roots=roots,
        nodes=tuple(nodes),
        edges=(),
        frontier_refs=tuple(sorted(set(reasons))),
        complete=complete,
    )
    graph = ProgramGraph(snapshot)
    return graph, {
        "metrics": parse_metrics,
        "reason_codes": sorted(set(reasons)),
        "graph_cid": graph.graph_id,
    }


# ---------------------------------------------------------------------------
# Default stage runners
# ---------------------------------------------------------------------------


def default_freeze_runner(context: PilotStageContext) -> StageRunnerResult:
    forest = freeze_repository_descriptors(context.config)
    context.forest = forest
    forest_payload = (
        forest.to_portable_dict()
        if hasattr(forest, "to_portable_dict")
        else {
            "forest_id": forest.forest_id,
            "descriptors": [
                item.to_portable_dict()
                if hasattr(item, "to_portable_dict")
                else {"alias": item.alias, "tree": item.tree, "commit": item.commit}
                for item in forest.descriptors
            ],
            "policy_cid": forest.policy_cid,
        }
    )
    context.artifacts_partial["forest_cid"] = forest.forest_id
    return StageRunnerResult(
        artifact_cid=forest.forest_id,
        metrics={"repositories": len(forest.descriptors)},
        payload=forest_payload,
        write_name="forest.json",
    )


def default_inventory_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest", "inventory requires freeze")
    limits = context.config.inventory_limits or InventoryLimits()
    index = scan_inventory(context.forest, limits=limits)
    context.index = index
    inventory_reasons: list[str] = []
    incomplete = False
    required = set(context.config.require_exhaustive_aliases)
    for repo in index.repositories:
        if not repo.exhaustive or repo.omitted_entry_count:
            incomplete = True
            inventory_reasons.extend(repo.reason_codes)
            inventory_reasons.append(f"incomplete:{repo.repository_alias}")
        if repo.repository_alias in required and (
            not repo.exhaustive or repo.omitted_entry_count
        ):
            raise SymbolicAssurancePilotError(
                "incomplete_inventory",
                f"required exhaustive inventory incomplete for {repo.repository_alias!r}",
            )
    context.incomplete_inventory = incomplete
    if incomplete:
        context.reason_codes.append("incomplete_inventory")
    context.artifacts_partial["inventory_cid"] = index.inventory_cid
    payload = (
        index.to_portable_dict()
        if hasattr(index, "to_portable_dict")
        else {"inventory_cid": index.inventory_cid, "entries": len(index.entries)}
    )
    return StageRunnerResult(
        artifact_cid=index.inventory_cid,
        status=PilotConclusion.INCOMPLETE if incomplete else PilotConclusion.PASSED,
        reason_codes=tuple(sorted(set(inventory_reasons))),
        metrics={
            "entries": len(index.entries),
            "repositories": len(index.repositories),
        },
        payload=payload,
        write_name="inventory.json",
    )


def default_scan_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None or context.index is None:
        raise SymbolicAssurancePilotError(
            "missing_inventory", "scan requires freeze and inventory"
        )
    admitted = admitted_entries_for_pilot(
        context.index, context.config.admission_policy
    )
    context.admitted = admitted
    coverage_bundle = build_coverage_manifest(
        forest=context.forest,
        index=context.index,
        admitted=admitted,
        profile=context.profile,
    )
    context.coverage_bundle = coverage_bundle
    context.artifacts_partial["manifest_cid"] = coverage_bundle["manifest_cid"]
    context.artifacts_partial["coverage_cid"] = coverage_bundle["coverage_cid"]
    return StageRunnerResult(
        artifact_cid=coverage_bundle["manifest_cid"],
        metrics={
            "admitted": len(admitted),
            "primary": coverage_bundle["primary_file_count"],
            "closure": coverage_bundle["closure_file_count"],
        },
        payload=coverage_bundle["manifest"],
        write_name="manifest.json",
        extra_writes={"coverage.json": coverage_bundle["coverage"]},
        extras=coverage_bundle,
    )


def default_graph_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest", "graph requires freeze")
    graph, graph_meta = build_pilot_program_graph(
        forest=context.forest,
        admitted=context.admitted,
        profile=context.profile,
        max_parse=context.config.max_admitted_parse,
    )
    context.graph = graph
    context.artifacts_partial["graph_cid"] = graph_meta["graph_cid"]
    return StageRunnerResult(
        artifact_cid=graph_meta["graph_cid"],
        status=(
            PilotConclusion.INCOMPLETE
            if graph_meta["reason_codes"]
            else PilotConclusion.PASSED
        ),
        reason_codes=tuple(graph_meta["reason_codes"]),
        metrics={
            "nodes": len(graph.nodes),
            **{f"parse_{key}": value for key, value in graph_meta["metrics"].items()},
        },
        payload=graph.to_dict(),
        write_name="graph.json",
        graph=graph,
    )


def default_contract_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest", "contract requires freeze")
    descriptors = _descriptor_map(context.forest)
    findings: list[PilotFinding] = []
    broken = 0
    inconclusive = 0
    for entry in context.admitted:
        text = _read_entry_text(
            entry,
            descriptors,
            parser_suffixes=context.profile.parser_suffixes,
        )
        if text is None:
            continue
        if context.profile.broken_contract_marker in text:
            broken += 1
            expected = _identity(
                {"expected": "pilot-contract", "path": entry.relative_path}
            )
            observed = _identity(
                {
                    "observed": context.profile.broken_contract_marker,
                    "path": entry.relative_path,
                    "blob": entry.content_sha256,
                }
            )
            finding = PilotFinding(
                finding_cid=_identity(
                    {
                        "status": "contract_broken",
                        "path": entry.relative_path,
                        "alias": entry.repository_alias,
                        "blob": entry.content_sha256,
                    }
                ),
                status="contract_broken",
                severity="high",
                summary=(
                    "Pilot fixture marks an explicit contract break for "
                    f"{entry.relative_path}"
                ),
                repository_alias=entry.repository_alias,
                relative_path=entry.relative_path,
                repository_id=entry.repository_id,
                executable=True,
                symbols=(entry.relative_path,),
                interfaces=(
                    f"pilot://{entry.repository_alias}/{entry.relative_path}",
                ),
                expected_contract_cid=expected,
                observed_contract_cid=observed,
                root_cause_family="pilot-seeded-contract-break",
                remediation_scope=(entry.relative_path,),
            )
            findings.append(finding)
        elif context.profile.inconclusive_marker in text:
            inconclusive += 1
            finding = PilotFinding(
                finding_cid=_identity(
                    {
                        "status": "inconclusive",
                        "path": entry.relative_path,
                        "alias": entry.repository_alias,
                        "blob": entry.content_sha256,
                    }
                ),
                status="inconclusive",
                severity="low",
                summary=(
                    "Pilot fixture is explicitly inconclusive for "
                    f"{entry.relative_path}"
                ),
                repository_alias=entry.repository_alias,
                relative_path=entry.relative_path,
                repository_id=entry.repository_id,
                executable=False,
                symbols=(entry.relative_path,),
                interfaces=(
                    f"pilot://{entry.repository_alias}/{entry.relative_path}",
                ),
                expected_contract_cid=_identity(
                    {"expected": "unresolved", "path": entry.relative_path}
                ),
                observed_contract_cid=_identity(
                    {"observed": "inconclusive", "path": entry.relative_path}
                ),
                root_cause_family="pilot-inconclusive",
                remediation_scope=(entry.relative_path,),
            )
            findings.append(finding)

    context.findings = findings
    context.inconclusive_count = inconclusive
    contract_payload = {
        "schema": context.profile.contract_schema,
        "forest_id": context.forest.forest_id,
        "graph_cid": context.graph.graph_id if context.graph is not None else "",
        "broken_count": broken,
        "inconclusive_count": inconclusive,
        "finding_cids": [item.finding_cid for item in findings],
    }
    contract_cid = _identity(contract_payload)
    context.contract_cid = contract_cid
    return StageRunnerResult(
        artifact_cid=contract_cid,
        metrics={
            "broken": broken,
            "inconclusive": inconclusive,
            "findings": len(findings),
        },
        payload=contract_payload,
        write_name="contract.json",
        findings=tuple(findings),
        inconclusive_count=inconclusive,
        finding_count=len(findings),
    )


def default_cache_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest", "cache requires freeze")
    inventory_cid = context.artifacts_partial.get("inventory_cid", "")
    graph_cid = context.artifacts_partial.get("graph_cid", "")
    contract_cid = context.contract_cid
    kinds = ("inventory", "graph", "contract", "proof")
    stored: list[str] = []
    for kind in kinds:
        body = {
            "component": kind,
            "inventory_cid": inventory_cid,
            "graph_cid": graph_cid,
            "contract_cid": contract_cid,
            "forest_id": context.forest.forest_id,
            "requirement_id": context.profile.requirement_id,
        }
        receipt = {
            "schema": f"{context.profile.cache_schema}-receipt",
            "status": "success",
            "component": kind,
            "body_cid": _identity(body),
            "body": body,
        }
        stored.append(_identity(receipt))
    payload = {
        "schema": context.profile.cache_schema,
        "forest_id": context.forest.forest_id,
        "stored": stored,
        "component_kinds": list(kinds),
    }
    cache_cid = _identity(payload)
    context.cache_cid = cache_cid
    context.artifacts_partial["cache_cid"] = cache_cid
    return StageRunnerResult(
        artifact_cid=cache_cid,
        metrics={"stored": len(stored)},
        payload=payload,
        write_name="cache_receipt.json",
    )


def default_proof_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest", "proof requires freeze")
    inventory_cid = context.artifacts_partial.get("inventory_cid", "")
    graph_cid = context.artifacts_partial.get("graph_cid", "")
    contract_cid = context.contract_cid
    zk_payload = {
        "schema": context.profile.zk_shadow_schema,
        "forest_id": context.forest.forest_id,
        "backend_mode": "shadow",
        "authoritative": False,
        "public_input_digest": _identity(
            {
                "inventory_cid": inventory_cid,
                "graph_cid": graph_cid,
                "contract_cid": contract_cid,
            }
        ),
        "semantic_proof": False,
    }
    zk_shadow_cid = _identity(zk_payload)
    proof_payload = {
        "schema": context.profile.proof_schema,
        "forest_id": context.forest.forest_id,
        "zk_shadow_cid": zk_shadow_cid,
        "authoritative": False,
        "claim_level": "zk_trace_attested",
        "does_not_prove_semantics": True,
    }
    proof_cid = _identity(proof_payload)
    context.proof_cid = proof_cid
    context.zk_shadow_cid = zk_shadow_cid
    context.artifacts_partial["proof_cid"] = proof_cid
    context.artifacts_partial["zk_shadow_cid"] = zk_shadow_cid
    context.extras["zk_payload"] = zk_payload
    context.extras["proof_payload"] = proof_payload
    return StageRunnerResult(
        artifact_cid=proof_cid,
        reason_codes=("shadow_non_authoritative",),
        metrics={"authoritative": 0},
        payload=proof_payload,
        write_name="proof.json",
        extra_writes={"zk_shadow.json": zk_payload},
        extras={"zk_shadow_cid": zk_shadow_cid},
    )


def default_zk_shadow_runner(context: PilotStageContext) -> StageRunnerResult:
    zk_shadow_cid = context.zk_shadow_cid or context.artifacts_partial.get(
        "zk_shadow_cid", ""
    )
    if not zk_shadow_cid:
        # Allow standalone injection to still produce a shadow receipt.
        result = default_proof_runner(context)
        zk_shadow_cid = context.zk_shadow_cid
        # proof stage already wrote; zk stage just receipts the shadow cid
        _ = result
    payload = context.extras.get("zk_payload") or {
        "schema": context.profile.zk_shadow_schema,
        "forest_id": context.forest.forest_id if context.forest else "",
        "zk_shadow_cid": zk_shadow_cid,
        "authoritative": False,
    }
    return StageRunnerResult(
        artifact_cid=zk_shadow_cid,
        reason_codes=("shadow_non_authoritative",),
        metrics={"authoritative": 0},
        payload=payload,
        write_name="zk_shadow.json",
    )


def default_findings_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest", "findings requires freeze")
    findings = list(context.findings)
    admitted = [item for item in findings if item.executable]
    packets: list[dict[str, Any]] = []
    for index, finding in enumerate(admitted[: context.profile.max_board_tasks], start=1):
        packet = {
            "finding_cid": finding.finding_cid,
            "packet_id": f"R-{index:04d}",
            "content_id": _identity(
                {
                    "packet": index,
                    "finding": finding.finding_cid,
                    "path": finding.relative_path,
                }
            ),
            "status": "complete",
        }
        packets.append(packet)
    context.repair_packets = packets
    ledger_payload = {
        "schema": context.profile.findings_schema,
        "forest_id": context.forest.forest_id,
        "finding_cids": [item.finding_cid for item in findings],
        "admitted_cids": [item.finding_cid for item in admitted],
        "projection_cid": _identity(
            {"findings": [item.finding_cid for item in findings]}
        ),
        "repair_packets": packets,
    }
    finding_ledger_cid = _identity(ledger_payload)
    context.finding_ledger_cid = finding_ledger_cid
    context.artifacts_partial["finding_ledger_cid"] = finding_ledger_cid
    return StageRunnerResult(
        artifact_cid=finding_ledger_cid,
        metrics={
            "findings": len(findings),
            "admitted": len(admitted),
        },
        payload=ledger_payload,
        write_name="findings.json",
        findings=tuple(findings),
        finding_count=len(findings),
        admitted_count=len(admitted),
        repair_packets=tuple(packets),
    )


def default_taskboard_runner(context: PilotStageContext) -> StageRunnerResult:
    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest", "taskboard requires freeze")
    findings = list(context.findings)
    executable = [item for item in findings if item.executable]
    reviews = [item for item in findings if not item.executable]
    tasks = []
    for index, finding in enumerate(executable[: context.profile.max_board_tasks], start=1):
        tasks.append(
            {
                "task_id": f"{context.profile.task_id or 'PILOT'}-T-{index:04d}",
                "finding_cid": finding.finding_cid,
                "status": "admitted",
                "executable": True,
                "path": finding.relative_path,
                "summary": finding.summary,
            }
        )
    review_items = []
    for index, finding in enumerate(reviews[: context.profile.max_board_tasks], start=1):
        review_items.append(
            {
                "review_id": f"{context.profile.task_id or 'PILOT'}-R-{index:04d}",
                "finding_cid": finding.finding_cid,
                "status": "review",
                "executable": False,
                "path": finding.relative_path,
                "summary": finding.summary,
            }
        )
    board_json = {
        "schema": f"{context.profile.schema.rsplit('@', 1)[0]}-taskboard@1",
        "board_namespace": context.profile.board_namespace,
        "goal_id": context.profile.objective_id,
        "task_id": context.profile.task_id,
        "forest_id": context.forest.forest_id,
        "tasks": tasks,
        "reviews": review_items,
        "authorizes_repair": False,
        "is_completion_evidence": False,
        "provider_calls": 0,
        "source_mutations": 0,
    }
    board_md_lines = [
        f"# {context.profile.board_title}",
        "",
        f"goal_id: `{context.profile.objective_id}`",
        f"board_namespace: `{context.profile.board_namespace}`",
        f"tasks: `{len(tasks)}`",
        f"reviews: `{len(review_items)}`",
        "authorizes_repair: `false`",
        "is_completion_evidence: `false`",
        "",
    ]
    for task in tasks:
        board_md_lines.append(f"- [ ] {task['task_id']}: {task['summary']}")
    for review in review_items:
        board_md_lines.append(f"- [ ] review {review['review_id']}: {review['summary']}")
    board_markdown = "\n".join(board_md_lines) + "\n"
    taskboard_cid = _identity(board_json)
    board_markdown_cid = _identity({"markdown": board_markdown})
    context.board_json = board_json
    context.board_markdown = board_markdown
    context.board_markdown_cid = board_markdown_cid
    context.taskboard_cid = taskboard_cid
    context.executable_task_count = len(tasks)
    context.review_count = len(review_items)
    context.artifacts_partial["taskboard_cid"] = taskboard_cid
    return StageRunnerResult(
        artifact_cid=taskboard_cid,
        metrics={
            "executable": len(tasks),
            "reviews": len(review_items),
            "repair_packets": len(context.repair_packets),
        },
        payload=board_json,
        write_name="taskboard.json",
        board_json=board_json,
        board_markdown=board_markdown,
        board_markdown_cid=board_markdown_cid,
        executable_task_count=len(tasks),
        review_count=len(review_items),
        extra_writes={"taskboard.md": board_markdown},
    )


def default_publish_runner(context: PilotStageContext) -> StageRunnerResult:
    required = (
        "forest_cid",
        "manifest_cid",
        "coverage_cid",
        "inventory_cid",
        "graph_cid",
        "cache_cid",
        "proof_cid",
        "zk_shadow_cid",
        "finding_ledger_cid",
        "taskboard_cid",
    )
    missing = [name for name in required if not context.artifacts_partial.get(name)]
    if missing:
        raise SymbolicAssurancePilotError(
            "incomplete_inventory",
            f"publish missing artifact fields: {missing}",
        )
    artifacts = PilotArtifactSet(
        forest_cid=context.artifacts_partial["forest_cid"],
        manifest_cid=context.artifacts_partial["manifest_cid"],
        coverage_cid=context.artifacts_partial["coverage_cid"],
        inventory_cid=context.artifacts_partial["inventory_cid"],
        graph_cid=context.artifacts_partial["graph_cid"],
        cache_cid=context.artifacts_partial["cache_cid"],
        proof_cid=context.artifacts_partial["proof_cid"],
        zk_shadow_cid=context.artifacts_partial["zk_shadow_cid"],
        finding_ledger_cid=context.artifacts_partial["finding_ledger_cid"],
        taskboard_cid=context.artifacts_partial["taskboard_cid"],
        schema=context.profile.artifact_set_schema,
    )
    context.extras["artifacts"] = artifacts
    payload = artifacts.to_dict()
    return StageRunnerResult(
        artifact_cid=_identity(payload),
        metrics={"artifact_count": 10},
        payload=payload,
        write_name="artifacts.json",
    )


_DEFAULT_STAGE_RUNNERS: Final[Mapping[PilotStage, StageRunner]] = {
    PilotStage.FREEZE: default_freeze_runner,
    PilotStage.INVENTORY: default_inventory_runner,
    PilotStage.SCAN: default_scan_runner,
    PilotStage.GRAPH: default_graph_runner,
    PilotStage.CONTRACT: default_contract_runner,
    PilotStage.CACHE: default_cache_runner,
    PilotStage.PROOF: default_proof_runner,
    PilotStage.ZK_SHADOW: default_zk_shadow_runner,
    PilotStage.FINDINGS: default_findings_runner,
    PilotStage.TASKBOARD: default_taskboard_runner,
    PilotStage.PUBLISH: default_publish_runner,
}


def _resolve_stage_runner(config: PilotConfig, stage: PilotStage) -> StageRunner:
    if config.stage_runners:
        for key in (stage.value, stage.name, stage):
            runner = config.stage_runners.get(str(key))  # type: ignore[arg-type]
            if runner is not None:
                return runner
    return _DEFAULT_STAGE_RUNNERS[stage]


def _apply_stage_result(
    context: PilotStageContext,
    stage: PilotStage,
    result: StageRunnerResult,
) -> StageReceipt:
    if stage in {item.stage for item in context.stages}:
        raise SymbolicAssurancePilotError(
            "duplicate_stage",
            f"stage {stage.value} already executed",
        )
    if result.status is PilotConclusion.FAILED:
        context.reason_codes.extend(result.reason_codes)
    if result.findings:
        context.findings = list(result.findings)
    if result.inconclusive_count:
        context.inconclusive_count = result.inconclusive_count
    if result.executable_task_count:
        context.executable_task_count = result.executable_task_count
    if result.review_count:
        context.review_count = result.review_count
    if result.board_json is not None:
        context.board_json = dict(result.board_json)
    if result.board_markdown:
        context.board_markdown = result.board_markdown
    if result.board_markdown_cid:
        context.board_markdown_cid = result.board_markdown_cid
    if result.repair_packets:
        context.repair_packets = [dict(item) for item in result.repair_packets]
    if result.graph is not None:
        context.graph = result.graph
    if result.extras:
        context.extras.update(result.extras)

    input_cids: list[str] = []
    if context.stages:
        input_cids.append(context.stages[-1].artifact_cid)

    receipt = StageReceipt(
        stage=stage,
        status=result.status,
        artifact_cid=result.artifact_cid,
        input_cids=tuple(input_cids),
        reason_codes=tuple(result.reason_codes),
        metrics=dict(result.metrics or {}),
        schema=context.profile.stage_receipt_schema,
    )
    context.stages.append(receipt)

    if context.artifact_dir is not None:
        if result.write_name and result.payload is not None:
            target = context.artifact_dir / result.write_name
            if isinstance(result.payload, str):
                _atomic_write_text(target, result.payload)
            else:
                _atomic_write_json(target, result.payload)
        for name, payload in dict(result.extra_writes or {}).items():
            target = context.artifact_dir / name
            if isinstance(payload, str):
                _atomic_write_text(target, payload)
            else:
                _atomic_write_json(target, payload)
    return receipt


# ---------------------------------------------------------------------------
# Board rendering / orchestration / verification
# ---------------------------------------------------------------------------


def render_findings_board_document(
    *,
    profile: PilotProgramProfile,
    report_context: Mapping[str, Any],
    taskboard_markdown: str = "",
) -> str:
    """Render the durable findings board markdown from profile identities."""

    artifacts = dict(report_context.get("artifacts") or {})
    packets = list(report_context.get("repair_packets") or [])
    lines = [
        f"# {profile.board_title}",
        "",
        f"Generated by `{profile.producer}` ({profile.task_id} / {profile.objective_id}).",
        "This board is diagnostic and **does not authorize repair or completion**.",
        "",
        "## Pilot receipt",
        "",
        f"- objective_id: `{profile.objective_id}`",
        f"- task_id: `{profile.task_id}`",
        f"- evidence: `{profile.evidence}`",
        f"- board_namespace: `{profile.board_namespace}`",
        f"- mode: `{report_context.get('mode', '')}`",
        f"- conclusion: `{report_context.get('conclusion', '')}`",
        f"- forest_id: `{report_context.get('forest_id', '')}`",
        f"- admitted_file_count: `{report_context.get('admitted_file_count', 0)}`",
        f"- primary_file_count: `{report_context.get('primary_file_count', 0)}`",
        f"- closure_file_count: `{report_context.get('closure_file_count', 0)}`",
        f"- finding_count: `{report_context.get('finding_count', 0)}`",
        f"- executable_task_count: `{report_context.get('executable_task_count', 0)}`",
        f"- review_count: `{report_context.get('review_count', 0)}`",
        f"- provider_calls: `0`",
        f"- source_mutations: `0`",
        f"- authorizes_repair: `false`",
        f"- is_completion_evidence: `false`",
        "",
        "## Published artifact CIDs",
        "",
    ]
    for key in (
        "forest_cid",
        "manifest_cid",
        "coverage_cid",
        "inventory_cid",
        "graph_cid",
        "cache_cid",
        "proof_cid",
        "zk_shadow_cid",
        "finding_ledger_cid",
        "taskboard_cid",
        "report_cid",
    ):
        value = artifacts.get(key, "")
        if value:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Repair packets", ""])
    if packets:
        for packet in packets[:MAX_BOARD_TASKS]:
            lines.append(
                f"- `{packet.get('packet_id', '')}` finding=`{packet.get('finding_cid', '')}` "
                f"status=`{packet.get('status', '')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Taskboard projection", ""])
    if taskboard_markdown.strip():
        lines.append(taskboard_markdown.rstrip())
    else:
        lines.append("_empty taskboard_")
    lines.append("")
    return "\n".join(lines)


def execute_pilot(
    config: PilotConfig,
    *,
    mode: PilotMode = PilotMode.DRY_RUN,
) -> SymbolicAssurancePilotReport:
    """Run the full deterministic pilot pipeline and return the report."""

    _assert_no_provider_surface()
    if mode is PilotMode.VERIFY:
        raise SymbolicAssurancePilotError(
            "invalid_mode",
            "execute_pilot does not perform verify; use verify_pilot",
        )
    if not isinstance(config, PilotConfig):
        raise SymbolicAssurancePilotError("invalid_config", "PilotConfig required")

    artifact_dir: Path | None = None
    if config.write_artifacts:
        artifact_dir = config.resolved_artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)

    context = PilotStageContext(
        config=config,
        mode=mode,
        profile=config.profile,
        artifact_dir=artifact_dir,
    )

    stage_order = (
        PilotStage.FREEZE,
        PilotStage.INVENTORY,
        PilotStage.SCAN,
        PilotStage.GRAPH,
        PilotStage.CONTRACT,
        PilotStage.CACHE,
        PilotStage.PROOF,
        PilotStage.ZK_SHADOW,
        PilotStage.FINDINGS,
        PilotStage.TASKBOARD,
        PilotStage.PUBLISH,
    )
    for stage in stage_order:
        runner = _resolve_stage_runner(config, stage)
        result = runner(context)
        if not isinstance(result, StageRunnerResult):
            raise SymbolicAssurancePilotError(
                "invalid_stage_result",
                f"stage {stage.value} runner must return StageRunnerResult",
            )
        if result.status is PilotConclusion.FAILED:
            _apply_stage_result(context, stage, result)
            raise SymbolicAssurancePilotError(
                "stage_failed",
                f"stage {stage.value} failed: {list(result.reason_codes)}",
            )
        _apply_stage_result(context, stage, result)

    if context.forest is None:
        raise SymbolicAssurancePilotError("missing_forest")
    artifacts = context.extras.get("artifacts")
    if not isinstance(artifacts, PilotArtifactSet):
        raise SymbolicAssurancePilotError("incomplete_inventory", "missing artifact set")

    tree_bindings = {
        descriptor.alias: descriptor.tree for descriptor in context.forest.descriptors
    }
    commit_bindings = {
        descriptor.alias: descriptor.commit for descriptor in context.forest.descriptors
    }

    if context.incomplete_inventory:
        conclusion = PilotConclusion.INCOMPLETE
    else:
        conclusion = PilotConclusion.PASSED

    primary_count = int(context.coverage_bundle.get("primary_file_count") or 0)
    closure_count = int(context.coverage_bundle.get("closure_file_count") or 0)

    board_md = render_findings_board_document(
        profile=config.profile,
        report_context={
            "forest_id": context.forest.forest_id,
            "artifacts": artifacts.to_dict(),
            "executable_task_count": context.executable_task_count,
            "review_count": context.review_count,
            "finding_count": len(context.findings),
            "admitted_file_count": len(context.admitted),
            "primary_file_count": primary_count,
            "closure_file_count": closure_count,
            "mode": mode.value,
            "conclusion": conclusion.value,
            "repair_packets": context.repair_packets,
        },
        taskboard_markdown=context.board_markdown,
    )
    board_markdown_cid = _identity({"markdown": board_md})

    if config.write_findings_board:
        board_path = config.resolved_findings_board_path()
        if len(board_md.encode("utf-8")) > MAX_FINDINGS_BOARD_BYTES:
            raise SymbolicAssurancePilotError(
                "board_bound_exceeded",
                f"findings board exceeds {MAX_FINDINGS_BOARD_BYTES} bytes",
            )
        _atomic_write_text(board_path, board_md)

    report = SymbolicAssurancePilotReport(
        schema=config.profile.schema,
        version=config.profile.version,
        objective_id=config.profile.objective_id,
        task_id=config.profile.task_id,
        requirement_id=config.profile.requirement_id,
        mode=mode,
        conclusion=conclusion,
        forest_id=context.forest.forest_id,
        tree_bindings=tree_bindings,
        commit_bindings=commit_bindings,
        stages=tuple(context.stages),
        artifacts=artifacts,
        admitted_file_count=len(context.admitted),
        primary_file_count=primary_count,
        closure_file_count=closure_count,
        finding_count=len(context.findings),
        executable_task_count=context.executable_task_count,
        review_count=context.review_count,
        inconclusive_count=context.inconclusive_count,
        provider_calls=0,
        source_mutations=0,
        reason_codes=tuple(sorted(set(context.reason_codes))),
        board_markdown_cid=board_markdown_cid,
        board_namespace=config.profile.board_namespace,
        policy_revision=config.profile.policy_revision,
        evidence=config.profile.evidence,
        profile_cid=config.profile.profile_cid,
    )
    published_artifacts = PilotArtifactSet(
        forest_cid=artifacts.forest_cid,
        manifest_cid=artifacts.manifest_cid,
        coverage_cid=artifacts.coverage_cid,
        inventory_cid=artifacts.inventory_cid,
        graph_cid=artifacts.graph_cid,
        cache_cid=artifacts.cache_cid,
        proof_cid=artifacts.proof_cid,
        zk_shadow_cid=artifacts.zk_shadow_cid,
        finding_ledger_cid=artifacts.finding_ledger_cid,
        taskboard_cid=artifacts.taskboard_cid,
        report_cid=report.report_cid,
        schema=config.profile.artifact_set_schema,
    )

    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "artifacts.json", published_artifacts.to_dict())
        _atomic_write_json(artifact_dir / "report.json", report.to_dict())
        _atomic_write_text(artifact_dir / "findings_board.md", board_md)

    _assert_no_provider_surface()
    return report


def dry_run_pilot(config: PilotConfig) -> SymbolicAssurancePilotReport:
    """Dry-run mode: full pipeline, no provider calls, no source mutation."""

    return execute_pilot(config, mode=PilotMode.DRY_RUN)


def verify_pilot_report(
    report: SymbolicAssurancePilotReport | Mapping[str, Any],
    *,
    config: PilotConfig | None = None,
    recompute: bool = True,
) -> SymbolicAssurancePilotReport:
    """Verify a pilot report without provider calls or source mutation.

    When ``recompute`` is true and ``config`` is supplied, the forest is frozen
    again and tree/commit bindings must match the report.  Inventory
    completeness, artifact canonicality, and zero provider/mutation counters
    are always enforced.
    """

    _assert_no_provider_surface()
    if isinstance(report, Mapping):
        report = SymbolicAssurancePilotReport.from_dict(report)
    if not isinstance(report, SymbolicAssurancePilotReport):
        raise PilotVerificationError("invalid_report", "report type is invalid")

    if report.provider_calls != 0:
        raise PilotVerificationError("provider_call_forbidden")
    if report.source_mutations != 0:
        raise PilotVerificationError("source_mutation_forbidden")
    if report.authorizes_repair or report.is_completion_evidence:
        raise PilotVerificationError("authority_drift")
    if report.artifacts is None:
        raise PilotVerificationError("incomplete_inventory", "missing artifact set")

    reloaded = SymbolicAssurancePilotReport.from_dict(report.to_dict())
    if reloaded.report_cid != report.report_cid:
        raise PilotVerificationError(
            "noncanonical_artifact",
            "report is not canonical under re-encode",
        )

    required_stages = {stage for stage in PilotStage}
    observed_stages = [item.stage for item in report.stages]
    if len(observed_stages) != len(set(observed_stages)):
        raise PilotVerificationError(
            "duplicate_stage",
            "report contains duplicate stages",
        )
    missing = required_stages - set(observed_stages)
    if missing:
        raise PilotVerificationError(
            "incomplete_inventory",
            f"missing stages: {sorted(item.value for item in missing)}",
        )

    for stage in report.stages:
        if not stage.artifact_cid:
            raise PilotVerificationError(
                "stale_evidence",
                f"stage {stage.stage.value} missing artifact CID",
            )
        if StageReceipt.from_dict(stage.to_dict()).receipt_cid != stage.receipt_cid:
            raise PilotVerificationError(
                "noncanonical_artifact",
                f"stage {stage.stage.value} is non-canonical",
            )

    if recompute and config is not None:
        live = freeze_repository_descriptors(config)
        if live.forest_id != report.forest_id:
            raise PilotVerificationError(
                "changed_trees",
                "live forest_id does not match frozen pilot report",
            )
        live_trees = {
            descriptor.alias: descriptor.tree for descriptor in live.descriptors
        }
        live_commits = {
            descriptor.alias: descriptor.commit for descriptor in live.descriptors
        }
        for alias, tree in report.tree_bindings.items():
            if live_trees.get(alias) != tree:
                raise PilotVerificationError(
                    "changed_trees",
                    f"tree for {alias!r} changed since pilot freeze",
                )
        for alias, commit in report.commit_bindings.items():
            if live_commits.get(alias) != commit:
                raise PilotVerificationError(
                    "changed_trees",
                    f"commit for {alias!r} changed since pilot freeze",
                )

        limits = config.inventory_limits or InventoryLimits()
        index = scan_inventory(live, limits=limits)
        if index.inventory_cid != report.artifacts.inventory_cid:
            raise PilotVerificationError(
                "stale_evidence",
                "inventory_cid drifted under recompute",
            )
        required = set(config.require_exhaustive_aliases)
        for repo in index.repositories:
            if repo.repository_alias in required and (
                not repo.exhaustive or repo.omitted_entry_count
            ):
                raise PilotVerificationError(
                    "incomplete_inventory",
                    f"required repository {repo.repository_alias!r} inventory incomplete",
                )
            if not repo.exhaustive and report.conclusion == PilotConclusion.PASSED:
                raise PilotVerificationError(
                    "incomplete_inventory",
                    f"repository {repo.repository_alias!r} inventory incomplete",
                )

        verify_config = config.without_writes()
        recomputed = execute_pilot(verify_config, mode=PilotMode.DRY_RUN)
        if recomputed.forest_id != report.forest_id:
            raise PilotVerificationError("changed_trees", "recomputed forest drifted")
        if recomputed.artifacts is None:
            raise PilotVerificationError(
                "incomplete_inventory", "recompute missing artifacts"
            )
        for field_name in (
            "manifest_cid",
            "coverage_cid",
            "inventory_cid",
            "graph_cid",
            "cache_cid",
            "proof_cid",
            "zk_shadow_cid",
            "finding_ledger_cid",
            "taskboard_cid",
        ):
            expected = getattr(report.artifacts, field_name)
            observed = getattr(recomputed.artifacts, field_name)
            if expected != observed:
                raise PilotVerificationError(
                    "stale_evidence",
                    f"{field_name} is not reproducible "
                    f"(expected {expected}, got {observed})",
                )

    verified_payload = dict(report.to_dict())
    verified_payload.pop("report_cid", None)
    verified_payload["mode"] = PilotMode.VERIFY.value
    return SymbolicAssurancePilotReport.from_dict(verified_payload)


def verify_pilot(
    config: PilotConfig,
    *,
    report_path: Path | None = None,
) -> SymbolicAssurancePilotReport:
    """Verify mode entrypoint.

    When ``report_path`` is provided, load and verify that report against the
    live forest.  Otherwise dry-run into a temporary artifact directory and
    verify the freshly produced report (hermetic self-check).
    """

    if report_path is not None:
        payload = _load_json(Path(report_path))
        return verify_pilot_report(payload, config=config, recompute=True)

    with tempfile.TemporaryDirectory(prefix="symbolic-assurance-pilot-verify-") as tmp:
        tmp_path = Path(tmp)
        allowed = config.allowed_output_roots or (tmp_path,)
        run_config = PilotConfig(
            profile=config.profile,
            admission_policy=config.admission_policy,
            repositories=config.repositories,
            forest_policy=config.forest_policy,
            forest_builder=config.forest_builder,
            stage_runners=config.stage_runners,
            artifact_dir=tmp_path / "artifacts",
            findings_board_path=tmp_path / "findings.todo.md",
            inventory_limits=config.inventory_limits,
            max_admitted_parse=config.max_admitted_parse,
            write_artifacts=True,
            write_findings_board=True,
            require_exhaustive_aliases=config.require_exhaustive_aliases,
            allowed_output_roots=tuple(allowed) + (tmp_path,),
            sole_write_alias=config.sole_write_alias,
        )
        report = dry_run_pilot(run_config)
        return verify_pilot_report(report, config=run_config, recompute=True)


__all__ = [
    "MAX_ADMITTED_PARSE",
    "MAX_FINDINGS_BOARD_BYTES",
    "MAX_REPORT_BYTES",
    "ForestBuilder",
    "PilotArtifactSet",
    "PilotConclusion",
    "PilotConfig",
    "PilotFinding",
    "PilotMode",
    "PilotProgramProfile",
    "PilotStage",
    "PilotStageContext",
    "PilotVerificationError",
    "RepositoryAdmissionPolicy",
    "StageReceipt",
    "StageRunner",
    "StageRunnerResult",
    "SymbolicAssurancePilotError",
    "SymbolicAssurancePilotReport",
    "admitted_entries_for_pilot",
    "build_coverage_manifest",
    "build_pilot_program_graph",
    "default_cache_runner",
    "default_contract_runner",
    "default_findings_runner",
    "default_freeze_runner",
    "default_graph_runner",
    "default_inventory_runner",
    "default_proof_runner",
    "default_publish_runner",
    "default_scan_runner",
    "default_taskboard_runner",
    "default_zk_shadow_runner",
    "dry_run_pilot",
    "execute_pilot",
    "freeze_repository_descriptors",
    "render_findings_board_document",
    "scan_inventory",
    "verify_pilot",
    "verify_pilot_report",
]
