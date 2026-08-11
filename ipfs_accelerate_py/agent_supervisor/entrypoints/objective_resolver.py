"""Objective, plan, task-source, and output resolution (ASE-007).

This module owns objective/task-source *selection and default construction*
for prompt-only entrypoints.  It reuses content-addressed identity helpers and
task-source kind vocabulary without materializing projections, editing boards,
or executing tasks.

Design rules enforced here:

- exact integrity-checked run bindings win over discovery and defaults;
- absent unique intent creates a content-addressed prompt objective (and a
  pending plan identity) rather than guessing a board by filename or title;
- multiple integrity-checked compatible objectives or boards are reported as
  explicit ambiguities without silent selection;
- DuckDB plus Markdown mirror (dual) is preferred when DuckDB capability is
  available; Markdown-only degradation is typed and explicit;
- projection outputs default under the platform state root and never dirty the
  source repository checkout unless a separately authorized override is
  present (and even then paths are never accepted from prompt text);
- prompt prose, board titles, and untrusted path labels are non-authoritative.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

from .contracts import (
    DecisionEffect,
    EntrypointContractError,
    OutputMode,
    ResolutionDisposition,
    ResolutionSource,
    RevalidationRule,
    TargetCandidate,
    TargetInferenceDecision,
    TaskSourceKind,
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
OBJECTIVE_EVIDENCE_SCHEMA: Final = f"{SCHEMA_PREFIX}/objective-evidence@1"
OBJECTIVE_RESOLUTION_SCHEMA: Final = f"{SCHEMA_PREFIX}/objective-resolution@1"
OBJECTIVE_CANDIDATE_SCHEMA: Final = f"{SCHEMA_PREFIX}/objective-candidate@1"
TASK_SOURCE_CANDIDATE_SCHEMA: Final = f"{SCHEMA_PREFIX}/task-source-candidate@1"
RUN_OBJECTIVE_BINDING_SCHEMA: Final = f"{SCHEMA_PREFIX}/run-objective-binding@1"
PROMPT_OBJECTIVE_SCHEMA: Final = f"{SCHEMA_PREFIX}/prompt-objective@1"
PROMPT_OBJECTIVE_REVISION_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/prompt-objective-revision@1"
)
PROMPT_PLAN_PLACEHOLDER_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/prompt-plan-placeholder@1"
)
DEFAULT_TASK_SOURCE_IDENTITY_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/default-task-source-identity@1"
)
OUTPUT_POLICY_SCHEMA: Final = f"{SCHEMA_PREFIX}/output-policy@1"

OBJECTIVE_AND_TASK_SOURCE_RESOLUTION_REQUIREMENT_ID: Final = (
    "agent_supervisor.entrypoints.objective_resolver.v1"
)

OBJECTIVE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "objective",
    "plan",
    "task_source",
    "output",
)

DEFAULT_MARKDOWN_RELATIVE: Final = "projections/tasks.md"
DEFAULT_DUCKDB_RELATIVE: Final = "projections/tasks.duckdb"

# Source precedence mirrors the prompt-entrypoint plan (lower wins).
SOURCE_PRECEDENCE: Final[Mapping[ResolutionSource, int]] = {
    ResolutionSource.CANONICAL_REQUEST: 10,
    ResolutionSource.EXPLICIT_OVERRIDE: 20,
    ResolutionSource.EXISTING_RUN: 30,
    ResolutionSource.AUTHENTICATED_TRANSPORT: 40,
    ResolutionSource.SIGNED_PROFILE: 50,
    ResolutionSource.REPOSITORY_HINT: 70,
    ResolutionSource.DISCOVERY: 80,
    ResolutionSource.BUILTIN_DEFAULT: 90,
}

_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_PATH_INJECTION_MARKERS: Final[tuple[str, ...]] = (
    "\n",
    "\r",
    "\x00",
    "\\",
)


class ObjectiveResolverError(EntrypointContractError):
    """Raised when objective/task-source evidence is malformed or unsafe."""


class OutputDegradationCode(str, Enum):
    """Typed optional degradation of the preferred dual projection policy."""

    NONE = "none"
    DUCKDB_UNAVAILABLE = "duckdb_unavailable"
    MARKDOWN_ONLY = "markdown_only"
    EXPLICIT_MARKDOWN = "explicit_markdown"
    EXPLICIT_DUCKDB = "explicit_duckdb"


class TaskSourceSelectionAction(str, Enum):
    """Closed task-source selection outcome."""

    BIND_EXISTING = "bind_existing"
    CREATE_DEFAULT = "create_default"
    REPORT_AMBIGUOUS = "report_ambiguous"
    DENIED = "denied"


def _require_nonempty(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ObjectiveResolverError(f"{name} is required")
    return text


def _require_cid(value: Any, name: str) -> str:
    text = _require_nonempty(value, name)
    # Accept multiformats CIDs (lowercase base32) or short fixture tokens that
    # still look identity-like. TargetInferenceDecision validation still
    # requires real CIDv1 for decision evidence_cid fields.
    if not re.fullmatch(r"[a-z0-9]{8,}", text) and not re.fullmatch(
        r"[A-Za-z0-9:._+/-]{8,}", text
    ):
        raise ObjectiveResolverError(f"{name} is not a valid identity")
    return text


def _token(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not text or not _TOKEN_RE.fullmatch(text):
        raise ObjectiveResolverError(f"{name} must be a closed token")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ObjectiveResolverError(f"{name} must be a boolean")
    return value


def _absolute_posix_path(value: Any, name: str) -> str:
    text = _require_nonempty(value, name)
    if any(marker in text for marker in _PATH_INJECTION_MARKERS):
        raise ObjectiveResolverError(f"{name} contains forbidden path characters")
    if not text.startswith("/") or "\\" in text:
        raise ObjectiveResolverError(f"{name} must be an absolute POSIX path")
    normalized = posixpath.normpath(text)
    if normalized != text or any(part == ".." for part in text.split("/")):
        raise ObjectiveResolverError(f"{name} must be lexically normalized")
    if text == "/":
        raise ObjectiveResolverError(f"{name} cannot be the filesystem root")
    return text


def _optional_absolute_path(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _absolute_posix_path(text, name)


def _is_path_under(path: str, root: str) -> bool:
    """Return True when ``path`` is ``root`` or a descendant of ``root``."""

    if not path or not root:
        return False
    try:
        common = posixpath.commonpath((path, root))
    except ValueError:
        return False
    return common == root


def _enum_member(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        raise ObjectiveResolverError(
            f"{name} must be one of {[item.value for item in enum_type]}"
        ) from exc


def content_addressed_prompt_objective(
    prompt_cid: str,
    *,
    repository_id: str = "",
    run_namespace: str = "",
) -> tuple[str, str, str]:
    """Construct objective, revision, and pending-plan CIDs from prompt intent.

    The identities are pure functions of the prompt content address (plus
    optional repository/run binding) so identical prompts replay to the same
    objective identity without consulting board titles or filenames.
    """

    prompt = _require_cid(prompt_cid, "prompt_cid")
    objective_payload: dict[str, Any] = {
        "schema": PROMPT_OBJECTIVE_SCHEMA,
        "prompt_cid": prompt,
        "kind": "prompt_intent",
        "requirement_id": OBJECTIVE_AND_TASK_SOURCE_RESOLUTION_REQUIREMENT_ID,
    }
    if repository_id:
        objective_payload["repository_id"] = str(repository_id)
    if run_namespace:
        objective_payload["run_namespace"] = str(run_namespace)
    objective_cid = cid_for_dag_json(objective_payload)
    revision_payload = {
        "schema": PROMPT_OBJECTIVE_REVISION_SCHEMA,
        "objective_cid": objective_cid,
        "revision_index": 1,
        "prompt_cid": prompt,
    }
    objective_revision_cid = cid_for_dag_json(revision_payload)
    plan_payload = {
        "schema": PROMPT_PLAN_PLACEHOLDER_SCHEMA,
        "objective_revision_cid": objective_revision_cid,
        "status": "pending_materialization",
    }
    plan_cid = cid_for_dag_json(plan_payload)
    return objective_cid, objective_revision_cid, plan_cid


def default_projection_paths(state_root: str) -> tuple[str, str]:
    """Return default Markdown and DuckDB projection paths under ``state_root``."""

    root = _absolute_posix_path(state_root, "state_root")
    markdown = f"{root}/{DEFAULT_MARKDOWN_RELATIVE}"
    duckdb = f"{root}/{DEFAULT_DUCKDB_RELATIVE}"
    return markdown, duckdb


def default_task_source_identity(
    *,
    state_root: str,
    kind: TaskSourceKind,
    markdown_path: str,
    duckdb_path: str,
    objective_revision_cid: str,
) -> tuple[str, str]:
    """Content-address a default task-source identity and initial revision."""

    payload = {
        "schema": DEFAULT_TASK_SOURCE_IDENTITY_SCHEMA,
        "state_root": state_root,
        "kind": kind.value,
        "markdown_path": markdown_path,
        "duckdb_path": duckdb_path,
        "objective_revision_cid": objective_revision_cid,
    }
    task_source_cid = cid_for_dag_json(payload)
    revision_cid = cid_for_dag_json(
        {
            "schema": f"{SCHEMA_PREFIX}/default-task-source-revision@1",
            "task_source_cid": task_source_cid,
            "revision_index": 1,
        }
    )
    return task_source_cid, revision_cid


def _candidate(
    *,
    field_name: str,
    value: str,
    source: ResolutionSource,
    evidence_cid: str,
    confidence_ppm: int = 1_000_000,
    rejection_reason: str = "",
) -> TargetCandidate:
    return TargetCandidate(
        field_name=field_name,
        value=value,
        source=source,
        source_precedence=SOURCE_PRECEDENCE[source],
        evidence_cid=evidence_cid,
        confidence_ppm=confidence_ppm,
        rejection_reason=rejection_reason,
    )


def _decision(
    *,
    field_name: str,
    disposition: ResolutionDisposition,
    selected_value: str,
    selected_source: ResolutionSource,
    evidence_cid: str,
    candidates: Sequence[TargetCandidate],
    reason_codes: Sequence[str],
    effect: DecisionEffect,
    override_accepted: bool = False,
    revalidation_rule: RevalidationRule = RevalidationRule.BEFORE_MUTATION,
) -> TargetInferenceDecision:
    return TargetInferenceDecision(
        field_name=field_name,
        disposition=disposition,
        selected_value=selected_value,
        selected_source=selected_source,
        source_precedence=SOURCE_PRECEDENCE[selected_source],
        evidence_cid=evidence_cid,
        candidates=tuple(candidates),
        reason_codes=tuple(reason_codes),
        effect=effect,
        override_accepted=override_accepted,
        fresh_until_ms=0,
        revalidation_rule=revalidation_rule,
    )


@dataclass(frozen=True)
class ObjectiveCandidateEvidence:
    """One integrity-checked (or explicitly unverified) objective observation.

    Titles and board filenames are accepted only for diagnostics and never
    participate in selection identity.
    """

    objective_cid: str
    objective_revision_cid: str
    plan_cid: str = ""
    board_id: str = ""
    title: str = ""
    integrity_verified: bool = True
    active: bool = True
    compatible: bool = True
    run_bound: bool = False
    evidence_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_cid",
            _require_cid(self.objective_cid, "objective_cid"),
        )
        object.__setattr__(
            self,
            "objective_revision_cid",
            _require_cid(self.objective_revision_cid, "objective_revision_cid"),
        )
        plan = str(self.plan_cid or "").strip()
        object.__setattr__(
            self, "plan_cid", _require_cid(plan, "plan_cid") if plan else ""
        )
        object.__setattr__(self, "board_id", str(self.board_id or "").strip())
        object.__setattr__(self, "title", str(self.title or "").strip())
        object.__setattr__(
            self,
            "integrity_verified",
            _bool(self.integrity_verified, "integrity_verified"),
        )
        object.__setattr__(self, "active", _bool(self.active, "active"))
        object.__setattr__(
            self, "compatible", _bool(self.compatible, "compatible")
        )
        object.__setattr__(self, "run_bound", _bool(self.run_bound, "run_bound"))
        evidence = str(self.evidence_cid or "").strip()
        if evidence:
            object.__setattr__(
                self, "evidence_cid", _require_cid(evidence, "evidence_cid")
            )
        else:
            object.__setattr__(
                self,
                "evidence_cid",
                cid_for_dag_json(
                    {
                        "schema": OBJECTIVE_CANDIDATE_SCHEMA,
                        "objective_cid": self.objective_cid,
                        "objective_revision_cid": self.objective_revision_cid,
                        "plan_cid": self.plan_cid,
                    }
                ),
            )

    @property
    def viable(self) -> bool:
        return (
            self.integrity_verified
            and self.active
            and self.compatible
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBJECTIVE_CANDIDATE_SCHEMA,
            "objective_cid": self.objective_cid,
            "objective_revision_cid": self.objective_revision_cid,
            "plan_cid": self.plan_cid,
            "board_id": self.board_id,
            "integrity_verified": self.integrity_verified,
            "active": self.active,
            "compatible": self.compatible,
            "run_bound": self.run_bound,
            "evidence_cid": self.evidence_cid,
            # title deliberately omitted from durable identity
        }


@dataclass(frozen=True)
class TaskSourceCandidateEvidence:
    """One integrity-checked task-source / board observation."""

    task_source_cid: str
    task_source_revision_cid: str
    kind: TaskSourceKind
    path: str = ""
    markdown_path: str = ""
    duckdb_path: str = ""
    integrity_verified: bool = True
    compatible: bool = True
    run_bound: bool = False
    board_filename: str = ""
    evidence_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_source_cid",
            _require_cid(self.task_source_cid, "task_source_cid"),
        )
        object.__setattr__(
            self,
            "task_source_revision_cid",
            _require_cid(
                self.task_source_revision_cid, "task_source_revision_cid"
            ),
        )
        object.__setattr__(
            self, "kind", _enum_member(self.kind, TaskSourceKind, "kind")
        )
        for name in ("path", "markdown_path", "duckdb_path"):
            object.__setattr__(
                self, name, _optional_absolute_path(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "integrity_verified",
            _bool(self.integrity_verified, "integrity_verified"),
        )
        object.__setattr__(
            self, "compatible", _bool(self.compatible, "compatible")
        )
        object.__setattr__(self, "run_bound", _bool(self.run_bound, "run_bound"))
        object.__setattr__(
            self, "board_filename", str(self.board_filename or "").strip()
        )
        evidence = str(self.evidence_cid or "").strip()
        if evidence:
            object.__setattr__(
                self, "evidence_cid", _require_cid(evidence, "evidence_cid")
            )
        else:
            object.__setattr__(
                self,
                "evidence_cid",
                cid_for_dag_json(
                    {
                        "schema": TASK_SOURCE_CANDIDATE_SCHEMA,
                        "task_source_cid": self.task_source_cid,
                        "task_source_revision_cid": (
                            self.task_source_revision_cid
                        ),
                        "kind": self.kind.value,
                        "path": self.path,
                    }
                ),
            )

    @property
    def viable(self) -> bool:
        return self.integrity_verified and self.compatible

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_CANDIDATE_SCHEMA,
            "task_source_cid": self.task_source_cid,
            "task_source_revision_cid": self.task_source_revision_cid,
            "kind": self.kind.value,
            "path": self.path,
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "integrity_verified": self.integrity_verified,
            "compatible": self.compatible,
            "run_bound": self.run_bound,
            "board_filename": self.board_filename,
            "evidence_cid": self.evidence_cid,
        }


@dataclass(frozen=True)
class RunObjectiveBinding:
    """Exact run-bound objective, plan, task-source, and output identities.

    When integrity-checked, this binding is authoritative for a new
    status/steer/follow invocation against that run.
    """

    run_id: str
    objective_cid: str
    objective_revision_cid: str
    plan_cid: str
    task_source_cid: str
    task_source_revision_cid: str
    task_source_kind: TaskSourceKind
    output_mode: OutputMode
    markdown_path: str = ""
    duckdb_path: str = ""
    integrity_verified: bool = True
    evidence_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_cid(self.run_id, "run_id"))
        for name in (
            "objective_cid",
            "objective_revision_cid",
            "plan_cid",
            "task_source_cid",
            "task_source_revision_cid",
        ):
            object.__setattr__(
                self, name, _require_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "task_source_kind",
            _enum_member(self.task_source_kind, TaskSourceKind, "task_source_kind"),
        )
        object.__setattr__(
            self,
            "output_mode",
            _enum_member(self.output_mode, OutputMode, "output_mode"),
        )
        for name in ("markdown_path", "duckdb_path"):
            object.__setattr__(
                self, name, _optional_absolute_path(getattr(self, name), name)
            )
        mode = self.output_mode
        if mode in {OutputMode.MARKDOWN, OutputMode.BOTH} and not self.markdown_path:
            raise ObjectiveResolverError(
                "run binding output mode requires markdown_path"
            )
        if mode in {OutputMode.DUCKDB, OutputMode.BOTH} and not self.duckdb_path:
            raise ObjectiveResolverError(
                "run binding output mode requires duckdb_path"
            )
        object.__setattr__(
            self,
            "integrity_verified",
            _bool(self.integrity_verified, "integrity_verified"),
        )
        evidence = str(self.evidence_cid or "").strip()
        if evidence:
            object.__setattr__(
                self, "evidence_cid", _require_cid(evidence, "evidence_cid")
            )
        else:
            object.__setattr__(
                self,
                "evidence_cid",
                cid_for_dag_json(
                    {
                        "schema": RUN_OBJECTIVE_BINDING_SCHEMA,
                        "run_id": self.run_id,
                        "objective_cid": self.objective_cid,
                        "task_source_cid": self.task_source_cid,
                    }
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_OBJECTIVE_BINDING_SCHEMA,
            "run_id": self.run_id,
            "objective_cid": self.objective_cid,
            "objective_revision_cid": self.objective_revision_cid,
            "plan_cid": self.plan_cid,
            "task_source_cid": self.task_source_cid,
            "task_source_revision_cid": self.task_source_revision_cid,
            "task_source_kind": self.task_source_kind.value,
            "output_mode": self.output_mode.value,
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "integrity_verified": self.integrity_verified,
            "evidence_cid": self.evidence_cid,
        }


@dataclass(frozen=True)
class ObjectiveResolutionEvidence:
    """Frozen evidence for deterministic objective/plan/task-source/output resolution.

    Prompt text is accepted only to prove it cannot influence selection and is
    excluded from the evidence content identity.
    """

    repository_root: str
    state_root: str
    prompt_cid: str
    repository_id: str = ""
    run_namespace: str = ""
    run_binding: RunObjectiveBinding | None = None
    objective_candidates: tuple[ObjectiveCandidateEvidence, ...] = ()
    task_source_candidates: tuple[TaskSourceCandidateEvidence, ...] = ()
    explicit_objective_cid: str = ""
    explicit_objective_revision_cid: str = ""
    explicit_plan_cid: str = ""
    explicit_task_source_cid: str = ""
    explicit_task_source_revision_cid: str = ""
    explicit_task_source_kind: str = ""
    explicit_markdown_path: str = ""
    explicit_duckdb_path: str = ""
    output_mode_hint: str = ""
    duckdb_available: bool = True
    allow_repository_output_paths: bool = False
    prompt_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_root",
            _absolute_posix_path(self.repository_root, "repository_root"),
        )
        object.__setattr__(
            self,
            "state_root",
            _absolute_posix_path(self.state_root, "state_root"),
        )
        if _is_path_under(self.state_root, self.repository_root):
            raise ObjectiveResolverError(
                "state_root must remain outside the source repository checkout"
            )
        object.__setattr__(
            self, "prompt_cid", _require_cid(self.prompt_cid, "prompt_cid")
        )
        object.__setattr__(
            self, "repository_id", str(self.repository_id or "").strip()
        )
        ns = str(self.run_namespace or "").strip().lower()
        if ns and not _TOKEN_RE.fullmatch(ns):
            raise ObjectiveResolverError("run_namespace must be a closed token")
        object.__setattr__(self, "run_namespace", ns)

        if self.run_binding is not None and not isinstance(
            self.run_binding, RunObjectiveBinding
        ):
            raise ObjectiveResolverError(
                "run_binding must be RunObjectiveBinding or None"
            )

        if isinstance(self.objective_candidates, (str, bytes)) or not isinstance(
            self.objective_candidates, Sequence
        ):
            raise ObjectiveResolverError(
                "objective_candidates must be a sequence"
            )
        objectives = tuple(
            item
            if isinstance(item, ObjectiveCandidateEvidence)
            else ObjectiveCandidateEvidence(**item)  # type: ignore[arg-type]
            for item in self.objective_candidates
        )
        object.__setattr__(self, "objective_candidates", objectives)

        if isinstance(self.task_source_candidates, (str, bytes)) or not isinstance(
            self.task_source_candidates, Sequence
        ):
            raise ObjectiveResolverError(
                "task_source_candidates must be a sequence"
            )
        task_sources = tuple(
            item
            if isinstance(item, TaskSourceCandidateEvidence)
            else TaskSourceCandidateEvidence(**item)  # type: ignore[arg-type]
            for item in self.task_source_candidates
        )
        object.__setattr__(self, "task_source_candidates", task_sources)

        for name in (
            "explicit_objective_cid",
            "explicit_objective_revision_cid",
            "explicit_plan_cid",
            "explicit_task_source_cid",
            "explicit_task_source_revision_cid",
        ):
            raw = str(getattr(self, name) or "").strip()
            if raw:
                object.__setattr__(self, name, _require_cid(raw, name))
            else:
                object.__setattr__(self, name, "")

        kind = str(self.explicit_task_source_kind or "").strip().lower()
        if kind:
            _ = _enum_member(kind, TaskSourceKind, "explicit_task_source_kind")
        object.__setattr__(self, "explicit_task_source_kind", kind)

        for name in ("explicit_markdown_path", "explicit_duckdb_path"):
            object.__setattr__(
                self, name, _optional_absolute_path(getattr(self, name), name)
            )

        hint = str(self.output_mode_hint or "").strip().lower()
        if hint:
            _ = _enum_member(hint, OutputMode, "output_mode_hint")
        object.__setattr__(self, "output_mode_hint", hint)

        object.__setattr__(
            self,
            "duckdb_available",
            _bool(self.duckdb_available, "duckdb_available"),
        )
        object.__setattr__(
            self,
            "allow_repository_output_paths",
            _bool(
                self.allow_repository_output_paths,
                "allow_repository_output_paths",
            ),
        )
        # Prompt body is non-authoritative and excluded from content_id.
        object.__setattr__(self, "prompt_text", str(self.prompt_text or ""))

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBJECTIVE_EVIDENCE_SCHEMA,
            "requirement_id": (
                OBJECTIVE_AND_TASK_SOURCE_RESOLUTION_REQUIREMENT_ID
            ),
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "prompt_cid": self.prompt_cid,
            "repository_id": self.repository_id,
            "run_namespace": self.run_namespace,
            "run_binding": (
                None if self.run_binding is None else self.run_binding.to_dict()
            ),
            "objective_candidates": [
                item.to_dict() for item in self.objective_candidates
            ],
            "task_source_candidates": [
                item.to_dict() for item in self.task_source_candidates
            ],
            "explicit_objective_cid": self.explicit_objective_cid,
            "explicit_objective_revision_cid": (
                self.explicit_objective_revision_cid
            ),
            "explicit_plan_cid": self.explicit_plan_cid,
            "explicit_task_source_cid": self.explicit_task_source_cid,
            "explicit_task_source_revision_cid": (
                self.explicit_task_source_revision_cid
            ),
            "explicit_task_source_kind": self.explicit_task_source_kind,
            "explicit_markdown_path": self.explicit_markdown_path,
            "explicit_duckdb_path": self.explicit_duckdb_path,
            "output_mode_hint": self.output_mode_hint,
            "duckdb_available": self.duckdb_available,
            "allow_repository_output_paths": self.allow_repository_output_paths,
            # prompt_text omitted deliberately
        }


@dataclass(frozen=True)
class ObjectiveBinding:
    """Resolved objective and plan identities."""

    objective_cid: str
    objective_revision_cid: str
    plan_cid: str
    created_from_prompt: bool
    selected_source: ResolutionSource

    def __post_init__(self) -> None:
        for name in ("objective_cid", "objective_revision_cid", "plan_cid"):
            object.__setattr__(
                self, name, _require_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "created_from_prompt",
            _bool(self.created_from_prompt, "created_from_prompt"),
        )
        if not isinstance(self.selected_source, ResolutionSource):
            raise ObjectiveResolverError("selected_source is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_cid": self.objective_cid,
            "objective_revision_cid": self.objective_revision_cid,
            "plan_cid": self.plan_cid,
            "created_from_prompt": self.created_from_prompt,
            "selected_source": self.selected_source.value,
        }


@dataclass(frozen=True)
class TaskSourceBinding:
    """Resolved task-source identity, kind, and projection paths."""

    task_source_cid: str
    task_source_revision_cid: str
    kind: TaskSourceKind
    path: str
    markdown_path: str
    duckdb_path: str
    created_default: bool
    selected_source: ResolutionSource
    action: TaskSourceSelectionAction

    def __post_init__(self) -> None:
        for name in ("task_source_cid", "task_source_revision_cid"):
            object.__setattr__(
                self, name, _require_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self, "kind", _enum_member(self.kind, TaskSourceKind, "kind")
        )
        for name in ("path", "markdown_path", "duckdb_path"):
            object.__setattr__(
                self, name, _optional_absolute_path(getattr(self, name), name)
            )
        if not self.path:
            raise ObjectiveResolverError("task_source path is required")
        object.__setattr__(
            self,
            "created_default",
            _bool(self.created_default, "created_default"),
        )
        if not isinstance(self.selected_source, ResolutionSource):
            raise ObjectiveResolverError("selected_source is invalid")
        object.__setattr__(
            self,
            "action",
            _enum_member(self.action, TaskSourceSelectionAction, "action"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_source_cid": self.task_source_cid,
            "task_source_revision_cid": self.task_source_revision_cid,
            "kind": self.kind.value,
            "path": self.path,
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "created_default": self.created_default,
            "selected_source": self.selected_source.value,
            "action": self.action.value,
        }


@dataclass(frozen=True)
class OutputPolicy:
    """Resolved output mode and projection paths that avoid dirtying the repo."""

    SCHEMA: ClassVar[str] = OUTPUT_POLICY_SCHEMA

    output_mode: OutputMode
    markdown_path: str
    duckdb_path: str
    outside_source_checkout: bool
    degradation: OutputDegradationCode
    selected_source: ResolutionSource

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_mode",
            _enum_member(self.output_mode, OutputMode, "output_mode"),
        )
        for name in ("markdown_path", "duckdb_path"):
            object.__setattr__(
                self, name, _optional_absolute_path(getattr(self, name), name)
            )
        mode = self.output_mode
        if mode in {OutputMode.MARKDOWN, OutputMode.BOTH} and not self.markdown_path:
            raise ObjectiveResolverError("output mode requires markdown_path")
        if mode in {OutputMode.DUCKDB, OutputMode.BOTH} and not self.duckdb_path:
            raise ObjectiveResolverError("output mode requires duckdb_path")
        object.__setattr__(
            self,
            "outside_source_checkout",
            _bool(self.outside_source_checkout, "outside_source_checkout"),
        )
        if not self.outside_source_checkout:
            raise ObjectiveResolverError(
                "output policy must keep projections outside the source checkout"
            )
        object.__setattr__(
            self,
            "degradation",
            _enum_member(self.degradation, OutputDegradationCode, "degradation"),
        )
        if not isinstance(self.selected_source, ResolutionSource):
            raise ObjectiveResolverError("selected_source is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "output_mode": self.output_mode.value,
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "outside_source_checkout": self.outside_source_checkout,
            "degradation": self.degradation.value,
            "selected_source": self.selected_source.value,
        }


@dataclass(frozen=True)
class ObjectiveResolution:
    """Complete objective, plan, task-source, and output resolution result."""

    SCHEMA: ClassVar[str] = OBJECTIVE_RESOLUTION_SCHEMA

    decisions: tuple[TargetInferenceDecision, ...]
    evidence_cid: str
    objective: ObjectiveBinding | None
    task_source: TaskSourceBinding | None
    output: OutputPolicy | None
    unresolved_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    objective_candidates_considered: tuple[ObjectiveCandidateEvidence, ...]
    task_source_candidates_considered: tuple[TaskSourceCandidateEvidence, ...]
    prompt_intent_ignored: bool = True
    created_content_addressed_objective: bool = False
    dual_projection_selected: bool = False
    markdown_degradation: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.decisions, (str, bytes)) or not isinstance(
            self.decisions, Sequence
        ):
            raise ObjectiveResolverError("decisions must be a sequence")
        decisions = tuple(
            item
            if isinstance(item, TargetInferenceDecision)
            else TargetInferenceDecision.from_dict(item)
            for item in self.decisions
        )
        names = tuple(item.field_name for item in decisions)
        if set(names) != set(OBJECTIVE_FIELD_NAMES):
            missing = set(OBJECTIVE_FIELD_NAMES).difference(names)
            extra = set(names).difference(OBJECTIVE_FIELD_NAMES)
            raise ObjectiveResolverError(
                f"objective decisions have missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        if len(names) != len(set(names)):
            raise ObjectiveResolverError(
                "objective decisions contain duplicate fields"
            )
        decisions = tuple(sorted(decisions, key=lambda item: item.field_name))
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        if self.objective is not None and not isinstance(
            self.objective, ObjectiveBinding
        ):
            raise ObjectiveResolverError("objective must be ObjectiveBinding")
        if self.task_source is not None and not isinstance(
            self.task_source, TaskSourceBinding
        ):
            raise ObjectiveResolverError(
                "task_source must be TaskSourceBinding"
            )
        if self.output is not None and not isinstance(self.output, OutputPolicy):
            raise ObjectiveResolverError("output must be OutputPolicy")
        expected_unresolved = tuple(
            sorted(item.field_name for item in decisions if item.unresolved)
        )
        unresolved = tuple(str(item) for item in self.unresolved_fields)
        if tuple(sorted(unresolved)) != expected_unresolved:
            raise ObjectiveResolverError(
                "unresolved_fields must exactly match unresolved decisions"
            )
        object.__setattr__(self, "unresolved_fields", expected_unresolved)
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes),
        )
        object.__setattr__(
            self,
            "objective_candidates_considered",
            tuple(self.objective_candidates_considered),
        )
        object.__setattr__(
            self,
            "task_source_candidates_considered",
            tuple(self.task_source_candidates_considered),
        )
        for name in (
            "prompt_intent_ignored",
            "created_content_addressed_objective",
            "dual_projection_selected",
            "markdown_degradation",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))

    @property
    def unique(self) -> bool:
        return not self.unresolved_fields and self.objective is not None

    def decision(self, field_name: str) -> TargetInferenceDecision:
        for item in self.decisions:
            if item.field_name == field_name:
                return item
        raise KeyError(field_name)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "requirement_id": (
                OBJECTIVE_AND_TASK_SOURCE_RESOLUTION_REQUIREMENT_ID
            ),
            "decisions": [item.to_dict() for item in self.decisions],
            "evidence_cid": self.evidence_cid,
            "objective": (
                None if self.objective is None else self.objective.to_dict()
            ),
            "task_source": (
                None
                if self.task_source is None
                else self.task_source.to_dict()
            ),
            "output": None if self.output is None else self.output.to_dict(),
            "unresolved_fields": list(self.unresolved_fields),
            "reason_codes": list(self.reason_codes),
            "objective_candidates_considered": [
                item.to_dict() for item in self.objective_candidates_considered
            ],
            "task_source_candidates_considered": [
                item.to_dict()
                for item in self.task_source_candidates_considered
            ],
            "prompt_intent_ignored": self.prompt_intent_ignored,
            "created_content_addressed_objective": (
                self.created_content_addressed_objective
            ),
            "dual_projection_selected": self.dual_projection_selected,
            "markdown_degradation": self.markdown_degradation,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


class ObjectiveResolver:
    """Select a run-bound or unique compatible objective, else create intent."""

    def resolve_binding(
        self, evidence: ObjectiveResolutionEvidence
    ) -> tuple[
        ObjectiveBinding | None,
        TargetInferenceDecision,
        TargetInferenceDecision,
        tuple[str, ...],
        bool,
    ]:
        """Return objective/plan binding and the two field decisions."""

        if not isinstance(evidence, ObjectiveResolutionEvidence):
            raise ObjectiveResolverError(
                "evidence must be ObjectiveResolutionEvidence"
            )
        evidence_cid = evidence.content_id
        reasons: list[str] = []

        # 1) Exact integrity-checked run binding wins.
        binding = evidence.run_binding
        if binding is not None and binding.integrity_verified:
            reasons.append("exact_run_binding_selected")
            objective = ObjectiveBinding(
                objective_cid=binding.objective_cid,
                objective_revision_cid=binding.objective_revision_cid,
                plan_cid=binding.plan_cid,
                created_from_prompt=False,
                selected_source=ResolutionSource.EXISTING_RUN,
            )
            obj_decision = _decision(
                field_name="objective",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=objective.objective_cid,
                selected_source=ResolutionSource.EXISTING_RUN,
                evidence_cid=binding.evidence_cid,
                candidates=(
                    _candidate(
                        field_name="objective",
                        value=objective.objective_cid,
                        source=ResolutionSource.EXISTING_RUN,
                        evidence_cid=binding.evidence_cid,
                    ),
                ),
                reason_codes=("exact_run_binding_selected",),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            plan_decision = _decision(
                field_name="plan",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=objective.plan_cid,
                selected_source=ResolutionSource.EXISTING_RUN,
                evidence_cid=binding.evidence_cid,
                candidates=(
                    _candidate(
                        field_name="plan",
                        value=objective.plan_cid,
                        source=ResolutionSource.EXISTING_RUN,
                        evidence_cid=binding.evidence_cid,
                    ),
                ),
                reason_codes=("exact_run_binding_selected",),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            return objective, obj_decision, plan_decision, tuple(reasons), False

        if binding is not None and not binding.integrity_verified:
            reasons.append("run_binding_integrity_unverified")

        # 2) Explicit objective override.
        if evidence.explicit_objective_cid:
            revision = (
                evidence.explicit_objective_revision_cid
                or cid_for_dag_json(
                    {
                        "schema": PROMPT_OBJECTIVE_REVISION_SCHEMA,
                        "objective_cid": evidence.explicit_objective_cid,
                        "revision_index": 1,
                        "source": "explicit_override",
                    }
                )
            )
            plan = evidence.explicit_plan_cid or cid_for_dag_json(
                {
                    "schema": PROMPT_PLAN_PLACEHOLDER_SCHEMA,
                    "objective_revision_cid": revision,
                    "status": "explicit_binding",
                }
            )
            objective = ObjectiveBinding(
                objective_cid=evidence.explicit_objective_cid,
                objective_revision_cid=revision,
                plan_cid=plan,
                created_from_prompt=False,
                selected_source=ResolutionSource.EXPLICIT_OVERRIDE,
            )
            reasons.append("explicit_objective_override")
            alt_candidates = [
                _candidate(
                    field_name="objective",
                    value=objective.objective_cid,
                    source=ResolutionSource.EXPLICIT_OVERRIDE,
                    evidence_cid=evidence_cid,
                )
            ]
            for item in evidence.objective_candidates:
                if item.objective_cid == objective.objective_cid:
                    continue
                alt_candidates.append(
                    _candidate(
                        field_name="objective",
                        value=item.objective_cid,
                        source=ResolutionSource.DISCOVERY,
                        evidence_cid=item.evidence_cid,
                        confidence_ppm=250_000,
                        rejection_reason="superseded_by_explicit_override",
                    )
                )
            obj_decision = _decision(
                field_name="objective",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=objective.objective_cid,
                selected_source=ResolutionSource.EXPLICIT_OVERRIDE,
                evidence_cid=evidence_cid,
                candidates=alt_candidates,
                reason_codes=("explicit_objective_override",),
                effect=DecisionEffect.IDENTITY_ONLY,
                override_accepted=True,
            )
            plan_decision = _decision(
                field_name="plan",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=objective.plan_cid,
                selected_source=ResolutionSource.EXPLICIT_OVERRIDE,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="plan",
                        value=objective.plan_cid,
                        source=ResolutionSource.EXPLICIT_OVERRIDE,
                        evidence_cid=evidence_cid,
                    ),
                ),
                reason_codes=("explicit_objective_override",),
                effect=DecisionEffect.IDENTITY_ONLY,
                override_accepted=True,
            )
            return objective, obj_decision, plan_decision, tuple(reasons), False

        # 3) Unique viable discovered objective.
        viable = [item for item in evidence.objective_candidates if item.viable]
        nonviable = [
            item for item in evidence.objective_candidates if not item.viable
        ]
        if nonviable:
            reasons.append("nonviable_objective_candidates_rejected")
        if len(viable) > 1:
            reasons.append("multiple_compatible_objectives")
            candidates = [
                _candidate(
                    field_name="objective",
                    value=item.objective_cid,
                    source=(
                        ResolutionSource.EXISTING_RUN
                        if item.run_bound
                        else ResolutionSource.DISCOVERY
                    ),
                    evidence_cid=item.evidence_cid,
                    confidence_ppm=500_000,
                )
                for item in viable
            ]
            # Titles/board filenames alone never produce a unique pick.
            obj_decision = _decision(
                field_name="objective",
                disposition=ResolutionDisposition.AMBIGUOUS,
                selected_value="",
                selected_source=ResolutionSource.DISCOVERY,
                evidence_cid=evidence_cid,
                candidates=candidates,
                reason_codes=(
                    "multiple_compatible_objectives",
                    "board_titles_non_authoritative",
                ),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            plan_decision = _decision(
                field_name="plan",
                disposition=ResolutionDisposition.AMBIGUOUS,
                selected_value="",
                selected_source=ResolutionSource.DISCOVERY,
                evidence_cid=evidence_cid,
                candidates=[
                    _candidate(
                        field_name="plan",
                        value=item.plan_cid or item.objective_revision_cid,
                        source=ResolutionSource.DISCOVERY,
                        evidence_cid=item.evidence_cid,
                        confidence_ppm=500_000,
                    )
                    for item in viable
                    if (item.plan_cid or item.objective_revision_cid)
                ]
                or (
                    _candidate(
                        field_name="plan",
                        value=viable[0].objective_revision_cid,
                        source=ResolutionSource.DISCOVERY,
                        evidence_cid=viable[0].evidence_cid,
                    ),
                    _candidate(
                        field_name="plan",
                        value=viable[1].objective_revision_cid,
                        source=ResolutionSource.DISCOVERY,
                        evidence_cid=viable[1].evidence_cid,
                    ),
                ),
                reason_codes=("multiple_compatible_objectives",),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            return None, obj_decision, plan_decision, tuple(reasons), False

        if len(viable) == 1:
            selected = viable[0]
            plan_cid = selected.plan_cid or cid_for_dag_json(
                {
                    "schema": PROMPT_PLAN_PLACEHOLDER_SCHEMA,
                    "objective_revision_cid": selected.objective_revision_cid,
                    "status": "discovered_without_plan",
                }
            )
            source = (
                ResolutionSource.EXISTING_RUN
                if selected.run_bound
                else ResolutionSource.DISCOVERY
            )
            objective = ObjectiveBinding(
                objective_cid=selected.objective_cid,
                objective_revision_cid=selected.objective_revision_cid,
                plan_cid=plan_cid,
                created_from_prompt=False,
                selected_source=source,
            )
            reasons.append("unique_compatible_objective")
            alt = [
                _candidate(
                    field_name="objective",
                    value=selected.objective_cid,
                    source=source,
                    evidence_cid=selected.evidence_cid,
                )
            ]
            for item in nonviable:
                reject = (
                    "integrity_unverified"
                    if not item.integrity_verified
                    else "inactive_or_incompatible"
                )
                alt.append(
                    _candidate(
                        field_name="objective",
                        value=item.objective_cid,
                        source=ResolutionSource.DISCOVERY,
                        evidence_cid=item.evidence_cid,
                        confidence_ppm=0,
                        rejection_reason=reject,
                    )
                )
            obj_decision = _decision(
                field_name="objective",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=objective.objective_cid,
                selected_source=source,
                evidence_cid=selected.evidence_cid,
                candidates=alt,
                reason_codes=("unique_compatible_objective",),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            plan_decision = _decision(
                field_name="plan",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=objective.plan_cid,
                selected_source=source,
                evidence_cid=selected.evidence_cid,
                candidates=(
                    _candidate(
                        field_name="plan",
                        value=objective.plan_cid,
                        source=source,
                        evidence_cid=selected.evidence_cid,
                    ),
                ),
                reason_codes=("unique_compatible_objective",),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            return objective, obj_decision, plan_decision, tuple(reasons), False

        # 4) Absent intent → content-addressed prompt objective.
        objective_cid, revision_cid, plan_cid = content_addressed_prompt_objective(
            evidence.prompt_cid,
            repository_id=evidence.repository_id,
            run_namespace=evidence.run_namespace,
        )
        objective = ObjectiveBinding(
            objective_cid=objective_cid,
            objective_revision_cid=revision_cid,
            plan_cid=plan_cid,
            created_from_prompt=True,
            selected_source=ResolutionSource.BUILTIN_DEFAULT,
        )
        reasons.append("content_addressed_prompt_objective_created")
        reasons.append("absent_unique_objective_intent")
        obj_decision = _decision(
            field_name="objective",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=objective.objective_cid,
            selected_source=ResolutionSource.BUILTIN_DEFAULT,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="objective",
                    value=objective.objective_cid,
                    source=ResolutionSource.BUILTIN_DEFAULT,
                    evidence_cid=evidence_cid,
                ),
            ),
            reason_codes=(
                "content_addressed_prompt_objective_created",
                "absent_unique_objective_intent",
            ),
            effect=DecisionEffect.IDENTITY_ONLY,
        )
        plan_decision = _decision(
            field_name="plan",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=objective.plan_cid,
            selected_source=ResolutionSource.BUILTIN_DEFAULT,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="plan",
                    value=objective.plan_cid,
                    source=ResolutionSource.BUILTIN_DEFAULT,
                    evidence_cid=evidence_cid,
                ),
            ),
            reason_codes=(
                "pending_plan_identity_from_prompt_objective",
            ),
            effect=DecisionEffect.IDENTITY_ONLY,
        )
        return objective, obj_decision, plan_decision, tuple(reasons), True


class OutputPolicyResolver:
    """Resolve output mode and projection paths under the platform state root."""

    def resolve(
        self,
        evidence: ObjectiveResolutionEvidence,
        *,
        bound_markdown: str = "",
        bound_duckdb: str = "",
        bound_mode: OutputMode | None = None,
        bound_source: ResolutionSource | None = None,
    ) -> tuple[OutputPolicy | None, TargetInferenceDecision, tuple[str, ...]]:
        if not isinstance(evidence, ObjectiveResolutionEvidence):
            raise ObjectiveResolverError(
                "evidence must be ObjectiveResolutionEvidence"
            )
        evidence_cid = evidence.content_id
        reasons: list[str] = []
        markdown_default, duckdb_default = default_projection_paths(
            evidence.state_root
        )

        # Exact run binding paths win when provided.
        if bound_mode is not None and bound_source is not None:
            markdown = bound_markdown or markdown_default
            duckdb = bound_duckdb or duckdb_default
            if not self._paths_safe(evidence, markdown, duckdb):
                reasons.append("run_binding_output_paths_dirty_repository")
                return (
                    None,
                    _decision(
                        field_name="output",
                        disposition=ResolutionDisposition.DENIED,
                        selected_value="",
                        selected_source=bound_source,
                        evidence_cid=evidence_cid,
                        candidates=(
                            _candidate(
                                field_name="output",
                                value=bound_mode.value,
                                source=bound_source,
                                evidence_cid=evidence_cid,
                                rejection_reason="output_paths_inside_repository",
                            ),
                        ),
                        reason_codes=tuple(reasons),
                        effect=DecisionEffect.CONFIGURATION,
                    ),
                    tuple(reasons),
                )
            reasons.append("exact_run_binding_output")
            policy = OutputPolicy(
                output_mode=bound_mode,
                markdown_path=markdown if bound_mode is not OutputMode.DUCKDB else "",
                duckdb_path=duckdb if bound_mode is not OutputMode.MARKDOWN else "",
                outside_source_checkout=True,
                degradation=OutputDegradationCode.NONE,
                selected_source=bound_source,
            )
            # Normalize empty paths for single-mode bindings.
            if bound_mode is OutputMode.MARKDOWN:
                policy = OutputPolicy(
                    output_mode=bound_mode,
                    markdown_path=markdown,
                    duckdb_path="",
                    outside_source_checkout=True,
                    degradation=OutputDegradationCode.NONE,
                    selected_source=bound_source,
                )
            elif bound_mode is OutputMode.DUCKDB:
                policy = OutputPolicy(
                    output_mode=bound_mode,
                    markdown_path="",
                    duckdb_path=duckdb,
                    outside_source_checkout=True,
                    degradation=OutputDegradationCode.NONE,
                    selected_source=bound_source,
                )
            else:
                policy = OutputPolicy(
                    output_mode=bound_mode,
                    markdown_path=markdown,
                    duckdb_path=duckdb,
                    outside_source_checkout=True,
                    degradation=OutputDegradationCode.NONE,
                    selected_source=bound_source,
                )
            decision = _decision(
                field_name="output",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=policy.output_mode.value,
                selected_source=bound_source,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="output",
                        value=policy.output_mode.value,
                        source=bound_source,
                        evidence_cid=evidence_cid,
                    ),
                ),
                reason_codes=("exact_run_binding_output",),
                effect=DecisionEffect.CONFIGURATION,
            )
            return policy, decision, tuple(reasons)

        # Explicit path overrides (must stay outside the repository by default).
        explicit_markdown = evidence.explicit_markdown_path
        explicit_duckdb = evidence.explicit_duckdb_path
        if explicit_markdown or explicit_duckdb:
            markdown = explicit_markdown or markdown_default
            duckdb = explicit_duckdb or duckdb_default
            if not self._paths_safe(evidence, markdown, duckdb):
                reasons.append("explicit_output_paths_dirty_repository")
                return (
                    None,
                    _decision(
                        field_name="output",
                        disposition=ResolutionDisposition.DENIED,
                        selected_value="",
                        selected_source=ResolutionSource.EXPLICIT_OVERRIDE,
                        evidence_cid=evidence_cid,
                        candidates=(
                            _candidate(
                                field_name="output",
                                value=OutputMode.BOTH.value,
                                source=ResolutionSource.EXPLICIT_OVERRIDE,
                                evidence_cid=evidence_cid,
                                rejection_reason="output_paths_inside_repository",
                            ),
                        ),
                        reason_codes=tuple(reasons),
                        effect=DecisionEffect.CONFIGURATION,
                        override_accepted=False,
                    ),
                    tuple(reasons),
                )
            mode, degradation, mode_reasons = self._select_mode(
                evidence,
                prefer_explicit=True,
            )
            reasons.extend(mode_reasons)
            reasons.append("explicit_output_paths")
            policy = self._policy_for_mode(
                mode=mode,
                markdown=markdown,
                duckdb=duckdb,
                degradation=degradation,
                source=ResolutionSource.EXPLICIT_OVERRIDE,
            )
            decision = _decision(
                field_name="output",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=policy.output_mode.value,
                selected_source=ResolutionSource.EXPLICIT_OVERRIDE,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="output",
                        value=policy.output_mode.value,
                        source=ResolutionSource.EXPLICIT_OVERRIDE,
                        evidence_cid=evidence_cid,
                    ),
                ),
                reason_codes=tuple(dict.fromkeys(reasons)),
                effect=DecisionEffect.CONFIGURATION,
                override_accepted=True,
            )
            return policy, decision, tuple(dict.fromkeys(reasons))

        mode, degradation, mode_reasons = self._select_mode(
            evidence, prefer_explicit=False
        )
        reasons.extend(mode_reasons)
        reasons.append("state_root_projection_defaults")
        policy = self._policy_for_mode(
            mode=mode,
            markdown=markdown_default,
            duckdb=duckdb_default,
            degradation=degradation,
            source=ResolutionSource.BUILTIN_DEFAULT,
        )
        disposition = (
            ResolutionDisposition.DEFAULTED
            if not evidence.output_mode_hint
            else ResolutionDisposition.UNIQUE
        )
        source = (
            ResolutionSource.EXPLICIT_OVERRIDE
            if evidence.output_mode_hint
            and mode.value == evidence.output_mode_hint
            and (
                (mode is OutputMode.MARKDOWN)
                or (mode is OutputMode.DUCKDB and evidence.duckdb_available)
                or (mode is OutputMode.BOTH and evidence.duckdb_available)
            )
            else ResolutionSource.BUILTIN_DEFAULT
        )
        if source is ResolutionSource.EXPLICIT_OVERRIDE:
            disposition = ResolutionDisposition.UNIQUE
            reasons.append("output_mode_hint_accepted")
        else:
            if evidence.output_mode_hint and mode.value != evidence.output_mode_hint:
                reasons.append("output_mode_hint_degraded")
            disposition = ResolutionDisposition.DEFAULTED
            source = ResolutionSource.BUILTIN_DEFAULT

        candidates = [
            _candidate(
                field_name="output",
                value=policy.output_mode.value,
                source=source,
                evidence_cid=evidence_cid,
            )
        ]
        # Record the preferred dual option as a rejected alternative when degraded.
        if policy.output_mode is not OutputMode.BOTH and not evidence.duckdb_available:
            candidates.append(
                _candidate(
                    field_name="output",
                    value=OutputMode.BOTH.value,
                    source=ResolutionSource.BUILTIN_DEFAULT,
                    evidence_cid=evidence_cid,
                    confidence_ppm=0,
                    rejection_reason="duckdb_unavailable",
                )
            )

        decision = _decision(
            field_name="output",
            disposition=disposition,
            selected_value=policy.output_mode.value,
            selected_source=source,
            evidence_cid=evidence_cid,
            candidates=candidates,
            reason_codes=tuple(dict.fromkeys(reasons)),
            effect=DecisionEffect.CONFIGURATION,
            override_accepted=source is ResolutionSource.EXPLICIT_OVERRIDE,
        )
        return policy, decision, tuple(dict.fromkeys(reasons))

    def _paths_safe(
        self,
        evidence: ObjectiveResolutionEvidence,
        markdown: str,
        duckdb: str,
    ) -> bool:
        for path in (markdown, duckdb):
            if not path:
                continue
            if (
                _is_path_under(path, evidence.repository_root)
                and not evidence.allow_repository_output_paths
            ):
                return False
            if (
                not _is_path_under(path, evidence.repository_root)
                and not _is_path_under(path, evidence.state_root)
                and not evidence.allow_repository_output_paths
            ):
                # Paths must live under the state root unless they are the
                # empty single-mode counterpart.
                return False
        return True

    def _select_mode(
        self,
        evidence: ObjectiveResolutionEvidence,
        *,
        prefer_explicit: bool,
    ) -> tuple[OutputMode, OutputDegradationCode, list[str]]:
        _ = prefer_explicit
        reasons: list[str] = []
        hint = evidence.output_mode_hint
        if hint:
            requested = OutputMode(hint)
            if requested is OutputMode.BOTH and not evidence.duckdb_available:
                reasons.append("duckdb_unavailable_markdown_degradation")
                return (
                    OutputMode.MARKDOWN,
                    OutputDegradationCode.DUCKDB_UNAVAILABLE,
                    reasons,
                )
            if requested is OutputMode.DUCKDB and not evidence.duckdb_available:
                reasons.append("duckdb_unavailable_markdown_degradation")
                return (
                    OutputMode.MARKDOWN,
                    OutputDegradationCode.DUCKDB_UNAVAILABLE,
                    reasons,
                )
            if requested is OutputMode.MARKDOWN:
                reasons.append("explicit_markdown_output")
                return (
                    OutputMode.MARKDOWN,
                    OutputDegradationCode.EXPLICIT_MARKDOWN,
                    reasons,
                )
            if requested is OutputMode.DUCKDB:
                reasons.append("explicit_duckdb_output")
                return (
                    OutputMode.DUCKDB,
                    OutputDegradationCode.EXPLICIT_DUCKDB,
                    reasons,
                )
            reasons.append("explicit_both_output")
            return OutputMode.BOTH, OutputDegradationCode.NONE, reasons

        if evidence.duckdb_available:
            reasons.append("duckdb_plus_markdown_mirror_selected")
            return OutputMode.BOTH, OutputDegradationCode.NONE, reasons
        reasons.append("duckdb_unavailable_markdown_degradation")
        return (
            OutputMode.MARKDOWN,
            OutputDegradationCode.DUCKDB_UNAVAILABLE,
            reasons,
        )

    def _policy_for_mode(
        self,
        *,
        mode: OutputMode,
        markdown: str,
        duckdb: str,
        degradation: OutputDegradationCode,
        source: ResolutionSource,
    ) -> OutputPolicy:
        if mode is OutputMode.MARKDOWN:
            return OutputPolicy(
                output_mode=mode,
                markdown_path=markdown,
                duckdb_path="",
                outside_source_checkout=True,
                degradation=degradation,
                selected_source=source,
            )
        if mode is OutputMode.DUCKDB:
            return OutputPolicy(
                output_mode=mode,
                markdown_path="",
                duckdb_path=duckdb,
                outside_source_checkout=True,
                degradation=degradation,
                selected_source=source,
            )
        return OutputPolicy(
            output_mode=mode,
            markdown_path=markdown,
            duckdb_path=duckdb,
            outside_source_checkout=True,
            degradation=degradation,
            selected_source=source,
        )


class TaskSourceResolver:
    """Select a run-bound or unique task source, else construct state-root defaults."""

    def resolve_binding(
        self,
        evidence: ObjectiveResolutionEvidence,
        *,
        objective: ObjectiveBinding | None,
        output: OutputPolicy | None,
    ) -> tuple[
        TaskSourceBinding | None,
        TargetInferenceDecision,
        tuple[str, ...],
    ]:
        if not isinstance(evidence, ObjectiveResolutionEvidence):
            raise ObjectiveResolverError(
                "evidence must be ObjectiveResolutionEvidence"
            )
        evidence_cid = evidence.content_id
        reasons: list[str] = []

        binding = evidence.run_binding
        if binding is not None and binding.integrity_verified:
            path = binding.duckdb_path or binding.markdown_path
            if binding.task_source_kind is TaskSourceKind.MARKDOWN:
                path = binding.markdown_path or binding.duckdb_path
            elif binding.task_source_kind is TaskSourceKind.DUCKDB:
                path = binding.duckdb_path or binding.markdown_path
            else:
                path = binding.duckdb_path or binding.markdown_path
            task = TaskSourceBinding(
                task_source_cid=binding.task_source_cid,
                task_source_revision_cid=binding.task_source_revision_cid,
                kind=binding.task_source_kind,
                path=path,
                markdown_path=binding.markdown_path,
                duckdb_path=binding.duckdb_path,
                created_default=False,
                selected_source=ResolutionSource.EXISTING_RUN,
                action=TaskSourceSelectionAction.BIND_EXISTING,
            )
            reasons.append("exact_run_binding_task_source")
            decision = _decision(
                field_name="task_source",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=task.task_source_cid,
                selected_source=ResolutionSource.EXISTING_RUN,
                evidence_cid=binding.evidence_cid,
                candidates=(
                    _candidate(
                        field_name="task_source",
                        value=task.task_source_cid,
                        source=ResolutionSource.EXISTING_RUN,
                        evidence_cid=binding.evidence_cid,
                    ),
                ),
                reason_codes=("exact_run_binding_task_source",),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            return task, decision, tuple(reasons)

        if evidence.explicit_task_source_cid:
            kind = (
                TaskSourceKind(evidence.explicit_task_source_kind)
                if evidence.explicit_task_source_kind
                else (
                    TaskSourceKind.DUAL
                    if evidence.duckdb_available
                    else TaskSourceKind.MARKDOWN
                )
            )
            markdown, duckdb = default_projection_paths(evidence.state_root)
            markdown = evidence.explicit_markdown_path or markdown
            duckdb = evidence.explicit_duckdb_path or duckdb
            if output is not None:
                markdown = output.markdown_path or markdown
                duckdb = output.duckdb_path or duckdb
            path = (
                duckdb
                if kind in {TaskSourceKind.DUCKDB, TaskSourceKind.DUAL}
                else markdown
            )
            revision = (
                evidence.explicit_task_source_revision_cid
                or cid_for_dag_json(
                    {
                        "schema": f"{SCHEMA_PREFIX}/explicit-task-source-revision@1",
                        "task_source_cid": evidence.explicit_task_source_cid,
                        "revision_index": 1,
                    }
                )
            )
            task = TaskSourceBinding(
                task_source_cid=evidence.explicit_task_source_cid,
                task_source_revision_cid=revision,
                kind=kind,
                path=path,
                markdown_path=markdown if kind is not TaskSourceKind.DUCKDB else "",
                duckdb_path=duckdb if kind is not TaskSourceKind.MARKDOWN else "",
                created_default=False,
                selected_source=ResolutionSource.EXPLICIT_OVERRIDE,
                action=TaskSourceSelectionAction.BIND_EXISTING,
            )
            reasons.append("explicit_task_source_override")
            decision = _decision(
                field_name="task_source",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=task.task_source_cid,
                selected_source=ResolutionSource.EXPLICIT_OVERRIDE,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="task_source",
                        value=task.task_source_cid,
                        source=ResolutionSource.EXPLICIT_OVERRIDE,
                        evidence_cid=evidence_cid,
                    ),
                ),
                reason_codes=("explicit_task_source_override",),
                effect=DecisionEffect.IDENTITY_ONLY,
                override_accepted=True,
            )
            return task, decision, tuple(reasons)

        viable = [item for item in evidence.task_source_candidates if item.viable]
        nonviable = [
            item for item in evidence.task_source_candidates if not item.viable
        ]
        if nonviable:
            reasons.append("nonviable_task_source_candidates_rejected")
        if len(viable) > 1:
            reasons.append("multiple_compatible_task_sources")
            reasons.append("board_filenames_non_authoritative")
            candidates = [
                _candidate(
                    field_name="task_source",
                    value=item.task_source_cid,
                    source=(
                        ResolutionSource.EXISTING_RUN
                        if item.run_bound
                        else ResolutionSource.DISCOVERY
                    ),
                    evidence_cid=item.evidence_cid,
                    confidence_ppm=500_000,
                )
                for item in viable
            ]
            decision = _decision(
                field_name="task_source",
                disposition=ResolutionDisposition.AMBIGUOUS,
                selected_value="",
                selected_source=ResolutionSource.DISCOVERY,
                evidence_cid=evidence_cid,
                candidates=candidates,
                reason_codes=tuple(reasons),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            return None, decision, tuple(reasons)

        if len(viable) == 1:
            selected = viable[0]
            source = (
                ResolutionSource.EXISTING_RUN
                if selected.run_bound
                else ResolutionSource.DISCOVERY
            )
            markdown = selected.markdown_path
            duckdb = selected.duckdb_path
            path = selected.path or duckdb or markdown
            if not path:
                markdown, duckdb = default_projection_paths(evidence.state_root)
                path = (
                    duckdb
                    if selected.kind
                    in {TaskSourceKind.DUCKDB, TaskSourceKind.DUAL}
                    else markdown
                )
            # Ensure selected projections do not dirty the repository.
            for check in (path, markdown, duckdb):
                if (
                    check
                    and _is_path_under(check, evidence.repository_root)
                    and not evidence.allow_repository_output_paths
                ):
                    reasons.append(
                        "discovered_task_source_inside_repository_rejected"
                    )
                    decision = _decision(
                        field_name="task_source",
                        disposition=ResolutionDisposition.DENIED,
                        selected_value="",
                        selected_source=source,
                        evidence_cid=selected.evidence_cid,
                        candidates=(
                            _candidate(
                                field_name="task_source",
                                value=selected.task_source_cid,
                                source=source,
                                evidence_cid=selected.evidence_cid,
                                rejection_reason=(
                                    "task_source_inside_repository"
                                ),
                            ),
                        ),
                        reason_codes=tuple(reasons),
                        effect=DecisionEffect.IDENTITY_ONLY,
                    )
                    return None, decision, tuple(reasons)
            task = TaskSourceBinding(
                task_source_cid=selected.task_source_cid,
                task_source_revision_cid=selected.task_source_revision_cid,
                kind=selected.kind,
                path=path,
                markdown_path=markdown,
                duckdb_path=duckdb,
                created_default=False,
                selected_source=source,
                action=TaskSourceSelectionAction.BIND_EXISTING,
            )
            reasons.append("unique_compatible_task_source")
            alt = [
                _candidate(
                    field_name="task_source",
                    value=selected.task_source_cid,
                    source=source,
                    evidence_cid=selected.evidence_cid,
                )
            ]
            for item in nonviable:
                reject = (
                    "integrity_unverified"
                    if not item.integrity_verified
                    else "incompatible_task_source"
                )
                alt.append(
                    _candidate(
                        field_name="task_source",
                        value=item.task_source_cid,
                        source=ResolutionSource.DISCOVERY,
                        evidence_cid=item.evidence_cid,
                        confidence_ppm=0,
                        rejection_reason=reject,
                    )
                )
            decision = _decision(
                field_name="task_source",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=task.task_source_cid,
                selected_source=source,
                evidence_cid=selected.evidence_cid,
                candidates=alt,
                reason_codes=("unique_compatible_task_source",),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            return task, decision, tuple(reasons)

        # Default construction under state_root. Prefer dual when DuckDB exists.
        if objective is None or output is None:
            reasons.append("cannot_default_task_source_without_objective_output")
            decision = _decision(
                field_name="task_source",
                disposition=ResolutionDisposition.UNAVAILABLE,
                selected_value="",
                selected_source=ResolutionSource.BUILTIN_DEFAULT,
                evidence_cid=evidence_cid,
                candidates=(),
                reason_codes=tuple(reasons),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
            return None, decision, tuple(reasons)

        if output.output_mode is OutputMode.BOTH:
            kind = TaskSourceKind.DUAL
            reasons.append("default_dual_task_source")
        elif output.output_mode is OutputMode.DUCKDB:
            kind = TaskSourceKind.DUCKDB
            reasons.append("default_duckdb_task_source")
        else:
            kind = TaskSourceKind.MARKDOWN
            reasons.append("default_markdown_task_source")
            if not evidence.duckdb_available:
                reasons.append("typed_markdown_degradation")

        markdown = output.markdown_path or default_projection_paths(
            evidence.state_root
        )[0]
        duckdb = output.duckdb_path or default_projection_paths(evidence.state_root)[1]
        path = (
            duckdb
            if kind in {TaskSourceKind.DUCKDB, TaskSourceKind.DUAL}
            else markdown
        )
        task_source_cid, revision_cid = default_task_source_identity(
            state_root=evidence.state_root,
            kind=kind,
            markdown_path=markdown,
            duckdb_path=duckdb,
            objective_revision_cid=objective.objective_revision_cid,
        )
        task = TaskSourceBinding(
            task_source_cid=task_source_cid,
            task_source_revision_cid=revision_cid,
            kind=kind,
            path=path,
            markdown_path=markdown if kind is not TaskSourceKind.DUCKDB else "",
            duckdb_path=duckdb if kind is not TaskSourceKind.MARKDOWN else "",
            created_default=True,
            selected_source=ResolutionSource.BUILTIN_DEFAULT,
            action=TaskSourceSelectionAction.CREATE_DEFAULT,
        )
        decision = _decision(
            field_name="task_source",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=task.task_source_cid,
            selected_source=ResolutionSource.BUILTIN_DEFAULT,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="task_source",
                    value=task.task_source_cid,
                    source=ResolutionSource.BUILTIN_DEFAULT,
                    evidence_cid=evidence_cid,
                ),
            ),
            reason_codes=tuple(dict.fromkeys(reasons)),
            effect=DecisionEffect.IDENTITY_ONLY,
        )
        return task, decision, tuple(dict.fromkeys(reasons))


class ObjectivePlanTaskSourceResolver:
    """Compose objective, task-source, and output policy resolvers."""

    def __init__(
        self,
        *,
        objective_resolver: ObjectiveResolver | None = None,
        task_source_resolver: TaskSourceResolver | None = None,
        output_policy_resolver: OutputPolicyResolver | None = None,
    ) -> None:
        self.objective_resolver = objective_resolver or ObjectiveResolver()
        self.task_source_resolver = task_source_resolver or TaskSourceResolver()
        self.output_policy_resolver = (
            output_policy_resolver or OutputPolicyResolver()
        )

    def resolve(
        self, evidence: ObjectiveResolutionEvidence
    ) -> ObjectiveResolution:
        if not isinstance(evidence, ObjectiveResolutionEvidence):
            raise ObjectiveResolverError(
                "evidence must be ObjectiveResolutionEvidence"
            )
        evidence_cid = evidence.content_id
        all_reasons: list[str] = []
        if evidence.prompt_text:
            all_reasons.append("prompt_text_ignored")

        (
            objective,
            objective_decision,
            plan_decision,
            objective_reasons,
            created,
        ) = self.objective_resolver.resolve_binding(evidence)
        all_reasons.extend(objective_reasons)

        # If objective is ambiguous, still attempt output defaults but mark
        # task_source unresolved consistently.
        bound_mode: OutputMode | None = None
        bound_source: ResolutionSource | None = None
        bound_markdown = ""
        bound_duckdb = ""
        if (
            evidence.run_binding is not None
            and evidence.run_binding.integrity_verified
        ):
            bound_mode = evidence.run_binding.output_mode
            bound_source = ResolutionSource.EXISTING_RUN
            bound_markdown = evidence.run_binding.markdown_path
            bound_duckdb = evidence.run_binding.duckdb_path

        output, output_decision, output_reasons = (
            self.output_policy_resolver.resolve(
                evidence,
                bound_markdown=bound_markdown,
                bound_duckdb=bound_duckdb,
                bound_mode=bound_mode,
                bound_source=bound_source,
            )
        )
        all_reasons.extend(output_reasons)

        task_source, task_decision, task_reasons = (
            self.task_source_resolver.resolve_binding(
                evidence,
                objective=objective,
                output=output,
            )
        )
        all_reasons.extend(task_reasons)

        # When objective is ambiguous, force task_source ambiguity if it was
        # about to default (do not invent a board while objective is unclear).
        if (
            objective is None
            and objective_decision.disposition
            is ResolutionDisposition.AMBIGUOUS
            and task_source is not None
            and task_source.created_default
        ):
            all_reasons.append("task_source_deferred_until_objective_unique")
            # Prefer reporting objective candidates as the ambiguity surface.
            if len(evidence.task_source_candidates) >= 2:
                task_source = None
                task_decision = _decision(
                    field_name="task_source",
                    disposition=ResolutionDisposition.AMBIGUOUS,
                    selected_value="",
                    selected_source=ResolutionSource.DISCOVERY,
                    evidence_cid=evidence_cid,
                    candidates=[
                        _candidate(
                            field_name="task_source",
                            value=item.task_source_cid,
                            source=ResolutionSource.DISCOVERY,
                            evidence_cid=item.evidence_cid,
                            confidence_ppm=500_000,
                        )
                        for item in evidence.task_source_candidates
                        if item.viable
                    ][:64]
                    or [
                        _candidate(
                            field_name="task_source",
                            value=evidence.task_source_candidates[0].task_source_cid,
                            source=ResolutionSource.DISCOVERY,
                            evidence_cid=evidence.task_source_candidates[
                                0
                            ].evidence_cid,
                        ),
                        _candidate(
                            field_name="task_source",
                            value=evidence.task_source_candidates[1].task_source_cid,
                            source=ResolutionSource.DISCOVERY,
                            evidence_cid=evidence.task_source_candidates[
                                1
                            ].evidence_cid,
                        ),
                    ],
                    reason_codes=(
                        "task_source_deferred_until_objective_unique",
                        "multiple_compatible_objectives",
                    ),
                    effect=DecisionEffect.IDENTITY_ONLY,
                )
            else:
                task_source = None
                task_decision = _decision(
                    field_name="task_source",
                    disposition=ResolutionDisposition.UNAVAILABLE,
                    selected_value="",
                    selected_source=ResolutionSource.DISCOVERY,
                    evidence_cid=evidence_cid,
                    candidates=(),
                    reason_codes=(
                        "task_source_deferred_until_objective_unique",
                    ),
                    effect=DecisionEffect.IDENTITY_ONLY,
                )

        decisions = (
            objective_decision,
            plan_decision,
            task_decision,
            output_decision,
        )
        unresolved = tuple(
            sorted(item.field_name for item in decisions if item.unresolved)
        )
        dual = bool(
            task_source is not None
            and task_source.kind is TaskSourceKind.DUAL
        ) or bool(
            output is not None and output.output_mode is OutputMode.BOTH
        )
        markdown_degraded = bool(
            output is not None
            and output.degradation
            in {
                OutputDegradationCode.DUCKDB_UNAVAILABLE,
                OutputDegradationCode.MARKDOWN_ONLY,
            }
        )

        # Drop binding objects when their field is unresolved.
        if objective_decision.unresolved:
            objective = None
        if task_decision.unresolved:
            task_source = None
        if output_decision.unresolved:
            output = None

        return ObjectiveResolution(
            decisions=decisions,
            evidence_cid=evidence_cid,
            objective=objective,
            task_source=task_source,
            output=output,
            unresolved_fields=unresolved,
            reason_codes=tuple(dict.fromkeys(all_reasons)),
            objective_candidates_considered=evidence.objective_candidates,
            task_source_candidates_considered=evidence.task_source_candidates,
            prompt_intent_ignored=True,
            created_content_addressed_objective=created
            and objective is not None,
            dual_projection_selected=dual and not unresolved,
            markdown_degradation=markdown_degraded,
        )


def resolve_objective_plan_and_output(
    evidence: ObjectiveResolutionEvidence,
) -> ObjectiveResolution:
    """Module-level convenience wrapper for full ASE-007 resolution."""

    return ObjectivePlanTaskSourceResolver().resolve(evidence)


def resolve_objectives(
    evidence: ObjectiveResolutionEvidence,
) -> ObjectiveResolution:
    """Alias matching the ObjectiveResolver naming in the backlog interfaces."""

    return resolve_objective_plan_and_output(evidence)


__all__ = [
    "DEFAULT_DUCKDB_RELATIVE",
    "DEFAULT_MARKDOWN_RELATIVE",
    "OBJECTIVE_AND_TASK_SOURCE_RESOLUTION_REQUIREMENT_ID",
    "OBJECTIVE_EVIDENCE_SCHEMA",
    "OBJECTIVE_FIELD_NAMES",
    "OBJECTIVE_RESOLUTION_SCHEMA",
    "SOURCE_PRECEDENCE",
    "ObjectiveBinding",
    "ObjectiveCandidateEvidence",
    "ObjectivePlanTaskSourceResolver",
    "ObjectiveResolution",
    "ObjectiveResolutionEvidence",
    "ObjectiveResolver",
    "ObjectiveResolverError",
    "OutputDegradationCode",
    "OutputPolicy",
    "OutputPolicyResolver",
    "RunObjectiveBinding",
    "TaskSourceBinding",
    "TaskSourceCandidateEvidence",
    "TaskSourceResolver",
    "TaskSourceSelectionAction",
    "content_addressed_prompt_objective",
    "default_projection_paths",
    "default_task_source_identity",
    "resolve_objective_plan_and_output",
    "resolve_objectives",
]
