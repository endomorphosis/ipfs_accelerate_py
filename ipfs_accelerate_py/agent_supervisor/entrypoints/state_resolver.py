"""Platform state, namespace, and active-run resolution (ASE-006).

This module owns state/run *location inference only*. It does not open DuckDB,
persist a run registry, adopt processes, or select objectives (ASE-007/ASE-012).

Design rules enforced here:

- platform state defaults **outside** the source checkout;
- the default state root is stable for the same ``repository_id``;
- forks (distinct repository identities) and worktrees (when isolation is
  required) receive collision-resistant, separated namespaces;
- active-run adoption requires integrity-checked registry evidence;
- directory names, PID/status files, and prompt text are non-authoritative;
- exactly one exact compatible healthy candidate is adopted; multiple,
  incompatible, or stale candidates are reported without guessing.
"""

from __future__ import annotations

import hashlib
import os
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
    ResolutionDisposition,
    ResolutionSource,
    RevalidationRule,
    RunHealth,
    RunState,
    TargetCandidate,
    TargetInferenceDecision,
)

STATE_AND_OBJECTIVE_RESOLUTION_REQUIREMENT_ID: Final = (
    "agent_supervisor.entrypoints.state_resolver.v1"
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
STATE_EVIDENCE_SCHEMA: Final = f"{SCHEMA_PREFIX}/state-evidence@1"
STATE_RESOLUTION_SCHEMA: Final = f"{SCHEMA_PREFIX}/state-resolution@1"
RUN_CANDIDATE_EVIDENCE_SCHEMA: Final = f"{SCHEMA_PREFIX}/run-candidate-evidence@1"
RUN_CANDIDATE_RESOLUTION_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/run-candidate-resolution@1"
)

PLATFORM_PRODUCT: Final = "ipfs_accelerate_py"
PLATFORM_COMPONENT: Final = "agent_supervisor"
PLATFORM_STATE_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_HOME"
XDG_STATE_HOME_ENV: Final = "XDG_STATE_HOME"

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

TERMINAL_RUN_STATES: Final[frozenset[RunState]] = frozenset(
    {
        RunState.COMPLETED,
        RunState.CANCELLED,
        RunState.FAILED,
        RunState.REJECTED,
        RunState.QUARANTINED,
    }
)

ADOPTABLE_RUN_STATES: Final[frozenset[RunState]] = frozenset(
    {
        RunState.RECEIVED,
        RunState.RESOLVING,
        RunState.RESOLVED,
        RunState.PREVIEWING,
        RunState.ADMITTED,
        RunState.NEEDS_INPUT,
        RunState.AUTHORIZING,
        RunState.MATERIALIZING,
        RunState.STARTING,
        RunState.ADOPTING,
        RunState.RUNNING,
        RunState.BLOCKED,
        RunState.DRAINED,
    }
)

ADOPTABLE_HEALTH: Final[frozenset[RunHealth]] = frozenset(
    {
        RunHealth.HEALTHY,
        RunHealth.DEGRADED,
    }
)

_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_PATH_INJECTION_MARKERS: Final[tuple[str, ...]] = (
    "\n",
    "\r",
    "\x00",
    "\\",
)


class StateResolverError(EntrypointContractError):
    """Raised when state or run-candidate evidence is malformed or unsafe."""


class RunCandidateClass(str, Enum):
    """Classification of one observed run candidate against the target."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    STALE = "stale"
    UNVERIFIED = "unverified"


class RunAdoptionAction(str, Enum):
    """Closed adoption decision produced by :class:`RunCandidateResolver`."""

    ADOPT = "adopt"
    CREATE = "create"
    REPORT_AMBIGUOUS = "report_ambiguous"
    REPORT_STALE_OR_INCOMPATIBLE = "report_stale_or_incompatible"
    DENIED = "denied"


class WorktreeIsolationMode(str, Enum):
    """Whether run namespaces bind a checkout identity."""

    SHARED_REPOSITORY = "shared_repository"
    ISOLATE_CHECKOUT = "isolate_checkout"


def _cid(label: str, payload: Mapping[str, Any] | None = None) -> str:
    body: dict[str, Any] = {"label": label}
    if payload is not None:
        body["payload"] = dict(payload)
    return cid_for_dag_json(body)


def _require_nonempty(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise StateResolverError(f"{name} is required")
    return text


def _require_cid(value: Any, name: str) -> str:
    text = _require_nonempty(value, name)
    if not re.fullmatch(r"[A-Za-z0-9:._+/-]{8,}", text):
        raise StateResolverError(f"{name} is not a valid identity")
    return text


def _token(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not text or not _TOKEN_RE.fullmatch(text):
        raise StateResolverError(f"{name} must be a closed token")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise StateResolverError(f"{name} must be a boolean")
    return value


def _absolute_posix_path(value: Any, name: str) -> str:
    text = _require_nonempty(value, name)
    if any(marker in text for marker in _PATH_INJECTION_MARKERS):
        raise StateResolverError(f"{name} contains forbidden path characters")
    if not text.startswith("/") or "\\" in text:
        raise StateResolverError(f"{name} must be an absolute POSIX path")
    normalized = posixpath.normpath(text)
    if normalized != text or any(part == ".." for part in text.split("/")):
        raise StateResolverError(f"{name} must be lexically normalized")
    if text == "/":
        raise StateResolverError(f"{name} cannot be the filesystem root")
    return text


def _is_path_under(path: str, root: str) -> bool:
    """Return True when ``path`` is ``root`` or a descendant of ``root``."""

    if not path or not root:
        return False
    try:
        common = posixpath.commonpath((path, root))
    except ValueError:
        return False
    return common == root


def _safe_identity_segment(identity: str) -> str:
    """Return a filesystem-safe, collision-resistant segment for ``identity``."""

    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    # Keep a short readable prefix when the identity already looks token-like.
    prefix = re.sub(r"[^a-z0-9]+", "-", identity.lower())[:24].strip("-")
    if not prefix:
        prefix = "id"
    return f"{prefix}-{digest[:24]}"


def default_platform_state_home(
    *,
    environ: Mapping[str, str] | None = None,
    home_directory: str | None = None,
) -> str:
    """Resolve the platform state home **outside** any source checkout.

    Precedence:

    1. ``IPFS_ACCELERATE_AGENT_STATE_HOME`` (explicit platform home);
    2. ``$XDG_STATE_HOME/ipfs_accelerate_py/agent_supervisor``;
    3. ``$HOME/.local/state/ipfs_accelerate_py/agent_supervisor``.
    """

    env = environ if environ is not None else os.environ
    explicit = str(env.get(PLATFORM_STATE_ENV, "") or "").strip()
    if explicit:
        return _absolute_posix_path(explicit, PLATFORM_STATE_ENV)

    xdg = str(env.get(XDG_STATE_HOME_ENV, "") or "").strip()
    if xdg:
        base = _absolute_posix_path(xdg, XDG_STATE_HOME_ENV)
        return f"{base}/{PLATFORM_PRODUCT}/{PLATFORM_COMPONENT}"

    if home_directory is not None:
        home = _absolute_posix_path(home_directory, "home_directory")
    else:
        home = str(env.get("HOME", "") or "").strip()
        if not home:
            # Last-resort portable default that remains outside typical checkouts.
            home = "/var/tmp"
        home = _absolute_posix_path(home, "HOME")
    return f"{home}/.local/state/{PLATFORM_PRODUCT}/{PLATFORM_COMPONENT}"


def repository_state_root(
    repository_id: str,
    *,
    platform_state_home: str | None = None,
    environ: Mapping[str, str] | None = None,
    home_directory: str | None = None,
) -> str:
    """Return the stable, repository-keyed state root under the platform home."""

    repo_id = _require_nonempty(repository_id, "repository_id")
    home = platform_state_home or default_platform_state_home(
        environ=environ,
        home_directory=home_directory,
    )
    home = _absolute_posix_path(home, "platform_state_home")
    segment = _safe_identity_segment(repo_id)
    return f"{home}/repositories/{segment}"


def derive_run_namespace(
    *,
    repository_id: str,
    checkout_id: str = "",
    isolation: WorktreeIsolationMode = WorktreeIsolationMode.SHARED_REPOSITORY,
    board_namespace: str = "",
) -> str:
    """Derive a collision-resistant run namespace token.

    Forks with distinct ``repository_id`` values always diverge. Linked
    worktrees share a namespace under ``SHARED_REPOSITORY`` and diverge under
    ``ISOLATE_CHECKOUT`` (checkout-bound identity).
    """

    repo_id = _require_nonempty(repository_id, "repository_id")
    mode = isolation
    if not isinstance(mode, WorktreeIsolationMode):
        try:
            mode = WorktreeIsolationMode(str(mode).strip().lower())
        except ValueError as exc:
            raise StateResolverError(
                f"unknown worktree isolation mode {isolation!r}"
            ) from exc

    payload: dict[str, Any] = {
        "schema": f"{SCHEMA_PREFIX}/run-namespace@1",
        "repository_id": repo_id,
        "isolation": mode.value,
    }
    if mode is WorktreeIsolationMode.ISOLATE_CHECKOUT:
        checkout = _require_nonempty(checkout_id, "checkout_id")
        payload["checkout_id"] = checkout
    board = str(board_namespace or "").strip().lower()
    if board:
        if not _TOKEN_RE.fullmatch(board):
            raise StateResolverError("board_namespace must be a closed token")
        payload["board_namespace"] = board

    digest = hashlib.sha256(
        cid_for_dag_json(payload).encode("utf-8")
    ).hexdigest()
    return f"run-ns:{digest[:32]}"


def _candidate(
    *,
    field_name: str,
    value: str,
    source: ResolutionSource,
    precedence: int,
    evidence_cid: str,
    confidence_ppm: int = 1_000_000,
    rejection_reason: str = "",
) -> TargetCandidate:
    return TargetCandidate(
        field_name=field_name,
        value=value,
        source=source,
        source_precedence=precedence,
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
    source_precedence: int,
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
        source_precedence=source_precedence,
        evidence_cid=evidence_cid,
        candidates=tuple(candidates),
        reason_codes=tuple(reason_codes),
        effect=effect,
        override_accepted=override_accepted,
        fresh_until_ms=0,
        revalidation_rule=revalidation_rule,
    )


@dataclass(frozen=True)
class StateResolutionEvidence:
    """Frozen evidence for deterministic state-root and namespace resolution.

    Prompt text is accepted only to prove it cannot influence the result and is
    excluded from the evidence content identity.
    """

    repository_id: str
    repository_root: str
    checkout_id: str = ""
    explicit_state_root: str = ""
    signed_profile_state_root: str = ""
    signed_profile_cid: str = ""
    existing_run_state_root: str = ""
    existing_run_evidence_cid: str = ""
    repository_hint_state_root: str = ""
    board_namespace: str = ""
    isolation: WorktreeIsolationMode = WorktreeIsolationMode.SHARED_REPOSITORY
    platform_state_home: str = ""
    prompt_text: str = ""
    environ: Mapping[str, str] | None = None
    home_directory: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_id",
            _require_nonempty(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "repository_root",
            _absolute_posix_path(self.repository_root, "repository_root"),
        )
        checkout = str(self.checkout_id or "").strip()
        object.__setattr__(self, "checkout_id", checkout)

        for name in (
            "explicit_state_root",
            "signed_profile_state_root",
            "existing_run_state_root",
            "repository_hint_state_root",
            "platform_state_home",
        ):
            raw = str(getattr(self, name) or "").strip()
            if raw:
                object.__setattr__(
                    self, name, _absolute_posix_path(raw, name)
                )
            else:
                object.__setattr__(self, name, "")

        profile_cid = str(self.signed_profile_cid or "").strip()
        if self.signed_profile_state_root and not profile_cid:
            raise StateResolverError(
                "signed_profile_state_root requires signed_profile_cid"
            )
        if profile_cid:
            object.__setattr__(
                self, "signed_profile_cid", _require_cid(profile_cid, "signed_profile_cid")
            )
        else:
            object.__setattr__(self, "signed_profile_cid", "")

        run_evidence = str(self.existing_run_evidence_cid or "").strip()
        if self.existing_run_state_root and not run_evidence:
            raise StateResolverError(
                "existing_run_state_root requires existing_run_evidence_cid"
            )
        if run_evidence:
            object.__setattr__(
                self,
                "existing_run_evidence_cid",
                _require_cid(run_evidence, "existing_run_evidence_cid"),
            )
        else:
            object.__setattr__(self, "existing_run_evidence_cid", "")

        board = str(self.board_namespace or "").strip().lower()
        if board and not _TOKEN_RE.fullmatch(board):
            raise StateResolverError("board_namespace must be a closed token")
        object.__setattr__(self, "board_namespace", board)

        isolation = self.isolation
        if not isinstance(isolation, WorktreeIsolationMode):
            try:
                isolation = WorktreeIsolationMode(str(isolation).strip().lower())
            except ValueError as exc:
                raise StateResolverError(
                    f"unknown worktree isolation mode {self.isolation!r}"
                ) from exc
            object.__setattr__(self, "isolation", isolation)
        if (
            isolation is WorktreeIsolationMode.ISOLATE_CHECKOUT
            and not self.checkout_id
        ):
            raise StateResolverError(
                "isolate_checkout requires a non-empty checkout_id"
            )

        object.__setattr__(self, "prompt_text", str(self.prompt_text or ""))
        if self.environ is not None and not isinstance(self.environ, Mapping):
            raise StateResolverError("environ must be a mapping when provided")
        if self.home_directory is not None:
            object.__setattr__(
                self,
                "home_directory",
                _absolute_posix_path(self.home_directory, "home_directory"),
            )

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        # Prompt text and ambient environ are deliberately omitted so they
        # cannot change evidence identity or resolution.
        return {
            "schema": STATE_EVIDENCE_SCHEMA,
            "repository_id": self.repository_id,
            "repository_root": self.repository_root,
            "checkout_id": self.checkout_id,
            "explicit_state_root": self.explicit_state_root,
            "signed_profile_state_root": self.signed_profile_state_root,
            "signed_profile_cid": self.signed_profile_cid,
            "existing_run_state_root": self.existing_run_state_root,
            "existing_run_evidence_cid": self.existing_run_evidence_cid,
            "repository_hint_state_root": self.repository_hint_state_root,
            "board_namespace": self.board_namespace,
            "isolation": self.isolation.value,
            "platform_state_home": self.platform_state_home,
            "home_directory": self.home_directory or "",
        }


@dataclass(frozen=True)
class StateResolution:
    """Resolved platform state root, run namespace, and field decisions."""

    SCHEMA: ClassVar[str] = STATE_RESOLUTION_SCHEMA

    state_root: str
    run_namespace: str
    platform_state_home: str
    repository_id: str
    checkout_id: str
    isolation: WorktreeIsolationMode
    state_root_decision: TargetInferenceDecision
    run_namespace_decision: TargetInferenceDecision
    evidence_cid: str
    reason_codes: tuple[str, ...]
    outside_source_checkout: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state_root", _absolute_posix_path(self.state_root, "state_root")
        )
        object.__setattr__(
            self,
            "platform_state_home",
            _absolute_posix_path(self.platform_state_home, "platform_state_home"),
        )
        object.__setattr__(
            self, "run_namespace", _token(self.run_namespace, "run_namespace")
        )
        object.__setattr__(
            self,
            "repository_id",
            _require_nonempty(self.repository_id, "repository_id"),
        )
        object.__setattr__(self, "checkout_id", str(self.checkout_id or ""))
        if not isinstance(self.isolation, WorktreeIsolationMode):
            raise StateResolverError("isolation must be a WorktreeIsolationMode")
        if not isinstance(self.state_root_decision, TargetInferenceDecision):
            raise StateResolverError(
                "state_root_decision must be TargetInferenceDecision"
            )
        if not isinstance(self.run_namespace_decision, TargetInferenceDecision):
            raise StateResolverError(
                "run_namespace_decision must be TargetInferenceDecision"
            )
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes),
        )
        object.__setattr__(
            self,
            "outside_source_checkout",
            _bool(self.outside_source_checkout, "outside_source_checkout"),
        )
        if not self.outside_source_checkout:
            raise StateResolverError(
                "resolved state_root must remain outside the source checkout"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "state_root": self.state_root,
            "run_namespace": self.run_namespace,
            "platform_state_home": self.platform_state_home,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "isolation": self.isolation.value,
            "state_root_decision": self.state_root_decision.to_dict(),
            "run_namespace_decision": self.run_namespace_decision.to_dict(),
            "evidence_cid": self.evidence_cid,
            "reason_codes": list(self.reason_codes),
            "outside_source_checkout": self.outside_source_checkout,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


class StateRootResolver:
    """Resolve a platform state root and collision-resistant run namespace."""

    def resolve(self, evidence: StateResolutionEvidence) -> StateResolution:
        if not isinstance(evidence, StateResolutionEvidence):
            raise StateResolverError(
                "evidence must be StateResolutionEvidence"
            )
        evidence_cid = evidence.content_id
        platform_home = evidence.platform_state_home or default_platform_state_home(
            environ=evidence.environ,
            home_directory=evidence.home_directory,
        )
        platform_home = _absolute_posix_path(platform_home, "platform_state_home")

        # Collect admissible candidates first, then select by precedence so
        # non-selected alternatives always carry typed rejection reasons.
        # Reject in-checkout locations; repository-local defaults are the gap
        # this resolver closes.
        def _outside_checkout_or_reason(path: str) -> str:
            if _is_path_under(path, evidence.repository_root):
                return "state_root_inside_source_checkout"
            if path == evidence.repository_root:
                return "state_root_inside_source_checkout"
            return ""

        # Each entry: (source, value, evidence_ref, hard_rejection_or_empty)
        considered: list[
            tuple[ResolutionSource, str, str, str]
        ] = []
        reasons: list[str] = []

        if evidence.explicit_state_root:
            reject = _outside_checkout_or_reason(evidence.explicit_state_root)
            considered.append(
                (
                    ResolutionSource.EXPLICIT_OVERRIDE,
                    evidence.explicit_state_root,
                    evidence_cid,
                    reject,
                )
            )
            if reject:
                reasons.append("explicit_state_root_inside_checkout_rejected")
            else:
                reasons.append("explicit_state_root_accepted")

        if evidence.existing_run_state_root:
            reject = _outside_checkout_or_reason(evidence.existing_run_state_root)
            considered.append(
                (
                    ResolutionSource.EXISTING_RUN,
                    evidence.existing_run_state_root,
                    evidence.existing_run_evidence_cid,
                    reject,
                )
            )
            if reject:
                reasons.append("existing_run_state_root_inside_checkout_rejected")
            else:
                reasons.append("existing_run_state_root_considered")

        if evidence.signed_profile_state_root:
            reject = _outside_checkout_or_reason(evidence.signed_profile_state_root)
            considered.append(
                (
                    ResolutionSource.SIGNED_PROFILE,
                    evidence.signed_profile_state_root,
                    evidence.signed_profile_cid,
                    reject,
                )
            )
            if reject:
                reasons.append("signed_profile_state_root_inside_checkout_rejected")
            else:
                reasons.append("signed_profile_state_root_considered")

        if evidence.repository_hint_state_root:
            # Repository files may only *hint*; they never place state and are
            # never authoritative for the selected state root.
            reject = _outside_checkout_or_reason(evidence.repository_hint_state_root)
            if not reject:
                reject = "repository_hint_non_authoritative_for_state_root"
                reasons.append("repository_hint_state_root_non_authoritative")
            else:
                reasons.append("repository_hint_state_root_inside_checkout_rejected")
            considered.append(
                (
                    ResolutionSource.REPOSITORY_HINT,
                    evidence.repository_hint_state_root,
                    evidence_cid,
                    reject,
                )
            )

        builtin_root = repository_state_root(
            evidence.repository_id,
            platform_state_home=platform_home,
        )
        reject_builtin = _outside_checkout_or_reason(builtin_root)
        considered.append(
            (
                ResolutionSource.BUILTIN_DEFAULT,
                builtin_root,
                evidence_cid,
                reject_builtin,
            )
        )
        if reject_builtin:
            reasons.append("platform_state_home_inside_checkout")
        else:
            reasons.append("platform_repository_keyed_default")

        # Select the highest-precedence admissible candidate (lowest rank).
        admissible = [
            item for item in considered if not item[3]
        ]
        if not admissible:
            raise StateResolverError(
                "platform state home resolves inside the source checkout"
            )
        admissible.sort(key=lambda item: SOURCE_PRECEDENCE[item[0]])
        selected_source, selected_root, selected_evidence, _ = admissible[0]

        disposition = ResolutionDisposition.DEFAULTED
        override_accepted = False
        if selected_source is ResolutionSource.EXPLICIT_OVERRIDE:
            disposition = ResolutionDisposition.UNIQUE
            override_accepted = True
        elif selected_source in {
            ResolutionSource.EXISTING_RUN,
            ResolutionSource.SIGNED_PROFILE,
        }:
            disposition = ResolutionDisposition.UNIQUE

        candidates: list[TargetCandidate] = []
        for source, value, evidence_ref, hard_reject in considered:
            if source is selected_source and value == selected_root and not hard_reject:
                rejection = ""
            elif hard_reject:
                rejection = hard_reject
            else:
                rejection = "superseded_by_higher_precedence_source"
            candidates.append(
                _candidate(
                    field_name="state_root",
                    value=value,
                    source=source,
                    precedence=SOURCE_PRECEDENCE[source],
                    evidence_cid=evidence_ref,
                    confidence_ppm=1_000_000 if not rejection else 0,
                    rejection_reason=rejection,
                )
            )

        if _is_path_under(selected_root, evidence.repository_root):
            raise StateResolverError(
                "resolved state_root must remain outside the source checkout"
            )

        state_decision = _decision(
            field_name="state_root",
            disposition=disposition,
            selected_value=selected_root,
            selected_source=selected_source,
            source_precedence=SOURCE_PRECEDENCE[selected_source],
            evidence_cid=evidence_cid,
            candidates=candidates,
            reason_codes=tuple(reasons),
            effect=DecisionEffect.CONFIGURATION,
            override_accepted=override_accepted,
        )

        run_namespace = derive_run_namespace(
            repository_id=evidence.repository_id,
            checkout_id=evidence.checkout_id,
            isolation=evidence.isolation,
            board_namespace=evidence.board_namespace,
        )
        ns_reasons = [
            "collision_resistant_repository_namespace",
            f"isolation_{evidence.isolation.value}",
        ]
        if evidence.board_namespace:
            ns_reasons.append("board_namespace_bound")
        if evidence.isolation is WorktreeIsolationMode.ISOLATE_CHECKOUT:
            ns_reasons.append("checkout_bound_namespace")
        else:
            ns_reasons.append("worktrees_share_repository_namespace")

        # Prompt text cannot select state roots or namespaces.
        if evidence.prompt_text:
            ns_reasons.append("prompt_text_ignored")
            reasons.append("prompt_text_ignored")

        ns_decision = _decision(
            field_name="run_namespace",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=run_namespace,
            selected_source=ResolutionSource.BUILTIN_DEFAULT,
            source_precedence=SOURCE_PRECEDENCE[ResolutionSource.BUILTIN_DEFAULT],
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="run_namespace",
                    value=run_namespace,
                    source=ResolutionSource.BUILTIN_DEFAULT,
                    precedence=SOURCE_PRECEDENCE[ResolutionSource.BUILTIN_DEFAULT],
                    evidence_cid=evidence_cid,
                ),
            ),
            reason_codes=tuple(ns_reasons),
            effect=DecisionEffect.IDENTITY_ONLY,
        )

        return StateResolution(
            state_root=selected_root,
            run_namespace=run_namespace,
            platform_state_home=platform_home,
            repository_id=evidence.repository_id,
            checkout_id=evidence.checkout_id,
            isolation=evidence.isolation,
            state_root_decision=state_decision,
            run_namespace_decision=ns_decision,
            evidence_cid=evidence_cid,
            reason_codes=tuple(dict.fromkeys(reasons + ns_reasons)),
            outside_source_checkout=True,
        )


@dataclass(frozen=True)
class RunCandidateEvidence:
    """One integrity-checked (or explicitly unverified) run observation.

    Candidates without a registry integrity CID are classified ``unverified``
    and never adopted. Directory names and PID/status files alone are not
    sufficient authority.
    """

    run_id: str
    run_namespace: str
    repository_id: str
    checkout_id: str = ""
    state: RunState = RunState.RUNNING
    health: RunHealth = RunHealth.HEALTHY
    registry_integrity_cid: str = ""
    objective_cid: str = ""
    profile_cid: str = ""
    state_revision_cid: str = ""
    observed_from_directory_name: bool = False
    observed_from_pid_file: bool = False
    stale_marker: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_cid(self.run_id, "run_id"))
        object.__setattr__(
            self, "run_namespace", _token(self.run_namespace, "run_namespace")
        )
        object.__setattr__(
            self,
            "repository_id",
            _require_nonempty(self.repository_id, "repository_id"),
        )
        object.__setattr__(self, "checkout_id", str(self.checkout_id or "").strip())

        state = self.state
        if not isinstance(state, RunState):
            try:
                state = RunState(str(state).strip().lower())
            except ValueError as exc:
                raise StateResolverError(f"unknown run state {self.state!r}") from exc
            object.__setattr__(self, "state", state)

        health = self.health
        if not isinstance(health, RunHealth):
            try:
                health = RunHealth(str(health).strip().lower())
            except ValueError as exc:
                raise StateResolverError(f"unknown run health {self.health!r}") from exc
            object.__setattr__(self, "health", health)

        integrity = str(self.registry_integrity_cid or "").strip()
        if integrity:
            object.__setattr__(
                self,
                "registry_integrity_cid",
                _require_cid(integrity, "registry_integrity_cid"),
            )
        else:
            object.__setattr__(self, "registry_integrity_cid", "")

        for name in ("objective_cid", "profile_cid", "state_revision_cid"):
            raw = str(getattr(self, name) or "").strip()
            if raw:
                object.__setattr__(self, name, _require_cid(raw, name))
            else:
                object.__setattr__(self, name, "")

        for name in (
            "observed_from_directory_name",
            "observed_from_pid_file",
            "stale_marker",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_CANDIDATE_EVIDENCE_SCHEMA,
            "run_id": self.run_id,
            "run_namespace": self.run_namespace,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "state": self.state.value,
            "health": self.health.value,
            "registry_integrity_cid": self.registry_integrity_cid,
            "objective_cid": self.objective_cid,
            "profile_cid": self.profile_cid,
            "state_revision_cid": self.state_revision_cid,
            "observed_from_directory_name": self.observed_from_directory_name,
            "observed_from_pid_file": self.observed_from_pid_file,
            "stale_marker": self.stale_marker,
        }


@dataclass(frozen=True)
class ClassifiedRunCandidate:
    """A run candidate plus its closed classification and reason codes."""

    candidate: RunCandidateEvidence
    classification: RunCandidateClass
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "classification": self.classification.value,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RunCandidateResolution:
    """Adoption decision over integrity-checked run candidates."""

    SCHEMA: ClassVar[str] = RUN_CANDIDATE_RESOLUTION_SCHEMA

    action: RunAdoptionAction
    disposition: ResolutionDisposition
    selected_run_id: str
    target_run_namespace: str
    target_repository_id: str
    target_checkout_id: str
    classified: tuple[ClassifiedRunCandidate, ...]
    alternatives: tuple[ClassifiedRunCandidate, ...]
    decision: TargetInferenceDecision
    reason_codes: tuple[str, ...]
    evidence_cid: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, RunAdoptionAction):
            raise StateResolverError("action must be a RunAdoptionAction")
        if not isinstance(self.disposition, ResolutionDisposition):
            raise StateResolverError("disposition must be a ResolutionDisposition")
        selected = str(self.selected_run_id or "").strip()
        if selected:
            object.__setattr__(
                self, "selected_run_id", _require_cid(selected, "selected_run_id")
            )
        else:
            object.__setattr__(self, "selected_run_id", "")
        object.__setattr__(
            self,
            "target_run_namespace",
            _token(self.target_run_namespace, "target_run_namespace"),
        )
        object.__setattr__(
            self,
            "target_repository_id",
            _require_nonempty(self.target_repository_id, "target_repository_id"),
        )
        object.__setattr__(
            self, "target_checkout_id", str(self.target_checkout_id or "")
        )
        object.__setattr__(self, "classified", tuple(self.classified))
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        if not isinstance(self.decision, TargetInferenceDecision):
            raise StateResolverError("decision must be TargetInferenceDecision")
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )

        if self.action is RunAdoptionAction.ADOPT:
            if not self.selected_run_id:
                raise StateResolverError("adopt action requires selected_run_id")
            if self.disposition is not ResolutionDisposition.UNIQUE:
                raise StateResolverError("adopt action requires unique disposition")
        if self.action is RunAdoptionAction.REPORT_AMBIGUOUS:
            if self.selected_run_id:
                raise StateResolverError(
                    "ambiguous action cannot select a run without authority"
                )
            if self.disposition is not ResolutionDisposition.AMBIGUOUS:
                raise StateResolverError(
                    "report_ambiguous requires ambiguous disposition"
                )
        if self.action is RunAdoptionAction.CREATE:
            if self.selected_run_id:
                raise StateResolverError("create action cannot adopt a run_id")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "action": self.action.value,
            "disposition": self.disposition.value,
            "selected_run_id": self.selected_run_id,
            "target_run_namespace": self.target_run_namespace,
            "target_repository_id": self.target_repository_id,
            "target_checkout_id": self.target_checkout_id,
            "classified": [item.to_dict() for item in self.classified],
            "alternatives": [item.to_dict() for item in self.alternatives],
            "decision": self.decision.to_dict(),
            "reason_codes": list(self.reason_codes),
            "evidence_cid": self.evidence_cid,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class RunCandidateResolutionRequest:
    """Target identity plus frozen candidate population for adoption."""

    repository_id: str
    run_namespace: str
    checkout_id: str = ""
    isolation: WorktreeIsolationMode = WorktreeIsolationMode.SHARED_REPOSITORY
    candidates: tuple[RunCandidateEvidence, ...] = ()
    explicit_run_id: str = ""
    expected_objective_cid: str = ""
    expected_profile_cid: str = ""
    prompt_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_id",
            _require_nonempty(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self, "run_namespace", _token(self.run_namespace, "run_namespace")
        )
        object.__setattr__(self, "checkout_id", str(self.checkout_id or "").strip())
        isolation = self.isolation
        if not isinstance(isolation, WorktreeIsolationMode):
            try:
                isolation = WorktreeIsolationMode(str(isolation).strip().lower())
            except ValueError as exc:
                raise StateResolverError(
                    f"unknown worktree isolation mode {self.isolation!r}"
                ) from exc
            object.__setattr__(self, "isolation", isolation)
        if (
            isolation is WorktreeIsolationMode.ISOLATE_CHECKOUT
            and not self.checkout_id
        ):
            raise StateResolverError(
                "isolate_checkout requires a non-empty checkout_id"
            )

        normalized: list[RunCandidateEvidence] = []
        if not isinstance(self.candidates, Sequence) or isinstance(
            self.candidates, (str, bytes)
        ):
            raise StateResolverError("candidates must be a sequence")
        for index, item in enumerate(self.candidates):
            if not isinstance(item, RunCandidateEvidence):
                raise StateResolverError(
                    f"candidates[{index}] must be RunCandidateEvidence"
                )
            normalized.append(item)
        object.__setattr__(self, "candidates", tuple(normalized))

        explicit = str(self.explicit_run_id or "").strip()
        if explicit:
            object.__setattr__(
                self, "explicit_run_id", _require_cid(explicit, "explicit_run_id")
            )
        else:
            object.__setattr__(self, "explicit_run_id", "")

        for name in ("expected_objective_cid", "expected_profile_cid"):
            raw = str(getattr(self, name) or "").strip()
            if raw:
                object.__setattr__(self, name, _require_cid(raw, name))
            else:
                object.__setattr__(self, name, "")

        object.__setattr__(self, "prompt_text", str(self.prompt_text or ""))

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_PREFIX}/run-candidate-request@1",
            "repository_id": self.repository_id,
            "run_namespace": self.run_namespace,
            "checkout_id": self.checkout_id,
            "isolation": self.isolation.value,
            "candidates": [item.to_dict() for item in self.candidates],
            "explicit_run_id": self.explicit_run_id,
            "expected_objective_cid": self.expected_objective_cid,
            "expected_profile_cid": self.expected_profile_cid,
            # prompt_text omitted from identity
        }


def classify_run_candidate(
    candidate: RunCandidateEvidence,
    *,
    target_repository_id: str,
    target_run_namespace: str,
    target_checkout_id: str = "",
    isolation: WorktreeIsolationMode = WorktreeIsolationMode.SHARED_REPOSITORY,
    expected_objective_cid: str = "",
    expected_profile_cid: str = "",
) -> ClassifiedRunCandidate:
    """Classify one candidate without selecting among peers."""

    if not isinstance(candidate, RunCandidateEvidence):
        raise StateResolverError("candidate must be RunCandidateEvidence")

    reasons: list[str] = []

    # Non-authoritative signals never yield a compatible adoption.
    if candidate.observed_from_directory_name:
        reasons.append("directory_name_non_authoritative")
    if candidate.observed_from_pid_file:
        reasons.append("pid_file_non_authoritative")
    if not candidate.registry_integrity_cid:
        reasons.append("missing_registry_integrity_cid")
        return ClassifiedRunCandidate(
            candidate=candidate,
            classification=RunCandidateClass.UNVERIFIED,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    if candidate.stale_marker:
        reasons.append("stale_marker")
    if candidate.state in TERMINAL_RUN_STATES:
        reasons.append(f"terminal_state_{candidate.state.value}")
    if candidate.health in {RunHealth.TERMINAL, RunHealth.UNHEALTHY, RunHealth.UNKNOWN}:
        reasons.append(f"health_{candidate.health.value}")
    if candidate.state not in ADOPTABLE_RUN_STATES:
        reasons.append(f"non_adoptable_state_{candidate.state.value}")
    if candidate.health not in ADOPTABLE_HEALTH:
        reasons.append(f"non_adoptable_health_{candidate.health.value}")

    if reasons and (
        candidate.stale_marker
        or candidate.state in TERMINAL_RUN_STATES
        or candidate.health is RunHealth.TERMINAL
        or candidate.health is RunHealth.UNHEALTHY
    ):
        return ClassifiedRunCandidate(
            candidate=candidate,
            classification=RunCandidateClass.STALE,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    identity_mismatch = False
    if candidate.repository_id != target_repository_id:
        reasons.append("repository_id_mismatch")
        identity_mismatch = True
    if candidate.run_namespace != target_run_namespace:
        reasons.append("run_namespace_mismatch")
        identity_mismatch = True
    if isolation is WorktreeIsolationMode.ISOLATE_CHECKOUT:
        if candidate.checkout_id != target_checkout_id:
            reasons.append("checkout_id_mismatch")
            identity_mismatch = True
    if expected_objective_cid and candidate.objective_cid:
        if candidate.objective_cid != expected_objective_cid:
            reasons.append("objective_cid_mismatch")
            identity_mismatch = True
    if expected_profile_cid and candidate.profile_cid:
        if candidate.profile_cid != expected_profile_cid:
            reasons.append("profile_cid_mismatch")
            identity_mismatch = True

    if identity_mismatch:
        return ClassifiedRunCandidate(
            candidate=candidate,
            classification=RunCandidateClass.INCOMPATIBLE,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    if reasons:
        # Residual non-stale issues (e.g. degraded handled above as adoptable).
        # If we still have non-adoptable flags, treat as stale rather than guess.
        return ClassifiedRunCandidate(
            candidate=candidate,
            classification=RunCandidateClass.STALE,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    return ClassifiedRunCandidate(
        candidate=candidate,
        classification=RunCandidateClass.COMPATIBLE,
        reason_codes=("exact_compatible_match",),
    )


class RunCandidateResolver:
    """Classify and adopt at most one exact compatible run candidate."""

    def resolve(
        self, request: RunCandidateResolutionRequest
    ) -> RunCandidateResolution:
        if not isinstance(request, RunCandidateResolutionRequest):
            raise StateResolverError(
                "request must be RunCandidateResolutionRequest"
            )
        evidence_cid = request.content_id
        classified: list[ClassifiedRunCandidate] = []
        for candidate in request.candidates:
            classified.append(
                classify_run_candidate(
                    candidate,
                    target_repository_id=request.repository_id,
                    target_run_namespace=request.run_namespace,
                    target_checkout_id=request.checkout_id,
                    isolation=request.isolation,
                    expected_objective_cid=request.expected_objective_cid,
                    expected_profile_cid=request.expected_profile_cid,
                )
            )

        # Deterministic ordering by run_id for stable receipts.
        classified.sort(key=lambda item: item.candidate.run_id)

        compatible = [
            item
            for item in classified
            if item.classification is RunCandidateClass.COMPATIBLE
        ]
        stale = [
            item
            for item in classified
            if item.classification is RunCandidateClass.STALE
        ]
        incompatible = [
            item
            for item in classified
            if item.classification is RunCandidateClass.INCOMPATIBLE
        ]
        unverified = [
            item
            for item in classified
            if item.classification is RunCandidateClass.UNVERIFIED
        ]

        reasons: list[str] = []
        if request.prompt_text:
            reasons.append("prompt_text_ignored")

        selected_run_id = ""
        action = RunAdoptionAction.CREATE
        disposition = ResolutionDisposition.DEFAULTED
        selected_source = ResolutionSource.BUILTIN_DEFAULT
        override_accepted = False
        # Synthetic value used when no existing run is adopted.
        create_value = "create-new-run"

        if request.explicit_run_id:
            matches = [
                item
                for item in classified
                if item.candidate.run_id == request.explicit_run_id
            ]
            if not matches:
                action = RunAdoptionAction.DENIED
                disposition = ResolutionDisposition.DENIED
                reasons.append("explicit_run_id_not_found")
            else:
                match = matches[0]
                if match.classification is RunCandidateClass.COMPATIBLE:
                    selected_run_id = match.candidate.run_id
                    action = RunAdoptionAction.ADOPT
                    disposition = ResolutionDisposition.UNIQUE
                    selected_source = ResolutionSource.EXPLICIT_OVERRIDE
                    override_accepted = True
                    reasons.append("explicit_run_id_adopted")
                elif match.classification is RunCandidateClass.STALE:
                    action = RunAdoptionAction.DENIED
                    disposition = ResolutionDisposition.DENIED
                    reasons.append("explicit_run_id_stale")
                elif match.classification is RunCandidateClass.INCOMPATIBLE:
                    action = RunAdoptionAction.DENIED
                    disposition = ResolutionDisposition.DENIED
                    reasons.append("explicit_run_id_incompatible")
                else:
                    action = RunAdoptionAction.DENIED
                    disposition = ResolutionDisposition.DENIED
                    reasons.append("explicit_run_id_unverified")
        elif len(compatible) == 1:
            selected_run_id = compatible[0].candidate.run_id
            action = RunAdoptionAction.ADOPT
            disposition = ResolutionDisposition.UNIQUE
            selected_source = ResolutionSource.EXISTING_RUN
            reasons.append("unique_compatible_run_adopted")
        elif len(compatible) > 1:
            action = RunAdoptionAction.REPORT_AMBIGUOUS
            disposition = ResolutionDisposition.AMBIGUOUS
            reasons.append("multiple_compatible_runs")
            reasons.append("no_guess_among_compatible_candidates")
        elif classified:
            # Never adopt non-compatible candidates. Report them as alternatives
            # and default to creating a new run without guessing.
            if stale:
                reasons.append("only_stale_candidates")
            if incompatible:
                reasons.append("only_incompatible_candidates")
            if unverified:
                reasons.append("only_unverified_candidates")
            reasons.append("create_new_run_without_adoption")
            reasons.append("default_create_new_run")
            action = RunAdoptionAction.CREATE
            disposition = ResolutionDisposition.DEFAULTED
            selected_source = ResolutionSource.BUILTIN_DEFAULT
        else:
            action = RunAdoptionAction.CREATE
            disposition = ResolutionDisposition.DEFAULTED
            selected_source = ResolutionSource.BUILTIN_DEFAULT
            reasons.append("no_existing_run_candidates")
            reasons.append("default_create_new_run")

        decision_candidates: list[TargetCandidate] = []
        for item in classified:
            # Registry observations are always EXISTING_RUN. Explicit adoption
            # adds a separate EXPLICIT_OVERRIDE candidate below so the selected
            # value/source pair remains unique.
            is_selected_existing = (
                disposition is ResolutionDisposition.UNIQUE
                and selected_source is ResolutionSource.EXISTING_RUN
                and selected_run_id
                and item.candidate.run_id == selected_run_id
            )
            if is_selected_existing:
                rejection = ""
            elif item.reason_codes:
                rejection = item.reason_codes[0]
            else:
                rejection = item.classification.value
            if (
                selected_source is ResolutionSource.EXPLICIT_OVERRIDE
                and item.candidate.run_id == selected_run_id
                and not rejection
            ):
                rejection = "observed_as_existing_run"
            decision_candidates.append(
                _candidate(
                    field_name="run",
                    value=item.candidate.run_id,
                    source=ResolutionSource.EXISTING_RUN,
                    precedence=SOURCE_PRECEDENCE[ResolutionSource.EXISTING_RUN],
                    evidence_cid=item.candidate.registry_integrity_cid
                    or item.candidate.content_id,
                    confidence_ppm=1_000_000 if not rejection else 0,
                    rejection_reason=rejection,
                )
            )

        if (
            disposition is ResolutionDisposition.UNIQUE
            and selected_source is ResolutionSource.EXPLICIT_OVERRIDE
            and selected_run_id
        ):
            decision_candidates.append(
                _candidate(
                    field_name="run",
                    value=selected_run_id,
                    source=ResolutionSource.EXPLICIT_OVERRIDE,
                    precedence=SOURCE_PRECEDENCE[ResolutionSource.EXPLICIT_OVERRIDE],
                    evidence_cid=evidence_cid,
                )
            )

        selected_value = selected_run_id
        if action is RunAdoptionAction.CREATE:
            selected_value = create_value
            decision_candidates.append(
                _candidate(
                    field_name="run",
                    value=create_value,
                    source=ResolutionSource.BUILTIN_DEFAULT,
                    precedence=SOURCE_PRECEDENCE[ResolutionSource.BUILTIN_DEFAULT],
                    evidence_cid=evidence_cid,
                )
            )
            # Existing observations remain alternatives with rejections.
            decision_candidates = [
                item
                if item.value != create_value
                or item.source is ResolutionSource.BUILTIN_DEFAULT
                else item
                for item in decision_candidates
            ]
            # Ensure every non-create candidate is rejected.
            repaired: list[TargetCandidate] = []
            for item in decision_candidates:
                if (
                    item.value == create_value
                    and item.source is ResolutionSource.BUILTIN_DEFAULT
                ):
                    repaired.append(item)
                elif item.rejection_reason:
                    repaired.append(item)
                else:
                    repaired.append(
                        _candidate(
                            field_name="run",
                            value=item.value,
                            source=item.source,
                            precedence=item.source_precedence,
                            evidence_cid=item.evidence_cid,
                            confidence_ppm=0,
                            rejection_reason="not_adopted_create_new_run",
                        )
                    )
            decision_candidates = repaired
        elif disposition in {
            ResolutionDisposition.AMBIGUOUS,
            ResolutionDisposition.DENIED,
            ResolutionDisposition.UNAVAILABLE,
        }:
            selected_value = ""
            # Unresolved dispositions cannot select; ensure rejections exist
            # for bookkeeping where helpful (not required by contract).
            repaired = []
            for item in decision_candidates:
                if item.rejection_reason:
                    repaired.append(item)
                else:
                    repaired.append(
                        _candidate(
                            field_name="run",
                            value=item.value,
                            source=item.source,
                            precedence=item.source_precedence,
                            evidence_cid=item.evidence_cid,
                            confidence_ppm=0,
                            rejection_reason=(
                                "ambiguous_compatible_candidate"
                                if disposition is ResolutionDisposition.AMBIGUOUS
                                else "not_selected"
                            ),
                        )
                    )
            decision_candidates = repaired
        elif (
            disposition is ResolutionDisposition.UNIQUE
            and selected_source is ResolutionSource.EXISTING_RUN
        ):
            # Reject non-selected peers.
            repaired = []
            for item in decision_candidates:
                if item.value == selected_run_id and not item.rejection_reason:
                    repaired.append(item)
                elif item.rejection_reason:
                    repaired.append(item)
                else:
                    repaired.append(
                        _candidate(
                            field_name="run",
                            value=item.value,
                            source=item.source,
                            precedence=item.source_precedence,
                            evidence_cid=item.evidence_cid,
                            confidence_ppm=0,
                            rejection_reason="not_selected_unique_compatible",
                        )
                    )
            decision_candidates = repaired

        alternatives = tuple(
            item
            for item in classified
            if item.candidate.run_id != selected_run_id
        )

        decision = _decision(
            field_name="run",
            disposition=disposition,
            selected_value=selected_value,
            selected_source=selected_source
            if selected_value
            else ResolutionSource.BUILTIN_DEFAULT,
            source_precedence=SOURCE_PRECEDENCE[
                selected_source
                if selected_value
                else ResolutionSource.BUILTIN_DEFAULT
            ],
            evidence_cid=evidence_cid,
            candidates=decision_candidates,
            reason_codes=tuple(dict.fromkeys(reasons)),
            effect=DecisionEffect.IDENTITY_ONLY,
            override_accepted=override_accepted,
        )

        return RunCandidateResolution(
            action=action,
            disposition=disposition,
            selected_run_id=selected_run_id,
            target_run_namespace=request.run_namespace,
            target_repository_id=request.repository_id,
            target_checkout_id=request.checkout_id,
            classified=tuple(classified),
            alternatives=alternatives,
            decision=decision,
            reason_codes=tuple(dict.fromkeys(reasons)),
            evidence_cid=evidence_cid,
        )


def resolve_state(evidence: StateResolutionEvidence) -> StateResolution:
    """Module-level convenience wrapper around :class:`StateRootResolver`."""

    return StateRootResolver().resolve(evidence)


def resolve_run_candidates(
    request: RunCandidateResolutionRequest,
) -> RunCandidateResolution:
    """Module-level convenience wrapper around :class:`RunCandidateResolver`."""

    return RunCandidateResolver().resolve(request)


def resolve_platform_state_and_runs(
    state_evidence: StateResolutionEvidence,
    *,
    run_candidates: Sequence[RunCandidateEvidence] = (),
    explicit_run_id: str = "",
    expected_objective_cid: str = "",
    expected_profile_cid: str = "",
) -> tuple[StateResolution, RunCandidateResolution]:
    """Resolve state root/namespace then classify active-run candidates."""

    state = resolve_state(state_evidence)
    run_resolution = resolve_run_candidates(
        RunCandidateResolutionRequest(
            repository_id=state.repository_id,
            run_namespace=state.run_namespace,
            checkout_id=state.checkout_id,
            isolation=state.isolation,
            candidates=tuple(run_candidates),
            explicit_run_id=explicit_run_id,
            expected_objective_cid=expected_objective_cid,
            expected_profile_cid=expected_profile_cid,
            prompt_text=state_evidence.prompt_text,
        )
    )
    return state, run_resolution


__all__ = [
    "ADOPTABLE_HEALTH",
    "ADOPTABLE_RUN_STATES",
    "PLATFORM_COMPONENT",
    "PLATFORM_PRODUCT",
    "PLATFORM_STATE_ENV",
    "RUN_CANDIDATE_EVIDENCE_SCHEMA",
    "RUN_CANDIDATE_RESOLUTION_SCHEMA",
    "SOURCE_PRECEDENCE",
    "STATE_AND_OBJECTIVE_RESOLUTION_REQUIREMENT_ID",
    "STATE_EVIDENCE_SCHEMA",
    "STATE_RESOLUTION_SCHEMA",
    "TERMINAL_RUN_STATES",
    "XDG_STATE_HOME_ENV",
    "ClassifiedRunCandidate",
    "RunAdoptionAction",
    "RunCandidateClass",
    "RunCandidateEvidence",
    "RunCandidateResolution",
    "RunCandidateResolutionRequest",
    "RunCandidateResolver",
    "StateResolution",
    "StateResolutionEvidence",
    "StateResolverError",
    "StateRootResolver",
    "WorktreeIsolationMode",
    "classify_run_candidate",
    "default_platform_state_home",
    "derive_run_namespace",
    "repository_state_root",
    "resolve_platform_state_and_runs",
    "resolve_run_candidates",
    "resolve_state",
]
