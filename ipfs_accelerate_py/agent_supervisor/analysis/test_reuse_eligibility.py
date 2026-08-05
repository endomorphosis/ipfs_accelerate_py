"""Conservative reuse eligibility classification (PTR-022).

``TestReuseEligibilityDecision@1`` composes static and runtime dependency
evidence with repository-forest, dirty-state, parameter, and adapter policy.
It classifies items into the closed eligibility lattice and never authorizes a
skip from heuristic similarity.

Authority doctrine (fail-closed):

* rollout v1 always binds the full admitted repository-forest CID when an item
  is classified reusable under any positive class;
* incomplete static/runtime analysis, uncontrolled effects, unsupported
  parameters, unaccounted dirty state, and missing effect/snapshot adapters
  always produce ``RUN`` with class ``non_reusable``;
* no similarity score, embedding, model verdict, runtime-overlap fraction, or
  unchanged-line heuristic may elevate eligibility or authorize reuse;
* this module never emits ``SKIP``; certificate verification remains a later,
  separate authority chain.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    ContentIdentity,
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_runtime_dependency_trace import (
    RuntimeTestDependencyTrace,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_static_dependency_trace import (
    StaticTestDependencyTrace,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    EligibilityClass,
    ReuseAction,
    ReuseReasonCode,
)

TEST_REUSE_ELIGIBILITY_DECISION_INTERFACE: Final = "TestReuseEligibilityDecision@1"
TEST_REUSE_ELIGIBILITY_EVALUATOR_INTERFACE: Final = "TestReuseEligibilityEvaluator@1"
TEST_REUSE_ELIGIBILITY_POLICY_INTERFACE: Final = "TestReuseEligibilityPolicy@1"

TEST_REUSE_ELIGIBILITY_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-reuse-eligibility-decision@1"
)
TEST_REUSE_ELIGIBILITY_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-reuse-eligibility-policy@1"
)

# v1 always binds the admitted repository forest for reusable classes.
ROLLOUT_SCOPE_REPOSITORY_FOREST: Final = "repository_forest"
DEFAULT_ROLLOUT_SCOPE: Final = ROLLOUT_SCOPE_REPOSITORY_FOREST

_CID_RE: Final = re.compile(r"^b[a-z2-7]{20,}$")
_SAFE_REASON_RE: Final = re.compile(r"^[A-Za-z0-9_.:@/+-]{1,128}$")
_SAFE_ADAPTER_RE: Final = re.compile(r"^[A-Za-z0-9_.:@/+-]{1,128}$")
_MAX_REASONS: Final = 64
_MAX_DIAGNOSTIC_KEYS: Final = 32
_MAX_DIAGNOSTIC_CHARS: Final = 512
_MAX_ADAPTERS: Final = 64
_MAX_SNAPSHOTS: Final = 64
_MAX_UNACCOUNTED_PATHS: Final = 256

# Static frontiers that always block positive eligibility when present.
_BLOCKING_STATIC_FRONTIERS: Final = frozenset(
    {
        "analysis_bound",
        "ambiguous_fixture",
        "ambiguous_import",
        "dynamic_import",
        "missing_file",
        "missing_test_symbol",
        "native_code",
        "opaque_decorator",
        "parse_error",
        "indexed_parse_error",
        "reflection",
        "stale_ast_index",
        "unresolved_fixture",
    }
)

# Effect kinds that require an explicit reviewed adapter to leave pure.
_ADAPTER_REQUIRING_EFFECTS: Final = frozenset(
    {
        "subprocess",
        "network",
        "clock",
        "randomness",
        "environment",
        "hardware",
        "filesystem_write",
        "filesystem",
        "service",
    }
)

# Heuristic fields that must never authorize reuse (ignored for elevation).
_FORBIDDEN_HEURISTIC_KEYS: Final = frozenset(
    {
        "similarity",
        "similarity_score",
        "similarity_threshold",
        "embedding_score",
        "model_verdict",
        "runtime_overlap",
        "unchanged_line_heuristic",
        "ast_similarity",
        "heuristic_score",
    }
)


class TestReuseEligibilityError(ValueError):
    """Invalid eligibility input or policy material."""

    __test__ = False


class EligibilityDenyReason(str, Enum):  # noqa: UP042 - project supports older Python
    """Typed deny reasons retained on the decision payload."""

    MISSING_REPOSITORY_FOREST = "missing_repository_forest"
    MISSING_STATIC_TRACE = "missing_static_trace"
    MISSING_RUNTIME_TRACE = "missing_runtime_trace"
    INCOMPLETE_STATIC_ANALYSIS = "incomplete_static_analysis"
    INCOMPLETE_RUNTIME_ANALYSIS = "incomplete_runtime_analysis"
    UNCONTROLLED_EFFECT = "uncontrolled_effect"
    MISSING_EFFECT_ADAPTER = "missing_effect_adapter"
    MISSING_SNAPSHOT_ADAPTER = "missing_snapshot_adapter"
    UNSUPPORTED_PARAMETERS = "unsupported_parameters"
    UNACCOUNTED_DIRTY_STATE = "unaccounted_dirty_state"
    HEURISTIC_SIMILARITY_REJECTED = "heuristic_similarity_rejected"
    POLICY_EXCLUSION = "policy_exclusion"
    MALFORMED_EVIDENCE = "malformed_evidence"
    EMPTY_TRACE_IDENTITY = "empty_trace_identity"


@dataclass(frozen=True)
class TestReuseEligibilityPolicy:
    """Bounded, versioned policy for conservative eligibility classification."""

    __test__: ClassVar[bool] = False

    rollout_scope: str = DEFAULT_ROLLOUT_SCOPE
    allow_pure: bool = True
    allow_snapshot_bound: bool = True
    allow_repository_forest_bound: bool = True
    require_runtime_trace: bool = True
    require_static_trace: bool = True
    # v1: even pure classifications still bind the admitted forest.
    bind_repository_forest: bool = True
    # Similarity inputs are recorded and always deny; never elevate.
    reject_heuristic_similarity: bool = True

    def __post_init__(self) -> None:
        scope = str(self.rollout_scope or "").strip()
        if scope != ROLLOUT_SCOPE_REPOSITORY_FOREST:
            raise TestReuseEligibilityError(
                "rollout v1 admits only repository_forest scope"
            )
        for name in (
            "allow_pure",
            "allow_snapshot_bound",
            "allow_repository_forest_bound",
            "require_runtime_trace",
            "require_static_trace",
            "bind_repository_forest",
            "reject_heuristic_similarity",
        ):
            if type(getattr(self, name)) is not bool:
                raise TestReuseEligibilityError(f"{name} must be a boolean")
        object.__setattr__(self, "rollout_scope", scope)

    @property
    def interface(self) -> str:
        return TEST_REUSE_ELIGIBILITY_POLICY_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TEST_REUSE_ELIGIBILITY_POLICY_SCHEMA,
            "interface": TEST_REUSE_ELIGIBILITY_POLICY_INTERFACE,
            "rollout_scope": self.rollout_scope,
            "allow_pure": self.allow_pure,
            "allow_snapshot_bound": self.allow_snapshot_bound,
            "allow_repository_forest_bound": self.allow_repository_forest_bound,
            "require_runtime_trace": self.require_runtime_trace,
            "require_static_trace": self.require_static_trace,
            "bind_repository_forest": self.bind_repository_forest,
            "reject_heuristic_similarity": self.reject_heuristic_similarity,
        }


@dataclass(frozen=True)
class DirtyStateEvidence:
    """Dirty / generated working-tree facts relevant to eligibility."""

    __test__: ClassVar[bool] = False

    dirty: bool = False
    dirty_overlay_cid: str = ""
    dirty_accounted: bool = True
    unaccounted_paths: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.dirty) is not bool or type(self.dirty_accounted) is not bool:
            raise TestReuseEligibilityError("dirty flags must be booleans")
        overlay = str(self.dirty_overlay_cid or "").strip()
        if overlay and not _is_public_identity(overlay):
            raise TestReuseEligibilityError("dirty_overlay_cid is not a public identity")
        paths = tuple(
            _safe_repo_path(item) for item in (self.unaccounted_paths or ())
        )
        if len(paths) > _MAX_UNACCOUNTED_PATHS:
            raise TestReuseEligibilityError("unaccounted_paths exceeds bound")
        reasons = tuple(_safe_reason(item) for item in (self.reason_codes or ()))
        object.__setattr__(self, "dirty_overlay_cid", overlay)
        object.__setattr__(self, "unaccounted_paths", paths)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def unaccounted(self) -> bool:
        if self.unaccounted_paths:
            return True
        if self.dirty and not self.dirty_accounted:
            return True
        if self.dirty and not self.dirty_overlay_cid:
            return True
        if any(
            code
            in {
                "dirty_overlay_truncated",
                "dirty_path_escape",
                "unaccounted_untracked",
                "unaccounted_generated",
            }
            for code in self.reason_codes
        ):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "dirty_accounted": self.dirty_accounted,
            "unaccounted_paths": list(self.unaccounted_paths),
            "reason_codes": list(self.reason_codes),
            "unaccounted": self.unaccounted,
        }


@dataclass(frozen=True)
class TestReuseEligibilityDecision:
    """Immutable content-addressed eligibility classification."""

    __test__: ClassVar[bool] = False

    content_identity: ContentIdentity
    retained_canonical_bytes: bytes
    eligibility_class: EligibilityClass
    action: ReuseAction
    reason_codes: tuple[str, ...]
    repository_forest_cid: str
    static_trace_root_cid: str
    runtime_trace_root_cid: str
    rollout_scope: str
    reusable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.content_identity, ContentIdentity):
            raise TestReuseEligibilityError("decision requires ContentIdentity")
        if type(self.retained_canonical_bytes) is not bytes:
            raise TestReuseEligibilityError("retained_canonical_bytes must be exact bytes")
        if self.retained_canonical_bytes != self.content_identity.canonical_bytes:
            raise TestReuseEligibilityError("decision bytes do not match ContentIdentity")
        if not isinstance(self.eligibility_class, EligibilityClass):
            raise TestReuseEligibilityError("eligibility_class must be EligibilityClass")
        if not isinstance(self.action, ReuseAction):
            raise TestReuseEligibilityError("action must be ReuseAction")
        if self.action is ReuseAction.SKIP:
            raise TestReuseEligibilityError(
                "eligibility decision never authorizes SKIP"
            )
        if type(self.reusable) is not bool:
            raise TestReuseEligibilityError("reusable must be a boolean")
        if self.reusable and self.eligibility_class is EligibilityClass.NON_REUSABLE:
            raise TestReuseEligibilityError(
                "non_reusable class cannot be marked reusable"
            )
        if (
            self.reusable
            and self.rollout_scope == ROLLOUT_SCOPE_REPOSITORY_FOREST
            and not self.repository_forest_cid
        ):
            raise TestReuseEligibilityError(
                "v1 reusable decisions must bind repository_forest_cid"
            )
        reasons = tuple(self.reason_codes)
        if len(reasons) != len(set(reasons)):
            raise TestReuseEligibilityError("reason_codes must be unique")
        if len(reasons) > _MAX_REASONS:
            raise TestReuseEligibilityError("reason_codes exceed bound")
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def interface(self) -> str:
        return TEST_REUSE_ELIGIBILITY_DECISION_INTERFACE

    @property
    def schema(self) -> str:
        return TEST_REUSE_ELIGIBILITY_DECISION_SCHEMA

    @property
    def cid(self) -> str:
        return self.content_identity.cid

    @property
    def decision_cid(self) -> str:
        return self.cid

    @property
    def canonical_bytes(self) -> bytes:
        return self.retained_canonical_bytes

    @property
    def is_run(self) -> bool:
        return self.action is ReuseAction.RUN

    @property
    def is_skip(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.retained_canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise TestReuseEligibilityError("eligibility decision bytes are not an object")
        return value

    def verify(self) -> TestReuseEligibilityDecision:
        self.content_identity.verify()
        if canonical_json_bytes(self.to_dict()) != self.retained_canonical_bytes:
            raise TestReuseEligibilityError("eligibility decision bytes are not canonical")
        return self

    def as_reuse_reason(self) -> ReuseReasonCode:
        """Map to the plugin/cache :class:`ReuseReasonCode` vocabulary."""

        if self.reusable:
            # Eligibility alone never authorizes a hit; callers still look up.
            return ReuseReasonCode.UNKNOWN
        if EligibilityDenyReason.INCOMPLETE_STATIC_ANALYSIS.value in self.reason_codes:
            return ReuseReasonCode.INCOMPLETE_TRACE
        if EligibilityDenyReason.INCOMPLETE_RUNTIME_ANALYSIS.value in self.reason_codes:
            return ReuseReasonCode.INCOMPLETE_TRACE
        if EligibilityDenyReason.UNSUPPORTED_PARAMETERS.value in self.reason_codes:
            return ReuseReasonCode.UNSUPPORTED
        return ReuseReasonCode.ELIGIBILITY_DENIED


@dataclass(frozen=True)
class _EvaluationContext:
    policy: TestReuseEligibilityPolicy
    repository_forest_cid: str
    static_trace_root_cid: str
    runtime_trace_root_cid: str
    static_complete: bool
    runtime_complete: bool
    static_frontier_kinds: tuple[str, ...]
    static_effect_kinds: tuple[str, ...]
    runtime_reasons: tuple[str, ...]
    runtime_profile: str
    runtime_services: tuple[dict[str, Any], ...]
    runtime_capabilities: tuple[dict[str, Any], ...]
    runtime_subprocesses: tuple[dict[str, Any], ...]
    runtime_environment: tuple[dict[str, Any], ...]
    runtime_policies: tuple[dict[str, Any], ...]
    effect_adapters: tuple[str, ...]
    snapshot_adapters: Mapping[str, str]
    parameters_supported: bool
    parameter_non_reusable_reason: str
    dirty: DirtyStateEvidence
    heuristic_inputs_present: tuple[str, ...]
    policy_exclusion_reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _safe_reason(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text or not _SAFE_REASON_RE.fullmatch(text):
        raise TestReuseEligibilityError("reason code is not bounded public text")
    return text


def _safe_adapter(value: Any) -> str:
    text = str(value or "").strip()
    if not text or not _SAFE_ADAPTER_RE.fullmatch(text):
        raise TestReuseEligibilityError("adapter name is not bounded public text")
    return text


def _safe_repo_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw or raw.startswith("/") or ".." in raw.split("/"):
        raise TestReuseEligibilityError("path must be repository-relative and contained")
    if len(raw) > _MAX_DIAGNOSTIC_CHARS:
        raise TestReuseEligibilityError("path exceeds bounded length")
    return raw


def _is_public_identity(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 256:
        return False
    # Accept canonical CIDv1/base32 and stable public digests used by forest.
    if _CID_RE.fullmatch(text):
        return True
    if re.fullmatch(r"^[A-Za-z0-9_.:@/+-]{8,256}$", text):
        return True
    return False


def _normalize_adapters(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TestReuseEligibilityError("effect_adapters must be a sequence of names")
    if len(values) > _MAX_ADAPTERS:
        raise TestReuseEligibilityError("effect_adapters exceeds bound")
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        name = _safe_adapter(raw)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    # Sorted for content-address stability independent of marker/declaration order.
    return tuple(sorted(ordered))


def _normalize_snapshot_adapters(
    values: Mapping[str, str] | None,
) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TestReuseEligibilityError("snapshot_adapters must be a mapping")
    if len(values) > _MAX_SNAPSHOTS:
        raise TestReuseEligibilityError("snapshot_adapters exceeds bound")
    result: dict[str, str] = {}
    for key, identity in values.items():
        name = _safe_adapter(key)
        text = str(identity or "").strip()
        if not _is_public_identity(text):
            raise TestReuseEligibilityError(
                f"snapshot adapter identity for {name} is not public"
            )
        result[name] = text
    return dict(sorted(result.items()))


def _bounded_diagnostic_value(item: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return str(item)[:_MAX_DIAGNOSTIC_CHARS]
    if item is None or isinstance(item, bool):
        return item
    if isinstance(item, int) and not isinstance(item, bool):
        return item
    if isinstance(item, str):
        return item[:_MAX_DIAGNOSTIC_CHARS]
    if isinstance(item, Mapping):
        nested: dict[str, Any] = {}
        for key, value in list(item.items())[:_MAX_DIAGNOSTIC_KEYS]:
            if type(key) is not str or not key:
                continue
            nested[key[:128]] = _bounded_diagnostic_value(value, depth=depth + 1)
        return dict(sorted(nested.items()))
    if isinstance(item, (list, tuple)):
        return [
            _bounded_diagnostic_value(entry, depth=depth + 1)
            for entry in list(item)[:32]
        ]
    return str(item)[:_MAX_DIAGNOSTIC_CHARS]


def _bounded_diagnostics(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TestReuseEligibilityError("diagnostics must be a mapping")
    if len(value) > _MAX_DIAGNOSTIC_KEYS:
        raise TestReuseEligibilityError("diagnostics exceeds key bound")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str or not key or len(key) > 128:
            raise TestReuseEligibilityError("diagnostic keys must be bounded strings")
        if key in _FORBIDDEN_HEURISTIC_KEYS:
            # Heuristic keys never enter diagnostics as authority material.
            continue
        result[key] = _bounded_diagnostic_value(item)
    return dict(sorted(result.items()))


def _extract_forest_cid(
    repository_forest_cid: str | None,
    repository_forest: Any,
) -> str:
    if repository_forest_cid not in (None, ""):
        text = str(repository_forest_cid).strip()
        if not _is_public_identity(text):
            raise TestReuseEligibilityError("repository_forest_cid is not a public identity")
        return text
    if repository_forest is None:
        return ""
    if isinstance(repository_forest, Mapping):
        for key in ("forest_id", "repository_forest_cid", "cid", "content_id"):
            value = repository_forest.get(key)
            if value not in (None, ""):
                text = str(value).strip()
                if _is_public_identity(text):
                    return text
        return ""
    for attr in ("forest_id", "repository_forest_cid", "cid", "content_id"):
        value = getattr(repository_forest, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value not in (None, ""):
            text = str(value).strip()
            if _is_public_identity(text):
                return text
    return ""


def _trace_cid(trace: Any) -> str:
    if trace is None:
        return ""
    for attr in ("trace_cid", "root_cid", "cid"):
        value = getattr(trace, attr, None)
        if value not in (None, ""):
            return str(value)
    identity = getattr(trace, "content_identity", None)
    if identity is not None:
        cid = getattr(identity, "cid", "")
        if cid:
            return str(cid)
    return ""


def _static_facts(trace: StaticTestDependencyTrace | None) -> tuple[
    bool, str, tuple[str, ...], tuple[str, ...]
]:
    if trace is None:
        return False, "", (), ()
    if not isinstance(trace, StaticTestDependencyTrace):
        raise TestReuseEligibilityError("static_trace must be StaticTestDependencyTrace")
    cid = _trace_cid(trace)
    frontiers = tuple(sorted({item.kind for item in trace.unknown_frontier}))
    effects: list[str] = []
    try:
        payload = trace.to_dict()
        edges = payload.get("dependencies", {}).get("edges", ())
        for edge in edges:
            if isinstance(edge, Mapping) and edge.get("kind") == "effect":
                target = str(edge.get("target_symbol") or edge.get("target") or "").strip()
                if target:
                    effects.append(target)
    except Exception as exc:
        raise TestReuseEligibilityError("static_trace payload is unreadable") from exc
    return bool(trace.complete), cid, frontiers, tuple(sorted(set(effects)))


def _runtime_facts(trace: RuntimeTestDependencyTrace | None) -> tuple[
    bool,
    str,
    tuple[str, ...],
    str,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    if trace is None:
        return False, "", (), "pure", (), (), (), (), ()
    if not isinstance(trace, RuntimeTestDependencyTrace):
        raise TestReuseEligibilityError(
            "runtime_trace must be RuntimeTestDependencyTrace"
        )
    cid = _trace_cid(trace)
    reasons = tuple(sorted(str(item) for item in (trace.completeness_reasons or ())))
    try:
        payload = trace.to_dict() if trace.retained_canonical_bytes else {}
    except Exception as exc:
        raise TestReuseEligibilityError("runtime_trace payload is unreadable") from exc
    if not isinstance(payload, dict):
        raise TestReuseEligibilityError("runtime_trace payload is not an object")
    profile = str(payload.get("eligibility_profile") or "pure")
    deps = payload.get("dependencies") or {}
    if not isinstance(deps, Mapping):
        deps = {}

    def _rows(name: str) -> tuple[dict[str, Any], ...]:
        raw = deps.get(name) or ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        rows: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                rows.append(dict(item))
        return tuple(rows)

    return (
        bool(trace.complete),
        cid,
        reasons,
        profile,
        _rows("services"),
        _rows("capabilities"),
        _rows("subprocesses"),
        _rows("environment"),
        _rows("policies"),
    )


def _detect_heuristics(extra: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not extra:
        return ()
    found = []
    for key in extra:
        name = str(key)
        if name in _FORBIDDEN_HEURISTIC_KEYS or name.endswith("_similarity"):
            found.append(name)
    return tuple(sorted(set(found)))


def _adapter_covers(effect: str, adapters: Sequence[str]) -> bool:
    """Return True when a declared adapter covers the effect kind."""

    if effect in adapters:
        return True
    # Common aliases from pytest markers / plan vocabulary.
    aliases = {
        "filesystem_write": ("filesystem", "fs", "file"),
        "filesystem": ("filesystem_write", "fs", "file"),
        "environment": ("env", "environ"),
        "subprocess": ("process", "tool"),
        "network": ("http", "socket"),
        "hardware": ("gpu", "accelerator", "cuda"),
        "service": ("services", "external"),
        "clock": ("time",),
        "randomness": ("random", "rng"),
    }
    for candidate in aliases.get(effect, ()):
        if candidate in adapters:
            return True
    # Prefix / suffix match for reviewed names like env.TZ or service.postgres.
    for adapter in adapters:
        if adapter == effect or adapter.startswith(effect + ".") or adapter.endswith("." + effect):
            return True
    return False


class TestReuseEligibilityEvaluator:
    """Fail-closed classifier for pure / snapshot / forest / non-reusable items."""

    __test__ = False

    def __init__(
        self,
        *,
        policy: TestReuseEligibilityPolicy | None = None,
        identity_minter: Any = mint_content_identity,
    ) -> None:
        self.policy = policy or TestReuseEligibilityPolicy()
        if not callable(identity_minter):
            raise TestReuseEligibilityError("identity_minter must be callable")
        self._identity_minter = identity_minter

    @property
    def interface(self) -> str:
        return TEST_REUSE_ELIGIBILITY_EVALUATOR_INTERFACE

    def evaluate(
        self,
        *,
        static_trace: StaticTestDependencyTrace | None = None,
        runtime_trace: RuntimeTestDependencyTrace | None = None,
        repository_forest_cid: str | None = None,
        repository_forest: Any = None,
        effect_adapters: Sequence[str] | None = (),
        snapshot_adapters: Mapping[str, str] | None = None,
        parameters_supported: bool = True,
        parameter_non_reusable_reason: str = "",
        dirty_state: DirtyStateEvidence | Mapping[str, Any] | None = None,
        policy_exclusion_reason: str = "",
        diagnostics: Mapping[str, Any] | None = None,
        # Explicitly accepted so callers cannot smuggle authority via kwargs.
        similarity_score: Any = None,
        embedding_score: Any = None,
        model_verdict: Any = None,
        runtime_overlap: Any = None,
        unchanged_line_heuristic: Any = None,
        **rejected_heuristics: Any,
    ) -> TestReuseEligibilityDecision:
        """Classify eligibility; unsafe inputs always return RUN / non_reusable."""

        try:
            context = self._build_context(
                static_trace=static_trace,
                runtime_trace=runtime_trace,
                repository_forest_cid=repository_forest_cid,
                repository_forest=repository_forest,
                effect_adapters=effect_adapters,
                snapshot_adapters=snapshot_adapters,
                parameters_supported=parameters_supported,
                parameter_non_reusable_reason=parameter_non_reusable_reason,
                dirty_state=dirty_state,
                policy_exclusion_reason=policy_exclusion_reason,
                diagnostics=diagnostics,
                similarity_score=similarity_score,
                embedding_score=embedding_score,
                model_verdict=model_verdict,
                runtime_overlap=runtime_overlap,
                unchanged_line_heuristic=unchanged_line_heuristic,
                rejected_heuristics=rejected_heuristics,
            )
            return self._decide(context)
        except TestReuseEligibilityError as exc:
            return self._deny(
                reasons=(EligibilityDenyReason.MALFORMED_EVIDENCE.value,),
                repository_forest_cid=str(repository_forest_cid or ""),
                static_trace_root_cid=_trace_cid(static_trace),
                runtime_trace_root_cid=_trace_cid(runtime_trace),
                diagnostics={"error": type(exc).__name__, "detail": str(exc)[:_MAX_DIAGNOSTIC_CHARS]},
            )

    def _build_context(
        self,
        *,
        static_trace: StaticTestDependencyTrace | None,
        runtime_trace: RuntimeTestDependencyTrace | None,
        repository_forest_cid: str | None,
        repository_forest: Any,
        effect_adapters: Sequence[str] | None,
        snapshot_adapters: Mapping[str, str] | None,
        parameters_supported: bool,
        parameter_non_reusable_reason: str,
        dirty_state: DirtyStateEvidence | Mapping[str, Any] | None,
        policy_exclusion_reason: str,
        diagnostics: Mapping[str, Any] | None,
        similarity_score: Any,
        embedding_score: Any,
        model_verdict: Any,
        runtime_overlap: Any,
        unchanged_line_heuristic: Any,
        rejected_heuristics: Mapping[str, Any],
    ) -> _EvaluationContext:
        if type(parameters_supported) is not bool:
            raise TestReuseEligibilityError("parameters_supported must be a boolean")

        forest_cid = _extract_forest_cid(repository_forest_cid, repository_forest)
        (
            static_complete,
            static_cid,
            frontiers,
            effects,
        ) = _static_facts(static_trace)
        (
            runtime_complete,
            runtime_cid,
            runtime_reasons,
            runtime_profile,
            services,
            capabilities,
            subprocesses,
            environment,
            policies,
        ) = _runtime_facts(runtime_trace)

        if dirty_state is None:
            dirty = DirtyStateEvidence()
        elif isinstance(dirty_state, DirtyStateEvidence):
            dirty = dirty_state
        elif isinstance(dirty_state, Mapping):
            dirty = DirtyStateEvidence(
                dirty=bool(dirty_state.get("dirty", False)),
                dirty_overlay_cid=str(dirty_state.get("dirty_overlay_cid") or ""),
                dirty_accounted=bool(dirty_state.get("dirty_accounted", True)),
                unaccounted_paths=tuple(dirty_state.get("unaccounted_paths") or ()),
                reason_codes=tuple(dirty_state.get("reason_codes") or ()),
            )
        else:
            raise TestReuseEligibilityError("dirty_state must be DirtyStateEvidence or mapping")

        heuristic_probe = {
            "similarity_score": similarity_score,
            "embedding_score": embedding_score,
            "model_verdict": model_verdict,
            "runtime_overlap": runtime_overlap,
            "unchanged_line_heuristic": unchanged_line_heuristic,
            **dict(rejected_heuristics or {}),
            **dict(diagnostics or {}),
        }
        present = []
        for key, value in heuristic_probe.items():
            if value is None:
                continue
            if key in _FORBIDDEN_HEURISTIC_KEYS or str(key).endswith("_similarity"):
                present.append(str(key))
        present_tuple = tuple(sorted(set(present)))

        exclusion = str(policy_exclusion_reason or "").strip()
        if exclusion and not _SAFE_REASON_RE.fullmatch(exclusion):
            raise TestReuseEligibilityError("policy_exclusion_reason is not public text")

        param_reason = str(parameter_non_reusable_reason or "").strip()
        if param_reason and not _SAFE_REASON_RE.fullmatch(param_reason):
            raise TestReuseEligibilityError(
                "parameter_non_reusable_reason is not public text"
            )

        return _EvaluationContext(
            policy=self.policy,
            repository_forest_cid=forest_cid,
            static_trace_root_cid=static_cid,
            runtime_trace_root_cid=runtime_cid,
            static_complete=static_complete,
            runtime_complete=runtime_complete,
            static_frontier_kinds=frontiers,
            static_effect_kinds=effects,
            runtime_reasons=runtime_reasons,
            runtime_profile=runtime_profile,
            runtime_services=services,
            runtime_capabilities=capabilities,
            runtime_subprocesses=subprocesses,
            runtime_environment=environment,
            runtime_policies=policies,
            effect_adapters=_normalize_adapters(effect_adapters),
            snapshot_adapters=_normalize_snapshot_adapters(snapshot_adapters),
            parameters_supported=parameters_supported,
            parameter_non_reusable_reason=param_reason,
            dirty=dirty,
            heuristic_inputs_present=present_tuple,
            policy_exclusion_reason=exclusion,
            diagnostics=_bounded_diagnostics(diagnostics),
        )

    def _decide(self, context: _EvaluationContext) -> TestReuseEligibilityDecision:
        deny_reasons: list[str] = []
        diagnostics = dict(context.diagnostics)

        if context.heuristic_inputs_present and context.policy.reject_heuristic_similarity:
            deny_reasons.append(EligibilityDenyReason.HEURISTIC_SIMILARITY_REJECTED.value)
            diagnostics["rejected_heuristics"] = list(context.heuristic_inputs_present)

        if context.policy_exclusion_reason:
            deny_reasons.append(EligibilityDenyReason.POLICY_EXCLUSION.value)
            diagnostics["policy_exclusion_reason"] = context.policy_exclusion_reason

        if context.policy.require_static_trace and not context.static_trace_root_cid:
            deny_reasons.append(EligibilityDenyReason.MISSING_STATIC_TRACE.value)
        if context.policy.require_runtime_trace and not context.runtime_trace_root_cid:
            deny_reasons.append(EligibilityDenyReason.MISSING_RUNTIME_TRACE.value)

        if not context.parameters_supported or context.parameter_non_reusable_reason:
            deny_reasons.append(EligibilityDenyReason.UNSUPPORTED_PARAMETERS.value)
            if context.parameter_non_reusable_reason:
                diagnostics["parameter_non_reusable_reason"] = (
                    context.parameter_non_reusable_reason
                )

        if context.dirty.unaccounted:
            deny_reasons.append(EligibilityDenyReason.UNACCOUNTED_DIRTY_STATE.value)
            diagnostics["dirty"] = context.dirty.to_dict()

        if context.policy.bind_repository_forest and not context.repository_forest_cid:
            deny_reasons.append(EligibilityDenyReason.MISSING_REPOSITORY_FOREST.value)

        # Adapter / effect closure. Static tracers record effects as
        # ``uncontrolled_effect`` frontiers; a reviewed adapter closes that
        # frontier for eligibility without treating similarity as evidence.
        missing_adapters: list[str] = []
        missing_snapshots: list[str] = []
        uncontrolled_effects: list[str] = []

        if context.static_effect_kinds:
            diagnostics["static_effect_kinds"] = list(context.static_effect_kinds)

        for effect in context.static_effect_kinds:
            if not effect:
                continue
            covered = _adapter_covers(effect, context.effect_adapters)
            if not covered:
                missing_adapters.append(effect)
                uncontrolled_effects.append(effect)

        for service in context.runtime_services:
            name = str(service.get("name") or service.get("service") or "service")
            adapter = str(service.get("adapter_identity") or "")
            snapshot = str(service.get("snapshot_identity") or "")
            if not adapter:
                missing_adapters.append(f"service:{name}")
            if not snapshot and name not in context.snapshot_adapters:
                missing_snapshots.append(f"service:{name}")

        for capability in context.runtime_capabilities:
            name = str(
                capability.get("name") or capability.get("capability") or "capability"
            )
            adapter = str(capability.get("adapter_identity") or "")
            state = str(capability.get("state_identity") or "")
            if not adapter:
                missing_adapters.append(f"capability:{name}")
            if not state and name not in context.snapshot_adapters:
                missing_snapshots.append(f"capability:{name}")

        for subprocess in context.runtime_subprocesses:
            tool = str(subprocess.get("tool_identity") or "")
            executable = str(subprocess.get("executable") or "subprocess")
            if not tool or tool == "unadmitted":
                missing_adapters.append(f"subprocess:{executable}")
                uncontrolled_effects.append("subprocess")
            elif not _adapter_covers("subprocess", context.effect_adapters):
                missing_adapters.append("subprocess")

        if context.runtime_environment and not _adapter_covers(
            "environment", context.effect_adapters
        ):
            if context.runtime_profile == "pure":
                missing_adapters.append("environment")

        needs_snapshot_closure = (
            context.runtime_profile == "snapshot_bound"
            or bool(context.snapshot_adapters)
            or bool(context.runtime_services)
            or bool(context.runtime_capabilities)
            or bool(context.runtime_policies)
        )
        if needs_snapshot_closure:
            for effect in context.static_effect_kinds:
                if effect not in _ADAPTER_REQUIRING_EFFECTS:
                    continue
                if effect in context.snapshot_adapters:
                    continue
                if effect in {"clock", "randomness"} and any(
                    str(row.get("kind")) == effect for row in context.runtime_policies
                ):
                    continue
                if effect in {"service", "hardware"}:
                    continue
                missing_snapshots.append(effect)

        # Static completeness for eligibility: adapter-closed effect frontiers
        # do not count as residual unknown analysis.  All other frontiers do.
        residual_frontiers = [
            kind
            for kind in context.static_frontier_kinds
            if kind != "uncontrolled_effect"
        ]
        effect_frontiers_closed = not uncontrolled_effects and not missing_adapters
        if "uncontrolled_effect" in context.static_frontier_kinds:
            if not effect_frontiers_closed:
                deny_reasons.append(EligibilityDenyReason.UNCONTROLLED_EFFECT.value)
                diagnostics["uncontrolled_effects"] = sorted(set(uncontrolled_effects))
            # residual excludes uncontrolled_effect when closed by adapters
            if effect_frontiers_closed:
                residual_frontiers = [
                    kind
                    for kind in residual_frontiers
                    if kind != "uncontrolled_effect"
                ]

        static_eligibility_complete = (
            bool(context.static_trace_root_cid)
            and not residual_frontiers
            and (
                context.static_complete
                or (
                    "uncontrolled_effect" in context.static_frontier_kinds
                    and effect_frontiers_closed
                    and not residual_frontiers
                )
            )
        )
        # When the only incompleteness was uncontrolled_effect and adapters close
        # every effect, treat static analysis as eligibility-complete.
        if (
            context.static_trace_root_cid
            and not context.static_complete
            and set(context.static_frontier_kinds) <= {"uncontrolled_effect"}
            and effect_frontiers_closed
        ):
            static_eligibility_complete = True

        if context.static_trace_root_cid and not static_eligibility_complete:
            deny_reasons.append(EligibilityDenyReason.INCOMPLETE_STATIC_ANALYSIS.value)
            diagnostics["static_frontier_kinds"] = list(context.static_frontier_kinds)
            blocking = sorted(set(residual_frontiers) & _BLOCKING_STATIC_FRONTIERS)
            if blocking:
                diagnostics["blocking_static_frontiers"] = blocking

        if context.runtime_trace_root_cid and not context.runtime_complete:
            deny_reasons.append(EligibilityDenyReason.INCOMPLETE_RUNTIME_ANALYSIS.value)
            diagnostics["runtime_completeness_reasons"] = list(context.runtime_reasons)

        if missing_adapters:
            deny_reasons.append(EligibilityDenyReason.MISSING_EFFECT_ADAPTER.value)
            diagnostics["missing_effect_adapters"] = sorted(set(missing_adapters))
            if uncontrolled_effects:
                deny_reasons.append(EligibilityDenyReason.UNCONTROLLED_EFFECT.value)
                diagnostics["uncontrolled_effects"] = sorted(set(uncontrolled_effects))
        if missing_snapshots:
            deny_reasons.append(EligibilityDenyReason.MISSING_SNAPSHOT_ADAPTER.value)
            diagnostics["missing_snapshot_adapters"] = sorted(set(missing_snapshots))

        # Deduplicate deny reasons while preserving stable order.
        ordered_denies: list[str] = []
        for reason in deny_reasons:
            if reason not in ordered_denies:
                ordered_denies.append(reason)

        if ordered_denies:
            return self._deny(
                reasons=tuple(ordered_denies),
                repository_forest_cid=context.repository_forest_cid,
                static_trace_root_cid=context.static_trace_root_cid,
                runtime_trace_root_cid=context.runtime_trace_root_cid,
                diagnostics=diagnostics,
                dirty=context.dirty,
                effect_adapters=context.effect_adapters,
                snapshot_adapters=context.snapshot_adapters,
                runtime_profile=context.runtime_profile,
            )

        # Positive classification (still never SKIP).
        eligibility_class = self._classify_positive(context)
        if eligibility_class is EligibilityClass.PURE and not context.policy.allow_pure:
            eligibility_class = EligibilityClass.REPOSITORY_FOREST_BOUND
        if (
            eligibility_class is EligibilityClass.SNAPSHOT_BOUND
            and not context.policy.allow_snapshot_bound
        ):
            eligibility_class = EligibilityClass.REPOSITORY_FOREST_BOUND
        if (
            eligibility_class is EligibilityClass.REPOSITORY_FOREST_BOUND
            and not context.policy.allow_repository_forest_bound
        ):
            return self._deny(
                reasons=(EligibilityDenyReason.POLICY_EXCLUSION.value,),
                repository_forest_cid=context.repository_forest_cid,
                static_trace_root_cid=context.static_trace_root_cid,
                runtime_trace_root_cid=context.runtime_trace_root_cid,
                diagnostics={**diagnostics, "policy": "repository_forest_bound_disabled"},
            )

        return self._admit(
            eligibility_class=eligibility_class,
            context=context,
            diagnostics=diagnostics,
        )

    def _classify_positive(self, context: _EvaluationContext) -> EligibilityClass:
        """Choose pure / snapshot_bound / repository_forest_bound for safe items."""

        has_external = bool(
            context.static_effect_kinds
            or context.runtime_services
            or context.runtime_subprocesses
            or context.runtime_capabilities
            or context.runtime_environment
            or context.runtime_policies
        )
        if context.runtime_profile == "snapshot_bound" or (
            has_external
            and (
                context.snapshot_adapters
                or context.runtime_services
                or context.runtime_capabilities
                or context.runtime_policies
            )
        ):
            # Controlled external state with adapters → snapshot_bound.
            if context.snapshot_adapters or context.runtime_services or context.runtime_policies:
                return EligibilityClass.SNAPSHOT_BOUND

        if not has_external and context.runtime_profile == "pure":
            # Deterministic closure; v1 still binds the forest on the decision.
            return EligibilityClass.PURE

        # Safe default for rollout v1.
        return EligibilityClass.REPOSITORY_FOREST_BOUND

    def _admit(
        self,
        *,
        eligibility_class: EligibilityClass,
        context: _EvaluationContext,
        diagnostics: Mapping[str, Any],
    ) -> TestReuseEligibilityDecision:
        reason_codes = (
            "eligible",
            f"class:{eligibility_class.value}",
            f"scope:{context.policy.rollout_scope}",
        )
        payload = self._payload(
            eligibility_class=eligibility_class,
            action=ReuseAction.RUN,
            reusable=True,
            reason_codes=reason_codes,
            repository_forest_cid=context.repository_forest_cid,
            static_trace_root_cid=context.static_trace_root_cid,
            runtime_trace_root_cid=context.runtime_trace_root_cid,
            rollout_scope=context.policy.rollout_scope,
            diagnostics={
                **dict(diagnostics),
                "effect_adapters": list(context.effect_adapters),
                "snapshot_adapters": dict(context.snapshot_adapters),
                "runtime_profile": context.runtime_profile,
                "dirty": context.dirty.to_dict(),
                "policy": context.policy.to_dict(),
            },
        )
        return self._mint_decision(payload)

    def _deny(
        self,
        *,
        reasons: Sequence[str],
        repository_forest_cid: str = "",
        static_trace_root_cid: str = "",
        runtime_trace_root_cid: str = "",
        diagnostics: Mapping[str, Any] | None = None,
        dirty: DirtyStateEvidence | None = None,
        effect_adapters: Sequence[str] = (),
        snapshot_adapters: Mapping[str, str] | None = None,
        runtime_profile: str = "",
    ) -> TestReuseEligibilityDecision:
        ordered = tuple(dict.fromkeys(str(item) for item in reasons if item))
        if not ordered:
            ordered = (EligibilityDenyReason.MALFORMED_EVIDENCE.value,)
        diag = dict(diagnostics or {})
        if dirty is not None:
            diag.setdefault("dirty", dirty.to_dict())
        if effect_adapters:
            diag.setdefault("effect_adapters", list(effect_adapters))
        if snapshot_adapters:
            diag.setdefault("snapshot_adapters", dict(snapshot_adapters))
        if runtime_profile:
            diag.setdefault("runtime_profile", runtime_profile)
        diag.setdefault("policy", self.policy.to_dict())
        payload = self._payload(
            eligibility_class=EligibilityClass.NON_REUSABLE,
            action=ReuseAction.RUN,
            reusable=False,
            reason_codes=ordered,
            repository_forest_cid=repository_forest_cid,
            static_trace_root_cid=static_trace_root_cid,
            runtime_trace_root_cid=runtime_trace_root_cid,
            rollout_scope=self.policy.rollout_scope,
            diagnostics=diag,
        )
        return self._mint_decision(payload)

    def _payload(
        self,
        *,
        eligibility_class: EligibilityClass,
        action: ReuseAction,
        reusable: bool,
        reason_codes: Sequence[str],
        repository_forest_cid: str,
        static_trace_root_cid: str,
        runtime_trace_root_cid: str,
        rollout_scope: str,
        diagnostics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": TEST_REUSE_ELIGIBILITY_DECISION_SCHEMA,
            "interface": TEST_REUSE_ELIGIBILITY_DECISION_INTERFACE,
            "eligibility_class": eligibility_class.value,
            "action": action.value,
            "reusable": reusable,
            "reason_codes": list(reason_codes),
            "repository_forest_cid": repository_forest_cid,
            "static_trace_root_cid": static_trace_root_cid,
            "runtime_trace_root_cid": runtime_trace_root_cid,
            "rollout_scope": rollout_scope,
            "binds_repository_forest": bool(repository_forest_cid) and reusable,
            "authorizes_skip": False,
            "diagnostics": _bounded_diagnostics(diagnostics),
        }

    def _mint_decision(self, payload: Mapping[str, Any]) -> TestReuseEligibilityDecision:
        expected = canonical_json_bytes(payload)
        identity = self._identity_minter(dict(payload))
        if not isinstance(identity, ContentIdentity):
            raise TestReuseEligibilityError("identity provider did not return ContentIdentity")
        identity.verify()
        if identity.canonical_bytes != expected:
            raise TestReuseEligibilityError(
                "identity provider canonical bytes do not match decision"
            )
        return TestReuseEligibilityDecision(
            content_identity=identity,
            retained_canonical_bytes=expected,
            eligibility_class=EligibilityClass(payload["eligibility_class"]),
            action=ReuseAction(payload["action"]),
            reason_codes=tuple(payload["reason_codes"]),
            repository_forest_cid=str(payload["repository_forest_cid"]),
            static_trace_root_cid=str(payload["static_trace_root_cid"]),
            runtime_trace_root_cid=str(payload["runtime_trace_root_cid"]),
            rollout_scope=str(payload["rollout_scope"]),
            reusable=bool(payload["reusable"]),
        )


def evaluate_reuse_eligibility(
    *,
    static_trace: StaticTestDependencyTrace | None = None,
    runtime_trace: RuntimeTestDependencyTrace | None = None,
    repository_forest_cid: str | None = None,
    repository_forest: Any = None,
    effect_adapters: Sequence[str] | None = (),
    snapshot_adapters: Mapping[str, str] | None = None,
    parameters_supported: bool = True,
    parameter_non_reusable_reason: str = "",
    dirty_state: DirtyStateEvidence | Mapping[str, Any] | None = None,
    policy: TestReuseEligibilityPolicy | None = None,
    policy_exclusion_reason: str = "",
    diagnostics: Mapping[str, Any] | None = None,
    **heuristics: Any,
) -> TestReuseEligibilityDecision:
    """Module-level convenience wrapper around :class:`TestReuseEligibilityEvaluator`."""

    evaluator = TestReuseEligibilityEvaluator(policy=policy)
    return evaluator.evaluate(
        static_trace=static_trace,
        runtime_trace=runtime_trace,
        repository_forest_cid=repository_forest_cid,
        repository_forest=repository_forest,
        effect_adapters=effect_adapters,
        snapshot_adapters=snapshot_adapters,
        parameters_supported=parameters_supported,
        parameter_non_reusable_reason=parameter_non_reusable_reason,
        dirty_state=dirty_state,
        policy_exclusion_reason=policy_exclusion_reason,
        diagnostics=diagnostics,
        **heuristics,
    )


__all__ = [
    "DEFAULT_ROLLOUT_SCOPE",
    "ROLLOUT_SCOPE_REPOSITORY_FOREST",
    "TEST_REUSE_ELIGIBILITY_DECISION_INTERFACE",
    "TEST_REUSE_ELIGIBILITY_DECISION_SCHEMA",
    "TEST_REUSE_ELIGIBILITY_EVALUATOR_INTERFACE",
    "TEST_REUSE_ELIGIBILITY_POLICY_INTERFACE",
    "TEST_REUSE_ELIGIBILITY_POLICY_SCHEMA",
    "DirtyStateEvidence",
    "EligibilityDenyReason",
    "TestReuseEligibilityDecision",
    "TestReuseEligibilityError",
    "TestReuseEligibilityEvaluator",
    "TestReuseEligibilityPolicy",
    "evaluate_reuse_eligibility",
]
