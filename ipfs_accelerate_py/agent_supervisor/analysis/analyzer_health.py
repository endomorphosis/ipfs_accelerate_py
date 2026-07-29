"""Fail-closed health classification for backlog analyzers.

Analyzer health is deliberately independent from whether an analyzer produced
findings.  A broken analyzer can produce an empty result just as easily as a
complete one, so only a healthy, complete scan may be used as exhaustion
evidence.  This module contains no refinery imports; analyzers provide their
inventory and a small fixture callback, which keeps canaries deterministic and
avoids a second implementation of analyzer syntax.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


ANALYZER_HEALTH_SCHEMA = "ipfs_accelerate_py/agent-supervisor/analyzer-health@1"
ANALYZER_CANARY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/analyzer-canaries@1"
ANALYSIS_ESCALATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/analysis-escalation@1"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class AnalyzerHealthStatus(str, Enum):
    """Ordered analyzer health states used by refill receipts."""

    HEALTHY = "healthy"
    PARTIAL = "partial"
    UNHEALTHY = "unhealthy"


class AnalysisEscalationStage(str, Enum):
    """Ordered analysis stages used when the healthy backlog is too small."""

    INCREMENTAL_STATIC = "incremental_static"
    EXHAUSTIVE_AST = "exhaustive_ast"
    LLM_ROUTER = "llm_router"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class AnalysisEscalationStatus(str, Enum):
    """Terminal state of one stage or the complete escalation run."""

    SATISFIED = "satisfied"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    LIMITED = "limited"
    FAILED = "failed"
    ANALYSIS_INCONCLUSIVE = "analysis_inconclusive"


@dataclass(frozen=True)
class AnalysisEscalationPolicy:
    """Finite policy for low-backlog discovery.

    The limits are per daemon cycle. ``router_calls_per_window`` also applies
    to caller-supplied timestamps from earlier cycles, making the otherwise
    stateless policy suitable for a persisted rate-limit ledger.
    """

    backlog_target: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_BACKLOG_TARGET", "5")
        )
    )
    max_incremental_candidates: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_INCREMENTAL", "20")
        )
    )
    max_ast_records: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_AST_RECORDS", "50000")
        )
    )
    max_ast_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_AST_BYTES", "67108864")
        )
    )
    max_router_calls: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_ROUTER_CALLS", "2")
        )
    )
    router_calls_per_window: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_ROUTER_RATE", "4")
        )
    )
    router_window_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_ROUTER_WINDOW_SECONDS", "3600")
        )
    )
    max_router_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_ROUTER_TOKENS", "8192")
        )
    )
    max_router_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_ROUTER_RETRIES", "1")
        )
    )
    max_novel_proposals: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_NOVEL_PROPOSALS", "5")
        )
    )
    max_rejected_candidates: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MAX_REJECTIONS", "100")
        )
    )
    min_confidence: float = field(
        default_factory=lambda: float(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MIN_CONFIDENCE", "0.65")
        )
    )
    min_novelty: float = field(
        default_factory=lambda: float(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYSIS_MIN_NOVELTY", "0.35")
        )
    )

    def __post_init__(self) -> None:
        positive = (
            "backlog_target",
            "max_ast_records",
            "max_ast_bytes",
            "router_window_seconds",
            "max_router_tokens",
        )
        nonnegative = (
            "max_incremental_candidates",
            "max_router_calls",
            "router_calls_per_window",
            "max_router_retries",
            "max_novel_proposals",
            "max_rejected_candidates",
        )
        for name in positive:
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
            object.__setattr__(self, name, value)
        for name in nonnegative:
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("min_confidence", "min_novelty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)

    @classmethod
    def from_value(
        cls, value: "AnalysisEscalationPolicy | Mapping[str, Any] | None"
    ) -> "AnalysisEscalationPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("analysis policy must be AnalysisEscalationPolicy or a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in value if key not in known)
        if unknown:
            raise ValueError(f"unknown analysis policy fields: {', '.join(unknown)}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# A descriptive alias for callers that think of the policy primarily as a
# collection of circuit breakers.
AnalysisEscalationLimits = AnalysisEscalationPolicy


@dataclass(frozen=True)
class AnalysisEscalationRecord:
    """Auditable evidence for exactly one escalation stage."""

    stage: AnalysisEscalationStage | str
    status: AnalysisEscalationStatus | str
    cost: Mapping[str, Any] = field(default_factory=dict)
    scope: Mapping[str, Any] = field(default_factory=dict)
    novelty: float = 0.0
    confidence: float = 0.0
    rejected_candidates: tuple[Mapping[str, Any], ...] = ()
    objective_terms: tuple[str, ...] = ()
    accepted_candidates: tuple[Mapping[str, Any], ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", AnalysisEscalationStage(str(getattr(self.stage, "value", self.stage))))
        object.__setattr__(self, "status", AnalysisEscalationStatus(str(getattr(self.status, "value", self.status))))
        for name in ("novelty", "confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        terms = tuple(dict.fromkeys(str(item).strip() for item in self.objective_terms if str(item).strip()))
        object.__setattr__(self, "objective_terms", terms)
        object.__setattr__(self, "cost", dict(self.cost))
        object.__setattr__(self, "scope", dict(self.scope))
        object.__setattr__(self, "rejected_candidates", tuple(dict(item) for item in self.rejected_candidates))
        object.__setattr__(self, "accepted_candidates", tuple(dict(item) for item in self.accepted_candidates))
        object.__setattr__(self, "reason", str(self.reason or ""))

    @property
    def objective_terms_attempted(self) -> tuple[str, ...]:
        return self.objective_terms

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "cost": dict(self.cost),
            "scope": dict(self.scope),
            "novelty": self.novelty,
            "confidence": self.confidence,
            "rejected_candidates": [dict(item) for item in self.rejected_candidates],
            "objective_terms": list(self.objective_terms),
            "objective_terms_attempted": list(self.objective_terms),
            "accepted_candidates": [dict(item) for item in self.accepted_candidates],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AnalyzerHealthThresholds:
    """Configurable failure budgets for one analyzer scan.

    Non-zero parser failures and incomplete root discovery remain ``partial``
    even when they fit inside their configured budget.  A budget therefore
    controls escalation to ``unhealthy``; it never converts incomplete
    evidence into healthy exhaustion evidence.
    """

    require_canaries: bool = field(
        default_factory=lambda: _env_bool("IPFS_ACCELERATE_AGENT_ANALYZER_REQUIRE_CANARIES", True)
    )
    max_parser_failures: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYZER_MAX_PARSER_FAILURES", "10")
        )
    )
    max_parser_failure_ratio: float = field(
        default_factory=lambda: float(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYZER_MAX_PARSER_FAILURE_RATIO", "1")
        )
    )
    max_excluded_file_ratio: float = field(
        default_factory=lambda: float(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYZER_MAX_EXCLUDED_FILE_RATIO", "0.95")
        )
    )
    min_git_root_discovery_ratio: float = field(
        default_factory=lambda: float(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYZER_MIN_GIT_ROOT_RATIO", "1")
        )
    )
    require_git_root: bool = field(
        default_factory=lambda: _env_bool("IPFS_ACCELERATE_AGENT_ANALYZER_REQUIRE_GIT_ROOT", True)
    )
    min_git_roots: int = field(
        default_factory=lambda: int(
            os.environ.get("IPFS_ACCELERATE_AGENT_ANALYZER_MIN_GIT_ROOTS", "1")
        )
    )
    require_complete_funnel: bool = True

    def __post_init__(self) -> None:
        if self.max_parser_failures < 0:
            raise ValueError("max_parser_failures must be non-negative")
        if self.min_git_roots < 0:
            raise ValueError("min_git_roots must be non-negative")
        for name in (
            "max_parser_failure_ratio",
            "max_excluded_file_ratio",
            "min_git_root_discovery_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)

    @classmethod
    def from_value(
        cls, value: "AnalyzerHealthThresholds | Mapping[str, Any] | None"
    ) -> "AnalyzerHealthThresholds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("health_thresholds must be AnalyzerHealthThresholds or a mapping")
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown analyzer health thresholds: {', '.join(unknown)}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyzerCanaryFixture:
    """One in-memory fixture tied to an analyzer version and parser path."""

    fixture_id: str
    parser_path: str
    relative_path: str
    source: str
    expected_finding_kinds: tuple[str, ...] = ()


# Adding an analyzer version requires adding its complete fixture set here.
# The v1 set is the cross product of every supported finding kind and both
# parser paths (ordinary line scanning and Markdown/RST fenced-block scanning).
_CODEBASE_V1_CANARIES = (
    AnalyzerCanaryFixture(
        "line-source-annotated-followup", "line_source", "annotation.py",
        "# TODO: canary annotation\n", ("annotated_followup",),
    ),
    AnalyzerCanaryFixture(
        "line-source-swallowed-exception", "line_source", "exception.py",
        "try:\n    work()\nexcept Exception:\n    pass\n", ("swallowed_exception",),
    ),
    AnalyzerCanaryFixture(
        "line-source-placeholder", "line_source", "placeholder.py",
        "raise NotImplementedError\n", ("placeholder_runtime_path",),
    ),
    AnalyzerCanaryFixture(
        "markdown-annotated-followup", "markdown_fenced", "annotation.md",
        "TODO: visible canary\n```python\n# TODO: fenced canary must be ignored\n```\n",
        ("annotated_followup",),
    ),
    AnalyzerCanaryFixture(
        "markdown-swallowed-exception", "markdown_fenced", "exception.md",
        "except Exception:\n    pass\n```python\nexcept Exception:\n    pass\n```\n",
        ("swallowed_exception",),
    ),
    AnalyzerCanaryFixture(
        "markdown-placeholder", "markdown_fenced", "placeholder.md",
        "raise NotImplementedError\n```python\nraise NotImplementedError\n```\n",
        ("placeholder_runtime_path",),
    ),
)

ANALYZER_CANARY_FIXTURES: Mapping[str, tuple[AnalyzerCanaryFixture, ...]] = {
    "codebase-annotation-analyzer/v1": _CODEBASE_V1_CANARIES,
}

ANALYZER_SUPPORTED_FINDING_KINDS: Mapping[str, frozenset[str]] = {
    "codebase-annotation-analyzer/v1": frozenset(
        {"annotated_followup", "swallowed_exception", "placeholder_runtime_path"}
    ),
}
ANALYZER_SUPPORTED_PARSER_PATHS: Mapping[str, frozenset[str]] = {
    "codebase-annotation-analyzer/v1": frozenset({"line_source", "markdown_fenced"}),
}


def validate_canary_registry(analyzer_version: str | None = None) -> tuple[str, ...]:
    """Return missing/extra canary coverage errors for registered versions."""

    versions = (
        (str(analyzer_version),)
        if analyzer_version is not None
        else tuple(sorted(set(ANALYZER_SUPPORTED_FINDING_KINDS) | set(ANALYZER_SUPPORTED_PARSER_PATHS)))
    )
    errors: list[str] = []
    for version in versions:
        kinds = ANALYZER_SUPPORTED_FINDING_KINDS.get(version)
        paths = ANALYZER_SUPPORTED_PARSER_PATHS.get(version)
        fixtures = ANALYZER_CANARY_FIXTURES.get(version, ())
        if kinds is None or paths is None or not fixtures:
            errors.append(f"{version}:missing_registry")
            continue
        observed = {
            (fixture.parser_path, kind)
            for fixture in fixtures
            for kind in fixture.expected_finding_kinds
        }
        expected = {(parser_path, kind) for parser_path in paths for kind in kinds}
        for parser_path, kind in sorted(expected - observed):
            errors.append(f"{version}:missing:{parser_path}:{kind}")
        for parser_path, kind in sorted(observed - expected):
            errors.append(f"{version}:unsupported:{parser_path}:{kind}")
    return tuple(errors)


@dataclass(frozen=True)
class AnalyzerCanaryResult:
    fixture_id: str
    parser_path: str
    expected_finding_kinds: tuple[str, ...]
    observed_finding_kinds: tuple[str, ...]
    parser_failure: str = ""

    @property
    def passed(self) -> bool:
        return not self.parser_failure and self.observed_finding_kinds == self.expected_finding_kinds

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "parser_path": self.parser_path,
            "expected_finding_kinds": list(self.expected_finding_kinds),
            "observed_finding_kinds": list(self.observed_finding_kinds),
            "parser_failure": self.parser_failure,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class AnalyzerCanaryReport:
    analyzer_version: str
    results: tuple[AnalyzerCanaryResult, ...]
    registry_present: bool = True
    registry_errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.registry_present
            and not self.registry_errors
            and bool(self.results)
            and all(item.passed for item in self.results)
        )

    @property
    def missing_fixture_ids(self) -> tuple[str, ...]:
        return tuple(item.fixture_id for item in self.results if not item.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYZER_CANARY_SCHEMA,
            "analyzer_version": self.analyzer_version,
            "registry_present": self.registry_present,
            "registry_errors": list(self.registry_errors),
            "passed": self.passed,
            "fixture_count": len(self.results),
            "failed_fixture_ids": list(self.missing_fixture_ids),
            "results": [item.to_dict() for item in self.results],
        }


CanaryAnalyzer = Callable[[str, str], tuple[Sequence[Any], str, str]]


def run_analyzer_canaries(
    analyzer_version: str,
    analyze_fixture: CanaryAnalyzer,
) -> AnalyzerCanaryReport:
    """Run the registered deterministic fixtures through the real analyzer.

    ``analyze_fixture`` returns ``(findings, parser_path, parser_failure)``.
    Findings may be mappings or objects exposing a ``kind`` attribute.
    """

    registry_errors = validate_canary_registry(str(analyzer_version))
    fixtures = ANALYZER_CANARY_FIXTURES.get(str(analyzer_version), ())
    if not fixtures:
        return AnalyzerCanaryReport(
            str(analyzer_version),
            (),
            registry_present=False,
            registry_errors=registry_errors,
        )
    results: list[AnalyzerCanaryResult] = []
    for fixture in fixtures:
        try:
            findings, parser_path, parser_failure = analyze_fixture(
                fixture.source, fixture.relative_path
            )
            kinds = tuple(
                str(item.get("kind", "") if isinstance(item, Mapping) else getattr(item, "kind", ""))
                for item in findings
            )
            failure = str(parser_failure or "")
            if str(parser_path) != fixture.parser_path:
                failure = failure or (
                    f"parser_path_mismatch:expected={fixture.parser_path},observed={parser_path}"
                )
        except Exception as exc:  # canaries must classify analyzer exceptions, not leak them
            kinds = ()
            failure = f"{type(exc).__name__}: {exc}"
        results.append(
            AnalyzerCanaryResult(
                fixture.fixture_id,
                fixture.parser_path,
                fixture.expected_finding_kinds,
                kinds,
                failure,
            )
        )
    return AnalyzerCanaryReport(
        str(analyzer_version),
        tuple(results),
        registry_present=not registry_errors,
        registry_errors=registry_errors,
    )


@dataclass(frozen=True)
class AnalyzerHealthReport:
    status: AnalyzerHealthStatus
    reasons: tuple[str, ...]
    thresholds: AnalyzerHealthThresholds
    metrics: Mapping[str, Any]

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.status is AnalyzerHealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYZER_HEALTH_SCHEMA,
            "status": self.status.value,
            "healthy": self.status is AnalyzerHealthStatus.HEALTHY,
            "safe_for_completion_reasoning": self.safe_for_completion_reasoning,
            "reasons": list(self.reasons),
            "thresholds": self.thresholds.to_dict(),
            "metrics": dict(self.metrics),
        }


_COUNTER_ALIASES: Mapping[str, tuple[str, ...]] = {
    "raw_candidate_count": ("raw_candidate_count", "raw_candidates"),
    "seen_candidate_count": ("seen_candidate_count", "seen_candidates"),
    "deduplicated_candidate_count": (
        "deduplicated_candidate_count", "deduplicated_candidates"
    ),
    "rejected_candidate_count": ("rejected_candidate_count", "rejected_candidates"),
    "appended_task_count": ("appended_task_count", "appended_tasks"),
    "tracked_file_count": ("tracked_file_count", "tracked_files"),
    "eligible_file_count": ("eligible_file_count", "eligible_files"),
    "excluded_file_count": ("excluded_file_count", "excluded_files"),
    "parsed_file_count": ("parsed_file_count", "parsed_files"),
    "cache_hit_count": ("cache_hit_count", "cache_hits"),
    "parser_failure_count": ("parser_failure_count", "parser_failures"),
    "git_root_count": ("git_root_count", "git_roots"),
    "expected_git_root_count": ("expected_git_root_count", "expected_git_roots"),
}

_FUNNEL_KEYS = (
    "raw_candidate_count",
    "seen_candidate_count",
    "deduplicated_candidate_count",
    "rejected_candidate_count",
    "appended_task_count",
)


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value: Any = 0
    for alias in _COUNTER_ALIASES.get(key, (key,)):
        if alias in mapping:
            value = mapping.get(alias)
            break
    if isinstance(value, bool):
        return -1
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return -1


def impossible_candidate_funnel(inventory: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic invariant failures in inventory accounting."""

    failures: list[str] = []
    values = {key: _integer(inventory, key) for key in _FUNNEL_KEYS}
    for key, value in values.items():
        if value < 0:
            failures.append(f"negative_or_invalid_{key}")
    if failures:
        return tuple(failures)
    raw = values["raw_candidate_count"]
    seen = values["seen_candidate_count"]
    deduplicated = values["deduplicated_candidate_count"]
    rejected = values["rejected_candidate_count"]
    appended = values["appended_task_count"]
    if seen + deduplicated + rejected + appended != raw:
        failures.append("raw_candidates_do_not_balance")

    tracked = _integer(inventory, "tracked_file_count")
    eligible = _integer(inventory, "eligible_file_count")
    excluded = _integer(inventory, "excluded_file_count")
    parsed = _integer(inventory, "parsed_file_count")
    cache_hits = _integer(inventory, "cache_hit_count")
    parser_failures = _integer(inventory, "parser_failure_count")
    git_roots = _integer(inventory, "git_root_count")
    expected_git_roots = _integer(inventory, "expected_git_root_count")
    for key, value in (
        ("tracked_file_count", tracked),
        ("eligible_file_count", eligible),
        ("excluded_file_count", excluded),
        ("parsed_file_count", parsed),
        ("cache_hit_count", cache_hits),
        ("parser_failure_count", parser_failures),
        ("git_root_count", git_roots),
        ("expected_git_root_count", expected_git_roots),
    ):
        if value < 0:
            failures.append(f"negative_or_invalid_{key}")
    if tracked >= 0 and eligible >= 0 and excluded >= 0 and eligible + excluded > tracked:
        failures.append("tracked_files_do_not_balance")
    if eligible >= 0 and parsed >= 0 and cache_hits >= 0 and parser_failures >= 0:
        if parsed + cache_hits + parser_failures > eligible:
            failures.append("eligible_files_do_not_balance")
    if raw > 0 and parsed + cache_hits == 0:
        failures.append("candidates_without_parsed_files")
    return tuple(dict.fromkeys(failures))


def classify_analyzer_health(
    inventory: Mapping[str, Any],
    *,
    canaries: AnalyzerCanaryReport | Mapping[str, Any] | None,
    thresholds: AnalyzerHealthThresholds | Mapping[str, Any] | None = None,
) -> AnalyzerHealthReport:
    """Classify one scan from canary, coverage, parser, and funnel evidence."""

    policy = AnalyzerHealthThresholds.from_value(thresholds)
    unhealthy: list[str] = []
    partial: list[str] = []

    if isinstance(canaries, AnalyzerCanaryReport):
        canaries_present = canaries.registry_present and bool(canaries.results)
        canaries_passed = canaries.passed
    elif isinstance(canaries, Mapping):
        try:
            fixture_count = int(canaries.get("fixture_count", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            fixture_count = 0
        canaries_present = (
            bool(canaries.get("registry_present", True))
            and not canaries.get("registry_errors")
            and fixture_count > 0
        )
        canaries_passed = bool(canaries.get("passed", False))
    else:
        canaries_present = False
        canaries_passed = False
    if not canaries_present:
        (unhealthy if policy.require_canaries else partial).append("missing_canaries")
    elif not canaries_passed:
        unhealthy.append("canary_failure")

    expected_roots = max(0, _integer(inventory, "expected_git_root_count"))
    discovered_roots = max(0, _integer(inventory, "git_root_count"))
    if policy.require_git_root and discovered_roots < policy.min_git_roots:
        unhealthy.append("no_git_roots_discovered")
    root_ratio = discovered_roots / expected_roots if expected_roots else (1.0 if discovered_roots else 0.0)
    if expected_roots and discovered_roots < expected_roots:
        if root_ratio < policy.min_git_root_discovery_ratio:
            unhealthy.append("git_root_discovery_below_budget")
        else:
            partial.append("incomplete_git_root_discovery")

    eligible = max(0, _integer(inventory, "eligible_file_count"))
    parser_failures = max(0, _integer(inventory, "parser_failure_count"))
    parser_ratio = parser_failures / eligible if eligible else (1.0 if parser_failures else 0.0)
    if parser_failures:
        if (
            parser_failures > policy.max_parser_failures
            or parser_ratio > policy.max_parser_failure_ratio
        ):
            unhealthy.append("parser_failure_budget_exceeded")
        else:
            partial.append("parser_failures_within_budget")

    tracked = max(0, _integer(inventory, "tracked_file_count"))
    excluded = max(0, _integer(inventory, "excluded_file_count"))
    excluded_ratio = excluded / tracked if tracked else 0.0
    if tracked and excluded_ratio > policy.max_excluded_file_ratio:
        unhealthy.append("excluded_file_budget_exceeded")

    inventoried = max(0, _integer(inventory, "eligible_file_count")) + excluded
    processed = (
        max(0, _integer(inventory, "parsed_file_count"))
        + max(0, _integer(inventory, "cache_hit_count"))
        + parser_failures
    )
    if inventoried < tracked:
        partial.append("unclassified_tracked_files")
    if processed < eligible:
        partial.append("unparsed_eligible_files")
    if not bool(inventory.get("scan_complete", inventory.get("coverage_complete", True))):
        partial.append("scan_incomplete")

    funnel_failures = impossible_candidate_funnel(inventory)
    if funnel_failures and policy.require_complete_funnel:
        unhealthy.extend(f"impossible_funnel:{reason}" for reason in funnel_failures)
    elif funnel_failures:
        partial.extend(f"incomplete_funnel:{reason}" for reason in funnel_failures)

    if unhealthy:
        status = AnalyzerHealthStatus.UNHEALTHY
        reasons = tuple(dict.fromkeys(unhealthy + partial))
    elif partial:
        status = AnalyzerHealthStatus.PARTIAL
        reasons = tuple(dict.fromkeys(partial))
    else:
        status = AnalyzerHealthStatus.HEALTHY
        reasons = ()
    metrics = {
        "canaries_present": canaries_present,
        "canaries_passed": canaries_passed,
        "git_root_discovery_ratio": root_ratio,
        "parser_failure_ratio": parser_ratio,
        "excluded_file_ratio": excluded_ratio,
        "funnel_failure_count": len(funnel_failures),
    }
    return AnalyzerHealthReport(status, reasons, policy, metrics)


__all__ = [
    "ANALYSIS_ESCALATION_SCHEMA",
    "ANALYZER_CANARY_FIXTURES",
    "ANALYZER_CANARY_SCHEMA",
    "ANALYZER_HEALTH_SCHEMA",
    "ANALYZER_SUPPORTED_FINDING_KINDS",
    "ANALYZER_SUPPORTED_PARSER_PATHS",
    "AnalyzerCanaryFixture",
    "AnalyzerCanaryReport",
    "AnalyzerCanaryResult",
    "AnalyzerHealthReport",
    "AnalyzerHealthStatus",
    "AnalyzerHealthThresholds",
    "AnalysisEscalationLimits",
    "AnalysisEscalationPolicy",
    "AnalysisEscalationRecord",
    "AnalysisEscalationStage",
    "AnalysisEscalationStatus",
    "classify_analyzer_health",
    "impossible_candidate_funnel",
    "run_analyzer_canaries",
    "validate_canary_registry",
]
