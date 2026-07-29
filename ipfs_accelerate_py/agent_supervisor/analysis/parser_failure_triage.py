"""Deterministic parser-failure triage for SCA-231 / SCAEV231TRIAGE.

Classifies every typed parse failure into a content-addressed cluster keyed by
parser identity, normalized reason, and path family.  Reviewed exclusion
policies may remove non-contract fixture/archive/generated artifacts from the
parser-failure budget, but they are fail-closed against MCP and runtime
contract surfaces.  Analyzer-side repairs ship with positive and negative
fixtures; thresholds (max 10 failures / 1 percent) are never weakened.

Triage is permanently non-authoritative: it cannot satisfy a repair task
(SCA-232..SCA-237), cannot replace zero-model row receipts, and cannot stand
in for the fresh health / publication authority owned exclusively by SCA-512.
Exclusions here are budget projections only; they do not resolve retained
failures or authorize analyzer-health publication.

Source bodies never appear on triage receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .analyzer_health import AnalyzerHealthStatus
from .polyglot_ast_health import (
    DEFAULT_LANGUAGE_THRESHOLDS,
    FailureCluster,
    LanguageHealthThresholds,
    PathDispositionRecord,
    PathParseOutcome,
    PolyglotASTHealthReport,
    assess_polyglot_ast_health,
    classify_path_disposition,
    cluster_failures,
    load_coverage_rows,
    report_contains_source_body,
    typed_reason_code,
)


PARSER_FAILURE_TRIAGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/parser-failure-triage@1"
)
PARSER_FAILURE_TRIAGE_INTERFACE = "ParserFailureTriage@1"
PARSER_FAILURE_TRIAGE_EVIDENCE = "SCAEV231TRIAGE"

# Reviewed whole-tree budget — must stay at least as strict as the published gate.
REVIEWED_MAX_PARSER_FAILURES = 10
REVIEWED_MAX_PARSER_FAILURE_RATIO = 0.01

_DEFAULT_MAX_CLUSTER_SAMPLES = 8
_DEFAULT_MAX_MEMBER_PATHS = 64

# Path fragments that identify MCP / runtime contract surfaces.  Reviewed
# exclusions may never hide these.
_PROTECTED_SURFACE_MARKERS: tuple[str, ...] = (
    "/mcp/",
    "/mcp++/",
    "/mcpplusplus/",
    "mcpclient",
    "mockmcp",
    "mcp_server",
    "mcp-server",
    "tools_dispatch",
    "/runtime/",
    "runtime_mcp",
    "runtime-mcp",
)

_PROTECTED_BASENAME_MARKERS: tuple[str, ...] = (
    "mockmcpclient",
    "mcpclient",
    "mcpserver",
)

# Auto-converted Python→TypeScript garbage markers (classification only).
_AUTO_CONVERTED_MARKERS: tuple[str, ...] = (
    "automatically converted from python",
    "conversion fidelity might not be 100%",
    "converted import { {",
    "from \"python:",
)

_TS_CODE_RE = re.compile(r"\bTS\d{3,5}\b")
_SHEBANG_RE = re.compile(
    r"^#!\s*(?:/usr/bin/env\s+)?(?:/bin/)?(?:bash|sh|zsh|python[0-9.]*)\b"
    r"|^#!\s*/(?:usr/)?bin/(?:env\s+)?(?:bash|sh|zsh|python[0-9.]*)\b"
)


class ParserFailureTriageError(ValueError):
    """Invalid triage input, policy violation, or report serialization."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "parser_failure_triage_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class ClusterDispositionKind(str, Enum):
    """Closed vocabulary for how a failure cluster is handled."""

    GENUINE_SOURCE_DEFECT = "genuine_source_defect"
    INTENTIONALLY_INVALID_FIXTURE = "intentionally_invalid_fixture"
    GENERATED_ARTIFACT = "generated_artifact"
    VENDOR_OR_ARCHIVE_ARTIFACT = "vendor_or_archive_artifact"
    UNSUPPORTED_OR_MISCLASSIFIED = "unsupported_or_misclassified"
    OVERSIZED_ARTIFACT = "oversized_artifact"
    PARSER_DEFECT = "parser_defect"
    REVIEWED_EXCLUSION = "reviewed_exclusion"


class TriageAction(str, Enum):
    """Budget impact of a disposition."""

    COUNT_AS_FAILURE = "count_as_failure"
    EXCLUDE_FROM_BUDGET = "exclude_from_budget"
    RECLASSIFY_NOT_ELIGIBLE = "reclassify_not_eligible"
    APPLY_PARSER_REPAIR = "apply_parser_repair"


@dataclass(frozen=True)
class ReviewedExclusionRule:
    """One explicit, reviewed exclusion rule.

    Rules never match protected MCP/runtime surfaces even when path/reason
    predicates would otherwise match.
    """

    rule_id: str
    description: str
    path_prefixes: tuple[str, ...] = ()
    path_contains: tuple[str, ...] = ()
    path_suffixes: tuple[str, ...] = ()
    basename_contains: tuple[str, ...] = ()
    reason_prefixes: tuple[str, ...] = ()
    reason_contains: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    disposition: ClusterDispositionKind = (
        ClusterDispositionKind.REVIEWED_EXCLUSION
    )
    action: TriageAction = TriageAction.EXCLUDE_FROM_BUDGET
    reviewed: bool = True

    def __post_init__(self) -> None:
        if not self.rule_id or not str(self.rule_id).startswith("policy:"):
            raise ParserFailureTriageError(
                "reviewed exclusion rule_id must start with 'policy:'",
                reason_code="invalid_policy_rule_id",
            )
        if not self.reviewed:
            raise ParserFailureTriageError(
                "exclusion rules must be marked reviewed=True",
                reason_code="unreviewed_policy",
            )
        if not self.description.strip():
            raise ParserFailureTriageError(
                "exclusion rules require a description",
                reason_code="invalid_policy_description",
            )

    def matches(
        self,
        *,
        path: str,
        language: str,
        reason_code: str,
        raw_reason: str,
    ) -> bool:
        path_n = _normalize_path(path)
        if is_protected_contract_surface(path_n):
            return False

        def _prefix_hit(prefix: str) -> bool:
            p = prefix.rstrip("/")
            return path_n == p or path_n.startswith(p + "/")

        if self.path_prefixes and not any(
            _prefix_hit(prefix) for prefix in self.path_prefixes
        ):
            return False
        if self.path_contains and not any(
            fragment in path_n for fragment in self.path_contains
        ):
            return False
        if self.path_suffixes and not any(
            path_n.endswith(suffix) for suffix in self.path_suffixes
        ):
            return False
        if self.basename_contains:
            base = Path(path_n).name.casefold()
            if not any(token in base for token in self.basename_contains):
                return False
        if self.languages:
            lang = _normalize_language(language)
            allowed = {_normalize_language(item) for item in self.languages}
            if lang not in allowed:
                return False
        reason_blob = f"{reason_code}\n{raw_reason}".casefold()
        if self.reason_prefixes:
            raw_cf = (raw_reason or reason_code or "").casefold()
            code_cf = (reason_code or "").casefold()
            if not any(
                raw_cf.startswith(prefix.casefold())
                or code_cf.startswith(prefix.casefold())
                for prefix in self.reason_prefixes
            ):
                return False
        if self.reason_contains and not any(
            token.casefold() in reason_blob for token in self.reason_contains
        ):
            return False
        # At least one positive predicate beyond the universal protected check.
        has_predicate = bool(
            self.path_prefixes
            or self.path_contains
            or self.path_suffixes
            or self.basename_contains
            or self.reason_prefixes
            or self.reason_contains
            or self.languages
        )
        return has_predicate

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "path_prefixes": list(self.path_prefixes),
            "path_contains": list(self.path_contains),
            "path_suffixes": list(self.path_suffixes),
            "basename_contains": list(self.basename_contains),
            "reason_prefixes": list(self.reason_prefixes),
            "reason_contains": list(self.reason_contains),
            "languages": list(self.languages),
            "disposition": self.disposition.value,
            "action": self.action.value,
            "reviewed": self.reviewed,
        }


# Explicit reviewed policy catalogue.  New exclusions require a new rule_id.
DEFAULT_REVIEWED_EXCLUSION_POLICY: tuple[ReviewedExclusionRule, ...] = (
    ReviewedExclusionRule(
        rule_id="policy:auto-converted-ipfs-accelerate-js-test-fixtures",
        description=(
            "Broken Python→TypeScript auto-conversions under "
            "ipfs_accelerate_js/test are intentionally invalid fixtures, not "
            "MCP/runtime contract sources."
        ),
        path_prefixes=(
            "ipfs_accelerate_js/test/unit/",
            "ipfs_accelerate_js/test/browser/",
            "ipfs_accelerate_js/test/performance/",
        ),
        disposition=ClusterDispositionKind.INTENTIONALLY_INVALID_FIXTURE,
        action=TriageAction.EXCLUDE_FROM_BUDGET,
    ),
    ReviewedExclusionRule(
        rule_id="policy:web-legacy-archive-broken-artifacts",
        description=(
            "web/legacy-archive retains intentionally broken and historical "
            "front-end sources outside the active contract surface."
        ),
        path_prefixes=("web/legacy-archive/",),
        disposition=ClusterDispositionKind.VENDOR_OR_ARCHIVE_ARTIFACT,
        action=TriageAction.EXCLUDE_FROM_BUDGET,
    ),
    ReviewedExclusionRule(
        rule_id="policy:generated-ast-export-artifacts",
        description=(
            "docs/ast_exports holds generated AST dumps; oversize or corrupt "
            "entries are generated artifacts, not parser defects."
        ),
        path_prefixes=("docs/ast_exports/",),
        disposition=ClusterDispositionKind.GENERATED_ARTIFACT,
        action=TriageAction.RECLASSIFY_NOT_ELIGIBLE,
    ),
    ReviewedExclusionRule(
        rule_id="policy:benchmark-result-samples",
        description=(
            "benchmark-results sample files are diagnostic dumps, not "
            "contract-bearing sources."
        ),
        path_prefixes=("benchmark-results/",),
        disposition=ClusterDispositionKind.GENERATED_ARTIFACT,
        action=TriageAction.RECLASSIFY_NOT_ELIGIBLE,
    ),
    ReviewedExclusionRule(
        rule_id="policy:web-platform-test-output-generated",
        description=(
            "test/web_platform_test_output holds generated run output, not "
            "authoritative sources."
        ),
        path_prefixes=("test/web_platform_test_output/",),
        disposition=ClusterDispositionKind.GENERATED_ARTIFACT,
        action=TriageAction.EXCLUDE_FROM_BUDGET,
    ),
    ReviewedExclusionRule(
        rule_id="policy:non-mcp-test-mock-stubs",
        description=(
            "test/mocks stubs (excluding MCP surfaces) are intentionally "
            "incomplete compatibility shims."
        ),
        path_prefixes=("test/mocks/",),
        disposition=ClusterDispositionKind.INTENTIONALLY_INVALID_FIXTURE,
        action=TriageAction.EXCLUDE_FROM_BUDGET,
    ),
    ReviewedExclusionRule(
        rule_id="policy:named-broken-fixtures",
        description=(
            "Paths whose basename marks them broken are intentionally invalid "
            "fixtures retained for regression."
        ),
        basename_contains=("broken", "-old.", ".old."),
        disposition=ClusterDispositionKind.INTENTIONALLY_INVALID_FIXTURE,
        action=TriageAction.EXCLUDE_FROM_BUDGET,
    ),
    ReviewedExclusionRule(
        rule_id="policy:fixed-web-platform-broken-fixtures",
        description=(
            "test/fixed_web_platform retains historical broken Python fixtures."
        ),
        path_prefixes=("test/fixed_web_platform/",),
        disposition=ClusterDispositionKind.INTENTIONALLY_INVALID_FIXTURE,
        action=TriageAction.EXCLUDE_FROM_BUDGET,
    ),
    ReviewedExclusionRule(
        rule_id="policy:oversized-source-byte-bound",
        description=(
            "Sources exceeding the reviewed UTF-8 byte bound are typed as "
            "oversized artifacts rather than free-form parse failures."
        ),
        reason_contains=("file_bytes_exceeded",),
        disposition=ClusterDispositionKind.OVERSIZED_ARTIFACT,
        action=TriageAction.RECLASSIFY_NOT_ELIGIBLE,
    ),
    ReviewedExclusionRule(
        rule_id="policy:misclassified-shebang-extension",
        description=(
            "Files whose contents begin with a non-JS/TS shebang but carry a "
            "parser-eligible extension are misclassified extension artifacts."
        ),
        reason_contains=("misclassified_extension", "invalid character"),
        path_contains=("/utils/run_web_platform_integration_tests.js",),
        disposition=ClusterDispositionKind.UNSUPPORTED_OR_MISCLASSIFIED,
        action=TriageAction.RECLASSIFY_NOT_ELIGIBLE,
    ),
)


@dataclass(frozen=True)
class ParserRepairFixture:
    """One positive or negative fixture for an analyzer-side repair."""

    fixture_id: str
    language: str
    source: str
    expect_success: bool
    repair_id: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        # Source is retained only on in-memory fixtures for execution; the
        # serialized form stores a digest so reports stay body-free.
        digest = "sha256:" + hashlib.sha256(
            self.source.encode("utf-8")
        ).hexdigest()
        return {
            "fixture_id": self.fixture_id,
            "language": self.language,
            "expect_success": self.expect_success,
            "repair_id": self.repair_id,
            "source_sha256": digest,
            "source_byte_length": len(self.source.encode("utf-8")),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ParserRepair:
    """Minimal analyzer repair with positive and negative fixtures."""

    repair_id: str
    description: str
    kind: str
    positive_fixtures: tuple[ParserRepairFixture, ...]
    negative_fixtures: tuple[ParserRepairFixture, ...]

    def __post_init__(self) -> None:
        if not self.positive_fixtures or not self.negative_fixtures:
            raise ParserFailureTriageError(
                f"repair {self.repair_id} requires positive and negative fixtures",
                reason_code="repair_fixtures_required",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_id": self.repair_id,
            "description": self.description,
            "kind": self.kind,
            "positive_fixtures": [item.to_dict() for item in self.positive_fixtures],
            "negative_fixtures": [item.to_dict() for item in self.negative_fixtures],
        }


def default_parser_repairs() -> tuple[ParserRepair, ...]:
    """Reviewed analyzer repairs with positive/negative fixtures."""

    reason_norm = ParserRepair(
        repair_id="repair:normalize-typescript-diagnostic-codes",
        description=(
            "Collapse volatile TypeScript diagnostic text into a stable "
            "ordered set of TSxxxx codes so cluster identity is content-addressed "
            "and independent of column drift."
        ),
        kind="reason_normalization",
        positive_fixtures=(
            ParserRepairFixture(
                fixture_id="ts-reason-norm-positive",
                language="typescript",
                source='export function run(input: string): string { return input; }\n',
                expect_success=True,
                repair_id="repair:normalize-typescript-diagnostic-codes",
                notes="Valid TS source must not enter a failure cluster.",
            ),
        ),
        negative_fixtures=(
            ParserRepairFixture(
                fixture_id="ts-reason-norm-negative",
                language="typescript",
                source='const x = "unterminated\n',
                expect_success=False,
                repair_id="repair:normalize-typescript-diagnostic-codes",
                notes="Unterminated string remains a typed parse failure.",
            ),
        ),
    )
    shebang = ParserRepair(
        repair_id="repair:detect-shebang-extension-mismatch",
        description=(
            "Detect bash/python shebang bodies under .js/.ts extensions and "
            "reclassify them as unsupported/misclassified rather than silent "
            "success or untyped failure."
        ),
        kind="extension_mismatch",
        positive_fixtures=(
            ParserRepairFixture(
                fixture_id="shebang-js-positive",
                language="javascript",
                source="export function run(x) { return x; }\n",
                expect_success=True,
                repair_id="repair:detect-shebang-extension-mismatch",
                notes="Real JS modules remain parser-eligible successes.",
            ),
        ),
        negative_fixtures=(
            ParserRepairFixture(
                fixture_id="shebang-js-negative",
                language="javascript",
                source="#!/bin/bash\nset -e\necho hello\n",
                expect_success=False,
                repair_id="repair:detect-shebang-extension-mismatch",
                notes="Bash shebang under .js must not be labeled success.",
            ),
        ),
    )
    protected = ParserRepair(
        repair_id="repair:protected-mcp-surface-never-excluded",
        description=(
            "MCP/runtime contract surfaces remain in the failure budget even "
            "when surrounding test trees are excluded by reviewed policy."
        ),
        kind="policy_guard",
        positive_fixtures=(
            ParserRepairFixture(
                fixture_id="mcp-guard-positive-fixture-tree",
                language="typescript",
                source="// intentionally invalid fixture under test/unit\nconst x = \"\n",
                expect_success=False,
                repair_id="repair:protected-mcp-surface-never-excluded",
                notes="Non-MCP test fixtures may be excluded by policy.",
            ),
        ),
        negative_fixtures=(
            ParserRepairFixture(
                fixture_id="mcp-guard-negative-mockmcp",
                language="javascript",
                source="export const mockMCPClient = {\n  call(tool) { return tool\n}\n",
                expect_success=False,
                repair_id="repair:protected-mcp-surface-never-excluded",
                notes="mockMCPClient syntax errors must remain budgeted failures.",
            ),
        ),
    )
    return (reason_norm, shebang, protected)


@dataclass(frozen=True)
class FailureMember:
    """One parse-failure row bound into a triage cluster (body-free)."""

    path: str
    language: str
    parser_identity: str
    reason_code: str
    raw_reason: str
    content_digest: str = ""
    row_id: str = ""
    path_family: str = ""
    actionable_family: str = ""
    protected_surface: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(self, "language", _normalize_language(self.language))
        object.__setattr__(
            self, "parser_identity", str(self.parser_identity or "").strip()
        )
        object.__setattr__(
            self,
            "reason_code",
            str(self.reason_code or "").strip() or "parse_failure",
        )
        object.__setattr__(self, "raw_reason", str(self.raw_reason or "")[:512])
        object.__setattr__(
            self, "content_digest", str(self.content_digest or "").strip()
        )
        object.__setattr__(self, "row_id", str(self.row_id or "").strip())
        family = self.path_family or path_family_for(self.path)
        object.__setattr__(self, "path_family", family)
        # Actionable family is best-effort outside the pinned 258 diagnostic set.
        if self.actionable_family:
            object.__setattr__(
                self, "actionable_family", str(self.actionable_family).strip()
            )
        else:
            try:
                object.__setattr__(
                    self, "actionable_family", actionable_repair_family(self.path)
                )
            except ParserFailureTriageError:
                object.__setattr__(self, "actionable_family", "")
        object.__setattr__(
            self,
            "protected_surface",
            bool(self.protected_surface)
            or is_protected_contract_surface(self.path),
        )

    @property
    def member_id(self) -> str:
        return _identity(
            "failure-member",
            {
                "path": self.path,
                "language": self.language,
                "parser_identity": self.parser_identity,
                "reason_code": self.reason_code,
                "content_digest": self.content_digest,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "path": self.path,
            "language": self.language,
            "parser_identity": self.parser_identity,
            "reason_code": self.reason_code,
            "raw_reason": self.raw_reason,
            "content_digest": self.content_digest,
            "row_id": self.row_id,
            "path_family": self.path_family,
            "actionable_family": self.actionable_family,
            "protected_surface": self.protected_surface,
        }


@dataclass(frozen=True)
class TriageCluster:
    """Content-addressed cluster of equivalent parse failures."""

    parser_identity: str
    reason_code: str
    path_family: str
    language: str
    count: int
    disposition: ClusterDispositionKind
    action: TriageAction
    policy_rule_id: str = ""
    sample_paths: tuple[str, ...] = ()
    member_ids: tuple[str, ...] = ()
    protected_member_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parser_identity", str(self.parser_identity or "").strip()
        )
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "parse_failure").strip()
        )
        object.__setattr__(self, "path_family", str(self.path_family or "").strip())
        object.__setattr__(self, "language", _normalize_language(self.language))
        object.__setattr__(self, "count", int(self.count))
        if self.count < 1:
            raise ParserFailureTriageError(
                "cluster count must be at least 1",
                reason_code="invalid_cluster",
            )
        object.__setattr__(
            self,
            "disposition",
            ClusterDispositionKind(
                str(getattr(self.disposition, "value", self.disposition))
            ),
        )
        object.__setattr__(
            self,
            "action",
            TriageAction(str(getattr(self.action, "value", self.action))),
        )
        object.__setattr__(
            self, "policy_rule_id", str(self.policy_rule_id or "").strip()
        )
        object.__setattr__(
            self,
            "sample_paths",
            tuple(str(item) for item in self.sample_paths if str(item)),
        )
        object.__setattr__(
            self,
            "member_ids",
            tuple(str(item) for item in self.member_ids if str(item)),
        )
        object.__setattr__(
            self, "protected_member_count", int(self.protected_member_count)
        )
        if (
            self.action
            in {
                TriageAction.EXCLUDE_FROM_BUDGET,
                TriageAction.RECLASSIFY_NOT_ELIGIBLE,
            }
            and self.protected_member_count > 0
        ):
            raise ParserFailureTriageError(
                "exclusion clusters cannot include protected MCP/runtime surfaces",
                reason_code="protected_surface_exclusion",
                details={"path_family": self.path_family},
            )

    @property
    def cluster_id(self) -> str:
        return _identity(
            "parser-failure-cluster",
            {
                "parser_identity": self.parser_identity,
                "reason_code": self.reason_code,
                "path_family": self.path_family,
                "language": self.language,
            },
        )

    @property
    def counts_toward_budget(self) -> bool:
        return self.action is TriageAction.COUNT_AS_FAILURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "parser_identity": self.parser_identity,
            "reason_code": self.reason_code,
            "path_family": self.path_family,
            "language": self.language,
            "count": self.count,
            "disposition": self.disposition.value,
            "action": self.action.value,
            "counts_toward_budget": self.counts_toward_budget,
            "policy_rule_id": self.policy_rule_id,
            "sample_paths": list(self.sample_paths),
            "member_ids": list(self.member_ids),
            "protected_member_count": self.protected_member_count,
        }


@dataclass(frozen=True)
class MemberAssignment:
    """Per-path triage assignment."""

    member: FailureMember
    cluster_id: str
    disposition: ClusterDispositionKind
    action: TriageAction
    policy_rule_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.member.path,
            "member_id": self.member.member_id,
            "cluster_id": self.cluster_id,
            "disposition": self.disposition.value,
            "action": self.action.value,
            "policy_rule_id": self.policy_rule_id,
            "protected_surface": self.member.protected_surface,
            "path_family": self.member.path_family,
            "actionable_family": self.member.actionable_family,
            "language": self.member.language,
            "reason_code": self.member.reason_code,
            "parser_identity": self.member.parser_identity,
        }


@dataclass(frozen=True)
class HealthGateProjection:
    """Projected parser-failure budget after triage (thresholds unchanged)."""

    eligible_path_count: int
    residual_failure_count: int
    excluded_failure_count: int
    reclassified_count: int
    residual_failure_ratio: float
    max_parser_failures: int
    max_parser_failure_ratio: float
    meets_gate: bool
    status: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_path_count": self.eligible_path_count,
            "residual_failure_count": self.residual_failure_count,
            "excluded_failure_count": self.excluded_failure_count,
            "reclassified_count": self.reclassified_count,
            "residual_failure_ratio": self.residual_failure_ratio,
            "max_parser_failures": self.max_parser_failures,
            "max_parser_failure_ratio": self.max_parser_failure_ratio,
            "meets_gate": self.meets_gate,
            "status": self.status,
            "reasons": list(self.reasons),
            "thresholds_unchanged": True,
            # Projection only — never publication or SCA-512 authority.
            "non_authoritative": True,
            "authoritative_health_owner": "SCA-512",
            "satisfies_fresh_health_authority": False,
            "satisfies_repair_task": False,
        }


@dataclass(frozen=True)
class ParserFailureTriageReport:
    """Content-addressed triage receipt for one diagnostic index."""

    source_index_id: str
    failure_count: int
    cluster_count: int
    clusters: tuple[TriageCluster, ...]
    assignments: tuple[MemberAssignment, ...]
    policy: tuple[ReviewedExclusionRule, ...]
    repairs: tuple[ParserRepair, ...]
    health_gate: HealthGateProjection
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str = PARSER_FAILURE_TRIAGE_EVIDENCE
    schema: str = PARSER_FAILURE_TRIAGE_SCHEMA
    unassigned_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(self, "policy", tuple(self.policy))
        object.__setattr__(self, "repairs", tuple(self.repairs))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "failure_count", int(self.failure_count))
        object.__setattr__(self, "cluster_count", int(self.cluster_count))
        object.__setattr__(self, "unassigned_count", int(self.unassigned_count))
        if self.failure_count != len(self.assignments):
            raise ParserFailureTriageError(
                "every failure must receive exactly one assignment",
                reason_code="incomplete_assignment",
                details={
                    "failure_count": self.failure_count,
                    "assignment_count": len(self.assignments),
                },
            )
        if self.unassigned_count != 0:
            raise ParserFailureTriageError(
                "unassigned failures are not permitted",
                reason_code="unassigned_failures",
            )
        if self.cluster_count != len(self.clusters):
            raise ParserFailureTriageError(
                "cluster_count must match clusters",
                reason_code="cluster_count_mismatch",
            )
        # Every assignment must map to a known cluster.
        cluster_ids = {item.cluster_id for item in self.clusters}
        for assignment in self.assignments:
            if assignment.cluster_id not in cluster_ids:
                raise ParserFailureTriageError(
                    f"assignment cluster missing: {assignment.cluster_id}",
                    reason_code="dangling_cluster_ref",
                )
        # Protected surfaces must never be excluded.
        for assignment in self.assignments:
            if assignment.member.protected_surface and assignment.action in {
                TriageAction.EXCLUDE_FROM_BUDGET,
                TriageAction.RECLASSIFY_NOT_ELIGIBLE,
            }:
                raise ParserFailureTriageError(
                    "protected MCP/runtime surface cannot be excluded",
                    reason_code="protected_surface_exclusion",
                    details={"path": assignment.member.path},
                )
        payload = self.to_dict(include_identity=False)
        if report_contains_source_body(payload):
            raise ParserFailureTriageError(
                "triage report embeds a source body",
                reason_code="source_body_forbidden",
            )

    @property
    def complete(self) -> bool:
        return (
            self.failure_count == len(self.assignments)
            and self.unassigned_count == 0
            and all(item.cluster_id for item in self.assignments)
        )

    def content_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict(include_identity=False))

    def content_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.content_bytes()).hexdigest()

    def content_identity(self) -> dict[str, Any]:
        digest = self.content_digest()
        identity: dict[str, Any] = {
            "profile": "strict-dag-json-v1",
            "digest": digest,
            "byte_length": len(self.content_bytes()),
            "validated": False,
            "cid": "",
        }
        try:
            from .content_identity_bridge import identify_strict_artifact

            bound = identify_strict_artifact(self.to_dict(include_identity=False))
            if hasattr(bound, "to_dict"):
                payload = bound.to_dict()
            elif isinstance(bound, Mapping):
                payload = dict(bound)
            else:
                payload = {}
            if payload:
                identity.update(
                    {
                        "digest": str(payload.get("digest") or digest),
                        "cid": str(payload.get("cid") or ""),
                        "byte_length": int(
                            payload.get("byte_length") or identity["byte_length"]
                        ),
                        "validated": bool(payload.get("validated", False)),
                    }
                )
        except Exception as exc:  # identity is best-effort
            identity["identity_error"] = f"{type(exc).__name__}: {exc}"
        return identity

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "interface": PARSER_FAILURE_TRIAGE_INTERFACE,
            "evidence_id": self.evidence_id,
            "source_index_id": self.source_index_id,
            "failure_count": self.failure_count,
            "cluster_count": self.cluster_count,
            "unassigned_count": self.unassigned_count,
            "complete": self.complete,
            # SCA-231 is classification-only; SCA-512 owns fresh health authority.
            "non_authoritative": True,
            "completion_authoritative": False,
            "satisfies_repair_task": False,
            "satisfies_fresh_health_authority": False,
            "authoritative_health_owner": "SCA-512",
            "metrics": dict(self.metrics),
            "health_gate": self.health_gate.to_dict(),
            "policy": [item.to_dict() for item in self.policy],
            "repairs": [item.to_dict() for item in self.repairs],
            "clusters": [item.to_dict() for item in self.clusters],
            # Compact path→cluster ledger for audit; full member detail is
            # recoverable from cluster samples + source index.
            "assignments": [item.to_dict() for item in self.assignments],
        }
        if include_identity:
            payload["content_identity"] = self.content_identity()
        return payload


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity(prefix: str, value: Any) -> str:
    return (
        f"{prefix}:sha256:"
        + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    )


def _normalize_path(path: str) -> str:
    text = str(path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _normalize_language(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    if not raw:
        return ""
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    aliases = {
        "cjs": "javascript",
        "mjs": "javascript",
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
    }
    return aliases.get(raw, raw)


def is_protected_contract_surface(path: str) -> bool:
    """True when a path is an MCP/runtime contract surface that cannot be excluded."""

    normalized = "/" + _normalize_path(path).casefold()
    if not normalized.endswith("/"):
        # Ensure marker search works for basename-only matches.
        pass
    for marker in _PROTECTED_SURFACE_MARKERS:
        if marker in normalized:
            return True
    base = Path(_normalize_path(path)).name.casefold()
    stem = Path(base).stem.casefold()
    for marker in _PROTECTED_BASENAME_MARKERS:
        if marker in base or marker in stem:
            return True
    return False


def path_family_for(path: str) -> str:
    """Deterministic path-family key for clustering."""

    normalized = _normalize_path(path)
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return ""
    name = parts[-1]
    # HuggingFace auto-converted unit tests collapse to one family.
    if (
        len(parts) >= 4
        and parts[0] == "ipfs_accelerate_js"
        and parts[1] == "test"
        and parts[2] == "unit"
        and name.startswith("test_hf_")
    ):
        return "ipfs_accelerate_js/test/unit/test_hf_*"
    if (
        len(parts) >= 3
        and parts[0] == "ipfs_accelerate_js"
        and parts[1] == "test"
        and parts[2] == "browser"
    ):
        return "ipfs_accelerate_js/test/browser/*"
    if (
        len(parts) >= 3
        and parts[0] == "ipfs_accelerate_js"
        and parts[1] == "test"
        and parts[2] == "unit"
    ):
        return "ipfs_accelerate_js/test/unit/*"
    if (
        len(parts) >= 3
        and parts[0] == "ipfs_accelerate_js"
        and parts[1] == "test"
        and parts[2] == "performance"
    ):
        return "ipfs_accelerate_js/test/performance/*"
    if parts[0] == "web" and len(parts) >= 2 and parts[1] == "legacy-archive":
        return "web/legacy-archive/*"
    if parts[0] == "docs" and len(parts) >= 2 and parts[1] == "ast_exports":
        return "docs/ast_exports/*"
    if parts[0] == "benchmark-results":
        return "benchmark-results/*"
    if parts[0] == "test" and len(parts) >= 2:
        return f"test/{parts[1]}/*"
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}/*"
    return f"{parts[0]}/*"


def actionable_repair_family(path: str) -> str:
    """Bounded SCA-232..SCA-237 repair-family key for one retained failure path.

    Mirrors the content-addressed backlog families so triage and repair tasks
    share a single deterministic partition.  Unknown paths raise rather than
    invent a silent catch-all.
    """

    text = _normalize_path(path)
    active_js = {
        "ipfs_accelerate_js/src/utils/run_web_platform_integration_tests.js",
        "test/mocks/stubs/chai-stub.js",
        "test/unit/cli/chat-command.test.js",
        "test/utils/mockMCPClient.js",
    }
    python_paths = {
        "ipfs_accelerate_js/test/performance/webgpu_optimizer/run_benchmarks.py",
        "test/fixed_web_platform/cross_browser_model_sharding.py",
        "test/web_platform_test_output/test_hf_bert.py",
    }
    structured = {
        "benchmark-results/sample-baseline.json",
        "docs/ast_exports/full_asts/python/swissknife_old/"
        "ipfs_transformers.py.ast.json",
    }
    if text.startswith("ipfs_accelerate_js/test/unit/"):
        return "UNIT"
    if text.startswith("ipfs_accelerate_js/test/browser/"):
        return "BROWSER"
    if text in active_js:
        return "ACTIVEJS"
    if text in python_paths:
        return "PYTHON"
    if text in structured:
        return "STRUCTURED"
    if text.startswith("web/legacy-archive/"):
        return "LEGACY"
    raise ParserFailureTriageError(
        f"unclassified parser failure path for repair family: {text}",
        reason_code="unclassified_repair_family",
        details={"path": text},
    )


def normalize_cluster_reason(raw_reason: Any, *, fallback: str = "parse_failure") -> str:
    """Stable cluster reason independent of column/line drift."""

    text = str(raw_reason or "").strip()
    if not text:
        return fallback
    lower = text.casefold()
    if "file_bytes_exceeded" in lower:
        return "file_bytes_exceeded"
    if "jsondecodeerror" in lower or lower.startswith("json_decode"):
        return "json_decode_error"
    if "indentationerror" in lower:
        return "python_indentation_error"
    if "syntaxerror" in lower:
        return "python_syntax_error"
    if "misclassified_extension" in lower or "shebang_extension_mismatch" in lower:
        return "misclassified_extension"
    ts_codes = _TS_CODE_RE.findall(text)
    if ts_codes or lower.startswith("typescript_parse_error"):
        ordered: list[str] = []
        for code in ts_codes:
            if code not in ordered:
                ordered.append(code)
        if ordered:
            return "typescript_parse_error:" + "|".join(ordered[:12])
        return "typescript_parse_error"
    # Fall back to polyglot typed head, then sanitize.
    head = typed_reason_code(text, fallback=fallback)
    return head[:160] or fallback


def detect_shebang_extension_mismatch(source: str, path: str = "") -> bool:
    """True when body shebang disagrees with a JS/TS/Python path extension."""

    first = (source or "").lstrip("\ufeff").splitlines()[:1]
    if not first:
        return False
    line = first[0].strip()
    if not _SHEBANG_RE.match(line):
        return False
    suffix = Path(_normalize_path(path)).suffix.casefold() if path else ""
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        return True
    if suffix == ".py" and re.search(r"\b(?:bash|sh|zsh)\b", line):
        return True
    return False


def classify_member_disposition(
    member: FailureMember,
    *,
    policy: Sequence[ReviewedExclusionRule] = DEFAULT_REVIEWED_EXCLUSION_POLICY,
    source_hint: str = "",
) -> tuple[ClusterDispositionKind, TriageAction, str]:
    """Assign disposition/action for one failure member."""

    if member.protected_surface:
        return (
            ClusterDispositionKind.GENUINE_SOURCE_DEFECT,
            TriageAction.COUNT_AS_FAILURE,
            "",
        )

    if source_hint and detect_shebang_extension_mismatch(source_hint, member.path):
        return (
            ClusterDispositionKind.UNSUPPORTED_OR_MISCLASSIFIED,
            TriageAction.RECLASSIFY_NOT_ELIGIBLE,
            "policy:misclassified-shebang-extension",
        )

    for rule in policy:
        if rule.matches(
            path=member.path,
            language=member.language,
            reason_code=member.reason_code,
            raw_reason=member.raw_reason,
        ):
            return rule.disposition, rule.action, rule.rule_id

    # Heuristic (still non-excluding without policy): oversized reasons already
    # covered by policy.  Residual defaults to genuine defect.
    if member.reason_code == "file_bytes_exceeded":
        return (
            ClusterDispositionKind.OVERSIZED_ARTIFACT,
            TriageAction.RECLASSIFY_NOT_ELIGIBLE,
            "policy:oversized-source-byte-bound",
        )

    # Misnamed bash-as-js under src/utils without protected markers.
    base = Path(member.path).name.casefold()
    if base.endswith(".js") and "run_" in base and "test" in base:
        if "typescript_parse_error:TS1127" in member.reason_code or (
            "invalid character" in member.raw_reason.casefold()
        ):
            return (
                ClusterDispositionKind.UNSUPPORTED_OR_MISCLASSIFIED,
                TriageAction.RECLASSIFY_NOT_ELIGIBLE,
                "policy:misclassified-shebang-extension",
            )

    return (
        ClusterDispositionKind.GENUINE_SOURCE_DEFECT,
        TriageAction.COUNT_AS_FAILURE,
        "",
    )


def failure_rows_from_coverage(
    rows: Iterable[Mapping[str, Any] | PathDispositionRecord],
) -> tuple[dict[str, Any], ...]:
    """Select parse-failure rows from a coverage/index ledger."""

    failures: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, PathDispositionRecord):
            if row.outcome is not PathParseOutcome.BOUNDED_FAILURE:
                continue
            payload = row.to_dict()
        elif isinstance(row, Mapping):
            payload = dict(row)
            status = str(
                payload.get("parser_status")
                or payload.get("status")
                or payload.get("outcome")
                or ""
            ).casefold()
            disposition = classify_path_disposition(payload)
            if disposition.outcome is not PathParseOutcome.BOUNDED_FAILURE and (
                status not in {"parse_failure", "failure", "failed", "error", "bounded_failure"}
            ):
                continue
            if disposition.outcome is PathParseOutcome.BOUNDED_FAILURE:
                payload.setdefault("parser_status", "parse_failure")
        else:
            raise ParserFailureTriageError(
                "coverage rows must be mappings",
                reason_code="invalid_row",
            )
        path = str(payload.get("path") or "").strip()
        if not path:
            continue
        failures.append(payload)
    return tuple(failures)


def member_from_row(row: Mapping[str, Any]) -> FailureMember:
    """Build a body-free failure member from one index/coverage row."""

    path = _normalize_path(str(row.get("path") or ""))
    if not path:
        raise ParserFailureTriageError(
            "failure row missing path",
            reason_code="missing_path",
        )
    raw_reason = str(
        row.get("parser_reason")
        or row.get("reason_code")
        or row.get("parse_error")
        or ""
    )
    language = _normalize_language(row.get("language") or "")
    parser_identity = str(row.get("parser_identity") or "").strip()
    reason_code = normalize_cluster_reason(raw_reason)
    return FailureMember(
        path=path,
        language=language,
        parser_identity=parser_identity,
        reason_code=reason_code,
        raw_reason=raw_reason,
        content_digest=str(row.get("content_digest") or "").strip(),
        row_id=str(row.get("row_id") or "").strip(),
        path_family=path_family_for(path),
        protected_surface=is_protected_contract_surface(path),
    )


def cluster_parser_failures(
    members: Sequence[FailureMember],
    *,
    policy: Sequence[ReviewedExclusionRule] = DEFAULT_REVIEWED_EXCLUSION_POLICY,
    max_samples: int = _DEFAULT_MAX_CLUSTER_SAMPLES,
) -> tuple[tuple[TriageCluster, ...], tuple[MemberAssignment, ...]]:
    """Cluster members and assign dispositions deterministically."""

    if max_samples < 0:
        raise ParserFailureTriageError(
            "max_samples must be non-negative",
            reason_code="invalid_cluster_limits",
        )

    buckets: dict[
        tuple[str, str, str, str], list[tuple[FailureMember, ClusterDispositionKind, TriageAction, str]]
    ] = {}
    provisional: list[
        tuple[FailureMember, ClusterDispositionKind, TriageAction, str]
    ] = []

    for member in members:
        disposition, action, rule_id = classify_member_disposition(
            member, policy=policy
        )
        # Protected surfaces force count-as-failure regardless of bucket peers.
        if member.protected_surface:
            disposition = ClusterDispositionKind.GENUINE_SOURCE_DEFECT
            action = TriageAction.COUNT_AS_FAILURE
            rule_id = ""
        provisional.append((member, disposition, action, rule_id))
        key = (
            member.parser_identity,
            member.reason_code,
            member.path_family,
            member.language,
        )
        buckets.setdefault(key, []).append(
            (member, disposition, action, rule_id)
        )

    clusters: list[TriageCluster] = []
    assignments: list[MemberAssignment] = []
    # When a bucket mixes protected and excludable members, split by action.
    for key in sorted(
        buckets,
        key=lambda item: (
            -len(buckets[item]),
            item[2],
            item[1],
            item[3],
            item[0],
        ),
    ):
        entries = buckets[key]
        # Group by (disposition, action, policy_rule) so exclusions stay pure.
        sub: dict[
            tuple[ClusterDispositionKind, TriageAction, str],
            list[FailureMember],
        ] = {}
        for member, disposition, action, rule_id in entries:
            sub.setdefault((disposition, action, rule_id), []).append(member)
        for (disposition, action, rule_id), group in sorted(
            sub.items(),
            key=lambda pair: (
                -len(pair[1]),
                pair[0][0].value,
                pair[0][1].value,
                pair[0][2],
            ),
        ):
            samples = tuple(item.path for item in group[:max_samples])
            member_ids = tuple(item.member_id for item in group[:max_samples])
            protected_count = sum(1 for item in group if item.protected_surface)
            # If exclusion slipped through with protected members, force failure.
            final_action = action
            final_disposition = disposition
            final_rule = rule_id
            if protected_count and action in {
                TriageAction.EXCLUDE_FROM_BUDGET,
                TriageAction.RECLASSIFY_NOT_ELIGIBLE,
            }:
                final_action = TriageAction.COUNT_AS_FAILURE
                final_disposition = ClusterDispositionKind.GENUINE_SOURCE_DEFECT
                final_rule = ""
            cluster = TriageCluster(
                parser_identity=key[0],
                reason_code=key[1],
                path_family=key[2],
                language=key[3],
                count=len(group),
                disposition=final_disposition,
                action=final_action,
                policy_rule_id=final_rule,
                sample_paths=samples,
                member_ids=member_ids,
                protected_member_count=protected_count,
            )
            clusters.append(cluster)
            for member in group:
                assignments.append(
                    MemberAssignment(
                        member=member,
                        cluster_id=cluster.cluster_id,
                        disposition=final_disposition,
                        action=final_action,
                        policy_rule_id=final_rule,
                    )
                )

    # Stable order: budgeted failures first, then by count desc.
    clusters_sorted = tuple(
        sorted(
            clusters,
            key=lambda item: (
                0 if item.counts_toward_budget else 1,
                -item.count,
                item.path_family,
                item.reason_code,
                item.cluster_id,
            ),
        )
    )
    assignment_by_path = {item.member.path: item for item in assignments}
    assignments_sorted = tuple(
        assignment_by_path[path]
        for path in sorted(assignment_by_path)
    )
    return clusters_sorted, assignments_sorted


def project_health_gate(
    *,
    eligible_path_count: int,
    assignments: Sequence[MemberAssignment],
    max_parser_failures: int = REVIEWED_MAX_PARSER_FAILURES,
    max_parser_failure_ratio: float = REVIEWED_MAX_PARSER_FAILURE_RATIO,
) -> HealthGateProjection:
    """Project residual failure budget without changing thresholds."""

    if max_parser_failures < 0:
        raise ParserFailureTriageError(
            "max_parser_failures must be non-negative",
            reason_code="invalid_threshold",
        )
    if not 0.0 <= float(max_parser_failure_ratio) <= 1.0:
        raise ParserFailureTriageError(
            "max_parser_failure_ratio must be between 0 and 1",
            reason_code="invalid_threshold",
        )
    residual = sum(
        1 for item in assignments if item.action is TriageAction.COUNT_AS_FAILURE
    )
    excluded = sum(
        1
        for item in assignments
        if item.action is TriageAction.EXCLUDE_FROM_BUDGET
    )
    reclassified = sum(
        1
        for item in assignments
        if item.action is TriageAction.RECLASSIFY_NOT_ELIGIBLE
    )
    # Eligible denominator: original eligible minus reclassified-out paths.
    adjusted_eligible = max(0, int(eligible_path_count) - reclassified)
    ratio = (
        residual / adjusted_eligible
        if adjusted_eligible
        else (1.0 if residual else 0.0)
    )
    reasons: list[str] = []
    meets = True
    if residual > max_parser_failures or ratio > max_parser_failure_ratio:
        meets = False
        reasons.append("parser_failure_budget_exceeded")
    if any(
        item.member.protected_surface
        and item.action is not TriageAction.COUNT_AS_FAILURE
        for item in assignments
    ):
        meets = False
        reasons.append("protected_surface_excluded")
    status = "healthy" if meets and residual == 0 else (
        "healthy" if meets else "unhealthy"
    )
    if meets and residual:
        # Within budget but non-zero remains a partial signal for promotion.
        status = "partial"
        reasons.append("parser_failures_within_budget")
    return HealthGateProjection(
        eligible_path_count=adjusted_eligible,
        residual_failure_count=residual,
        excluded_failure_count=excluded,
        reclassified_count=reclassified,
        residual_failure_ratio=ratio,
        max_parser_failures=int(max_parser_failures),
        max_parser_failure_ratio=float(max_parser_failure_ratio),
        meets_gate=meets,
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def apply_triage_to_rows(
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[MemberAssignment],
) -> tuple[dict[str, Any], ...]:
    """Rewrite coverage rows so excluded/reclassified failures leave the budget.

    Excluded fixtures become ``parser_status=excluded`` / not eligible.
    Reclassified artifacts become ``unsupported``.  Residual failures and all
    successes are preserved.  Malformed source is never relabeled success.
    """

    by_path = {item.member.path: item for item in assignments}
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        path = _normalize_path(str(payload.get("path") or ""))
        assignment = by_path.get(path)
        if assignment is None:
            rewritten.append(payload)
            continue
        if assignment.action is TriageAction.COUNT_AS_FAILURE:
            # Keep typed failure; never upgrade to success.
            payload["parser_status"] = "parse_failure"
            payload.setdefault(
                "parser_reason",
                assignment.member.raw_reason or assignment.member.reason_code,
            )
            payload["triage_cluster_id"] = assignment.cluster_id
            payload["triage_disposition"] = assignment.disposition.value
            payload["triage_action"] = assignment.action.value
            if assignment.policy_rule_id:
                payload["policy_rule"] = assignment.policy_rule_id
            rewritten.append(payload)
            continue
        if assignment.action is TriageAction.EXCLUDE_FROM_BUDGET:
            payload["parser_status"] = "excluded"
            payload["disposition_kind"] = "excluded"
            payload["reason_code"] = (
                assignment.policy_rule_id
                or assignment.disposition.value
                or "reviewed_exclusion"
            )
            payload["parser_reason"] = payload["reason_code"]
            payload["policy_rule"] = (
                assignment.policy_rule_id or "policy:reviewed-exclusion"
            )
            payload["triage_cluster_id"] = assignment.cluster_id
            payload["triage_disposition"] = assignment.disposition.value
            payload["triage_action"] = assignment.action.value
            rewritten.append(payload)
            continue
        if assignment.action is TriageAction.RECLASSIFY_NOT_ELIGIBLE:
            payload["parser_status"] = "unsupported"
            payload["disposition_kind"] = "unsupported"
            payload["reason_code"] = (
                assignment.disposition.value or "unsupported_or_misclassified"
            )
            payload["parser_reason"] = payload["reason_code"]
            payload["policy_rule"] = (
                assignment.policy_rule_id or "policy:reviewed-reclassify"
            )
            payload["triage_cluster_id"] = assignment.cluster_id
            payload["triage_disposition"] = assignment.disposition.value
            payload["triage_action"] = assignment.action.value
            rewritten.append(payload)
            continue
        # APPLY_PARSER_REPAIR currently keeps failure until a real parse succeeds.
        payload["parser_status"] = "parse_failure"
        payload["triage_cluster_id"] = assignment.cluster_id
        payload["triage_disposition"] = assignment.disposition.value
        payload["triage_action"] = assignment.action.value
        rewritten.append(payload)
    return tuple(rewritten)


def run_parser_repair_fixtures(
    repairs: Sequence[ParserRepair] | None = None,
) -> dict[str, Any]:
    """Execute positive/negative fixture expectations for analyzer repairs.

    Fixtures are evaluated with local classifiers (no Node process, no model).
    """

    active = tuple(repairs or default_parser_repairs())
    results: list[dict[str, Any]] = []
    all_passed = True
    for repair in active:
        for fixture in (*repair.positive_fixtures, *repair.negative_fixtures):
            passed = _evaluate_repair_fixture(fixture)
            if not passed:
                all_passed = False
            results.append(
                {
                    "repair_id": repair.repair_id,
                    "fixture_id": fixture.fixture_id,
                    "expect_success": fixture.expect_success,
                    "passed": passed,
                    "language": fixture.language,
                }
            )
    return {
        "passed": all_passed and bool(results),
        "fixture_count": len(results),
        "results": results,
    }


def _evaluate_repair_fixture(fixture: ParserRepairFixture) -> bool:
    """Evaluate one in-memory repair fixture without retaining it on reports."""

    source = fixture.source
    language = _normalize_language(fixture.language)
    if fixture.repair_id == "repair:detect-shebang-extension-mismatch":
        mismatch = detect_shebang_extension_mismatch(
            source, path=f"fixture.{_ext_for_language(language)}"
        )
        if fixture.expect_success:
            return not mismatch and _stdlib_parse_ok(source, language)
        # Negative: either mismatch detected or parse fails — never silent success.
        if mismatch:
            return True
        return not _stdlib_parse_ok(source, language)

    if fixture.repair_id == "repair:normalize-typescript-diagnostic-codes":
        if fixture.expect_success:
            return _stdlib_parse_ok(source, language)
        # Negative: invent a diagnostic string and ensure normalization is stable.
        if not _stdlib_parse_ok(source, language):
            synthetic = (
                "typescript_parse_error:TS1002@9:79:Unterminated string literal."
                "|TS1005@14:5:'{' expected."
            )
            a = normalize_cluster_reason(synthetic)
            b = normalize_cluster_reason(
                "typescript_parse_error:TS1002@99:1:Unterminated string literal."
                "|TS1005@100:2:'{' expected."
            )
            return a == b == "typescript_parse_error:TS1002|TS1005"
        return False

    if fixture.repair_id == "repair:protected-mcp-surface-never-excluded":
        if fixture.fixture_id.endswith("mockmcp") or "mockMCP" in fixture.fixture_id:
            path = "test/utils/mockMCPClient.js"
            assert is_protected_contract_surface(path)
            member = FailureMember(
                path=path,
                language="javascript",
                parser_identity="parser:fixture",
                reason_code="typescript_parse_error:TS1005",
                raw_reason="typescript_parse_error:TS1005@1:1:',' expected.",
            )
            disposition, action, rule = classify_member_disposition(member)
            # Must count as failure with no exclusion rule.
            return (
                action is TriageAction.COUNT_AS_FAILURE
                and disposition is ClusterDispositionKind.GENUINE_SOURCE_DEFECT
                and rule == ""
                and not fixture.expect_success
            )
        # Non-MCP invalid fixture may be excluded by policy.
        path = "ipfs_accelerate_js/test/unit/test_hf_example.ts"
        member = FailureMember(
            path=path,
            language="typescript",
            parser_identity="parser:fixture",
            reason_code="typescript_parse_error:TS1002|TS1005",
            raw_reason="typescript_parse_error:TS1002@1:1:Unterminated string literal.",
        )
        disposition, action, rule = classify_member_disposition(member)
        return (
            action is TriageAction.EXCLUDE_FROM_BUDGET
            and rule.startswith("policy:")
            and not is_protected_contract_surface(path)
        )

    # Default: stdlib parse expectation.
    ok = _stdlib_parse_ok(source, language)
    return ok if fixture.expect_success else (not ok)


def _ext_for_language(language: str) -> str:
    return {
        "typescript": "ts",
        "tsx": "tsx",
        "javascript": "js",
        "jsx": "jsx",
        "python": "py",
        "json": "json",
    }.get(_normalize_language(language), "txt")


def _stdlib_parse_ok(source: str, language: str) -> bool:
    lang = _normalize_language(language)
    if lang == "python":
        try:
            compile(source, "<fixture>", "exec")
            return True
        except SyntaxError:
            return False
    if lang in {"json", "json-schema", "openapi-json"}:
        try:
            json.loads(source)
            return True
        except json.JSONDecodeError:
            return False
    if lang in {"javascript", "typescript", "jsx", "tsx"}:
        # Lightweight structural checks without claiming TS compiler authority.
        if detect_shebang_extension_mismatch(
            source, path=f"x.{_ext_for_language(lang)}"
        ):
            return False
        stripped = source.strip()
        if not stripped:
            return False
        if stripped.count('"') % 2 == 1 or stripped.count("'") % 2 == 1:
            return False
        if stripped.count("{") != stripped.count("}"):
            return False
        if stripped.count("(") != stripped.count(")"):
            return False
        return True
    return False


def triage_parser_failures(
    rows: Iterable[Mapping[str, Any] | PathDispositionRecord],
    *,
    source_index_id: str = "",
    eligible_path_count: int | None = None,
    policy: Sequence[ReviewedExclusionRule] = DEFAULT_REVIEWED_EXCLUSION_POLICY,
    repairs: Sequence[ParserRepair] | None = None,
    max_parser_failures: int = REVIEWED_MAX_PARSER_FAILURES,
    max_parser_failure_ratio: float = REVIEWED_MAX_PARSER_FAILURE_RATIO,
) -> ParserFailureTriageReport:
    """Classify every parse failure into one deterministic cluster."""

    # Thresholds must not be weaker than the reviewed gate.
    if max_parser_failures > REVIEWED_MAX_PARSER_FAILURES:
        raise ParserFailureTriageError(
            "max_parser_failures cannot exceed reviewed gate of "
            f"{REVIEWED_MAX_PARSER_FAILURES}",
            reason_code="threshold_weakened",
        )
    if max_parser_failure_ratio > REVIEWED_MAX_PARSER_FAILURE_RATIO:
        raise ParserFailureTriageError(
            "max_parser_failure_ratio cannot exceed reviewed gate of "
            f"{REVIEWED_MAX_PARSER_FAILURE_RATIO}",
            reason_code="threshold_weakened",
        )

    row_list = list(rows)
    failures = failure_rows_from_coverage(row_list)
    members = tuple(member_from_row(row) for row in failures)
    clusters, assignments = cluster_parser_failures(members, policy=policy)

    if eligible_path_count is None:
        # Count eligible as success + failure dispositions in the input ledger.
        eligible = 0
        for row in row_list:
            disposition = classify_path_disposition(row)
            if disposition.outcome in {
                PathParseOutcome.SUCCESS,
                PathParseOutcome.BOUNDED_FAILURE,
            }:
                eligible += 1
        eligible_path_count = eligible

    health = project_health_gate(
        eligible_path_count=int(eligible_path_count),
        assignments=assignments,
        max_parser_failures=max_parser_failures,
        max_parser_failure_ratio=max_parser_failure_ratio,
    )
    active_repairs = tuple(repairs if repairs is not None else default_parser_repairs())
    fixture_report = run_parser_repair_fixtures(active_repairs)

    disposition_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for assignment in assignments:
        key = assignment.disposition.value
        disposition_counts[key] = disposition_counts.get(key, 0) + 1
        family = assignment.member.actionable_family or "UNCLASSIFIED"
        family_counts[family] = family_counts.get(family, 0) + 1

    # Bounded repair-family manifest: one row per family with member paths.
    repair_family_manifest = []
    for family in sorted(family_counts):
        member_paths = sorted(
            item.member.path
            for item in assignments
            if (item.member.actionable_family or "UNCLASSIFIED") == family
        )
        cluster_ids = sorted(
            {
                item.cluster_id
                for item in assignments
                if (item.member.actionable_family or "UNCLASSIFIED") == family
            }
        )
        repair_family_manifest.append(
            {
                "family": family,
                "failure_count": family_counts[family],
                "cluster_ids": cluster_ids,
                "member_paths": member_paths,
            }
        )

    metrics = {
        "input_row_count": len(row_list),
        "failure_count": len(members),
        "cluster_count": len(clusters),
        "protected_failure_count": sum(
            1 for item in members if item.protected_surface
        ),
        "budgeted_failure_count": health.residual_failure_count,
        "excluded_failure_count": health.excluded_failure_count,
        "reclassified_count": health.reclassified_count,
        "disposition_counts": disposition_counts,
        "actionable_family_counts": dict(sorted(family_counts.items())),
        "repair_family_manifest": repair_family_manifest,
        "repair_fixtures_passed": fixture_report["passed"],
        "repair_fixture_count": fixture_report["fixture_count"],
        "max_parser_failures": max_parser_failures,
        "max_parser_failure_ratio": max_parser_failure_ratio,
        "thresholds_unchanged": True,
        "non_authoritative": True,
        "satisfies_fresh_health_authority": False,
        "satisfies_repair_task": False,
    }
    return ParserFailureTriageReport(
        source_index_id=str(source_index_id or ""),
        failure_count=len(members),
        cluster_count=len(clusters),
        clusters=clusters,
        assignments=assignments,
        policy=tuple(policy),
        repairs=active_repairs,
        health_gate=health,
        metrics=metrics,
        unassigned_count=0,
    )


def assess_health_after_triage(
    rows: Sequence[Mapping[str, Any]],
    *,
    assignments: Sequence[MemberAssignment] | None = None,
    report: ParserFailureTriageReport | None = None,
    run_canaries: bool = False,
    repair_authority: bool = False,
    thresholds: Mapping[str, LanguageHealthThresholds | Mapping[str, Any]]
    | None = None,
) -> PolyglotASTHealthReport:
    """Re-assess polyglot AST health after applying triage dispositions."""

    if assignments is None:
        if report is None:
            report = triage_parser_failures(rows)
        assignments = report.assignments
    rewritten = apply_triage_to_rows(rows, assignments)
    return assess_polyglot_ast_health(
        rewritten,
        thresholds=thresholds,
        run_canaries=run_canaries,
        repair_authority=repair_authority,
    )


def load_index_document(
    path: str | os.PathLike[str],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Load coverage/index rows plus document metadata."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return tuple(
            dict(item) for item in payload if isinstance(item, Mapping)
        ), {}
    if not isinstance(payload, Mapping):
        raise ParserFailureTriageError(
            "index document must be an object or array",
            reason_code="invalid_index_document",
        )
    rows = load_coverage_rows(target)
    meta = {
        "index_id": str(payload.get("index_id") or ""),
        "snapshot_id": str(
            payload.get("snapshot_id")
            or (payload.get("snapshot") or {}).get("snapshot_id")
            or ""
        ),
        "stats": dict(payload.get("stats") or {}),
        "health": dict(payload.get("health") or {}),
        "schema": str(payload.get("schema") or ""),
    }
    return rows, meta


def build_triage_from_index(
    index_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    policy: Sequence[ReviewedExclusionRule] = DEFAULT_REVIEWED_EXCLUSION_POLICY,
    max_parser_failures: int = REVIEWED_MAX_PARSER_FAILURES,
    max_parser_failure_ratio: float = REVIEWED_MAX_PARSER_FAILURE_RATIO,
) -> ParserFailureTriageReport:
    """Triage parse failures from a diagnostic repository index or coverage ledger."""

    rows, meta = load_index_document(index_path)
    stats = meta.get("stats") or {}
    eligible = stats.get("eligible_parser_path_count")
    if eligible is None:
        eligible = stats.get("eligible_path_count")
    report = triage_parser_failures(
        rows,
        source_index_id=str(meta.get("index_id") or ""),
        eligible_path_count=int(eligible) if eligible is not None else None,
        policy=policy,
        max_parser_failures=max_parser_failures,
        max_parser_failure_ratio=max_parser_failure_ratio,
    )
    if output_path is not None:
        write_parser_failure_triage_report(report, output_path)
    return report


def write_parser_failure_triage_report(
    report: ParserFailureTriageReport,
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically write a body-free triage report."""

    target = Path(path)
    payload = report.to_dict(include_identity=True)
    if report_contains_source_body(payload):
        raise ParserFailureTriageError(
            "refusing to write triage report with source body",
            reason_code="source_body_forbidden",
        )
    encoded = _canonical_json_bytes(payload) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload["content_identity"]


def polyglot_clusters_from_triage(
    report: ParserFailureTriageReport,
) -> tuple[FailureCluster, ...]:
    """Project triage clusters into polyglot FailureCluster rows (budgeted only)."""

    clusters: list[FailureCluster] = []
    for item in report.clusters:
        if not item.counts_toward_budget:
            continue
        clusters.append(
            FailureCluster(
                language=item.language,
                reason_code=item.reason_code,
                parser_identity=item.parser_identity,
                count=item.count,
                sample_disposition_ids=(),
                sample_paths=item.sample_paths,
            )
        )
    return tuple(clusters)


__all__ = [
    "DEFAULT_REVIEWED_EXCLUSION_POLICY",
    "PARSER_FAILURE_TRIAGE_EVIDENCE",
    "PARSER_FAILURE_TRIAGE_INTERFACE",
    "PARSER_FAILURE_TRIAGE_SCHEMA",
    "REVIEWED_MAX_PARSER_FAILURES",
    "REVIEWED_MAX_PARSER_FAILURE_RATIO",
    "ClusterDispositionKind",
    "FailureMember",
    "HealthGateProjection",
    "MemberAssignment",
    "ParserFailureTriageError",
    "ParserFailureTriageReport",
    "ParserRepair",
    "ParserRepairFixture",
    "ReviewedExclusionRule",
    "TriageAction",
    "TriageCluster",
    "actionable_repair_family",
    "apply_triage_to_rows",
    "assess_health_after_triage",
    "build_triage_from_index",
    "classify_member_disposition",
    "cluster_parser_failures",
    "default_parser_repairs",
    "detect_shebang_extension_mismatch",
    "failure_rows_from_coverage",
    "is_protected_contract_surface",
    "load_index_document",
    "member_from_row",
    "normalize_cluster_reason",
    "path_family_for",
    "polyglot_clusters_from_triage",
    "project_health_gate",
    "run_parser_repair_fixtures",
    "triage_parser_failures",
    "write_parser_failure_triage_report",
]
