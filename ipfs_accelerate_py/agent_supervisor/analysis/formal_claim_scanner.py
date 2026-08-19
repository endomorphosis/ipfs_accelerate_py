"""Repository-wide ambiguous-claim scanner (FACP-019).

Detects unqualified success / support / verification / currency / production
claim fields on production-shaped surfaces, emits source-bound findings with
an abstract repair trace and roadmap repair family, distinguishes typed FCA
compatibility aliases from forbidden generic fields, and applies a low-noise
allowlist that is forbidden from suppressing seeded corpus defects.

This module does not edit sources. Naming alone is never a defect: findings
require a claim-shaped assignment or literal binding.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Optional, Sequence, Union

SCHEMA: Final[str] = "facp/ambiguous-claim-scan@1"
EVIDENCE_SCHEMA: Final[str] = "facp/ambiguous-claim-scan@1"
VOCAB_SCHEMA: Final[str] = "facp/formal-claim-algebra-v1@1"
TASK_ID: Final[str] = "FACP-019"
GOAL_ID: Final[str] = "FACP-G130"
BUNDLE: Final[str] = "facp/fca/scanner"
SCANNER_VERSION: Final[str] = "formal-claim-scanner/v1"

# Normative forbidden generic fields on migrated production paths (§8.1).
FORBIDDEN_GENERIC_FIELDS: Final[frozenset[str]] = frozenset(
    {"success", "available", "supported", "verified", "proven"}
)

# Additional evidence-subset tokens that are ambiguous when used as unqualified
# production claim fields (FACP-019 evidence subset).
AMBIGUOUS_EVIDENCE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "authorized",
        "allowed",
        "current",
        "production",
        "capability",
        "mock",
        "simulation",
        "fallback",
        "cid",
        "api_available",
        "hwtest",
    }
)

# Typed FCA predicates / dimension-qualified spellings. These are explicit
# compatibility aliases and MUST NOT be classified as ambiguous generics.
TYPED_COMPATIBILITY_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "production_supported",
        "effect_successful",
        "proof_reusable",
        "receipt_authoritative",
        "release_admissible",
        "proof.verified",
        "proof.candidate",
        "proof.none",
        "proof.refuted",
        "proof.unknown",
        "proof.verifier_unavailable",
        "effect.observed",
        "effect.started",
        "effect.failed",
        "effect.externally_unknown",
        "effect.not_started",
        "origin.live_observed",
        "origin.hermetic_observed",
        "origin.simulated",
        "origin.fixture",
        "origin.declared",
        "origin.absent",
        "authority.valid",
        "authority.denied",
        "authority.absent",
        "authority.unchecked",
        "policy.allowed",
        "policy.denied",
        "policy.allowed_with_obligations",
        "policy.indeterminate",
        "policy.unchecked",
        "freshness.current",
        "freshness.stale",
        "freshness.superseded",
        "freshness.withdrawn",
        "environment.live",
        "environment.hermetic",
        "environment.conditional",
        "integrity.digest_valid",
        "integrity.signature_valid",
        "integrity.structurally_valid",
        "integrity.unchecked",
        "review.human_reviewed",
        "review.machine_reviewed",
        "review.unreviewed",
    }
)

ROADMAP_DEFECT_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "false_success",
        "mock_capability",
        "pseudo_cid",
        "import_effect",
        "browser_authority",
        "mutable_dependency",
        "stale_proof",
        "missing_recovery",
        "license_conflict",
        "hermetic_to_live",
        "secret_flow",
        "canonicalization_conflict",
        "total_assurance_ladder",
    }
)

_FIELD_REPAIR_FAMILY: Final[Mapping[str, str]] = {
    "success": "false_success",
    "available": "false_success",
    "supported": "false_success",
    "verified": "false_success",
    "proven": "false_success",
    "api_available": "false_success",
    "authorized": "browser_authority",
    "allowed": "browser_authority",
    "current": "stale_proof",
    "production": "hermetic_to_live",
    "capability": "mock_capability",
    "mock": "mock_capability",
    "simulation": "mock_capability",
    "fallback": "missing_recovery",
    "cid": "pseudo_cid",
    "hwtest": "mock_capability",
}

_TRUTHY_STRINGS: Final[frozenset[str]] = frozenset(
    {
        "true",
        "yes",
        "success",
        "ok",
        "available",
        "supported",
        "verified",
        "proven",
        "authorized",
        "allowed",
        "current",
        "production",
        "real",
        "live",
    }
)

_SCAN_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".pyi", ".json"})
_SKIP_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
        ".eggs",
    }
)


class FormalClaimScannerError(ValueError):
    """Malformed scanner input or corpus binding."""


class ClaimKind(str, Enum):
    FORBIDDEN_GENERIC = "forbidden_generic"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    TYPED_COMPATIBILITY_ALIAS = "typed_compatibility_alias"
    NOT_A_CLAIM = "not_a_claim"


class FindingDisposition(str, Enum):
    REJECT = "reject"
    COMPATIBILITY_ALIAS = "compatibility_alias"
    ALLOWLISTED = "allowlisted"
    CORPUS_BOUND = "corpus_bound"


@dataclass(frozen=True)
class SourceSpan:
    """Exact source locus for one ambiguous claim."""

    path: str
    start_line: int
    end_line: int
    symbol: str = ""
    excerpt: str = ""
    column: int = 0

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise FormalClaimScannerError("source span path is required")
        if (
            isinstance(self.start_line, bool)
            or not isinstance(self.start_line, int)
            or self.start_line < 1
        ):
            raise FormalClaimScannerError("source span start_line must be >= 1")
        if (
            isinstance(self.end_line, bool)
            or not isinstance(self.end_line, int)
            or self.end_line < self.start_line
        ):
            raise FormalClaimScannerError("source span end_line must be >= start_line")

    def overlaps(self, other: "SourceSpan") -> bool:
        if _normalize_relpath(self.path) != _normalize_relpath(other.path):
            return False
        return not (self.end_line < other.start_line or other.end_line < self.start_line)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "excerpt": self.excerpt,
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceSpan":
        return cls(
            path=str(payload.get("path") or ""),
            start_line=int(payload.get("start_line") or payload.get("line_start") or 0),
            end_line=int(
                payload.get("end_line")
                or payload.get("line_end")
                or payload.get("start_line")
                or payload.get("line_start")
                or 0
            ),
            symbol=str(payload.get("symbol") or ""),
            excerpt=str(payload.get("excerpt") or payload.get("quote") or ""),
            column=int(payload.get("column") or 0),
        )


@dataclass(frozen=True)
class AbstractTraceStep:
    """One abstract location on the claim-to-surface path."""

    kind: str
    label: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "detail": self.detail}


@dataclass(frozen=True)
class AbstractTrace:
    """Bounded abstract claim-to-surface trace (never a full repository dump)."""

    steps: tuple[AbstractTraceStep, ...]
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise FormalClaimScannerError("abstract trace requires at least one step")
        if len(self.steps) > 32:
            raise FormalClaimScannerError("abstract trace exceeds step bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary or self.steps[0].label,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class AmbiguousClaimFinding:
    """One source-bound ambiguous claim finding."""

    finding_id: str
    field_name: str
    claim_kind: ClaimKind
    disposition: FindingDisposition
    repair_family: str
    source_span: SourceSpan
    abstract_trace: AbstractTrace
    value_repr: str = ""
    message: str = ""
    corpus_seed_id: str = ""
    corpus_defect_id: str = ""
    roadmap_seed: bool = False
    allowlist_entry_id: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id.strip():
            raise FormalClaimScannerError("finding_id is required")
        if self.repair_family and self.repair_family not in ROADMAP_DEFECT_FAMILIES:
            raise FormalClaimScannerError(
                f"unknown repair family: {self.repair_family!r}"
            )
        if self.disposition is FindingDisposition.COMPATIBILITY_ALIAS:
            if self.claim_kind is not ClaimKind.TYPED_COMPATIBILITY_ALIAS:
                raise FormalClaimScannerError(
                    "compatibility_alias disposition requires typed alias kind"
                )

    @property
    def is_corpus_defect(self) -> bool:
        return bool(self.corpus_seed_id) or bool(self.roadmap_seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "field_name": self.field_name,
            "claim_kind": self.claim_kind.value,
            "disposition": self.disposition.value,
            "repair_family": self.repair_family,
            "source_span": self.source_span.to_dict(),
            "abstract_trace": self.abstract_trace.to_dict(),
            "value_repr": self.value_repr,
            "message": self.message,
            "corpus_seed_id": self.corpus_seed_id,
            "corpus_defect_id": self.corpus_defect_id,
            "roadmap_seed": self.roadmap_seed,
            "allowlist_entry_id": self.allowlist_entry_id,
        }


@dataclass(frozen=True)
class AllowlistEntry:
    """Low-noise suppression rule. Never applies to corpus-bound defects."""

    entry_id: str
    reason: str
    path_suffix: str = ""
    field_name: str = ""
    symbol: str = ""

    def __post_init__(self) -> None:
        if not self.entry_id.strip():
            raise FormalClaimScannerError("allowlist entry_id is required")
        if not self.reason.strip():
            raise FormalClaimScannerError("allowlist reason is required")

    def matches(self, finding: AmbiguousClaimFinding) -> bool:
        if self.field_name and self.field_name != finding.field_name:
            return False
        if self.symbol and self.symbol != finding.source_span.symbol:
            return False
        if self.path_suffix:
            path = _normalize_relpath(finding.source_span.path)
            suffix = _normalize_relpath(self.path_suffix)
            if not (path == suffix or path.endswith("/" + suffix) or path.endswith(suffix)):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "reason": self.reason,
            "path_suffix": self.path_suffix,
            "field_name": self.field_name,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class AmbiguousClaimAllowlist:
    """Allowlist that refuses to suppress seeded corpus defects."""

    entries: tuple[AllowlistEntry, ...] = ()

    def matching_entry(
        self, finding: AmbiguousClaimFinding
    ) -> Optional[AllowlistEntry]:
        for entry in self.entries:
            if entry.matches(finding):
                return entry
        return None

    def may_suppress(self, finding: AmbiguousClaimFinding) -> bool:
        """Return True only for non-corpus findings that match an entry."""

        if finding.is_corpus_defect:
            return False
        return self.matching_entry(finding) is not None

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}


@dataclass(frozen=True)
class AmbiguousClaimScanReport:
    """Deterministic scan report for one tree or source set."""

    findings: tuple[AmbiguousClaimFinding, ...]
    scanned_paths: tuple[str, ...]
    allowlisted_finding_ids: tuple[str, ...] = ()
    corpus_seed_ids_bound: tuple[str, ...] = ()
    scanner_version: str = SCANNER_VERSION
    schema: str = SCHEMA

    @property
    def reject_findings(self) -> tuple[AmbiguousClaimFinding, ...]:
        return tuple(
            item
            for item in self.findings
            if item.disposition
            in {FindingDisposition.REJECT, FindingDisposition.CORPUS_BOUND}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_schema": EVIDENCE_SCHEMA,
            "vocab_schema": VOCAB_SCHEMA,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
            "scanner_version": self.scanner_version,
            "scanned_path_count": len(self.scanned_paths),
            "scanned_paths": list(self.scanned_paths),
            "finding_count": len(self.findings),
            "reject_count": len(self.reject_findings),
            "allowlisted_finding_ids": list(self.allowlisted_finding_ids),
            "corpus_seed_ids_bound": list(self.corpus_seed_ids_bound),
            "findings": [item.to_dict() for item in self.findings],
        }


def classify_field_name(name: str) -> ClaimKind:
    """Classify a field/identifier spelling without treating naming as a defect."""

    raw = str(name or "").strip()
    if not raw:
        return ClaimKind.NOT_A_CLAIM
    lowered = raw.casefold()
    if lowered in {item.casefold() for item in TYPED_COMPATIBILITY_ALIASES}:
        return ClaimKind.TYPED_COMPATIBILITY_ALIAS
    # Underscore / dotted aliases that normalize to typed predicates.
    compact = lowered.replace("-", "_")
    if compact in {item.casefold() for item in TYPED_COMPATIBILITY_ALIASES}:
        return ClaimKind.TYPED_COMPATIBILITY_ALIAS
    if lowered in FORBIDDEN_GENERIC_FIELDS:
        return ClaimKind.FORBIDDEN_GENERIC
    if lowered in AMBIGUOUS_EVIDENCE_TOKENS:
        return ClaimKind.AMBIGUOUS_EVIDENCE
    return ClaimKind.NOT_A_CLAIM


def repair_family_for_field(field_name: str, *, context: str = "") -> str:
    """Map a claim field (+ optional context) onto a roadmap repair family."""

    lowered = str(field_name or "").casefold()
    ctx = str(context or "").casefold()
    if "mock" in ctx or "simul" in ctx:
        if lowered in {"available", "supported", "capability", "hwtest", "api_available"}:
            return "mock_capability"
    if lowered == "cid" or "cid" in ctx and "sha" in ctx:
        return "pseudo_cid"
    family = _FIELD_REPAIR_FAMILY.get(lowered)
    if family is None:
        raise FormalClaimScannerError(f"no repair family for field {field_name!r}")
    return family


def _normalize_relpath(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _excerpt_line(source: str, lineno: int, limit: int = 160) -> str:
    lines = source.splitlines()
    if lineno < 1 or lineno > len(lines):
        return ""
    return lines[lineno - 1].strip()[:limit]


def _const_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
        return {"True": True, "False": False, "None": None}[node.id]
    return None


def _is_claim_value(value: Any, *, field_name: str = "") -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        if text.casefold() in _TRUTHY_STRINGS:
            return True
        # Identity / provenance fields are claim-shaped for any non-empty string
        # (raw hashes presented as CIDs, mock labels, fallback markers, etc.).
        if field_name.casefold() in {
            "cid",
            "mock",
            "simulation",
            "fallback",
            "capability",
            "production",
        }:
            return True
    return False


def _value_repr(value: Any) -> str:
    if isinstance(value, str):
        text = value if len(value) <= 64 else value[:61] + "..."
        return json.dumps(text)
    return repr(value)


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _build_trace(
    *,
    field_name: str,
    span: SourceSpan,
    enclosing: Sequence[str],
    module_path: str,
) -> AbstractTrace:
    steps: list[AbstractTraceStep] = [
        AbstractTraceStep(
            kind="claim_site",
            label=f"{field_name}@{span.path}:{span.start_line}",
            detail=span.excerpt,
        )
    ]
    for name in enclosing[-3:]:
        steps.append(
            AbstractTraceStep(kind="enclosing_scope", label=name, detail=module_path)
        )
    steps.append(
        AbstractTraceStep(
            kind="module_surface",
            label=_normalize_relpath(module_path),
            detail="production-shaped module surface",
        )
    )
    summary = " -> ".join(step.label for step in steps)
    return AbstractTrace(steps=tuple(steps), summary=summary)


def _finding_id(span: SourceSpan, field_name: str) -> str:
    return (
        f"finding:{_normalize_relpath(span.path)}:{span.start_line}:"
        f"{span.column}:{field_name}"
    )


@dataclass
class _PythonClaimVisitor(ast.NodeVisitor):
    """AST visitor that records claim-shaped bindings, not bare names."""

    path: str
    source: str
    findings: list[AmbiguousClaimFinding] = field(default_factory=list)
    _scope_stack: list[str] = field(default_factory=list)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Dict(self, node: ast.Dict) -> None:
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                continue
            key = _const_value(key_node)
            if not isinstance(key, str):
                continue
            self._maybe_record(key, value_node, key_node)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg:
            self._maybe_record(node.arg, node.value, node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._visit_assignment_target(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._visit_assignment_target(node.target, node.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Detect writing through obj["success"] handled in Assign; reading alone
        # is naming and intentionally ignored.
        self.generic_visit(node)

    def _visit_assignment_target(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Name):
            # Bare local names are naming-only unless assigned a claim literal
            # AND the name itself is a forbidden/ambiguous field. Still require
            # claim-shaped value so `success = helper()` without literal is not
            # auto-classified from naming alone when value is opaque.
            self._maybe_record(target.id, value, target, require_literal=True)
            return
        if isinstance(target, ast.Attribute):
            self._maybe_record(target.attr, value, target, require_literal=True)
            return
        if isinstance(target, ast.Subscript):
            slice_node = target.slice
            key = _const_value(slice_node)
            if isinstance(key, str):
                self._maybe_record(key, value, target, require_literal=True)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
                value.elts
            ):
                for left, right in zip(target.elts, value.elts):
                    self._visit_assignment_target(left, right)

    def _maybe_record(
        self,
        field_name: str,
        value_node: ast.AST,
        locus: ast.AST,
        *,
        require_literal: bool = False,
    ) -> None:
        kind = classify_field_name(field_name)
        if kind in {ClaimKind.NOT_A_CLAIM, ClaimKind.TYPED_COMPATIBILITY_ALIAS}:
            return
        const = _const_value(value_node)
        if const is None:
            # Non-literal values: still flag dict/kw claim keys that bind the
            # forbidden field into an API-shaped structure when the value node
            # is an obvious truthy Name/Call only if require_literal is False.
            if require_literal:
                return
            # Keyword/dict keys with non-constants are still claim-shaped API
            # fields on migrated paths (reject-on-migrated-path).
            value_repr = _qualified_name(value_node) or type(value_node).__name__
        else:
            if not _is_claim_value(const, field_name=field_name):
                return
            value_repr = _value_repr(const)

        lineno = getattr(locus, "lineno", None) or getattr(value_node, "lineno", 1)
        col = getattr(locus, "col_offset", 0) or 0
        end_lineno = getattr(locus, "end_lineno", None) or lineno
        symbol = ".".join(self._scope_stack) if self._scope_stack else ""
        excerpt = _excerpt_line(self.source, int(lineno))
        span = SourceSpan(
            path=self.path,
            start_line=int(lineno),
            end_line=int(end_lineno),
            symbol=symbol,
            excerpt=excerpt,
            column=int(col),
        )
        context = " ".join(self._scope_stack) + " " + excerpt
        family = repair_family_for_field(field_name, context=context)
        trace = _build_trace(
            field_name=field_name,
            span=span,
            enclosing=self._scope_stack,
            module_path=self.path,
        )
        self.findings.append(
            AmbiguousClaimFinding(
                finding_id=_finding_id(span, field_name),
                field_name=field_name,
                claim_kind=kind,
                disposition=FindingDisposition.REJECT,
                repair_family=family,
                source_span=span,
                abstract_trace=trace,
                value_repr=value_repr,
                message=(
                    f"unqualified {kind.value} field {field_name!r} bound at "
                    f"{span.path}:{span.start_line}"
                ),
            )
        )


def scan_python_source(
    source: str,
    *,
    path: str = "<memory>.py",
) -> tuple[AmbiguousClaimFinding, ...]:
    """Scan one Python source string for claim-shaped ambiguous fields."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise FormalClaimScannerError(f"python source failed to parse: {path}: {exc}") from exc
    visitor = _PythonClaimVisitor(path=_normalize_relpath(path), source=source)
    visitor.visit(tree)
    return _dedupe_findings(visitor.findings)


def scan_json_source(
    source: str,
    *,
    path: str = "<memory>.json",
) -> tuple[AmbiguousClaimFinding, ...]:
    """Scan one JSON document for forbidden/ambiguous claim keys with claim values."""

    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise FormalClaimScannerError(f"json source failed to parse: {path}: {exc}") from exc
    findings: list[AmbiguousClaimFinding] = []
    for key_path, key, value, line_hint in _walk_json_claims(payload):
        kind = classify_field_name(key)
        if kind in {ClaimKind.NOT_A_CLAIM, ClaimKind.TYPED_COMPATIBILITY_ALIAS}:
            continue
        if not _is_claim_value(value, field_name=key):
            continue
        # Best-effort line from raw text search for the key.
        line = _json_key_line(source, key, line_hint)
        excerpt = _excerpt_line(source, line) or f"{key}: {_value_repr(value)}"
        span = SourceSpan(
            path=_normalize_relpath(path),
            start_line=line,
            end_line=line,
            symbol=key_path,
            excerpt=excerpt,
            column=0,
        )
        findings.append(
            AmbiguousClaimFinding(
                finding_id=_finding_id(span, key),
                field_name=key,
                claim_kind=kind,
                disposition=FindingDisposition.REJECT,
                repair_family=repair_family_for_field(key, context=key_path),
                source_span=span,
                abstract_trace=_build_trace(
                    field_name=key,
                    span=span,
                    enclosing=[key_path] if key_path else [],
                    module_path=path,
                ),
                value_repr=_value_repr(value),
                message=f"unqualified JSON claim field {key!r} at {path}:{line}",
            )
        )
    return _dedupe_findings(findings)


def _walk_json_claims(
    value: Any, prefix: str = ""
) -> Iterator[tuple[str, str, Any, int]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_s = str(key)
            path = f"{prefix}.{key_s}" if prefix else key_s
            yield path, key_s, child, 1
            yield from _walk_json_claims(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json_claims(child, f"{prefix}[{index}]")


def _json_key_line(source: str, key: str, default: int) -> int:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:')
    for index, line in enumerate(source.splitlines(), start=1):
        if pattern.search(line):
            return index
    return max(1, int(default))


def scan_path(path: Union[str, Path], *, root: Union[str, Path, None] = None) -> tuple[AmbiguousClaimFinding, ...]:
    """Scan a single file path."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FormalClaimScannerError(f"scan path is not a file: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    rel = _relative_to_root(file_path, root)
    suffix = file_path.suffix.casefold()
    if suffix in {".py", ".pyi"}:
        return scan_python_source(text, path=rel)
    if suffix == ".json":
        return scan_json_source(text, path=rel)
    return ()


def scan_tree(
    root: Union[str, Path],
    *,
    relative_paths: Optional[Sequence[str]] = None,
    allowlist: Optional[AmbiguousClaimAllowlist] = None,
    corpus_entries: Optional[Sequence[Mapping[str, Any]]] = None,
) -> AmbiguousClaimScanReport:
    """Scan a repository tree (or an explicit relative path subset)."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FormalClaimScannerError(f"scan root is not a directory: {root_path}")

    paths = (
        [_normalize_relpath(item) for item in relative_paths]
        if relative_paths is not None
        else list(_iter_scan_paths(root_path))
    )

    findings: list[AmbiguousClaimFinding] = []
    scanned: list[str] = []
    for rel in paths:
        abs_path = root_path / rel
        if not abs_path.is_file():
            continue
        if abs_path.suffix.casefold() not in _SCAN_SUFFIXES:
            continue
        scanned.append(rel)
        findings.extend(scan_path(abs_path, root=root_path))

    bound = bind_corpus_seeds(tuple(findings), corpus_entries or ())
    applied = apply_allowlist(bound, allowlist or AmbiguousClaimAllowlist())
    allowlisted_ids = tuple(
        item.finding_id
        for item in applied
        if item.disposition is FindingDisposition.ALLOWLISTED
    )
    corpus_ids = tuple(
        sorted(
            {
                item.corpus_seed_id
                for item in applied
                if item.corpus_seed_id
            }
        )
    )
    return AmbiguousClaimScanReport(
        findings=applied,
        scanned_paths=tuple(scanned),
        allowlisted_finding_ids=allowlisted_ids,
        corpus_seed_ids_bound=corpus_ids,
    )


def _iter_scan_paths(root: Path) -> Iterator[str]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.casefold() not in _SCAN_SUFFIXES:
            continue
        yield _normalize_relpath(str(path.relative_to(root)))


def _relative_to_root(path: Path, root: Union[str, Path, None]) -> str:
    if root is None:
        return _normalize_relpath(str(path))
    root_path = Path(root).resolve()
    try:
        return _normalize_relpath(str(path.resolve().relative_to(root_path)))
    except ValueError:
        return _normalize_relpath(str(path))


def _dedupe_findings(
    findings: Iterable[AmbiguousClaimFinding],
) -> tuple[AmbiguousClaimFinding, ...]:
    seen: set[str] = set()
    ordered: list[AmbiguousClaimFinding] = []
    for item in findings:
        if item.finding_id in seen:
            continue
        seen.add(item.finding_id)
        ordered.append(item)
    return tuple(ordered)


def load_defect_corpus(path: Union[str, Path]) -> tuple[dict[str, Any], ...]:
    """Load FACP-008 defect corpus JSONL entries."""

    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FormalClaimScannerError(f"defect corpus not found: {corpus_path}")
    entries: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        corpus_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormalClaimScannerError(
                f"defect corpus line {line_no} is not JSON: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise FormalClaimScannerError(
                f"defect corpus line {line_no} must be an object"
            )
        entries.append(dict(payload))
    return tuple(entries)


def findings_for_corpus_entry(
    entry: Mapping[str, Any],
    *,
    repo_root: Union[str, Path, None] = None,
) -> tuple[AmbiguousClaimFinding, ...]:
    """Materialize corpus-bound findings for one seeded defect entry.

    Each source span becomes a finding carrying the corpus repair family and an
    abstract trace derived from the seed metadata. When ``repo_root`` is set and
    the span path exists, the scanner also re-parses the live source so claim
    sites still present in the tree are attached.
    """

    seed_id = str(entry.get("seed_id") or "")
    defect_id = str(entry.get("defect_id") or "")
    family = str(entry.get("family") or "")
    if family not in ROADMAP_DEFECT_FAMILIES:
        # Some inventory families are already roadmap names; fall back carefully.
        raise FormalClaimScannerError(
            f"corpus entry {seed_id or defect_id!r} has unknown family {family!r}"
        )
    spans_raw = entry.get("source_spans") or []
    if not isinstance(spans_raw, Sequence) or isinstance(spans_raw, (str, bytes)):
        raise FormalClaimScannerError(
            f"corpus entry {seed_id!r} source_spans must be a sequence"
        )

    call_flow = entry.get("call_flow_path") or entry.get("call_flow") or []
    flow_labels = [str(item) for item in call_flow] if isinstance(call_flow, Sequence) else []

    findings: list[AmbiguousClaimFinding] = []
    for index, span_payload in enumerate(spans_raw):
        if not isinstance(span_payload, Mapping):
            continue
        span = SourceSpan.from_dict(span_payload)
        field_name = _infer_field_name_from_span(span, family=family, entry=entry)
        kind = classify_field_name(field_name)
        if kind is ClaimKind.TYPED_COMPATIBILITY_ALIAS:
            # Corpus seeds describe defects; never rebadge them as aliases.
            kind = ClaimKind.FORBIDDEN_GENERIC
        if kind is ClaimKind.NOT_A_CLAIM:
            kind = (
                ClaimKind.FORBIDDEN_GENERIC
                if family in {"false_success", "mock_capability"}
                else ClaimKind.AMBIGUOUS_EVIDENCE
            )
            if not field_name:
                field_name = {
                    "false_success": "success",
                    "mock_capability": "available",
                    "pseudo_cid": "cid",
                    "browser_authority": "authorized",
                    "stale_proof": "current",
                    "hermetic_to_live": "production",
                    "missing_recovery": "fallback",
                }.get(family, family)

        steps = [
            AbstractTraceStep(
                kind="seed_span",
                label=f"{span.path}:{span.start_line}-{span.end_line}",
                detail=span.excerpt,
            )
        ]
        for label in flow_labels[:6]:
            steps.append(AbstractTraceStep(kind="call_flow", label=label))
        steps.append(
            AbstractTraceStep(
                kind="repair_family",
                label=family,
                detail=str(entry.get("expected_illegal_promotion") or ""),
            )
        )
        trace = AbstractTrace(
            steps=tuple(steps),
            summary=f"{seed_id or defect_id}:{family}",
        )
        finding = AmbiguousClaimFinding(
            finding_id=f"corpus:{seed_id or defect_id}:{index}:{field_name}",
            field_name=field_name,
            claim_kind=kind,
            disposition=FindingDisposition.CORPUS_BOUND,
            repair_family=family,
            source_span=span,
            abstract_trace=trace,
            value_repr=span.excerpt,
            message=str(entry.get("title") or seed_id or defect_id),
            corpus_seed_id=seed_id,
            corpus_defect_id=defect_id,
            roadmap_seed=bool(entry.get("roadmap_seed", True)),
        )
        findings.append(finding)

        if repo_root is not None:
            abs_path = Path(repo_root) / span.path
            if abs_path.is_file() and abs_path.suffix.casefold() in {".py", ".pyi"}:
                try:
                    live = scan_path(abs_path, root=repo_root)
                except FormalClaimScannerError:
                    live = ()
                for live_finding in live:
                    if live_finding.source_span.overlaps(span):
                        findings.append(
                            AmbiguousClaimFinding(
                                finding_id=f"{live_finding.finding_id}:corpus:{seed_id}",
                                field_name=live_finding.field_name,
                                claim_kind=live_finding.claim_kind,
                                disposition=FindingDisposition.CORPUS_BOUND,
                                repair_family=family,
                                source_span=live_finding.source_span,
                                abstract_trace=AbstractTrace(
                                    steps=live_finding.abstract_trace.steps
                                    + (
                                        AbstractTraceStep(
                                            kind="corpus_bind",
                                            label=seed_id or defect_id,
                                            detail=family,
                                        ),
                                    ),
                                    summary=live_finding.abstract_trace.summary,
                                ),
                                value_repr=live_finding.value_repr,
                                message=live_finding.message,
                                corpus_seed_id=seed_id,
                                corpus_defect_id=defect_id,
                                roadmap_seed=True,
                            )
                        )
    return _dedupe_findings(findings)


def _infer_field_name_from_span(
    span: SourceSpan,
    *,
    family: str,
    entry: Mapping[str, Any],
) -> str:
    excerpt = (span.excerpt or "").casefold()
    title = str(entry.get("title") or "").casefold()
    blob = f"{excerpt} {title} {span.symbol.casefold()}"
    for name in (
        "api_available",
        "success",
        "available",
        "supported",
        "verified",
        "proven",
        "authorized",
        "allowed",
        "current",
        "production",
        "capability",
        "mock",
        "simulation",
        "fallback",
        "cid",
    ):
        if re.search(rf"\b{re.escape(name)}\b", blob):
            return name
    return {
        "false_success": "success",
        "mock_capability": "available",
        "pseudo_cid": "cid",
        "browser_authority": "authorized",
        "stale_proof": "current",
        "hermetic_to_live": "production",
        "missing_recovery": "fallback",
        "import_effect": "success",
        "total_assurance_ladder": "verified",
        "canonicalization_conflict": "cid",
        "secret_flow": "success",
        "license_conflict": "allowed",
        "mutable_dependency": "current",
    }.get(family, "success")


def bind_corpus_seeds(
    findings: Sequence[AmbiguousClaimFinding],
    corpus_entries: Sequence[Mapping[str, Any]],
) -> tuple[AmbiguousClaimFinding, ...]:
    """Attach corpus seed ids to overlapping live findings and append missing seeds."""

    if not corpus_entries:
        return tuple(findings)

    corpus_findings: list[AmbiguousClaimFinding] = []
    for entry in corpus_entries:
        corpus_findings.extend(findings_for_corpus_entry(entry))

    bound: list[AmbiguousClaimFinding] = []
    matched_seed_ids: set[str] = set()
    for finding in findings:
        matched = next(
            (
                corpus_finding
                for corpus_finding in corpus_findings
                if finding.source_span.overlaps(corpus_finding.source_span)
            ),
            None,
        )
        if matched is None:
            bound.append(finding)
            continue
        matched_seed_ids.add(matched.corpus_seed_id)
        bound.append(
            AmbiguousClaimFinding(
                finding_id=finding.finding_id,
                field_name=finding.field_name,
                claim_kind=finding.claim_kind,
                disposition=FindingDisposition.CORPUS_BOUND,
                repair_family=matched.repair_family,
                source_span=finding.source_span,
                abstract_trace=AbstractTrace(
                    steps=finding.abstract_trace.steps
                    + tuple(
                        step
                        for step in matched.abstract_trace.steps
                        if step.kind in {"call_flow", "repair_family"}
                    ),
                    summary=finding.abstract_trace.summary,
                ),
                value_repr=finding.value_repr,
                message=finding.message,
                corpus_seed_id=matched.corpus_seed_id,
                corpus_defect_id=matched.corpus_defect_id,
                roadmap_seed=True,
            )
        )

    # Preserve one representative span per unmatched seed so corpus defects
    # remain visible when live AST re-detection misses them.
    seen_orphan_seeds: set[str] = set()
    for corpus_finding in corpus_findings:
        seed_id = corpus_finding.corpus_seed_id
        if not seed_id or seed_id in matched_seed_ids or seed_id in seen_orphan_seeds:
            continue
        seen_orphan_seeds.add(seed_id)
        bound.append(corpus_finding)
    return _dedupe_findings(bound)


def apply_allowlist(
    findings: Sequence[AmbiguousClaimFinding],
    allowlist: AmbiguousClaimAllowlist,
) -> tuple[AmbiguousClaimFinding, ...]:
    """Apply allowlist suppressions without ever dropping corpus defects."""

    result: list[AmbiguousClaimFinding] = []
    for finding in findings:
        if finding.is_corpus_defect:
            # Hard rule: corpus defects cannot be allowlisted away.
            result.append(finding)
            continue
        entry = allowlist.matching_entry(finding)
        if entry is None:
            result.append(finding)
            continue
        result.append(
            AmbiguousClaimFinding(
                finding_id=finding.finding_id,
                field_name=finding.field_name,
                claim_kind=finding.claim_kind,
                disposition=FindingDisposition.ALLOWLISTED,
                repair_family=finding.repair_family,
                source_span=finding.source_span,
                abstract_trace=finding.abstract_trace,
                value_repr=finding.value_repr,
                message=finding.message,
                corpus_seed_id=finding.corpus_seed_id,
                corpus_defect_id=finding.corpus_defect_id,
                roadmap_seed=finding.roadmap_seed,
                allowlist_entry_id=entry.entry_id,
            )
        )
    return tuple(result)


def scan_seeded_corpus(
    *,
    corpus_path: Union[str, Path],
    repo_root: Union[str, Path, None] = None,
    seed_ids: Optional[Sequence[str]] = None,
    allowlist: Optional[AmbiguousClaimAllowlist] = None,
) -> AmbiguousClaimScanReport:
    """Emit corpus-bound findings for seeded defect spans.

    This is the primary acceptance path for FACP-019: every selected seed yields
    source spans with abstract traces and repair families. An allowlist may be
    supplied but cannot suppress those corpus findings.
    """

    entries = load_defect_corpus(corpus_path)
    if seed_ids is not None:
        wanted = {str(item) for item in seed_ids}
        entries = tuple(
            entry for entry in entries if str(entry.get("seed_id") or "") in wanted
        )

    findings: list[AmbiguousClaimFinding] = []
    for entry in entries:
        findings.extend(
            findings_for_corpus_entry(entry, repo_root=repo_root)
        )
    applied = apply_allowlist(
        _dedupe_findings(findings), allowlist or AmbiguousClaimAllowlist()
    )
    # Corpus findings must remain reject/corpus_bound after allowlist.
    for item in applied:
        if item.is_corpus_defect and item.disposition is FindingDisposition.ALLOWLISTED:
            raise FormalClaimScannerError(
                "allowlist illegally suppressed corpus defect "
                f"{item.corpus_seed_id or item.finding_id}"
            )
    seed_ids_bound = tuple(
        sorted({item.corpus_seed_id for item in applied if item.corpus_seed_id})
    )
    paths = tuple(
        sorted({item.source_span.path for item in applied})
    )
    return AmbiguousClaimScanReport(
        findings=applied,
        scanned_paths=paths,
        allowlisted_finding_ids=tuple(
            item.finding_id
            for item in applied
            if item.disposition is FindingDisposition.ALLOWLISTED
        ),
        corpus_seed_ids_bound=seed_ids_bound,
    )


def describe_compatibility_alias(name: str) -> Optional[dict[str, Any]]:
    """Return metadata when ``name`` is a typed compatibility alias."""

    if classify_field_name(name) is not ClaimKind.TYPED_COMPATIBILITY_ALIAS:
        return None
    return {
        "name": name,
        "claim_kind": ClaimKind.TYPED_COMPATIBILITY_ALIAS.value,
        "disposition": FindingDisposition.COMPATIBILITY_ALIAS.value,
        "vocab_schema": VOCAB_SCHEMA,
        "ambiguous": False,
    }


__all__ = [
    "AMBIGUOUS_EVIDENCE_TOKENS",
    "AbstractTrace",
    "AbstractTraceStep",
    "AllowlistEntry",
    "AmbiguousClaimAllowlist",
    "AmbiguousClaimFinding",
    "AmbiguousClaimScanReport",
    "BUNDLE",
    "ClaimKind",
    "EVIDENCE_SCHEMA",
    "FORBIDDEN_GENERIC_FIELDS",
    "FindingDisposition",
    "FormalClaimScannerError",
    "GOAL_ID",
    "ROADMAP_DEFECT_FAMILIES",
    "SCHEMA",
    "SCANNER_VERSION",
    "SourceSpan",
    "TASK_ID",
    "TYPED_COMPATIBILITY_ALIASES",
    "VOCAB_SCHEMA",
    "apply_allowlist",
    "bind_corpus_seeds",
    "classify_field_name",
    "describe_compatibility_alias",
    "findings_for_corpus_entry",
    "load_defect_corpus",
    "repair_family_for_field",
    "scan_json_source",
    "scan_path",
    "scan_python_source",
    "scan_seeded_corpus",
    "scan_tree",
]
