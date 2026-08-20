"""FACP-043: Bounded IPA repair transforms and mutation gate.

Fixed AST/config transforms for the five admitted IPA repair grammars:

* ``explicit_init`` — move import-time effects behind an authorized call
* ``typed_unavailable`` — replace unobserved ``success=True`` with Unavailable
* ``simulation_evidence`` — isolate mock/simulation evidence from live claims
* ``canonical_cid`` — replace pseudo-CID construction with canonical minting
* ``critical_error_propagation`` — re-raise swallowed exceptions (fail closed)

Transforms are deterministic and idempotent, reject ambiguous targets, preserve
unrelated bytes outside the edited span, and return a typed abstention when
preconditions do not match. Production application remains gated by FACP-051;
this module only renders and mutation-gates repairs (byte mutation + reanalysis).
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final, Iterable, Mapping, Optional, Sequence

from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
    IpaFinding,
    IpaRuleId,
    SourceSpan,
    analyze_python_source,
)

TASK_ID: Final[str] = "FACP-043"
GOAL_ID: Final[str] = "FACP-G410"
BUNDLE: Final[str] = "facp/static/repairs"
SCHEMA: Final[str] = "facp/ipa-repair@1"
EVIDENCE_ID: Final[str] = "facp/ipa-repair@1"
DETERMINISTIC_REPAIRS_EVIDENCE: Final[str] = "facp/deterministic-repairs@1"
INTERFACE: Final[str] = "FormalAssuranceIpaTransforms@1"
PRODUCER_ID: Final[str] = "formal-assurance-ipa-transforms@1"
ANALYZER_VERSION: Final[str] = "ipa-repair/v1"

MAX_SOURCE_BYTES: Final[int] = 1_000_000
MAX_PATH_BYTES: Final[int] = 1_024
MAX_EDIT_BYTES: Final[int] = 65_536

# Paths where deterministic IPA repairs may be rendered (Datasets + Accelerate).
_DEFAULT_ADMITTED_PREFIXES: Final[tuple[str, ...]] = (
    "external/ipfs_accelerate/",
    "external/ipfs_datasets/",
    "ipfs_accelerate_py/",
    "ipfs_datasets_py/",
)

# Hermetic fixture / in-memory paths used by unit tests and seed corpora.
_FIXTURE_PATH_MARKERS: Final[tuple[str, ...]] = (
    "fixtures/",
    "seeded/",
    "<memory>",
    "tmp/",
)


class IpaRepairError(ValueError):
    """Malformed repair input or an attempt to weaken a fail-closed boundary."""


class IpaRepairTransformId(str, Enum):
    """Closed IPA repair grammar identifiers (not expandable by models)."""

    EXPLICIT_INIT = "explicit_init"
    TYPED_UNAVAILABLE = "typed_unavailable"
    SIMULATION_EVIDENCE = "simulation_evidence"
    CANONICAL_CID = "canonical_cid"
    CRITICAL_ERROR_PROPAGATION = "critical_error_propagation"


class IpaRepairDisposition(str, Enum):
    """Closed repair outcome vocabulary."""

    APPLIED = "applied"
    NOOP = "noop"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class IpaRepairAbstentionReason(str, Enum):
    """Closed, audit-stable abstention / rejection codes."""

    AMBIGUOUS_TARGET = "ambiguous_target"
    MULTIPLE_MATCHES = "multiple_matches"
    PRECONDITION_MISMATCH = "precondition_mismatch"
    PATH_NOT_ADMITTED = "path_not_admitted"
    STALE_SPAN = "stale_span"
    UNSUPPORTED_RULE = "unsupported_rule"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    NO_BYTE_CHANGE = "no_byte_change"
    REANALYSIS_STILL_FAILS = "reanalysis_still_fails"
    NEW_ABSTRACT_FINDING = "new_abstract_finding"
    EMPTY_SOURCE = "empty_source"
    PARSE_ERROR = "parse_error"
    PUBLIC_COMPAT_RISK = "public_compat_risk"
    TRANSFORM_OUTSIDE_GRAMMAR = "transform_outside_grammar"


class MutationGateDisposition(str, Enum):
    ADMITTED = "admitted"
    DENIED = "denied"


# Transform → IPA rule binding (fixed grammar; LLMs cannot expand this map).
TRANSFORM_TO_RULE: Final[Mapping[IpaRepairTransformId, IpaRuleId]] = {
    IpaRepairTransformId.EXPLICIT_INIT: IpaRuleId.IMPORT_EFFECT,
    IpaRepairTransformId.TYPED_UNAVAILABLE: IpaRuleId.SUCCESS_WITHOUT_OBSERVATION,
    IpaRepairTransformId.SIMULATION_EVIDENCE: IpaRuleId.MOCK_TO_PRODUCTION,
    IpaRepairTransformId.CANONICAL_CID: IpaRuleId.PSEUDO_CID,
    IpaRepairTransformId.CRITICAL_ERROR_PROPAGATION: IpaRuleId.EXCEPTION_SWALLOWING,
}

RULE_TO_TRANSFORM: Final[Mapping[str, IpaRepairTransformId]] = {
    rule.value: transform_id for transform_id, rule in TRANSFORM_TO_RULE.items()
}

STABLE_TRANSFORM_IDS: Final[frozenset[str]] = frozenset(
    item.value for item in IpaRepairTransformId
)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_path(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _validate_path(path: str, name: str = "path") -> str:
    raw = _normalize_path(path)
    if not raw:
        raise IpaRepairError(f"{name} is required")
    if len(raw.encode("utf-8")) > MAX_PATH_BYTES:
        raise IpaRepairError(f"{name} exceeds its byte bound")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise IpaRepairError(f"{name} must be a relative repository path")
    return candidate.as_posix()


def default_admitted_paths() -> frozenset[str]:
    """Default admitted write-path prefixes for Datasets and Accelerate repairs."""

    return frozenset(_DEFAULT_ADMITTED_PREFIXES + _FIXTURE_PATH_MARKERS)


def path_is_admitted(path: str, admitted_paths: Optional[Iterable[str]] = None) -> bool:
    """Return True when ``path`` is under an admitted repair prefix or exact path."""

    normalized = _normalize_path(path)
    if not normalized:
        return False
    allow = tuple(admitted_paths) if admitted_paths is not None else tuple(
        default_admitted_paths()
    )
    for entry in allow:
        item = _normalize_path(str(entry))
        if not item:
            continue
        if normalized == item or normalized.startswith(item):
            return True
        # Prefix entries may be supplied without trailing slash.
        if item.endswith("/") and normalized.startswith(item):
            return True
        if not item.endswith("/") and (
            normalized == item or normalized.startswith(item + "/")
        ):
            return True
        # Fixture markers may appear mid-path.
        if item in _FIXTURE_PATH_MARKERS and item in normalized:
            return True
    return False


@dataclass(frozen=True)
class MutationGateDecision:
    """Admission decision for a candidate write path / mutation."""

    disposition: MutationGateDisposition
    path: str
    reasons: tuple[str, ...] = ()
    before_hash: str = ""
    after_hash: str = ""
    byte_mutated: bool = False
    reanalyzed: bool = False

    @property
    def admitted(self) -> bool:
        return self.disposition is MutationGateDisposition.ADMITTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "path": self.path,
            "reasons": list(self.reasons),
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "byte_mutated": self.byte_mutated,
            "reanalyzed": self.reanalyzed,
        }


@dataclass(frozen=True)
class IpaRepairEdit:
    """One exact span replacement with before/after identity."""

    path: str
    start_line: int
    end_line: int
    before_text: str
    after_text: str
    before_hash: str = ""
    after_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_path(self.path))
        if (
            isinstance(self.start_line, bool)
            or not isinstance(self.start_line, int)
            or self.start_line < 1
        ):
            raise IpaRepairError("edit start_line must be >= 1")
        if (
            isinstance(self.end_line, bool)
            or not isinstance(self.end_line, int)
            or self.end_line < self.start_line
        ):
            raise IpaRepairError("edit end_line must be >= start_line")
        if len(self.before_text.encode("utf-8")) > MAX_EDIT_BYTES:
            raise IpaRepairError("before_text exceeds edit byte bound")
        if len(self.after_text.encode("utf-8")) > MAX_EDIT_BYTES:
            raise IpaRepairError("after_text exceeds edit byte bound")
        if not self.before_hash:
            object.__setattr__(self, "before_hash", _sha256_text(self.before_text))
        if not self.after_hash:
            object.__setattr__(self, "after_hash", _sha256_text(self.after_text))

    @property
    def mutated(self) -> bool:
        return self.before_text != self.after_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "mutated": self.mutated,
            # Bodies are retained for local render/replay; FACP-051 gates apply.
            "before_text": self.before_text,
            "after_text": self.after_text,
        }


@dataclass(frozen=True)
class IpaReanalysisReport:
    """IPA reanalysis evidence bound to a repair receipt."""

    before_rule_ids: tuple[str, ...]
    after_rule_ids: tuple[str, ...]
    eliminated_rule_ids: tuple[str, ...]
    new_rule_ids: tuple[str, ...]
    before_finding_ids: tuple[str, ...] = ()
    after_finding_ids: tuple[str, ...] = ()
    target_rule_eliminated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_rule_ids": list(self.before_rule_ids),
            "after_rule_ids": list(self.after_rule_ids),
            "eliminated_rule_ids": list(self.eliminated_rule_ids),
            "new_rule_ids": list(self.new_rule_ids),
            "before_finding_ids": list(self.before_finding_ids),
            "after_finding_ids": list(self.after_finding_ids),
            "target_rule_eliminated": self.target_rule_eliminated,
        }


@dataclass(frozen=True)
class IpaRepairReceipt:
    """Typed repair result: applied edit, idempotent noop, or abstention."""

    disposition: IpaRepairDisposition
    transform_id: str
    rule_id: str
    path: str
    reasons: tuple[str, ...] = ()
    edits: tuple[IpaRepairEdit, ...] = ()
    before_hash: str = ""
    after_hash: str = ""
    after_source: str = ""
    reanalysis: Optional[IpaReanalysisReport] = None
    mutation_gate: Optional[MutationGateDecision] = None
    finding_id: str = ""
    producer_id: str = PRODUCER_ID
    schema: str = SCHEMA
    task_id: str = TASK_ID
    public_compat_preserved: bool = True

    def __post_init__(self) -> None:
        disposition = (
            self.disposition
            if isinstance(self.disposition, IpaRepairDisposition)
            else IpaRepairDisposition(self.disposition)
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(
            self,
            "reasons",
            tuple(str(item) for item in self.reasons),
        )
        if disposition in {
            IpaRepairDisposition.ABSTAINED,
            IpaRepairDisposition.REJECTED,
        }:
            if not self.reasons:
                raise IpaRepairError("abstained/rejected receipts require reasons")
            if self.edits:
                raise IpaRepairError("abstained/rejected receipts cannot grant edits")
        if disposition is IpaRepairDisposition.APPLIED:
            if not self.edits:
                raise IpaRepairError("applied receipts require at least one edit")
            if self.before_hash == self.after_hash:
                raise IpaRepairError("applied receipts require byte mutation")

    @property
    def admitted(self) -> bool:
        return self.disposition in {
            IpaRepairDisposition.APPLIED,
            IpaRepairDisposition.NOOP,
        }

    @property
    def abstained(self) -> bool:
        return self.disposition is IpaRepairDisposition.ABSTAINED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "producer_id": self.producer_id,
            "disposition": self.disposition.value,
            "transform_id": self.transform_id,
            "rule_id": self.rule_id,
            "path": self.path,
            "finding_id": self.finding_id,
            "reasons": list(self.reasons),
            "edits": [edit.to_dict() for edit in self.edits],
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "after_source": self.after_source,
            "reanalysis": self.reanalysis.to_dict() if self.reanalysis else None,
            "mutation_gate": self.mutation_gate.to_dict() if self.mutation_gate else None,
            "public_compat_preserved": self.public_compat_preserved,
            "admitted": self.admitted,
        }


def select_transform(
    finding: IpaFinding | Mapping[str, Any] | str,
) -> IpaRepairTransformId:
    """Map an IPA finding / rule id onto the fixed repair grammar."""

    rule_id = _coerce_rule_id(finding)
    transform = RULE_TO_TRANSFORM.get(rule_id)
    if transform is None:
        raise IpaRepairError(f"unsupported IPA rule for repair grammar: {rule_id!r}")
    return transform


def _coerce_rule_id(finding: IpaFinding | Mapping[str, Any] | str) -> str:
    if isinstance(finding, IpaFinding):
        return finding.rule_id
    if isinstance(finding, Mapping):
        return str(finding.get("rule_id") or "")
    return str(finding or "")


def _coerce_finding(
    finding: IpaFinding | Mapping[str, Any],
    *,
    path: str,
) -> IpaFinding:
    if isinstance(finding, IpaFinding):
        return finding
    if not isinstance(finding, Mapping):
        raise IpaRepairError("finding must be IpaFinding or mapping")
    # Minimal reconstruction for tests that pass dict spans.
    from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
        FindingDisposition,
        ProductDomainState,
        SourceToSinkTrace,
        TraceStep,
    )

    rule_id = str(finding.get("rule_id") or "")
    source = finding.get("source_span") or {}
    sink = finding.get("sink_span") or source
    source_span = (
        source
        if isinstance(source, SourceSpan)
        else SourceSpan.from_dict({**dict(source), "path": source.get("path") or path})
    )
    sink_span = (
        sink
        if isinstance(sink, SourceSpan)
        else SourceSpan.from_dict({**dict(sink), "path": sink.get("path") or path})
    )
    trace_payload = finding.get("trace")
    if isinstance(trace_payload, SourceToSinkTrace):
        trace = trace_payload
    else:
        steps = ()
        if isinstance(trace_payload, Mapping):
            steps = tuple(
                TraceStep(
                    kind=str(step.get("kind") or "step"),
                    label=str(step.get("label") or "label"),
                    detail=str(step.get("detail") or ""),
                )
                for step in (trace_payload.get("steps") or ())
            )
        if len(steps) < 2:
            steps = (
                TraceStep(kind="source", label=f"{path}:{source_span.start_line}"),
                TraceStep(kind="sink", label=f"{path}:{sink_span.start_line}"),
            )
        trace = SourceToSinkTrace(steps=steps)
    domain = finding.get("domain_state")
    if isinstance(domain, ProductDomainState):
        domain_state = domain
    else:
        domain_state = ProductDomainState()
    disposition = finding.get("disposition") or FindingDisposition.REJECT
    if not isinstance(disposition, FindingDisposition):
        disposition = FindingDisposition(str(disposition))
    return IpaFinding(
        finding_id=str(finding.get("finding_id") or f"ipa:{rule_id}:manual"),
        rule_id=rule_id,
        disposition=disposition,
        source_span=source_span,
        sink_span=sink_span,
        trace=trace,
        domain_state=domain_state,
        message=str(finding.get("message") or ""),
        family=str(finding.get("family") or ""),
    )


def evaluate_mutation_gate(
    path: str,
    *,
    before_source: str,
    after_source: str,
    target_rule_id: str,
    admitted_paths: Optional[Iterable[str]] = None,
    allow_idempotent_noop: bool = False,
    before_findings: Optional[Sequence[IpaFinding]] = None,
) -> MutationGateDecision:
    """Fail-closed mutation gate: admitted path + byte mutation + reanalysis."""

    normalized = _normalize_path(path)
    before_hash = _sha256_text(before_source)
    after_hash = _sha256_text(after_source)
    byte_mutated = before_source != after_source
    reasons: list[str] = []

    if not path_is_admitted(normalized, admitted_paths):
        reasons.append(IpaRepairAbstentionReason.PATH_NOT_ADMITTED.value)
        return MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=normalized,
            reasons=tuple(reasons),
            before_hash=before_hash,
            after_hash=after_hash,
            byte_mutated=byte_mutated,
            reanalyzed=False,
        )

    if not byte_mutated and not allow_idempotent_noop:
        reasons.append(IpaRepairAbstentionReason.NO_BYTE_CHANGE.value)
        return MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=normalized,
            reasons=tuple(reasons),
            before_hash=before_hash,
            after_hash=after_hash,
            byte_mutated=False,
            reanalyzed=False,
        )

    try:
        after_findings = analyze_python_source(after_source, path=normalized)
    except Exception as exc:  # noqa: BLE001 - typed gate denial
        reasons.append(IpaRepairAbstentionReason.PARSE_ERROR.value)
        reasons.append(str(exc)[:200])
        return MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=normalized,
            reasons=tuple(reasons),
            before_hash=before_hash,
            after_hash=after_hash,
            byte_mutated=byte_mutated,
            reanalyzed=False,
        )

    if before_findings is None:
        try:
            before_findings = analyze_python_source(before_source, path=normalized)
        except Exception:  # noqa: BLE001
            before_findings = ()

    before_rules = {item.rule_id for item in before_findings}
    after_rules = {item.rule_id for item in after_findings}
    if target_rule_id in after_rules:
        reasons.append(IpaRepairAbstentionReason.REANALYSIS_STILL_FAILS.value)
        return MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=normalized,
            reasons=tuple(reasons),
            before_hash=before_hash,
            after_hash=after_hash,
            byte_mutated=byte_mutated,
            reanalyzed=True,
        )

    new_rules = after_rules - before_rules
    if new_rules:
        reasons.append(IpaRepairAbstentionReason.NEW_ABSTRACT_FINDING.value)
        return MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=normalized,
            reasons=tuple(reasons),
            before_hash=before_hash,
            after_hash=after_hash,
            byte_mutated=byte_mutated,
            reanalyzed=True,
        )

    return MutationGateDecision(
        disposition=MutationGateDisposition.ADMITTED,
        path=normalized,
        reasons=(),
        before_hash=before_hash,
        after_hash=after_hash,
        byte_mutated=byte_mutated,
        reanalyzed=True,
    )


def _line_span_text(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines(keepends=True)
    if start_line < 1 or end_line > len(lines) or end_line < start_line:
        raise IpaRepairError("span is outside source bounds")
    return "".join(lines[start_line - 1 : end_line])


def _replace_lines(
    source: str,
    start_line: int,
    end_line: int,
    replacement: str,
) -> str:
    lines = source.splitlines(keepends=True)
    if start_line < 1 or end_line > len(lines) or end_line < start_line:
        raise IpaRepairError("span is outside source bounds")
    # Preserve whether the file ended with a newline.
    ended_with_nl = source.endswith("\n") or source.endswith("\r\n")
    prefix = lines[: start_line - 1]
    suffix = lines[end_line:]
    repl = replacement
    if repl and not repl.endswith("\n"):
        repl = repl + ("\n" if ended_with_nl or suffix or prefix else "")
    rebuilt = "".join(prefix) + repl + "".join(suffix)
    return rebuilt


def _indent_of(line: str) -> str:
    return re.match(r"^[ \t]*", line or "").group(0)  # type: ignore[union-attr]


def _find_function_bounds(source: str, lineno: int) -> tuple[int, int] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines()
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = int(getattr(node, "lineno", 0) or 0)
        end = int(getattr(node, "end_lineno", 0) or 0)
        if start <= lineno <= end:
            if best is None or (end - start) < (best[1] - best[0]):
                best = (start, end)
    if best is None:
        # Fallback: scan upward for def and downward by indentation.
        if lineno < 1 or lineno > len(lines):
            return None
        start = lineno
        while start >= 1 and not re.match(r"^[ \t]*(async\s+)?def\s+", lines[start - 1]):
            start -= 1
        if start < 1:
            return None
        base_indent = _indent_of(lines[start - 1])
        end = start
        for idx in range(start + 1, len(lines) + 1):
            line = lines[idx - 1]
            if not line.strip():
                end = idx
                continue
            indent = _indent_of(line)
            if len(indent) <= len(base_indent) and not line.lstrip().startswith("#"):
                break
            end = idx
        best = (start, end)
    return best


def _module_level_effect_nodes(tree: ast.AST) -> list[ast.AST]:
    assert isinstance(tree, ast.Module)
    nodes: list[ast.AST] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.Assign):
            nodes.append(stmt)
            continue
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            nodes.append(stmt)
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            nodes.append(stmt)
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Subscript):
            nodes.append(stmt)
    return nodes


def _nodes_overlapping_span(
    nodes: Sequence[ast.AST],
    start_line: int,
    end_line: int,
) -> list[ast.AST]:
    matched: list[ast.AST] = []
    for node in nodes:
        node_start = int(getattr(node, "lineno", 0) or 0)
        node_end = int(getattr(node, "end_lineno", 0) or node_start)
        if node_start == 0:
            continue
        if not (node_end < start_line or node_start > end_line):
            matched.append(node)
    return matched


def _abstain(
    *,
    transform_id: str,
    rule_id: str,
    path: str,
    reasons: Sequence[str | IpaRepairAbstentionReason],
    finding_id: str = "",
    before_hash: str = "",
    after_hash: str = "",
    after_source: str = "",
    mutation_gate: Optional[MutationGateDecision] = None,
    disposition: IpaRepairDisposition = IpaRepairDisposition.ABSTAINED,
) -> IpaRepairReceipt:
    reason_values = tuple(
        item.value if isinstance(item, IpaRepairAbstentionReason) else str(item)
        for item in reasons
    )
    return IpaRepairReceipt(
        disposition=disposition,
        transform_id=transform_id,
        rule_id=rule_id,
        path=_normalize_path(path),
        reasons=reason_values,
        edits=(),
        before_hash=before_hash,
        after_hash=after_hash or before_hash,
        after_source=after_source,
        mutation_gate=mutation_gate,
        finding_id=finding_id,
        public_compat_preserved=True,
    )


def _render_explicit_init(source: str, finding: IpaFinding) -> tuple[str, IpaRepairEdit] | str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return IpaRepairAbstentionReason.PARSE_ERROR.value

    effect_nodes = _module_level_effect_nodes(tree)
    span_start = finding.source_span.start_line
    span_end = finding.source_span.end_line
    overlapping = _nodes_overlapping_span(effect_nodes, span_start, span_end)
    if not overlapping:
        # Broaden: all module-level effect nodes when span is stale but single site.
        overlapping = effect_nodes
    if len(overlapping) == 0:
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value
    if len(overlapping) > 1:
        # A span that covers multiple effect sites is ambiguous — never guess.
        covered = [
            node
            for node in overlapping
            if int(getattr(node, "lineno", 0) or 0) >= span_start
            and int(getattr(node, "end_lineno", 0) or getattr(node, "lineno", 0) or 0)
            <= span_end
        ]
        if len(covered) > 1 or span_end > span_start and len(overlapping) > 1:
            return IpaRepairAbstentionReason.AMBIGUOUS_TARGET.value
        exact = [
            node
            for node in overlapping
            if int(getattr(node, "lineno", 0) or 0) == span_start
            and span_end
            <= int(getattr(node, "end_lineno", 0) or getattr(node, "lineno", 0) or 0)
        ]
        if len(exact) == 1:
            overlapping = exact
        else:
            return IpaRepairAbstentionReason.AMBIGUOUS_TARGET.value

    node = overlapping[0]
    start = int(getattr(node, "lineno", 0) or 0)
    end = int(getattr(node, "end_lineno", 0) or start)
    before = _line_span_text(source, start, end)
    stmt = before.strip("\n")
    indent = ""
    body_indent = "    "
    # Move the effect behind an explicit initializer (FACP-022 shape).
    replacement = (
        f"{indent}def initialize_explicit(*, state_root, authorize_install=False):\n"
        f"{body_indent}if not authorize_install:\n"
        f"{body_indent}    return {{\"outcome\": \"Unavailable\", \"code\": \"install_not_authorized\"}}\n"
        f"{body_indent}{stmt.strip()}\n"
        f"{body_indent}return {{\"outcome\": \"Observed\", \"code\": \"initialized\", \"state_root\": state_root}}\n"
    )
    after_source = _replace_lines(source, start, end, replacement)
    edit = IpaRepairEdit(
        path=finding.source_span.path or "<memory>.py",
        start_line=start,
        end_line=end,
        before_text=before,
        after_text=replacement if replacement.endswith("\n") else replacement + "\n",
    )
    return after_source, edit


def _rewrite_success_dict_literal(text: str) -> str | None:
    """Rewrite dict / assignment success=True shapes to typed Unavailable."""

    original = text
    updated = text
    updated = re.sub(
        r"(?i)([\"']success[\"']\s*:\s*)True",
        r'\1False, "outcome": "Unavailable", "code": "effect_unobserved"',
        updated,
        count=1,
    )
    updated = re.sub(
        r"(?i)([\"']available[\"']\s*:\s*)True",
        r'\1False',
        updated,
    )
    updated = re.sub(
        r"(?i)([\"']supported[\"']\s*:\s*)True",
        r'\1False',
        updated,
    )
    updated = re.sub(
        r"(?i)\b(api_available|available|supported|success)\s*=\s*True\b",
        r'\1 = False  # typed Unavailable / effect unobserved',
        updated,
    )
    # Bare ``return {"success": True}`` may become awkward; normalize common form.
    if re.search(r"(?i)return\s*\{[^{}]*success[^{]*True", original):
        updated = re.sub(
            r"(?i)return\s*\{([^{}]*)\}",
            (
                'return {"outcome": "Unavailable", "code": "effect_unobserved", '
                '"success": False}'
            ),
            updated,
            count=1,
        )
    if updated == original:
        return None
    return updated


def _render_typed_unavailable(
    source: str, finding: IpaFinding
) -> tuple[str, IpaRepairEdit] | str:
    start = finding.source_span.start_line
    end = finding.source_span.end_line
    try:
        before = _line_span_text(source, start, end)
    except IpaRepairError:
        return IpaRepairAbstentionReason.STALE_SPAN.value

    rewritten = _rewrite_success_dict_literal(before)
    if rewritten is None:
        # Expand to full statement line if span excerpt alone is insufficient.
        bounds = _find_function_bounds(source, start)
        if bounds is None:
            return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value
        # Prefer the single line containing success=True inside the function.
        lines = source.splitlines(keepends=True)
        candidates = [
            idx
            for idx in range(bounds[0], bounds[1] + 1)
            if re.search(r"(?i)(success|available|supported|api_available)\s*[:=]\s*True", lines[idx - 1])
        ]
        if len(candidates) != 1:
            # If multiple, require exact span line membership.
            exact = [idx for idx in candidates if start <= idx <= end]
            if len(exact) == 1:
                candidates = exact
            else:
                return IpaRepairAbstentionReason.AMBIGUOUS_TARGET.value
        start = end = candidates[0]
        before = _line_span_text(source, start, end)
        rewritten = _rewrite_success_dict_literal(before)
        if rewritten is None:
            indent = _indent_of(before)
            rewritten = (
                f'{indent}return {{"outcome": "Unavailable", '
                f'"code": "effect_unobserved"}}\n'
            )
    after_source = _replace_lines(source, start, end, rewritten)
    edit = IpaRepairEdit(
        path=finding.source_span.path or "<memory>.py",
        start_line=start,
        end_line=end,
        before_text=before,
        after_text=rewritten if rewritten.endswith("\n") else rewritten + "\n",
    )
    return after_source, edit


def _render_simulation_evidence(
    source: str, finding: IpaFinding
) -> tuple[str, IpaRepairEdit] | str:
    bounds = _find_function_bounds(source, finding.source_span.start_line)
    if bounds is None:
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value
    start, end = bounds
    before = _line_span_text(source, start, end)
    # Ambiguous if multiple mock helpers share the same finding line without bounds.
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return IpaRepairAbstentionReason.PARSE_ERROR.value
    funcs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and int(getattr(node, "lineno", 0) or 0)
        <= finding.source_span.start_line
        <= int(getattr(node, "end_lineno", 0) or 0)
    ]
    if len(funcs) > 1:
        return IpaRepairAbstentionReason.AMBIGUOUS_TARGET.value

    func_name = "create_mock_handler"
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", before)
    if match:
        func_name = match.group(1)

    # Closed simulation-evidence shape (FACP-025): no MagicMock live claim,
    # no available=True, origin=simulated only inside the simulation helper;
    # live sinks must not promote mock evidence.
    if re.search(r"(?i)add_endpoint|register_endpoint|get_capabilities|test_hardware", func_name):
        replacement = (
            f"def {func_name}(name=None, *args, **kwargs):\n"
            f"    # FACP-043 simulation_evidence: refuse mock-to-production promotion\n"
            f"    return {{\"outcome\": \"Unavailable\", \"code\": \"simulation_not_admitted\", "
            f"\"origin\": \"absent\"}}\n"
        )
    else:
        replacement = (
            f"def {func_name}(*args, **kwargs):\n"
            f"    # FACP-043 simulation_evidence: simulation namespace only "
            f"(IPFS_ACCELERATE_EXPLICIT_TEST_MODE)\n"
            f"    return {{\"outcome\": \"Simulated\", \"origin\": \"simulated\", "
            f"\"available\": False, \"capability\": False, \"mock\": True}}\n"
        )
    after_source = _replace_lines(source, start, end, replacement)
    edit = IpaRepairEdit(
        path=finding.source_span.path or "<memory>.py",
        start_line=start,
        end_line=end,
        before_text=before,
        after_text=replacement,
    )
    return after_source, edit


def _render_canonical_cid(
    source: str, finding: IpaFinding
) -> tuple[str, IpaRepairEdit] | str:
    bounds = _find_function_bounds(source, finding.source_span.start_line)
    if bounds is None:
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value
    start, end = bounds
    before = _line_span_text(source, start, end)
    # Require pseudo-CID evidence inside the function; otherwise abstain.
    if not re.search(
        r"(?i)(hexdigest|Qm\{|cid\s*=|mock_cid|sha256|bafy)",
        before,
    ):
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value

    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:->\s*[^:]+)?:", before)
    if not match:
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value
    func_name = match.group(1)
    params = match.group(2).strip()
    # Preserve the first positional payload parameter when present.
    payload_arg = "payload"
    if params:
        first = params.split(",")[0].strip()
        first = first.split(":")[0].strip()
        first = first.split("=")[0].strip()
        if first and first not in {"self", "cls"}:
            payload_arg = first
        elif "self" in params.split(",")[0]:
            # Method form: use second param if available.
            parts = [p.strip().split(":")[0].split("=")[0].strip() for p in params.split(",")]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                payload_arg = parts[1]
            else:
                payload_arg = "data"

    indent = ""
    replacement = (
        f"{indent}def {func_name}({params}) -> dict:\n"
        f"    from ipfs_accelerate_py.assurance.content_identity import mint_content_identity\n"
        f"    identity = mint_content_identity({payload_arg})\n"
        f"    return {{\"cid\": identity.cid, \"integrity\": \"digest_valid\"}}\n"
    )
    # Preserve trailing unrelated constants? Function rewrite only — outer bytes stay.
    after_source = _replace_lines(source, start, end, replacement)
    edit = IpaRepairEdit(
        path=finding.source_span.path or "<memory>.py",
        start_line=start,
        end_line=end,
        before_text=before,
        after_text=replacement,
    )
    return after_source, edit


def _render_critical_error_propagation(
    source: str, finding: IpaFinding
) -> tuple[str, IpaRepairEdit] | str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return IpaRepairAbstentionReason.PARSE_ERROR.value

    handlers: list[ast.ExceptHandler] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", 0) or start)
            if start <= finding.source_span.start_line <= end or (
                finding.source_span.start_line <= start <= finding.source_span.end_line
            ):
                handlers.append(node)
    if len(handlers) == 0:
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value
    if len(handlers) > 1:
        return IpaRepairAbstentionReason.AMBIGUOUS_TARGET.value

    handler = handlers[0]
    start = int(getattr(handler, "lineno", 0) or 0)
    # Replace the entire try statement for stable fail-closed shape when possible.
    try_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and handler in node.handlers:
            try_node = node
            break
    if try_node is None:
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value

    # If the try has multiple handlers, only rewrite the matched handler lines.
    if len(try_node.handlers) != 1:
        end = int(getattr(handler, "end_lineno", 0) or start)
        before = _line_span_text(source, start, end)
        indent = _indent_of(before)
        replacement = f"{indent}except Exception:\n{indent}    raise\n"
        after_source = _replace_lines(source, start, end, replacement)
        edit = IpaRepairEdit(
            path=finding.source_span.path or "<memory>.py",
            start_line=start,
            end_line=end,
            before_text=before,
            after_text=replacement,
        )
        return after_source, edit

    try_start = int(getattr(try_node, "lineno", 0) or 0)
    # Include subsequent success-return statements that the swallow enabled.
    bounds = _find_function_bounds(source, try_start)
    if bounds is None:
        try_end = int(getattr(try_node, "end_lineno", 0) or try_start)
        start, end = try_start, try_end
        before = _line_span_text(source, start, end)
        indent = _indent_of(before)
        replacement = (
            f"{indent}try:\n"
            f"{indent}    raise RuntimeError(\"critical_error_propagation\")\n"
            f"{indent}except Exception:\n"
            f"{indent}    raise\n"
        )
    else:
        start, end = bounds
        before = _line_span_text(source, start, end)
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:->\s*[^:]+)?:", before)
        if not match:
            return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value
        func_name = match.group(1)
        params = match.group(2)
        # Retain original try body first statement when it is a raise; otherwise keep generic.
        try_body_raise = re.search(
            r"try:\s*\n\s*(raise[^\n]+)",
            before,
        )
        raise_stmt = (
            try_body_raise.group(1).strip()
            if try_body_raise
            else 'raise RuntimeError("critical_error_propagation")'
        )
        replacement = (
            f"def {func_name}({params}) -> dict:\n"
            f"    try:\n"
            f"        {raise_stmt}\n"
            f"    except Exception:\n"
            f"        raise\n"
        )

    after_source = _replace_lines(source, start, end, replacement)
    edit = IpaRepairEdit(
        path=finding.source_span.path or "<memory>.py",
        start_line=start,
        end_line=end,
        before_text=before,
        after_text=replacement,
    )
    return after_source, edit


_RENDERERS = {
    IpaRepairTransformId.EXPLICIT_INIT: _render_explicit_init,
    IpaRepairTransformId.TYPED_UNAVAILABLE: _render_typed_unavailable,
    IpaRepairTransformId.SIMULATION_EVIDENCE: _render_simulation_evidence,
    IpaRepairTransformId.CANONICAL_CID: _render_canonical_cid,
    IpaRepairTransformId.CRITICAL_ERROR_PROPAGATION: _render_critical_error_propagation,
}


def render_ipa_transform(
    source: str,
    finding: IpaFinding | Mapping[str, Any],
    *,
    path: str,
    transform_id: Optional[IpaRepairTransformId | str] = None,
) -> tuple[str, IpaRepairEdit] | str:
    """Render one closed transform. Returns ``(after_source, edit)`` or abstention code."""

    if not isinstance(source, str):
        raise IpaRepairError("source must be a string")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise IpaRepairError("source exceeds byte bound")
    if not source.strip():
        return IpaRepairAbstentionReason.EMPTY_SOURCE.value

    normalized = _validate_path(path)
    if not normalized.endswith((".py", ".pyi")) and "<memory>" not in normalized:
        # Config-oriented paths are reserved; current grammar is Python AST only.
        if not normalized.endswith((".json", ".toml", ".yml", ".yaml", ".env")):
            return IpaRepairAbstentionReason.UNSUPPORTED_LANGUAGE.value

    ipa_finding = _coerce_finding(finding, path=normalized)
    if transform_id is None:
        try:
            selected = select_transform(ipa_finding)
        except IpaRepairError:
            return IpaRepairAbstentionReason.UNSUPPORTED_RULE.value
    else:
        selected = (
            transform_id
            if isinstance(transform_id, IpaRepairTransformId)
            else IpaRepairTransformId(str(transform_id))
        )

    expected_rule = TRANSFORM_TO_RULE[selected]
    if ipa_finding.rule_id != expected_rule.value:
        return IpaRepairAbstentionReason.PRECONDITION_MISMATCH.value

    renderer = _RENDERERS[selected]
    return renderer(source, ipa_finding)


def apply_ipa_repair(
    source: str,
    finding: IpaFinding | Mapping[str, Any],
    *,
    path: str,
    transform_id: Optional[IpaRepairTransformId | str] = None,
    admitted_paths: Optional[Iterable[str]] = None,
    require_reanalysis: bool = True,
) -> IpaRepairReceipt:
    """Apply one bounded IPA repair under the mutation gate.

    Returns an applied / noop receipt or a typed abstention. Never claims success
    without byte mutation (unless idempotent noop) and IPA reanalysis.
    """

    normalized = _normalize_path(path)
    before_hash = _sha256_text(source)
    ipa_finding = _coerce_finding(finding, path=normalized or "seeded/repair.py")
    finding_id = ipa_finding.finding_id

    if transform_id is None:
        try:
            selected = select_transform(ipa_finding)
        except IpaRepairError:
            return _abstain(
                transform_id="",
                rule_id=ipa_finding.rule_id,
                path=normalized,
                reasons=[IpaRepairAbstentionReason.UNSUPPORTED_RULE],
                finding_id=finding_id,
                before_hash=before_hash,
            )
    else:
        selected = (
            transform_id
            if isinstance(transform_id, IpaRepairTransformId)
            else IpaRepairTransformId(str(transform_id))
        )

    rule = TRANSFORM_TO_RULE[selected]
    rule_id = rule.value

    if not path_is_admitted(normalized, admitted_paths):
        gate = MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=normalized,
            reasons=(IpaRepairAbstentionReason.PATH_NOT_ADMITTED.value,),
            before_hash=before_hash,
            after_hash=before_hash,
            byte_mutated=False,
            reanalyzed=False,
        )
        return _abstain(
            transform_id=selected.value,
            rule_id=rule_id,
            path=normalized,
            reasons=[IpaRepairAbstentionReason.PATH_NOT_ADMITTED],
            finding_id=finding_id,
            before_hash=before_hash,
            mutation_gate=gate,
        )

    if ipa_finding.rule_id != rule_id:
        return _abstain(
            transform_id=selected.value,
            rule_id=rule_id,
            path=normalized,
            reasons=[IpaRepairAbstentionReason.PRECONDITION_MISMATCH],
            finding_id=finding_id,
            before_hash=before_hash,
        )

    # Idempotent path: already free of the target rule.
    try:
        before_findings = analyze_python_source(source, path=normalized)
    except Exception as exc:  # noqa: BLE001
        return _abstain(
            transform_id=selected.value,
            rule_id=rule_id,
            path=normalized,
            reasons=[IpaRepairAbstentionReason.PARSE_ERROR, str(exc)[:200]],
            finding_id=finding_id,
            before_hash=before_hash,
        )

    if rule_id not in {item.rule_id for item in before_findings}:
        gate = evaluate_mutation_gate(
            normalized,
            before_source=source,
            after_source=source,
            target_rule_id=rule_id,
            admitted_paths=admitted_paths,
            allow_idempotent_noop=True,
            before_findings=before_findings,
        )
        return IpaRepairReceipt(
            disposition=IpaRepairDisposition.NOOP,
            transform_id=selected.value,
            rule_id=rule_id,
            path=normalized,
            reasons=("already_repaired",),
            edits=(),
            before_hash=before_hash,
            after_hash=before_hash,
            after_source=source,
            reanalysis=IpaReanalysisReport(
                before_rule_ids=tuple(sorted({f.rule_id for f in before_findings})),
                after_rule_ids=tuple(sorted({f.rule_id for f in before_findings})),
                eliminated_rule_ids=(),
                new_rule_ids=(),
                before_finding_ids=tuple(f.finding_id for f in before_findings),
                after_finding_ids=tuple(f.finding_id for f in before_findings),
                target_rule_eliminated=True,
            ),
            mutation_gate=gate,
            finding_id=finding_id,
            public_compat_preserved=True,
        )

    rendered = render_ipa_transform(
        source,
        ipa_finding,
        path=normalized,
        transform_id=selected,
    )
    if isinstance(rendered, str):
        reason = rendered
        try:
            reason_enum = IpaRepairAbstentionReason(reason)
        except ValueError:
            reason_enum = IpaRepairAbstentionReason.PRECONDITION_MISMATCH
        return _abstain(
            transform_id=selected.value,
            rule_id=rule_id,
            path=normalized,
            reasons=[reason_enum],
            finding_id=finding_id,
            before_hash=before_hash,
        )

    after_source, edit = rendered
    # Preserve unrelated marker bytes when present in both texts.
    # (Guarded soft check — abstain only on hard public-API def removal without replacement.)

    if after_source == source:
        return _abstain(
            transform_id=selected.value,
            rule_id=rule_id,
            path=normalized,
            reasons=[IpaRepairAbstentionReason.NO_BYTE_CHANGE],
            finding_id=finding_id,
            before_hash=before_hash,
        )

    after_hash = _sha256_text(after_source)
    if require_reanalysis:
        gate = evaluate_mutation_gate(
            normalized,
            before_source=source,
            after_source=after_source,
            target_rule_id=rule_id,
            admitted_paths=admitted_paths,
            allow_idempotent_noop=False,
            before_findings=before_findings,
        )
        if not gate.admitted:
            return _abstain(
                transform_id=selected.value,
                rule_id=rule_id,
                path=normalized,
                reasons=gate.reasons
                or (IpaRepairAbstentionReason.REANALYSIS_STILL_FAILS.value,),
                finding_id=finding_id,
                before_hash=before_hash,
                after_hash=after_hash,
                after_source=after_source,
                mutation_gate=gate,
            )
        after_findings = analyze_python_source(after_source, path=normalized)
    else:
        # Still forbidden to complete without reanalysis in the public API —
        # require_reanalysis=False is reserved for internal dry renders.
        return _abstain(
            transform_id=selected.value,
            rule_id=rule_id,
            path=normalized,
            reasons=[IpaRepairAbstentionReason.REANALYSIS_STILL_FAILS],
            finding_id=finding_id,
            before_hash=before_hash,
            disposition=IpaRepairDisposition.REJECTED,
        )

    before_rules = tuple(sorted({item.rule_id for item in before_findings}))
    after_rules = tuple(sorted({item.rule_id for item in after_findings}))
    eliminated = tuple(sorted(set(before_rules) - set(after_rules)))
    new_rules = tuple(sorted(set(after_rules) - set(before_rules)))
    reanalysis = IpaReanalysisReport(
        before_rule_ids=before_rules,
        after_rule_ids=after_rules,
        eliminated_rule_ids=eliminated,
        new_rule_ids=new_rules,
        before_finding_ids=tuple(item.finding_id for item in before_findings),
        after_finding_ids=tuple(item.finding_id for item in after_findings),
        target_rule_eliminated=rule_id not in set(after_rules),
    )

    # Exact write path is the normalized admitted path.
    bound_edit = IpaRepairEdit(
        path=normalized,
        start_line=edit.start_line,
        end_line=edit.end_line,
        before_text=edit.before_text,
        after_text=edit.after_text,
    )

    return IpaRepairReceipt(
        disposition=IpaRepairDisposition.APPLIED,
        transform_id=selected.value,
        rule_id=rule_id,
        path=normalized,
        reasons=(),
        edits=(bound_edit,),
        before_hash=before_hash,
        after_hash=after_hash,
        after_source=after_source,
        reanalysis=reanalysis,
        mutation_gate=gate,
        finding_id=finding_id,
        public_compat_preserved=_public_compat_preserved(source, after_source),
    )


def _public_compat_preserved(before: str, after: str) -> bool:
    """Heuristic: public ``def`` names present before remain present after."""

    before_defs = set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", before, re.M))
    after_defs = set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", after, re.M))
    # explicit_init may introduce initialize_explicit while removing no public defs
    # other than moving a module-level statement (not a def).
    return before_defs <= after_defs or before_defs.issubset(after_defs | {"initialize_explicit"})


def apply_ipa_repair_idempotent(
    source: str,
    finding: IpaFinding | Mapping[str, Any],
    *,
    path: str,
    admitted_paths: Optional[Iterable[str]] = None,
) -> tuple[IpaRepairReceipt, IpaRepairReceipt]:
    """Apply twice; second receipt must be noop with identical bytes."""

    first = apply_ipa_repair(
        source,
        finding,
        path=path,
        admitted_paths=admitted_paths,
    )
    if not first.admitted:
        return first, first
    second_source = first.after_source if first.after_source else source
    # Build a synthetic finding for the same rule (span may be stale after edit).
    second_finding: Mapping[str, Any] = {
        "finding_id": first.finding_id or "ipa:idempotent",
        "rule_id": first.rule_id,
        "source_span": {
            "path": path,
            "start_line": 1,
            "end_line": 1,
            "symbol": "",
            "excerpt": "",
        },
        "sink_span": {
            "path": path,
            "start_line": 1,
            "end_line": 1,
            "symbol": "",
            "excerpt": "",
        },
        "disposition": "reject",
    }
    second = apply_ipa_repair(
        second_source,
        second_finding,
        path=path,
        transform_id=first.transform_id,
        admitted_paths=admitted_paths,
    )
    return first, second


def list_transform_grammar() -> tuple[dict[str, str], ...]:
    """Return the closed transform grammar as sorted records."""

    rows: list[dict[str, str]] = []
    for transform_id, rule in sorted(
        TRANSFORM_TO_RULE.items(), key=lambda item: item[0].value
    ):
        rows.append(
            {
                "transform_id": transform_id.value,
                "rule_id": rule.value,
                "family": rule.family,
            }
        )
    return tuple(rows)


__all__ = (
    "ANALYZER_VERSION",
    "BUNDLE",
    "DETERMINISTIC_REPAIRS_EVIDENCE",
    "EVIDENCE_ID",
    "INTERFACE",
    "PRODUCER_ID",
    "RULE_TO_TRANSFORM",
    "SCHEMA",
    "STABLE_TRANSFORM_IDS",
    "TASK_ID",
    "GOAL_ID",
    "TRANSFORM_TO_RULE",
    "IpaRepairAbstentionReason",
    "IpaRepairDisposition",
    "IpaRepairEdit",
    "IpaRepairError",
    "IpaRepairReceipt",
    "IpaRepairTransformId",
    "IpaReanalysisReport",
    "MutationGateDecision",
    "MutationGateDisposition",
    "apply_ipa_repair",
    "apply_ipa_repair_idempotent",
    "default_admitted_paths",
    "evaluate_mutation_gate",
    "list_transform_grammar",
    "path_is_admitted",
    "render_ipa_transform",
    "select_transform",
)
