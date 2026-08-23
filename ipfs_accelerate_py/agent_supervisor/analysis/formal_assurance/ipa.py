"""IPA — Import Purity and Capability Abstract Interpreter (FACP-042).

Extends existing AST / provenance / Datalog adapters with bounded product
domains for effects, trust/origin, outcomes, and identity. Detects:

* import-time installation / network / process / mutation
* mock-to-production flow
* success without effect observation
* exception swallowing
* raw / pseudo-CID construction

Every finding carries a source-to-sink trace and a stable rule ID. When Souffle
is unavailable the analyzer emits a typed capability record and continues via a
hermetic reference Datalog evaluator (analysis is never skipped). CEGAR
refinement may drop spurious imprecise paths without suppressing corpus seeds.

This module does not import analyzed packages, auto-install Souffle, or emit
full source bodies.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Optional, Sequence, Union

SCHEMA: Final[str] = "facp/ipa-analysis@1"
EVIDENCE_SCHEMA: Final[str] = "facp/ipa-analysis@1"
TASK_ID: Final[str] = "FACP-042"
GOAL_ID: Final[str] = "FACP-G410"
BUNDLE: Final[str] = "facp/static/ipa"
ANALYZER_VERSION: Final[str] = "ipa/v1"
HERMETIC_EVALUATOR_ID: Final[str] = "ipa.hermetic_reference_evaluator/v1"
SOUFFLE_TOOL_ID: Final[str] = "tool:souffle"

# IPA-relevant defect families from the FACP-008 corpus.
IPA_CORPUS_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "import_effect",
        "mock_capability",
        "false_success",
        "pseudo_cid",
    }
)

_SCAN_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".pyi", ".ts", ".tsx"})
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

_IMPORT_EFFECT_CALLEES: Final[frozenset[str]] = frozenset(
    {
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
        "os.environ.__setitem__",
        "os.putenv",
        "os.makedirs",
        "os.mkdir",
        "os.remove",
        "os.unlink",
        "shutil.rmtree",
        "pathlib.Path.mkdir",
        "pip.main",
        "urllib.request.urlopen",
        "requests.get",
        "requests.post",
        "socket.create_connection",
    }
)

_MOCK_SOURCE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "MagicMock",
        "Mock",
        "AsyncMock",
        "create_autospec",
        "patch",
    }
)

_MOCK_HELPER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(create_mock|mock_|_mock|MagicMock|MockWorker|mock_handler|"
    r"create_cuda_mock|mock_ipfs|MockIPFS)"
)

_LIVE_SINK_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(add_endpoint|register_endpoint|get_capabilities|test_hardware|"
    r"hwtest|production|live_observed|capability|api_available|"
    r"is_available|supported)"
)

_SUCCESS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "success",
        "available",
        "supported",
        "verified",
        "proven",
        "api_available",
        "status",
    }
)

_PSEUDO_CID_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(hexdigest\(\)|Qm\{|f[\"']Qm|bafy|cid\s*=\s*[\"'][0-9a-f]{16,}|"
    r"hashlib\.sha256|truncated.?sha|pseudo.?cid|mock_cid)"
)

_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{64}$")
_QM_FAKE_RE: Final[re.Pattern[str]] = re.compile(r"^Qm[0-9A-Za-z]{0,46}$")
_BAFY_FAKE_RE: Final[re.Pattern[str]] = re.compile(r"^bafy[0-9a-z]{10,}$", re.I)


class IpaError(ValueError):
    """Malformed IPA input, corpus binding, or refinement request."""


class IpaRuleId(str, Enum):
    """Stable rule identifiers for IPA product-domain violations."""

    IMPORT_EFFECT = "ipa.rule.import_effect"
    MOCK_TO_PRODUCTION = "ipa.rule.mock_to_production_flow"
    SUCCESS_WITHOUT_OBSERVATION = "ipa.rule.success_without_observation"
    EXCEPTION_SWALLOWING = "ipa.rule.exception_swallowing"
    PSEUDO_CID = "ipa.rule.pseudo_cid_construction"

    @property
    def family(self) -> str:
        return _RULE_TO_FAMILY[self]


_RULE_TO_FAMILY: Final[Mapping[IpaRuleId, str]] = {
    IpaRuleId.IMPORT_EFFECT: "import_effect",
    IpaRuleId.MOCK_TO_PRODUCTION: "mock_capability",
    IpaRuleId.SUCCESS_WITHOUT_OBSERVATION: "false_success",
    IpaRuleId.EXCEPTION_SWALLOWING: "false_success",
    IpaRuleId.PSEUDO_CID: "pseudo_cid",
}

_FAMILY_TO_RULE: Final[Mapping[str, IpaRuleId]] = {
    "import_effect": IpaRuleId.IMPORT_EFFECT,
    "mock_capability": IpaRuleId.MOCK_TO_PRODUCTION,
    "false_success": IpaRuleId.SUCCESS_WITHOUT_OBSERVATION,
    "pseudo_cid": IpaRuleId.PSEUDO_CID,
}

STABLE_RULE_IDS: Final[frozenset[str]] = frozenset(item.value for item in IpaRuleId)


class EffectAbstract(str, Enum):
    """Effect product-domain lattice (coarse, bounded)."""

    BOTTOM = "bottom"
    PURE = "pure"
    OBSERVED = "observed"
    STARTED = "started"
    EXTERNALLY_UNKNOWN = "externally_unknown"
    MUTATING = "mutating"
    NETWORK = "network"
    PROCESS = "process"
    INSTALL = "install"
    TOP = "top"


class TrustAbstract(str, Enum):
    """Trust / origin product-domain lattice."""

    BOTTOM = "bottom"
    ABSENT = "absent"
    DECLARED = "declared"
    FIXTURE = "fixture"
    SIMULATED = "simulated"
    HERMETIC_OBSERVED = "hermetic_observed"
    LIVE_OBSERVED = "live_observed"
    TOP = "top"


class ResultAbstract(str, Enum):
    """Outcome / result product-domain lattice."""

    BOTTOM = "bottom"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    ATTEMPTED = "attempted"
    UNKNOWN = "unknown"
    SUCCESS_CLAIMED = "success_claimed"
    OBSERVED_SUCCESS = "observed_success"
    VERIFIED = "verified"
    TOP = "top"


class IdentityAbstract(str, Enum):
    """Content-identity product-domain lattice."""

    BOTTOM = "bottom"
    ABSENT = "absent"
    RAW_HASH = "raw_hash"
    PSEUDO_CID = "pseudo_cid"
    STRUCTURAL_CID = "structural_cid"
    VERIFIED_CID = "verified_cid"
    TOP = "top"


class FindingDisposition(str, Enum):
    REJECT = "reject"
    CORPUS_BOUND = "corpus_bound"
    SPURIOUS_CANDIDATE = "spurious_candidate"
    REFINED_AWAY = "refined_away"


class SouffleStatus(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"
    UNKNOWN = "unknown"


class CapabilityDisposition(str, Enum):
    AVAILABLE = "available"
    TYPED_CAPABILITY_GAP = "typed_capability_gap"
    DEFER_CAPABILITY = "defer_capability"


@dataclass(frozen=True)
class ProductDomainState:
    """One abstract state in the effect × trust × result × identity product."""

    effect: EffectAbstract = EffectAbstract.BOTTOM
    trust: TrustAbstract = TrustAbstract.BOTTOM
    result: ResultAbstract = ResultAbstract.BOTTOM
    identity: IdentityAbstract = IdentityAbstract.BOTTOM

    def join(self, other: "ProductDomainState") -> "ProductDomainState":
        return ProductDomainState(
            effect=_join_enum(self.effect, other.effect, EffectAbstract),
            trust=_join_enum(self.trust, other.trust, TrustAbstract),
            result=_join_enum(self.result, other.result, ResultAbstract),
            identity=_join_enum(self.identity, other.identity, IdentityAbstract),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "effect": self.effect.value,
            "trust": self.trust.value,
            "result": self.result.value,
            "identity": self.identity.value,
        }


def _join_enum(left: Enum, right: Enum, enum_cls: type[Enum]) -> Enum:
    order = list(enum_cls)
    return order[max(order.index(left), order.index(right))]


@dataclass(frozen=True)
class SourceSpan:
    """Exact source locus for one IPA finding (never a full body dump)."""

    path: str
    start_line: int
    end_line: int
    symbol: str = ""
    excerpt: str = ""
    column: int = 0

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise IpaError("source span path is required")
        if (
            isinstance(self.start_line, bool)
            or not isinstance(self.start_line, int)
            or self.start_line < 1
        ):
            raise IpaError("source span start_line must be >= 1")
        if (
            isinstance(self.end_line, bool)
            or not isinstance(self.end_line, int)
            or self.end_line < self.start_line
        ):
            raise IpaError("source span end_line must be >= start_line")
        if len(self.excerpt.encode("utf-8")) > 512:
            object.__setattr__(self, "excerpt", self.excerpt[:200] + "...")

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
class TraceStep:
    """One bounded hop on a source-to-sink path."""

    kind: str
    label: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise IpaError("trace step kind is required")
        if not self.label.strip():
            raise IpaError("trace step label is required")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "label": self.label, "detail": self.detail}


@dataclass(frozen=True)
class SourceToSinkTrace:
    """Bounded source-to-sink explanation attached to every finding."""

    steps: tuple[TraceStep, ...]
    summary: str = ""
    source_label: str = ""
    sink_label: str = ""

    def __post_init__(self) -> None:
        if len(self.steps) < 2:
            raise IpaError("source-to-sink trace requires at least source and sink")
        if len(self.steps) > 32:
            raise IpaError("source-to-sink trace exceeds step bound")
        if not self.source_label:
            object.__setattr__(self, "source_label", self.steps[0].label)
        if not self.sink_label:
            object.__setattr__(self, "sink_label", self.steps[-1].label)
        if not self.summary:
            object.__setattr__(
                self,
                "summary",
                f"{self.source_label} -> {self.sink_label}",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "source_label": self.source_label,
            "sink_label": self.sink_label,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class IpaFinding:
    """One IPA product-domain violation with rule ID and source-to-sink trace."""

    finding_id: str
    rule_id: str
    disposition: FindingDisposition
    source_span: SourceSpan
    sink_span: SourceSpan
    trace: SourceToSinkTrace
    domain_state: ProductDomainState
    message: str = ""
    family: str = ""
    corpus_seed_id: str = ""
    corpus_defect_id: str = ""
    roadmap_seed: bool = False
    refinement_note: str = ""
    imprecise: bool = False

    def __post_init__(self) -> None:
        if not self.finding_id.strip():
            raise IpaError("finding_id is required")
        if self.rule_id not in STABLE_RULE_IDS:
            raise IpaError(f"unknown IPA rule_id: {self.rule_id!r}")
        if not self.family:
            object.__setattr__(self, "family", IpaRuleId(self.rule_id).family)

    @property
    def is_corpus_seed(self) -> bool:
        return bool(self.corpus_seed_id) or bool(self.roadmap_seed)

    @property
    def rule(self) -> IpaRuleId:
        return IpaRuleId(self.rule_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "rule_id": self.rule_id,
            "family": self.family,
            "disposition": self.disposition.value,
            "source_span": self.source_span.to_dict(),
            "sink_span": self.sink_span.to_dict(),
            "trace": self.trace.to_dict(),
            "domain_state": self.domain_state.to_dict(),
            "message": self.message,
            "corpus_seed_id": self.corpus_seed_id,
            "corpus_defect_id": self.corpus_defect_id,
            "roadmap_seed": self.roadmap_seed,
            "refinement_note": self.refinement_note,
            "imprecise": self.imprecise,
        }


@dataclass(frozen=True)
class SouffleCapabilityRecord:
    """Typed capability record for the Souffle Datalog engine.

    When Souffle is absent, analysis continues via ``reference_evaluator`` —
    never by skipping analysis or auto-installing the tool.
    """

    tool: str = "souffle"
    tool_id: str = SOUFFLE_TOOL_ID
    status: SouffleStatus = SouffleStatus.ABSENT
    disposition: CapabilityDisposition = CapabilityDisposition.TYPED_CAPABILITY_GAP
    available: bool = False
    version: Optional[str] = None
    executable_path: Optional[str] = None
    reference_evaluator: str = HERMETIC_EVALUATOR_ID
    analysis_backend: str = "hermetic_reference_evaluator"
    role: str = "datalog_engine"
    prohibited_compensation: tuple[str, ...] = (
        "auto_install",
        "import_time_installation",
        "worker_time_installation",
        "skip_analysis",
        "simulated_proof",
    )
    assumptions: tuple[str, ...] = ()
    schema: str = "facp/souffle-capability@1"

    def __post_init__(self) -> None:
        if self.available and self.status is SouffleStatus.ABSENT:
            raise IpaError("available Souffle cannot have absent status")
        if (
            not self.available
            and self.disposition is CapabilityDisposition.AVAILABLE
        ):
            raise IpaError("unavailable Souffle cannot claim available disposition")
        if not self.available and not self.reference_evaluator.strip():
            raise IpaError(
                "unavailable Souffle requires a hermetic reference evaluator id"
            )
        if "skip_analysis" not in self.prohibited_compensation:
            object.__setattr__(
                self,
                "prohibited_compensation",
                tuple(self.prohibited_compensation) + ("skip_analysis",),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tool": self.tool,
            "tool_id": self.tool_id,
            "status": self.status.value,
            "disposition": self.disposition.value,
            "available": self.available,
            "version": self.version,
            "executable_path": self.executable_path,
            "reference_evaluator": self.reference_evaluator,
            "analysis_backend": self.analysis_backend,
            "role": self.role,
            "prohibited_compensation": list(self.prohibited_compensation),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class SpuriousPathRefinement:
    """CEGAR refinement constraint that may eliminate an imprecise path.

    Refinements never apply to corpus-bound seeds.
    """

    refinement_id: str
    finding_id: str
    reason: str
    constraint: str = ""

    def __post_init__(self) -> None:
        if not self.refinement_id.strip():
            raise IpaError("refinement_id is required")
        if not self.finding_id.strip():
            raise IpaError("refinement finding_id is required")
        if not self.reason.strip():
            raise IpaError("refinement reason is required")

    def to_dict(self) -> dict[str, str]:
        return {
            "refinement_id": self.refinement_id,
            "finding_id": self.finding_id,
            "reason": self.reason,
            "constraint": self.constraint,
        }


@dataclass(frozen=True)
class IpaAnalysisReport:
    """Deterministic IPA analysis report for one tree, source, or corpus run."""

    findings: tuple[IpaFinding, ...]
    scanned_paths: tuple[str, ...] = ()
    corpus_seed_ids_bound: tuple[str, ...] = ()
    souffle_capability: Optional[SouffleCapabilityRecord] = None
    analysis_backend: str = "hermetic_reference_evaluator"
    refined_away_finding_ids: tuple[str, ...] = ()
    analyzer_version: str = ANALYZER_VERSION
    schema: str = SCHEMA

    @property
    def active_findings(self) -> tuple[IpaFinding, ...]:
        return tuple(
            item
            for item in self.findings
            if item.disposition is not FindingDisposition.REFINED_AWAY
        )

    @property
    def rule_ids(self) -> frozenset[str]:
        return frozenset(item.rule_id for item in self.active_findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_schema": EVIDENCE_SCHEMA,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "bundle": BUNDLE,
            "analyzer_version": self.analyzer_version,
            "analysis_backend": self.analysis_backend,
            "scanned_path_count": len(self.scanned_paths),
            "scanned_paths": list(self.scanned_paths),
            "finding_count": len(self.findings),
            "active_finding_count": len(self.active_findings),
            "corpus_seed_ids_bound": list(self.corpus_seed_ids_bound),
            "refined_away_finding_ids": list(self.refined_away_finding_ids),
            "souffle_capability": (
                self.souffle_capability.to_dict() if self.souffle_capability else None
            ),
            "findings": [item.to_dict() for item in self.findings],
        }


# ---------------------------------------------------------------------------
# Souffle capability + hermetic reference Datalog evaluator
# ---------------------------------------------------------------------------


def probe_souffle_capability(
    *,
    search_path: Optional[Sequence[str]] = None,
    force_absent: bool = False,
) -> SouffleCapabilityRecord:
    """Probe for a native Souffle binary without installing or downloading.

    Absent hosts receive a typed capability gap and the hermetic reference
    evaluator identity so analysis can continue.
    """

    if force_absent:
        return _absent_souffle_record(
            assumptions=(
                "Souffle probe forced absent for hermetic analysis path.",
                "Analysis continues via hermetic reference evaluator.",
            )
        )

    path_dirs = list(search_path) if search_path is not None else None
    executable = shutil.which("souffle", path=os.pathsep.join(path_dirs) if path_dirs else None)
    if executable is None:
        return _absent_souffle_record(
            assumptions=(
                "souffle is absent on the host PATH; never auto-install.",
                "Analysis continues via hermetic reference evaluator.",
            )
        )

    version = _read_souffle_version(executable)
    return SouffleCapabilityRecord(
        status=SouffleStatus.PRESENT,
        disposition=CapabilityDisposition.AVAILABLE,
        available=True,
        version=version,
        executable_path=executable,
        reference_evaluator=HERMETIC_EVALUATOR_ID,
        analysis_backend="souffle",
        assumptions=(
            f"Native souffle observed at {executable}.",
            "Hermetic reference evaluator remains available as fallback.",
        ),
    )


def _absent_souffle_record(*, assumptions: tuple[str, ...]) -> SouffleCapabilityRecord:
    return SouffleCapabilityRecord(
        status=SouffleStatus.ABSENT,
        disposition=CapabilityDisposition.TYPED_CAPABILITY_GAP,
        available=False,
        version=None,
        executable_path=None,
        reference_evaluator=HERMETIC_EVALUATOR_ID,
        analysis_backend="hermetic_reference_evaluator",
        assumptions=assumptions,
    )


def _read_souffle_version(executable: str) -> Optional[str]:
    # Cheap --version probe only. Never auto-install; never import analyzed pkgs.
    import subprocess

    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    if not text:
        return None
    first = text.splitlines()[0].strip()
    return first[:128] if first else None


@dataclass(frozen=True)
class DatalogAtom:
    """One ground or schematic Datalog atom ``Pred(arg0, ...).``"""

    predicate: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.predicate.strip():
            raise IpaError("datalog atom predicate is required")
        if len(self.args) > 16:
            raise IpaError("datalog atom arity exceeds bound")

    @property
    def is_ground(self) -> bool:
        return all(not _is_variable(arg) for arg in self.args)

    def to_tuple(self) -> tuple[str, ...]:
        return (self.predicate, *self.args)

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "args": list(self.args)}


@dataclass(frozen=True)
class DatalogRule:
    """Horn clause ``head :- body0, body1, ...`` with stable rule_id metadata."""

    rule_id: str
    head: DatalogAtom
    body: tuple[DatalogAtom, ...] = ()

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise IpaError("datalog rule_id is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "head": self.head.to_dict(),
            "body": [atom.to_dict() for atom in self.body],
        }


@dataclass(frozen=True)
class DatalogEvaluationResult:
    """Fixed-point relations from the hermetic reference evaluator."""

    relations: Mapping[str, frozenset[tuple[str, ...]]]
    derived_rule_ids: tuple[str, ...]
    evaluator_id: str = HERMETIC_EVALUATOR_ID
    iterations: int = 0

    def facts(self, predicate: str) -> frozenset[tuple[str, ...]]:
        return self.relations.get(predicate, frozenset())

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "iterations": self.iterations,
            "derived_rule_ids": list(self.derived_rule_ids),
            "relations": {
                name: [list(row) for row in sorted(rows)]
                for name, rows in sorted(self.relations.items())
            },
        }


def _is_variable(token: str) -> bool:
    return bool(token) and token[0].isupper()


class HermeticReferenceEvaluator:
    """Bounded bottom-up Datalog evaluator used when Souffle is unavailable.

    Supports positive Horn clauses with variables (uppercase identifiers) and
    ground facts. Evaluation is deterministic and iteration-bounded.
    """

    def __init__(self, *, max_iterations: int = 64, max_facts: int = 10_000) -> None:
        if max_iterations < 1 or max_iterations > 10_000:
            raise IpaError("max_iterations out of bounds")
        if max_facts < 1 or max_facts > 1_000_000:
            raise IpaError("max_facts out of bounds")
        self.max_iterations = max_iterations
        self.max_facts = max_facts
        self.evaluator_id = HERMETIC_EVALUATOR_ID

    def evaluate(
        self,
        facts: Sequence[DatalogAtom],
        rules: Sequence[DatalogRule],
    ) -> DatalogEvaluationResult:
        relations: dict[str, set[tuple[str, ...]]] = {}
        for fact in facts:
            if not fact.is_ground:
                raise IpaError(f"fact must be ground: {fact!r}")
            relations.setdefault(fact.predicate, set()).add(fact.args)
        derived_rules: set[str] = set()
        iterations = 0
        changed = True
        while changed and iterations < self.max_iterations:
            changed = False
            iterations += 1
            for rule in rules:
                for binding in self._match_body(rule.body, relations):
                    head_args = tuple(
                        binding.get(arg, arg) if _is_variable(arg) else arg
                        for arg in rule.head.args
                    )
                    if any(_is_variable(arg) for arg in head_args):
                        continue
                    bucket = relations.setdefault(rule.head.predicate, set())
                    before = len(bucket)
                    bucket.add(head_args)
                    if len(bucket) > before:
                        changed = True
                        derived_rules.add(rule.rule_id)
                    total = sum(len(rows) for rows in relations.values())
                    if total > self.max_facts:
                        raise IpaError("hermetic datalog evaluation exceeded fact bound")
        frozen = {
            name: frozenset(rows) for name, rows in sorted(relations.items())
        }
        return DatalogEvaluationResult(
            relations=frozen,
            derived_rule_ids=tuple(sorted(derived_rules)),
            evaluator_id=self.evaluator_id,
            iterations=iterations,
        )

    def _match_body(
        self,
        body: Sequence[DatalogAtom],
        relations: Mapping[str, set[tuple[str, ...]]],
    ) -> Iterator[dict[str, str]]:
        if not body:
            yield {}
            return

        def rec(index: int, binding: dict[str, str]) -> Iterator[dict[str, str]]:
            if index >= len(body):
                yield dict(binding)
                return
            atom = body[index]
            for row in sorted(relations.get(atom.predicate, set())):
                if len(row) != len(atom.args):
                    continue
                next_binding = dict(binding)
                ok = True
                for schema, value in zip(atom.args, row):
                    if _is_variable(schema):
                        existing = next_binding.get(schema)
                        if existing is None:
                            next_binding[schema] = value
                        elif existing != value:
                            ok = False
                            break
                    elif schema != value:
                        ok = False
                        break
                if ok:
                    yield from rec(index + 1, next_binding)

        yield from rec(0, {})


def default_ipa_datalog_rules() -> tuple[DatalogRule, ...]:
    """Souffle-compatible IPA product-domain rules for the hermetic evaluator."""

    return (
        DatalogRule(
            rule_id=IpaRuleId.MOCK_TO_PRODUCTION.value,
            head=DatalogAtom("IpaViolation", ("X", "Y", IpaRuleId.MOCK_TO_PRODUCTION.value)),
            body=(
                DatalogAtom("MockSource", ("X",)),
                DatalogAtom("FlowsTo", ("X", "Y")),
                DatalogAtom("LiveSink", ("Y",)),
            ),
        ),
        DatalogRule(
            rule_id=IpaRuleId.SUCCESS_WITHOUT_OBSERVATION.value,
            head=DatalogAtom(
                "IpaViolation",
                ("X", "Y", IpaRuleId.SUCCESS_WITHOUT_OBSERVATION.value),
            ),
            body=(
                DatalogAtom("SuccessClaim", ("X",)),
                DatalogAtom("FlowsTo", ("X", "Y")),
                DatalogAtom("UnobservedEffect", ("Y",)),
            ),
        ),
        DatalogRule(
            rule_id=IpaRuleId.PSEUDO_CID.value,
            head=DatalogAtom("IpaViolation", ("X", "Y", IpaRuleId.PSEUDO_CID.value)),
            body=(
                DatalogAtom("RawHash", ("X",)),
                DatalogAtom("FlowsTo", ("X", "Y")),
                DatalogAtom("CidSink", ("Y",)),
            ),
        ),
        DatalogRule(
            rule_id=IpaRuleId.IMPORT_EFFECT.value,
            head=DatalogAtom("IpaViolation", ("X", "Y", IpaRuleId.IMPORT_EFFECT.value)),
            body=(
                DatalogAtom("ModuleTopLevel", ("X",)),
                DatalogAtom("EffectfulCall", ("X", "Y")),
            ),
        ),
        DatalogRule(
            rule_id=IpaRuleId.EXCEPTION_SWALLOWING.value,
            head=DatalogAtom(
                "IpaViolation",
                ("X", "Y", IpaRuleId.EXCEPTION_SWALLOWING.value),
            ),
            body=(
                DatalogAtom("SwallowedException", ("X",)),
                DatalogAtom("FlowsTo", ("X", "Y")),
                DatalogAtom("SuccessClaim", ("Y",)),
            ),
        ),
        # Transitive flow closure for CEGAR / imprecise dispatch.
        DatalogRule(
            rule_id="ipa.rule.flow_transitive",
            head=DatalogAtom("FlowsTo", ("X", "Z")),
            body=(
                DatalogAtom("FlowsTo", ("X", "Y")),
                DatalogAtom("FlowsTo", ("Y", "Z")),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Path / text helpers
# ---------------------------------------------------------------------------


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


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _const_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
        return {"True": True, "False": False, "None": None}[node.id]
    return None


def _finding_id(rule: IpaRuleId, span: SourceSpan, sink: SourceSpan) -> str:
    digest = hashlib.sha256(
        f"{rule.value}|{_normalize_relpath(span.path)}|{span.start_line}|"
        f"{span.column}|{_normalize_relpath(sink.path)}|{sink.start_line}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return f"ipa:{rule.value}:{digest}"


def _make_trace(
    *,
    source: SourceSpan,
    sink: SourceSpan,
    rule: IpaRuleId,
    intermediate: Sequence[TraceStep] = (),
) -> SourceToSinkTrace:
    steps: list[TraceStep] = [
        TraceStep(
            kind="source",
            label=f"{source.path}:{source.start_line}",
            detail=source.excerpt or source.symbol,
        ),
        *list(intermediate)[:28],
        TraceStep(
            kind="sink",
            label=f"{sink.path}:{sink.start_line}",
            detail=sink.excerpt or sink.symbol or rule.family,
        ),
    ]
    # Ensure distinct source/sink even when spans coincide.
    if len(steps) == 2 and steps[0].label == steps[1].label:
        steps.insert(
            1,
            TraceStep(kind="rule", label=rule.value, detail=rule.family),
        )
    return SourceToSinkTrace(steps=tuple(steps))


def _domain_for_rule(rule: IpaRuleId) -> ProductDomainState:
    if rule is IpaRuleId.IMPORT_EFFECT:
        return ProductDomainState(
            effect=EffectAbstract.MUTATING,
            trust=TrustAbstract.DECLARED,
            result=ResultAbstract.ATTEMPTED,
            identity=IdentityAbstract.ABSENT,
        )
    if rule is IpaRuleId.MOCK_TO_PRODUCTION:
        return ProductDomainState(
            effect=EffectAbstract.PURE,
            trust=TrustAbstract.SIMULATED,
            result=ResultAbstract.SUCCESS_CLAIMED,
            identity=IdentityAbstract.ABSENT,
        )
    if rule is IpaRuleId.SUCCESS_WITHOUT_OBSERVATION:
        return ProductDomainState(
            effect=EffectAbstract.EXTERNALLY_UNKNOWN,
            trust=TrustAbstract.DECLARED,
            result=ResultAbstract.SUCCESS_CLAIMED,
            identity=IdentityAbstract.ABSENT,
        )
    if rule is IpaRuleId.EXCEPTION_SWALLOWING:
        return ProductDomainState(
            effect=EffectAbstract.EXTERNALLY_UNKNOWN,
            trust=TrustAbstract.DECLARED,
            result=ResultAbstract.SUCCESS_CLAIMED,
            identity=IdentityAbstract.ABSENT,
        )
    if rule is IpaRuleId.PSEUDO_CID:
        return ProductDomainState(
            effect=EffectAbstract.PURE,
            trust=TrustAbstract.SIMULATED,
            result=ResultAbstract.SUCCESS_CLAIMED,
            identity=IdentityAbstract.PSEUDO_CID,
        )
    return ProductDomainState()


# ---------------------------------------------------------------------------
# Python AST product-domain analysis
# ---------------------------------------------------------------------------


@dataclass
class _PythonIpaVisitor(ast.NodeVisitor):
    path: str
    source: str
    findings: list[IpaFinding] = field(default_factory=list)
    _scope_stack: list[str] = field(default_factory=list)
    _module_level: bool = True
    _function_depth: int = 0
    facts: list[DatalogAtom] = field(default_factory=list)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self._function_depth += 1
        prev = self._module_level
        self._module_level = False
        self.generic_visit(node)
        self._module_level = prev
        self._function_depth -= 1
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = _qualified_name(node.func)
        short = name.rsplit(".", 1)[-1] if name else ""
        lineno = int(getattr(node, "lineno", 1) or 1)
        symbol = ".".join(self._scope_stack) if self._scope_stack else "<module>"
        excerpt = _excerpt_line(self.source, lineno)
        span = SourceSpan(
            path=self.path,
            start_line=lineno,
            end_line=int(getattr(node, "end_lineno", None) or lineno),
            symbol=symbol,
            excerpt=excerpt,
            column=int(getattr(node, "col_offset", 0) or 0),
        )
        node_id = f"{self.path}:{lineno}:{name or short or 'call'}"

        if self._module_level and (
            name in _IMPORT_EFFECT_CALLEES
            or short in {"system", "run", "Popen", "urlopen", "mkdir", "makedirs"}
            or "pip" in name.casefold()
            or (name.endswith("environ.__setitem__"))
        ):
            self._record(
                IpaRuleId.IMPORT_EFFECT,
                source=span,
                sink=span,
                message=f"import-time effectful call {name or short!r}",
            )
            self.facts.append(DatalogAtom("ModuleTopLevel", (node_id,)))
            self.facts.append(DatalogAtom("EffectfulCall", (node_id, name or short)))

        if short in _MOCK_SOURCE_NAMES or _MOCK_HELPER_RE.search(name or short or ""):
            self.facts.append(DatalogAtom("MockSource", (node_id,)))
            # Conservative: mock construction near a live sink name is a flow.
            if _LIVE_SINK_RE.search(symbol) or _LIVE_SINK_RE.search(excerpt):
                sink = replace(span, excerpt=excerpt)
                self._record(
                    IpaRuleId.MOCK_TO_PRODUCTION,
                    source=span,
                    sink=sink,
                    message=f"mock source {name or short!r} reaches live sink context",
                    intermediate=(
                        TraceStep(kind="mock_source", label=name or short),
                        TraceStep(kind="live_sink_context", label=symbol or excerpt),
                    ),
                )
                self.facts.append(DatalogAtom("LiveSink", (node_id,)))
                self.facts.append(DatalogAtom("FlowsTo", (node_id, node_id)))

        if name.endswith("hexdigest") or short == "hexdigest" or "sha256" in name:
            self.facts.append(DatalogAtom("RawHash", (node_id,)))
            if _PSEUDO_CID_RE.search(excerpt) or "cid" in excerpt.casefold():
                self._record(
                    IpaRuleId.PSEUDO_CID,
                    source=span,
                    sink=span,
                    message="raw hash / pseudo-CID construction",
                    intermediate=(
                        TraceStep(kind="raw_hash", label=name or short),
                        TraceStep(kind="cid_sink", label=symbol or "cid"),
                    ),
                )
                self.facts.append(DatalogAtom("CidSink", (node_id,)))
                self.facts.append(DatalogAtom("FlowsTo", (node_id, node_id)))

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._visit_binding(node.targets, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._visit_binding([node.target], node.value, node)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                continue
            key = _const_value(key_node)
            if isinstance(key, str):
                self._maybe_success_or_cid(key, value_node, key_node)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg:
            self._maybe_success_or_cid(node.arg, node.value, node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # obj["IPFS_DATASETS_AUTO_INSTALL"] = ... handled via Assign.
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            if self._handler_swallows(handler):
                lineno = int(getattr(handler, "lineno", 1) or 1)
                symbol = ".".join(self._scope_stack) if self._scope_stack else "<module>"
                excerpt = _excerpt_line(self.source, lineno)
                span = SourceSpan(
                    path=self.path,
                    start_line=lineno,
                    end_line=int(getattr(handler, "end_lineno", None) or lineno),
                    symbol=symbol,
                    excerpt=excerpt or "except: pass",
                    column=int(getattr(handler, "col_offset", 0) or 0),
                )
                sink = span
                # Look for nearby success claims in the same function body text.
                success_sink = self._nearby_success_span(lineno) or span
                self._record(
                    IpaRuleId.EXCEPTION_SWALLOWING,
                    source=span,
                    sink=success_sink,
                    message="exception swallowed; success path not fail-closed",
                    intermediate=(
                        TraceStep(kind="swallowed_exception", label=symbol or "except"),
                        TraceStep(
                            kind="success_continuation",
                            label=success_sink.excerpt or "success",
                        ),
                    ),
                )
                node_id = f"{self.path}:{lineno}:except"
                self.facts.append(DatalogAtom("SwallowedException", (node_id,)))
                self.facts.append(
                    DatalogAtom(
                        "SuccessClaim",
                        (f"{success_sink.path}:{success_sink.start_line}:success",),
                    )
                )
                self.facts.append(
                    DatalogAtom(
                        "FlowsTo",
                        (
                            node_id,
                            f"{success_sink.path}:{success_sink.start_line}:success",
                        ),
                    )
                )
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # Module-level env mutation via subscript assign is an Assign; keep generic.
        self.generic_visit(node)

    def _visit_binding(
        self,
        targets: Sequence[ast.AST],
        value: ast.AST,
        locus: ast.AST,
    ) -> None:
        for target in targets:
            if isinstance(target, ast.Name):
                self._maybe_success_or_cid(target.id, value, locus)
                if self._module_level and target.id.casefold() in {
                    "auto_install",
                    "installer",
                }:
                    self._record_import_assign(target, value, locus)
            elif isinstance(target, ast.Attribute):
                self._maybe_success_or_cid(target.attr, value, locus)
                if self._module_level and target.attr == "environ":
                    self._record_import_assign(target, value, locus)
            elif isinstance(target, ast.Subscript):
                key = _const_value(target.slice)
                if isinstance(key, str):
                    self._maybe_success_or_cid(key, value, locus)
                    if self._module_level and (
                        "AUTO_INSTALL" in key
                        or key.startswith("IPFS_")
                        or key == "PATH"
                    ):
                        self._record_import_assign(target, value, locus)
            elif isinstance(target, (ast.Tuple, ast.List)):
                if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
                    value.elts
                ):
                    for left, right in zip(target.elts, value.elts):
                        self._visit_binding([left], right, locus)

    def _record_import_assign(
        self, target: ast.AST, value: ast.AST, locus: ast.AST
    ) -> None:
        lineno = int(getattr(locus, "lineno", 1) or 1)
        symbol = ".".join(self._scope_stack) if self._scope_stack else "<module>"
        excerpt = _excerpt_line(self.source, lineno)
        span = SourceSpan(
            path=self.path,
            start_line=lineno,
            end_line=int(getattr(locus, "end_lineno", None) or lineno),
            symbol=symbol,
            excerpt=excerpt,
            column=int(getattr(locus, "col_offset", 0) or 0),
        )
        self._record(
            IpaRuleId.IMPORT_EFFECT,
            source=span,
            sink=span,
            message="import-time environment / installer mutation",
        )

    def _maybe_success_or_cid(
        self, field_name: str, value_node: ast.AST, locus: ast.AST
    ) -> None:
        lowered = field_name.casefold()
        lineno = int(getattr(locus, "lineno", None) or getattr(value_node, "lineno", 1) or 1)
        symbol = ".".join(self._scope_stack) if self._scope_stack else "<module>"
        excerpt = _excerpt_line(self.source, lineno)
        span = SourceSpan(
            path=self.path,
            start_line=lineno,
            end_line=int(
                getattr(locus, "end_lineno", None)
                or getattr(value_node, "end_lineno", None)
                or lineno
            ),
            symbol=symbol,
            excerpt=excerpt,
            column=int(getattr(locus, "col_offset", 0) or 0),
        )
        const = _const_value(value_node)
        node_id = f"{self.path}:{lineno}:{field_name}"

        if lowered in _SUCCESS_KEYS and const is True:
            self.facts.append(DatalogAtom("SuccessClaim", (node_id,)))
            self.facts.append(DatalogAtom("UnobservedEffect", (node_id,)))
            self.facts.append(DatalogAtom("FlowsTo", (node_id, node_id)))
            # Hardcoded true support / availability is success without observation.
            self._record(
                IpaRuleId.SUCCESS_WITHOUT_OBSERVATION,
                source=span,
                sink=span,
                message=f"unobserved success/support claim {field_name!r}=True",
                intermediate=(
                    TraceStep(kind="success_claim", label=field_name),
                    TraceStep(kind="unobserved_effect", label=symbol or field_name),
                ),
            )
            if _MOCK_HELPER_RE.search(symbol) or _MOCK_HELPER_RE.search(excerpt):
                self._record(
                    IpaRuleId.MOCK_TO_PRODUCTION,
                    source=span,
                    sink=span,
                    message=f"mock-origin value bound as {field_name!r}",
                    intermediate=(
                        TraceStep(kind="mock_source", label=symbol or excerpt),
                        TraceStep(kind="live_sink", label=field_name),
                    ),
                )
                self.facts.append(DatalogAtom("MockSource", (node_id,)))
                self.facts.append(DatalogAtom("LiveSink", (node_id,)))

        if lowered in {"cid", "content_id", "ipfs_cid"}:
            text = const if isinstance(const, str) else ""
            value_text = text or _qualified_name(value_node) or excerpt
            if (
                (text and (_HEX64_RE.match(text) or _QM_FAKE_RE.match(text) or _BAFY_FAKE_RE.match(text)))
                or _PSEUDO_CID_RE.search(value_text)
                or "hexdigest" in value_text
                or "mock_cid" in value_text.casefold()
            ):
                self.facts.append(DatalogAtom("RawHash", (node_id,)))
                self.facts.append(DatalogAtom("CidSink", (node_id,)))
                self.facts.append(DatalogAtom("FlowsTo", (node_id, node_id)))
                self._record(
                    IpaRuleId.PSEUDO_CID,
                    source=span,
                    sink=span,
                    message=f"pseudo-CID / raw hash bound as {field_name!r}",
                    intermediate=(
                        TraceStep(kind="raw_hash", label=value_text[:80]),
                        TraceStep(kind="cid_sink", label=field_name),
                    ),
                )

        # f"Qm{...}" or similar formatted pseudo CIDs assigned to any name.
        if isinstance(value_node, ast.JoinedStr) and lowered in {
            "cid",
            "mock_cid",
            "content_id",
            "key",
        }:
            self._record(
                IpaRuleId.PSEUDO_CID,
                source=span,
                sink=span,
                message=f"formatted pseudo-CID assigned to {field_name!r}",
            )

    def _handler_swallows(self, handler: ast.ExceptHandler) -> bool:
        body = handler.body
        if not body:
            return True
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            return True
        if len(body) == 1 and isinstance(body[0], ast.Return):
            # return True / return {"success": True}
            value = body[0].value
            const = _const_value(value) if value is not None else None
            if const is True or const is None:
                return True
            if isinstance(value, ast.Dict):
                for key_node, value_node in zip(value.keys, value.values):
                    key = _const_value(key_node) if key_node else None
                    if (
                        isinstance(key, str)
                        and key.casefold() in _SUCCESS_KEYS
                        and _const_value(value_node) is True
                    ):
                        return True
        # except Exception: continue / bare continue
        if all(isinstance(stmt, (ast.Pass, ast.Continue)) for stmt in body):
            return True
        return False

    def _nearby_success_span(self, lineno: int) -> Optional[SourceSpan]:
        lines = self.source.splitlines()
        for offset in range(0, 8):
            idx = lineno - 1 + offset
            if idx < 0 or idx >= len(lines):
                break
            text = lines[idx]
            if re.search(r"(?i)(success|available|supported)\s*[:=]\s*True", text):
                return SourceSpan(
                    path=self.path,
                    start_line=idx + 1,
                    end_line=idx + 1,
                    symbol=".".join(self._scope_stack),
                    excerpt=text.strip()[:160],
                )
        return None

    def _record(
        self,
        rule: IpaRuleId,
        *,
        source: SourceSpan,
        sink: SourceSpan,
        message: str,
        intermediate: Sequence[TraceStep] = (),
        imprecise: bool = False,
    ) -> None:
        disposition = (
            FindingDisposition.SPURIOUS_CANDIDATE
            if imprecise
            else FindingDisposition.REJECT
        )
        self.findings.append(
            IpaFinding(
                finding_id=_finding_id(rule, source, sink),
                rule_id=rule.value,
                disposition=disposition,
                source_span=source,
                sink_span=sink,
                trace=_make_trace(
                    source=source,
                    sink=sink,
                    rule=rule,
                    intermediate=intermediate,
                ),
                domain_state=_domain_for_rule(rule),
                message=message,
                imprecise=imprecise,
            )
        )


def analyze_python_source(
    source: str,
    *,
    path: str = "<memory>.py",
) -> tuple[IpaFinding, ...]:
    """Analyze one Python source string with IPA product domains."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise IpaError(f"python source failed to parse: {path}: {exc}") from exc
    visitor = _PythonIpaVisitor(path=_normalize_relpath(path), source=source)
    visitor.visit(tree)
    # Close product-domain facts through the hermetic evaluator for stable rule IDs.
    evaluator = HermeticReferenceEvaluator()
    evaluation = evaluator.evaluate(visitor.facts, default_ipa_datalog_rules())
    datalog_findings = _findings_from_datalog(
        evaluation, path=_normalize_relpath(path), source=source
    )
    return _dedupe_findings(tuple(visitor.findings) + datalog_findings)


def analyze_typescript_source(
    source: str,
    *,
    path: str = "<memory>.ts",
) -> tuple[IpaFinding, ...]:
    """Lightweight TypeScript IPA scan (regex/product-domain heuristics).

    Full TS AST adapters remain owned by existing program_ast adapters; this
    path only detects IPA seed patterns without trusting naming alone.
    """

    findings: list[IpaFinding] = []
    rel = _normalize_relpath(path)
    lines = source.splitlines()
    for lineno, text in enumerate(lines, start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith("//"):
            continue
        span = SourceSpan(
            path=rel,
            start_line=lineno,
            end_line=lineno,
            symbol="",
            excerpt=stripped[:160],
        )
        if re.search(
            r"(?i)(child_process|execSync|spawnSync|fs\.mkdirSync|process\.env\.[A-Z0-9_]+\s*=)",
            stripped,
        ) and _is_likely_module_scope(lines, lineno):
            findings.append(
                _simple_finding(
                    IpaRuleId.IMPORT_EFFECT,
                    span,
                    "import/module-scope effectful TypeScript call",
                )
            )
        if re.search(r"(?i)(jest\.fn|vi\.mock|createMock|MagicMock)", stripped) and (
            re.search(r"(?i)(available|capability|supported)\s*:\s*true", stripped)
            or _LIVE_SINK_RE.search(stripped)
        ):
            findings.append(
                _simple_finding(
                    IpaRuleId.MOCK_TO_PRODUCTION,
                    span,
                    "mock-origin value on live TypeScript sink",
                )
            )
        if re.search(
            r"(?i)(success|available|supported)\s*:\s*true", stripped
        ) and not re.search(r"(?i)(observed|verified|effect)", stripped):
            findings.append(
                _simple_finding(
                    IpaRuleId.SUCCESS_WITHOUT_OBSERVATION,
                    span,
                    "success/support true without observation",
                )
            )
        if re.search(r"(?i)(cid\s*[:=].*(Qm|bafy|[0-9a-f]{32,})|createHash\(['\"]sha256)", stripped):
            findings.append(
                _simple_finding(
                    IpaRuleId.PSEUDO_CID,
                    span,
                    "pseudo-CID / raw hash TypeScript construction",
                )
            )
        if re.search(r"(?i)catch\s*\([^)]*\)\s*\{\s*\}", stripped) or re.search(
            r"(?i)catch\s*\([^)]*\)\s*\{\s*return\s*\{\s*success\s*:\s*true",
            stripped,
        ):
            findings.append(
                _simple_finding(
                    IpaRuleId.EXCEPTION_SWALLOWING,
                    span,
                    "exception swallowed on TypeScript success path",
                )
            )
    return _dedupe_findings(findings)


def _is_likely_module_scope(lines: Sequence[str], lineno: int) -> bool:
    # Heuristic: no unmatched "{" opening a function/class before this line
    # that would indent the statement — keep conservative (prefer false pos.
    # only when clearly top-level by indentation).
    line = lines[lineno - 1]
    return len(line) == len(line.lstrip()) or line.startswith("export ")


def _simple_finding(rule: IpaRuleId, span: SourceSpan, message: str) -> IpaFinding:
    return IpaFinding(
        finding_id=_finding_id(rule, span, span),
        rule_id=rule.value,
        disposition=FindingDisposition.REJECT,
        source_span=span,
        sink_span=span,
        trace=_make_trace(source=span, sink=span, rule=rule),
        domain_state=_domain_for_rule(rule),
        message=message,
    )


def _findings_from_datalog(
    evaluation: DatalogEvaluationResult,
    *,
    path: str,
    source: str,
) -> tuple[IpaFinding, ...]:
    findings: list[IpaFinding] = []
    for row in sorted(evaluation.facts("IpaViolation")):
        if len(row) < 3:
            continue
        src_id, sink_id, rule_id = row[0], row[1], row[2]
        if rule_id not in STABLE_RULE_IDS:
            continue
        rule = IpaRuleId(rule_id)
        src_line = _line_from_node_id(src_id)
        sink_line = _line_from_node_id(sink_id)
        source_span = SourceSpan(
            path=path,
            start_line=src_line,
            end_line=src_line,
            excerpt=_excerpt_line(source, src_line),
            symbol=src_id.rsplit(":", 1)[-1][:80],
        )
        sink_span = SourceSpan(
            path=path,
            start_line=sink_line,
            end_line=sink_line,
            excerpt=_excerpt_line(source, sink_line),
            symbol=sink_id.rsplit(":", 1)[-1][:80],
        )
        findings.append(
            IpaFinding(
                finding_id=_finding_id(rule, source_span, sink_span),
                rule_id=rule.value,
                disposition=FindingDisposition.REJECT,
                source_span=source_span,
                sink_span=sink_span,
                trace=_make_trace(
                    source=source_span,
                    sink=sink_span,
                    rule=rule,
                    intermediate=(
                        TraceStep(kind="datalog", label=rule.value),
                        TraceStep(kind="evaluator", label=evaluation.evaluator_id),
                    ),
                ),
                domain_state=_domain_for_rule(rule),
                message=f"hermetic datalog derived {rule.value}",
            )
        )
    return tuple(findings)


def _line_from_node_id(node_id: str) -> int:
    parts = str(node_id).split(":")
    for part in parts:
        if part.isdigit():
            return max(1, int(part))
    return 1


def analyze_path(
    path: Union[str, Path],
    *,
    root: Union[str, Path, None] = None,
) -> tuple[IpaFinding, ...]:
    """Analyze a single source file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise IpaError(f"analyze path is not a file: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    rel = _relative_to_root(file_path, root)
    suffix = file_path.suffix.casefold()
    if suffix in {".py", ".pyi"}:
        return analyze_python_source(text, path=rel)
    if suffix in {".ts", ".tsx"}:
        return analyze_typescript_source(text, path=rel)
    return ()


def analyze_tree(
    root: Union[str, Path],
    *,
    relative_paths: Optional[Sequence[str]] = None,
    corpus_entries: Optional[Sequence[Mapping[str, Any]]] = None,
    souffle_capability: Optional[SouffleCapabilityRecord] = None,
    refinements: Optional[Sequence[SpuriousPathRefinement]] = None,
) -> IpaAnalysisReport:
    """Analyze a repository tree (or an explicit relative path subset)."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise IpaError(f"analyze root is not a directory: {root_path}")

    capability = souffle_capability or probe_souffle_capability()
    paths = (
        [_normalize_relpath(item) for item in relative_paths]
        if relative_paths is not None
        else list(_iter_scan_paths(root_path))
    )

    findings: list[IpaFinding] = []
    scanned: list[str] = []
    for rel in paths:
        abs_path = root_path / rel
        if not abs_path.is_file():
            continue
        if abs_path.suffix.casefold() not in _SCAN_SUFFIXES:
            continue
        scanned.append(rel)
        findings.extend(analyze_path(abs_path, root=root_path))

    if corpus_entries:
        findings.extend(
            finding
            for entry in corpus_entries
            if str(entry.get("family") or "") in IPA_CORPUS_FAMILIES
            for finding in findings_for_corpus_entry(entry, repo_root=root_path)
        )

    deduped = _dedupe_findings(findings)
    refined, refined_ids = refine_spurious_paths(deduped, refinements or ())
    corpus_ids = tuple(
        sorted({item.corpus_seed_id for item in refined if item.corpus_seed_id})
    )
    backend = (
        "souffle"
        if capability.available
        else capability.analysis_backend or "hermetic_reference_evaluator"
    )
    return IpaAnalysisReport(
        findings=refined,
        scanned_paths=tuple(scanned),
        corpus_seed_ids_bound=corpus_ids,
        souffle_capability=capability,
        analysis_backend=backend,
        refined_away_finding_ids=refined_ids,
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


def _dedupe_findings(findings: Iterable[IpaFinding]) -> tuple[IpaFinding, ...]:
    seen: set[str] = set()
    ordered: list[IpaFinding] = []
    for item in findings:
        if item.finding_id in seen:
            continue
        seen.add(item.finding_id)
        ordered.append(item)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Corpus binding
# ---------------------------------------------------------------------------


def load_defect_corpus(path: Union[str, Path]) -> tuple[dict[str, Any], ...]:
    """Load FACP-008 defect corpus JSONL entries."""

    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise IpaError(f"defect corpus not found: {corpus_path}")
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
            raise IpaError(f"defect corpus line {line_no} is not JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise IpaError(f"defect corpus line {line_no} must be an object")
        entries.append(dict(payload))
    return tuple(entries)


def ipa_corpus_entries(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Filter corpus entries to IPA product-domain families."""

    return tuple(
        dict(entry)
        for entry in entries
        if str(entry.get("family") or "") in IPA_CORPUS_FAMILIES
    )


def findings_for_corpus_entry(
    entry: Mapping[str, Any],
    *,
    repo_root: Union[str, Path, None] = None,
) -> tuple[IpaFinding, ...]:
    """Materialize corpus-bound IPA findings with source-to-sink traces."""

    seed_id = str(entry.get("seed_id") or "")
    defect_id = str(entry.get("defect_id") or "")
    family = str(entry.get("family") or "")
    if family not in IPA_CORPUS_FAMILIES:
        raise IpaError(
            f"corpus entry {seed_id or defect_id!r} family {family!r} is not IPA-relevant"
        )
    rule = _FAMILY_TO_RULE[family]
    # Prefer exception_swallowing when the seed text indicates it.
    blob = " ".join(
        [
            str(entry.get("title") or ""),
            str(entry.get("scenario") or ""),
            str(entry.get("category") or ""),
        ]
    ).casefold()
    if "swallow" in blob or "except" in blob and "pass" in blob:
        rule = IpaRuleId.EXCEPTION_SWALLOWING

    spans_raw = entry.get("source_spans") or []
    if not isinstance(spans_raw, Sequence) or isinstance(spans_raw, (str, bytes)):
        raise IpaError(f"corpus entry {seed_id!r} source_spans must be a sequence")
    if not spans_raw:
        raise IpaError(f"corpus entry {seed_id!r} requires at least one source span")

    call_flow = entry.get("call_flow_path") or entry.get("call_flow") or []
    flow_labels = (
        [str(item) for item in call_flow]
        if isinstance(call_flow, Sequence) and not isinstance(call_flow, (str, bytes))
        else []
    )

    findings: list[IpaFinding] = []
    spans = [SourceSpan.from_dict(item) for item in spans_raw if isinstance(item, Mapping)]
    if not spans:
        raise IpaError(f"corpus entry {seed_id!r} has no usable source spans")

    source = spans[0]
    sink = spans[-1]
    intermediate: list[TraceStep] = [
        TraceStep(
            kind="seed_span",
            label=f"{source.path}:{source.start_line}-{source.end_line}",
            detail=source.excerpt,
        )
    ]
    for label in flow_labels[:6]:
        intermediate.append(TraceStep(kind="call_flow", label=label))
    for mid in spans[1:-1][:4]:
        intermediate.append(
            TraceStep(
                kind="intermediate_span",
                label=f"{mid.path}:{mid.start_line}",
                detail=mid.excerpt,
            )
        )
    intermediate.append(
        TraceStep(
            kind="rule",
            label=rule.value,
            detail=str(entry.get("expected_illegal_promotion") or family),
        )
    )
    if source.path == sink.path and source.start_line == sink.start_line and len(spans) == 1:
        # Single-span seeds still need a distinct sink hop for source-to-sink.
        sink = replace(
            sink,
            excerpt=(sink.excerpt or "") + f" [sink:{family}]",
        )

    finding = IpaFinding(
        finding_id=f"ipa:corpus:{seed_id or defect_id}:{rule.value}",
        rule_id=rule.value,
        disposition=FindingDisposition.CORPUS_BOUND,
        source_span=source,
        sink_span=sink,
        trace=_make_trace(
            source=source,
            sink=sink,
            rule=rule,
            intermediate=intermediate,
        ),
        domain_state=_domain_for_rule(rule),
        message=str(entry.get("title") or seed_id or defect_id),
        corpus_seed_id=seed_id,
        corpus_defect_id=defect_id,
        roadmap_seed=bool(entry.get("roadmap_seed", True)),
    )
    findings.append(finding)

    if repo_root is not None:
        abs_path = Path(repo_root) / source.path
        if abs_path.is_file() and abs_path.suffix.casefold() in {".py", ".pyi"}:
            try:
                live = analyze_path(abs_path, root=repo_root)
            except IpaError:
                live = ()
            for live_finding in live:
                if live_finding.rule_id != rule.value and live_finding.family != family:
                    # Still bind overlapping live findings of any IPA rule on the span.
                    if not live_finding.source_span.overlaps(source):
                        continue
                if live_finding.source_span.overlaps(source) or live_finding.sink_span.overlaps(
                    sink
                ):
                    findings.append(
                        replace(
                            live_finding,
                            finding_id=f"{live_finding.finding_id}:corpus:{seed_id}",
                            disposition=FindingDisposition.CORPUS_BOUND,
                            corpus_seed_id=seed_id,
                            corpus_defect_id=defect_id,
                            roadmap_seed=True,
                            trace=SourceToSinkTrace(
                                steps=live_finding.trace.steps
                                + (
                                    TraceStep(
                                        kind="corpus_bind",
                                        label=seed_id or defect_id,
                                        detail=family,
                                    ),
                                ),
                                summary=live_finding.trace.summary,
                            ),
                        )
                    )
    return _dedupe_findings(findings)


def analyze_seeded_corpus(
    *,
    corpus_path: Union[str, Path],
    repo_root: Union[str, Path, None] = None,
    seed_ids: Optional[Sequence[str]] = None,
    souffle_capability: Optional[SouffleCapabilityRecord] = None,
    refinements: Optional[Sequence[SpuriousPathRefinement]] = None,
) -> IpaAnalysisReport:
    """Analyze every IPA-relevant seeded defect from the FACP-008 corpus."""

    entries = ipa_corpus_entries(load_defect_corpus(corpus_path))
    if seed_ids is not None:
        wanted = {str(item) for item in seed_ids}
        entries = tuple(entry for entry in entries if str(entry.get("seed_id")) in wanted)
        missing = wanted - {str(entry.get("seed_id")) for entry in entries}
        if missing:
            raise IpaError(f"requested IPA seed ids not found: {sorted(missing)}")

    capability = souffle_capability or probe_souffle_capability()
    findings: list[IpaFinding] = []
    scanned: list[str] = []
    for entry in entries:
        findings.extend(findings_for_corpus_entry(entry, repo_root=repo_root))
        for span in entry.get("source_spans") or []:
            if isinstance(span, Mapping) and span.get("path"):
                scanned.append(_normalize_relpath(str(span["path"])))

    deduped = _dedupe_findings(findings)
    refined, refined_ids = refine_spurious_paths(deduped, refinements or ())
    corpus_ids = tuple(
        sorted({item.corpus_seed_id for item in refined if item.corpus_seed_id})
    )
    # Sanity: every requested/IPA seed must remain present after refinement.
    bound_active = {
        item.corpus_seed_id
        for item in refined
        if item.corpus_seed_id
        and item.disposition is not FindingDisposition.REFINED_AWAY
    }
    expected_ids = {str(entry.get("seed_id")) for entry in entries}
    missing_after = expected_ids - bound_active
    if missing_after:
        raise IpaError(
            "IPA corpus analysis lost seeds after refinement: "
            + ", ".join(sorted(missing_after))
        )

    backend = (
        "souffle"
        if capability.available
        else capability.analysis_backend or "hermetic_reference_evaluator"
    )
    return IpaAnalysisReport(
        findings=refined,
        scanned_paths=tuple(sorted(set(scanned))),
        corpus_seed_ids_bound=corpus_ids,
        souffle_capability=capability,
        analysis_backend=backend,
        refined_away_finding_ids=refined_ids,
    )


# ---------------------------------------------------------------------------
# CEGAR refinement (spurious paths only; never suppress seeds)
# ---------------------------------------------------------------------------


def refine_spurious_paths(
    findings: Sequence[IpaFinding],
    refinements: Sequence[SpuriousPathRefinement],
) -> tuple[tuple[IpaFinding, ...], tuple[str, ...]]:
    """Apply CEGAR refinements to imprecise/spurious paths.

    Corpus seeds are never refined away, even if a refinement targets them.
    """

    by_id = {item.finding_id: item for item in findings}
    refined_away: list[str] = []
    updates: dict[str, IpaFinding] = {}

    for refinement in refinements:
        target = by_id.get(refinement.finding_id)
        if target is None:
            continue
        if target.is_corpus_seed:
            # Explicit no-op: seeds survive refinement attempts.
            updates[target.finding_id] = replace(
                target,
                refinement_note=(
                    f"refinement {refinement.refinement_id} refused: corpus seed"
                ),
            )
            continue
        if (
            target.disposition is FindingDisposition.SPURIOUS_CANDIDATE
            or target.imprecise
            or target.disposition is FindingDisposition.REJECT
        ):
            updates[target.finding_id] = replace(
                target,
                disposition=FindingDisposition.REFINED_AWAY,
                refinement_note=f"{refinement.refinement_id}: {refinement.reason}",
                imprecise=True,
            )
            refined_away.append(target.finding_id)

    result: list[IpaFinding] = []
    for item in findings:
        result.append(updates.get(item.finding_id, item))
    return tuple(result), tuple(sorted(set(refined_away)))


def mark_imprecise(
    finding: IpaFinding,
    *,
    note: str = "imprecise dynamic dispatch",
) -> IpaFinding:
    """Mark a non-seed finding as a spurious CEGAR candidate."""

    if finding.is_corpus_seed:
        return finding
    return replace(
        finding,
        disposition=FindingDisposition.SPURIOUS_CANDIDATE,
        imprecise=True,
        refinement_note=note,
    )


def analyze_with_capability(
    source: str,
    *,
    path: str = "<memory>.py",
    language: str = "python",
    souffle_capability: Optional[SouffleCapabilityRecord] = None,
) -> IpaAnalysisReport:
    """Analyze one source buffer, always attaching a Souffle capability record."""

    capability = souffle_capability or probe_souffle_capability()
    if language.casefold() in {"python", "py"}:
        findings = analyze_python_source(source, path=path)
    elif language.casefold() in {"typescript", "ts", "tsx"}:
        findings = analyze_typescript_source(source, path=path)
    else:
        raise IpaError(f"unsupported language: {language!r}")

    backend = (
        "souffle"
        if capability.available
        else capability.analysis_backend or "hermetic_reference_evaluator"
    )
    # Even when Souffle is absent, hermetic evaluation already ran inside
    # analyze_python_source — analysis is not skipped.
    return IpaAnalysisReport(
        findings=findings,
        scanned_paths=(_normalize_relpath(path),),
        corpus_seed_ids_bound=(),
        souffle_capability=capability,
        analysis_backend=backend,
    )


__all__ = [
    "ANALYZER_VERSION",
    "BUNDLE",
    "CapabilityDisposition",
    "DatalogAtom",
    "DatalogEvaluationResult",
    "DatalogRule",
    "EffectAbstract",
    "EVIDENCE_SCHEMA",
    "FindingDisposition",
    "GOAL_ID",
    "HERMETIC_EVALUATOR_ID",
    "HermeticReferenceEvaluator",
    "IPA_CORPUS_FAMILIES",
    "IdentityAbstract",
    "IpaAnalysisReport",
    "IpaError",
    "IpaFinding",
    "IpaRuleId",
    "ProductDomainState",
    "ResultAbstract",
    "SCHEMA",
    "SOUFFLE_TOOL_ID",
    "STABLE_RULE_IDS",
    "SourceSpan",
    "SourceToSinkTrace",
    "SouffleCapabilityRecord",
    "SouffleStatus",
    "SpuriousPathRefinement",
    "TASK_ID",
    "TraceStep",
    "TrustAbstract",
    "analyze_path",
    "analyze_python_source",
    "analyze_seeded_corpus",
    "analyze_tree",
    "analyze_typescript_source",
    "analyze_with_capability",
    "default_ipa_datalog_rules",
    "findings_for_corpus_entry",
    "ipa_corpus_entries",
    "load_defect_corpus",
    "mark_imprecise",
    "probe_souffle_capability",
    "refine_spurious_paths",
]
