"""Case records and independently declared oracles for SCG-040.

Interface surface: SemanticGovernorFixtureCorpus@1 case payloads.

Oracles are reviewed fixture authority. They describe scanner-visible
symbols/paths and expected omission/outcome labels. They are never derived
from governor, harness, model, or receipt observations under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CASE_SCHEMA = "scg/fixture-case@1"
SCANNER_VIEW_SCHEMA = "scg/scanner-view@1"
OMISSION_ORACLE_SCHEMA = "scg/omission-oracle@1"
OUTCOME_ORACLE_SCHEMA = "scg/outcome-oracle@1"

PARTITIONS = ("calibration", "development", "held_out")

TASK_FAMILIES = (
    "local_bug",
    "exception",
    "api_migration",
    "schema_migration",
    "state",
    "configuration",
    "fixture",
    "dynamic_import",
    "monkey_patch",
    "generated",
    "plugin",
    "refactor",
    "documentation",
    "proof",
)

ADVERSARIAL_SCENARIOS = (
    "hidden_callee_side_effect",
    "caller_exception_contract",
    "config_flag",
    "pytest_fixture",
    "serializer",
    "generated_interface",
    "stale_capsule",
    "confidence_misclassification",
    "opaque_dynamic_import",
    "behavior_only_dependency",
    "security_invariant",
    "migration_path",
    "misleading_comment",
    "prompt_injection",
    "selected_pass_full_fail",
    "test_pass_formal_fail",
    "raw_correct_compressed_wrong",
    "both_context_model_failure",
)

CONFIDENCE_CLASSES = frozenset({"exact", "conservative", "heuristic", "opaque"})
OUTCOME_LABELS = frozenset(
    {
        "sufficient",
        "insufficient_omission",
        "insufficient_model",
        "inconclusive",
        "human_review_required",
        "reject_stale",
        "reject_injection",
        "verification_conflict",
    }
)
DIAGNOSIS_LABELS = frozenset(
    {
        "none",
        "omission",
        "model_insufficiency",
        "stale_artifact",
        "confidence_error",
        "security",
        "injection",
        "verification_conflict",
        "dynamic_opacity",
    }
)
PATH_OPS = frozenset({"replace", "add", "delete", "rename"})


class FixtureCorpusError(ValueError):
    """Closed fixture-record violation."""


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FixtureCorpusError(f"{name} must be a nonempty trimmed string")
    return value


def _sorted_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_text(item, f"{name}[]") for item in values)
    if len(set(items)) != len(items):
        raise FixtureCorpusError(f"{name} must not contain duplicates")
    ordered = tuple(sorted(items))
    if ordered != items:
        raise FixtureCorpusError(f"{name} must be sorted")
    return ordered


@dataclass(frozen=True)
class PathOperation:
    """Single path mutation applied to the shared base tree."""

    op: str
    path: str
    content: str | None = None
    from_path: str | None = None

    def __post_init__(self) -> None:
        op = _text(self.op, "op")
        if op not in PATH_OPS:
            raise FixtureCorpusError(f"unsupported op {op!r}")
        path = _text(self.path, "path")
        if path.startswith("/") or ".." in path.split("/"):
            raise FixtureCorpusError(f"illegal path {path!r}")
        content = self.content
        from_path = self.from_path
        if op in {"replace", "add"}:
            if type(content) is not str:
                raise FixtureCorpusError(f"{op} requires string content")
        else:
            if content is not None:
                raise FixtureCorpusError(f"{op} must not carry content")
        if op == "rename":
            if type(from_path) is not str or not from_path:
                raise FixtureCorpusError("rename requires from_path")
            if from_path.startswith("/") or ".." in from_path.split("/"):
                raise FixtureCorpusError(f"illegal from_path {from_path!r}")
        else:
            if from_path is not None:
                raise FixtureCorpusError(f"{op} must not carry from_path")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "from_path", from_path)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.op, "path": self.path}
        if self.content is not None:
            payload["content"] = self.content
        if self.from_path is not None:
            payload["from_path"] = self.from_path
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathOperation":
        return cls(
            op=str(payload["op"]),
            path=str(payload["path"]),
            content=payload.get("content"),
            from_path=payload.get("from_path"),
        )


@dataclass(frozen=True)
class ScannerView:
    """Independently declared scanner-visible identity for a case.

    Fields mirror canonical scanner vocabulary (qualified symbol names,
    repository-relative paths, confidence classes). Values are reviewed
    fixture authority and must stay consistent with a byte-level scan of the
    materialised tree; they are not harvested from the governor under test.
    """

    changed_paths: tuple[str, ...]
    changed_symbols: tuple[str, ...]
    primary_symbol: str
    dependency_symbols: tuple[str, ...]
    context_symbols: tuple[str, ...]
    confidence: str
    opaque_symbols: tuple[str, ...]
    relation_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        paths = _sorted_unique(self.changed_paths, "changed_paths")
        symbols = _sorted_unique(self.changed_symbols, "changed_symbols")
        primary = _text(self.primary_symbol, "primary_symbol")
        if primary not in symbols:
            raise FixtureCorpusError("primary_symbol must appear in changed_symbols")
        deps = _sorted_unique(self.dependency_symbols, "dependency_symbols")
        context = _sorted_unique(self.context_symbols, "context_symbols")
        confidence = _text(self.confidence, "confidence")
        if confidence not in CONFIDENCE_CLASSES:
            raise FixtureCorpusError(f"unsupported confidence {confidence!r}")
        opaque = _sorted_unique(self.opaque_symbols, "opaque_symbols")
        relations = _sorted_unique(self.relation_kinds, "relation_kinds")
        object.__setattr__(self, "changed_paths", paths)
        object.__setattr__(self, "changed_symbols", symbols)
        object.__setattr__(self, "primary_symbol", primary)
        object.__setattr__(self, "dependency_symbols", deps)
        object.__setattr__(self, "context_symbols", context)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "opaque_symbols", opaque)
        object.__setattr__(self, "relation_kinds", relations)

    def symbol_universe(self) -> frozenset[str]:
        """Scanner-derived identities available for omission/outcome oracles."""

        return frozenset(
            set(self.changed_symbols)
            | set(self.dependency_symbols)
            | set(self.context_symbols)
            | set(self.opaque_symbols)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCANNER_VIEW_SCHEMA,
            "changed_paths": list(self.changed_paths),
            "changed_symbols": list(self.changed_symbols),
            "primary_symbol": self.primary_symbol,
            "dependency_symbols": list(self.dependency_symbols),
            "context_symbols": list(self.context_symbols),
            "confidence": self.confidence,
            "opaque_symbols": list(self.opaque_symbols),
            "relation_kinds": list(self.relation_kinds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScannerView":
        schema = payload.get("schema")
        if schema is not None and schema != SCANNER_VIEW_SCHEMA:
            raise FixtureCorpusError(f"unsupported scanner view schema {schema!r}")
        return cls(
            changed_paths=tuple(payload["changed_paths"]),
            changed_symbols=tuple(payload["changed_symbols"]),
            primary_symbol=str(payload["primary_symbol"]),
            dependency_symbols=tuple(payload.get("dependency_symbols") or ()),
            context_symbols=tuple(payload.get("context_symbols") or ()),
            confidence=str(payload["confidence"]),
            opaque_symbols=tuple(payload.get("opaque_symbols") or ()),
            relation_kinds=tuple(payload.get("relation_kinds") or ()),
        )


@dataclass(frozen=True)
class OmissionOracle:
    """Independently declared intentional omission / inclusion oracle."""

    critical_omitted_symbols: tuple[str, ...]
    noncritical_omitted_symbols: tuple[str, ...]
    compressed_includes: tuple[str, ...]
    compressed_omits: tuple[str, ...]
    intentional_critical: bool
    expansion_targets: tuple[str, ...]

    def __post_init__(self) -> None:
        critical = _sorted_unique(
            self.critical_omitted_symbols, "critical_omitted_symbols"
        )
        noncritical = _sorted_unique(
            self.noncritical_omitted_symbols, "noncritical_omitted_symbols"
        )
        if set(critical) & set(noncritical):
            raise FixtureCorpusError(
                "critical and noncritical omission sets must be disjoint"
            )
        includes = _sorted_unique(self.compressed_includes, "compressed_includes")
        omits = _sorted_unique(self.compressed_omits, "compressed_omits")
        if set(includes) & set(omits):
            raise FixtureCorpusError(
                "compressed_includes and compressed_omits must be disjoint"
            )
        if type(self.intentional_critical) is not bool:
            raise FixtureCorpusError("intentional_critical must be a bool")
        if self.intentional_critical and not critical:
            raise FixtureCorpusError(
                "intentional_critical requires critical_omitted_symbols"
            )
        expansion = _sorted_unique(self.expansion_targets, "expansion_targets")
        object.__setattr__(self, "critical_omitted_symbols", critical)
        object.__setattr__(self, "noncritical_omitted_symbols", noncritical)
        object.__setattr__(self, "compressed_includes", includes)
        object.__setattr__(self, "compressed_omits", omits)
        object.__setattr__(self, "expansion_targets", expansion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OMISSION_ORACLE_SCHEMA,
            "critical_omitted_symbols": list(self.critical_omitted_symbols),
            "noncritical_omitted_symbols": list(self.noncritical_omitted_symbols),
            "compressed_includes": list(self.compressed_includes),
            "compressed_omits": list(self.compressed_omits),
            "intentional_critical": self.intentional_critical,
            "expansion_targets": list(self.expansion_targets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OmissionOracle":
        schema = payload.get("schema")
        if schema is not None and schema != OMISSION_ORACLE_SCHEMA:
            raise FixtureCorpusError(f"unsupported omission schema {schema!r}")
        return cls(
            critical_omitted_symbols=tuple(
                payload.get("critical_omitted_symbols") or ()
            ),
            noncritical_omitted_symbols=tuple(
                payload.get("noncritical_omitted_symbols") or ()
            ),
            compressed_includes=tuple(payload.get("compressed_includes") or ()),
            compressed_omits=tuple(payload.get("compressed_omits") or ()),
            intentional_critical=bool(payload["intentional_critical"]),
            expansion_targets=tuple(payload.get("expansion_targets") or ()),
        )


@dataclass(frozen=True)
class OutcomeOracle:
    """Independently declared expected governor outcome for a case."""

    expected_outcome: str
    expected_diagnosis: str
    automatic_accept_allowed: bool
    reason_codes: tuple[str, ...]
    selected_tests: tuple[str, ...]
    full_suite_tests: tuple[str, ...]
    proof_obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        outcome = _text(self.expected_outcome, "expected_outcome")
        if outcome not in OUTCOME_LABELS:
            raise FixtureCorpusError(f"unsupported expected_outcome {outcome!r}")
        diagnosis = _text(self.expected_diagnosis, "expected_diagnosis")
        if diagnosis not in DIAGNOSIS_LABELS:
            raise FixtureCorpusError(f"unsupported expected_diagnosis {diagnosis!r}")
        if type(self.automatic_accept_allowed) is not bool:
            raise FixtureCorpusError("automatic_accept_allowed must be a bool")
        # Critical fail-closed outcomes never auto-accept.
        if outcome in {
            "insufficient_omission",
            "human_review_required",
            "reject_stale",
            "reject_injection",
            "verification_conflict",
        } and self.automatic_accept_allowed:
            raise FixtureCorpusError(
                f"{outcome} must not allow automatic acceptance"
            )
        reasons = _sorted_unique(self.reason_codes, "reason_codes")
        selected = _sorted_unique(self.selected_tests, "selected_tests")
        full = _sorted_unique(self.full_suite_tests, "full_suite_tests")
        if not set(selected).issubset(set(full)):
            raise FixtureCorpusError(
                "selected_tests must be a subset of full_suite_tests"
            )
        proofs = _sorted_unique(self.proof_obligations, "proof_obligations")
        object.__setattr__(self, "expected_outcome", outcome)
        object.__setattr__(self, "expected_diagnosis", diagnosis)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "selected_tests", selected)
        object.__setattr__(self, "full_suite_tests", full)
        object.__setattr__(self, "proof_obligations", proofs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OUTCOME_ORACLE_SCHEMA,
            "expected_outcome": self.expected_outcome,
            "expected_diagnosis": self.expected_diagnosis,
            "automatic_accept_allowed": self.automatic_accept_allowed,
            "reason_codes": list(self.reason_codes),
            "selected_tests": list(self.selected_tests),
            "full_suite_tests": list(self.full_suite_tests),
            "proof_obligations": list(self.proof_obligations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutcomeOracle":
        schema = payload.get("schema")
        if schema is not None and schema != OUTCOME_ORACLE_SCHEMA:
            raise FixtureCorpusError(f"unsupported outcome schema {schema!r}")
        return cls(
            expected_outcome=str(payload["expected_outcome"]),
            expected_diagnosis=str(payload["expected_diagnosis"]),
            automatic_accept_allowed=bool(payload["automatic_accept_allowed"]),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            selected_tests=tuple(payload.get("selected_tests") or ()),
            full_suite_tests=tuple(payload.get("full_suite_tests") or ()),
            proof_obligations=tuple(payload.get("proof_obligations") or ()),
        )


@dataclass(frozen=True)
class FixtureCase:
    """One partitioned controlled fixture case with independent oracles."""

    case_id: str
    partition: str
    family: str
    description: str
    operations: tuple[PathOperation, ...]
    scanner_view: ScannerView
    omission: OmissionOracle
    outcome: OutcomeOracle
    adversarial_scenario: str | None = None
    production_eligible: bool = False

    def __post_init__(self) -> None:
        case_id = _text(self.case_id, "case_id")
        partition = _text(self.partition, "partition")
        if partition not in PARTITIONS:
            raise FixtureCorpusError(f"unsupported partition {partition!r}")
        family = _text(self.family, "family")
        if family not in TASK_FAMILIES:
            raise FixtureCorpusError(f"unsupported family {family!r}")
        description = _text(self.description, "description")
        if not self.operations:
            raise FixtureCorpusError(f"{case_id}: operations must be non-empty")
        if not isinstance(self.scanner_view, ScannerView):
            raise FixtureCorpusError("scanner_view must be a ScannerView")
        if not isinstance(self.omission, OmissionOracle):
            raise FixtureCorpusError("omission must be an OmissionOracle")
        if not isinstance(self.outcome, OutcomeOracle):
            raise FixtureCorpusError("outcome must be an OutcomeOracle")
        scenario = self.adversarial_scenario
        if scenario is not None:
            scenario = _text(scenario, "adversarial_scenario")
            if scenario not in ADVERSARIAL_SCENARIOS:
                raise FixtureCorpusError(
                    f"unsupported adversarial_scenario {scenario!r}"
                )
        if type(self.production_eligible) is not bool:
            raise FixtureCorpusError("production_eligible must be a bool")
        if self.production_eligible:
            raise FixtureCorpusError(
                "fixture cases are oracle/replay only; production_eligible must be false"
            )
        # Omission symbols must be scanner-derived identities.
        scanner_universe = set(self.scanner_view.symbol_universe())
        for group_name, group in (
            ("critical_omitted_symbols", self.omission.critical_omitted_symbols),
            (
                "noncritical_omitted_symbols",
                self.omission.noncritical_omitted_symbols,
            ),
            ("compressed_includes", self.omission.compressed_includes),
            ("compressed_omits", self.omission.compressed_omits),
            ("expansion_targets", self.omission.expansion_targets),
        ):
            unknown = set(group) - scanner_universe
            if unknown:
                raise FixtureCorpusError(
                    f"{case_id}: {group_name} not scanner-derived: "
                    f"{sorted(unknown)}"
                )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "adversarial_scenario", scenario)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CASE_SCHEMA,
            "case_id": self.case_id,
            "partition": self.partition,
            "family": self.family,
            "description": self.description,
            "operations": [op.to_dict() for op in self.operations],
            "scanner_view": self.scanner_view.to_dict(),
            "omission": self.omission.to_dict(),
            "outcome": self.outcome.to_dict(),
            "adversarial_scenario": self.adversarial_scenario,
            "production_eligible": self.production_eligible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixtureCase":
        schema = payload.get("schema")
        if schema is not None and schema != CASE_SCHEMA:
            raise FixtureCorpusError(f"unsupported case schema {schema!r}")
        return cls(
            case_id=str(payload["case_id"]),
            partition=str(payload["partition"]),
            family=str(payload["family"]),
            description=str(payload["description"]),
            operations=tuple(
                PathOperation.from_dict(item) for item in payload["operations"]
            ),
            scanner_view=ScannerView.from_dict(payload["scanner_view"]),
            omission=OmissionOracle.from_dict(payload["omission"]),
            outcome=OutcomeOracle.from_dict(payload["outcome"]),
            adversarial_scenario=payload.get("adversarial_scenario"),
            production_eligible=bool(payload.get("production_eligible", False)),
        )
