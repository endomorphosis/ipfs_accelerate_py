"""Frozen, lineage-safe paired residual benchmark contracts.

The benchmark is an evaluation artifact, not a training admission or a source
of authority. Its compact case records carry identities rather than prompts,
hidden-test bodies, or provider output. Roots cover the complete catalog and
every partition, preventing an unobserved held-out/adversarial substitution.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    canonical_id,
    required_text,
    strict_fields,
)

MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-intelligence-benchmark-manifest@1"
)
CASE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-frozen-benchmark-case@1"
LINEAGE_ROOT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-frozen-benchmark-lineage@1"
PARTITIONS: Final[tuple[str, ...]] = ("training", "development", "held_out", "adversarial")
REQUIRED_KINDS: Final[tuple[str, ...]] = (
    "boundary",
    "negative",
    "cross_repository",
    "unknown_ood",
)
REQUIRED_BINDINGS: Final[tuple[str, ...]] = (
    "repository_states",
    "objective_revisions",
    "operation_catalog",
    "provider_policy",
    "tokenizer",
    "model_versions",
    "fault_schedule",
    "validation_policy",
)
IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_identity",
    "objective_identity",
    "operation_catalog_identity",
    "provider_policy_identity",
    "tokenizer_identity",
    "model_identity",
    "fault_identity",
    "validation_identity",
)
_HIDDEN_PARTITIONS: Final[frozenset[str]] = frozenset({"held_out", "adversarial"})
_KIND_DISPOSITIONS: Final[dict[str, ExpertDisposition]] = {
    "boundary": ExpertDisposition.ABSTAIN,
    "negative": ExpertDisposition.REJECT_INPUT,
    "cross_repository": ExpertDisposition.REJECT_INPUT,
    "unknown_ood": ExpertDisposition.OUT_OF_DISTRIBUTION,
}


def _strict_json_loads(text: str, *, noun: str) -> Any:
    """Decode JSON while rejecting duplicate object keys at every depth."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ResidualIntelligenceError(f"{noun} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ResidualIntelligenceError(f"invalid {noun} JSON: {exc.msg}") from exc


def _text_mapping(value: Any, *, name: str, keys: Sequence[str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ResidualIntelligenceError(f"{name} must bind exactly: {', '.join(keys)}")
    return {key: required_text(value[key], f"{name}.{key}") for key in keys}


@dataclass(frozen=True)
class FrozenBenchmarkCase:
    """One fully pinned benchmark case without any private test body."""

    family: ResidualTaskFamily
    partition: str
    kind: str
    case_id: str
    lineage_group: str
    repository_identity: str
    objective_identity: str
    operation_catalog_identity: str
    provider_policy_identity: str
    tokenizer_identity: str
    model_identity: str
    fault_identity: str
    validation_identity: str
    cross_repository_identity: str = ""
    hidden_test: bool = False
    expected_disposition: ExpertDisposition = ExpertDisposition.ABSTAIN
    schema: str = CASE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "family",
            "partition",
            "kind",
            "case_id",
            "lineage_group",
            *IDENTITY_FIELDS,
            "cross_repository_identity",
            "hidden_test",
            "expected_disposition",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != CASE_SCHEMA:
            raise ResidualIntelligenceError("unsupported frozen benchmark case schema")
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        partition = required_text(self.partition, "partition")
        if partition not in PARTITIONS:
            raise ResidualIntelligenceError(f"unknown partition: {partition}")
        object.__setattr__(self, "partition", partition)
        kind = required_text(self.kind, "kind")
        if kind not in REQUIRED_KINDS:
            raise ResidualIntelligenceError(f"unknown case kind: {kind}")
        object.__setattr__(self, "kind", kind)
        for field in ("case_id", "lineage_group", *IDENTITY_FIELDS):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        cross_repository = (
            required_text(self.cross_repository_identity, "cross_repository_identity")
            if self.cross_repository_identity
            else ""
        )
        if kind == "cross_repository":
            if not cross_repository or cross_repository == self.repository_identity:
                raise ResidualIntelligenceError(
                    "cross-repository cases require a distinct repository identity"
                )
        elif cross_repository:
            raise ResidualIntelligenceError(
                "only cross-repository cases may bind a second repository"
            )
        object.__setattr__(self, "cross_repository_identity", cross_repository)
        if type(self.hidden_test) is not bool:
            raise ResidualIntelligenceError("hidden_test must be boolean")
        if self.hidden_test != (partition in _HIDDEN_PARTITIONS):
            raise ResidualIntelligenceError(
                "hidden_test must be true exactly for held-out and adversarial partitions"
            )
        disposition = ExpertDisposition(self.expected_disposition)
        if disposition is not _KIND_DISPOSITIONS[kind]:
            raise ResidualIntelligenceError(
                f"{kind} cases must expect {_KIND_DISPOSITIONS[kind].value}"
            )
        object.__setattr__(self, "expected_disposition", disposition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "family": self.family.value,
            "partition": self.partition,
            "kind": self.kind,
            "case_id": self.case_id,
            "lineage_group": self.lineage_group,
            **{field: getattr(self, field) for field in IDENTITY_FIELDS},
            "cross_repository_identity": self.cross_repository_identity,
            "hidden_test": self.hidden_test,
            "expected_disposition": self.expected_disposition.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenBenchmarkCase":
        strict_fields(payload, allowed=cls._FIELDS, required=cls._FIELDS, noun="frozen benchmark case")
        return cls(
            schema=payload.get("schema"), family=payload.get("family"),
            partition=payload.get("partition"), kind=payload.get("kind"),
            case_id=payload.get("case_id"), lineage_group=payload.get("lineage_group"),
            repository_identity=payload.get("repository_identity"),
            objective_identity=payload.get("objective_identity"),
            operation_catalog_identity=payload.get("operation_catalog_identity"),
            provider_policy_identity=payload.get("provider_policy_identity"),
            tokenizer_identity=payload.get("tokenizer_identity"),
            model_identity=payload.get("model_identity"), fault_identity=payload.get("fault_identity"),
            validation_identity=payload.get("validation_identity"),
            cross_repository_identity=payload.get("cross_repository_identity"),
            hidden_test=payload.get("hidden_test"), expected_disposition=payload.get("expected_disposition"),
        )


def case_catalog_root(cases: Sequence[FrozenBenchmarkCase]) -> str:
    """Return the canonical identity of the complete ordered-independent catalog."""

    return canonical_id({
        "schema": CASE_SCHEMA,
        "cases": [item.to_dict() for item in sorted(cases, key=lambda item: item.case_id)],
    })


def partition_roots(cases: Sequence[FrozenBenchmarkCase]) -> dict[str, str]:
    return {
        partition: case_catalog_root(tuple(item for item in cases if item.partition == partition))
        for partition in PARTITIONS
    }


def lineage_root(cases: Sequence[FrozenBenchmarkCase]) -> str:
    return canonical_id({
        "schema": LINEAGE_ROOT_SCHEMA,
        "assignments": [
            {"case_id": item.case_id, "lineage_group": item.lineage_group, "partition": item.partition}
            for item in sorted(cases, key=lambda item: item.case_id)
        ],
    })


@dataclass(frozen=True)
class ResidualBenchmarkManifest:
    """Pinned benchmark metadata and roots for all benchmark views."""

    families: tuple[ResidualTaskFamily, ...]
    partitions: tuple[str, ...]
    frozen_root: str
    source_revision: str
    source_tree: str
    case_catalog_root: str
    partition_roots: Mapping[str, str]
    split_root: str
    program_identifier: str = "agent-supervisor-verified-residual-intelligence-foundry-v1"
    owner_task: str = "VRIF-030"
    status: str = "staged_not_qualified"
    training_admission: str = "training_unavailable"
    schema: str = MANIFEST_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset({
        "schema", "program_identifier", "status", "owner_task", "source_revision", "source_tree",
        "partitions", "required_case_kinds", "task_families", "frozen_bindings_required_before_qualification",
        "frozen_roots", "training_admission",
        "weights_committed", "large_corpus_committed", "promotion_evidence",
    })

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual benchmark manifest schema")
        families = tuple(ResidualTaskFamily(item) for item in self.families)
        if len(families) != len(set(families)) or set(families) != set(ResidualTaskFamily):
            raise ResidualIntelligenceError("benchmark task families must be the exact closed taxonomy")
        object.__setattr__(self, "families", families)
        partitions = tuple(required_text(item, "partition") for item in self.partitions)
        if partitions != PARTITIONS:
            raise ResidualIntelligenceError("benchmark partitions must be exact")
        object.__setattr__(self, "partitions", partitions)
        for field in (
            "frozen_root", "source_revision", "source_tree", "case_catalog_root", "split_root",
            "program_identifier", "owner_task", "status", "training_admission",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        object.__setattr__(self, "partition_roots", _text_mapping(
            self.partition_roots, name="partition_roots", keys=PARTITIONS,
        ))
        if self.program_identifier != "agent-supervisor-verified-residual-intelligence-foundry-v1":
            raise ResidualIntelligenceError("unexpected benchmark program identifier")
        if self.owner_task != "VRIF-030" or self.status != "staged_not_qualified":
            raise ResidualIntelligenceError("benchmark manifest cannot claim qualification")
        if self.training_admission != "training_unavailable":
            raise ResidualIntelligenceError("benchmark does not grant training admission")

    def to_dict(self, *, include_frozen_root: bool = True) -> dict[str, Any]:
        roots: dict[str, Any] = {
            "case_catalog": self.case_catalog_root,
            "partitions": dict(self.partition_roots),
            "semantic_lineage": self.split_root,
        }
        if include_frozen_root:
            roots["benchmark"] = self.frozen_root
        return {
            "schema": self.schema, "program_identifier": self.program_identifier,
            "status": self.status, "owner_task": self.owner_task,
            "source_revision": self.source_revision, "source_tree": self.source_tree,
            "partitions": list(self.partitions), "required_case_kinds": list(REQUIRED_KINDS),
            "task_families": [item.value for item in self.families], "frozen_roots": roots,
            "frozen_bindings_required_before_qualification": list(REQUIRED_BINDINGS),
            "training_admission": self.training_admission, "weights_committed": False,
            "large_corpus_committed": False, "promotion_evidence": False,
        }

    @property
    def computed_frozen_root(self) -> str:
        return canonical_id(self.to_dict(include_frozen_root=False))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualBenchmarkManifest":
        strict_fields(payload, allowed=cls._FIELDS, required=cls._FIELDS, noun="benchmark manifest")
        if payload.get("required_case_kinds") != list(REQUIRED_KINDS):
            raise ResidualIntelligenceError("benchmark required case kinds must be exact")
        if payload.get("frozen_bindings_required_before_qualification") != list(REQUIRED_BINDINGS):
            raise ResidualIntelligenceError("benchmark frozen binding requirements must be exact")
        if any(payload.get(field) is not False for field in (
            "weights_committed", "large_corpus_committed", "promotion_evidence"
        )):
            raise ResidualIntelligenceError("benchmark cannot claim weights, corpus, or promotion evidence")
        roots = payload.get("frozen_roots")
        if not isinstance(roots, Mapping):
            raise ResidualIntelligenceError("benchmark frozen_roots must be an object")
        strict_fields(roots, allowed={"benchmark", "case_catalog", "partitions", "semantic_lineage"},
                      required={"benchmark", "case_catalog", "partitions", "semantic_lineage"},
                      noun="benchmark frozen_roots")
        return cls(
            schema=payload.get("schema"), program_identifier=payload.get("program_identifier"),
            status=payload.get("status"), owner_task=payload.get("owner_task"),
            source_revision=payload.get("source_revision"), source_tree=payload.get("source_tree"),
            partitions=tuple(payload.get("partitions") or ()), families=tuple(payload.get("task_families") or ()),
            frozen_root=roots.get("benchmark"), case_catalog_root=roots.get("case_catalog"),
            partition_roots=roots.get("partitions"), split_root=roots.get("semantic_lineage"),
            training_admission=payload.get("training_admission"),
        )


def validate_frozen_benchmark(manifest: ResidualBenchmarkManifest, cases: Sequence[FrozenBenchmarkCase]) -> None:
    """Fail closed unless the catalog is a complete, pinned paired population."""

    typed = tuple(cases)
    if any(not isinstance(item, FrozenBenchmarkCase) for item in typed):
        raise ResidualIntelligenceError("benchmark cases must be typed frozen cases")
    expected = {
        (family, partition, kind)
        for family in manifest.families for partition in PARTITIONS for kind in REQUIRED_KINDS
    }
    actual = {(item.family, item.partition, item.kind) for item in typed}
    if actual != expected or len(typed) != len(expected):
        missing = sorted(
            f"{family.value}/{partition}/{kind}" for family, partition, kind in expected - actual
        )
        raise ResidualIntelligenceError(
            "benchmark coverage must contain one case for every family/partition/kind; "
            f"missing={missing}; duplicate_or_extra={len(typed) - len(actual)}"
        )
    ids = [item.case_id for item in typed]
    if len(ids) != len(set(ids)):
        raise ResidualIntelligenceError("benchmark contains duplicate case identities")
    lineage_partitions: dict[str, set[str]] = defaultdict(set)
    for item in typed:
        lineage_partitions[item.lineage_group].add(item.partition)
    mixed = sorted(group for group, partitions in lineage_partitions.items() if len(partitions) != 1)
    if mixed:
        raise ResidualIntelligenceError(
            f"semantic lineage crosses benchmark partitions: {', '.join(mixed)}"
        )
    if manifest.case_catalog_root != case_catalog_root(typed):
        raise ResidualIntelligenceError("benchmark case catalog root does not verify")
    if dict(manifest.partition_roots) != partition_roots(typed):
        raise ResidualIntelligenceError("benchmark partition roots do not verify")
    if manifest.split_root != lineage_root(typed):
        raise ResidualIntelligenceError("benchmark semantic lineage root does not verify")
    if manifest.frozen_root != manifest.computed_frozen_root:
        raise ResidualIntelligenceError("benchmark frozen root does not verify")


@dataclass(frozen=True)
class PairedBenchmarkRunner:
    """Evaluate the exact frozen population with both baseline result vectors."""

    def evaluate(self, manifest: ResidualBenchmarkManifest, cases: Sequence[FrozenBenchmarkCase], *,
                 prior: Mapping[str, int], current: Mapping[str, int]) -> dict[str, Any]:
        validate_frozen_benchmark(manifest, cases)
        if not isinstance(prior, Mapping) or not isinstance(current, Mapping):
            raise ResidualIntelligenceError("paired benchmark results must be mappings")
        if set(prior) != set(current):
            raise ResidualIntelligenceError("paired benchmark results must use identical metrics")
        for label, result_set in (("prior", prior), ("current", current)):
            for metric, value in result_set.items():
                required_text(metric, f"{label} metric", max_bytes=256)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ResidualIntelligenceError(
                        f"{label}.{metric} must be a non-negative integer"
                    )
        denominators = {
            family.value: sum(1 for item in cases if item.family is family)
            for family in manifest.families
        }
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/residual-paired-benchmark-result@1",
            "frozen_root": manifest.frozen_root, "case_catalog_root": manifest.case_catalog_root,
            "partition_roots": dict(manifest.partition_roots), "prior": dict(prior),
            "current": dict(current), "denominators": denominators,
            "total_denominator": sum(denominators.values()), "candidate_only": True,
        }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a strict manifest payload without treating it as validated evidence."""

    payload = _strict_json_loads(path.read_text(encoding="utf-8"), noun="benchmark manifest")
    if not isinstance(payload, dict):
        raise ResidualIntelligenceError("benchmark manifest must be an object")
    return payload


def load_cases(path: Path) -> tuple[FrozenBenchmarkCase, ...]:
    """Load strict JSONL cases; blank lines are rejected to preserve the root."""

    cases: list[FrozenBenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ResidualIntelligenceError(f"benchmark cases line {line_number} is blank")
        payload = _strict_json_loads(line, noun=f"benchmark case line {line_number}")
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError(f"benchmark case line {line_number} must be an object")
        cases.append(FrozenBenchmarkCase.from_dict(payload))
    if not cases:
        raise ResidualIntelligenceError("benchmark cases cannot be empty")
    return tuple(cases)


def load_frozen_benchmark(manifest_path: Path, cases_path: Path) -> tuple[ResidualBenchmarkManifest, tuple[FrozenBenchmarkCase, ...]]:
    """Load and verify the complete immutable benchmark bundle."""

    manifest = ResidualBenchmarkManifest.from_dict(load_manifest(manifest_path))
    cases = load_cases(cases_path)
    validate_frozen_benchmark(manifest, cases)
    return manifest, cases


__all__ = (
    "CASE_SCHEMA", "IDENTITY_FIELDS", "LINEAGE_ROOT_SCHEMA", "MANIFEST_SCHEMA", "PARTITIONS",
    "REQUIRED_BINDINGS", "REQUIRED_KINDS", "FrozenBenchmarkCase", "PairedBenchmarkRunner", "ResidualBenchmarkManifest",
    "case_catalog_root", "lineage_root", "load_cases", "load_frozen_benchmark", "load_manifest",
    "partition_roots", "validate_frozen_benchmark",
)
