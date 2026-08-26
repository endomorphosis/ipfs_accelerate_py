"""Owner-exact frozen residual benchmark contracts.

The VRIF program has no admitted training rows, learned tokenizer, or model.
Its benchmark is therefore a content-addressed, all-abstain record rather than
evidence of learned capability. This module mirrors the independent owner
construction so producer artifacts can be checked without importing operator
state or authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from .contracts import (
    PROGRAM_ID,
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    bounded_json_mapping,
    required_text,
    strict_fields,
)

MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-intelligence-benchmark-manifest@1"
)
CASE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-frozen-benchmark-case@2"
LINEAGE_ROOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-benchmark-lineage-group@1"
)
FAULT_SCHEDULE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-benchmark-fault-schedule@1"
)
UNAVAILABLE_INPUT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-benchmark-unavailable-input@1"
)
PAIRED_BASELINE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-paired-benchmark-baseline@2"
)
BENCHMARK_FREEZE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-benchmark-freeze@2"
)
PARTITIONS: Final[tuple[str, ...]] = (
    "training",
    "development",
    "held_out",
    "adversarial",
)
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
BASE_BINDINGS: Final[tuple[str, ...]] = tuple(
    item for item in REQUIRED_BINDINGS if item != "fault_schedule"
)
# Retained as a useful public description of the identities carried by @2 cases.
IDENTITY_FIELDS: Final[tuple[str, ...]] = ("group_id", "input_identity")

_HIDDEN_PARTITIONS: Final[frozenset[str]] = frozenset({"held_out", "adversarial"})
_INPUT_DISPOSITION: Final = "payload_unavailable_training_unavailable"
_EVALUATION_DISPOSITION: Final = "all_abstain_not_run"
_COMPARISON_DISPOSITION: Final = "identical_no_candidate_training_unavailable"
_SHA256_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}")

_CASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "family",
        "partition",
        "kind",
        "hidden_test",
        "group_id",
        "input_identity",
        "input_disposition",
        "expected_outcome",
        "case_id",
    }
)
_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "program_identifier",
        "status",
        "owner_task",
        "source_revision",
        "partitions",
        "required_case_kinds",
        "task_families",
        "training_admission",
        "weights_committed",
        "large_corpus_committed",
        "promotion_evidence",
        "benchmark_freeze",
    }
)
_SOURCE_FIELDS: Final[frozenset[str]] = frozenset({"commit", "tree"})
_SCHEDULE_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {"family", "partition", "kind", "hidden_test", "group_id"}
)
_FAULT_SCHEDULE_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema", "source_tree", "split_root", "entries", "schedule_id"}
)
_SCORE_FIELDS: Final[frozenset[str]] = frozenset(
    {"accept", "abstain", "total", "denominators_by_family"}
)
_PAIRED_BASELINE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "prior_source",
        "evaluated_source",
        "comparison_disposition",
        "case_payload_disposition",
        "evaluation_disposition",
        "case_count",
        "case_root",
        "binding_set_id",
        "before",
        "after",
        "candidate_only",
        "training_performed",
        "paired_baseline_id",
    }
)
_BENCHMARK_FREEZE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "state",
        "source",
        "case_payload_disposition",
        "evaluation_disposition",
        "bindings",
        "binding_set_id",
        "fault_schedule",
        "case_count",
        "case_root",
        "paired_baseline",
        "freeze_id",
    }
)


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


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResidualIntelligenceError("benchmark value is not canonical JSON") from exc


def sha256_identity(value: Any) -> str:
    """Return the exact identity format used by the independent VRIF owner."""

    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _identity_text(value: Any, name: str) -> str:
    text = required_text(value, name)
    if _SHA256_ID_RE.fullmatch(text) is None:
        raise ResidualIntelligenceError(f"{name} must be a sha256 identity")
    return text


def _git_object(value: Any, name: str) -> str:
    text = required_text(value, name)
    if _GIT_OBJECT_RE.fullmatch(text) is None:
        raise ResidualIntelligenceError(f"{name} must be a 40-character Git object identity")
    return text


def _task_families(values: Any) -> tuple[ResidualTaskFamily, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("task_families must be a sequence")
    try:
        families = tuple(ResidualTaskFamily(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ResidualIntelligenceError("task_families contains an unknown family") from exc
    expected = tuple(ResidualTaskFamily)
    if families != expected:
        raise ResidualIntelligenceError(
            "benchmark task_families must be the exact closed taxonomy in canonical order"
        )
    return families


def _text_bindings(value: Any, *, include_fault_schedule: bool) -> dict[str, str]:
    keys = REQUIRED_BINDINGS if include_fault_schedule else BASE_BINDINGS
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ResidualIntelligenceError(
            "benchmark bindings must bind exactly: " + ", ".join(keys)
        )
    return {key: _identity_text(value[key], f"bindings.{key}") for key in keys}


@dataclass(frozen=True)
class FrozenBenchmarkCase:
    """One owner-computable unavailable-input case from the @2 schedule."""

    family: ResidualTaskFamily
    partition: str
    kind: str
    hidden_test: bool
    group_id: str
    input_identity: str
    input_disposition: str
    expected_outcome: ExpertDisposition
    case_id: str
    schema: str = CASE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = _CASE_FIELDS

    def __post_init__(self) -> None:
        if self.schema != CASE_SCHEMA:
            raise ResidualIntelligenceError("unsupported frozen benchmark case schema")
        try:
            family = ResidualTaskFamily(self.family)
        except (TypeError, ValueError) as exc:
            raise ResidualIntelligenceError("frozen benchmark case family is unknown") from exc
        object.__setattr__(self, "family", family)
        partition = required_text(self.partition, "partition")
        kind = required_text(self.kind, "kind")
        if partition not in PARTITIONS:
            raise ResidualIntelligenceError(f"unknown partition: {partition}")
        expected_kind = REQUIRED_KINDS[PARTITIONS.index(partition)]
        if kind != expected_kind:
            raise ResidualIntelligenceError(
                f"{partition} cases must use the owner-scheduled {expected_kind} kind"
            )
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "kind", kind)
        if type(self.hidden_test) is not bool:
            raise ResidualIntelligenceError("hidden_test must be boolean")
        if self.hidden_test != (partition in _HIDDEN_PARTITIONS):
            raise ResidualIntelligenceError(
                "hidden_test must be true exactly for held-out and adversarial partitions"
            )
        object.__setattr__(self, "group_id", _identity_text(self.group_id, "group_id"))
        object.__setattr__(
            self,
            "input_identity",
            _identity_text(self.input_identity, "input_identity"),
        )
        if self.input_disposition != _INPUT_DISPOSITION:
            raise ResidualIntelligenceError(
                "benchmark inputs must remain unavailable because training is unavailable"
            )
        try:
            outcome = ExpertDisposition(self.expected_outcome)
        except (TypeError, ValueError) as exc:
            raise ResidualIntelligenceError("benchmark expected_outcome is unknown") from exc
        if outcome is not ExpertDisposition.CAPABILITY_UNAVAILABLE:
            raise ResidualIntelligenceError(
                "frozen benchmark cases must expect CAPABILITY_UNAVAILABLE"
            )
        object.__setattr__(self, "expected_outcome", outcome)
        object.__setattr__(self, "case_id", _identity_text(self.case_id, "case_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "family": self.family.value,
            "partition": self.partition,
            "kind": self.kind,
            "hidden_test": self.hidden_test,
            "group_id": self.group_id,
            "input_identity": self.input_identity,
            "input_disposition": self.input_disposition,
            "expected_outcome": self.expected_outcome.value,
            "case_id": self.case_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrozenBenchmarkCase:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS,
            noun="frozen benchmark case",
        )
        return cls(
            schema=payload.get("schema"),
            family=payload.get("family"),
            partition=payload.get("partition"),
            kind=payload.get("kind"),
            hidden_test=payload.get("hidden_test"),
            group_id=payload.get("group_id"),
            input_identity=payload.get("input_identity"),
            input_disposition=payload.get("input_disposition"),
            expected_outcome=payload.get("expected_outcome"),
            case_id=payload.get("case_id"),
        )


def build_frozen_benchmark_contract(
    *,
    task_families: Sequence[str | ResidualTaskFamily],
    source_commit: str,
    source_tree: str,
    split_root: str,
    base_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact pure value independently reconstructed by the owner."""

    families = _task_families(task_families)
    commit = _git_object(source_commit, "source_commit")
    tree = _git_object(source_tree, "source_tree")
    split = required_text(split_root, "split_root")
    bindings = _text_bindings(base_bindings, include_fault_schedule=False)

    schedule_entries: list[dict[str, Any]] = []
    for family in families:
        for partition, kind in zip(PARTITIONS, REQUIRED_KINDS, strict=True):
            group_body = {
                "schema": LINEAGE_ROOT_SCHEMA,
                "family": family.value,
                "partition": partition,
                "kind": kind,
                "source_tree": tree,
                "split_root": split,
            }
            schedule_entries.append(
                {
                    "family": family.value,
                    "partition": partition,
                    "kind": kind,
                    "hidden_test": partition in _HIDDEN_PARTITIONS,
                    "group_id": sha256_identity(group_body),
                }
            )
    fault_schedule: dict[str, Any] = {
        "schema": FAULT_SCHEDULE_SCHEMA,
        "source_tree": tree,
        "split_root": split,
        "entries": schedule_entries,
    }
    fault_schedule["schedule_id"] = sha256_identity(fault_schedule)
    bindings["fault_schedule"] = str(fault_schedule["schedule_id"])
    binding_set_id = sha256_identity(bindings)

    cases: list[dict[str, Any]] = []
    for scheduled in schedule_entries:
        input_contract = {
            "schema": UNAVAILABLE_INPUT_SCHEMA,
            "family": scheduled["family"],
            "partition": scheduled["partition"],
            "kind": scheduled["kind"],
            "group_id": scheduled["group_id"],
            "source_tree": tree,
            "disposition": _INPUT_DISPOSITION,
        }
        case_body: dict[str, Any] = {
            "schema": CASE_SCHEMA,
            **scheduled,
            "input_identity": sha256_identity(input_contract),
            "input_disposition": _INPUT_DISPOSITION,
            "expected_outcome": ExpertDisposition.CAPABILITY_UNAVAILABLE.value,
        }
        case_body["case_id"] = sha256_identity(
            {**case_body, "freeze_binding_set_id": binding_set_id}
        )
        cases.append(case_body)

    case_root = sha256_identity(cases)
    denominators = {family.value: len(PARTITIONS) for family in families}
    scores = {
        "accept": 0,
        "abstain": len(cases),
        "total": len(cases),
        "denominators_by_family": denominators,
    }
    source = {"commit": commit, "tree": tree}
    paired_baseline: dict[str, Any] = {
        "schema": PAIRED_BASELINE_SCHEMA,
        "prior_source": source,
        "evaluated_source": source,
        "comparison_disposition": _COMPARISON_DISPOSITION,
        "case_payload_disposition": _INPUT_DISPOSITION,
        "evaluation_disposition": _EVALUATION_DISPOSITION,
        "case_count": len(cases),
        "case_root": case_root,
        "binding_set_id": binding_set_id,
        "before": scores,
        "after": scores,
        "candidate_only": True,
        "training_performed": False,
    }
    paired_baseline["paired_baseline_id"] = sha256_identity(paired_baseline)
    benchmark_freeze: dict[str, Any] = {
        "schema": BENCHMARK_FREEZE_SCHEMA,
        "state": "frozen",
        "source": source,
        "case_payload_disposition": _INPUT_DISPOSITION,
        "evaluation_disposition": _EVALUATION_DISPOSITION,
        "bindings": bindings,
        "binding_set_id": binding_set_id,
        "fault_schedule": fault_schedule,
        "case_count": len(cases),
        "case_root": case_root,
        "paired_baseline": paired_baseline,
    }
    benchmark_freeze["freeze_id"] = sha256_identity(benchmark_freeze)
    return {
        "partitions": list(PARTITIONS),
        "case_kinds": list(REQUIRED_KINDS),
        "cases": cases,
        "scores": scores,
        "fault_schedule": fault_schedule,
        "bindings": bindings,
        "binding_set_id": binding_set_id,
        "paired_baseline": paired_baseline,
        "benchmark_freeze": benchmark_freeze,
    }


def _validate_source(value: Any, *, noun: str) -> dict[str, str]:
    strict_fields(value, allowed=_SOURCE_FIELDS, required=_SOURCE_FIELDS, noun=noun)
    return {
        "commit": _git_object(value.get("commit"), f"{noun}.commit"),
        "tree": _git_object(value.get("tree"), f"{noun}.tree"),
    }


def _validate_score_shape(value: Any, *, noun: str) -> None:
    strict_fields(value, allowed=_SCORE_FIELDS, required=_SCORE_FIELDS, noun=noun)
    for field in ("accept", "abstain", "total"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ResidualIntelligenceError(f"{noun}.{field} must be a non-negative integer")
    denominators = value.get("denominators_by_family")
    if not isinstance(denominators, Mapping):
        raise ResidualIntelligenceError(f"{noun}.denominators_by_family must be an object")
    try:
        denominator_families = {ResidualTaskFamily(item) for item in denominators}
    except (TypeError, ValueError) as exc:
        raise ResidualIntelligenceError(
            f"{noun}.denominators_by_family contains an unknown family"
        ) from exc
    if denominator_families != set(ResidualTaskFamily):
        raise ResidualIntelligenceError(
            f"{noun}.denominators_by_family must cover the exact closed taxonomy"
        )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in denominators.values()
    ):
        raise ResidualIntelligenceError(
            f"{noun}.denominators_by_family values must be non-negative integers"
        )


def _validate_freeze_shape(value: Any) -> dict[str, Any]:
    freeze = bounded_json_mapping(value, "benchmark_freeze")
    strict_fields(
        freeze,
        allowed=_BENCHMARK_FREEZE_FIELDS,
        required=_BENCHMARK_FREEZE_FIELDS,
        noun="benchmark freeze",
    )
    if freeze.get("schema") != BENCHMARK_FREEZE_SCHEMA or freeze.get("state") != "frozen":
        raise ResidualIntelligenceError("unsupported benchmark freeze schema or state")
    _validate_source(freeze.get("source"), noun="benchmark freeze source")
    if freeze.get("case_payload_disposition") != _INPUT_DISPOSITION:
        raise ResidualIntelligenceError("benchmark freeze case payload disposition is not exact")
    if freeze.get("evaluation_disposition") != _EVALUATION_DISPOSITION:
        raise ResidualIntelligenceError("benchmark freeze evaluation disposition is not exact")
    _text_bindings(freeze.get("bindings"), include_fault_schedule=True)
    _identity_text(freeze.get("binding_set_id"), "benchmark freeze binding_set_id")
    _identity_text(freeze.get("case_root"), "benchmark freeze case_root")
    _identity_text(freeze.get("freeze_id"), "benchmark freeze freeze_id")
    case_count = freeze.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 0:
        raise ResidualIntelligenceError("benchmark freeze case_count must be a non-negative integer")

    schedule = freeze.get("fault_schedule")
    strict_fields(
        schedule,
        allowed=_FAULT_SCHEDULE_FIELDS,
        required=_FAULT_SCHEDULE_FIELDS,
        noun="benchmark fault schedule",
    )
    if schedule.get("schema") != FAULT_SCHEDULE_SCHEMA:
        raise ResidualIntelligenceError("unsupported benchmark fault schedule schema")
    _git_object(schedule.get("source_tree"), "benchmark fault schedule source_tree")
    required_text(schedule.get("split_root"), "benchmark fault schedule split_root")
    _identity_text(schedule.get("schedule_id"), "benchmark fault schedule schedule_id")
    entries = schedule.get("entries")
    if not isinstance(entries, list):
        raise ResidualIntelligenceError("benchmark fault schedule entries must be a list")
    for index, entry in enumerate(entries):
        strict_fields(
            entry,
            allowed=_SCHEDULE_ENTRY_FIELDS,
            required=_SCHEDULE_ENTRY_FIELDS,
            noun=f"benchmark fault schedule entry {index}",
        )

    paired = freeze.get("paired_baseline")
    strict_fields(
        paired,
        allowed=_PAIRED_BASELINE_FIELDS,
        required=_PAIRED_BASELINE_FIELDS,
        noun="paired benchmark baseline",
    )
    if paired.get("schema") != PAIRED_BASELINE_SCHEMA:
        raise ResidualIntelligenceError("unsupported paired benchmark baseline schema")
    _validate_source(paired.get("prior_source"), noun="paired baseline prior_source")
    _validate_source(paired.get("evaluated_source"), noun="paired baseline evaluated_source")
    for field in ("case_root", "binding_set_id", "paired_baseline_id"):
        _identity_text(paired.get(field), f"paired benchmark baseline {field}")
    _validate_score_shape(paired.get("before"), noun="paired benchmark baseline before")
    _validate_score_shape(paired.get("after"), noun="paired benchmark baseline after")
    return freeze


@dataclass(frozen=True)
class ResidualBenchmarkManifest:
    """The exact 13-field producer manifest consumed by the owner."""

    families: tuple[ResidualTaskFamily, ...]
    source_revision: str
    benchmark_freeze: Mapping[str, Any]
    partitions: tuple[str, ...] = PARTITIONS
    required_case_kinds: tuple[str, ...] = REQUIRED_KINDS
    program_identifier: str = PROGRAM_ID
    owner_task: str = "VRIF-030"
    status: str = "staged_not_qualified"
    training_admission: str = "training_unavailable"
    weights_committed: bool = False
    large_corpus_committed: bool = False
    promotion_evidence: bool = False
    schema: str = MANIFEST_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = _MANIFEST_FIELDS

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual benchmark manifest schema")
        if self.program_identifier != PROGRAM_ID:
            raise ResidualIntelligenceError("unexpected benchmark program identifier")
        if self.owner_task != "VRIF-030" or self.status != "staged_not_qualified":
            raise ResidualIntelligenceError("benchmark manifest cannot claim qualification")
        if self.training_admission != "training_unavailable":
            raise ResidualIntelligenceError("benchmark does not grant training admission")
        if any(
            type(item) is not bool or item
            for item in (
                self.weights_committed,
                self.large_corpus_committed,
                self.promotion_evidence,
            )
        ):
            raise ResidualIntelligenceError(
                "benchmark cannot claim weights, corpus, or promotion evidence"
            )
        object.__setattr__(self, "families", _task_families(self.families))
        if tuple(self.partitions) != PARTITIONS:
            raise ResidualIntelligenceError("benchmark partitions must be exact")
        if tuple(self.required_case_kinds) != REQUIRED_KINDS:
            raise ResidualIntelligenceError("benchmark required case kinds must be exact")
        object.__setattr__(self, "partitions", PARTITIONS)
        object.__setattr__(self, "required_case_kinds", REQUIRED_KINDS)
        object.__setattr__(
            self,
            "source_revision",
            _git_object(self.source_revision, "source_revision"),
        )
        object.__setattr__(self, "benchmark_freeze", _validate_freeze_shape(self.benchmark_freeze))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "program_identifier": self.program_identifier,
            "status": self.status,
            "owner_task": self.owner_task,
            "source_revision": self.source_revision,
            "partitions": list(self.partitions),
            "required_case_kinds": list(self.required_case_kinds),
            "task_families": [item.value for item in self.families],
            "training_admission": self.training_admission,
            "weights_committed": self.weights_committed,
            "large_corpus_committed": self.large_corpus_committed,
            "promotion_evidence": self.promotion_evidence,
            "benchmark_freeze": dict(self.benchmark_freeze),
        }

    @property
    def frozen_root(self) -> str:
        """Compatibility alias for the @2 freeze identity."""

        return str(self.benchmark_freeze["freeze_id"])

    @property
    def computed_frozen_root(self) -> str:
        body = dict(self.benchmark_freeze)
        body.pop("freeze_id", None)
        return sha256_identity(body)

    @property
    def case_catalog_root(self) -> str:
        """Compatibility alias for the ordered @2 case root."""

        return str(self.benchmark_freeze["case_root"])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualBenchmarkManifest:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS,
            noun="benchmark manifest",
        )
        return cls(
            schema=payload.get("schema"),
            program_identifier=payload.get("program_identifier"),
            status=payload.get("status"),
            owner_task=payload.get("owner_task"),
            source_revision=payload.get("source_revision"),
            partitions=tuple(payload.get("partitions") or ()),
            required_case_kinds=tuple(payload.get("required_case_kinds") or ()),
            families=tuple(payload.get("task_families") or ()),
            training_admission=payload.get("training_admission"),
            weights_committed=payload.get("weights_committed"),
            large_corpus_committed=payload.get("large_corpus_committed"),
            promotion_evidence=payload.get("promotion_evidence"),
            benchmark_freeze=payload.get("benchmark_freeze"),
        )


def build_frozen_benchmark(
    *,
    task_families: Sequence[str | ResidualTaskFamily],
    source_commit: str,
    source_tree: str,
    split_root: str,
    base_bindings: Mapping[str, str],
) -> tuple[ResidualBenchmarkManifest, tuple[FrozenBenchmarkCase, ...]]:
    """Build typed manifest and cases using the owner-exact pure contract."""

    contract = build_frozen_benchmark_contract(
        task_families=task_families,
        source_commit=source_commit,
        source_tree=source_tree,
        split_root=split_root,
        base_bindings=base_bindings,
    )
    manifest = ResidualBenchmarkManifest(
        families=tuple(ResidualTaskFamily(item) for item in task_families),
        source_revision=source_commit,
        benchmark_freeze=contract["benchmark_freeze"],
    )
    cases = tuple(FrozenBenchmarkCase.from_dict(item) for item in contract["cases"])
    validate_frozen_benchmark(manifest, cases)
    return manifest, cases


def case_catalog_root(cases: Sequence[FrozenBenchmarkCase]) -> str:
    """Return the owner-exact identity of the ordered case catalog."""

    return sha256_identity([item.to_dict() for item in cases])


def partition_roots(cases: Sequence[FrozenBenchmarkCase]) -> dict[str, str]:
    """Return diagnostic identities; these are not manifest authority fields."""

    return {
        partition: sha256_identity(
            [item.to_dict() for item in cases if item.partition == partition]
        )
        for partition in PARTITIONS
    }


def lineage_root(cases: Sequence[FrozenBenchmarkCase]) -> str:
    """Return a diagnostic identity of the owner-scheduled lineage assignments."""

    return sha256_identity(
        [
            {
                "case_id": item.case_id,
                "group_id": item.group_id,
                "partition": item.partition,
            }
            for item in cases
        ]
    )


def validate_frozen_benchmark(
    manifest: ResidualBenchmarkManifest,
    cases: Sequence[FrozenBenchmarkCase],
) -> None:
    """Fail closed unless manifest and cases equal the owner's reconstruction."""

    if not isinstance(manifest, ResidualBenchmarkManifest):
        raise ResidualIntelligenceError("benchmark manifest must be typed")
    typed = tuple(cases)
    if any(not isinstance(item, FrozenBenchmarkCase) for item in typed):
        raise ResidualIntelligenceError("benchmark cases must be typed frozen cases")
    freeze = manifest.benchmark_freeze
    source = freeze["source"]
    if source["commit"] != manifest.source_revision:
        raise ResidualIntelligenceError("benchmark source revision does not match frozen source")
    bindings = dict(freeze["bindings"])
    base_bindings = {key: bindings[key] for key in BASE_BINDINGS}
    schedule = freeze["fault_schedule"]
    expected = build_frozen_benchmark_contract(
        task_families=manifest.families,
        source_commit=manifest.source_revision,
        source_tree=source["tree"],
        split_root=schedule["split_root"],
        base_bindings=base_bindings,
    )
    expected_freeze = expected["benchmark_freeze"]
    if freeze["bindings"] != expected_freeze["bindings"]:
        raise ResidualIntelligenceError("benchmark frozen bindings do not verify")
    if freeze["binding_set_id"] != expected_freeze["binding_set_id"]:
        raise ResidualIntelligenceError("benchmark frozen binding set does not verify")
    if freeze["fault_schedule"] != expected_freeze["fault_schedule"]:
        raise ResidualIntelligenceError("benchmark fault schedule does not verify")
    if freeze["paired_baseline"] != expected_freeze["paired_baseline"]:
        raise ResidualIntelligenceError("paired benchmark baseline does not verify")
    actual_cases = [item.to_dict() for item in typed]
    if actual_cases != expected["cases"]:
        raise ResidualIntelligenceError(
            "benchmark cases do not match the owner-computable 96-case schedule"
        )
    if freeze != expected_freeze:
        raise ResidualIntelligenceError("benchmark freeze does not verify")


@dataclass(frozen=True)
class PairedBenchmarkRunner:
    """Return only the exact all-abstain, not-run paired baseline."""

    def evaluate(
        self,
        manifest: ResidualBenchmarkManifest,
        cases: Sequence[FrozenBenchmarkCase],
        *,
        prior: Mapping[str, Any] | None = None,
        current: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_frozen_benchmark(manifest, cases)
        baseline = dict(manifest.benchmark_freeze["paired_baseline"])
        expected_scores = baseline["before"]
        if prior is not None and dict(prior) != expected_scores:
            raise ResidualIntelligenceError(
                "prior scores must equal the frozen all-abstain baseline"
            )
        if current is not None and dict(current) != expected_scores:
            raise ResidualIntelligenceError(
                "current scores must equal the frozen all-abstain baseline"
            )
        if (prior is None) != (current is None):
            raise ResidualIntelligenceError("paired scores must be provided together")
        return baseline


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a strict manifest payload without treating it as owner evidence."""

    payload = _strict_json_loads(path.read_text(encoding="utf-8"), noun="benchmark manifest")
    if not isinstance(payload, dict):
        raise ResidualIntelligenceError("benchmark manifest must be an object")
    return payload


def load_cases(path: Path) -> tuple[FrozenBenchmarkCase, ...]:
    """Load exact JSONL cases; blank lines are rejected because order is bound."""

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


def load_frozen_benchmark(
    manifest_path: Path,
    cases_path: Path,
) -> tuple[ResidualBenchmarkManifest, tuple[FrozenBenchmarkCase, ...]]:
    """Load and independently verify the complete immutable benchmark bundle."""

    manifest = ResidualBenchmarkManifest.from_dict(load_manifest(manifest_path))
    cases = load_cases(cases_path)
    validate_frozen_benchmark(manifest, cases)
    return manifest, cases


__all__ = (
    "BASE_BINDINGS",
    "BENCHMARK_FREEZE_SCHEMA",
    "CASE_SCHEMA",
    "FAULT_SCHEDULE_SCHEMA",
    "IDENTITY_FIELDS",
    "LINEAGE_ROOT_SCHEMA",
    "MANIFEST_SCHEMA",
    "PAIRED_BASELINE_SCHEMA",
    "PARTITIONS",
    "REQUIRED_BINDINGS",
    "REQUIRED_KINDS",
    "FrozenBenchmarkCase",
    "PairedBenchmarkRunner",
    "ResidualBenchmarkManifest",
    "build_frozen_benchmark",
    "build_frozen_benchmark_contract",
    "case_catalog_root",
    "lineage_root",
    "load_cases",
    "load_frozen_benchmark",
    "load_manifest",
    "partition_roots",
    "sha256_identity",
    "validate_frozen_benchmark",
)
