"""Frozen, lineage-safe paired residual benchmark contracts.

Published benchmark records carry commitments only.  Raw test inputs, and in
particular held-out and adversarial bodies, are deliberately not Git content.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .contracts import ExpertDisposition, ResidualIntelligenceError, ResidualTaskFamily, required_text

MANIFEST_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-intelligence-benchmark-manifest@1"
CASE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-frozen-benchmark-case@1"
RESULT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-paired-benchmark-result@1"
PARTITIONS: Final[tuple[str, ...]] = ("training", "development", "held_out", "adversarial")
REQUIRED_KINDS: Final[tuple[str, ...]] = ("boundary", "negative", "cross_repository", "unknown_ood")
IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_identity", "objective_identity", "catalog_identity", "provider_identity",
    "tokenizer_identity", "model_identity", "fault_identity", "validation_identity",
)
_PREFIX: Final = "sha256:"


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _identity(value: Any, name: str) -> str:
    text = required_text(value, name, max_bytes=256)
    digest = text[len(_PREFIX) :] if text.startswith(_PREFIX) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ResidualIntelligenceError("{} must be a lowercase sha256 identity".format(name))
    return text


def _identities(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("{} must be an identity sequence".format(name))
    result = tuple(_identity(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ResidualIntelligenceError("{} must be non-empty and unique".format(name))
    return result


@dataclass(frozen=True)
class BenchmarkBindings:
    """The exact non-payload identities shared by every published case."""

    repository_identity: str
    objective_identity: str
    catalog_identity: str
    provider_identity: str
    tokenizer_identity: str
    model_identity: str
    fault_identity: str
    validation_identity: str
    cross_repository_identities: tuple[str, ...]
    schema: str = "ipfs_accelerate_py/agent-supervisor/residual-benchmark-bindings@1"

    def __post_init__(self) -> None:
        if self.schema != "ipfs_accelerate_py/agent-supervisor/residual-benchmark-bindings@1":
            raise ResidualIntelligenceError("unsupported benchmark bindings schema")
        for field in IDENTITY_FIELDS:
            object.__setattr__(self, field, _identity(getattr(self, field), field))
        cross = _identities(self.cross_repository_identities, "cross_repository_identities")
        if self.repository_identity in cross:
            raise ResidualIntelligenceError("cross repositories must differ from benchmark repository")
        object.__setattr__(self, "cross_repository_identities", cross)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {"schema": self.schema, **{field: getattr(self, field) for field in IDENTITY_FIELDS}}
        result["cross_repository_identities"] = list(self.cross_repository_identities)
        if include_id:
            result["binding_set_id"] = self.binding_set_id
        return result

    @property
    def binding_set_id(self) -> str:
        return _digest(self.to_dict(include_id=False))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkBindings:
        expected = {"schema", "binding_set_id", *IDENTITY_FIELDS, "cross_repository_identities"}
        if set(payload) != expected:
            raise ResidualIntelligenceError("benchmark bindings have an invalid field set")
        result = cls(
            schema=str(payload["schema"]),
            cross_repository_identities=tuple(payload["cross_repository_identities"]),
            **{field: str(payload[field]) for field in IDENTITY_FIELDS},
        )
        if str(payload["binding_set_id"]) != result.binding_set_id:
            raise ResidualIntelligenceError("benchmark binding set identity mismatch")
        return result


@dataclass(frozen=True)
class FrozenBenchmarkCase:
    """A payload-free immutable benchmark coordinate."""

    family: ResidualTaskFamily
    partition: str
    kind: str
    case_id: str
    group_id: str
    semantic_lineage_id: str
    input_identity: str
    expected_outcome: ExpertDisposition
    hidden_test: bool
    repository_identity: str
    objective_identity: str
    catalog_identity: str
    provider_identity: str
    tokenizer_identity: str
    model_identity: str
    fault_identity: str
    validation_identity: str
    admission_identity: str
    permitted_use: str
    schema: str = CASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASE_SCHEMA:
            raise ResidualIntelligenceError("unsupported frozen benchmark case schema")
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        partition = required_text(self.partition, "partition")
        kind = required_text(self.kind, "kind")
        if partition not in PARTITIONS or kind not in REQUIRED_KINDS:
            raise ResidualIntelligenceError("unknown benchmark partition or kind")
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "case_id", required_text(self.case_id, "case_id", max_bytes=256))
        for field in ("group_id", "semantic_lineage_id", "input_identity", "admission_identity", *IDENTITY_FIELDS):
            object.__setattr__(self, field, _identity(getattr(self, field), field))
        object.__setattr__(self, "expected_outcome", ExpertDisposition(self.expected_outcome))
        if type(self.hidden_test) is not bool:
            raise ResidualIntelligenceError("hidden_test must be boolean")
        expected_use = "training" if partition == "training" else "evaluation"
        if self.permitted_use != expected_use:
            raise ResidualIntelligenceError("case permitted use does not match its partition")
        if (partition in {"held_out", "adversarial"}) != self.hidden_test:
            raise ResidualIntelligenceError("hidden-test denial must match held-out/adversarial partitions")
        if kind == "unknown_ood" and self.expected_outcome is not ExpertDisposition.OUT_OF_DISTRIBUTION:
            raise ResidualIntelligenceError("unknown OOD cases must require OUT_OF_DISTRIBUTION")

    def to_dict(self, *, include_commitment: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema, "family": self.family.value, "partition": self.partition,
            "kind": self.kind, "case_id": self.case_id, "group_id": self.group_id,
            "semantic_lineage_id": self.semantic_lineage_id, "input_identity": self.input_identity,
            "expected_outcome": self.expected_outcome.value, "hidden_test": self.hidden_test,
            **{field: getattr(self, field) for field in IDENTITY_FIELDS},
            "admission_identity": self.admission_identity, "permitted_use": self.permitted_use,
        }
        if include_commitment:
            result["case_commitment"] = self.case_commitment
        return result

    @property
    def case_commitment(self) -> str:
        return _digest(self.to_dict(include_commitment=False))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrozenBenchmarkCase:
        expected = {
            "schema", "case_commitment", "family", "partition", "kind", "case_id", "group_id",
            "semantic_lineage_id", "input_identity", "expected_outcome", "hidden_test",
            *IDENTITY_FIELDS, "admission_identity", "permitted_use",
        }
        if set(payload) != expected:
            raise ResidualIntelligenceError("frozen benchmark case has an invalid field set")
        result = cls(
            schema=str(payload["schema"]), family=ResidualTaskFamily(str(payload["family"])),
            partition=str(payload["partition"]), kind=str(payload["kind"]), case_id=str(payload["case_id"]),
            group_id=str(payload["group_id"]), semantic_lineage_id=str(payload["semantic_lineage_id"]),
            input_identity=str(payload["input_identity"]),
            expected_outcome=ExpertDisposition(str(payload["expected_outcome"])),
            hidden_test=payload["hidden_test"], admission_identity=str(payload["admission_identity"]),
            permitted_use=str(payload["permitted_use"]),
            **{field: str(payload[field]) for field in IDENTITY_FIELDS},
        )
        if str(payload["case_commitment"]) != result.case_commitment:
            raise ResidualIntelligenceError("frozen benchmark case commitment mismatch")
        return result


@dataclass(frozen=True)
class ResidualBenchmarkManifest:
    families: tuple[ResidualTaskFamily, ...]
    partitions: tuple[str, ...]
    bindings: BenchmarkBindings
    case_root: str
    frozen_root: str
    paired_baseline_identity: str
    hidden_test_commitment: str
    schema: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual benchmark manifest schema")
        families = tuple(ResidualTaskFamily(item) for item in self.families)
        if families != tuple(ResidualTaskFamily):
            raise ResidualIntelligenceError("benchmark must contain every family exactly once in catalog order")
        object.__setattr__(self, "families", families)
        if tuple(self.partitions) != PARTITIONS:
            raise ResidualIntelligenceError("benchmark partitions must be exact")
        if not isinstance(self.bindings, BenchmarkBindings):
            raise ResidualIntelligenceError("benchmark manifest requires typed bindings")
        for field in ("case_root", "frozen_root", "paired_baseline_identity", "hidden_test_commitment"):
            object.__setattr__(self, field, _identity(getattr(self, field), field))

    def root_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "task_families": [family.value for family in self.families],
            "partitions": list(self.partitions), "binding_set_id": self.bindings.binding_set_id,
            "case_root": self.case_root, "paired_baseline_identity": self.paired_baseline_identity,
            "hidden_test_commitment": self.hidden_test_commitment,
        }

    @property
    def expected_frozen_root(self) -> str:
        return _digest(self.root_payload())

    def verify_frozen_root(self) -> None:
        if self.frozen_root != self.expected_frozen_root:
            raise ResidualIntelligenceError("frozen root does not bind manifest metadata")


def case_root(cases: Sequence[FrozenBenchmarkCase]) -> str:
    return _digest({
        "schema": CASE_SCHEMA,
        "cases": [case.to_dict() for case in sorted(cases, key=lambda item: item.case_id)],
    })


def validate_frozen_benchmark(manifest: ResidualBenchmarkManifest, cases: Sequence[FrozenBenchmarkCase]) -> None:
    """Fail closed on incomplete coverage, lineage leakage, or altered bindings."""

    manifest.verify_frozen_root()
    cases = tuple(cases)
    if not cases or any(not isinstance(case, FrozenBenchmarkCase) for case in cases):
        raise ResidualIntelligenceError("benchmark cases must be non-empty typed records")
    if len({case.case_id for case in cases}) != len(cases):
        raise ResidualIntelligenceError("benchmark contains duplicate case identifiers")
    if manifest.case_root != case_root(cases):
        raise ResidualIntelligenceError("benchmark case root mismatch")
    groups: dict[str, set[str]] = defaultdict(set)
    coverage = Counter((case.family, case.partition, case.kind) for case in cases)
    hidden_inputs = []
    for case in cases:
        groups[case.group_id].add(case.partition)
        if case.kind == "cross_repository":
            if case.repository_identity not in manifest.bindings.cross_repository_identities:
                raise ResidualIntelligenceError("cross-repository case uses an unbound repository")
        elif case.repository_identity != manifest.bindings.repository_identity:
            raise ResidualIntelligenceError("non-cross-repository case uses an unbound repository")
        for field in IDENTITY_FIELDS[1:]:
            if getattr(case, field) != getattr(manifest.bindings, field):
                raise ResidualIntelligenceError("case {} does not match frozen bindings".format(field))
        if case.hidden_test:
            hidden_inputs.append(case.input_identity)
    if any(len(partitions) != 1 for partitions in groups.values()):
        raise ResidualIntelligenceError("semantic lineage group crosses benchmark partitions")
    for family in manifest.families:
        for partition in manifest.partitions:
            for kind in REQUIRED_KINDS:
                if not coverage[(family, partition, kind)]:
                    raise ResidualIntelligenceError("benchmark has incomplete family/partition/kind coverage")
    if manifest.hidden_test_commitment != _digest({"hidden_input_identities": sorted(hidden_inputs)}):
        raise ResidualIntelligenceError("hidden-test commitment mismatch")


@dataclass(frozen=True)
class PairedBenchmarkRunner:
    """Compare prior and current candidates on the identical frozen case set."""

    def evaluate(
        self, manifest: ResidualBenchmarkManifest, cases: Sequence[FrozenBenchmarkCase], *,
        prior: Mapping[str, str | Mapping[str, Any]], current: Mapping[str, str | Mapping[str, Any]],
    ) -> dict[str, Any]:
        validate_frozen_benchmark(manifest, cases)
        expected_ids = {case.case_id for case in cases}
        if set(prior) != expected_ids or set(current) != expected_ids:
            raise ResidualIntelligenceError("paired evaluations must score every exact frozen case")

        def parse(value: str | Mapping[str, Any]) -> ExpertDisposition:
            candidate = value.get("outcome") if isinstance(value, Mapping) else value
            try:
                return ExpertDisposition(str(candidate))
            except ValueError as exc:
                raise ResidualIntelligenceError("paired evaluation has an invalid disposition") from exc

        families: Counter[str] = Counter()
        partitions: Counter[str] = Counter()
        prior_correct: Counter[str] = Counter()
        current_correct: Counter[str] = Counter()
        paired: dict[str, dict[str, Any]] = {}
        for case in cases:
            family = case.family.value
            families[family] += 1
            partitions[case.partition] += 1
            before, after = parse(prior[case.case_id]), parse(current[case.case_id])
            before_ok, after_ok = before is case.expected_outcome, after is case.expected_outcome
            prior_correct[family] += int(before_ok)
            current_correct[family] += int(after_ok)
            paired[case.case_id] = {
                "expected": case.expected_outcome.value, "prior": before.value, "current": after.value,
                "prior_correct": before_ok, "current_correct": after_ok,
            }
        return {
            "schema": RESULT_SCHEMA, "frozen_root": manifest.frozen_root,
            "binding_set_id": manifest.bindings.binding_set_id,
            "paired_baseline_identity": manifest.paired_baseline_identity,
            "denominators": {
                "all_cases": len(cases), "by_family": dict(sorted(families.items())),
                "by_partition": dict(sorted(partitions.items())),
            },
            "prior": {"correct_by_family": dict(sorted(prior_correct.items())), "scored_cases": len(prior)},
            "current": {"correct_by_family": dict(sorted(current_correct.items())), "scored_cases": len(current)},
            "paired_cases": paired, "candidate_only": True,
        }


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidualIntelligenceError("cannot load benchmark manifest") from exc
    if not isinstance(payload, dict):
        raise ResidualIntelligenceError("benchmark manifest must be an object")
    return payload


def load_cases(path: Path) -> tuple[FrozenBenchmarkCase, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ResidualIntelligenceError("cannot load benchmark cases") from exc
    if not lines or any(not line.strip() for line in lines):
        raise ResidualIntelligenceError("benchmark cases must be non-empty canonical JSONL")
    records = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResidualIntelligenceError("invalid benchmark JSONL") from exc
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("benchmark case must be an object")
        records.append(FrozenBenchmarkCase.from_dict(payload))
    return tuple(records)


def manifest_from_dict(payload: Mapping[str, Any]) -> ResidualBenchmarkManifest:
    required = {
        "schema", "task_families", "partitions", "bindings", "case_root", "frozen_root",
        "paired_baseline", "hidden_test_commitment",
    }
    if required - set(payload):
        raise ResidualIntelligenceError("benchmark manifest is missing required frozen fields")
    bindings, baseline = payload["bindings"], payload["paired_baseline"]
    if not isinstance(bindings, Mapping) or not isinstance(baseline, Mapping):
        raise ResidualIntelligenceError("benchmark bindings and baseline must be objects")
    if set(baseline) != {"baseline_identity", "candidate_only"} or baseline["candidate_only"] is not True:
        raise ResidualIntelligenceError("paired baseline must remain candidate-only and identity-bound")
    return ResidualBenchmarkManifest(
        schema=str(payload["schema"]), families=tuple(payload["task_families"]),
        partitions=tuple(payload["partitions"]), bindings=BenchmarkBindings.from_dict(bindings),
        case_root=str(payload["case_root"]), frozen_root=str(payload["frozen_root"]),
        paired_baseline_identity=str(baseline["baseline_identity"]),
        hidden_test_commitment=str(payload["hidden_test_commitment"]),
    )
