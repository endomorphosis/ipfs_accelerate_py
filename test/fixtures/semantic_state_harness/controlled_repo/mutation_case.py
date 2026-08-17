"""Mutation cases and independently declared fixture oracles.

Every mutation must carry separate oracles for:

* changed symbols
* Merkle node impact
* invalidation / selected tests / proof obligations
* receipt freshness
* confidence class and raw-source requirement

Oracles are reviewed fixture authority, not observations derived from running
the harness against the mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

MUTATION_CASE_SCHEMA = "ipfs_accelerate_py/semantic-state/mutation-case@1"
FIXTURE_ORACLE_SCHEMA = "ipfs_accelerate_py/semantic-state/fixture-oracle@1"

CONFIDENCE_CLASSES = frozenset({"exact", "conservative", "heuristic", "opaque"})
FRESHNESS_DISPOSITIONS = frozenset(
    {
        "current",
        "stale",
        "require_rescan",
        "reject_stale",
        "cas_reject",
        "interrupted",
        "concurrent_fence",
        "not_applicable",
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
class ChangedSymbolOracle:
    """Independently declared changed-symbol oracle."""

    symbol_ids: tuple[str, ...]
    primary_symbol_id: str
    change_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        symbols = _sorted_unique(self.symbol_ids, "symbol_ids")
        primary = _text(self.primary_symbol_id, "primary_symbol_id")
        if primary not in symbols:
            raise FixtureCorpusError("primary_symbol_id must appear in symbol_ids")
        kinds = _sorted_unique(self.change_kinds, "change_kinds")
        object.__setattr__(self, "symbol_ids", symbols)
        object.__setattr__(self, "primary_symbol_id", primary)
        object.__setattr__(self, "change_kinds", kinds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_ids": list(self.symbol_ids),
            "primary_symbol_id": self.primary_symbol_id,
            "change_kinds": list(self.change_kinds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChangedSymbolOracle":
        return cls(
            symbol_ids=tuple(payload["symbol_ids"]),
            primary_symbol_id=str(payload["primary_symbol_id"]),
            change_kinds=tuple(payload["change_kinds"]),
        )


@dataclass(frozen=True)
class MerkleOracle:
    """Independently declared Merkle-node impact oracle."""

    changed_node_ids: tuple[str, ...]
    affected_path_ids: tuple[str, ...]
    root_changes: bool

    def __post_init__(self) -> None:
        nodes = _sorted_unique(self.changed_node_ids, "changed_node_ids")
        paths = _sorted_unique(self.affected_path_ids, "affected_path_ids")
        if type(self.root_changes) is not bool:
            raise FixtureCorpusError("root_changes must be a bool")
        object.__setattr__(self, "changed_node_ids", nodes)
        object.__setattr__(self, "affected_path_ids", paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_node_ids": list(self.changed_node_ids),
            "affected_path_ids": list(self.affected_path_ids),
            "root_changes": self.root_changes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MerkleOracle":
        return cls(
            changed_node_ids=tuple(payload["changed_node_ids"]),
            affected_path_ids=tuple(payload["affected_path_ids"]),
            root_changes=bool(payload["root_changes"]),
        )


@dataclass(frozen=True)
class InvalidationOracle:
    """Independently declared invalidation, test-selection, and proof oracle."""

    invalidation_symbol_ids: tuple[str, ...]
    selected_test_node_ids: tuple[str, ...]
    proof_obligation_ids: tuple[str, ...]
    full_suite_test_node_ids: tuple[str, ...]
    fallback: str
    expected_false_negatives: int

    def __post_init__(self) -> None:
        inv = _sorted_unique(self.invalidation_symbol_ids, "invalidation_symbol_ids")
        selected = _sorted_unique(
            self.selected_test_node_ids, "selected_test_node_ids"
        )
        proofs = _sorted_unique(self.proof_obligation_ids, "proof_obligation_ids")
        full = _sorted_unique(
            self.full_suite_test_node_ids, "full_suite_test_node_ids"
        )
        fallback = _text(self.fallback, "fallback")
        if fallback not in {"none", "full_pytest", "full_proofs", "both"}:
            raise FixtureCorpusError(f"unsupported fallback {fallback!r}")
        if type(self.expected_false_negatives) is not int:
            raise FixtureCorpusError("expected_false_negatives must be an int")
        if self.expected_false_negatives < 0:
            raise FixtureCorpusError("expected_false_negatives must be >= 0")
        # Selected tests must be a subset of the full suite oracle.
        if not set(selected).issubset(set(full)):
            raise FixtureCorpusError(
                "selected_test_node_ids must be a subset of full_suite_test_node_ids"
            )
        object.__setattr__(self, "invalidation_symbol_ids", inv)
        object.__setattr__(self, "selected_test_node_ids", selected)
        object.__setattr__(self, "proof_obligation_ids", proofs)
        object.__setattr__(self, "full_suite_test_node_ids", full)
        object.__setattr__(self, "fallback", fallback)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalidation_symbol_ids": list(self.invalidation_symbol_ids),
            "selected_test_node_ids": list(self.selected_test_node_ids),
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "full_suite_test_node_ids": list(self.full_suite_test_node_ids),
            "fallback": self.fallback,
            "expected_false_negatives": self.expected_false_negatives,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvalidationOracle":
        return cls(
            invalidation_symbol_ids=tuple(payload["invalidation_symbol_ids"]),
            selected_test_node_ids=tuple(payload["selected_test_node_ids"]),
            proof_obligation_ids=tuple(payload["proof_obligation_ids"]),
            full_suite_test_node_ids=tuple(payload["full_suite_test_node_ids"]),
            fallback=str(payload["fallback"]),
            expected_false_negatives=int(payload["expected_false_negatives"]),
        )


@dataclass(frozen=True)
class ReceiptFreshnessOracle:
    """Independently declared receipt-freshness oracle."""

    disposition: str
    accepts_stale_receipt: bool
    binds_tree_cid: bool
    binds_config_cid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        disposition = _text(self.disposition, "disposition")
        if disposition not in FRESHNESS_DISPOSITIONS:
            raise FixtureCorpusError(f"unsupported disposition {disposition!r}")
        if type(self.accepts_stale_receipt) is not bool:
            raise FixtureCorpusError("accepts_stale_receipt must be a bool")
        if type(self.binds_tree_cid) is not bool:
            raise FixtureCorpusError("binds_tree_cid must be a bool")
        if type(self.binds_config_cid) is not bool:
            raise FixtureCorpusError("binds_config_cid must be a bool")
        reasons = _sorted_unique(self.reason_codes, "reason_codes")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "accepts_stale_receipt": self.accepts_stale_receipt,
            "binds_tree_cid": self.binds_tree_cid,
            "binds_config_cid": self.binds_config_cid,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReceiptFreshnessOracle":
        return cls(
            disposition=str(payload["disposition"]),
            accepts_stale_receipt=bool(payload["accepts_stale_receipt"]),
            binds_tree_cid=bool(payload["binds_tree_cid"]),
            binds_config_cid=bool(payload["binds_config_cid"]),
            reason_codes=tuple(payload["reason_codes"]),
        )


@dataclass(frozen=True)
class ConfidenceOracle:
    """Independently declared confidence and raw-source oracle."""

    confidence: str
    raw_source_required: bool
    raw_source_symbol_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        confidence = _text(self.confidence, "confidence")
        if confidence not in CONFIDENCE_CLASSES:
            raise FixtureCorpusError(f"unsupported confidence {confidence!r}")
        if type(self.raw_source_required) is not bool:
            raise FixtureCorpusError("raw_source_required must be a bool")
        symbols = _sorted_unique(
            self.raw_source_symbol_ids, "raw_source_symbol_ids"
        )
        reasons = _sorted_unique(self.reason_codes, "reason_codes")
        if self.raw_source_required and not symbols:
            raise FixtureCorpusError(
                "raw_source_required requires raw_source_symbol_ids"
            )
        if not self.raw_source_required and symbols:
            raise FixtureCorpusError(
                "raw_source_symbol_ids must be empty when raw_source_required is false"
            )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "raw_source_symbol_ids", symbols)
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "raw_source_required": self.raw_source_required,
            "raw_source_symbol_ids": list(self.raw_source_symbol_ids),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConfidenceOracle":
        return cls(
            confidence=str(payload["confidence"]),
            raw_source_required=bool(payload["raw_source_required"]),
            raw_source_symbol_ids=tuple(payload["raw_source_symbol_ids"]),
            reason_codes=tuple(payload["reason_codes"]),
        )


@dataclass(frozen=True)
class FixtureOracle:
    """Complete independent oracle bundle for one mutation."""

    changed_symbol: ChangedSymbolOracle
    merkle: MerkleOracle
    invalidation: InvalidationOracle
    receipt_freshness: ReceiptFreshnessOracle
    confidence: ConfidenceOracle

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FIXTURE_ORACLE_SCHEMA,
            "changed_symbol": self.changed_symbol.to_dict(),
            "merkle": self.merkle.to_dict(),
            "invalidation": self.invalidation.to_dict(),
            "receipt_freshness": self.receipt_freshness.to_dict(),
            "confidence": self.confidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixtureOracle":
        schema = payload.get("schema")
        if schema is not None and schema != FIXTURE_ORACLE_SCHEMA:
            raise FixtureCorpusError(f"unsupported oracle schema {schema!r}")
        return cls(
            changed_symbol=ChangedSymbolOracle.from_dict(payload["changed_symbol"]),
            merkle=MerkleOracle.from_dict(payload["merkle"]),
            invalidation=InvalidationOracle.from_dict(payload["invalidation"]),
            receipt_freshness=ReceiptFreshnessOracle.from_dict(
                payload["receipt_freshness"]
            ),
            confidence=ConfidenceOracle.from_dict(payload["confidence"]),
        )


@dataclass(frozen=True)
class PathOperation:
    """One deterministic path-level mutation applied to the base tree."""

    op: str
    path: str
    content: str | None = None
    from_path: str | None = None

    def __post_init__(self) -> None:
        op = _text(self.op, "op")
        if op not in PATH_OPS:
            raise FixtureCorpusError(f"unsupported path op {op!r}")
        path = _text(self.path, "path")
        if path.startswith("/") or ".." in path.split("/"):
            raise FixtureCorpusError(f"path must be relative and non-escaping: {path}")
        content = self.content
        from_path = self.from_path
        if op in {"replace", "add"}:
            if type(content) is not str:
                raise FixtureCorpusError(f"{op} requires string content")
            if from_path is not None:
                raise FixtureCorpusError(f"{op} does not accept from_path")
        elif op == "delete":
            if content is not None or from_path is not None:
                raise FixtureCorpusError("delete accepts only path")
        elif op == "rename":
            if type(from_path) is not str or not from_path.strip():
                raise FixtureCorpusError("rename requires from_path")
            if from_path.startswith("/") or ".." in from_path.split("/"):
                raise FixtureCorpusError(
                    f"from_path must be relative and non-escaping: {from_path}"
                )
            if content is not None:
                raise FixtureCorpusError("rename does not accept content")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "path", path)
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
class MutationCase:
    """One base-to-mutated fixture case with independent oracles."""

    case_id: str
    category: str
    description: str
    operations: tuple[PathOperation, ...]
    oracle: FixtureOracle
    # Post-scan source-race payload must never be admitted into a context pack.
    source_race_bytes_forbidden: bool
    # Unrelated formatting and similar noise must remain a bounded change set.
    change_is_bounded: bool
    # Paths whose post-scan bytes must never appear in pack targets.
    pack_excluded_paths: tuple[str, ...]
    # Harness-level scenario labels (CAS, interruption, concurrent, ...).
    harness_scenario: str | None = None
    production_eligible: bool = False

    def __post_init__(self) -> None:
        case_id = _text(self.case_id, "case_id")
        category = _text(self.category, "category")
        description = _text(self.description, "description")
        if not self.operations and self.harness_scenario is None:
            raise FixtureCorpusError(
                f"{case_id}: operations empty without harness_scenario"
            )
        ops = tuple(self.operations)
        if not isinstance(self.oracle, FixtureOracle):
            raise FixtureCorpusError("oracle must be a FixtureOracle")
        if type(self.source_race_bytes_forbidden) is not bool:
            raise FixtureCorpusError("source_race_bytes_forbidden must be a bool")
        if type(self.change_is_bounded) is not bool:
            raise FixtureCorpusError("change_is_bounded must be a bool")
        excluded = _sorted_unique(self.pack_excluded_paths, "pack_excluded_paths")
        scenario = self.harness_scenario
        if scenario is not None:
            scenario = _text(scenario, "harness_scenario")
        if type(self.production_eligible) is not bool:
            raise FixtureCorpusError("production_eligible must be a bool")
        # Oracle candidates are never production-eligible model output.
        if self.production_eligible:
            raise FixtureCorpusError(
                "mutation cases are oracle/replay fixtures; production_eligible must be false"
            )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "operations", ops)
        object.__setattr__(self, "pack_excluded_paths", excluded)
        object.__setattr__(self, "harness_scenario", scenario)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_CASE_SCHEMA,
            "case_id": self.case_id,
            "category": self.category,
            "description": self.description,
            "operations": [op.to_dict() for op in self.operations],
            "oracle": self.oracle.to_dict(),
            "source_race_bytes_forbidden": self.source_race_bytes_forbidden,
            "change_is_bounded": self.change_is_bounded,
            "pack_excluded_paths": list(self.pack_excluded_paths),
            "harness_scenario": self.harness_scenario,
            "production_eligible": self.production_eligible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MutationCase":
        schema = payload.get("schema")
        if schema is not None and schema != MUTATION_CASE_SCHEMA:
            raise FixtureCorpusError(f"unsupported mutation schema {schema!r}")
        return cls(
            case_id=str(payload["case_id"]),
            category=str(payload["category"]),
            description=str(payload["description"]),
            operations=tuple(
                PathOperation.from_dict(item) for item in payload["operations"]
            ),
            oracle=FixtureOracle.from_dict(payload["oracle"]),
            source_race_bytes_forbidden=bool(payload["source_race_bytes_forbidden"]),
            change_is_bounded=bool(payload["change_is_bounded"]),
            pack_excluded_paths=tuple(payload.get("pack_excluded_paths") or ()),
            harness_scenario=payload.get("harness_scenario"),
            production_eligible=bool(payload.get("production_eligible", False)),
        )
