"""Strict, non-authoritative final-tree release reporting.

This module validates a report *about* an inspected tree.  It does not inspect
Git, execute a benchmark, create a checkpoint, promote an expert, accept a
proof, or complete a goal.  Those effects remain with their existing owners.

The report schema is intentionally the same closed projection consumed by the
VRIF owner.  Keeping a second, more permissive projection here would let a
report pass its public model while remaining unusable as root-goal evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    bounded_json_mapping,
    required_text,
)

REPORT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-intelligence-release-report@2"
_EXACT_UNSUPPORTED_CLAIMS: Final[tuple[str, ...]] = (
    "learned",
    "verified",
    "safe",
    "autonomous",
    "token-efficient",
    "production-ready",
)
FORBIDDEN_CLAIMS: Final[frozenset[str]] = frozenset(_EXACT_UNSUPPORTED_CLAIMS)
_EXACT_BLOCKERS: Final[tuple[str, ...]] = ("training_unavailable",)
_EXACT_NOT_RUN: Final[tuple[str, ...]] = (
    "gpu_live_qualification",
    "promotion",
    "training",
)
_TASK_FAMILIES: Final[tuple[str, ...]] = tuple(item.value for item in ResidualTaskFamily)
_CAPABILITY_UNAVAILABLE: Final = ExpertDisposition.CAPABILITY_UNAVAILABLE.value
_TREE_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_SHA256_ID_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPORT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "start_tree",
        "end_tree",
        "corpus_admission_id",
        "expert_dispositions",
        "before",
        "after",
        "costs",
        "promotion_eligible",
        "rollback_target",
        "gaps",
        "producer_artifacts",
        "files_symbols",
        "corpus_rights_splits",
        "architecture_tokenizer_checkpoint",
        "proof_validation",
        "drift",
        "rollback_blocker_eligibility",
    }
)
_DECLARED_OUTPUT_PATHS: Final[tuple[str, ...]] = (
    "docs/architecture/residual_intelligence_inventory/final_release_report.json",
    "docs/architecture/residual_intelligence_inventory/final_release_report.md",
    "test/api/residual_intelligence/test_release_report.py",
)
_REQUIRED_REPORT_PATHS: Final[tuple[str, ...]] = _DECLARED_OUTPUT_PATHS[:2]
_DECLARED_SYMBOLS: Final[tuple[str, ...]] = (
    "ResidualIntelligenceReleaseReport",
    "ResidualGapReport",
    "validate_release_claims",
)
_PRODUCER_TASK_ALIASES: Final[tuple[str, ...]] = (
    "VRIF-028",
    "VRIF-029",
    "VRIF-030",
    "VRIF-031",
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    return bounded_json_mapping(value, name)


def _git_oid(value: Any, name: str) -> str:
    result = required_text(value, name, max_bytes=40)
    if not _TREE_RE.fullmatch(result):
        raise ResidualIntelligenceError(f"{name} must be an exact 40-character Git object identity")
    return result


def _sha256_identity(value: Any, name: str) -> str:
    result = required_text(value, name, max_bytes=71)
    if not _SHA256_ID_RE.fullmatch(result):
        raise ResidualIntelligenceError(f"{name} must be a sha256 identity")
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def render_vrif_release_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the exact human companion for one closed VRIF machine report."""

    sections: list[tuple[str, Any]] = [
        (
            "Lineage",
            {
                "start_tree": report.get("start_tree"),
                "end_tree": report.get("end_tree"),
            },
        ),
        ("Files and Symbols", report.get("files_symbols")),
        ("Corpus Rights and Splits", report.get("corpus_rights_splits")),
        (
            "Architecture, Tokenizer, Checkpoint, and Training",
            report.get("architecture_tokenizer_checkpoint"),
        ),
        ("Expert Dispositions", report.get("expert_dispositions")),
        (
            "Before/After Denominators",
            {"before": report.get("before"), "after": report.get("after")},
        ),
        ("Costs and Break-even", report.get("costs")),
        ("Proof and Validation", report.get("proof_validation")),
        ("Drift", report.get("drift")),
        (
            "Rollback, Blockers, and Eligibility",
            report.get("rollback_blocker_eligibility"),
        ),
        ("Unsupported Gaps", report.get("gaps")),
    ]
    rendered = [
        "# VRIF Final Release Report\n\n",
        "This report is non-authoritative and cannot promote a residual expert.\n\n",
    ]
    for title, payload in sections:
        rendered.extend(
            (
                f"## {title}\n\n",
                "```json\n",
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "\n```\n\n",
            )
        )
    rendered.extend(
        (
            "## Complete Machine Report\n\n",
            "```json\n",
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "\n```\n",
        )
    )
    return "".join(rendered)


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ResidualIntelligenceError(f"{name} must be a non-negative integer")
    return value


def _require_exact_mapping_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    keys: frozenset[str],
) -> None:
    observed = set(value)
    unknown = sorted(observed - keys)
    missing = sorted(keys - observed)
    if unknown:
        raise ResidualIntelligenceError(f"{name} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise ResidualIntelligenceError(f"{name} is missing required fields: {', '.join(missing)}")


def _text_list(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ResidualIntelligenceError(f"{name} must be a list of strings")
    result = [required_text(item, f"{name} item") for item in value]
    if not allow_empty and not result:
        raise ResidualIntelligenceError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ResidualIntelligenceError(f"{name} contains duplicate entries")
    return result


def _require_literal(value: Any, expected: Any, name: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ResidualIntelligenceError(f"{name} does not match the closed report contract")


@dataclass(frozen=True)
class ResidualGapReport:
    """Explicit blockers, nonclaims, and checks that were not performed."""

    blockers: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    not_run: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("blockers", "unsupported_claims", "not_run"):
            values = tuple(required_text(item, f"{name} item") for item in getattr(self, name))
            if len(values) != len(set(values)):
                raise ResidualIntelligenceError(f"{name} contains duplicate entries")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "blockers": list(self.blockers),
            "unsupported_claims": list(self.unsupported_claims),
            "not_run": list(self.not_run),
        }


@dataclass(frozen=True)
class ResidualIntelligenceReleaseReport:
    """The exact bounded report projection consumed by the VRIF owner."""

    start_tree: str
    end_tree: str
    corpus_admission_id: str
    expert_dispositions: Mapping[str, Any]
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    costs: Mapping[str, Any]
    promotion_eligible: bool
    rollback_target: str
    gaps: ResidualGapReport
    producer_artifacts: Mapping[str, Any]
    files_symbols: Mapping[str, Any]
    corpus_rights_splits: Mapping[str, Any]
    architecture_tokenizer_checkpoint: Mapping[str, Any]
    proof_validation: Mapping[str, Any]
    drift: Mapping[str, Any]
    rollback_blocker_eligibility: Mapping[str, Any]
    schema: str = REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPORT_SCHEMA:
            raise ResidualIntelligenceError("unsupported release report schema")
        object.__setattr__(self, "start_tree", _git_oid(self.start_tree, "start_tree"))
        object.__setattr__(self, "end_tree", _git_oid(self.end_tree, "end_tree"))
        if self.start_tree == self.end_tree:
            raise ResidualIntelligenceError(
                "start_tree and end_tree must identify distinct snapshots"
            )
        object.__setattr__(
            self,
            "corpus_admission_id",
            required_text(self.corpus_admission_id, "corpus_admission_id"),
        )
        object.__setattr__(
            self,
            "rollback_target",
            _git_oid(self.rollback_target, "rollback_target"),
        )
        if type(self.promotion_eligible) is not bool:
            raise ResidualIntelligenceError("promotion_eligible must be boolean")
        if self.promotion_eligible:
            raise ResidualIntelligenceError("release reports cannot promote")
        if not isinstance(self.gaps, ResidualGapReport):
            raise ResidualIntelligenceError("gaps must be a ResidualGapReport")
        for name in (
            "expert_dispositions",
            "before",
            "after",
            "costs",
            "producer_artifacts",
            "files_symbols",
            "corpus_rights_splits",
            "architecture_tokenizer_checkpoint",
            "proof_validation",
            "drift",
            "rollback_blocker_eligibility",
        ):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "start_tree": self.start_tree,
            "end_tree": self.end_tree,
            "corpus_admission_id": self.corpus_admission_id,
            "expert_dispositions": dict(self.expert_dispositions),
            "before": dict(self.before),
            "after": dict(self.after),
            "costs": dict(self.costs),
            "promotion_eligible": False,
            "rollback_target": self.rollback_target,
            "gaps": self.gaps.to_dict(),
            "producer_artifacts": dict(self.producer_artifacts),
            "files_symbols": dict(self.files_symbols),
            "corpus_rights_splits": dict(self.corpus_rights_splits),
            "architecture_tokenizer_checkpoint": dict(self.architecture_tokenizer_checkpoint),
            "proof_validation": dict(self.proof_validation),
            "drift": dict(self.drift),
            "rollback_blocker_eligibility": dict(self.rollback_blocker_eligibility),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualIntelligenceReleaseReport:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("release report must be an object")
        unknown = sorted(str(key) for key in payload if key not in _REPORT_FIELDS)
        missing = sorted(_REPORT_FIELDS - set(payload))
        if unknown:
            raise ResidualIntelligenceError(
                f"release report contains unknown fields: {', '.join(unknown)}"
            )
        if missing:
            raise ResidualIntelligenceError(
                f"release report is missing required fields: {', '.join(missing)}"
            )
        gaps = payload.get("gaps")
        if not isinstance(gaps, Mapping) or set(gaps) != {
            "blockers",
            "unsupported_claims",
            "not_run",
        }:
            raise ResidualIntelligenceError(
                "gaps must bind blockers, unsupported_claims, and not_run"
            )
        blockers = _text_list(gaps.get("blockers"), "gaps.blockers")
        unsupported_claims = _text_list(gaps.get("unsupported_claims"), "gaps.unsupported_claims")
        not_run = _text_list(gaps.get("not_run"), "gaps.not_run")
        return cls(
            schema=payload.get("schema"),
            start_tree=payload.get("start_tree"),
            end_tree=payload.get("end_tree"),
            corpus_admission_id=payload.get("corpus_admission_id"),
            expert_dispositions=payload.get("expert_dispositions"),
            before=payload.get("before"),
            after=payload.get("after"),
            costs=payload.get("costs"),
            promotion_eligible=payload.get("promotion_eligible"),
            rollback_target=payload.get("rollback_target"),
            gaps=ResidualGapReport(
                blockers=tuple(blockers),
                unsupported_claims=tuple(unsupported_claims),
                not_run=tuple(not_run),
            ),
            producer_artifacts=payload.get("producer_artifacts"),
            files_symbols=payload.get("files_symbols"),
            corpus_rights_splits=payload.get("corpus_rights_splits"),
            architecture_tokenizer_checkpoint=payload.get("architecture_tokenizer_checkpoint"),
            proof_validation=payload.get("proof_validation"),
            drift=payload.get("drift"),
            rollback_blocker_eligibility=payload.get("rollback_blocker_eligibility"),
        )


def _validate_scores(value: Mapping[str, Any], *, name: str) -> None:
    _require_exact_mapping_keys(
        value,
        name=name,
        keys=frozenset({"accept", "abstain", "total", "denominators_by_family"}),
    )
    denominators = _mapping(value["denominators_by_family"], f"{name}.denominators_by_family")
    if set(denominators) != set(_TASK_FAMILIES):
        raise ResidualIntelligenceError(
            f"{name}.denominators_by_family must enumerate the exact task families"
        )
    if any(
        _nonnegative_int(count, f"{name}.denominators_by_family.{family}") != 4
        for family, count in denominators.items()
    ):
        raise ResidualIntelligenceError(
            f"{name}.denominators_by_family must retain four cases per family"
        )
    total = _nonnegative_int(value["total"], f"{name}.total")
    if total != sum(denominators.values()):
        raise ResidualIntelligenceError(f"{name}.total loses its family denominators")
    if (
        _nonnegative_int(value["accept"], f"{name}.accept") != 0
        or _nonnegative_int(value["abstain"], f"{name}.abstain") != total
    ):
        raise ResidualIntelligenceError(
            f"{name} must preserve the all-abstain, not-run benchmark result"
        )


def _validate_producer_artifacts(value: Mapping[str, Any]) -> str:
    name = "producer_artifacts"
    _require_exact_mapping_keys(
        value,
        name=name,
        keys=frozenset({"schema", "digest_algorithm", "tasks", "bundle_id"}),
    )
    _require_literal(
        value["schema"],
        "ipfs_accelerate_py/agent-supervisor/goal-terminal-producer-artifacts@1",
        f"{name}.schema",
    )
    _require_literal(value["digest_algorithm"], "sha256", f"{name}.digest_algorithm")
    tasks = value["tasks"]
    if not isinstance(tasks, list) or len(tasks) != len(_PRODUCER_TASK_ALIASES):
        raise ResidualIntelligenceError(f"{name}.tasks must enumerate the exact producer task set")
    observed_aliases: list[str] = []
    for index, task in enumerate(tasks):
        task_name = f"{name}.tasks[{index}]"
        task = _mapping(task, task_name)
        _require_exact_mapping_keys(
            task,
            name=task_name,
            keys=frozenset({"task_alias", "artifacts", "bundle_id"}),
        )
        alias = required_text(task["task_alias"], f"{task_name}.task_alias")
        observed_aliases.append(alias)
        artifacts = task["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ResidualIntelligenceError(f"{task_name}.artifacts must not be empty")
        paths: list[str] = []
        for artifact_index, artifact in enumerate(artifacts):
            artifact_name = f"{task_name}.artifacts[{artifact_index}]"
            artifact = _mapping(artifact, artifact_name)
            _require_exact_mapping_keys(
                artifact,
                name=artifact_name,
                keys=frozenset({"path", "blob_identity"}),
            )
            paths.append(required_text(artifact["path"], f"{artifact_name}.path"))
            _sha256_identity(artifact["blob_identity"], f"{artifact_name}.blob_identity")
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ResidualIntelligenceError(
                f"{task_name}.artifacts must contain sorted unique paths"
            )
        bundle_id = _sha256_identity(task["bundle_id"], f"{task_name}.bundle_id")
        task_body = dict(task)
        task_body.pop("bundle_id")
        if bundle_id != _canonical_sha256(task_body):
            raise ResidualIntelligenceError(f"{task_name}.bundle_id does not bind its artifacts")
    if tuple(observed_aliases) != _PRODUCER_TASK_ALIASES:
        raise ResidualIntelligenceError(f"{name}.tasks must use the exact ordered producer aliases")
    bundle_id = _sha256_identity(value["bundle_id"], f"{name}.bundle_id")
    body = dict(value)
    body.pop("bundle_id")
    if bundle_id != _canonical_sha256(body):
        raise ResidualIntelligenceError(f"{name}.bundle_id does not bind the task bundles")
    return bundle_id


def validate_release_claims(
    report: ResidualIntelligenceReleaseReport,
) -> ResidualIntelligenceReleaseReport:
    """Fail closed unless the report matches the owner's non-release semantics."""

    if not isinstance(report, ResidualIntelligenceReleaseReport):
        raise ResidualIntelligenceError("validate_release_claims requires a typed release report")
    if report.promotion_eligible:
        raise ResidualIntelligenceError("release reports cannot promote")
    if report.gaps.blockers != _EXACT_BLOCKERS:
        raise ResidualIntelligenceError(
            "release report blockers must be exactly training_unavailable"
        )
    if report.gaps.unsupported_claims != _EXACT_UNSUPPORTED_CLAIMS:
        raise ResidualIntelligenceError(
            "release report must enumerate every unsupported claim token in owner order"
        )
    if report.gaps.not_run != _EXACT_NOT_RUN:
        raise ResidualIntelligenceError(
            "release report not-run evidence must match the owner contract"
        )

    if set(report.expert_dispositions) != set(_TASK_FAMILIES) or any(
        value != _CAPABILITY_UNAVAILABLE for value in report.expert_dispositions.values()
    ):
        raise ResidualIntelligenceError(
            "expert_dispositions must map every task family to CAPABILITY_UNAVAILABLE"
        )

    _validate_scores(report.before, name="before")
    _validate_scores(report.after, name="after")
    if dict(report.before) != dict(report.after):
        raise ResidualIntelligenceError(
            "before and after must retain the same frozen benchmark scores"
        )
    _require_exact_mapping_keys(
        report.costs,
        name="costs",
        keys=frozenset({"tokens", "break_even"}),
    )
    if any(
        _nonnegative_int(report.costs[field], f"costs.{field}") != 0
        for field in ("tokens", "break_even")
    ):
        raise ResidualIntelligenceError("costs must be exactly zero tokens and zero break-even")

    producer_bundle_id = _validate_producer_artifacts(report.producer_artifacts)

    _require_exact_mapping_keys(
        report.files_symbols,
        name="files_symbols",
        keys=frozenset(
            {
                "disposition",
                "declared_output_paths",
                "required_report_paths",
                "declared_symbols",
                "producer_artifact_bundle_id",
            }
        ),
    )
    _require_literal(
        report.files_symbols["disposition"],
        "current_tracked_blobs_bound",
        "files_symbols.disposition",
    )
    declared_outputs = _text_list(
        report.files_symbols["declared_output_paths"],
        "files_symbols.declared_output_paths",
    )
    if tuple(declared_outputs) != _DECLARED_OUTPUT_PATHS:
        raise ResidualIntelligenceError(
            "files_symbols.declared_output_paths does not match VRIF-032"
        )
    required_reports = _text_list(
        report.files_symbols["required_report_paths"],
        "files_symbols.required_report_paths",
    )
    if tuple(required_reports) != _REQUIRED_REPORT_PATHS:
        raise ResidualIntelligenceError(
            "files_symbols.required_report_paths does not match the report pair"
        )
    declared_symbols = _text_list(
        report.files_symbols["declared_symbols"],
        "files_symbols.declared_symbols",
    )
    if tuple(declared_symbols) != _DECLARED_SYMBOLS:
        raise ResidualIntelligenceError(
            "files_symbols.declared_symbols does not match the public release API"
        )
    if (
        _sha256_identity(
            report.files_symbols["producer_artifact_bundle_id"],
            "files_symbols.producer_artifact_bundle_id",
        )
        != producer_bundle_id
    ):
        raise ResidualIntelligenceError("files_symbols does not bind the producer artifact bundle")

    _require_exact_mapping_keys(
        report.corpus_rights_splits,
        name="corpus_rights_splits",
        keys=frozenset(
            {
                "disposition",
                "admission_id",
                "corpus_root",
                "source_rights_root",
                "split_root",
                "partitions",
                "hidden_test_bodies_accessed",
                "privacy_disposition",
            }
        ),
    )
    corpus = report.corpus_rights_splits
    _require_literal(
        corpus["disposition"],
        "training_unavailable",
        "corpus_rights_splits.disposition",
    )
    if corpus["admission_id"] != report.corpus_admission_id:
        raise ResidualIntelligenceError(
            "corpus_rights_splits.admission_id does not match corpus_admission_id"
        )
    for field_name in ("corpus_root", "source_rights_root", "split_root"):
        required_text(corpus[field_name], f"corpus_rights_splits.{field_name}")
    _require_literal(
        corpus["partitions"],
        ["training", "development", "held_out", "adversarial"],
        "corpus_rights_splits.partitions",
    )
    _require_literal(
        corpus["hidden_test_bodies_accessed"],
        False,
        "corpus_rights_splits.hidden_test_bodies_accessed",
    )
    _require_literal(
        corpus["privacy_disposition"],
        "public_report_bounded",
        "corpus_rights_splits.privacy_disposition",
    )

    _require_exact_mapping_keys(
        report.architecture_tokenizer_checkpoint,
        name="architecture_tokenizer_checkpoint",
        keys=frozenset({"disposition", "architecture", "tokenizer", "checkpoint", "training"}),
    )
    _require_literal(
        dict(report.architecture_tokenizer_checkpoint),
        {
            "disposition": "training_unavailable",
            "architecture": "not_selected",
            "tokenizer": "no_learned_tokenizer_admitted",
            "checkpoint": "not_created",
            "training": "not_attempted",
        },
        "architecture_tokenizer_checkpoint",
    )

    proof = report.proof_validation
    _require_exact_mapping_keys(
        proof,
        name="proof_validation",
        keys=frozenset(
            {
                "disposition",
                "validation_commands",
                "producer_artifact_bundle_id",
                "benchmark_freeze_id",
                "benchmark_case_root",
                "benchmark_binding_set_id",
                "paired_baseline_id",
                "benchmark_case_payload_disposition",
                "benchmark_evaluation_disposition",
                "producer_database_portal_validations",
                "terminal_database_portal_validation",
                "report_authoritative",
            }
        ),
    )
    _require_literal(
        proof["disposition"],
        "owner_receipts_required",
        "proof_validation.disposition",
    )
    commands = proof["validation_commands"]
    if not isinstance(commands, list) or not commands:
        raise ResidualIntelligenceError("proof_validation.validation_commands must not be empty")
    for index, command in enumerate(commands):
        _text_list(command, f"proof_validation.validation_commands[{index}]")
    if (
        _sha256_identity(
            proof["producer_artifact_bundle_id"],
            "proof_validation.producer_artifact_bundle_id",
        )
        != producer_bundle_id
    ):
        raise ResidualIntelligenceError(
            "proof_validation does not bind the producer artifact bundle"
        )
    for field_name in (
        "benchmark_freeze_id",
        "benchmark_case_root",
        "benchmark_binding_set_id",
        "paired_baseline_id",
    ):
        _sha256_identity(proof[field_name], f"proof_validation.{field_name}")
    _require_literal(
        proof["benchmark_case_payload_disposition"],
        "payload_unavailable_training_unavailable",
        "proof_validation.benchmark_case_payload_disposition",
    )
    _require_literal(
        proof["benchmark_evaluation_disposition"],
        "all_abstain_not_run",
        "proof_validation.benchmark_evaluation_disposition",
    )
    for field_name in (
        "producer_database_portal_validations",
        "terminal_database_portal_validation",
    ):
        _require_literal(proof[field_name], "required", f"proof_validation.{field_name}")
    _require_literal(
        proof["report_authoritative"],
        False,
        "proof_validation.report_authoritative",
    )

    _require_exact_mapping_keys(
        report.drift,
        name="drift",
        keys=frozenset(
            {
                "disposition",
                "reference_tree",
                "evaluated_tree",
                "checkpoint_available",
                "detectors_run",
                "reason_codes",
            }
        ),
    )
    drift = report.drift
    _require_literal(
        drift["disposition"],
        "not_run_training_unavailable",
        "drift.disposition",
    )
    if drift["reference_tree"] != report.start_tree or drift["evaluated_tree"] != report.end_tree:
        raise ResidualIntelligenceError(
            "drift tree bindings do not match the inspected report trees"
        )
    _require_literal(drift["checkpoint_available"], False, "drift.checkpoint_available")
    _require_literal(drift["detectors_run"], [], "drift.detectors_run")
    _require_literal(
        drift["reason_codes"],
        ["no_admitted_checkpoint", "training_unavailable"],
        "drift.reason_codes",
    )

    rollback = report.rollback_blocker_eligibility
    _require_exact_mapping_keys(
        rollback,
        name="rollback_blocker_eligibility",
        keys=frozenset(
            {
                "promotion_eligible",
                "rollback_target",
                "blockers",
                "not_run",
                "report_authority",
            }
        ),
    )
    _require_literal(
        rollback["promotion_eligible"],
        False,
        "rollback_blocker_eligibility.promotion_eligible",
    )
    if rollback["rollback_target"] != report.rollback_target:
        raise ResidualIntelligenceError(
            "rollback_blocker_eligibility.rollback_target does not match rollback_target"
        )
    _require_literal(
        rollback["blockers"],
        list(_EXACT_BLOCKERS),
        "rollback_blocker_eligibility.blockers",
    )
    _require_literal(
        rollback["not_run"],
        list(_EXACT_NOT_RUN),
        "rollback_blocker_eligibility.not_run",
    )
    _require_literal(
        rollback["report_authority"],
        "non_authoritative",
        "rollback_blocker_eligibility.report_authority",
    )
    return report


__all__ = (
    "FORBIDDEN_CLAIMS",
    "REPORT_SCHEMA",
    "ResidualGapReport",
    "ResidualIntelligenceReleaseReport",
    "render_vrif_release_report_markdown",
    "validate_release_claims",
)
