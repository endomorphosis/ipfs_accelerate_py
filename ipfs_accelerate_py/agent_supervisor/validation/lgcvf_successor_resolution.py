"""Validate the append-only LGCVF successor resolution.

The original ``lgcvf-successor-tasks@1`` object remains immutable.  This
module binds a terminal R&D disposition to each of its three tasks without
turning self-verification into release or production authority.

Authority receipt signature and issuer policy belong to a separate validator.
Callers inject that validator through :class:`AuthorityReceiptValidator`; its
content-addressed verdict is sealed into this resolution.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

PREDECESSOR_SCHEMA: Final[str] = "lgcvf-successor-tasks@1"
RESOLUTION_SCHEMA: Final[str] = "lgcvf-successor-resolution@1"
AUTHORITY_EVIDENCE_SCHEMA: Final[str] = (
    "lgcvf-successor-authority-resolution-evidence@1"
)
S003_EVIDENCE_SCHEMA: Final[str] = "lgcvf-s003-completion-evidence@1"
VALIDATION_EVIDENCE_SCHEMA: Final[str] = "lgcvf-exact-validation-evidence@1"
AUTHORITY_CONTEXT_SCHEMA: Final[str] = "lgcvf-successor-authority-context@1"
AUTHORITY_VALIDATION_SCHEMA: Final[str] = "lgcvf-successor-authority-validation@1"

EXPECTED_TASK_IDS: Final[tuple[str, ...]] = (
    "LGCVF-S001",
    "LGCVF-S002",
    "LGCVF-S003",
)
EXPECTED_DISPOSITIONS: Final[dict[str, str]] = {
    "LGCVF-S001": "self_verified_r_and_d",
    "LGCVF-S002": "production_declined_r_and_d",
    "LGCVF-S003": "completed",
}
EXPECTED_PREDECESSOR_STATUSES: Final[dict[str, str]] = {
    "LGCVF-S001": "blocked_external_authority",
    "LGCVF-S002": "blocked_manual",
    "LGCVF-S003": "todo",
}
EXPECTED_DEPENDENCIES: Final[dict[str, tuple[str, ...]]] = {
    "LGCVF-S001": (),
    "LGCVF-S002": ("LGCVF-S001",),
    "LGCVF-S003": ("LGCVF-S001",),
}
DERIVED_STATES: Final[dict[str, bool]] = {
    "task_implementation_complete": True,
    "objective_complete": False,
    "release_qualified": False,
    "production_authorized": False,
}
MANDATORY_LIMITATIONS: Final[tuple[str, ...]] = (
    "S001 is self-verified R&D evidence, not independent release qualification",
    "S002 explicitly declines production authorization",
    "objective completion and release qualification remain unclaimed",
)

_CID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^baguqeera[a-z2-7]{52}$")
_GIT_OBJECT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40,64}$")

AuthorityReceiptValidator = Callable[..., Mapping[str, Any]]


class LgcvfSuccessorResolutionError(RuntimeError):
    """The successor resolution is malformed, stale, or unsupported."""


def _mapping(value: Any, *, noun: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LgcvfSuccessorResolutionError(f"{noun} is not an object")
    return dict(value)


def _closed(value: Any, fields: set[str], *, noun: str) -> dict[str, Any]:
    result = _mapping(value, noun=noun)
    if set(result) != fields:
        raise LgcvfSuccessorResolutionError(f"{noun} fields differ")
    return result


def _cid(value: Any, *, noun: str) -> str:
    if not isinstance(value, str) or _CID_PATTERN.fullmatch(value) is None:
        raise LgcvfSuccessorResolutionError(f"{noun} is not a canonical CID")
    return value


def _git_object(value: Any, *, noun: str) -> str:
    if not isinstance(value, str) or _GIT_OBJECT_PATTERN.fullmatch(value) is None:
        raise LgcvfSuccessorResolutionError(f"{noun} is not a Git object ID")
    return value


def _verify_identity(value: Mapping[str, Any], field: str, *, noun: str) -> str:
    claimed = _cid(value.get(field), noun=f"{noun} {field}")
    body = {key: item for key, item in value.items() if key != field}
    if content_identity(body) != claimed:
        raise LgcvfSuccessorResolutionError(f"{noun} content identity differs")
    return claimed


def _source_roots(value: Any, *, noun: str) -> dict[str, dict[str, str]]:
    roots = _closed(
        value,
        {"ipfs_accelerate_py", "ipfs_datasets_py"},
        noun=noun,
    )
    accelerator = _closed(
        roots["ipfs_accelerate_py"],
        {"head", "tree"},
        noun=f"{noun} accelerator",
    )
    datasets = _closed(
        roots["ipfs_datasets_py"],
        {"head", "tree", "gitlink"},
        noun=f"{noun} datasets",
    )
    normalized = {
        "ipfs_accelerate_py": {
            "head": _git_object(accelerator["head"], noun="accelerator head"),
            "tree": _git_object(accelerator["tree"], noun="accelerator tree"),
        },
        "ipfs_datasets_py": {
            "head": _git_object(datasets["head"], noun="datasets head"),
            "tree": _git_object(datasets["tree"], noun="datasets tree"),
            "gitlink": _git_object(datasets["gitlink"], noun="datasets gitlink"),
        },
    }
    if (
        normalized["ipfs_datasets_py"]["head"]
        != normalized["ipfs_datasets_py"]["gitlink"]
    ):
        raise LgcvfSuccessorResolutionError(
            "datasets head differs from the accelerator gitlink"
        )
    return normalized


def _validate_predecessor(
    value: Mapping[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]]:
    predecessor = _mapping(value, noun="predecessor successor tasks")
    if predecessor.get("schema") != PREDECESSOR_SCHEMA:
        raise LgcvfSuccessorResolutionError("predecessor successor schema differs")
    predecessor_cid = _verify_identity(
        predecessor,
        "successor_tasks_cid",
        noun="predecessor successor tasks",
    )
    for field in ("objective_complete", "release_qualified", "production_authorized"):
        if predecessor.get(field) is not False:
            raise LgcvfSuccessorResolutionError(
                f"predecessor raises unsupported authority: {field}"
            )
    tasks = predecessor.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(EXPECTED_TASK_IDS):
        raise LgcvfSuccessorResolutionError("predecessor task population differs")
    indexed: dict[str, dict[str, Any]] = {}
    for task in tasks:
        current = _mapping(task, noun="predecessor task")
        task_id = current.get("task_id")
        if task_id not in EXPECTED_TASK_IDS or task_id in indexed:
            raise LgcvfSuccessorResolutionError("predecessor task identity differs")
        _verify_identity(current, "task_cid", noun=str(task_id))
        dependencies = current.get("depends_on")
        if (
            current.get("status") != EXPECTED_PREDECESSOR_STATUSES[str(task_id)]
            or not isinstance(dependencies, list)
            or tuple(dependencies) != EXPECTED_DEPENDENCIES[str(task_id)]
        ):
            raise LgcvfSuccessorResolutionError(
                f"{task_id} predecessor status or dependencies differ"
            )
        indexed[str(task_id)] = current
    if tuple(item.get("task_id") for item in tasks) != EXPECTED_TASK_IDS:
        raise LgcvfSuccessorResolutionError(
            "predecessor tasks are not canonically ordered"
        )
    return predecessor_cid, indexed


def _validate_qualification(value: Mapping[str, Any], *, plan_cid: str) -> str:
    qualification = _mapping(value, noun="qualification result")
    qualification_cid = _verify_identity(
        qualification,
        "result_cid",
        noun="qualification result",
    )
    if (
        qualification.get("schema") != "lgcvf-independent-hermetic-qualification@1"
        or qualification.get("plan_cid") != plan_cid
        or qualification.get("passed") is not True
        or qualification.get("test_qualification_complete") is not True
    ):
        raise LgcvfSuccessorResolutionError("qualification binding is unsuccessful")
    for field in DERIVED_STATES:
        if qualification.get(field) is not False:
            raise LgcvfSuccessorResolutionError(
                f"qualification raises unsupported state: {field}"
            )
    return qualification_cid


def _validate_benchmark(value: Mapping[str, Any]) -> str:
    benchmark = _mapping(value, noun="benchmark result")
    benchmark_cid = _verify_identity(benchmark, "report_cid", noun="benchmark result")
    if (
        benchmark.get("schema") != "lgcvf-symbolic-displacement-benchmark@1"
        or benchmark.get("production_authoritative") is not False
        or benchmark.get("release_qualified") is not False
        or benchmark.get("production_authorized") is not False
        or benchmark.get("overall_disposition")
        not in {
            "partial",
            "no_go",
            "development_targets_met",
        }
    ):
        raise LgcvfSuccessorResolutionError("benchmark authority binding differs")
    return benchmark_cid


def _authority_context(
    *,
    task_id: str,
    disposition: str,
    predecessor_cid: str,
    predecessor_task_cid: str,
    plan_cid: str,
    qualification_cid: str,
    benchmark_cid: str,
    source_roots: Mapping[str, Any],
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "schema": AUTHORITY_CONTEXT_SCHEMA,
        "task_id": task_id,
        "disposition": disposition,
        "predecessor_successor_tasks_cid": predecessor_cid,
        "predecessor_task_cid": predecessor_task_cid,
        "plan_cid": plan_cid,
        "qualification_result_cid": qualification_cid,
        "benchmark_report_cid": benchmark_cid,
        "source_roots": dict(source_roots),
    }
    context["context_cid"] = content_identity(context)
    return context


def _authority_verdict(
    validator: AuthorityReceiptValidator,
    *,
    task_id: str,
    disposition: str,
    receipt: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_value = _mapping(receipt, noun=f"{task_id} authority receipt")
    receipt_cid = _verify_identity(
        receipt_value,
        "receipt_cid",
        noun=f"{task_id} authority receipt",
    )
    try:
        raw = validator(
            task_id=task_id,
            disposition=disposition,
            receipt=receipt_value,
            context=dict(context),
        )
    except LgcvfSuccessorResolutionError:
        raise
    except Exception as exc:
        raise LgcvfSuccessorResolutionError(
            f"{task_id} authority validator failed: {type(exc).__name__}"
        ) from exc
    verdict = _closed(
        raw,
        {
            "schema",
            "valid",
            "signed",
            "task_id",
            "disposition",
            "receipt_cid",
            "context_cid",
            "release_qualified",
            "production_authorized",
            "validation_cid",
        },
        noun=f"{task_id} authority verdict",
    )
    claimed = _verify_identity(
        verdict,
        "validation_cid",
        noun=f"{task_id} authority verdict",
    )
    if (
        verdict.get("schema") != AUTHORITY_VALIDATION_SCHEMA
        or verdict.get("valid") is not True
        or verdict.get("signed") is not True
        or verdict.get("task_id") != task_id
        or verdict.get("disposition") != disposition
        or verdict.get("receipt_cid") != receipt_cid
        or verdict.get("context_cid") != context.get("context_cid")
        or verdict.get("release_qualified") is not False
        or verdict.get("production_authorized") is not False
    ):
        raise LgcvfSuccessorResolutionError(
            f"{task_id} authority verdict does not admit the exact receipt"
        )
    if (
        receipt_value.get("disposition") != disposition
        or receipt_value.get("release_qualified") is not False
        or receipt_value.get("production_authorized") is not False
    ):
        raise LgcvfSuccessorResolutionError(
            f"{task_id} receipt disposition or authority ceiling differs"
        )
    assert claimed == verdict["validation_cid"]
    return verdict


def _command_manifest_path(command: str, *, owning_repository: str) -> str:
    try:
        arguments = shlex.split(command)
    except ValueError as exc:
        raise LgcvfSuccessorResolutionError(
            "S003 validation command is malformed"
        ) from exc
    candidates = [item.split("::", 1)[0] for item in arguments if ".py" in item]
    if len(candidates) != 1:
        raise LgcvfSuccessorResolutionError(
            "S003 validation command does not identify one Python test file"
        )
    candidate = PurePosixPath(candidates[0])
    if candidate.is_absolute() or ".." in candidate.parts:
        raise LgcvfSuccessorResolutionError("S003 validation path is unsafe")
    parts = candidate.parts
    if parts and parts[0] == owning_repository:
        parts = parts[1:]
    if not parts:
        raise LgcvfSuccessorResolutionError("S003 validation path is empty")
    normalized = PurePosixPath(*parts).as_posix()
    if not normalized.startswith("tests/"):
        raise LgcvfSuccessorResolutionError(
            "S003 validation is outside the datasets test tree"
        )
    return normalized


def _s003_suite(qualification: Mapping[str, Any]) -> dict[str, Any]:
    suites = qualification.get("suites")
    if not isinstance(suites, list):
        raise LgcvfSuccessorResolutionError("qualification suite population is absent")
    matches = [
        _mapping(item, noun="qualification suite")
        for item in suites
        if isinstance(item, Mapping)
        and item.get("suite_id") == "fixed_datasets_semantics"
    ]
    if len(matches) != 1:
        raise LgcvfSuccessorResolutionError(
            "fixed datasets qualification suite is absent or ambiguous"
        )
    suite = matches[0]
    if (
        suite.get("passed") is not True
        or suite.get("exit_code") != 0
        or any(
            suite.get(field) != 0
            for field in (
                "failed_count",
                "skipped_count",
                "xfailed_count",
                "xpassed_count",
                "error_count",
            )
        )
        or isinstance(suite.get("passed_count"), bool)
        or not isinstance(suite.get("passed_count"), int)
        or int(suite["passed_count"]) < 1
    ):
        raise LgcvfSuccessorResolutionError(
            "fixed datasets qualification suite is not an exact pass"
        )
    _cid(suite.get("observation_cid"), noun="datasets suite observation CID")
    _cid(suite.get("nodeids_cid"), noun="datasets suite node IDs CID")
    return suite


def _build_s003_evidence(
    task: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    qualification_cid: str,
    source_roots: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if task.get("owning_repository") != "ipfs_datasets_py":
        raise LgcvfSuccessorResolutionError("S003 owner differs")
    commands = task.get("validation")
    if not isinstance(commands, list) or not commands:
        raise LgcvfSuccessorResolutionError("S003 has no exact validation commands")
    suite = _s003_suite(qualification)
    manifest = _mapping(suite.get("manifest"), noun="datasets suite manifest")
    if manifest.get("owner_root") != "ipfs_datasets_py":
        raise LgcvfSuccessorResolutionError("datasets suite owner differs")
    manifest_paths = manifest.get("paths")
    if not isinstance(manifest_paths, list) or any(
        not isinstance(item, str) for item in manifest_paths
    ):
        raise LgcvfSuccessorResolutionError("datasets suite manifest paths differ")
    validations: list[dict[str, Any]] = []
    for command in commands:
        if not isinstance(command, str) or not command.strip():
            raise LgcvfSuccessorResolutionError("S003 validation command is invalid")
        manifest_path = _command_manifest_path(
            command,
            owning_repository="ipfs_datasets_py",
        )
        if manifest_paths.count(manifest_path) != 1:
            raise LgcvfSuccessorResolutionError(
                "S003 validation is not uniquely present in protected qualification"
            )
        record: dict[str, Any] = {
            "schema": VALIDATION_EVIDENCE_SCHEMA,
            "command": command,
            "manifest_path": manifest_path,
            "passed": True,
            "exit_code": 0,
            "qualification_observation_cid": suite["observation_cid"],
        }
        record["validation_cid"] = content_identity(record)
        validations.append(record)
    datasets = source_roots["ipfs_datasets_py"]
    evidence: dict[str, Any] = {
        "schema": S003_EVIDENCE_SCHEMA,
        "datasets_commit": datasets["head"],
        "datasets_tree": datasets["tree"],
        "qualification_result_cid": qualification_cid,
        "qualification_suite_id": "fixed_datasets_semantics",
        "qualification_suite_nodeids_cid": suite["nodeids_cid"],
        "qualification_observation_cid": suite["observation_cid"],
        "validations": validations,
    }
    evidence["evidence_cid"] = content_identity(evidence)
    return evidence


def _validate_s003_evidence(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    evidence = _mapping(observed, noun="S003 completion evidence")
    _verify_identity(evidence, "evidence_cid", noun="S003 completion evidence")
    validations = evidence.get("validations")
    if isinstance(validations, list):
        for index, value in enumerate(validations):
            record = _mapping(value, noun=f"S003 validation {index}")
            _verify_identity(record, "validation_cid", noun=f"S003 validation {index}")
    if evidence != dict(expected):
        raise LgcvfSuccessorResolutionError(
            "S003 completion evidence differs from protected qualification"
        )


def _authority_evidence(
    *,
    task_id: str,
    disposition: str,
    receipt: Mapping[str, Any],
    validator: AuthorityReceiptValidator,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = _authority_verdict(
        validator,
        task_id=task_id,
        disposition=disposition,
        receipt=receipt,
        context=context,
    )
    evidence: dict[str, Any] = {
        "schema": AUTHORITY_EVIDENCE_SCHEMA,
        "authority_receipt_cid": verdict["receipt_cid"],
        "authority_context_cid": context["context_cid"],
        "authority_validation_cid": verdict["validation_cid"],
    }
    evidence["evidence_cid"] = content_identity(evidence)
    return evidence


def build_successor_resolution(
    *,
    predecessor: Mapping[str, Any],
    qualification: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    source_roots: Mapping[str, Any],
    authority_receipts: Mapping[str, Mapping[str, Any]],
    authority_validator: AuthorityReceiptValidator,
    limitations: Sequence[str] = MANDATORY_LIMITATIONS,
) -> dict[str, Any]:
    """Build and immediately revalidate one append-only resolution object."""

    predecessor_cid, predecessor_tasks = _validate_predecessor(predecessor)
    plan_cid = _cid(predecessor.get("plan_cid"), noun="plan CID")
    qualification_cid = _validate_qualification(qualification, plan_cid=plan_cid)
    benchmark_cid = _validate_benchmark(benchmark)
    if (
        predecessor.get("qualification_cid") != qualification_cid
        or predecessor.get("benchmark_cid") != benchmark_cid
    ):
        raise LgcvfSuccessorResolutionError(
            "predecessor evidence CIDs differ from current evidence"
        )
    roots = _source_roots(source_roots, noun="source roots")
    resolved_tasks: list[dict[str, Any]] = []
    for task_id in EXPECTED_TASK_IDS:
        original = predecessor_tasks[task_id]
        disposition = EXPECTED_DISPOSITIONS[task_id]
        context = _authority_context(
            task_id=task_id,
            disposition=disposition,
            predecessor_cid=predecessor_cid,
            predecessor_task_cid=str(original["task_cid"]),
            plan_cid=plan_cid,
            qualification_cid=qualification_cid,
            benchmark_cid=benchmark_cid,
            source_roots=roots,
        )
        if task_id in {"LGCVF-S001", "LGCVF-S002"}:
            receipt = authority_receipts.get(task_id)
            if not isinstance(receipt, Mapping):
                raise LgcvfSuccessorResolutionError(
                    f"{task_id} signed authority receipt is absent"
                )
            evidence = _authority_evidence(
                task_id=task_id,
                disposition=disposition,
                receipt=receipt,
                validator=authority_validator,
                context=context,
            )
        else:
            evidence = _build_s003_evidence(
                original,
                qualification=qualification,
                qualification_cid=qualification_cid,
                source_roots=roots,
            )
        resolution: dict[str, Any] = {
            "task_id": task_id,
            "predecessor_task_cid": original["task_cid"],
            "disposition": disposition,
            "depends_on": list(EXPECTED_DEPENDENCIES[task_id]),
            "evidence": evidence,
        }
        resolution["task_resolution_cid"] = content_identity(resolution)
        resolved_tasks.append(resolution)
    limitation_list = list(limitations)
    value: dict[str, Any] = {
        "schema": RESOLUTION_SCHEMA,
        "predecessor_successor_tasks_cid": predecessor_cid,
        "plan_cid": plan_cid,
        "qualification_result_cid": qualification_cid,
        "benchmark_report_cid": benchmark_cid,
        "source_roots": roots,
        "tasks": resolved_tasks,
        "derived_states": dict(DERIVED_STATES),
        "limitations": limitation_list,
    }
    value["resolution_cid"] = content_identity(value)
    validate_successor_resolution(
        value,
        predecessor=predecessor,
        qualification=qualification,
        benchmark=benchmark,
        expected_source_roots=roots,
        authority_receipts=authority_receipts,
        authority_validator=authority_validator,
    )
    return value


def validate_successor_resolution(
    value: Mapping[str, Any],
    *,
    predecessor: Mapping[str, Any],
    qualification: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    expected_source_roots: Mapping[str, Any],
    authority_receipts: Mapping[str, Mapping[str, Any]],
    authority_validator: AuthorityReceiptValidator,
) -> dict[str, Any]:
    """Fail closed unless ``value`` resolves the exact predecessor and roots."""

    resolution = _closed(
        value,
        {
            "schema",
            "predecessor_successor_tasks_cid",
            "plan_cid",
            "qualification_result_cid",
            "benchmark_report_cid",
            "source_roots",
            "tasks",
            "derived_states",
            "limitations",
            "resolution_cid",
        },
        noun="successor resolution",
    )
    if resolution.get("schema") != RESOLUTION_SCHEMA:
        raise LgcvfSuccessorResolutionError("successor resolution schema differs")
    resolution_cid = _verify_identity(
        resolution,
        "resolution_cid",
        noun="successor resolution",
    )
    predecessor_cid, predecessor_tasks = _validate_predecessor(predecessor)
    plan_cid = _cid(predecessor.get("plan_cid"), noun="plan CID")
    qualification_cid = _validate_qualification(qualification, plan_cid=plan_cid)
    benchmark_cid = _validate_benchmark(benchmark)
    roots = _source_roots(
        resolution.get("source_roots"), noun="resolution source roots"
    )
    expected_roots = _source_roots(expected_source_roots, noun="expected source roots")
    expected_bindings = {
        "predecessor_successor_tasks_cid": predecessor_cid,
        "plan_cid": plan_cid,
        "qualification_result_cid": qualification_cid,
        "benchmark_report_cid": benchmark_cid,
    }
    if any(
        resolution.get(field) != expected
        for field, expected in expected_bindings.items()
    ):
        raise LgcvfSuccessorResolutionError("successor resolution root binding differs")
    if (
        predecessor.get("qualification_cid") != qualification_cid
        or predecessor.get("benchmark_cid") != benchmark_cid
        or roots != expected_roots
    ):
        raise LgcvfSuccessorResolutionError(
            "successor resolution evidence root differs"
        )
    if resolution.get("derived_states") != DERIVED_STATES:
        raise LgcvfSuccessorResolutionError("successor derived states differ")
    limitations = resolution.get("limitations")
    if (
        not isinstance(limitations, list)
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
        or len(limitations) != len(set(limitations))
        or not set(MANDATORY_LIMITATIONS).issubset(limitations)
    ):
        raise LgcvfSuccessorResolutionError("successor limitations are incomplete")
    tasks = resolution.get("tasks")
    if (
        not isinstance(tasks, list)
        or tuple(
            item.get("task_id") if isinstance(item, Mapping) else None for item in tasks
        )
        != EXPECTED_TASK_IDS
    ):
        raise LgcvfSuccessorResolutionError("successor resolution task order differs")
    resolved: set[str] = set()
    authority_cids: set[str] = set()
    for task_id, raw in zip(EXPECTED_TASK_IDS, tasks, strict=True):
        task = _closed(
            raw,
            {
                "task_id",
                "predecessor_task_cid",
                "disposition",
                "depends_on",
                "evidence",
                "task_resolution_cid",
            },
            noun=f"{task_id} resolution",
        )
        _verify_identity(task, "task_resolution_cid", noun=f"{task_id} resolution")
        dependencies = EXPECTED_DEPENDENCIES[task_id]
        if (
            task.get("task_id") != task_id
            or task.get("predecessor_task_cid")
            != predecessor_tasks[task_id]["task_cid"]
            or task.get("disposition") != EXPECTED_DISPOSITIONS[task_id]
            or task.get("depends_on") != list(dependencies)
            or any(dependency not in resolved for dependency in dependencies)
        ):
            raise LgcvfSuccessorResolutionError(
                f"{task_id} resolution identity, disposition, or dependency differs"
            )
        if task_id in {"LGCVF-S001", "LGCVF-S002"}:
            evidence = _closed(
                task.get("evidence"),
                {
                    "schema",
                    "authority_receipt_cid",
                    "authority_context_cid",
                    "authority_validation_cid",
                    "evidence_cid",
                },
                noun=f"{task_id} authority evidence",
            )
            _verify_identity(
                evidence, "evidence_cid", noun=f"{task_id} authority evidence"
            )
            if evidence.get("schema") != AUTHORITY_EVIDENCE_SCHEMA:
                raise LgcvfSuccessorResolutionError(
                    f"{task_id} authority evidence schema differs"
                )
            receipt = authority_receipts.get(task_id)
            if not isinstance(receipt, Mapping):
                raise LgcvfSuccessorResolutionError(
                    f"{task_id} signed authority receipt is absent"
                )
            context = _authority_context(
                task_id=task_id,
                disposition=EXPECTED_DISPOSITIONS[task_id],
                predecessor_cid=predecessor_cid,
                predecessor_task_cid=str(predecessor_tasks[task_id]["task_cid"]),
                plan_cid=plan_cid,
                qualification_cid=qualification_cid,
                benchmark_cid=benchmark_cid,
                source_roots=roots,
            )
            verdict = _authority_verdict(
                authority_validator,
                task_id=task_id,
                disposition=EXPECTED_DISPOSITIONS[task_id],
                receipt=receipt,
                context=context,
            )
            if (
                evidence.get("authority_receipt_cid") != verdict["receipt_cid"]
                or evidence.get("authority_context_cid") != context["context_cid"]
                or evidence.get("authority_validation_cid") != verdict["validation_cid"]
                or verdict["receipt_cid"] in authority_cids
            ):
                raise LgcvfSuccessorResolutionError(
                    f"{task_id} authority evidence differs or is replayed"
                )
            authority_cids.add(str(verdict["receipt_cid"]))
        else:
            expected_evidence = _build_s003_evidence(
                predecessor_tasks[task_id],
                qualification=qualification,
                qualification_cid=qualification_cid,
                source_roots=roots,
            )
            _validate_s003_evidence(
                _mapping(task.get("evidence"), noun="S003 completion evidence"),
                expected_evidence,
            )
        resolved.add(task_id)
    return {
        "schema": "lgcvf-successor-resolution-validation@1",
        "valid": True,
        "resolution_cid": resolution_cid,
        "predecessor_successor_tasks_cid": predecessor_cid,
        "resolved_task_ids": list(EXPECTED_TASK_IDS),
        **DERIVED_STATES,
    }


__all__ = (
    "AUTHORITY_CONTEXT_SCHEMA",
    "AUTHORITY_VALIDATION_SCHEMA",
    "DERIVED_STATES",
    "EXPECTED_DISPOSITIONS",
    "EXPECTED_TASK_IDS",
    "MANDATORY_LIMITATIONS",
    "RESOLUTION_SCHEMA",
    "AuthorityReceiptValidator",
    "LgcvfSuccessorResolutionError",
    "build_successor_resolution",
    "validate_successor_resolution",
)
