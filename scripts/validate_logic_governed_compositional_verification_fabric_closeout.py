#!/usr/bin/env python3
"""Validate LGCVF release and implementation reports against current evidence.

This protected judge validates content and lineage; it cannot grant external
qualification or operator authorization.  In the current hermetic cohort a
truthful partial/no-go is valid task output while a release/production claim
fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)

PLAN_CID: Final[str] = "baguqeerabxn5kkewz44v4chz6vbt3kcozfj4rvhh4gpdp54645blhemhvloq"
RELEASE_PATH: Final[Path] = (
    ROOT / "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_RELEASE.md"
)
IMPLEMENTATION_PATH: Final[Path] = (
    ROOT
    / "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md"
)
SUCCESSORS_PATH: Final[Path] = (
    ROOT
    / "data/agent_supervisor/logic_governed_compositional_verification_fabric"
    / "successor_tasks.json"
)
BENCHMARK_PATH: Final[Path] = (
    ROOT
    / "data/agent_supervisor/logic_governed_compositional_verification_fabric"
    / "benchmark_result.json"
)
QUALIFICATION_PATH: Final[Path] = (
    ROOT
    / "data/agent_supervisor/logic_governed_compositional_verification_fabric"
    / "independent_qualification_result.json"
)
QUALIFICATION_VALIDATOR_PATH: Final[Path] = (
    ROOT / "scripts/qualify_logic_governed_compositional_verification_fabric.py"
)
BENCHMARK_VALIDATOR_PATH: Final[Path] = (
    ROOT / "scripts/benchmark_lgcvf_symbolic_displacement.py"
)
# A protected qualification worker has its own 900-second CPU ceiling. The
# outer judge must also cover sandbox setup, teardown, and the benchmark
# fixture replay that follows its nested qualification.
PROTECTED_REPLAY_TIMEOUT_SECONDS: Final[int] = 1_200
_EVIDENCE_ONLY_PATHS: Final[frozenset[str]] = frozenset(
    {
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/independent_qualification_result.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/external_qualification_receipt.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/production_authorization_receipt.json",
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/successor_tasks.json",
        "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md",
        "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_RELEASE.md",
    }
)


class CloseoutValidationError(RuntimeError):
    """A report or its evidence lineage is invalid."""


@dataclass(frozen=True)
class _ReplayedEvidence:
    """Evidence admitted only after an isolated protected reconstruction."""

    benchmark: dict[str, Any]
    benchmark_cid: str
    benchmark_authority_cid: str
    benchmark_replay_cid: str
    qualification: dict[str, Any]
    qualification_cid: str
    qualification_authority_cid: str
    qualification_replay_cid: str
    benchmark_validator_sha256: str
    qualification_validator_sha256: str


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloseoutValidationError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise CloseoutValidationError(f"{label} root is not an object")
    return value


def _read_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloseoutValidationError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise CloseoutValidationError(f"{label} root is not an object")
    return value, encoded


def _require_unchanged(path: Path, expected: bytes, *, label: str) -> None:
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise CloseoutValidationError(f"{label} disappeared during replay: {exc}") from exc
    if observed != expected:
        raise CloseoutValidationError(f"{label} changed during protected replay")


def _run_protected_validator(
    path: Path,
    arguments: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Run an exact repository-owned judge in an isolated interpreter."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CloseoutValidationError(f"{label} validator is unavailable: {exc}") from exc
    if resolved != path.absolute() or not resolved.is_file():
        raise CloseoutValidationError(f"{label} validator path is not an exact regular file")
    try:
        completed = subprocess.run(
            (sys.executable, str(resolved), *arguments),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=PROTECTED_REPLAY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CloseoutValidationError(f"{label} protected replay failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2_000:]
        raise CloseoutValidationError(
            f"{label} protected replay returned {completed.returncode}: {detail}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CloseoutValidationError(
            f"{label} protected replay did not emit one JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise CloseoutValidationError(f"{label} protected replay root is not an object")
    return value


_QUALIFICATION_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "plan_cid",
        "predecessor_plan_cid",
        "cohort",
        "candidate_suites_are_self_authority",
        "independent_fixed_manifest_executed",
        "checkout_fingerprint_cid",
        "checkout_unchanged",
        "passed",
        "totals",
        "suites",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "limitations",
        "result_cid",
    }
)
_QUALIFICATION_SUITE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "suite_id",
        "manifest",
        "collected",
        "passed_count",
        "failed_count",
        "skipped_count",
        "xfailed_count",
        "xpassed_count",
        "error_count",
        "nodeids_cid",
        "exit_code",
        "passed",
        "isolation",
        "duration_ms",
        "transcript_sha256",
        "failure_tail",
        "observation_cid",
    }
)


def _qualification_authority_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return every reproducible authority field, rejecting open schemas.

    Per-run duration and transcript identities are retained losslessly by each
    result/observation CID, which the closeout report binds separately for the
    stored and fresh receipts.  They are not equality compared because a fresh
    pytest receipt is deliberately a different observation.
    """

    if set(value) != _QUALIFICATION_TOP_LEVEL_FIELDS:
        raise CloseoutValidationError("qualification fields differ from the closed schema")
    observations = value.get("suites")
    if not isinstance(observations, list):
        raise CloseoutValidationError("qualification suite population is absent")
    suites: list[dict[str, Any]] = []
    suite_fields = (
        "schema",
        "suite_id",
        "manifest",
        "isolation",
        "collected",
        "passed_count",
        "failed_count",
        "skipped_count",
        "xfailed_count",
        "xpassed_count",
        "error_count",
        "nodeids_cid",
        "exit_code",
        "passed",
    )
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise CloseoutValidationError(
                f"qualification suite observation {index} is not an object"
            )
        if set(observation) != _QUALIFICATION_SUITE_FIELDS:
            raise CloseoutValidationError(
                f"qualification suite observation {index} fields differ from the closed schema"
            )
        suites.append({field: observation.get(field) for field in suite_fields})
    stable_fields = (
        "schema",
        "plan_cid",
        "predecessor_plan_cid",
        "cohort",
        "candidate_suites_are_self_authority",
        "independent_fixed_manifest_executed",
        "checkout_fingerprint_cid",
        "checkout_unchanged",
        "passed",
        "totals",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "limitations",
    )
    return {field: value.get(field) for field in stable_fields} | {"suites": suites}


_BENCHMARK_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "cohort",
        "production_authoritative",
        "release_qualified",
        "production_authorized",
        "overall_disposition",
        "execution_evidence",
        "pairing",
        "task_class_coverage",
        "paired_result",
        "thresholds",
        "excluded_cohorts",
        "limitations",
        "reproducible_projection_cid",
        "report_cid",
    }
)
_BENCHMARK_EXECUTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "vertical_result_cid",
        "artifact_cid",
        "artifact_verification_receipt_cid",
        "fresh_execution_receipts_reproducible",
    }
)


def _benchmark_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the benchmark producer's projection for its declared CID only."""

    stable_fields = (
        "schema",
        "interface",
        "cohort",
        "production_authoritative",
        "release_qualified",
        "production_authorized",
        "overall_disposition",
        "pairing",
        "task_class_coverage",
        "paired_result",
        "thresholds",
        "excluded_cohorts",
        "limitations",
    )
    return {field: value.get(field) for field in stable_fields}


def _benchmark_authority_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the closed, reproducible benchmark authority surface.

    A vertical result CID is a fresh, run-specific receipt and therefore need
    not equal the earlier stored run.  The artifact and independent verifier
    receipt identities are deterministic authority bindings and must replay
    exactly; omitting the whole execution-evidence object would let forged
    artifact lineage pass.
    """

    if set(value) != _BENCHMARK_TOP_LEVEL_FIELDS:
        raise CloseoutValidationError("benchmark fields differ from the closed schema")
    execution = value.get("execution_evidence")
    if not isinstance(execution, Mapping) or set(execution) != _BENCHMARK_EXECUTION_FIELDS:
        raise CloseoutValidationError("benchmark execution evidence differs from the closed schema")
    authority_execution = {
        field: execution.get(field)
        for field in (
            "artifact_cid",
            "artifact_verification_receipt_cid",
            "fresh_execution_receipts_reproducible",
        )
    }
    return {
        key: item
        for key, item in value.items()
        if key not in {"execution_evidence", "report_cid"}
    } | {"execution_evidence": authority_execution}


def _verify_self_identity(value: Mapping[str, Any], field: str, *, label: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or not claimed:
        raise CloseoutValidationError(f"{label} has no {field}")
    body = {key: item for key, item in value.items() if key != field}
    if content_identity(body) != claimed:
        raise CloseoutValidationError(f"{label} {field} differs")
    return claimed


def _sha256_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CloseoutValidationError(f"required report is unreadable: {path}") from exc
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    """Return one bounded Git observation or fail closed."""

    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CloseoutValidationError(f"Git source observation failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1_000:]
        raise CloseoutValidationError(
            f"Git source observation returned {completed.returncode}: {detail}"
        )
    if len(completed.stdout) > 16 * 1024 * 1024:
        raise CloseoutValidationError("Git source observation exceeds its bound")
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(root, *arguments).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise CloseoutValidationError("Git source observation is not UTF-8") from exc


def _semantic_source_revision(root: Path = ROOT) -> tuple[str, str]:
    """Return the latest commit/tree that changed a non-evidence input.

    Report, successor, benchmark, and qualification commits are deliberately
    skipped. This keeps the source baseline reconstructable after those
    evidence files are committed and avoids requiring a report to predict the
    commit or tree that will contain itself.
    """

    revisions = _git_text(root, "rev-list", "--first-parent", "HEAD").splitlines()
    if not revisions:
        raise CloseoutValidationError("repository has no source revision")
    if len(revisions) > 100_000:
        raise CloseoutValidationError("repository source history exceeds its bound")
    for revision in revisions:
        lineage = _git_text(root, "rev-list", "--parents", "-n", "1", revision).split()
        if not lineage or lineage[0] != revision:
            raise CloseoutValidationError("repository source lineage is invalid")
        if len(lineage) > 1:
            changed_raw = _git_bytes(
                root,
                "diff",
                "--no-ext-diff",
                "--name-only",
                "-z",
                lineage[1],
                revision,
            )
        else:
            changed_raw = _git_bytes(
                root,
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-z",
                revision,
            )
        try:
            changed = {
                item.decode("utf-8")
                for item in changed_raw.split(b"\0")
                if item
            }
        except UnicodeDecodeError as exc:
            raise CloseoutValidationError("repository path is not UTF-8") from exc
        if changed - _EVIDENCE_ONLY_PATHS:
            tree = _git_text(root, "rev-parse", f"{revision}^{{tree}}")
            if (
                re.fullmatch(r"[0-9a-f]{40,64}", revision) is None
                or re.fullmatch(r"[0-9a-f]{40,64}", tree) is None
            ):
                raise CloseoutValidationError("repository source identity is invalid")
            return revision, tree
    raise CloseoutValidationError("repository history contains only evidence outputs")


def _tree_entry(root: Path, revision: str, path: str) -> tuple[str, str, str] | None:
    raw = _git_bytes(root, "ls-tree", "-z", revision, "--", path)
    entries = [item for item in raw.split(b"\0") if item]
    if not entries:
        return None
    if len(entries) != 1:
        raise CloseoutValidationError(f"source tree has ambiguous entry: {path}")
    try:
        metadata, observed_path = entries[0].decode("utf-8").split("\t", 1)
        mode, kind, object_id = metadata.split(" ", 2)
    except (UnicodeDecodeError, ValueError) as exc:
        raise CloseoutValidationError(f"source tree entry is invalid: {path}") from exc
    if observed_path != path or re.fullmatch(r"[0-9a-f]{40,64}", object_id) is None:
        raise CloseoutValidationError(f"source tree entry differs: {path}")
    return mode, kind, object_id


def _current_repository_truth(
    qualification: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Reconstruct exact source revisions and the datasets repository topology."""

    protected_input_cid = qualification.get("checkout_fingerprint_cid")
    if not isinstance(protected_input_cid, str) or not protected_input_cid.startswith("b"):
        raise CloseoutValidationError("qualification protected-input identity is absent")
    source_head, source_tree = _semantic_source_revision(root)
    datasets_path = root / "ipfs_datasets_py"
    if not datasets_path.is_dir():
        raise CloseoutValidationError("ipfs_datasets_py source directory is unavailable")
    entry = _tree_entry(root, source_head, "ipfs_datasets_py")
    nested_git = (datasets_path / ".git").exists()
    revisions: dict[str, dict[str, str]] = {
        "ipfs_accelerate_py": {
            "head": source_head,
            "tree": source_tree,
            "protected_input_cid": protected_input_cid,
        }
    }
    topology: dict[str, dict[str, str]] = {
        "ipfs_accelerate_py": {"kind": "repository_root", "path": "."}
    }
    if entry is not None and entry[0] == "160000" and entry[1] == "commit":
        if not nested_git:
            raise CloseoutValidationError("datasets gitlink checkout is unavailable")
        datasets_head = _git_text(datasets_path, "rev-parse", "HEAD")
        datasets_tree = _git_text(datasets_path, "rev-parse", "HEAD^{tree}")
        if datasets_head != entry[2]:
            raise CloseoutValidationError("datasets checkout and gitlink revisions differ")
        revisions["ipfs_datasets_py"] = {
            "head": datasets_head,
            "tree": datasets_tree,
            "gitlink": entry[2],
            "protected_input_cid": protected_input_cid,
        }
        topology["ipfs_datasets_py"] = {
            "kind": "git_submodule",
            "path": "ipfs_datasets_py",
        }
    elif nested_git:
        datasets_head = _git_text(datasets_path, "rev-parse", "HEAD")
        datasets_tree = _git_text(datasets_path, "rev-parse", "HEAD^{tree}")
        revisions["ipfs_datasets_py"] = {
            "head": datasets_head,
            "tree": datasets_tree,
            "protected_input_cid": protected_input_cid,
        }
        topology["ipfs_datasets_py"] = {
            "kind": "nested_independent_repository",
            "path": "ipfs_datasets_py",
        }
    else:
        if entry is None or entry[1] != "tree":
            raise CloseoutValidationError("integrated datasets source is absent from source tree")
        subtree_marker = _git_text(
            root,
            "log",
            "--all",
            "-n",
            "1",
            "--format=%H",
            "--grep=^git-subtree-dir: ipfs_datasets_py$",
        )
        revisions["ipfs_datasets_py"] = {
            "head": source_head,
            "tree": entry[2],
            "protected_input_cid": protected_input_cid,
        }
        topology["ipfs_datasets_py"] = {
            "kind": "subtree" if subtree_marker else "physically_integrated",
            "path": "ipfs_datasets_py",
        }
    return revisions, topology


def _field(text: str, label: str) -> str:
    matches = re.findall(rf"^- {re.escape(label)}:\s*(\S.*?)\s*$", text, re.MULTILINE)
    if not matches:
        raise CloseoutValidationError(f"report field is missing: {label}")
    if len(matches) != 1:
        raise CloseoutValidationError(f"report field is duplicated: {label}")
    return matches[0].strip()


_FIELD_LINE: Final[re.Pattern[str]] = re.compile(
    r"^- ([A-Za-z][A-Za-z0-9 /_-]*):\s*(\S.*?)\s*$", re.MULTILINE
)
_PLACEHOLDERS: Final[frozenset[str]] = frozenset(
    {"evidence", "evidence.", "n/a", "none", "placeholder", "tbd", "todo", "x"}
)


def _report_sections(text: str, headings: Sequence[str]) -> tuple[str, dict[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    positions: dict[str, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        if title in positions:
            raise CloseoutValidationError(f"report heading is duplicated: {title}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        positions[title] = (match.end(), end)
    missing = [heading for heading in headings if heading not in positions]
    if missing:
        raise CloseoutValidationError(f"report heading is missing: {missing[0]}")
    unexpected = sorted(set(positions) - set(headings))
    if unexpected:
        raise CloseoutValidationError(f"report heading is unexpected: {unexpected[0]}")
    ordered_offsets = [positions[heading][0] for heading in headings]
    if ordered_offsets != sorted(ordered_offsets):
        raise CloseoutValidationError("report headings are out of order")
    first_heading = min(match.start() for match in matches) if matches else len(text)
    sections = {
        heading: text[start:end].strip() for heading, (start, end) in positions.items()
    }
    for heading in headings:
        if not sections[heading]:
            raise CloseoutValidationError(f"report section is empty: {heading}")
    return text[:first_heading], sections


def _closed_fields(
    text: str,
    expected: Sequence[str],
    *,
    label: str,
    allow_document_title: bool = False,
) -> dict[str, str]:
    fields: dict[str, str] = {}
    matches = list(_FIELD_LINE.finditer(text))
    for match in matches:
        name, value = match.group(1).strip(), match.group(2).strip()
        if name in fields:
            raise CloseoutValidationError(f"{label} field is duplicated: {name}")
        fields[name] = value
    expected_set = set(expected)
    if set(fields) != expected_set:
        absent = sorted(expected_set - set(fields))
        extra = sorted(set(fields) - expected_set)
        detail = f"missing {absent[0]}" if absent else f"unexpected {extra[0]}"
        raise CloseoutValidationError(f"{label} typed fields differ: {detail}")
    residual = text
    for match in reversed(matches):
        residual = residual[: match.start()] + residual[match.end() :]
    remaining = [line.strip() for line in residual.splitlines() if line.strip()]
    if allow_document_title and remaining and re.fullmatch(r"# [^#].+", remaining[0]):
        remaining = remaining[1:]
    if remaining:
        raise CloseoutValidationError(
            f"{label} contains untyped content: {remaining[0][:120]}"
        )
    return fields


def _substantive_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise CloseoutValidationError(f"{label} is not a string")
    normalized = value.strip()
    if (
        len(normalized) < 8
        or normalized.casefold() in _PLACEHOLDERS
        or "\n" in normalized
        or "\r" in normalized
    ):
        raise CloseoutValidationError(f"{label} is not substantive")
    return normalized


def _json_field(fields: Mapping[str, str], name: str, *, label: str) -> Any:
    try:
        return json.loads(fields[name])
    except (KeyError, json.JSONDecodeError) as exc:
        raise CloseoutValidationError(f"{label} field is not valid JSON: {name}") from exc


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CloseoutValidationError(f"{label} is not a non-empty list")
    result = [_substantive_string(item, label=label) for item in value]
    if len(result) != len(set(result)):
        raise CloseoutValidationError(f"{label} contains duplicates")
    return result


def _string_mapping(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise CloseoutValidationError(f"{label} is not a non-empty object")
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _substantive_string(key, label=f"{label} key")
        result[normalized_key] = _substantive_string(item, label=f"{label} value")
    return result


def _reject_unsupported_authority_claims(text: str) -> None:
    patterns = (
        r"\bobjective\s+(?:is\s+)?complete\b",
        r"\brelease\s+(?:is\s+)?qualified\b",
        r"\bproduction\s+(?:is\s+)?authorized\b",
    )
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise CloseoutValidationError("report contains an unsupported positive authority claim")


def _evidence() -> _ReplayedEvidence:
    benchmark, benchmark_bytes = _read_snapshot(BENCHMARK_PATH, label="benchmark result")
    qualification, qualification_bytes = _read_snapshot(
        QUALIFICATION_PATH, label="qualification result"
    )

    replayed_qualification = _run_protected_validator(
        QUALIFICATION_VALIDATOR_PATH,
        ("--check",),
        label="qualification",
    )
    _require_unchanged(
        QUALIFICATION_PATH,
        qualification_bytes,
        label="qualification result",
    )
    qualification_authority = _qualification_authority_evidence(qualification)
    replayed_qualification_authority = _qualification_authority_evidence(
        replayed_qualification
    )
    if qualification_authority != replayed_qualification_authority:
        raise CloseoutValidationError(
            "qualification authority reconstruction differs from stored evidence"
        )

    replayed_benchmark = _run_protected_validator(
        BENCHMARK_VALIDATOR_PATH,
        ("--check", "--output", str(BENCHMARK_PATH), "--json"),
        label="benchmark",
    )
    _require_unchanged(BENCHMARK_PATH, benchmark_bytes, label="benchmark result")
    _require_unchanged(
        QUALIFICATION_PATH,
        qualification_bytes,
        label="qualification result",
    )
    benchmark_authority = _benchmark_authority_evidence(benchmark)
    replayed_benchmark_authority = _benchmark_authority_evidence(replayed_benchmark)
    if benchmark_authority != replayed_benchmark_authority:
        raise CloseoutValidationError(
            "benchmark authority reconstruction differs from stored evidence"
        )

    benchmark_cid = _verify_self_identity(benchmark, "report_cid", label="benchmark result")
    benchmark_replay_cid = _verify_self_identity(
        replayed_benchmark, "report_cid", label="replayed benchmark result"
    )
    if benchmark.get("schema") != "lgcvf-symbolic-displacement-benchmark@1":
        raise CloseoutValidationError("benchmark schema differs")
    if benchmark.get("cohort") != "hermetic_local_execution":
        raise CloseoutValidationError("benchmark cohort is not hermetic local execution")
    if benchmark.get("production_authoritative") is not False:
        raise CloseoutValidationError("benchmark raises production authority")
    if any(
        benchmark.get(field) is not False
        for field in ("release_qualified", "production_authorized")
    ):
        raise CloseoutValidationError("benchmark raises release or production authority")

    qualification_cid = _verify_self_identity(
        qualification, "result_cid", label="qualification result"
    )
    qualification_replay_cid = _verify_self_identity(
        replayed_qualification,
        "result_cid",
        label="replayed qualification result",
    )
    if qualification.get("schema") != "lgcvf-independent-hermetic-qualification@1":
        raise CloseoutValidationError("qualification schema differs")
    if qualification.get("plan_cid") != PLAN_CID or qualification.get("passed") is not True:
        raise CloseoutValidationError("qualification result is stale or unsuccessful")
    expected_qualification_states = {
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "production_authoritative": False,
    }
    for field, wanted in expected_qualification_states.items():
        if qualification.get(field) is not wanted:
            raise CloseoutValidationError(
                f"qualification completion or authority state differs: {field}"
            )
    return _ReplayedEvidence(
        benchmark=benchmark,
        benchmark_cid=benchmark_cid,
        benchmark_authority_cid=content_identity(benchmark_authority),
        benchmark_replay_cid=benchmark_replay_cid,
        qualification=qualification,
        qualification_cid=qualification_cid,
        qualification_authority_cid=content_identity(qualification_authority),
        qualification_replay_cid=qualification_replay_cid,
        benchmark_validator_sha256=_sha256_file(BENCHMARK_VALIDATOR_PATH),
        qualification_validator_sha256=_sha256_file(QUALIFICATION_VALIDATOR_PATH),
    )


def validate_release(path: Path | None = None) -> dict[str, Any]:
    path = RELEASE_PATH if path is None else path
    return _validate_release(path, _evidence())


def _validate_release(path: Path, evidence: _ReplayedEvidence) -> dict[str, Any]:
    benchmark = evidence.benchmark
    benchmark_cid = evidence.benchmark_cid
    qualification_cid = evidence.qualification_cid
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CloseoutValidationError(f"release report is unreadable: {exc}") from exc
    _reject_unsupported_authority_claims(text)
    preamble, sections = _report_sections(
        text, ("Evidence", "Disposition", "Blockers", "Limitations")
    )
    _closed_fields(
        preamble,
        (),
        label="release report preamble",
        allow_document_title=True,
    )
    evidence_fields = _closed_fields(
        sections["Evidence"],
        (
            "Formal plan CID",
            "Qualification result CID",
            "Qualification authority CID",
            "Qualification suite node IDs",
            "Benchmark result CID",
            "Benchmark authority CID",
            "Evidence cohort",
        ),
        label="release Evidence section",
    )
    expected_evidence = {
        "Formal plan CID": PLAN_CID,
        "Qualification result CID": qualification_cid,
        "Qualification authority CID": evidence.qualification_authority_cid,
        "Benchmark result CID": benchmark_cid,
        "Benchmark authority CID": evidence.benchmark_authority_cid,
        "Evidence cohort": "hermetic_local_execution",
    }
    for label, value in expected_evidence.items():
        if evidence_fields[label] != value:
            raise CloseoutValidationError(f"release report field differs: {label}")
    expected_nodeids = [
        item.get("nodeids_cid")
        for item in evidence.qualification.get("suites", [])
        if isinstance(item, Mapping)
    ]
    if (
        not expected_nodeids
        or _json_field(
            evidence_fields,
            "Qualification suite node IDs",
            label="release Evidence section",
        )
        != expected_nodeids
    ):
        raise CloseoutValidationError("release qualification suite node identities differ")

    disposition_fields = _closed_fields(
        sections["Disposition"],
        (
            "Disposition",
            "Task implementation",
            "Test success",
            "Objective completion",
            "Release qualification",
            "Production authorization",
            "Threshold comparison",
        ),
        label="release Disposition section",
    )
    expected_states = {
        "Task implementation": "incomplete",
        "Test success": "passed_hermetic",
        "Objective completion": "incomplete",
        "Release qualification": "not_qualified",
        "Production authorization": "not_authorized",
    }
    for label, value in expected_states.items():
        if disposition_fields[label] != value:
            raise CloseoutValidationError(f"release report field differs: {label}")
    if _json_field(
        disposition_fields,
        "Threshold comparison",
        label="release Disposition section",
    ) != benchmark.get("thresholds"):
        raise CloseoutValidationError("release threshold comparison differs from benchmark")

    blocker_fields = _closed_fields(
        sections["Blockers"],
        ("External authority gate", "Manual authority gate"),
        label="release Blockers section",
    )
    expected_blockers = {
        "External authority gate": "blocked_external_authority",
        "Manual authority gate": "blocked_manual",
    }
    for label, value in expected_blockers.items():
        if blocker_fields[label] != value:
            raise CloseoutValidationError(f"release report field differs: {label}")
    limitation_fields = _closed_fields(
        sections["Limitations"],
        ("Limitations",),
        label="release Limitations section",
    )
    _string_list(
        _json_field(limitation_fields, "Limitations", label="release Limitations section"),
        label="release limitations",
    )

    disposition = disposition_fields["Disposition"]
    benchmark_disposition = str(benchmark.get("overall_disposition") or "")
    permitted = {"partial", "no_go"}
    expected_disposition = (
        benchmark_disposition
        if benchmark_disposition in permitted
        else "partial"
        if benchmark_disposition == "development_targets_met"
        else ""
    )
    if disposition not in permitted or disposition != expected_disposition:
        raise CloseoutValidationError("release disposition differs from benchmark evidence")
    result = {
        "schema": "lgcvf-release-report-validation@1",
        "valid": True,
        "report_sha256": _sha256_file(path),
        "plan_cid": PLAN_CID,
        "benchmark_cid": benchmark_cid,
        "benchmark_authority_cid": evidence.benchmark_authority_cid,
        "benchmark_replay_cid": evidence.benchmark_replay_cid,
        "qualification_cid": qualification_cid,
        "qualification_authority_cid": evidence.qualification_authority_cid,
        "qualification_replay_cid": evidence.qualification_replay_cid,
        "benchmark_validator_sha256": evidence.benchmark_validator_sha256,
        "qualification_validator_sha256": evidence.qualification_validator_sha256,
        "disposition": disposition,
        "release_qualified": False,
        "production_authorized": False,
    }
    result["validation_cid"] = content_identity(result)
    return result


_IMPLEMENTATION_HEADINGS: Final[tuple[str, ...]] = (
    "A. Exact source revisions and repository topology",
    "B. Pre-existing implemented capabilities",
    "C. Verified gaps",
    "D. Architecture decisions and authority boundaries",
    "E. Files changed by repository",
    "F. Public interfaces added or extended",
    "G. Tests and exact results",
    "H. Vertical-slice trace and receipt identities",
    "I. Benchmark metrics",
    "J. Model and context displacement",
    "K. Remaining risks and production blockers",
    "L. Next minimal machine-executable tasks",
)

_IMPLEMENTATION_SECTION_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    _IMPLEMENTATION_HEADINGS[0]: ("Source revisions", "Repository topology"),
    _IMPLEMENTATION_HEADINGS[1]: ("Reused capabilities",),
    _IMPLEMENTATION_HEADINGS[2]: ("Verified gaps",),
    _IMPLEMENTATION_HEADINGS[3]: ("Completion states",),
    _IMPLEMENTATION_HEADINGS[4]: ("Files changed by repository",),
    _IMPLEMENTATION_HEADINGS[5]: ("Public interfaces",),
    _IMPLEMENTATION_HEADINGS[6]: ("Test commands", "Exact test results"),
    _IMPLEMENTATION_HEADINGS[7]: ("Vertical receipt identities",),
    _IMPLEMENTATION_HEADINGS[8]: ("Benchmark disposition", "Thresholds"),
    _IMPLEMENTATION_HEADINGS[9]: ("Displacement evidence",),
    _IMPLEMENTATION_HEADINGS[10]: ("Remaining risks", "Production blockers"),
    _IMPLEMENTATION_HEADINGS[11]: ("Successor task IDs", "Successor tasks CID"),
}

_COMPLETION_STATES: Final[dict[str, bool]] = {
    "task_implementation_complete": False,
    "test_qualification_complete": True,
    "objective_complete": False,
    "release_qualified": False,
    "production_authorized": False,
}
_REQUIRED_REPORT_REPOSITORIES: Final[frozenset[str]] = frozenset(
    {"ipfs_accelerate_py", "ipfs_datasets_py"}
)


def _files_by_repository(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict) or not value:
        raise CloseoutValidationError("files changed by repository is not a non-empty object")
    result: dict[str, list[str]] = {}
    for repository, raw_paths in value.items():
        name = _substantive_string(repository, label="changed-file repository")
        paths = _string_list(raw_paths, label=f"{name} changed files")
        for raw_path in paths:
            candidate = PurePosixPath(raw_path)
            if candidate.is_absolute() or ".." in candidate.parts or raw_path.endswith("/"):
                raise CloseoutValidationError(f"changed-file path is unsafe: {raw_path}")
        result[name] = paths
    return result


def _validate_successors(
    value: Mapping[str, Any],
    *,
    benchmark_cid: str,
    qualification_cid: str,
    release_sha256: str,
) -> str:
    expected_top_level = {
        "schema",
        "plan_cid",
        "benchmark_cid",
        "qualification_cid",
        "release_report_sha256",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "tasks",
        "successor_tasks_cid",
    }
    if set(value) != expected_top_level:
        raise CloseoutValidationError("successor task artifact fields differ from the closed schema")
    if value.get("schema") != "lgcvf-successor-tasks@1":
        raise CloseoutValidationError("successor task schema differs")
    expected = {
        "plan_cid": PLAN_CID,
        "benchmark_cid": benchmark_cid,
        "qualification_cid": qualification_cid,
        "release_report_sha256": release_sha256,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise CloseoutValidationError(f"successor authority binding differs: {field}")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise CloseoutValidationError("successor task list is empty")
    if len(tasks) > 100:
        raise CloseoutValidationError("successor task list exceeds its bound")
    expected_task_fields = {
        "task_id",
        "task_cid",
        "title",
        "status",
        "owning_repository",
        "depends_on",
        "outputs",
        "validation",
        "acceptance",
        "reason_codes",
    }
    seen: set[str] = set()
    dependencies: dict[str, tuple[str, ...]] = {}
    statuses: set[str] = set()
    output_owners: dict[str, str] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise CloseoutValidationError(f"successor task {index} is not an object")
        if set(task) != expected_task_fields:
            raise CloseoutValidationError(
                f"successor task {index} fields differ from the closed schema"
            )
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not re.fullmatch(r"LGCVF-S\d{3}", task_id):
            raise CloseoutValidationError(f"successor task {index} identity is invalid")
        if task_id in seen:
            raise CloseoutValidationError(f"duplicate successor task: {task_id}")
        seen.add(task_id)
        status = task.get("status")
        if status not in {
            "todo",
            "blocked_manual",
            "blocked_external_authority",
        }:
            raise CloseoutValidationError(f"{task_id}: invalid successor status")
        statuses.add(str(status))
        _substantive_string(task.get("title"), label=f"{task_id}: title")
        repository = _substantive_string(
            task.get("owning_repository"), label=f"{task_id}: owning_repository"
        )
        if repository not in _REQUIRED_REPORT_REPOSITORIES:
            raise CloseoutValidationError(f"{task_id}: owning_repository is invalid")
        for field in ("depends_on", "outputs", "validation", "acceptance", "reason_codes"):
            items = task.get(field)
            if not isinstance(items, list) or (field != "depends_on" and not items):
                raise CloseoutValidationError(f"{task_id}: {field} is invalid")
            if any(not isinstance(item, str) or not item.strip() for item in items):
                raise CloseoutValidationError(f"{task_id}: {field} contains invalid values")
            if len(items) != len(set(items)):
                raise CloseoutValidationError(f"{task_id}: {field} contains duplicates")
        raw_dependencies = tuple(str(item) for item in task["depends_on"])
        if any(re.fullmatch(r"LGCVF-S\d{3}", item) is None for item in raw_dependencies):
            raise CloseoutValidationError(f"{task_id}: depends_on identity is invalid")
        dependencies[task_id] = raw_dependencies
        outputs = [
            _substantive_string(item, label=f"{task_id}: output") for item in task["outputs"]
        ]
        for output in outputs:
            candidate = PurePosixPath(output)
            if candidate.is_absolute() or ".." in candidate.parts or output.endswith("/"):
                raise CloseoutValidationError(f"{task_id}: output path is unsafe")
            owner = output_owners.get(output)
            if owner is not None:
                raise CloseoutValidationError(
                    f"successor output {output} is owned by both {owner} and {task_id}"
                )
            output_owners[output] = task_id
        for item in task["validation"]:
            command = _substantive_string(item, label=f"{task_id}: validation")
            if re.match(r"^(?:python(?:3(?:\.\d+)*)?|pytest|bash|sh|git)\s+\S", command) is None:
                raise CloseoutValidationError(
                    f"{task_id}: validation is not a machine-executable command"
                )
        for item in task["acceptance"]:
            _substantive_string(item, label=f"{task_id}: acceptance")
        reason_codes = [str(item) for item in task["reason_codes"]]
        if any(re.fullmatch(r"[a-z][a-z0-9_:-]{3,}", item) is None for item in reason_codes):
            raise CloseoutValidationError(f"{task_id}: reason_codes contains invalid values")
        if status in {"blocked_manual", "blocked_external_authority"} and status not in reason_codes:
            raise CloseoutValidationError(f"{task_id}: blocker status lacks its reason code")
        claimed_task_cid = task.get("task_cid")
        task_body = {key: item for key, item in task.items() if key != "task_cid"}
        if not isinstance(claimed_task_cid, str) or content_identity(task_body) != claimed_task_cid:
            raise CloseoutValidationError(f"{task_id}: task content identity differs")
    ordered_ids = [str(task["task_id"]) for task in tasks]
    expected_ids = [f"LGCVF-S{index:03d}" for index in range(1, len(tasks) + 1)]
    if ordered_ids != expected_ids:
        raise CloseoutValidationError("successor task identities are not contiguous and ordered")
    for task_id, task_dependencies in dependencies.items():
        unknown = set(task_dependencies) - seen
        if unknown:
            raise CloseoutValidationError(
                f"{task_id}: dependency is outside the successor task closure: {sorted(unknown)[0]}"
            )
        if task_id in task_dependencies:
            raise CloseoutValidationError(f"{task_id}: successor dependency is self-referential")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise CloseoutValidationError("successor task dependencies contain a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in dependencies[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ordered_ids:
        visit(task_id)
    for required_status in ("blocked_manual", "blocked_external_authority"):
        if required_status not in statuses:
            raise CloseoutValidationError(
                f"successor tasks do not preserve the {required_status} blocker"
            )
    claimed = value.get("successor_tasks_cid")
    body = {key: item for key, item in value.items() if key != "successor_tasks_cid"}
    if not isinstance(claimed, str) or content_identity(body) != claimed:
        raise CloseoutValidationError("successor task content identity differs")
    return claimed


def validate_implementation(
    report_path: Path = IMPLEMENTATION_PATH,
    successors_path: Path = SUCCESSORS_PATH,
) -> dict[str, Any]:
    evidence = _evidence()
    release = _validate_release(RELEASE_PATH, evidence)
    benchmark_cid = evidence.benchmark_cid
    qualification_cid = evidence.qualification_cid
    try:
        text = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CloseoutValidationError(f"implementation report is unreadable: {exc}") from exc
    _reject_unsupported_authority_claims(text)
    preamble, sections = _report_sections(text, _IMPLEMENTATION_HEADINGS)
    preamble_fields = _closed_fields(
        preamble,
        (
            "Formal plan CID",
            "Qualification result CID",
            "Qualification authority CID",
            "Benchmark result CID",
            "Benchmark authority CID",
            "Release report SHA256",
            "Task implementation",
            "Test success",
            "Objective completion",
            "Release qualification",
            "Production authorization",
        ),
        label="implementation report preamble",
        allow_document_title=True,
    )
    expected_preamble = {
        "Formal plan CID": PLAN_CID,
        "Qualification result CID": qualification_cid,
        "Qualification authority CID": evidence.qualification_authority_cid,
        "Benchmark result CID": benchmark_cid,
        "Benchmark authority CID": evidence.benchmark_authority_cid,
        "Release report SHA256": release["report_sha256"],
        "Task implementation": "incomplete",
        "Test success": "passed_hermetic",
        "Objective completion": "incomplete",
        "Release qualification": "not_qualified",
        "Production authorization": "not_authorized",
    }
    for label, value in expected_preamble.items():
        if preamble_fields[label] != value:
            raise CloseoutValidationError(f"implementation report field differs: {label}")
    successors = _load_object(successors_path, label="successor tasks")
    successor_cid = _validate_successors(
        successors,
        benchmark_cid=benchmark_cid,
        qualification_cid=qualification_cid,
        release_sha256=release["report_sha256"],
    )
    typed_sections = {
        heading: _closed_fields(
            sections[heading],
            fields,
            label=f"implementation section {heading}",
        )
        for heading, fields in _IMPLEMENTATION_SECTION_FIELDS.items()
    }

    section_a = typed_sections[_IMPLEMENTATION_HEADINGS[0]]
    source_revisions = _json_field(
        section_a,
        "Source revisions",
        label="implementation section A",
    )
    repository_topology = _json_field(
        section_a,
        "Repository topology",
        label="implementation section A",
    )
    expected_revisions, expected_topology = _current_repository_truth(
        evidence.qualification
    )
    if source_revisions != expected_revisions:
        raise CloseoutValidationError("implementation source revisions differ from Git truth")
    if repository_topology != expected_topology:
        raise CloseoutValidationError("implementation repository topology differs from Git truth")
    _string_list(
        _json_field(
            typed_sections[_IMPLEMENTATION_HEADINGS[1]],
            "Reused capabilities",
            label="implementation section B",
        ),
        label="reused capabilities",
    )
    _string_list(
        _json_field(
            typed_sections[_IMPLEMENTATION_HEADINGS[2]],
            "Verified gaps",
            label="implementation section C",
        ),
        label="verified gaps",
    )
    completion_states = _json_field(
        typed_sections[_IMPLEMENTATION_HEADINGS[3]],
        "Completion states",
        label="implementation section D",
    )
    if completion_states != _COMPLETION_STATES:
        raise CloseoutValidationError("implementation completion states differ")
    changed_files = _files_by_repository(
        _json_field(
            typed_sections[_IMPLEMENTATION_HEADINGS[4]],
            "Files changed by repository",
            label="implementation section E",
        )
    )
    if set(changed_files) != _REQUIRED_REPORT_REPOSITORIES:
        raise CloseoutValidationError("implementation changed-file repositories are incomplete")
    _string_list(
        _json_field(
            typed_sections[_IMPLEMENTATION_HEADINGS[5]],
            "Public interfaces",
            label="implementation section F",
        ),
        label="public interfaces",
    )
    section_g = typed_sections[_IMPLEMENTATION_HEADINGS[6]]
    _string_list(
        _json_field(section_g, "Test commands", label="implementation section G"),
        label="test commands",
    )
    if _json_field(section_g, "Exact test results", label="implementation section G") != (
        evidence.qualification.get("totals")
    ):
        raise CloseoutValidationError("implementation exact test results differ")
    execution_evidence = evidence.benchmark.get("execution_evidence")
    if not isinstance(execution_evidence, Mapping):
        raise CloseoutValidationError("benchmark execution evidence is absent")
    if _json_field(
        typed_sections[_IMPLEMENTATION_HEADINGS[7]],
        "Vertical receipt identities",
        label="implementation section H",
    ) != dict(execution_evidence):
        raise CloseoutValidationError("implementation vertical receipt identities differ")
    section_i = typed_sections[_IMPLEMENTATION_HEADINGS[8]]
    if section_i["Benchmark disposition"] != str(
        evidence.benchmark.get("overall_disposition") or ""
    ):
        raise CloseoutValidationError("implementation benchmark disposition differs")
    if _json_field(section_i, "Thresholds", label="implementation section I") != (
        evidence.benchmark.get("thresholds")
    ):
        raise CloseoutValidationError("implementation benchmark thresholds differ")
    pairing = evidence.benchmark.get("pairing")
    paired_result = evidence.benchmark.get("paired_result")
    if not isinstance(pairing, Mapping) or not isinstance(paired_result, Mapping):
        raise CloseoutValidationError("benchmark displacement evidence is absent")
    model_invocations = pairing.get("model_invocation_count")
    comparison = paired_result.get("comparison")
    if (
        isinstance(model_invocations, bool)
        or not isinstance(model_invocations, int)
        or model_invocations < 0
        or not isinstance(comparison, Mapping)
        or not comparison
    ):
        raise CloseoutValidationError("benchmark displacement evidence is invalid")
    expected_displacement = {
        "model_invocation_count": model_invocations,
        "context_comparison": dict(comparison),
    }
    if _json_field(
        typed_sections[_IMPLEMENTATION_HEADINGS[9]],
        "Displacement evidence",
        label="implementation section J",
    ) != expected_displacement:
        raise CloseoutValidationError("implementation displacement evidence differs")
    section_k = typed_sections[_IMPLEMENTATION_HEADINGS[10]]
    _string_list(
        _json_field(section_k, "Remaining risks", label="implementation section K"),
        label="remaining risks",
    )
    if _json_field(
        section_k, "Production blockers", label="implementation section K"
    ) != ["blocked_external_authority", "blocked_manual"]:
        raise CloseoutValidationError("implementation production blockers differ")
    section_l = typed_sections[_IMPLEMENTATION_HEADINGS[11]]
    successor_ids = [str(task["task_id"]) for task in successors["tasks"]]
    if _json_field(
        section_l, "Successor task IDs", label="implementation section L"
    ) != successor_ids:
        raise CloseoutValidationError("implementation successor task identities differ")
    if section_l["Successor tasks CID"] != successor_cid:
        raise CloseoutValidationError("implementation report successor CID differs")
    result = {
        "schema": "lgcvf-implementation-report-validation@1",
        "valid": True,
        "report_sha256": _sha256_file(report_path),
        "release_validation_cid": release["validation_cid"],
        "successor_tasks_cid": successor_cid,
        "benchmark_cid": benchmark_cid,
        "benchmark_authority_cid": evidence.benchmark_authority_cid,
        "benchmark_replay_cid": evidence.benchmark_replay_cid,
        "qualification_cid": qualification_cid,
        "qualification_authority_cid": evidence.qualification_authority_cid,
        "qualification_replay_cid": evidence.qualification_replay_cid,
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    result["validation_cid"] = content_identity(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("release", "implementation"))
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_release() if args.kind == "release" else validate_implementation()
    except (CloseoutValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": type(exc).__name__, "reason": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
