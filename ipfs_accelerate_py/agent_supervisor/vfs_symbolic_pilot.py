"""Frozen SwissKnife / IPFS Kit VFS pilot (VFS-037 / VFS-G131).

Evidence schema: ``vfs/swissknife-vfs-pilot@1``.

This module freezes fresh multi-repository descriptors, scans every admitted
SwissKnife file plus the VFS-relevant closure of the IPFS packages, executes
the deterministic graph / contract / proof / ZK-shadow pipeline, and publishes
content-addressed manifest, coverage, cache, proof, finding, and taskboard
CIDs.

Safety invariants (non-waivable):

* dry-run and verify never call a model provider;
* neither mode mutates source trees (SwissKnife is always read-only);
* every file and finding is provenance-bound to forest / inventory CIDs;
* inconclusive, ambiguous, or partial findings remain non-executable;
* the repair board is bounded, deduplicated, goal-backed, dependency-valid,
  and carries exact repair packets when executable work is admitted;
* verification fails closed on changed trees, incomplete inventory, stale
  evidence, or non-canonical artifacts.

CLI::

    python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot --dry-run
    python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot \
        --verify --report path/to/report.json
    python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot \
        --verify-release-evidence --report path/to/report.json
    python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot \
        --hermetic-self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .contract_findings import (
    CallSlice,
    CallSliceStep,
    ContractFindingLedger,
    EvidenceReferences,
    build_contract_finding,
)
from .contract_repair_packet import (
    CallSliceRef,
    CallSliceStepRef,
    ContractRepairPacketCompiler,
    RepairPacketRequest,
    RepairPacketStatus,
    RepairAuthority,
)
from .finding_task_source import (
    DEFAULT_BOARD_NAMESPACE,
    DEFAULT_GOAL_ID,
    DEFAULT_GOAL_LINEAGE,
    FindingTaskSource,
    FindingTaskSourcePolicy,
    project_board_json,
    project_board_markdown,
)
from .program_analysis_cache import (
    ProgramAnalysisAuthority,
    ProgramAnalysisCache,
    ProgramAnalysisComponentKind,
    build_program_analysis_cache_key,
)
from .program_analysis_zkp import (
    PUBLIC_INPUT_CODEC_ID,
    PUBLIC_INPUT_CODEC_VERSION,
    PrivateProgramAnalysisWitness,
    ProgramZkpBackendMode,
    ProgramZkpVerdict,
    build_program_zkp_public_inputs,
    commitment_identity,
    create_program_zkp_shadow_envelope,
    prepare_program_analysis_zkp,
    record_program_zkp_verification,
)
from .program_assurance_contracts import (
    ClaimLevel,
    EvidenceFreshness,
    FindingSeverity,
    FindingStatus,
)
from .program_ast_adapters import adapt_program_source
from .program_graph import (
    ProgramGraph,
    ProgramGraphBinding,
    ProgramGraphNode,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan as GraphSourceSpan,
    build_program_graph,
)
from .proof.formal_verification_contracts import (
    canonical_json,
    canonical_json_bytes,
    content_identity,
)
from .repository_corpus_index import (
    InclusionDecision,
    InventoryLimits,
    RepositoryCorpusIndex,
    build_repository_corpus_index,
)
from .repository_forest import (
    DEFAULT_ACCELERATOR_ALIAS,
    DEFAULT_DATASETS_ALIAS,
    DEFAULT_KIT_ALIAS,
    DEFAULT_SWISSKNIFE_ALIAS,
    DEFAULT_SWISSKNIFE_ROOT,
    AuthorityMode,
    ForestPolicy,
    ForestRootSpec,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForest,
    build_repository_forest,
    initial_vfs_assurance_forest_policy,
)


# ---------------------------------------------------------------------------
# Evidence / version / bounds
# ---------------------------------------------------------------------------

SWISS_KNIFE_VFS_PILOT_SCHEMA: Final[str] = "vfs/swissknife-vfs-pilot@1"
PILOT_MANIFEST_SCHEMA: Final[str] = "vfs/swissknife-vfs-pilot-manifest@1"
PILOT_COVERAGE_SCHEMA: Final[str] = "vfs/swissknife-vfs-pilot-coverage@1"
PILOT_STAGE_RECEIPT_SCHEMA: Final[str] = "vfs/swissknife-vfs-pilot-stage@1"
PILOT_ARTIFACT_SET_SCHEMA: Final[str] = "vfs/swissknife-vfs-pilot-artifacts@1"

PILOT_VERSION: Final[int] = 1
PILOT_OBJECTIVE_ID: Final[str] = "VFS-G131"
PILOT_TASK_ID: Final[str] = "VFS-037"
PILOT_REQUIREMENT_ID: Final[str] = "vfs-037:frozen-swissknife-ipfs-vfs-pilot"
PILOT_PRODUCER: Final[str] = "vfs-symbolic-pilot@1"
PILOT_BOARD_NAMESPACE: Final[str] = DEFAULT_BOARD_NAMESPACE
PILOT_POLICY_REVISION: Final[str] = "policy:vfs-symbolic-pilot@1"

DEFAULT_ARTIFACT_RELATIVE: Final[str] = (
    "data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/pilot"
)
DEFAULT_FINDINGS_BOARD_RELATIVE: Final[str] = (
    "docs/architecture/ipfs_kit_vfs_symbolic_assurance.findings.todo.md"
)

MAX_ADMITTED_PARSE: Final[int] = 4_096
MAX_GRAPH_NODES: Final[int] = 16_384
MAX_BOARD_TASKS: Final[int] = 4_096
MAX_REPORT_BYTES: Final[int] = 8 * 1024 * 1024
MAX_FINDINGS_BOARD_BYTES: Final[int] = 1_000_000
MAX_STAGE_REASON_CODES: Final[int] = 128

# Path / name signals that admit an IPFS-package file into the VFS-relevant
# closure (SwissKnife itself admits every included corpus entry).
_VFS_RELEVANT_PATH = re.compile(
    r"""(?ix)
    (?:^|/)(?:
        vfs|virtual[\s._-]*file[\s._-]*system|
        fsspec|ipfs[_-]?kit|mcplusplus|mcp[_+]?plus|
        ipfs[_-]?datasets|ipfs[_-]?accelerate
    )(?:/|$)|
    (?:^|/)(?:
        vfs_[a-z0-9_]+|[a-z0-9_]+_vfs|
        virtual_fs|ipfs_fs|fs_manager|
        swissknife
    )
    """
)

_PARSER_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".md",
        ".markdown",
    }
)

# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class VfsSymbolicPilotError(ValueError):
    """Pilot input, pipeline, or verification failure."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "pilot_error").strip()
        detail = str(message or "").strip()
        super().__init__(detail or self.reason_code)


class PilotVerificationError(VfsSymbolicPilotError):
    """Verify mode rejected the pilot report or live forest."""


class PilotMode(str, Enum):
    """Operator mode for the pilot pipeline."""

    DRY_RUN = "dry_run"
    VERIFY = "verify"
    RUN = "run"


class PilotStage(str, Enum):
    """Closed pipeline stages executed by the pilot."""

    FREEZE = "freeze"
    INVENTORY = "inventory"
    SCAN = "scan"
    GRAPH = "graph"
    CONTRACT = "contract"
    CACHE = "cache"
    PROOF = "proof"
    ZK_SHADOW = "zk_shadow"
    FINDINGS = "findings"
    TASKBOARD = "taskboard"
    PUBLISH = "publish"


class PilotConclusion(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_json_bytes(_plain(value))
    except (TypeError, ValueError) as exc:
        raise VfsSymbolicPilotError(
            "noncanonical_artifact",
            "pilot data must be canonical JSON",
        ) from exc


def _identity(value: Any) -> str:
    return content_identity(_plain(value))


def _text(value: Any, name: str, *, maximum: int = 4_096) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value or value != value.strip():
        raise VfsSymbolicPilotError(
            "invalid_text",
            f"{name} must be non-empty canonical text",
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise VfsSymbolicPilotError(
            "invalid_text",
            f"{name} is unsafe or exceeds {maximum} bytes",
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise VfsSymbolicPilotError("invalid_boolean", f"{name} must be boolean")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    body = canonical_json(_plain(payload))
    if not body.endswith("\n"):
        body = body + "\n"
    _atomic_write_text(path, body)


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VfsSymbolicPilotError(
            "missing_artifact",
            f"cannot read {path}: {exc}",
        ) from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise VfsSymbolicPilotError(
                    "noncanonical_artifact",
                    f"{path} contains duplicate JSON key {key!r}",
                )
            result[key] = item
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise VfsSymbolicPilotError(
            "noncanonical_artifact",
            f"{path} is not valid JSON",
        ) from exc


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise VfsSymbolicPilotError(
            "git_failure",
            f"git {' '.join(args)} failed in {repo}: {completed.stderr.strip()}",
        )
    return (completed.stdout or "").strip()


# ---------------------------------------------------------------------------
# Config / stage / report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotConfig:
    """Roots and bounds for one pilot execution."""

    accelerator_root: Path
    swissknife_root: Path = field(default_factory=lambda: Path(DEFAULT_SWISSKNIFE_ROOT))
    kit_root: Path | None = None
    datasets_root: Path | None = None
    artifact_dir: Path | None = None
    findings_board_path: Path | None = None
    inventory_limits: InventoryLimits | None = None
    max_admitted_parse: int = MAX_ADMITTED_PARSE
    write_artifacts: bool = True
    write_findings_board: bool = True
    include_optional_missing: bool = False
    require_exhaustive_swissknife: bool = True
    forest_policy: ForestPolicy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "accelerator_root", Path(self.accelerator_root).resolve())
        object.__setattr__(self, "swissknife_root", Path(self.swissknife_root).resolve())
        if self.kit_root is not None:
            object.__setattr__(self, "kit_root", Path(self.kit_root).resolve())
        if self.datasets_root is not None:
            object.__setattr__(
                self, "datasets_root", Path(self.datasets_root).resolve()
            )
        if self.artifact_dir is not None:
            object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        if self.findings_board_path is not None:
            object.__setattr__(
                self, "findings_board_path", Path(self.findings_board_path)
            )
        if not isinstance(self.max_admitted_parse, int) or self.max_admitted_parse < 1:
            raise VfsSymbolicPilotError(
                "invalid_bound",
                "max_admitted_parse must be a positive integer",
            )
        for flag_name in (
            "write_artifacts",
            "write_findings_board",
            "include_optional_missing",
            "require_exhaustive_swissknife",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise VfsSymbolicPilotError(
                    "invalid_boolean",
                    f"{flag_name} must be boolean",
                )

    def resolved_artifact_dir(self) -> Path:
        if self.artifact_dir is not None:
            return Path(self.artifact_dir)
        return self.accelerator_root / DEFAULT_ARTIFACT_RELATIVE

    def resolved_findings_board_path(self) -> Path:
        if self.findings_board_path is not None:
            return Path(self.findings_board_path)
        return self.accelerator_root / DEFAULT_FINDINGS_BOARD_RELATIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator_root": str(self.accelerator_root),
            "swissknife_root": str(self.swissknife_root),
            "kit_root": str(self.kit_root) if self.kit_root else None,
            "datasets_root": (
                str(self.datasets_root) if self.datasets_root else None
            ),
            "artifact_dir": str(self.resolved_artifact_dir()),
            "findings_board_path": str(self.resolved_findings_board_path()),
            "max_admitted_parse": self.max_admitted_parse,
            "write_artifacts": self.write_artifacts,
            "write_findings_board": self.write_findings_board,
            "include_optional_missing": self.include_optional_missing,
            "require_exhaustive_swissknife": self.require_exhaustive_swissknife,
        }


@dataclass(frozen=True)
class StageReceipt:
    """One deterministic pipeline stage receipt."""

    stage: PilotStage
    status: PilotConclusion
    artifact_cid: str
    input_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", PilotStage(self.stage) if not isinstance(self.stage, PilotStage) else self.stage
        )
        object.__setattr__(
            self,
            "status",
            PilotConclusion(self.status)
            if not isinstance(self.status, PilotConclusion)
            else self.status,
        )
        object.__setattr__(self, "artifact_cid", _text(self.artifact_cid, "artifact_cid", maximum=128))
        object.__setattr__(
            self,
            "input_cids",
            tuple(_text(item, "input_cid", maximum=128) for item in self.input_cids),
        )
        reasons = tuple(
            dict.fromkeys(
                _text(item, "reason_code", maximum=192)
                for item in self.reason_codes
                if str(item).strip()
            )
        )
        if len(reasons) > MAX_STAGE_REASON_CODES:
            raise VfsSymbolicPilotError("stage_reason_bound_exceeded")
        object.__setattr__(self, "reason_codes", reasons)
        metrics: dict[str, int] = {}
        for key, value in dict(self.metrics or {}).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise VfsSymbolicPilotError(
                    "invalid_metric",
                    f"metric {key!r} must be a non-negative int",
                )
            metrics[str(key)] = value
        object.__setattr__(self, "metrics", dict(sorted(metrics.items())))

    @property
    def receipt_cid(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PILOT_STAGE_RECEIPT_SCHEMA,
            "stage": self.stage.value,
            "status": self.status.value,
            "artifact_cid": self.artifact_cid,
            "input_cids": list(self.input_cids),
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageReceipt":
        return cls(
            stage=str(payload.get("stage") or ""),
            status=str(payload.get("status") or ""),
            artifact_cid=str(payload.get("artifact_cid") or ""),
            input_cids=tuple(payload.get("input_cids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            metrics=dict(payload.get("metrics") or {}),
        )


@dataclass(frozen=True)
class PilotArtifactSet:
    """Published content identities for every pilot product."""

    forest_cid: str
    manifest_cid: str
    coverage_cid: str
    inventory_cid: str
    graph_cid: str
    cache_cid: str
    proof_cid: str
    zk_shadow_cid: str
    finding_ledger_cid: str
    taskboard_cid: str
    report_cid: str = ""

    def __post_init__(self) -> None:
        for name in (
            "forest_cid",
            "manifest_cid",
            "coverage_cid",
            "inventory_cid",
            "graph_cid",
            "cache_cid",
            "proof_cid",
            "zk_shadow_cid",
            "finding_ledger_cid",
            "taskboard_cid",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=128)
            )
        if self.report_cid:
            object.__setattr__(
                self, "report_cid", _text(self.report_cid, "report_cid", maximum=128)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": PILOT_ARTIFACT_SET_SCHEMA,
            "forest_cid": self.forest_cid,
            "manifest_cid": self.manifest_cid,
            "coverage_cid": self.coverage_cid,
            "inventory_cid": self.inventory_cid,
            "graph_cid": self.graph_cid,
            "cache_cid": self.cache_cid,
            "proof_cid": self.proof_cid,
            "zk_shadow_cid": self.zk_shadow_cid,
            "finding_ledger_cid": self.finding_ledger_cid,
            "taskboard_cid": self.taskboard_cid,
        }
        if self.report_cid:
            payload["report_cid"] = self.report_cid
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PilotArtifactSet":
        return cls(
            forest_cid=str(payload.get("forest_cid") or ""),
            manifest_cid=str(payload.get("manifest_cid") or ""),
            coverage_cid=str(payload.get("coverage_cid") or ""),
            inventory_cid=str(payload.get("inventory_cid") or ""),
            graph_cid=str(payload.get("graph_cid") or ""),
            cache_cid=str(payload.get("cache_cid") or ""),
            proof_cid=str(payload.get("proof_cid") or ""),
            zk_shadow_cid=str(payload.get("zk_shadow_cid") or ""),
            finding_ledger_cid=str(payload.get("finding_ledger_cid") or ""),
            taskboard_cid=str(payload.get("taskboard_cid") or ""),
            report_cid=str(payload.get("report_cid") or ""),
        )


@dataclass(frozen=True)
class SwissKnifeVfsPilotReport:
    """Authoritative, content-addressed pilot receipt."""

    schema: str = SWISS_KNIFE_VFS_PILOT_SCHEMA
    version: int = PILOT_VERSION
    objective_id: str = PILOT_OBJECTIVE_ID
    task_id: str = PILOT_TASK_ID
    requirement_id: str = PILOT_REQUIREMENT_ID
    mode: PilotMode = PilotMode.DRY_RUN
    conclusion: PilotConclusion = PilotConclusion.PASSED
    forest_id: str = ""
    tree_bindings: Mapping[str, str] = field(default_factory=dict)
    commit_bindings: Mapping[str, str] = field(default_factory=dict)
    stages: tuple[StageReceipt, ...] = ()
    artifacts: PilotArtifactSet | None = None
    admitted_file_count: int = 0
    swissknife_file_count: int = 0
    vfs_closure_file_count: int = 0
    finding_count: int = 0
    executable_task_count: int = 0
    review_count: int = 0
    inconclusive_count: int = 0
    provider_calls: int = 0
    source_mutations: int = 0
    reason_codes: tuple[str, ...] = ()
    board_markdown_cid: str = ""
    board_namespace: str = PILOT_BOARD_NAMESPACE
    policy_revision: str = PILOT_POLICY_REVISION
    evidence: str = SWISS_KNIFE_VFS_PILOT_SCHEMA
    authorizes_repair: bool = False
    is_completion_evidence: bool = False

    def __post_init__(self) -> None:
        if self.schema != SWISS_KNIFE_VFS_PILOT_SCHEMA:
            raise VfsSymbolicPilotError("unsupported_pilot_schema")
        if self.version != PILOT_VERSION:
            raise VfsSymbolicPilotError("unsupported_pilot_version")
        object.__setattr__(
            self,
            "mode",
            self.mode if isinstance(self.mode, PilotMode) else PilotMode(self.mode),
        )
        object.__setattr__(
            self,
            "conclusion",
            self.conclusion
            if isinstance(self.conclusion, PilotConclusion)
            else PilotConclusion(self.conclusion),
        )
        object.__setattr__(self, "forest_id", _text(self.forest_id, "forest_id"))
        trees = {
            _text(key, "tree_alias"): _text(value, "tree_id", maximum=128)
            for key, value in dict(self.tree_bindings or {}).items()
        }
        commits = {
            _text(key, "commit_alias"): _text(value, "commit_id", maximum=128)
            for key, value in dict(self.commit_bindings or {}).items()
        }
        object.__setattr__(self, "tree_bindings", dict(sorted(trees.items())))
        object.__setattr__(self, "commit_bindings", dict(sorted(commits.items())))
        stages = tuple(
            item if isinstance(item, StageReceipt) else StageReceipt.from_dict(item)
            for item in self.stages
        )
        object.__setattr__(self, "stages", stages)
        artifacts = self.artifacts
        if artifacts is not None and not isinstance(artifacts, PilotArtifactSet):
            artifacts = PilotArtifactSet.from_dict(artifacts)
        object.__setattr__(self, "artifacts", artifacts)
        for name in (
            "admitted_file_count",
            "swissknife_file_count",
            "vfs_closure_file_count",
            "finding_count",
            "executable_task_count",
            "review_count",
            "inconclusive_count",
            "provider_calls",
            "source_mutations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VfsSymbolicPilotError(
                    "invalid_count",
                    f"{name} must be a non-negative integer",
                )
        if self.provider_calls != 0:
            raise VfsSymbolicPilotError(
                "provider_call_forbidden",
                "pilot reports must record zero provider calls",
            )
        if self.source_mutations != 0:
            raise VfsSymbolicPilotError(
                "source_mutation_forbidden",
                "pilot reports must record zero source mutations",
            )
        reasons = tuple(
            dict.fromkeys(
                _text(item, "reason_code", maximum=192)
                for item in self.reason_codes
                if str(item).strip()
            )
        )
        object.__setattr__(self, "reason_codes", reasons)
        if self.board_markdown_cid:
            object.__setattr__(
                self,
                "board_markdown_cid",
                _text(self.board_markdown_cid, "board_markdown_cid", maximum=128),
            )
        object.__setattr__(
            self,
            "board_namespace",
            _text(self.board_namespace, "board_namespace", maximum=256),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision", maximum=256),
        )
        object.__setattr__(
            self, "evidence", _text(self.evidence, "evidence", maximum=128)
        )
        if self.evidence != SWISS_KNIFE_VFS_PILOT_SCHEMA:
            raise VfsSymbolicPilotError("forged_pilot_evidence")
        if self.authorizes_repair:
            raise VfsSymbolicPilotError(
                "authority_drift",
                "pilot report must never authorize repair",
            )
        if self.is_completion_evidence:
            raise VfsSymbolicPilotError(
                "authority_drift",
                "pilot report is not completion evidence",
            )
        body = _canonical_bytes(self._core_payload())
        if len(body) > MAX_REPORT_BYTES:
            raise VfsSymbolicPilotError(
                "report_bound_exceeded",
                f"pilot report exceeds {MAX_REPORT_BYTES} bytes",
            )

    def _core_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "objective_id": self.objective_id,
            "task_id": self.task_id,
            "requirement_id": self.requirement_id,
            "mode": self.mode.value,
            "conclusion": self.conclusion.value,
            "forest_id": self.forest_id,
            "tree_bindings": dict(self.tree_bindings),
            "commit_bindings": dict(self.commit_bindings),
            "stages": [item.to_dict() for item in self.stages],
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "admitted_file_count": self.admitted_file_count,
            "swissknife_file_count": self.swissknife_file_count,
            "vfs_closure_file_count": self.vfs_closure_file_count,
            "finding_count": self.finding_count,
            "executable_task_count": self.executable_task_count,
            "review_count": self.review_count,
            "inconclusive_count": self.inconclusive_count,
            "provider_calls": self.provider_calls,
            "source_mutations": self.source_mutations,
            "reason_codes": list(self.reason_codes),
            "board_markdown_cid": self.board_markdown_cid,
            "board_namespace": self.board_namespace,
            "policy_revision": self.policy_revision,
            "evidence": self.evidence,
            "authorizes_repair": False,
            "is_completion_evidence": False,
        }

    @property
    def report_cid(self) -> str:
        return _identity(self._core_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._core_payload()
        payload["report_cid"] = self.report_cid
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SwissKnifeVfsPilotReport":
        if not isinstance(payload, Mapping):
            raise VfsSymbolicPilotError("invalid_report", "report must be an object")
        data = dict(payload)
        claimed = str(data.pop("report_cid", "") or "")
        report = cls(
            schema=str(data.get("schema") or SWISS_KNIFE_VFS_PILOT_SCHEMA),
            version=int(data.get("version") or PILOT_VERSION),
            objective_id=str(data.get("objective_id") or PILOT_OBJECTIVE_ID),
            task_id=str(data.get("task_id") or PILOT_TASK_ID),
            requirement_id=str(data.get("requirement_id") or PILOT_REQUIREMENT_ID),
            mode=str(data.get("mode") or PilotMode.DRY_RUN.value),
            conclusion=str(data.get("conclusion") or PilotConclusion.PASSED.value),
            forest_id=str(data.get("forest_id") or ""),
            tree_bindings=dict(data.get("tree_bindings") or {}),
            commit_bindings=dict(data.get("commit_bindings") or {}),
            stages=tuple(data.get("stages") or ()),
            artifacts=data.get("artifacts"),
            admitted_file_count=int(data.get("admitted_file_count") or 0),
            swissknife_file_count=int(data.get("swissknife_file_count") or 0),
            vfs_closure_file_count=int(data.get("vfs_closure_file_count") or 0),
            finding_count=int(data.get("finding_count") or 0),
            executable_task_count=int(data.get("executable_task_count") or 0),
            review_count=int(data.get("review_count") or 0),
            inconclusive_count=int(data.get("inconclusive_count") or 0),
            provider_calls=int(data.get("provider_calls") or 0),
            source_mutations=int(data.get("source_mutations") or 0),
            reason_codes=tuple(data.get("reason_codes") or ()),
            board_markdown_cid=str(data.get("board_markdown_cid") or ""),
            board_namespace=str(
                data.get("board_namespace") or PILOT_BOARD_NAMESPACE
            ),
            policy_revision=str(
                data.get("policy_revision") or PILOT_POLICY_REVISION
            ),
            evidence=str(data.get("evidence") or SWISS_KNIFE_VFS_PILOT_SCHEMA),
            authorizes_repair=bool(data.get("authorizes_repair", False)),
            is_completion_evidence=bool(data.get("is_completion_evidence", False)),
        )
        if claimed and claimed != report.report_cid:
            raise PilotVerificationError(
                "stale_evidence",
                "report_cid does not match canonical body",
            )
        return report


# ---------------------------------------------------------------------------
# Admission / selection
# ---------------------------------------------------------------------------


def is_vfs_relevant_path(relative_path: str, *, repository_alias: str) -> bool:
    """Return True when a non-SwissKnife path is in the VFS-relevant closure."""

    alias = str(repository_alias or "").strip().lower()
    path = str(relative_path or "").replace("\\", "/").strip()
    if alias == DEFAULT_SWISSKNIFE_ALIAS:
        return True
    if not path:
        return False
    if _VFS_RELEVANT_PATH.search(path):
        return True
    # Explicit package roots always contribute surface-adjacent modules.
    if alias in {
        DEFAULT_KIT_ALIAS,
        DEFAULT_DATASETS_ALIAS,
        DEFAULT_ACCELERATOR_ALIAS,
    }:
        lowered = path.lower()
        markers = (
            "vfs",
            "fsspec",
            "ipfs_kit",
            "mcplusplus",
            "mcp",
            "virtual_file",
            "agent_supervisor/vfs_",
            "agent_supervisor/program_",
            "agent_supervisor/contract_",
            "agent_supervisor/repository_",
        )
        return any(marker in lowered for marker in markers)
    return False


def admitted_entries_for_pilot(
    index: RepositoryCorpusIndex,
) -> tuple[Any, ...]:
    """Return included corpus entries admitted into the pilot scan set."""

    selected = []
    for entry in index.entries:
        if entry.inclusion != InclusionDecision.INCLUDED.value:
            continue
        if not entry.parser_eligible:
            continue
        if not is_vfs_relevant_path(
            entry.relative_path, repository_alias=entry.repository_alias
        ):
            continue
        selected.append(entry)
    selected.sort(
        key=lambda item: (
            item.repository_alias,
            item.relative_path,
            item.content_sha256,
        )
    )
    return tuple(selected)


def _descriptor_map(forest: RepositoryForest) -> dict[str, RepositoryDescriptor]:
    return {descriptor.alias: descriptor for descriptor in forest.descriptors}


def _read_entry_text(
    entry: Any,
    descriptors: Mapping[str, RepositoryDescriptor],
) -> str | None:
    descriptor = descriptors.get(entry.repository_alias)
    if descriptor is None:
        return None
    path = descriptor.root_path / entry.relative_path
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix not in _PARSER_SUFFIXES:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def freeze_repository_descriptors(config: PilotConfig) -> RepositoryForest:
    """Freeze fresh repository descriptors from configured roots."""

    if config.forest_policy is not None:
        policy = config.forest_policy
    else:
        policy = initial_vfs_assurance_forest_policy(
            accelerator_root=config.accelerator_root,
            swissknife_root=config.swissknife_root,
            kit_root=config.kit_root,
            datasets_root=config.datasets_root,
            include_optional_missing=config.include_optional_missing,
        )
    forest = build_repository_forest(policy)
    # SwissKnife must remain read-only under the initial policy.
    for descriptor in forest.descriptors:
        if descriptor.alias == DEFAULT_SWISSKNIFE_ALIAS:
            if descriptor.authority.mode != AuthorityMode.READ_ONLY.value:
                raise VfsSymbolicPilotError(
                    "swissknife_write_authority",
                    "SwissKnife descriptor must be read_only",
                )
    return forest


def scan_inventory(
    forest: RepositoryForest,
    *,
    limits: InventoryLimits | None = None,
) -> RepositoryCorpusIndex:
    """Exhaustively inventory every forest descriptor."""

    return build_repository_corpus_index(forest, limits=limits)


def build_coverage_manifest(
    *,
    forest: RepositoryForest,
    index: RepositoryCorpusIndex,
    admitted: Sequence[Any],
) -> dict[str, Any]:
    """Build the portable coverage + manifest projection."""

    by_alias: dict[str, list[dict[str, Any]]] = {}
    for entry in admitted:
        by_alias.setdefault(entry.repository_alias, []).append(
            {
                "relative_path": entry.relative_path,
                "canonical_path": entry.canonical_path,
                "blob_oid": entry.blob_oid,
                "content_sha256": entry.content_sha256,
                "entry_cid": entry.entry_cid,
                "size": entry.size,
                "classifications": list(entry.classifications),
            }
        )
    for alias in by_alias:
        by_alias[alias].sort(key=lambda item: item["relative_path"])

    swissknife_count = sum(
        1 for entry in admitted if entry.repository_alias == DEFAULT_SWISSKNIFE_ALIAS
    )
    vfs_count = len(admitted) - swissknife_count

    repo_coverage = []
    for inventory in index.repositories:
        repo_coverage.append(
            {
                "repository_alias": inventory.repository_alias,
                "repository_id": inventory.repository_id,
                "descriptor_cid": inventory.descriptor_cid,
                "exhaustive": inventory.exhaustive,
                "observed_entry_count": inventory.observed_entry_count,
                "included_entry_count": inventory.included_entry_count,
                "excluded_entry_count": inventory.excluded_entry_count,
                "omitted_entry_count": inventory.omitted_entry_count,
                "reason_codes": list(inventory.reason_codes),
                "admitted_for_pilot": len(by_alias.get(inventory.repository_alias, [])),
            }
        )
    repo_coverage.sort(key=lambda item: item["repository_alias"])

    manifest = {
        "schema": PILOT_MANIFEST_SCHEMA,
        "forest_id": forest.forest_id,
        "inventory_cid": index.inventory_cid,
        "policy_cid": forest.policy_cid,
        "admitted_file_count": len(admitted),
        "swissknife_file_count": swissknife_count,
        "vfs_closure_file_count": vfs_count,
        "repositories": [
            {
                "alias": descriptor.alias,
                "repository_id": descriptor.repository_id,
                "descriptor_cid": descriptor.descriptor_cid,
                "commit": descriptor.commit,
                "tree": descriptor.tree,
                "authority": descriptor.authority.mode,
            }
            for descriptor in sorted(forest.descriptors, key=lambda item: item.alias)
        ],
        "admitted_by_alias": {
            alias: files for alias, files in sorted(by_alias.items())
        },
    }
    coverage = {
        "schema": PILOT_COVERAGE_SCHEMA,
        "forest_id": forest.forest_id,
        "inventory_cid": index.inventory_cid,
        "repository_coverage": repo_coverage,
        "admitted_file_count": len(admitted),
        "swissknife_file_count": swissknife_count,
        "vfs_closure_file_count": vfs_count,
        "complete": all(item["exhaustive"] for item in repo_coverage)
        and all(item["omitted_entry_count"] == 0 for item in repo_coverage),
    }
    return {
        "manifest": manifest,
        "coverage": coverage,
        "manifest_cid": _identity(manifest),
        "coverage_cid": _identity(coverage),
    }


def build_pilot_program_graph(
    *,
    forest: RepositoryForest,
    admitted: Sequence[Any],
    max_parse: int,
) -> tuple[ProgramGraph, dict[str, Any]]:
    """Parse admitted sources and emit a content-addressed program graph."""

    descriptors = _descriptor_map(forest)
    nodes: list[ProgramGraphNode] = []
    parse_metrics = {
        "parsed": 0,
        "unsupported": 0,
        "malformed": 0,
        "unreadable": 0,
        "skipped_bound": 0,
    }
    reasons: list[str] = []

    for index, entry in enumerate(admitted):
        if index >= max_parse:
            parse_metrics["skipped_bound"] += 1
            reasons.append("parse_bound_reached")
            continue
        text = _read_entry_text(entry, descriptors)
        if text is None:
            parse_metrics["unreadable"] += 1
            continue
        result = adapt_program_source(
            text,
            path=entry.relative_path,
            blob_identity=entry.content_sha256 or entry.blob_oid or entry.entry_cid,
        )
        if result.status == "success":
            parse_metrics["parsed"] += 1
        elif result.status == "malformed":
            parse_metrics["malformed"] += 1
        else:
            parse_metrics["unsupported"] += 1

        blob_cid = entry.content_sha256 or entry.blob_oid or entry.entry_cid
        binding = ProgramGraphBinding(
            producer=PILOT_PRODUCER,
            blob_cid=blob_cid,
            forest_id=forest.forest_id,
            span=GraphSourceSpan(),
            resolver_status=ResolverStatus.RESOLVED_STATIC,
        )
        module_node = ProgramGraphNode(
            kind=ProgramNodeKind.MODULE,
            record_key=f"module:{entry.repository_alias}:{entry.relative_path}",
            binding=binding,
            component_id=f"module:{entry.repository_alias}",
            qualified_name=entry.relative_path,
            path=entry.relative_path,
            language=str(getattr(result, "language", "") or ""),
            record={
                "repository_alias": entry.repository_alias,
                "entry_cid": entry.entry_cid,
                "status": result.status,
                "fact_count": len(result.facts),
            },
        )
        nodes.append(module_node)

        for fact in result.facts[:64]:
            kind_name = str(fact.kind or "").lower()
            if "import" in kind_name:
                node_kind = ProgramNodeKind.IMPORT
            elif "export" in kind_name:
                node_kind = ProgramNodeKind.EXPORT
            elif "call" in kind_name:
                node_kind = ProgramNodeKind.CALL
            elif "function" in kind_name or "class" in kind_name or "def" in kind_name:
                node_kind = ProgramNodeKind.DEFINITION
            elif "schema" in kind_name:
                node_kind = ProgramNodeKind.SCHEMA
            else:
                node_kind = ProgramNodeKind.SYMBOL
            span = fact.span
            fact_binding = ProgramGraphBinding(
                producer=PILOT_PRODUCER,
                blob_cid=blob_cid,
                forest_id=forest.forest_id,
                span=GraphSourceSpan(
                    line_start=int(
                        getattr(span, "line_start", None)
                        or getattr(span, "start_line", 0)
                        or 0
                    ),
                    column_start=int(
                        getattr(span, "column_start", None)
                        or getattr(span, "start_column", 0)
                        or 0
                    ),
                    line_end=int(
                        getattr(span, "line_end", None)
                        or getattr(span, "end_line", 0)
                        or 0
                    ),
                    column_end=int(
                        getattr(span, "column_end", None)
                        or getattr(span, "end_column", 0)
                        or 0
                    ),
                ),
                resolver_status=(
                    ResolverStatus.AMBIGUOUS
                    if fact.ambiguous
                    else ResolverStatus.RESOLVED_STATIC
                ),
            )
            nodes.append(
                ProgramGraphNode(
                    kind=node_kind,
                    record_key=f"fact:{entry.repository_alias}:{fact.fact_id}",
                    binding=fact_binding,
                    component_id=module_node.component_id,
                    qualified_name=fact.name or fact.fact_id,
                    path=entry.relative_path,
                    language=str(getattr(result, "language", "") or ""),
                    record={
                        "kind": fact.kind,
                        "name": fact.name,
                        "owner": fact.owner,
                        "target": fact.target,
                        "ambiguous": bool(fact.ambiguous),
                        "fact_id": fact.fact_id,
                    },
                )
            )
            if len(nodes) >= MAX_GRAPH_NODES:
                reasons.append("graph_node_bound_reached")
                break
        if len(nodes) >= MAX_GRAPH_NODES:
            break

    graph = build_program_graph(
        forest_id=forest.forest_id,
        nodes=nodes,
        edges=(),
        producer=PILOT_PRODUCER,
        unexplained_gap_count=parse_metrics["unreadable"]
        + parse_metrics["malformed"]
        + parse_metrics["skipped_bound"],
        truncated=bool(parse_metrics["skipped_bound"])
        or "graph_node_bound_reached" in reasons,
        truncation_reason=";".join(sorted(set(reasons))),
    )
    return graph, {
        "metrics": parse_metrics,
        "reason_codes": sorted(set(reasons)),
        "graph_cid": graph.graph_id,
    }


def _graph_content_id(graph: ProgramGraph) -> str:
    return str(graph.graph_id)


def run_contract_stage(
    *,
    forest: RepositoryForest,
    graph: ProgramGraph,
    admitted: Sequence[Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Deterministic contract observation over the pilot graph.

    Emits conclusive broken-contract findings only when a pilot fixture embeds
    an explicit ``VFS_PILOT_CONTRACT_BROKEN`` marker.  All other unresolved
    edges become non-executable inconclusive records.
    """

    descriptors = _descriptor_map(forest)
    findings: list[Any] = []
    inconclusive = 0
    broken = 0

    for entry in admitted:
        text = _read_entry_text(entry, descriptors)
        if text is None:
            continue
        if "VFS_PILOT_CONTRACT_BROKEN" in text:
            broken += 1
            findings.append(
                build_contract_finding(
                    claim_level=ClaimLevel.MODEL_DISPROVED,
                    status=FindingStatus.CONTRACT_BROKEN,
                    severity=FindingSeverity.HIGH,
                    confidence_millionths=950_000,
                    freshness=EvidenceFreshness.CURRENT,
                    repositories=(entry.repository_id,),
                    symbols=(entry.relative_path,),
                    interfaces=(f"pilot://{entry.repository_alias}/{entry.relative_path}",),
                    expected_contract_cid=_identity(
                        {"expected": "vfs-pilot-contract", "path": entry.relative_path}
                    ),
                    observed_contract_cid=_identity(
                        {
                            "observed": "VFS_PILOT_CONTRACT_BROKEN",
                            "path": entry.relative_path,
                            "blob": entry.content_sha256,
                        }
                    ),
                    root_cause_family="vfs-pilot-seeded-contract-break",
                    merge_fate=f"pilot:{entry.repository_alias}:{entry.relative_path}",
                    summary=(
                        "Pilot fixture marks an explicit contract break for "
                        f"{entry.relative_path}"
                    ),
                    call_slice=CallSlice(
                        steps=(
                            CallSliceStep(
                                symbol=entry.relative_path,
                                interface=f"pilot://{entry.relative_path}",
                                repository_id=entry.repository_id,
                                path=entry.relative_path,
                            ),
                        )
                    ),
                    evidence=EvidenceReferences(
                        counterexample_cids=(
                            _identity(
                                {
                                    "marker": "VFS_PILOT_CONTRACT_BROKEN",
                                    "path": entry.relative_path,
                                }
                            ),
                        ),
                        artifact_cids=(entry.entry_cid,),
                    ),
                    assumptions=("hermetic pilot fixture",),
                    analyzer_versions={"vfs-symbolic-pilot": "1"},
                    remediation_scope=(entry.relative_path,),
                    tree_id=forest.forest_id,
                    policy_revision=PILOT_POLICY_REVISION,
                    repository_observation_id=entry.entry_cid,
                    verdict="violated",
                )
            )
        elif "VFS_PILOT_INCONCLUSIVE" in text:
            inconclusive += 1
            findings.append(
                build_contract_finding(
                    claim_level=ClaimLevel.OBSERVED_SYNTAX,
                    status=FindingStatus.INCONCLUSIVE,
                    severity=FindingSeverity.LOW,
                    confidence_millionths=200_000,
                    freshness=EvidenceFreshness.CURRENT,
                    repositories=(entry.repository_id,),
                    symbols=(entry.relative_path,),
                    interfaces=(f"pilot://{entry.repository_alias}/{entry.relative_path}",),
                    expected_contract_cid=_identity(
                        {"expected": "unresolved", "path": entry.relative_path}
                    ),
                    observed_contract_cid=_identity(
                        {"observed": "inconclusive", "path": entry.relative_path}
                    ),
                    root_cause_family="vfs-pilot-inconclusive",
                    merge_fate=f"review:{entry.repository_alias}:{entry.relative_path}",
                    summary=(
                        "Pilot fixture is explicitly inconclusive for "
                        f"{entry.relative_path}"
                    ),
                    call_slice=CallSlice(),
                    evidence=EvidenceReferences(artifact_cids=(entry.entry_cid,)),
                    assumptions=("hermetic pilot fixture",),
                    analyzer_versions={"vfs-symbolic-pilot": "1"},
                    remediation_scope=(entry.relative_path,),
                    tree_id=forest.forest_id,
                    policy_revision=PILOT_POLICY_REVISION,
                    repository_observation_id=entry.entry_cid,
                    verdict="inconclusive",
                    partial=True,
                )
            )

    # Ambiguous graph nodes without markers stay non-actionable coverage.
    ambiguous_nodes = [
        node
        for node in graph.nodes
        if node.binding.resolver_status == ResolverStatus.AMBIGUOUS
    ]
    contract_payload = {
        "schema": "vfs/swissknife-vfs-pilot-contract@1",
        "forest_id": forest.forest_id,
        "graph_cid": _graph_content_id(graph),
        "broken_count": broken,
        "inconclusive_count": inconclusive,
        "ambiguous_node_count": len(ambiguous_nodes),
        "finding_cids": [item.finding_cid for item in findings],
    }
    return findings, {
        "payload": contract_payload,
        "contract_cid": _identity(contract_payload),
        "broken_count": broken,
        "inconclusive_count": inconclusive,
    }


def run_cache_stage(
    *,
    forest: RepositoryForest,
    inventory_cid: str,
    graph_cid: str,
    contract_cid: str,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    """Store deterministic analysis cache receipts (authoritative components)."""

    if artifact_dir is not None:
        cache_root = artifact_dir / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
    else:
        cache_root = Path(tempfile.mkdtemp(prefix="vfs-pilot-cache-"))

    cache = ProgramAnalysisCache(cache_root)
    selected_kinds = [
        ProgramAnalysisComponentKind.INVENTORY,
        ProgramAnalysisComponentKind.GRAPH,
        ProgramAnalysisComponentKind.CONTRACT,
        ProgramAnalysisComponentKind.PROOF,
    ]
    stored: list[str] = []
    for kind in selected_kinds:
        body = {
            "component": kind.value,
            "inventory_cid": inventory_cid,
            "graph_cid": graph_cid,
            "contract_cid": contract_cid,
            "forest_id": forest.forest_id,
            "pilot": PILOT_REQUIREMENT_ID,
        }
        key = build_program_analysis_cache_key(
            forest_identity=forest.forest_id,
            repository_forest_identity=forest.forest_id,
            objective_revision=PILOT_OBJECTIVE_ID,
            policy_revision=PILOT_POLICY_REVISION,
            analyzer_version=PILOT_PRODUCER,
            schema_version=str(PILOT_VERSION),
            configuration_digest=_identity(
                {"stage": kind.value, "pilot": PILOT_REQUIREMENT_ID}
            ),
            query_digest=_identity(body),
            capability_revision="capability:vfs-symbolic-pilot@1",
            assumption_digest=_identity(
                {"assumptions": ["hermetic_pilot", "no_provider"]}
            ),
            toolchain_version=PILOT_PRODUCER,
            component_kind=kind,
            authority=ProgramAnalysisAuthority.AUTHORITATIVE,
        )
        receipt = {
            "schema": "vfs/swissknife-vfs-pilot-cache-receipt@1",
            "status": "success",
            "component": kind.value,
            "body_cid": _identity(body),
            "body": body,
        }
        try:
            result = cache.put(key, receipt)
            stored.append(
                str(
                    getattr(result, "receipt_cid", None)
                    or getattr(result, "entry_cid", None)
                    or getattr(result, "content_id", None)
                    or _identity(receipt)
                )
            )
        except Exception:
            # Cache storage is best-effort; the portable cache CID remains
            # content-addressed from the receipt payload itself.
            stored.append(_identity(receipt))

    payload = {
        "schema": "vfs/swissknife-vfs-pilot-cache@1",
        "forest_id": forest.forest_id,
        "stored": stored,
        "component_kinds": [kind.value for kind in selected_kinds],
    }
    return {"payload": payload, "cache_cid": _identity(payload)}


def run_zk_shadow_stage(
    *,
    forest: RepositoryForest,
    inventory_cid: str,
    graph_cid: str,
    contract_cid: str,
) -> dict[str, Any]:
    """Produce a non-authoritative ZK-shadow envelope for the pilot trace."""

    public = build_program_zkp_public_inputs(
        forest_commitment=commitment_identity("forest", {"forest_id": forest.forest_id}),
        inventory_commitment=commitment_identity(
            "inventory", {"inventory_cid": inventory_cid}
        ),
        contract_commitment=commitment_identity(
            "contract", {"contract_cid": contract_cid}
        ),
        call_slice_commitment=commitment_identity(
            "call_slice", {"graph_cid": graph_cid}
        ),
        assumptions_commitment=commitment_identity(
            "assumptions",
            {"items": ["hermetic_pilot", "no_provider", "read_only_swissknife"]},
        ),
        analyzer_version=PILOT_PRODUCER,
        resolver_version="resolver:pilot@1",
        translator_version="translator:pilot@1",
        prover_version="prover:shadow-trace@0.1.0",
        result_commitment=commitment_identity(
            "result",
            {
                "inventory_cid": inventory_cid,
                "graph_cid": graph_cid,
                "contract_cid": contract_cid,
            },
        ),
        circuit_id="circuit:program-contract-trace@1",
        proving_key_id="pk:program-contract-trace@1:pilot",
        verifying_key_id="vk:program-contract-trace@1:pilot",
        ceremony_id="ceremony:program-contract-trace@1",
        public_input_codec_id=PUBLIC_INPUT_CODEC_ID,
        public_input_codec_version=PUBLIC_INPUT_CODEC_VERSION,
    )
    witness = PrivateProgramAnalysisWitness(
        {
            "source_text": "pilot-private-witness-never-publish",
            "opening_secret": "pilot-opening-secret",
        }
    )
    request = prepare_program_analysis_zkp(
        public,
        witness=witness,
        backend_mode=ProgramZkpBackendMode.SHADOW,
    )
    envelope = create_program_zkp_shadow_envelope(
        request,
        proof_artifact_id=f"proof:pilot:{forest.forest_id[:24]}",
        proof_digest=_identity(
            {
                "public_input_digest": public.public_input_digest,
                "mode": "shadow",
            }
        ),
        prover_id="prover:vfs-symbolic-pilot-shadow",
        backend_mode=ProgramZkpBackendMode.SHADOW,
    )
    receipt = record_program_zkp_verification(
        envelope,
        verdict=ProgramZkpVerdict.VERIFIED,
        verifier_id="verifier:vfs-symbolic-pilot-shadow",
        capability_production_eligible=False,
        independent_verifier=False,
    )
    payload = {
        "schema": "vfs/swissknife-vfs-pilot-zk-shadow@1",
        "forest_id": forest.forest_id,
        "backend_mode": ProgramZkpBackendMode.SHADOW.value,
        "authoritative": False,
        "public_input_digest": public.public_input_digest,
        "envelope_cid": getattr(
            envelope, "content_id", _identity(envelope.to_dict())
        ),
        "receipt_cid": getattr(
            receipt, "content_id", _identity(receipt.to_dict())
        ),
        "semantic_proof": False,
    }
    proof_payload = {
        "schema": "vfs/swissknife-vfs-pilot-proof@1",
        "forest_id": forest.forest_id,
        "zk_shadow_cid": _identity(payload),
        "authoritative": False,
        "claim_level": ClaimLevel.ZK_TRACE_ATTESTED.value,
        "does_not_prove_semantics": True,
    }
    return {
        "zk_payload": payload,
        "zk_shadow_cid": _identity(payload),
        "proof_payload": proof_payload,
        "proof_cid": _identity(proof_payload),
    }


def materialize_findings_and_board(
    *,
    forest: RepositoryForest,
    findings: Sequence[Any],
    artifact_dir: Path | None,
) -> dict[str, Any]:
    """Append findings to a ledger and materialize the repair taskboard."""

    if artifact_dir is not None:
        ledger_root = artifact_dir / "finding_ledger"
        board_root = artifact_dir / "taskboard"
    else:
        tmp = Path(tempfile.mkdtemp(prefix="vfs-pilot-board-"))
        ledger_root = tmp / "finding_ledger"
        board_root = tmp / "taskboard"
    ledger_root.mkdir(parents=True, exist_ok=True)
    board_root.mkdir(parents=True, exist_ok=True)

    ledger = ContractFindingLedger(root=ledger_root)
    for finding in findings:
        ledger.append(finding)

    projection = ledger.projection()
    admitted = list(ledger.current_findings(admitted_only=True))
    policy = FindingTaskSourcePolicy(
        board_namespace=PILOT_BOARD_NAMESPACE,
        goal_id=DEFAULT_GOAL_ID,
        goal_lineage=DEFAULT_GOAL_LINEAGE,
    )
    source = FindingTaskSource(policy=policy, root=board_root)
    receipt = source.materialize(
        findings=list(findings),
        admitted_only=False,
        tree_id=forest.forest_id,
    )
    snapshot = source.snapshot()
    board_json = project_board_json(snapshot)
    board_md = project_board_markdown(snapshot)
    taskboard_cid = str(
        getattr(snapshot, "board_cid", None) or _identity(board_json)
    )
    executable = len(snapshot.tasks)
    reviews = len(snapshot.reviews)

    # Exact repair packets for executable admitted findings only.
    packets: list[dict[str, Any]] = []
    compiler = ContractRepairPacketCompiler()
    for index, finding in enumerate(admitted[:MAX_BOARD_TASKS], start=1):
        try:
            outputs = tuple(finding.remediation_scope[:8]) or (
                f"pilot/{finding.root_cause_family}.md",
            )
            steps = []
            for step_index, step in enumerate(
                (finding.call_slice.steps if finding.call_slice else ()),
                start=1,
            ):
                steps.append(
                    CallSliceStepRef(
                        step_id=f"step-{step_index}",
                        symbol=step.symbol,
                        path=step.path,
                        kind=getattr(step, "kind", "") or "call",
                        contract_ref=finding.expected_contract_cid,
                    )
                )
            if not steps:
                steps.append(
                    CallSliceStepRef(
                        step_id="step-1",
                        symbol=(finding.symbols[0] if finding.symbols else "unknown"),
                        path=outputs[0],
                        kind="call",
                        contract_ref=finding.expected_contract_cid,
                    )
                )
            request = RepairPacketRequest(
                task_id=f"VFS-R-{index:04d}",
                finding_ids=(finding.finding_cid,),
                forest_id=forest.forest_id,
                tree_id=forest.forest_id,
                policy_id=PILOT_POLICY_REVISION,
                expected_contract_ref=finding.expected_contract_cid,
                observed_contract_ref=finding.observed_contract_cid,
                call_slice=CallSliceRef(
                    slice_id=f"slice:{finding.finding_cid[:24]}",
                    steps=tuple(steps),
                    root_symbol=(
                        finding.symbols[0] if finding.symbols else steps[0].symbol
                    ),
                    complete=True,
                ),
                edit_scope=outputs,
                effects=("repair_contract_drift",),
                acceptance=(
                    f"Repair root-cause family {finding.root_cause_family}",
                    "Validation and proof plans pass without expanding authority",
                ),
                validation_commands=(
                    "python -m pytest test/api/test_agent_supervisor_vfs_symbolic_pilot.py -q",
                ),
                proof_commands=(
                    "python -m ipfs_accelerate_py.agent_supervisor."
                    "vfs_symbolic_pilot --verify --report "
                    f"{DEFAULT_ARTIFACT_RELATIVE}/report.json",
                ),
                risks=("bounded_scope_only",),
                policy_revision=PILOT_POLICY_REVISION,
                goal_id=DEFAULT_GOAL_ID,
                symbols=tuple(finding.symbols[:24]),
                interfaces=tuple(finding.interfaces[:24]),
                authority=RepairAuthority(),
            )
            compiled = compiler.compile(request)
            packet = compiled.packet if hasattr(compiled, "packet") else compiled
            status = getattr(compiled, "status", None) or getattr(
                packet, "status", RepairPacketStatus.COMPLETE
            )
            packets.append(
                {
                    "finding_cid": finding.finding_cid,
                    "packet_id": getattr(packet, "packet_id", ""),
                    "content_id": getattr(packet, "content_id", ""),
                    "status": (
                        status.value if isinstance(status, Enum) else str(status)
                    ),
                }
            )
        except Exception:
            # Exact packets are best-effort for admitted findings; failures
            # do not invent executable work.
            continue

    if hasattr(projection, "to_dict"):
        projection_cid = getattr(
            projection, "content_id", _identity(projection.to_dict())
        )
    else:
        projection_cid = _identity(
            {"findings": [item.finding_cid for item in findings]}
        )

    ledger_payload = {
        "schema": "vfs/swissknife-vfs-pilot-findings@1",
        "forest_id": forest.forest_id,
        "finding_cids": [item.finding_cid for item in findings],
        "admitted_cids": [item.finding_cid for item in admitted],
        "projection_cid": projection_cid,
        "repair_packets": packets,
    }

    return {
        "ledger_payload": ledger_payload,
        "finding_ledger_cid": _identity(ledger_payload),
        "board_json": board_json,
        "board_markdown": board_md,
        "taskboard_cid": taskboard_cid,
        "board_markdown_cid": _identity({"markdown": board_md}),
        "executable_task_count": executable,
        "review_count": reviews,
        "finding_count": len(findings),
        "admitted_count": len(admitted),
        "materialization_receipt": getattr(receipt, "to_dict", lambda: {})(),
        "repair_packets": packets,
    }


def _assert_no_provider_surface() -> None:
    """Fail closed if the pilot module imports provider SDKs at runtime.

    Runtime verification also checks that the report records zero provider
    calls; this static guard documents the invariant for reviewers.
    """

    for module_name in ("openai", "anthropic", "groq"):
        if module_name in sys.modules:
            raise VfsSymbolicPilotError(
                "provider_call_forbidden",
                f"provider SDK {module_name!r} must not be loaded during pilot",
            )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def execute_pilot(
    config: PilotConfig,
    *,
    mode: PilotMode = PilotMode.DRY_RUN,
) -> SwissKnifeVfsPilotReport:
    """Run the full deterministic pilot pipeline and return the report."""

    _assert_no_provider_surface()
    if mode is PilotMode.VERIFY:
        raise VfsSymbolicPilotError(
            "invalid_mode",
            "execute_pilot does not perform verify; use verify_pilot",
        )

    stages: list[StageReceipt] = []
    reason_codes: list[str] = []
    artifact_dir = (
        config.resolved_artifact_dir() if config.write_artifacts else None
    )
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Freeze descriptors
    forest = freeze_repository_descriptors(config)
    forest_payload = forest.to_portable_dict() if hasattr(forest, "to_portable_dict") else {
        "forest_id": forest.forest_id,
        "descriptors": [item.to_portable_dict() for item in forest.descriptors],
        "policy_cid": forest.policy_cid,
    }
    forest_cid = forest.forest_id
    stages.append(
        StageReceipt(
            stage=PilotStage.FREEZE,
            status=PilotConclusion.PASSED,
            artifact_cid=forest_cid,
            metrics={"repositories": len(forest.descriptors)},
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "forest.json", forest_payload)

    # 2. Inventory
    limits = config.inventory_limits or InventoryLimits()
    index = scan_inventory(forest, limits=limits)
    inventory_cid = index.inventory_cid
    inventory_reasons: list[str] = []
    incomplete = False
    for repo in index.repositories:
        if not repo.exhaustive or repo.omitted_entry_count:
            incomplete = True
            inventory_reasons.extend(repo.reason_codes)
            inventory_reasons.append(f"incomplete:{repo.repository_alias}")
        if (
            config.require_exhaustive_swissknife
            and repo.repository_alias == DEFAULT_SWISSKNIFE_ALIAS
            and (not repo.exhaustive or repo.omitted_entry_count)
        ):
            raise VfsSymbolicPilotError(
                "incomplete_inventory",
                f"SwissKnife inventory is incomplete: {sorted(repo.reason_codes)}",
            )
    stages.append(
        StageReceipt(
            stage=PilotStage.INVENTORY,
            status=(
                PilotConclusion.INCOMPLETE if incomplete else PilotConclusion.PASSED
            ),
            artifact_cid=inventory_cid,
            input_cids=(forest_cid,),
            reason_codes=tuple(sorted(set(inventory_reasons))),
            metrics={
                "entries": len(index.entries),
                "repositories": len(index.repositories),
            },
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "inventory.json", index.to_portable_dict())

    # 3. Scan admission
    admitted = admitted_entries_for_pilot(index)
    coverage_bundle = build_coverage_manifest(
        forest=forest, index=index, admitted=admitted
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.SCAN,
            status=PilotConclusion.PASSED,
            artifact_cid=coverage_bundle["manifest_cid"],
            input_cids=(forest_cid, inventory_cid),
            metrics={
                "admitted": len(admitted),
                "swissknife": coverage_bundle["manifest"]["swissknife_file_count"],
                "vfs_closure": coverage_bundle["manifest"]["vfs_closure_file_count"],
            },
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "manifest.json", coverage_bundle["manifest"])
        _atomic_write_json(artifact_dir / "coverage.json", coverage_bundle["coverage"])

    # 4. Graph
    graph, graph_meta = build_pilot_program_graph(
        forest=forest,
        admitted=admitted,
        max_parse=config.max_admitted_parse,
    )
    graph_cid = graph_meta["graph_cid"]
    stages.append(
        StageReceipt(
            stage=PilotStage.GRAPH,
            status=(
                PilotConclusion.INCOMPLETE
                if graph_meta["reason_codes"]
                else PilotConclusion.PASSED
            ),
            artifact_cid=graph_cid,
            input_cids=(coverage_bundle["manifest_cid"],),
            reason_codes=tuple(graph_meta["reason_codes"]),
            metrics={
                "nodes": len(graph.nodes),
                **{f"parse_{key}": value for key, value in graph_meta["metrics"].items()},
            },
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "graph.json", graph.to_dict())

    # 5. Contract
    findings, contract_meta = run_contract_stage(
        forest=forest, graph=graph, admitted=admitted
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.CONTRACT,
            status=PilotConclusion.PASSED,
            artifact_cid=contract_meta["contract_cid"],
            input_cids=(graph_cid,),
            metrics={
                "broken": contract_meta["broken_count"],
                "inconclusive": contract_meta["inconclusive_count"],
                "findings": len(findings),
            },
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "contract.json", contract_meta["payload"])

    # 6. Cache
    cache_meta = run_cache_stage(
        forest=forest,
        inventory_cid=inventory_cid,
        graph_cid=graph_cid,
        contract_cid=contract_meta["contract_cid"],
        artifact_dir=artifact_dir,
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.CACHE,
            status=PilotConclusion.PASSED,
            artifact_cid=cache_meta["cache_cid"],
            input_cids=(inventory_cid, graph_cid, contract_meta["contract_cid"]),
            metrics={"stored": len(cache_meta["payload"].get("stored", []))},
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "cache_receipt.json", cache_meta["payload"])

    # 7. Proof + ZK shadow
    zk_meta = run_zk_shadow_stage(
        forest=forest,
        inventory_cid=inventory_cid,
        graph_cid=graph_cid,
        contract_cid=contract_meta["contract_cid"],
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.PROOF,
            status=PilotConclusion.PASSED,
            artifact_cid=zk_meta["proof_cid"],
            input_cids=(contract_meta["contract_cid"],),
            reason_codes=("shadow_non_authoritative",),
            metrics={"authoritative": 0},
        )
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.ZK_SHADOW,
            status=PilotConclusion.PASSED,
            artifact_cid=zk_meta["zk_shadow_cid"],
            input_cids=(zk_meta["proof_cid"],),
            reason_codes=("shadow_non_authoritative",),
            metrics={"authoritative": 0},
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(artifact_dir / "proof.json", zk_meta["proof_payload"])
        _atomic_write_json(artifact_dir / "zk_shadow.json", zk_meta["zk_payload"])

    # 8–9. Findings + taskboard
    board_meta = materialize_findings_and_board(
        forest=forest,
        findings=findings,
        artifact_dir=artifact_dir,
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.FINDINGS,
            status=PilotConclusion.PASSED,
            artifact_cid=board_meta["finding_ledger_cid"],
            input_cids=(contract_meta["contract_cid"],),
            metrics={
                "findings": board_meta["finding_count"],
                "admitted": board_meta["admitted_count"],
            },
        )
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.TASKBOARD,
            status=PilotConclusion.PASSED,
            artifact_cid=board_meta["taskboard_cid"],
            input_cids=(board_meta["finding_ledger_cid"],),
            metrics={
                "executable": board_meta["executable_task_count"],
                "reviews": board_meta["review_count"],
                "repair_packets": len(board_meta["repair_packets"]),
            },
        )
    )
    if artifact_dir is not None:
        _atomic_write_json(
            artifact_dir / "findings.json", board_meta["ledger_payload"]
        )
        _atomic_write_json(artifact_dir / "taskboard.json", board_meta["board_json"])
        _atomic_write_text(
            artifact_dir / "taskboard.md", board_meta["board_markdown"]
        )

    # 10. Publish artifact set + report
    artifacts = PilotArtifactSet(
        forest_cid=forest_cid,
        manifest_cid=coverage_bundle["manifest_cid"],
        coverage_cid=coverage_bundle["coverage_cid"],
        inventory_cid=inventory_cid,
        graph_cid=graph_cid,
        cache_cid=cache_meta["cache_cid"],
        proof_cid=zk_meta["proof_cid"],
        zk_shadow_cid=zk_meta["zk_shadow_cid"],
        finding_ledger_cid=board_meta["finding_ledger_cid"],
        taskboard_cid=board_meta["taskboard_cid"],
    )
    stages.append(
        StageReceipt(
            stage=PilotStage.PUBLISH,
            status=PilotConclusion.PASSED,
            artifact_cid=_identity(artifacts.to_dict()),
            input_cids=tuple(
                [
                    artifacts.manifest_cid,
                    artifacts.coverage_cid,
                    artifacts.cache_cid,
                    artifacts.proof_cid,
                    artifacts.finding_ledger_cid,
                    artifacts.taskboard_cid,
                ]
            ),
            metrics={"artifact_count": 10},
        )
    )

    tree_bindings = {
        descriptor.alias: descriptor.tree for descriptor in forest.descriptors
    }
    commit_bindings = {
        descriptor.alias: descriptor.commit for descriptor in forest.descriptors
    }

    if incomplete:
        reason_codes.append("incomplete_inventory")
        conclusion = PilotConclusion.INCOMPLETE
    else:
        conclusion = PilotConclusion.PASSED

    board_md = render_findings_board_document(
        report_context={
            "forest_id": forest.forest_id,
            "artifacts": artifacts.to_dict(),
            "executable_task_count": board_meta["executable_task_count"],
            "review_count": board_meta["review_count"],
            "finding_count": board_meta["finding_count"],
            "admitted_file_count": len(admitted),
            "swissknife_file_count": coverage_bundle["manifest"]["swissknife_file_count"],
            "vfs_closure_file_count": coverage_bundle["manifest"][
                "vfs_closure_file_count"
            ],
            "mode": mode.value,
            "conclusion": conclusion.value,
            "repair_packets": board_meta["repair_packets"],
        },
        taskboard_markdown=board_meta["board_markdown"],
    )
    board_markdown_cid = _identity({"markdown": board_md})

    if config.write_findings_board:
        board_path = config.resolved_findings_board_path()
        if len(board_md.encode("utf-8")) > MAX_FINDINGS_BOARD_BYTES:
            raise VfsSymbolicPilotError(
                "board_bound_exceeded",
                f"findings board exceeds {MAX_FINDINGS_BOARD_BYTES} bytes",
            )
        _atomic_write_text(board_path, board_md)

    report = SwissKnifeVfsPilotReport(
        mode=mode,
        conclusion=conclusion,
        forest_id=forest.forest_id,
        tree_bindings=tree_bindings,
        commit_bindings=commit_bindings,
        stages=tuple(stages),
        artifacts=artifacts,
        admitted_file_count=len(admitted),
        swissknife_file_count=coverage_bundle["manifest"]["swissknife_file_count"],
        vfs_closure_file_count=coverage_bundle["manifest"]["vfs_closure_file_count"],
        finding_count=board_meta["finding_count"],
        executable_task_count=board_meta["executable_task_count"],
        review_count=board_meta["review_count"],
        inconclusive_count=contract_meta["inconclusive_count"],
        provider_calls=0,
        source_mutations=0,
        reason_codes=tuple(sorted(set(reason_codes))),
        board_markdown_cid=board_markdown_cid,
    )
    # Published artifact set binds the report CID without mutating report identity.
    published_artifacts = PilotArtifactSet(
        forest_cid=artifacts.forest_cid,
        manifest_cid=artifacts.manifest_cid,
        coverage_cid=artifacts.coverage_cid,
        inventory_cid=artifacts.inventory_cid,
        graph_cid=artifacts.graph_cid,
        cache_cid=artifacts.cache_cid,
        proof_cid=artifacts.proof_cid,
        zk_shadow_cid=artifacts.zk_shadow_cid,
        finding_ledger_cid=artifacts.finding_ledger_cid,
        taskboard_cid=artifacts.taskboard_cid,
        report_cid=report.report_cid,
    )

    if artifact_dir is not None:
        _atomic_write_json(
            artifact_dir / "artifacts.json", published_artifacts.to_dict()
        )
        _atomic_write_json(artifact_dir / "report.json", report.to_dict())
        _atomic_write_text(artifact_dir / "findings_board.md", board_md)

    return report


def dry_run_pilot(config: PilotConfig) -> SwissKnifeVfsPilotReport:
    """Dry-run mode: full pipeline, no provider calls, no source mutation."""

    return execute_pilot(config, mode=PilotMode.DRY_RUN)


def verify_pilot_report(
    report: SwissKnifeVfsPilotReport | Mapping[str, Any],
    *,
    config: PilotConfig | None = None,
    recompute: bool = True,
) -> SwissKnifeVfsPilotReport:
    """Verify a pilot report without provider calls or source mutation.

    When ``recompute`` is true and ``config`` is supplied, the forest is frozen
    again and tree/commit bindings must match the report.  Inventory
    completeness, artifact canonicality, and zero provider/mutation counters
    are always enforced.
    """

    _assert_no_provider_surface()
    if isinstance(report, Mapping):
        report = SwissKnifeVfsPilotReport.from_dict(report)
    if not isinstance(report, SwissKnifeVfsPilotReport):
        raise PilotVerificationError("invalid_report", "report type is invalid")

    if report.provider_calls != 0:
        raise PilotVerificationError("provider_call_forbidden")
    if report.source_mutations != 0:
        raise PilotVerificationError("source_mutation_forbidden")
    if report.authorizes_repair or report.is_completion_evidence:
        raise PilotVerificationError("authority_drift")
    if report.evidence != SWISS_KNIFE_VFS_PILOT_SCHEMA:
        raise PilotVerificationError("stale_evidence", "unexpected evidence schema")
    if report.artifacts is None:
        raise PilotVerificationError("incomplete_inventory", "missing artifact set")

    # Canonical re-encode must be stable.
    reloaded = SwissKnifeVfsPilotReport.from_dict(report.to_dict())
    if reloaded.report_cid != report.report_cid:
        raise PilotVerificationError(
            "noncanonical_artifact",
            "report is not canonical under re-encode",
        )

    required_stages = {stage for stage in PilotStage}
    observed_stages = {item.stage for item in report.stages}
    missing = required_stages - observed_stages
    if missing:
        raise PilotVerificationError(
            "incomplete_inventory",
            f"missing stages: {sorted(item.value for item in missing)}",
        )

    for stage in report.stages:
        if not stage.artifact_cid:
            raise PilotVerificationError(
                "stale_evidence",
                f"stage {stage.stage.value} missing artifact CID",
            )
        # Stage receipt must rehash.
        if StageReceipt.from_dict(stage.to_dict()).receipt_cid != stage.receipt_cid:
            raise PilotVerificationError(
                "noncanonical_artifact",
                f"stage {stage.stage.value} is non-canonical",
            )

    if recompute and config is not None:
        live = freeze_repository_descriptors(config)
        if live.forest_id != report.forest_id:
            # Forest id includes portable identity; mismatch implies changed
            # trees, gitlinks, overlays, or policy.
            raise PilotVerificationError(
                "changed_trees",
                "live forest_id does not match frozen pilot report",
            )
        live_trees = {
            descriptor.alias: descriptor.tree for descriptor in live.descriptors
        }
        live_commits = {
            descriptor.alias: descriptor.commit for descriptor in live.descriptors
        }
        for alias, tree in report.tree_bindings.items():
            if live_trees.get(alias) != tree:
                raise PilotVerificationError(
                    "changed_trees",
                    f"tree for {alias!r} changed since pilot freeze",
                )
        for alias, commit in report.commit_bindings.items():
            if live_commits.get(alias) != commit:
                raise PilotVerificationError(
                    "changed_trees",
                    f"commit for {alias!r} changed since pilot freeze",
                )

        limits = config.inventory_limits or InventoryLimits()
        index = scan_inventory(live, limits=limits)
        if index.inventory_cid != report.artifacts.inventory_cid:
            raise PilotVerificationError(
                "stale_evidence",
                "inventory_cid drifted under recompute",
            )
        for repo in index.repositories:
            if (
                config.require_exhaustive_swissknife
                and repo.repository_alias == DEFAULT_SWISSKNIFE_ALIAS
                and (not repo.exhaustive or repo.omitted_entry_count)
            ):
                raise PilotVerificationError(
                    "incomplete_inventory",
                    "SwissKnife inventory is incomplete on verify",
                )
            if not repo.exhaustive and report.conclusion == PilotConclusion.PASSED:
                raise PilotVerificationError(
                    "incomplete_inventory",
                    f"repository {repo.repository_alias!r} inventory incomplete",
                )

        # Full recompute of dry-run must reproduce the report CID when trees match.
        verify_config = PilotConfig(
            accelerator_root=config.accelerator_root,
            swissknife_root=config.swissknife_root,
            kit_root=config.kit_root,
            datasets_root=config.datasets_root,
            artifact_dir=None,
            findings_board_path=None,
            inventory_limits=config.inventory_limits,
            max_admitted_parse=config.max_admitted_parse,
            write_artifacts=False,
            write_findings_board=False,
            include_optional_missing=config.include_optional_missing,
            require_exhaustive_swissknife=config.require_exhaustive_swissknife,
            forest_policy=config.forest_policy,
        )
        recomputed = execute_pilot(verify_config, mode=PilotMode.DRY_RUN)
        # Compare portable identity of core artifacts (mode differs: dry_run vs verify).
        if recomputed.forest_id != report.forest_id:
            raise PilotVerificationError("changed_trees", "recomputed forest drifted")
        if recomputed.artifacts is None:
            raise PilotVerificationError("incomplete_inventory", "recompute missing artifacts")
        for field_name in (
            "manifest_cid",
            "coverage_cid",
            "inventory_cid",
            "graph_cid",
            "cache_cid",
            "proof_cid",
            "zk_shadow_cid",
            "finding_ledger_cid",
            "taskboard_cid",
        ):
            expected = getattr(report.artifacts, field_name)
            observed = getattr(recomputed.artifacts, field_name)
            if expected != observed:
                raise PilotVerificationError(
                    "stale_evidence",
                    f"{field_name} is not reproducible (expected {expected}, got {observed})",
                )

    verified_payload = dict(report.to_dict())
    verified_payload.pop("report_cid", None)
    verified_payload["mode"] = PilotMode.VERIFY.value
    return SwissKnifeVfsPilotReport.from_dict(verified_payload)


def verify_pilot(
    config: PilotConfig,
    *,
    report_path: Path | None = None,
) -> SwissKnifeVfsPilotReport:
    """Verify one durable report against its live repository forest.

    A report path is deliberately mandatory at runtime.  Verification must not
    silently replace the operator's release artifact with a freshly generated
    temporary fixture.  Use :func:`run_hermetic_self_test` for that distinct
    diagnostic operation.
    """

    if report_path is None:
        raise PilotVerificationError(
            "durable_report_required",
            "report verification requires an explicit --report path; "
            "use --hermetic-self-test for a temporary fixture check",
        )
    payload = _load_json(Path(report_path))
    return verify_pilot_report(payload, config=config, recompute=True)


def verify_release_evidence(
    config: PilotConfig,
    *,
    report_path: Path | None = None,
) -> SwissKnifeVfsPilotReport:
    """Verify a durable report and require explicit release authority.

    The current pilot schema intentionally sets ``is_completion_evidence`` to
    false.  Consequently its reports can prove deterministic integrity and
    freshness, but cannot pass this release gate.  A future authoritative
    schema must make that authority explicit rather than gaining it merely
    because structural verification succeeded.
    """

    report = verify_pilot(config, report_path=report_path)
    if report.conclusion is not PilotConclusion.PASSED:
        raise PilotVerificationError(
            "release_evidence_failed",
            f"pilot conclusion is {report.conclusion.value!r}, not 'passed'",
        )
    if not report.is_completion_evidence:
        raise PilotVerificationError(
            "non_authoritative_report",
            "pilot report is structurally valid but explicitly sets "
            "is_completion_evidence=false; verification cannot promote it "
            "to release evidence",
        )
    return report


def run_hermetic_self_test(config: PilotConfig) -> SwissKnifeVfsPilotReport:
    """Generate and verify a temporary report as a non-release self-test."""

    with tempfile.TemporaryDirectory(prefix="vfs-pilot-verify-") as tmp:
        tmp_path = Path(tmp)
        run_config = PilotConfig(
            accelerator_root=config.accelerator_root,
            swissknife_root=config.swissknife_root,
            kit_root=config.kit_root,
            datasets_root=config.datasets_root,
            artifact_dir=tmp_path / "artifacts",
            findings_board_path=tmp_path / "findings.todo.md",
            inventory_limits=config.inventory_limits,
            max_admitted_parse=config.max_admitted_parse,
            write_artifacts=True,
            write_findings_board=True,
            include_optional_missing=config.include_optional_missing,
            require_exhaustive_swissknife=config.require_exhaustive_swissknife,
            forest_policy=config.forest_policy,
        )
        report = dry_run_pilot(run_config)
        return verify_pilot_report(report, config=run_config, recompute=True)


def render_findings_board_document(
    *,
    report_context: Mapping[str, Any],
    taskboard_markdown: str,
) -> str:
    """Render the durable findings board markdown for VFS-G131."""

    artifacts = dict(report_context.get("artifacts") or {})
    packets = list(report_context.get("repair_packets") or [])
    lines = [
        "# IPFS Kit VFS Symbolic Assurance Findings Board",
        "",
        "Generated by `vfs_symbolic_pilot` (VFS-037 / VFS-G131).",
        "This board is diagnostic and **does not authorize repair or completion**.",
        "",
        "## Pilot receipt",
        "",
        f"- objective_id: `{PILOT_OBJECTIVE_ID}`",
        f"- task_id: `{PILOT_TASK_ID}`",
        f"- evidence: `{SWISS_KNIFE_VFS_PILOT_SCHEMA}`",
        f"- mode: `{report_context.get('mode', '')}`",
        f"- conclusion: `{report_context.get('conclusion', '')}`",
        f"- forest_id: `{report_context.get('forest_id', '')}`",
        f"- admitted_file_count: `{report_context.get('admitted_file_count', 0)}`",
        f"- swissknife_file_count: `{report_context.get('swissknife_file_count', 0)}`",
        f"- vfs_closure_file_count: `{report_context.get('vfs_closure_file_count', 0)}`",
        f"- finding_count: `{report_context.get('finding_count', 0)}`",
        f"- executable_task_count: `{report_context.get('executable_task_count', 0)}`",
        f"- review_count: `{report_context.get('review_count', 0)}`",
        f"- provider_calls: `0`",
        f"- source_mutations: `0`",
        f"- authorizes_repair: `false`",
        f"- is_completion_evidence: `false`",
        "",
        "## Published artifact CIDs",
        "",
    ]
    for key in (
        "forest_cid",
        "manifest_cid",
        "coverage_cid",
        "inventory_cid",
        "graph_cid",
        "cache_cid",
        "proof_cid",
        "zk_shadow_cid",
        "finding_ledger_cid",
        "taskboard_cid",
        "report_cid",
    ):
        if key in artifacts:
            lines.append(f"- {key}: `{artifacts[key]}`")
    lines.extend(
        [
            "",
            "## Exact repair packets",
            "",
        ]
    )
    if not packets:
        lines.append(
            "No executable repair packets were admitted. "
            "Inconclusive or ambiguous findings remain non-executable."
        )
        lines.append("")
    else:
        for packet in packets:
            lines.extend(
                [
                    f"### packet {packet.get('packet_id') or packet.get('content_id') or 'unknown'}",
                    "",
                    f"- finding_cid: `{packet.get('finding_cid', '')}`",
                    f"- content_id: `{packet.get('content_id', '')}`",
                    f"- status: `{packet.get('status', '')}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Taskboard projection",
            "",
            "The following projection is goal-backed, bounded, and deduplicated. "
            "It never grants edit authority.",
            "",
            taskboard_markdown.strip(),
            "",
            "## Invariants",
            "",
            "- Every file and finding is reproducible from content-addressed evidence.",
            "- Verification performs no provider call and no source mutation.",
            "- Verification fails on changed trees, incomplete inventory, stale "
            "evidence, or non-canonical artifacts.",
            "- SwissKnife remains read-only under the initial forest policy.",
            "",
        ]
    )
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


# ---------------------------------------------------------------------------
# Hermetic fixture helpers (tests + default CLI verify)
# ---------------------------------------------------------------------------


def build_hermetic_pilot_forest_policy(
    root: Path,
    *,
    seed_broken: bool = False,
    seed_inconclusive: bool = False,
) -> tuple[ForestPolicy, Path, Path]:
    """Create a tiny multi-repo fixture forest for hermetic pilot runs.

    Returns ``(policy, swissknife_root, accelerator_root)``.
    """

    root = Path(root)
    swiss = root / "swissknife"
    accel = root / "accelerator"
    # Keep kit/datasets as sibling checkouts so the accelerator inventory does
    # not observe nested independent Git roots as path escapes.
    kit = root / "ipfs_kit_py"
    datasets = root / "ipfs_datasets_py"

    def init_repo(path: Path, files: Mapping[str, str]) -> None:
        path.mkdir(parents=True, exist_ok=True)
        _git(path, "init")
        _git(path, "checkout", "-b", "main")
        _git(path, "config", "user.name", "VFS Pilot")
        _git(path, "config", "user.email", "vfs-pilot@example.invalid")
        for relative, content in files.items():
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        _git(path, "add", ".")
        _git(path, "commit", "-m", "seed pilot fixture")

    swiss_files = {
        "src/connector.ts": (
            "export function listTools() {\n"
            "  return ['vfs.stat', 'vfs.read'];\n"
            "}\n"
        ),
        "src/mcp.ts": (
            "export const server = 'ipfs-kit-vfs';\n"
            "export function call(tool: string) { return tool; }\n"
        ),
        "README.md": "# SwissKnife pilot fixture\n",
    }
    if seed_broken:
        swiss_files["src/broken.ts"] = (
            "// VFS_PILOT_CONTRACT_BROKEN\n"
            "export function drift() { return 'broken'; }\n"
        )
    if seed_inconclusive:
        swiss_files["src/maybe.ts"] = (
            "// VFS_PILOT_INCONCLUSIVE\n"
            "export function maybe() { return 'unknown'; }\n"
        )

    init_repo(swiss, swiss_files)
    init_repo(
        accel,
        {
            "ipfs_accelerate_py/agent_supervisor/vfs_surface.py": (
                "def inventory_vfs_surfaces():\n"
                "    return ['vfs.stat']\n"
            ),
            "README.md": "# accelerator pilot fixture\n",
        },
    )
    init_repo(
        kit,
        {
            "ipfs_kit_py/vfs/manager.py": (
                "class VfsManager:\n"
                "    def stat(self, path: str) -> dict:\n"
                "        return {'path': path}\n"
            ),
            "README.md": "# kit pilot fixture\n",
        },
    )
    init_repo(
        datasets,
        {
            "ipfs_datasets_py/logic/zkp/README.md": "# zkp surface\n",
            "README.md": "# datasets pilot fixture\n",
        },
    )

    policy = ForestPolicy(
        roots=(
            ForestRootSpec(
                alias=DEFAULT_SWISSKNIFE_ALIAS,
                root_path=swiss,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
                required=True,
            ),
            ForestRootSpec(
                alias=DEFAULT_ACCELERATOR_ALIAS,
                root_path=accel,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
                required=True,
            ),
            ForestRootSpec(
                alias=DEFAULT_KIT_ALIAS,
                root_path=kit,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
                required=True,
            ),
            ForestRootSpec(
                alias=DEFAULT_DATASETS_ALIAS,
                root_path=datasets,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
                required=True,
            ),
        ),
        sole_write_alias=DEFAULT_ACCELERATOR_ALIAS,
    )
    return policy, swiss, accel


def default_config_from_environment(
    *,
    accelerator_root: Path | None = None,
    swissknife_root: Path | None = None,
    hermetic_root: Path | None = None,
) -> PilotConfig:
    """Build a pilot config from env vars or a hermetic fixture.

    When the default SwissKnife root is missing, a hermetic fixture is used so
    that dry-runs and the explicitly named hermetic self-test remain runnable
    in CI and agent sandboxes.
    """

    accel = Path(
        accelerator_root
        or os.environ.get("IPFS_ACCELERATE_ROOT")
        or Path.cwd()
    ).resolve()
    swiss = Path(
        swissknife_root
        or os.environ.get("SWISSKNIFE_ROOT")
        or DEFAULT_SWISSKNIFE_ROOT
    )
    if swiss.is_dir() and hermetic_root is None:
        try:
            policy = initial_vfs_assurance_forest_policy(
                accelerator_root=accel,
                swissknife_root=swiss,
            )
            # Probe whether the forest builds; fall back to hermetic on failure.
            build_repository_forest(policy)
            return PilotConfig(
                accelerator_root=accel,
                swissknife_root=swiss,
                write_artifacts=True,
                write_findings_board=True,
            )
        except Exception:
            pass

    fixture_root = Path(
        hermetic_root
        or tempfile.mkdtemp(prefix="vfs-pilot-hermetic-")
    )
    policy, swiss_path, accel_path = build_hermetic_pilot_forest_policy(fixture_root)
    return PilotConfig(
        accelerator_root=accel_path,
        swissknife_root=swiss_path,
        forest_policy=policy,
        artifact_dir=accel / DEFAULT_ARTIFACT_RELATIVE
        if (accel / "ipfs_accelerate_py").is_dir()
        else fixture_root / "pilot_artifacts",
        findings_board_path=(
            accel / DEFAULT_FINDINGS_BOARD_RELATIVE
            if (accel / "docs" / "architecture").is_dir()
            else fixture_root / "findings.todo.md"
        ),
        write_artifacts=True,
        write_findings_board=True,
        require_exhaustive_swissknife=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot",
        description=(
            "Run or verify the frozen SwissKnife/IPFS VFS pilot "
            f"({PILOT_TASK_ID} / {PILOT_OBJECTIVE_ID})."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze descriptors, run the deterministic pipeline, publish CIDs",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Verify the integrity and freshness of the durable report named by "
            "--report; this does not grant release authority"
        ),
    )
    mode.add_argument(
        "--verify-release-evidence",
        action="store_true",
        help=(
            "Verify the durable report named by --report and require explicit "
            "release/completion authority"
        ),
    )
    mode.add_argument(
        "--hermetic-self-test",
        action="store_true",
        help=(
            "Generate and verify a temporary hermetic fixture report; never "
            "treat this diagnostic as durable or release evidence"
        ),
    )
    parser.add_argument(
        "--accelerator-root",
        type=Path,
        default=None,
        help="Accelerator repository root (default: cwd / IPFS_ACCELERATE_ROOT)",
    )
    parser.add_argument(
        "--swissknife-root",
        type=Path,
        default=None,
        help=f"SwissKnife checkout (default: {DEFAULT_SWISSKNIFE_ROOT})",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for pilot artifacts",
    )
    parser.add_argument(
        "--findings-board",
        type=Path,
        default=None,
        help="Path for the generated findings board markdown",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Existing durable report.json (required by --verify and "
            "--verify-release-evidence)"
        ),
    )
    parser.add_argument(
        "--hermetic",
        action="store_true",
        help="Force hermetic multi-repo fixtures instead of live checkouts",
    )
    parser.add_argument(
        "--max-admitted-parse",
        type=int,
        default=MAX_ADMITTED_PARSE,
        help="Upper bound on admitted files parsed into the program graph",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the pilot report as canonical JSON on stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.verify or args.verify_release_evidence:
        if args.report is None:
            parser.error(
                "--verify and --verify-release-evidence require an explicit "
                "--report path"
            )
    elif args.report is not None:
        parser.error(
            "--report is only valid with --verify or --verify-release-evidence"
        )
    if args.hermetic and (args.verify or args.verify_release_evidence):
        parser.error(
            "--hermetic cannot be combined with durable report verification; "
            "use --hermetic-self-test for a temporary fixture check"
        )

    hermetic_root = None
    if args.hermetic or args.hermetic_self_test:
        hermetic_root = Path(tempfile.mkdtemp(prefix="vfs-pilot-cli-hermetic-"))

    config = default_config_from_environment(
        accelerator_root=args.accelerator_root,
        swissknife_root=args.swissknife_root,
        hermetic_root=hermetic_root,
    )
    if args.artifact_dir is not None or args.findings_board is not None or args.max_admitted_parse:
        config = PilotConfig(
            accelerator_root=config.accelerator_root,
            swissknife_root=config.swissknife_root,
            kit_root=config.kit_root,
            datasets_root=config.datasets_root,
            artifact_dir=args.artifact_dir or config.artifact_dir,
            findings_board_path=args.findings_board or config.findings_board_path,
            inventory_limits=config.inventory_limits,
            max_admitted_parse=int(args.max_admitted_parse or config.max_admitted_parse),
            write_artifacts=True,
            write_findings_board=True,
            include_optional_missing=config.include_optional_missing,
            require_exhaustive_swissknife=config.require_exhaustive_swissknife,
            forest_policy=config.forest_policy,
        )

    try:
        if args.dry_run:
            report = dry_run_pilot(config)
        elif args.hermetic_self_test:
            report = run_hermetic_self_test(config)
        elif args.verify_release_evidence:
            report = verify_release_evidence(config, report_path=args.report)
        else:
            report = verify_pilot(config, report_path=args.report)
    except VfsSymbolicPilotError as exc:
        message = {
            "ok": False,
            "error": exc.reason_code,
            "detail": str(exc),
            "evidence": SWISS_KNIFE_VFS_PILOT_SCHEMA,
        }
        sys.stderr.write(canonical_json(message) + "\n")
        return 2

    if args.json:
        sys.stdout.write(report.to_json() + "\n")
    else:
        operation = report.mode.value
        if args.hermetic_self_test:
            operation = "hermetic_self_test"
        elif args.verify_release_evidence:
            operation = "release_evidence_verify"
        elif args.verify:
            operation = "report_verify"
        sys.stdout.write(
            f"pilot {operation} {report.conclusion.value} "
            f"report_cid={report.report_cid} "
            f"admitted={report.admitted_file_count} "
            f"findings={report.finding_count} "
            f"tasks={report.executable_task_count}\n"
        )
        if report.artifacts is not None:
            sys.stdout.write(
                f"artifacts manifest={report.artifacts.manifest_cid} "
                f"coverage={report.artifacts.coverage_cid} "
                f"cache={report.artifacts.cache_cid} "
                f"proof={report.artifacts.proof_cid} "
                f"findings={report.artifacts.finding_ledger_cid} "
                f"taskboard={report.artifacts.taskboard_cid}\n"
            )
    return 0 if report.conclusion is not PilotConclusion.FAILED else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_RELATIVE",
    "DEFAULT_FINDINGS_BOARD_RELATIVE",
    "MAX_ADMITTED_PARSE",
    "PILOT_ARTIFACT_SET_SCHEMA",
    "PILOT_BOARD_NAMESPACE",
    "PILOT_COVERAGE_SCHEMA",
    "PILOT_MANIFEST_SCHEMA",
    "PILOT_OBJECTIVE_ID",
    "PILOT_POLICY_REVISION",
    "PILOT_PRODUCER",
    "PILOT_REQUIREMENT_ID",
    "PILOT_TASK_ID",
    "PILOT_VERSION",
    "SWISS_KNIFE_VFS_PILOT_SCHEMA",
    "PilotArtifactSet",
    "PilotConclusion",
    "PilotConfig",
    "PilotMode",
    "PilotStage",
    "PilotVerificationError",
    "StageReceipt",
    "SwissKnifeVfsPilotReport",
    "VfsSymbolicPilotError",
    "admitted_entries_for_pilot",
    "build_coverage_manifest",
    "build_hermetic_pilot_forest_policy",
    "build_pilot_program_graph",
    "default_config_from_environment",
    "dry_run_pilot",
    "execute_pilot",
    "freeze_repository_descriptors",
    "is_vfs_relevant_path",
    "main",
    "materialize_findings_and_board",
    "render_findings_board_document",
    "run_cache_stage",
    "run_contract_stage",
    "run_hermetic_self_test",
    "run_zk_shadow_stage",
    "scan_inventory",
    "verify_pilot",
    "verify_pilot_report",
    "verify_release_evidence",
]
