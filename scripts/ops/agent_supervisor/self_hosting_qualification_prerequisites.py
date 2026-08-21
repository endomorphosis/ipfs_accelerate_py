#!/usr/bin/env python3
"""Read-only prerequisite-state observer for self-hosting qualification (SHQ-G006).

Observes the ten named prerequisite systems and records current repository,
commit, API mapping, focused-test selector, board state, and evidence time.

This module is intentionally non-authoritative:

* ordinary observation always succeeds with an honest snapshot, even when
  upstream systems are incomplete;
* ``require-terminal`` mode fails closed unless every row is terminal;
* no row claims release from prompt text or branch name alone;
* versioned functional interfaces (for example ``ContextPacker`` /
  ``pack_context`` standing in for planned ``ContextPackBuilder``) are
  recognized only through the explicit compatibility map — missing facade
  classes are never manufactured.

Interfaces: ``observe_prerequisite_releases``, ``PrerequisiteObservation``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

INTERFACE_ID: Final[str] = "PrerequisiteObservation@1"
OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "self-hosting-qualification-prerequisite-observation@1"
)
OBSERVATION_VERSION: Final[str] = "1"
GOAL_ID: Final[str] = "SHQ-G006"
BUNDLE_ID: Final[str] = "agent-supervisor/self-hosting/prerequisite-observer"

# Observation never authorizes mutation, completion, or release admission.
OBSERVATION_IS_COMPLETION_EVIDENCE: Final[bool] = False
OBSERVATION_IS_PROOF_EVIDENCE: Final[bool] = False
OBSERVATION_AUTHORIZES_MUTATION: Final[bool] = False
OBSERVATION_AUTHORIZES_RELEASE: Final[bool] = False

DEFAULT_OUTPUT_RELATIVE: Final[str] = (
    "artifacts/agent_supervisor/self_hosting_qualification/"
    "prerequisite_observation.json"
)

_STATUS_LINE_RE = re.compile(
    r"^- Status:\s*(?P<status>completed|todo|blocked|in_progress|review|cancelled)\s*$",
    re.IGNORECASE,
)
_TASK_HEADING_RE = re.compile(r"^##\s+(?P<task_id>[A-Z]+-\d+)\b")


class ObservationMode(str, Enum):
    """CLI / API observation modes."""

    OBSERVE = "observe"
    REQUIRE_TERMINAL = "require-terminal"


class PrerequisiteStatus(str, Enum):
    """Closed taxonomy for one prerequisite row."""

    RELEASED = "released"
    IN_FLIGHT = "in-flight"
    MISSING = "missing"
    MISMATCHED_NAME = "mismatched-name"
    UNVERIFIABLE = "unverifiable"


class BoardState(str, Enum):
    """Closed taxonomy for owner-board observation."""

    TERMINAL = "terminal"
    IN_FLIGHT = "in-flight"
    ABSENT = "absent"
    NOT_CONFIGURED = "not_configured"
    UNREADABLE = "unreadable"


class ApiResolution(str, Enum):
    """How a public API surface was resolved."""

    EXACT = "exact"
    COMPATIBILITY_MAP = "compatibility_map"
    ABSENT = "absent"
    UNVERIFIABLE = "unverifiable"


# ---------------------------------------------------------------------------
# Compatibility map (explicit; do not invent facades)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompatibilityEntry:
    """One planned-name → versioned implementation mapping."""

    planned_name: str
    implementation_symbols: tuple[str, ...]
    interface_ids: tuple[str, ...]
    module_paths: tuple[str, ...]
    rationale: str


# Planned names that must never be synthesized as empty facade classes.
COMPATIBILITY_MAP: Final[tuple[CompatibilityEntry, ...]] = (
    CompatibilityEntry(
        planned_name="ContextPackBuilder",
        implementation_symbols=("ContextPacker", "pack_context"),
        interface_ids=("ContextPack@1",),
        module_paths=(
            "ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py",
        ),
        rationale=(
            "Existing implementation is named ContextPacker / pack_context "
            "with versioned interface ContextPack@1; admit the mapping rather "
            "than manufacturing a ContextPackBuilder facade."
        ),
    ),
    CompatibilityEntry(
        planned_name="SemanticCapsuleCompiler",
        implementation_symbols=(
            "compile_semantic_capsule",
            "compile_semantic_capsules",
            "SEMANTIC_CAPSULE_COMPILER_INTERFACE",
        ),
        interface_ids=("SemanticCapsuleCompiler@1",),
        module_paths=(
            "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/"
            "semantic_state/capsules.py",
        ),
        rationale=(
            "Functional @1 API is exported via compile_semantic_capsule* and "
            "SEMANTIC_CAPSULE_COMPILER_INTERFACE; no class facade is required."
        ),
    ),
)


def compatibility_map_as_dict() -> list[dict[str, Any]]:
    """Public projection of the explicit compatibility map."""

    return [
        {
            "planned_name": entry.planned_name,
            "implementation_symbols": list(entry.implementation_symbols),
            "interface_ids": list(entry.interface_ids),
            "module_paths": list(entry.module_paths),
            "rationale": entry.rationale,
        }
        for entry in COMPATIBILITY_MAP
    ]


def _compat_for(planned_name: str) -> CompatibilityEntry | None:
    for entry in COMPATIBILITY_MAP:
        if entry.planned_name == planned_name:
            return entry
    return None


# ---------------------------------------------------------------------------
# Prerequisite catalog (exactly ten systems)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrerequisiteSpec:
    """Static catalog entry for one named prerequisite system."""

    prerequisite_id: str
    repository: str
    repository_root_relative: str
    module_paths: tuple[str, ...]
    expected_symbols: tuple[str, ...]
    interface_ids: tuple[str, ...]
    test_selectors: tuple[str, ...]
    board_paths: tuple[str, ...]
    notes: str = ""


# Order is stable and matches the plan's ten-prerequisite table.
PREREQUISITE_CATALOG: Final[tuple[PrerequisiteSpec, ...]] = (
    PrerequisiteSpec(
        prerequisite_id="IncrementalSemanticIndex",
        repository="ipfs_datasets_py",
        repository_root_relative="ipfs_datasets_py",
        module_paths=(
            "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/"
            "semantic_index/index.py",
        ),
        expected_symbols=("IncrementalSemanticIndex", "scan_repository"),
        interface_ids=(),
        test_selectors=(
            "ipfs_datasets_py/tests/unit/logic/software_contracts/"
            "semantic_index/test_api.py",
        ),
        board_paths=(),
        notes="Datasets-owned incremental semantic index facade.",
    ),
    PrerequisiteSpec(
        prerequisite_id="SemanticCapsuleCompiler",
        repository="ipfs_datasets_py",
        repository_root_relative="ipfs_datasets_py",
        module_paths=(
            "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/"
            "semantic_state/capsules.py",
        ),
        expected_symbols=(
            "compile_semantic_capsule",
            "compile_semantic_capsules",
            "SEMANTIC_CAPSULE_COMPILER_INTERFACE",
        ),
        interface_ids=("SemanticCapsuleCompiler@1",),
        test_selectors=(
            "ipfs_datasets_py/tests/unit/logic/software_contracts/"
            "semantic_state/test_capsules.py",
        ),
        board_paths=(),
        notes=(
            "Versioned functional @1 API; the explicit compatibility map "
            "documents that no cosmetic class facade is required."
        ),
    ),
    PrerequisiteSpec(
        prerequisite_id="ContextPackBuilder",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(
            "ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py",
        ),
        expected_symbols=("ContextPackBuilder",),
        interface_ids=("ContextPack@1",),
        test_selectors=("test/api/semantic_state/test_context_pack.py",),
        board_paths=("docs/architecture/semantic_compression_harness.todo.md",),
        notes=(
            "Planned name ContextPackBuilder maps explicitly to ContextPacker "
            "/ pack_context; do not manufacture a facade class."
        ),
    ),
    PrerequisiteSpec(
        prerequisite_id="VerificationReceiptCache",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(
            "ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py",
        ),
        expected_symbols=("VerificationReceiptCache",),
        interface_ids=("VerificationReceiptCache@1",),
        test_selectors=(
            "test/api/test_agent_supervisor_verification_receipt_cache.py",
        ),
        board_paths=(
            "docs/architecture/incremental_verification_planner.todo.md",
        ),
        notes="Owned by the incremental verification planner board.",
    ),
    PrerequisiteSpec(
        prerequisite_id="IncrementalVerificationPlanner",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(
            "ipfs_accelerate_py/agent_supervisor/verification/planner.py",
        ),
        expected_symbols=("IncrementalVerificationPlanner",),
        interface_ids=("IncrementalVerificationPlanner@1",),
        test_selectors=(
            "test/api/test_agent_supervisor_incremental_verification_planner.py",
        ),
        board_paths=(
            "docs/architecture/incremental_verification_planner.todo.md",
        ),
        notes="Owned by the incremental verification planner board.",
    ),
    PrerequisiteSpec(
        prerequisite_id="ModelRoutePlanner",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(
            "ipfs_accelerate_py/agent_supervisor/verification/model_route.py",
        ),
        expected_symbols=("ModelRoutePlanner",),
        interface_ids=("ModelRoutePlanner@1",),
        test_selectors=(
            "test/api/test_agent_supervisor_verification_model_route.py",
        ),
        board_paths=(
            "docs/architecture/incremental_verification_planner.todo.md",
        ),
        notes="Owned by the incremental verification planner board.",
    ),
    PrerequisiteSpec(
        prerequisite_id="VerifiedGuiOptimizer",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(),
        expected_symbols=("VerifiedGuiOptimizer",),
        interface_ids=(),
        test_selectors=(),
        board_paths=(),
        notes=(
            "No released public symbol or terminal owner board bound in this "
            "tree; observation must stay non-authoritative."
        ),
    ),
    PrerequisiteSpec(
        prerequisite_id="IncrementalProofSealer",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(),
        expected_symbols=("IncrementalProofSealer",),
        interface_ids=(),
        test_selectors=(),
        board_paths=(),
        notes=(
            "No released public symbol or terminal owner board bound in this "
            "tree; observation must stay non-authoritative."
        ),
    ),
    PrerequisiteSpec(
        prerequisite_id="SemanticCompressionGovernor",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(
            "ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py",
            "docs/architecture/semantic_compression_governor.objectives.md",
        ),
        expected_symbols=("SemanticCompressionGovernor",),
        interface_ids=("SemanticCompressionGovernor@1",),
        test_selectors=("test/api/test_semantic_compression_governor_board.py",),
        board_paths=("docs/architecture/semantic_compression_governor.todo.md",),
        notes=(
            "Owning supervisor board remains the release authority; harness "
            "adjacent symbols do not upgrade this system to released."
        ),
    ),
    PrerequisiteSpec(
        prerequisite_id="AdversarialAssuranceEngine",
        repository="ipfs_accelerate_py",
        repository_root_relative=".",
        module_paths=(),
        expected_symbols=("AdversarialAssuranceEngine",),
        interface_ids=(),
        test_selectors=(),
        board_paths=(),
        notes=(
            "No exact released symbol or terminal board found; do not "
            "substitute a new engine."
        ),
    ),
)


def prerequisite_catalog() -> tuple[PrerequisiteSpec, ...]:
    """Return the frozen ten-prerequisite catalog."""

    return PREREQUISITE_CATALOG


# ---------------------------------------------------------------------------
# Observation models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolHit:
    """One AST-level definition or assignment of a searched name."""

    symbol: str
    path: str
    line: int
    kind: str


@dataclass(frozen=True, slots=True)
class BoardObservation:
    """Parsed owner-board lifecycle counters."""

    paths: tuple[str, ...]
    state: str
    task_count: int
    completed: int
    open: int
    blocked: int
    other: int
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": list(self.paths),
            "state": self.state,
            "task_count": self.task_count,
            "completed": self.completed,
            "open": self.open,
            "blocked": self.blocked,
            "other": self.other,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ApiObservation:
    """API / compatibility resolution for one prerequisite."""

    resolution: str
    expected_symbols: tuple[str, ...]
    interface_ids: tuple[str, ...]
    found_symbols: tuple[str, ...]
    hits: tuple[SymbolHit, ...]
    compatibility: Mapping[str, Any] | None
    module_paths: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "resolved": self.resolution
            in {
                ApiResolution.EXACT.value,
                ApiResolution.COMPATIBILITY_MAP.value,
            },
            "expected_symbols": list(self.expected_symbols),
            "interface_ids": list(self.interface_ids),
            "found_symbols": list(self.found_symbols),
            "hits": [
                {
                    "symbol": hit.symbol,
                    "path": hit.path,
                    "line": hit.line,
                    "kind": hit.kind,
                }
                for hit in self.hits
            ],
            "compatibility": (
                dict(self.compatibility) if self.compatibility is not None else None
            ),
            "module_paths": list(self.module_paths),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class TestObservation:
    """Focused test selector presence (not execution)."""

    selectors: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]
    all_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectors": list(self.selectors),
            "present": list(self.present),
            "missing": list(self.missing),
            "all_present": self.all_present,
        }


@dataclass(frozen=True, slots=True)
class PrerequisiteRow:
    """One bound observation row."""

    prerequisite_id: str
    repository: str
    commit: str | None
    status: str
    api: ApiObservation
    tests: TestObservation
    board: BoardObservation
    evidence_time: str
    notes: str
    limitations: tuple[str, ...] = ()
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "prerequisite_id": self.prerequisite_id,
            "repository": self.repository,
            "commit": self.commit,
            "status": self.status,
            "terminal": self.terminal,
            "api": self.api.to_dict(),
            "tests": self.tests.to_dict(),
            "board": self.board.to_dict(),
            "evidence_time": self.evidence_time,
            "notes": self.notes,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class PrerequisiteObservation:
    """Aggregate observation envelope for all ten prerequisites."""

    schema: str
    interface: str
    version: str
    goal_id: str
    bundle: str
    mode: str
    evidence_time: str
    repo_root: str
    outer_commit: str | None
    rows: tuple[PrerequisiteRow, ...]
    compatibility_map: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]
    limitations: tuple[str, ...]
    is_completion_evidence: bool = OBSERVATION_IS_COMPLETION_EVIDENCE
    is_proof_evidence: bool = OBSERVATION_IS_PROOF_EVIDENCE
    authorizes_mutation: bool = OBSERVATION_AUTHORIZES_MUTATION
    authorizes_release: bool = OBSERVATION_AUTHORIZES_RELEASE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "version": self.version,
            "goal_id": self.goal_id,
            "bundle": self.bundle,
            "mode": self.mode,
            "evidence_time": self.evidence_time,
            "repo_root": self.repo_root,
            "outer_commit": self.outer_commit,
            "rows": [row.to_dict() for row in self.rows],
            "compatibility_map": list(self.compatibility_map),
            "summary": dict(self.summary),
            "limitations": list(self.limitations),
            "is_completion_evidence": self.is_completion_evidence,
            "is_proof_evidence": self.is_proof_evidence,
            "authorizes_mutation": self.authorizes_mutation,
            "authorizes_release": self.authorizes_release,
        }


# ---------------------------------------------------------------------------
# Path / git helpers
# ---------------------------------------------------------------------------


def repo_root_from(root: Path | None = None) -> Path:
    """Resolve the superproject root."""

    if root is not None:
        return root.expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def observe_git_commit(repo_path: Path) -> tuple[str | None, tuple[str, ...]]:
    """Return HEAD commit for ``repo_path`` without treating branch as release."""

    if not repo_path.is_dir():
        return None, (f"repository path missing: {repo_path}",)
    code, out, err = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    if code != 0 or not out:
        detail = err or out or f"git rev-parse failed with {code}"
        return None, (detail,)
    # Explicitly ignore branch / symbolic-ref names for release claims.
    return out, ()


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# AST symbol discovery (read-only, no imports of product code)
# ---------------------------------------------------------------------------


def _iter_python_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix == ".py":
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*.py")):
            if any(part.startswith(".") for part in child.parts):
                continue
            yield child


def _symbol_hits_in_source(
    source: str,
    *,
    path: str,
    wanted: set[str],
) -> list[SymbolHit]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []

    hits: list[SymbolHit] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            hits.append(
                SymbolHit(
                    symbol=node.name,
                    path=path,
                    line=int(getattr(node, "lineno", 0) or 0),
                    kind="class",
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            hits.append(
                SymbolHit(
                    symbol=node.name,
                    path=path,
                    line=int(getattr(node, "lineno", 0) or 0),
                    kind="function",
                )
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    hits.append(
                        SymbolHit(
                            symbol=target.id,
                            path=path,
                            line=int(getattr(node, "lineno", 0) or 0),
                            kind="assign",
                        )
                    )
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id in wanted:
                hits.append(
                    SymbolHit(
                        symbol=target.id,
                        path=path,
                        line=int(getattr(node, "lineno", 0) or 0),
                        kind="ann_assign",
                    )
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in wanted:
                hits.append(
                    SymbolHit(
                        symbol=node.value,
                        path=path,
                        line=int(getattr(node, "lineno", 0) or 0),
                        kind="interface_literal",
                    )
                )
    return hits


def find_symbols(
    root: Path,
    module_paths: Sequence[str],
    symbols: Sequence[str],
) -> tuple[tuple[SymbolHit, ...], tuple[str, ...]]:
    """Locate definitions / interface literals for ``symbols`` under ``module_paths``."""

    wanted = {name for name in symbols if name}
    if not wanted:
        return (), ()

    hits: list[SymbolHit] = []
    limitations: list[str] = []

    if not module_paths:
        return (), ("no module paths configured for symbol search",)

    for relative in module_paths:
        path = root / relative
        if not path.exists():
            limitations.append(f"module path missing: {relative}")
            continue
        if path.is_file() and path.suffix != ".py":
            # Non-Python paths (objectives markdown) may still carry interface
            # string mentions; scan as text for interface literals only.
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                limitations.append(f"unreadable path {relative}: {exc}")
                continue
            for name in sorted(wanted):
                if name in text:
                    hits.append(
                        SymbolHit(
                            symbol=name,
                            path=relative,
                            line=0,
                            kind="text_mention",
                        )
                    )
            continue

        for py_file in _iter_python_files(path):
            try:
                source = py_file.read_text(encoding="utf-8")
            except OSError as exc:
                limitations.append(f"unreadable python file {py_file}: {exc}")
                continue
            rel = str(py_file.relative_to(root)).replace("\\", "/")
            hits.extend(
                _symbol_hits_in_source(source, path=rel, wanted=wanted)
            )

    # Stable, unique by (symbol, path, kind)
    dedup: dict[tuple[str, str, str], SymbolHit] = {}
    for hit in hits:
        key = (hit.symbol, hit.path, hit.kind)
        if key not in dedup:
            dedup[key] = hit
    ordered = tuple(
        sorted(
            dedup.values(),
            key=lambda item: (item.symbol, item.path, item.line, item.kind),
        )
    )
    return ordered, tuple(limitations)


# ---------------------------------------------------------------------------
# Board parsing
# ---------------------------------------------------------------------------


def parse_board_statuses(text: str) -> dict[str, int]:
    """Count task status rows from a supervisor markdown board."""

    counts = {
        "completed": 0,
        "open": 0,
        "blocked": 0,
        "other": 0,
        "task_count": 0,
    }
    current_task: str | None = None
    for raw_line in text.splitlines():
        heading = _TASK_HEADING_RE.match(raw_line.strip())
        if heading:
            current_task = heading.group("task_id")
            counts["task_count"] += 1
            continue
        match = _STATUS_LINE_RE.match(raw_line.strip())
        if not match or current_task is None:
            continue
        status = match.group("status").lower()
        if status == "completed":
            counts["completed"] += 1
        elif status in {"todo", "in_progress", "review"}:
            counts["open"] += 1
        elif status == "blocked":
            counts["blocked"] += 1
        elif status == "cancelled":
            counts["other"] += 1
        else:
            counts["other"] += 1
        current_task = None  # only the first status line per task block
    return counts


def observe_board(root: Path, board_paths: Sequence[str]) -> BoardObservation:
    """Observe owner-board lifecycle without treating branch names as release."""

    if not board_paths:
        return BoardObservation(
            paths=(),
            state=BoardState.NOT_CONFIGURED.value,
            task_count=0,
            completed=0,
            open=0,
            blocked=0,
            other=0,
            limitations=("no owner board paths configured",),
        )

    totals = {
        "completed": 0,
        "open": 0,
        "blocked": 0,
        "other": 0,
        "task_count": 0,
    }
    present: list[str] = []
    limitations: list[str] = []

    unreadable = False
    for relative in board_paths:
        path = root / relative
        if path.exists() and not path.is_file():
            limitations.append(f"board path not a file: {relative}")
            unreadable = True
            continue
        if not path.is_file():
            limitations.append(f"board path missing: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            limitations.append(f"board unreadable {relative}: {exc}")
            unreadable = True
            continue
        present.append(relative)
        counts = parse_board_statuses(text)
        for key in totals:
            totals[key] += counts[key]

    if not present:
        if unreadable or any("unreadable" in item for item in limitations):
            state = BoardState.UNREADABLE.value
        else:
            state = BoardState.ABSENT.value
        return BoardObservation(
            paths=tuple(board_paths),
            state=state,
            task_count=0,
            completed=0,
            open=0,
            blocked=0,
            other=0,
            limitations=tuple(limitations),
        )

    open_work = totals["open"] + totals["blocked"]
    if totals["task_count"] == 0:
        state = BoardState.UNREADABLE.value
        limitations = limitations + ("board contained no task status rows",)
    elif open_work == 0 and totals["completed"] > 0:
        state = BoardState.TERMINAL.value
    else:
        state = BoardState.IN_FLIGHT.value

    return BoardObservation(
        paths=tuple(present),
        state=state,
        task_count=totals["task_count"],
        completed=totals["completed"],
        open=totals["open"],
        blocked=totals["blocked"],
        other=totals["other"],
        limitations=tuple(limitations),
    )


# ---------------------------------------------------------------------------
# Per-prerequisite observation
# ---------------------------------------------------------------------------


def observe_tests(root: Path, selectors: Sequence[str]) -> TestObservation:
    present: list[str] = []
    missing: list[str] = []
    for selector in selectors:
        if (root / selector).is_file():
            present.append(selector)
        else:
            missing.append(selector)
    all_present = bool(selectors) and not missing
    if not selectors:
        all_present = False
    return TestObservation(
        selectors=tuple(selectors),
        present=tuple(present),
        missing=tuple(missing),
        all_present=all_present,
    )


_DEFINITION_KINDS = frozenset(
    {"class", "function", "assign", "ann_assign", "interface_literal"}
)


def observe_api(root: Path, spec: PrerequisiteSpec) -> ApiObservation:
    """Resolve expected symbols, falling back only via the explicit map."""

    limitations: list[str] = []
    primary_hits, primary_limits = find_symbols(
        root,
        spec.module_paths,
        spec.expected_symbols + spec.interface_ids,
    )
    limitations.extend(primary_limits)

    definition_hits = tuple(
        hit for hit in primary_hits if hit.kind in _DEFINITION_KINDS
    )
    expected_names = set(spec.expected_symbols) | set(spec.interface_ids)
    exact_names = {
        hit.symbol for hit in definition_hits if hit.symbol in expected_names
    }

    has_expected_def = any(
        hit.symbol in spec.expected_symbols
        and hit.kind in {"class", "function", "assign", "ann_assign"}
        for hit in definition_hits
    )
    has_interface = any(
        hit.symbol in spec.interface_ids
        and hit.kind in {"interface_literal", "assign", "ann_assign"}
        for hit in definition_hits
    )

    # Exact resolution requires a real definition for at least one expected
    # symbol, or a versioned interface literal when no compatibility rename
    # applies for this planned name.
    compat = _compat_for(spec.prerequisite_id)
    if has_expected_def or (has_interface and compat is None):
        return ApiObservation(
            resolution=ApiResolution.EXACT.value,
            expected_symbols=spec.expected_symbols,
            interface_ids=spec.interface_ids,
            found_symbols=tuple(sorted(exact_names)),
            hits=definition_hits,
            compatibility=None,
            module_paths=spec.module_paths,
            limitations=tuple(limitations),
        )

    if compat is not None:
        search_paths = compat.module_paths or spec.module_paths
        search_symbols = compat.implementation_symbols + compat.interface_ids
        compat_hits, compat_limits = find_symbols(
            root, search_paths, search_symbols
        )
        limitations.extend(compat_limits)
        definition_compat = tuple(
            hit for hit in compat_hits if hit.kind in _DEFINITION_KINDS
        )
        found = tuple(sorted({hit.symbol for hit in definition_compat}))
        if definition_compat:
            return ApiObservation(
                resolution=ApiResolution.COMPATIBILITY_MAP.value,
                expected_symbols=spec.expected_symbols,
                interface_ids=compat.interface_ids,
                found_symbols=found,
                hits=definition_compat,
                compatibility={
                    "planned_name": compat.planned_name,
                    "implementation_symbols": list(
                        compat.implementation_symbols
                    ),
                    "interface_ids": list(compat.interface_ids),
                    "module_paths": list(compat.module_paths),
                    "rationale": compat.rationale,
                },
                module_paths=search_paths,
                limitations=tuple(limitations),
            )
        limitations.append(
            f"compatibility map entry present for {spec.prerequisite_id} "
            "but implementation symbols were not found"
        )

    if definition_hits or primary_hits:
        limitations.append(
            "symbol mentions found without a resolvable public definition"
        )

    return ApiObservation(
        resolution=ApiResolution.ABSENT.value,
        expected_symbols=spec.expected_symbols,
        interface_ids=spec.interface_ids,
        found_symbols=(),
        hits=(),
        compatibility=None,
        module_paths=spec.module_paths,
        limitations=tuple(limitations),
    )


def derive_status(
    *,
    api: ApiObservation,
    tests: TestObservation,
    board: BoardObservation,
    commit: str | None,
    commit_limitations: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    """Derive a single status value under fail-closed release rules."""

    limitations: list[str] = []

    if commit is None and commit_limitations:
        # Missing commit makes release claims unverifiable when a repository
        # root was expected; still allow honest missing/in-flight when the
        # whole repository path is absent.
        if board.state == BoardState.UNREADABLE.value:
            return PrerequisiteStatus.UNVERIFIABLE.value, tuple(commit_limitations)
        if api.resolution == ApiResolution.UNVERIFIABLE.value:
            return PrerequisiteStatus.UNVERIFIABLE.value, tuple(commit_limitations)

    if board.state == BoardState.UNREADABLE.value and board.paths:
        return (
            PrerequisiteStatus.UNVERIFIABLE.value,
            tuple(board.limitations) or ("owner board unreadable",),
        )

    if api.resolution == ApiResolution.COMPATIBILITY_MAP.value:
        # Name mismatch is reported explicitly; release is never inferred
        # from the planned name alone.
        limitations.append(
            "resolved only through the explicit compatibility map"
        )
        return PrerequisiteStatus.MISMATCHED_NAME.value, tuple(limitations)

    if api.resolution == ApiResolution.ABSENT.value:
        if board.state == BoardState.IN_FLIGHT.value:
            return PrerequisiteStatus.IN_FLIGHT.value, (
                "owner board has open work and public API is absent",
            )
        if board.state == BoardState.TERMINAL.value:
            return PrerequisiteStatus.MISSING.value, (
                "owner board is terminal but public API is absent",
            )
        return PrerequisiteStatus.MISSING.value, (
            "public API absent and no terminal owner evidence",
        )

    # Exact API resolution
    assert api.resolution == ApiResolution.EXACT.value
    if board.state == BoardState.TERMINAL.value and tests.all_present and commit:
        return PrerequisiteStatus.RELEASED.value, ()
    if board.state == BoardState.TERMINAL.value and not tests.all_present:
        return PrerequisiteStatus.IN_FLIGHT.value, (
            "owner board terminal but focused tests are incomplete",
        )
    if board.state == BoardState.IN_FLIGHT.value:
        return PrerequisiteStatus.IN_FLIGHT.value, (
            "owner board still has open or blocked work",
        )
    if board.state in {
        BoardState.NOT_CONFIGURED.value,
        BoardState.ABSENT.value,
    }:
        # API present without a bound terminal board: never claim released.
        return PrerequisiteStatus.IN_FLIGHT.value, (
            "public API present but owner board is not bound as terminal",
        )
    if not commit:
        return PrerequisiteStatus.UNVERIFIABLE.value, (
            "repository commit could not be resolved",
        )
    return PrerequisiteStatus.IN_FLIGHT.value, ()


def row_is_terminal(row: PrerequisiteRow) -> bool:
    """Whether a row is terminal enough for require-terminal mode."""

    if row.status == PrerequisiteStatus.RELEASED.value:
        return bool(row.commit) and row.tests.all_present and row.api.to_dict()["resolved"]
    if row.status == PrerequisiteStatus.MISMATCHED_NAME.value:
        # Mapped names require explicit map + terminal board + tests + commit.
        return (
            row.api.resolution == ApiResolution.COMPATIBILITY_MAP.value
            and row.board.state == BoardState.TERMINAL.value
            and row.tests.all_present
            and bool(row.commit)
            and row.api.compatibility is not None
        )
    return False


def observe_one(
    root: Path,
    spec: PrerequisiteSpec,
    *,
    evidence_time: str,
    commit_cache: dict[str, tuple[str | None, tuple[str, ...]]],
) -> PrerequisiteRow:
    """Observe a single catalog entry."""

    repo_key = spec.repository_root_relative
    if repo_key not in commit_cache:
        repo_path = root if repo_key in {".", ""} else root / repo_key
        commit_cache[repo_key] = observe_git_commit(repo_path)
    commit, commit_limits = commit_cache[repo_key]

    api = observe_api(root, spec)
    tests = observe_tests(root, spec.test_selectors)
    board = observe_board(root, spec.board_paths)
    status, status_limits = derive_status(
        api=api,
        tests=tests,
        board=board,
        commit=commit,
        commit_limitations=commit_limits,
    )
    limitations = tuple(
        dict.fromkeys(
            [
                *commit_limits,
                *api.limitations,
                *board.limitations,
                *status_limits,
            ]
        )
    )
    provisional = PrerequisiteRow(
        prerequisite_id=spec.prerequisite_id,
        repository=spec.repository,
        commit=commit,
        status=status,
        api=api,
        tests=tests,
        board=board,
        evidence_time=evidence_time,
        notes=spec.notes,
        limitations=limitations,
        terminal=False,
    )
    return PrerequisiteRow(
        prerequisite_id=provisional.prerequisite_id,
        repository=provisional.repository,
        commit=provisional.commit,
        status=provisional.status,
        api=provisional.api,
        tests=provisional.tests,
        board=provisional.board,
        evidence_time=provisional.evidence_time,
        notes=provisional.notes,
        limitations=provisional.limitations,
        terminal=row_is_terminal(provisional),
    )


def _summary_for(rows: Sequence[PrerequisiteRow]) -> dict[str, Any]:
    by_status: dict[str, int] = {status.value: 0 for status in PrerequisiteStatus}
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
    terminal_ids = [row.prerequisite_id for row in rows if row.terminal]
    nonterminal_ids = [row.prerequisite_id for row in rows if not row.terminal]
    return {
        "prerequisite_count": len(rows),
        "by_status": by_status,
        "terminal_count": len(terminal_ids),
        "terminal_ids": terminal_ids,
        "nonterminal_ids": nonterminal_ids,
        "all_terminal": bool(rows) and not nonterminal_ids,
    }


def observe_prerequisite_releases(
    repo_root: Path | str | None = None,
    *,
    mode: str | ObservationMode = ObservationMode.OBSERVE,
    evidence_time: str | None = None,
    catalog: Sequence[PrerequisiteSpec] | None = None,
) -> PrerequisiteObservation:
    """Observe all prerequisites and return a bound ``PrerequisiteObservation``.

    Ordinary ``observe`` mode always returns an honest snapshot. Callers that
    need fail-closed terminal admission must use ``require-terminal`` and
    inspect ``summary["all_terminal"]`` or call :func:`assert_terminal`.
    """

    root = repo_root_from(Path(repo_root) if repo_root is not None else None)
    mode_value = (
        mode.value if isinstance(mode, ObservationMode) else str(mode)
    )
    if mode_value not in {
        ObservationMode.OBSERVE.value,
        ObservationMode.REQUIRE_TERMINAL.value,
    }:
        raise ValueError(f"unsupported observation mode: {mode_value}")

    when = evidence_time or _utc_now_iso()
    specs = tuple(catalog) if catalog is not None else prerequisite_catalog()
    if len(specs) != 10:
        raise ValueError(
            f"prerequisite catalog must contain exactly ten systems, got {len(specs)}"
        )

    outer_commit, outer_limits = observe_git_commit(root)
    commit_cache: dict[str, tuple[str | None, tuple[str, ...]]] = {
        ".": (outer_commit, outer_limits),
    }

    rows = tuple(
        observe_one(
            root,
            spec,
            evidence_time=when,
            commit_cache=commit_cache,
        )
        for spec in specs
    )
    summary = _summary_for(rows)
    limitations = list(outer_limits)
    limitations.append(
        "observation is non-authoritative; SHQ-G010 owns release admission"
    )
    limitations.append(
        "release is never inferred from prompt text or git branch names alone"
    )

    return PrerequisiteObservation(
        schema=OBSERVATION_SCHEMA,
        interface=INTERFACE_ID,
        version=OBSERVATION_VERSION,
        goal_id=GOAL_ID,
        bundle=BUNDLE_ID,
        mode=mode_value,
        evidence_time=when,
        repo_root=str(root),
        outer_commit=outer_commit,
        rows=rows,
        compatibility_map=tuple(compatibility_map_as_dict()),
        summary=summary,
        limitations=tuple(limitations),
    )


class NonTerminalPrerequisiteError(RuntimeError):
    """Raised when require-terminal mode sees incomplete prerequisite rows."""

    def __init__(
        self,
        observation: PrerequisiteObservation,
        *,
        nonterminal_ids: Sequence[str],
    ) -> None:
        self.observation = observation
        self.nonterminal_ids = tuple(nonterminal_ids)
        joined = ", ".join(self.nonterminal_ids) or "(none)"
        super().__init__(
            "require-terminal mode failed closed; non-terminal prerequisites: "
            f"{joined}"
        )


def assert_terminal(observation: PrerequisiteObservation) -> PrerequisiteObservation:
    """Fail closed unless every row is terminal."""

    nonterminal = [
        row.prerequisite_id for row in observation.rows if not row.terminal
    ]
    if nonterminal:
        raise NonTerminalPrerequisiteError(
            observation, nonterminal_ids=nonterminal
        )
    return observation


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def observation_to_json(
    observation: PrerequisiteObservation,
    *,
    indent: int = 2,
) -> str:
    """Serialize an observation deterministically."""

    return json.dumps(
        observation.to_dict(),
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"


def write_observation_artifact(
    observation: PrerequisiteObservation,
    output_path: Path | str,
) -> Path:
    """Atomically write the observation JSON artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = observation_to_json(observation)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Observe self-hosting qualification prerequisite release state "
            "(SHQ-G006). Read-only; never claims release from branch names."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Superproject repository root (default: auto-detect).",
    )
    parser.add_argument(
        "--mode",
        choices=[
            ObservationMode.OBSERVE.value,
            ObservationMode.REQUIRE_TERMINAL.value,
        ],
        default=ObservationMode.OBSERVE.value,
        help=(
            "observe: always emit an honest snapshot (default). "
            "require-terminal: fail closed unless every row is terminal."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON artifact path. Defaults to "
            f"{DEFAULT_OUTPUT_RELATIVE} when omitted and writing is requested "
            "via --write / require-terminal workflows."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the observation JSON artifact.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON stdout dump (still writes when --write).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = repo_root_from(args.repo_root) if args.repo_root is not None else repo_root_from()

    observation = observe_prerequisite_releases(root, mode=args.mode)

    should_write = bool(args.write) or args.output is not None
    if should_write:
        output = (
            Path(args.output)
            if args.output is not None
            else root / DEFAULT_OUTPUT_RELATIVE
        )
        if not output.is_absolute():
            output = (root / output).resolve()
        write_observation_artifact(observation, output)
        if not args.quiet:
            print(f"wrote {output}", file=sys.stderr)

    if not args.quiet:
        sys.stdout.write(observation_to_json(observation))

    if args.mode == ObservationMode.REQUIRE_TERMINAL.value:
        try:
            assert_terminal(observation)
        except NonTerminalPrerequisiteError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
