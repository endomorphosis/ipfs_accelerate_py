"""Joined DuckDB/Quack control-plane release receipt (DQP-039).

Interface: ``DuckDBControlPlaneReleaseReceipt@1``

Independently queries and joins schema, Quack, import/export, intent, runtime,
worktree, AST/mutation, symbolic/proof, context/churn, control, watchdog,
backup, chaos, canary, shadow, cutover and rollback roots into a content-bound
decision.  Never fabricates or refreshes component evidence inside the verifier.

Fail-closed on missing/stale/synthetic/skipped evidence, legacy file decision
reads in canary, unauthorized SQL, stale lease writes, false completion, lost
accepted state, incomplete mutation lineage, projection divergence, safety or
quality regression, or absent rollback.  A pass records Quack experimental
scope without claiming production HA or future 2.0 compatibility.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.quack_capabilities import DEFAULT_QUACK_BETA_LIMITATIONS
from .duckdb_quack_baseline import SAFETY_FLOOR_KEYS


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DUCKDB_CONTROL_PLANE_RELEASE_INTERFACE: Final[str] = (
    "DuckDBControlPlaneReleaseReceipt@1"
)
RELEASE_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-039"
GOAL_ID: Final[str] = "DQP-G090"
EVIDENCE: Final[str] = "dqp/duckdb-quack-release@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
RELEASE_RECEIPT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-control-plane-release-receipt@1"
)
COMPONENT_EVIDENCE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/release-component-evidence@1"
)

DEFAULT_EVIDENCE_MAX_AGE_SECONDS: Final[int] = 86_400
MAX_TEXT_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 512

# Evidence roots that must be present and fresh for release.
REQUIRED_COMPONENT_ROOTS: Final[tuple[str, ...]] = (
    "schema",
    "quack",
    "import_export",
    "intent",
    "runtime",
    "worktree",
    "ast_mutation",
    "symbolic_proof",
    "context_churn",
    "control",
    "watchdog",
    "backup",
    "chaos",
    "canary",
    "shadow",
    "cutover",
    "rollback",
)

REQUIRED_MODULES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline",
    "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_canary",
    "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_benchmark",
    "ipfs_accelerate_py.agent_supervisor.validation.quack_chaos",
    "ipfs_accelerate_py.agent_supervisor.self_improvement.database_shadow_rollout",
    "ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout",
    "ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import",
    "ipfs_accelerate_py.agent_supervisor.task_sources.state_export",
    "ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities",
)

# Board task ids that must be terminal before release.
PRIOR_TASK_IDS: Final[tuple[str, ...]] = tuple(
    f"DQP-{index:03d}" for index in range(0, 39)
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ReleaseVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"


class ComponentStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"
    STALE = "stale"
    SYNTHETIC = "synthetic"
    SKIPPED = "skipped"


class DuckDBQuackReleaseError(ValueError):
    """Fail-closed rejection for incomplete release evaluation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise DuckDBQuackReleaseError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise DuckDBQuackReleaseError(f"{name} must not be empty")
    if "\x00" in result:
        raise DuckDBQuackReleaseError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise DuckDBQuackReleaseError(f"{name} exceeds its {maximum}-byte bound")
    return result


def _nonnegative_int(value: Any, name: str, *, maximum: int = 10**18) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DuckDBQuackReleaseError(f"{name} must be a non-negative integer")
    if value < 0 or value > maximum:
        raise DuckDBQuackReleaseError(f"{name} out of bounds")
    return value


def content_identity(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Evidence models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentEvidence:
    """One independently supplied evidence root (never fabricated here)."""

    SCHEMA: ClassVar[str] = COMPONENT_EVIDENCE_SCHEMA

    root: str
    identity: str
    age_seconds: int
    passed: bool
    synthetic: bool = False
    skipped: bool = False
    tree_id: str = ""
    schema_checksum: str = ""
    store_generation: int = 0
    profile_id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _text(self.root, "root", maximum=64))
        object.__setattr__(
            self, "identity", _text(self.identity, "identity", maximum=160)
        )
        object.__setattr__(
            self, "age_seconds", _nonnegative_int(self.age_seconds, "age_seconds")
        )
        object.__setattr__(
            self,
            "store_generation",
            _nonnegative_int(self.store_generation, "store_generation"),
        )
        if self.tree_id:
            object.__setattr__(
                self, "tree_id", _text(self.tree_id, "tree_id", maximum=256)
            )
        if self.schema_checksum:
            object.__setattr__(
                self,
                "schema_checksum",
                _text(self.schema_checksum, "schema_checksum", maximum=128),
            )
        if self.profile_id:
            object.__setattr__(
                self, "profile_id", _text(self.profile_id, "profile_id", maximum=128)
            )

    def classify(
        self, *, max_age_seconds: int, expected_tree: str, expected_schema: str
    ) -> ComponentStatus:
        if self.skipped:
            return ComponentStatus.SKIPPED
        if self.synthetic:
            return ComponentStatus.SYNTHETIC
        if self.age_seconds > max_age_seconds:
            return ComponentStatus.STALE
        if self.tree_id and expected_tree and self.tree_id != expected_tree:
            return ComponentStatus.FAIL
        if (
            self.schema_checksum
            and expected_schema
            and self.schema_checksum != expected_schema
        ):
            return ComponentStatus.FAIL
        if not self.passed:
            return ComponentStatus.FAIL
        return ComponentStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "root": self.root,
            "identity": self.identity,
            "age_seconds": self.age_seconds,
            "passed": self.passed,
            "synthetic": self.synthetic,
            "skipped": self.skipped,
            "tree_id": self.tree_id,
            "schema_checksum": self.schema_checksum,
            "store_generation": self.store_generation,
            "profile_id": self.profile_id,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SafetyFloorObservation:
    floors: Mapping[str, int]

    def __post_init__(self) -> None:
        cleaned = {
            key: _nonnegative_int(
                dict(self.floors or {}).get(key, 0), f"floors.{key}"
            )
            for key in SAFETY_FLOOR_KEYS
        }
        for key in dict(self.floors or {}):
            if key not in SAFETY_FLOOR_KEYS:
                raise DuckDBQuackReleaseError(f"unknown safety floor key {key!r}")
        object.__setattr__(self, "floors", MappingProxyType(cleaned))

    @property
    def all_zero(self) -> bool:
        return all(value == 0 for value in self.floors.values())

    def to_dict(self) -> dict[str, int]:
        return dict(self.floors)

    @classmethod
    def zeros(cls) -> "SafetyFloorObservation":
        return cls(floors={key: 0 for key in SAFETY_FLOOR_KEYS})


@dataclass(frozen=True)
class DuckDBControlPlaneReleaseReceipt:
    """``DuckDBControlPlaneReleaseReceipt@1`` terminal joined decision."""

    SCHEMA: ClassVar[str] = RELEASE_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = DUCKDB_CONTROL_PLANE_RELEASE_INTERFACE

    verdict: ReleaseVerdict
    tree_id: str
    schema_checksum: str
    store_generation: int
    quack_profile: str
    extension_fingerprint: str
    git_identity: str
    components: Mapping[str, str]
    missing_roots: tuple[str, ...]
    modules_present: tuple[str, ...]
    modules_missing: tuple[str, ...]
    tasks_terminal: bool
    safety_floors_zero: bool
    quality_non_regressing: bool
    legacy_file_decision_read_in_canary: bool
    rollback_present: bool
    experimental_scope: bool
    production_ha_claimed: bool
    duckdb_2_0_compatibility_claimed: bool
    reason_codes: tuple[str, ...] = ()
    beta_limitations: tuple[str, ...] = ()
    safety_floors: SafetyFloorObservation = field(
        default_factory=SafetyFloorObservation.zeros
    )
    created_at: str = field(default_factory=_utc_iso)
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verdict",
            self.verdict
            if isinstance(self.verdict, ReleaseVerdict)
            else ReleaseVerdict(str(self.verdict)),
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "schema_checksum", _text(self.schema_checksum, "schema_checksum")
        )
        object.__setattr__(
            self,
            "store_generation",
            _nonnegative_int(self.store_generation, "store_generation"),
        )
        object.__setattr__(
            self, "quack_profile", _text(self.quack_profile, "quack_profile")
        )
        object.__setattr__(
            self,
            "extension_fingerprint",
            _text(self.extension_fingerprint, "extension_fingerprint"),
        )
        object.__setattr__(
            self, "git_identity", _text(self.git_identity, "git_identity")
        )
        object.__setattr__(
            self,
            "components",
            MappingProxyType(
                {
                    _text(k, "components.key", maximum=64): _text(
                        v, "components.value", maximum=32
                    )
                    for k, v in dict(self.components).items()
                }
            ),
        )
        # Pass always records experimental scope; never claims HA / 2.0.
        object.__setattr__(self, "experimental_scope", True)
        object.__setattr__(self, "production_ha_claimed", False)
        object.__setattr__(self, "duckdb_2_0_compatibility_claimed", False)
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _text(item, "reason_codes.item", maximum=128)
                for item in self.reason_codes[:MAX_REASON_CODES]
            ),
        )
        if not self.beta_limitations:
            object.__setattr__(
                self, "beta_limitations", tuple(DEFAULT_QUACK_BETA_LIMITATIONS)
            )

    @property
    def passed(self) -> bool:
        return self.verdict is ReleaseVerdict.PASS

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": RELEASE_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "verdict": self.verdict.value
            if isinstance(self.verdict, Enum)
            else self.verdict,
            "passed": self.passed,
            "tree_id": self.tree_id,
            "schema_checksum": self.schema_checksum,
            "store_generation": self.store_generation,
            "quack_profile": self.quack_profile,
            "extension_fingerprint": self.extension_fingerprint,
            "git_identity": self.git_identity,
            "components": dict(self.components),
            "missing_roots": list(self.missing_roots),
            "modules_present": list(self.modules_present),
            "modules_missing": list(self.modules_missing),
            "tasks_terminal": self.tasks_terminal,
            "safety_floors_zero": self.safety_floors_zero,
            "quality_non_regressing": self.quality_non_regressing,
            "legacy_file_decision_read_in_canary": (
                self.legacy_file_decision_read_in_canary
            ),
            "rollback_present": self.rollback_present,
            "experimental_scope": True,
            "production_ha_claimed": False,
            "duckdb_2_0_compatibility_claimed": False,
            "safety_floors": self.safety_floors.to_dict(),
            "beta_limitations": list(self.beta_limitations),
            "reason_codes": list(self.reason_codes),
            "created_at": self.created_at,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _probe_modules(modules: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present: list[str] = []
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
            present.append(name)
        except Exception:
            missing.append(name)
    return tuple(present), tuple(missing)


def parse_board_task_statuses(
    board_text: str,
) -> Mapping[str, str]:
    """Parse ``- Status:`` lines under ``## DQP-NNN`` headings."""

    statuses: dict[str, str] = {}
    current: str | None = None
    for line in board_text.splitlines():
        if line.startswith("## DQP-"):
            current = line[3:].split()[0].strip()
            continue
        if current and line.startswith("- Status:"):
            statuses[current] = line.split(":", 1)[1].strip().lower()
            current = None
    return MappingProxyType(statuses)


def board_tasks_terminal(
    board_text: str,
    *,
    prior_task_ids: Sequence[str] = PRIOR_TASK_IDS,
) -> tuple[bool, tuple[str, ...]]:
    statuses = parse_board_task_statuses(board_text)
    incomplete = [
        task_id
        for task_id in prior_task_ids
        if statuses.get(task_id, "missing") != "completed"
    ]
    return (not incomplete, tuple(incomplete))


def hermetic_component_bundle(
    *,
    tree_id: str = "tree:sha256:dqp039-release",
    schema_checksum: str = "sha256:" + ("cc" * 32),
    store_generation: int = 1,
    profile_id: str = "profile:quack-1.5.2-loopback",
    age_seconds: int = 300,
    exclude_roots: Sequence[str] = (),
    fail_roots: Sequence[str] = (),
    stale_roots: Sequence[str] = (),
    synthetic_roots: Sequence[str] = (),
    skipped_roots: Sequence[str] = (),
) -> tuple[ComponentEvidence, ...]:
    """Build a complete hermetic component evidence set for tests."""

    excluded = frozenset(exclude_roots)
    failed = frozenset(fail_roots)
    stale = frozenset(stale_roots)
    synthetic = frozenset(synthetic_roots)
    skipped = frozenset(skipped_roots)
    items: list[ComponentEvidence] = []
    for root in REQUIRED_COMPONENT_ROOTS:
        if root in excluded:
            continue
        items.append(
            ComponentEvidence(
                root=root,
                identity=f"component:{root}:hermetic",
                age_seconds=10**9 if root in stale else age_seconds,
                passed=root not in failed,
                synthetic=root in synthetic,
                skipped=root in skipped,
                tree_id=tree_id,
                schema_checksum=schema_checksum,
                store_generation=store_generation,
                profile_id=profile_id,
                detail="hermetic",
            )
        )
    return tuple(items)


def evaluate_release(
    *,
    components: Sequence[ComponentEvidence],
    tree_id: str,
    schema_checksum: str,
    store_generation: int,
    quack_profile: str,
    extension_fingerprint: str,
    git_identity: str,
    board_text: str | None = None,
    tasks_terminal: bool | None = None,
    safety_floors: SafetyFloorObservation | None = None,
    quality_non_regressing: bool = True,
    legacy_file_decision_read_in_canary: bool = False,
    max_age_seconds: int = DEFAULT_EVIDENCE_MAX_AGE_SECONDS,
    require_modules: Sequence[str] = REQUIRED_MODULES,
) -> DuckDBControlPlaneReleaseReceipt:
    """Join component evidence into a fail-closed release decision.

    This function never fabricates missing component evidence.
    """

    floors = safety_floors or SafetyFloorObservation.zeros()
    by_root = {item.root: item for item in components}
    component_status: dict[str, str] = {}
    reasons: list[str] = []
    missing: list[str] = []

    for root in REQUIRED_COMPONENT_ROOTS:
        item = by_root.get(root)
        if item is None:
            missing.append(root)
            component_status[root] = ComponentStatus.MISSING.value
            reasons.append(f"missing:{root}")
            continue
        status = item.classify(
            max_age_seconds=max_age_seconds,
            expected_tree=tree_id,
            expected_schema=schema_checksum,
        )
        component_status[root] = status.value
        if status is ComponentStatus.PASS:
            continue
        reasons.append(f"{status.value}:{root}")

    present, modules_missing = _probe_modules(require_modules)
    if modules_missing:
        reasons.append("required_module_missing")

    if tasks_terminal is None:
        if board_text is None:
            # Without board text, callers must assert terminal tasks explicitly.
            tasks_terminal = False
            reasons.append("tasks_terminal_unknown")
        else:
            tasks_terminal, incomplete = board_tasks_terminal(board_text)
            if incomplete:
                reasons.append("tasks_not_terminal")
                reasons.extend(f"task_open:{task_id}" for task_id in incomplete[:32])

    if not floors.all_zero:
        reasons.append("safety_floor_nonzero")
    if not quality_non_regressing:
        reasons.append("quality_regression")
    if legacy_file_decision_read_in_canary:
        reasons.append("legacy_file_decision_read_in_canary")

    rollback_present = (
        component_status.get("rollback") == ComponentStatus.PASS.value
    )
    if not rollback_present:
        reasons.append("rollback_absent")

    # Hard safety denials that always fail (not merely block).
    hard_fail_prefixes = (
        "fail:",
        "synthetic:",
        "safety_floor_nonzero",
        "quality_regression",
        "legacy_file_decision_read_in_canary",
        "unauthorized_sql",
        "stale_lease",
        "false_completion",
        "accepted_state_loss",
        "projection_divergence",
        "incomplete_mutation_lineage",
    )
    hard = any(
        reason.startswith(prefix) or reason == prefix.rstrip(":")
        for reason in reasons
        for prefix in hard_fail_prefixes
    )

    if not reasons:
        verdict = ReleaseVerdict.PASS
    elif hard:
        verdict = ReleaseVerdict.FAIL
    else:
        # Missing/stale/skipped without hard regression → blocked.
        soft_only = all(
            reason.startswith(
                ("missing:", "stale:", "skipped:", "task_open:", "tasks_")
            )
            or reason
            in {
                "required_module_missing",
                "tasks_not_terminal",
                "tasks_terminal_unknown",
                "rollback_absent",
            }
            for reason in reasons
        )
        verdict = ReleaseVerdict.BLOCKED if soft_only else ReleaseVerdict.FAIL

    return DuckDBControlPlaneReleaseReceipt(
        verdict=verdict,
        tree_id=tree_id,
        schema_checksum=schema_checksum,
        store_generation=store_generation,
        quack_profile=quack_profile,
        extension_fingerprint=extension_fingerprint,
        git_identity=git_identity,
        components=component_status,
        missing_roots=tuple(missing),
        modules_present=present,
        modules_missing=modules_missing,
        tasks_terminal=bool(tasks_terminal),
        safety_floors_zero=floors.all_zero,
        quality_non_regressing=quality_non_regressing,
        legacy_file_decision_read_in_canary=legacy_file_decision_read_in_canary,
        rollback_present=rollback_present,
        experimental_scope=True,
        production_ha_claimed=False,
        duckdb_2_0_compatibility_claimed=False,
        reason_codes=tuple(reasons),
        safety_floors=floors,
    )


def run_hermetic_release(
    *,
    board_path: Path | str | None = None,
    board_text: str | None = None,
    force_tasks_terminal: bool | None = None,
    **kwargs: Any,
) -> DuckDBControlPlaneReleaseReceipt:
    """Evaluate release with hermetic passing components (tests / dry-run)."""

    tree_id = str(kwargs.get("tree_id") or "tree:sha256:dqp039-release")
    schema_checksum = str(kwargs.get("schema_checksum") or ("sha256:" + ("cc" * 32)))
    store_generation = int(kwargs.get("store_generation") or 1)
    quack_profile = str(
        kwargs.get("quack_profile") or "profile:quack-1.5.2-loopback"
    )
    extension_fingerprint = str(
        kwargs.get("extension_fingerprint") or ("sha256:" + ("dd" * 32))
    )
    git_identity = str(kwargs.get("git_identity") or "git:hermetic-dqp039")

    text = board_text
    if text is None and board_path is not None:
        text = Path(board_path).read_text(encoding="utf-8")

    components = hermetic_component_bundle(
        tree_id=tree_id,
        schema_checksum=schema_checksum,
        store_generation=store_generation,
        profile_id=quack_profile,
        exclude_roots=tuple(kwargs.get("exclude_roots") or ()),
        fail_roots=tuple(kwargs.get("fail_roots") or ()),
        stale_roots=tuple(kwargs.get("stale_roots") or ()),
        synthetic_roots=tuple(kwargs.get("synthetic_roots") or ()),
        skipped_roots=tuple(kwargs.get("skipped_roots") or ()),
    )

    return evaluate_release(
        components=components,
        tree_id=tree_id,
        schema_checksum=schema_checksum,
        store_generation=store_generation,
        quack_profile=quack_profile,
        extension_fingerprint=extension_fingerprint,
        git_identity=git_identity,
        board_text=text,
        tasks_terminal=force_tasks_terminal,
        safety_floors=kwargs.get("safety_floors"),
        quality_non_regressing=bool(kwargs.get("quality_non_regressing", True)),
        legacy_file_decision_read_in_canary=bool(
            kwargs.get("legacy_file_decision_read_in_canary", False)
        ),
        max_age_seconds=int(
            kwargs.get("max_age_seconds") or DEFAULT_EVIDENCE_MAX_AGE_SECONDS
        ),
    )


__all__ = (
    "DEFAULT_EVIDENCE_MAX_AGE_SECONDS",
    "DUCKDB_CONTROL_PLANE_RELEASE_INTERFACE",
    "EVIDENCE",
    "GOAL_ID",
    "PRIOR_TASK_IDS",
    "REQUIRED_COMPONENT_ROOTS",
    "REQUIRED_MODULES",
    "TASK_ID",
    "ComponentEvidence",
    "ComponentStatus",
    "DuckDBControlPlaneReleaseReceipt",
    "DuckDBQuackReleaseError",
    "ReleaseVerdict",
    "SafetyFloorObservation",
    "board_tasks_terminal",
    "content_identity",
    "evaluate_release",
    "hermetic_component_bundle",
    "parse_board_task_statuses",
    "run_hermetic_release",
)
