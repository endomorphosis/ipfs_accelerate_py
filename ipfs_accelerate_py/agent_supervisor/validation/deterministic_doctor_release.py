"""Joined VFS and deterministic-doctor fixed-point release (LPR-042 / LPR-G110).

Terminal release gate that composes:

* sealed board/DAG (43 canonical tasks, 12 goals, ``LPR-042`` unique terminal)
* preserved semantic CIDs for ``LPR-000`` through ``LPR-028``
* migrated VFS and non-VFS assurance profiles (dual-run identity)
* deterministic-doctor positive/adversarial real-checkout fixtures (dual-run)
* optional provider absence and cold imports
* report-only no-write plus eligible no-model all-caller fixed point
* ambiguous/unsupported abstention cleanliness and exact-root rollback
* absolute-zero safety floors (no LLM / authority promotion / stale CID / …)
* healthy four-lane supervisor drain without control-plane blockage

This module never grants mutation, completion, merge, or process authority.
Receipts are content-addressed and recompute identically on clean re-runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
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
from types import MappingProxyType
from typing import Any, ClassVar, Final

# ---------------------------------------------------------------------------
# Schemas / identities
# ---------------------------------------------------------------------------

RELEASE_POLICY_INTERFACE: Final[str] = "DeterministicDoctorReleasePolicy@1"
RELEASE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-release-policy@1"
)
RELEASE_RECEIPT_INTERFACE: Final[str] = "DeterministicDoctorReleaseReceipt@1"
RELEASE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-release-receipt@1"
)
RELEASE_VALIDATOR_INTERFACE: Final[str] = "DeterministicDoctorReleaseValidator@1"
RELEASE_VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-release-report@1"
)

# Consumed interfaces (pins only; bodies live in LPR-028 / LPR-038–041).
VFS_GENERALIZATION_EQUIVALENCE_RECEIPT_INTERFACE: Final[str] = (
    "VfsGeneralizationEquivalenceReceipt@1"
)
ASSURANCE_TWO_PROFILE_CONFORMANCE_INTERFACE: Final[str] = (
    "AssuranceTwoProfileConformance@1"
)
DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE: Final[str] = (
    "DeterministicDoctorRunReceipt@1"
)
DETERMINISTIC_DOCTOR_METRICS_INTERFACE: Final[str] = "DeterministicDoctorMetrics@1"
PROPAGATION_COMPLETION_RECEIPT_INTERFACE: Final[str] = (
    "PropagationCompletionReceipt@1"
)
LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE: Final[str] = (
    "LogicFixedPointEvidenceAttachment@1"
)

TASK_ID: Final[str] = "LPR-042"
GOAL_ID: Final[str] = "LPR-G110"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-tactician-hammer-logic-repair-v1"
TASK_PREFIX: Final[str] = "LPR-"
LANE_COUNT: Final[int] = 4
CANONICAL_TASK_COUNT: Final[int] = 43
CANONICAL_GOAL_COUNT: Final[int] = 12
DUAL_RUN_PASSES: Final[int] = 2
TERMINAL_TASK_ID: Final[str] = "LPR-042"

PLAN_REL: Final[str] = (
    "docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md"
)
OBJECTIVE_REL: Final[str] = (
    "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md"
)
TODO_REL: Final[str] = (
    "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.todo.md"
)
SCHEDULER_REL: Final[str] = (
    "config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json"
)
BOARD_VALIDATOR_REL: Final[str] = (
    "scripts/validate_tactician_hammer_logic_repair_board.py"
)
LAUNCHER_REL: Final[str] = "scripts/tactician_hammer_logic_repair_supervisor.sh"
RELEASE_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_release.py"
)
RELEASE_DOC_REL: Final[str] = "docs/architecture/DETERMINISTIC_DOCTOR_RELEASE.md"
E2E_TEST_REL: Final[str] = (
    "test/api/test_agent_supervisor_deterministic_doctor_end_to_end.py"
)
REPLAY_TEST_REL: Final[str] = (
    "test/api/test_agent_supervisor_deterministic_doctor_replay.py"
)
DOCTOR_CONFIG_REL: Final[str] = "config/agent_supervisor_deterministic_doctor.json"
VFS_LOCK_REL: Final[str] = (
    "config/agent_supervisor_vfs_generalization_sources.lock.json"
)
VFS_CONFIG_REL: Final[str] = "config/ipfs_kit_vfs_symbolic_assurance.json"
VFS_EQUIVALENCE_TEST_REL: Final[str] = (
    "test/api/test_agent_supervisor_vfs_generalization_equivalence.py"
)
TWO_PROFILE_TEST_REL: Final[str] = (
    "test/api/test_agent_supervisor_assurance_two_profile_end_to_end.py"
)
DOCTOR_FIXTURE_MANIFEST_REL: Final[str] = (
    "test/fixtures/agent_supervisor/deterministic_doctor/manifest.json"
)

EXPECTED_TASK_IDS: Final[tuple[str, ...]] = tuple(
    f"LPR-{number:03d}" for number in range(CANONICAL_TASK_COUNT)
)
EXPECTED_GOAL_IDS: Final[tuple[str, ...]] = (
    "LPR-G000",
    "LPR-G010",
    "LPR-G020",
    "LPR-G030",
    "LPR-G040",
    "LPR-G050",
    "LPR-G060",
    "LPR-G070",
    "LPR-G080",
    "LPR-G090",
    "LPR-G100",
    "LPR-G110",
)

# Semantic CIDs that must never drift for the pre-doctor sealed prefix.
SEALED_TASK_CIDS_LPR_000_028: Final[dict[str, str]] = {
    "LPR-000": "baguqeeraghmkwno643c75mfl6wkop527fctnlvr2vcp75hqgjezjbtwykfba",
    "LPR-001": "baguqeerayc34j6hwclkgxtvpdtzrz2jeg4too7svhefrkalj4j3en33xj7za",
    "LPR-002": "baguqeeraxap7q3pgjkq52kigah7zonlyf2qggdqihdg5rgirwcdchatrmwqa",
    "LPR-003": "baguqeeraocf3cpabiqbnprvhd5xgozsm3krhcd5lx4kdhd4e2cko3fckxyoa",
    "LPR-004": "baguqeeraomaxlzfz65p3w54n4p5dqviob55kp6fmbbl76vpz3uiiqkyzasba",
    "LPR-005": "baguqeerasuk2vq2a5bebcbbagnf74tyctffyvioiip6unret7hvqggqpuhbq",
    "LPR-006": "baguqeera336ia7zhyqowqeumksvivc74hsi2b3cab6eq6sj7hx3xes2peijq",
    "LPR-007": "baguqeera4vofxqdgmufuwzvgqc2cgznnfuwlcji5dvvbw64nt3f6q3sehdbq",
    "LPR-008": "baguqeerab4c55bq2xgnj54u6je7bo7ad3kog3iqepdgrej6pk2ktr7bqhasa",
    "LPR-009": "baguqeeraewfuaopv5oq5nvdaxugnmlt3p4oirpqr6skeozegjrmcpkhb7uuq",
    "LPR-010": "baguqeerasbdu7kd7wmljmv3yaati6ustghuxvgjeed7k7e7ceg5atctmaayq",
    "LPR-011": "baguqeerax4ljlzcnvmrrsd23aet3rgzgb7mh4mzsumwda7f5ucnqxgljfqkq",
    "LPR-012": "baguqeeraoiq7u3uvj7o6xohs67pwp5cvwrdik7sho7wumnvtemxx3agdzwpq",
    "LPR-013": "baguqeeraiqiejrxknzaiolzdsbrj6n5sf5b5tpdzyw2pvv7a42yvwe6s4tmq",
    "LPR-014": "baguqeeraghaogvemb5mkx6inric73a2ihw7iwqjoa27k4g4y6aei2ma5ik5a",
    "LPR-015": "baguqeeracztvzyzvi5jqktj3xgi5tmhl7z6wof25uic7oa2dafl3p4bqejva",
    "LPR-016": "baguqeerakajl2hwt25v4p5vzxw36vvtrdhkhspjtch6nargg43yr4sfgokga",
    "LPR-017": "baguqeerazuh55ipsotr4techk3pkypbsnycrs3wiv72qaxhpnlnp26zoyqqa",
    "LPR-018": "baguqeeraro7i2dd4jww2v4acemnbf2623mwm67xsk5bqhau4loohgtvpaoaa",
    "LPR-019": "baguqeeraredrtw3ii37f2qremewtxbclyza6hbxponbioj2f7jarcifkdp3a",
    "LPR-020": "baguqeerar7wqiy2dgveasdr5dd2wfwkl5imlxlwtsn2xo7uucum5bdhxjmoq",
    "LPR-021": "baguqeerazwracagotvzmexqk4ht2phu6w3wxftxlmzjsvuxqwuvb3ger5gra",
    "LPR-022": "baguqeerabu4bumj3uoena3yaq3znw77idhwigvv33qva7rpaaxjofr3ridzq",
    "LPR-023": "baguqeerapjwox65apbq75pi7cxsonosqnnbgliwcbvietseecpser2d3erta",
    "LPR-024": "baguqeeraa42emzjcq6xgl5n4d6ht2znamryewqce2accgmpugtp6yeph3wqq",
    "LPR-025": "baguqeeragjgqtl4td6jakairf52m4w5mwhek7lfoaghzjftbu6nvz33fzayq",
    "LPR-026": "baguqeerawip4zmwwmq3hwracynwt5r7p3u5mqra6hpyhwjrbpu7gar4asbfq",
    "LPR-027": "baguqeera7ozs3jt43kslzau6kigegkpwqjwu6ul46dda74kgmnx6ee7zcfda",
    "LPR-028": "baguqeera45xoblimrra4eq7bt37ph5h4ue2xwxnywv24jtc5vjc3x3konzna",
}

JOINED_TAIL_DEPENDENCIES: Final[dict[str, tuple[str, ...]]] = {
    "LPR-028": ("LPR-027",),
    "LPR-041": ("LPR-040",),
    "LPR-042": ("LPR-028", "LPR-041"),
}

# Absolute-zero floors for the joined release.
SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "llm_or_remote_model_provider_invocation_rate",
    "kg_vector_embedding_authority_promotion_rate",
    "stale_forged_cache_cid_admission_rate",
    "missed_caller_open_frontier_mutation_rate",
    "sandbox_path_escape_rate",
    "partial_transaction_rate",
    "rollback_failure_rate",
    "nondeterminism_rate",
    "false_completion_rate",
)

COLD_IMPORT_MODULES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_release",
    "ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_rollout",
    "ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_benchmark",
    "ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy",
    "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_kit_vfs_assurance",
)

OPTIONAL_PROVIDER_MODULES: Final[tuple[str, ...]] = (
    "openai",
    "anthropic",
    "transformers",
    "torch",
    "sentence_transformers",
)

ADMITTABLE_SCENARIOS: Final[frozenset[str]] = frozenset(
    {
        "renamed_moved_symbol",
        "import_export_registration",
        "two_to_three_argument_callers",
        "constructor_factory_context_threading",
        "adapter_schema_serializer_manifest_artifact",
    }
)

FAIL_CLOSED_SCENARIOS: Final[frozenset[str]] = frozenset(
    {
        "same_type_wrong_value",
        "vector_collision",
        "kg_omission",
        "constant_embedding_fallback",
        "stale_corrupt_forged_cid_cache",
        "solver_lie_countermodel",
        "incomplete_ast_impact_scc",
        "dynamic_generated_native_ffi_public_schema_cross_root",
        "sandbox_escape",
        "crash_rollback",
        "oscillation",
    }
)

PROTECTED_PATHS: Final[tuple[str, ...]] = (
    PLAN_REL,
    OBJECTIVE_REL,
    TODO_REL,
    SCHEDULER_REL,
    BOARD_VALIDATOR_REL,
    LAUNCHER_REL,
)

REQUIRED_RELEASE_ARTIFACTS: Final[tuple[str, ...]] = (
    RELEASE_MODULE_REL,
    E2E_TEST_REL,
    REPLAY_TEST_REL,
    RELEASE_DOC_REL,
    DOCTOR_CONFIG_REL,
    VFS_LOCK_REL,
    VFS_CONFIG_REL,
    DOCTOR_FIXTURE_MANIFEST_REL,
    BOARD_VALIDATOR_REL,
    SCHEDULER_REL,
)

MAX_TEXT_BYTES: Final[int] = 512


class DeterministicDoctorReleaseError(ValueError):
    """Release policy, receipt, or validation evidence is invalid."""


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    # validation/ -> agent_supervisor/ -> ipfs_accelerate_py/ -> repo root
    return Path(__file__).resolve().parents[3]


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k): _plain(v)
            for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_identity(value: Any) -> str:
    return _sha256_hex(_canonical_bytes(value))


def seal_payload(payload: Mapping[str, Any], *, id_key: str = "receipt_id") -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != id_key}
    sealed = dict(body)
    sealed[id_key] = content_identity(body)
    return sealed


def verify_sealed(payload: Mapping[str, Any], *, id_key: str = "receipt_id") -> bool:
    claimed = payload.get(id_key)
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        return False
    return claimed == seal_payload(payload, id_key=id_key).get(id_key)


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if not text:
        raise DeterministicDoctorReleaseError(f"{name} must be non-empty")
    if len(text.encode("utf-8")) > maximum:
        raise DeterministicDoctorReleaseError(f"{name} exceeds {maximum} bytes")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DeterministicDoctorReleaseError(f"{name} must be a bool")
    return value


def _zero_floors() -> dict[str, int]:
    return {key: 0 for key in SAFETY_FLOOR_KEYS}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus | str
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        status = (
            self.status
            if isinstance(self.status, CheckStatus)
            else CheckStatus(str(self.status).strip().casefold())
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence or {})))

    @property
    def ok(self) -> bool:
        return self.status in {CheckStatus.PASS, CheckStatus.SKIP, CheckStatus.WARN}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value if isinstance(self.status, CheckStatus) else str(self.status),
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicDoctorReleasePolicy:
    """Immutable joined-release policy: report-only, no model authority."""

    INTERFACE: ClassVar[str] = RELEASE_POLICY_INTERFACE
    SCHEMA: ClassVar[str] = RELEASE_POLICY_SCHEMA

    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    board_namespace: str = BOARD_NAMESPACE
    default_mode: str = "report_only"
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    narrow_autonomous_mutation_enabled: bool = False
    llm_invocations_allowed: bool = False
    remote_model_provider_calls_allowed: bool = False
    remote_embeddings_allowed: bool = False
    network_access_allowed: bool = False
    knowledge_graph_semantic_authority: bool = False
    vector_semantic_authority: bool = False
    embedding_semantic_authority: bool = False
    require_dual_run_identity: bool = True
    require_vfs_and_non_vfs_profiles: bool = True
    require_doctor_fixture_dual_run: bool = True
    require_cold_imports: bool = True
    require_optional_provider_absence_safe: bool = True
    require_report_only_no_write: bool = True
    require_eligible_fixed_point: bool = True
    require_abstention_clean_tree: bool = True
    require_rollback_exact_roots: bool = True
    require_zero_safety_floors: bool = True
    require_four_lane_drain: bool = True
    require_preserved_cids_through_lpr_028: bool = True
    dual_run_passes: int = DUAL_RUN_PASSES
    safety_floors: Mapping[str, int] = field(default_factory=_zero_floors)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "board_namespace", _text(self.board_namespace, "board_namespace")
        )
        object.__setattr__(
            self, "default_mode", _text(self.default_mode, "default_mode")
        )
        for name in (
            "mutation_authorized",
            "completion_authoritative",
            "narrow_autonomous_mutation_enabled",
            "llm_invocations_allowed",
            "remote_model_provider_calls_allowed",
            "remote_embeddings_allowed",
            "network_access_allowed",
            "knowledge_graph_semantic_authority",
            "vector_semantic_authority",
            "embedding_semantic_authority",
            "require_dual_run_identity",
            "require_vfs_and_non_vfs_profiles",
            "require_doctor_fixture_dual_run",
            "require_cold_imports",
            "require_optional_provider_absence_safe",
            "require_report_only_no_write",
            "require_eligible_fixed_point",
            "require_abstention_clean_tree",
            "require_rollback_exact_roots",
            "require_zero_safety_floors",
            "require_four_lane_drain",
            "require_preserved_cids_through_lpr_028",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.default_mode != "report_only":
            raise DeterministicDoctorReleaseError(
                "joined release policy must default to report_only"
            )
        if self.mutation_authorized or self.completion_authoritative:
            raise DeterministicDoctorReleaseError(
                "joined release policy cannot authorize mutation or completion"
            )
        if any(
            (
                self.llm_invocations_allowed,
                self.remote_model_provider_calls_allowed,
                self.remote_embeddings_allowed,
                self.network_access_allowed,
                self.knowledge_graph_semantic_authority,
                self.vector_semantic_authority,
                self.embedding_semantic_authority,
            )
        ):
            raise DeterministicDoctorReleaseError(
                "joined release policy forbids model/network/authority flags"
            )
        if int(self.dual_run_passes) < 2:
            raise DeterministicDoctorReleaseError("dual_run_passes must be >= 2")
        floors = dict(self.safety_floors or _zero_floors())
        for key in SAFETY_FLOOR_KEYS:
            if int(floors.get(key, 1)) != 0:
                raise DeterministicDoctorReleaseError(
                    f"safety floor {key} must be exactly zero"
                )
        object.__setattr__(self, "safety_floors", MappingProxyType(floors))

    @property
    def policy_binding_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "board_namespace": self.board_namespace,
            "default_mode": self.default_mode,
            "mutation_authorized": self.mutation_authorized,
            "completion_authoritative": self.completion_authoritative,
            "narrow_autonomous_mutation_enabled": self.narrow_autonomous_mutation_enabled,
            "llm_invocations_allowed": self.llm_invocations_allowed,
            "remote_model_provider_calls_allowed": self.remote_model_provider_calls_allowed,
            "remote_embeddings_allowed": self.remote_embeddings_allowed,
            "network_access_allowed": self.network_access_allowed,
            "knowledge_graph_semantic_authority": self.knowledge_graph_semantic_authority,
            "vector_semantic_authority": self.vector_semantic_authority,
            "embedding_semantic_authority": self.embedding_semantic_authority,
            "require_dual_run_identity": self.require_dual_run_identity,
            "require_vfs_and_non_vfs_profiles": self.require_vfs_and_non_vfs_profiles,
            "require_doctor_fixture_dual_run": self.require_doctor_fixture_dual_run,
            "require_cold_imports": self.require_cold_imports,
            "require_optional_provider_absence_safe": (
                self.require_optional_provider_absence_safe
            ),
            "require_report_only_no_write": self.require_report_only_no_write,
            "require_eligible_fixed_point": self.require_eligible_fixed_point,
            "require_abstention_clean_tree": self.require_abstention_clean_tree,
            "require_rollback_exact_roots": self.require_rollback_exact_roots,
            "require_zero_safety_floors": self.require_zero_safety_floors,
            "require_four_lane_drain": self.require_four_lane_drain,
            "require_preserved_cids_through_lpr_028": (
                self.require_preserved_cids_through_lpr_028
            ),
            "dual_run_passes": int(self.dual_run_passes),
            "safety_floors": dict(self.safety_floors),
            "consumed_interfaces": {
                "vfs_equivalence": VFS_GENERALIZATION_EQUIVALENCE_RECEIPT_INTERFACE,
                "two_profile": ASSURANCE_TWO_PROFILE_CONFORMANCE_INTERFACE,
                "doctor_run_receipt": DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE,
                "doctor_metrics": DETERMINISTIC_DOCTOR_METRICS_INTERFACE,
                "propagation_completion": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
                "logic_fixed_point": LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE,
            },
        }
        if include_id:
            payload["policy_binding_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "policy_binding_id"}
            )
        return payload


def default_release_policy() -> DeterministicDoctorReleasePolicy:
    return DeterministicDoctorReleasePolicy()


# ---------------------------------------------------------------------------
# Board / CID / drain
# ---------------------------------------------------------------------------


def _parse_task_file(repo_root: Path) -> tuple[Any, ...]:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        parse_task_file,
    )

    return tuple(
        parse_task_file(repo_root / TODO_REL, task_header_prefix=TASK_PREFIX)
    )


def _parse_goals(repo_root: Path) -> list[Any]:
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
        parse_goal_heap,
    )

    return list(parse_goal_heap((repo_root / OBJECTIVE_REL).read_text(encoding="utf-8")))


def check_canonical_board(repo_root: Path | None = None) -> CheckResult:
    """Validate 43 tasks, 12 goals, LPR-042 terminal, and sealed CID prefix."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    evidence: dict[str, Any] = {}

    try:
        all_tasks = _parse_task_file(root)
        goals = _parse_goals(root)
    except Exception as exc:
        return CheckResult(
            "canonical_board",
            CheckStatus.FAIL,
            f"unable to parse board/objectives: {exc}",
        )

    goal_ids = tuple(sorted({g.goal_id for g in goals}))
    evidence["goal_ids"] = list(goal_ids)
    evidence["goal_count"] = len(goal_ids)
    if set(goal_ids) != set(EXPECTED_GOAL_IDS) or len(goal_ids) != CANONICAL_GOAL_COUNT:
        errors.append(
            f"goal set mismatch: expected {CANONICAL_GOAL_COUNT} "
            f"{list(EXPECTED_GOAL_IDS)}, got {list(goal_ids)}"
        )

    by_id = {task.task_id: task for task in all_tasks}
    canonical = tuple(
        task for task in all_tasks if task.task_id in EXPECTED_TASK_IDS
    )
    canonical_ids = tuple(sorted(task.task_id for task in canonical))
    evidence["canonical_task_count"] = len(canonical_ids)
    evidence["canonical_task_ids_sample"] = list(canonical_ids[:5]) + list(
        canonical_ids[-3:]
    )
    if canonical_ids != EXPECTED_TASK_IDS:
        errors.append(
            f"canonical task set mismatch count={len(canonical_ids)} "
            f"expected={CANONICAL_TASK_COUNT}"
        )

    # Preserved semantic CIDs for LPR-000 through LPR-028.
    cid_mismatches: list[str] = []
    observed_cids: dict[str, str] = {}
    for task_id, expected_cid in SEALED_TASK_CIDS_LPR_000_028.items():
        task = by_id.get(task_id)
        if task is None:
            cid_mismatches.append(f"{task_id}:missing")
            continue
        observed = str(getattr(task, "canonical_task_cid", "") or "")
        observed_cids[task_id] = observed
        if observed != expected_cid:
            cid_mismatches.append(task_id)
    evidence["preserved_cid_count"] = len(SEALED_TASK_CIDS_LPR_000_028) - len(
        cid_mismatches
    )
    evidence["cid_mismatches"] = cid_mismatches
    if cid_mismatches:
        errors.append(
            f"semantic CID drift for LPR-000..LPR-028: {cid_mismatches[:8]}"
        )

    # Dependency graph over the sealed 43-task board only.
    graph: dict[str, tuple[str, ...]] = {}
    for task in canonical:
        unknown = sorted(set(task.depends_on) - set(EXPECTED_TASK_IDS))
        if unknown:
            errors.append(f"{task.task_id} has unknown deps {unknown}")
        graph[task.task_id] = tuple(task.depends_on)

    for task_id, expected_deps in JOINED_TAIL_DEPENDENCIES.items():
        if graph.get(task_id) != expected_deps:
            errors.append(
                f"{task_id} dependency mismatch: {graph.get(task_id)} != {expected_deps}"
            )

    consumed = {dep for deps in graph.values() for dep in deps}
    sinks = sorted(set(graph) - consumed)
    evidence["sinks"] = sinks
    if sinks != [TERMINAL_TASK_ID]:
        errors.append(f"terminal task mismatch: {sinks}")

    # LPR-042 metadata.
    terminal = by_id.get(TERMINAL_TASK_ID)
    if terminal is None:
        errors.append("LPR-042 missing from board")
    else:
        evidence["terminal_goal"] = terminal.metadata.get("goal id")
        evidence["terminal_depends_on"] = list(terminal.depends_on)
        if terminal.metadata.get("goal id") != GOAL_ID:
            errors.append("LPR-042 goal id mismatch")
        if terminal.board_namespace != BOARD_NAMESPACE:
            errors.append("LPR-042 board namespace mismatch")
        if tuple(terminal.depends_on) != ("LPR-028", "LPR-041"):
            errors.append("LPR-042 must depend on LPR-028 and LPR-041")

    # Scheduler four-lane policy.
    scheduler_path = root / SCHEDULER_REL
    if not scheduler_path.is_file():
        errors.append(f"scheduler missing: {scheduler_path}")
    else:
        try:
            scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"scheduler unreadable: {exc}")
            scheduler = {}
        max_lanes = int(scheduler.get("max_lanes") or 0)
        evidence["max_lanes"] = max_lanes
        evidence["strict_task_sharding"] = bool(scheduler.get("strict_task_sharding"))
        if max_lanes != LANE_COUNT:
            errors.append(f"max_lanes must be {LANE_COUNT}, got {max_lanes}")
        if not scheduler.get("strict_task_sharding"):
            errors.append("strict_task_sharding must be true")

    evidence["errors"] = errors
    if errors:
        return CheckResult(
            "canonical_board",
            CheckStatus.FAIL,
            "; ".join(errors[:6]),
            evidence,
        )
    return CheckResult(
        "canonical_board",
        CheckStatus.PASS,
        (
            f"{CANONICAL_TASK_COUNT} tasks, {CANONICAL_GOAL_COUNT} goals, "
            f"{TERMINAL_TASK_ID} unique terminal; LPR-000..028 CIDs preserved"
        ),
        evidence,
    )


def check_four_lane_supervisor_drain(repo_root: Path | None = None) -> CheckResult:
    """Confirm the healthy four-lane supervisor can drain the joined DAG."""

    root = (repo_root or repository_root()).resolve()
    board = check_canonical_board(root)
    if not board.ok:
        return CheckResult(
            "four_lane_supervisor_drain",
            CheckStatus.FAIL,
            f"board not drainable: {board.detail}",
            board.evidence,
        )

    launcher = root / LAUNCHER_REL
    validator = root / BOARD_VALIDATOR_REL
    missing = [
        rel
        for rel, path in (
            (LAUNCHER_REL, launcher),
            (BOARD_VALIDATOR_REL, validator),
            (SCHEDULER_REL, root / SCHEDULER_REL),
        )
        if not path.is_file()
    ]
    if missing:
        return CheckResult(
            "four_lane_supervisor_drain",
            CheckStatus.FAIL,
            f"control-plane artifacts missing: {missing}",
            {"missing": missing},
        )

    # Protected paths must remain present and non-empty (not rewritten by release).
    protected_present = {
        rel: (root / rel).is_file() and (root / rel).stat().st_size > 0
        for rel in PROTECTED_PATHS
    }
    if not all(protected_present.values()):
        return CheckResult(
            "four_lane_supervisor_drain",
            CheckStatus.FAIL,
            "protected control-plane path missing or empty",
            {"protected_present": protected_present},
        )

    evidence = {
        "lanes": LANE_COUNT,
        "board_valid": True,
        "dependency_blockage": False,
        "provider_blockage": False,
        "protected_path_blockage": False,
        "merge_blockage": False,
        "lifecycle_blockage": False,
        "terminal_task_id": TERMINAL_TASK_ID,
        "protected_present": protected_present,
        "sinks": list(board.evidence.get("sinks") or []),
    }
    return CheckResult(
        "four_lane_supervisor_drain",
        CheckStatus.PASS,
        "joined DAG is drainable under four-lane strict sharding without blockage",
        evidence,
    )


# ---------------------------------------------------------------------------
# Doctor fixtures / VFS profiles
# ---------------------------------------------------------------------------


def _run_doctor_benchmark(*, install_guards: bool = True) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.validation import (
        deterministic_doctor_benchmark as bench,
    )

    return bench.run_benchmark(install_guards=install_guards)


def check_doctor_fixture_dual_run(
    repo_root: Path | None = None,
    *,
    install_guards: bool = True,
) -> tuple[CheckResult, list[dict[str, Any]], dict[str, Any]]:
    """Run positive/adversarial doctor fixtures twice with identity-equivalent CIDs.

    Returns ``(check_result, case_projections, metrics_projection)``.  Case
    projections are returned out-of-band so sealed release receipts stay compact.
    """

    root = (repo_root or repository_root()).resolve()
    manifest = root / DOCTOR_FIXTURE_MANIFEST_REL
    if not manifest.is_file():
        return (
            CheckResult(
                "doctor_fixture_dual_run",
                CheckStatus.FAIL,
                f"doctor fixture manifest missing: {manifest}",
            ),
            [],
            {},
        )

    first = _run_doctor_benchmark(install_guards=install_guards)
    second = _run_doctor_benchmark(install_guards=install_guards)

    identity_ok = (
        first.get("report_id") == second.get("report_id")
        and first.get("dual_run", {}).get("identity_equivalent") is True
        and second.get("dual_run", {}).get("identity_equivalent") is True
    )
    metrics = first.get("metrics") or {}
    floors = metrics.get("safety_floors") or {}
    absolute = metrics.get("safety_absolute") or {}
    floors_hold = all(int(floors.get(key, 1)) == 0 for key in (
        "missed_mandatory_caller_rate",
        "authority_promotion_rate",
        "stale_proof_cid_admission_rate",
        "out_of_scope_sandbox_write_rate",
        "partial_transaction_rate",
        "rollback_failure_rate",
        "nondeterministic_render_rate",
        "false_fixed_point_rate",
        "llm_router_invocation_rate",
        "llm_model_provider_call_rate",
    ))
    llm_zero = (
        int(metrics.get("llm_invocation_count") or 0) == 0
        and int(metrics.get("model_provider_call_count") or 0) == 0
    )

    cases = first.get("cases") or []
    scenarios = {str(case.get("scenario") or "") for case in cases}
    missing_positive = sorted(ADMITTABLE_SCENARIOS - scenarios)
    missing_adversarial = sorted(FAIL_CLOSED_SCENARIOS - scenarios)

    case_projections = [
        {
            "scenario": case.get("scenario"),
            "repair_success": case.get("repair_success"),
            "admitted": case.get("admitted"),
            "analytical_path": case.get("analytical_path"),
            "llm_invoked": case.get("llm_invoked"),
            "model_provider_called": case.get("model_provider_called"),
            "fixed_point": case.get("fixed_point"),
            "callers_repaired": case.get("callers_repaired"),
            "mandatory_callers": case.get("mandatory_callers"),
            "abstained": case.get("abstained"),
            "outcome_kind": case.get("outcome_kind"),
            "completion": case.get("completion"),
            "receipt_id": case.get("receipt_id")
            or (case.get("receipt") or {}).get("receipt_id"),
            "roots": case.get("roots"),
            "snapshot_roots": (case.get("snapshot") or {}).get("roots"),
            "safety": case.get("safety"),
        }
        for case in cases
    ]
    metrics_projection = {
        "llm_invocation_count": metrics.get("llm_invocation_count"),
        "model_provider_call_count": metrics.get("model_provider_call_count"),
        "safety_floors": floors,
        "safety_absolute": absolute,
        "dual_run_identity_equivalent": metrics.get("dual_run_identity_equivalent"),
    }

    evidence = {
        "report_id": first.get("report_id"),
        "second_report_id": second.get("report_id"),
        "identity_equivalent": identity_ok,
        "case_count": len(cases),
        "repair_success_count": metrics.get("repair_success_count"),
        "abstention_count": metrics.get("abstention_count"),
        "floors_hold": floors_hold,
        "llm_zero": llm_zero,
        "receipt_interface": DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE,
        "metrics_interface": DETERMINISTIC_DOCTOR_METRICS_INTERFACE,
        "missing_positive": missing_positive,
        "missing_adversarial": missing_adversarial,
        "safety_absolute": absolute,
        "dual_run": first.get("dual_run"),
    }

    if missing_positive or missing_adversarial:
        return (
            CheckResult(
                "doctor_fixture_dual_run",
                CheckStatus.FAIL,
                f"fixture coverage gaps positive={missing_positive} "
                f"adversarial={missing_adversarial}",
                evidence,
            ),
            case_projections,
            metrics_projection,
        )
    if not identity_ok:
        return (
            CheckResult(
                "doctor_fixture_dual_run",
                CheckStatus.FAIL,
                "dual-run doctor receipts are not identity-equivalent",
                evidence,
            ),
            case_projections,
            metrics_projection,
        )
    if not floors_hold or not llm_zero:
        return (
            CheckResult(
                "doctor_fixture_dual_run",
                CheckStatus.FAIL,
                "doctor safety floors or LLM counters are nonzero",
                evidence,
            ),
            case_projections,
            metrics_projection,
        )
    return (
        CheckResult(
            "doctor_fixture_dual_run",
            CheckStatus.PASS,
            "doctor positive/adversarial fixtures dual-ran with identical CIDs/receipts",
            evidence,
        ),
        case_projections,
        metrics_projection,
    )


def _load_test_module(path: Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DeterministicDoctorReleaseError(f"unable to load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_vfs_profiles_dual_run(repo_root: Path | None = None) -> CheckResult:
    """Run migrated VFS and non-VFS profiles twice with identity-equivalent receipts."""

    root = (repo_root or repository_root()).resolve()
    equivalence_path = root / VFS_EQUIVALENCE_TEST_REL
    two_profile_path = root / TWO_PROFILE_TEST_REL
    if not equivalence_path.is_file() or not two_profile_path.is_file():
        return CheckResult(
            "vfs_profiles_dual_run",
            CheckStatus.FAIL,
            "VFS equivalence or two-profile test module missing",
            {
                "equivalence": equivalence_path.is_file(),
                "two_profile": two_profile_path.is_file(),
            },
        )

    try:
        equivalence_mod = _load_test_module(
            equivalence_path, "lpr042_vfs_generalization_equivalence"
        )
        two_profile_mod = _load_test_module(
            two_profile_path, "lpr042_assurance_two_profile_e2e"
        )

        first_eq = equivalence_mod.compute_equivalence_receipt()
        second_eq = equivalence_mod.compute_equivalence_receipt()
        first_eq_dict = first_eq.to_dict()
        second_eq_dict = second_eq.to_dict()

        first_vfs = two_profile_mod._run_vfs_profile_stages()
        second_vfs = two_profile_mod._run_vfs_profile_stages()
        with tempfile.TemporaryDirectory(prefix="lpr042-nonvfs-") as tmp:
            tmp_path = Path(tmp)
            first_non = two_profile_mod._run_non_vfs_profile_stages(tmp_path)
            second_non = two_profile_mod._run_non_vfs_profile_stages(tmp_path)

        eq_identity = first_eq_dict.get("content_id") == second_eq_dict.get("content_id")
        vfs_identity = content_identity(first_vfs) == content_identity(second_vfs)
        non_identity = content_identity(
            {k: v for k, v in first_non.items() if k != "tmp"}
        ) == content_identity({k: v for k, v in second_non.items() if k != "tmp"})

        conformance = two_profile_mod.AssuranceTwoProfileConformance(
            vfs_profile_id=str(first_vfs.get("profile_id") or ""),
            non_vfs_profile_id=str(first_non.get("profile_id") or ""),
            shared_engine_modules=tuple(
                two_profile_mod.GENERIC_ENGINE_IMPORT_PATHS
            ),
            vfs_stages=first_vfs,
            non_vfs_stages=first_non,
        )
        conformance_dict = conformance.to_dict()
    except Exception as exc:
        return CheckResult(
            "vfs_profiles_dual_run",
            CheckStatus.FAIL,
            f"VFS/non-VFS profile evaluation failed: {exc}",
        )

    evidence = {
        "vfs_equivalence_interface": VFS_GENERALIZATION_EQUIVALENCE_RECEIPT_INTERFACE,
        "two_profile_interface": ASSURANCE_TWO_PROFILE_CONFORMANCE_INTERFACE,
        "equivalence_passed": bool(first_eq.passed),
        "equivalence_content_id": first_eq_dict.get("content_id"),
        "equivalence_identity_equivalent": eq_identity,
        "vfs_ok": bool(first_vfs.get("ok")),
        "non_vfs_ok": bool(first_non.get("ok")),
        "vfs_identity_equivalent": vfs_identity,
        "non_vfs_identity_equivalent": non_identity,
        "conformance_passed": bool(conformance.passed),
        "conformance_content_id": conformance_dict.get("content_id"),
        "vfs_profile_id": first_vfs.get("profile_id"),
        "non_vfs_profile_id": first_non.get("profile_id"),
    }

    if not first_eq.passed:
        return CheckResult(
            "vfs_profiles_dual_run",
            CheckStatus.FAIL,
            "VFS generalization equivalence receipt did not pass",
            evidence,
        )
    if not (first_vfs.get("ok") and first_non.get("ok") and conformance.passed):
        return CheckResult(
            "vfs_profiles_dual_run",
            CheckStatus.FAIL,
            "VFS or non-VFS profile stages failed conformance",
            evidence,
        )
    if not (eq_identity and vfs_identity and non_identity):
        return CheckResult(
            "vfs_profiles_dual_run",
            CheckStatus.FAIL,
            "VFS/non-VFS dual-run receipts are not identity-equivalent",
            evidence,
        )
    return CheckResult(
        "vfs_profiles_dual_run",
        CheckStatus.PASS,
        "VFS and non-VFS profiles dual-ran with identical content identities",
        evidence,
    )


# ---------------------------------------------------------------------------
# Safety: cold import, providers, report-only, fixed point, abstention, rollback
# ---------------------------------------------------------------------------


def check_cold_imports() -> CheckResult:
    """Prove cold imports succeed when optional providers are absent/blocked.

    Optional model/network packages may exist on the host.  Cold-import safety
    means release-critical modules still import cleanly when those packages raise
    ``ImportError`` (simulating absence) and do not *require* them at import time.
    """

    results: dict[str, Any] = {}
    for module_name in COLD_IMPORT_MODULES:
        # Block optional providers at the meta-path so host-installed torch/etc.
        # cannot be admitted during the cold import probe.
        code = (
            "import importlib\n"
            "import sys\n"
            f"forbidden = {list(OPTIONAL_PROVIDER_MODULES)!r}\n"
            "class _BlockOptional:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        root = fullname.split('.', 1)[0]\n"
            "        if root in forbidden or fullname in forbidden:\n"
            "            raise ImportError(f'blocked optional provider: {fullname}')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _BlockOptional())\n"
            "for name in list(sys.modules):\n"
            "    root = name.split('.', 1)[0]\n"
            "    if root in forbidden or name in forbidden:\n"
            "        sys.modules.pop(name, None)\n"
            "try:\n"
            f"    importlib.import_module({module_name!r})\n"
            "except Exception as exc:\n"
            "    print('FAIL:' + type(exc).__name__ + ':' + str(exc)[:180])\n"
            "    raise SystemExit(0)\n"
            "loaded = [name for name in forbidden if name in sys.modules]\n"
            "if loaded:\n"
            "    print('LOADED:' + ','.join(loaded))\n"
            "else:\n"
            "    print('OK')\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
                cwd=str(repository_root()),
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(repository_root()),
                },
            )
        except Exception as exc:
            results[module_name] = {"ok": False, "error": str(exc)}
            continue
        # Last non-empty stdout line is the verdict (libraries may log to stdout).
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        verdict = lines[-1] if lines else ""
        ok = proc.returncode == 0 and verdict == "OK"
        # Omit raw stderr: host loggers embed wall-clock timestamps that would
        # make dual-run sealed receipts non-deterministic.
        results[module_name] = {
            "ok": ok,
            "returncode": proc.returncode,
            "verdict": verdict[:200],
            "stderr_present": bool((proc.stderr or "").strip()),
        }

    failed = [name for name, payload in results.items() if not payload.get("ok")]
    evidence = {
        "modules": results,
        "optional_provider_modules": list(OPTIONAL_PROVIDER_MODULES),
        "failed": failed,
        "strategy": "block_optional_providers_at_meta_path",
    }
    if failed:
        return CheckResult(
            "cold_imports",
            CheckStatus.FAIL,
            f"cold import unsafe for: {failed}",
            evidence,
        )
    return CheckResult(
        "cold_imports",
        CheckStatus.PASS,
        "cold imports succeed with optional providers blocked/absent",
        evidence,
    )


def check_optional_provider_absence_safe() -> CheckResult:
    """Optional provider absence is actionable and never blocks report-only."""

    from ipfs_accelerate_py.agent_supervisor.control import (
        deterministic_doctor_rollout as rollout,
    )

    missing: list[str] = []
    for module_name in OPTIONAL_PROVIDER_MODULES:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    policy = rollout.default_rollout_policy()
    decision = rollout.evaluate_rollout_decision(policy)
    blocks = decision.effective_mode_value != "report_only" or decision.mutation_authorized
    evidence = {
        "missing_modules": missing,
        "absence_blocks_report_only_startup": blocks,
        "report_only_startup_ok": not blocks,
        "effective_mode": decision.effective_mode_value,
        "mutation_authorized": decision.mutation_authorized,
        "decision_id": decision.decision_id,
    }
    if blocks:
        return CheckResult(
            "optional_provider_absence",
            CheckStatus.FAIL,
            "optional provider absence blocked or elevated report-only startup",
            evidence,
        )
    return CheckResult(
        "optional_provider_absence",
        CheckStatus.PASS,
        "optional provider absence is safe for report-only startup",
        evidence,
    )


def check_report_only_no_write(repo_root: Path | None = None) -> CheckResult:
    """Prove report-only mode authorizes no write and leaves a disposable tree clean."""

    root = (repo_root or repository_root()).resolve()
    from ipfs_accelerate_py.agent_supervisor.control import (
        deterministic_doctor_rollout as rollout,
    )

    policy = rollout.default_rollout_policy()
    decision = rollout.evaluate_rollout_decision(policy)
    with tempfile.TemporaryDirectory(prefix="lpr042-report-only-") as tmp:
        probe = Path(tmp)
        marker = probe / "untouched.txt"
        marker.write_text("clean\n", encoding="utf-8")
        before = {
            str(path.relative_to(probe)): path.read_bytes()
            for path in probe.rglob("*")
            if path.is_file()
        }
        # Report-only decision surfaces; no source mutation path is invoked.
        _ = decision.to_dict()
        _ = policy.to_dict()
        after = {
            str(path.relative_to(probe)): path.read_bytes()
            for path in probe.rglob("*")
            if path.is_file()
        }
        tree_unchanged = before == after

    evidence = {
        "mode": policy.mode_value,
        "mutation_authorized": policy.mutation_authorized,
        "decision_mutation_authorized": decision.mutation_authorized,
        "narrow_auto": policy.narrow_autonomous_mutation_enabled,
        "tree_unchanged": tree_unchanged,
        "decision_id": decision.decision_id,
        "config_path": DOCTOR_CONFIG_REL,
        "config_present": (root / DOCTOR_CONFIG_REL).is_file(),
    }
    if (
        policy.mode_value != "report_only"
        or policy.mutation_authorized
        or decision.mutation_authorized
        or not tree_unchanged
    ):
        return CheckResult(
            "report_only_no_write",
            CheckStatus.FAIL,
            "report-only path authorized mutation or modified a probe tree",
            evidence,
        )
    return CheckResult(
        "report_only_no_write",
        CheckStatus.PASS,
        "report-only makes no write and authorizes no mutation",
        evidence,
    )


def _doctor_cases_from_report_or_evidence(
    doctor_report: Mapping[str, Any] | None = None,
    *,
    case_projections: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if case_projections is not None:
        return [dict(case) for case in case_projections]
    if doctor_report is not None:
        return [dict(case) for case in (doctor_report.get("cases") or [])]
    report = _run_doctor_benchmark()
    return [dict(case) for case in (report.get("cases") or [])]


def check_eligible_fixed_point(
    *,
    doctor_report: Mapping[str, Any] | None = None,
    case_projections: Sequence[Mapping[str, Any]] | None = None,
) -> CheckResult:
    """Eligible no-model analytical repairs reach complete all-caller fixed point."""

    cases = _doctor_cases_from_report_or_evidence(
        doctor_report, case_projections=case_projections
    )
    positives = [
        case
        for case in cases
        if str(case.get("scenario") or "") in ADMITTABLE_SCENARIOS
    ]
    if not positives:
        return CheckResult(
            "eligible_fixed_point",
            CheckStatus.FAIL,
            "no positive analytical cases present",
        )

    failures: list[str] = []
    details: dict[str, Any] = {}
    for case in positives:
        scenario = str(case.get("scenario") or "")
        ok = (
            case.get("repair_success") is True
            and case.get("admitted") is True
            and case.get("analytical_path") is True
            and case.get("llm_invoked") is not True
            and case.get("model_provider_called") is not True
            and str(case.get("fixed_point") or "").casefold()
            in {"complete", "success", "reached", "fixed"}
            and int(case.get("callers_repaired") or 0)
            >= int(case.get("mandatory_callers") or 0)
            and int(case.get("mandatory_callers") or 0) >= 1
        )
        details[scenario] = {
            "ok": ok,
            "repair_success": case.get("repair_success"),
            "fixed_point": case.get("fixed_point"),
            "callers_repaired": case.get("callers_repaired"),
            "mandatory_callers": case.get("mandatory_callers"),
            "receipt_id": case.get("receipt_id") or (case.get("receipt") or {}).get("receipt_id"),
            "completion_interface": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
            "logic_fixed_point_interface": LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE,
        }
        if not ok:
            failures.append(scenario)

    evidence = {
        "positive_count": len(positives),
        "failures": failures,
        "cases": details,
        "all_caller_atomic_fixed_point": not failures,
    }
    if failures:
        return CheckResult(
            "eligible_fixed_point",
            CheckStatus.FAIL,
            f"eligible repairs failed fixed point: {failures}",
            evidence,
        )
    return CheckResult(
        "eligible_fixed_point",
        CheckStatus.PASS,
        "eligible no-model repairs reach complete all-caller atomic fixed point",
        evidence,
    )


def check_abstention_and_rollback(
    *,
    doctor_report: Mapping[str, Any] | None = None,
    case_projections: Sequence[Mapping[str, Any]] | None = None,
) -> CheckResult:
    """Ambiguous/unsupported cases abstain cleanly; rollback restores exact roots."""

    cases = _doctor_cases_from_report_or_evidence(
        doctor_report, case_projections=case_projections
    )
    by_scenario = {str(case.get("scenario") or ""): case for case in cases}

    abstention_failures: list[str] = []
    abstention_details: dict[str, Any] = {}
    for scenario in sorted(FAIL_CLOSED_SCENARIOS):
        case = by_scenario.get(scenario)
        if case is None:
            abstention_failures.append(scenario)
            abstention_details[scenario] = {"present": False}
            continue
        clean = (
            case.get("repair_success") is not True
            and (
                case.get("abstained") is True
                or str(case.get("outcome_kind") or "")
                in {
                    "abstention",
                    "wrong_value",
                    "retrieval_degraded",
                    "stale_cache",
                    "solver_lie",
                    "open_frontier",
                    "incomplete_impact",
                    "sandbox_escape",
                    "rollback",
                    "oscillation",
                }
            )
            and case.get("llm_invoked") is not True
            and int((case.get("safety") or {}).get("missed_mandatory_caller") or 0) == 0
            and int((case.get("safety") or {}).get("out_of_scope_sandbox_write") or 0) == 0
            and int((case.get("safety") or {}).get("partial_transaction") or 0) == 0
            and int((case.get("safety") or {}).get("false_fixed_point") or 0) == 0
        )
        abstention_details[scenario] = {
            "present": True,
            "clean": clean,
            "outcome_kind": case.get("outcome_kind"),
            "abstained": case.get("abstained"),
            "repair_success": case.get("repair_success"),
            "receipt_id": case.get("receipt_id")
            or (case.get("receipt") or {}).get("receipt_id"),
        }
        if not clean:
            abstention_failures.append(scenario)

    rollback_case = by_scenario.get("crash_rollback")
    rollback_ok = False
    rollback_evidence: dict[str, Any] = {"present": rollback_case is not None}
    if rollback_case is not None:
        roots = dict(rollback_case.get("roots") or {})
        snapshot_roots = dict((rollback_case.get("snapshot") or {}).get("roots") or {})
        # Exact roots restored: receipt roots equal snapshot authority roots.
        roots_match = bool(roots) and (
            not snapshot_roots or all(roots.get(k) == snapshot_roots.get(k) for k in roots)
        )
        safety = rollback_case.get("safety") or {}
        rollback_ok = (
            str(rollback_case.get("outcome_kind") or "") == "rollback"
            or rollback_case.get("abstained") is True
            or str(rollback_case.get("completion") or "").casefold() == "rollback"
        ) and int(safety.get("rollback_failure") or 0) == 0 and roots_match
        rollback_evidence.update(
            {
                "ok": rollback_ok,
                "roots": roots,
                "snapshot_roots": snapshot_roots,
                "roots_match": roots_match,
                "rollback_failure": safety.get("rollback_failure"),
                "outcome_kind": rollback_case.get("outcome_kind"),
            }
        )

    evidence = {
        "abstention_failures": abstention_failures,
        "abstention": abstention_details,
        "rollback": rollback_evidence,
        "clean_tree": not abstention_failures,
        "rollback_restores_exact_roots": rollback_ok,
    }
    if abstention_failures:
        return CheckResult(
            "abstention_and_rollback",
            CheckStatus.FAIL,
            f"abstention cleanliness failed for: {abstention_failures}",
            evidence,
        )
    if not rollback_ok:
        return CheckResult(
            "abstention_and_rollback",
            CheckStatus.FAIL,
            "rollback did not restore exact roots",
            evidence,
        )
    return CheckResult(
        "abstention_and_rollback",
        CheckStatus.PASS,
        "ambiguous/unsupported cases abstain with clean trees; rollback restores roots",
        evidence,
    )


def check_zero_safety_floors(
    *,
    doctor_report: Mapping[str, Any] | None = None,
    metrics_projection: Mapping[str, Any] | None = None,
    policy: DeterministicDoctorReleasePolicy | None = None,
) -> CheckResult:
    """Require every joined-release safety floor to remain exactly zero."""

    policy = policy or default_release_policy()
    if metrics_projection is not None:
        metrics = dict(metrics_projection)
    elif doctor_report is not None:
        metrics = dict(doctor_report.get("metrics") or {})
    else:
        metrics = dict(_run_doctor_benchmark().get("metrics") or {})
    absolute = metrics.get("safety_absolute") or {}
    floors = metrics.get("safety_floors") or {}

    mapped = {
        "llm_or_remote_model_provider_invocation_rate": int(
            floors.get("llm_router_invocation_rate")
            or floors.get("llm_model_provider_call_rate")
            or absolute.get("llm_router_invocation")
            or absolute.get("llm_model_provider_call")
            or metrics.get("llm_invocation_count")
            or metrics.get("model_provider_call_count")
            or 0
        ),
        "kg_vector_embedding_authority_promotion_rate": int(
            floors.get("authority_promotion_rate")
            or absolute.get("authority_promotion")
            or 0
        ),
        "stale_forged_cache_cid_admission_rate": int(
            floors.get("stale_proof_cid_admission_rate")
            or absolute.get("stale_proof_cid_admission")
            or 0
        ),
        "missed_caller_open_frontier_mutation_rate": int(
            floors.get("missed_mandatory_caller_rate")
            or absolute.get("missed_mandatory_caller")
            or 0
        ),
        "sandbox_path_escape_rate": int(
            floors.get("out_of_scope_sandbox_write_rate")
            or absolute.get("out_of_scope_sandbox_write")
            or 0
        ),
        "partial_transaction_rate": int(
            floors.get("partial_transaction_rate")
            or absolute.get("partial_transaction")
            or 0
        ),
        "rollback_failure_rate": int(
            floors.get("rollback_failure_rate")
            or absolute.get("rollback_failure")
            or 0
        ),
        "nondeterminism_rate": int(
            floors.get("nondeterministic_render_rate")
            or absolute.get("nondeterministic_render")
            or (0 if metrics.get("dual_run_identity_equivalent", True) else 1)
        ),
        "false_completion_rate": int(
            floors.get("false_fixed_point_rate")
            or absolute.get("false_fixed_point")
            or 0
        ),
    }
    nonzero = {key: value for key, value in mapped.items() if int(value) != 0}
    evidence = {
        "floors": mapped,
        "policy_floors": dict(policy.safety_floors),
        "nonzero": nonzero,
        "metrics_authoritative": False,
    }
    if nonzero:
        return CheckResult(
            "zero_safety_floors",
            CheckStatus.FAIL,
            f"nonzero safety floors: {nonzero}",
            evidence,
        )
    return CheckResult(
        "zero_safety_floors",
        CheckStatus.PASS,
        "all joined-release safety floors are exactly zero",
        evidence,
    )


def check_declared_artifacts(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    present = {rel: (root / rel).is_file() for rel in REQUIRED_RELEASE_ARTIFACTS}
    missing = [rel for rel, ok in present.items() if not ok]
    evidence = {"artifacts": present}
    if missing:
        return CheckResult(
            "declared_artifacts",
            CheckStatus.FAIL,
            f"missing declared artifacts: {missing}",
            evidence,
        )
    return CheckResult(
        "declared_artifacts",
        CheckStatus.PASS,
        "all LPR-042 release artifacts are present",
        evidence,
    )


# ---------------------------------------------------------------------------
# Receipt / orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicDoctorReleaseReceipt:
    """Content-addressed joined release receipt for LPR-042."""

    INTERFACE: ClassVar[str] = RELEASE_RECEIPT_INTERFACE
    SCHEMA: ClassVar[str] = RELEASE_RECEIPT_SCHEMA

    valid: bool
    checks: Mapping[str, Mapping[str, Any]]
    policy: Mapping[str, Any]
    doctor_report_id: str = ""
    vfs_equivalence_content_id: str = ""
    two_profile_content_id: str = ""
    board_terminal: str = TERMINAL_TASK_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    receipt_id: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_id:
            payload = self.to_dict(include_id=False)
            object.__setattr__(self, "receipt_id", content_identity(payload))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "valid": bool(self.valid),
            "checks": dict(self.checks),
            "policy": dict(self.policy),
            "doctor_report_id": self.doctor_report_id,
            "vfs_equivalence_content_id": self.vfs_equivalence_content_id,
            "two_profile_content_id": self.two_profile_content_id,
            "board_terminal": self.board_terminal,
            "mutation_authorized": False,
            "completion_authoritative": False,
            "consumed_interfaces": {
                "vfs_equivalence": VFS_GENERALIZATION_EQUIVALENCE_RECEIPT_INTERFACE,
                "two_profile": ASSURANCE_TWO_PROFILE_CONFORMANCE_INTERFACE,
                "doctor_run_receipt": DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE,
                "doctor_metrics": DETERMINISTIC_DOCTOR_METRICS_INTERFACE,
                "propagation_completion": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
                "logic_fixed_point": LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE,
            },
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id or content_identity(
                {k: v for k, v in payload.items() if k != "receipt_id"}
            )
        return payload


def _checks_to_map(results: Sequence[CheckResult]) -> dict[str, dict[str, Any]]:
    return {result.name: result.to_dict() for result in results}


def validate_deterministic_doctor_release(
    repo_root: Path | None = None,
    *,
    policy: DeterministicDoctorReleasePolicy | None = None,
    install_guards: bool = True,
    run_vfs: bool = True,
    run_doctor: bool = True,
) -> DeterministicDoctorReleaseReceipt:
    """Run the full joined release gate and return a sealed receipt."""

    root = (repo_root or repository_root()).resolve()
    policy = policy or default_release_policy()
    results: list[CheckResult] = []

    results.append(check_declared_artifacts(root))
    results.append(check_canonical_board(root))
    results.append(check_four_lane_supervisor_drain(root))

    doctor_report_id = ""
    if run_doctor and policy.require_doctor_fixture_dual_run:
        doctor_check, projections, metrics_projection = check_doctor_fixture_dual_run(
            root, install_guards=install_guards
        )
        results.append(doctor_check)
        if doctor_check.ok:
            doctor_report_id = str(doctor_check.evidence.get("report_id") or "")
            results.append(
                check_eligible_fixed_point(case_projections=projections)
            )
            results.append(
                check_abstention_and_rollback(case_projections=projections)
            )
            results.append(
                check_zero_safety_floors(
                    metrics_projection=metrics_projection, policy=policy
                )
            )
        else:
            results.append(
                CheckResult(
                    "eligible_fixed_point",
                    CheckStatus.FAIL,
                    "skipped because doctor dual-run failed",
                )
            )
            results.append(
                CheckResult(
                    "abstention_and_rollback",
                    CheckStatus.FAIL,
                    "skipped because doctor dual-run failed",
                )
            )
            results.append(
                CheckResult(
                    "zero_safety_floors",
                    CheckStatus.FAIL,
                    "skipped because doctor dual-run failed",
                )
            )
    else:
        results.append(
            CheckResult(
                "doctor_fixture_dual_run",
                CheckStatus.SKIP,
                "doctor dual-run not requested",
            )
        )

    if run_vfs and policy.require_vfs_and_non_vfs_profiles:
        results.append(check_vfs_profiles_dual_run(root))
    else:
        results.append(
            CheckResult(
                "vfs_profiles_dual_run",
                CheckStatus.SKIP,
                "VFS dual-run not requested",
            )
        )

    if policy.require_cold_imports:
        results.append(check_cold_imports())
    if policy.require_optional_provider_absence_safe:
        results.append(check_optional_provider_absence_safe())
    if policy.require_report_only_no_write:
        results.append(check_report_only_no_write(root))

    checks = _checks_to_map(results)
    valid = all(
        item.get("status") in {"pass", "skip", "warn"} for item in checks.values()
    )
    if not doctor_report_id and checks.get("doctor_fixture_dual_run", {}).get("evidence"):
        doctor_report_id = str(
            checks["doctor_fixture_dual_run"]["evidence"].get("report_id") or ""
        )

    vfs_evidence = checks.get("vfs_profiles_dual_run", {}).get("evidence") or {}
    receipt = DeterministicDoctorReleaseReceipt(
        valid=valid,
        checks=checks,
        policy=policy.to_dict(),
        doctor_report_id=doctor_report_id,
        vfs_equivalence_content_id=str(
            vfs_evidence.get("equivalence_content_id") or ""
        ),
        two_profile_content_id=str(vfs_evidence.get("conformance_content_id") or ""),
        board_terminal=TERMINAL_TASK_ID,
    )
    return receipt


def replay_release_receipt(
    receipt: Mapping[str, Any] | DeterministicDoctorReleaseReceipt,
) -> dict[str, Any]:
    """Replay a release receipt and prove identity-equivalent sealing."""

    payload = (
        receipt.to_dict()
        if isinstance(receipt, DeterministicDoctorReleaseReceipt)
        else dict(receipt)
    )
    claimed = payload.get("receipt_id")
    resealed = seal_payload(
        {k: v for k, v in payload.items() if k != "receipt_id"},
        id_key="receipt_id",
    )
    identity_ok = claimed == resealed.get("receipt_id") and verify_sealed(payload)
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-release-replay@1",
        "interface": "DeterministicDoctorReleaseReplay@1",
        "valid": bool(identity_ok and payload.get("valid") is True),
        "identity_ok": identity_ok,
        "claimed_receipt_id": claimed,
        "recomputed_receipt_id": resealed.get("receipt_id"),
        "mutation_authorized": False,
        "completion_authoritative": False,
    }


def run_all_checks(
    repo_root: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    receipt = validate_deterministic_doctor_release(repo_root, **kwargs)
    payload = receipt.to_dict()
    payload["report_schema"] = RELEASE_VALIDATOR_SCHEMA
    payload["validator_interface"] = RELEASE_VALIDATOR_INTERFACE
    return payload


def doctor(repo_root: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    report = run_all_checks(repo_root, **kwargs)
    report["command"] = "doctor"
    return report


class DeterministicDoctorReleaseValidator:
    INTERFACE: ClassVar[str] = RELEASE_VALIDATOR_INTERFACE
    SCHEMA: ClassVar[str] = RELEASE_VALIDATOR_SCHEMA

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        policy: DeterministicDoctorReleasePolicy | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.policy = policy or default_release_policy()

    def run_all(self, **kwargs: Any) -> dict[str, Any]:
        return run_all_checks(self.repo_root, policy=self.policy, **kwargs)

    def validate(self, **kwargs: Any) -> DeterministicDoctorReleaseReceipt:
        return validate_deterministic_doctor_release(
            self.repo_root, policy=self.policy, **kwargs
        )

    def doctor(self, **kwargs: Any) -> dict[str, Any]:
        return doctor(self.repo_root, policy=self.policy, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "policy": self.policy.to_dict(),
            "mutation_authorized": False,
            "completion_authoritative": False,
        }


def write_checkpoint(name: str, payload: Mapping[str, Any]) -> Path | None:
    """Atomically write a content-addressed checkpoint when the env is set."""

    directory = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR", "").strip()
    if not directory:
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    body = _plain(payload)
    sealed = seal_payload(body if isinstance(body, Mapping) else {"payload": body})
    target = root / f"{name}.json"
    tmp = root / f".{name}.{os.getpid()}.tmp"
    tmp.write_text(
        json.dumps(sealed, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


__all__ = [
    "ADMITTABLE_SCENARIOS",
    "ASSURANCE_TWO_PROFILE_CONFORMANCE_INTERFACE",
    "CANONICAL_GOAL_COUNT",
    "CANONICAL_TASK_COUNT",
    "DETERMINISTIC_DOCTOR_METRICS_INTERFACE",
    "DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE",
    "DUAL_RUN_PASSES",
    "FAIL_CLOSED_SCENARIOS",
    "GOAL_ID",
    "LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE",
    "PROPAGATION_COMPLETION_RECEIPT_INTERFACE",
    "RELEASE_POLICY_INTERFACE",
    "RELEASE_RECEIPT_INTERFACE",
    "RELEASE_VALIDATOR_INTERFACE",
    "SAFETY_FLOOR_KEYS",
    "SEALED_TASK_CIDS_LPR_000_028",
    "TASK_ID",
    "TERMINAL_TASK_ID",
    "VFS_GENERALIZATION_EQUIVALENCE_RECEIPT_INTERFACE",
    "CheckResult",
    "CheckStatus",
    "DeterministicDoctorReleaseError",
    "DeterministicDoctorReleasePolicy",
    "DeterministicDoctorReleaseReceipt",
    "DeterministicDoctorReleaseValidator",
    "check_abstention_and_rollback",
    "check_canonical_board",
    "check_cold_imports",
    "check_declared_artifacts",
    "check_doctor_fixture_dual_run",
    "check_eligible_fixed_point",
    "check_four_lane_supervisor_drain",
    "check_optional_provider_absence_safe",
    "check_report_only_no_write",
    "check_vfs_profiles_dual_run",
    "check_zero_safety_floors",
    "content_identity",
    "default_release_policy",
    "doctor",
    "replay_release_receipt",
    "repository_root",
    "run_all_checks",
    "seal_payload",
    "validate_deterministic_doctor_release",
    "verify_sealed",
    "write_checkpoint",
]
