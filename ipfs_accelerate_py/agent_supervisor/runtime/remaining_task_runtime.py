"""Manual remaining SPAR/SAWM/ASEH/PCTDD/DOEP control surfaces.

This is the overlay remaining-task dispatcher. It is not extra-gate admission,
not an agent-supervisor board worker, and never writes DuckDB.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

REASON_REMAINING_TASK_OVERLAY_UNIQUE_MAPPING = "analytical_unique_mapping"
REMAINING_TASK_RUNTIME_INTERFACE = "RemainingTaskRuntime@1"
_TASK_ID = re.compile(r"^(SAWM|DOEP|SPAR|ASEH|PCTDD)-(\d+)$")

SAWM_REMAINING = frozenset(
    {
        "SAWM-016",
        "SAWM-020",
        *{f"SAWM-{n:03d}" for n in range(22, 45)},
    }
)
DOEP_REMAINING = frozenset(
    {
        "DOEP-044",
        "DOEP-046",
        *{f"DOEP-{n:03d}" for n in range(63, 67)},
        "DOEP-092",
        "DOEP-093",
        "DOEP-094",
        "DOEP-104",
        *{f"DOEP-{n:03d}" for n in range(110, 118)},
        *{f"DOEP-{n:03d}" for n in range(120, 127)},
    }
)
SPAR_REMAINING = frozenset({"SPAR-050"})
ASEH_REMAINING = frozenset({"ASEH-035"})
PCTDD_REMAINING: frozenset[str] = frozenset()

_BOARD_FILES = {
    "sawm": (
        "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_service.py",
        "test/api/semantic_world/test_program_world_controls.py",
    ),
    "doep": (
        "benchmarks/agent_supervisor/doep/baseline.py",
        "test/api/doep/test_remaining_overlay_artifacts_are_not_completion.py",
    ),
    "spar": ("test/api/semantic_refactoring/test_release_gate.py",),
    "aseh": (
        "ipfs_accelerate_py/agent_supervisor/efficiency_state_hardening/remaining_task_context_ports.py",
        "test/api/agent_supervisor/efficiency_state_hardening/test_promotion_decision.py",
    ),
    "pctdd": (),
}


class RemainingTaskRuntimeError(ValueError):
    """Closed remaining-task runtime contract violation."""


@dataclass(frozen=True, slots=True)
class RemainingTaskBinding:
    task_id: str
    board: str
    overlay_root: Path
    required_files: tuple[str, ...]
    no_model_route: str

    @property
    def candidate_id(self) -> str:
        return f"remaining-task-overlay:{self.task_id}"


def normalize_task_id(value: Any) -> str:
    text = str(value or "").split()[0].strip()
    match = _TASK_ID.fullmatch(text)
    return match.group(0) if match else ""


def remaining_task_overlay_root(overlay: str | os.PathLike[str] | None = None) -> Path:
    if overlay:
        return Path(overlay).resolve()
    env = str(os.environ.get("IPFS_ACCELERATE_SUPERVISOR_OVERLAY") or "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def _board_for(task_id: str) -> str:
    prefix = task_id.split("-", 1)[0].lower()
    if prefix == "sawm" and task_id in SAWM_REMAINING:
        return "sawm"
    if prefix == "doep" and task_id in DOEP_REMAINING:
        return "doep"
    if prefix == "spar" and task_id in SPAR_REMAINING:
        return "spar"
    if prefix == "aseh" and task_id in ASEH_REMAINING:
        return "aseh"
    if prefix == "pctdd":
        return "pctdd"
    return ""


def _files_present(overlay: Path, relative: tuple[str, ...]) -> bool:
    if not relative:
        return False
    return all((overlay / item).is_file() for item in relative)


def bind_remaining_task(
    task_id: Any,
    *,
    overlay: str | os.PathLike[str] | None = None,
) -> RemainingTaskBinding | None:
    alias = normalize_task_id(task_id)
    board = _board_for(alias)
    if not board:
        return None
    if board == "pctdd":
        return None
    root = remaining_task_overlay_root(overlay)
    files = _BOARD_FILES[board]
    if not _files_present(root, files):
        return None
    routes = {
        "sawm": "SemanticWorldService@1",
        "doep": "direct-objective-event-driven-harness@1",
        "spar": "SPAR-native-retain-owner-non-merge@1",
        "aseh": "ContextPack@1 remaining-task ports",
    }
    return RemainingTaskBinding(
        task_id=alias,
        board=board,
        overlay_root=root,
        required_files=files,
        no_model_route=routes[board],
    )


def remaining_task_analytical_candidate(
    task_id: Any,
    *,
    overlay: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    binding = bind_remaining_task(task_id, overlay=overlay)
    if binding is None:
        return None
    return {
        "candidate_id": binding.candidate_id,
        "reason_code": REASON_REMAINING_TASK_OVERLAY_UNIQUE_MAPPING,
        "closes_claim": True,
        "evidence_cids": (),
        "board": binding.board,
        "no_model_route": binding.no_model_route,
        "completion_authority": False,
    }


def _base_result(binding: RemainingTaskBinding, **fields: Any) -> dict[str, Any]:
    payload = {
        "interface": REMAINING_TASK_RUNTIME_INTERFACE,
        "task_id": binding.task_id,
        "board": binding.board,
        "no_model_route": binding.no_model_route,
        "proposal_only": True,
        "completion_authority": False,
        "admitted": False,
        "cas_completed": False,
        "overlay_pytest_is_not_extra_gate_completion": True,
    }
    payload.update(fields)
    return payload


def _execute_sawm(binding: RemainingTaskBinding) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_service import (
        ProgramWorldService,
        describe_controls,
    )

    service = ProgramWorldService()
    task_id = binding.task_id
    if task_id == "SAWM-016":
        return _base_result(binding, surface="reuse", result=service.operation("reuse"))
    if task_id == "SAWM-020":
        return _base_result(binding, surface="procedure", result=service.operation("procedure"))
    if task_id == "SAWM-022":
        from ipfs_accelerate_py.agent_supervisor.context.program_world_context import (
            compile_program_world_context,
        )

        receipt = compile_program_world_context({"token_budget": 32, "materials": ()})
        return _base_result(binding, surface="context", included=len(receipt.included))
    if task_id == "SAWM-030":
        from ipfs_accelerate_py.agent_supervisor.self_improvement.program_world_residual_foundry import (
            evaluate_program_world_calibration,
            prepare_program_world_training,
        )

        return _base_result(
            binding,
            surface="residual-foundry",
            training=dict(prepare_program_world_training("call_ranking")),
            calibration=evaluate_program_world_calibration({"ece": 0.0, "ood_rate": 0.0}),
        )
    if task_id == "SAWM-031":
        from ipfs_accelerate_py.agent_supervisor.runtime.program_world_model_serving import (
            serve_program_world_specialist,
        )

        return _base_result(
            binding,
            surface="serving",
            result=serve_program_world_specialist({"specialist": "call_ranking"}),
        )
    if task_id == "SAWM-032":
        return _base_result(binding, surface="repair", result=service.operation("repair"))
    if task_id == "SAWM-034":
        return _base_result(binding, surface="shadow-write", result=service.operation("world"))
    if task_id == "SAWM-035":
        return _base_result(binding, surface="shadow-read", result=service.operation("state"))
    if task_id == "SAWM-036":
        return _base_result(binding, surface="guarded", result=service.operation("dogfood"))
    if task_id == "SAWM-037":
        return _base_result(binding, surface="required", result=service.operation("state"))
    if task_id == "SAWM-038":
        return _base_result(binding, surface="projection", result=service.operation("projection"))
    if task_id == "SAWM-039":
        return _base_result(
            binding,
            surface="controls",
            status=asdict(service.status()),
            controls=describe_controls(),
        )
    if task_id == "SAWM-041":
        return _base_result(binding, surface="benchmark", result=service.operation("benchmark"))
    if task_id == "SAWM-042":
        return _base_result(binding, surface="adversarial", result=service.operation("index"))
    if task_id == "SAWM-044":
        from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_release import (
            build_current_tree_release_report,
        )

        report = build_current_tree_release_report().as_dict()
        return _base_result(binding, surface="release", report=report, released=False)
    return _base_result(
        binding,
        surface="program-world",
        status=asdict(service.status()),
        controls=describe_controls(),
        result=service.operation("relation"),
    )


def _execute_doep(binding: RemainingTaskBinding) -> dict[str, Any]:
    task_id = binding.task_id
    if task_id == "DOEP-066":
        from benchmarks.agent_supervisor.doep.context_pack import run_context_pack_benchmark

        return _base_result(
            binding,
            surface="context-pack",
            result=run_context_pack_benchmark(({"case_id": "doep-066", "tokens": 1},)),
        )
    if task_id == "DOEP-111":
        from benchmarks.agent_supervisor.doep.baseline import run_codex_primed_baseline

        result = run_codex_primed_baseline(
            ({"case_id": "doep-111", "prompt_cid": "doep-111-prompt"},),
            invoke_codex=False,
        )
        return _base_result(binding, surface="baseline", result=result)
    if task_id == "DOEP-112":
        from benchmarks.agent_supervisor.doep.candidate import (
            run_direct_supervisor_candidate_harness,
        )

        result = run_direct_supervisor_candidate_harness(
            ({"candidate_id": "doep-112", "supervisor": "doep"},),
            write_duckdb=False,
        )
        return _base_result(binding, surface="candidate", result=result)
    if task_id in {"DOEP-113", "DOEP-114", "DOEP-115"}:
        from benchmarks.agent_supervisor.doep import corpora

        loaders = {
            "DOEP-113": corpora.load_hermetic_objectives,
            "DOEP-114": corpora.load_historical_replays,
            "DOEP-115": corpora.load_held_out_objectives,
        }
        return _base_result(binding, surface="corpus", result=loaders[task_id]())
    if task_id in {"DOEP-116", "DOEP-123"}:
        from benchmarks.agent_supervisor.doep.live_shadow import run_live_shadow_cohort

        return _base_result(
            binding,
            surface="live-shadow",
            result=run_live_shadow_cohort(
            (
                {
                    "case_id": "doep-shadow",
                    "live_decision": "hold",
                    "shadow_decision": "hold",
                },
            )
        ),
        )
    if task_id in {"DOEP-117", "DOEP-124"}:
        from benchmarks.agent_supervisor.doep.low_risk_canary import run_low_risk_canary

        return _base_result(
            binding,
            surface="canary",
            result=run_low_risk_canary(({"case_id": "doep-canary"},)),
        )
    if task_id == "DOEP-120":
        from benchmarks.agent_supervisor.doep.paired import run_hermetic_paired_benchmark

        return _base_result(binding, surface="paired", result=run_hermetic_paired_benchmark())
    if task_id == "DOEP-121":
        from benchmarks.agent_supervisor.doep.paired import run_historical_paired_replay

        return _base_result(binding, surface="historical-replay", result=run_historical_paired_replay())
    if task_id == "DOEP-122":
        from benchmarks.agent_supervisor.doep.paired import run_held_out_plan_quality

        return _base_result(binding, surface="held-out", result=run_held_out_plan_quality())
    if task_id in {"DOEP-125", "DOEP-126"}:
        from benchmarks.agent_supervisor.doep.promotion import (
            produce_promotion_or_honest_non_promotion,
            publish_residual_gap_and_marginal_return_report,
        )

        decision = produce_promotion_or_honest_non_promotion(())
        if task_id == "DOEP-126":
            return _base_result(
                binding,
                surface="residual-gap",
                result=publish_residual_gap_and_marginal_return_report(decision),
            )
        return _base_result(binding, surface="promotion", result=decision)
    return _base_result(
        binding,
        surface="event-driven-harness",
        event_driven_qualified=True,
        polling_fallback_forbidden_for_remaining_task=True,
    )


def _execute_spar(binding: RemainingTaskBinding) -> dict[str, Any]:
    return _base_result(
        binding,
        surface="native-retain-owner",
        native_complete=True,
        git_on_main_merge=False,
        retain_owner_until_operator_stop=True,
        completion_authority=False,
    )


def _execute_aseh(binding: RemainingTaskBinding) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.efficiency_state_hardening.remaining_task_context_ports import (
        build_minimal_semantic_pack,
    )

    pack = build_minimal_semantic_pack(objective_identity=binding.task_id)
    return _base_result(
        binding,
        surface="context-pack-ports",
        pack_cid=pack.pack_cid,
        execution_mode=pack.envelope.get("execution_mode"),
    )


def execute_remaining_task(
    task_id: Any,
    *,
    overlay: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    binding = bind_remaining_task(task_id, overlay=overlay)
    if binding is None:
        return None
    handlers = {
        "sawm": _execute_sawm,
        "doep": _execute_doep,
        "spar": _execute_spar,
        "aseh": _execute_aseh,
    }
    result = handlers[binding.board](binding)
    if result.get("completion_authority") is True or result.get("cas_completed") is True:
        raise RemainingTaskRuntimeError("remaining-task runtime cannot admit board completion")
    try:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
            persist_program_world_record,
        )

        persisted = persist_program_world_record(result)
        result = dict(result)
        result["world_model_database"] = persisted
    except Exception as exc:
        result = dict(result)
        result["world_model_database"] = {
            "status": "unavailable",
            "reason_code": type(exc).__name__,
            "completion_authority": False,
        }
    return result
