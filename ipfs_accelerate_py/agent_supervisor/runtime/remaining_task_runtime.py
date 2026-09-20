"""Bind remaining SPAR/SAWM/ASEH/PCTDD/DOEP board theory to extra-gate runtime.

Remaining-task modules live on the overlay. Extra-gate workers currently
import the sealed nested accelerate (`python3 -P` plus nested PYTHONPATH) and
override a no-model kernel abstention into Grok/Codex. This module is the
typed remaining-task control surface those workers should call instead.

It never writes DuckDB, never CAS-completes a task, and never treats overlay
pytest as extra-gate admission.
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


def remaining_task_pythonpath_entries(overlay: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    return (str(remaining_task_overlay_root(overlay)),)


def prepend_remaining_task_overlay_pythonpath(
    overlay: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Put overlay remaining-task modules first for extra-gate children.

    ``python3 -P`` still honors PYTHONPATH. Extra-gate parent pin-only /
    sealed observation still insert their own ``sys.path[0]``. This is not
    an ExecStart wrap and does not exclusive-open DuckDB.
    """

    root = str(remaining_task_overlay_root(overlay))
    target = os.environ if environ is None else environ
    current = str(target.get("PYTHONPATH") or "")
    parts = [item for item in current.split(os.pathsep) if item and item != root]
    updated = os.pathsep.join((root, *parts))
    target["PYTHONPATH"] = updated
    return updated


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
    if binding.task_id == "SAWM-044":
        from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_release import (
            build_current_tree_release_report,
        )

        report = build_current_tree_release_report().as_dict()
        return _base_result(binding, surface="release", report=report, released=False)
    if binding.task_id in {"SAWM-039"}:
        return _base_result(
            binding,
            surface="controls",
            status=asdict(service.status()),
            controls=describe_controls(),
        )
    if binding.task_id == "SAWM-041":
        result = service.operation("benchmark")
        return _base_result(binding, surface="benchmark", result=result)
    if binding.task_id == "SAWM-016":
        result = service.operation("reuse")
        return _base_result(binding, surface="reuse", result=result)
    return _base_result(
        binding,
        surface="program-world",
        status=asdict(service.status()),
        controls=describe_controls(),
    )


def _execute_doep(binding: RemainingTaskBinding) -> dict[str, Any]:
    task_id = binding.task_id
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
        raise RemainingTaskRuntimeError("remaining-task runtime cannot admit extra-gate completion")
    return result
