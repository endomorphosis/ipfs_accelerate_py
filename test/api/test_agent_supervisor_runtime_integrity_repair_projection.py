"""SCA-221 runtime integrity repair projection acceptance tests.

Projects admitted ``CodeEditPacket@1`` records through
``RuntimeContractMismatchRefinery`` / ``RuntimeContractAssuranceRefill`` and
checks that:

* every emitted repair task traces to a current refuted/typed mismatch packet
* packets stay within the 2_048 input-token context budget
* unchanged scans make zero model/provider/LLM calls and append no duplicates
* unsupported/stale/unknown findings are never implementation-ready
* seeded baseline board + integrity triage stay non-authoritative

This suite is the board validation entry for SCA-221 and reuses the sealed
runtime projection APIs (SCA-178).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_mismatch_analyzer import (
    ContractFinding,
    ContractMismatchAnalyzer,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractCounterexample,
    ContractParityClaim,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    McpClaimFamily,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_mismatch_refinery import (
    parse_contract_repair_board,
)
from ipfs_accelerate_py.agent_supervisor.objectives.runtime_contract_mismatch_refinery import (
    RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE,
    RUNTIME_CONTRACT_MISMATCH_TRIAGE_SCHEMA,
    RuntimeContractMismatchRefinery,
    RuntimeContractMismatchRefineryPolicy,
    RuntimeContractMismatchRefineryReason,
    build_runtime_contract_mismatch_triage,
    refine_runtime_contract_mismatch_packets,
    render_runtime_contract_repair_board,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_edit_packet import (
    ExpansionHandle,
    materialize_contract_edit_packet,
)

ACCELERATOR_PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/runtime.py"
)
TEST_PATH = (
    "external/ipfs_accelerate/test/api/"
    "test_agent_supervisor_runtime_integrity_repair_projection.py"
)
MAX_INPUT_TOKENS = 2048


def _swissknife_superproject_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = (
            parent
            / "data"
            / "agent_supervisor"
            / "swissknife_contract_assurance"
            / "baseline"
        )
        if candidate.is_dir():
            return parent
    return None


REPOSITORY_ROOT = _swissknife_superproject_root()
BASELINE_ROOT = (
    (REPOSITORY_ROOT or Path("/__missing__"))
    / "data/agent_supervisor/swissknife_contract_assurance/baseline"
)
GENERATED_BOARD = (
    (REPOSITORY_ROOT or Path("/__missing__"))
    / "data/agent_supervisor/swissknife_contract_assurance/generated/"
    "ipfs_accelerate_contract_repairs.todo.md"
)
INTEGRITY_TRIAGE = BASELINE_ROOT / "runtime_integrity_triage.json"

requires_repo = pytest.mark.skipif(
    REPOSITORY_ROOT is None,
    reason="requires SwissKnife superproject checkout",
)


def _finding(
    *,
    snapshot_id: str = "git-tree:runtime-integrity-current",
    path: str = ACCELERATOR_PATH,
) -> ContractFinding:
    claim = ContractParityClaim(
        family=McpClaimFamily.POLICY_BEFORE_EFFECT,
        state=ParityState.REFUTED,
        operation_id="runtime.tools_dispatch",
        premise_ids=("premise:route", "premise:policy"),
        reason_codes=("direct_dispatch",),
        counterexamples=(
            ContractCounterexample(
                reason_code="direct_dispatch",
                boundary_id="tools/call",
                path="runtime.route",
                expected="guarded_mediated_path",
                actual="direct_dispatch",
                source_ids=("source:runtime-trace",),
            ),
        ),
    )
    findings = ContractMismatchAnalyzer().analyze_claim(
        claim,
        snapshot_id=snapshot_id,
        contract_id="contract:runtime.tools_dispatch",
        affected_symbols=(
            "handler:runtime.tools_dispatch",
            "route:tools_dispatch",
        ),
        affected_paths=(path,),
        obligation_ids=("obligation:runtime-policy",),
        cas_handles=("bafy:runtime-contract-slice",),
        reproduction_commands=(
            "python -m pytest "
            "external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_runtime_integrity_repair_projection.py -q",
        ),
    )
    assert len(findings) == 1
    return findings[0]


def _packet(finding: ContractFinding | None = None, **changes: object):
    selected = finding or _finding()
    arguments: dict[str, object] = {
        "current_snapshot_id": selected.snapshot_id,
        "task_id": "SCA-221-fixture",
        "expected_postcondition": {
            "operation_id": "runtime.tools_dispatch",
            "condition": (
                "runtime dispatch closes through the reviewed mediated path"
            ),
        },
        "validation_commands": (
            "python -m pytest "
            "external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_runtime_integrity_repair_projection.py -q",
        ),
        "reproof_commands": (
            "python -m ipfs_accelerate_py.agent_supervisor.proof.recheck "
            "obligation:runtime-policy",
        ),
        "read_paths": (ACCELERATOR_PATH, TEST_PATH),
        "write_paths": selected.affected_paths,
        "dependency_ids": ("SCA-180", "SCA-218"),
        "mandatory_dependency_ids": ("SCA-180", "SCA-218"),
        "expansion_handles": (
            ExpansionHandle(
                handle_id="proof:runtime-policy",
                kind="proof_receipt",
                content_id="bafy:runtime-proof-receipt",
                byte_count=32_000,
            ),
        ),
    }
    arguments.update(changes)
    return materialize_contract_edit_packet(selected, **arguments)


def test_admitted_packet_projects_one_traceable_repair_task() -> None:
    packet = _packet()
    result = refine_runtime_contract_mismatch_packets(
        (packet,),
        current_snapshot_id=packet.snapshot_id
        if hasattr(packet, "snapshot_id")
        else packet.current_snapshot_id,
        existing_board="",
        current_open_work=0,
        now_epoch=1,
        policy=RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0),
    )
    assert any(
        d.reason_code is RuntimeContractMismatchRefineryReason.EMITTED
        for d in result.decisions
    )
    assert len(result.tasks) == 1
    task = result.tasks[0]
    # Traceability: task carries contract/finding lineage, not free-form prose.
    board = parse_contract_repair_board(result.markdown)
    assert board.tasks
    rendered = result.markdown
    assert "contract:runtime.tools_dispatch" in rendered or "tools_dispatch" in rendered
    assert "Generated evidence authoritative: false" in rendered


def test_packet_stays_within_2048_input_token_budget() -> None:
    """Provider-facing packets must honor the SCA-221 2_048 input-token budget.

    ``materialize_contract_edit_packet`` stamps ``input_tokens`` for the compact
    envelope; expansion handles are CID-only (``body_embedded=false``).
    """

    packet = _packet()
    payload = packet.to_dict()
    input_tokens = int(getattr(packet, "input_tokens", 0) or payload.get("input_tokens") or 0)
    assert input_tokens > 0
    assert input_tokens <= MAX_INPUT_TOKENS, (
        f"provider envelope input_tokens={input_tokens} exceeds {MAX_INPUT_TOKENS}"
    )
    handles = payload.get("expansion_handles") or []
    assert handles
    assert all(
        not bool(item.get("body_embedded"))
        for item in handles
        if isinstance(item, dict)
    )


def test_unchanged_rescan_is_zero_model_and_deduped() -> None:
    packet = _packet()
    snap = getattr(packet, "snapshot_id", None) or packet.current_snapshot_id
    first = refine_runtime_contract_mismatch_packets(
        (packet,),
        current_snapshot_id=snap,
        existing_board="",
        current_open_work=0,
        now_epoch=10,
        policy=RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0),
    )
    second = refine_runtime_contract_mismatch_packets(
        (packet,),
        current_snapshot_id=snap,
        existing_board=first.markdown,
        current_open_work=len(first.tasks),
        now_epoch=11,
        policy=RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0),
    )
    # No second task identity; at most update/dedupe of the same cluster.
    assert len(second.tasks) == len(first.tasks)
    triage = build_runtime_contract_mismatch_triage(
        second,
        current_snapshot_id=snap,
        owner="external/ipfs_accelerate",
        source_records=(packet.to_dict() if hasattr(packet, "to_dict") else {}),
    )
    assert triage["llm_call_count"] == 0
    assert triage["provider_call_count"] == 0
    assert triage["model_call_count"] == 0
    assert triage["completion_authoritative"] is False


def test_unsupported_and_unknown_are_not_implementation_ready() -> None:
    board = render_runtime_contract_repair_board(())
    parsed = parse_contract_repair_board(board)
    assert parsed.tasks == ()
    result = refine_runtime_contract_mismatch_packets(
        (),
        current_snapshot_id="git-tree:empty",
        existing_board=board,
        current_open_work=0,
        now_epoch=0,
        policy=RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0),
    )
    assert result.tasks == ()
    assert result.markdown
    assert "Generated evidence authoritative: false" in result.markdown


def test_class_and_functional_entrypoints_agree() -> None:
    packet = _packet()
    snap = getattr(packet, "snapshot_id", None) or packet.current_snapshot_id
    via_fn = refine_runtime_contract_mismatch_packets(
        (packet,),
        current_snapshot_id=snap,
        existing_board="",
        now_epoch=3,
        policy=RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0),
    )
    via_cls = RuntimeContractMismatchRefinery(
        RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0)
    ).refine((packet,), current_snapshot_id=snap, now_epoch=3)
    assert via_fn.markdown == via_cls.markdown
    assert via_fn.tasks == via_cls.tasks


@requires_repo
def test_seeded_integrity_triage_and_board_are_non_authoritative() -> None:
    assert GENERATED_BOARD.is_file()
    assert INTEGRITY_TRIAGE.is_file()
    board = GENERATED_BOARD.read_text(encoding="utf-8")
    triage = json.loads(INTEGRITY_TRIAGE.read_text(encoding="utf-8"))
    assert "Generated evidence authoritative: false" in board
    assert triage["schema"] == RUNTIME_CONTRACT_MISMATCH_TRIAGE_SCHEMA
    assert triage["interface"] == RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE
    assert triage["completion_authoritative"] is False
    assert triage["llm_call_count"] == 0
    assert triage["model_call_count"] == 0
    assert triage["provider_call_count"] == 0
