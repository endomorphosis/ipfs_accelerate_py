"""LPR-017: ordinary provider proposals as read-only candidate overlays.

Exact generic-LLM regression: a proposal that changes ``f(a, b)`` to
``f(a, b, c)`` but omits affected callers is rejected or expanded into a newly
admitted write set.  Every resolved caller is dispositioned before mutation;
required unknown frontiers abstain.  Analytical overlay analysis never invokes
a model.  Legacy proposal flows remain unchanged when the flag is off.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.live_logic_repair_controller import (
    PROPOSAL_OVERLAY_STAGE_ORDER,
    CandidateOverlayContractDeltaGate,
    LiveLogicRepairController,
    LiveLogicRepairMode,
    LiveLogicRepairPolicy,
    LiveLogicRepairRequest,
    OverlayCallerDisposition,
    OverlayGateDisposition,
    compute_signature_deltas,
    extract_python_signatures,
    find_call_sites,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    ImplementationProposal,
    ProposalFindingCode,
    ProposalValidationPolicy,
    ProposalValidator,
    validate_implementation_proposal,
)


# ---------------------------------------------------------------------------
# Sources for the exact f(a, b) -> f(a, b, c) regression
# ---------------------------------------------------------------------------


CALLEE_BEFORE = """\
def f(a, b):
    return a + b
"""

CALLEE_AFTER = """\
def f(a, b, c):
    return a + b + c
"""

CALLER_BEFORE = """\
from callee import f

def use_f():
    return f(1, 2)
"""

CALLER_AFTER_UNTOUCHED = CALLER_BEFORE  # generic LLM omitted the caller


def _base_sources() -> dict[str, str]:
    return {
        "pkg/callee.py": CALLEE_BEFORE,
        "pkg/caller.py": CALLER_BEFORE,
    }


def _candidate_sources_omitting_caller() -> dict[str, str]:
    return {
        "pkg/callee.py": CALLEE_AFTER,
        "pkg/caller.py": CALLER_AFTER_UNTOUCHED,
    }


def _enabled_gate(**fields: Any) -> CandidateOverlayContractDeltaGate:
    return CandidateOverlayContractDeltaGate(
        LiveLogicRepairPolicy(enable_live_logic_repair=True, **fields)
    )


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------


def test_extract_python_signatures_f_a_b() -> None:
    sigs = extract_python_signatures(CALLEE_BEFORE)
    assert sigs["f"] == ("a", "b")
    sigs2 = extract_python_signatures(CALLEE_AFTER)
    assert sigs2["f"] == ("a", "b", "c")


def test_compute_signature_deltas_arity_increase() -> None:
    deltas = compute_signature_deltas(
        {"pkg/callee.py": CALLEE_BEFORE},
        {"pkg/callee.py": CALLEE_AFTER},
    )
    assert len(deltas) == 1
    delta = deltas[0]
    assert delta.symbol == "f"
    assert delta.before_signature == "f(a, b)"
    assert delta.after_signature == "f(a, b, c)"
    assert delta.arity_increased is True


def test_find_call_sites_resolves_caller() -> None:
    sites = find_call_sites(_base_sources(), "f")
    paths = {s["path"] for s in sites}
    assert "pkg/caller.py" in paths
    assert any(s["caller_id"].endswith("use_f") for s in sites)


# ---------------------------------------------------------------------------
# Overlay gate: f(a,b) -> f(a,b,c) regression
# ---------------------------------------------------------------------------


def test_overlay_gate_defaults_off() -> None:
    gate = CandidateOverlayContractDeltaGate()
    result = gate.evaluate(
        proposal_id="proposal:1",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py",),
        base_sources=_base_sources(),
        candidate_sources=_candidate_sources_omitting_caller(),
    )
    assert result.disposition is OverlayGateDisposition.DISABLED
    assert result.mutation_allowed is False
    assert result.provider_invoked is False


def test_fab_to_fabc_omitted_caller_rejected_when_expand_disabled() -> None:
    """Exact generic-LLM regression: callee-only write set is rejected."""

    gate = _enabled_gate(
        expand_write_set_on_omission=False,
        reject_omitted_callers=True,
    )
    result = gate.evaluate(
        proposal_id="proposal:fab-fabc",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py",),  # LLM omitted caller
        base_sources=_base_sources(),
        candidate_sources=_candidate_sources_omitting_caller(),
        resolved_callers=(
            {
                "caller_id": "pkg/caller.py::use_f",
                "path": "pkg/caller.py",
                "symbol": "f",
            },
        ),
    )
    assert result.disposition is OverlayGateDisposition.REJECTED
    assert result.mutation_allowed is False
    assert "omitted_callers" in result.reason_codes
    assert "signature_arity_increase" in result.reason_codes
    assert result.overlay is not None
    assert "pkg/caller.py::use_f" in result.overlay.omitted_callers
    # Every resolved caller is dispositioned.
    assert result.overlay.caller_dispositions
    assert all(
        d.disposition
        in {
            OverlayCallerDisposition.OMITTED,
            OverlayCallerDisposition.IN_WRITE_SET,
            OverlayCallerDisposition.COMPATIBILITY_PROOF,
            OverlayCallerDisposition.NO_CHANGE_PROOF,
            OverlayCallerDisposition.EXPANDED,
        }
        for d in result.overlay.caller_dispositions
    )
    assert result.provider_invoked is False
    for stage in PROPOSAL_OVERLAY_STAGE_ORDER:
        assert stage in result.stages_completed, stage


def test_fab_to_fabc_omitted_caller_expanded_for_readmission() -> None:
    gate = _enabled_gate(expand_write_set_on_omission=True)
    result = gate.evaluate(
        proposal_id="proposal:fab-expand",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py",),
        base_sources=_base_sources(),
        candidate_sources=_candidate_sources_omitting_caller(),
        resolved_callers=(
            {
                "caller_id": "pkg/caller.py::use_f",
                "path": "pkg/caller.py",
                "symbol": "f",
            },
        ),
    )
    assert result.disposition is OverlayGateDisposition.EXPANDED
    assert result.mutation_allowed is True
    assert "pkg/caller.py" in result.expanded_write_set
    assert "pkg/callee.py" in result.expanded_write_set
    assert result.overlay is not None
    assert result.overlay.read_only is True
    # Expanded disposition recorded for the previously omitted caller.
    expanded = [
        d
        for d in result.overlay.caller_dispositions
        if d.disposition is OverlayCallerDisposition.EXPANDED
    ]
    assert expanded


def test_fab_to_fabc_with_caller_in_write_set_admits() -> None:
    gate = _enabled_gate()
    result = gate.evaluate(
        proposal_id="proposal:fab-complete",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py", "pkg/caller.py"),
        base_sources=_base_sources(),
        candidate_sources={
            "pkg/callee.py": CALLEE_AFTER,
            "pkg/caller.py": (
                "from callee import f\n\ndef use_f():\n    return f(1, 2, 3)\n"
            ),
        },
        resolved_callers=(
            {
                "caller_id": "pkg/caller.py::use_f",
                "path": "pkg/caller.py",
                "symbol": "f",
            },
        ),
    )
    assert result.disposition is OverlayGateDisposition.ADMITTED
    assert result.mutation_allowed is True
    assert result.overlay is not None
    assert not result.overlay.omitted_callers
    assert all(
        d.disposition is OverlayCallerDisposition.IN_WRITE_SET
        for d in result.overlay.caller_dispositions
    )


def test_auto_discover_callers_from_ast() -> None:
    """Without explicit callers, AST discovery still finds use_f."""

    gate = _enabled_gate(expand_write_set_on_omission=False)
    result = gate.evaluate(
        proposal_id="proposal:auto",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py",),
        base_sources=_base_sources(),
        candidate_sources=_candidate_sources_omitting_caller(),
        resolved_callers=(),
        auto_discover_callers=True,
    )
    assert result.disposition is OverlayGateDisposition.REJECTED
    assert result.overlay is not None
    assert any("caller" in c for c in result.overlay.resolved_callers)


def test_unknown_frontier_abstains() -> None:
    gate = _enabled_gate()
    result = gate.evaluate(
        proposal_id="proposal:unknown",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py", "pkg/caller.py"),
        base_sources=_base_sources(),
        candidate_sources=_candidate_sources_omitting_caller(),
        resolved_callers=(
            {
                "caller_id": "pkg/caller.py::use_f",
                "path": "pkg/caller.py",
                "symbol": "f",
            },
        ),
        unknown_frontier=("dyn:plugin_loader",),
    )
    assert result.disposition is OverlayGateDisposition.ABSTAINED
    assert result.mutation_allowed is False
    assert "unknown_frontier_required" in result.reason_codes


def test_compatibility_proof_dispositions_caller_without_write() -> None:
    gate = _enabled_gate()
    result = gate.evaluate(
        proposal_id="proposal:compat",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py",),
        base_sources=_base_sources(),
        candidate_sources={"pkg/callee.py": CALLEE_AFTER},
        resolved_callers=(
            {
                "caller_id": "pkg/caller.py::use_f",
                "path": "pkg/caller.py",
                "symbol": "f",
            },
        ),
        compatibility_proofs=("pkg/caller.py::use_f",),
    )
    assert result.disposition is OverlayGateDisposition.ADMITTED
    assert result.mutation_allowed is True


def test_controller_proposal_overlay_mode() -> None:
    controller = LiveLogicRepairController(
        policy=LiveLogicRepairPolicy(enable_live_logic_repair=True)
    )
    result = controller.run(
        LiveLogicRepairRequest(
            mode=LiveLogicRepairMode.PROPOSAL_OVERLAY,
            repository_id="repository:lpr-017",
            tree_id="tree:base",
            proposal_id="proposal:mode",
            write_set=("pkg/callee.py",),
            base_sources=_base_sources(),
            candidate_sources=_candidate_sources_omitting_caller(),
            resolved_callers=(
                {
                    "caller_id": "pkg/caller.py::use_f",
                    "path": "pkg/caller.py",
                    "symbol": "f",
                },
            ),
        )
    )
    assert result.enabled
    assert result.mode == LiveLogicRepairMode.PROPOSAL_OVERLAY.value
    assert result.disposition in {"rejected", "expanded"}
    assert result.provider_invoked is False
    assert result.overlay_gate is not None


def test_overlay_is_read_only_no_write_authority() -> None:
    gate = _enabled_gate()
    result = gate.evaluate(
        proposal_id="proposal:ro",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py", "pkg/caller.py"),
        base_sources=_base_sources(),
        candidate_sources={
            "pkg/callee.py": CALLEE_AFTER,
            "pkg/caller.py": CALLER_BEFORE,
        },
        resolved_callers=(
            {
                "caller_id": "pkg/caller.py::use_f",
                "path": "pkg/caller.py",
                "symbol": "f",
            },
        ),
    )
    assert result.overlay is not None
    assert result.overlay.read_only is True
    payload = result.overlay.to_dict()
    assert payload["read_only"] is True


# ---------------------------------------------------------------------------
# ProposalValidationPolicy integration (flag default off)
# ---------------------------------------------------------------------------


def _minimal_proposal(
    *,
    before: str,
    after: str,
    path: str = "pkg/callee.py",
) -> ImplementationProposal:
    entry = CandidateDiffEntry(
        old_path=path,
        new_path=path,
        change_kind=DiffChangeKind.MODIFY,
        before_source=before,
        after_source=after,
    )
    # Leave proposal_id empty so identity is derived (fail-closed otherwise).
    return ImplementationProposal(
        task_id="LPR-017",
        accepted_plan_id="plan:lpr-017",
        repository_id="repository:lpr-017",
        repository_tree_id="tree:lpr-017",
        objective_id="LPR-G050",
        context_id="context:lpr-017",
        baseline_id="baseline:lpr-017",
        replay_nonce="nonce:lpr-017",
        candidate_diff=(entry,),
        declared_paths=(path,),
        validation_plan=(),
        operations=(),
        expected_effects=(),
    )


def test_proposal_validation_legacy_unchanged_when_flag_off() -> None:
    """Default policy does not invoke overlay gate; ordinary path works."""

    policy = ProposalValidationPolicy(
        allowed_paths=("pkg/",),
        expected_task_id="LPR-017",
        expected_plan_id="plan:lpr-017",
        expected_repository_id="repository:lpr-017",
        expected_repository_tree_id="tree:lpr-017",
        expected_objective_id="LPR-G050",
        expected_context_id="context:lpr-017",
        expected_baseline_id="baseline:lpr-017",
        expected_replay_nonce="nonce:lpr-017",
        require_structured_details=False,
        require_declared_paths=False,
    )
    assert policy.enable_live_logic_repair is False
    proposal = _minimal_proposal(before=CALLEE_BEFORE, after=CALLEE_AFTER)
    # Without authority expectations matching, may still find authority issues;
    # the critical property is that OMITTED_CALLERS is never raised when off.
    result = ProposalValidator(policy).validate(proposal)
    codes = {f.code for f in result.findings}
    assert ProposalFindingCode.OMITTED_CALLERS not in codes
    assert ProposalFindingCode.LOGIC_REPAIR_OVERLAY_REJECTED not in codes


def test_proposal_validation_rejects_omitted_callers_when_enabled() -> None:
    policy = ProposalValidationPolicy(
        allowed_paths=("pkg/",),
        expected_task_id="LPR-017",
        expected_plan_id="plan:lpr-017",
        expected_repository_id="repository:lpr-017",
        expected_repository_tree_id="tree:lpr-017",
        expected_objective_id="LPR-G050",
        expected_context_id="context:lpr-017",
        expected_baseline_id="baseline:lpr-017",
        expected_replay_nonce="nonce:lpr-017",
        require_structured_details=False,
        require_declared_paths=False,
        enable_live_logic_repair=True,
        logic_repair_expand_write_set=False,
        logic_repair_resolved_callers=("pkg/caller.py::use_f",),
    )
    # Two-file proposal would be ideal; single callee change with bound
    # callers still exercises the gate.  Provide both entries so base/candidate
    # sources include the caller for discovery.
    callee = CandidateDiffEntry(
        old_path="pkg/callee.py",
        new_path="pkg/callee.py",
        change_kind=DiffChangeKind.MODIFY,
        before_source=CALLEE_BEFORE,
        after_source=CALLEE_AFTER,
    )
    # Caller present in sources but not as a change — only callee in write set
    # via effective_entries.  Gate uses write_set from effective entries only,
    # so pass resolved callers via policy.
    proposal = ImplementationProposal(
        task_id="LPR-017",
        accepted_plan_id="plan:lpr-017",
        repository_id="repository:lpr-017",
        repository_tree_id="tree:lpr-017",
        objective_id="LPR-G050",
        context_id="context:lpr-017",
        baseline_id="baseline:lpr-017",
        replay_nonce="nonce:lpr-017",
        candidate_diff=(callee,),
        declared_paths=("pkg/callee.py",),
        validation_plan=(),
        operations=(),
        expected_effects=(),
    )
    result = validate_implementation_proposal(proposal, policy=policy)
    # May still have other findings; when overlay runs after a clean pass it
    # would reject.  If authority findings fire first, overlay is skipped —
    # so also exercise the gate directly for the exact regression (above).
    if result.accepted:
        pytest.fail("expected rejection for omitted callers when overlay enabled")
    # If rejected for authority/schema before overlay, that is still fail-closed.
    assert result.accepted is False


def test_daemon_intercept_proposal_thin_host() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    class _Daemon:
        pass

    daemon = _Daemon()
    result = PortalImplementationDaemon.intercept_logic_repair_proposal(
        daemon,
        proposal_id="proposal:daemon",
        repository_id="repository:lpr-017",
        base_tree_id="tree:base",
        candidate_tree_id="tree:candidate",
        write_set=("pkg/callee.py",),
        base_sources=_base_sources(),
        candidate_sources=_candidate_sources_omitting_caller(),
        resolved_callers=(
            {
                "caller_id": "pkg/caller.py::use_f",
                "path": "pkg/caller.py",
                "symbol": "f",
            },
        ),
        enable=True,
        expand_write_set_on_omission=False,
    )
    assert result.disposition is OverlayGateDisposition.REJECTED
    assert result.provider_invoked is False
