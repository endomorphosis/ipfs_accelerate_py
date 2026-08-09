"""DCR-033: reconstruct proofs and preserve minimal counterexamples.

Acceptance:
* Unreconstructable proofs become invalid.
* Provider ``verified`` alone never mints reconstruction.
* Fabricated proof children and incomplete terms fail closed.
* Every refutation minimizes and replays against a bound graph and live
  transcript without inferred observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.kernel_reconstruction import (
    COUNTEREXAMPLE_INTERFACE,
    DEFAULT_KERNEL_VERSION,
    PROOF_KERNEL_RECEIPT_INTERFACE,
    Counterexample,
    CounterexampleReplayStatus,
    KernelReconstructionError,
    ProofClaim,
    ProofKernelReceipt,
    ReconstructionStatus,
    materialize_proof_kernel_reconstruction_artifact,
    minimize_counterexample,
    proof_term_digest,
    reconstruct_proof,
    replay_counterexample,
    write_proof_kernel_reconstruction_artifact,
)


def _graph() -> dict:
    return {
        "graph_root": "baguqeeragraphroot0000000000000000000000000000000000001",
        "edges": [
            {
                "edge_id": "edge:route-to-dispatcher",
                "kind": "route_to_dispatcher",
                "consumer_id": "swissknife/ipfs_accelerate_py/expected.only.tool",
                "resolution": "expected_only",
            },
            {
                "edge_id": "edge:handler-to-effect",
                "kind": "handler_to_effect",
                "consumer_id": "swissknife/ipfs_accelerate_py/accelerate.inference",
                "resolution": "resolved",
            },
        ],
    }


def _transcript() -> dict:
    return {
        "exchanges": [
            {
                "receipt_cid": "receipt:live-tools-list",
                "role": "accelerate",
                "method": "tools/list",
                "terminal_state": "passed",
            },
            {
                "receipt_cid": "receipt:live-unknown-call",
                "role": "accelerate",
                "method": "tools/call",
                "terminal_state": "refuted",
            },
        ]
    }


def test_interfaces_are_declared() -> None:
    assert PROOF_KERNEL_RECEIPT_INTERFACE == "ProofKernelReceipt@1"
    assert COUNTEREXAMPLE_INTERFACE == "Counterexample@1"
    assert ProofKernelReceipt.INTERFACE == PROOF_KERNEL_RECEIPT_INTERFACE
    assert Counterexample.INTERFACE == COUNTEREXAMPLE_INTERFACE


def test_reconstruct_proof_accepts_independent_bound_term() -> None:
    term = {
        "theorem": "contract_edge_sound",
        "steps": ["intro", "exact h"],
        "conclusion": "True",
    }
    claim = ProofClaim(
        obligation_id="obligation:dcr033:sound",
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        kernel_version=DEFAULT_KERNEL_VERSION,
        root_ids=("ipfs-accelerate", "swissknife"),
        tree_id="tree:test",
        graph_root="graph:test",
        independent=True,
        proof_children=tuple(
            proof_term_digest(step) for step in term["steps"]
        ),
    )
    receipt = reconstruct_proof(
        claim,
        expected_root_ids=("ipfs-accelerate", "swissknife"),
        expected_tree_id="tree:test",
        expected_graph_root="graph:test",
    )
    assert receipt.valid
    assert receipt.status is ReconstructionStatus.RECONSTRUCTED
    assert receipt.reconstructed is True
    assert receipt.independent is True
    assert receipt.proof_term_digest == proof_term_digest(term)
    assert "independent_reconstruction_ok" in receipt.reason_codes


def test_unreconstructable_proof_becomes_invalid() -> None:
    claim = ProofClaim(
        obligation_id="obligation:dcr033:broken",
        proof_term="theorem broken : False := by sorry",
        certificate_digest="sha256:deadbeef",
        independent=True,
    )
    receipt = reconstruct_proof(claim)
    assert not receipt.valid
    assert receipt.status is ReconstructionStatus.INVALID
    assert "incomplete_proof" in receipt.reason_codes
    assert "certificate_digest_mismatch" in receipt.reason_codes


def test_provider_verified_without_independence_is_invalid() -> None:
    term = "theorem ok : True := by trivial"
    claim = ProofClaim(
        obligation_id="obligation:dcr033:provider-only",
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        provider_status="verified",
        independent=False,
    )
    receipt = reconstruct_proof(claim)
    assert receipt.status is ReconstructionStatus.INVALID
    assert "provider_status_not_independent" in receipt.reason_codes
    assert "independent_reconstruction_required" in receipt.reason_codes


def test_fabricated_proof_children_are_rejected() -> None:
    term = {"steps": ["alpha", "beta"], "conclusion": "P"}
    claim = ProofClaim(
        obligation_id="obligation:dcr033:fabricated",
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        independent=True,
        proof_children=("sha256:fabricated-child",),
    )
    receipt = reconstruct_proof(claim)
    assert receipt.status is ReconstructionStatus.INVALID
    assert "fabricated_proof_children" in receipt.reason_codes


def test_independent_checker_can_admit_reconstruction() -> None:
    term = "exact True.intro"
    claim = ProofClaim(
        obligation_id="obligation:dcr033:checker",
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        independent=False,
    )
    receipt = reconstruct_proof(
        claim,
        independent_checker=lambda _claim: True,
    )
    assert receipt.valid
    assert receipt.independent is True


def test_independent_checker_rejection_invalidates() -> None:
    term = "exact True.intro"
    claim = ProofClaim(
        obligation_id="obligation:dcr033:checker-reject",
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        independent=True,
    )
    receipt = reconstruct_proof(
        claim,
        independent_checker=lambda _claim: False,
    )
    assert receipt.status is ReconstructionStatus.INVALID
    assert "independent_checker_rejected" in receipt.reason_codes


def test_refutation_marked_term_is_refuted_not_proved() -> None:
    term = "refuted by counterexample model {x = 1}"
    claim = ProofClaim(
        obligation_id="obligation:dcr033:refuted-term",
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        independent=True,
    )
    receipt = reconstruct_proof(claim)
    assert receipt.status is ReconstructionStatus.REFUTED
    assert not receipt.reconstructed


def test_minimize_counterexample_replays_against_graph_and_transcript() -> None:
    seed = {
        "obligation_id": "obligation:dcr033:missing-tool",
        "violated_property": "tools.call.expected.only.tool",
        "summary": "expected tool absent from live tools/list",
        "edge_id": "edge:route-to-dispatcher",
        "edge_kind": "route_to_dispatcher",
        "consumer_id": "swissknife/ipfs_accelerate_py/expected.only.tool",
        "receipt_cid": "receipt:live-tools-list",
        "role": "accelerate",
        "method": "tools/list",
        "timestamp": "2026-08-09T00:00:00Z",
        "raw_output": "DROP THIS",
        "expected_outcome": "must not survive",
        "stdout": "noise",
        "reason_code": "tool_absent_from_live_tools_list",
    }
    minimized = minimize_counterexample(
        seed,
        graph=_graph(),
        transcript=_transcript(),
        require_replay=True,
    )
    assert minimized.minimized is True
    assert minimized.inferred_observations is False
    assert "edge:route-to-dispatcher" in minimized.graph_edge_ids
    assert "receipt:live-tools-list" in minimized.transcript_receipt_ids
    assert "raw_output" not in minimized.witness
    assert "timestamp" not in minimized.witness
    assert "expected_outcome" not in minimized.witness
    assert "stdout" not in minimized.witness

    replay = replay_counterexample(
        minimized,
        graph=_graph(),
        transcript=_transcript(),
    )
    assert replay.replayed
    assert replay.status is CounterexampleReplayStatus.REPLAYED
    assert "edge:route-to-dispatcher" in replay.matched_graph_edge_ids
    assert "receipt:live-tools-list" in replay.matched_transcript_receipt_ids


def test_inferred_observations_are_rejected() -> None:
    with pytest.raises(KernelReconstructionError, match="inferred"):
        minimize_counterexample(
            {
                "obligation_id": "obligation:dcr033:inferred",
                "violated_property": "p",
                "summary": "inferred",
                "edge_id": "edge:route-to-dispatcher",
                "receipt_cid": "receipt:live-tools-list",
                "inferred_observation": True,
            },
            graph=_graph(),
            transcript=_transcript(),
        )


def test_replay_fails_closed_without_graph_anchor() -> None:
    ce = Counterexample(
        obligation_id="obligation:dcr033:no-graph",
        violated_property="p",
        summary="missing edge",
        witness={"edge_id": "edge:does-not-exist"},
        graph_edge_ids=("edge:does-not-exist",),
        transcript_receipt_ids=("receipt:live-tools-list",),
    )
    replay = replay_counterexample(
        ce,
        graph=_graph(),
        transcript=_transcript(),
    )
    assert not replay.replayed
    assert replay.status is CounterexampleReplayStatus.MISSING_GRAPH_ANCHOR


def test_replay_fails_closed_without_transcript_anchor() -> None:
    ce = Counterexample(
        obligation_id="obligation:dcr033:no-transcript",
        violated_property="p",
        summary="missing receipt",
        witness={"edge_id": "edge:route-to-dispatcher"},
        graph_edge_ids=("edge:route-to-dispatcher",),
        transcript_receipt_ids=("receipt:missing",),
    )
    replay = replay_counterexample(
        ce,
        graph=_graph(),
        transcript=_transcript(),
    )
    assert not replay.replayed
    assert replay.status is CounterexampleReplayStatus.MISSING_TRANSCRIPT_ANCHOR


def test_counterexample_requires_anchors() -> None:
    with pytest.raises(KernelReconstructionError, match="anchors"):
        Counterexample(
            obligation_id="obligation:dcr033:no-anchor",
            violated_property="p",
            summary="no anchors",
            witness={"x": 1},
        )


def test_receipt_round_trip() -> None:
    term = "rfl"
    claim = ProofClaim(
        obligation_id="obligation:dcr033:roundtrip",
        proof_term=term,
        certificate_digest=proof_term_digest(term),
        independent=True,
        root_ids=("ipfs-accelerate",),
    )
    receipt = reconstruct_proof(claim)
    restored = ProofKernelReceipt.from_dict(receipt.to_record())
    assert restored.content_id == receipt.content_id
    assert restored.valid is receipt.valid


def test_artifact_materialization(tmp_path: Path) -> None:
    term = "exact True.intro"
    receipt = reconstruct_proof(
        ProofClaim(
            obligation_id="obligation:dcr033:artifact",
            proof_term=term,
            certificate_digest=proof_term_digest(term),
            independent=True,
            root_ids=("ipfs-accelerate", "swissknife"),
            tree_id="tree:artifact",
        )
    )
    counterexample = minimize_counterexample(
        {
            "obligation_id": "obligation:dcr033:artifact-ce",
            "violated_property": "tools.call.expected.only.tool",
            "summary": "route blocked",
            "edge_id": "edge:route-to-dispatcher",
            "receipt_cid": "receipt:live-unknown-call",
        },
        graph=_graph(),
        transcript=_transcript(),
    )
    artifact = materialize_proof_kernel_reconstruction_artifact(
        tree_id="tree:artifact",
        receipts=(receipt,),
        counterexamples=(counterexample,),
    )
    assert artifact["task_id"] == "DCR-033"
    assert artifact["interface"] == PROOF_KERNEL_RECEIPT_INTERFACE
    assert artifact["counterexample_interface"] == COUNTEREXAMPLE_INTERFACE
    assert artifact["reconstructed_count"] == 1
    assert artifact["counterexample_count"] == 1
    assert artifact["acceptance"]["unreconstructable_proof_becomes_invalid"] is True
    assert (
        artifact["acceptance"][
            "refutations_replay_against_bound_graph_and_transcript"
        ]
        is True
    )

    path = write_proof_kernel_reconstruction_artifact(
        tmp_path / "proof-kernel-reconstruction.json",
        artifact,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["task_id"] == "DCR-033"
    assert loaded["reconstructed_count"] == 1
