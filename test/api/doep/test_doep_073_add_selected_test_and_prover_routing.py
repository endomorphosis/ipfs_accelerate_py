"""Independent current-tree checks for DOEP-073 selected-test and prover routing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import (
    SELECTED_TEST_PROVER_ROUTE_INTERFACE,
    SELECTED_TEST_PROVER_ROUTE_SCHEMA,
    SelectedTestProverRoute,
    TaskProposalRouterError,
    route_selected_test_and_prover,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    MultiProverRouter,
    PortfolioPlan,
    PropertyKind,
    PropertyObligation,
    route_obligation,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    AnalysisKind,
    ModelRouteFacts,
)
from ipfs_accelerate_py.agent_supervisor.verification.selection import (
    AffectedVerificationSelection,
    FallbackMode,
    VerificationCatalog,
    select_affected_verification,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
ROUTER_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/planning/task_proposal_router.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-073.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-073.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/task_proposal_router.py",
    "test/api/doep/test_doep_073_add_selected_test_and_prover_routing.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-073.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-073.json",
)
TASK_CID = "sha256:ab5418a0219be9ce0a4b04908aa12d2e04586a849025efa843de9d20920cc331"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}

TEST_A = "test/api/test_mod.py::test_fn"
TEST_B = "test/api/test_other.py::test_unrelated"
PROOF_A = content_identity({"artifact": "doep-073-proof-a", "schema": "fixture-artifact@1"})


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _edge(
    source: str,
    target: str,
    kind: str,
    *,
    disposition: str = "exact",
    opaque: bool = False,
    critical: bool = True,
    edge_id: str,
) -> dict[str, object]:
    return {
        "source": source,
        "target": target,
        "kind": kind,
        "disposition": disposition,
        "truncated": False,
        "opaque": opaque,
        "uncovered": False,
        "critical": critical,
        "edge_id": edge_id,
    }


def _catalog(**overrides: object) -> VerificationCatalog:
    base: dict[str, object] = {
        "tests": [TEST_A, TEST_B],
        "proof_obligations": [PROOF_A],
        "proof_obligation_dependencies": {PROOF_A: ["pkg.mod.fn"]},
    }
    base.update(overrides)
    return VerificationCatalog(**base)


def _obligation() -> PropertyObligation:
    return PropertyObligation(
        obligation_id=PROOF_A,
        property_kind=PropertyKind.FINITE_CONSTRAINT,
        statement="reviewed finite constraint for DOEP-073",
        premise_ids=("premise:a",),
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_routes_through_canonical_selection_and_multi_prover_without_execution() -> None:
    assert route_selected_test_and_prover.__module__.endswith("task_proposal_router")
    assert select_affected_verification.__module__.endswith("verification.selection")
    assert route_obligation.__module__.endswith("proof.multi_prover_router")
    assert MultiProverRouter.__module__.endswith("proof.multi_prover_router")

    route = route_selected_test_and_prover(
        changed_symbols=("pkg.mod.fn",),
        edges=(
            _edge("pkg.mod.fn", TEST_A, "tested_by", edge_id="t1"),
            _edge("pkg.mod.fn", PROOF_A, "proved_by", edge_id="p1"),
        ),
        catalog=_catalog(),
        proof_obligations=(_obligation(),),
    )
    assert isinstance(route, SelectedTestProverRoute)
    assert isinstance(route.selection, AffectedVerificationSelection)
    assert route.selection.fallback_mode is FallbackMode.EXACT
    assert route.selected_tests == (TEST_A,)
    assert route.affected_proof_obligation_cids == (PROOF_A,)
    assert len(route.portfolio_plans) == 1
    assert isinstance(route.portfolio_plans[0], PortfolioPlan)
    assert route.prover_ids == ("z3", "cvc5")
    assert isinstance(route.model_route_facts, ModelRouteFacts)
    assert route.model_route_facts.analysis_kind is AnalysisKind.LOCALIZED_EXACT
    assert route.model_route_facts.dependency_cone_size >= 1
    assert route.test_execution_count == 0
    assert route.prover_execution_count == 0
    assert route.proof_authoritative is False
    assert route.to_dict()["schema"] == SELECTED_TEST_PROVER_ROUTE_SCHEMA
    assert route.to_dict()["interface"] == SELECTED_TEST_PROVER_ROUTE_INTERFACE
    assert '"source":' not in json.dumps(route.to_dict())


def test_critical_uncertainty_and_missing_obligation_bodies_fail_closed() -> None:
    broadened = route_selected_test_and_prover(
        changed_symbols=("pkg.mod.fn",),
        edges=(
            _edge(
                "pkg.mod.fn",
                "pkg.dyn",
                "depends_on",
                disposition="opaque",
                opaque=True,
                edge_id="o1",
            ),
        ),
        # Full-suite fallback expands known tests; keep proofs verified-empty so
        # this case exercises uncertainty routing without inventing obligations.
        catalog=_catalog(proof_obligations=(), proof_obligation_dependencies={}),
    )
    assert broadened.selection.broader_selection_required is True
    assert broadened.selection.full_suite_required is True
    assert broadened.model_route_facts.full_suite_required is True
    assert broadened.model_route_facts.full_suite_pending is True
    assert broadened.portfolio_plans == ()

    with pytest.raises(TaskProposalRouterError, match="lack supplied obligation bodies"):
        route_selected_test_and_prover(
            changed_symbols=("pkg.mod.fn",),
            edges=(_edge("pkg.mod.fn", PROOF_A, "proved_by", edge_id="p-missing"),),
            catalog=_catalog(),
        )

    with pytest.raises(TaskProposalRouterError, match="malformed selection input"):
        route_selected_test_and_prover(
            changed_symbols=("pkg.mod.fn",),
            edges=("not-an-edge",),
            catalog=_catalog(),
        )


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-073"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "route_selected_test_and_prover"
    assert manifest["canonical_extension"]["carrier"] == "SelectedTestProverRoute"
    assert "select_affected_verification" in manifest["canonical_extension"]["binding"]
    assert "route_obligation" in manifest["canonical_extension"]["binding"]
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(ROUTER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
