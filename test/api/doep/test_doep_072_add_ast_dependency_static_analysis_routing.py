"""Independent current-tree checks for DOEP-072 analysis routing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    AST_DEPENDENCY_STATIC_ROUTE_SCHEMA,
    AnalysisASTIndexError,
    StaticAnalysisState,
    build_analysis_ast_index,
    route_ast_dependency_static_analysis,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import build_python_ast_blob_record
from ipfs_accelerate_py.agent_supervisor.verification.model_route import AnalysisKind


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/analysis_ast_index.py"
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-072.json"
RECEIPT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-072.json"
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/analysis/analysis_ast_index.py",
    "test/api/doep/test_doep_072_add_ast_dependency_static_analysis_routing.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-072.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-072.json",
)
TASK_CID = "sha256:da9259c71470767b7c97927a0e91dab572b75e0f20b3bfe3d9071249e98550f9"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {"commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f", "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7"},
    "ipfs_datasets_py": {"commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7", "tree": "456e09b51d6a07a3a5873436df24054768195320"},
    "ipfs_kit_py": {"commit": "b6c65ba732733d7e33852713ba18aa3b12235668", "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"},
    "lift_coding": {"commit": "bb8869ed72eb7002434345d9969efee729c4f7f6", "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42"},
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _index():
    return build_analysis_ast_index(
        [
            ("pkg/api.py", build_python_ast_blob_record("from pkg.service import run\n\ndef dispatch():\n    return run()\n", blob_identity="blob:api")),
            ("pkg/service.py", build_python_ast_blob_record("from pkg.storage import read\n\ndef run():\n    return read()\n", blob_identity="blob:service")),
            ("pkg/storage.py", build_python_ast_blob_record("def read():\n    return 1\n", blob_identity="blob:storage")),
        ]
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_routes_through_canonical_ast_index_and_existing_model_facts() -> None:
    index = _index()
    route = route_ast_dependency_static_analysis(
        index,
        paths=("pkg/api.py",),
        query="dispatch run",
        static_analysis_state=StaticAnalysisState.PASSED,
        static_check_ids=("python-ast",),
    )
    assert route.index_id == index.index_id
    assert route.dependency_paths == ("pkg/api.py", "pkg/service.py", "pkg/storage.py")
    assert route.model_route_facts.analysis_kind is AnalysisKind.LOCALIZED_EXACT
    assert route.model_route_facts.dependency_cone_size == 3
    assert route.model_route_facts.changed_file_count == 1
    assert route.model_route_facts.opaque_dependency_count == 0
    assert route.ast_evidence.evidence
    assert route.to_dict()["schema"] == AST_DEPENDENCY_STATIC_ROUTE_SCHEMA
    assert '"source":' not in json.dumps(route.to_dict())


def test_static_failure_and_declared_opaque_dependency_fail_closed_to_existing_ladder() -> None:
    route = _index().route_ast_dependency_static_analysis(
        paths=("pkg/api.py",),
        static_analysis_state="failed",
        opaque_dependencies=("generated.vendor.runtime",),
    )
    assert route.model_route_facts.analysis_kind is AnalysisKind.OPAQUE
    assert route.model_route_facts.unresolved_obligation_count == 1
    assert route.model_route_facts.opaque_dependency_count == 1
    assert route.unresolved_dependencies == ("generated.vendor.runtime",)

    with pytest.raises(AnalysisASTIndexError, match="absent from the current snapshot"):
        _index().route_ast_dependency_static_analysis(paths=("missing.py",))
    with pytest.raises(AnalysisASTIndexError, match="requires check identities"):
        _index().route_ast_dependency_static_analysis(static_analysis_state="passed")


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in ((manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"), (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1")):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-072"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(INDEX_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == "pending_independent_fenced_supervisor"
