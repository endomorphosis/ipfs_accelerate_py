"""Independent current-tree checks for DOEP-071 exact receipt reuse."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from ipfs_accelerate_py.agent_supervisor.analysis.test_reuse_eligibility import (
    EXACT_RECEIPT_REUSE_INTERFACE,
    lookup_exact_receipt_reuse,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    CacheReuseDisposition,
)
from ipfs_accelerate_py.agent_supervisor.verification.receipt_cache import (
    VerificationReceiptCache,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
ELIGIBILITY_PATH = ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/test_reuse_eligibility.py"
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-071.json"
RECEIPT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-071.json"
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/analysis/test_reuse_eligibility.py",
    "test/api/doep/test_doep_071_add_exact_receipt_reuse.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-071.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-071.json",
)
TASK_CID = "sha256:80f10b52904acf7f9d3ada85c9fab68586d6425192ba7449702d455a53ae2bcd"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {"commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f", "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7"},
    "ipfs_datasets_py": {"commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7", "tree": "456e09b51d6a07a3a5873436df24054768195320"},
    "ipfs_kit_py": {"commit": "b6c65ba732733d7e33852713ba18aa3b12235668", "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"},
    "lift_coding": {"commit": "bb8869ed72eb7002434345d9969efee729c4f7f6", "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42"},
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_fixture_module(filename: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ACCELERATE_ROOT / "test/api" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_exact_reuse_delegates_to_the_existing_canonical_cache(tmp_path: Path) -> None:
    cache_fixtures = _load_fixture_module(
        "test_agent_supervisor_verification_receipt_cache.py", "doep071_cache_fixtures"
    )
    eligibility_fixtures = _load_fixture_module(
        "test_agent_supervisor_test_reuse_eligibility.py", "doep071_eligibility_fixtures"
    )
    key = cache_fixtures._key()
    cache = cache_fixtures._cache(tmp_path)
    receipt = cache_fixtures._type_check_receipt(key, label="doep-071")
    eligibility = eligibility_fixtures.evaluate_reuse_eligibility(
        static_trace=eligibility_fixtures._static_pure(tmp_path),
        runtime_trace=eligibility_fixtures._runtime_complete(tmp_path),
        repository_forest_cid=eligibility_fixtures._forest_cid("doep-071"),
    )
    assert eligibility.reusable is True
    assert cache.admit(receipt).success is True

    reused = lookup_exact_receipt_reuse(
        eligibility=eligibility, receipt_cache=cache, receipt_key=key
    )
    assert reused.disposition is CacheReuseDisposition.REUSED
    assert reused.candidate_receipt is not None
    assert reused.candidate_receipt.receipt_id == receipt.receipt_id

    changed_key = cache_fixtures._key(tool_version="1.18.3")
    miss = lookup_exact_receipt_reuse(
        eligibility=eligibility, receipt_cache=cache, receipt_key=changed_key
    )
    assert miss.disposition is CacheReuseDisposition.MISSING


def test_ineligible_decision_cannot_consult_or_reuse_a_receipt(tmp_path: Path) -> None:
    cache_fixtures = _load_fixture_module(
        "test_agent_supervisor_verification_receipt_cache.py", "doep071_denial_fixtures"
    )
    eligibility_fixtures = _load_fixture_module(
        "test_agent_supervisor_test_reuse_eligibility.py", "doep071_denial_eligibility"
    )
    key = cache_fixtures._key()
    cache = cache_fixtures._cache(tmp_path)
    assert cache.admit(cache_fixtures._type_check_receipt(key, label="denied")).success
    denied = eligibility_fixtures.evaluate_reuse_eligibility(
        repository_forest_cid=eligibility_fixtures._forest_cid("denied")
    )
    decision = lookup_exact_receipt_reuse(
        eligibility=denied, receipt_cache=cache, receipt_key=key
    )
    assert decision.disposition is CacheReuseDisposition.POLICY_REJECTED
    assert decision.candidate_receipt is None


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in ((manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"), (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1")):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-071"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert EXACT_RECEIPT_REUSE_INTERFACE == "ExactReceiptReuse@1"
    assert VerificationReceiptCache.__module__ == "ipfs_accelerate_py.agent_supervisor.verification.receipt_cache"
    assert manifest["canonical_extension"]["carrier"] == "VerificationReceiptCache"
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(ELIGIBILITY_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
