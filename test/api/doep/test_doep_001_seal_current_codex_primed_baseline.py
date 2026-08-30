"""Independent evidence that DOEP-001 sealed the Codex-primed baseline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "benchmarks/agent_supervisor/doep/baseline_manifest.py"
TEST_PATH = (
    ROOT / "test/api/doep/test_doep_001_seal_current_codex_primed_baseline.py"
)
OUTPUT_MANIFEST_PATH = (
    ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-001.json"
)
RECEIPT_PATH = (
    ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-001.json"
)

DECLARED_OUTPUTS = (
    "benchmarks/agent_supervisor/doep/baseline_manifest.py",
    "test/api/doep/test_doep_001_seal_current_codex_primed_baseline.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-001.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-001.json",
)

EXPECTED_BASES = {
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
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
}

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict), path
    assert raw == _canonical(payload), f"{path} must be canonical JSON"
    return payload


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "doep_baseline_manifest", MANIFEST_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_content_digest(path: Path, omit_keys: set[str]) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    filtered = {key: value for key, value in payload.items() if key not in omit_keys}
    return _sha256_bytes(
        json.dumps(
            filtered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )


def _expected_output_digests() -> dict[str, str]:
    return {
        DECLARED_OUTPUTS[0]: _sha256_file(ROOT / DECLARED_OUTPUTS[0]),
        DECLARED_OUTPUTS[1]: _sha256_file(ROOT / DECLARED_OUTPUTS[1]),
        DECLARED_OUTPUTS[2]: _json_content_digest(
            OUTPUT_MANIFEST_PATH,
            {
                "changed_path_digest",
                "output_digests",
                "output_digest_method",
                "receipt_cid",
            },
        ),
        DECLARED_OUTPUTS[3]: _json_content_digest(
            RECEIPT_PATH,
            {
                "changed_path_digest",
                "output_content_digests",
                "receipt_cid",
                "required_evidence",
            },
        ),
    }


def _changed_path_digest() -> str:
    digests = _expected_output_digests()
    entries = [
        {"path": path, "sha256": digests[path]} for path in DECLARED_OUTPUTS
    ]
    return _sha256_bytes(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def test_declared_outputs_exist() -> None:
    for relative in DECLARED_OUTPUTS:
        path = ROOT / relative
        assert path.is_file(), f"missing declared output: {relative}"
        assert path.stat().st_size > 0


def test_baseline_manifest_seals_codex_primed_identity() -> None:
    module = _load_baseline_module()
    manifest = module.build_baseline_manifest()
    module.assert_sealed_identity(manifest)

    assert manifest["schema"].endswith("codex-primed-baseline-manifest@1")
    assert manifest["plan_revision"] == "DOEP-PLAN-V5"
    assert manifest["plan_cid"] == (
        "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
    )
    assert manifest["task_id"] == "DOEP-001"
    assert manifest["task_cid"] == (
        "sha256:940f797121528fe86fe8f86fb808d3cb8c60e6bdd20b20e3439b45bc11a94b74"
    )
    assert manifest["mode"] == "codex_primed"
    assert manifest["baseline"]["base_repositories"] == EXPECTED_BASES
    assert manifest["measurements"]["status"] == "not_run"
    assert manifest["promotion_eligible"] is False
    assert manifest["competing_subsystem_created"] is False
    assert manifest["authority"] is False
    assert SHA256_RE.fullmatch(manifest["manifest_cid"])

    env = manifest["sealed_validation_environment"]
    assert env["path"] == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert env["python_interpreter"] == "/usr/bin/python3.12"
    assert env["formal_toolchain_deployment_identity"] == (
        "fa9916ef2e4a927ae633309de7ef09b9d85e798e9a6711ae13bc502c6015e0c8"
    )

    rebuilt = module.build_baseline_manifest()
    assert rebuilt == manifest
    assert rebuilt["manifest_cid"] == manifest["manifest_cid"]


def test_output_manifest_indexes_exact_outputs() -> None:
    output = _load_json(OUTPUT_MANIFEST_PATH)
    assert output["schema"] == (
        "ipfs_accelerate_py.agent_supervisor.doep.task-output-manifest@1"
    )
    assert output["task_id"] == "DOEP-001"
    assert output["plan_revision"] == "DOEP-PLAN-V5"
    assert output["plan_cid"] == (
        "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
    )
    assert output["task_cid"] == (
        "sha256:940f797121528fe86fe8f86fb808d3cb8c60e6bdd20b20e3439b45bc11a94b74"
    )
    assert output["primary_output"] == DECLARED_OUTPUTS[0]
    assert output["exact_outputs"] == list(DECLARED_OUTPUTS)
    assert output["receipt_output"] == DECLARED_OUTPUTS[3]
    assert output["outputs_present"] is True
    assert output["completion_authoritative"] is False

    digests = output["output_digests"]
    assert isinstance(digests, dict)
    assert digests == _expected_output_digests()
    assert output["source_digests"][DECLARED_OUTPUTS[0]] == digests[DECLARED_OUTPUTS[0]]
    assert output["source_digests"][DECLARED_OUTPUTS[1]] == digests[DECLARED_OUTPUTS[1]]


def test_candidate_receipt_carries_required_evidence() -> None:
    receipt = _load_json(RECEIPT_PATH)
    assert receipt["schema"] == (
        "ipfs_accelerate_py.agent_supervisor.doep.task-receipt@1"
    )
    assert receipt["task_id"] == "DOEP-001"
    assert receipt["title"] == "Seal current Codex-primed baseline"
    assert receipt["board_namespace"] == (
        "agent-supervisor-direct-objective-and-event-driven-planning-v1"
    )
    assert receipt["plan_revision"] == "DOEP-PLAN-V5"
    assert receipt["plan_cid"] == (
        "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
    )
    assert receipt["task_cid"] == (
        "sha256:940f797121528fe86fe8f86fb808d3cb8c60e6bdd20b20e3439b45bc11a94b74"
    )
    assert receipt["status"] == "implemented"
    assert receipt["completion_authoritative"] is False
    assert receipt["worker_completion_insufficient"] is True
    assert receipt["authority"] is False
    assert receipt["promotion_eligible"] is False

    assert receipt["write_scope"] == list(DECLARED_OUTPUTS)
    assert receipt["changed_paths"] == list(DECLARED_OUTPUTS)
    assert receipt["exact_outputs"] == list(DECLARED_OUTPUTS)
    assert receipt["outputs_present"] is True

    bases = receipt["base_repositories"]
    assert bases == EXPECTED_BASES
    for identity in bases.values():
        assert SHA1_RE.fullmatch(identity["commit"])
        assert SHA1_RE.fullmatch(identity["tree"])

    digest = receipt["changed_path_digest"]
    assert SHA256_RE.fullmatch(digest)
    assert digest == _changed_path_digest()
    assert receipt["output_content_digests"] == _expected_output_digests()

    evidence = receipt["required_evidence"]
    assert evidence["source_commit_tree_gitlinks"] is True
    assert evidence["changed_path_digest"] == digest
    assert evidence["test_results"]["command"] == [
        "python3",
        "-m",
        "pytest",
        "test/api/doep/test_doep_001_seal_current_codex_primed_baseline.py",
        "-q",
    ]
    assert evidence["test_results"]["status"] == "declared_independent_pytest"
    assert evidence["proof_selection"]["selection_status"] == "verified_empty"
    assert SHA256_RE.fullmatch(evidence["receipt_cid"])
    assert evidence["receipt_cid"] == receipt["receipt_cid"]
    assert isinstance(evidence["limitations"], list)
    assert evidence["limitations"]
    assert evidence["verifier_admission"] == (
        "candidate_receipt_only; independent validation required"
    )

    sealed = receipt["sealed_baseline"]
    assert sealed["mode"] == "codex_primed"
    assert sealed["manifest_path"] == DECLARED_OUTPUTS[0]
    assert sealed["manifest_sha256"] == _sha256_file(MANIFEST_PATH)
    assert sealed["measurements_status"] == "not_run"
    assert sealed["competing_subsystem_created"] is False

    assert receipt["validation"]["profile"] == (
        "doep-validation/DOEP-PLAN-V5/DOEP-001@1"
    )
    assert "truth_safety_authority_or_promotion_gate_weakened" not in receipt.get(
        "postcondition_violations", []
    )
    assert receipt["postconditions_hold"] is True
    assert receipt["gates_weakened"] is False

    body = {key: value for key, value in receipt.items() if key != "receipt_cid"}
    evidence_body = dict(body["required_evidence"])
    evidence_body.pop("receipt_cid", None)
    body["required_evidence"] = evidence_body
    expected_cid = _sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    assert receipt["receipt_cid"] == expected_cid


def test_output_manifest_and_receipt_agree() -> None:
    output = _load_json(OUTPUT_MANIFEST_PATH)
    receipt = _load_json(RECEIPT_PATH)
    assert output["task_cid"] == receipt["task_cid"]
    assert output["plan_cid"] == receipt["plan_cid"]
    assert output["exact_outputs"] == receipt["exact_outputs"]
    assert output["output_digests"] == receipt["output_content_digests"]
    assert output["output_digests"] == _expected_output_digests()
    assert output["receipt_cid"] == receipt["receipt_cid"]
    assert output["changed_path_digest"] == receipt["changed_path_digest"]
    assert output["changed_path_digest"] == _changed_path_digest()


def test_no_competing_subsystem_and_no_promotion_claim() -> None:
    module = _load_baseline_module()
    manifest = module.build_baseline_manifest()
    receipt = _load_json(RECEIPT_PATH)
    output = _load_json(OUTPUT_MANIFEST_PATH)

    assert manifest["competing_subsystem_created"] is False
    assert receipt["sealed_baseline"]["competing_subsystem_created"] is False
    assert output.get("competing_subsystem_created") is False
    assert manifest["promotion_eligible"] is False
    assert receipt["promotion_eligible"] is False
    assert output.get("promotion_eligible") is False
    assert TEST_PATH.is_file()
    assert MANIFEST_PATH.is_file()
