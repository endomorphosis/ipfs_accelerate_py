"""Fail-closed contract tests for the PCCE v0.1 required CI workflow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ACCELERATOR_ROOT = Path(__file__).resolve().parents[2]
OUTER_ROOT = ACCELERATOR_ROOT.parents[1]
WORKFLOW = ACCELERATOR_ROOT / ".github/workflows/proof-context-v0.1.yml"
VERIFIER = ACCELERATOR_ROOT / "scripts/proof_context/verify_release_ci.py"
MANIFEST = OUTER_ROOT / "artifacts/proof_carrying_context_engine/ci/required_jobs.json"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pcce080_verify_release_ci", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier() -> ModuleType:
    return _load_verifier()


def _mutated_workflow(tmp_path: Path, old: str, new: str) -> Path:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count(old) >= 1, old
    path = tmp_path / "workflow.yml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_exact_required_workflow_and_manifest_validate(verifier: ModuleType) -> None:
    workflow = verifier.validate_workflow(WORKFLOW)
    manifest = verifier.validate_manifest(MANIFEST, workflow)

    assert workflow["required_job_count"] == 9
    assert [job["job_id"] for job in workflow["jobs"]] == list(verifier.JOB_ORDER)
    assert workflow["jobs"][-1]["needs"] == list(verifier.UPSTREAM_JOBS)
    assert workflow["error_swallow_audit"] == "passed"
    assert workflow["skip_audit"] == "passed"
    assert workflow["immutable_action_pin_audit"] == "passed"
    assert workflow["bounded_evidence_upload_audit"] == "passed"
    assert manifest["value"]["task_authority"]["live_task_cid"] == (
        "baguqeerax65k5eln5awrgysj33pt6gsatg2po2leftgrmg25m3f47ojhnnga"
    )


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        (
            "        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            "        uses: actions/checkout@v7",
            "mutable action reference",
        ),
        (
            "          persist-credentials: false",
            "          persist-credentials: true",
            "credential persistence",
        ),
        (
            "          ref: ${{ github.sha }}",
            "          ref: main",
            "mutable candidate ref",
        ),
        (
            "          set -euo pipefail",
            "          set +e",
            "error swallowing",
        ),
        (
            "            test/proof_context/security/test_adversarial_concurrency.py \\",
            "            test/proof_context/security/test_adversarial_patch_and_agent.py \\",
            "required security population removed",
        ),
        (
            "      - dependency-license",
            "      - receipt-seal",
            "aggregate dependency removed",
        ),
        (
            "            --require-qualified",
            "            --require-qualified || true",
            "qualification failure swallowed",
        ),
    ],
)
def test_workflow_mutations_fail_closed(
    verifier: ModuleType, tmp_path: Path, old: str, new: str, reason: str
) -> None:
    path = _mutated_workflow(tmp_path, old, new)
    with pytest.raises(verifier.ContractError):
        verifier.validate_workflow(path)
    assert reason


def test_no_required_job_can_skip_or_soft_fail(verifier: ModuleType) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error" not in text
    assert "fail-fast: false" not in text
    assert "|| true" not in text
    assert "set +e" not in text
    assert "--deselect" not in text
    assert "--continue-on-collection-errors" not in text
    assert text.count("if: ${{ always() }}") == 10
    assert text.count(verifier.UPLOAD_ACTION) == 9
    assert text.count("if-no-files-found: error") == 9


def test_external_authority_is_explicitly_unavailable() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    qualification = manifest["qualification"]
    authority = manifest["external_authority"]

    assert qualification == {
        "decision": "NO-GO",
        "external_ci_authority_available": False,
        "local_contract_verified": True,
        "release_qualified": False,
        "waivers": [],
    }
    assert authority["status"] == "unavailable-not-observed"
    assert authority["workflow_run_id"] is None
    assert authority["workflow_run_url"] is None
    assert authority["current_head_sha"] is None
    assert authority["branch_ruleset_id"] is None
    assert authority["branch_ruleset_url"] is None
    assert authority["job_logs"] == []
    assert authority["dependency_license_findings"] is None
    assert all(
        job["external_result"]
        == {
            "check_run_id": None,
            "conclusion": None,
            "log_cid": None,
            "log_sha256": None,
            "status": "unavailable-not-run",
            "url": None,
        }
        for job in manifest["required_jobs"]
    )


def test_local_contract_success_does_not_upgrade_qualification(
    verifier: ModuleType,
) -> None:
    assert verifier.main(["--workflow", str(WORKFLOW), "--manifest", str(MANIFEST)]) == 0
    assert (
        verifier.main(
            [
                "--workflow",
                str(WORKFLOW),
                "--manifest",
                str(MANIFEST),
                "--require-qualified",
            ]
        )
        == verifier.EXIT_NOT_QUALIFIED
    )


def test_input_preflight_requires_exact_immutable_identities(verifier: ModuleType) -> None:
    assert verifier.main(["--check-inputs"]) == verifier.EXIT_NOT_QUALIFIED
    assert (
        verifier.main(
            [
                "--check-inputs",
                "--evidence-run-id",
                "123456",
                "--outer-commit",
                "43457c396be7a9116152e4414dadc4625eff2c2e",
            ]
        )
        == 0
    )
    assert (
        verifier.main(
            [
                "--check-inputs",
                "--evidence-run-id",
                "latest",
                "--outer-commit",
                "main",
            ]
        )
        == verifier.EXIT_NOT_QUALIFIED
    )


def test_job_evidence_is_bounded_deterministic_and_nonqualifying(
    verifier: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "nested" / "security.json"
    args = [
        "--write-job-evidence",
        str(path),
        "--job-id",
        "security",
        "--phase",
        "completed",
        "--head-sha",
        "0" * 40,
    ]
    assert verifier.main(args) == 0
    first = path.read_bytes()
    assert verifier.main(args) == 0
    assert path.read_bytes() == first
    value = json.loads(first)
    assert value["local_command_sequence_completed"] is True
    assert value["release_qualification_claimed"] is False
    assert value["check_name"] == "PCCE / security"


def test_aggregate_requires_all_eight_successes(verifier: ModuleType) -> None:
    success = {job_id: {"result": "success", "outputs": {}} for job_id in verifier.UPSTREAM_JOBS}
    assert verifier.main(["--aggregate-needs", json.dumps(success)]) == 0

    failed = json.loads(json.dumps(success))
    failed["security"]["result"] = "failure"
    assert verifier.main(["--aggregate-needs", json.dumps(failed)]) == verifier.EXIT_NOT_QUALIFIED


def test_manifest_is_canonical_and_binds_exact_local_files(verifier: ModuleType) -> None:
    raw = MANIFEST.read_bytes()
    value = json.loads(raw)
    assert raw == verifier._canonical_json_bytes(value)
    assert value["file_identities"]["workflow"] == verifier._descriptor(
        "external/ipfs_accelerate/.github/workflows/proof-context-v0.1.yml",
        WORKFLOW.read_bytes(),
    )
    assert value["file_identities"]["verifier"] == verifier._descriptor(
        "external/ipfs_accelerate/scripts/proof_context/verify_release_ci.py",
        VERIFIER.read_bytes(),
    )
    assert value["file_identities"]["contract_test"] == verifier._descriptor(
        "external/ipfs_accelerate/test/proof_context/test_release_ci_contract.py",
        Path(__file__).read_bytes(),
    )
