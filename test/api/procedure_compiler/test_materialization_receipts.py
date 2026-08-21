from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/materialize_agent_supervisor_procedure_compiler_program.py"


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pcpc_materializer_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qualification(module: ModuleType) -> dict[str, object]:
    commands = []
    for argv in module.QUALIFICATION_COMMANDS:
        commands.append(
            {
                "argv": list(argv),
                "returncode": 0,
                "elapsed_ms": 1,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "stdout_sha256": "0" * 64,
                "stderr_sha256": "0" * 64,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
    payload: dict[str, object] = {
        "schema": ("ipfs_accelerate_py/agent-supervisor/procedure-compiler-p0-qualification@2"),
        "program": module.PROGRAM,
        "repository_commit": "commit-current",
        "repository_tree": "tree-current",
        "branch": module.BRANCH,
        "commands": commands,
        "p0_tasks": list(module.P0_TASKS),
        "test_evidence_class": "current_tree_hermetic",
        "simulated": False,
    }
    payload["qualification_cid"] = module.content_identity(payload)
    return payload


def test_qualification_recomputes_identity_and_rejects_receipt_shaped_edit() -> None:
    module = _load_materializer()
    receipt = _qualification(module)
    assert module._stored_qualification_receipt_is_intact(
        receipt, head="commit-current", tree="tree-current"
    )

    receipt["test_evidence_class"] = "current_tree_hermetic_but_forged"
    assert not module._stored_qualification_receipt_is_intact(
        receipt, head="commit-current", tree="tree-current"
    )


def test_materialization_receipt_identity_must_be_recomputed() -> None:
    module = _load_materializer()
    receipt = {"program": module.PROGRAM, "ready_task_ids": ["PCPC-009"]}
    receipt["receipt_cid"] = module.content_identity(receipt)
    assert module._has_valid_embedded_identity(receipt, identity_field="receipt_cid")

    receipt["ready_task_ids"] = []
    assert not module._has_valid_embedded_identity(receipt, identity_field="receipt_cid")


def test_existing_materialization_cannot_bypass_fresh_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()

    def reject_stale_or_fabricated_evidence() -> dict[str, object]:
        raise module.MaterializationError("fresh qualification required")

    monkeypatch.setattr(module, "_qualify_exact_tree", reject_stale_or_fabricated_evidence)
    with pytest.raises(module.MaterializationError, match="fresh qualification required"):
        module.verify_existing()


def test_existing_materialization_reopens_with_recomputed_current_plan_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    database_path = tmp_path / "control.duckdb"
    database_path.touch()
    expected_plan_cid = "baguqeera" + "a" * 48
    observed: dict[str, object] = {}

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_qualify_exact_tree",
        lambda: {
            "qualification_cid": "baguqeera" + "b" * 48,
            "repository_commit": "commit-current",
            "repository_tree": "tree-current",
            "simulated": False,
        },
    )
    monkeypatch.setattr(
        module,
        "_git",
        lambda *args: (
            "commit-current" if args[-1] == "HEAD" else "tree-current"
        ),
    )
    monkeypatch.setattr(
        module,
        "_read_json",
        lambda path: (
            {
                "database_program": {"store_id": "control.duckdb"},
                "runtime_paths": {"evidence": "evidence"},
            }
            if Path(path).name
            == Path(module.CONFIG_RELATIVE).name
            else {}
        ),
    )
    monkeypatch.setattr(
        module,
        "_population",
        lambda **kwargs: ({}, expected_plan_cid),
    )

    class CapturedPlanRoot(RuntimeError):
        pass

    def capture_source(*args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        raise CapturedPlanRoot

    monkeypatch.setattr(module, "DatabaseTaskSource", capture_source)

    with pytest.raises(CapturedPlanRoot):
        module.verify_existing()
    assert observed["repository_tree_id"] == "tree-current"
    assert observed["plan_root_cid"] == expected_plan_cid


def _command(argv: list[str], *, returncode: int, output: str) -> dict[str, object]:
    return {
        "argv": argv,
        "returncode": returncode,
        "elapsed_ms": 1,
        "stdout_bytes": len(output.encode("utf-8")),
        "stderr_bytes": 0,
        "stdout_sha256": "1" * 64,
        "stderr_sha256": "0" * 64,
        "stdout_tail": output,
        "stderr_tail": "",
    }


def _producer_fixture(module: ModuleType) -> tuple[dict[str, object], dict[str, object]]:
    failing = {
        "reason_code": "typed_known_failure",
        "required_output_fragments": ["test_known_failure", "KeyError: 'known'"],
        "signature": "known deterministic failure",
    }
    baseline: dict[str, object] = {
        "sibling_release_bindings": [],
        "test_producers": [
            {
                "producer_id": "TP-PASS",
                "command": ["python", "-m", "pytest", "-q", "pass.py"],
                "expected": {
                    "collected": 2,
                    "passed": 2,
                    "failed": 0,
                    "errors": 0,
                    "returncode": 0,
                },
                "source_bindings": [
                    {
                        "path": "pass.py",
                        "blob_id": "1" * 40,
                        "current_blob_id": "3" * 40,
                    }
                ],
            },
            {
                "producer_id": "TP-FAIL",
                "command": ["python", "-m", "pytest", "-q", "fail.py"],
                "expected": {
                    "collected": 2,
                    "passed": 1,
                    "failed": 1,
                    "errors": 0,
                    "returncode": 1,
                },
                "expected_failure": failing,
                "source_bindings": [{"path": "fail.py", "blob_id": "2" * 40}],
            },
        ],
    }
    inventory: dict[str, object] = {
        "dispositions": [
            {
                "authority": "AvailableAuthority",
                "status": "available_with_caveats",
                "test_producer_bindings": ["TP-PASS", "TP-FAIL"],
            },
            {
                "authority": "MissingAuthority",
                "status": "missing",
                "test_producer_bindings": [],
            },
        ]
    }
    return baseline, inventory


def _gitlink_fixture(module: ModuleType) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/procedure-compiler-exact-gitlink-checkouts@1"
        ),
        "program": module.PROGRAM,
        "repository_commit": "commit-current",
        "repository_tree": "tree-current",
        "bindings": [],
        "binding_count": 0,
        "auto_updated": False,
        "simulated": False,
    }
    payload["gitlink_receipt_cid"] = module.content_identity(payload)
    return payload


def test_current_prerequisite_execution_binds_exact_counts_and_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    baseline, inventory = _producer_fixture(module)

    def execute(argv: list[str], *, timeout: int) -> tuple[dict[str, object], str]:
        assert timeout == module.PREREQUISITE_PRODUCER_TIMEOUT_SECONDS
        if argv[-1] == "pass.py":
            output, returncode = "..\n2 passed in 0.01s\n", 0
        else:
            output, returncode = (
                "FAILED fail.py::test_known_failure - KeyError: 'known'\n"
                "1 failed, 1 passed in 0.02s\n",
                1,
            )
        return _command(argv, returncode=returncode, output=output), output

    monkeypatch.setattr(module, "_captured_command_receipt", execute)
    execution = module._execute_prerequisite_test_producers(
        baseline=baseline,
        inventory=inventory,
        head="commit-current",
        tree="tree-current",
        gitlinks=_gitlink_fixture(module),
    )
    assert execution["producer_count"] == 2
    assert execution["typed_expected_failure_count"] == 1
    assert execution["authority_count"] == 2
    assert execution["all_declared_outcomes_matched"] is True
    by_producer = {item["producer_id"]: item for item in execution["producer_receipts"]}
    assert by_producer["TP-PASS"]["source_blob_ids"] == ["3" * 40]
    by_authority = {item["authority"]: item for item in execution["authority_receipts"]}
    assert by_authority["AvailableAuthority"]["producer_ids"] == ["TP-FAIL", "TP-PASS"]
    assert (
        by_authority["MissingAuthority"]["evidence_disposition"]
        == "not_applicable_missing_authority"
    )

    def read_fixture(path: Path) -> dict[str, object]:
        return baseline if path.name == "baseline.json" else inventory

    monkeypatch.setattr(module, "_read_json", read_fixture)
    assert module._stored_prerequisite_execution_is_intact(
        execution, head="commit-current", tree="tree-current"
    )

    forged = copy.deepcopy(execution)
    producer = forged["producer_receipts"][0]
    old_cid = producer.pop("producer_receipt_cid")
    producer["unknown_normative_field"] = "not admitted"
    producer["producer_receipt_cid"] = module.content_identity(producer)
    for authority in forged["authority_receipts"]:
        if old_cid in authority["producer_receipt_cids"]:
            authority["producer_receipt_cids"] = [
                producer["producer_receipt_cid"] if item == old_cid else item
                for item in authority["producer_receipt_cids"]
            ]
            authority.pop("authority_receipt_cid")
            authority["authority_receipt_cid"] = module.content_identity(authority)
    forged.pop("execution_cid")
    forged["execution_cid"] = module.content_identity(forged)
    assert not module._stored_prerequisite_execution_is_intact(
        forged, head="commit-current", tree="tree-current"
    )


def test_current_prerequisite_execution_rejects_count_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    baseline, inventory = _producer_fixture(module)

    def drifted(argv: list[str], *, timeout: int) -> tuple[dict[str, object], str]:
        del timeout
        output = "3 passed in 0.01s\n"
        return _command(argv, returncode=0, output=output), output

    monkeypatch.setattr(module, "_captured_command_receipt", drifted)
    with pytest.raises(module.MaterializationError, match="producer TP-PASS drifted"):
        module._execute_prerequisite_test_producers(
            baseline=baseline,
            inventory=inventory,
            head="commit-current",
            tree="tree-current",
            gitlinks=_gitlink_fixture(module),
        )


def test_current_prerequisite_execution_rejects_untyped_same_count_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    baseline, inventory = _producer_fixture(module)
    baseline = copy.deepcopy(baseline)
    baseline["test_producers"] = [baseline["test_producers"][1]]
    inventory = {
        "dispositions": [
            {
                "authority": "AvailableAuthority",
                "status": "available_with_caveats",
                "test_producer_bindings": ["TP-FAIL"],
            }
        ]
    }

    def wrong_failure(argv: list[str], *, timeout: int) -> tuple[dict[str, object], str]:
        del timeout
        output = "FAILED fail.py::test_other - RuntimeError: other\n1 failed, 1 passed in 0.01s\n"
        return _command(argv, returncode=1, output=output), output

    monkeypatch.setattr(module, "_captured_command_receipt", wrong_failure)
    with pytest.raises(module.MaterializationError, match="typed expected failure fragments"):
        module._execute_prerequisite_test_producers(
            baseline=baseline,
            inventory=inventory,
            head="commit-current",
            tree="tree-current",
            gitlinks=_gitlink_fixture(module),
        )


def test_exact_gitlink_checkout_is_a_read_only_qualification_precondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    commit = "a" * 40
    baseline = {
        "sibling_release_bindings": [
            {"binding_id": "gitlink:sibling", "path": "sibling", "gitlink_commit": commit}
        ]
    }
    monkeypatch.setattr(module, "_object_id", lambda revision, path: commit)
    monkeypatch.setattr(
        module,
        "_git",
        lambda *args: f"-{commit} sibling" if args[:2] == ("submodule", "status") else "",
    )
    with pytest.raises(module.MaterializationError, match="exact sibling checkout required"):
        module._verify_exact_gitlink_checkouts(baseline, head="commit-current", tree="tree-current")
