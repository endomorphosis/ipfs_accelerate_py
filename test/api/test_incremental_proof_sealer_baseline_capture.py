from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_incremental_proof_sealer_baselines.py"
SPEC = importlib.util.spec_from_file_location("ips_baseline_capture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)


EXPECTED_IDS = {
    "accelerate": [
        "accelerate-proof-focused-core-15",
        "accelerate-proof-focused-wide-36",
        "accelerate-proof-reuse-migration",
        "accelerate-proof-reuse-cross-repo",
    ],
    "datasets": [
        "datasets-zkp-focused-current",
        "datasets-zkp-unit-wide-current",
        "datasets-proof-cache-adapters",
        "datasets-zkp-broad-safe-current",
    ],
    "kit": [
        "kit-proof-certificate",
        "kit-reuse-capabilities",
        "kit-profile-d",
        "kit-coordination",
        "kit-modern-wal",
        "kit-proof-reuse-bootstrap",
        "kit-agent-receipts",
        "kit-iroh-release",
        "kit-release-receipt",
    ],
}


TRANSCRIPT = b"""============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-8.4.1, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 6 items / 1 deselected / 5 selected

tests/test_proof.py::test_pass PASSED                                    [ 20%]
tests/test_proof.py::test_fail FAILED                                    [ 40%]
tests/test_proof.py::test_skip SKIPPED (optional key unavailable)        [ 60%]
tests/test_proof.py::test_xfail XFAIL (known limitation)                 [ 80%]
tests/test_proof.py::test_xpass XPASS                                    [100%]

=================================== FAILURES ===================================
tests/test_proof.py:4: AssertionError: no
=========================== short test summary info ============================
FAILED tests/test_proof.py::test_fail - AssertionError: no
SKIPPED [1] tests/test_proof.py:5: optional key unavailable
XFAIL tests/test_proof.py::test_xfail - known limitation
XPASS tests/test_proof.py::test_xpass
===== 1 failed, 1 passed, 1 skipped, 1 deselected, 1 xfailed, 1 xpassed in 0.12s =====
"""


def test_suite_registry_is_fixed_bounded_and_contains_audited_slices() -> None:
    assert {
        name: [suite.id for suite in suites]
        for name, suites in capture.SUITES_BY_REPOSITORY.items()
    } == EXPECTED_IDS
    assert len(capture.SUITES) == 17
    assert capture.ENVIRONMENT_POLICY_ID == (
        "incremental-proof-sealer-controlled-offline-pytest@3"
    )
    assert capture.GIT_ENVIRONMENT_POLICY_ID == (
        "incremental-proof-sealer-fixed-git-environment@2"
    )
    assert capture.SCHEMA_VERSION == "incremental-proof-sealer-baseline-receipt@4"
    for suite in capture.SUITES:
        assert suite.timeout_seconds in range(1, 601)
        assert suite.argv_template[:3] == ("{python}", "-m", "pytest")
        assert "--basetemp={basetemp}" in suite.argv_template
        assert "cache_dir={cache_dir}" in suite.argv_template
        assert "no:cacheprovider" not in suite.argv_template
        assert not any(
            "http://" in token or "https://" in token for token in suite.argv_template
        )
        assert not any(
            token in {"pip", "install", "cargo", "setup"}
            for token in suite.argv_template
        )
        assert "historical" in suite.observation_note
        assert "not reconstruct" in suite.observation_note
    assert capture.SUITES_BY_ID["kit-proof-reuse-bootstrap"].timeout_seconds == 300
    assert len(capture.ACCELERATE_FOCUSED_PATHS[:15]) == 15
    assert capture.ACCELERATE_FOCUSED_PATHS[14].endswith(
        "test_agent_supervisor_code_proof_attestation_policy.py"
    )


def test_suite_registry_exactly_pins_safe_current_slices_and_exclusions() -> None:
    core = capture.SUITES_BY_ID["accelerate-proof-focused-core-15"]
    wide = capture.SUITES_BY_ID["accelerate-proof-focused-wide-36"]
    assert core.test_args[:15] == capture.ACCELERATE_FOCUSED_PATHS[:15]
    assert wide.test_args[:36] == capture.ACCELERATE_FOCUSED_PATHS
    assert core.test_args[-2:] == wide.test_args[-2:]
    assert capture.SUITES_BY_ID["accelerate-proof-reuse-migration"].test_args == (
        "test/api/test_proof_reuse_v4_publication_integration.py",
        "test/api/test_proof_reuse_runtime_activation_e2e.py",
        "test/api/test_proof_reuse_runtime_composition.py",
        "test/api/test_pytest_proof_reuse_item_identity.py",
        "test/api/test_pytest_proof_reuse_lookup.py",
        "test/api/test_pytest_proof_reuse_plugin.py",
        "test/api/test_pytest_proof_reuse_receipt.py",
        "test/api/test_pytest_proof_reuse_xdist.py",
    )
    focused = capture.SUITES_BY_ID["datasets-zkp-focused-current"]
    assert focused.test_args == (
        "tests/unit/logic/zkp",
        "tests/integration/test_provekit_zkp.py",
        "tests/integration/test_groth16_local_evm_verification.py",
        "tests/integration/logic/test_proof_receipt_attestation.py",
        "-k",
        "not test_nargo_check_when_toolchain_available",
    )
    assert capture.SUITES_BY_ID["datasets-zkp-unit-wide-current"].test_args == (
        "tests/unit_tests/logic/zkp",
        "-k",
        "not test_import_backends_quiet and not test_py_ecc_not_imported_on_backends_import",
    )
    broad = capture.SUITES_BY_ID["datasets-zkp-broad-safe-current"]
    assert "tests/mcp/unit/test_mcplusplus_spec_session50.py" in broad.test_args
    assert "tests/mcp/integration/test_profile_f_ceremony_p2p.py" in broad.test_args
    assert "tests/mcp/integration/test_profile_d_policy_p2p.py" in broad.test_args
    assert (
        "tests/contract/processors/wallets/test_worldcoin_differential.py"
        in broad.test_args
    )
    assert "tests/integration/test_pdf_form_agent.py" in broad.test_args
    assert broad.test_args[-2:] == (
        "-k",
        "not test_profile_e_node_starts_with_the_installed_multiformats_runtime and not test_nargo_check_when_toolchain_available and not test_import_backends_quiet and not test_py_ecc_not_imported_on_backends_import",
    )
    assert capture.UNMATERIALIZED_OUTER_GITLINKS == {
        "docs/fastmcp": "1d932cc778a24cc0bf46fc4baad8306d4fed9c4b",
        "docs/mcp-python-sdk": "0da9a074d09267a927d72faa58c26d828f0f8edb",
        "ipfs_accelerate_py/mcplusplus": "15c1816d6c63a2b11edd505704f6a04a9abc6167",
        "ipfs_model_manager_py": "f6151d2113f42e75ea7d83a1b2362fc97e55e44d",
        "ipfs_transformers_py": "b397988ed9e3e656475c1cf4417b84efdb95daf3",
        "test/doc-builder": "6108e850ae1cf2f71bb0815a600bcd50c39abfa7",
        "test/huggingface_doc_builder": "6108e850ae1cf2f71bb0815a600bcd50c39abfa7",
        "test/huggingface_transformers": "44752c8dd99f3fb0da23006dc4fde4a07d9c417f",
    }


def test_parse_pytest_log_independently_counts_and_names_nonpasses() -> None:
    parsed = capture.parse_pytest_log(TRANSCRIPT)
    assert parsed["collected_count"] == 6
    assert parsed["collection_complete"] is True
    assert parsed["outcome_counts"] == {
        "passed": 1,
        "failed": 1,
        "errors": 0,
        "skipped": 1,
        "deselected": 1,
        "xfailed": 1,
        "xpassed": 1,
        "selected": 5,
    }
    assert [(item["status"], item["node_id"]) for item in parsed["non_pass_nodes"]] == [
        ("failed", "tests/test_proof.py::test_fail"),
        ("skipped", "tests/test_proof.py::test_skip"),
        ("xfailed", "tests/test_proof.py::test_xfail"),
        ("xpassed", "tests/test_proof.py::test_xpass"),
    ]
    assert parsed["summary_line"].startswith("=====")


def test_collection_error_is_observed_but_not_called_complete() -> None:
    raw = b"""============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-8.4.1, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 0 items / 1 error
ERROR collecting tests/test_receipt.py
=============================== 1 error in 0.10s ===============================
"""
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collected_count"] == 0
    assert parsed["collection_complete"] is False
    assert parsed["outcome_counts"]["errors"] == 1
    assert parsed["outcome_counts"]["selected"] == 1
    assert parsed["non_pass_nodes"] == [
        {
            "status": "error",
            "node_id": "collecting tests/test_receipt.py",
            "detail": "ERROR collecting tests/test_receipt.py",
        }
    ]


def test_decorated_collection_error_and_short_summary_are_one_exact_nonpass() -> None:
    raw = b"""============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-9.1.1, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 0 items / 1 error
________________ ERROR collecting tests/test_receipt.py _________________
E   ImportError: unavailable collection dependency
=========================== short test summary info ============================
ERROR tests/test_receipt.py
=============================== 1 error in 0.10s ===============================
"""
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collection_complete"] is False
    assert parsed["non_pass_nodes"] == [
        {
            "status": "error",
            "node_id": "collecting tests/test_receipt.py",
            "detail": (
                "________________ ERROR collecting tests/test_receipt.py "
                "_________________"
            ),
        }
    ]


def test_collection_error_count_and_summary_only_are_canonicalized() -> None:
    raw = b"""============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-9.1.1, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 0 items / 1 error
=========================== short test summary info ============================
ERROR tests/test_receipt.py
=============================== 1 error in 0.10s ===============================
"""
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collection_complete"] is False
    assert parsed["non_pass_nodes"] == [
        {
            "status": "error",
            "node_id": "collecting tests/test_receipt.py",
            "detail": "ERROR tests/test_receipt.py",
        }
    ]


def test_deselection_only_collection_remains_complete() -> None:
    raw = b"""============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-9.1.1, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 10 items / 2 deselected / 8 selected
tests/test_receipt.py::test_0 PASSED
tests/test_receipt.py::test_1 PASSED
tests/test_receipt.py::test_2 PASSED
tests/test_receipt.py::test_3 PASSED
tests/test_receipt.py::test_4 PASSED
tests/test_receipt.py::test_5 PASSED
tests/test_receipt.py::test_6 PASSED
tests/test_receipt.py::test_7 PASSED
======================== 8 passed, 2 deselected in 0.10s ========================
"""
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collected_count"] == 10
    assert parsed["collection_complete"] is True
    assert parsed["outcome_counts"]["selected"] == 8
    assert parsed["outcome_counts"]["deselected"] == 2
    assert parsed["non_pass_nodes"] == []


def test_runtime_item_error_remains_a_complete_collection() -> None:
    raw = b"""============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-9.1.1, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 1 item
tests/test_receipt.py::test_runtime ERROR
=========================== short test summary info ============================
ERROR tests/test_receipt.py::test_runtime - RuntimeError: observed
=============================== 1 error in 0.10s ===============================
"""
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collection_complete"] is True
    assert parsed["collected_count"] == 1
    assert parsed["non_pass_nodes"] == [
        {
            "status": "error",
            "node_id": "tests/test_receipt.py::test_runtime",
            "detail": "tests/test_receipt.py::test_runtime ERROR",
        }
    ]


def test_collection_skip_is_retained_as_an_exact_nonpass_location() -> None:
    raw = b"""============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-8.4.1, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 0 items / 1 skipped
=========================== short test summary info ============================
SKIPPED [1] tests/test_optional.py:7: unavailable dependency
============================== 1 skipped in 0.10s ==============================
"""
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collection_complete"] is False
    assert parsed["outcome_counts"]["selected"] == 1
    assert parsed["non_pass_nodes"] == [
        {
            "status": "skipped",
            "node_id": "tests/test_optional.py:7",
            "detail": "SKIPPED [1] tests/test_optional.py:7: unavailable dependency",
        }
    ]


@pytest.mark.parametrize(
    "raw",
    [
        b"20 static checks passed\n",
        b"================ test session starts ================\nno tests ran in 0.01s\n",
        b"================ test session starts ================\ncollected 2 items\n",
    ],
)
def test_parse_rejects_static_zero_count_and_summaryless_evidence(raw: bytes) -> None:
    with pytest.raises(capture.BaselineError):
        capture.parse_pytest_log(raw)


def test_receipt_digest_is_canonical_self_digest_and_detects_tampering() -> None:
    first = {"b": [2, 1], "a": {"value": "proof"}, "receipt_digest": "ignored"}
    reordered = {"receipt_digest": "different", "a": {"value": "proof"}, "b": [2, 1]}
    assert capture.receipt_digest(first) == capture.receipt_digest(reordered)
    expected = (
        "sha256:" + hashlib.sha256(b'{"a":{"value":"proof"},"b":[2,1]}').hexdigest()
    )
    assert capture.receipt_digest(first) == expected
    reordered["a"]["value"] = "tampered"
    assert capture.receipt_digest(first) != capture.receipt_digest(reordered)


def _valid_command(tmp_path: Path) -> tuple[object, dict[str, object]]:
    suite = capture.SUITES_BY_ID["kit-proof-certificate"]
    python_info = capture._python_metadata()
    pytest_info = capture._pytest_metadata()
    transcript = TRANSCRIPT.replace(
        b"Python 3.12.0", f"Python {python_info['version'].split()[0]}".encode()
    ).replace(b"pytest-8.4.1", f"pytest-{pytest_info['version']}".encode())
    workspace_relative = (
        capture.ARTIFACT_RELATIVE_ROOT / "work" / "capture-1" / suite.id
    ).as_posix()
    workspace = tmp_path / workspace_relative
    log_relative = (
        capture.ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-capture-1.log"
    ).as_posix()
    log_path = tmp_path / log_relative
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(transcript)
    environment = {
        "policy_id": capture.ENVIRONMENT_POLICY_ID,
        "variables": capture._environment(tmp_path, workspace_relative),
    }
    argv = capture._resolved_argv(
        suite, python_info["executable"], workspace / "pytest"
    )
    parsed = capture.parse_pytest_log(transcript)
    command: dict[str, object] = {
        "id": suite.id,
        "evidence_type": "pytest_execution_observation",
        "suite_definition_digest": capture.suite_definition_digest(suite),
        "command_digest": capture._sha256(
            capture._canonical_bytes(
                {
                    "id": suite.id,
                    "argv": argv,
                    "cwd": suite.cwd,
                    "environment": environment,
                }
            )
        ),
        "argv": argv,
        "cwd": suite.cwd,
        "workspace_relative_path": workspace_relative,
        "environment": environment,
        "python": python_info,
        "pytest": pytest_info,
        "started_at": "2026-08-11T13:00:00.000000Z",
        "finished_at": "2026-08-11T13:00:01.000000Z",
        "duration_ns": 1_000_000_000,
        "timeout_seconds": suite.timeout_seconds,
        "capture_status": "completed",
        "exit_code": 1,
        **parsed,
        "parse_error": None,
        "log": {
            "relative_path": log_relative,
            "bytes": len(transcript),
            "sha256": capture._sha256(transcript),
        },
        "assurance": capture._assurance_payload(process_observed=True, aggregate=False),
    }
    return suite, command


def _collection_abort_transcript(
    shape_id: str,
    *,
    collected: int,
    errors: int,
    deselected: int,
) -> tuple[bytes, list[dict[str, str]]]:
    python_info = capture._python_metadata()
    pytest_info = capture._pytest_metadata()
    safe_shape = shape_id.replace("-", "_")
    nodes = [
        f"tests/{safe_shape}/test_collection_{index:02d}.py"
        for index in range(errors)
    ]
    headings = [
        f"________________ ERROR collecting {node} _________________"
        for node in nodes
    ]
    summary_nodes = [f"ERROR {node}" for node in nodes]
    collection_parts = [f"{errors} {'error' if errors == 1 else 'errors'}"]
    final_parts = list(collection_parts)
    if deselected:
        collection_parts.append(f"{deselected} deselected")
        final_parts.append(f"{deselected} deselected")
    raw = "\n".join(
        (
            "============================= test session starts ==============================",
            (
                f"platform linux -- Python {python_info['version'].split()[0]}, "
                f"pytest-{pytest_info['version']}, pluggy-1.6.0"
            ),
            "rootdir: /fixed",
            (
                f"collecting ... collected {collected} items / "
                + " / ".join(collection_parts)
            ),
            *headings,
            "=========================== short test summary info ============================",
            *summary_nodes,
            "================ " + ", ".join(final_parts) + " in 0.10s ================",
            "",
        )
    ).encode()
    expected = [
        {
            "status": "error",
            "node_id": f"collecting {node}",
            "detail": heading,
        }
        for node, heading in zip(nodes, headings, strict=True)
    ]
    return raw, expected


@pytest.mark.parametrize(
    ("shape_id", "collected", "errors", "deselected"),
    [
        ("datasets-zkp-focused-current", 34, 6, 1),
        ("datasets-zkp-unit-wide-current", 93, 40, 0),
        ("datasets-proof-cache-adapters", 130, 3, 0),
        ("datasets-zkp-broad-safe-current", 377, 52, 1),
        ("kit-agent-receipts", 0, 1, 0),
    ],
)
def test_real_collection_abort_shapes_self_validate_as_incomplete_observations(
    tmp_path: Path,
    shape_id: str,
    collected: int,
    errors: int,
    deselected: int,
) -> None:
    suite, command = _valid_command(tmp_path)
    raw, expected_nonpasses = _collection_abort_transcript(
        shape_id,
        collected=collected,
        errors=errors,
        deselected=deselected,
    )
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collected_count"] == collected
    assert parsed["collection_complete"] is False
    assert parsed["outcome_counts"]["errors"] == errors
    assert parsed["outcome_counts"]["deselected"] == deselected
    assert parsed["non_pass_nodes"] == expected_nonpasses

    log_path = tmp_path / command["log"]["relative_path"]
    log_path.write_bytes(raw)
    command.update(parsed)
    command["exit_code"] = 2
    command["log"] = {
        "relative_path": command["log"]["relative_path"],
        "bytes": len(raw),
        "sha256": capture._sha256(raw),
    }
    capture._validate_command(
        tmp_path,
        suite,
        command,
        capture._python_metadata(),
        capture._pytest_metadata(),
    )


def test_command_validation_rehashes_and_reparses_retained_log(tmp_path: Path) -> None:
    suite, command = _valid_command(tmp_path)
    capture._validate_command(
        tmp_path,
        suite,
        command,
        capture._python_metadata(),
        capture._pytest_metadata(),
    )


def test_command_validation_accepts_honest_collection_level_skip(
    tmp_path: Path,
) -> None:
    suite, command = _valid_command(tmp_path)
    python_info = capture._python_metadata()
    pytest_info = capture._pytest_metadata()
    raw = f"""============================= test session starts ==============================
platform linux -- Python {python_info["version"].split()[0]}, pytest-{pytest_info["version"]}, pluggy-1.6.0
rootdir: /fixed
collecting ... collected 0 items / 1 skipped
=========================== short test summary info ============================
SKIPPED [1] tests/test_optional.py:7: unavailable dependency
============================== 1 skipped in 0.10s ==============================
""".encode()
    log_path = tmp_path / command["log"]["relative_path"]
    log_path.write_bytes(raw)
    parsed = capture.parse_pytest_log(raw)
    assert parsed["collection_complete"] is False
    for key in (
        "outcome_counts",
        "collected_count",
        "collection_complete",
        "non_pass_nodes",
        "summary_line",
    ):
        command[key] = parsed[key]
    command["exit_code"] = 0
    command["log"] = {
        "relative_path": command["log"]["relative_path"],
        "bytes": len(raw),
        "sha256": capture._sha256(raw),
    }
    capture._validate_command(tmp_path, suite, command, python_info, pytest_info)


@pytest.mark.parametrize("mutation", ["argv", "counts", "assurance", "log"])
def test_command_validation_rejects_tampered_or_overstated_evidence(
    tmp_path: Path, mutation: str
) -> None:
    suite, command = _valid_command(tmp_path)
    if mutation == "argv":
        command["argv"] = [*command["argv"], "--collect-only"]
    elif mutation == "counts":
        command["outcome_counts"] = copy.deepcopy(command["outcome_counts"])
        command["outcome_counts"]["passed"] = 999
    elif mutation == "assurance":
        command["assurance"] = copy.deepcopy(command["assurance"])
        command["assurance"]["test_execution_cryptographically_proven"] = True
    else:
        log_path = tmp_path / command["log"]["relative_path"]
        log_path.write_bytes(TRANSCRIPT + b"tampered\n")
    with pytest.raises(capture.BaselineError):
        capture._validate_command(
            tmp_path,
            suite,
            command,
            capture._python_metadata(),
            capture._pytest_metadata(),
        )


def test_cli_has_no_arbitrary_command_or_output_path_inputs() -> None:
    parser = capture.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["capture", "--command", "pytest evil.py"])
    with pytest.raises(SystemExit):
        parser.parse_args(["validate-only", "--path", "/tmp/forged.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(["capture", "--repository", "kit"])
    assert parser.parse_args(["capture", "--repository", "all"]).repository == "all"
    assert (
        parser.parse_args(["validate-only", "--repository", "kit"]).repository
        == "kit"
    )
    assert parser.parse_args(["render-pins"]).action == "render-pins"


def test_suite_definition_digest_has_reviewed_fixed_preimage() -> None:
    suite = capture.SUITES_BY_ID["kit-iroh-release"]
    payload = capture.suite_definition_payload(suite)
    assert set(payload) == {
        "id",
        "repository",
        "cwd",
        "argv_template",
        "environment_policy_id",
        "timeout_seconds",
        "observation_note",
    }
    assert (
        capture.suite_definition_digest(suite)
        == "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    changed = capture.dataclasses.replace(suite, observation_note="different claim")
    assert capture.suite_definition_digest(changed) != capture.suite_definition_digest(
        suite
    )


def test_reviewed_registry_helper_is_pure_ordered_and_claim_bound() -> None:
    payload = capture.reviewed_suite_registry()
    assert payload["schema_version"] == capture.SUITE_REGISTRY_SCHEMA_VERSION
    assert payload["environment_policy_id"] == capture.ENVIRONMENT_POLICY_ID
    assert {
        repository: [suite["id"] for suite in suites]
        for repository, suites in payload["repositories"].items()
    } == EXPECTED_IDS
    assert all(
        "not reconstruct" in suite["observation_note"]
        for suites in payload["repositories"].values()
        for suite in suites
    )


def test_protected_registry_is_canonical_and_drift_rejected(tmp_path: Path) -> None:
    registry = capture.reviewed_suite_registry()
    path = tmp_path / capture.SUITE_REGISTRY_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(capture._canonical_bytes(registry) + b"\n")
    assert capture._validate_protected_suite_registry(tmp_path) == registry
    registry["repositories"]["kit"][0]["timeout_seconds"] += 1
    path.write_bytes(capture._canonical_bytes(registry) + b"\n")
    with pytest.raises(capture.BaselineError, match="compiled projection"):
        capture._validate_protected_suite_registry(tmp_path)


def test_environment_is_fixed_controlled_offline_and_auto_install_disabled(
    tmp_path: Path,
) -> None:
    relative = (
        capture.ARTIFACT_RELATIVE_ROOT / "work" / "capture" / "suite"
    ).as_posix()
    environment = capture._environment(tmp_path, relative)
    expected_gates = {
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
        "IPFS_DATASETS_ENABLE_GROTH16": "0",
        "IPFS_DATASETS_RUN_GROTH16_EVM": "0",
        "IPFS_DATASETS_RUN_PROVEKIT_TESTS": "0",
        "IPFS_DATASETS_PY_AUTO_GROTH16_BUILD": "0",
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
        "IPFS_ACCEL_AUTO_INSTALL": "0",
    }
    assert {key: environment[key] for key in expected_gates} == expected_gates
    assert environment["PATH"] == capture.FIXED_EXECUTABLE_PATH
    assert environment["IPFS_PATH"] == str(tmp_path / relative / "ipfs-repo")
    assert environment["PYTHONPYCACHEPREFIX"] == str(tmp_path / relative / "pycache")
    assert environment["HYPOTHESIS_STORAGE_DIRECTORY"] == str(
        tmp_path / relative / "hypothesis"
    )
    assert environment["PYTEST_ADDOPTS"] == (
        f"--benchmark-storage=file://{tmp_path / relative / 'pytest-benchmark'}"
    )
    assert environment["PIP_NO_INDEX"] == "1"
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in environment


def test_readonly_git_commands_ignore_inherited_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "baseline@example.invalid")
    _git(tmp_path, "config", "user.name", "Baseline Test")
    (tmp_path / "source.py").write_text("bound\n", encoding="utf-8")
    _git(tmp_path, "add", "source.py")
    _git(tmp_path, "commit", "--quiet", "-m", "bound")
    expected = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker-git-dir"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "attacker-objects"))
    observed_environments: list[dict[str, str]] = []
    real_popen = capture.subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> object:
        observed_environments.append(dict(kwargs["env"]))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(capture.subprocess, "Popen", recording_popen)
    assert capture._git_text(tmp_path, "rev-parse", "HEAD") == expected
    assert observed_environments[-1]["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert observed_environments[-1]["GIT_OPTIONAL_LOCKS"] == "0"
    assert "GIT_DIR" not in observed_environments[-1]
    assert "GIT_INDEX_FILE" not in observed_environments[-1]
    assert "GIT_OBJECT_DIRECTORY" not in observed_environments[-1]

    capture._run_local_git_materialization(
        [sys.executable, "-c", "pass"], tmp_path
    )
    assert observed_environments[-1]["GIT_NO_REPLACE_OBJECTS"] == "1"


@pytest.mark.parametrize(
    "relative",
    [
        "/tmp/receipt.json",
        "../receipt.json",
        "artifacts/../receipt.json",
        r"artifacts\\receipt.json",
        "./artifacts/receipt.json",
    ],
)
def test_artifact_paths_reject_absolute_traversal_and_noncanonical_forms(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(capture.BaselineError):
        capture._artifact_path(
            tmp_path, relative, label="test artifact", allow_missing=True
        )


def test_artifact_paths_reject_every_symlink_component(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)
    relative = (capture.ARTIFACT_RELATIVE_ROOT / "logs" / "receipt.log").as_posix()
    with pytest.raises(capture.BaselineError, match="symlink"):
        capture._artifact_path(
            tmp_path, relative, label="test artifact", allow_missing=True
        )


def test_atomic_write_rejects_symlink_destination(tmp_path: Path) -> None:
    logs = capture._ensure_artifact_directory(
        tmp_path, (capture.ARTIFACT_RELATIVE_ROOT / "logs").as_posix()
    )
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"outside")
    (logs / "receipt.log").symlink_to(outside)
    with pytest.raises(capture.BaselineError, match="symlink"):
        capture._atomic_write(
            tmp_path,
            (capture.ARTIFACT_RELATIVE_ROOT / "logs" / "receipt.log").as_posix(),
            b"replacement",
        )
    assert outside.read_bytes() == b"outside"


def test_dirfd_io_rejects_intermediate_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_relative = (capture.ARTIFACT_RELATIVE_ROOT / "logs").as_posix()
    logs = capture._ensure_artifact_directory(tmp_path, logs_relative)
    relative = (capture.ARTIFACT_RELATIVE_ROOT / "logs" / "receipt.log").as_posix()
    held_logs = logs.with_name("logs-held")
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open_relative = capture._open_relative_directory
    swapped = False

    def swap_after_parent_open(
        root_descriptor: int,
        parts: object,
        *,
        create: bool,
        label: str,
    ) -> int:
        nonlocal swapped
        descriptor = real_open_relative(
            root_descriptor, parts, create=create, label=label
        )
        if label == "atomic output parent" and not swapped:
            swapped = True
            logs.rename(held_logs)
            logs.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(capture, "_open_relative_directory", swap_after_parent_open)
    with pytest.raises(capture.AtomicWriteError) as raised:
        capture._atomic_write(tmp_path, relative, b"held-parent-only")
    assert raised.value.replaced is True
    assert not (outside / "receipt.log").exists()
    assert (held_logs / "receipt.log").read_bytes() == b"held-parent-only"


def test_dirfd_reads_reject_intermediate_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_relative = (capture.ARTIFACT_RELATIVE_ROOT / "logs").as_posix()
    logs = capture._ensure_artifact_directory(tmp_path, logs_relative)
    (logs / "receipt.log").write_bytes(b"trusted")
    held_logs = logs.with_name("logs-held")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "receipt.log").write_bytes(b"attacker")
    real_open_relative = capture._open_relative_directory
    swapped = False

    def swap_after_parent_open(
        root_descriptor: int,
        parts: object,
        *,
        create: bool,
        label: str,
    ) -> int:
        nonlocal swapped
        descriptor = real_open_relative(
            root_descriptor, parts, create=create, label=label
        )
        if label == "artifact file parent" and not swapped:
            swapped = True
            logs.rename(held_logs)
            logs.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(capture, "_open_relative_directory", swap_after_parent_open)
    relative = (capture.ARTIFACT_RELATIVE_ROOT / "logs" / "receipt.log").as_posix()
    with pytest.raises(capture.BaselineError, match="parent"):
        capture._safe_artifact_file(tmp_path, relative, maximum=1024)
    assert (outside / "receipt.log").read_bytes() == b"attacker"


def test_dirfd_fixed_file_read_rejects_intermediate_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "reviewed.json").write_bytes(b"trusted")
    held_config = tmp_path / "config-held"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "reviewed.json").write_bytes(b"attacker")
    real_open_relative = capture._open_relative_directory
    swapped = False

    def swap_after_parent_open(
        root_descriptor: int,
        parts: object,
        *,
        create: bool,
        label: str,
    ) -> int:
        nonlocal swapped
        descriptor = real_open_relative(
            root_descriptor, parts, create=create, label=label
        )
        if label == "fixed repository file parent" and not swapped:
            swapped = True
            config.rename(held_config)
            config.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(capture, "_open_relative_directory", swap_after_parent_open)
    with pytest.raises(capture.BaselineError, match="parent"):
        capture._safe_fixed_repository_file(
            tmp_path, "config/reviewed.json", maximum=1024
        )
    assert (outside / "reviewed.json").read_bytes() == b"attacker"


def test_dirfd_directory_creation_rejects_intermediate_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = (capture.ARTIFACT_RELATIVE_ROOT / "work" / "capture").as_posix()
    target = tmp_path / relative
    held_target = target.with_name("capture-held")
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open_relative = capture._open_relative_directory
    swapped = False

    def swap_after_creation(
        root_descriptor: int,
        parts: object,
        *,
        create: bool,
        label: str,
    ) -> int:
        nonlocal swapped
        descriptor = real_open_relative(
            root_descriptor, parts, create=create, label=label
        )
        if label == "artifact directory" and not swapped:
            swapped = True
            target.rename(held_target)
            target.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(capture, "_open_relative_directory", swap_after_creation)
    with pytest.raises(capture.BaselineError, match="artifact directory"):
        capture._ensure_artifact_directory(tmp_path, relative)
    assert list(outside.iterdir()) == []


def test_bounded_capture_terminates_process_tree_on_output_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capture, "MAX_LOG_BYTES", 1024)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('x'*8192); sys.stdout.flush(); time.sleep(30)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    raw, status, exit_code = capture._communicate_bounded(process, 5)
    assert status == "output_limit_exceeded"
    assert exit_code == 125
    assert len(raw) <= capture.MAX_LOG_BYTES
    assert b"OUTPUT_LIMIT_EXCEEDED" in raw
    assert process.poll() is not None


def test_bounded_capture_times_out_when_child_closes_stdout_then_sleeps() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import os,time; os.close(1); time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    raw, status, exit_code = capture._communicate_bounded(process, 0.2)
    assert status == "timed_out"
    assert exit_code == 124
    assert b"BASELINE_CAPTURE_TIMEOUT" in raw
    assert process.poll() is not None


def test_bounded_capture_terminates_residual_child_after_parent_exits() -> None:
    program = """
import subprocess
import sys

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import os,time; os.close(1); os.close(2); time.sleep(30)",
])
print(child.pid, flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    raw, status, exit_code = capture._communicate_bounded(process, 5)
    child_pid = int(raw.splitlines()[0])
    assert status == "residual_process_terminated"
    assert exit_code == 126
    assert b"BASELINE_CAPTURE_RESIDUAL_PROCESS_TERMINATED" in raw

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{child_pid}/stat")
        if not stat_path.exists():
            break
        if stat_path.read_text(encoding="ascii").split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("residual child remained live after capture returned")


def test_bounded_capture_has_no_fast_exit_process_group_race() -> None:
    for _ in range(20):
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        raw, status, exit_code = capture._communicate_bounded(process, 2)
        assert raw == b""
        assert status == "completed"
        assert exit_code == 0


def _invoke_bounded_git_helper(kind: str, argv: list[str], cwd: Path) -> object:
    if kind == "readonly":
        return capture._run_readonly(argv, cwd=cwd)
    assert kind == "materialization"
    return capture._run_local_git_materialization(argv, cwd)


@pytest.mark.parametrize("kind", ["readonly", "materialization"])
def test_git_helpers_bound_time_and_terminate_process_tree(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "GIT_READ_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(capture, "GIT_MATERIALIZE_TIMEOUT_SECONDS", 0.2)
    started = time.monotonic()
    with pytest.raises(capture.BaselineError, match="timed_out"):
        _invoke_bounded_git_helper(
            kind,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
        )
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("kind", ["readonly", "materialization"])
def test_git_helpers_bound_combined_stdout_and_stderr(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "MAX_GIT_OUTPUT_BYTES", 1024)
    program = (
        "import sys,time; "
        "sys.stdout.write('o'*8192); sys.stdout.flush(); "
        "sys.stderr.write('e'*8192); sys.stderr.flush(); time.sleep(30)"
    )
    with pytest.raises(capture.BaselineError, match="output_limit_exceeded"):
        _invoke_bounded_git_helper(
            kind, [sys.executable, "-c", program], tmp_path
        )


@pytest.mark.parametrize("kind", ["readonly", "materialization"])
def test_git_helpers_terminate_residual_child_after_parent_exit(
    kind: str, tmp_path: Path
) -> None:
    child_pid_path = tmp_path / f"{kind}-child.pid"
    program = """
import os
import pathlib
import subprocess
import sys

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import os,time; os.close(1); os.close(2); time.sleep(30)",
])
pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")
"""
    with pytest.raises(capture.BaselineError, match="residual_process_terminated"):
        _invoke_bounded_git_helper(
            kind,
            [sys.executable, "-c", program, str(child_pid_path)],
            tmp_path,
        )
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{child_pid}/stat")
        if not stat_path.exists() or stat_path.read_text(encoding="ascii").split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("bounded Git helper left a residual child live")


@pytest.mark.parametrize("kind", ["readonly", "materialization"])
def test_git_helpers_have_no_fast_exit_process_group_race(
    kind: str, tmp_path: Path
) -> None:
    for _ in range(10):
        result = _invoke_bounded_git_helper(
            kind, [sys.executable, "-c", "pass"], tmp_path
        )
        if kind == "readonly":
            assert result.returncode == 0


@pytest.mark.parametrize(
    "secret",
    [
        b"-----BEGIN PRIVATE KEY-----",
        b"authorization: abcdefghijklmnopqrstuvwxyz1234",
        b"ghp_abcdefghijklmnopqrstuvwxyz123456",
    ],
)
def test_public_log_secret_scan_is_fail_closed(secret: bytes) -> None:
    with pytest.raises(capture.BaselineError, match="secret scan"):
        capture._assert_public_log_safe(b"prefix\n" + secret + b"\nsuffix")


def test_kit_capture_rejects_ipfs_from_fixed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "ipfs_kit_py").mkdir()
    monkeypatch.setattr(
        capture.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/ipfs"
    )
    with pytest.raises(capture.BaselineError, match="fixed PATH resolves ipfs"):
        capture._assert_no_live_ipfs(tmp_path)


def test_kit_capture_rejects_executable_repository_ipfs_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "ipfs_kit_py" / "vendor" / "bin" / "ipfs"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setattr(capture.shutil, "which", lambda *_args, **_kwargs: None)
    with pytest.raises(capture.BaselineError, match="executable IPFS candidate"):
        capture._assert_no_live_ipfs(tmp_path)


def test_exit_five_is_never_accepted_as_pytest_evidence(tmp_path: Path) -> None:
    suite, command = _valid_command(tmp_path)
    command["exit_code"] = 5
    with pytest.raises(capture.BaselineError, match="pytest exit code"):
        capture._validate_command(
            tmp_path,
            suite,
            command,
            capture._python_metadata(),
            capture._pytest_metadata(),
        )


def test_internally_inconsistent_collection_count_is_rejected(tmp_path: Path) -> None:
    suite, command = _valid_command(tmp_path)
    log_path = tmp_path / command["log"]["relative_path"]
    raw = log_path.read_bytes().replace(b"collected 6 items", b"collected 60 items")
    log_path.write_bytes(raw)
    parsed = capture.parse_pytest_log(raw)
    for key in (
        "outcome_counts",
        "collected_count",
        "collection_complete",
        "non_pass_nodes",
        "summary_line",
    ):
        command[key] = parsed[key]
    command["log"] = {
        "relative_path": command["log"]["relative_path"],
        "bytes": len(raw),
        "sha256": capture._sha256(raw),
    }
    with pytest.raises(capture.BaselineError, match="counts disagree"):
        capture._validate_command(
            tmp_path,
            suite,
            command,
            capture._python_metadata(),
            capture._pytest_metadata(),
        )


def test_safe_cleanup_unlinks_workspace_symlink_without_following_target(
    tmp_path: Path,
) -> None:
    capture_id = "20260811T120000.000000Z-1"
    workspace = capture._ensure_artifact_directory(
        tmp_path,
        (capture.ARTIFACT_RELATIVE_ROOT / "work" / capture_id).as_posix(),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    witness = outside / "must-remain.txt"
    witness.write_text("retained", encoding="utf-8")
    (workspace / "external").symlink_to(outside, target_is_directory=True)
    capture._safe_cleanup_capture_workspace(tmp_path, capture_id)
    assert not workspace.exists()
    assert witness.read_text(encoding="utf-8") == "retained"


def test_read_only_execution_hardening_contains_cache_writes_and_cleans_safely(
    tmp_path: Path,
) -> None:
    capture_id = "20260811T120000.000000Z-1"
    workspace = capture._ensure_artifact_directory(
        tmp_path,
        (capture.ARTIFACT_RELATIVE_ROOT / "work" / capture_id).as_posix(),
    )
    execution_root = workspace / "source"
    kit_root = execution_root / "ipfs_kit_py"
    git_directory = execution_root / ".git" / "objects"
    kit_root.mkdir(parents=True)
    git_directory.mkdir(parents=True)
    executable = execution_root / "tracked-tool"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    sitecustomize = kit_root / "sitecustomize.py"
    sitecustomize.write_text("MARKER = 'loaded'\n", encoding="utf-8")
    (git_directory / "tracked-object").write_bytes(b"object")
    outside = tmp_path / "outside.txt"
    outside.write_text("retained", encoding="utf-8")
    outside.chmod(0o600)
    (execution_root / "external").symlink_to(outside)

    capture._harden_execution_tree_read_only(execution_root)

    assert stat.S_IMODE(execution_root.stat().st_mode) == 0o555
    assert stat.S_IMODE(executable.stat().st_mode) == 0o555
    assert stat.S_IMODE(sitecustomize.stat().st_mode) == 0o444
    assert stat.S_IMODE(git_directory.stat().st_mode) == 0o555
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "assert __import__('sitecustomize').MARKER == 'loaded'; "
                "\ntry: Path('.benchmarks').mkdir()\nexcept OSError: pass"
            ),
        ],
        cwd=execution_root,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": capture.FIXED_EXECUTABLE_PATH,
            "PYTHONPATH": str(kit_root),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    assert not (kit_root / "__pycache__").exists()
    assert not (execution_root / ".benchmarks").exists()
    with pytest.raises(PermissionError):
        (execution_root / "unexpected-output").write_text("blocked", encoding="utf-8")

    capture._safe_cleanup_capture_workspace(tmp_path, capture_id)
    assert not workspace.exists()
    assert outside.read_text(encoding="utf-8") == "retained"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600


def test_read_only_execution_hardening_rejects_regular_hardlink_without_chmod(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path / "source"
    execution_root.mkdir()
    external = tmp_path / "external.txt"
    external_bytes = b"external bytes\n"
    external.write_bytes(external_bytes)
    external.chmod(0o640)
    os.link(external, execution_root / "tracked.txt")

    with pytest.raises(capture.BaselineError, match="regular leaf is hardlinked"):
        capture._harden_execution_tree_read_only(execution_root)

    assert external.read_bytes() == external_bytes
    assert stat.S_IMODE(external.stat().st_mode) == 0o640


def test_safe_cleanup_rejects_artifact_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_id = "20260811T120000.000000Z-1"
    workspace = capture._ensure_artifact_directory(
        tmp_path,
        (capture.ARTIFACT_RELATIVE_ROOT / "work" / capture_id).as_posix(),
    )
    (workspace / "temporary.txt").write_text("remove only held", encoding="utf-8")
    artifact_root = tmp_path / capture.ARTIFACT_RELATIVE_ROOT
    held_artifact_root = artifact_root.with_name("baseline_receipts-held")
    outside = tmp_path / "outside"
    outside_workspace = outside / "work" / capture_id
    outside_workspace.mkdir(parents=True)
    witness = outside_workspace / "must-remain.txt"
    witness.write_text("retained", encoding="utf-8")
    real_try_open = capture._try_open_relative_directory
    swapped = False

    def swap_after_work_open(
        root_descriptor: int, parts: object, *, label: str
    ) -> int | None:
        nonlocal swapped
        descriptor = real_try_open(root_descriptor, parts, label=label)
        if label == "capture cleanup parent" and descriptor is not None and not swapped:
            swapped = True
            artifact_root.rename(held_artifact_root)
            artifact_root.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(capture, "_try_open_relative_directory", swap_after_work_open)
    with pytest.raises(capture.BaselineError, match="cleanup parent"):
        capture._safe_cleanup_capture_workspace(tmp_path, capture_id)
    assert witness.read_text(encoding="utf-8") == "retained"
    assert not (held_artifact_root / "work" / capture_id).exists()


def _fake_source_snapshots() -> dict[str, dict[str, object]]:
    return {
        repository: {
            "repository": repository,
            "planning_revision": capture.PLANNING_REVISIONS[repository],
            "planning_tree": capture.PLANNING_TREES[repository],
            "tested_revision": (str(index + 1) * 40)[:40],
            "tested_tree": (str(index + 4) * 40)[:40],
            "ignored_sensitive_fingerprint": {},
            "clean": True,
            "untracked_paths": [],
        }
        for index, repository in enumerate(capture.REPOSITORY_PATHS)
    }


def _write_pristine_scheduler(root: Path) -> None:
    path = root / capture.SCHEDULER_CONFIG_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        capture._canonical_bytes(
            {
                "operator_baseline_receipts": {},
                "protected_paths": list(capture.PRE_CAPTURE_PROTECTED_PATHS),
            }
        )
        + b"\n"
    )


def test_capture_self_validation_failure_publishes_no_receipts_or_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pristine_scheduler(tmp_path)
    snapshots = _fake_source_snapshots()
    monkeypatch.setattr(capture, "_all_source_snapshots", lambda _root: snapshots)
    monkeypatch.setattr(capture, "_validate_protected_suite_registry", lambda _root: {})
    monkeypatch.setattr(capture, "_validate_source_bindings", lambda *_args: None)
    monkeypatch.setattr(
        capture, "_materialize_execution_trees", lambda *_args: tmp_path
    )
    monkeypatch.setattr(
        capture, "_harden_execution_tree_read_only", lambda *_args: None
    )
    monkeypatch.setattr(capture, "_assert_execution_trees_clean", lambda *_args: {})
    real_atomic_write = capture._atomic_write

    def invalid_command(
        repo_root: Path,
        _execution_root: Path,
        suite: object,
        capture_id: str,
        *_args: object,
    ) -> dict[str, object]:
        relative = (
            capture.ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
        ).as_posix()
        real_atomic_write(repo_root, relative, b"bounded diagnostic\n")
        return {"assurance": {"process_observed": True}}

    monkeypatch.setattr(capture, "_capture_command", invalid_command)
    published = False

    def forbidden_publish(*_args: object, **_kwargs: object) -> Path:
        nonlocal published
        published = True
        raise AssertionError("invalid candidate was published")

    monkeypatch.setattr(capture, "_create_once", forbidden_publish)
    with pytest.raises(capture.BaselineError, match="interpreter binding"):
        capture.capture_repositories(tmp_path, tuple(capture.REPOSITORY_PATHS))
    assert published is False
    receipt_directory = tmp_path / capture.ARTIFACT_RELATIVE_ROOT
    assert not list(receipt_directory.glob("*.json"))
    assert not (receipt_directory / capture.CAPTURE_LOCK_NAME).exists()
    logs = receipt_directory / "logs"
    assert not list(logs.glob("*.log"))


def test_second_receipt_failure_is_unadmitted_and_rerun_requires_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pristine_scheduler(tmp_path)
    snapshots = _fake_source_snapshots()
    monkeypatch.setattr(capture, "_all_source_snapshots", lambda _root: snapshots)
    monkeypatch.setattr(capture, "_validate_protected_suite_registry", lambda _root: {})
    monkeypatch.setattr(
        capture, "_materialize_execution_trees", lambda *_args: tmp_path
    )
    monkeypatch.setattr(
        capture, "_harden_execution_tree_read_only", lambda *_args: None
    )
    monkeypatch.setattr(capture, "_assert_execution_trees_clean", lambda *_args: {})
    real_atomic_write = capture._atomic_write
    real_create_once = capture._create_once

    def observed_command(
        repo_root: Path,
        _execution_root: Path,
        suite: object,
        capture_id: str,
        *_args: object,
    ) -> dict[str, object]:
        relative = (
            capture.ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
        ).as_posix()
        real_atomic_write(repo_root, relative, b"bounded diagnostic\n")
        return {"assurance": {"process_observed": True}}

    monkeypatch.setattr(capture, "_capture_command", observed_command)
    monkeypatch.setattr(capture, "_self_validate_payload", lambda *_args: None)

    def fail_second_receipt(repo_root: Path, relative: str, data: bytes) -> Path:
        if relative.endswith("/datasets.json"):
            raise capture.AtomicWriteError("injected second receipt failure", replaced=False)
        return real_create_once(repo_root, relative, data)

    monkeypatch.setattr(capture, "_create_once", fail_second_receipt)
    with pytest.raises(capture.AtomicWriteError, match="second receipt"):
        capture.capture_repositories(tmp_path, tuple(capture.REPOSITORY_PATHS))
    receipt_directory = tmp_path / capture.ARTIFACT_RELATIVE_ROOT
    assert (receipt_directory / "accelerate.json").is_file()
    assert not (receipt_directory / "datasets.json").exists()
    assert not (receipt_directory / "kit.json").exists()
    logs = list((receipt_directory / "logs").glob("*.log"))
    assert len(logs) == len(capture.SUITES)
    assert (receipt_directory / capture.CAPTURE_LOCK_NAME).is_file()
    with pytest.raises(capture.BaselineError, match="quarantine or repair"):
        capture.capture_repositories(tmp_path, tuple(capture.REPOSITORY_PATHS))


def test_create_once_never_replaces_a_racing_regular_file(tmp_path: Path) -> None:
    relative = (capture.ARTIFACT_RELATIVE_ROOT / "accelerate.json").as_posix()
    capture._ensure_artifact_directory(
        tmp_path, capture.ARTIFACT_RELATIVE_ROOT.as_posix()
    )
    capture._create_once(tmp_path, relative, b"first\n")
    with pytest.raises(capture.AtomicWriteError) as raised:
        capture._create_once(tmp_path, relative, b"second\n")
    assert raised.value.replaced is False
    assert (tmp_path / relative).read_bytes() == b"first\n"


def test_capture_lock_serializes_two_concurrent_contenders(tmp_path: Path) -> None:
    capture._ensure_artifact_directory(
        tmp_path, capture.ARTIFACT_RELATIVE_ROOT.as_posix()
    )
    start = threading.Barrier(2)
    attempted = threading.Barrier(2)

    def contend(index: int) -> str:
        start.wait(timeout=5)
        held: object | None = None
        try:
            held = capture._acquire_capture_lock(
                tmp_path, f"20260811T12000{index}.000000Z-{1000 + index}"
            )
            outcome = "acquired"
        except capture.BaselineError as exc:
            assert "quarantine or repair" in str(exc)
            outcome = "refused"
        attempted.wait(timeout=5)
        if held is not None:
            capture._release_capture_lock(held)
        return outcome

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(contend, (0, 1)))
    assert sorted(outcomes) == ["acquired", "refused"]
    assert not (
        tmp_path / capture.ARTIFACT_RELATIVE_ROOT / capture.CAPTURE_LOCK_NAME
    ).exists()


def test_stale_capture_lock_refuses_public_capture_until_operator_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture._ensure_artifact_directory(
        tmp_path, capture.ARTIFACT_RELATIVE_ROOT.as_posix()
    )
    stale = capture._acquire_capture_lock(
        tmp_path, "20260811T120000.000000Z-4242"
    )
    lock_path = (
        tmp_path / capture.ARTIFACT_RELATIVE_ROOT / capture.CAPTURE_LOCK_NAME
    )
    lock_raw = lock_path.read_bytes()
    lock_payload = json.loads(lock_raw)
    assert lock_raw == capture._canonical_bytes(lock_payload) + b"\n"
    assert set(lock_payload) == {
        "schema_version",
        "operation",
        "capture_id",
        "owner_pid",
        "created_at",
    }
    assert lock_payload["schema_version"] == capture.CAPTURE_LOCK_SCHEMA_VERSION
    assert len(lock_raw) <= capture.MAX_CAPTURE_LOCK_BYTES
    capture._close_capture_lock(stale)
    monkeypatch.setattr(capture, "_validate_protected_suite_registry", lambda _root: {})
    with pytest.raises(capture.BaselineError, match="quarantine or repair"):
        capture.capture_repositories(tmp_path, tuple(capture.REPOSITORY_PATHS))
    assert lock_path.is_file()


def test_cli_capture_all_is_one_source_epoch_one_shot_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pristine_scheduler(tmp_path)
    snapshots = _fake_source_snapshots()
    snapshot_calls = 0

    def one_epoch(_root: Path) -> dict[str, dict[str, object]]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return copy.deepcopy(snapshots)

    monkeypatch.setattr(capture, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(capture, "_validate_protected_suite_registry", lambda _root: {})
    monkeypatch.setattr(capture, "_all_source_snapshots", one_epoch)
    monkeypatch.setattr(capture, "_assert_no_live_ipfs", lambda _root: None)
    monkeypatch.setattr(
        capture, "_materialize_execution_trees", lambda *_args: tmp_path
    )
    monkeypatch.setattr(
        capture, "_harden_execution_tree_read_only", lambda *_args: None
    )
    monkeypatch.setattr(capture, "_assert_execution_trees_clean", lambda *_args: {})
    monkeypatch.setattr(capture, "_self_validate_payload", lambda *_args: None)

    def observed_command(
        repo_root: Path,
        _execution_root: Path,
        suite: object,
        capture_id: str,
        *_args: object,
    ) -> dict[str, object]:
        relative = (
            capture.ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
        ).as_posix()
        capture._atomic_write(repo_root, relative, b"observed\n")
        return {
            "id": suite.id,
            "assurance": capture._assurance_payload(
                process_observed=True, aggregate=False
            ),
        }

    monkeypatch.setattr(capture, "_capture_command", observed_command)
    assert capture.main(["capture", "--repository", "all"]) == 0
    assert not (
        tmp_path / capture.ARTIFACT_RELATIVE_ROOT / capture.CAPTURE_LOCK_NAME
    ).exists()
    assert snapshot_calls == 2
    source_maps = []
    for repository in capture.REPOSITORY_PATHS:
        payload = json.loads(
            (
                tmp_path / capture.ARTIFACT_RELATIVE_ROOT / f"{repository}.json"
            ).read_text(encoding="utf-8")
        )
        source_maps.append(payload["source_revisions"])
    assert source_maps[0] == source_maps[1] == source_maps[2]
    receipt_bytes = {
        repository: (
            tmp_path / capture.ARTIFACT_RELATIVE_ROOT / f"{repository}.json"
        ).read_bytes()
        for repository in capture.REPOSITORY_PATHS
    }
    assert capture.main(["capture", "--repository", "all"]) == 1
    assert receipt_bytes == {
        repository: (
            tmp_path / capture.ARTIFACT_RELATIVE_ROOT / f"{repository}.json"
        ).read_bytes()
        for repository in capture.REPOSITORY_PATHS
    }


def test_kit_live_ipfs_preflight_runs_before_any_suite_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pristine_scheduler(tmp_path)
    snapshots = _fake_source_snapshots()
    monkeypatch.setattr(capture, "_all_source_snapshots", lambda _root: snapshots)
    monkeypatch.setattr(capture, "_validate_protected_suite_registry", lambda _root: {})
    command_started = False

    def mark_command(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal command_started
        command_started = True
        return {}

    monkeypatch.setattr(capture, "_capture_command", mark_command)
    monkeypatch.setattr(
        capture,
        "_assert_no_live_ipfs",
        lambda _root: (_ for _ in ()).throw(capture.BaselineError("live ipfs")),
    )
    with pytest.raises(capture.BaselineError, match="live ipfs"):
        capture.capture_repositories(tmp_path, tuple(capture.REPOSITORY_PATHS))
    assert command_started is False


def test_gitignore_exposes_only_canonical_receipts_and_public_logs() -> None:
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    prefix = "artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts"
    for repository in ("accelerate", "datasets", "kit"):
        assert f"!/{prefix}/{repository}.json" in content
    assert f"!/{prefix}/logs/*.log" in content
    assert f"/{prefix}/work/" in content
    assert "!config/incremental_proof_sealer_baseline_suite_registry.json" in content
    required_outputs = (
        "artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json",
        "artifacts/agent_supervisor/incremental_proof_sealer/summary.json",
        "artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json",
        "artifacts/agent_supervisor/incremental_proof_sealer/release_validation.log",
    )
    for relative in required_outputs:
        assert f"!/{relative}" in content
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", relative],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 1, f"required output remains ignored: {relative}"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _initialize_repository(path: Path, filename: str) -> tuple[str, str]:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "--quiet")
    _git(path, "config", "user.email", "baseline@example.invalid")
    _git(path, "config", "user.name", "Baseline Test")
    (path / filename).write_text("captured\n", encoding="utf-8")
    _git(path, "add", filename)
    _git(path, "commit", "--quiet", "-m", "captured")
    revision = _git(path, "rev-parse", "HEAD")
    tree = _git(path, "rev-parse", "HEAD^{tree}")
    return revision, tree


@pytest.mark.parametrize("mechanism", ("replacement-ref", "legacy-grafts"))
def test_git_object_replacement_is_disabled_and_explicitly_rejected(
    tmp_path: Path, mechanism: str
) -> None:
    first_revision, first_tree = _initialize_repository(tmp_path, "source.py")
    if mechanism == "replacement-ref":
        (tmp_path / "source.py").write_text("replacement\n", encoding="utf-8")
        _git(tmp_path, "add", "source.py")
        _git(tmp_path, "commit", "--quiet", "-m", "replacement")
        replacement_revision = _git(tmp_path, "rev-parse", "HEAD")
        replacement_tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
        _git(tmp_path, "replace", first_revision, replacement_revision)
        assert _git(tmp_path, "rev-parse", f"{first_revision}^{{tree}}") == replacement_tree
        assert capture._git_text(
            tmp_path, "rev-parse", f"{first_revision}^{{tree}}"
        ) == first_tree
        message = "replacement refs"
    else:
        grafts = tmp_path / ".git" / "info" / "grafts"
        grafts.write_text(first_revision + "\n", encoding="ascii")
        message = "legacy Git grafts"

    with pytest.raises(capture.BaselineError, match=message):
        capture._assert_no_git_object_replacement(tmp_path, "test repository")


def test_materialized_execution_trees_reject_untracked_sitecustomize_and_special(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture,
        "UNMATERIALIZED_GITLINKS",
        {repository: {} for repository in capture.REPOSITORY_PATHS},
    )
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "baseline@example.invalid")
    _git(tmp_path, "config", "user.name", "Baseline Test")
    (tmp_path / ".gitignore").write_text(
        "ipfs_datasets_py/\nipfs_kit_py/\nsitecustomize.py\n*.fifo\n",
        encoding="utf-8",
    )
    (tmp_path / "accelerate.py").write_text("captured\n", encoding="utf-8")
    revisions: dict[str, str] = {}
    trees: dict[str, str] = {}
    revisions["datasets"], trees["datasets"] = _initialize_repository(
        tmp_path / "ipfs_datasets_py", "datasets.py"
    )
    revisions["kit"], trees["kit"] = _initialize_repository(
        tmp_path / "ipfs_kit_py", "kit.py"
    )
    _git(
        tmp_path,
        "add",
        "-f",
        ".gitignore",
        "accelerate.py",
        "ipfs_datasets_py",
        "ipfs_kit_py",
    )
    _git(tmp_path, "commit", "--quiet", "-m", "outer with reviewed gitlinks")
    revisions["accelerate"] = _git(tmp_path, "rev-parse", "HEAD")
    trees["accelerate"] = _git(tmp_path, "rev-parse", "HEAD^{tree}")
    snapshots = {
        repository: {
            "tested_revision": revisions[repository],
            "tested_tree": trees[repository],
        }
        for repository in capture.REPOSITORY_PATHS
    }
    replacement_commit = _git(
        tmp_path,
        "commit-tree",
        trees["accelerate"],
        "-p",
        revisions["accelerate"],
        "-m",
        "replacement",
    )
    _git(tmp_path, "replace", revisions["accelerate"], replacement_commit)
    rejected_capture_id = "20260811T115959.000000Z-1"
    with pytest.raises(capture.BaselineError, match="replacement refs"):
        capture._materialize_execution_trees(
            tmp_path, rejected_capture_id, snapshots
        )
    capture._safe_cleanup_capture_workspace(tmp_path, rejected_capture_id)
    _git(tmp_path, "replace", "-d", revisions["accelerate"])

    capture_id = "20260811T120000.000000Z-1"
    execution_root = capture._materialize_execution_trees(
        tmp_path, capture_id, snapshots
    )
    try:
        baseline = capture._assert_execution_trees_clean(execution_root, snapshots)
        for relative in capture.REPOSITORY_PATHS.values():
            target = execution_root / relative
            assert _git(target, "remote") == ""
            alternates = target / ".git" / "objects" / "info" / "alternates"
            assert not alternates.exists()

        grafts = execution_root / ".git" / "info" / "grafts"
        grafts.write_text(revisions["accelerate"] + "\n", encoding="ascii")
        with pytest.raises(capture.BaselineError, match="legacy Git grafts"):
            capture._assert_execution_trees_clean(
                execution_root, snapshots, baseline
            )
        grafts.unlink()

        tracked = execution_root / "accelerate.py"
        _git(execution_root, "update-index", "--assume-unchanged", "accelerate.py")
        tracked.write_text("evil = 1\n", encoding="utf-8")
        assert _git(execution_root, "status", "--porcelain") == ""
        with pytest.raises(capture.BaselineError, match="exact Git tree"):
            capture._assert_execution_trees_clean(execution_root, snapshots, baseline)
        tracked.write_text("captured\n", encoding="utf-8")
        _git(execution_root, "update-index", "--no-assume-unchanged", "accelerate.py")

        injected = execution_root / "sitecustomize.py"
        injected.write_text("raise RuntimeError('must not load')\n", encoding="utf-8")
        with pytest.raises(capture.BaselineError, match="not clean"):
            capture._assert_execution_trees_clean(execution_root, snapshots, baseline)
        injected.unlink()
        special = execution_root / "injected.fifo"
        os.mkfifo(special)
        with pytest.raises(capture.BaselineError, match="tracked leaves"):
            capture._assert_execution_trees_clean(execution_root, snapshots, baseline)
        special.unlink()

        nested_dotgit = execution_root / "tracked_dir" / ".git" / "config"
        nested_dotgit.parent.mkdir(parents=True)
        nested_dotgit.write_text("[core]\n", encoding="utf-8")
        with pytest.raises(capture.BaselineError, match="tracked leaves"):
            capture._assert_execution_trees_clean(execution_root, snapshots, baseline)
    finally:
        capture._safe_cleanup_capture_workspace(tmp_path, capture_id)
    assert not (
        tmp_path / capture.ARTIFACT_RELATIVE_ROOT / "work" / capture_id
    ).exists()


def _write_structural_present_receipt(
    root: Path,
    repository: str,
    revisions: dict[str, str],
    trees: dict[str, str],
    capture_id: str = "20260811T120000.000000Z-1",
) -> dict[str, object]:
    commands: list[dict[str, object]] = []
    for suite in capture.SUITES_BY_REPOSITORY[repository]:
        log_relative = (
            capture.ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
        ).as_posix()
        log_raw = f"public retained log for {suite.id}\n".encode()
        log_path = root / log_relative
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(log_raw)
        commands.append(
            {
                "id": suite.id,
                "suite_definition_digest": capture.suite_definition_digest(suite),
                "log": {
                    "relative_path": log_relative,
                    "bytes": len(log_raw),
                    "sha256": capture._sha256(log_raw),
                },
            }
        )
    task_id = {"accelerate": "IPS-001", "datasets": "IPS-002", "kit": "IPS-003"}[
        repository
    ]
    payload: dict[str, object] = {
        "schema_version": capture.SCHEMA_VERSION,
        "operator_origin": capture.OPERATOR_ORIGIN,
        "repository": repository,
        "task_id": task_id,
        "capture_id": capture_id,
        "captured_at": "2026-08-11T12:00:00.000000Z",
        "required_command_ids": [
            suite.id for suite in capture.SUITES_BY_REPOSITORY[repository]
        ],
        "planning_revision": capture.PLANNING_REVISIONS[repository],
        "planning_tree": capture.PLANNING_TREES[repository],
        "source_revision": revisions[repository],
        "source_tree": trees[repository],
        "execution_head": revisions["accelerate"],
        "execution_tree": trees["accelerate"],
        "source_revisions": revisions,
        "source_trees": trees,
        "source_clean_before": {name: True for name in revisions},
        "source_clean_after": {name: True for name in revisions},
        "ignored_sensitive_inputs": {
            "policy_id": capture.IGNORED_INPUT_POLICY_ID,
            "repositories": {
                name: capture._ignored_sensitive_binding({}) for name in revisions
            },
        },
        "git_environment_policy_id": capture.GIT_ENVIRONMENT_POLICY_ID,
        "commands": commands,
        "assurance": capture._assurance_payload(process_observed=True, aggregate=True),
        "receipt_digest": "",
    }
    payload["receipt_digest"] = capture.receipt_digest(payload)
    receipt_path = root / capture.ARTIFACT_RELATIVE_ROOT / f"{repository}.json"
    receipt_path.write_bytes(capture._canonical_bytes(payload) + b"\n")
    return payload


def test_render_pins_is_read_only_and_rejects_tampered_or_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pristine_scheduler(tmp_path)
    monkeypatch.setattr(capture, "_validate_protected_suite_registry", lambda _root: {})
    monkeypatch.setattr(capture, "_self_validate_payload", lambda *_args: None)
    revisions = {
        repository: (str(index + 1) * 40)[:40]
        for index, repository in enumerate(capture.REPOSITORY_PATHS)
    }
    trees = {
        repository: (str(index + 4) * 40)[:40]
        for index, repository in enumerate(capture.REPOSITORY_PATHS)
    }
    payloads = {
        repository: _write_structural_present_receipt(
            tmp_path, repository, revisions, trees
        )
        for repository in capture.REPOSITORY_PATHS
    }
    scheduler_before = (tmp_path / capture.SCHEDULER_CONFIG_RELATIVE).read_bytes()
    projection = capture.render_pin_projection(tmp_path)
    assert set(projection) == {"operator_baseline_receipts", "protected_paths"}
    assert set(projection["operator_baseline_receipts"]) == {
        "IPS-001",
        "IPS-002",
        "IPS-003",
    }
    expected_evidence = {
        (capture.ARTIFACT_RELATIVE_ROOT / f"{name}.json").as_posix()
        for name in capture.REPOSITORY_PATHS
    } | {
        command["log"]["relative_path"]
        for payload in payloads.values()
        for command in payload["commands"]
    }
    assert set(projection["protected_paths"]) == (
        set(capture.PRE_CAPTURE_PROTECTED_PATHS) | expected_evidence
    )
    assert (tmp_path / capture.SCHEDULER_CONFIG_RELATIVE).read_bytes() == scheduler_before

    stale_lock = (
        tmp_path / capture.ARTIFACT_RELATIVE_ROOT / capture.CAPTURE_LOCK_NAME
    )
    stale_lock.write_bytes(b'{"ambiguous":true}\n')
    with pytest.raises(capture.BaselineError, match="active or stale capture lock"):
        capture.render_pin_projection(tmp_path)
    stale_lock.unlink()

    logs = tmp_path / capture.ARTIFACT_RELATIVE_ROOT / "logs"
    held_logs = logs.with_name("logs-held")
    outside = tmp_path / "outside-logs"
    outside.mkdir()
    with monkeypatch.context() as swap_patch:
        real_try_open = capture._try_open_relative_directory
        swapped = False

        def swap_after_log_directory_open(
            root_descriptor: int, parts: object, *, label: str
        ) -> int | None:
            nonlocal swapped
            descriptor = real_try_open(root_descriptor, parts, label=label)
            if (
                label == "present receipt log directory"
                and descriptor is not None
                and not swapped
            ):
                swapped = True
                logs.rename(held_logs)
                logs.symlink_to(outside, target_is_directory=True)
            return descriptor

        swap_patch.setattr(
            capture, "_try_open_relative_directory", swap_after_log_directory_open
        )
        with pytest.raises(capture.BaselineError, match="log directory"):
            capture.render_pin_projection(tmp_path)
    logs.unlink()
    held_logs.rename(logs)

    tampered_log = tmp_path / payloads["kit"]["commands"][0]["log"]["relative_path"]
    original_log = tampered_log.read_bytes()
    tampered_log.write_bytes(original_log + b"tampered\n")
    with pytest.raises(capture.BaselineError, match="binding"):
        capture.render_pin_projection(tmp_path)
    tampered_log.write_bytes(original_log)

    (tmp_path / capture.ARTIFACT_RELATIVE_ROOT / "kit.json").unlink()
    with pytest.raises(capture.BaselineError):
        capture.render_pin_projection(tmp_path)


def test_current_relevance_allows_only_pin_and_evidence_paths_after_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The outer repository ignores the two independently versioned nested repositories.
    tmp_path.mkdir(exist_ok=True)
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "baseline@example.invalid")
    _git(tmp_path, "config", "user.name", "Baseline Test")
    (tmp_path / ".gitignore").write_text(
        "ipfs_datasets_py/\nipfs_kit_py/\nbuild/\n", encoding="utf-8"
    )
    base_config = {
        "schema": "test-scheduler@1",
        "operator_baseline_receipts": {},
        "protected_paths": ["scripts/capture_incremental_proof_sealer_baselines.py"],
    }
    config = tmp_path / capture.SCHEDULER_CONFIG_RELATIVE
    config.parent.mkdir()
    config.write_text(json.dumps(base_config), encoding="utf-8")
    (tmp_path / "accelerate.py").write_text("captured = True\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "accelerate.py", "config")
    _git(tmp_path, "commit", "--quiet", "-m", "captured")
    accelerate_revision = _git(tmp_path, "rev-parse", "HEAD")
    accelerate_tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
    datasets_revision, datasets_tree = _initialize_repository(
        tmp_path / "ipfs_datasets_py", "datasets.py"
    )
    kit_revision, kit_tree = _initialize_repository(tmp_path / "ipfs_kit_py", "kit.py")
    revisions = {
        "accelerate": accelerate_revision,
        "datasets": datasets_revision,
        "kit": kit_revision,
    }
    trees = {
        "accelerate": accelerate_tree,
        "datasets": datasets_tree,
        "kit": kit_tree,
    }
    monkeypatch.setattr(capture, "PLANNING_REVISIONS", revisions)
    monkeypatch.setattr(capture, "PLANNING_TREES", trees)
    # Commit one complete three-repository bundle, then recapture all three.
    # The superseded logs are deleted only after replacement and remain in Git
    # history at the second capture's exact source revision.
    first_payloads = {
        repository: _write_structural_present_receipt(
            tmp_path, repository, revisions, trees
        )
        for repository in capture.REPOSITORY_PATHS
    }
    _, pinned_paths, pins = capture._present_receipt_evidence_contract(tmp_path)
    pinned_config = copy.deepcopy(base_config)
    pinned_config["operator_baseline_receipts"] = pins
    pinned_config["protected_paths"] = [
        *base_config["protected_paths"],
        *sorted(pinned_paths),
    ]
    config.write_text(json.dumps(pinned_config), encoding="utf-8")
    _git(tmp_path, "add", "config", "artifacts")
    _git(tmp_path, "commit", "--quiet", "-m", "commit evidence and pin")
    assert _git(tmp_path, "rev-parse", "HEAD") != accelerate_revision

    for repository, payload in first_payloads.items():
        capture._validate_source_bindings(tmp_path, repository, payload)
        capture._validate_current_relevance(tmp_path, payload)

    recapture_revision = _git(tmp_path, "rev-parse", "HEAD")
    recapture_tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
    recapture_revisions = {**revisions, "accelerate": recapture_revision}
    recapture_trees = {**trees, "accelerate": recapture_tree}
    superseded_logs = {
        command["log"]["relative_path"]
        for payload in first_payloads.values()
        for command in payload["commands"]
    }
    second_payloads = {
        repository: _write_structural_present_receipt(
            tmp_path,
            repository,
            recapture_revisions,
            recapture_trees,
            "20260811T120001.000000Z-2",
        )
        for repository in capture.REPOSITORY_PATHS
    }
    for relative in superseded_logs:
        (tmp_path / relative).unlink()
    _, second_pinned_paths, second_pins = capture._present_receipt_evidence_contract(
        tmp_path
    )
    pinned_config = copy.deepcopy(pinned_config)
    pinned_config["operator_baseline_receipts"] = second_pins
    pinned_config["protected_paths"] = sorted(
        (set(pinned_config["protected_paths"]) - superseded_logs) | second_pinned_paths
    )
    config.write_text(json.dumps(pinned_config), encoding="utf-8")
    _git(tmp_path, "add", "-A", "config", "artifacts")
    _git(tmp_path, "commit", "--quiet", "-m", "recapture all evidence")

    for repository, payload in second_payloads.items():
        capture._validate_source_bindings(tmp_path, repository, payload)
        capture._validate_current_relevance(tmp_path, payload)
    receipt_source = second_payloads["accelerate"]

    stale_lock = (
        tmp_path / capture.ARTIFACT_RELATIVE_ROOT / capture.CAPTURE_LOCK_NAME
    )
    stale_lock.write_bytes(b'{"ambiguous":true}\n')
    with pytest.raises(capture.BaselineError, match="active or stale capture lock"):
        capture._validate_current_relevance(tmp_path, receipt_source)
    stale_lock.unlink()

    ignored_key = tmp_path / "build" / "proving_key.bin"
    ignored_key.parent.mkdir()
    ignored_key.write_bytes(b"A" * 32)
    capture._validate_current_relevance(tmp_path, receipt_source)
    ignored_key.write_bytes(b"B" * 32)
    capture._validate_current_relevance(tmp_path, receipt_source)
    ignored_key.unlink()
    ignored_key.parent.rmdir()

    extra_log = (
        tmp_path
        / capture.ARTIFACT_RELATIVE_ROOT
        / "logs"
        / "accelerate-proof-focused-core-15-20260811T120001.000000Z-1.log"
    )
    extra_log.write_text("valid-looking but unreferenced\n", encoding="utf-8")
    with pytest.raises(capture.BaselineError, match="orphan logs"):
        capture._validate_current_relevance(tmp_path, receipt_source)
    extra_log.unlink()

    arbitrary_config = copy.deepcopy(pinned_config)
    arbitrary_config["arbitrary_post_capture_field"] = True
    config.write_text(json.dumps(arbitrary_config), encoding="utf-8")
    with pytest.raises(
        capture.BaselineError, match="not limited to exact receipt pins"
    ):
        capture._validate_current_relevance(tmp_path, receipt_source)
    config.write_text(json.dumps(pinned_config), encoding="utf-8")

    # A later source edit leaves the historical receipt internally bound but
    # makes it inadmissible for the current checkout.
    (tmp_path / "accelerate.py").write_text(
        "captured = True\nchanged = True\n", encoding="utf-8"
    )
    _git(tmp_path, "add", "accelerate.py")
    _git(tmp_path, "commit", "--quiet", "-m", "change source")
    capture._validate_source_bindings(tmp_path, "accelerate", receipt_source)
    with pytest.raises(capture.BaselineError, match="outside post-capture evidence"):
        capture._validate_current_relevance(tmp_path, receipt_source)
