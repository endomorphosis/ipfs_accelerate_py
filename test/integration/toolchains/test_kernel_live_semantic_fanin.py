"""Live semantic fan-in for Lean, Rocq, and Isabelle kernels (FVT-057 / FVT-G206).

``KernelLiveSemanticFanIn@1``

Require each installed proof kernel to check its own generated source and retain
assumptions, imports/session, theorem, mutation, and output digests. Acceptance:

* Lean, Rocq, and Isabelle independently execute a valid theorem, false theorem,
  hypothesis/conclusion mutation, deterministic replay, malformed source,
  timeout, and forbidden admit/axiom-oracle checks;
* Isabelle's live source/session helper is exercised rather than only offline
  fixtures;
* receipts bind exact kernel, dependency, source, imports/session, assumptions,
  theorem, and output digests;
* no advisor or sibling kernel substitutes for the selected kernel.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LEAN_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "lean.py"
ROCQ_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "rocq.py"
ISABELLE_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "isabelle.py"
CERTIFICATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_kernel_live_certificate.json"
)

FANIN_INTERFACE = "KernelLiveSemanticFanIn@1"
FANIN_SCHEMA = "kernel-live-semantic-fanin/v1"
FANIN_GOAL_ID = "FVT-G206"
FANIN_TASK_ID = "FVT-057"

REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "fail_closed",
}

REQUIRED_KERNELS = ("lean", "rocq", "isabelle")


def _ensure_import_paths() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    _ensure_import_paths()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lean_cert():
    return _load_module(LEAN_CERT_PATH, "tools_logic_certification_lean_fanin")


@pytest.fixture(scope="module")
def rocq_cert():
    return _load_module(ROCQ_CERT_PATH, "tools_logic_certification_rocq_fanin")


@pytest.fixture(scope="module")
def isabelle_cert():
    return _load_module(ISABELLE_CERT_PATH, "tools_logic_certification_isabelle_fanin")


@pytest.fixture(scope="module")
def lean_contribution(lean_cert) -> dict[str, Any]:
    return lean_cert.build_live_fanin_contribution(
        repo_root=REPO_ROOT,
        env=lean_cert.offline_env(os.environ),
    )


@pytest.fixture(scope="module")
def rocq_contribution(rocq_cert) -> dict[str, Any]:
    return rocq_cert.build_live_fanin_contribution(
        repo_root=REPO_ROOT,
        env=rocq_cert.offline_env(os.environ),
    )


@pytest.fixture(scope="module")
def isabelle_contribution(isabelle_cert) -> dict[str, Any]:
    return isabelle_cert.build_live_fanin_contribution(
        repo_root=REPO_ROOT,
        env=isabelle_cert.offline_env(os.environ),
    )


@pytest.fixture(scope="module")
def assembled_certificate(
    lean_cert,
    lean_contribution,
    rocq_contribution,
    isabelle_contribution,
) -> dict[str, Any]:
    return lean_cert.assemble_kernel_live_fanin_certificate(
        {
            "lean": lean_contribution,
            "rocq": rocq_contribution,
            "isabelle": isabelle_contribution,
        },
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Expected outputs / interface contract
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert LEAN_CERT_PATH.is_file()
    assert ROCQ_CERT_PATH.is_file()
    assert ISABELLE_CERT_PATH.is_file()
    assert CERTIFICATE_PATH.is_file()
    assert Path(__file__).is_file()


def test_fanin_constants(lean_cert, rocq_cert, isabelle_cert) -> None:
    for mod in (lean_cert, rocq_cert, isabelle_cert):
        assert mod.FANIN_INTERFACE == FANIN_INTERFACE
        assert mod.FANIN_SCHEMA_VERSION == FANIN_SCHEMA
        assert mod.FANIN_GOAL_ID == FANIN_GOAL_ID
        assert mod.FANIN_TASK_ID == FANIN_TASK_ID
        assert set(mod.REQUIRED_FANIN_CASE_KINDS) == REQUIRED_CASE_KINDS
        recipes = mod.live_fanin_case_recipes()
        kinds = {str(item["kind"]) for item in recipes}
        assert REQUIRED_CASE_KINDS <= kinds


def test_certificate_schema() -> None:
    payload = json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == FANIN_SCHEMA
    assert payload["interface"] == FANIN_INTERFACE
    assert payload["goal_id"] == FANIN_GOAL_ID
    assert payload["task_id"] == FANIN_TASK_ID
    assert payload["lane_id"] == "kernel"
    policy = payload["policy"]
    assert policy["own_kernel_only"] is True
    assert policy["sibling_kernel_substitution_forbidden"] is True
    assert policy["advisor_substitution_forbidden"] is True
    assert policy["isabelle_live_source_session_helper_required"] is True
    assert set(policy["required_case_kinds"]) == REQUIRED_CASE_KINDS
    assert list(payload["kernel_ids"]) == list(REQUIRED_KERNELS)
    for kernel_id in REQUIRED_KERNELS:
        kernel = payload["kernels"][kernel_id]
        assert kernel["kernel_id"] == kernel_id
        assert kernel["fanin_interface"] == FANIN_INTERFACE
        assert kernel["sibling_kernel_substitution"] is False
        assert kernel["advisor_substitution"] is False
        assert set(kernel["required_case_kinds"]) == REQUIRED_CASE_KINDS
        assert "bindings" in kernel
        assert kernel["bindings"]["authority"]["selected_kernel"] == kernel_id


# ---------------------------------------------------------------------------
# Per-kernel live fan-in
# ---------------------------------------------------------------------------


def _assert_contribution_shape(contrib: dict[str, Any], *, kernel_id: str) -> None:
    assert contrib["kernel_id"] == kernel_id
    assert contrib["fanin_interface"] == FANIN_INTERFACE
    assert contrib["goal_id"] == FANIN_GOAL_ID
    assert contrib["task_id"] == FANIN_TASK_ID
    assert contrib["sibling_kernel_substitution"] is False
    assert contrib["advisor_substitution"] is False
    assert contrib["network_used"] is False
    assert contrib["install_attempted"] is False
    assert contrib["download_attempted"] is False
    kinds = {case["kind"] for case in contrib["cases"]}
    assert REQUIRED_CASE_KINDS <= kinds
    bindings = contrib["bindings"]
    assert bindings["authority"]["selected_kernel"] == kernel_id
    assert bindings["authority"]["sibling_kernel_substitution_forbidden"] is True
    assert bindings["authority"]["advisor_substitution_forbidden"] is True
    assert "source" in bindings and bindings["source"]["source_digest"]
    assert "theorem" in bindings
    assert "assumptions" in bindings
    assert "imports" in bindings
    assert "output" in bindings
    assert contrib["contribution_digest_sha256"]
    assert len(contrib["contribution_digest_sha256"]) == 64


def _assert_case_matrix(contrib: dict[str, Any]) -> None:
    by_id = {case["case_id"]: case for case in contrib["cases"]}
    checks = {check["check_id"].split(".")[-1]: check for check in contrib["checks"]}

    positive = by_id["true_theorem"]
    assert positive["accepted"] is True
    assert checks["true_theorem"]["status"] == "passed"

    for case_id in (
        "false_proof",
        "hypothesis_mutation",
        "conclusion_mutation",
        "malformed_source",
        "timeout_case",
    ):
        assert by_id[case_id]["accepted"] is False
        assert checks[case_id]["status"] == "passed"

    replay = by_id["deterministic_replay"]
    assert replay["accepted"] is True
    assert replay["source_digest"] == positive["source_digest"]
    assert checks["deterministic_replay"]["status"] == "passed"

    # Forbidden admit / axiom-oracle style escapes (kernel-specific ids).
    fail_closed_ids = [
        case_id
        for case_id, case in by_id.items()
        if case["kind"] == "fail_closed"
    ]
    assert fail_closed_ids
    for case_id in fail_closed_ids:
        assert by_id[case_id]["accepted"] is False
        assert checks[case_id]["status"] == "passed"


def test_lean_live_fanin_matrix(lean_contribution: dict[str, Any]) -> None:
    _assert_contribution_shape(lean_contribution, kernel_id="lean")
    if not lean_contribution.get("usable"):
        pytest.skip(f"Lean pin unavailable: {lean_contribution.get('block_reasons')}")
    assert lean_contribution["live_source_helper"] == "check_lean_source"
    assert lean_contribution["live_executed"] is True
    _assert_case_matrix(lean_contribution)
    assert lean_contribution["fanin_passed"] is True
    assert lean_contribution["bindings"]["locked_toolchain"]
    assert lean_contribution["bindings"]["authority"]["not_rocq"] is True
    assert lean_contribution["bindings"]["authority"]["not_isabelle"] is True


def test_rocq_live_fanin_matrix(rocq_contribution: dict[str, Any]) -> None:
    _assert_contribution_shape(rocq_contribution, kernel_id="rocq")
    if not rocq_contribution.get("usable"):
        pytest.skip(f"Rocq pin unavailable: {rocq_contribution.get('block_reasons')}")
    assert rocq_contribution["live_source_helper"] == "check_rocq_source_live"
    assert rocq_contribution["live_executed"] is True
    _assert_case_matrix(rocq_contribution)
    assert rocq_contribution["fanin_passed"] is True
    assert rocq_contribution["bindings"]["package_identity"]
    assert rocq_contribution["bindings"]["authority"]["not_lean"] is True
    assert rocq_contribution["bindings"]["authority"]["not_isabelle"] is True
    assert rocq_contribution["bindings"]["authority"]["opam_cannot_promote_kernel_lane"] is True


def test_isabelle_live_source_helper_and_matrix(
    isabelle_contribution: dict[str, Any],
) -> None:
    _assert_contribution_shape(isabelle_contribution, kernel_id="isabelle")
    if not isabelle_contribution.get("usable"):
        pytest.skip(
            f"Isabelle pin unavailable: {isabelle_contribution.get('block_reasons')}"
        )
    # Acceptance explicitly requires the live helper, not offline fixtures only.
    assert isabelle_contribution["live_source_helper"] == "check_isabelle_source_live"
    assert isabelle_contribution["live_source_helper_exercised"] is True
    assert isabelle_contribution["live_executed"] is True
    _assert_case_matrix(isabelle_contribution)
    assert isabelle_contribution["fanin_passed"] is True
    session = isabelle_contribution["bindings"]["session"]
    assert session["name"]
    assert "build" in session["process_command_template"]
    assert isabelle_contribution["bindings"]["authority"]["not_lean"] is True
    assert isabelle_contribution["bindings"]["authority"]["not_rocq"] is True
    assert isabelle_contribution["bindings"]["authority"]["hammer_is_proposal_only"] is True


def test_no_sibling_kernel_substitution(
    lean_contribution: dict[str, Any],
    rocq_contribution: dict[str, Any],
    isabelle_contribution: dict[str, Any],
) -> None:
    """Each contribution selects only its own kernel; no cross-substitution."""

    mapping = {
        "lean": lean_contribution,
        "rocq": rocq_contribution,
        "isabelle": isabelle_contribution,
    }
    for kernel_id, contrib in mapping.items():
        assert contrib["kernel_id"] == kernel_id
        assert contrib["sibling_kernel_substitution"] is False
        assert contrib["advisor_substitution"] is False
        authority = contrib["bindings"]["authority"]
        assert authority["selected_kernel"] == kernel_id
        assert authority["not_advisor"] is True
        # Executable path, when present, must not be a sibling kernel binary.
        path = str(contrib.get("executable_path") or "").lower()
        if not path:
            continue
        if kernel_id == "lean":
            assert "coqc" not in path and "isabelle" not in path
        elif kernel_id == "rocq":
            assert "lean" not in Path(path).name and "isabelle" not in path
        elif kernel_id == "isabelle":
            assert "lean" not in Path(path).name and "coqc" not in path


# ---------------------------------------------------------------------------
# Aggregated certificate
# ---------------------------------------------------------------------------


def test_assembled_certificate_binds_all_kernels(
    assembled_certificate: dict[str, Any],
    lean_contribution: dict[str, Any],
    rocq_contribution: dict[str, Any],
    isabelle_contribution: dict[str, Any],
) -> None:
    cert = assembled_certificate
    assert cert["interface"] == FANIN_INTERFACE
    assert cert["schema_version"] == FANIN_SCHEMA
    assert cert["goal_id"] == FANIN_GOAL_ID
    assert cert["task_id"] == FANIN_TASK_ID
    assert list(cert["kernel_ids"]) == list(REQUIRED_KERNELS)
    assert cert["policy"]["sibling_kernel_substitution_forbidden"] is True
    assert cert["policy"]["advisor_substitution_forbidden"] is True
    assert cert["policy"]["isabelle_live_source_session_helper_required"] is True
    assert cert["receipt_digest_sha256"]
    assert len(cert["receipt_digest_sha256"]) == 64

    for kernel_id, contrib in (
        ("lean", lean_contribution),
        ("rocq", rocq_contribution),
        ("isabelle", isabelle_contribution),
    ):
        bound = cert["kernels"][kernel_id]
        assert bound["kernel_id"] == kernel_id
        assert bound["fanin_passed"] == contrib["fanin_passed"]
        assert bound["contribution_digest_sha256"] == contrib["contribution_digest_sha256"]
        kernel_binding = cert["bindings"]["kernels"][kernel_id]
        assert "source_digest" in kernel_binding
        assert "output_digest" in kernel_binding
        assert "theorem" in kernel_binding

    if all(
        contrib.get("fanin_passed")
        for contrib in (lean_contribution, rocq_contribution, isabelle_contribution)
    ):
        assert cert["all_kernels_passed"] is True
        assert cert["production_certified"] is True
        assert cert["promotion_blocked"] is False
        assert cert["block_reasons"] == []


def test_checked_in_certificate_matches_live_when_all_usable(
    assembled_certificate: dict[str, Any],
    lean_contribution: dict[str, Any],
    rocq_contribution: dict[str, Any],
    isabelle_contribution: dict[str, Any],
) -> None:
    """When all kernels are live-usable, the checked-in certificate is current."""

    if not all(
        contrib.get("fanin_passed")
        for contrib in (lean_contribution, rocq_contribution, isabelle_contribution)
    ):
        pytest.skip("Not all kernels usable; schema contract covered separately")

    on_disk = json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))
    assert on_disk["interface"] == FANIN_INTERFACE
    assert on_disk["all_kernels_passed"] is True
    assert on_disk["production_certified"] is True
    for kernel_id in REQUIRED_KERNELS:
        assert on_disk["kernels"][kernel_id]["fanin_passed"] is True
        assert on_disk["kernels"][kernel_id]["live_executed"] is True
        if kernel_id == "isabelle":
            assert on_disk["kernels"][kernel_id]["live_source_helper_exercised"] is True
        # Case matrix present on disk.
        kinds = {case["kind"] for case in on_disk["kernels"][kernel_id]["cases"]}
        assert REQUIRED_CASE_KINDS <= kinds

    # Live assembly must also report full pass under the same environment.
    assert assembled_certificate["all_kernels_passed"] is True
    assert assembled_certificate["production_certified"] is True


def test_sibling_substitution_fails_closed(lean_cert, lean_contribution) -> None:
    poisoned = dict(lean_contribution)
    poisoned["kernel_id"] = "lean"
    poisoned = {
        **poisoned,
        "bindings": {
            **dict(lean_contribution.get("bindings") or {}),
            "authority": {
                **dict((lean_contribution.get("bindings") or {}).get("authority") or {}),
                "selected_kernel": "rocq",  # deliberate sibling claim
            },
        },
    }
    cert = lean_cert.assemble_kernel_live_fanin_certificate(
        {
            "lean": poisoned,
            "rocq": {
                "kernel_id": "rocq",
                "fanin_passed": True,
                "sibling_kernel_substitution": False,
                "advisor_substitution": False,
                "bindings": {"authority": {"selected_kernel": "rocq"}},
                "contribution_digest_sha256": "0" * 64,
            },
            "isabelle": {
                "kernel_id": "isabelle",
                "fanin_passed": True,
                "sibling_kernel_substitution": False,
                "advisor_substitution": False,
                "bindings": {"authority": {"selected_kernel": "isabelle"}},
                "contribution_digest_sha256": "1" * 64,
            },
        },
        repo_root=REPO_ROOT,
    )
    assert cert["production_certified"] is False
    assert cert["promotion_blocked"] is True
    assert any(
        reason.startswith("sibling_substitution:") for reason in cert["block_reasons"]
    )
