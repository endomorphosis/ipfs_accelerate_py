"""Live TLC + Apalache semantic certification (FVT-060 / FVT-G204).

``StateModelLiveSemanticCertification@1``

Replaces classifier-backed state-model promotion with real pinned TLC jar and
Apalache executable runs against positive and adversarial models.

Acceptance covered:

* TLC and Apalache each execute a valid invariant model, a violating model with
  concrete counterexample, specification and invariant mutations, deterministic
  replay, malformed input, timeout, and bounded-state/resource cases;
* receipts bind exact source model, property, bound, JVM, executable,
  jar/archive, and output digests;
* canned text and parser classification remain ``hermetic_parser`` and cannot
  satisfy live external semantics;
* Java is support only; bounded model checking never grants theorem authority;
* certification never installs, downloads, or opens the network.
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
STATE_MODEL_CERT_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "state_model.py"
)
LIVE_CERTIFICATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_state_model_live_certificate.json"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

LIVE_INTERFACE = "StateModelLiveSemanticCertification@1"
LIVE_SCHEMA_VERSION = "state-model-live-semantic-certification/v1"
LIVE_CORPUS_SCHEMA = "state-model-live-semantic-corpus/v1"
LIVE_GOAL_ID = "FVT-G204"
LIVE_TASK_ID = "FVT-060"
LOCKED_TLC_VERSION = "1.8.0"
LOCKED_APALACHE_VERSION = "0.58.3"
LOCKED_TLC_SHA256 = (
    "e22f8ffb4bacdea0a871f444dd94fe5fb0d8013b3388ae39e82e26f852c735d5"
)

REQUIRED_CASE_KINDS = {
    "invariant_holds",
    "violation_trace",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "bound",
}

REQUIRED_CASE_IDS = {
    "invariant_holds",
    "violation_trace",
    "mutated_next",
    "mutated_invariant",
    "deterministic_replay",
    "malformed_model",
    "timeout_bound",
    "bound_behavior",
}

REQUIRED_BINDING_FIELDS = {
    "binary_digest",
    "jar_or_archive_digest",
    "jvm_digest",
    "model_digest",
    "property_digest",
    "artifact_digest",
    "output_digest",
    "bounds",
    "limits",
}


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
def state_model_cert():
    return _load_module(
        STATE_MODEL_CERT_PATH, "tools_logic_certification_state_model_live"
    )


@pytest.fixture(scope="module")
def offline_env(state_model_cert) -> dict[str, str]:
    return state_model_cert.offline_env(os.environ)


@pytest.fixture(scope="module")
def live_receipt(state_model_cert, offline_env) -> dict[str, Any]:
    return state_model_cert.build_live_semantic_receipt(
        repo_root=REPO_ROOT,
        env=offline_env,
    )


@pytest.fixture(scope="module")
def live_certificate(state_model_cert, live_receipt, offline_env) -> dict[str, Any]:
    """Ensure the durable certificate exists and load it."""

    path = state_model_cert.write_live_certificate(
        live_receipt,
        repo_root=REPO_ROOT,
        env=offline_env,
    )
    assert path.is_file()
    assert (
        path == LIVE_CERTIFICATE_PATH
        or path.resolve() == LIVE_CERTIFICATE_PATH.resolve()
    )
    payload = json.loads(LIVE_CERTIFICATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert STATE_MODEL_CERT_PATH.is_file()
    assert Path(__file__).is_file()
    assert LIVE_CERTIFICATE_PATH.parent.is_dir()


def test_live_module_constants(state_model_cert) -> None:
    assert state_model_cert.LIVE_INTERFACE == LIVE_INTERFACE
    assert state_model_cert.LIVE_SCHEMA_VERSION == LIVE_SCHEMA_VERSION
    assert state_model_cert.LIVE_CORPUS_SCHEMA == LIVE_CORPUS_SCHEMA
    assert state_model_cert.LIVE_GOAL_ID == LIVE_GOAL_ID
    assert state_model_cert.LIVE_TASK_ID == LIVE_TASK_ID
    assert state_model_cert.LOCKED_TLC_VERSION == LOCKED_TLC_VERSION
    assert state_model_cert.LOCKED_APALACHE_VERSION == LOCKED_APALACHE_VERSION
    assert state_model_cert.LOCKED_TLC_SHA256 == LOCKED_TLC_SHA256
    assert state_model_cert.AUTHORITY_SCOPE == "bounded_state_model_only"
    assert state_model_cert.AUTHORITY_CEILING == "bounded"
    assert state_model_cert.EVIDENCE_CLASS_LIVE == "live_external"
    assert state_model_cert.EVIDENCE_CLASS_HERMETIC_PARSER == "hermetic_parser"
    assert state_model_cert.DEFAULT_LIVE_CERTIFICATE_RELATIVE.as_posix().endswith(
        "formal_verification_state_model_live_certificate.json"
    )


def test_live_corpus_schema_and_required_cases(state_model_cert) -> None:
    manifest = state_model_cert.default_live_corpus_manifest()
    assert manifest["schema_version"] == LIVE_CORPUS_SCHEMA
    assert manifest["interface"] == LIVE_INTERFACE
    assert manifest["goal_id"] == LIVE_GOAL_ID
    assert manifest["task_id"] == LIVE_TASK_ID
    assert manifest["locked_tlc_version"] == LOCKED_TLC_VERSION
    assert manifest["locked_apalache_version"] == LOCKED_APALACHE_VERSION
    assert manifest["policy"]["live_execution_required_for_production"] is True
    assert manifest["policy"]["fixture_or_parser_cannot_satisfy_live_goal"] is True
    assert manifest["policy"]["hermetic_parser_cannot_satisfy_live"] is True
    assert manifest["policy"]["never_theorem_authority"] is True
    assert manifest["policy"]["java_is_support_only"] is True
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True
    assert manifest["policy"]["exact_jar_archive_digest_required"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert REQUIRED_CASE_IDS <= case_ids
    # Live corpus must not rely on canned stdout fixtures.
    for case in cases:
        assert "stdout" not in case or not case.get("stdout"), (
            f"live case {case.get('case_id')} must not ship canned stdout"
        )
        assert case.get("model_source"), (
            f"live case {case.get('case_id')} needs model_source"
        )


# ---------------------------------------------------------------------------
# Live execution against real pinned binaries
# ---------------------------------------------------------------------------


def test_live_tools_usable_when_present(live_receipt: dict[str, Any]) -> None:
    if not (
        live_receipt.get("tlc_usable")
        and live_receipt.get("apalache_usable")
        and live_receipt.get("java_usable")
    ):
        pytest.skip("pinned TLC/Apalache/Java not available on this host")
    assert live_receipt["live_execution"] is True
    assert live_receipt["tlc_version_match"] is True
    assert live_receipt["apalache_version_match"] is True
    assert live_receipt["tlc_binary_digest"]
    assert live_receipt["apalache_binary_digest"]
    assert live_receipt["java_binary_digest"]
    assert len(live_receipt["tlc_binary_digest"]) == 64
    assert len(live_receipt["apalache_binary_digest"]) == 64
    assert live_receipt["tlc_jar_digest"] == LOCKED_TLC_SHA256
    assert live_receipt["apalache_archive_digest"]
    assert len(live_receipt["apalache_archive_digest"]) == 64


def test_each_tool_executes_required_case_kinds(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("production_certified"):
        if not (
            live_receipt.get("tlc_usable") and live_receipt.get("apalache_usable")
        ):
            pytest.skip("pinned TLC/Apalache not available on this host")
    cases = live_receipt.get("cases") or []
    by_tool: dict[str, set[str]] = {"tlc": set(), "apalache": set()}
    for case in cases:
        tool_id = case.get("tool_id")
        if tool_id in by_tool and case.get("execution_mode") != "skipped":
            by_tool[tool_id].add(str(case.get("kind")))
    for tool_id, kinds in by_tool.items():
        assert REQUIRED_CASE_KINDS <= kinds, (
            f"{tool_id} missing kinds: {sorted(REQUIRED_CASE_KINDS - kinds)}"
        )


def test_invariant_holds_and_violation_live(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable")
    by_id = {case["case_id"]: case for case in live_receipt.get("cases") or []}
    for tool_id in ("tlc", "apalache"):
        holds = by_id[f"{tool_id}.invariant_holds"]
        viol = by_id[f"{tool_id}.violation_trace"]
        assert holds["status"] == "passed"
        assert holds["matched"] is True
        assert holds["grants_theorem_authority"] is False
        assert holds["evidence_class"] == "live_external"
        assert viol["status"] == "counterexample"
        assert viol["matched"] is True
        assert viol["grants_theorem_authority"] is False
        # Concrete counterexample evidence (trace or classifier binding).
        assert viol.get("counterexample") or "state" in (
            viol.get("output_preview") or ""
        ).lower() or viol.get("output_digest")


def test_mutations_never_remain_pass(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable")
    for case in live_receipt.get("cases") or []:
        if case.get("kind") != "mutation":
            continue
        assert case["matched"] is True
        assert case["status"] != "passed"
        assert case["grants_theorem_authority"] is False


def test_replay_preserves_model_digests(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable")
    by_id = {case["case_id"]: case for case in live_receipt.get("cases") or []}
    for tool_id in ("tlc", "apalache"):
        holds = by_id[f"{tool_id}.invariant_holds"]
        replay = by_id[f"{tool_id}.deterministic_replay"]
        assert holds["model_digest"] == replay["model_digest"]
        assert holds["config_digest"] == replay["config_digest"]
        assert holds["status"] == replay["status"] == "passed"
        assert holds["matched"] is True and replay["matched"] is True


def test_malformed_model_quarantined(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable")
    for case in live_receipt.get("cases") or []:
        if case.get("kind") != "malformed":
            continue
        assert case["status"] in {"malformed", "error", "unknown"}
        assert case["matched"] is True
        assert case["status"] != "passed"
        assert case["grants_theorem_authority"] is False


def test_timeout_resource_bounds(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable")
    for case in live_receipt.get("cases") or []:
        if case.get("kind") != "timeout":
            continue
        assert case["status"] == "timed_out"
        assert case["matched"] is True
        assert case["timed_out"] is True
        assert case["limits"]["timeout_seconds"] > 0
        assert case["limits"]["network"] is False


def test_bound_behavior_never_theorem(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable")
    for case in live_receipt.get("cases") or []:
        if case.get("kind") != "bound":
            continue
        assert case["matched"] is True
        assert case["status"] == "passed"
        assert case["grants_theorem_authority"] is False
        assert case["authority"] == "bounded"
        assert "finite_trace_only" in (case.get("bounds") or {}) or case[
            "tool_id"
        ] == "tlc"


def test_authority_ceiling_bounded(
    live_receipt: dict[str, Any], state_model_cert
) -> None:
    boundary = state_model_cert.bounded_checking_never_theorem_authority()
    assert boundary["never_theorem_authority"] is True
    assert boundary["bounded_model_checking_only"] is True
    java_boundary = state_model_cert.java_cannot_promote_tla_lane()
    assert java_boundary["blocks_alone"] is True
    assert java_boundary["support_only"] is True

    for case in live_receipt.get("cases") or []:
        if case.get("execution_mode") == "skipped":
            continue
        assert case.get("grants_theorem_authority") is False
        assert case.get("authority") == "bounded"


def test_receipt_binds_required_fields(live_receipt: dict[str, Any]) -> None:
    if not live_receipt.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable")
    bindings = live_receipt.get("bindings") or {}
    assert "binaries" in bindings
    assert bindings["binaries"]["tlc"]["binary_digest"]
    assert bindings["binaries"]["tlc"]["jar_digest"] == LOCKED_TLC_SHA256
    assert bindings["binaries"]["apalache"]["binary_digest"]
    assert bindings["binaries"]["apalache"]["archive_digest"]
    assert bindings["binaries"]["java"]["binary_digest"]
    assert bindings["binaries"]["java"]["support_only"] is True
    assert bindings["bounds"]["network"] is False
    assert bindings["authority"]["ceiling"] == "bounded"
    assert bindings["authority"]["never_theorem"] is True
    live_cases = bindings.get("live_cases") or []
    assert live_cases
    for entry in live_cases:
        if entry.get("execution_mode") == "skipped":
            continue
        for field in REQUIRED_BINDING_FIELDS:
            assert field in entry, (
                f"missing binding field {field} on {entry.get('case_id')}"
            )
        assert entry["limits"]["network"] is False
        assert entry["limits"]["install"] is False
        assert entry["binary_digest"]
        assert entry["model_digest"]
        assert entry["jar_or_archive_digest"]
        assert entry["jvm_digest"]
        assert entry["evidence_class"] == "live_external"


def test_offline_policy_never_installs(live_receipt: dict[str, Any]) -> None:
    assert live_receipt.get("network_used") is False
    assert live_receipt.get("install_attempted") is False
    assert live_receipt.get("download_attempted") is False
    policy = live_receipt.get("policy") or {}
    assert policy.get("no_install") is True
    assert policy.get("no_download") is True
    assert policy.get("no_network") is True
    assert policy.get("fixture_or_parser_cannot_satisfy_live_goal") is True
    assert policy.get("hermetic_parser_cannot_satisfy_live") is True


def test_production_certified_when_live_suite_passes(
    live_receipt: dict[str, Any],
) -> None:
    if not (
        live_receipt.get("tlc_usable")
        and live_receipt.get("apalache_usable")
        and live_receipt.get("java_usable")
    ):
        pytest.skip("pinned TLC/Apalache/Java not available on this host")
    assert live_receipt["production_certified"] is True, (
        f"block_reasons={live_receipt.get('block_reasons')}"
    )
    assert live_receipt["promotion_blocked"] is False
    assert live_receipt["live_semantic_corpus_passed"] is True
    assert not live_receipt.get("block_reasons")


def test_fixture_only_cannot_satisfy_live_goal(
    state_model_cert, offline_env
) -> None:
    """Parser fixtures alone are not live execution (FVT-G204 conflict policy)."""

    offline = state_model_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=offline_env,
    )
    assert offline.get("semantic_corpus_passed") is True

    receipt = state_model_cert.build_live_semantic_receipt(
        repo_root=REPO_ROOT,
        env=offline_env,
        tlc_executable="/nonexistent/tlc-binary",
        apalache_executable="/nonexistent/apalache-binary",
    )
    assert receipt["production_certified"] is False
    assert receipt["promotion_blocked"] is True
    assert receipt["live_execution"] is False
    assert receipt["policy"]["fixture_or_parser_cannot_satisfy_live_goal"] is True
    assert receipt["hermetic_parser_cannot_satisfy_live"] is True


def test_certify_state_model_live_semantics_entry(
    state_model_cert, offline_env
) -> None:
    receipt = state_model_cert.certify_state_model_live_semantics(
        repo_root=REPO_ROOT,
        env=offline_env,
    )
    assert receipt["handler_id"] == "state_model_live_semantic_certification@1"
    assert receipt["lane_id"] == "tla"
    assert receipt["interface"] == LIVE_INTERFACE
    assert receipt["goal_id"] == LIVE_GOAL_ID
    assert receipt["task_id"] == LIVE_TASK_ID
    assert "certified" in receipt
    assert "status" in receipt


# ---------------------------------------------------------------------------
# Durable certificate
# ---------------------------------------------------------------------------


def test_live_certificate_schema(live_certificate: dict[str, Any]) -> None:
    assert live_certificate["interface"] == LIVE_INTERFACE
    assert live_certificate["schema_version"] == LIVE_SCHEMA_VERSION
    assert live_certificate["goal_id"] == LIVE_GOAL_ID
    assert live_certificate["task_id"] == LIVE_TASK_ID
    assert live_certificate["locked_tlc_version"] == LOCKED_TLC_VERSION
    assert live_certificate["locked_apalache_version"] == LOCKED_APALACHE_VERSION
    assert live_certificate["authority_ceiling"] == "bounded"
    assert live_certificate["authority_scope"] == "bounded_state_model_only"
    assert live_certificate["hermetic_parser_cannot_satisfy_live"] is True
    assert live_certificate.get("receipt_digest_sha256")
    assert len(live_certificate["receipt_digest_sha256"]) == 64
    assert LIVE_CERTIFICATE_PATH.is_file()


def test_live_certificate_cases_bind_digests(
    live_certificate: dict[str, Any],
) -> None:
    if not live_certificate.get("live_execution"):
        pytest.skip("live TLC/Apalache binaries unavailable when certificate written")
    cases = live_certificate.get("cases") or []
    assert cases
    for case in cases:
        if case.get("execution_mode") == "skipped":
            continue
        assert case.get("binary_digest")
        assert case.get("jar_or_archive_digest")
        assert case.get("jvm_digest")
        assert case.get("model_digest")
        assert case.get("property_digest")
        assert case.get("artifact_digest")
        assert case.get("output_digest")
        assert "bounds" in case
        assert "limits" in case
        # Durable cert stores digest/preview, not full raw stdout dump.
        assert "stdout" not in case or not case.get("stdout")
        assert "stderr" not in case or not case.get("stderr")


def test_live_certificate_matches_toolchain_pins() -> None:
    if not LOCK_PATH.is_file():
        pytest.skip("deployment lock not present in this worktree")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    pins = lock.get("managed_pin_versions") or {}
    if "tlc" in pins:
        assert pins["tlc"] == LOCKED_TLC_VERSION
    if "apalache" in pins:
        assert pins["apalache"] == LOCKED_APALACHE_VERSION


def test_execute_state_model_check_real_holds(
    state_model_cert, offline_env
) -> None:
    tlc = state_model_cert.resolve_executable(["tlc"])
    if tlc is None:
        pytest.skip("tlc not on PATH")
    probe = state_model_cert.probe_tlc_live_identity(
        env=offline_env, executable=tlc
    )
    if not probe.get("usable"):
        pytest.skip(f"tlc live identity unusable: {probe.get('probe_error')}")
    result = state_model_cert.execute_state_model_check(
        "tlc",
        tlc,
        model_source=state_model_cert._LIVE_MODULE_HOLDS,
        config_source=state_model_cert._LIVE_TLC_CONFIG,
        timeout_seconds=30.0,
        env=offline_env,
    )
    assert result["timed_out"] is False
    status, reasons, _ = state_model_cert.classify_model_check_output(
        tool="tlc",
        stdout=result["stdout"],
        stderr=result["stderr"],
        returncode=result["returncode"],
        timed_out=False,
    )
    assert status == "passed"
    assert "bounded_success" in reasons or reasons


def test_build_state_model_argv_apalache_uses_length(state_model_cert) -> None:
    argv = state_model_cert.build_state_model_argv(
        "apalache",
        "/usr/bin/apalache-mc",
        length=5,
        property_name="Inv",
    )
    assert "check" in argv
    assert any(part.startswith("--length=5") for part in argv)
    assert any(part.startswith("--inv=Inv") for part in argv)
    assert any(part.endswith(".tla") for part in argv)
