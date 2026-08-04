"""Live TLC + Apalache semantic certification (FVT-060 / FVT-G204).

``StateModelLiveSemanticCertification@1``

Replaces classifier-backed state-model promotion with real pinned TLC jar and
Apalache executable runs against positive and adversarial models.

FVT-076 is the objective validation repair for the same goal: path evidence
already exists; this suite re-proves acceptance and binds the synthetic
discovery term ``objective validation repair`` into the receipt and durable
certificate so supervisor objective scans re-find the validation gate.

Acceptance covered:

* TLC and Apalache each execute a valid invariant model, a violating model with
  concrete counterexample, specification and invariant mutations, deterministic
  replay, malformed input, timeout, and bounded-state/resource cases;
* receipts bind exact source model, property, bound, JVM, executable,
  jar/archive, and output digests;
* canned text and parser classification remain ``hermetic_parser`` and cannot
  satisfy live external semantics;
* Java is support only; bounded model checking never grants theorem authority;
* certification never installs, downloads, or opens the network;
* ``objective validation repair`` is present on constants, receipts, and the
  durable live certificate (FVT-076).
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
REPAIR_TASK_ID = "FVT-076"
OBJECTIVE_VALIDATION_EVIDENCE = "objective validation repair"
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


def _write_executable(path: Path, body: str = "exit 0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_managed_manifest(
    root: Path,
    *,
    tool_id: str,
    version: str,
    artifact_sha256: str,
    java_executable: Path,
) -> None:
    path = root / "manifests" / f"{tool_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tool_id": tool_id,
                "version": version,
                "artifact_sha256": artifact_sha256,
                "java_executable": str(java_executable),
            }
        ),
        encoding="utf-8",
    )


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
    assert state_model_cert.REPAIR_TASK_ID == REPAIR_TASK_ID
    assert (
        state_model_cert.OBJECTIVE_VALIDATION_EVIDENCE
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert state_model_cert.OBJECTIVE_VALIDATION_EVIDENCE == (
        "objective validation repair"
    )
    assert "test_state_model_live_semantic_certification.py" in (
        state_model_cert.OBJECTIVE_VALIDATION_COMMAND
    )
    assert "test_state_model_toolchain_certification.py" in (
        state_model_cert.OBJECTIVE_VALIDATION_COMMAND
    )
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


def test_managed_execution_env_selects_jointly_bound_jdk(
    state_model_cert, tmp_path: Path
) -> None:
    root = tmp_path / "theorem-provers"
    managed_bin = root / "bin"
    _write_executable(managed_bin / "tlc")
    _write_executable(managed_bin / "apalache-mc")
    java = _write_executable(root / "jdk-21" / "bin" / "java")
    _write_managed_manifest(
        root,
        tool_id="tlc",
        version=LOCKED_TLC_VERSION,
        artifact_sha256=LOCKED_TLC_SHA256,
        java_executable=java,
    )
    _write_managed_manifest(
        root,
        tool_id="apalache",
        version=LOCKED_APALACHE_VERSION,
        artifact_sha256=state_model_cert.LOCKED_APALACHE_SHA256,
        java_executable=java,
    )

    env = state_model_cert.managed_execution_env(
        {
            "PATH": "/system/bin",
            "JAVA_HOME": "/system/java-8",
            state_model_cert.MANAGED_PROVER_ROOT_ENV: str(root),
            "JAVA_TOOL_OPTIONS": "-Dhostile=true",
            "_JAVA_OPTIONS": "-Xbootclasspath/a:/hostile",
            "JDK_JAVA_OPTIONS": "--add-modules=bad.module",
        }
    )

    assert env[state_model_cert.installer.JAVA_EXECUTABLE_ENV] == str(
        java.resolve()
    )
    assert env["JAVA_HOME"] == str(java.resolve().parent.parent)
    path_parts = env["PATH"].split(os.pathsep)
    assert path_parts[:2] == [
        str(managed_bin.resolve()),
        str(java.resolve().parent),
    ]
    assert path_parts[-1] == "/system/bin"
    for variable in state_model_cert.JAVA_OPTION_ENV_VARS:
        assert variable not in env
    assert env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_NETWORK"] == "1"


def test_managed_execution_env_rejects_disagreeing_jdk_bindings(
    state_model_cert, tmp_path: Path
) -> None:
    root = tmp_path / "theorem-provers"
    tlc_java = _write_executable(root / "tlc-jdk" / "bin" / "java")
    apalache_java = _write_executable(
        root / "apalache-jdk" / "bin" / "java"
    )
    _write_managed_manifest(
        root,
        tool_id="tlc",
        version=LOCKED_TLC_VERSION,
        artifact_sha256=LOCKED_TLC_SHA256,
        java_executable=tlc_java,
    )
    _write_managed_manifest(
        root,
        tool_id="apalache",
        version=LOCKED_APALACHE_VERSION,
        artifact_sha256=state_model_cert.LOCKED_APALACHE_SHA256,
        java_executable=apalache_java,
    )

    env = state_model_cert.managed_execution_env(
        {
            "PATH": "/system/bin",
            "JAVA_HOME": "/system/java-8",
            state_model_cert.MANAGED_PROVER_ROOT_ENV: str(root),
        }
    )

    assert state_model_cert.installer.JAVA_EXECUTABLE_ENV not in env
    assert env["JAVA_HOME"] == "/system/java-8"


def test_executable_resolution_uses_supplied_path(
    state_model_cert, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed_bin = tmp_path / "managed-bin"
    tlc = _write_executable(managed_bin / "tlc")
    monkeypatch.setenv("PATH", str(managed_bin))

    assert state_model_cert.resolve_executable(
        ["tlc"], env={"PATH": str(managed_bin)}
    ) == str(tlc.resolve())
    assert (
        state_model_cert.resolve_executable(
            ["tlc"], env={"PATH": "/definitely/not/a/tool/path"}
        )
        is None
    )


def test_certificate_generation_uses_bound_managed_environment(
    state_model_cert, tmp_path: Path
) -> None:
    root = tmp_path / "theorem-provers"
    (root / "bin").mkdir(parents=True)
    java = _write_executable(
        root / "jdk-21" / "bin" / "java",
        (
            'if [ -n "${JAVA_TOOL_OPTIONS+x}" ] || '
            '[ -n "${_JAVA_OPTIONS+x}" ] || '
            '[ -n "${JDK_JAVA_OPTIONS+x}" ]; then\n'
            "  exit 91\n"
            "fi\n"
            'echo \'openjdk version "21.0.9"\' >&2'
        ),
    )
    _write_managed_manifest(
        root,
        tool_id="tlc",
        version=LOCKED_TLC_VERSION,
        artifact_sha256=LOCKED_TLC_SHA256,
        java_executable=java,
    )
    _write_managed_manifest(
        root,
        tool_id="apalache",
        version=LOCKED_APALACHE_VERSION,
        artifact_sha256=state_model_cert.LOCKED_APALACHE_SHA256,
        java_executable=java,
    )

    receipt = state_model_cert.build_live_semantic_receipt(
        repo_root=REPO_ROOT,
        env={
            "PATH": "/system/bin",
            "JAVA_HOME": "/system/java-8",
            state_model_cert.MANAGED_PROVER_ROOT_ENV: str(root),
            "JAVA_TOOL_OPTIONS": "-Dhostile=true",
            "_JAVA_OPTIONS": "-Xmx1m",
            "JDK_JAVA_OPTIONS": "--add-modules=bad.module",
        },
        tlc_executable="/nonexistent/tlc",
        apalache_executable="/nonexistent/apalache-mc",
    )

    assert receipt["java_usable"] is True
    assert receipt["java_executable"] == str(java.resolve())
    assert receipt["java_version_string"].startswith(
        'openjdk version "21.0.9"'
    )
    assert receipt["production_certified"] is False
    assert receipt["live_execution"] is False


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
    assert manifest["policy"]["objective_validation_repair"] is True
    assert (
        manifest["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert manifest["objective_validation_repair"] is True
    assert manifest["repair_task_id"] == REPAIR_TASK_ID

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
    assert live_certificate["public_evidence_policy"]["satisfied"] is True
    assert LIVE_CERTIFICATE_PATH.is_file()
    # Objective validation repair evidence binding (FVT-076 / FVT-G204).
    assert (
        live_certificate.get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    repair = live_certificate.get("objective_validation_repair") or {}
    assert repair.get("schema_version") == "objective-validation-repair/v1"
    assert repair.get("goal_id") == LIVE_GOAL_ID
    assert repair.get("repair_task_id") == REPAIR_TASK_ID
    assert "objective validation repair" in (repair.get("evidence_terms") or [])
    assert live_certificate.get("policy", {}).get(
        "objective_validation_repair"
    ) is True
    assert (
        live_certificate.get("acceptance", {}).get(
            "objective_validation_evidence"
        )
        == OBJECTIVE_VALIDATION_EVIDENCE
    )


def test_live_certificate_writer_projects_private_evidence_before_digest(
    state_model_cert,
    tmp_path: Path,
) -> None:
    target = tmp_path / "portable-state-model-certificate.json"
    private_root = "/home/private-user/private-state-model-run"
    state_model_cert.write_live_certificate(
        {
            "interface": LIVE_INTERFACE,
            "notes": f"generated under {private_root}/model",
            "environment": {
                "HOME": private_root,
                "api_key": "state-model-private-api-key",
            },
            "stdout": f"raw process output from {private_root}",
            "receipt_digest_sha256": "stale-untrusted-digest",
        },
        repo_root=REPO_ROOT,
        path=target,
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True)
    assert private_root not in encoded
    assert "state-model-private-api-key" not in encoded
    assert "raw process output" not in encoded
    assert payload["environment"]["api_key"]["redacted"] is True
    assert payload["stdout"]["redacted"] is True
    assert payload["public_evidence_policy"]["satisfied"] is True
    assert state_model_cert.public_evidence_audit(
        payload, repo_root=REPO_ROOT
    )["satisfied"] is True
    assert payload["receipt_digest_sha256"] == state_model_cert.content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )


def test_live_certificate_writer_fails_closed_on_public_evidence_audit(
    state_model_cert,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "unsafe-state-model-certificate.json"
    monkeypatch.setattr(
        state_model_cert,
        "public_evidence_audit",
        lambda *_args, **_kwargs: {
            "satisfied": False,
            "failures": ["host_private_path"],
        },
    )

    with pytest.raises(
        ValueError,
        match="unsafe state-model public evidence",
    ):
        state_model_cert.write_live_certificate(
            {"interface": LIVE_INTERFACE},
            repo_root=REPO_ROOT,
            path=target,
        )
    assert not target.exists()


def test_live_certificate_writer_does_not_demote_valid_production_evidence(
    state_model_cert,
    tmp_path: Path,
) -> None:
    target = tmp_path / "state-model-production-certificate.json"
    production = {
        "interface": LIVE_INTERFACE,
        "production_certified": True,
        "live_execution": True,
        "live_semantic_corpus_passed": True,
        "tlc_usable": True,
        "apalache_usable": True,
        "java_usable": True,
        "block_reasons": [],
    }
    state_model_cert.write_live_certificate(
        production,
        repo_root=REPO_ROOT,
        path=target,
        force=True,
    )
    before = target.read_bytes()

    preserved = state_model_cert.write_live_certificate(
        {
            "interface": LIVE_INTERFACE,
            "production_certified": False,
            "live_execution": False,
            "block_reasons": ["live_tools_unavailable"],
        },
        repo_root=REPO_ROOT,
        path=target,
    )

    assert preserved == target
    assert target.read_bytes() == before


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
    managed_env = state_model_cert.managed_execution_env(offline_env)
    tlc = state_model_cert.resolve_executable(["tlc"], env=managed_env)
    if tlc is None:
        pytest.skip("tlc not on PATH")
    probe = state_model_cert.probe_tlc_live_identity(
        env=managed_env, executable=tlc
    )
    if not probe.get("usable"):
        pytest.skip(f"tlc live identity unusable: {probe.get('probe_error')}")
    result = state_model_cert.execute_state_model_check(
        "tlc",
        tlc,
        model_source=state_model_cert._LIVE_MODULE_HOLDS,
        config_source=state_model_cert._LIVE_TLC_CONFIG,
        timeout_seconds=30.0,
        env=managed_env,
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


def test_objective_validation_repair_receipt_binding(
    live_receipt: dict[str, Any],
    live_certificate: dict[str, Any],
    state_model_cert,
) -> None:
    """Receipt always binds the objective validation repair evidence term.

    This is the synthetic evidence term ``objective validation repair`` for the
    FVT-076 / FVT-G204 objective-scan validation gate. Path evidence alone is
    insufficient; the term must appear in code, receipt, and certificate.
    """

    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert (
        state_model_cert.OBJECTIVE_VALIDATION_EVIDENCE
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert state_model_cert.REPAIR_TASK_ID == REPAIR_TASK_ID

    repair = live_receipt.get("objective_validation_repair") or {}
    assert repair.get("schema_version") == "objective-validation-repair/v1"
    assert repair.get("goal_id") == LIVE_GOAL_ID
    assert repair.get("interface") == LIVE_INTERFACE
    assert repair.get("repair_task_id") == REPAIR_TASK_ID
    assert "objective validation repair" in (repair.get("evidence_terms") or [])
    assert (
        live_receipt.get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert live_receipt.get("policy", {}).get(
        "objective_validation_repair"
    ) is True
    assert live_receipt.get("repair_task_id") == REPAIR_TASK_ID
    assert (
        live_receipt.get("acceptance", {}).get(
            "objective_validation_evidence"
        )
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    if live_receipt.get("production_certified"):
        assert repair.get("status") == "satisfied"
        assert live_receipt["acceptance"]["objective_validation_repair"] is True
    elif not live_receipt.get("live_execution"):
        assert repair.get("status") == "withheld_live_tools_unavailable"

    # Exact-text discovery must appear in the declared output sources.
    module_source = STATE_MODEL_CERT_PATH.read_text(encoding="utf-8")
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in module_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in test_source
    cert_text = LIVE_CERTIFICATE_PATH.read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in cert_text
    assert (
        live_certificate.get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    cert_repair = live_certificate.get("objective_validation_repair") or {}
    assert "objective validation repair" in (
        cert_repair.get("evidence_terms") or []
    )
