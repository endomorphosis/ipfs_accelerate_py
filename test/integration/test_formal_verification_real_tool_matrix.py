"""Real multi-prover matrix certification (FVT-030 / FVT-G060).

Exercises ``tools/logic/certify_formal_verification_toolchains.py`` and the
``FormalVerificationToolchainCertificate@1`` receipt.

Acceptance covered:

* available tools pass live positive/negative/mutation/replay checks with
  exact identities;
* absent/mismatched lanes are explicit skips/unavailable and block only their
  promotion;
* PATH shims are not usability;
* certification performs no download/network/install and quarantines
  disagreement.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFIER_PATH = REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
CERTIFICATE_PATH = (
    REPO_ROOT / "docs" / "architecture" / "formal_verification_toolchain_certificate.json"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "FormalVerificationToolchainCertificate@1"
SCHEMA_VERSION = "formal-verification-toolchain-certificate/v1"
GOAL_ID = "FVT-G060"
TASK_ID = "FVT-030"

REQUIRED_LANES = {
    "smt",
    "tla",
    "datalog_secpal",
    "protocol",
    "hyperltl",
    "atp",
    "hammer",
    "kernel",
    "runtime_mtl",
    "attestation",
}

CHECK_KINDS = {"positive", "negative", "mutation", "replay"}


def _load_certifier():
    assert CERTIFIER_PATH.is_file(), f"missing certifier: {CERTIFIER_PATH}"
    spec = importlib.util.spec_from_file_location(
        "certify_formal_verification_toolchains",
        CERTIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load_certifier()


@pytest.fixture(scope="module")
def certificate(certifier) -> dict[str, Any]:
    return certifier.build_certificate(repo_root=REPO_ROOT)


# ---------------------------------------------------------------------------
# Artifact presence / schema
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert CERTIFIER_PATH.is_file()
    assert LOCK_PATH.is_file()
    assert CERTIFICATE_PATH.is_file(), (
        "checked-in certificate missing; run "
        "python tools/logic/certify_formal_verification_toolchains.py"
    )


def test_certificate_schema_and_identity(certificate: dict[str, Any]) -> None:
    assert certificate["schema_version"] == SCHEMA_VERSION
    assert certificate["interface"] == INTERFACE
    assert certificate["goal_id"] == GOAL_ID
    assert certificate["task_id"] == TASK_ID
    assert certificate["binding_mode"] == "offline_pinned_live_lanes"
    assert certificate["certificate_digest_sha256"]
    assert len(certificate["certificate_digest_sha256"]) == 64


def test_checked_in_certificate_matches_interface() -> None:
    payload = json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["interface"] == INTERFACE
    assert payload["goal_id"] == GOAL_ID
    assert payload["task_id"] == TASK_ID
    assert "tools" in payload and isinstance(payload["tools"], list)
    assert "property_lanes" in payload and isinstance(payload["property_lanes"], list)
    assert "certification_policy" in payload


def test_certification_policy_is_fail_closed(certificate: dict[str, Any]) -> None:
    policy = certificate["certification_policy"]
    for key in (
        "forbid_install",
        "forbid_download",
        "forbid_network",
        "forbid_curl_pipe_shell",
        "path_presence_is_not_usability",
        "require_exact_pin_match_for_production_certification",
        "shim_toolchain_mismatch_fails_closed",
        "absent_lanes_block_only_own_promotion",
        "optional_tools_not_mandatory_for_unrelated_properties",
        "quarantine_disagreement",
        "synthetic_evidence_cannot_certify_production",
    ):
        assert policy[key] is True, key

    lock_policy = policy["lock_offline_verification_policy"]
    for key in (
        "forbid_install",
        "forbid_download",
        "forbid_network",
        "path_presence_is_not_usability",
    ):
        assert lock_policy[key] is True, key


# ---------------------------------------------------------------------------
# Property lanes
# ---------------------------------------------------------------------------


def test_all_required_property_lanes_present(certificate: dict[str, Any]) -> None:
    lanes = {lane["lane_id"]: lane for lane in certificate["property_lanes"]}
    assert set(lanes) >= REQUIRED_LANES
    for lane_id, lane in lanes.items():
        assert lane["property_class"]
        assert isinstance(lane["tool_ids"], list) and lane["tool_ids"]
        assert isinstance(lane["authority_tool_ids"], list)
        assert lane["authority_tool_ids"]
        assert set(lane["authority_tool_ids"]) <= set(lane["tool_ids"])
        assert isinstance(lane["unavailable_tool_ids"], list)
        assert isinstance(lane["blocked_tool_ids"], list)
        assert isinstance(lane["certified_tool_ids"], list)
        assert isinstance(lane["certified_authority_tool_ids"], list)
        assert set(lane["certified_authority_tool_ids"]) <= set(
            lane["authority_tool_ids"]
        )
        assert lane["promotion_ready"] is bool(
            lane["certified_authority_tool_ids"]
        ) and not bool(lane["disagreement_quarantine_ids"])
        # Unavailable tools must only appear in their own blocked/unavailable sets.
        for tool_id in lane["unavailable_tool_ids"]:
            assert tool_id in lane["tool_ids"]
            assert tool_id in lane["blocked_tool_ids"] or tool_id not in lane[
                "certified_tool_ids"
            ]


def test_absent_tools_do_not_fail_unrelated_lanes(
    certificate: dict[str, Any],
) -> None:
    """Optional / missing tools only block their own promotion."""

    tools = {entry["tool_id"]: entry for entry in certificate["tools"]}
    lanes = {lane["lane_id"]: lane for lane in certificate["property_lanes"]}

    # Pick any unavailable tool and ensure other lanes without it can still
    # be promotion_ready when they have a certified member.
    unavailable = [
        tid for tid, entry in tools.items() if entry.get("unavailable")
    ]
    assert unavailable, "expected at least one unavailable optional tool on host"

    for lane_id, lane in lanes.items():
        foreign_unavailable = [
            tid for tid in unavailable if tid not in lane["tool_ids"]
        ]
        # Foreign unavailable tools must not appear in this lane's blocked set.
        for tid in foreign_unavailable:
            assert tid not in lane["blocked_tool_ids"]
            assert tid not in lane["unavailable_tool_ids"]

    # SMT lane must not be failed solely because ATP tools are missing.
    smt = lanes["smt"]
    atp_missing = [
        tid
        for tid in lanes["atp"]["tool_ids"]
        if tools[tid].get("unavailable")
    ]
    if atp_missing and smt["certified_tool_ids"]:
        assert smt["promotion_ready"] is True


def test_support_runtimes_cannot_promote_authority_lanes(certifier) -> None:
    ToolCertification = certifier.ToolCertification
    lanes = certifier.certify_property_lanes(
        {
            "java": ToolCertification(
                tool_id="java",
                production_certified=True,
                promotion_blocked=False,
            ),
            "maude": ToolCertification(
                tool_id="maude",
                production_certified=True,
                promotion_blocked=False,
            ),
        },
        (),
    )
    by_id = {lane.lane_id: lane for lane in lanes}

    assert by_id["tla"].certified_tool_ids == ["java"]
    assert by_id["tla"].certified_authority_tool_ids == []
    assert by_id["tla"].promotion_ready is False
    assert by_id["protocol"].certified_tool_ids == ["maude"]
    assert by_id["protocol"].certified_authority_tool_ids == []
    assert by_id["protocol"].promotion_ready is False


# ---------------------------------------------------------------------------
# Tool identity / PATH shim policy
# ---------------------------------------------------------------------------


def test_path_presence_is_not_usability_encoded(
    certifier, certificate: dict[str, Any]
) -> None:
    assert certifier.detect_lean_shim_toolchain_mismatch(
        "leanprover/lean4:v4.32.2",
        ["leanprover/lean4:v4.31.0"],
    )
    assert not certifier.detect_lean_shim_toolchain_mismatch(
        "leanprover/lean4:v4.31.0",
        ["leanprover/lean4:v4.31.0"],
    )

    # Every tool that is only path-present without identity must not be usable.
    for entry in certificate["tools"]:
        if entry.get("path_present") and not entry.get("identity_probed"):
            assert entry["usable"] is False
            assert entry["production_certified"] is False
            assert entry["promotion_blocked"] is True
            assert "path_presence_without_identity_probe" in entry["block_reasons"] or entry[
                "evidence_class"
            ] in {"path_shim", "unavailable"}


def test_lean_probe_selects_already_installed_locked_toolchain(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locked = "leanprover/lean4:v4.31.0"
    observed_envs: list[dict[str, str]] = []
    monkeypatch.setattr(
        certifier,
        "resolve_executable",
        lambda _candidates: "/fixture/lean",
    )
    monkeypatch.setattr(
        certifier,
        "list_elan_installed_toolchains",
        lambda _env=None: [locked, "leanprover/lean4:v4.32.2"],
    )

    def bounded_run(argv, *, timeout, env, **_kwargs):
        observed_envs.append(dict(env))
        version = "4.31.0" if env.get("ELAN_TOOLCHAIN") == locked else "4.32.2"
        return certifier.subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"Lean (version {version}, fixture)\n",
            stderr="",
        )

    monkeypatch.setattr(certifier, "bounded_run", bounded_run)
    result = certifier.probe_tool_identity(
        {
            "tool_id": "lean",
            "availability": "managed_pin",
            "executable_candidates": ["lean"],
            "offline_probe": {
                "argv": ["--version"],
                "locked_toolchain": locked,
            },
        },
        env=certifier.offline_env({"PATH": "/fixture"}),
    )

    assert observed_envs[0]["ELAN_TOOLCHAIN"] == locked
    assert observed_envs[0]["ELAN_NO_AUTO_INSTALL"] == "1"
    assert result["version_string"].startswith("Lean (version 4.31.0")
    assert result["selected_toolchain"] == locked
    assert result["shim_toolchain_mismatch"] is False


def test_symbolicai_probe_binds_distribution_to_symai_without_import(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []

    def distribution_version(name: str) -> str:
        observed.append(("distribution", name))
        return "1.14.0"

    def find_spec(name: str) -> object:
        observed.append(("module", name))
        return object()

    monkeypatch.setattr(
        certifier.importlib.metadata,
        "version",
        distribution_version,
    )
    monkeypatch.setattr(certifier.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(
        certifier,
        "bounded_run",
        lambda *_args, **_kwargs: pytest.fail(
            "SymbolicAI availability must not import symai"
        ),
    )

    available, identity = certifier._probe_in_process_module(
        "symbolicai",
        env=certifier.offline_env({}),
    )

    assert available is True
    assert identity == (
        "python-distribution:symbolicai==1.14.0;module:symai"
    )
    assert observed == [
        ("distribution", "symbolicai"),
        ("module", "symai"),
    ]


def test_version_mismatch_blocks_production_certification(certifier) -> None:
    assert certifier.detect_locked_version_mismatch(
        "1.3.3", "This is cvc5 version 1.2.0"
    )
    assert not certifier.detect_locked_version_mismatch(
        "1.3.3", "This is cvc5 version 1.3.3 [git]"
    )
    assert not certifier.detect_locked_version_mismatch(
        ">=4.12.0,<5.0.0", "Z3 version 4.16.0 - 64 bit"
    )
    assert certifier.detect_locked_version_mismatch(
        ">=4.12.0,<5.0.0", "Z3 version 3.1.0"
    )
    assert certifier.detect_locked_version_mismatch(
        "v4.31.0", "Lean (version 4.32.2"
    )


def test_every_tool_has_four_check_slots(certificate: dict[str, Any]) -> None:
    for entry in certificate["tools"]:
        kinds = {check["kind"] for check in entry["checks"]}
        assert kinds == CHECK_KINDS, entry["tool_id"]
        for check in entry["checks"]:
            assert check["status"] in {
                "passed",
                "failed",
                "skipped",
                "unavailable",
            }


# ---------------------------------------------------------------------------
# Live SMT matrix (available tools)
# ---------------------------------------------------------------------------


def test_available_smt_tools_pass_live_matrix(certificate: dict[str, Any]) -> None:
    tools = {entry["tool_id"]: entry for entry in certificate["tools"]}
    live_smt = []
    for tool_id in ("z3", "cvc5"):
        entry = tools[tool_id]
        if entry.get("unavailable"):
            # Explicit skip — must not be production_certified.
            assert entry["production_certified"] is False
            assert entry["promotion_blocked"] is True
            continue
        if not entry.get("usable"):
            # Version/shim mismatch: identity may exist but promotion blocked.
            assert entry["production_certified"] is False
            assert entry["promotion_blocked"] is True
            continue
        live_smt.append(tool_id)
        by_kind = {check["kind"]: check for check in entry["checks"]}
        for kind in CHECK_KINDS:
            assert by_kind[kind]["status"] == "passed", (
                f"{tool_id}.{kind}: {by_kind[kind]}"
            )
        assert entry["identity_probed"] is True
        assert entry["version_string"]
        assert entry["production_certified"] is True
        assert entry["promotion_blocked"] is False

    # On this program's audit hosts at least one SMT solver is expected; if
    # the hermetic environment truly has none, the lane must still be explicit.
    smt_lane = next(
        lane for lane in certificate["property_lanes"] if lane["lane_id"] == "smt"
    )
    if live_smt:
        assert smt_lane["promotion_ready"] is True
        assert set(live_smt) <= set(smt_lane["certified_tool_ids"])
    else:
        assert smt_lane["certified_tool_ids"] == []
        assert smt_lane["promotion_ready"] is False


def test_production_certified_implies_usable_and_pin_match(
    certificate: dict[str, Any],
) -> None:
    for entry in certificate["tools"]:
        if not entry["production_certified"]:
            continue
        assert entry["usable"] is True
        assert entry["installed"] is True
        assert entry["identity_probed"] is True
        assert entry["locked_version_mismatch"] is False
        assert entry["shim_toolchain_mismatch"] is False
        assert entry["unavailable"] is False
        assert entry["promotion_blocked"] is False
        assert all(check["status"] == "passed" for check in entry["checks"])


def test_identity_only_checks_never_claim_production_certification(
    certificate: dict[str, Any],
) -> None:
    for entry in certificate["tools"]:
        statuses = {check["status"] for check in entry["checks"]}
        if "skipped" in statuses:
            assert entry["production_certified"] is False


def test_bundled_in_process_lanes_are_usable_but_not_semantically_certified(
    certificate: dict[str, Any],
) -> None:
    tools = {entry["tool_id"]: entry for entry in certificate["tools"]}
    for tool_id in (
        "datalog-authorization",
        "secpal-authorization",
        "runtime-mtl",
    ):
        entry = tools[tool_id]
        assert entry["installed"] is True
        assert entry["usable"] is True
        assert entry["unavailable"] is False
        assert entry["production_certified"] is False
        assert "live_checks_incomplete_or_failed" in entry["block_reasons"]


# ---------------------------------------------------------------------------
# Disagreement quarantine
# ---------------------------------------------------------------------------


def test_disagreement_quarantine_blocks_promotion(certifier) -> None:
    """Synthetic sat vs unsat outcomes must quarantine and demote both tools."""

    CheckResult = certifier.CheckResult
    ToolCertification = certifier.ToolCertification

    z3 = ToolCertification(
        tool_id="z3",
        usable=True,
        production_certified=True,
        promotion_blocked=False,
        checks=[
            CheckResult("z3.positive", "positive", "passed", "unsat", "unsat"),
        ],
    )
    cvc5 = ToolCertification(
        tool_id="cvc5",
        usable=True,
        production_certified=True,
        promotion_blocked=False,
        checks=[
            # Disagreement: reports sat on the same positive obligation.
            CheckResult("cvc5.positive", "positive", "failed", "unsat", "sat"),
        ],
    )
    quarantine = certifier.quarantine_smt_disagreement(
        {"z3": z3, "cvc5": cvc5}
    )
    assert quarantine is not None
    assert quarantine.status == "quarantined"
    assert quarantine.reason == "cross_provider_disagreement"
    assert set(quarantine.promotion_blocked_tool_ids) == {"z3", "cvc5"}

    # Agreement does not quarantine.
    cvc5_agree = ToolCertification(
        tool_id="cvc5",
        usable=True,
        production_certified=True,
        checks=[
            CheckResult("cvc5.positive", "positive", "passed", "unsat", "unsat"),
        ],
    )
    assert (
        certifier.quarantine_smt_disagreement({"z3": z3, "cvc5": cvc5_agree})
        is None
    )


def test_live_certificate_quarantines_are_structured(
    certificate: dict[str, Any],
) -> None:
    for item in certificate["disagreement_quarantines"]:
        assert item["status"] == "quarantined"
        assert item["quarantine_id"]
        assert item["lane_id"]
        assert isinstance(item["outcomes"], dict)
        assert item["promotion_blocked_tool_ids"]
        # Quarantined tools must not remain production-certified.
        tools = {entry["tool_id"]: entry for entry in certificate["tools"]}
        for tool_id in item["promotion_blocked_tool_ids"]:
            assert tools[tool_id]["production_certified"] is False
            assert tools[tool_id]["promotion_blocked"] is True


# ---------------------------------------------------------------------------
# Offline / no install / no network
# ---------------------------------------------------------------------------


def test_certifier_env_blocks_install_and_network(certifier) -> None:
    env = certifier.offline_env(
        {
            "PATH": os.environ.get("PATH", ""),
            "ELAN_NO_AUTO_INSTALL": "0",
        }
    )
    assert env["PIP_NO_INDEX"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_INSTALL"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_NETWORK"] == "1"
    assert env["ELAN_NO_AUTO_INSTALL"] == "1"
    assert env.get("NPM_CONFIG_OFFLINE") == "true"


def test_lock_forbids_network_during_verification() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    policy = lock["offline_verification_policy"]
    assert policy["forbid_install"] is True
    assert policy["forbid_download"] is True
    assert policy["forbid_network"] is True
    for entry in lock["tools"]:
        assert entry["network_during_verification"] is False
        assert entry["install_during_verification"] is False
        assert entry["download_during_verification"] is False


def test_cli_main_writes_certificate(certifier, tmp_path: Path) -> None:
    output = tmp_path / "certificate.json"
    rc = certifier.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert rc == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["interface"] == INTERFACE
    assert payload["goal_id"] == GOAL_ID


def test_promotion_summary_consistent(certificate: dict[str, Any]) -> None:
    promotion = certificate["promotion"]
    tools = {entry["tool_id"]: entry for entry in certificate["tools"]}

    for tool_id in promotion["production_certified_tool_ids"]:
        assert tools[tool_id]["production_certified"] is True

    for tool_id in promotion["unavailable_tool_ids"]:
        assert tools[tool_id]["unavailable"] is True
        assert tool_id not in promotion["production_certified_tool_ids"]

    for tool_id, reasons in promotion["blocked_tool_ids"].items():
        assert tools[tool_id]["promotion_blocked"] is True
        assert isinstance(reasons, list)
        assert tools[tool_id]["production_certified"] is False
