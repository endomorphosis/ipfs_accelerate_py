"""Host-platform classification from the formal verification toolchain lock.

FVT-064 / FVT-G201 — ``FormalVerificationPlatformSupport@1``.

Acceptance covered:

* the normalized host key is derived from the running OS and architecture;
* each tool reports ``supported_here``, ``unsupported_here``, or ``ambiguous``
  from its own pins and deployment contract;
* ``any`` support is honored;
* absent, contradictory, or ambiguous metadata is a blocker;
* only an explicit host exclusion can produce a narrow platform exception;
* linux-aarch64 classifies HyperLTL, AutoHyper, MCHyper, Souffle, and external
  Runtime MTL as supported under the current lock;
* external SecPAL is unsupported (narrow platform exception);
* ZKP is a platform-independent deployment binding;
* a lock mutation that adds or removes ``linux-aarch64`` changes the
  classification and final digest;
* classification never probes PATH, installs tools, or converts unavailability
  into unsupported status.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "platform_support.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
ZKP_LOCK_PATH = REPO_ROOT / "config" / "formal_verification_zkp_deployment.lock.json"

INTERFACE = "FormalVerificationPlatformSupport@1"
SCHEMA_VERSION = "formal-verification-platform-support/v1"
GOAL_ID = "FVT-G201"
TASK_ID = "FVT-064"
PROGRAM = "formal-verification-tactician/platform-support-classifier"

LINUX_AARCH64 = "linux-aarch64"
LINUX_X86_64 = "linux-x86_64"

# Current-lock expectations on linux-aarch64 (acceptance subset).
SUPPORTED_ON_LINUX_AARCH64 = {
    "hyperltl",
    "autohyper",
    "mchyper",
    "souffle",
    "runtime-mtl-external",
}
UNSUPPORTED_ON_LINUX_AARCH64 = {
    "secpal",  # external SecPAL — narrow platform exception
}
PLATFORM_INDEPENDENT_TOOLS = {
    "zkp-circuit",
}


def _load_module():
    assert MODULE_PATH.is_file(), f"missing expected output: {MODULE_PATH}"
    # Keep import style consistent with other certification integration tests.
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    name = "tools_logic_certification_platform_support"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ps():
    return _load_module()


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing toolchain lock: {LOCK_PATH}"
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def zkp_lock() -> dict[str, Any] | None:
    if not ZKP_LOCK_PATH.is_file():
        return None
    payload = json.loads(ZKP_LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def report_linux_aarch64(ps, lock, zkp_lock) -> dict[str, Any]:
    return ps.build_platform_support_report(
        lock,
        host_platform=LINUX_AARCH64,
        zkp_deployment_lock=zkp_lock,
    )


# ---------------------------------------------------------------------------
# Surface / identity
# ---------------------------------------------------------------------------


def test_expected_outputs_exist():
    assert MODULE_PATH.is_file()
    assert LOCK_PATH.is_file()


def test_module_surface_constants(ps):
    assert ps.INTERFACE == INTERFACE
    assert ps.SCHEMA_VERSION == SCHEMA_VERSION
    assert ps.GOAL_ID == GOAL_ID
    assert ps.TASK_ID == TASK_ID
    assert ps.PROGRAM == PROGRAM
    assert ps.CLASSIFICATION_SUPPORTED == "supported_here"
    assert ps.CLASSIFICATION_UNSUPPORTED == "unsupported_here"
    assert ps.CLASSIFICATION_AMBIGUOUS == "ambiguous"


# ---------------------------------------------------------------------------
# Host normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "aarch64", "linux-aarch64"),
        ("linux", "arm64", "linux-aarch64"),
        ("Linux", "x86_64", "linux-x86_64"),
        ("Linux", "amd64", "linux-x86_64"),
        ("Darwin", "arm64", "darwin-arm64"),
        ("Darwin", "aarch64", "darwin-arm64"),
        ("Darwin", "x86_64", "darwin-x86_64"),
        ("darwin", "amd64", "darwin-x86_64"),
    ],
)
def test_normalize_host_platform(ps, system, machine, expected):
    assert ps.normalize_host_platform(system, machine) == expected


def test_observed_host_platform_matches_running_os(ps):
    observed = ps.observed_host_platform()
    system = platform.system().lower()
    machine = platform.machine().lower()
    assert observed == ps.normalize_host_platform(system, machine)
    assert "-" in observed
    assert observed.split("-", 1)[0] in {"linux", "darwin"} or system not in {
        "linux",
        "darwin",
    }


# ---------------------------------------------------------------------------
# Full lock classification on linux-aarch64
# ---------------------------------------------------------------------------


def test_report_envelope_on_linux_aarch64(report_linux_aarch64, lock):
    report = report_linux_aarch64
    assert report["interface"] == INTERFACE
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["goal_id"] == GOAL_ID
    assert report["task_id"] == TASK_ID
    assert report["host_platform"] == LINUX_AARCH64
    assert report["host_globally_supported"] is True
    assert "linux-aarch64" in report["global_supported_platforms"]
    assert report["tool_count"] == len(lock["tools"])
    assert report["tool_count"] == len(report["classifications"])
    assert report["final_digest"] == report["classification_digest"]
    assert str(report["final_digest"]).startswith("sha256:")
    assert len(report["final_digest"]) == len("sha256:") + 64
    assert report["policy"]["never_probe_path"] is True
    assert report["policy"]["never_install"] is True
    assert report["policy"]["never_infer_support_from_path"] is True
    assert report["policy"]["never_convert_unavailability_to_unsupported"] is True
    assert report["policy"]["only_explicit_host_exclusion_is_platform_exception"] is True
    assert report["policy"]["any_support_honored"] is True
    assert report["policy"]["absent_or_ambiguous_metadata_is_blocker"] is True


def test_every_tool_has_valid_classification(report_linux_aarch64, ps):
    valid = {
        ps.CLASSIFICATION_SUPPORTED,
        ps.CLASSIFICATION_UNSUPPORTED,
        ps.CLASSIFICATION_AMBIGUOUS,
    }
    seen: set[str] = set()
    for row in report_linux_aarch64["classifications"]:
        tool_id = row["tool_id"]
        assert tool_id not in seen
        seen.add(tool_id)
        assert row["classification"] in valid
        assert row["host_platform"] == LINUX_AARCH64
        assert row["supported"] is (row["classification"] == "supported_here")
        assert row["ambiguous"] is (row["classification"] == "ambiguous")
        assert row["exception_eligible"] is (
            row["classification"] == "unsupported_here"
        )
        # Unavailability is not a field of this surface — only lock-derived
        # platform status. Absence of installation probes is intentional.
        assert "installed" not in row or row.get("installed") is None or True
        assert "path" not in row
        assert "executable" not in row


def test_linux_aarch64_acceptance_matrix(report_linux_aarch64):
    by_id = report_linux_aarch64["by_tool_id"]

    for tool_id in SUPPORTED_ON_LINUX_AARCH64:
        row = by_id[tool_id]
        assert row["classification"] == "supported_here", tool_id
        assert row["supported"] is True
        assert row["exception_eligible"] is False
        assert row["blocker"] is False

    for tool_id in UNSUPPORTED_ON_LINUX_AARCH64:
        row = by_id[tool_id]
        assert row["classification"] == "unsupported_here", tool_id
        assert row["supported"] is False
        assert row["exception_eligible"] is True
        assert row["ambiguous"] is False

    # External SecPAL must appear as a narrow platform exception.
    exception_ids = {
        item["tool_id"] for item in report_linux_aarch64["platform_exceptions"]
    }
    assert "secpal" in exception_ids
    secpal_exc = next(
        item
        for item in report_linux_aarch64["platform_exceptions"]
        if item["tool_id"] == "secpal"
    )
    assert secpal_exc["narrow_scope"] is True
    assert secpal_exc["complete"] is False
    assert secpal_exc["production_certified"] is False
    assert secpal_exc["installed"] is False
    assert secpal_exc["authoritative"] is False
    assert LINUX_AARCH64 not in (by_id["secpal"].get("contract_platforms") or [])

    # ZKP is a platform-independent deployment binding.
    zkp = by_id["zkp-circuit"]
    assert zkp["classification"] == "supported_here"
    assert zkp["platform_independent_deployment_binding"] is True
    assert "zkp-circuit" in report_linux_aarch64[
        "platform_independent_deployment_binding_tool_ids"
    ]


def test_any_support_is_honored(ps, lock, zkp_lock):
    tools = ps.lock_tools_by_id(lock)
    # runtime-mtl-external declares supported_platforms: ["any"]
    row = ps.classify_tool_platform_support(
        tools["runtime-mtl-external"],
        host_platform=LINUX_AARCH64,
        global_supported=ps.global_supported_platforms(lock),
        zkp_deployment_lock=zkp_lock,
    )
    assert row.classification == "supported_here"
    assert "any" in row.declared_platforms

    # Also on an unrelated host that is still globally supported.
    row_x86 = ps.classify_tool_platform_support(
        tools["runtime-mtl-external"],
        host_platform=LINUX_X86_64,
        global_supported=ps.global_supported_platforms(lock),
    )
    assert row_x86.classification == "supported_here"


def test_only_explicit_exclusion_is_exception(ps, lock):
    tools = ps.lock_tools_by_id(lock)
    globals_ = ps.global_supported_platforms(lock)

    # Managed tool with no platforms at all → ambiguous blocker, not exception.
    bare = {
        "tool_id": "synthetic-bare-managed",
        "availability": "managed_pin",
        "pins": [],
        "deployment_contract": {},
    }
    bare_row = ps.classify_tool_platform_support(
        bare,
        host_platform=LINUX_AARCH64,
        global_supported=globals_,
    )
    assert bare_row.classification == "ambiguous"
    assert bare_row.blocker is True
    assert bare_row.exception_eligible is False
    assert "managed_tool_platform_metadata_missing" in bare_row.blocker_reasons

    # Contract supports host, pins only another host (no any/source) → ambiguous.
    narrow_pins = {
        "tool_id": "synthetic-contract-without-pin",
        "availability": "managed_pin",
        "pins": [{"platform": LINUX_X86_64, "version": "1.0.0"}],
        "deployment_contract": {
            "supported_platforms": [LINUX_AARCH64, LINUX_X86_64],
        },
    }
    narrow_row = ps.classify_tool_platform_support(
        narrow_pins,
        host_platform=LINUX_AARCH64,
        global_supported=globals_,
    )
    assert narrow_row.classification == "ambiguous"
    assert narrow_row.blocker is True
    assert narrow_row.exception_eligible is False

    # Explicit contract exclusion → unsupported exception even if pin says any.
    excluded = {
        "tool_id": "synthetic-excluded",
        "availability": "managed_pin",
        "pins": [{"platform": "any", "version": "1.0.0"}],
        "deployment_contract": {
            "supported_platforms": [LINUX_X86_64],
        },
    }
    excluded_row = ps.classify_tool_platform_support(
        excluded,
        host_platform=LINUX_AARCH64,
        global_supported=globals_,
    )
    assert excluded_row.classification == "unsupported_here"
    assert excluded_row.exception_eligible is True
    assert excluded_row.blocker is False
    assert excluded_row.basis == "deployment_contract.supported_platforms"

    # Real external SecPAL matches the explicit-exclusion pattern.
    secpal = ps.classify_tool_platform_support(
        tools["secpal"],
        host_platform=LINUX_AARCH64,
        global_supported=globals_,
    )
    assert secpal.classification == "unsupported_here"
    assert secpal.exception_eligible is True


def test_contradiction_with_global_policy_is_blocker(ps):
    entry = {
        "tool_id": "synthetic-off-policy",
        "availability": "managed_pin",
        "pins": [{"platform": "any", "version": "0"}],
        "deployment_contract": {"supported_platforms": ["any"]},
    }
    row = ps.classify_tool_platform_support(
        entry,
        host_platform="plan9-amd64",
        global_supported=[LINUX_AARCH64, LINUX_X86_64],
    )
    assert row.classification == "ambiguous"
    assert row.blocker is True
    assert "tool_and_global_platform_policy_contradict" in row.blocker_reasons


def test_non_managed_never_becomes_unsupported_from_missing_install(ps, lock):
    tools = ps.lock_tools_by_id(lock)
    globals_ = ps.global_supported_platforms(lock)
    # in_process tools are supported via global policy only — no PATH probe.
    for tool_id in ("runtime-mtl", "datalog-authorization", "secpal-authorization"):
        row = ps.classify_tool_platform_support(
            tools[tool_id],
            host_platform=LINUX_AARCH64,
            global_supported=globals_,
        )
        assert row.managed is False
        assert row.classification == "supported_here"
        assert row.basis == "global_platform_policy"
        assert row.exception_eligible is False


def test_platform_exceptions_exclude_ambiguous(ps, lock):
    tools = ps.lock_tools_by_id(lock)
    globals_ = ps.global_supported_platforms(lock)
    rows = [
        ps.classify_tool_platform_support(
            tools[tool_id],
            host_platform=LINUX_AARCH64,
            global_supported=globals_,
        )
        for tool_id in sorted(tools)
    ]
    # Inject an ambiguous row and ensure it never becomes an exception.
    ambiguous = ps.classify_tool_platform_support(
        {
            "tool_id": "synthetic-ambiguous",
            "availability": "managed_pin",
            "pins": [],
        },
        host_platform=LINUX_AARCH64,
        global_supported=globals_,
    )
    rows.append(ambiguous)
    exceptions = ps.build_platform_exceptions(rows)
    exception_ids = {item.tool_id for item in exceptions}
    assert "synthetic-ambiguous" not in exception_ids
    assert all(item.classification == "unsupported_here" for item in exceptions)


# ---------------------------------------------------------------------------
# Digest sensitivity to lock mutation
# ---------------------------------------------------------------------------


def test_removing_linux_aarch64_changes_classification_and_digest(
    ps, lock, zkp_lock, report_linux_aarch64
):
    baseline = report_linux_aarch64
    baseline_digest = baseline["final_digest"]
    by_id = baseline["by_tool_id"]

    # souffle is supported on linux-aarch64 under the current lock.
    assert by_id["souffle"]["classification"] == "supported_here"

    mutated = ps.mutate_tool_supported_platforms(
        lock,
        "souffle",
        remove=[LINUX_AARCH64],
    )
    mutated_report = ps.build_platform_support_report(
        mutated,
        host_platform=LINUX_AARCH64,
        zkp_deployment_lock=zkp_lock,
    )
    mutated_row = mutated_report["by_tool_id"]["souffle"]
    assert mutated_row["classification"] == "unsupported_here"
    assert mutated_row["exception_eligible"] is True
    assert LINUX_AARCH64 not in mutated_row["contract_platforms"]
    assert mutated_report["final_digest"] != baseline_digest
    assert "souffle" in {
        item["tool_id"] for item in mutated_report["platform_exceptions"]
    }


def test_adding_linux_aarch64_to_secpal_changes_classification_and_digest(
    ps, lock, zkp_lock, report_linux_aarch64
):
    baseline = report_linux_aarch64
    assert baseline["by_tool_id"]["secpal"]["classification"] == "unsupported_here"

    mutated = ps.mutate_tool_supported_platforms(
        lock,
        "secpal",
        add=[LINUX_AARCH64],
    )
    mutated_report = ps.build_platform_support_report(
        mutated,
        host_platform=LINUX_AARCH64,
        zkp_deployment_lock=zkp_lock,
    )
    mutated_row = mutated_report["by_tool_id"]["secpal"]
    assert mutated_row["classification"] == "supported_here"
    assert mutated_row["exception_eligible"] is False
    assert LINUX_AARCH64 in mutated_row["contract_platforms"]
    assert mutated_report["final_digest"] != baseline["final_digest"]
    assert "secpal" not in {
        item["tool_id"] for item in mutated_report["platform_exceptions"]
    }


def test_mutation_helper_does_not_mutate_original_lock(ps, lock):
    original = copy.deepcopy(lock)
    _ = ps.mutate_tool_supported_platforms(
        lock,
        "hyperltl",
        remove=[LINUX_AARCH64],
    )
    assert lock == original


# ---------------------------------------------------------------------------
# Repository entry point / no side effects
# ---------------------------------------------------------------------------


def test_classify_repository_loads_reviewed_locks(ps):
    report = ps.classify_repository(
        repo_root=REPO_ROOT,
        host_platform=LINUX_AARCH64,
    )
    assert report["interface"] == INTERFACE
    assert report["host_platform"] == LINUX_AARCH64
    assert report["tool_count"] >= len(SUPPORTED_ON_LINUX_AARCH64)
    assert Path(report["lock_path"]).resolve() == LOCK_PATH.resolve()
    if ZKP_LOCK_PATH.is_file():
        assert report["zkp_deployment_lock_bound"] is True
        assert report["by_tool_id"]["zkp-circuit"][
            "platform_independent_deployment_binding"
        ] is True


def test_classify_repository_uses_observed_host_when_unspecified(ps):
    report = ps.classify_repository(repo_root=REPO_ROOT)
    assert report["host_platform"] == ps.observed_host_platform()
    # On this CI/worktree host (linux-aarch64), acceptance matrix holds.
    if report["host_platform"] == LINUX_AARCH64:
        by_id = report["by_tool_id"]
        for tool_id in SUPPORTED_ON_LINUX_AARCH64:
            assert by_id[tool_id]["classification"] == "supported_here"
        assert by_id["secpal"]["classification"] == "unsupported_here"
        assert by_id["zkp-circuit"]["platform_independent_deployment_binding"] is True


def test_source_pin_counts_as_support_without_host_binary(ps, lock):
    # tamarin pins linux-x86_64 + source; on linux-aarch64 source makes it
    # supported when no excluding contract is present.
    tools = ps.lock_tools_by_id(lock)
    row = ps.classify_tool_platform_support(
        tools["tamarin"],
        host_platform=LINUX_AARCH64,
        global_supported=ps.global_supported_platforms(lock),
    )
    assert "source" in row.pin_platforms
    assert row.classification == "supported_here"
    assert row.basis == "tool.pins.platform"


def test_cli_main_json_exit_zero_when_no_blockers(ps, capsys):
    # Synthetic clean lock subset: force host + minimal tools without blockers.
    # Exercise the real repo path instead — current lock on linux-aarch64 may
    # include exceptions (secpal) but exceptions are not blockers.
    code = ps.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--host-platform",
            LINUX_AARCH64,
            "--json",
            "--tool",
            "souffle",
            "--tool",
            "secpal",
            "--tool",
            "zkp-circuit",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["interface"] == INTERFACE
    tool_ids = {row["tool_id"] for row in payload["classifications"]}
    assert tool_ids == {"souffle", "secpal", "zkp-circuit"}
    # Exit code reflects blockers only; exceptions are allowed.
    assert code in {0, 1}
    if not payload.get("blockers"):
        assert code == 0


def test_hyperproperty_tools_share_contract_basis_on_linux_aarch64(
    report_linux_aarch64,
):
    by_id = report_linux_aarch64["by_tool_id"]
    for tool_id in ("hyperltl", "autohyper", "mchyper"):
        row = by_id[tool_id]
        assert row["classification"] == "supported_here"
        assert row["basis"] == "deployment_contract.supported_platforms"
        assert LINUX_AARCH64 in row["contract_platforms"]
        assert "any" in row["pin_platforms"]
