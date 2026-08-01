"""FormalVerificationDeploymentLock@2 packaging gate (FVT-041 / FVT-G110).

Proves that declared external-tool gaps and incomplete managed pins are replaced
by reviewed deployment contracts with version/license/platform/source/checksum
or immutable package-lock identities and installer entries; ZKP uses a
secret-safe deployment-artifact schema; unsupported platforms fail closed;
installs are user-local and require explicit opt-in; imports, discovery, tests,
and offline certification never install, download, network, or mutate system
package managers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
DATASETS_ROOT = REPO_ROOT / "ipfs_datasets_py"
INSTALLERS_REGISTRY_PATH = (
    DATASETS_ROOT
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "installers"
    / "registry.py"
)

DEPLOYMENT_LOCK_INTERFACE = "FormalVerificationDeploymentLock@2"
DEPLOYMENT_LOCK_SCHEMA = "formal-verification-deployment-lock/v2"
INSTALLER_REGISTRY_INTERFACE = "FormalVerificationInstallerRegistry@1"
OFFLINE_LOCK_INTERFACE = "OfflineToolchainLock@1"
PACKAGING_GATE_INTERFACE = "FormalVerificationPackagingGate@1"
GOAL_ID = "FVT-G110"
TASK_ID = "FVT-041"

# Acceptance tools that must carry reviewed identities + installer entries.
REVIEWED_ACCEPTANCE_TOOLS = (
    "tlc",
    "hyperltl",
    "autohyper",
    "mchyper",
    "souffle",
    "secpal",
    "runtime-mtl-external",
    "vampire",
    "lean",
    "coq",
    "isabelle",
    "opam",
    "symbolicai",
    "ergoai",
    "zkp-circuit",
)

# Former InstallGapKind values that the deployment lock must replace.
REPLACED_GAP_IDS = (
    "tlc",
    "hyper_tools",
    "datalog_secpal_external",
    "runtime_mtl_external",
    "circuit_witness",
)

# Identity kinds that satisfy "checksum or immutable package-lock identity".
IMMUTABLE_IDENTITY_KINDS = frozenset(
    {
        "release_archive",
        "immutable_git_commit",
        "immutable_release_tag",
        "immutable_source_tag",
        "immutable_toolchain_identity",
        "opam_package",
        "python_package",
        "typescript_package",
        "operator_bound_artifact",
        "deployment_artifact_schema",
        "in_process",
        "host_runtime",
    }
)


def _ensure_datasets_on_path() -> None:
    root = str(DATASETS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_lock() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing deployment lock: {LOCK_PATH}"
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _tools_by_id(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tools = lock["tools"]
    assert isinstance(tools, list)
    by_id: dict[str, dict[str, Any]] = {}
    for entry in tools:
        assert isinstance(entry, dict)
        tool_id = entry["tool_id"]
        assert tool_id not in by_id
        by_id[tool_id] = entry
    return by_id


def _import_installer_registry():
    _ensure_datasets_on_path()
    from ipfs_datasets_py.logic.backends.installers import registry as installer_registry

    return installer_registry


# ---------------------------------------------------------------------------
# Lock interface and gap replacement
# ---------------------------------------------------------------------------


def test_deployment_lock_interface_and_schema_for_install_gap_closure() -> None:
    lock = _load_lock()
    assert lock["interface"] == DEPLOYMENT_LOCK_INTERFACE
    assert lock["schema_version"] == DEPLOYMENT_LOCK_SCHEMA
    assert lock["packaging_gate_interface"] == PACKAGING_GATE_INTERFACE
    assert lock["offline_toolchain_lock_interface"] == OFFLINE_LOCK_INTERFACE
    assert lock["installer_registry_interface"] == INSTALLER_REGISTRY_INTERFACE
    assert lock["goal_id"] == GOAL_ID
    assert lock["task_id"] == TASK_ID
    assert lock["binding_mode"] == "reviewed_deployment_lock"


def test_replaced_install_gaps_are_closed_in_lock_metadata() -> None:
    lock = _load_lock()
    replaced = lock["replaced_install_gaps"]
    assert isinstance(replaced, dict)
    for gap_id in REPLACED_GAP_IDS:
        assert gap_id in replaced, gap_id
        assert "tool_ids" in replaced[gap_id]
        assert str(replaced[gap_id]["status"]).startswith("replaced_")

    by_id = _tools_by_id(lock)
    # No tool entry may still carry a live gap_id after deployment-lock review.
    for tool_id, entry in by_id.items():
        assert entry.get("gap_id") in (None, ""), (
            f"{tool_id} still declares gap_id={entry.get('gap_id')!r}"
        )


def test_acceptance_tools_have_reviewed_pin_license_source_and_installer() -> None:
    by_id = _tools_by_id(_load_lock())
    for tool_id in REVIEWED_ACCEPTANCE_TOOLS:
        entry = by_id[tool_id]
        assert entry["license"], f"{tool_id} missing license"
        assert entry["source"], f"{tool_id} missing source"
        assert entry["installer_entry"], f"{tool_id} missing installer_entry"
        assert entry["installer_entry"].startswith("ensure_"), tool_id
        assert entry["identity_kind"] in IMMUTABLE_IDENTITY_KINDS, tool_id
        assert entry["install_scope"] == "user_local"
        assert entry["requires_explicit_opt_in"] is True
        assert entry["network_during_verification"] is False
        assert entry["install_during_verification"] is False
        assert entry["download_during_verification"] is False

        pins = entry["pins"]
        assert isinstance(pins, list) and pins, f"{tool_id} missing pins"
        for pin in pins:
            assert pin["version"], f"{tool_id} pin missing version"
            assert pin["platform"], f"{tool_id} pin missing platform"
            # Checksum OR immutable identity kind satisfies acceptance.
            if pin.get("is_checksummed"):
                assert pin.get("sha256") and len(pin["sha256"]) == 64
                assert pin.get("artifact_url")
            else:
                assert entry["identity_kind"] in IMMUTABLE_IDENTITY_KINDS


def test_managed_pins_include_platform_or_checksum_inventory() -> None:
    lock = _load_lock()
    inventory = lock["checksummed_release_inventory"]
    managed = lock["managed_pin_versions"]
    # Incomplete pins from OfflineToolchainLock@1 must now carry digests or
    # immutable identities for the high-priority tools.
    for tool_id in ("cvc5", "vampire", "isabelle", "opam"):
        assert tool_id in managed
        assert tool_id in inventory
        item = inventory[tool_id]
        assert item["version"] == managed[tool_id] or managed[tool_id].startswith(
            item["version"]
        ) or item["version"] == managed[tool_id]
        if item.get("sha256"):
            assert len(item["sha256"]) == 64
            assert item.get("url")
        else:
            assert item.get("identity_kind") or item.get("platforms")

    # Platform matrices for multi-arch binaries.
    for tool_id in ("cvc5", "vampire", "isabelle", "opam"):
        platforms = inventory[tool_id].get("platforms") or {}
        if platforms:
            for platform, meta in platforms.items():
                assert platform
                assert meta.get("url")
                assert meta.get("sha256") and len(meta["sha256"]) == 64


def test_zkp_secret_safe_deployment_artifact_schema_forbids_private_pins() -> None:
    lock = _load_lock()
    zkp = lock["zkp_deployment_artifact_schema"]
    assert zkp["interface"] == "ZkpDeploymentArtifactSchema@1"
    assert zkp["secret_safe"] is True
    assert zkp["forbid_private_witness_in_lock"] is True
    assert zkp["forbid_proving_key_bytes_in_lock"] is True
    assert zkp["forbid_trapdoor_material_in_lock"] is True
    forbidden = set(zkp["forbidden_fields"])
    for name in ("private_witness", "proving_key", "trapdoor", "secret"):
        assert name in forbidden
    allowed = set(zkp["allowed_fields"])
    for name in ("circuit_id", "circuit_public_digest", "sha256", "artifact_uri"):
        assert name in allowed

    zkp_tool = _tools_by_id(lock)["zkp-circuit"]
    assert zkp_tool["identity_kind"] == "deployment_artifact_schema"
    contract = zkp_tool["deployment_contract"]
    assert contract["secret_safe"] is True
    assert contract["forbid_private_witness_in_lock"] is True
    assert contract["replaces_gap_id"] == "circuit_witness"
    # Lock text must not embed private witness material.
    raw = LOCK_PATH.read_text(encoding="utf-8").lower()
    for banned in ("private_witness=", "proving_key_pem", "trapdoor_bytes"):
        assert banned not in raw


# ---------------------------------------------------------------------------
# Install policy, platform fail-closed, offline guarantees
# ---------------------------------------------------------------------------


def test_install_policy_is_user_local_explicit_and_offline_safe() -> None:
    lock = _load_lock()
    install_policy = lock["install_policy"]
    for key in (
        "never_on_import",
        "never_on_capability_discovery",
        "requires_explicit_yes",
        "requires_pin_or_declared_gap",
        "requires_checksum_for_managed_artifacts",
        "forbid_system_package_mutation_in_tests",
        "forbid_curl_pipe_shell",
        "user_local_only",
    ):
        assert install_policy[key] is True, key
    assert "ipfs_datasets_py" in install_policy["install_root"]
    assert install_policy["install_root"].startswith("~")

    offline = lock["offline_verification_policy"]
    for key in (
        "forbid_install",
        "forbid_download",
        "forbid_network",
        "forbid_curl_pipe_shell",
        "forbid_system_package_mutation",
        "require_exact_pin_match_for_production_certification",
        "shim_toolchain_mismatch_fails_closed",
    ):
        assert offline[key] is True, key


def test_unsupported_platform_policy_fails_closed_on_install() -> None:
    lock = _load_lock()
    policy = lock["platform_policy"]
    assert policy["unsupported_platforms_fail_closed"] is True
    supported = set(policy["supported_platforms"])
    assert "linux-x86_64" in supported
    assert "windows-x86_64" not in supported
    assert "refuse" in policy["unsupported_effect"]

    rules = lock["detection_rules"]
    assert "unsupported_platform" in rules
    assert rules["unsupported_platform"]["effect"]["install"] == "refuse_explicitly"


# ---------------------------------------------------------------------------
# Installer registry alignment
# ---------------------------------------------------------------------------


def test_installer_registry_module_exists_and_is_side_effect_free() -> None:
    assert INSTALLERS_REGISTRY_PATH.is_file(), INSTALLERS_REGISTRY_PATH
    registry = _import_installer_registry()
    assert registry.FORMAL_VERIFICATION_INSTALLER_REGISTRY_INTERFACE == (
        INSTALLER_REGISTRY_INTERFACE
    )
    assert registry.FORMAL_VERIFICATION_DEPLOYMENT_LOCK_INTERFACE == (
        DEPLOYMENT_LOCK_INTERFACE
    )
    assert registry.GOAL_ID == GOAL_ID
    assert registry.TASK_ID == TASK_ID
    assert registry.registry_side_effect_free_on_import() is True
    assert registry.install_is_forbidden_on_import() is True
    assert registry.network_forbidden_during_offline_certification() is True
    assert registry.system_package_mutation_forbidden_in_tests() is True


def test_installer_registry_covers_acceptance_tools_and_replaced_gaps() -> None:
    registry = _import_installer_registry()
    registry.reset_default_installer_registry()
    reg = registry.default_installer_registry()
    registry.assert_acceptance_tools_have_installer_entries(reg)

    tool_ids = set(reg.list_tool_ids())
    for tool_id in REVIEWED_ACCEPTANCE_TOOLS:
        assert tool_id in tool_ids
        entry = reg.get(tool_id)
        assert entry.ensure_name.startswith("ensure_")
        assert entry.license
        assert entry.source
        assert entry.user_local_only is True
        assert entry.requires_explicit_yes is True
        assert entry.never_on_import is True
        assert "installers." in entry.module_path

    replaced = {entry.replaces_gap_id for entry in reg.entries_replacing_gaps()}
    for gap_id in REPLACED_GAP_IDS:
        assert gap_id in replaced, gap_id


def test_installer_registry_aligned_with_deployment_lock_pins() -> None:
    registry = _import_installer_registry()
    registry.clear_deployment_lock_cache()
    registry.reset_default_installer_registry()
    lock = registry.load_deployment_lock(REPO_ROOT)
    registry.assert_deployment_lock_contract(lock)
    registry.assert_registry_aligned_with_lock(lock, registry=registry.default_installer_registry())

    # Cross-check lock installer_entry strings.
    by_id = _tools_by_id(dict(lock))
    for entry in registry.list_installer_entries():
        assert by_id[entry.tool_id]["installer_entry"] == entry.ensure_name


def test_install_authorization_requires_yes_pin_checksum_and_supported_platform() -> None:
    registry = _import_installer_registry()
    registry.reset_default_installer_registry()

    with pytest.raises(registry.InstallerRegistryError, match="yes=True"):
        registry.authorize_installer_entry_install("vampire", yes=False)

    with pytest.raises(registry.InstallerRegistryError, match="explicit"):
        registry.authorize_installer_entry_install(
            "vampire", yes=True, explicit_call=False
        )

    with pytest.raises(registry.InstallerRegistryError, match="import"):
        registry.authorize_installer_entry_install(
            "vampire", yes=True, import_context=True
        )

    with pytest.raises(registry.InstallerRegistryError, match="capability discovery"):
        registry.authorize_installer_entry_install(
            "vampire", yes=True, capability_discovery=True
        )

    with pytest.raises(registry.InstallerRegistryError, match="checksum"):
        registry.authorize_installer_entry_install(
            "vampire", yes=True, checksum_verified=False
        )

    with pytest.raises(registry.InstallerRegistryError, match="system package"):
        registry.authorize_installer_entry_install(
            "vampire",
            yes=True,
            system_package_mutation=True,
            test_mode=True,
        )

    with pytest.raises(registry.InstallerRegistryError, match="unsupported platform"):
        registry.authorize_installer_entry_install(
            "vampire", yes=True, platform="windows-x86_64"
        )

    # Happy path: authorized metadata only — no install side effect.
    entry = registry.authorize_installer_entry_install(
        "vampire",
        yes=True,
        checksum_verified=True,
        platform="linux-x86_64",
    )
    assert entry.tool_id == "vampire"
    assert entry.ensure_name == "ensure_vampire"


def test_gap_replacement_tools_have_deployment_contracts_not_live_gaps() -> None:
    by_id = _tools_by_id(_load_lock())
    for tool_id, expected_gap in (
        ("tlc", "tlc"),
        ("hyperltl", "hyper_tools"),
        ("autohyper", "hyper_tools"),
        ("mchyper", "hyper_tools"),
        ("souffle", "datalog_secpal_external"),
        ("secpal", "datalog_secpal_external"),
        ("runtime-mtl-external", "runtime_mtl_external"),
        ("zkp-circuit", "circuit_witness"),
    ):
        entry = by_id[tool_id]
        assert entry.get("gap_id") in (None, "")
        contract = entry["deployment_contract"]
        assert contract["status"] == "reviewed"
        assert contract["replaces_gap_id"] == expected_gap
        assert contract["unsupported_platforms_fail_closed"] is True
        assert entry["installer_entry"]


def test_runtime_mtl_vendor_launcher_matches_locked_path_candidate() -> None:
    _ensure_datasets_on_path()
    from ipfs_datasets_py.logic.backends.installers import runtime_mtl

    entry = _tools_by_id(_load_lock())["runtime-mtl-external"]
    assert runtime_mtl.MANAGED_EXECUTABLE_NAME in entry["executable_candidates"]
    metadata = runtime_mtl.describe_runtime_mtl_installer()
    assert (
        metadata["vendor"]["managed_executable_name"]
        == runtime_mtl.MANAGED_EXECUTABLE_NAME
    )
    assert (
        metadata["policy"]["vendor_launcher_is_digest_bound_and_path_visible"]
        is True
    )


def test_lock_stable_modules_include_installer_registry_path() -> None:
    lock = _load_lock()
    modules = set(lock["stable_python_modules"])
    assert "ipfs_datasets_py.logic.backends.installers.registry" in modules
    assert INSTALLERS_REGISTRY_PATH.is_file()
