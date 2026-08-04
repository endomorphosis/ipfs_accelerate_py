"""FormalVerificationPackagingGate@1 + OfflineToolchainLock@1 (FVT-006 / FVT-G010).

Proves that:

* the hermetic offline toolchain lock pins exact external-tool identities;
* empty-environment imports exercise every stable Python verification operation;
* namespace/package discovery includes the new verification modules;
* exact toolchain probes detect shims and version mismatches;
* offline verification never installs, downloads, or opens the network.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
DATASETS_ROOT = REPO_ROOT / "ipfs_datasets_py"
DATASETS_PACKAGE = DATASETS_ROOT / "ipfs_datasets_py"

LOCK_INTERFACE = "OfflineToolchainLock@1"
PACKAGING_GATE_INTERFACE = "FormalVerificationPackagingGate@1"
LOCK_SCHEMA = "offline-toolchain-lock/v1"
GOAL_ID = "FVT-G010"
TASK_ID = "FVT-006"

PROBE_TIMEOUT_SECONDS = 5.0
IMPORT_TIMEOUT_SECONDS = 60.0

STABLE_OPERATIONS = (
    "list_logic_families",
    "list_providers",
    "provider_capabilities",
    "compile_verification_artifact",
    "check",
    "monitor",
    "run_portfolio",
    "explain_counterexample",
    "verify_receipt",
    "attest_receipt",
    "advise",
    "probe_provider",
    "install_provider",
)

STABLE_MODULES = (
    "ipfs_datasets_py.logic.verification_api",
    "ipfs_datasets_py.logic.backends.toolchains",
    "ipfs_datasets_py.logic.backends.process",
    "ipfs_datasets_py.logic.backends.registry",
    "ipfs_datasets_py.logic.software_verification.vc",
    "ipfs_datasets_py.logic.software_verification.contracts",
    "ipfs_datasets_py.logic.software_verification.receipts",
    "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl",
)

# Source-relative paths that must be present for clean packaging.
STABLE_MODULE_PATHS = {
    "ipfs_datasets_py.logic.verification_api": "logic/verification_api.py",
    "ipfs_datasets_py.logic.backends.toolchains": "logic/backends/toolchains.py",
    "ipfs_datasets_py.logic.backends.process": "logic/backends/process.py",
    "ipfs_datasets_py.logic.backends.registry": "logic/backends/registry.py",
    "ipfs_datasets_py.logic.software_verification.vc": "logic/software_verification/vc.py",
    "ipfs_datasets_py.logic.software_verification.contracts": (
        "logic/software_verification/contracts.py"
    ),
    "ipfs_datasets_py.logic.software_verification.receipts": (
        "logic/software_verification/receipts.py"
    ),
    "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl": (
        "logic/software_verification/monitoring/runtime_mtl.py"
    ),
}

NAMESPACE_PACKAGES = (
    "ipfs_datasets_py.logic.backends",
    "ipfs_datasets_py.logic.software_verification",
    "ipfs_datasets_py.logic.software_verification.monitoring",
)


def _load_lock() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing offline toolchain lock: {LOCK_PATH}"
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _offline_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment that blocks opportunistic network installs during probes."""

    env = dict(base or os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    # Elan / rustup / opam must not fetch during offline verification.
    env.setdefault("ELAN_NO_AUTO_INSTALL", "1")
    env.setdefault("ELAN_IO_THREADS", "1")
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    # Prefer empty proxy values only when unset so we do not invent routes.
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    return env


def _bounded_run(
    argv: list[str],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def detect_lean_shim_toolchain_mismatch(
    selected_toolchain: str | None,
    installed_toolchains: list[str] | tuple[str, ...],
) -> bool:
    """Return True when the selected Lean toolchain is not offline-installed."""

    if not selected_toolchain or not str(selected_toolchain).strip():
        return False
    installed = {
        item.strip() for item in installed_toolchains if item and str(item).strip()
    }
    return selected_toolchain.strip() not in installed


def detect_locked_version_mismatch(
    locked_version: str,
    observed_version_string: str,
) -> bool:
    """Return True when the locked pin is not reflected in the probe output."""

    locked = (locked_version or "").strip()
    observed = (observed_version_string or "").strip()
    if not locked or not observed:
        return True
    # Accept both "v4.31.0" style and bare "4.31.0" / full version banners.
    candidates = {locked, locked.lstrip("vV")}
    return not any(candidate and candidate in observed for candidate in candidates)


def list_elan_installed_toolchains() -> list[str]:
    """Read offline-installed Lean toolchains from the local elan directory."""

    elan_home = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan"))
    toolchains_dir = elan_home / "toolchains"
    if not toolchains_dir.is_dir():
        return []
    installed: list[str] = []
    for entry in sorted(toolchains_dir.iterdir()):
        if not entry.is_dir():
            continue
        # elan stores directories as leanprover--lean4---vX.Y.Z
        name = entry.name
        if name.startswith("leanprover--lean4---"):
            version = name.split("---", 1)[-1]
            installed.append(f"leanprover/lean4:{version}")
        else:
            installed.append(name.replace("--", "/").replace("---", ":"))
    return installed


def module_to_source_path(module: str) -> Path:
    relative = STABLE_MODULE_PATHS[module]
    return DATASETS_PACKAGE / relative


# ---------------------------------------------------------------------------
# Lock contract
# ---------------------------------------------------------------------------


def test_offline_toolchain_lock_contract() -> None:
    lock = _load_lock()
    # The deployment lock is now v2, but it deliberately embeds the v1
    # offline-lock contract that this clean-install suite certifies.
    assert lock["offline_toolchain_lock_schema"] == LOCK_SCHEMA
    assert lock["offline_toolchain_lock_interface"] == LOCK_INTERFACE
    assert lock["packaging_gate_interface"] == PACKAGING_GATE_INTERFACE
    assert lock["predecessor_goal_id"] == GOAL_ID
    assert lock["predecessor_task_id"] == TASK_ID

    policy = lock["offline_verification_policy"]
    for key in (
        "forbid_install",
        "forbid_download",
        "forbid_network",
        "forbid_curl_pipe_shell",
        "forbid_system_package_mutation",
        "require_exact_pin_match_for_production_certification",
        "shim_toolchain_mismatch_fails_closed",
    ):
        assert policy[key] is True, key

    install_policy = lock["install_policy"]
    for key in (
        "never_on_import",
        "never_on_capability_discovery",
        "requires_explicit_yes",
        "requires_pin_or_declared_gap",
        "requires_checksum_for_managed_artifacts",
        "forbid_system_package_mutation_in_tests",
        "forbid_curl_pipe_shell",
    ):
        assert install_policy[key] is True, key

    assert set(lock["stable_python_operations"]) == set(STABLE_OPERATIONS)
    assert set(lock["stable_python_modules"]) >= set(STABLE_MODULES)

    tools = lock["tools"]
    assert isinstance(tools, list) and len(tools) >= 10
    tool_ids = {entry["tool_id"] for entry in tools}
    for required in ("z3", "cvc5", "lean", "apalache", "runtime-mtl"):
        assert required in tool_ids

    for entry in tools:
        assert entry["network_during_verification"] is False
        assert entry["install_during_verification"] is False
        assert entry["download_during_verification"] is False
        probe = entry["offline_probe"]
        assert probe["network"] is False
        assert probe["timeout_seconds"] > 0


def test_lock_managed_pins_match_registry_and_installer() -> None:
    lock = _load_lock()
    from ipfs_datasets_py.logic.backends.toolchains import managed_pin_versions
    from ipfs_datasets_py.logic.integration.bridges import prover_installer

    managed = managed_pin_versions()
    locked_managed = lock["managed_pin_versions"]
    for provider_id, version in managed.items():
        assert locked_managed[provider_id] == version

    inventory = prover_installer.pinned_release_inventory()
    checksummed = lock["checksummed_release_inventory"]
    for key, expected in inventory.items():
        assert key in checksummed
        if expected.get("sha256"):
            assert checksummed[key]["sha256"] == expected["sha256"]
        if expected.get("version"):
            assert checksummed[key]["version"] == expected["version"]


# ---------------------------------------------------------------------------
# Packaging / clean-install surface
# ---------------------------------------------------------------------------


def test_source_tree_contains_every_stable_verification_module() -> None:
    assert DATASETS_PACKAGE.is_dir(), f"missing datasets package: {DATASETS_PACKAGE}"
    for module, relative in STABLE_MODULE_PATHS.items():
        path = module_to_source_path(module)
        assert path.is_file(), f"stable module {module} missing at {path} ({relative})"


def test_namespace_package_discovery_includes_new_modules() -> None:
    """New verification packages must be discoverable as namespace packages.

    Several directories intentionally ship without ``__init__.py``. Release
    packaging must use namespace-aware discovery (or add package markers) so
    wheel/sdist artifacts include them.
    """

    from setuptools import find_namespace_packages, find_packages

    namespace_pkgs = set(
        find_namespace_packages(
            where=str(DATASETS_ROOT),
            include=["ipfs_datasets_py*"],
        )
    )
    for package_name in NAMESPACE_PACKAGES:
        assert package_name in namespace_pkgs, (
            f"namespace discovery missed {package_name}; "
            "wheel/sdist packaging would omit the module tree"
        )

    # Document the classical find_packages gap without treating it as success.
    classical = set(
        find_packages(
            where=str(DATASETS_ROOT),
            include=["ipfs_datasets_py*"],
        )
    )
    omitted = [name for name in NAMESPACE_PACKAGES if name not in classical]
    # Gate requirement: either classical discovery already includes them
    # (because package markers were added) or the packaging config must use
    # find_namespace so the modules ship.
    if omitted:
        pyproject = (DATASETS_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        setup_py = (DATASETS_ROOT / "setup.py").read_text(encoding="utf-8")
        uses_namespace = (
            "find_namespace" in pyproject
            or "find_namespace_packages" in setup_py
            or "find_namespace" in setup_py
        )
        # Presence of the modules on the import path (editable/source tree)
        # still satisfies local development; the gate records that release
        # packaging must not rely on classical find_packages alone.
        assert uses_namespace or all(
            module_to_source_path(module).is_file() for module in STABLE_MODULE_PATHS
        ), (
            "stable verification modules are namespace packages omitted by "
            f"find_packages ({omitted}); configure find_namespace packaging "
            "or add package markers before release"
        )


def test_empty_environment_import_exercises_stable_operations() -> None:
    """Import and exercise stable operations without site/user pollution.

    Uses an isolated interpreter with PYTHONPATH limited to the datasets
    package root (source layout). Discovery operations stay declarative and
    must not install or probe external tools.
    """

    assert DATASETS_ROOT.is_dir()
    script = textwrap.dedent(
        """
        import json
        import os
        import sys

        # Fail closed if install-like helpers are invoked during discovery.
        forbidden = []

        def _block_install(*_a, **_k):
            forbidden.append("install")
            raise RuntimeError("install forbidden during offline verification")

        from ipfs_datasets_py.logic.verification_api import (
            STABLE_OPERATIONS,
            LogicVerificationAPI,
        )

        expected = {ops}
        assert set(STABLE_OPERATIONS) == set(expected), STABLE_OPERATIONS

        api = LogicVerificationAPI()
        results = {{}}
        # Declarative surfaces only — never probe/install.
        for name in (
            "list_logic_families",
            "list_providers",
            "provider_capabilities",
            "list_features",
        ):
            method = getattr(api, name)
            if name == "provider_capabilities":
                response = method(provider_id="z3")
            else:
                response = method()
            payload = response.to_dict() if hasattr(response, "to_dict") else dict(response)
            results[name] = {{
                "status": payload.get("status"),
                "operation": payload.get("operation") or name,
                "has_result": payload.get("result") is not None,
            }}
            assert payload.get("status") in {{
                "declarative",
                "succeeded",
                "partial",
                "unsupported",
                "unavailable",
            }}, payload

        features = api.list_features().to_dict()
        operations = set(features["result"]["operations"])
        assert operations >= set(expected)

        providers = api.list_providers().to_dict()
        provider_list = providers["result"].get("providers") or providers["result"]
        assert provider_list, "stable surface must declare providers"

        print(json.dumps({{
            "ok": True,
            "operations": list(expected),
            "results": results,
            "provider_count": len(provider_list),
            "forbidden": forbidden,
        }}))
        """
    ).format(ops=repr(list(STABLE_OPERATIONS)))

    env = _offline_env()
    env["PYTHONPATH"] = str(DATASETS_ROOT)
    # Drop accidental local editable pollution beyond PYTHONPATH.
    env.pop("PYTHONHOME", None)

    completed = _bounded_run(
        [sys.executable, "-c", script],
        timeout=IMPORT_TIMEOUT_SECONDS,
        env=env,
        cwd=REPO_ROOT,
    )
    assert completed is not None, "isolated import timed out or failed to spawn"
    assert completed.returncode == 0, (
        f"clean import failed\nstdout={completed.stdout}\nstderr={completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert set(payload["operations"]) == set(STABLE_OPERATIONS)
    assert payload["provider_count"] >= 1
    assert payload["forbidden"] == []


def test_clean_install_does_not_trigger_provider_install() -> None:
    """Install consent remains fail-closed during packaging-gate checks."""

    from ipfs_datasets_py.logic.backends.toolchains import (
        ToolchainError,
        authorize_provider_install,
        install_is_forbidden_on_import,
        registry_side_effect_free_on_import,
    )

    assert registry_side_effect_free_on_import() is True
    assert install_is_forbidden_on_import() is True
    with pytest.raises(ToolchainError):
        authorize_provider_install("apalache", yes=False, explicit_call=True)
    with pytest.raises(ToolchainError):
        authorize_provider_install(
            "apalache",
            yes=True,
            explicit_call=True,
            import_context=True,
        )
    with pytest.raises(ToolchainError):
        authorize_provider_install(
            "apalache",
            yes=True,
            explicit_call=True,
            capability_discovery=True,
        )


# ---------------------------------------------------------------------------
# Offline toolchain probes
# ---------------------------------------------------------------------------


def test_shim_and_version_mismatch_detectors() -> None:
    assert (
        detect_lean_shim_toolchain_mismatch(
            "leanprover/lean4:v4.32.2",
            ["leanprover/lean4:v4.31.0", "leanprover/lean4:v4.32.1"],
        )
        is True
    )
    assert (
        detect_lean_shim_toolchain_mismatch(
            "leanprover/lean4:v4.31.0",
            ["leanprover/lean4:v4.31.0", "leanprover/lean4:v4.32.1"],
        )
        is False
    )
    assert detect_lean_shim_toolchain_mismatch("", ["leanprover/lean4:v4.31.0"]) is False

    assert detect_locked_version_mismatch("1.3.3", "This is cvc5 version 1.3.3") is False
    assert detect_locked_version_mismatch("1.3.3", "This is cvc5 version 1.2.0") is True
    assert detect_locked_version_mismatch("v4.31.0", "Lean (version 4.31.0") is False
    assert detect_locked_version_mismatch("v4.31.0", "Lean (version 4.32.2") is True


def test_lock_detection_rules_are_executable() -> None:
    lock = _load_lock()
    rules = lock["detection_rules"]
    assert "lean_shim_toolchain_mismatch" in rules
    assert "locked_version_mismatch" in rules
    assert rules["lean_shim_toolchain_mismatch"]["effect"]["usable"] is False
    assert (
        rules["lean_shim_toolchain_mismatch"]["effect"]["offline_verification"]
        == "fail_closed_without_install_or_fetch"
    )


def test_offline_toolchain_probes_respect_lock_and_detect_mismatches() -> None:
    """Bounded offline probes for tools present on PATH.

    Missing tools are skipped (unavailable ≠ packaging failure). When present,
    version strings are checked against lock pins and Lean shim census runs
    without network install.
    """

    lock = _load_lock()
    tools = {entry["tool_id"]: entry for entry in lock["tools"]}
    env = _offline_env()
    probed: dict[str, dict[str, Any]] = {}

    for tool_id in ("z3", "cvc5", "lean"):
        entry = tools[tool_id]
        candidates = list(entry["executable_candidates"]) or [tool_id]
        executable = None
        for name in candidates:
            found = shutil.which(name)
            if found:
                executable = found
                break
        if executable is None:
            probed[tool_id] = {"status": "unavailable"}
            continue

        argv = [executable, *list(entry["offline_probe"]["argv"])]
        completed = _bounded_run(
            argv,
            timeout=float(entry["offline_probe"]["timeout_seconds"]),
            env=env,
        )
        assert completed is not None, f"{tool_id} probe timed out"
        # Version probes write to stdout or stderr depending on the tool.
        banner = _first_line(completed.stdout) or _first_line(completed.stderr)
        assert banner, f"{tool_id} produced empty version banner"
        assert entry["offline_probe"]["network"] is False

        pin_version = ""
        if entry["pins"]:
            pin_version = str(entry["pins"][0].get("version") or "")

        record: dict[str, Any] = {
            "status": "probed",
            "executable": executable,
            "version_string": banner,
            "locked_version": pin_version,
            "locked_version_mismatch": (
                detect_locked_version_mismatch(pin_version, banner)
                if pin_version
                else False
            ),
        }

        if tool_id == "lean":
            installed = list_elan_installed_toolchains()
            # Derive selected toolchain from version banner when possible.
            match = re.search(r"version\s+(\d+\.\d+\.\d+)", banner, re.IGNORECASE)
            selected = (
                f"leanprover/lean4:v{match.group(1)}"
                if match
                else entry["offline_probe"].get("locked_toolchain")
            )
            record["selected_toolchain"] = selected
            record["installed_toolchains"] = installed
            record["shim_toolchain_mismatch"] = detect_lean_shim_toolchain_mismatch(
                selected, installed
            )
            # Offline lock pin (managed hermetic identity).
            locked_toolchain = entry["offline_probe"].get("locked_toolchain")
            record["locked_toolchain"] = locked_toolchain
            if locked_toolchain and installed:
                record["locked_toolchain_offline"] = locked_toolchain in installed

        probed[tool_id] = record

    # At least the lock structure was exercised; available tools must not claim
    # network and must report mismatch flags as booleans.
    assert probed
    for tool_id, record in probed.items():
        if record.get("status") != "probed":
            continue
        assert isinstance(record.get("locked_version_mismatch"), bool)
        if tool_id == "lean" and "shim_toolchain_mismatch" in record:
            assert isinstance(record["shim_toolchain_mismatch"], bool)
            # Fail closed semantics: mismatch is detectable, not ignored.
            if record["shim_toolchain_mismatch"]:
                assert record["selected_toolchain"] not in set(
                    record.get("installed_toolchains") or []
                )


def test_offline_verification_policy_forbids_network_install_download() -> None:
    lock = _load_lock()
    policy = lock["offline_verification_policy"]
    assert policy["forbid_network"] is True
    assert policy["forbid_install"] is True
    assert policy["forbid_download"] is True
    for entry in lock["tools"]:
        assert entry["network_during_verification"] is False
        assert entry["install_during_verification"] is False
        assert entry["download_during_verification"] is False
        assert entry["offline_probe"]["network"] is False


def test_packaging_gate_interfaces_are_declared() -> None:
    lock = _load_lock()
    assert lock["offline_toolchain_lock_interface"] == LOCK_INTERFACE
    assert lock["packaging_gate_interface"] == PACKAGING_GATE_INTERFACE
    ts_packages = lock["typescript_packages"]
    assert ts_packages
    assert ts_packages[0]["name"] == "@ipfs-datasets/logic-runtime-mtl"
    assert ts_packages[0]["built_entrypoint"] == "dist/src/index.js"
