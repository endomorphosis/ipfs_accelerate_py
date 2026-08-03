"""ErgoAI live toolchain contract tests (FVT-085 / FVT-G218).

``ErgoAILiveToolchainContract@1``

Proves the genuine ErgoAI advisor-toolchain path:

* the lock binds the official ErgoAI 3.0 distribution (release tag, digests,
  license/acquisition conditions, XSB and build/runtime dependencies,
  supported OS/arch matrix, entry point, and identity probe);
* explicit lazy installation is staged, checksum-verified, atomic, relocatable,
  and offline after acquisition (never on import / never during certification);
* the bounded live semantic adapter covers entailment, non-entailment,
  contradiction, rule/query mutation, deterministic replay, malformed input,
  timeout, and resource-bound cases;
* results remain proposal/candidate evidence under an advisory authority
  ceiling and never elevate to theorem authority;
* hermetic shims and simulation-mode wrapper fixtures are not live vendor
  execution, and certification never installs, downloads, or opens the network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "installers"
    / "advisors.py"
)
WRAPPER_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "flogic"
    / "ergoai_wrapper.py"
)
ADVISORS_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "advisors.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "ErgoAILiveToolchainContract@1"
SCHEMA_VERSION = "ergoai-live-toolchain-contract/v1"
GOAL_ID = "FVT-G218"
TASK_ID = "FVT-085"
LOCKED_ERGOAI_VERSION = "3.0"
LOCKED_RELEASE_TAG = "v3.0_release"
LOCKED_SHA256 = (
    "46f9747db118567a7da50f70b439e35ee36ea02c3dfde971a57c77a8ce94aa01"
)
LOCKED_SIZE = 53_064_767

REQUIRED_CASE_KINDS = {
    "entailment",
    "non_entailment",
    "contradiction",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "resource_bound",
}

REQUIRED_LOCK_ACQUISITION_KEYS = {
    "requires_explicit_opt_in",
    "checksum_required_before_execute",
    "download_during_certification_forbidden",
    "user_local_only",
    "offline_after_acquisition",
    "atomic_staged_install",
    "relocatable_install_root",
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


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _full_matrix_fixture_script() -> str:
    """Deterministic fake ErgoAI that implements the full live case matrix."""

    return textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from pathlib import Path
        import re
        import sys
        import time

        if len(sys.argv) > 1 and sys.argv[1] in {"--version", "-v", "version"}:
            print("ErgoAI 3.0 (managed linux-aarch64; v3.0_release)")
            raise SystemExit(0)

        data = sys.stdin.read()
        match = re.search(r"load\\{'([^']+)'\\}", data)
        program = Path(match.group(1)).read_text(encoding="utf-8") if match else ""

        if "fvt_loop" in program or "fvt_loop" in data:
            while True:
                time.sleep(1.0)

        if "fvt_ergo_resource_bound" in data or "fvt_ergo_resource_marker" in program:
            sys.stdout.write("X" * 4096 + "\\n")
            print("Yes")
            raise SystemExit(0)

        if "not %% valid" in program or "{{" in program:
            print("++Error")
            print("syntax error: malformed ergo input")
            raise SystemExit(1)

        if "fvt_ergo_contradiction" in data or "fvt_ergo_contradiction" in program:
            print("No")
            raise SystemExit(0)

        if "fvt_ergo_absent" in data or "fvt_ergo_mutated" in program:
            print("Yes")
            print("No")
            raise SystemExit(0)

        print("Yes")
        print("Yes")
        """
    )


def _materialize_live_fixture(root: Path, installer) -> Path:
    executable = root / "bin" / "ergoai"
    _write_executable(executable, _full_matrix_fixture_script())

    xsb = (
        root
        / "advisors"
        / "ergoai"
        / "3.0"
        / "vendor"
        / "XSB"
        / "config"
        / "aarch64-unknown-linux-gnu"
        / "bin"
        / "xsb"
    )
    xsb.parent.mkdir(parents=True, exist_ok=True)
    xsb.write_bytes(b"fixture-xsb-aarch64")
    xsb.chmod(0o755)

    release = root / "downloads" / "ergoAI_3.0.run"
    release.parent.mkdir(parents=True, exist_ok=True)
    release.write_bytes(b"fixture-official-release-for-contract")
    release_digest = hashlib.sha256(release.read_bytes()).hexdigest()

    identity_path = root / "advisors" / "ergoai" / "3.0" / "identity.json"
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "ergoai-managed-vendor-identity/v1",
        "tool_id": "ergoai",
        "version": "3.0",
        "selected_platform": "linux-aarch64",
        "release_tag": LOCKED_RELEASE_TAG,
        "release_url": installer.ERGOAI_RELEASE_URL,
        "release_artifact_path": str(release),
        "release_artifact_sha256": release_digest,
        "release_artifact_size_bytes": release.stat().st_size,
        "vendor_executable": str(executable),
        "vendor_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "xsb_executable": str(xsb),
        "xsb_executable_sha256": hashlib.sha256(xsb.read_bytes()).hexdigest(),
        "xsb_configuration": "aarch64-unknown-linux-gnu",
        "launcher": str(executable),
        "launcher_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "identity_digest_sha256": "fixture-identity",
        "license_components": ["Apache-2.0", "LGPL-2.0"],
        "checksum_verified": True,
        "is_live_vendor": True,
        "is_hermetic_advisor_shim": False,
        "grants_proof_authority": False,
    }
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    return executable


@pytest.fixture(scope="module")
def installer():
    assert INSTALLER_PATH.is_file(), f"missing expected output: {INSTALLER_PATH}"
    _ensure_import_paths()
    from ipfs_datasets_py.logic.backends.installers import advisors as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def advisors_cert():
    return _load_module(ADVISORS_CERT_PATH, "tools_logic_certification_advisors_live")


@pytest.fixture(scope="module")
def wrapper_mod():
    assert WRAPPER_PATH.is_file(), f"missing expected output: {WRAPPER_PATH}"
    _ensure_import_paths()
    from ipfs_datasets_py.logic.flogic import ergoai_wrapper as wrapper

    return wrapper


@pytest.fixture(scope="module")
def lock_document() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing expected output: {LOCK_PATH}"
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def structural_contract(advisors_cert) -> dict[str, Any]:
    return advisors_cert.build_ergoai_live_toolchain_contract(
        repo_root=REPO_ROOT,
        env=advisors_cert.offline_env(os.environ),
        run_semantics=False,
    )


@pytest.fixture
def live_fixture(installer, advisors_cert, tmp_path, monkeypatch):
    root = tmp_path / "ergoai-live-fixture"
    executable = _materialize_live_fixture(root, installer)
    # Align provenance validators with the fixture release bytes.
    release = root / "downloads" / "ergoAI_3.0.run"
    monkeypatch.setattr(
        installer,
        "ERGOAI_RELEASE_SHA256",
        hashlib.sha256(release.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        installer,
        "ERGOAI_RELEASE_SIZE_BYTES",
        release.stat().st_size,
    )
    receipt = advisors_cert.build_ergoai_live_toolchain_contract(
        repo_root=REPO_ROOT,
        install_root=root,
        executable=executable,
        platform_key="linux-aarch64",
        env=advisors_cert.offline_env(os.environ),
        run_semantics=True,
        timeout=5.0,
    )
    return {
        "root": root,
        "executable": executable,
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert LOCK_PATH.is_file()
    assert INSTALLER_PATH.is_file()
    assert WRAPPER_PATH.is_file()
    assert ADVISORS_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, advisors_cert, wrapper_mod) -> None:
    assert installer.ERGOAI_VERSION == LOCKED_ERGOAI_VERSION
    assert installer.ERGOAI_RELEASE_TAG == LOCKED_RELEASE_TAG
    assert installer.ERGOAI_RELEASE_SHA256 == LOCKED_SHA256
    assert installer.ERGOAI_RELEASE_SIZE_BYTES == LOCKED_SIZE
    assert set(installer.ERGOAI_LIVE_SEMANTIC_CASE_KINDS) == REQUIRED_CASE_KINDS
    assert "xsb" in installer.ERGOAI_RUNTIME_DEPENDENCIES
    assert installer.ERGOAI_ENTRY_POINT == "runergo"

    assert advisors_cert.ERGOAI_LIVE_TOOLCHAIN_INTERFACE == INTERFACE
    assert advisors_cert.ERGOAI_LIVE_TOOLCHAIN_SCHEMA == SCHEMA_VERSION
    assert advisors_cert.ERGOAI_LIVE_TOOLCHAIN_GOAL_ID == GOAL_ID
    assert advisors_cert.ERGOAI_LIVE_TOOLCHAIN_TASK_ID == TASK_ID
    assert set(advisors_cert.ERGOAI_LIVE_CASE_KINDS) == REQUIRED_CASE_KINDS

    assert wrapper_mod.LIVE_TOOLCHAIN_INTERFACE == INTERFACE
    assert set(wrapper_mod.LIVE_CASE_KINDS) == REQUIRED_CASE_KINDS
    assert wrapper_mod.AUTHORITY_CEILING == "advisory"
    assert (
        wrapper_mod.EVIDENCE_CLASS
        == "proposal_or_candidate_until_independent_reconstruction"
    )


# ---------------------------------------------------------------------------
# Lock binding
# ---------------------------------------------------------------------------


def test_lock_binds_official_ergoai_distribution(lock_document) -> None:
    versions = lock_document.get("managed_pin_versions") or {}
    assert versions.get("ergoai") == LOCKED_ERGOAI_VERSION

    inventory = (lock_document.get("checksummed_release_inventory") or {}).get(
        "ergoai"
    )
    assert isinstance(inventory, dict)
    assert inventory["version"] == LOCKED_ERGOAI_VERSION
    assert inventory["sha256"] == LOCKED_SHA256
    assert inventory["release_tag"] == LOCKED_RELEASE_TAG
    assert inventory["artifact_size_bytes"] == LOCKED_SIZE
    assert inventory["entry_point"] == "runergo"
    assert inventory["identity_probe"]["argv"] == ["--version"]
    assert "xsb" in (inventory.get("runtime_dependencies") or {})
    for key in ("sh", "make", "gcc", "flex", "bison"):
        assert key in (inventory.get("build_dependencies") or {})
    acquisition = inventory.get("acquisition_conditions") or {}
    assert REQUIRED_LOCK_ACQUISITION_KEYS <= set(acquisition)
    assert set(inventory.get("platforms") or {}) == {
        "linux-x86_64",
        "linux-aarch64",
    }

    tools = {
        tool["tool_id"]: tool
        for tool in lock_document.get("tools") or ()
        if isinstance(tool, dict) and "tool_id" in tool
    }
    ergo = tools["ergoai"]
    assert ergo["installer_entry"] == "ensure_ergoai"
    assert ergo["license"] == "Apache-2.0"
    assert ergo["source"] == "https://github.com/ErgoAI/ErgoEngine"
    pins = ergo.get("pins") or []
    assert {pin["platform"] for pin in pins} == {"linux-x86_64", "linux-aarch64"}
    for pin in pins:
        assert pin["version"] == LOCKED_ERGOAI_VERSION
        assert pin["sha256"] == LOCKED_SHA256
        assert pin["is_checksummed"] is True
        assert pin["release_tag"] == LOCKED_RELEASE_TAG

    contract = ergo.get("deployment_contract") or {}
    assert contract["live_toolchain_contract_interface"] == INTERFACE
    assert contract["goal_id"] == GOAL_ID
    assert contract["task_id"] == TASK_ID
    assert contract["entry_point"] == "runergo"
    assert contract["authority_ceiling"] == "advisory"
    assert set(contract.get("live_semantic_checks_required") or []) == (
        REQUIRED_CASE_KINDS
    )
    assert "xsb" in (contract.get("runtime_dependencies") or {})
    lazy = contract.get("lazy_install") or {}
    assert lazy.get("staged") is True
    assert lazy.get("checksum_verified") is True
    assert lazy.get("atomic") is True
    assert lazy.get("relocatable") is True
    assert lazy.get("offline_after_acquisition") is True
    assert lazy.get("never_during_certification") is True


# ---------------------------------------------------------------------------
# Structural contract (offline, no live binary required)
# ---------------------------------------------------------------------------


def test_structural_contract_receipt(structural_contract) -> None:
    assert structural_contract["interface"] == INTERFACE
    assert structural_contract["schema_version"] == SCHEMA_VERSION
    assert structural_contract["goal_id"] == GOAL_ID
    assert structural_contract["task_id"] == TASK_ID
    assert structural_contract["structural_passed"] is True
    assert structural_contract["contract_passed"] is True
    assert structural_contract["grants_proof_authority"] is False
    assert structural_contract["grants_theorem_authority"] is False
    assert structural_contract["promotion_blocked"] is True
    assert structural_contract["authority_ceiling"] == "advisory"
    assert structural_contract["network_used"] is False
    assert structural_contract["install_attempted"] is False
    assert structural_contract["download_attempted"] is False
    assert structural_contract["live_vendor_execution"] is False
    assert set(structural_contract["case_kinds"]) == REQUIRED_CASE_KINDS
    assert not structural_contract["block_reasons"]


def test_lazy_install_is_fail_closed(installer, tmp_path) -> None:
    root = tmp_path / "lazy"
    refused = installer.ensure_ergoai(
        yes=False,
        strict=False,
        force=True,
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        hermetic_shim=True,
    )
    assert refused.status in {"blocked", "refused"}
    assert "yes_required" in refused.reason_codes
    assert refused.install_attempted is False
    assert refused.grants_proof_authority is False

    with pytest.raises(installer.AdvisorInstallerError):
        installer.authorize_plugin_install(
            "ergoai",
            yes=True,
            import_context=True,
        )

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("FORMAL_VERIFICATION_CERTIFY_OFFLINE", "1")
        offline = installer.ensure_ergoai(
            yes=True,
            strict=False,
            force=True,
            install_root=root / "offline",
            repo_root=REPO_ROOT,
            platform_key="linux-aarch64",
            hermetic_shim=False,
        )
    finally:
        monkeypatch.undo()
    assert not offline.ok
    assert offline.phase == "offline_policy"
    assert "offline_policy_blocks_live_install" in offline.reason_codes
    assert not (
        root / "offline" / "advisors" / "ergoai" / "3.0" / "identity.json"
    ).exists()


def test_wrapper_simulation_is_not_live_vendor(wrapper_mod) -> None:
    wrapper = wrapper_mod.ErgoAIWrapper(lazy_install=False)
    assert wrapper.simulation_mode is True or wrapper.binary is not None
    if wrapper.simulation_mode:
        adapter = wrapper.run_live_semantic_adapter(require_live_binary=True)
        assert adapter["live_vendor_execution"] is False
        assert adapter["passed"] is False
        assert adapter["grants_proof_authority"] is False
        assert adapter["authority_ceiling"] == "advisory"
        assert (
            adapter["evidence_class"]
            == "proposal_or_candidate_until_independent_reconstruction"
        )
        stats = wrapper.get_statistics()
        assert stats["grants_proof_authority"] is False
        assert stats["authority_ceiling"] == "advisory"


# ---------------------------------------------------------------------------
# Full live matrix through fixture executable
# ---------------------------------------------------------------------------


def test_semantic_matrix_through_fixture(installer, live_fixture) -> None:
    executable = live_fixture["executable"]
    semantics = installer.run_ergoai_semantic_checks(
        executable,
        timeout=5.0,
        include_extended=True,
        bound_timeout_seconds=0.15,
        max_output_bytes=256,
    )
    assert semantics["core_passed"] is True
    assert semantics["extended_passed"] is True
    assert semantics["passed"] is True
    assert semantics["replay_bound"] is True
    assert semantics["grants_proof_authority"] is False
    assert (
        semantics["evidence_class"]
        == "proposal_or_candidate_until_independent_reconstruction"
    )
    checks = semantics["checks"]
    for kind in REQUIRED_CASE_KINDS:
        assert checks[kind]["passed"] is True, kind
    assert checks["timeout"]["timed_out"] is True
    assert checks["resource_bound"]["resource_bound_enforced"] is True
    assert checks["malformed"]["verdict"] == "error"
    assert checks["entailment"]["verdict"] == "yes"
    assert checks["non_entailment"]["verdict"] == "no"
    assert checks["mutation"]["verdict"] == "no"
    assert checks["contradiction"]["verdict"] in {"no", "error", "unknown"}
    # Legacy aliases remain for role-certification fixtures.
    assert checks["positive"]["passed"] is True
    assert checks["negative"]["passed"] is True


def test_live_toolchain_contract_with_semantics(live_fixture) -> None:
    receipt = live_fixture["receipt"]
    assert receipt["interface"] == INTERFACE
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["structural_passed"] is True
    assert receipt["semantic_passed"] is True
    assert receipt["contract_passed"] is True
    assert receipt["grants_proof_authority"] is False
    assert receipt["promotion_blocked"] is True
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert not receipt["block_reasons"]
    by_id = {check["check_id"]: check for check in receipt["checks"]}
    for kind in REQUIRED_CASE_KINDS:
        assert by_id[f"ergoai.live_toolchain.case.{kind}"]["status"] == "passed"
    assert by_id["ergoai.live_toolchain.lock_binding"]["status"] == "passed"
    assert by_id["ergoai.live_toolchain.lazy_install_policy"]["status"] == "passed"
    assert by_id["ergoai.live_toolchain.wrapper_adapter"]["status"] == "passed"
    assert by_id["ergoai.live_toolchain.authority_boundary"]["status"] == "passed"


def test_wrapper_live_adapter_through_fixture(wrapper_mod, live_fixture) -> None:
    executable = live_fixture["executable"]
    wrapper = wrapper_mod.ErgoAIWrapper(binary=executable, lazy_install=False)
    assert wrapper.simulation_mode is False
    assert wrapper.is_live_vendor_execution() is True
    adapter = wrapper.run_live_semantic_adapter(
        timeout_seconds=5.0,
        bound_timeout_seconds=0.15,
        max_output_bytes=256,
        require_live_binary=True,
    )
    assert adapter["interface"] == INTERFACE
    assert adapter["live_vendor_execution"] is True
    assert adapter["passed"] is True
    assert adapter["grants_proof_authority"] is False
    assert adapter["authority_ceiling"] == "advisory"
    assert (
        adapter["evidence_class"]
        == "proposal_or_candidate_until_independent_reconstruction"
    )
    for kind in REQUIRED_CASE_KINDS:
        assert adapter["checks"][kind]["passed"] is True

    bounded = wrapper.evaluate_bounded_goal(
        "fvt_ergo_subject : fvt_ergo_expected",
        timeout_seconds=5.0,
    )
    assert bounded["grants_proof_authority"] is False
    assert bounded["authority_ceiling"] == "advisory"
    assert bounded["live_vendor_execution"] is True
    assert bounded["status"] in {"success", "failure", "error", "timeout", "unknown"}


def test_hermetic_shim_is_not_live_vendor_execution(
    installer, advisors_cert, tmp_path
) -> None:
    root = tmp_path / "hermetic"
    receipt = installer.ensure_ergoai(
        yes=True,
        strict=True,
        force=True,
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        hermetic_shim=True,
        test_mode=True,
    )
    assert receipt.ok
    probe = installer.probe_ergoai_identity(
        expected_version=LOCKED_ERGOAI_VERSION,
        install_root=root,
        require_managed_vendor=True,
        platform_key="linux-x86_64",
    )
    assert probe.get("is_hermetic_advisor_shim") is True
    assert probe.get("managed_vendor_provenance_verified") is False

    contract = advisors_cert.build_ergoai_live_toolchain_contract(
        repo_root=REPO_ROOT,
        install_root=root,
        executable=receipt.executable_path,
        platform_key="linux-x86_64",
        run_semantics=True,
        timeout=2.0,
    )
    # Structural pieces may pass, but hermetic shims never become live vendor.
    assert contract["live_vendor_execution"] is False
    assert contract["grants_proof_authority"] is False


def test_live_certifier_authority_ceiling(
    installer, advisors_cert, tmp_path, monkeypatch
) -> None:
    root = tmp_path / "certifier"
    executable = _materialize_live_fixture(root, installer)
    release = root / "downloads" / "ergoAI_3.0.run"
    monkeypatch.setattr(
        installer,
        "ERGOAI_RELEASE_SHA256",
        hashlib.sha256(release.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        installer,
        "ERGOAI_RELEASE_SIZE_BYTES",
        release.stat().st_size,
    )
    receipt = advisors_cert.certify_live_ergoai_vendor(
        executable=executable,
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key="linux-aarch64",
        timeout=5.0,
    )
    assert receipt["vendor_certified"] is True
    assert receipt["grants_proof_authority"] is False
    assert receipt["promotion_blocked"] is True
    assert receipt["authority_ceiling"] == "advisory"
    by_id = {check["check_id"]: check for check in receipt["checks"]}
    for kind in ("positive", "negative", "mutation", "replay"):
        assert by_id[f"advisors.ergoai_live.{kind}"]["status"] == "passed"
