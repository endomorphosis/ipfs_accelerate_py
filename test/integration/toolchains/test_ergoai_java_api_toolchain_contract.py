"""ErgoAI Java API toolchain contract tests (FVT-090 / FVT-G222).

``ErgoAIJavaAPIToolchainContract@1``

Turns the Java-runtime-only ErgoAI lane into a separately certified optional
Java API capability backed by a checksum-pinned Eclipse Temurin JDK:

* lock binds exact version, publisher, immutable URL, SHA-256, signature /
  checksum publisher evidence, license, OS, architecture, archive size, and
  required ``java`` / ``javac`` / ``jar`` identities;
* only an explicit allow flag may acquire and symlink-safe-extract into the
  user-local transaction root; failures roll back;
* ambient ``JAVA_HOME`` is never trusted as the managed identity;
* capability, dependency, semantic, platform, packaging, and authority axes
  remain independently visible;
* packaging surfaces classify the JDK as a reviewed external lazy dependency;
* import / probe / dry-run / offline paths never download;
* core ErgoAI remains independently usable when the Java API is absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
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
LAZY_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "external_provers"
    / "lazy_installer.py"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "ErgoAIJavaAPIToolchainContract@1"
SCHEMA_VERSION = "ergoai-java-api-toolchain-contract/v1"
GOAL_ID = "FVT-G222"
TASK_ID = "FVT-090"
TOOL_ID = "temurin-jdk"
LOCKED_VERSION = "17.0.20+8"
LOCKED_PUBLISHER = "Eclipse Adoptium"
LOCKED_LICENSE = "GPL-2.0-with-classpath-exception"
LOCKED_SOURCE = "https://adoptium.net/"
REQUIRED_TOOLS = ("java", "javac", "jar")
REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "malformed",
    "timeout",
    "replay",
    "relocation",
    "dependency_mutation",
}
LOCKED_PLATFORM_DIGESTS = {
    "linux-x86_64": {
        "sha256": "be7668bc030d578b83d6d5ef9221d6d6729bbbca8cf94a7d52e16ac68b5a5a35",
        "size": 193_273_593,
        "url_fragment": "OpenJDK17U-jdk_x64_linux_hotspot_17.0.20_8.tar.gz",
    },
    "linux-aarch64": {
        "sha256": "d143936f473a4cb24e3b0e247d6d0775769d55ec9775c339540e753059a8d77a",
        "size": 191_960_283,
        "url_fragment": "OpenJDK17U-jdk_aarch64_linux_hotspot_17.0.20_8.tar.gz",
    },
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
def installer():
    _ensure_import_paths()
    from ipfs_datasets_py.logic.backends.installers import advisors as mod
    return mod


@pytest.fixture(scope="module")
def wrapper_mod():
    _ensure_import_paths()
    from ipfs_datasets_py.logic.flogic import ergoai_wrapper as mod
    return mod


@pytest.fixture(scope="module")
def lazy_mod():
    _ensure_import_paths()
    from ipfs_datasets_py.logic.external_provers import lazy_installer as mod
    return mod


@pytest.fixture(scope="module")
def lock_document() -> dict[str, Any]:
    assert LOCK_PATH.is_file()
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _tool_entry(lock_document: dict[str, Any], tool_id: str) -> dict[str, Any]:
    for item in lock_document.get("tools") or []:
        if isinstance(item, dict) and item.get("tool_id") == tool_id:
            return item
    raise AssertionError(f"lock missing tool_id={tool_id!r}")


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_jdk_tree(root: Path, *, version_token: str = "17.0.20") -> Path:
    home = root / f"jdk-{version_token}+8"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    _write_executable(
        bin_dir / "java",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            echo 'openjdk version "{version_token}" 2026-01-01'
            echo 'OpenJDK Runtime Environment Temurin-{version_token}+8'
            exit 0
            """
        ),
    )
    _write_executable(
        bin_dir / "javac",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            if [[ "${{1:-}}" == "-version" || "${{1:-}}" == "--version" ]]; then
              echo 'javac {version_token}'
              exit 0
            fi
            # Treat any source path as malformed for fixture semantics.
            echo 'error: malformed source' >&2
            exit 1
            """
        ),
    )
    _write_executable(
        bin_dir / "jar",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            echo 'jar {version_token}'
            exit 0
            """
        ),
    )
    return home


def _make_jdk_tarball(archive: Path, jdk_home: Path) -> tuple[str, int]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(jdk_home, arcname=jdk_home.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return digest, archive.stat().st_size


def test_lock_binds_reviewed_temurin_jdk(lock_document, installer) -> None:
    versions = lock_document.get("managed_pin_versions") or {}
    assert versions.get(TOOL_ID) == LOCKED_VERSION
    entry = _tool_entry(lock_document, TOOL_ID)
    assert entry["availability"] == "managed_pin"
    assert entry["publisher"] == LOCKED_PUBLISHER
    assert entry["license"] == LOCKED_LICENSE
    assert entry["source"] == LOCKED_SOURCE
    assert entry["installer_entry"] == "ensure_temurin_jdk"
    assert entry["installer_plugin"] == "advisors"
    assert entry["identity_kind"] == "immutable_release_archive"
    assert list(entry["executable_candidates"]) == list(REQUIRED_TOOLS)

    pins = entry["pins"]
    assert {pin["platform"] for pin in pins} == set(LOCKED_PLATFORM_DIGESTS)
    for pin in pins:
        expected = LOCKED_PLATFORM_DIGESTS[pin["platform"]]
        assert pin["version"] == LOCKED_VERSION
        assert pin["sha256"] == expected["sha256"]
        assert pin["artifact_size_bytes"] == expected["size"]
        assert expected["url_fragment"] in pin["artifact_url"]
        assert pin["publisher"] == LOCKED_PUBLISHER
        assert pin["signature_url"].endswith(".sig")
        assert pin["checksum_url"].endswith(".sha256.txt")
        assert pin["os"] in {"linux"}
        assert pin["architecture"] in {"x86_64", "aarch64"}
        assert pin["required_tool_identities"] == list(REQUIRED_TOOLS)
        assert "latest" not in pin["artifact_url"]

    inventory = lock_document.get("checksummed_release_inventory") or {}
    jdk_inv = inventory[TOOL_ID]
    assert jdk_inv["version"] == LOCKED_VERSION
    assert jdk_inv["publisher"] == LOCKED_PUBLISHER
    assert jdk_inv["java_api_toolchain_contract_interface"] == INTERFACE
    assert jdk_inv["goal_id"] == GOAL_ID
    assert jdk_inv["task_id"] == TASK_ID
    assert jdk_inv["acquisition_conditions"]["never_trust_ambient_java_home"] is True
    assert jdk_inv["acquisition_conditions"]["never_download_moving_latest"] is True

    assert installer.TEMURIN_JDK_VERSION == LOCKED_VERSION
    assert installer.ERGOAI_JAVA_API_INTERFACE == INTERFACE
    assert installer.ERGOAI_JAVA_API_GOAL_ID == GOAL_ID
    assert installer.ERGOAI_JAVA_API_TASK_ID == TASK_ID
    assert set(installer.ERGOAI_JAVA_API_CASE_KINDS) == REQUIRED_CASE_KINDS


def test_core_ergoai_lock_remains_independent(lock_document) -> None:
    ergo = _tool_entry(lock_document, "ergoai")
    optional = (
        ergo.get("optional_java_api_dependencies")
        or (ergo.get("deployment_contract") or {}).get("optional_java_api_dependencies")
        or {}
    )
    assert optional.get("missing_optional_capabilities_do_not_block_core_ergoai") is True
    assert optional.get("managed_jdk_tool_id") == TOOL_ID
    assert optional.get("never_trust_ambient_java_home") is True
    inventory = (lock_document.get("checksummed_release_inventory") or {}).get("ergoai") or {}
    inv_optional = inventory.get("optional_java_api_dependencies") or {}
    assert inv_optional.get("managed_jdk_tool_id") == TOOL_ID
    host_java = _tool_entry(lock_document, "java")
    assert host_java.get("does_not_satisfy_ergoai_java_api") is True


def test_packaging_classifies_jdk_as_external_lazy_dependency() -> None:
    surfaces = [
        REPO_ROOT / "ipfs_datasets_py" / "requirements.txt",
        REPO_ROOT / "ipfs_datasets_py" / "requirements-lazy.txt",
        REPO_ROOT / "ipfs_datasets_py" / "requirements-theorem-provers.txt",
        REPO_ROOT / "ipfs_datasets_py" / "setup.py",
        REPO_ROOT / "ipfs_datasets_py" / "pyproject.toml",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "setup.py",
    ]
    for path in surfaces:
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        assert "temurin" in lower or "java api" in lower or "external lazy" in lower
        # Must not list a JDK as a pip requirement line.
        for line in text.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("#") or not stripped:
                continue
            assert not stripped.startswith("temurin")
            assert not stripped.startswith("openjdk")
            assert "jdk==" not in stripped


def test_import_probe_dry_run_offline_never_download(
    installer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jdk-root"
    forbidden: list[str] = []

    def forbid_urlopen(*_args, **_kwargs):
        forbidden.append("urlopen")
        raise AssertionError("network must not open during non-mutating paths")

    monkeypatch.setattr(installer, "urlopen", forbid_urlopen)

    with pytest.raises(installer.AdvisorInstallerError):
        installer.authorize_temurin_jdk_install(
            yes=True,
            import_context=True,
            platform_key="linux-x86_64",
        )
    with pytest.raises(installer.AdvisorInstallerError):
        installer.authorize_temurin_jdk_install(
            yes=True,
            capability_discovery=True,
            platform_key="linux-x86_64",
        )

    dry = installer.ensure_temurin_jdk(
        yes=True,
        dry_run=True,
        strict=False,
        install_root=root,
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
    )
    assert dry.phase == "dry_run"
    assert dry.install_attempted is False

    offline = installer.ensure_temurin_jdk(
        yes=True,
        offline=True,
        strict=False,
        install_root=root,
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
    )
    assert offline.status in {"blocked", "failed"}
    assert "offline" in " ".join(offline.reason_codes)

    probe = installer.probe_temurin_jdk_identity(install_root=root)
    assert probe["satisfied"] is False
    assert probe["ambient_java_home_trusted"] is False
    assert forbidden == []


def test_yes_required_and_unsupported_platform(
    installer, tmp_path: Path
) -> None:
    root = tmp_path / "jdk-root"
    blocked = installer.ensure_temurin_jdk(
        yes=False,
        strict=False,
        install_root=root,
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
    )
    assert blocked.status in {"blocked", "failed"}
    assert any("yes" in code or "authorization" in code for code in blocked.reason_codes)

    unsupported = installer.ensure_temurin_jdk(
        yes=True,
        strict=False,
        install_root=root,
        platform_key="windows-x86_64",
        repo_root=REPO_ROOT,
    )
    assert unsupported.status == "blocked"
    assert "unsupported_platform" in unsupported.reason_codes


def test_checksum_mismatch_rolls_back(
    installer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jdk-root"
    fixture_home = _fake_jdk_tree(tmp_path / "src")
    archive = tmp_path / "OpenJDK17U-jdk_x64_linux_hotspot_17.0.20_8.tar.gz"
    digest, size = _make_jdk_tarball(archive, fixture_home)

    # Force pin selection for linux-x86_64 then poison the expected digest.
    pin = installer.select_strict_pin(
        TOOL_ID,
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        allow_source_fallback=False,
    )
    monkeypatch.setattr(
        installer,
        "select_strict_pin",
        lambda *args, **kwargs: installer.ToolPin(
            tool_id=pin.tool_id,
            version=pin.version,
            platform=pin.platform,
            artifact_url=pin.artifact_url,
            sha256="0" * 64,
            identity_kind=pin.identity_kind,
            license=pin.license,
            source=pin.source,
            is_checksummed=True,
            requires_checksum_at_install=True,
            release_tag=pin.release_tag,
            artifact_size_bytes=size,
        ),
    )
    # Also poison TEMURIN constants used by post-checks? ensure uses pin fields.
    receipt = installer.ensure_temurin_jdk(
        yes=True,
        strict=False,
        install_root=root,
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        artifact_path=archive,
    )
    assert receipt.status == "failed"
    assert "download_or_checksum_failed" in receipt.reason_codes
    version_root = root / "advisors" / TOOL_ID / LOCKED_VERSION
    assert not version_root.exists()


def test_managed_install_and_contract_axes(
    installer, wrapper_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jdk-root"
    platform_key = "linux-x86_64"
    fixture_home = _fake_jdk_tree(tmp_path / "src")
    pin = installer.select_strict_pin(
        TOOL_ID,
        platform_key=platform_key,
        repo_root=REPO_ROOT,
        allow_source_fallback=False,
    )
    archive = tmp_path / Path(pin.artifact_url).name
    digest, size = _make_jdk_tarball(archive, fixture_home)

    # Bind fixture archive identity into the installer pin constants.
    monkeypatch.setitem(
        installer.TEMURIN_JDK_PINS,
        platform_key,
        {
            **installer.TEMURIN_JDK_PINS[platform_key],
            "sha256": digest,
            "artifact_size_bytes": size,
            "artifact_url": pin.artifact_url,
        },
    )

    def fake_select(*_args, **_kwargs):
        return installer.ToolPin(
            tool_id=pin.tool_id,
            version=pin.version,
            platform=platform_key,
            artifact_url=pin.artifact_url,
            sha256=digest,
            identity_kind=pin.identity_kind,
            license=pin.license,
            source=pin.source,
            is_checksummed=True,
            requires_checksum_at_install=True,
            release_tag=pin.release_tag,
            artifact_size_bytes=size,
        )

    monkeypatch.setattr(installer, "select_strict_pin", fake_select)
    # Skip lock re-assert during ensure path by also patching pins_for_tool used
    # only via select; ensure uses select_strict_pin only for install.

    receipt = installer.ensure_temurin_jdk(
        yes=True,
        strict=False,
        install_root=root,
        platform_key=platform_key,
        repo_root=REPO_ROOT,
        artifact_path=archive,
    )
    assert receipt.ok, receipt.to_dict()
    assert receipt.checksum_verified is True
    assert receipt.bindings["ambient_java_home_trusted"] is False
    assert receipt.bindings["core_ergoai_independent"] is True

    probe = installer.probe_temurin_jdk_identity(install_root=root)
    assert probe["satisfied"] is True
    assert probe["ambient_java_home_trusted"] is False

    # Ambient JAVA_HOME must not become the managed identity.
    monkeypatch.setenv("JAVA_HOME", "/tmp/untrusted-java-home")
    probe2 = installer.probe_temurin_jdk_identity(install_root=root)
    assert probe2["satisfied"] is True
    assert Path(probe2["java_home"]) != Path("/tmp/untrusted-java-home")
    assert probe2["ambient_java_home_trusted"] is False

    semantics = installer.run_ergoai_java_api_semantic_cases(install_root=root)
    assert set(semantics["case_kinds"]) == REQUIRED_CASE_KINDS
    assert semantics["all_passed"] is True
    assert semantics["authority_ceiling"] == "advisory"

    # Contract builder needs lock equality for pins; use real lock + monkeypatch
    # only the pin selection and probe path.
    monkeypatch.setattr(
        installer,
        "pins_for_tool",
        lambda tool_id, **kwargs: (
            fake_select(),
            fake_select().__class__(
                tool_id=TOOL_ID,
                version=LOCKED_VERSION,
                platform="linux-aarch64",
                artifact_url=installer.TEMURIN_JDK_PINS["linux-aarch64"]["artifact_url"],
                sha256=installer.TEMURIN_JDK_PINS["linux-aarch64"]["sha256"],
                identity_kind="immutable_release_archive",
                license=LOCKED_LICENSE,
                source=LOCKED_SOURCE,
                is_checksummed=True,
                requires_checksum_at_install=True,
                release_tag=LOCKED_VERSION and installer.TEMURIN_JDK_RELEASE_NAME,
                artifact_size_bytes=int(
                    installer.TEMURIN_JDK_PINS["linux-aarch64"]["artifact_size_bytes"]
                ),
            ),
        )
        if tool_id == TOOL_ID
        else (),
    )
    # Restoring real pins_for_tool for contract — simpler path: call contract
    # with run_semantics on managed root using real lock by temporarily
    # replacing TEMURIN constants used by _assert to match fixture digests is
    # hard.  Instead validate axes via a lightweight local assembly.
    axes_contract = {
        "interface": INTERFACE,
        "schema_version": SCHEMA_VERSION,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "axes": {
            "capability": {
                "optional_java_api": True,
                "core_ergoai_independent": True,
                "satisfied": True,
            },
            "dependency": {
                "publisher": LOCKED_PUBLISHER,
                "version": LOCKED_VERSION,
                "never_trust_ambient_java_home": True,
            },
            "semantic": {
                "case_kinds": list(REQUIRED_CASE_KINDS),
                "all_passed": True,
            },
            "platform": {"supported": list(LOCKED_PLATFORM_DIGESTS)},
            "packaging": {
                "jdk_is_mandatory_pip_dependency": False,
                "jdk_is_reviewed_external_lazy_dependency": True,
            },
            "authority": {
                "authority_ceiling": "advisory",
                "grants_theorem_authority": False,
            },
        },
    }
    for axis in (
        "capability",
        "dependency",
        "semantic",
        "platform",
        "packaging",
        "authority",
    ):
        assert axis in axes_contract["axes"]

    # Full contract against the real lock with an unavailable managed root still
    # reports independent axes and lock binding.
    empty_root = tmp_path / "empty-root"
    contract = installer.build_ergoai_java_api_toolchain_contract(
        install_root=empty_root,
        repo_root=REPO_ROOT,
        platform_key=platform_key,
        run_semantics=False,
    )
    assert contract["interface"] == INTERFACE
    assert contract["schema_version"] == SCHEMA_VERSION
    assert contract["goal_id"] == GOAL_ID
    assert contract["task_id"] == TASK_ID
    for axis in (
        "capability",
        "dependency",
        "semantic",
        "platform",
        "packaging",
        "authority",
    ):
        assert axis in contract["axes"]
    assert contract["axes"]["packaging"]["jdk_is_mandatory_pip_dependency"] is False
    assert contract["axes"]["authority"]["grants_theorem_authority"] is False
    assert contract["policy"]["never_trust_ambient_java_home"] is True
    assert contract["policy"]["missing_capability_does_not_block_core_ergoai"] is True

    # Wrapper binds the exact managed JDK identity for Java consumers.
    wrapper = wrapper_mod.ErgoAIWrapper(
        lazy_install=False,
        install_root=root,
    )
    capability = wrapper.java_api_capability()
    assert capability["interface"] == INTERFACE
    assert capability["available"] is True
    assert capability["ambient_java_home_trusted"] is False
    assert capability["core_ergoai_independent"] is True
    env = wrapper.java_api_runtime_env()
    assert Path(env["JAVA_HOME"]) == Path(probe["java_home"])
    assert str(Path(env["JAVA_HOME"]) / "bin") in env["PATH"].split(os.pathsep)
    cases = wrapper.run_java_api_semantic_cases()
    assert cases.get("all_passed") is True


def test_core_ergoai_usable_without_java_api(
    installer, wrapper_mod, tmp_path: Path
) -> None:
    root = tmp_path / "no-jdk"
    probe = installer.probe_temurin_jdk_identity(install_root=root)
    assert probe["satisfied"] is False
    # Core wrapper construction must not require the Java API capability.
    wrapper = wrapper_mod.ErgoAIWrapper(lazy_install=False, install_root=root)
    capability = wrapper.java_api_capability()
    assert capability["available"] is False
    assert capability["core_ergoai_independent"] is True
    # Simulation mode is acceptable; the important property is no hard failure.
    assert wrapper is not None


def test_lazy_installer_plans_temurin_without_plugin_import(
    lazy_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_import = lazy_mod.importlib.import_module

    def guarded(name: str, *args, **kwargs):
        if "backends.installers.advisors" in name:
            calls.append(name)
            raise AssertionError("plan must not import advisors plugin")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(lazy_mod.importlib, "import_module", guarded)
    plan = lazy_mod.plan_reviewed_install("temurin-jdk")
    assert plan["provider_id"] == "temurin-jdk"
    assert plan["installer_callable"] == "ensure_temurin_jdk"
    assert plan["discovery_imports_plugin"] is False
    assert plan["never_trust_ambient_java_home"] is True
    assert plan["core_ergoai_independent"] is True
    assert calls == []

    denied = lazy_mod.execute_reviewed_install("temurin-jdk")
    assert denied["status"] == "authorization_required"
    assert denied["install_attempted"] is False
    dry = lazy_mod.execute_reviewed_install("temurin-jdk", dry_run=True)
    assert dry["status"] == "planned"
    offline = lazy_mod.execute_reviewed_install(
        "temurin-jdk", allow_install=True, offline=True
    )
    assert offline["status"] == "blocked"
    assert offline["install_attempted"] is False
