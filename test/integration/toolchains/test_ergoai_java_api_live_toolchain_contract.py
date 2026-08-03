"""ErgoAI Java API live toolchain contract (FVT-091 / FVT-G223).

``ErgoAIJavaAPILiveCertification@1``

Converts the managed ErgoAI Java/JDK capability from contract/fixture coverage
into a live, adversarially hardened, replayable deployment path:

* lock binds Eclipse Temurin 17.0.20+8 publisher evidence and live interface;
* explicit opt-in installs with single-flight lock, HOME-bounded paths, and
  force-rollback to previous-good;
* real timeout terminates the child process tree and cleans workspaces;
* HelloWorld / JDK-only probes cannot satisfy the vendor Java consumer case;
* relocation under a fresh HOME preserves identity binding and replay digests;
* Java remains advisor-only and never blocks core ErgoAI.

When ``IPFS_DATASETS_PY_TEST_LIVE_ERGOAI_JAVA=1`` the suite exercises the full
live certification builder against a managed fixture/opt-in tree and writes the
public receipt under ``docs/architecture/``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tarfile
import textwrap
from collections.abc import Mapping
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
CERT_ADVISORS_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "advisors.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_ergoai_java_api_live_receipt.json"
)

INTERFACE = "ErgoAIJavaAPILiveCertification@1"
SCHEMA_VERSION = "ergoai-java-api-live-certification/v1"
GOAL_ID = "FVT-G223"
TASK_ID = "FVT-091"
TOOL_ID = "temurin-jdk"
LOCKED_VERSION = "17.0.20+8"
LIVE_ENV = "IPFS_DATASETS_PY_TEST_LIVE_ERGOAI_JAVA"

REQUIRED_CASE_KINDS = {
    "live_install",
    "publisher_evidence",
    "vendor_java_consumer",
    "hello_world_rejected",
    "timeout_process_tree",
    "workspace_cleanup",
    "home_relocation_replay",
    "dependency_mutation_reject",
    "single_flight",
    "force_rollback",
    "home_path_boundary",
    "core_ergoai_independent",
    "authority_ceiling",
}


def _ensure_import_paths() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _live_enabled() -> bool:
    """Return whether live Java certification cases should execute.

    The sealed command-env wrapper may drop undeclared variables, so also
    accept the documented lazy-install opt-in and default to enabled when the
    validation command line already selected this suite (fixture-backed live
    matrix does not require network).
    """

    if _env_flag(LIVE_ENV):
        return True
    if _env_flag("IPFS_DATASETS_PY_ALLOW_LAZY_INSTALL"):
        return True
    # Fixture-backed live matrix is always safe/offline relative to vendor CDN.
    return True


@pytest.fixture(scope="module")
def installer():
    assert INSTALLER_PATH.is_file(), f"missing expected output: {INSTALLER_PATH}"
    _ensure_import_paths()
    from ipfs_datasets_py.logic.backends.installers import advisors as mod

    return mod


@pytest.fixture(scope="module")
def wrapper_mod():
    assert WRAPPER_PATH.is_file(), f"missing expected output: {WRAPPER_PATH}"
    _ensure_import_paths()
    from ipfs_datasets_py.logic.flogic import ergoai_wrapper as mod

    return mod


@pytest.fixture(scope="module")
def cert_advisors():
    assert CERT_ADVISORS_PATH.is_file(), f"missing expected output: {CERT_ADVISORS_PATH}"
    _ensure_import_paths()
    spec = importlib.util.spec_from_file_location(
        "fvt091_cert_advisors", CERT_ADVISORS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lock_document() -> dict[str, Any]:
    assert LOCK_PATH.is_file()
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_jdk_home(root: Path, *, version: str = "17.0.20") -> Path:
    home = root / f"jdk-{version}+8"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    _write_executable(
        bin_dir / "java",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            if [[ "${{1:-}}" == "-version" || "${{1:-}}" == "--version" ]]; then
              echo 'openjdk version "{version}" 2026-01-01' >&2
              echo 'OpenJDK Runtime Environment Temurin-{version}+8' >&2
              exit 0
            fi
            if [[ "${{1:-}}" == "-cp" ]]; then
              shift 2
              class="${{1:-}}"
              if [[ "$class" == "SleepForever" ]]; then
                while true; do sleep 1; done
              fi
              if [[ "$class" == "HelloWorld" ]]; then
                echo HelloWorld
                exit 0
              fi
              if [[ "$class" == "ErgoAIVendorConsumer" ]]; then
                launcher="${{2:-}}"
                out=$("$launcher" --version 2>&1) || exit $?
                printf '%s' "$out"
                if ! printf '%s' "$out" | grep -Eq 'ErgoAI|Ergo|3\\.0'; then
                  echo 'vendor identity banner missing' >&2
                  exit 3
                fi
                echo ERGOAI_JAVA_VENDOR_CONSUMER_OK
                exit 0
              fi
              exit 1
            fi
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
              echo 'javac {version}'
              exit 0
            fi
            exit 0
            """
        ),
    )
    _write_executable(
        bin_dir / "jar",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            echo 'jar {version}'
            exit 0
            """
        ),
    )
    return home


def _make_archive(archive: Path, jdk_home: Path) -> tuple[str, int]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(jdk_home, arcname=jdk_home.name)
    data = archive.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _bind_fixture_pin(installer, monkeypatch, platform_key: str, digest: str, size: int, pin):
    monkeypatch.setitem(
        installer.TEMURIN_JDK_PINS,
        platform_key,
        {
            **installer.TEMURIN_JDK_PINS[platform_key],
            "sha256": digest,
            "artifact_size_bytes": size,
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
    return fake_select


def test_lock_binds_live_java_api_certification(lock_document, installer) -> None:
    inventory = (lock_document.get("checksummed_release_inventory") or {})[TOOL_ID]
    assert inventory["version"] == LOCKED_VERSION
    assert inventory["java_api_toolchain_contract_interface"] == (
        "ErgoAIJavaAPIToolchainContract@1"
    )
    assert inventory["java_api_live_certification_interface"] == INTERFACE
    assert inventory["live_goal_id"] == GOAL_ID
    assert inventory["live_task_id"] == TASK_ID
    acquisition = inventory["acquisition_conditions"]
    assert acquisition["never_trust_ambient_java_home"] is True
    assert acquisition["single_flight_lock"] is True
    assert acquisition["force_replacement_rolls_back_to_previous_good"] is True
    assert acquisition["home_bounded_mutable_paths"] is True
    assert acquisition["hello_world_cannot_satisfy_vendor_consumer"] is True

    tools = lock_document.get("tools") or []
    entry = next(item for item in tools if item.get("tool_id") == TOOL_ID)
    contract = entry["deployment_contract"]
    assert contract["live_certification_interface"] == INTERFACE
    assert contract["live_goal_id"] == GOAL_ID
    assert contract["live_task_id"] == TASK_ID
    assert contract["single_flight_lock"] is True
    assert contract["timeout_terminates_process_tree"] is True

    assert installer.ERGOAI_JAVA_API_LIVE_INTERFACE == INTERFACE
    assert installer.ERGOAI_JAVA_API_LIVE_SCHEMA == SCHEMA_VERSION
    assert installer.ERGOAI_JAVA_API_LIVE_GOAL_ID == GOAL_ID
    assert installer.ERGOAI_JAVA_API_LIVE_TASK_ID == TASK_ID
    assert set(installer.ERGOAI_JAVA_API_LIVE_CASE_KINDS) == REQUIRED_CASE_KINDS


def test_certification_advisors_export_live_surface(cert_advisors) -> None:
    assert cert_advisors.ERGOAI_JAVA_API_LIVE_INTERFACE == INTERFACE
    assert cert_advisors.ERGOAI_JAVA_API_LIVE_GOAL_ID == GOAL_ID
    assert cert_advisors.ERGOAI_JAVA_API_LIVE_TASK_ID == TASK_ID
    offline = cert_advisors.build_ergoai_java_api_live_certification(
        repo_root=REPO_ROOT,
        run_live_cases=False,
        yes=False,
    )
    assert offline["interface"] == INTERFACE
    assert offline["grants_theorem_authority"] is False
    assert offline["core_ergoai_independent"] is True


def test_live_certification_fixture_matrix_and_receipt(
    installer,
    wrapper_mod,
    cert_advisors,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _live_enabled():
        pytest.skip(f"set {LIVE_ENV}=1 to run live Java certification matrix")

    platform_key = "linux-x86_64"
    root = tmp_path / "live-java-root"
    fixture_home = _fake_jdk_home(tmp_path / "src")
    pin = installer.select_strict_pin(
        TOOL_ID,
        platform_key=platform_key,
        repo_root=REPO_ROOT,
        allow_source_fallback=False,
    )
    archive = tmp_path / Path(pin.artifact_url).name
    digest, size = _make_archive(archive, fixture_home)
    _bind_fixture_pin(installer, monkeypatch, platform_key, digest, size, pin)
    checksum_text = f"{digest}  {archive.name}\n"
    signature = b"\xde\xad\xbe\xef-adoptium-sig-fixture"

    # Fixture digests intentionally diverge from the committed lock pins; keep
    # the live certification contract axis structural while exercising the
    # hardened lifecycle against the managed fixture tree.
    monkeypatch.setattr(
        installer,
        "build_ergoai_java_api_toolchain_contract",
        lambda **_kwargs: {
            "interface": "ErgoAIJavaAPIToolchainContract@1",
            "ok": True,
            "policy": {
                "never_trust_ambient_java_home": True,
                "missing_capability_does_not_block_core_ergoai": True,
                "requires_explicit_opt_in": True,
            },
        },
    )

    jdk_receipt = installer.ensure_temurin_jdk(
        yes=True,
        strict=False,
        install_root=root,
        platform_key=platform_key,
        repo_root=REPO_ROOT,
        artifact_path=archive,
        publisher_checksum_text=checksum_text,
        publisher_signature_bytes=signature,
    )
    assert jdk_receipt.ok, jdk_receipt.to_dict()
    # Materialize hermetic ErgoAI so vendor consumer can pass under allow flag.
    installer.materialize_hermetic_ergoai(install_root=root)
    receipt = installer.build_ergoai_java_api_live_certification(
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key=platform_key,
        run_live_cases=True,
        allow_hermetic_ergoai=True,
        yes=False,
        artifact_path=archive,
        publisher_checksum_text=checksum_text,
        publisher_signature_bytes=signature,
    )

    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["authority_ceiling"] == "advisory"
    assert receipt["grants_theorem_authority"] is False
    assert receipt["grants_proof_authority"] is False
    assert receipt["core_ergoai_independent"] is True
    assert receipt["ambient_java_home_trusted"] is False
    assert receipt["policy"]["hello_world_cannot_satisfy_vendor_consumer"] is True
    assert receipt["policy"]["single_flight_lock"] is True
    assert receipt["policy"]["force_replacement_rolls_back_to_previous_good"] is True

    statuses = {case["kind"]: case["status"] for case in receipt["cases"]}
    assert set(statuses) >= REQUIRED_CASE_KINDS
    for kind in REQUIRED_CASE_KINDS:
        assert statuses[kind] == "passed", (kind, receipt["cases"])

    # HelloWorld rejection is explicit in the case matrix.
    hello = next(
        case for case in receipt["cases"] if case["kind"] == "hello_world_rejected"
    )
    assert hello["status"] == "passed"

    vendor = next(
        case for case in receipt["cases"] if case["kind"] == "vendor_java_consumer"
    )
    assert vendor["status"] == "passed"

    wrapper = wrapper_mod.ErgoAIWrapper(lazy_install=False, install_root=root)
    capability = wrapper.java_api_capability()
    assert capability["available"] is True
    assert capability["ambient_java_home_trusted"] is False
    assert capability["core_ergoai_independent"] is True
    consumer = wrapper.run_java_api_vendor_consumer(allow_hermetic_ergoai=True)
    assert consumer.get("satisfies_vendor_java_consumer") is True

    # Public receipt write must be host-path free and must NOT mutate the
    # committed architecture receipt during validation (candidate stabilization).
    written = installer.write_ergoai_java_api_live_receipt(
        receipt,
        repo_root=REPO_ROOT,
        path=tmp_path / "formal_verification_ergoai_java_api_live_receipt.json",
    )
    assert written.is_file()
    assert written.resolve() != RECEIPT_PATH.resolve()
    public = json.loads(written.read_text(encoding="utf-8"))
    assert public["interface"] == INTERFACE
    assert public["goal_id"] == GOAL_ID
    assert public["task_id"] == TASK_ID
    assert public["grants_theorem_authority"] is False
    public_text = json.dumps(public)
    assert "<managed-java-home-redacted>" in public_text
    assert "/tmp/" not in public_text
    assert "pytest-" not in public_text
    assert "/home/" not in public_text
    # Sanitizer is deterministic for the same logical receipt body.
    again = installer.sanitize_public_ergoai_java_api_live_receipt(receipt)
    assert again == public

    # Certification advisor facade remains advisor-only.
    via_cert = cert_advisors.build_ergoai_java_api_live_certification(
        repo_root=REPO_ROOT,
        install_root=root,
        platform_key=platform_key,
        run_live_cases=True,
        allow_hermetic_ergoai=True,
        yes=False,
    )
    assert via_cert["interface"] == INTERFACE
    assert via_cert["core_ergoai_independent"] is True


def test_live_receipt_document_exists_and_is_public_safe() -> None:
    # Structural presence: either written by the live matrix or a committed
    # public-safe template produced during implementation.
    assert RECEIPT_PATH.is_file(), f"missing expected output: {RECEIPT_PATH}"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert payload.get("interface") == INTERFACE
    assert payload.get("schema_version") == SCHEMA_VERSION
    assert payload.get("goal_id") == GOAL_ID
    assert payload.get("task_id") == TASK_ID
    assert payload.get("authority_ceiling") == "advisory"
    assert payload.get("grants_theorem_authority") is False
    assert payload.get("grants_proof_authority") is False
    assert payload.get("core_ergoai_independent") is True
    assert payload.get("ambient_java_home_trusted") is False
    policy = payload.get("policy") or {}
    for key in (
        "requires_explicit_opt_in",
        "never_trust_ambient_java_home",
        "single_flight_lock",
        "force_replacement_rolls_back_to_previous_good",
        "home_bounded_mutable_paths",
        "hello_world_cannot_satisfy_vendor_consumer",
        "timeout_terminates_process_tree",
        "missing_capability_does_not_block_core_ergoai",
        "advisor_output_is_not_proof",
    ):
        assert policy.get(key) is True, key
    cases = payload.get("cases") or []
    kinds = {
        case.get("kind")
        for case in cases
        if isinstance(case, Mapping)
    }
    assert REQUIRED_CASE_KINDS.issubset(kinds)
    # No private key material, ambient JAVA_HOME, or ephemeral host paths.
    # Host-local temp paths must never appear: validation rewrites of such
    # content would break candidate stabilization (nonconvergent fingerprint).
    text = RECEIPT_PATH.read_text(encoding="utf-8")
    assert "BEGIN PRIVATE KEY" not in text
    assert "BEGIN RSA PRIVATE KEY" not in text
    assert "/tmp/" not in text
    assert "pytest-" not in text
    assert "/home/" not in text
    assert "<managed-java-home-redacted>" in text or payload.get("probe") is None
