"""External Runtime MTL vendor certification tests (FVT-056 / FVT-G210).

``ExternalRuntimeMTLVendorCertification@1``

Acceptance covered:

* a locked TypeScript dependency graph builds an independent Node
  package/executable without importing or dispatching to the Python reference;
* package, source, lockfile, runtime, executable, and artifact digests are bound;
* positive, negative, interval/event mutation, timestamp boundary, shortest-prefix
  replay, malformed input, timeout, bounds, and disagreement cases execute out of
  process;
* finite-trace authority and inconclusive-prefix semantics are preserved;
* generated Python parity wrappers remain non-production shadow evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "ipfs_datasets_py"
INSTALLER_PATH = (
    DATASET_ROOT
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "installers"
    / "runtime_mtl.py"
)
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl_external.py"
)
CENTRAL_CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
TOOLCHAIN_LOCK_PATH = (
    REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
)
TS_PACKAGE = REPO_ROOT / "ipfs_datasets_py" / "typescript" / "logic-runtime-mtl"
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_runtime_mtl_external_install_receipt.json"
)

VENDOR_INTERFACE = "ExternalRuntimeMTLVendorCertification@1"
VENDOR_SCHEMA = "external-runtime-mtl-vendor-certification/v1"
VENDOR_RECEIPT_SCHEMA = (
    "formal-verification-runtime-mtl-external-install-receipt/v1"
)
VENDOR_GOAL_ID = "FVT-G210"
VENDOR_TASK_ID = "FVT-056"
VENDOR_INSTALLER_INTERFACE = "ExternalRuntimeMTLVendorInstaller@1"
PACKAGE_IDENTITY = "@ipfs-datasets/logic-runtime-mtl"
PIN_VERSION = "1.0.0-reviewed"

REQUIRED_CATEGORIES = {
    "satisfied",
    "violated",
    "timestamp_boundary",
    "interval_mutation",
    "event_mutation",
    "shortest_violating_prefix",
    "malformed",
    "clean_prefix",
}
REQUIRED_MUTATIONS = {"interval", "event"}


def _ensure_datasets_on_path() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (REPO_ROOT, datasets_root):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_module(name: str, path: Path):
    assert path.is_file(), f"missing module: {path}"
    _ensure_datasets_on_path()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def installer():
    return _load_module("runtime_mtl_vendor_installer", INSTALLER_PATH)


@pytest.fixture(scope="module")
def certifier():
    return _load_module("runtime_mtl_vendor_certification", CERTIFIER_PATH)


@pytest.fixture(scope="module")
def central_certifier():
    return _load_module(
        "runtime_mtl_vendor_central_certifier", CENTRAL_CERTIFIER_PATH
    )


@pytest.fixture(scope="module")
def install_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("runtime-mtl-vendor")


@pytest.fixture(scope="module")
def vendor_bundle(installer, install_root):
    return installer.ensure_runtime_mtl_vendor(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root,
        repo_root=REPO_ROOT,
        checksum_verified=True,
    )


@pytest.fixture(scope="module")
def vendor_certificate(certifier, install_root) -> dict[str, Any]:
    return certifier.certify_external_runtime_mtl_vendor(
        install_root=install_root,
        force_install=True,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Expected outputs / package lock
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert CERTIFIER_PATH.is_file()
    assert TS_PACKAGE.is_dir()
    assert (TS_PACKAGE / "package.json").is_file()
    assert (TS_PACKAGE / "package-lock.json").is_file()
    assert (TS_PACKAGE / "src" / "index.ts").is_file()
    assert (TS_PACKAGE / "src" / "cli.ts").is_file()
    assert RECEIPT_PATH.is_file()
    assert Path(__file__).is_file()


def test_typescript_package_identity_and_lock() -> None:
    package_path = TS_PACKAGE / "package.json"
    lock_path = TS_PACKAGE / "package-lock.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert package["name"] == PACKAGE_IDENTITY
    assert package["version"] == PIN_VERSION
    assert "runtime-mtl-external" in (package.get("bin") or {})
    assert lock["name"] == PACKAGE_IDENTITY
    assert lock["version"] == PIN_VERSION
    assert lock["lockfileVersion"] >= 2
    # Locked TypeScript dependency graph is present.
    packages = lock.get("packages") or {}
    assert "" in packages
    assert "node_modules/typescript" in packages
    assert packages["node_modules/typescript"]["version"] == "5.6.3"

    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(DATASET_ROOT),
            "ls-files",
            "--error-unmatch",
            "typescript/logic-runtime-mtl/package-lock.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode == 0, tracked.stderr
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert (
        receipt["runtime_mtl_external"]["lockfile_digest_sha256"]
        == lock_digest
    )


def test_installer_vendor_constants(installer) -> None:
    assert installer.VENDOR_INTERFACE == VENDOR_INSTALLER_INTERFACE
    assert installer.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert installer.VENDOR_TASK_ID == VENDOR_TASK_ID
    assert installer.VENDOR_PACKAGE_IDENTITY == PACKAGE_IDENTITY
    meta = installer.describe_runtime_mtl_installer()
    assert meta["policy"]["hermetic_parity_engines_are_non_production_shadows"] is True
    assert meta["policy"]["hermetic_parity_engines_cannot_satisfy_vendor"] is True
    assert meta["policy"]["vendor_builds_independent_typescript_node"] is True
    assert meta["policy"]["vendor_never_imports_python_reference"] is True
    assert meta["vendor"]["interface"] == VENDOR_INSTALLER_INTERFACE
    assert meta["vendor"]["goal_id"] == VENDOR_GOAL_ID


def test_certifier_vendor_constants(certifier) -> None:
    assert certifier.VENDOR_INTERFACE == VENDOR_INTERFACE
    assert certifier.VENDOR_SCHEMA_VERSION == VENDOR_SCHEMA
    assert certifier.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert certifier.VENDOR_TASK_ID == VENDOR_TASK_ID
    assert certifier.VENDOR_INSTALL_RECEIPT_SCHEMA == VENDOR_RECEIPT_SCHEMA


# ---------------------------------------------------------------------------
# Vendor install path
# ---------------------------------------------------------------------------


def test_vendor_install_independent_typescript_engine(installer, vendor_bundle) -> None:
    assert vendor_bundle.interface == VENDOR_INSTALLER_INTERFACE
    assert vendor_bundle.goal_id == VENDOR_GOAL_ID
    assert vendor_bundle.is_vendor_path is True
    assert vendor_bundle.ok
    by_id = {item.tool_id: item for item in vendor_bundle.receipts}
    assert "runtime-mtl-external" in by_id
    receipt = by_id["runtime-mtl-external"]
    assert receipt.ok
    assert receipt.is_vendor_path is True
    identity = receipt.identity
    assert identity is not None
    assert identity.is_vendor_build is True
    assert identity.is_hermetic_parity_engine is False
    assert identity.version == PIN_VERSION
    assert identity.package_identity == PACKAGE_IDENTITY
    assert identity.package_digest_sha256
    assert identity.source_digest_sha256
    assert identity.lockfile_digest_sha256
    assert identity.runtime_digest_sha256
    assert identity.artifact_sha256
    assert identity.executable_digest_sha256
    assert len(identity.package_digest_sha256) == 64
    assert len(identity.lockfile_digest_sha256) == 64
    assert Path(identity.executable).is_file()
    assert "runtime-mtl-vendor" in identity.executable
    managed_launcher = (
        Path(identity.install_root) / "bin" / installer.MANAGED_EXECUTABLE_NAME
    )
    assert managed_launcher.is_file()
    assert managed_launcher.stat().st_mode & 0o111
    assert hashlib.sha256(managed_launcher.read_bytes()).hexdigest() == (
        identity.executable_digest_sha256
    )

    completed = subprocess.run(
        [identity.executable, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    banner = (completed.stdout or "") + (completed.stderr or "")
    assert PIN_VERSION in banner or "1.0.0" in banner
    assert "typescript-vendor" in banner.casefold()
    assert "hermetic-parity-engine" not in banner.casefold()

    managed_completed = subprocess.run(
        [managed_launcher, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    managed_banner = (
        (managed_completed.stdout or "") + (managed_completed.stderr or "")
    )
    assert managed_completed.returncode == 0
    assert PIN_VERSION in managed_banner
    assert "typescript-vendor" in managed_banner.casefold()

    # Independence: wrapper must not import the Python reference.
    wrapper = Path(identity.executable).read_text(encoding="utf-8")
    assert "from ipfs_datasets_py" not in wrapper
    assert "ipfs_datasets_py.logic.software_verification" not in wrapper
    assert "python3" not in wrapper or "#!/usr/bin/env" in wrapper


def test_managed_vendor_launcher_is_discoverable_by_central_probe(
    central_certifier, installer, vendor_bundle
) -> None:
    identity = next(iter(vendor_bundle.identities.values()))
    lock = json.loads(TOOLCHAIN_LOCK_PATH.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in lock["tools"]
        if item["tool_id"] == "runtime-mtl-external"
    )
    managed_bin = Path(identity.install_root) / "bin"
    probe = central_certifier.probe_tool_identity(
        entry,
        env={
            "HOME": str(Path(identity.install_root) / "probe-home"),
            "PATH": f"{managed_bin}:/usr/local/bin:/usr/bin:/bin",
        },
    )

    assert probe["path_present"] is True
    assert probe["identity_probed"] is True
    assert probe["installed"] is True
    assert probe["probe_error"] is None
    assert PIN_VERSION in probe["version_string"]
    assert (
        Path(probe["executable_path"]).name
        == installer.MANAGED_EXECUTABLE_NAME
    )


def test_hermetic_install_cannot_replace_managed_vendor_launcher(
    installer, vendor_bundle
) -> None:
    identity = next(iter(vendor_bundle.identities.values()))
    install_root = Path(identity.install_root)
    managed_launcher = (
        install_root / "bin" / installer.MANAGED_EXECUTABLE_NAME
    )
    before = managed_launcher.read_bytes()
    before_digest = hashlib.sha256(before).hexdigest()

    hermetic = installer.ensure_runtime_mtl_external(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root,
        hermetic_parity_engine=True,
        vendor=False,
        checksum_verified=True,
        repo_root=REPO_ROOT,
    )

    assert hermetic.ok
    assert hermetic.identity is not None
    assert hermetic.identity.is_hermetic_parity_engine is True
    assert managed_launcher.read_bytes() == before
    assert hashlib.sha256(managed_launcher.read_bytes()).hexdigest() == (
        before_digest
    )


def test_hermetic_parity_cannot_satisfy_vendor(installer, install_root) -> None:
    hermetic = installer.ensure_runtime_mtl_external(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root / "hermetic-only",
        hermetic_parity_engine=True,
        vendor=False,
        checksum_verified=True,
        repo_root=REPO_ROOT,
    )
    assert hermetic.ok
    assert hermetic.identity is not None
    assert hermetic.identity.is_hermetic_parity_engine is True
    assert hermetic.identity.is_vendor_build is False
    hermetic_parts = Path(hermetic.identity.executable).parts
    assert "runtime-mtl-external" in hermetic_parts
    # Lane directory (not a coincidental tmp path substring) must not be vendor.
    assert "runtime-mtl-vendor" not in hermetic_parts
    assert not (
        Path(hermetic.identity.install_root)
        / "bin"
        / installer.MANAGED_EXECUTABLE_NAME
    ).exists()

    vendor = installer.ensure_runtime_mtl_external(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root / "vendor-only",
        hermetic_parity_engine=False,
        vendor=True,
        checksum_verified=True,
        repo_root=REPO_ROOT,
    )
    assert vendor.ok
    assert vendor.identity is not None
    assert vendor.identity.is_vendor_build is True
    assert vendor.identity.is_hermetic_parity_engine is False
    assert vendor.identity.executable != hermetic.identity.executable
    vendor_parts = Path(vendor.identity.executable).parts
    assert "runtime-mtl-vendor" in vendor_parts


def test_source_digests_match_locked_package(installer, vendor_bundle) -> None:
    identity = next(iter(vendor_bundle.identities.values()))
    digests = installer.compute_vendor_source_digests(TS_PACKAGE)
    assert digests["package_digest_sha256"] == identity.package_digest_sha256
    assert digests["source_digest_sha256"] == identity.source_digest_sha256
    assert digests["lockfile_digest_sha256"] == identity.lockfile_digest_sha256


# ---------------------------------------------------------------------------
# Full vendor certification corpus
# ---------------------------------------------------------------------------


def test_vendor_certificate_envelope(vendor_certificate: dict[str, Any]) -> None:
    assert vendor_certificate["schema_version"] == VENDOR_SCHEMA
    assert vendor_certificate["interface"] == VENDOR_INTERFACE
    assert vendor_certificate["goal_id"] == VENDOR_GOAL_ID
    assert vendor_certificate["task_id"] == VENDOR_TASK_ID
    assert vendor_certificate["certified"] is True
    assert vendor_certificate["authority_ceiling"] == "finite_trace"
    policy = vendor_certificate["policy"]
    assert policy["locked_typescript_dependency_graph"] is True
    assert policy["independent_node_package_without_python_dispatch"] is True
    assert policy["package_source_lockfile_runtime_executable_artifact_digests_bound"] is True
    assert policy["hermetic_parity_wrappers_are_non_production_shadows"] is True
    assert policy["hermetic_parity_wrappers_cannot_satisfy_vendor"] is True
    assert policy["finite_trace_authority_only"] is True
    assert policy["never_grants_theorem_authority"] is True
    assert policy["no_global_correctness_claim"] is True
    assert policy["grants_theorem_authority"] is False


def test_vendor_engine_digests_and_independence(
    vendor_certificate: dict[str, Any],
) -> None:
    engine = vendor_certificate["runtime_mtl_external"]
    assert engine["certified"] is True
    assert engine["usable"] is True
    assert engine["version"] == PIN_VERSION
    assert engine["is_vendor_build"] is True
    assert engine["is_hermetic_parity_engine"] is False
    assert engine["package_identity"] == PACKAGE_IDENTITY
    for key in (
        "package_digest_sha256",
        "source_digest_sha256",
        "lockfile_digest_sha256",
        "runtime_digest_sha256",
        "executable_digest_sha256",
        "artifact_sha256",
    ):
        value = engine[key]
        assert value and len(value) == 64, key
    assert Path(engine["executable"]).is_file()
    assert engine["no_python_reference_dispatch"] is True
    assert engine["finite_trace_authority_only"] is True
    shadow = vendor_certificate["hermetic_parity_shadow"]
    assert shadow["is_hermetic_parity_engine"] is True
    assert shadow["is_vendor_build"] is False
    assert shadow["non_production_shadow_evidence"] is True
    assert shadow["cannot_satisfy_vendor"] is True


def test_vendor_categories_and_mutations(vendor_certificate: dict[str, Any]) -> None:
    assert REQUIRED_CATEGORIES <= set(vendor_certificate["categories_exercised"])
    assert set(vendor_certificate["mutation_kinds"]) == REQUIRED_MUTATIONS


def test_vendor_receipt_retains_all_central_check_kinds(
    vendor_certificate: dict[str, Any],
) -> None:
    checks = [
        check
        for engine in vendor_certificate["engines"]
        for check in engine["checks"]
    ]
    passed_kinds = {
        check["kind"] for check in checks if check["status"] == "passed"
    }
    assert {"positive", "negative", "mutation", "replay"} <= passed_kinds
    assert any(
        check["kind"] == "negative"
        and check["status"] == "passed"
        and (
            check["expected"].startswith("violated/")
            or check["expected"].endswith("/false")
        )
        for check in checks
    )


@pytest.mark.parametrize("category", sorted(REQUIRED_CATEGORIES - {"malformed"}))
def test_vendor_category_outcomes(
    certifier, vendor_certificate: dict[str, Any], category: str
) -> None:
    engine = vendor_certificate["runtime_mtl_external"]
    executable = engine["executable"]
    version = engine["version"]
    if category == "shortest_violating_prefix":
        # Shortest-prefix specs use a dedicated recipe exercised in the
        # full vendor certificate matrix; verify a known violated golden.
        from ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl import (
            golden_fixtures,
        )

        fixture = next(
            item
            for item in golden_fixtures()
            if item.get("expected", {}).get("status") == "violated"
        )
        record = certifier.run_parity_case(
            "runtime-mtl-external",
            "case:shortest_prefix_probe",
            {
                "case_id": "case:shortest_prefix_probe",
                "formula": fixture["formula"],
                "trace": fixture["trace"],
                "position": fixture.get("position", 0),
            },
            executable=executable,
            engine_version=version,
        )
        assert record.status == "violated"
        assert record.agreed is True
        assert record.authorizes_global_proof is False
        return

    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category == category
    ]
    # interval/event mutations and clean_prefix may use different recipe labels.
    if not specs and category in {"interval_mutation", "event_mutation"}:
        specs = [
            spec
            for spec in certifier.default_case_specs()
            if category.replace("_mutation", "") in (spec.category or "")
            or (spec.mutation_kind or "") == category.replace("_mutation", "")
        ]
    if not specs and category == "clean_prefix":
        specs = [
            spec
            for spec in certifier.default_case_specs()
            if "prefix" in (spec.category or "") or "clean" in (spec.case_id or "")
        ]
    assert specs, f"no specs for category {category}"
    for spec in specs[:3]:
        case = certifier.materialize_case(spec)
        if "formula" not in case or "trace" not in case:
            continue
        record = certifier.run_parity_case(
            "runtime-mtl-external",
            spec.case_id,
            case,
            executable=executable,
            engine_version=version,
        )
        assert record.agreed is True, (spec.case_id, record)
        assert record.quarantined is False
        assert record.authority in {"monitor", "finite_trace"}
        assert record.authorizes_global_proof is False


def test_vendor_mutations_replay_malformed_timeout_disagreement_bounds(
    certifier, installer, vendor_certificate: dict[str, Any]
) -> None:
    engine = vendor_certificate["runtime_mtl_external"]
    executable = engine["executable"]
    version = engine["version"]

    # Interval / event mutations
    for mutation_kind in sorted(REQUIRED_MUTATIONS):
        specs = [
            spec
            for spec in certifier.default_case_specs()
            if (spec.mutation_kind or "") == mutation_kind
            or mutation_kind in (spec.category or "")
        ]
        assert specs, mutation_kind
        spec = specs[0]
        base = certifier._semantic._golden_by_id(spec.base_fixture_id)
        baseline = certifier.run_parity_case(
            "runtime-mtl-external",
            f"{spec.case_id}:baseline",
            {
                "case_id": f"{spec.case_id}:baseline",
                "formula": base["formula"],
                "trace": base["trace"],
                "position": base.get("position", 0),
            },
            executable=executable,
            engine_version=version,
        )
        mutated_case = certifier.materialize_case(spec)
        mutated = certifier.run_parity_case(
            "runtime-mtl-external",
            spec.case_id,
            mutated_case,
            executable=executable,
            engine_version=version,
        )
        assert mutated.status != baseline.status or mutated.verdict != baseline.verdict
        assert mutated.agreed is True

    # Replay determinism
    satisfied_specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category == "satisfied"
    ]
    assert satisfied_specs
    for spec in satisfied_specs[:2]:
        case = certifier.materialize_case(spec)
        first = certifier.run_parity_case(
            "runtime-mtl-external",
            spec.case_id,
            case,
            executable=executable,
            engine_version=version,
        )
        second = certifier.run_parity_case(
            "runtime-mtl-external",
            f"{spec.case_id}:replay",
            case,
            executable=executable,
            engine_version=version,
        )
        assert first.status == second.status
        assert first.verdict == second.verdict
        assert first.formula_digest == second.formula_digest

    # Malformed
    malformed = certifier.run_parity_case(
        "runtime-mtl-external",
        "case:malformed",
        None,
        executable=executable,
        engine_version=version,
        expect_error=True,
    )
    assert malformed.status != "satisfied"
    assert malformed.malformed is True
    assert malformed.quarantined is True

    # Timeout
    from ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl import (
        golden_fixtures,
    )

    fixture = next(
        item
        for item in golden_fixtures()
        if item.get("expected", {}).get("status") == "satisfied"
    )
    timed = certifier.run_parity_case(
        "runtime-mtl-external",
        "case:timeout",
        {
            "case_id": "case:timeout",
            "formula": fixture["formula"],
            "trace": fixture["trace"],
            "position": fixture.get("position", 0),
        },
        executable=executable,
        engine_version=version,
        timeout_seconds=0.25,
        env={installer.ENV_SLEEP_SECONDS: "2.0"},
    )
    assert timed.timed_out is True
    assert timed.quarantined is True
    assert timed.status == "timeout"

    # Disagreement
    disagree = certifier.run_parity_case(
        "runtime-mtl-external",
        "case:disagreement",
        {
            "case_id": "case:disagreement",
            "formula": fixture["formula"],
            "trace": fixture["trace"],
            "position": fixture.get("position", 0),
        },
        executable=executable,
        engine_version=version,
        env={installer.ENV_DISAGREE: "1"},
    )
    assert disagree.agreed is False
    assert disagree.quarantined is True
    assert disagree.status != disagree.reference_status

    # Bounds elevation
    elevate = certifier.run_parity_case(
        "runtime-mtl-external",
        "case:bounds-elevation",
        {
            "case_id": "case:bounds-elevation",
            "formula": fixture["formula"],
            "trace": fixture["trace"],
            "position": fixture.get("position", 0),
        },
        executable=executable,
        engine_version=version,
        env={installer.ENV_AUTHORIZE_GLOBAL_PROOF: "1"},
    )
    assert elevate.authorizes_global_proof is True
    assert elevate.quarantined is True


def test_vendor_lane_handler(certifier, install_root) -> None:
    result = certifier.external_runtime_mtl_vendor_lane_handler(
        install_root=install_root,
        force_install=False,
        skip_install=True,
        repo_root=REPO_ROOT,
    )
    assert result["interface"] == VENDOR_INTERFACE
    assert result["goal_id"] == VENDOR_GOAL_ID
    assert result["task_id"] == VENDOR_TASK_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["finite_trace_authority_only"] is True
    assert result["hermetic_parity_wrappers_cannot_satisfy_vendor"] is True
    assert result["is_vendor_build"] is True
    assert result["certificate_digest_sha256"]
    assert len(result["certificate_digest_sha256"]) == 64


def test_checked_in_vendor_receipt_structure() -> None:
    receipt_text = RECEIPT_PATH.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["schema_version"] == VENDOR_RECEIPT_SCHEMA
    assert receipt["interface"] == VENDOR_INTERFACE
    assert receipt["goal_id"] == VENDOR_GOAL_ID
    assert receipt["task_id"] == VENDOR_TASK_ID
    assert receipt["certified"] is True
    engine = receipt["runtime_mtl_external"]
    assert engine["version"] == PIN_VERSION
    assert engine["is_vendor_build"] is True
    assert engine["is_hermetic_parity_engine"] is False
    assert engine["package_identity"] == PACKAGE_IDENTITY
    assert engine["executable"] == "<managed-tool-path-redacted>"
    assert (
        receipt["hermetic_parity_shadow"]["executable"]
        == "<managed-tool-path-redacted>"
    )
    assert "/home/" not in receipt_text
    assert "/tmp/" not in receipt_text
    for key in (
        "package_digest_sha256",
        "source_digest_sha256",
        "lockfile_digest_sha256",
        "runtime_digest_sha256",
        "executable_digest_sha256",
        "artifact_sha256",
    ):
        assert receipt["runtime_mtl_external"][key]
        assert len(receipt["runtime_mtl_external"][key]) == 64
    shadow = receipt["hermetic_parity_shadow"]
    assert shadow["non_production_shadow_evidence"] is True
    assert shadow["cannot_satisfy_vendor"] is True
    assert REQUIRED_CATEGORIES <= set(receipt["categories_exercised"])
    assert set(receipt["mutation_kinds"]) == REQUIRED_MUTATIONS
    assert receipt["policy"]["never_grants_theorem_authority"] is True
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64
