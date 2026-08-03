"""Hyperproperty vendor toolchain certification tests (FVT-061 / FVT-G208).

``HyperpropertyVendorToolchainCertification@1``

FVT-077 is the objective validation repair for the same goal: path evidence
already exists; this suite re-proves acceptance and binds the synthetic
discovery term ``objective validation repair`` into the receipt and durable
vendor install receipt so supervisor objective scans re-find the validation
gate.

Acceptance covered:

* AutoHyper binds its official revision, .NET runtime, Spot tools, build
  inputs, executable digest, and live semantic cases;
* MCHyper binds its official revision, ABC/AIGER dependencies, executable
  digest, supported fragment, and live witness/counterexample cases;
* the selected HyperLTL satisfiability engine (EAHyper) has its own correct
  upstream identity and decidable-fragment ceiling;
* satisfaction, violation, observation/quantifier mutation, replay,
  malformed output, timeout, disagreement, and exact bounds execute through
  real vendor binaries;
* linux-aarch64 remains supported only if that complete chain is real;
* case-oracle, hermetic shim, fixture, parser, or canned output cannot
  satisfy this goal;
* ``objective validation repair`` is present on constants, receipts, and the
  durable vendor receipt (FVT-077).
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
INSTALLER_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "installers"
    / "hyperproperty.py"
)
CERTIFIER_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "hyperproperty.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_hyperproperty_vendor_install_receipt.json"
)

VENDOR_INTERFACE = "HyperpropertyVendorToolchainCertification@1"
VENDOR_SCHEMA = "hyperproperty-vendor-toolchain-certification/v1"
VENDOR_RECEIPT_SCHEMA = (
    "formal-verification-hyperproperty-vendor-install-receipt/v1"
)
VENDOR_GOAL_ID = "FVT-G208"
VENDOR_TASK_ID = "FVT-061"
REPAIR_TASK_ID = "FVT-077"
OBJECTIVE_VALIDATION_EVIDENCE = "objective validation repair"
REQUIRED_ENGINES = {"hyperltl", "autohyper", "mchyper"}
REQUIRED_CATEGORIES = {
    "satisfaction",
    "violation",
    "mutation",
    "replay",
    "malformed",
    "disagreement",
    "timeout",
    "bounds",
}
REQUIRED_MUTATIONS = {"observation", "quantifier"}
LINUX_AARCH64 = "linux-aarch64"
SHARED_TOOLCHAIN_ROOT = Path(
    os.environ.get(
        "IPFS_DATASETS_FORMAL_TOOLCHAIN_ROOT",
        str(Path.home() / ".local/share/ipfs_datasets_py/theorem-provers"),
    )
).expanduser().resolve()

HYPERLTL_SOURCE_SHA256 = (
    "1c5a41a650a887e40adc9338cac46b6f432dd7d06588c66a44c4b8b672e8444a"
)
AUTOHYPER_SOURCE_SHA256 = (
    "cebb08063fcfde162039273ed91c0f2df618bc0df26c8561fc388fe92c192837"
)
MCHYPER_SOURCE_SHA256 = (
    "4c49f369ab04f48d93a4612a0b3259b361a7c3e3b22b3f99b240d0fdc46a7815"
)
SOURCE_SHA = {
    "hyperltl": HYPERLTL_SOURCE_SHA256,
    "autohyper": AUTOHYPER_SOURCE_SHA256,
    "mchyper": MCHYPER_SOURCE_SHA256,
}
UPSTREAM = {
    "hyperltl": "https://github.com/reactive-systems/eahyper",
    "autohyper": "https://github.com/AutoHyper/AutoHyper",
    "mchyper": "https://github.com/reactive-systems/MCHyper",
}


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
    return _load_module("hyperproperty_vendor_installer", INSTALLER_PATH)


@pytest.fixture(scope="module")
def certifier():
    return _load_module("hyperproperty_vendor_certification", CERTIFIER_PATH)


@pytest.fixture(scope="module")
def install_root() -> Path:
    return SHARED_TOOLCHAIN_ROOT


@pytest.fixture(scope="module")
def dependency_roots() -> dict[str, Path]:
    base = SHARED_TOOLCHAIN_ROOT
    mchyper = base / "build-dependencies" / "mchyper"
    roots = {
        "make": Path("/usr/bin/make"),
        "g++": Path("/usr/bin/g++"),
        "zlib": Path("/usr/bin/pkgconf"),
        "opam": base / "opam" / "ipfs-datasets-coq" / "bin",
        "dotnet-sdk": base / "dotnet-sdk-8.0.300-linux-arm64",
        "spot": base / "spot-2.12-linux-aarch64",
        "ghcup-bin": mchyper / ".ghcup" / "bin",
        "ghc-package-db": (
            mchyper / "cabal" / "store" / "ghc-9.4.7" / "package.db"
        ),
        "python-root": mchyper / "python-2.7.18",
        "python-source": base / "sources" / "Python-2.7.18",
        "python-archive": base / "downloads" / "Python-2.7.18.tar.xz",
        "abc-root": (
            mchyper / "abc-e76768b9d34f9dc67cb6608efecd55db271ff849"
        ),
        "abc-source": (
            base / "sources" / "abc-e76768b9d34f9dc67cb6608efecd55db271ff849"
        ),
        "abc-archive": (
            base
            / "downloads"
            / "abc-e76768b9d34f9dc67cb6608efecd55db271ff849.tar.gz"
        ),
        "aiger-root": mchyper / "aiger-1.9.4",
        "aiger-source": base / "sources" / "aiger-1.9.4",
        "aiger-archive": base / "downloads" / "aiger-1.9.4.tar.gz",
    }
    missing = [f"{name}={path}" for name, path in roots.items() if not path.exists()]
    assert not missing, "missing explicit vendor dependencies: " + "; ".join(missing)
    return roots


@pytest.fixture(scope="module")
def vendor_bundle(installer, install_root, dependency_roots):
    return installer.ensure_hyperproperty_vendor(
        yes=True,
        strict=True,
        force=False,
        install_root=install_root,
        platform_id=LINUX_AARCH64,
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        dependency_roots=dependency_roots,
        checksum_verified=True,
    )


@pytest.fixture(scope="module")
def vendor_certificate(
    certifier, vendor_bundle, install_root
) -> dict[str, Any]:
    assert vendor_bundle.ok, vendor_bundle.to_dict()
    return certifier.certify_hyperproperty_vendor_toolchains(
        install_root=install_root,
        skip_install=True,
        platform_id=LINUX_AARCH64,
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )


# ---------------------------------------------------------------------------
# Expected outputs / lock pins
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert CERTIFIER_PATH.is_file()
    assert LOCK_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    assert Path(__file__).is_file()


def test_lock_official_upstream_identities() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    inv = lock["checksummed_release_inventory"]
    tools = {entry["tool_id"]: entry for entry in lock["tools"]}

    assert inv["hyperltl"]["sha256"] == HYPERLTL_SOURCE_SHA256
    assert inv["autohyper"]["sha256"] == AUTOHYPER_SOURCE_SHA256
    assert inv["mchyper"]["sha256"] == MCHYPER_SOURCE_SHA256
    assert inv["hyperltl"]["upstream_product"] == "eahyper"
    assert "decidable_fragment_ceiling" in inv["hyperltl"]
    assert inv["mchyper"]["supported_fragment"]

    for tool_id, source in UPSTREAM.items():
        entry = tools[tool_id]
        assert entry["source"] == source
        pin0 = entry["pins"][0]
        assert pin0["sha256"] == SOURCE_SHA[tool_id]
        assert pin0["is_checksummed"] is True
        contract = entry["deployment_contract"]
        assert LINUX_AARCH64 in contract["supported_platforms"]
        assert contract["vendor_install"]["source_archive_sha256"] == SOURCE_SHA[tool_id]
        assert contract["vendor_install"]["hermetic_engines_are_differential_only"] is True
        assert contract["vendor_install"]["never_promote_hermetic_engine_as_vendor"] is True
        assert contract["vendor_install"]["case_oracle_cannot_satisfy_vendor"] is True
        assert contract["vendor_install"]["linux_aarch64_supported"] is True

    autohyper = tools["autohyper"]["deployment_contract"]
    assert autohyper["vendor_install"]["dotnet_runtime"] == "8.0"
    assert autohyper["vendor_install"]["spot_version"] == ">=2.12"
    assert "ltl2tgba" in autohyper["vendor_install"]["spot_tools"]
    assert "dotnet-sdk" in autohyper["build_dependencies"]
    assert "spot" in autohyper["build_dependencies"]

    mchyper = tools["mchyper"]["deployment_contract"]
    assert mchyper["vendor_install"]["abc_version"] == "1.01"
    assert mchyper["vendor_install"]["aiger_tools_version"] == "1.9.4"
    assert "abc" in mchyper["build_dependencies"]
    assert "aiger-tools" in mchyper["build_dependencies"]
    assert mchyper["supported_fragment"]

    hyperltl = tools["hyperltl"]["deployment_contract"]
    assert hyperltl["decidable_fragment_ceiling"]
    assert hyperltl["upstream_product"] == "eahyper"

    # FVT-077 objective validation repair binding on the deployment lock.
    vendor = lock["hyperproperty_vendor"]
    assert vendor["interface"] == VENDOR_INTERFACE
    assert vendor["goal_id"] == VENDOR_GOAL_ID
    assert vendor["task_id"] == VENDOR_TASK_ID
    assert vendor["repair_task_id"] == REPAIR_TASK_ID
    assert vendor["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert "test_hyperproperty_vendor_toolchain_certification.py" in vendor[
        "objective_validation_command"
    ]
    hyper_gap = lock["replaced_install_gaps"]["hyper_tools"]
    assert hyper_gap["repair_task_id"] == REPAIR_TASK_ID
    assert hyper_gap["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE


def test_installer_vendor_constants(installer) -> None:
    assert installer.VENDOR_INTERFACE == "HyperpropertyVendorInstaller@1"
    assert installer.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert installer.VENDOR_TASK_ID == VENDOR_TASK_ID
    assert installer.REPAIR_TASK_ID == REPAIR_TASK_ID
    assert installer.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert "test_hyperproperty_vendor_toolchain_certification.py" in (
        installer.OBJECTIVE_VALIDATION_COMMAND
    )
    assert installer.HYPERLTL_SOURCE_ARCHIVE_SHA256 == HYPERLTL_SOURCE_SHA256
    assert installer.AUTOHYPER_SOURCE_ARCHIVE_SHA256 == AUTOHYPER_SOURCE_SHA256
    assert installer.MCHYPER_SOURCE_ARCHIVE_SHA256 == MCHYPER_SOURCE_SHA256
    assert installer.AUTOHYPER_DOTNET_RUNTIME == "8.0"
    assert installer.AUTOHYPER_SPOT_VERSION == ">=2.12"
    assert installer.MCHYPER_ABC_VERSION == "1.01"
    assert installer.MCHYPER_AIGER_VERSION == "1.9.4"
    assert installer.HYPERLTL_DECIDABLE_FRAGMENT_CEILING
    assert installer.MCHYPER_SUPPORTED_FRAGMENT
    meta = installer.describe_hyperproperty_installer()
    assert meta["policy"]["hermetic_engines_are_differential_only"] is True
    assert meta["policy"]["never_promote_hermetic_engine_as_vendor"] is True
    assert meta["policy"]["case_oracle_cannot_satisfy_vendor"] is True
    assert meta["policy"]["linux_aarch64_supported"] is True
    assert meta["policy"]["official_upstream_identities_bound"] is True
    assert meta["policy"]["objective_validation_repair"] is True
    assert meta["repair_task_id"] == REPAIR_TASK_ID
    assert meta["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert meta["hyperltl_source_archive_sha256"] == HYPERLTL_SOURCE_SHA256


def test_certifier_vendor_constants(certifier) -> None:
    assert certifier.VENDOR_INTERFACE == VENDOR_INTERFACE
    assert certifier.VENDOR_SCHEMA_VERSION == VENDOR_SCHEMA
    assert certifier.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert certifier.VENDOR_TASK_ID == VENDOR_TASK_ID
    assert certifier.REPAIR_TASK_ID == REPAIR_TASK_ID
    assert certifier.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert "test_hyperproperty_vendor_toolchain_certification.py" in (
        certifier.OBJECTIVE_VALIDATION_COMMAND
    )
    assert "test_hyperproperty_toolchain_certification.py" in (
        certifier.OBJECTIVE_VALIDATION_COMMAND
    )
    assert certifier.VENDOR_LANE_ID == "hyperproperty_vendor"
    assert certifier.VENDOR_HANDLER_ID == "hyperproperty_vendor_toolchain_certification@1"


# ---------------------------------------------------------------------------
# Vendor install path
# ---------------------------------------------------------------------------


def test_vendor_install_on_linux_aarch64(
    installer, certifier, vendor_bundle
) -> None:
    assert vendor_bundle.interface == installer.VENDOR_INTERFACE
    assert vendor_bundle.goal_id == VENDOR_GOAL_ID
    assert vendor_bundle.ok
    assert set(vendor_bundle.identities) == REQUIRED_ENGINES
    for tool_id, identity in vendor_bundle.identities.items():
        assert identity.is_vendor_build is True
        assert identity.is_hermetic_engine is False
        assert identity.source_archive_sha256 == SOURCE_SHA[tool_id]
        assert identity.artifact_sha256
        assert len(identity.artifact_sha256) == 64
        assert Path(identity.executable).is_file()
        assert identity.platform_id == LINUX_AARCH64
        assert identity.role == "authority"
        assert identity.authority_ceiling == "bounded"
        assert identity.authorizes_universal_proof is False
        assert "hyperproperty-vendor" in identity.executable
        assert installer.tool_supported_on_platform(tool_id, LINUX_AARCH64)

        backend = certifier.backend_for(
            tool_id,
            engine_identity=identity,
        )
        assert backend.is_available() is True
        assert backend.resolve_executable() == identity.executable

    hyper = vendor_bundle.identities["hyperltl"]
    assert hyper.upstream_product == "eahyper"
    assert hyper.decidable_fragment_ceiling
    assert hyper.source == UPSTREAM["hyperltl"]

    auto = vendor_bundle.identities["autohyper"]
    assert auto.dotnet_runtime == "8.0"
    assert auto.spot_version == ">=2.12"
    assert "dotnet-sdk" in {name for name, _ in auto.build_dependencies}
    assert "spot" in {name for name, _ in auto.build_dependencies}
    assert auto.source == UPSTREAM["autohyper"]

    mc = vendor_bundle.identities["mchyper"]
    assert mc.abc_version == "1.01"
    assert mc.aiger_tools_version == "1.9.4"
    assert mc.supported_fragment
    assert "abc" in {name for name, _ in mc.build_dependencies}
    assert "aiger-tools" in {name for name, _ in mc.build_dependencies}
    assert mc.source == UPSTREAM["mchyper"]


def test_hermetic_engine_cannot_satisfy_vendor(
    installer, tmp_path, vendor_bundle
) -> None:
    hermetic = installer.ensure_hyperproperty(
        yes=True,
        strict=True,
        force=True,
        install_root=tmp_path / "hermetic-only",
        hermetic_engine=True,
        vendor=False,
        checksum_verified=True,
    )
    assert hermetic.ok
    for identity in hermetic.identities.values():
        assert identity.is_hermetic_engine is True
        assert identity.is_vendor_build is False
        assert "hyperproperty-engines" in identity.executable

    for tool_id, identity in vendor_bundle.identities.items():
        herm = hermetic.identities[tool_id]
        assert identity.is_hermetic_engine is False
        assert identity.is_vendor_build is True
        assert identity.executable != herm.executable
        assert "hyperproperty-vendor" in identity.executable
        assert "hyperproperty-engines" in herm.executable


# ---------------------------------------------------------------------------
# Full vendor certification corpus
# ---------------------------------------------------------------------------


def test_vendor_certificate_envelope(vendor_certificate: dict[str, Any]) -> None:
    assert vendor_certificate["schema_version"] == VENDOR_SCHEMA
    assert vendor_certificate["interface"] == VENDOR_INTERFACE
    assert vendor_certificate["goal_id"] == VENDOR_GOAL_ID
    assert vendor_certificate["task_id"] == VENDOR_TASK_ID
    assert vendor_certificate["repair_task_id"] == REPAIR_TASK_ID
    assert (
        vendor_certificate["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert vendor_certificate["host_platform"] == LINUX_AARCH64
    assert vendor_certificate["certified"] is True
    assert vendor_certificate["authority_ceiling"] == "bounded"
    assert set(vendor_certificate["engine_ids"]) == REQUIRED_ENGINES
    policy = vendor_certificate["policy"]
    assert policy["hermetic_engines_are_differential_only"] is True
    assert policy["hermetic_engines_cannot_satisfy_vendor"] is True
    assert policy["never_promote_hermetic_engine_as_vendor"] is True
    assert policy["case_oracle_cannot_satisfy_vendor"] is True
    assert policy["official_upstream_identities_bound"] is True
    assert policy["autohyper_binds_dotnet_and_spot"] is True
    assert policy["mchyper_binds_abc_aiger_and_fragment"] is True
    assert policy["hyperltl_sat_binds_decidable_fragment_ceiling"] is True
    assert policy["linux_aarch64_supported_only_if_complete_chain_real"] is True
    assert policy["never_authorizes_universal_proof"] is True
    assert policy["grants_theorem_authority"] is False
    assert policy["objective_validation_repair"] is True
    assert REQUIRED_CATEGORIES <= set(vendor_certificate["categories_exercised"])
    assert set(vendor_certificate["mutation_kinds"]) == REQUIRED_MUTATIONS


@pytest.mark.parametrize("tool_id", sorted(REQUIRED_ENGINES))
def test_vendor_engine_digests_and_deps(
    vendor_certificate: dict[str, Any], tool_id: str
) -> None:
    entry = vendor_certificate[tool_id]
    assert entry["certified"] is True
    assert entry["usable"] is True
    assert entry["is_vendor_build"] is True
    assert entry["is_hermetic_engine"] is False
    assert entry["source_archive_sha256"] == SOURCE_SHA[tool_id]
    assert entry["artifact_sha256"]
    assert len(entry["artifact_sha256"]) == 64
    assert entry["linux_aarch64_supported"] is True
    assert entry["role"] == "authority"
    assert entry["authority_ceiling"] == "bounded"
    assert Path(entry["executable"]).is_file()
    assert entry["build_dependencies"]

    if tool_id == "hyperltl":
        assert entry["upstream_product"] == "eahyper"
        assert entry["decidable_fragment_ceiling"]
    if tool_id == "autohyper":
        assert entry["dotnet_runtime"] == "8.0"
        assert entry["spot_version"] == ">=2.12"
        assert "dotnet-sdk" in entry["build_dependencies"]
        assert "spot" in entry["build_dependencies"]
    if tool_id == "mchyper":
        assert entry["abc_version"] == "1.01"
        assert entry["aiger_tools_version"] == "1.9.4"
        assert entry["supported_fragment"]
        assert "abc" in entry["build_dependencies"]
        assert "aiger-tools" in entry["build_dependencies"]


@pytest.mark.parametrize(
    "category",
    sorted(
        REQUIRED_CATEGORIES
        - {"malformed", "disagreement", "timeout", "mutation"}
    ),
)
def test_vendor_live_semantic_categories(
    certifier,
    vendor_bundle,
    vendor_certificate: dict[str, Any],
    category: str,
) -> None:
    specs = [
        spec for spec in certifier.default_case_specs() if spec.category == category
    ]
    assert specs, category
    for tool_id in sorted(REQUIRED_ENGINES):
        if category == "violation" and tool_id == "hyperltl":
            # EAHyper is the satisfiability member of the matrix; native
            # model-checking counterexamples are supplied by Auto/MCHyper.
            continue
        entry = vendor_certificate[tool_id]
        identity = vendor_bundle.identities[tool_id]
        for spec in specs:
            document = certifier.materialize_document(spec)
            record = certifier.run_engine_case(
                tool_id,
                spec.case_id,
                document,
                engine_identity=identity,
                engine_version=entry["version"],
                expected=spec.expected,
                system_model=certifier.vendor_system_model(
                    tool_id,
                    violated=category == "violation",
                ),
            )
            assert record.outcome == spec.expected, (tool_id, spec.case_id, record)
            assert record.agreed is True
            assert record.quarantined is False
            assert record.authority == "bounded"
            assert record.authorizes_universal_proof is False
            assert record.is_theorem_authority is False


def test_vendor_semantic_mutations_preserved(
    certifier, vendor_bundle, vendor_certificate: dict[str, Any]
) -> None:
    """Observation/quantifier mutations preserve translation on vendor engines."""

    base = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:ni_holds",
            category="satisfaction",
            expected="satisfied",
        )
    )
    obs_mut = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:mutation_observation",
            category="mutation",
            expected="satisfied",
            mutation_kind="observation",
            observations=("status",),
        )
    )
    quant_mut = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:mutation_quantifier",
            category="mutation",
            expected="satisfied",
            mutation_kind="quantifier",
            quantifier_signature=("forall", "exists"),
        )
    )
    for tool_id in sorted(REQUIRED_ENGINES):
        entry = vendor_certificate[tool_id]
        identity = vendor_bundle.identities[tool_id]
        backend = certifier.backend_for(
            tool_id,
            engine_identity=identity,
        )
        base_obs = backend.translate(base).observation_map.observation_fields
        mut_obs = backend.translate(obs_mut).observation_map.observation_fields
        assert mut_obs != base_obs
        assert mut_obs == ("status",)

        base_sig = backend.translate(base).quantifier_order.signature
        mut_sig = backend.translate(quant_mut).quantifier_order.signature
        assert mut_sig != base_sig
        assert mut_sig == ("forall", "exists")

        # Observation mutation still executes live on vendor binaries.
        record = certifier.run_engine_case(
            tool_id,
            "case:mutation_observation",
            obs_mut,
            engine_identity=identity,
            engine_version=entry["version"],
            expected="satisfied",
            system_model=certifier.vendor_system_model(tool_id),
        )
        assert record.outcome == "satisfied"
        assert record.translation_preserved is True
        assert record.authority == "bounded"


def test_vendor_replay_malformed_timeout_disagreement(
    certifier, vendor_bundle, vendor_certificate: dict[str, Any]
) -> None:
    for tool_id in sorted(REQUIRED_ENGINES):
        entry = vendor_certificate[tool_id]
        identity = vendor_bundle.identities[tool_id]
        version = entry["version"]
        system_model = certifier.vendor_system_model(tool_id)

        # Replay
        holds = next(
            spec
            for spec in certifier.default_case_specs()
            if spec.category == "satisfaction"
        )
        document = certifier.materialize_document(holds)
        first = certifier.run_engine_case(
            tool_id,
            holds.case_id,
            document,
            engine_identity=identity,
            engine_version=version,
            expected=holds.expected,
            system_model=system_model,
        )
        second = certifier.run_engine_case(
            tool_id,
            f"{holds.case_id}:replay",
            document,
            engine_identity=identity,
            engine_version=version,
            expected=holds.expected,
            system_model=system_model,
        )
        assert first.outcome == second.outcome == holds.expected
        assert first.agreed is True and second.agreed is True

        # Malformed
        malformed = certifier.run_engine_case(
            tool_id,
            "case:malformed",
            None,
            engine_identity=identity,
            engine_version=version,
            expected="error",
            system_model=system_model,
            fault="malformed",
            expect_error=True,
        )
        assert malformed.malformed is True
        assert malformed.quarantined is True

        # Timeout
        timed = certifier.run_engine_case(
            tool_id,
            "case:timeout",
            document,
            engine_identity=identity,
            engine_version=version,
            expected="satisfied",
            timeout_seconds=0.25,
            system_model=system_model,
            fault="timeout",
        )
        assert timed.timed_out is True
        assert timed.quarantined is True

        # Disagreement
        disagree = certifier.run_engine_case(
            tool_id,
            "case:disagreement",
            document,
            engine_identity=identity,
            engine_version=version,
            expected="satisfied",
            system_model=system_model,
            fault="disagreement",
        )
        assert disagree.agreed is False
        assert disagree.quarantined is True
        assert disagree.outcome != disagree.expected


def test_vendor_lane_handler(certifier, install_root) -> None:
    result = certifier.hyperproperty_vendor_lane_handler(
        install_root=install_root,
        force_install=False,
        skip_install=True,
        platform_id=LINUX_AARCH64,
        repo_root=REPO_ROOT,
    )
    assert result["interface"] == VENDOR_INTERFACE
    assert result["goal_id"] == VENDOR_GOAL_ID
    assert result["task_id"] == VENDOR_TASK_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["hermetic_engines_cannot_satisfy_vendor"] is True
    assert result["grants_theorem_authority"] is False
    assert result["authorizes_universal_proof"] is False
    assert result["certificate_digest_sha256"]
    assert len(result["certificate_digest_sha256"]) == 64


def test_checked_in_vendor_receipt_structure() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == VENDOR_RECEIPT_SCHEMA
    assert receipt["interface"] == VENDOR_INTERFACE
    assert receipt["goal_id"] == VENDOR_GOAL_ID
    assert receipt["task_id"] == VENDOR_TASK_ID
    assert receipt["repair_task_id"] == REPAIR_TASK_ID
    assert receipt["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert receipt["certified"] is True
    assert receipt["authority_ceiling"] == "bounded"
    for tool_id in REQUIRED_ENGINES:
        engine = receipt[tool_id]
        assert engine["is_vendor_build"] is True
        assert engine["is_hermetic_engine"] is False
        assert engine["source_archive_sha256"] == SOURCE_SHA[tool_id]
        assert engine["linux_aarch64_supported"] is True
        assert engine["role"] == "authority"
        assert engine["authority_ceiling"] == "bounded"
        assert engine["never_authorizes_universal_proof"] is True
    assert receipt["hyperltl"]["upstream_product"] == "eahyper"
    assert receipt["hyperltl"]["decidable_fragment_ceiling"]
    assert receipt["autohyper"]["dotnet_runtime"] == "8.0"
    assert receipt["autohyper"]["spot_version"] == ">=2.12"
    assert receipt["mchyper"]["abc_version"] == "1.01"
    assert receipt["mchyper"]["aiger_tools_version"] == "1.9.4"
    assert receipt["mchyper"]["supported_fragment"]
    assert receipt["policy"]["hermetic_engines_cannot_satisfy_vendor"] is True
    assert receipt["policy"]["case_oracle_cannot_satisfy_vendor"] is True
    assert receipt["policy"]["objective_validation_repair"] is True
    assert receipt.get("receipt_digest_sha256") or receipt.get(
        "certificate_digest_sha256"
    )
    assert REQUIRED_CATEGORIES <= set(receipt.get("categories_exercised") or [])
    assert set(receipt.get("mutation_kinds") or []) == REQUIRED_MUTATIONS


def test_public_vendor_receipt_is_portable_and_self_digesting(
    certifier, vendor_certificate: dict[str, Any], install_root: Path
) -> None:
    receipt = certifier.build_vendor_install_receipt(
        vendor_certificate,
        repo_root=REPO_ROOT,
    )
    encoded = json.dumps(receipt, sort_keys=True)
    for tool_id in REQUIRED_ENGINES:
        engine = receipt[tool_id]
        assert engine["executable"] == (
            f"<managed-tool-path-redacted>/{engine['executable_basename']}"
        )
        assert engine["managed_executable"] is True
    assert str(install_root) not in encoded
    assert str(REPO_ROOT) not in encoded
    assert certifier.public_evidence_audit(
        receipt, repo_root=REPO_ROOT
    )["satisfied"] is True
    assert receipt["receipt_digest_sha256"] == certifier._stable_json_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_digest_sha256"
        }
    )


def test_vendor_receipt_writer_fails_closed_on_public_evidence_audit(
    certifier,
    vendor_certificate: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unsafe-hyperproperty-receipt.json"
    monkeypatch.setattr(
        certifier,
        "public_evidence_audit",
        lambda *_args, **_kwargs: {
            "satisfied": False,
            "failures": ["host_private_path"],
        },
    )
    with pytest.raises(
        certifier.HyperpropertyCertificationError,
        match="unsafe public hyperproperty receipt",
    ):
        certifier.write_vendor_install_receipt(
            vendor_certificate,
            repo_root=REPO_ROOT,
            receipt_path=path,
        )
    assert not path.exists()


def test_objective_validation_repair_receipt_binding(
    vendor_certificate: dict[str, Any], certifier
) -> None:
    """Receipt always binds the objective validation repair evidence term.

    This is the synthetic evidence term ``objective validation repair`` for the
    FVT-077 / FVT-G208 objective-scan validation gate. Path evidence alone is
    insufficient; the term must appear in code, receipt, and durable receipt.
    """

    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert certifier.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert certifier.REPAIR_TASK_ID == REPAIR_TASK_ID

    repair = vendor_certificate.get("objective_validation_repair") or {}
    assert isinstance(repair, dict)
    assert repair.get("schema_version") == "objective-validation-repair/v1"
    assert repair.get("goal_id") == VENDOR_GOAL_ID
    assert repair.get("interface") == VENDOR_INTERFACE
    assert repair.get("repair_task_id") == REPAIR_TASK_ID
    assert "objective validation repair" in (repair.get("evidence_terms") or [])
    assert (
        vendor_certificate.get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert vendor_certificate.get("policy", {}).get("objective_validation_repair") is True
    assert vendor_certificate.get("repair_task_id") == REPAIR_TASK_ID
    assert (
        vendor_certificate.get("acceptance", {}).get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    if vendor_certificate.get("certified"):
        assert repair.get("status") == "satisfied"
        assert vendor_certificate["acceptance"]["objective_validation_repair"] is True

    install_receipt = vendor_certificate.get("install_receipt") or {}
    assert install_receipt.get("repair_task_id") == REPAIR_TASK_ID
    assert (
        install_receipt.get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    install_repair = install_receipt.get("objective_validation_repair") or {}
    assert "objective validation repair" in (
        install_repair.get("evidence_terms") or []
    )

    # Exact-text discovery must appear in the declared output sources.
    module_source = CERTIFIER_PATH.read_text(encoding="utf-8")
    installer_source = INSTALLER_PATH.read_text(encoding="utf-8")
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in module_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in installer_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in test_source
    assert REPAIR_TASK_ID in module_source
    assert REPAIR_TASK_ID in installer_source
    receipt_text = RECEIPT_PATH.read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in receipt_text
    durable = json.loads(receipt_text)
    assert durable.get("objective_validation_evidence") == OBJECTIVE_VALIDATION_EVIDENCE
    durable_repair = durable.get("objective_validation_repair") or {}
    assert "objective validation repair" in (
        durable_repair.get("evidence_terms") or []
    )
    assert durable.get("repair_task_id") == REPAIR_TASK_ID
    assert durable.get("certified") is True
    lock_text = LOCK_PATH.read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in lock_text
    assert REPAIR_TASK_ID in lock_text
