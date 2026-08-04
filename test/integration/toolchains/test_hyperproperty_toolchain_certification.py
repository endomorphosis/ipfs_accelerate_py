"""HyperLTL / AutoHyper / MCHyper toolchain certification tests.

FVT-046 / FVT-G170 — ``HyperpropertyToolchainCertification@1``.

Acceptance covered:

* explicit strict installation selects reviewed HyperLTL, AutoHyper, and
  MCHyper artifacts;
* quantifiers and observation projections are preserved;
* satisfaction, violating trace tuples, semantic mutations, replay,
  malformed output, disagreement, timeout, and exact bounds pass;
* results retain declared bounded hyperproperty authority and cannot make
  universal claims beyond bounds.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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

INTERFACE = "HyperpropertyToolchainCertification@1"
SCHEMA_VERSION = "hyperproperty-toolchain-certification/v1"
INSTALLER_INTERFACE = "HyperpropertyInstaller@1"
GOAL_ID = "FVT-G170"
TASK_ID = "FVT-046"
LANE_ID = "hyperltl"
HANDLER_ID = "hyperproperty_toolchain_certification@1"

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
    return _load_module("hyperproperty_installer", INSTALLER_PATH)


@pytest.fixture(scope="module")
def certifier():
    return _load_module("hyperproperty_certification", CERTIFIER_PATH)


@pytest.fixture(scope="module")
def install_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("hyperproperty-engines")


@pytest.fixture(scope="module")
def install_bundle(installer, install_root):
    return installer.ensure_hyperproperty(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root,
        hermetic_engine=True,
        checksum_verified=True,
    )


@pytest.fixture(scope="module")
def certificate(certifier, install_bundle, install_root) -> dict[str, Any]:
    assert install_bundle.ok, install_bundle.to_dict()
    return certifier.certify_hyperproperty_toolchains(
        install_root=install_root,
        skip_install=True,
    )


# ---------------------------------------------------------------------------
# Artifact presence / identity
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert CERTIFIER_PATH.is_file()
    assert Path(__file__).is_file()


def test_installer_interface_and_pins(installer) -> None:
    assert installer.INTERFACE == INSTALLER_INTERFACE
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert set(installer.EXTERNAL_TOOLS) == REQUIRED_ENGINES
    meta = installer.describe_hyperproperty_installer()
    assert meta["policy"]["never_on_import"] is True
    assert meta["policy"]["requires_yes_true"] is True
    assert meta["policy"]["never_authorizes_universal_proof"] is True
    assert meta["policy"]["cannot_make_universal_claims_beyond_bounds"] is True
    assert meta["policy"]["authority_ceiling"] == "bounded"
    assert meta["gap_id"] == "hyper_tools"
    for tool_id in REQUIRED_ENGINES:
        pin = installer.pin_for_tool(tool_id)
        assert pin["version"]
        assert pin["license"]
        assert pin["source"]


def test_certifier_interface_constants(certifier) -> None:
    assert certifier.INTERFACE == INTERFACE
    assert certifier.SCHEMA_VERSION == SCHEMA_VERSION
    assert certifier.GOAL_ID == GOAL_ID
    assert certifier.TASK_ID == TASK_ID
    assert certifier.LANE_ID == LANE_ID
    assert certifier.HANDLER_ID == HANDLER_ID
    assert set(certifier.EXTERNAL_ENGINES) == REQUIRED_ENGINES
    assert certifier.AUTHORITY_CEILING == "bounded"


def test_strict_installation_selects_exact_pins(installer, install_bundle) -> None:
    assert install_bundle.ok
    assert install_bundle.gap_replaced == "hyper_tools"
    assert set(install_bundle.identities) == REQUIRED_ENGINES
    for tool_id, identity in install_bundle.identities.items():
        pin = installer.pin_for_tool(tool_id)
        assert identity.version == pin["version"]
        assert identity.role == "authority"
        assert identity.authority_ceiling == "bounded"
        assert identity.authorizes_universal_proof is False
        assert Path(identity.executable).is_file()
        completed = subprocess.run(
            [identity.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        banner = (completed.stdout or "") + (completed.stderr or "")
        assert pin["version"] in banner
        assert (
            tool_id in banner.casefold()
            or "hyper" in banner.casefold()
            or "hermetic" in banner.casefold()
        )


def test_install_refuses_without_yes(installer, install_root) -> None:
    receipt = installer.ensure_hyperltl(
        yes=False,
        strict=False,
        install_root=install_root / "no-yes",
    )
    assert not receipt.ok
    assert receipt.status == "refused"
    assert "requires_yes_true" in receipt.block_reasons


def test_install_forbidden_on_import(installer, install_root) -> None:
    receipt = installer.ensure_autohyper(
        yes=True,
        strict=False,
        install_root=install_root / "import-blocked",
        import_context=True,
    )
    assert not receipt.ok
    assert "forbidden_on_import" in receipt.block_reasons


def test_install_forbidden_on_capability_discovery(installer, install_root) -> None:
    receipt = installer.ensure_mchyper(
        yes=True,
        strict=False,
        install_root=install_root / "capability-blocked",
        capability_discovery=True,
    )
    assert not receipt.ok
    assert "forbidden_on_capability_discovery" in receipt.block_reasons


# ---------------------------------------------------------------------------
# Full hyperproperty toolchain certification
# ---------------------------------------------------------------------------


def test_all_engines_are_bounded_certified(certificate: dict[str, Any]) -> None:
    assert certificate["schema_version"] == SCHEMA_VERSION
    assert certificate["interface"] == INTERFACE
    assert certificate["goal_id"] == GOAL_ID
    assert certificate["task_id"] == TASK_ID
    assert certificate["lane_id"] == LANE_ID
    assert certificate["certified"] is True
    assert certificate["authority_ceiling"] == "bounded"
    assert certificate["forbids_theorem_authority"] is True
    assert certificate["forbids_universal_claims_beyond_bounds"] is True
    assert set(certificate["engine_ids"]) == REQUIRED_ENGINES
    assert certificate["policy"]["quantifiers_and_observation_projections_preserved"] is True
    assert certificate["policy"]["disagreement_quarantines_promotion"] is True
    assert certificate["policy"]["never_authorizes_universal_proof"] is True
    assert certificate["policy"]["cannot_make_universal_claims_beyond_bounds"] is True
    assert certificate["policy"]["grants_theorem_authority"] is False
    assert certificate["policy"]["authorizes_universal_proof"] is False

    engines = {item["engine_id"]: item for item in certificate["engines"]}
    assert set(engines) == REQUIRED_ENGINES
    for engine_id, entry in engines.items():
        assert entry["usable"] is True, engine_id
        assert entry["certified"] is True, (
            engine_id,
            entry.get("block_reasons"),
            [c for c in entry["checks"] if c["status"] != "passed"],
        )
        assert entry["role"] == "authority"
        assert entry["authority_ceiling"] == "bounded"
        assert entry["authorizes_universal_proof"] is False
        assert entry["block_reasons"] == []
        assert entry["checks"]
        assert all(check["status"] == "passed" for check in entry["checks"]), (
            engine_id,
            [c for c in entry["checks"] if c["status"] != "passed"],
        )
        assert all(check["is_theorem_authority"] is False for check in entry["checks"])
        assert all(
            check["authorizes_universal_proof"] is False for check in entry["checks"]
        )


def test_required_categories_and_mutations(certificate: dict[str, Any]) -> None:
    assert REQUIRED_CATEGORIES <= set(certificate["categories_exercised"])
    assert set(certificate["mutation_kinds"]) == REQUIRED_MUTATIONS


def test_quantifiers_and_observations_preserved(
    certifier, install_bundle
) -> None:
    for engine_id, identity in install_bundle.identities.items():
        document = certifier.materialize_document(
            certifier.CaseSpec(
                case_id="case:ni_holds",
                category="satisfaction",
                expected="satisfied",
            )
        )
        backend = certifier.backend_for(
            engine_id, executable=identity.executable
        )
        translation = backend.translate(document)
        assert translation.quantifier_order.signature == ("forall", "forall")
        assert translation.quantifier_order.variable_names == ("pi1", "pi2")
        assert translation.quantifier_order.matches_document(document)
        assert translation.observation_map.observation_fields == (
            "status",
            "public_token",
        )
        text = translation.formula_text
        assert text.index("forall pi1.") < text.index("forall pi2.")
        assert "status" in text
        assert "public_token" in text
        assert "secret" not in text


def test_satisfaction_and_violation_cases(certifier, install_bundle) -> None:
    holds = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:ni_holds",
            category="satisfaction",
            expected="satisfied",
        )
    )
    violated = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:ni_violated",
            category="violation",
            expected="violated",
            force_verdict="violated",
        )
    )
    for engine_id, identity in install_bundle.identities.items():
        sat = certifier.run_engine_case(
            engine_id,
            "case:ni_holds",
            holds,
            executable=identity.executable,
            engine_version=identity.version,
            expected="satisfied",
        )
        assert sat.outcome == "satisfied"
        assert sat.agreed is True
        assert sat.authorizes_universal_proof is False
        assert sat.authority == "bounded"

        viol = certifier.run_engine_case(
            engine_id,
            "case:ni_violated",
            violated,
            executable=identity.executable,
            engine_version=identity.version,
            expected="violated",
            force_verdict="violated",
        )
        assert viol.outcome == "violated"
        assert viol.agreed is True
        assert viol.counterexample_traces >= 2
        assert viol.authorizes_universal_proof is False


def test_semantic_mutations(certifier, install_bundle) -> None:
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
    for engine_id, identity in install_bundle.identities.items():
        backend = certifier.backend_for(engine_id, executable=identity.executable)
        base_obs = backend.translate(base).observation_map.observation_fields
        mut_obs = backend.translate(obs_mut).observation_map.observation_fields
        assert mut_obs != base_obs
        assert mut_obs == ("status",)

        base_sig = backend.translate(base).quantifier_order.signature
        mut_sig = backend.translate(quant_mut).quantifier_order.signature
        assert mut_sig != base_sig
        assert mut_sig == ("forall", "exists")


def test_replay_is_deterministic(certifier, install_bundle) -> None:
    document = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:replay_holds",
            category="replay",
            expected="satisfied",
        )
    )
    for engine_id, identity in install_bundle.identities.items():
        first = certifier.run_engine_case(
            engine_id,
            "case:replay_holds",
            document,
            executable=identity.executable,
            engine_version=identity.version,
            expected="satisfied",
        )
        second = certifier.run_engine_case(
            engine_id,
            "case:replay_holds:again",
            document,
            executable=identity.executable,
            engine_version=identity.version,
            expected="satisfied",
        )
        assert first.outcome == second.outcome == "satisfied"
        assert first.document_digest == second.document_digest
        assert first.agreed is second.agreed is True


def test_malformed_output_never_satisfies(certifier, install_bundle) -> None:
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_engine_case(
            engine_id,
            "case:malformed",
            None,
            executable=identity.executable,
            engine_version=identity.version,
            expected="error",
            expect_error=True,
        )
        assert record.outcome != "satisfied"
        assert record.malformed is True
        assert record.quarantined is True


def test_timeout_is_quarantined(certifier, install_bundle, installer) -> None:
    document = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:timeout",
            category="timeout",
            expected="satisfied",
        )
    )
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_engine_case(
            engine_id,
            "case:timeout",
            document,
            executable=identity.executable,
            engine_version=identity.version,
            expected="satisfied",
            timeout_seconds=0.25,
            env={installer.ENV_SLEEP_SECONDS: "2.0"},
        )
        assert record.timed_out is True
        assert record.quarantined is True
        assert record.outcome == "timeout"


def test_disagreement_quarantines_promotion(
    certifier, install_bundle, installer
) -> None:
    document = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:disagreement",
            category="disagreement",
            expected="satisfied",
        )
    )
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_engine_case(
            engine_id,
            "case:disagreement",
            document,
            executable=identity.executable,
            engine_version=identity.version,
            expected="satisfied",
            env={installer.ENV_DISAGREE: "1"},
        )
        assert record.agreed is False
        assert record.quarantined is True
        assert record.outcome != record.expected
        assert record.expected == "satisfied"


def test_exact_bounds_and_no_universal_claims(certifier, install_bundle) -> None:
    document = certifier.materialize_document(
        certifier.CaseSpec(
            case_id="case:bounds_exact",
            category="bounds",
            expected="satisfied",
            max_traces=4,
            max_pairs=8,
        )
    )
    assert document.self_composition_bound.max_traces == 4
    assert document.self_composition_bound.max_pairs == 8
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_engine_case(
            engine_id,
            "case:bounds_exact",
            document,
            executable=identity.executable,
            engine_version=identity.version,
            expected="satisfied",
        )
        assert record.outcome == "satisfied"
        assert record.authorizes_universal_proof is False
        assert record.authority == "bounded"
        # Backend path: satisfaction is hyperproperty authority, not theorem.
        backend = certifier.backend_for(engine_id, executable=identity.executable)
        outcome = backend.check(document)
        assert outcome.receipt.authorizes_universal_proof is False
        assert outcome.result.authority.value == "hyperproperty"
        assert outcome.result.translation_ceiling.value == "bounded"


def test_lane_handler_reports_certified(certifier, install_root) -> None:
    result = certifier.hyperproperty_lane_handler(
        install_root=install_root,
        skip_install=True,
    )
    assert result["lane_id"] == LANE_ID
    assert result["handler_id"] == HANDLER_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["authority_ceiling"] == "bounded"
    assert result["grants_theorem_authority"] is False
    assert result["authorizes_universal_proof"] is False
    assert set(result["engine_ids"]) == REQUIRED_ENGINES


def test_roles_bind_to_bounded_authority() -> None:
    _ensure_datasets_on_path()
    from ipfs_datasets_py.logic.backends.toolchain_roles import (
        ToolRole,
        ToolchainAuthorityCeiling,
        get_tool_role,
    )

    for tool_id in sorted(REQUIRED_ENGINES):
        role = get_tool_role(tool_id)
        assert role.role is ToolRole.AUTHORITY
        assert role.authority_ceiling is ToolchainAuthorityCeiling.BOUNDED
