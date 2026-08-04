"""External Datalog/SecPAL differential-shadow certification tests.

FVT-051 / FVT-G180 — ``ExternalAuthorizationShadowCertification@1``.

Acceptance covered:

* explicit strict installation selects exact external engines (souffle, secpal);
* allow/deny/unknown/conflict/delegation corpus passes differentially;
* rule and scope mutations change the verdict;
* replay is deterministic;
* malformed output and timeouts fail closed / quarantine;
* any disagreement quarantines promotion;
* external engines remain shadows while in-process references retain
  authorization authority.
"""

from __future__ import annotations

import importlib.util
import json
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
    / "authorization.py"
)
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "authorization_external.py"
)

INTERFACE = "ExternalAuthorizationShadowCertification@1"
SCHEMA_VERSION = "external-authorization-shadow-certification/v1"
INSTALLER_INTERFACE = "AuthorizationExternalInstaller@1"
GOAL_ID = "FVT-G180"
TASK_ID = "FVT-051"
LANE_ID = "datalog_secpal_external"
HANDLER_ID = "external_authorization_shadow_certification@1"

REQUIRED_ENGINES = {"souffle", "secpal"}
REQUIRED_CATEGORIES = {"allow", "deny", "unknown", "conflict", "delegation"}
REQUIRED_MUTATIONS = {"rule", "scope"}
REFERENCE_ENGINES = {"datalog-authorization", "secpal-authorization"}


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
    return _load_module("authorization_external_installer", INSTALLER_PATH)


@pytest.fixture(scope="module")
def certifier():
    return _load_module("authorization_external_certification", CERTIFIER_PATH)


@pytest.fixture(scope="module")
def install_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("authz-external-shadows")


@pytest.fixture(scope="module")
def install_bundle(installer, install_root):
    return installer.ensure_authorization_external(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root,
        hermetic_shadow=True,
        checksum_verified=True,
    )


@pytest.fixture(scope="module")
def certificate(certifier, install_bundle, install_root) -> dict[str, Any]:
    assert install_bundle.ok, install_bundle.to_dict()
    return certifier.certify_external_authorization_shadows(
        install_root=install_root,
        skip_install=True,
    )


# ---------------------------------------------------------------------------
# Artifact presence / identity
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert CERTIFIER_PATH.is_file()


def test_installer_interface_and_pins(installer) -> None:
    assert installer.INTERFACE == INSTALLER_INTERFACE
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert set(installer.EXTERNAL_TOOLS) == REQUIRED_ENGINES
    meta = installer.describe_authorization_installer()
    assert meta["policy"]["external_engines_are_shadows"] is True
    assert meta["policy"]["never_grants_authorization_authority"] is True
    assert meta["policy"]["never_on_import"] is True
    assert meta["policy"]["requires_yes_true"] is True
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
    assert set(certifier.REFERENCE_ENGINES) == REFERENCE_ENGINES
    assert certifier.SHADOW_AUTHORITY_CEILING == "none"


def test_strict_installation_selects_exact_pins(installer, install_bundle) -> None:
    assert install_bundle.ok
    assert install_bundle.gap_replaced == "datalog_secpal_external"
    assert set(install_bundle.identities) == REQUIRED_ENGINES
    for tool_id, identity in install_bundle.identities.items():
        pin = installer.pin_for_tool(tool_id)
        assert identity.version == pin["version"]
        assert identity.role == "shadow"
        assert identity.authority_ceiling == "none"
        assert Path(identity.executable).is_file()
        # Version probe must report the pinned identity.
        import subprocess

        completed = subprocess.run(
            [identity.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        banner = (completed.stdout or "") + (completed.stderr or "")
        assert pin["version"] in banner
        assert tool_id in banner.casefold() or "shadow" in banner.casefold()


def test_install_refuses_without_yes(installer, install_root) -> None:
    receipt = installer.ensure_souffle(
        yes=False,
        strict=False,
        install_root=install_root / "no-yes",
    )
    assert not receipt.ok
    assert receipt.status == "refused"
    assert "requires_yes_true" in receipt.block_reasons


def test_install_forbidden_on_import(installer, install_root) -> None:
    receipt = installer.ensure_secpal(
        yes=True,
        strict=False,
        install_root=install_root / "import-blocked",
        import_context=True,
    )
    assert not receipt.ok
    assert "forbidden_on_import" in receipt.block_reasons


# ---------------------------------------------------------------------------
# Full external shadow certification
# ---------------------------------------------------------------------------


def test_both_external_engines_are_shadow_certified(
    certificate: dict[str, Any],
) -> None:
    assert certificate["schema_version"] == SCHEMA_VERSION
    assert certificate["interface"] == INTERFACE
    assert certificate["goal_id"] == GOAL_ID
    assert certificate["task_id"] == TASK_ID
    assert certificate["lane_id"] == LANE_ID
    assert certificate["certified"] is True
    assert certificate["authority_ceiling"] == "none"
    assert certificate["forbids_theorem_authority"] is True
    assert certificate["forbids_authorization_authority_on_shadows"] is True
    assert set(certificate["engine_ids"]) == REQUIRED_ENGINES
    assert certificate["policy"]["external_engines_are_shadows"] is True
    assert certificate["policy"]["in_process_references_retain_authorization_authority"] is True
    assert certificate["policy"]["disagreement_quarantines_promotion"] is True
    assert certificate["policy"]["grants_theorem_authority"] is False
    assert certificate["policy"]["grants_authorization_decision_authority"] is False

    engines = {item["engine_id"]: item for item in certificate["engines"]}
    assert set(engines) == REQUIRED_ENGINES
    for engine_id, entry in engines.items():
        assert entry["usable"] is True, engine_id
        assert entry["certified"] is True, engine_id
        assert entry["is_shadow"] is True
        assert entry["role"] == "shadow"
        assert entry["authority_ceiling"] == "none"
        assert entry["block_reasons"] == []
        assert entry["checks"]
        assert all(check["status"] == "passed" for check in entry["checks"]), (
            engine_id,
            [c for c in entry["checks"] if c["status"] != "passed"],
        )
        assert all(check["is_theorem_authority"] is False for check in entry["checks"])
        assert all(
            check["is_authorization_authority"] is False for check in entry["checks"]
        )


def test_required_categories_and_mutations(certificate: dict[str, Any]) -> None:
    assert REQUIRED_CATEGORIES <= set(certificate["categories_exercised"])
    assert set(certificate["mutation_kinds"]) == REQUIRED_MUTATIONS


@pytest.mark.parametrize("category", sorted(REQUIRED_CATEGORIES))
def test_category_outcomes_agree_with_reference(
    certifier, install_bundle, category: str
) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category == category
    ]
    assert specs, category
    for engine_id, identity in install_bundle.identities.items():
        for spec in specs:
            document, query, expected = certifier.materialize_case(spec)
            record = certifier.run_shadow_case(
                engine_id,
                spec.case_id,
                document,
                query,
                executable=identity.executable,
                engine_version=identity.version,
            )
            assert record.outcome == expected, (engine_id, spec.case_id, record)
            assert record.reference_outcome == expected
            assert record.agreed is True
            assert record.quarantined is False
            assert record.authority == "none"
            assert record.is_theorem_authority is False
            assert record.is_authorization_authority is False


@pytest.mark.parametrize("mutation_kind", sorted(REQUIRED_MUTATIONS))
def test_rule_and_scope_mutations_change_verdict(
    certifier, install_bundle, installer, mutation_kind: str
) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category == "mutation" and spec.mutation_kind == mutation_kind
    ]
    assert specs, mutation_kind
    for engine_id, identity in install_bundle.identities.items():
        for spec in specs:
            base = certifier._semantic._fixture_by_id(spec.base_fixture_id)
            baseline = certifier.run_shadow_case(
                engine_id,
                f"{spec.case_id}:baseline",
                base.document,
                base.query,
                executable=identity.executable,
                engine_version=identity.version,
            )
            document, query, expected = certifier.materialize_case(spec)
            mutated = certifier.run_shadow_case(
                engine_id,
                spec.case_id,
                document,
                query,
                executable=identity.executable,
                engine_version=identity.version,
            )
            assert mutated.outcome != baseline.outcome, (
                engine_id,
                mutation_kind,
                baseline.outcome,
                mutated.outcome,
            )
            assert mutated.outcome == expected
            assert mutated.agreed is True
            assert mutated.policy_digest != baseline.policy_digest


def test_replay_is_deterministic(certifier, install_bundle) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category in {"deny", "unknown", "conflict"}
    ]
    assert specs
    for engine_id, identity in install_bundle.identities.items():
        for spec in specs:
            document, query, _ = certifier.materialize_case(spec)
            first = certifier.run_shadow_case(
                engine_id,
                spec.case_id,
                document,
                query,
                executable=identity.executable,
                engine_version=identity.version,
            )
            second = certifier.run_shadow_case(
                engine_id,
                f"{spec.case_id}:replay",
                document,
                query,
                executable=identity.executable,
                engine_version=identity.version,
            )
            assert first.outcome == second.outcome
            assert first.policy_digest == second.policy_digest
            assert first.agreed is second.agreed is True


def test_malformed_output_never_allows(certifier, install_bundle) -> None:
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_shadow_case(
            engine_id,
            "case:malformed",
            None,
            None,
            executable=identity.executable,
            engine_version=identity.version,
            expect_error=True,
        )
        assert record.outcome != "allow"
        assert record.malformed is True
        assert record.quarantined is True


def test_timeout_is_quarantined(certifier, install_bundle, installer) -> None:
    from ipfs_datasets_py.logic.backends.datalog.adapters import (
        DEFAULT_AUTHORIZATION_FIXTURES,
    )

    fixture = next(
        item for item in DEFAULT_AUTHORIZATION_FIXTURES if item.category == "allow"
    )
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_shadow_case(
            engine_id,
            "case:timeout",
            fixture.document,
            fixture.query,
            executable=identity.executable,
            engine_version=identity.version,
            timeout_seconds=0.25,
            env={installer.ENV_SLEEP_SECONDS: "2.0"},
        )
        assert record.timed_out is True
        assert record.quarantined is True
        assert record.outcome == "timeout"


def test_disagreement_quarantines_promotion(
    certifier, install_bundle, installer
) -> None:
    from ipfs_datasets_py.logic.backends.datalog.adapters import (
        DEFAULT_AUTHORIZATION_FIXTURES,
    )

    fixture = next(
        item for item in DEFAULT_AUTHORIZATION_FIXTURES if item.category == "allow"
    )
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_shadow_case(
            engine_id,
            "case:disagreement",
            fixture.document,
            fixture.query,
            executable=identity.executable,
            engine_version=identity.version,
            env={installer.ENV_DISAGREE: "1"},
        )
        assert record.agreed is False
        assert record.quarantined is True
        assert record.outcome != record.reference_outcome
        assert record.reference_outcome == "allow"


def test_references_retain_authorization_authority(certificate: dict[str, Any]) -> None:
    refs = certificate["reference_authority"]
    assert set(refs) == REFERENCE_ENGINES
    for engine_id, meta in refs.items():
        assert meta["authority_ceiling"] == "authorization"
        assert meta["retains_authorization_authority"] is True
        assert meta["role"] == "authority"


def test_lane_handler_reports_certified(certifier, install_root) -> None:
    # Install first so the handler can skip re-install races in parallel.
    certifier.certify_external_authorization_shadows(
        install_root=install_root,
        force_install=False,
    )
    result = certifier.external_authorization_lane_handler(
        install_root=install_root,
        skip_install=True,
    )
    assert result["lane_id"] == LANE_ID
    assert result["handler_id"] == HANDLER_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["authority_ceiling"] == "none"
    assert result["grants_theorem_authority"] is False
    assert result["grants_authorization_decision_authority"] is False
    assert result["external_engines_are_shadows"] is True
    assert set(result["engine_ids"]) == REQUIRED_ENGINES
    assert result["certificate_digest_sha256"]
    assert len(result["certificate_digest_sha256"]) == 64


def test_certificate_digest_is_stable(certifier, install_root) -> None:
    first = certifier.certify_external_authorization_shadows(
        install_root=install_root,
        skip_install=True,
    )
    second = certifier.certify_external_authorization_shadows(
        install_root=install_root,
        skip_install=True,
    )
    assert first["certificate_digest_sha256"] == second["certificate_digest_sha256"]
    assert first["certified"] is True
    assert second["certified"] is True


def test_installer_registry_entries_match(installer) -> None:
    from ipfs_datasets_py.logic.backends.installers.registry import (
        get_installer_entry,
    )

    for tool_id in REQUIRED_ENGINES:
        entry = get_installer_entry(tool_id)
        assert entry.family.value == "authorization"
        assert entry.ensure_name == f"ensure_{tool_id}"
        assert entry.replaces_gap_id == "datalog_secpal_external"
        assert entry.never_on_import is True
        assert entry.requires_explicit_yes is True
        assert entry.user_local_only is True


def test_shadow_shim_source_is_deterministic(installer) -> None:
    a = installer.build_shadow_shim_source(
        "souffle", "2.4.1", identity_file="/tmp/id.json"
    )
    b = installer.build_shadow_shim_source(
        "souffle", "2.4.1", identity_file="/tmp/id.json"
    )
    assert a == b
    assert "2.4.1" in a
    assert "souffle" in a
    # No unfilled format placeholders.
    assert "{tool_id" not in a
    assert "{version" not in a
