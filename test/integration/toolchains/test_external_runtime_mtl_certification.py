"""External Runtime MTL cross-runtime parity certification tests.

FVT-052 / FVT-G181 — ``ExternalRuntimeMTLCertification@1``.

Acceptance covered:

* explicit strict installation selects the exact external monitor pin;
* Python, TypeScript (when available), and external agree on satisfied/violated
  golden traces, boundary intervals, mutations, shortest-prefix replay,
  malformed input, and bounds;
* disagreement quarantines promotion;
* finite-trace authority is preserved and no global correctness claim is inferred.
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
    / "runtime_mtl.py"
)
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl_external.py"
)
SEMANTIC_CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl.py"
)

INTERFACE = "ExternalRuntimeMTLCertification@1"
SCHEMA_VERSION = "external-runtime-mtl-certification/v1"
INSTALLER_INTERFACE = "RuntimeMTLExternalInstaller@1"
GOAL_ID = "FVT-G181"
TASK_ID = "FVT-052"
LANE_ID = "runtime_mtl_external"
HANDLER_ID = "external_runtime_mtl_certification@1"
TOOL_ID = "runtime-mtl-external"
AUTHORITY_CEILING = "finite_trace"

REQUIRED_ENGINES = {"runtime-mtl-external"}
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
REFERENCE_ENGINES = {"runtime-mtl"}


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
    return _load_module("runtime_mtl_external_installer", INSTALLER_PATH)


@pytest.fixture(scope="module")
def certifier():
    return _load_module("runtime_mtl_external_certification", CERTIFIER_PATH)


@pytest.fixture(scope="module")
def install_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("runtime-mtl-external")


@pytest.fixture(scope="module")
def install_bundle(installer, install_root):
    return installer.ensure_runtime_mtl_external_bundle(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root,
        hermetic_parity_engine=True,
        checksum_verified=True,
    )


@pytest.fixture(scope="module")
def certificate(certifier, install_bundle, install_root) -> dict[str, Any]:
    assert install_bundle.ok, install_bundle.to_dict()
    return certifier.certify_external_runtime_mtl(
        install_root=install_root,
        skip_install=True,
        repo_root=REPO_ROOT,
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
    meta = installer.describe_runtime_mtl_installer()
    assert meta["policy"]["requires_yes_true"] is True
    assert meta["policy"]["never_on_import"] is True
    assert meta["policy"]["finite_trace_authority_only"] is True
    assert meta["policy"]["never_grants_theorem_authority"] is True
    assert meta["policy"]["no_global_correctness_claim"] is True
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
    assert certifier.AUTHORITY_CEILING == AUTHORITY_CEILING


def test_strict_installation_selects_exact_pin(installer, install_bundle) -> None:
    assert install_bundle.ok
    assert install_bundle.gap_replaced == "runtime_mtl_external"
    assert set(install_bundle.identities) == REQUIRED_ENGINES
    for tool_id, identity in install_bundle.identities.items():
        pin = installer.pin_for_tool(tool_id)
        assert identity.version == pin["version"]
        assert identity.role == "authority"
        assert identity.authority_ceiling == "finite_trace"
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
        assert "runtime-mtl" in banner.casefold() or "parity" in banner.casefold()


def test_install_refuses_without_yes(installer, install_root) -> None:
    receipt = installer.ensure_runtime_mtl_external(
        yes=False,
        strict=False,
        install_root=install_root / "no-yes",
    )
    assert not receipt.ok
    assert receipt.status == "refused"
    assert "requires_yes_true" in receipt.block_reasons


def test_install_forbidden_on_import(installer, install_root) -> None:
    receipt = installer.ensure_runtime_mtl_external(
        yes=True,
        strict=False,
        install_root=install_root / "import-blocked",
        import_context=True,
    )
    assert not receipt.ok
    assert "forbidden_on_import" in receipt.block_reasons


# ---------------------------------------------------------------------------
# Full external parity certification
# ---------------------------------------------------------------------------


def test_external_engine_is_parity_certified(certificate: dict[str, Any]) -> None:
    assert certificate["schema_version"] == SCHEMA_VERSION
    assert certificate["interface"] == INTERFACE
    assert certificate["goal_id"] == GOAL_ID
    assert certificate["task_id"] == TASK_ID
    assert certificate["lane_id"] == LANE_ID
    assert certificate["certified"] is True, certificate["summary"]
    assert certificate["authority_ceiling"] == AUTHORITY_CEILING
    assert certificate["forbids_theorem_authority"] is True
    assert certificate["forbids_global_correctness_claim"] is True
    assert set(certificate["engine_ids"]) == REQUIRED_ENGINES
    assert certificate["policy"]["finite_trace_authority_only"] is True
    assert certificate["policy"]["disagreement_quarantines_promotion"] is True
    assert certificate["policy"]["grants_theorem_authority"] is False
    assert certificate["policy"]["grants_global_correctness"] is False
    assert certificate["policy"]["python_typescript_external_parity"] is True

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
        assert entry["authority_ceiling"] == AUTHORITY_CEILING
        assert entry["block_reasons"] == []
        assert entry["checks"]
        assert all(check["status"] == "passed" for check in entry["checks"]), (
            engine_id,
            [c for c in entry["checks"] if c["status"] != "passed"],
        )
        assert all(check["is_theorem_authority"] is False for check in entry["checks"])
        assert all(
            check["authorizes_global_proof"] is False for check in entry["checks"]
        )


def test_required_categories_and_mutations(certificate: dict[str, Any]) -> None:
    assert REQUIRED_CATEGORIES <= set(certificate["categories_exercised"])
    assert set(certificate["mutation_kinds"]) == REQUIRED_MUTATIONS


def test_required_central_check_kinds_are_explicit_and_passed(
    certificate: dict[str, Any],
) -> None:
    checks = [
        check
        for engine in certificate["engines"]
        for check in engine["checks"]
    ]
    passed_kinds = {
        check["kind"] for check in checks if check["status"] == "passed"
    }
    assert {"positive", "negative", "mutation", "replay"} <= passed_kinds
    negative_checks = [
        check for check in checks if check["kind"] == "negative"
    ]
    assert negative_checks
    assert all(check["status"] == "passed" for check in negative_checks)
    assert all(
        check["expected"].startswith("violated/")
        or check["expected"].endswith("/false")
        for check in negative_checks
    )


@pytest.mark.parametrize(
    "category",
    sorted(
        {
            "satisfied",
            "violated",
            "timestamp_boundary",
            "clean_prefix",
            "malformed",
        }
    ),
)
def test_category_outcomes_agree_with_python(
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
            case = certifier.materialize_case(spec)
            if "formula" not in case:
                continue
            record = certifier.run_parity_case(
                engine_id,
                spec.case_id,
                case,
                executable=identity.executable,
                engine_version=identity.version,
            )
            assert record.reference_status == (
                spec.expected_status or record.reference_status
            )
            assert record.status == record.reference_status, (
                engine_id,
                spec.case_id,
                record.to_dict(),
            )
            assert record.verdict == record.reference_verdict
            assert record.agreed is True
            assert record.quarantined is False
            assert record.authorizes_global_proof is False
            assert record.authority == "monitor"


@pytest.mark.parametrize("mutation_kind", sorted(REQUIRED_MUTATIONS))
def test_interval_and_event_mutations_change_verdict(
    certifier, install_bundle, mutation_kind: str
) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.mutation_kind == mutation_kind
        or (
            mutation_kind in (spec.category or "")
            and "mutation" in (spec.category or "")
        )
    ]
    assert specs, mutation_kind
    for engine_id, identity in install_bundle.identities.items():
        for spec in specs:
            base = certifier._semantic._golden_by_id(spec.base_fixture_id)
            baseline = certifier.run_parity_case(
                engine_id,
                f"{spec.case_id}:baseline",
                {
                    "case_id": f"{spec.case_id}:baseline",
                    "formula": base["formula"],
                    "trace": base["trace"],
                    "position": base.get("position", 0),
                },
                executable=identity.executable,
                engine_version=identity.version,
            )
            mutated_case = certifier.materialize_case(spec)
            mutated = certifier.run_parity_case(
                engine_id,
                spec.case_id,
                mutated_case,
                executable=identity.executable,
                engine_version=identity.version,
            )
            assert (
                mutated.status != baseline.status
                or mutated.verdict != baseline.verdict
            ), (
                engine_id,
                mutation_kind,
                baseline.status,
                mutated.status,
            )
            assert mutated.status == spec.expected_status
            assert mutated.verdict == spec.expected_verdict
            assert mutated.agreed is True
            assert (
                mutated.formula_digest != baseline.formula_digest
                or mutated.trace_digest != baseline.trace_digest
            )


def test_shortest_prefix_replay_agrees(certifier, install_bundle) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.recipe == "shortest_violating_prefix_replay"
    ]
    assert specs
    for engine_id, identity in install_bundle.identities.items():
        for spec in specs:
            base = certifier._semantic._golden_by_id(spec.base_fixture_id)
            prefix, length, prefix_record = certifier._semantic.shortest_violating_prefix(
                base["formula"],
                base["trace"],
                position=int(base.get("position", 0)),
            )
            assert prefix is not None and length is not None
            assert prefix_record is not None
            assert prefix_record.status == "violated"
            first = certifier.run_parity_case(
                engine_id,
                f"{spec.case_id}:shortest",
                {
                    "case_id": f"{spec.case_id}:shortest",
                    "formula": base["formula"],
                    "trace": prefix,
                    "position": base.get("position", 0),
                },
                executable=identity.executable,
                engine_version=identity.version,
            )
            second = certifier.run_parity_case(
                engine_id,
                f"{spec.case_id}:replay",
                {
                    "case_id": f"{spec.case_id}:replay",
                    "formula": base["formula"],
                    "trace": prefix,
                    "position": base.get("position", 0),
                },
                executable=identity.executable,
                engine_version=identity.version,
            )
            assert first.status == "violated"
            assert first.agreed is True
            assert first.status == second.status
            assert first.verdict == second.verdict
            assert first.formula_digest == second.formula_digest


def test_replay_is_deterministic(certifier, install_bundle) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category in {"satisfied", "violated", "clean_prefix"}
    ]
    assert specs
    for engine_id, identity in install_bundle.identities.items():
        for spec in specs:
            case = certifier.materialize_case(spec)
            if "formula" not in case:
                continue
            first = certifier.run_parity_case(
                engine_id,
                spec.case_id,
                case,
                executable=identity.executable,
                engine_version=identity.version,
            )
            second = certifier.run_parity_case(
                engine_id,
                f"{spec.case_id}:replay",
                case,
                executable=identity.executable,
                engine_version=identity.version,
            )
            assert first.status == second.status
            assert first.verdict == second.verdict
            assert first.formula_digest == second.formula_digest
            assert first.agreed is second.agreed is True


def test_malformed_output_never_satisfies(certifier, install_bundle) -> None:
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_parity_case(
            engine_id,
            "case:malformed",
            None,
            executable=identity.executable,
            engine_version=identity.version,
            expect_error=True,
        )
        assert record.status != "satisfied"
        assert record.malformed is True
        assert record.quarantined is True
        assert record.authorizes_global_proof is False


def test_disagreement_quarantines_promotion(
    certifier, install_bundle, installer
) -> None:
    from ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl import (
        golden_fixtures,
    )

    fixture = next(
        item
        for item in golden_fixtures()
        if item.get("expected", {}).get("status") == "satisfied"
    )
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_parity_case(
            engine_id,
            "case:disagreement",
            {
                "case_id": "case:disagreement",
                "formula": fixture["formula"],
                "trace": fixture["trace"],
                "position": fixture.get("position", 0),
            },
            executable=identity.executable,
            engine_version=identity.version,
            env={installer.ENV_DISAGREE: "1"},
        )
        assert record.agreed is False
        assert record.quarantined is True
        assert (
            record.status != record.reference_status
            or record.verdict != record.reference_verdict
        )
        assert record.reference_status == "satisfied"


def test_global_proof_elevation_is_quarantined(
    certifier, install_bundle, installer
) -> None:
    from ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl import (
        golden_fixtures,
    )

    fixture = next(
        item
        for item in golden_fixtures()
        if item.get("expected", {}).get("status") == "satisfied"
    )
    for engine_id, identity in install_bundle.identities.items():
        record = certifier.run_parity_case(
            engine_id,
            "case:bounds-elevation",
            {
                "case_id": "case:bounds-elevation",
                "formula": fixture["formula"],
                "trace": fixture["trace"],
                "position": fixture.get("position", 0),
            },
            executable=identity.executable,
            engine_version=identity.version,
            env={installer.ENV_AUTHORIZE_GLOBAL_PROOF: "1"},
        )
        # Either the engine elevates and is quarantined, or refuses elevation.
        if record.authorizes_global_proof:
            assert record.quarantined is True
            assert record.agreed is False
        else:
            assert record.authorizes_global_proof is False


def test_references_retain_finite_trace_authority(certificate: dict[str, Any]) -> None:
    refs = certificate["reference_authority"]
    assert set(refs) == REFERENCE_ENGINES
    for engine_id, meta in refs.items():
        assert meta["authority_ceiling"] == AUTHORITY_CEILING
        assert meta["retains_finite_trace_authority"] is True
        assert meta["role"] == "authority"


def test_lane_handler_reports_certified(certifier, install_root) -> None:
    certifier.certify_external_runtime_mtl(
        install_root=install_root,
        force_install=False,
        repo_root=REPO_ROOT,
    )
    result = certifier.external_runtime_mtl_lane_handler(
        install_root=install_root,
        skip_install=True,
        repo_root=REPO_ROOT,
    )
    assert result["lane_id"] == LANE_ID
    assert result["handler_id"] == HANDLER_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["authority_ceiling"] == AUTHORITY_CEILING
    assert result["grants_theorem_authority"] is False
    assert result["grants_global_correctness"] is False
    assert result["finite_trace_authority_only"] is True
    assert set(result["engine_ids"]) == REQUIRED_ENGINES
    assert result["certificate_digest_sha256"]
    assert len(result["certificate_digest_sha256"]) == 64


def test_certificate_digest_is_stable(certifier, install_root) -> None:
    first = certifier.certify_external_runtime_mtl(
        install_root=install_root,
        skip_install=True,
        repo_root=REPO_ROOT,
    )
    second = certifier.certify_external_runtime_mtl(
        install_root=install_root,
        skip_install=True,
        repo_root=REPO_ROOT,
    )
    assert first["certificate_digest_sha256"] == second["certificate_digest_sha256"]
    assert first["certified"] is True
    assert second["certified"] is True


def test_does_not_edit_in_process_semantic_lane() -> None:
    """Conflict policy: external lane must not own/edit the G103 surface."""

    assert SEMANTIC_CERTIFIER_PATH.is_file()
    # Our declared outputs are distinct from the in-process semantic certifier.
    assert CERTIFIER_PATH != SEMANTIC_CERTIFIER_PATH
    assert INSTALLER_PATH.name == "runtime_mtl.py"
    # Semantic certifier still declares no external install.
    text = SEMANTIC_CERTIFIER_PATH.read_text(encoding="utf-8")
    assert "does not install the external Runtime MTL" in text or "FVT-G103" in text


def test_import_is_side_effect_free(installer) -> None:
    assert installer._import_side_effect_free() is True
