from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.prover_matrix_registry import (
    EXPECTED_PROVER_IDS,
    PROVER_MATRIX_DUCKDB_SCHEMA_VERSION,
    PROVER_MATRIX_SCHEMA_VERSION,
    BoundIdentity,
    CommandRequest,
    CommandResult,
    IdentityKind,
    ProverDefinition,
    ProverFixture,
    ProverMatrixProbeConfig,
    ProverMatrixRegistry,
    ProverState,
    SelfTestStatus,
    load_documentation_claims,
    prover_matrix_paths,
    query_prover_matrix,
    write_prover_matrix_projection,
)


EXPECTED_MATRIX = {
    "z3",
    "cvc5",
    "tla_tlc",
    "apalache",
    "datalog_secpal",
    "tamarin",
    "proverif",
    "hyperltl_autohyper_mchyper",
    "lean",
    "coq",
    "runtime_mtl",
    "dcec",
    "tdfol",
    "hammer",
    "vampire",
    "e",
    "isabelle",
    "shadowprover",
    "leanstral",
    "zkp_backends",
}


def _definition(
    *,
    prover_id: str = "testprover",
    conformant: bool = True,
    reconstruction: bool = True,
    fixture_authorities: tuple[str, ...] = ("accepted_property", "unreviewed"),
    maximum_authorities: tuple[str, ...] = ("accepted_property",),
) -> ProverDefinition:
    return ProverDefinition(
        prover_id=prover_id,
        display_name="Test Prover",
        family="test",
        executable_candidates=(prover_id,),
        package_modules=("test_provider.backend",),
        package_distributions=("test-provider",),
        fixture=ProverFixture(
            fixture_id=f"{prover_id}-fixture@1",
            model_text="assert true\n",
            file_name="fixture.logic",
            translator_id="test-translator",
            translator_version="4",
            semantic_profile_id="test-semantics",
            semantic_profile_version="2",
            args=("check", "{fixture}"),
            expected_output_all=("verified",),
            establishes_translation_conformance=conformant,
            establishes_reconstruction=reconstruction,
            authoritative_for=fixture_authorities,
        ),
        maximum_authoritative_for=maximum_authorities,
        documentation_labels=("Test Prover",),
    )


def _executable(tmp_path: Path, name: str = "testprover") -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class FakeRuntime:
    def __init__(
        self,
        executable: Path | None,
        *,
        smoke: CommandResult | None = None,
        package: bool = True,
    ) -> None:
        self.executable = executable
        self.smoke = smoke or CommandResult(0, stdout="VERIFIED\n")
        self.package = package
        self.requests: list[CommandRequest] = []

    def which(self, _candidate: str) -> str | None:
        return str(self.executable) if self.executable else None

    def find_spec(self, module: str) -> object | None:
        if self.package and module == "test_provider.backend":
            return SimpleNamespace(origin="/packages/test_provider/backend.py")
        return None

    @staticmethod
    def distribution_version(distribution: str) -> str:
        assert distribution == "test-provider"
        return "7.1"

    def run(self, request: CommandRequest) -> CommandResult:
        self.requests.append(request)
        if "--version" in request.command:
            return CommandResult(0, stdout="TestProver 3.2")
        return self.smoke


def _registry(
    definition: ProverDefinition,
    runtime: FakeRuntime,
    *,
    documentation_path: Path | None = None,
    max_self_tests: int = 64,
) -> ProverMatrixRegistry:
    return ProverMatrixRegistry(
        (definition,),
        config=ProverMatrixProbeConfig(
            documentation_path=documentation_path,
            max_self_tests=max_self_tests,
        ),
        which=runtime.which,
        find_spec=runtime.find_spec,
        distribution_version=runtime.distribution_version,
        command_runner=runtime.run,
        wall_clock=lambda: 1_700_000_000,
    )


def test_default_registry_covers_the_complete_required_prover_matrix() -> None:
    registry = ProverMatrixRegistry.default(
        config=ProverMatrixProbeConfig(run_self_tests=False),
        which=lambda _name: None,
        find_spec=lambda _module: None,
    )

    snapshot = registry.probe()

    assert EXPECTED_PROVER_IDS == EXPECTED_MATRIX
    assert set(snapshot.capabilities) == EXPECTED_MATRIX
    assert snapshot.schema_version == PROVER_MATRIX_SCHEMA_VERSION
    assert snapshot.bounded is True
    payload = snapshot.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    for entry in snapshot.entries:
        assert set(entry.states) == {state.value for state in ProverState}
        assert entry.absent
        assert entry.highest_state is ProverState.ABSENT


def test_successful_bounded_self_test_binds_every_identity_and_promotes_only_allowlisted_authority(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path)
    runtime = FakeRuntime(executable)
    snapshot = _registry(_definition(), runtime).probe()

    entry = snapshot.entry("testprover")
    assert entry.states == {
        "absent": False,
        "discovered": True,
        "versioned": True,
        "smoke_tested": True,
        "translation_conformant": True,
        "reconstruction_capable": True,
        "authoritative_for": ["accepted_property"],
    }
    assert entry.highest_state is ProverState.AUTHORITATIVE_FOR
    receipt = entry.receipt
    assert receipt is not None
    assert receipt.status is SelfTestStatus.PASSED
    assert receipt.command[0] == str(executable)
    assert receipt.command_identity.startswith("sha256:")
    assert receipt.receipt_id.startswith("sha256:")
    assert receipt.binding.binding_id.startswith("sha256:")
    binding = receipt.binding.to_dict()
    assert set(binding) == {
        "executable",
        "package",
        "model",
        "translator",
        "semantic_profile",
        "fixture",
        "binding_id",
    }
    assert binding["executable"]["metadata"]["file_sha256"].startswith("sha256:")
    assert binding["package"]["metadata"]["version"] == "7.1"
    assert binding["translator"]["metadata"]["version"] == "4"
    assert binding["semantic_profile"]["metadata"]["version"] == "2"
    smoke_request = runtime.requests[-1]
    assert smoke_request.timeout_seconds == pytest.approx(5.0)
    assert smoke_request.max_output_bytes == 64 * 1024
    assert Path(smoke_request.command[-1]).name == "fixture.logic"
    assert receipt.command[-1] == "{fixture}"


@pytest.mark.parametrize(
    ("smoke_result", "status"),
    [
        (CommandResult(2, stderr="invalid"), SelfTestStatus.FAILED),
        (CommandResult(None, timed_out=True), SelfTestStatus.TIMED_OUT),
        (CommandResult(None, error="isolated runner failed"), SelfTestStatus.ERROR),
    ],
)
def test_failed_timeout_and_error_receipts_fail_closed(
    tmp_path: Path,
    smoke_result: CommandResult,
    status: SelfTestStatus,
) -> None:
    runtime = FakeRuntime(_executable(tmp_path), smoke=smoke_result)

    entry = _registry(_definition(), runtime).probe().entry("testprover")

    assert entry.discovered and entry.versioned
    assert not entry.smoke_tested
    assert not entry.translation_conformant
    assert not entry.reconstruction_capable
    assert entry.authoritative_for == ()
    assert entry.receipt is not None
    assert entry.receipt.status is status
    assert entry.highest_state is ProverState.VERSIONED


def test_discovery_and_documentation_claims_are_not_runtime_evidence(
    tmp_path: Path,
) -> None:
    documentation = tmp_path / "prover_matrix.md"
    documentation.write_text(
        "# Matrix\n\n"
        "| Prover | Access path | Primary fit |\n"
        "| --- | --- | --- |\n"
        "| Test Prover | Source says installed | Everything |\n",
        encoding="utf-8",
    )
    runtime = FakeRuntime(None, package=False)
    claims = load_documentation_claims(documentation)

    entry = _registry(
        _definition(), runtime, documentation_path=documentation
    ).probe().entry("testprover")

    assert len(claims) == 1
    assert entry.absent and not entry.discovered
    assert not entry.versioned and not entry.smoke_tested
    assert len(entry.documentation_claims) == 1
    claim = entry.documentation_claims[0].to_dict()
    assert claim["evidence_class"] == "documentation_only"
    assert claim["runtime_evidence"] is False
    assert entry.to_dict()["documentation_is_runtime_evidence"] is False


def test_package_only_discovery_can_be_versioned_but_is_not_smoke_tested(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(None, package=True)

    entry = _registry(_definition(), runtime).probe().entry("testprover")

    assert not entry.absent
    assert entry.discovered and entry.versioned
    assert entry.package_module == "test_provider.backend"
    assert entry.package_version == "7.1"
    assert not entry.smoke_tested
    assert entry.receipt is None
    assert runtime.requests == []
    assert "no isolated executable fixture runner" in entry.reason


def test_no_self_test_and_budget_exhaustion_stop_at_versioned(
    tmp_path: Path,
) -> None:
    first = _definition(prover_id="first")
    second = _definition(prover_id="second")
    first_exe = _executable(tmp_path, "first")
    second_exe = _executable(tmp_path, "second")
    requests: list[CommandRequest] = []

    def which(name: str) -> str | None:
        return {"first": str(first_exe), "second": str(second_exe)}.get(name)

    def runner(request: CommandRequest) -> CommandResult:
        requests.append(request)
        if "--version" in request.command:
            return CommandResult(0, stdout="version 1")
        return CommandResult(0, stdout="verified")

    registry = ProverMatrixRegistry(
        (first, second),
        config=ProverMatrixProbeConfig(max_self_tests=1),
        which=which,
        find_spec=lambda _module: None,
        command_runner=runner,
    )

    snapshot = registry.probe()
    assert snapshot.entry("first").smoke_tested
    assert snapshot.entry("second").versioned
    assert not snapshot.entry("second").smoke_tested
    assert len([request for request in requests if "--version" not in request.command]) == 1

    no_tests = registry.probe(run_self_tests=False)
    assert all(entry.versioned and not entry.smoke_tested for entry in no_tests.entries)


def test_json_and_duckdb_projection_are_queryable_and_keep_claims_non_authoritative(
    tmp_path: Path,
) -> None:
    documentation = tmp_path / "prover_matrix.md"
    documentation.write_text(
        "| Prover | Access path | Primary fit |\n"
        "| --- | --- | --- |\n"
        "| Test Prover | Documented | Test checks |\n",
        encoding="utf-8",
    )
    runtime = FakeRuntime(_executable(tmp_path))
    snapshot = _registry(
        _definition(), runtime, documentation_path=documentation
    ).probe()

    payload = write_prover_matrix_projection(tmp_path / "matrix.json", snapshot)
    paths = prover_matrix_paths(tmp_path / "matrix.duckdb")

    assert paths.json_path.is_file()
    assert paths.duckdb_path.is_file()
    assert payload["query_store"]["schema_version"] == PROVER_MATRIX_DUCKDB_SCHEMA_VERSION
    on_disk = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert on_disk["snapshot_id"] == snapshot.snapshot_id
    assert on_disk["documentation_is_runtime_evidence"] is False

    capability_rows = query_prover_matrix(
        paths.json_path,
        columns=(
            "prover_id",
            "smoke_tested",
            "translation_conformant",
            "reconstruction_capable",
        ),
        where="prover_id = ?",
        parameters=("testprover",),
    )["rows"]
    assert capability_rows == [
        {
            "prover_id": "testprover",
            "smoke_tested": True,
            "translation_conformant": True,
            "reconstruction_capable": True,
        }
    ]
    components = query_prover_matrix(
        paths.duckdb_path,
        table="prover_components",
        columns=("component_kind", "content_identity"),
        where="prover_id = ?",
        parameters=("testprover",),
    )["rows"]
    assert {row["component_kind"] for row in components} == {
        kind.value for kind in IdentityKind
    }
    assert all(row["content_identity"].startswith("sha256:") for row in components)
    claims = query_prover_matrix(
        paths.duckdb_path,
        table="documentation_claims",
        columns=("evidence_class", "runtime_evidence"),
    )["rows"]
    assert claims == [
        {"evidence_class": "documentation_only", "runtime_evidence": False}
    ]
    catalog = query_prover_matrix(
        paths.duckdb_path,
        table="prover_matrix_catalog",
        columns=("documentation_is_runtime_evidence",),
    )["rows"]
    assert catalog == [{"documentation_is_runtime_evidence": False}]


def test_bound_identity_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="name and content_identity"):
        BoundIdentity(IdentityKind.MODEL, "", "sha256:x", True)
    with pytest.raises(ValueError, match="strict JSON"):
        BoundIdentity(
            IdentityKind.MODEL,
            "model",
            "sha256:x",
            True,
            {"invalid": float("nan")},
        )
    with pytest.raises(ValueError, match="between 1 and 1000"):
        query_prover_matrix("missing.json", limit=0)
