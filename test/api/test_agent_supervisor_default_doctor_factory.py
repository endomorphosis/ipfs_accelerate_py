"""Tests for DefaultDoctorFactory@1 / build_default_doctor_service (WPD-010)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorMode,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory import (
    DEFAULT_DOCTOR_FACTORY_EVIDENCE,
    DEFAULT_DOCTOR_FACTORY_INTERFACE,
    DEFAULT_DOCTOR_FACTORY_VERSION,
    DefaultDoctorCheckoutError,
    DefaultDoctorFactory,
    assert_no_llm_surface_loaded,
    build_default_doctor_factory,
    build_default_doctor_service,
)
from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service import (
    DeterministicDoctorService,
    DoctorServiceCapabilityCode,
    DoctorStageBackends,
)


@pytest.fixture(autouse=True)
def _hermetic_git_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Make fixture + live-analyze git safe under sealed validation HOMEs.

    The authoritative validation runner uses a private HOME and may execute as
    a different filesystem owner than the worktree.  Git then refuses both
    fixture setup and PlanningAnalysisFactory inventory with "dubious
    ownership".  A process-scoped global config (and author identity) keeps
    every git subprocess — including those inside live stage backends —
    usable for exact local roots without weakening Doctor safety gates.
    """

    config_root = tmp_path_factory.mktemp("doctor-factory-gitconfig")
    config_path = config_root / "config"
    config_path.write_text(
        "[user]\n"
        "\tname = Default Doctor Factory\n"
        "\temail = doctor-factory@example.invalid\n"
        "[safe]\n"
        "\tdirectory = *\n"
        "[init]\n"
        "\tdefaultBranch = main\n",
        encoding="utf-8",
    )
    empty_system = config_root / "system"
    empty_system.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config_path))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_system))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Default Doctor Factory")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "doctor-factory@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Default Doctor Factory")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "doctor-factory@example.invalid")
    monkeypatch.setenv("EMAIL", "doctor-factory@example.invalid")


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_AUTHOR_NAME", "Default Doctor Factory")
    env.setdefault("GIT_AUTHOR_EMAIL", "doctor-factory@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "Default Doctor Factory")
    env.setdefault("GIT_COMMITTER_EMAIL", "doctor-factory@example.invalid")
    return env


def _git(repository: Path, *arguments: str) -> str:
    resolved = str(repository.resolve())
    result = subprocess.run(
        (
            "git",
            "-c",
            f"safe.directory={resolved}",
            "-c",
            "safe.directory=*",
            "-C",
            resolved,
            *arguments,
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_git_env(),
    )
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repository(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True, exist_ok=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Default Doctor Factory")
    _git(repository, "config", "user.email", "doctor-factory@example.invalid")
    # Mirror the runtime fixture shape: semantic Python + structured data + docs.
    payload = files or {
        "app.py": "def add(left, right):\n    return left + right\n",
        "config.json": '{"enabled": true}\n',
        "README.md": "fixture\n",
    }
    for relative, body in payload.items():
        _write(repository / relative, body)
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def test_interfaces_and_evidence_key_are_stable() -> None:
    assert DEFAULT_DOCTOR_FACTORY_INTERFACE == "DefaultDoctorFactory@1"
    assert DEFAULT_DOCTOR_FACTORY_VERSION == 1
    assert DEFAULT_DOCTOR_FACTORY_EVIDENCE == "wpd/default-doctor-factory@1"
    discovery = DefaultDoctorFactory.discovery()
    assert discovery["interface"] == DEFAULT_DOCTOR_FACTORY_INTERFACE
    assert discovery["evidence_key"] == DEFAULT_DOCTOR_FACTORY_EVIDENCE
    assert discovery["llm_router_enabled"] is False
    assert discovery["automatic_fallback"] is False
    assert discovery["live_checkout_composition"] is True
    assert "diagnose" in discovery["stage_slots"]
    assert "plan" in discovery["stage_slots"]


def test_default_construction_loads_no_llm_surface() -> None:
    assert_no_llm_surface_loaded()
    factory = build_default_doctor_factory()
    service = factory.build()
    assert isinstance(service, DeterministicDoctorService)
    assert service.backends_available == ()
    assert_no_llm_surface_loaded()

    binding = factory.last_binding
    assert binding is not None
    assert binding.live_stages_bound is False
    assert "diagnose" in binding.capability_gaps
    assert "plan" in binding.capability_gaps
    assert "all_stage_slots_empty" in binding.notes


def test_missing_backend_yields_typed_abstention() -> None:
    service = build_default_doctor_service()
    assert_no_llm_surface_loaded()

    inspect = service.inspect(incident_id="incident:missing-inspect")
    assert inspect.abstained
    assert inspect.read_only is True
    assert inspect.changed is False
    assert (
        DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value in inspect.reason_codes
    )
    assert (
        DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value
        in inspect.reason_codes
    )
    assert "inspect_without_snapshot" in inspect.reason_codes

    planned = service.plan(incident_id="incident:missing-plan")
    assert planned.abstained
    assert planned.read_only is True
    assert (
        DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value in planned.reason_codes
    )
    assert (
        DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value
        in planned.reason_codes
    )

    # Repair without an enabled policy / transaction backend is a typed gap,
    # never a free-form provider fallback.
    repair = service.repair(incident_id="incident:missing-repair")
    assert not repair.succeeded
    assert repair.changed is False
    assert repair.disposition in {
        DoctorRepairDisposition.ABSTAIN,
        DoctorRepairDisposition.QUARANTINED,
        DoctorRepairDisposition.APPROVAL_REQUIRED,
    }
    assert any(
        code
        in {
            DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
            DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
            DoctorServiceCapabilityCode.POLICY_REJECTED.value,
            DoctorServiceCapabilityCode.POLICY_ABSTAINED.value,
            DoctorServiceCapabilityCode.POLICY_APPROVAL_REQUIRED.value,
        }
        for code in repair.reason_codes
    )


def test_build_default_doctor_service_inspect_and_plan_on_fixture(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    assert_no_llm_surface_loaded()

    service = build_default_doctor_service(checkout_root=repository)
    assert isinstance(service, DeterministicDoctorService)
    assert_no_llm_surface_loaded()

    # Live composition binds diagnose/plan (and deferred analytical stages).
    available = set(service.backends_available)
    assert "diagnose" in available
    assert "plan" in available
    # Explicit capability gaps remain typed (explain is not live-bound by default).
    assert "explain" not in available

    inspect = service.inspect(incident_id="incident:fixture-inspect")
    assert inspect.succeeded, (
        f"inspect expected SUPPORTED, got {inspect.disposition!r} "
        f"reasons={inspect.reason_codes!r} explanation={inspect.explanation!r}"
    )
    assert inspect.read_only is True
    assert inspect.changed is False
    assert inspect.disposition is DoctorRepairDisposition.SUPPORTED
    assert inspect.run_receipt is not None
    assert inspect.run_receipt.snapshot_id
    assert inspect.run_receipt.network_denied is True
    assert inspect.run_receipt.llm_router_invoked is False
    assert inspect.run_receipt.model_invocation_count == 0

    planned = service.plan(
        mode=DoctorMode.PLAN.value,
        incident_id="incident:fixture-plan",
    )
    # Plan produces a typed result: either a report or an actionable abstention
    # when analytical inputs remain open (never a free-form LLM fallback).
    assert planned.read_only is True
    assert planned.changed is False
    assert planned.disposition in {
        DoctorRepairDisposition.SUPPORTED,
        DoctorRepairDisposition.ABSTAIN,
        DoctorRepairDisposition.APPROVAL_REQUIRED,
    }, (
        f"plan disposition unexpected: {planned.disposition!r} "
        f"reasons={planned.reason_codes!r} explanation={planned.explanation!r}"
    )
    if planned.abstained:
        assert planned.reason_codes
        assert any(
            code
            in {
                DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
                DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                "plan_inputs_deferred",
            }
            or code.startswith("stage_unavailable:")
            for code in planned.reason_codes
        ), f"plan abstention reasons not actionable: {planned.reason_codes!r}"
        # Default live plan defers typed stage inputs without starting providers.
        assert planned.status.get("automatic_fallback") is False
    assert_no_llm_surface_loaded()


def test_factory_build_records_live_binding(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = DefaultDoctorFactory()
    service = factory.build(checkout_root=repository)

    binding = factory.last_binding
    assert binding is not None
    assert binding.live_stages_bound is True
    assert binding.checkout_root == str(repository.resolve())
    assert "diagnose" in binding.backends_available
    assert "plan" in binding.backends_available
    assert "live_checkout_stages_bound" in binding.notes
    assert factory.last_runtime is not None
    assert factory.last_runtime.service is service
    assert binding.binding_id


def test_explicit_empty_backends_skip_live_composition(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = build_default_doctor_factory()
    service = factory.build(
        checkout_root=repository,
        backends=DoctorStageBackends(),
    )
    assert service.backends_available == ()
    assert factory.last_binding is not None
    assert factory.last_binding.live_stages_bound is False
    assert factory.last_runtime is None

    result = service.inspect()
    assert result.abstained
    assert DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value in result.reason_codes


def test_bind_live_stages_false_leaves_capability_gaps(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    service = build_default_doctor_service(
        checkout_root=repository,
        bind_live_stages=False,
    )
    assert service.backends_available == ()
    result = service.plan()
    assert result.abstained


def test_missing_checkout_root_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(DefaultDoctorCheckoutError) as exc_info:
        build_default_doctor_service(checkout_root=missing)
    assert exc_info.value.reason_code == "checkout_unavailable"


def test_checkout_must_be_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x\n", encoding="utf-8")
    with pytest.raises(DefaultDoctorCheckoutError) as exc_info:
        build_default_doctor_service(checkout_root=file_path)
    assert exc_info.value.reason_code == "checkout_not_directory"


def test_checkout_not_allowlisted_raises(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    rejected_root = tmp_path / "rejected"
    allowed_root.mkdir()
    rejected_root.mkdir()
    allowed = _repository(allowed_root)
    rejected = _repository(rejected_root)
    with pytest.raises(DefaultDoctorCheckoutError) as exc_info:
        build_default_doctor_service(
            checkout_root=rejected,
            repository_allowlist=(allowed,),
        )
    assert exc_info.value.reason_code == "checkout_not_allowlisted"


def test_module_cold_import_does_not_pull_runtime() -> None:
    """Factory module load must stay control-local (runtime is lazy)."""

    import ast

    path = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "control"
        / "default_doctor_factory.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level_imports.append(node.module)
    joined = "\n".join(top_level_imports)
    assert "deterministic_doctor_runtime" not in joined
    assert "openai" not in joined
    assert "anthropic" not in joined
    assert "torch" not in joined
    assert "transformers" not in joined
    assert "llm_router" not in joined
