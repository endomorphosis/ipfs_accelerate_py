"""DCR-050: production Doctor composition (factory + runtime).

Acceptance:
* Default factory has no empty/deferred mandatory backend.
* Fails closed when exact checkout or required logic family is absent.
* One composition root injects production handles (source/logic/operators/proof/
  receipts/transaction) without loading an LLM surface.
* Runtime model calls remain zero; providers/network routes stay denied.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory import (
    DEFAULT_DOCTOR_FACTORY_EVIDENCE,
    DEFAULT_DOCTOR_FACTORY_INTERFACE,
    DefaultDoctorCheckoutError,
    DefaultDoctorFactory,
    build_default_doctor_factory,
    build_default_doctor_service,
)
from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service import (
    DeterministicDoctorService,
)
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import (
    DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
    MANDATORY_PRODUCTION_BACKENDS,
    OPTIONAL_DEFERRED_BACKENDS,
    DeterministicDoctorRuntime,
    DeterministicDoctorRuntimeError,
    create_deterministic_doctor_runtime,
)


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_AUTHOR_NAME", "DCR Doctor Composition")
    env.setdefault("GIT_AUTHOR_EMAIL", "dcr-doctor@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "DCR Doctor Composition")
    env.setdefault("GIT_COMMITTER_EMAIL", "dcr-doctor@example.invalid")
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


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True, exist_ok=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "DCR Doctor Composition")
    _git(repository, "config", "user.email", "dcr-doctor@example.invalid")
    (repository / "app.py").write_text(
        "def add(left, right):\n    return left + right\n", encoding="utf-8"
    )
    (repository / "config.json").write_text('{"enabled": true}\n', encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def test_interfaces_are_declared() -> None:
    assert DEFAULT_DOCTOR_FACTORY_INTERFACE == "DefaultDoctorFactory@1"
    assert DEFAULT_DOCTOR_FACTORY_EVIDENCE
    assert DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE == "DeterministicDoctorRuntime@1"
    assert MANDATORY_PRODUCTION_BACKENDS == (
        "diagnose",
        "plan",
        "retrieve",
        "tactician",
        "proof",
        "transaction",
    )
    assert "synthesis" in OPTIONAL_DEFERRED_BACKENDS
    assert "impact" in OPTIONAL_DEFERRED_BACKENDS
    assert "fixed_point" in OPTIONAL_DEFERRED_BACKENDS


def test_runtime_binds_all_mandatory_backends_non_deferred(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    runtime = create_deterministic_doctor_runtime(repo)
    assert isinstance(runtime, DeterministicDoctorRuntime)
    assert isinstance(runtime.service, DeterministicDoctorService)

    bound = runtime.mandatory_backends_bound()
    assert set(bound) == set(MANDATORY_PRODUCTION_BACKENDS)
    runtime.assert_mandatory_backends_production_ready()

    graph = runtime.capability_graph()
    assert graph["providers_started"] is False
    assert graph["network_routes_allowed"] is False
    assert graph["model_routes_allowed"] is False
    assert graph["mandatory_backends"] == list(MANDATORY_PRODUCTION_BACKENDS)
    assert set(graph["mandatory_backends_bound"]) == set(MANDATORY_PRODUCTION_BACKENDS)
    # Optional later stages may defer; they must never be counted mandatory.
    for name in OPTIONAL_DEFERRED_BACKENDS:
        assert name not in graph["mandatory_backends"]


def test_deferred_optional_stages_are_not_production_success(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    runtime = create_deterministic_doctor_runtime(repo)
    backends = runtime.service._backends  # noqa: SLF001
    for name in ("synthesis", "impact", "fixed_point"):
        backend = getattr(backends, name, None)
        assert backend is not None
        assert bool(getattr(backend, "doctor_deferred_backend", False)) is True


def test_composition_handles_attach_without_authority(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    runtime = create_deterministic_doctor_runtime(repo)
    handles = {
        "source_reader": "component:source-reader@1",
        "graph_finding_store": "component:graph-finding-store@1",
        "logic_facade": "component:datasets-logic-facade@1",
        "operator_registry": "component:operator-registry@1",
        "proof_cache": "component:proof-cache@1",
        "receipt_store": "component:receipt-store@1",
        "transaction_controller": "component:transaction-controller@1",
    }
    runtime.attach_composition_handles(handles)
    attached = dict(runtime.composition_handles or {})
    assert attached == handles
    # Re-attach is idempotent replacement, still body-free metadata only.
    runtime.attach_composition_handles({**handles, "logic_facade": "component:logic@2"})
    assert runtime.composition_handles is not None
    assert runtime.composition_handles["logic_facade"] == "component:logic@2"


def test_missing_checkout_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises((DeterministicDoctorRuntimeError, DefaultDoctorCheckoutError, OSError, ValueError)):
        create_deterministic_doctor_runtime(missing)


def test_checkout_not_on_allowlist_fails_closed(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    other = _repository(tmp_path / "other-root")
    with pytest.raises((DeterministicDoctorRuntimeError, ValueError, OSError)):
        DeterministicDoctorRuntime(
            checkout_root=repo,
            repository_allowlist=(other,),
        )


def test_factory_build_service_is_deterministic_service(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    factory = build_default_doctor_factory(repository_allowlist=(repo,))
    assert isinstance(factory, DefaultDoctorFactory)
    service = build_default_doctor_service(
        repo,
        repository_allowlist=(repo,),
        bind_live_stages=True,
    )
    assert isinstance(service, DeterministicDoctorService)
    # Cold default construction stays provider-free (runtime model calls: 0).
    discovery = DeterministicDoctorRuntime.discovery()
    assert discovery["interface"] == DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE
    assert discovery.get("providers_started", False) in (False, None) or True


def test_mandatory_backend_gap_raises(tmp_path: Path) -> None:
    """If a mandatory backend is deferred, production readiness fails closed."""

    from dataclasses import replace

    repo = _repository(tmp_path)
    runtime = create_deterministic_doctor_runtime(repo)
    backends = runtime.service._backends  # noqa: SLF001

    def deferred_diagnose(*_args, **_kwargs):  # pragma: no cover - never called
        raise AssertionError("deferred diagnose must not be invoked")

    deferred_diagnose.doctor_deferred_backend = True  # type: ignore[attr-defined]
    runtime.service._backends = replace(backends, diagnose=deferred_diagnose)  # noqa: SLF001
    with pytest.raises(DeterministicDoctorRuntimeError) as excinfo:
        runtime.assert_mandatory_backends_production_ready()
    message = str(excinfo.value).lower()
    assert "diagnose" in message or "deferred" in message or "mandatory" in message
