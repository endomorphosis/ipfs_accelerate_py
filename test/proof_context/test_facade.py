"""PCCE-020: provider-neutral ProofCarryingContextEngine facade."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from ipfs_accelerate_py.proof_context.compatibility import CompatibilityError
from ipfs_accelerate_py.proof_context.facade import (
    COMPATIBILITY_MATRIX_CONTENT_ID,
    CONTRACT_SCHEMA_PREFIX,
    CONTRACT_VERSION,
    ENGINE_RECORD_SCHEMA,
    EPIC_A_GATE_CONTENT_ID,
    EPIC_A_GATE_TASK,
    INSTANCE_OPERATIONS,
    OPERATION_CONTRACTS,
    OPERATIONS,
    PCCE_006_CONTENT_ID,
    PROVIDER_BOUND,
    SCHEMA,
    SIBLING_LAYOUT_REQUIRED,
    AssurancePort,
    EngineIdentities,
    EnginePorts,
    EngineRecord,
    ExecutionPort,
    FacadeError,
    PersistencePort,
    ProofCarryingContextEngine,
    ReportPort,
    RoutePort,
    SealingPort,
    SemanticPort,
    VerificationPort,
    public_signature_snapshot,
)

FROZEN_SIGNATURE_SNAPSHOT = {
    "open": {
        "parameters": ("repository", "ports", "identities", "mode"),
        "keyword_only": ("ports", "identities", "mode"),
        "return": "ProofCarryingContextEngine",
    },
    "scan": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "status": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "plan": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "context_pack": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "route": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "run": {
        "parameters": ("proposal",),
        "keyword_only": (),
        "return": "EngineRecord",
    },
    "verify": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "expand_context": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "assurance": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "seal": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "report": {"parameters": (), "keyword_only": (), "return": "EngineRecord"},
    "resume": {
        "parameters": ("checkpoint",),
        "keyword_only": (),
        "return": "EngineRecord",
    },
}


def _cid(label: str) -> str:
    body = "".join(ch if ch in "abcdefghijklmnopqrstuvwxyz234567" else "a" for ch in label)
    return "b" + (body + "a" * 58)[:58]


def _identities(**overrides: Any) -> EngineIdentities:
    values = {
        "repository_id": "example/ordinary-python-repo",
        "repository_state_cid": _cid("repostate"),
        "task_id": "PCCE-020",
        "run_id": "run-pcce-020",
        "trace_id": "trace-pcce-020",
        "contract_version": CONTRACT_VERSION,
        "patch_id": None,
        "artifact_id": None,
    }
    values.update(overrides)
    return EngineIdentities(**values)


def _record(
    operation: str,
    identities: EngineIdentities,
    *,
    artifact: str | None = None,
    payload: Mapping[str, Any] | None = None,
    provenance: str = "live",
    status: str = "succeeded",
) -> EngineRecord:
    return EngineRecord(
        schema=ENGINE_RECORD_SCHEMA,
        operation=operation,
        status=status,
        identities=identities,
        artifact_cid=artifact or _cid(operation),
        provenance=provenance,
        payload=payload or {},
    )


@dataclass
class FakeSemanticPort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)
    drift: EngineIdentities | None = None

    def _emit(self, operation: str, identities: EngineIdentities) -> EngineRecord:
        self.calls.append(operation)
        return _record(operation, self.drift or identities)

    def scan(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        return self._emit("scan", identities)

    def plan(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        return self._emit("plan", identities)

    def context_pack(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord:
        return self._emit("context-pack", identities)

    def expand_context(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord:
        return self._emit("expand-context", identities)


@dataclass
class FakePersistencePort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)

    def resume(
        self,
        identities: EngineIdentities,
        repository: Path,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> EngineRecord:
        self.calls.append("resume")
        payload = {"checkpoint": dict(checkpoint or {})}
        return _record("resume", identities, payload=payload)


@dataclass
class FakeRoutePort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)
    payload: Mapping[str, Any] = field(
        default_factory=lambda: {
            "tier": "small_local_model",
            "provider": "unspecified",
            "model": "capability-tier-only",
        }
    )

    def route(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        self.calls.append("route")
        return _record("route", identities, payload=self.payload)


@dataclass
class FakeExecutionPort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)
    patch_id: str = field(default_factory=lambda: _cid("patch"))

    def run(
        self,
        identities: EngineIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None = None,
    ) -> EngineRecord:
        self.calls.append("run")
        bound = EngineIdentities.from_mapping(
            {
                **identities.to_mapping(),
                "patch_id": self.patch_id,
                "artifact_id": _cid("run-artifact"),
            }
        )
        return _record(
            "run",
            bound,
            artifact=_cid("run-artifact"),
            payload={"proposal": dict(proposal or {})},
        )


@dataclass
class FakeVerificationPort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)

    def verify(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        self.calls.append("verify")
        return _record("verify", identities)


@dataclass
class FakeAssurancePort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)

    def assurance(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord:
        self.calls.append("assurance")
        return _record("assurance", identities)


@dataclass
class FakeSealingPort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)

    def seal(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        self.calls.append("seal")
        bound = EngineIdentities.from_mapping(
            {
                **identities.to_mapping(),
                "artifact_id": identities.artifact_id or _cid("seal-artifact"),
            }
        )
        return _record("seal", bound, artifact=_cid("seal-artifact"))


@dataclass
class FakeReportPort:
    identities: EngineIdentities
    calls: list[str] = field(default_factory=list)

    def status(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        self.calls.append("status")
        return _record("status", identities)

    def report(self, identities: EngineIdentities, repository: Path) -> EngineRecord:
        self.calls.append("report")
        return _record("report", identities)


def _ports(identities: EngineIdentities, **overrides: Any) -> EnginePorts:
    values = {
        "semantic": FakeSemanticPort(identities),
        "persistence": FakePersistencePort(identities),
        "route": FakeRoutePort(identities),
        "execution": FakeExecutionPort(identities),
        "verification": FakeVerificationPort(identities),
        "assurance": FakeAssurancePort(identities),
        "sealing": FakeSealingPort(identities),
        "report": FakeReportPort(identities),
    }
    values.update(overrides)
    return EnginePorts(**values)


def _ordinary_python_repo(root: Path) -> Path:
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    package = root / "src" / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return root


def test_cold_import_creates_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    imported = importlib.import_module("ipfs_accelerate_py.proof_context.facade")
    after = set(tmp_path.rglob("*"))
    assert after == before
    assert imported.SCHEMA == SCHEMA
    assert imported.SIBLING_LAYOUT_REQUIRED is False
    assert imported.PROVIDER_BOUND is False


def test_public_signature_snapshot_is_stable() -> None:
    snapshot = public_signature_snapshot()
    assert {key: dict(value) for key, value in snapshot.items()} == FROZEN_SIGNATURE_SNAPSHOT
    assert tuple(snapshot) == (
        "open",
        "scan",
        "status",
        "plan",
        "context_pack",
        "route",
        "run",
        "verify",
        "expand_context",
        "assurance",
        "seal",
        "report",
        "resume",
    )
    assert OPERATIONS == (
        "open",
        "scan",
        "status",
        "plan",
        "context-pack",
        "route",
        "run",
        "verify",
        "expand-context",
        "assurance",
        "seal",
        "report",
        "resume",
    )
    assert INSTANCE_OPERATIONS == OPERATIONS[1:]
    for method_name in (
        "scan",
        "status",
        "plan",
        "context_pack",
        "route",
        "run",
        "verify",
        "expand_context",
        "assurance",
        "seal",
        "report",
        "resume",
    ):
        assert callable(getattr(ProofCarryingContextEngine, method_name))


def test_open_ordinary_python_repository(tmp_path: Path) -> None:
    repo = _ordinary_python_repo(tmp_path)
    identities = _identities()
    engine = ProofCarryingContextEngine.open(
        repo,
        ports=_ports(identities),
        identities=identities,
    )
    assert engine.repository == repo
    assert engine.mode == "production"
    assert engine.identities.task_id == "PCCE-020"
    assert engine.identities.contract_version == "0.1"
    assert engine.provider_bound is False
    assert engine.sibling_layout_required is False
    assert not (repo / "ipfs_datasets_py").exists()
    assert not (repo / "ipfs_kit_py").exists()


def test_open_requires_injected_ports_and_directory(tmp_path: Path) -> None:
    identities = _identities()
    missing = tmp_path / "missing"
    with pytest.raises(FacadeError) as missing_exc:
        ProofCarryingContextEngine.open(
            missing,
            ports=_ports(identities),
            identities=identities,
        )
    assert missing_exc.value.reason == "invalid"
    with pytest.raises(FacadeError) as mode_exc:
        ProofCarryingContextEngine.open(
            tmp_path,
            ports=_ports(identities),
            identities=identities,
            mode="shadow",
        )
    assert mode_exc.value.reason == "unknown_field"
    with pytest.raises(FacadeError) as ports_exc:
        ProofCarryingContextEngine.open(
            tmp_path,
            ports=object(),  # type: ignore[arg-type]
            identities=identities,
        )
    assert ports_exc.value.reason == "malformed"


def test_fake_ports_cover_every_operation_and_preserve_identities(
    tmp_path: Path,
) -> None:
    repo = _ordinary_python_repo(tmp_path)
    identities = _identities()
    ports = _ports(identities)
    engine = ProofCarryingContextEngine.open(
        repo,
        ports=ports,
        identities=identities,
        mode="supervised",
    )
    scan = engine.scan()
    status = engine.status()
    plan = engine.plan()
    pack = engine.context_pack()
    route = engine.route()
    run = engine.run({"declared_files": ["src/demo/__init__.py"]})
    verify = engine.verify()
    expanded = engine.expand_context()
    assurance = engine.assurance()
    seal = engine.seal()
    report = engine.report()
    resume = engine.resume({"run_id": identities.run_id, "trace_id": identities.trace_id})

    records = {
        "scan": scan,
        "status": status,
        "plan": plan,
        "context-pack": pack,
        "route": route,
        "run": run,
        "verify": verify,
        "expand-context": expanded,
        "assurance": assurance,
        "seal": seal,
        "report": report,
        "resume": resume,
    }
    assert tuple(records) == INSTANCE_OPERATIONS
    for operation, record in records.items():
        assert record.operation == operation
        assert record.identities.repository_id == identities.repository_id
        assert record.identities.repository_state_cid == identities.repository_state_cid
        assert record.identities.task_id == identities.task_id
        assert record.identities.run_id == identities.run_id
        assert record.identities.trace_id == identities.trace_id
        assert record.identities.contract_version == CONTRACT_VERSION
        assert record.artifact_cid.startswith("b")
        assert OPERATION_CONTRACTS[operation].startswith(CONTRACT_SCHEMA_PREFIX)
        assert record.to_mapping()["identities"]["run_id"] == identities.run_id

    assert engine.identities.patch_id == ports.execution.patch_id
    assert verify.identities.patch_id == ports.execution.patch_id
    assert seal.identities.patch_id == ports.execution.patch_id
    assert ports.semantic.calls == ["scan", "plan", "context-pack", "expand-context"]
    assert ports.route.calls == ["route"]
    assert ports.execution.calls == ["run"]
    assert ports.verification.calls == ["verify"]
    assert ports.assurance.calls == ["assurance"]
    assert ports.sealing.calls == ["seal"]
    assert ports.report.calls == ["status", "report"]
    assert ports.persistence.calls == ["resume"]
    assert isinstance(ports.semantic, SemanticPort)
    assert isinstance(ports.persistence, PersistencePort)
    assert isinstance(ports.route, RoutePort)
    assert isinstance(ports.execution, ExecutionPort)
    assert isinstance(ports.verification, VerificationPort)
    assert isinstance(ports.assurance, AssurancePort)
    assert isinstance(ports.sealing, SealingPort)
    assert isinstance(ports.report, ReportPort)


def test_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    identities = _identities()
    drifted = _identities(task_id="OTHER-TASK")
    engine = ProofCarryingContextEngine.open(
        tmp_path,
        ports=_ports(identities, semantic=FakeSemanticPort(identities, drift=drifted)),
        identities=identities,
    )
    with pytest.raises(FacadeError) as exc:
        engine.scan()
    assert exc.value.reason == "identity_inconsistent"
    with pytest.raises(FacadeError) as payload_exc:
        engine.resume({"task_id": "OTHER-TASK"})
    assert payload_exc.value.reason == "identity_inconsistent"


def test_patch_and_artifact_identities_bind_once(tmp_path: Path) -> None:
    identities = _identities()
    execution = FakeExecutionPort(identities, patch_id=_cid("first-patch"))
    engine = ProofCarryingContextEngine.open(
        tmp_path,
        ports=_ports(identities, execution=execution),
        identities=identities,
    )
    first = engine.run()
    assert first.identities.patch_id == execution.patch_id
    execution.patch_id = _cid("second-patch")
    with pytest.raises(FacadeError) as exc:
        engine.run()
    assert exc.value.reason == "identity_inconsistent"


def test_production_rejects_mocks_pseudo_cids_and_simulated_promotion(
    tmp_path: Path,
) -> None:
    identities = _identities()
    mock_ports = EnginePorts(
        semantic=Mock(),
        persistence=Mock(),
        route=Mock(),
        execution=Mock(),
        verification=Mock(),
        assurance=Mock(),
        sealing=Mock(),
        report=Mock(),
    )
    with pytest.raises(CompatibilityError):
        ProofCarryingContextEngine.open(
            tmp_path,
            ports=mock_ports,
            identities=identities,
            mode="production",
        )
    with pytest.raises(CompatibilityError):
        _identities(repository_state_cid="sha256:deadbeef")
    simulated = FakeRoutePort(identities)
    original_route = simulated.route

    def _simulated_route(
        bound: EngineIdentities, repository: Path
    ) -> EngineRecord:
        record = original_route(bound, repository)
        return EngineRecord(
            schema=record.schema,
            operation=record.operation,
            status="succeeded",
            identities=record.identities,
            artifact_cid=record.artifact_cid,
            provenance="simulated",
            payload=record.payload,
        )

    simulated.route = _simulated_route  # type: ignore[method-assign]
    engine = ProofCarryingContextEngine.open(
        tmp_path,
        ports=_ports(identities, route=simulated),
        identities=identities,
        mode="production",
    )
    with pytest.raises(FacadeError) as exc:
        engine.route()
    assert exc.value.reason == "simulated_promoted"


def test_provider_neutral_ast_and_route_has_no_credentials(tmp_path: Path) -> None:
    source = Path(inspect.getfile(ProofCarryingContextEngine)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "openai" not in imported
    assert "anthropic" not in imported
    assert "ipfs_datasets_py" not in imported
    assert "ipfs_kit_py" not in imported
    assert PROVIDER_BOUND is False
    identities = _identities()
    engine = ProofCarryingContextEngine.open(
        tmp_path,
        ports=_ports(
            identities,
            route=FakeRoutePort(
                identities,
                payload={
                    "tier": "medium_model",
                    "credentials": {"token": "secret"},
                },
            ),
        ),
        identities=identities,
    )
    with pytest.raises(FacadeError) as exc:
        engine.route()
    assert exc.value.reason == "boundary_violation"
    clean = ProofCarryingContextEngine.open(
        tmp_path,
        ports=_ports(identities),
        identities=identities,
    )
    decision = clean.route()
    assert decision.payload["tier"] == "small_local_model"
    assert "credentials" not in decision.payload


def test_epic_a_contract_version_binding() -> None:
    assert CONTRACT_VERSION == "0.1"
    assert SCHEMA.endswith("v0.1")
    assert CONTRACT_SCHEMA_PREFIX == "pcce/proof-context/v0.1/"
    assert EPIC_A_GATE_TASK == "PCCE-011"
    assert EPIC_A_GATE_CONTENT_ID.startswith("sha256:")
    assert EPIC_A_GATE_CONTENT_ID.endswith("14ce6")
    assert PCCE_006_CONTENT_ID.endswith("43f37")
    assert COMPATIBILITY_MATRIX_CONTENT_ID.endswith("e920")
    for contract in OPERATION_CONTRACTS.values():
        assert contract.startswith(CONTRACT_SCHEMA_PREFIX)


def test_import_side_effect_trace_does_not_bind_sys_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    banned = {
        "openai",
        "anthropic",
        "ipfs_datasets_py.proof_context.provider",
        "ipfs_kit_py.proof_context.state_store",
    }
    before = {name: sys.modules.get(name) for name in banned}
    imported = importlib.import_module("ipfs_accelerate_py.proof_context.facade")
    for name in banned:
        assert sys.modules.get(name) is before.get(name)
    assert imported.PROVIDER_BOUND is False
    after = set(tmp_path.rglob("*"))
    assert after == set()


def test_cold_import_subprocess_is_hermetic(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(package_root), existing) if part
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HOME"] = str(tmp_path)
    before = set(tmp_path.rglob("*"))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ipfs_accelerate_py.proof_context.facade as facade; "
                "print(facade.SCHEMA); "
                "print(facade.PROVIDER_BOUND); "
                "print(facade.SIBLING_LAYOUT_REQUIRED)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    after = set(tmp_path.rglob("*"))
    assert after == before
    stdout = completed.stdout.splitlines()
    assert stdout[0] == SCHEMA
    assert stdout[1] == "False"
    assert stdout[2] == "False"
