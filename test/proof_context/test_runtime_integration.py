"""PCCE-025: stable lazy-loaded runtime package and external-patch path."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.proof_context.bootstrap import (
    DATASETS_PORT,
    KIT_PORT,
    RUNTIME_CID,
    RUNTIME_DESCRIPTOR,
    RUNTIME_SCHEMA,
    RuntimeOptions,
    create_ordinary_python_repository,
    open_runtime,
    runtime_descriptor,
)
from ipfs_accelerate_py.proof_context.dependencies import (
    Capability,
    DependencyUnavailable,
    require_production_capability,
    resolve_datasets,
    resolve_kit,
)
from ipfs_accelerate_py.proof_context.errors import (
    ERRORS,
    BoundaryViolationError,
    ProofContextError,
    UnavailableCapabilityError,
)
from ipfs_accelerate_py.proof_context.facade import (
    INSTANCE_OPERATIONS,
    OPERATION_CONTRACTS,
    OPERATIONS,
    PROVIDER_BOUND,
    SCHEMA,
    SIBLING_LAYOUT_REQUIRED,
    EngineRecord,
    FacadeError,
    ProofCarryingContextEngine,
    public_signature_snapshot,
)
from ipfs_accelerate_py.proof_context.lifecycle import LIFECYCLE_CID, STAGES
from ipfs_accelerate_py.proof_context.policy import POLICY_CID
from ipfs_accelerate_py.proof_context.recovery import RECOVERY_CID
from ipfs_accelerate_py.proof_context.results import RESULT_STATE_CID

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_API_PATH = (
    WORKSPACE_ROOT
    / "artifacts"
    / "proof_carrying_context_engine"
    / "runtime"
    / "runtime_api.json"
)
EXTERNAL_PATCH = {
    "declared_files": ["src/demo/__init__.py"],
    "files": {"src/demo/__init__.py": "VALUE = 2\n"},
    "adapter_id": "external-patch",
    "approver_id": "coordinator",
}


def _runtime(tmp_path: Path, **overrides: Any):
    repo = create_ordinary_python_repository(tmp_path / "repo")
    options = RuntimeOptions(
        kit_root=tmp_path / "kit",
        worktree_parent=tmp_path / "worktrees",
        **overrides,
    )
    bundle = open_runtime(repo, options=options)
    return repo, bundle


def test_cold_import_creates_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    imported = importlib.import_module("ipfs_accelerate_py.proof_context")
    after = set(tmp_path.rglob("*"))
    assert after == before
    assert imported.SCHEMA == SCHEMA
    assert imported.SCHEMA.endswith("v0.1")


def test_cold_import_does_not_bind_datasets_kit_or_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    banned = {
        "openai",
        "anthropic",
        "ipfs_datasets_py.proof_context.provider",
        "ipfs_kit_py.proof_context.state_store",
        "ipfs_accelerate_py.proof_context.facade",
        "ipfs_accelerate_py.proof_context.bootstrap",
        "ipfs_accelerate_py.proof_context.lifecycle",
        "ipfs_accelerate_py.proof_context.recovery",
    }
    sys.modules.pop("ipfs_accelerate_py.proof_context", None)
    before = {name: sys.modules.get(name) for name in banned}
    imported = importlib.import_module("ipfs_accelerate_py.proof_context")
    for name in banned:
        assert sys.modules.get(name) is before.get(name)
    assert imported.SCHEMA.endswith("v0.1")
    assert set(tmp_path.rglob("*")) == set()


def test_cold_import_subprocess_is_hermetic(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(package_root), existing) if part
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HOME"] = str(tmp_path)
    env["GIT_TERMINAL_PROMPT"] = "0"
    before = set(tmp_path.rglob("*"))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ipfs_accelerate_py.proof_context as port; "
                "print(port.SCHEMA); "
                "print('bootstrap' in port.__all__); "
                "print(hasattr(port, 'SCHEMA'))"
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


def test_package_exports_are_lazy_and_stable() -> None:
    port = importlib.import_module("ipfs_accelerate_py.proof_context")
    assert port.ProofCarryingContextEngine is ProofCarryingContextEngine
    assert callable(port.open_engine)
    assert callable(port.open_runtime)
    assert port.RUNTIME_CID == RUNTIME_CID
    assert port.LIFECYCLE_CID == LIFECYCLE_CID
    assert port.POLICY_CID == POLICY_CID
    assert port.RESULT_STATE_CID == RESULT_STATE_CID
    assert port.RECOVERY_CID == RECOVERY_CID
    assert port.SIBLING_LAYOUT_REQUIRED is False
    assert port.PROVIDER_BOUND is False


def test_bootstrap_ast_is_provider_neutral() -> None:
    source = Path(inspect.getfile(open_runtime)).read_text(encoding="utf-8")
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
    assert SIBLING_LAYOUT_REQUIRED is False


def test_open_initializes_ordinary_python_git_repository(tmp_path: Path) -> None:
    repo, bundle = _runtime(tmp_path)
    engine = bundle.engine
    assert isinstance(engine, ProofCarryingContextEngine)
    assert engine.repository == repo
    assert engine.mode == "production"
    assert engine.identities.contract_version == "0.1"
    assert engine.identities.repository_state_cid.startswith("b")
    assert (repo / "pyproject.toml").is_file()
    assert (repo / ".git").exists()
    assert (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (repo / "ipfs_datasets_py").exists()
    assert not (repo / "ipfs_kit_py").exists()
    assert bundle.descriptor["cid"] == RUNTIME_CID
    assert bundle.datasets.name == "datasets"
    assert bundle.kit.name == "kit"


def test_complete_external_patch_path_uses_isolated_worktree(tmp_path: Path) -> None:
    repo, bundle = _runtime(tmp_path)
    engine = bundle.engine
    before = (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8")
    head_before = bundle.session.canonical_head

    scan = engine.scan()
    plan = engine.plan()
    pack = engine.context_pack()
    route = engine.route()
    applied = engine.run(EXTERNAL_PATCH)
    verify = engine.verify()
    expanded = engine.expand_context()
    assurance = engine.assurance()
    seal = engine.seal()
    report = engine.report()
    status = engine.status()
    resumed = engine.resume()

    records = {
        "scan": scan,
        "plan": plan,
        "context-pack": pack,
        "route": route,
        "run": applied,
        "verify": verify,
        "expand-context": expanded,
        "assurance": assurance,
        "seal": seal,
        "report": report,
        "status": status,
        "resume": resumed,
    }
    for operation, record in records.items():
        assert isinstance(record, EngineRecord)
        assert record.operation == operation
        assert record.identities.repository_id == engine.identities.repository_id
        assert record.identities.run_id == engine.identities.run_id
        assert record.identities.trace_id == engine.identities.trace_id
        assert record.identities.contract_version == "0.1"
        assert record.artifact_cid.startswith("b")
        assert record.provenance != "simulated"
        if operation in OPERATION_CONTRACTS:
            assert OPERATION_CONTRACTS[operation].startswith("pcce/proof-context/v0.1/")

    assert applied.status == "succeeded"
    assert applied.payload["sealed"] is True
    assert applied.payload["published"] is True
    assert applied.payload["canonical_mutated"] is False
    assert applied.payload["canonical_head"] == head_before
    assert tuple(applied.payload["stages"]) == STAGES
    assert [item["stage"] for item in applied.payload["trace"]] == list(STAGES)
    assert applied.identities.patch_id
    assert applied.payload["seal_cid"]
    assert str(applied.payload["seal_cid"]).startswith("b")
    assert applied.payload["evidence_cid"].startswith("b")
    worktree = applied.payload["worktree"]
    assert worktree.get("disposable") is True
    assert worktree.get("canonical_mutated") is False
    assert worktree.get("worktree_id")
    assert worktree.get("target_ref") == "pcce-disposable"

    assert (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8") == before
    assert bundle.session.canonical_head == head_before
    worktree_path = worktree.get("worktree_path")
    if isinstance(worktree_path, str) and Path(worktree_path).exists():
        patched = Path(worktree_path) / "src" / "demo" / "__init__.py"
        assert patched.read_text(encoding="utf-8") == "VALUE = 2\n"

    assert verify.status == "succeeded"
    assert seal.status == "succeeded"
    assert report.status == "succeeded"
    assert resumed.status == "succeeded"
    assert resumed.payload["sealed"] is True
    assert engine.identities.patch_id == applied.identities.patch_id


def test_failure_modes_are_typed(tmp_path: Path) -> None:
    repo = create_ordinary_python_repository(tmp_path / "repo")

    simulated = open_runtime(
        repo,
        options=RuntimeOptions(
            kit_root=tmp_path / "kit-sim",
            worktree_parent=tmp_path / "wt-sim",
            fail_at="route",
            fail_provenance="simulated",
            fail_status="succeeded",
        ),
    )
    simulated_record = simulated.engine.run(EXTERNAL_PATCH)
    assert simulated_record.status == "simulated"
    assert simulated_record.payload.get("published") is not True
    assert simulated_record.status in ("simulated", "rejected")

    stale = open_runtime(
        repo,
        options=RuntimeOptions(
            kit_root=tmp_path / "kit-stale",
            worktree_parent=tmp_path / "wt-stale",
            fail_at="scan-semantic",
            fail_status="stale",
            fail_error="stale_root",
        ),
    )
    stale_record = stale.engine.run(EXTERNAL_PATCH)
    assert stale_record.status == "stale"
    assert stale_record.payload.get("published") is not True

    unavailable = open_runtime(
        repo,
        options=RuntimeOptions(
            kit_root=tmp_path / "kit-unavail",
            worktree_parent=tmp_path / "wt-unavail",
            fail_at="assurance",
            fail_status="unavailable",
            fail_error="unavailable_capability",
        ),
    )
    unavailable_record = unavailable.engine.run(EXTERNAL_PATCH)
    assert unavailable_record.status == "unavailable"
    assert unavailable_record.payload.get("published") is not True

    partial = open_runtime(
        repo,
        options=RuntimeOptions(
            kit_root=tmp_path / "kit-partial",
            worktree_parent=tmp_path / "wt-partial",
            fail_at="incremental-verify",
            fail_status="verification_failed",
            fail_error="verification_failed",
            fail_discard=True,
        ),
    )
    partial_record = partial.engine.run(EXTERNAL_PATCH)
    assert partial_record.status == "partial_effect"
    assert partial_record.payload.get("published") is not True

    bypass = open_runtime(
        repo,
        options=RuntimeOptions(
            kit_root=tmp_path / "kit-bypass",
            worktree_parent=tmp_path / "wt-bypass",
        ),
    )
    bypass_record = bypass.engine.run({**EXTERNAL_PATCH, "skip_stages": ["assurance"]})
    assert bypass_record.status in {"rejected", "invalid"}
    assert bypass_record.payload.get("published") is not True

    mutated = open_runtime(
        repo,
        options=RuntimeOptions(
            kit_root=tmp_path / "kit-mut",
            worktree_parent=tmp_path / "wt-mut",
            mark_canonical_mutated=True,
        ),
    )
    mutated_record = mutated.engine.run(EXTERNAL_PATCH)
    assert mutated_record.status in {"rejected", "invalid"}
    assert mutated_record.payload.get("published") is not True
    assert (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    missing = tmp_path / "missing"
    with pytest.raises(ProofContextError) as missing_exc:
        open_runtime(missing)
    assert missing_exc.value.code in ERRORS
    assert missing_exc.value.code == "malformed"

    with pytest.raises(FacadeError) as mode_exc:
        ProofCarryingContextEngine.open(
            repo,
            ports=bypass.engine.ports,
            identities=bypass.engine.identities,
            mode="shadow",
        )
    assert mode_exc.value.reason == "unknown_field"


def test_stale_resume_and_unavailable_capability_are_typed(tmp_path: Path) -> None:
    from ipfs_accelerate_py.proof_context.recovery import mint_idempotency_key

    repo, bundle = _runtime(tmp_path)
    attempt = bundle.session.attempt
    key = mint_idempotency_key(
        attempt_id=attempt.attempt_id,
        run_id=bundle.engine.identities.run_id,
        trace_id=bundle.engine.identities.trace_id,
        stage="scan-semantic",
        position="after",
        inbound_cid=None,
        generation=attempt.writer_generation,
    )
    bundle.session.store.put(
        {
            "attempt_id": attempt.attempt_id,
            "idempotency_key": key,
            "stage": "scan-semantic",
            "position": "after",
            "settled": False,
            "published": False,
        },
        writer_id=attempt.writer_id,
        generation=attempt.writer_generation,
        fence_token=attempt.fence_token,
    )
    bundle.session.store.invalidate_fence(attempt.attempt_id)
    resumed = bundle.engine.resume()
    assert resumed.status == "stale"
    assert resumed.payload.get("published") is not True

    cap = Capability(
        name="datasets",
        distribution="ipfs_datasets_py",
        available=False,
        reason="unavailable",
    )
    with pytest.raises(DependencyUnavailable):
        require_production_capability(cap)
    datasets = resolve_datasets()
    kit = resolve_kit()
    if not datasets.available:
        with pytest.raises((DependencyUnavailable, UnavailableCapabilityError)):
            open_runtime(
                repo,
                options=RuntimeOptions(
                    kit_root=tmp_path / "kit-req",
                    require_datasets=True,
                ),
            )
    if not kit.available:
        with pytest.raises((DependencyUnavailable, UnavailableCapabilityError)):
            open_runtime(
                repo,
                options=RuntimeOptions(
                    kit_root=tmp_path / "kit-req2",
                    require_kit=True,
                ),
            )


def test_undeclared_and_escape_paths_are_rejected(tmp_path: Path) -> None:
    repo, bundle = _runtime(tmp_path)
    escaped = bundle.engine.run(
        {
            "declared_files": ["../secret.py"],
            "files": {"../secret.py": "x = 1\n"},
            "adapter_id": "external-patch",
            "approver_id": "coordinator",
        }
    )
    assert escaped.status in {"rejected", "invalid"}
    undeclared = bundle.engine.run(
        {
            "declared_files": ["src/demo/__init__.py"],
            "files": {
                "src/demo/__init__.py": "VALUE = 2\n",
                "src/demo/secret.py": "hidden = 1\n",
            },
            "adapter_id": "external-patch",
            "approver_id": "coordinator",
        }
    )
    assert undeclared.status in {"rejected", "invalid"}
    assert (repo / "src" / "demo" / "__init__.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_runtime_descriptor_is_stable() -> None:
    descriptor = dict(runtime_descriptor())
    assert descriptor == dict(RUNTIME_DESCRIPTOR)
    assert descriptor["schema"] == RUNTIME_SCHEMA
    assert descriptor["cid"] == RUNTIME_CID
    assert descriptor["cid"].startswith("b")
    assert descriptor["interface"] == "ProofCarryingContextEngine@0.1"
    assert descriptor["operations"] == list(OPERATIONS)
    assert descriptor["instance_operations"] == list(INSTANCE_OPERATIONS)
    assert descriptor["stages"] == list(STAGES)
    assert descriptor["sibling_layout_required"] is False
    assert descriptor["provider_bound"] is False
    assert descriptor["datasets_port"] == DATASETS_PORT
    assert descriptor["kit_port"] == KIT_PORT
    assert descriptor["lifecycle_cid"] == LIFECYCLE_CID
    assert descriptor["policy_cid"] == POLICY_CID
    assert descriptor["result_state_cid"] == RESULT_STATE_CID
    assert descriptor["recovery_cid"] == RECOVERY_CID
    snapshot = public_signature_snapshot()
    encoded = {
        name: {
            "parameters": list(spec["parameters"]),
            "keyword_only": list(spec["keyword_only"]),
            "return": spec["return"],
        }
        for name, spec in snapshot.items()
    }
    assert descriptor["public_signature_snapshot"] == encoded
    assert RUNTIME_API_PATH.is_file()
    on_disk = json.loads(RUNTIME_API_PATH.read_text(encoding="utf-8"))
    for key, value in on_disk.items():
        assert descriptor[key] == value
    assert descriptor["lifecycle_cid_binding"] == on_disk["lifecycle_cid_binding"]
    assert descriptor["runtime_cid_binding"] == on_disk["runtime_cid_binding"]
    digest = hashlib.sha256(
        json.dumps(on_disk, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert len(digest) == 64


def test_lazy_public_names_cover_composed_surfaces() -> None:
    port = importlib.import_module("ipfs_accelerate_py.proof_context")
    for name in (
        "ProofCarryingContextEngine",
        "PatchLifecycle",
        "RecoveryCoordinator",
        "admit_mode",
        "ResultRecord",
        "ProofContextError",
        "resolve_datasets",
        "resolve_kit",
        "open_engine",
        "runtime_descriptor",
    ):
        assert name in port.__all__
        assert getattr(port, name) is not None
