"""Tests for lazy session-scoped default identity services (PTR-134).

Acceptance covered:
- read/write/readwrite modes yield admitted forest + locator + static components
  for a direct node without conftest service attributes or a test registry
- expensive stable inputs are built once per session
- dirty overlays and source changes invalidate identities
- explicit test injections override defaults
- off mode imports no optional provider
- unavailable / incomplete / exceptional components return non-reusable
  rather than aborting pytest
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
from ipfs_accelerate_py.testing.proof_reuse.default_identity_services import (
    ANALYSIS_AST_INDEX_PROVIDER_INTERFACE,
    DEFAULT_IDENTITY_SERVICE_FACTORY_INTERFACE,
    DEFAULT_ITEM_STATIC_IDENTITY_INTERFACE,
    PROOF_REUSE_SESSION_IDENTITY_INTERFACE,
    AnalysisASTIndexProvider,
    DefaultIdentityReason,
    DefaultIdentityServiceFactory,
    ProofReuseSessionIdentity,
    build_default_identity_services,
)
from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
    CurrentInputCompleteness,
    CurrentItemComponentInputs,
    ItemIdentityAssemblyReason,
    ItemIdentityAssemblyServices,
    assemble_and_attach_item_identity,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[2]


class _Item:
    """Minimal collected-node stand-in (no registry, no conftest attributes)."""

    def __init__(
        self,
        path: Path,
        *,
        fixture_names: tuple[str, ...] = (),
        parameterized: bool = False,
        parameter: Any = None,
    ) -> None:
        self.path = path
        self.nodeid = f"{path.name}::test_direct"
        self.name = "test_direct"
        self.originalname = "test_direct"
        self.fixturenames = fixture_names
        self.cls = None
        self.session = None
        self._markers: list[Any] = []
        if parameterized:
            self.nodeid += "[case]"
            self.callspec = SimpleNamespace(id="case", params={"value": parameter})

    def iter_markers(self, name: str | None = None):
        if name is None:
            return iter(self._markers)
        return (marker for marker in self._markers if marker.name == name)

    def get_closest_marker(self, name: str):
        return next(
            (marker for marker in reversed(self._markers) if marker.name == name),
            None,
        )


def _git(root: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "proof-reuse@example.invalid")
    _git(root, "config", "user.name", "Proof Reuse Test")
    test_path = root / "test_direct.py"
    test_path.write_text(
        "def test_direct():\n    value = 1\n    assert value == 1\n",
        encoding="utf-8",
    )
    _git(root, "add", "test_direct.py")
    _git(root, "commit", "-qm", "fixture")
    return root, test_path


def _factory(
    root: Path,
    *,
    mode: str | ProofReuseMode = "read",
    **overrides: Any,
) -> DefaultIdentityServiceFactory:
    return DefaultIdentityServiceFactory(
        mode=mode,
        root_path=root,
        sole_write_alias="repo",
        **overrides,
    )


@pytest.mark.parametrize("mode", ("read", "write", "readwrite"))
def test_enabled_modes_obtain_forest_locator_and_static_components(
    tmp_path: Path, mode: str
) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    factory = _factory(root, mode=mode)

    result = factory.obtain_static_identity(item)

    assert result.reason is DefaultIdentityReason.ADMITTED
    assert result.reusable is True
    assert result.action == "RUN"
    assert result.authorizes_skip is False
    assert result.interface == DEFAULT_ITEM_STATIC_IDENTITY_INTERFACE
    assert result.forest is not None
    assert result.forest_id
    assert result.forest_id == result.forest.forest_id
    assert result.forest.descriptors
    assert result.locator_artifact is not None
    assert result.locator_artifact.reusable is True
    assert result.locator_artifact.locator is not None
    assert result.components is not None
    assert result.components.reusable is True
    assert result.static_trace is not None
    assert result.component_inputs is not None
    # No conftest service attributes or registry required on the item.
    assert not hasattr(item, "_ipfs_proof_reuse_identity_services")
    assert not hasattr(item, "_proof_reuse_registry")


def test_session_memoizes_expensive_stable_inputs(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    factory = _factory(root, mode="read")
    session = factory.session_identity()
    item_a = _Item(path)
    item_b = _Item(path)

    first = factory.obtain_static_identity(item_a)
    second = factory.obtain_static_identity(item_b)

    assert first.reusable and second.reusable
    assert first.forest_id == second.forest_id
    assert session.forest_build_count == 1
    assert session.ast_index_build_count == 1
    assert session.policy_build_count == 1
    # Dependency inventory is built at most once per kind in the session.
    assert session.dependency_build_count <= 2


def test_dirty_overlay_invalidates_forest_identity(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    factory = _factory(root, mode="read")
    session = factory.session_identity()
    item = _Item(path)

    before = factory.obtain_static_identity(item)
    assert before.reusable
    before_forest_id = before.forest_id
    assert session.forest_build_count == 1

    dirty_file = root / "dirty_overlay.txt"
    dirty_file.write_text("dirty\n", encoding="utf-8")

    after = factory.obtain_static_identity(_Item(path))
    assert after.reusable
    assert after.forest_id != before_forest_id
    assert session.forest_build_count >= 2


def test_source_change_invalidates_ast_index_and_static_identity(
    tmp_path: Path,
) -> None:
    root, path = _repository(tmp_path)
    factory = _factory(root, mode="read")
    session = factory.session_identity()

    before = factory.obtain_static_identity(_Item(path))
    assert before.reusable
    before_trace = before.static_trace.trace_cid
    assert session.ast_index_build_count == 1

    path.write_text(
        "def test_direct():\n    value = 2\n    assert value == 2\n",
        encoding="utf-8",
    )

    after = factory.obtain_static_identity(_Item(path))
    assert after.reusable
    assert after.static_trace.trace_cid != before_trace
    assert session.ast_index_build_count >= 2


def test_explicit_injections_override_defaults(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    calls: list[str] = []

    def forest_override(received: Any) -> Any:
        calls.append("forest")
        # Delegate to a clean session forest so admission still succeeds.
        return ProofReuseSessionIdentity(
            mode=ProofReuseMode.READ,
            root_path=root,
            sole_write_alias="repo",
        ).forest(seed_path=path)

    def index_override(received: Any, descriptor: Any) -> Any:
        calls.append("index")
        session = ProofReuseSessionIdentity(
            mode=ProofReuseMode.READ,
            root_path=root,
            sole_write_alias="repo",
        )
        return AnalysisASTIndexProvider(session).provide(descriptor)

    factory = _factory(
        root,
        mode="read",
        repository_forest_provider=forest_override,
        analysis_index_provider=index_override,
    )
    result = factory.obtain_static_identity(item)

    assert result.reusable
    assert "forest" in calls
    assert "index" in calls

    # build_default_identity_services also honours explicit overrides.
    services = build_default_identity_services(
        mode="write",
        root_path=root,
        repository_forest_provider=forest_override,
    )
    assert isinstance(services, ItemIdentityAssemblyServices)
    assert services.repository_forest_provider is not None
    services.repository_forest_provider(item)
    assert calls.count("forest") >= 2


def test_off_mode_imports_no_optional_provider(tmp_path: Path) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import builtins
original = builtins.__import__
BLOCKED = (
    "multiformats",
    "ipfs_datasets_py",
    "jsonschema",
)
def guarded(name, *args, **kwargs):
    root = name.split(".", 1)[0]
    if name in BLOCKED or root in BLOCKED:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}", name=name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
""".lstrip(),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(ACCELERATE_ROOT), environment.get("PYTHONPATH", ""))
    )
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = "off"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from ipfs_accelerate_py.testing.proof_reuse."
                "default_identity_services import ("
                "    build_default_identity_services,"
                "    DefaultIdentityServiceFactory,"
                "); "
                "services = build_default_identity_services(mode='off'); "
                "assert services.repository_forest_provider is None; "
                "assert services.analysis_index_provider is None; "
                "factory = DefaultIdentityServiceFactory(mode='off'); "
                "result = factory.obtain_static_identity(object()); "
                "assert result.reusable is False; "
                "blocked = ("
                "    'multiformats', 'ipfs_datasets_py', 'jsonschema'"
                "); "
                "assert not any("
                "    n == b or n.startswith(b + '.') "
                "    for n in sys.modules for b in blocked"
                ")"
            ),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unavailable_and_exceptional_components_fail_open(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)

    def boom(_item: Any) -> Any:
        raise RuntimeError("forest exploded")

    factory = _factory(root, mode="read", repository_forest_provider=boom)
    result = factory.obtain_static_identity(item)

    assert result.reusable is False
    assert result.action == "RUN"
    assert result.authorizes_skip is False
    assert result.reason is DefaultIdentityReason.REPOSITORY_FOREST_UNAVAILABLE

    # Empty services from off-mode assembly fail open through the assembler.
    services = build_default_identity_services(mode="off")
    assembly = assemble_and_attach_item_identity(item, services)
    assert assembly.action == "RUN"
    assert assembly.authorizes_skip is False
    assert assembly.reason is ItemIdentityAssemblyReason.PROVIDER_UNAVAILABLE


def test_services_bundle_wires_session_providers(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    factory = _factory(root, mode="readwrite")
    services = factory.build_services()

    assert isinstance(services, ItemIdentityAssemblyServices)
    assert services.repository_forest_provider is not None
    assert services.analysis_index_provider is not None
    assert services.component_inputs_provider is not None
    assert services.policy_inputs_provider is not None
    assert services.identity_compiler is not None

    forest = services.repository_forest_provider(item)
    assert forest.forest_id
    descriptor = forest.descriptors[0]
    index = services.analysis_index_provider(item, descriptor)
    assert index is not None

    # Second call reuses the same session AST index.
    assert factory.session_identity().ast_index_build_count == 1
    services.analysis_index_provider(item, descriptor)
    assert factory.session_identity().ast_index_build_count == 1

    # Default runtime evidence is intentionally absent so full assembly fails
    # open to RUN without aborting collection.
    assembly = assemble_and_attach_item_identity(_Item(path), services)
    assert assembly.action == "RUN"
    assert assembly.reason in {
        ItemIdentityAssemblyReason.RUNTIME_EVIDENCE_UNAVAILABLE,
        ItemIdentityAssemblyReason.PROVIDER_UNAVAILABLE,
        ItemIdentityAssemblyReason.COMPONENTS_NON_REUSABLE,
        ItemIdentityAssemblyReason.ADMITTED_FOR_LOOKUP,
    }


def test_analysis_ast_index_provider_interface_and_memoization(
    tmp_path: Path,
) -> None:
    root, path = _repository(tmp_path)
    session = ProofReuseSessionIdentity(
        mode=ProofReuseMode.READ,
        root_path=root,
        sole_write_alias="repo",
    )
    provider = AnalysisASTIndexProvider(session)
    assert provider.interface == ANALYSIS_AST_INDEX_PROVIDER_INTERFACE
    forest = session.forest(seed_path=path)
    descriptor = forest.descriptors[0]

    first = provider.provide(descriptor)
    second = provider(None, descriptor)
    assert first is second
    assert provider.build_count == 1


def test_factory_interfaces_and_off_mode_static_identity(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    factory = _factory(root, mode="off")
    assert factory.interface == DEFAULT_IDENTITY_SERVICE_FACTORY_INTERFACE
    assert factory.enabled is False
    assert factory.session_identity().interface == PROOF_REUSE_SESSION_IDENTITY_INTERFACE

    result = factory.obtain_static_identity(_Item(path))
    assert result.reason is DefaultIdentityReason.MODE_OFF
    assert result.reusable is False
    assert result.to_dict()["action"] == "RUN"


def test_component_inventory_matches_item_fixtures(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path, fixture_names=("tmp_path",))
    factory = _factory(root, mode="read")
    result = factory.obtain_static_identity(item)

    # Fixtures without controlled value adapters become non-reusable components,
    # which must surface as non-reusable rather than aborting.
    assert result.component_inputs is not None
    names = tuple(
        sorted(str(record.get("name") or "") for record in result.component_inputs.fixtures)
    )
    assert names == ("tmp_path",)
    assert result.forest is not None
    assert result.action == "RUN"
    if result.reason is DefaultIdentityReason.COMPONENTS_NON_REUSABLE:
        assert result.components is not None
        assert result.components.reusable is False
    else:
        # If fixture collection somehow admits, still require forest + locator.
        assert result.reusable is True


def test_cold_module_import_is_inert(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ACCELERATE_ROOT), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "before = set(sys.modules); "
                "import ipfs_accelerate_py.testing.proof_reuse."
                "default_identity_services as mod; "
                "after = set(sys.modules) - before; "
                "assert mod.DEFAULT_IDENTITY_SERVICE_FACTORY_INTERFACE; "
                "forbidden = ("
                "    'multiformats',"
                "    'ipfs_datasets_py',"
                "    'jsonschema',"
                "    'ipfs_accelerate_py.agent_supervisor.repository_forest',"
                "    'ipfs_accelerate_py.agent_supervisor.analysis."
                "analysis_ast_index',"
                "); "
                "assert not any(name in after for name in forbidden), after"
            ),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_build_default_identity_services_idempotent_per_factory(
    tmp_path: Path,
) -> None:
    root, _path = _repository(tmp_path)
    factory = _factory(root, mode="read")
    first = factory.build_services()
    second = factory.build_services()
    assert first is second

    services = build_default_identity_services(mode="read", root_path=root)
    assert isinstance(services, ItemIdentityAssemblyServices)
    assert services.repository_forest_provider is not None
