"""Locator-first collection seed regressions (PTR-143).

Acceptance covered:
- read/write/readwrite: direct collected node receives a canonical collection
  seed and stable locator before any runtime trace exists
- parameterized nodes bind the exact canonical parameter-value CID
- collection performs no fixture or test call and attaches no final execution key
- explicit injected identity remains an override
- incomplete or exceptional static facts attach no lookup authority
- off mode retains cold import behaviour
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.test_identity_components import (
    TestIdentityComponents,
)
from ipfs_accelerate_py.testing.proof_reuse.collection_seed import (
    ITEM_COLLECTION_SEED_ATTRIBUTE,
    LOCATOR_FIRST_ASSEMBLER_INTERFACE,
    PROOF_REUSE_COLLECTION_SEED_INTERFACE,
    CollectionSeedReason,
    LocatorFirstItemIdentityAssembler,
    ProofReuseCollectionSeed,
    assemble_and_attach_collection_seed,
    build_collection_seed_from_static_identity,
)
from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
from ipfs_accelerate_py.testing.proof_reuse.default_identity_services import (
    DefaultIdentityReason,
    DefaultIdentityServiceFactory,
)
from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
    ITEM_EXECUTION_KEY_ATTRIBUTE,
    ITEM_LOCATOR_ATTRIBUTE,
    assemble_and_attach_item_identity,
    ItemIdentityAssemblyServices,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    CONFIG_ATTRIBUTE,
    IDENTITY_FACTORY_ATTRIBUTE,
    ProofReuseRuntimeComposition,
    get_proof_reuse_config,
    pytest_collection_modifyitems,
)
from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseConfig


ACCELERATE_ROOT = Path(__file__).resolve().parents[2]


class _Item:
    """Minimal collected-node stand-in (no registry, no fixture invocation)."""

    def __init__(
        self,
        path: Path,
        *,
        fixture_names: tuple[str, ...] = (),
        parameterized: bool = False,
        parameter: Any = None,
        parameter_id: str = "case",
    ) -> None:
        self.path = path
        self.nodeid = f"{path.name}::test_direct"
        self.name = "test_direct"
        self.originalname = "test_direct"
        self.fixturenames = fixture_names
        self.cls = None
        self.session = None
        self._markers: list[Any] = []
        self.fixture_call_count = 0
        self.test_call_count = 0
        if parameterized:
            self.nodeid += f"[{parameter_id}]"
            self.callspec = SimpleNamespace(
                id=parameter_id,
                params={"value": parameter},
            )

    def iter_markers(self, name: str | None = None):
        if name is None:
            return iter(self._markers)
        return (marker for marker in self._markers if marker.name == name)

    def get_closest_marker(self, name: str):
        return next(
            (marker for marker in reversed(self._markers) if marker.name == name),
            None,
        )

    def runtest(self) -> None:
        self.test_call_count += 1
        raise AssertionError("collection must not call the test body")

    def _request_fixture(self, name: str) -> Any:
        self.fixture_call_count += 1
        raise AssertionError(f"collection must not call fixture {name!r}")


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
def test_enabled_modes_attach_canonical_seed_and_stable_locator(
    tmp_path: Path, mode: str
) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    factory = _factory(root, mode=mode)

    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode=mode
    )

    assert seed.reason is CollectionSeedReason.ADMITTED
    assert seed.admitted is True
    assert seed.interface == PROOF_REUSE_COLLECTION_SEED_INTERFACE
    assert seed.action == "RUN"
    assert seed.authorizes_skip is False
    assert seed.authorizes_lookup is False
    assert seed.has_execution_key is False
    assert seed.seed_cid.startswith("b")
    assert seed.locator_cid.startswith("b")
    assert seed.forest_id
    assert seed.static_trace_root_cid
    assert seed.component_root_cid
    assert seed.parameterized is False
    assert seed.parameter_values_cid == ""

    attached = getattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE)
    assert isinstance(attached, ProofReuseCollectionSeed)
    assert attached.seed_cid == seed.seed_cid
    locator = getattr(item, ITEM_LOCATOR_ATTRIBUTE)
    assert locator is seed.locator
    assert locator.node_id.endswith("test_direct.py::test_direct")
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)
    assert not hasattr(item, "_ipfs_proof_reuse_lookup_request")
    assert item.fixture_call_count == 0
    assert item.test_call_count == 0


def test_parameterized_node_binds_exact_parameter_value_cid(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    parameter = {"model": "tiny", "shards": (1, 2)}
    item = _Item(path, parameterized=True, parameter=parameter, parameter_id="p0")
    factory = _factory(root, mode="read")

    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode="read"
    )

    assert seed.admitted is True
    assert seed.parameterized is True
    assert seed.parameter_id == "p0"
    assert seed.parameter_values_cid.startswith("b")

    # Locator binds the components parameter CID for the exact callspec map.
    static = factory.obtain_static_identity(item)
    assert static.reusable is True
    assert static.components is not None
    assert seed.parameter_values_cid == static.components.parameter_cid
    assert seed.locator.parameter_values_cid == static.components.parameter_cid
    # Sanity: independent compile of the same callspec map agrees.
    direct = TestIdentityComponents.compile(
        parameter_id="p0",
        parameter_value={"value": parameter},
    )
    assert seed.parameter_values_cid == direct.parameter_cid


def test_collection_never_invokes_fixtures_or_test_body(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path, fixture_names=("tmp_path",))
    factory = _factory(root, mode="readwrite")

    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode="readwrite"
    )

    # Incomplete fixtures may make the seed non-admitted, but must not call.
    assert item.fixture_call_count == 0
    assert item.test_call_count == 0
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)
    assert seed.authorizes_lookup is False
    assert seed.action == "RUN"


def test_explicit_injected_identity_remains_override(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    injected_locator = object()
    injected_key = object()
    item._ipfs_proof_reuse_locator = injected_locator
    item._ipfs_proof_reuse_execution_key = injected_key
    factory = _factory(root, mode="read")

    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode="read"
    )

    assert seed.reason is CollectionSeedReason.EXISTING_IDENTITY_OVERRIDE
    assert seed.admitted is False
    assert item._ipfs_proof_reuse_locator is injected_locator
    assert item._ipfs_proof_reuse_execution_key is injected_key
    assert not hasattr(item, "_ipfs_proof_reuse_lookup_request")


def test_manual_locator_without_seed_is_override(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    injected_locator = object()
    item._ipfs_proof_reuse_locator = injected_locator
    factory = _factory(root, mode="read")

    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode="read"
    )

    assert seed.reason is CollectionSeedReason.EXISTING_IDENTITY_OVERRIDE
    assert item._ipfs_proof_reuse_locator is injected_locator
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)


def test_incomplete_static_facts_attach_no_lookup_authority(
    tmp_path: Path,
) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)

    def boom(_item: Any) -> Any:
        raise RuntimeError("forest exploded")

    factory = _factory(root, mode="read", repository_forest_provider=boom)
    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode="read"
    )

    assert seed.admitted is False
    assert seed.authorizes_lookup is False
    assert seed.authorizes_skip is False
    assert seed.action == "RUN"
    assert seed.reason is CollectionSeedReason.STATIC_IDENTITY_INCOMPLETE
    assert not hasattr(item, ITEM_LOCATOR_ATTRIBUTE)
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)
    assert not hasattr(item, "_ipfs_proof_reuse_lookup_request")
    # Diagnostic seed may still be recorded.
    recorded = getattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, None)
    assert recorded is not None
    assert recorded.admitted is False


def test_exceptional_factory_fails_open(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)

    class _BrokenFactory:
        mode = ProofReuseMode.READ

        def obtain_static_identity(self, _item: Any) -> Any:
            raise RuntimeError("unexpected")

    seed = assemble_and_attach_collection_seed(
        item, factory=_BrokenFactory(), mode="read"
    )
    assert seed.reason is CollectionSeedReason.INTERNAL_ERROR_FAIL_OPEN
    assert seed.action == "RUN"
    assert seed.authorizes_lookup is False
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)


def test_off_mode_retains_cold_import_and_no_seed(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    factory = _factory(root, mode="off")

    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode="off"
    )
    assert seed.reason is CollectionSeedReason.MODE_OFF
    assert seed.admitted is False
    assert not hasattr(item, ITEM_LOCATOR_ATTRIBUTE)
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)

    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import builtins
original = builtins.__import__
BLOCKED = ("multiformats", "ipfs_datasets_py", "jsonschema")
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
                "from ipfs_accelerate_py.testing.proof_reuse import collection_seed; "
                "from ipfs_accelerate_py.testing.proof_reuse.default_identity_services "
                "import DefaultIdentityServiceFactory; "
                "factory = DefaultIdentityServiceFactory(mode='off'); "
                "seed = collection_seed.assemble_and_attach_collection_seed("
                "    object(), factory=factory, mode='off'"
                "); "
                "assert seed.admitted is False; "
                "assert seed.reason.value == 'mode_off'; "
                "blocked = ('multiformats', 'ipfs_datasets_py', 'jsonschema'); "
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


def test_seed_is_stable_for_identical_static_inputs(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    factory = _factory(root, mode="read")
    first = assemble_and_attach_collection_seed(
        _Item(path), factory=factory, mode="read"
    )
    second = assemble_and_attach_collection_seed(
        _Item(path), factory=factory, mode="read"
    )
    assert first.admitted and second.admitted
    assert first.seed_cid == second.seed_cid
    assert first.locator_cid == second.locator_cid
    assert first.parameter_values_cid == second.parameter_values_cid


def test_build_seed_from_static_identity_projection(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    factory = _factory(root, mode="read")
    static = factory.obtain_static_identity(_Item(path))
    assert static.reason is DefaultIdentityReason.ADMITTED

    seed = build_collection_seed_from_static_identity(static)
    assert seed.admitted is True
    assert seed.locator is static.locator_artifact.locator
    assert seed.static_identity is static
    assert seed.to_dict()["has_execution_key"] is False


def test_locator_first_assembler_interface(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    factory = _factory(root, mode="read")
    assembler = LocatorFirstItemIdentityAssembler(
        factory=factory, mode="read"
    )
    assert assembler.interface == LOCATOR_FIRST_ASSEMBLER_INTERFACE
    seed = assembler.assemble_and_attach(_Item(path))
    assert seed.admitted is True


def test_full_assembly_without_runtime_does_not_attach_execution_key(
    tmp_path: Path,
) -> None:
    """Collection seed + default services: locator present, no final key."""

    root, path = _repository(tmp_path)
    item = _Item(path)
    factory = _factory(root, mode="read")
    services = factory.build_services()

    seed = assemble_and_attach_collection_seed(
        item, factory=factory, mode="read"
    )
    assert seed.admitted is True
    assert getattr(item, ITEM_LOCATOR_ATTRIBUTE) is seed.locator

    assembly = assemble_and_attach_item_identity(item, services)
    # Default runtime evidence is absent → no lookup upgrade.
    assert assembly.action == "RUN"
    assert assembly.authorizes_skip is False
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)
    assert not hasattr(item, "_ipfs_proof_reuse_lookup_request")
    # Intermediate locator from the collection seed remains.
    assert getattr(item, ITEM_LOCATOR_ATTRIBUTE) is seed.locator
    assert getattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE).seed_cid == seed.seed_cid


def test_plugin_collection_attaches_seed_in_read_mode(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)

    class _Config:
        def __init__(self) -> None:
            self.rootpath = root

        def getoption(self, name: str, default: Any = None) -> Any:
            return default

        def getini(self, name: str) -> Any:
            return ""

    config = _Config()
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.READ),
    )
    item = _Item(path)
    item.nodeid = f"{path.relative_to(root).as_posix()}::test_direct"

    pytest_collection_modifyitems(config, [item])

    seed = getattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, None)
    assert isinstance(seed, ProofReuseCollectionSeed)
    # Real repo walk may admit or fail open; never attaches execution key.
    assert seed.authorizes_lookup is False
    assert seed.authorizes_skip is False
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)
    assert not hasattr(item, "_ipfs_proof_reuse_lookup_request")
    if seed.admitted:
        assert getattr(item, ITEM_LOCATOR_ATTRIBUTE) is seed.locator
        assert seed.locator_cid
        assert seed.seed_cid
    assert getattr(config, IDENTITY_FACTORY_ATTRIBUTE, None) is not None
    composition = getattr(config, "_ipfs_proof_reuse_runtime_composition", None)
    assert isinstance(composition, ProofReuseRuntimeComposition)
    # Off-path config helper remains readable.
    assert get_proof_reuse_config(config).mode is ProofReuseMode.READ


def test_plugin_off_mode_is_inert(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)

    class _Config:
        rootpath = root

        def getoption(self, name: str, default: Any = None) -> Any:
            return default

        def getini(self, name: str) -> Any:
            return ""

    config = _Config()
    setattr(config, CONFIG_ATTRIBUTE, ProofReuseConfig(mode=ProofReuseMode.OFF))
    item = _Item(path)

    pytest_collection_modifyitems(config, [item])

    assert not hasattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE)
    assert not hasattr(item, ITEM_LOCATOR_ATTRIBUTE)
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)


def test_missing_factory_fails_open_without_lookup(tmp_path: Path) -> None:
    root, path = _repository(tmp_path)
    item = _Item(path)
    seed = assemble_and_attach_collection_seed(
        item, factory=None, services=ItemIdentityAssemblyServices(), mode="read"
    )
    assert seed.reason is CollectionSeedReason.FACTORY_UNAVAILABLE
    assert seed.admitted is False
    assert not hasattr(item, ITEM_LOCATOR_ATTRIBUTE)
    assert not hasattr(item, ITEM_EXECUTION_KEY_ATTRIBUTE)
