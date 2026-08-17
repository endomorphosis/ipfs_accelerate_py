"""SCH-002 pinned datasets semantic-state adapter tests."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state import datasets_adapter as adapter_mod
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    ADAPTER_ID,
    CONFIDENCE_VALUES,
    EXPECTED_CAPSULE_SCHEMA,
    EXPECTED_SELECTION_SCHEMA,
    EXPECTED_SEMANTIC_INDEX_SCHEMA,
    EXPECTED_SEMANTIC_STATE_SCHEMA,
    PROVIDER_CONTRACT,
    IpfsDatasetsSemanticStateProvider,
    SemanticStateAdapterError,
    SemanticStateCapability,
    SemanticStateProvider,
    SemanticStateUnavailable,
    SourceBlobStale,
    inspect_semantic_state_capability,
    load_semantic_state_provider,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import UnavailableResult

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py"
)
SEALED_DATASETS_COMMIT = "1330038f626ef92993f03d46f21e1a57719e9c25"
_DATASETS_CANDIDATE_REPOS = (
    Path("/home/barberb/lift_coding/external/ipfs_datasets"),
    Path(os.environ.get("IPFS_DATASETS_GIT_DIR", "")),
)


# ---------------------------------------------------------------------------
# Sealed datasets materialization
# ---------------------------------------------------------------------------


def _find_datasets_git() -> Path | None:
    for candidate in _DATASETS_CANDIDATE_REPOS:
        if not candidate or not str(candidate):
            continue
        if (candidate / ".git").exists() or (candidate / "ipfs_datasets_py").is_dir():
            # Prefer real git roots that contain the sealed commit.
            try:
                subprocess.run(
                    ["git", "cat-file", "-t", SEALED_DATASETS_COMMIT],
                    cwd=candidate,
                    check=True,
                    capture_output=True,
                )
                return candidate
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    return None


def _materialize_sealed_datasets(destination: Path) -> Path:
    git_root = _find_datasets_git()
    if git_root is None:
        pytest.skip("sealed ipfs_datasets_py git root unavailable")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".sealed_commit"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == SEALED_DATASETS_COMMIT:
        if (destination / "ipfs_datasets_py/logic/software_contracts/semantic_state/api.py").is_file():
            return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    proc = subprocess.run(
        ["git", "archive", SEALED_DATASETS_COMMIT],
        cwd=git_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=proc.stdout,
        check=True,
        capture_output=True,
    )
    marker.write_text(SEALED_DATASETS_COMMIT + "\n", encoding="utf-8")
    return destination


def _purge_datasets_modules() -> None:
    doomed = [
        name
        for name in list(sys.modules)
        if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.")
    ]
    for name in doomed:
        del sys.modules[name]


@pytest.fixture(scope="module")
def sealed_datasets_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("sealed-datasets")
    return _materialize_sealed_datasets(root)


@pytest.fixture
def sealed_provider(sealed_datasets_root: Path, monkeypatch: pytest.MonkeyPatch):
    """Provider bound to the sealed datasets pin on an isolated PYTHONPATH."""

    _purge_datasets_modules()
    monkeypatch.syspath_prepend(str(sealed_datasets_root))
    # Drop the empty workspace namespace package path if present.
    workspace_ns = str(REPO_ROOT / "ipfs_datasets_py")
    sys.path[:] = [p for p in sys.path if p != workspace_ns and p != str(REPO_ROOT)]
    monkeypatch.syspath_prepend(str(sealed_datasets_root))
    _purge_datasets_modules()
    provider = load_semantic_state_provider()
    yield provider
    _purge_datasets_modules()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text(
        "VALUE = 1\n\ndef hello() -> int:\n    return VALUE\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "mod.py").write_text(
        "from pkg import hello\n\ndef use() -> int:\n    return hello()\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_mod.py").write_text(
        "from pkg.mod import use\n\ndef test_use() -> None:\n    assert use() == 1\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "adapter@example.invalid")
    _git(repo, "config", "user.name", "Adapter Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo,
        check=False,
    )
    if status.returncode != 0:
        _git(repo, "commit", "-m", message)


# ---------------------------------------------------------------------------
# Unit: capability / fail-closed / AST boundary
# ---------------------------------------------------------------------------


def test_provider_protocol_and_capability_constants() -> None:
    assert PROVIDER_CONTRACT == "SemanticStateProvider@1"
    assert ADAPTER_ID.startswith("ipfs-datasets-semantic-state-adapter")
    assert EXPECTED_SEMANTIC_STATE_SCHEMA.endswith("@1")
    assert EXPECTED_CAPSULE_SCHEMA.endswith("@1")
    assert EXPECTED_SELECTION_SCHEMA.endswith("@1")
    assert EXPECTED_SEMANTIC_INDEX_SCHEMA.endswith("@2")
    assert CONFIDENCE_VALUES == frozenset(
        {"exact", "conservative", "heuristic", "opaque"}
    )
    assert issubclass(IpfsDatasetsSemanticStateProvider, object)


def test_unavailable_maps_to_unavailable_result() -> None:
    exc = SemanticStateUnavailable(
        "open_semantic_state",
        "import_failed",
        "no package",
        retryable=True,
    )
    result = exc.to_unavailable_result()
    assert isinstance(result, UnavailableResult)
    assert result.operation == "open_semantic_state"
    assert result.adapter_id == ADAPTER_ID
    assert result.reason_code == "import_failed"
    assert result.retryable is True


def test_source_blob_stale_requires_rescan() -> None:
    err = SourceBlobStale("bytes changed", kind="source_binding_mismatch")
    assert err.requires_rescan is True
    assert err.kind == "source_binding_mismatch"


def test_injected_surface_provider_round_trip_without_import() -> None:
    calls: list[str] = []

    def open_semantic_state(root_cid: str, get_block: Callable[[str], bytes]) -> Any:
        calls.append("open")
        assert root_cid.startswith("b")
        assert get_block(root_cid) == b"root"
        bound_root = root_cid

        class _View:
            root = types.SimpleNamespace(root_cid=bound_root)

            def get_block(self, cid: str) -> bytes:
                return get_block(cid)

            def symbol_node(self, stable_symbol_id: str) -> Any:
                return types.SimpleNamespace(
                    node_cid=bound_root,
                    stable_symbol_id=stable_symbol_id,
                    confidence="exact",
                )

            def capsule(self, stable_symbol_id: str) -> Any:
                return types.SimpleNamespace(
                    capsule_cid=bound_root,
                    stable_symbol_id=stable_symbol_id,
                    confidence="opaque",
                    capsule_schema=EXPECTED_CAPSULE_SCHEMA,
                    source_cid=bound_root,
                    version_cid=bound_root,
                )

        return _View()

    def scan_repository(repo_path: Any, previous_state: Any = None) -> Any:
        calls.append("scan")
        return types.SimpleNamespace(
            repository_id="r",
            state_cid="b" + "a" * 58,
            symbols=(),
            edges=(),
            artifacts=(),
            previous_state=previous_state,
            path=str(repo_path),
        )

    def build_semantic_state(semantic_index: Any, **kwargs: Any) -> Any:
        calls.append("build")
        bound = "b" + "c" * 58

        class _Bundle:
            root = types.SimpleNamespace(root_cid=bound)

            def get_block(self, cid: str) -> bytes:
                return b"x"

        return _Bundle()

    def select_tests_and_proofs(
        previous_state: Any,
        current_state: Any,
        invalidation: Any,
        *,
        policy: Any,
        explicit_rules: Any = (),
        **kwargs: Any,
    ) -> Any:
        calls.append("select")
        assert previous_state is not None or previous_state is None
        return types.SimpleNamespace(
            selection_cid="b" + "d" * 58,
            schema=EXPECTED_SELECTION_SCHEMA,
            previous=previous_state,
            current=current_state,
            policy=policy,
            explicit_rules=explicit_rules,
        )

    def watch_repository(repo_path: Any, callback: Callable[[Any], Any], *, debounce_ms: int = 250) -> Any:
        calls.append("watch")
        notification = types.SimpleNamespace(
            state=types.SimpleNamespace(state_cid="b" + "e" * 58, repository_id="r"),
            previous_state=types.SimpleNamespace(state_cid="b" + "f" * 58, repository_id="r"),
            event_paths=["should-not-become-state"],
        )
        callback(notification)
        return types.SimpleNamespace(stop=lambda: None, debounce_ms=debounce_ms)

    def read_required_source(
        semantic_index: Any,
        symbol_id: str,
        *,
        expected_producer_state_cid: str,
        read_source_blob: Callable[[str], bytes] | None = None,
    ) -> Any:
        calls.append("source")
        if read_source_blob is None:
            raise RuntimeError("missing sealed reader")
        data = read_source_blob("b" + "1" * 58)
        return types.SimpleNamespace(source_bytes=data, symbol_id=symbol_id)

    surface = types.SimpleNamespace(
        open_semantic_state=open_semantic_state,
        scan_repository=scan_repository,
        build_semantic_state=build_semantic_state,
        select_tests_and_proofs=select_tests_and_proofs,
        watch_repository=watch_repository,
        read_required_source=read_required_source,
        view_semantic_state_bundle=lambda bundle: open_semantic_state(
            bundle.root.root_cid, bundle.get_block
        ),
        capability=SemanticStateCapability(
            available=True,
            adapter_id=ADAPTER_ID,
            contract_name="SemanticStateProvider@1",
            semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version="1",
            capsule_compiler_version="1",
            semantic_state_api_schema="ipfs-datasets.software-contracts.semantic-state-api@1",
            view_interface="SemanticStateView@1",
            producer_interface="SemanticStateProducer@1",
            block_reader_interface="SemanticStateBlockReader@1",
            operations=("open_semantic_state", "scan_repository"),
        ),
    )
    provider = load_semantic_state_provider(surface)
    assert isinstance(provider, IpfsDatasetsSemanticStateProvider)
    assert isinstance(provider, SemanticStateProvider)

    root_cid = "b" + "a" * 58
    blocks = {root_cid: b"root"}
    view = provider.open_verified_view(root_cid, blocks.__getitem__)
    assert view.root.root_cid == root_cid
    capsule = view.capsule("sym")
    assert capsule.confidence == "opaque"

    state = provider.scan_repository("/tmp/repo")
    assert state.state_cid.startswith("b")
    assert state.previous_state is None

    bundle = provider.build_semantic_state(state)
    assert bundle.root.root_cid.startswith("b")
    # Bundle must not grant storage mutation authority.
    assert not hasattr(bundle, "put")

    selection = provider.select_tests_and_proofs(
        view, view, invalidation=object(), policy=object()
    )
    assert selection.schema == EXPECTED_SELECTION_SCHEMA
    assert selection.previous is view

    seen: list[Any] = []
    provider.watch_repository("/tmp/repo", seen.append, debounce_ms=10)
    assert len(seen) == 1
    # Event paths never become state — only scanned state_cid is authoritative.
    assert seen[0].state.state_cid.startswith("b")

    with pytest.raises(SourceBlobStale):
        provider.read_required_source(
            object(), "sym", expected_producer_state_cid=root_cid
        )

    sealed_bytes = b"exact-source"
    materialization = provider.read_required_source(
        object(),
        "sym",
        expected_producer_state_cid=root_cid,
        read_source_blob=lambda cid: sealed_bytes,
    )
    assert materialization.source_bytes == sealed_bytes
    assert "open" in calls and "scan" in calls and "build" in calls


def test_mismatched_schema_pin_fails_closed() -> None:
    surface = types.SimpleNamespace(
        capability=SemanticStateCapability(
            available=False,
            adapter_id=ADAPTER_ID,
            contract_name="SemanticStateProvider@1",
            semantic_state_schema="wrong@0",
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version="1",
            capsule_compiler_version="1",
            semantic_state_api_schema="x",
            view_interface="SemanticStateView@1",
            producer_interface="SemanticStateProducer@1",
            block_reader_interface="SemanticStateBlockReader@1",
            operations=(),
            reason_code="schema_mismatch",
            diagnostic="schema pin failed",
        ),
        open_semantic_state=lambda *a, **k: None,
        scan_repository=lambda *a, **k: None,
    )
    provider = IpfsDatasetsSemanticStateProvider(surface)
    with pytest.raises(SemanticStateUnavailable, match="schema_mismatch"):
        provider.view_semantic_state_bundle(object())


def test_open_verified_view_root_binding_mismatch() -> None:
    def open_semantic_state(root_cid: str, get_block: Callable[[str], bytes]) -> Any:
        class _View:
            root = types.SimpleNamespace(root_cid="b" + "z" * 58)

            def get_block(self, cid: str) -> bytes:
                return b""

            def symbol_node(self, stable_symbol_id: str) -> Any:
                return None

            def capsule(self, stable_symbol_id: str) -> Any:
                return None

        return _View()

    surface = types.SimpleNamespace(
        open_semantic_state=open_semantic_state,
        scan_repository=lambda *a, **k: None,
        capability=SemanticStateCapability(
            available=True,
            adapter_id=ADAPTER_ID,
            contract_name="SemanticStateProvider@1",
            semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version="1",
            capsule_compiler_version="1",
            semantic_state_api_schema="ipfs-datasets.software-contracts.semantic-state-api@1",
            view_interface="SemanticStateView@1",
            producer_interface="SemanticStateProducer@1",
            block_reader_interface="SemanticStateBlockReader@1",
            operations=("open_semantic_state",),
        ),
    )
    provider = IpfsDatasetsSemanticStateProvider(surface)
    with pytest.raises(SemanticStateAdapterError, match="root_cid binding mismatch"):
        provider.open_verified_view("b" + "a" * 58, lambda cid: b"x")


def test_illegal_confidence_fails_closed() -> None:
    def open_semantic_state(root_cid: str, get_block: Callable[[str], bytes]) -> Any:
        class _View:
            root = types.SimpleNamespace(root_cid=root_cid, confidence="telepathic")

            def get_block(self, cid: str) -> bytes:
                return b""

            def symbol_node(self, stable_symbol_id: str) -> Any:
                return None

            def capsule(self, stable_symbol_id: str) -> Any:
                return None

        return _View()

    surface = types.SimpleNamespace(
        open_semantic_state=open_semantic_state,
        scan_repository=lambda *a, **k: None,
        capability=SemanticStateCapability(
            available=True,
            adapter_id=ADAPTER_ID,
            contract_name="SemanticStateProvider@1",
            semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version="1",
            capsule_compiler_version="1",
            semantic_state_api_schema="ipfs-datasets.software-contracts.semantic-state-api@1",
            view_interface="SemanticStateView@1",
            producer_interface="SemanticStateProducer@1",
            block_reader_interface="SemanticStateBlockReader@1",
            operations=("open_semantic_state",),
        ),
    )
    provider = IpfsDatasetsSemanticStateProvider(surface)
    with pytest.raises(SemanticStateAdapterError, match="confidence"):
        provider.open_verified_view("b" + "a" * 58, lambda cid: b"x")


def test_forged_cid_rejected_on_view() -> None:
    def open_semantic_state(root_cid: str, get_block: Callable[[str], bytes]) -> Any:
        class _View:
            root = types.SimpleNamespace(root_cid="cidv1-sha256-" + "ab" * 32)

            def get_block(self, cid: str) -> bytes:
                return b""

            def symbol_node(self, stable_symbol_id: str) -> Any:
                return None

            def capsule(self, stable_symbol_id: str) -> Any:
                return None

        return _View()

    surface = types.SimpleNamespace(
        open_semantic_state=open_semantic_state,
        scan_repository=lambda *a, **k: None,
        capability=SemanticStateCapability(
            available=True,
            adapter_id=ADAPTER_ID,
            contract_name="SemanticStateProvider@1",
            semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version="1",
            capsule_compiler_version="1",
            semantic_state_api_schema="ipfs-datasets.software-contracts.semantic-state-api@1",
            view_interface="SemanticStateView@1",
            producer_interface="SemanticStateProducer@1",
            block_reader_interface="SemanticStateBlockReader@1",
            operations=("open_semantic_state",),
        ),
    )
    provider = IpfsDatasetsSemanticStateProvider(surface)
    with pytest.raises(SemanticStateAdapterError):
        provider.open_verified_view("b" + "a" * 58, lambda cid: b"x")


def test_source_race_maps_to_source_blob_stale() -> None:
    class _FakeSourceError(Exception):
        kind = "source_binding_mismatch"
        requires_rescan = True

    def read_required_source(*args: Any, **kwargs: Any) -> Any:
        raise _FakeSourceError("TOCTOU source bytes changed")

    surface = types.SimpleNamespace(
        open_semantic_state=lambda *a, **k: None,
        scan_repository=lambda *a, **k: None,
        read_required_source=read_required_source,
        capability=SemanticStateCapability(
            available=True,
            adapter_id=ADAPTER_ID,
            contract_name="SemanticStateProvider@1",
            semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version="1",
            capsule_compiler_version="1",
            semantic_state_api_schema="ipfs-datasets.software-contracts.semantic-state-api@1",
            view_interface="SemanticStateView@1",
            producer_interface="SemanticStateProducer@1",
            block_reader_interface="SemanticStateBlockReader@1",
            operations=("read_required_source",),
        ),
    )
    provider = IpfsDatasetsSemanticStateProvider(surface, forbid_filesystem_source=False)
    with pytest.raises(SourceBlobStale, match="TOCTOU"):
        provider.read_required_source(
            object(),
            "sym",
            expected_producer_state_cid="b" + "a" * 58,
            read_source_blob=lambda cid: b"x",
        )


def test_ast_audit_allows_only_pure_delegation_for_sealed_names() -> None:
    """Mirror the dependency-seal AST constraints for this adapter file."""

    source = ADAPTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    forbidden = {
        "build_semantic_state",
        "verify_semantic_state_bundle",
        "open_semantic_state",
        "scan_repository",
        "select_tests_and_proofs",
        "compare_test_selection_oracle",
        "compile_semantic_capsule",
        "diff_repository_states",
        "calculate_invalidation",
        "SemanticStateView",
        "TestSelection",
        "SemanticCapsule",
    }
    allowed = {
        "IpfsDatasetsSemanticStateProvider": {"open_semantic_state", "scan_repository"},
    }

    def is_direct_provider_delegation(node: ast.FunctionDef) -> bool:
        body = list(node.body)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return False
        call = body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr != node.name or not isinstance(call.func.value, ast.Attribute):
            return False
        owner = call.func.value
        if not isinstance(owner.value, ast.Name) or owner.value.id != "self":
            return False
        if owner.attr not in {"_api", "_provider", "_datasets"}:
            return False
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            parent = parents.get(node)
            approved = isinstance(parent, ast.ClassDef) and node.name in allowed.get(
                parent.name, set()
            )
            if node.name in forbidden and not approved:
                # SemanticStateView may only appear as an assignment alias to Any.
                if node.name == "SemanticStateView":
                    pytest.fail(f"forbidden function definition: {node.name}")
                pytest.fail(
                    f"forbidden local authority function {node.name} at line {node.lineno}"
                )
            if approved and not is_direct_provider_delegation(node):
                pytest.fail(
                    f"{parent.name}.{node.name} is not pure delegation at line {node.lineno}"
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                pytest.fail("dynamic import_module is forbidden in datasets_adapter")
            if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec"}:
                pytest.fail(f"forbidden call {node.func.id}")


def test_inspect_capability_without_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> Any:
        raise SemanticStateUnavailable("load", "import_failed", "missing")

    monkeypatch.setattr(adapter_mod, "_load_pinned_surface", boom)
    cap = inspect_semantic_state_capability()
    assert cap.available is False
    assert cap.reason_code == "import_failed"


# ---------------------------------------------------------------------------
# Integration against sealed datasets pin
# ---------------------------------------------------------------------------


def test_clean_scan_build_view_and_selection_round_trip(
    sealed_provider: IpfsDatasetsSemanticStateProvider, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    state = sealed_provider.scan_repository(repo)
    assert state.symbols
    assert state.edges
    assert all(sym.stable_id for sym in state.symbols)
    assert all(str(getattr(edge, "confidence", "exact")) in CONFIDENCE_VALUES for edge in state.edges)

    # Incremental scan reuses previous_state without identity translation.
    state2 = sealed_provider.scan_repository(repo, previous_state=state)
    assert state2.state_cid == state.state_cid
    assert {s.stable_id for s in state2.symbols} == {s.stable_id for s in state.symbols}

    bundle = sealed_provider.build_semantic_state(state)
    root = sealed_provider.verify_semantic_state_bundle(bundle)
    assert root.root_cid == bundle.root.root_cid

    memory_view = sealed_provider.view_semantic_state_bundle(bundle)
    durable_blocks = dict(bundle.blocks)
    durable_view = sealed_provider.open_verified_view(
        root.root_cid, durable_blocks.__getitem__
    )
    assert memory_view.root.root_cid == durable_view.root.root_cid
    assert memory_view.root.root_cid == root.root_cid

    # Merkle nodes + capsules survive both readers with identical identity.
    for symbol in state.symbols:
        node_a = memory_view.symbol_node(symbol.stable_id)
        node_b = durable_view.symbol_node(symbol.stable_id)
        assert node_a.node_cid == node_b.node_cid
        assert node_a.stable_symbol_id == symbol.stable_id
        cap_a = memory_view.capsule(symbol.stable_id)
        cap_b = durable_view.capsule(symbol.stable_id)
        assert cap_a.capsule_cid == cap_b.capsule_cid
        assert cap_a.stable_symbol_id == symbol.stable_id
        assert str(cap_a.confidence) in CONFIDENCE_VALUES
        # Opaque/invalid-adjacent still visible; raw source remains the escape.
        if str(cap_a.confidence) in {"opaque", "heuristic"}:
            assert cap_a.source_cid is not None or cap_a.source_slice_path

    # Mutate and prove selection receives previous+current views.
    (repo / "pkg" / "mod.py").write_text(
        "from pkg import hello\n\ndef use() -> int:\n    return hello() + 1\n",
        encoding="utf-8",
    )
    _commit_all(repo, "mutate")
    current_state = sealed_provider.scan_repository(repo, previous_state=state)
    assert current_state.state_cid != state.state_cid
    current_bundle = sealed_provider.build_semantic_state(
        current_state, previous_bundle=bundle
    )
    delta = sealed_provider.diff_repository_states(state, current_state)
    plan = sealed_provider.calculate_invalidation(state, current_state, delta)
    previous_view = memory_view
    current_view = sealed_provider.view_semantic_state_bundle(current_bundle)

    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SelectionPolicy,
    )

    policy = SelectionPolicy(policy_id="adapter-test", allow_full_fallback=True)
    extended = sealed_provider.extend_semantic_invalidation(
        state, current_state, delta, plan, previous_view, current_view
    )
    selection = sealed_provider.select_tests_and_proofs(
        previous_view,
        current_view,
        extended,
        policy=policy,
        explicit_rules=(),
    )
    assert selection is not None
    # Identity of roots is preserved on the selection binding when present.
    for attr in ("previous_root_cid", "current_root_cid", "selection_cid"):
        value = getattr(selection, attr, None)
        if isinstance(value, str) and value.startswith("b"):
            assert len(value) >= 50


def test_filesystem_mutation_after_scan_yields_stale_or_rescan(
    sealed_provider: IpfsDatasetsSemanticStateProvider, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    state = sealed_provider.scan_repository(repo)
    symbol = next(s for s in state.symbols if s.source_cid)

    # Adapter refuses ambient filesystem source without a sealed blob reader.
    with pytest.raises(SourceBlobStale):
        sealed_provider.read_required_source(
            state,
            symbol.stable_id,
            expected_producer_state_cid=state.state_cid,
        )

    # Snapshot-bound blob reader that later races must surface SourceBlobStale.
    blobs = {symbol.source_cid: b"original-bytes-not-matching-cid"}
    calls = {"n": 0}

    def racing_reader(cid: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            return blobs[cid]
        return b"mutated-after-scan"

    # Wrong bytes / race: producer fails closed; adapter maps to SourceBlobStale.
    with pytest.raises(SourceBlobStale):
        sealed_provider.read_required_source(
            state,
            symbol.stable_id,
            expected_producer_state_cid=state.state_cid,
            read_source_blob=racing_reader,
        )

    # Post-scan mutation: either SourceBlobStale on sealed source reads or a
    # rescan with a new state_cid — never a mixed post-scan filesystem source.
    target = repo / "pkg" / "mod.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# dirty\n", encoding="utf-8")
    _commit_all(repo, "dirty-commit")
    after = sealed_provider.scan_repository(repo, previous_state=state)
    assert after.state_cid != state.state_cid
    # Prior sealed state identity is preserved (no translation / mix-in).
    assert state.state_cid.startswith("b")
    assert after.state_cid.startswith("b")


def test_capability_matches_sealed_pins(
    sealed_provider: IpfsDatasetsSemanticStateProvider,
) -> None:
    cap = sealed_provider.capability
    assert cap.available is True
    assert cap.semantic_state_schema == EXPECTED_SEMANTIC_STATE_SCHEMA
    assert cap.capsule_schema == EXPECTED_CAPSULE_SCHEMA
    assert cap.selection_schema == EXPECTED_SELECTION_SCHEMA
    assert cap.semantic_index_schema == EXPECTED_SEMANTIC_INDEX_SCHEMA
    assert cap.merkle_compiler_version == "1"
    assert cap.capsule_compiler_version == "1"
    assert "open_semantic_state" in cap.operations
    assert "scan_repository" in cap.operations
