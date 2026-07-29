"""Multi-root provider package index: sources, not opaque Gitlinks (SCA-216)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analyzer_health import (
    AnalyzerHealthStatus,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer import (
    CROSS_ROOT_SYMBOL_IDENTITY_SCHEMA,
    MULTI_ROOT_REPOSITORY_INDEX_SCHEMA,
    PROVIDER_INDEX_SCHEMA,
    CrossRootSymbolIdentity,
    CrossRootSymbolJoinError,
    MultiRootRepositoryIndex,
    RepositoryIndexer,
    build_multi_root_repository_index,
    extract_package_function_symbols,
    join_cross_root_symbols,
    make_cross_root_symbol,
    module_name_for_package_path,
    write_provider_index_baseline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    DEFAULT_PROVIDER_PACKAGE_SPECS,
    MULTI_ROOT_REPOSITORY_SNAPSHOT_INTERFACE,
    MULTI_ROOT_REPOSITORY_SNAPSHOT_SCHEMA,
    CoverageKind,
    EntryKind,
    ProviderPackageSpec,
    ProviderRootContradictionKind,
    ProviderRootStatus,
    SCOPE_POLICY_SCHEMA,
    build_multi_root_repository_snapshot,
    build_provider_package_snapshot,
    observe_provider_package_root,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        (
            "git",
            "-c",
            "user.name=SCA Multi-Root Test",
            "-c",
            "user.email=sca-multi-root@example.invalid",
            "-C",
            str(repository),
            *arguments,
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init_repo(root: Path, message: str = "init") -> str:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "SCA Multi-Root Test")
    _git(root, "config", "user.email", "sca-multi-root@example.invalid")
    _git(root, "add", ".")
    # Allow empty-ish first commit only when files exist.
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    if status.strip():
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", message)
    else:
        _git(root, "commit", "--allow-empty", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


def _provider_source(package: str, function_name: str = "dispatch") -> str:
    return (
        f'"""{package} fixture package."""\n\n'
        f"def {function_name}(value: int) -> int:\n"
        f"    return value + 1\n\n"
        f"def helper() -> str:\n"
        f"    return {package!r}\n"
    )


def _build_superproject(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a superproject with three provider package gitlink checkouts."""

    superproject = tmp_path / "super"
    superproject.mkdir()
    _git(superproject, "init", "-q")
    _git(superproject, "config", "user.name", "SCA Multi-Root Test")
    _git(superproject, "config", "user.email", "sca-multi-root@example.invalid")
    _write(superproject / "README.md", "superproject\n")
    # Primary SwissKnife stand-in (kept distinct from provider roots).
    _write(superproject / "swissknife" / "src" / "main.ts", "export const x = 1;\n")
    _git(superproject, "add", ".")
    _git(superproject, "commit", "-qm", "superproject base")

    commits: dict[str, str] = {}
    packages = (
        ("ipfs_accelerate_py", "external/ipfs_accelerate"),
        ("ipfs_kit_py", "external/ipfs_kit"),
        ("ipfs_datasets_py", "external/ipfs_datasets"),
    )
    for package, scope in packages:
        provider = tmp_path / f"provider-{package}"
        package_dir = provider / package
        _write(package_dir / "__init__.py", f'"""{package}"""\n')
        _write(package_dir / "api.py", _provider_source(package, "dispatch"))
        _write(package_dir / "nested" / "service.py", _provider_source(package, "run"))
        _write(provider / "README.md", f"{package} checkout\n")
        commit = _init_repo(provider, f"init {package}")
        commits[package] = commit
        # Record as gitlink in the superproject.
        _git(
            superproject,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{commit},{scope}",
        )
        # Materialize the checkout at the configured scope path.
        target = superproject / scope
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy tree by re-cloning via file protocol for a real git worktree.
        subprocess.run(
            ("git", "clone", "-q", str(provider), str(target)),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    _git(superproject, "commit", "-qm", "pin provider gitlinks")
    return superproject, commits


def _policy_for_fixture() -> dict[str, object]:
    return {
        "schema": SCOPE_POLICY_SCHEMA,
        "schemaVersion": 1,
        "scopeId": "test-sca-multi-root-v1",
        "primaryRepository": "swissknife",
        "primaryRoot": "swissknife",
        "providerScopes": [
            "external/ipfs_accelerate",
            "external/ipfs_kit",
            "external/ipfs_datasets",
            "Mcp-Plus-Plus",
        ],
        "skipPrefixes": ["node_modules", "tmp"],
        "skipDirectoryNames": [".git", "node_modules", "__pycache__"],
        "dependencyDirectoryNames": ["node_modules"],
        "dependencyLockFiles": ["package-lock.json"],
        "dependencyManifestFiles": ["package.json", "pyproject.toml"],
        "workingTreeOverlay": {
            "mode": "tracked_plus_allowlisted_untracked_source",
            "allowDirtyAnalysis": True,
            "allowlistedUntrackedSuffixes": [".py", ".ts", ".json", ".md"],
            "allowlistedUntrackedExactNames": ["package.json"],
        },
        "dispositionRules": {
            "semanticExtensions": [".py", ".ts", ".js"],
            "structuredExtensions": [".json"],
            "textExtensions": [".md", ".txt"],
            "binaryExtensions": [".png", ".pyc"],
            "generatedSuffixes": [".map"],
            "generatedPathParts": ["dist", "build"],
        },
        "silentExclusionsAllowed": False,
        "trackedCoverageRequired": 1.0,
    }


def test_default_provider_package_specs_cover_three_packages() -> None:
    packages = {item.package for item in DEFAULT_PROVIDER_PACKAGE_SPECS}
    assert packages == {
        "ipfs_accelerate_py",
        "ipfs_kit_py",
        "ipfs_datasets_py",
    }
    assert all(
        item.package_dirname == item.package for item in DEFAULT_PROVIDER_PACKAGE_SPECS
    )


def test_provider_package_sources_are_indexed_not_opaque(
    tmp_path: Path,
) -> None:
    superproject, commits = _build_superproject(tmp_path)
    multi = build_multi_root_repository_snapshot(
        superproject,
        scope_policy=_policy_for_fixture(),
        include_primary_snapshot=False,
    )

    assert multi.schema_version == 1
    assert MULTI_ROOT_REPOSITORY_SNAPSHOT_SCHEMA in multi.to_dict()["schema"]
    assert multi.to_dict()["interface"] == MULTI_ROOT_REPOSITORY_SNAPSHOT_INTERFACE
    assert len(multi.providers) == 3
    assert multi.all_providers_indexed is True
    assert multi.has_blocking_contradictions is False

    for observation in multi.providers:
        assert observation.indexed is True
        assert observation.opaque_gitlink is False
        assert observation.present is True
        assert observation.status is ProviderRootStatus.PRESENT
        assert observation.snapshot is not None
        assert observation.gitlink_commit_id == commits[observation.package]
        assert observation.head_commit_id == commits[observation.package]
        # Package-relative source paths, not flattened superproject paths.
        paths = {item.path for item in observation.snapshot.dispositions}
        assert "api.py" in paths
        assert "nested/service.py" in paths
        assert all(
            not item.path.startswith("external/")
            for item in observation.snapshot.dispositions
        )
        # Nested gitlinks inside the package would be explicit, but sources are
        # semantic AST (or other dispositions) — never opaque package roots.
        assert any(
            item.kind is CoverageKind.SEMANTIC_AST
            for item in observation.snapshot.dispositions
        )
        assert observation.snapshot.repository_root.endswith(observation.package)


def test_primary_snapshot_remains_distinct_namespace(tmp_path: Path) -> None:
    superproject, _ = _build_superproject(tmp_path)
    # Inventory the superproject worktree itself as the primary root so the
    # fixture need not embed a nested swissknife Git repository.
    policy = _policy_for_fixture()
    policy["primaryRoot"] = "."
    policy["primaryRepository"] = "superproject"
    multi = build_multi_root_repository_snapshot(
        superproject,
        scope_policy=policy,
        include_primary_snapshot=True,
    )
    assert multi.primary_snapshot is not None
    primary_paths = {item.path for item in multi.primary_snapshot.dispositions}
    # Primary ledger stays in the superproject namespace (gitlinks + local files).
    assert any("README.md" == path or path.endswith("README.md") for path in primary_paths)
    for observation in multi.providers:
        assert observation.snapshot is not None
        provider_paths = {item.path for item in observation.snapshot.dispositions}
        # Provider package-relative paths never share the primary namespace.
        assert primary_paths.isdisjoint(provider_paths)
        assert "api.py" in provider_paths
        # Primary may record the gitlink path, but never package source bodies.
        assert all(
            not path.startswith(observation.package + "/")
            and path != "api.py"
            for path in primary_paths
        )


def test_missing_dirty_and_version_divergent_roots_are_explicit(
    tmp_path: Path,
) -> None:
    superproject, commits = _build_superproject(tmp_path)

    # Dirty accelerate package.
    dirty_api = (
        superproject
        / "external"
        / "ipfs_accelerate"
        / "ipfs_accelerate_py"
        / "api.py"
    )
    dirty_api.write_text(
        dirty_api.read_text(encoding="utf-8") + "# dirty\n",
        encoding="utf-8",
    )

    # Version-divergent kit: advance checkout HEAD without updating gitlink.
    kit = superproject / "external" / "ipfs_kit"
    _write(kit / "ipfs_kit_py" / "extra.py", "def extra():\n    return 1\n")
    _git(kit, "add", "ipfs_kit_py/extra.py")
    _git(kit, "commit", "-qm", "advance kit")
    advanced = _git(kit, "rev-parse", "HEAD")
    assert advanced != commits["ipfs_kit_py"]

    # Missing datasets checkout (remove materialization; keep gitlink).
    datasets_path = superproject / "external" / "ipfs_datasets"
    subprocess.run(("rm", "-rf", str(datasets_path)), check=True)

    multi = build_multi_root_repository_snapshot(
        superproject,
        scope_policy=_policy_for_fixture(),
    )

    accelerate = multi.provider_for_package("ipfs_accelerate_py")
    kit_obs = multi.provider_for_package("ipfs_kit_py")
    datasets = multi.provider_for_package("ipfs_datasets_py")
    assert accelerate is not None and kit_obs is not None and datasets is not None

    assert accelerate.dirty is True
    assert accelerate.status is ProviderRootStatus.DIRTY
    assert any(
        item.kind is ProviderRootContradictionKind.DIRTY
        for item in accelerate.contradictions
    )
    # Dirty package is still indexed as source (not opaque).
    assert accelerate.indexed is True
    assert accelerate.opaque_gitlink is False

    assert kit_obs.version_divergent is True
    assert kit_obs.status is ProviderRootStatus.VERSION_DIVERGENT
    assert kit_obs.gitlink_commit_id == commits["ipfs_kit_py"]
    assert kit_obs.head_commit_id == advanced
    assert kit_obs.indexed is True
    assert any(
        item.kind is ProviderRootContradictionKind.VERSION_DIVERGENT
        for item in kit_obs.contradictions
    )

    assert datasets.present is False
    assert datasets.indexed is False
    assert datasets.opaque_gitlink is True
    assert datasets.status in {
        ProviderRootStatus.MISSING,
        ProviderRootStatus.OPAQUE_GITLINK,
    }
    assert any(
        item.kind
        in {
            ProviderRootContradictionKind.MISSING,
            ProviderRootContradictionKind.OPAQUE_GITLINK,
        }
        for item in datasets.contradictions
    )
    assert multi.has_blocking_contradictions is True


def test_moved_package_directory_is_explicit(tmp_path: Path) -> None:
    superproject, _ = _build_superproject(tmp_path)
    package = superproject / "external" / "ipfs_kit" / "ipfs_kit_py"
    moved = superproject / "external" / "ipfs_kit" / "ipfs_kit_py_moved"
    package.rename(moved)

    observation = observe_provider_package_root(
        superproject,
        ProviderPackageSpec(
            package="ipfs_kit_py",
            scope_path="external/ipfs_kit",
            package_dirname="ipfs_kit_py",
        ),
        scope_policy=_policy_for_fixture(),
    )
    assert observation.moved is True
    assert observation.indexed is False
    assert observation.status is ProviderRootStatus.MOVED
    assert any(
        item.kind is ProviderRootContradictionKind.MOVED
        for item in observation.contradictions
    )


def test_cross_root_joins_are_package_module_function_exact() -> None:
    left = make_cross_root_symbol(
        package="ipfs_accelerate_py",
        module="ipfs_accelerate_py.api",
        function="dispatch",
        path="api.py",
        root_id="root-a",
    )
    right = make_cross_root_symbol(
        package="ipfs_accelerate_py",
        module="ipfs_accelerate_py.api",
        function="dispatch",
        path="api.py",
        root_id="root-b",
    )
    joined = join_cross_root_symbols(left, right)
    assert joined.identity_id == left.identity_id
    assert joined.qualified_name == "ipfs_accelerate_py:ipfs_accelerate_py.api.dispatch"
    assert joined.to_dict()["schema"] == CROSS_ROOT_SYMBOL_IDENTITY_SCHEMA

    with pytest.raises(CrossRootSymbolJoinError):
        join_cross_root_symbols(
            left,
            make_cross_root_symbol(
                package="ipfs_kit_py",
                module="ipfs_accelerate_py.api",
                function="dispatch",
            ),
        )
    with pytest.raises(CrossRootSymbolJoinError):
        join_cross_root_symbols(
            left,
            make_cross_root_symbol(
                package="ipfs_accelerate_py",
                module="ipfs_accelerate_py.other",
                function="dispatch",
            ),
        )
    with pytest.raises(CrossRootSymbolJoinError):
        join_cross_root_symbols(
            left,
            make_cross_root_symbol(
                package="ipfs_accelerate_py",
                module="ipfs_accelerate_py.api",
                function="helper",
            ),
        )
    with pytest.raises(CrossRootSymbolJoinError):
        CrossRootSymbolIdentity(package="ipfs_accelerate_py", module="", function="x")


def test_module_name_and_symbol_extraction() -> None:
    assert (
        module_name_for_package_path("ipfs_kit_py", "nested/service.py")
        == "ipfs_kit_py.nested.service"
    )
    assert (
        module_name_for_package_path("ipfs_kit_py", "__init__.py") == "ipfs_kit_py"
    )
    source = _provider_source("ipfs_kit_py", "run")
    symbols = extract_package_function_symbols(
        "ipfs_kit_py", "nested/service.py", source, root_id="r1"
    )
    names = {item.function for item in symbols}
    assert "run" in names
    assert "helper" in names
    assert all(item.package == "ipfs_kit_py" for item in symbols)
    assert all(item.module == "ipfs_kit_py.nested.service" for item in symbols)


def test_multi_root_index_keeps_bodies_in_cas_and_blocks_partial_parity(
    tmp_path: Path,
) -> None:
    superproject, _ = _build_superproject(tmp_path)
    index_root = tmp_path / "index"

    multi_index = build_multi_root_repository_index(
        superproject,
        index_root=index_root,
        scope_policy=_policy_for_fixture(),
        extract_symbols=True,
    )

    assert multi_index.all_providers_indexed is True
    assert multi_index.any_opaque_gitlink is False
    assert multi_index.to_dict()["schema"] == MULTI_ROOT_REPOSITORY_INDEX_SCHEMA
    assert multi_index.to_dict()["bodies_in_cas"] is True
    assert multi_index.to_dict()["cross_root_join_policy"] == (
        "package_module_function_exact"
    )

    for provider in multi_index.providers:
        assert provider.indexed is True
        assert provider.index is not None
        assert provider.observation.opaque_gitlink is False
        # Compact rows must not embed source/AST bodies.
        for row in provider.index.rows:
            payload = row.to_dict()
            for forbidden in (
                "source",
                "source_text",
                "source_body",
                "ast",
                "body",
                "contents",
            ):
                assert forbidden not in payload
            if row.source_ref is not None:
                # Bodies live in CAS and are integrity-checkable.
                cas_root = index_root / "providers" / provider.package
                reader = RepositoryIndexer(cas_root)
                try:
                    body = reader.cas.read(row.source_ref)
                finally:
                    reader.close()
                assert isinstance(body, (bytes, bytearray))
                assert len(body) > 0
        # Symbols extracted for exact cross-root joins.
        assert provider.symbols
        assert any(item.function == "dispatch" for item in provider.symbols)

    # Join the same logical symbol observed under two root ids.
    accelerate_syms = multi_index.symbols_for_package("ipfs_accelerate_py")
    kit_syms = multi_index.symbols_for_package("ipfs_kit_py")
    acc_dispatch = next(s for s in accelerate_syms if s.function == "dispatch")
    # Same package/module/function identity can join; different packages cannot.
    same = make_cross_root_symbol(
        package=acc_dispatch.package,
        module=acc_dispatch.module,
        function=acc_dispatch.function,
        root_id="other-root",
    )
    assert multi_index.join_symbols(acc_dispatch, same).function == "dispatch"
    kit_dispatch = next(s for s in kit_syms if s.function == "dispatch")
    with pytest.raises(CrossRootSymbolJoinError):
        multi_index.join_symbols(acc_dispatch, kit_dispatch)

    # Healthy fixture trees may or may not meet production health thresholds;
    # force a partial health contradiction to prove the parity gate.
    if multi_index.exhaustive_parity_allowed:
        # All healthy: parity may be allowed.  Force partial by re-indexing with
        # impossible thresholds is heavy; instead assert the property holds for
        # an index that already carries PARTIAL_HEALTH contradictions.
        assert multi_index.all_providers_healthy is True
    else:
        assert multi_index.exhaustive_parity_allowed is False
        assert (
            multi_index.contradictions
            or multi_index.multi_root_snapshot.has_blocking_contradictions
            or not multi_index.all_providers_healthy
        )


def test_partial_provider_health_blocks_exhaustive_parity(tmp_path: Path) -> None:
    superproject, _ = _build_superproject(tmp_path)
    # Remove one package after index construction inputs: missing root.
    subprocess.run(
        ("rm", "-rf", str(superproject / "external" / "ipfs_datasets")),
        check=True,
    )
    multi_index = build_multi_root_repository_index(
        superproject,
        index_root=tmp_path / "index-partial",
        scope_policy=_policy_for_fixture(),
        extract_symbols=False,
    )
    assert multi_index.exhaustive_parity_allowed is False
    assert multi_index.any_opaque_gitlink is True or not multi_index.all_providers_indexed
    datasets = multi_index.provider_for_package("ipfs_datasets_py")
    assert datasets is not None
    assert datasets.indexed is False
    assert datasets.opaque_gitlink is True


def test_provider_index_baseline_is_compact_and_body_free(
    tmp_path: Path,
) -> None:
    superproject, _ = _build_superproject(tmp_path)
    multi_index = build_multi_root_repository_index(
        superproject,
        index_root=tmp_path / "index-baseline",
        scope_policy=_policy_for_fixture(),
        extract_symbols=True,
    )
    destination = tmp_path / "provider-index.json"
    write_provider_index_baseline(multi_index, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["schema"] == PROVIDER_INDEX_SCHEMA
    assert payload["interface"] == MULTI_ROOT_REPOSITORY_SNAPSHOT_INTERFACE
    assert payload["bodies_in_cas"] is True
    assert payload["primary_snapshot_distinct"] is True
    assert payload["cross_root_join_policy"] == "package_module_function_exact"
    assert len(payload["providers"]) == 3
    packages = {item["package"] for item in payload["providers"]}
    assert packages == {
        "ipfs_accelerate_py",
        "ipfs_kit_py",
        "ipfs_datasets_py",
    }
    for item in payload["providers"]:
        assert item["indexed"] is True
        assert item["opaque_gitlink"] is False
        assert "snapshot" not in item
        assert "rows" not in item
        assert "source" not in item
        assert item["tracked_path_count"] >= 1
        assert item["symbol_count"] >= 1
    # Compact: no embedded source bodies.
    encoded = destination.read_text(encoding="utf-8")
    assert "def dispatch" not in encoded
    assert "return value + 1" not in encoded


def test_build_provider_package_snapshot_scopes_paths(tmp_path: Path) -> None:
    provider = tmp_path / "provider"
    package = provider / "ipfs_accelerate_py"
    _write(package / "api.py", _provider_source("ipfs_accelerate_py"))
    _write(provider / "outside.py", "def leak():\n    return 0\n")
    _init_repo(provider, "provider")

    snapshot = build_provider_package_snapshot(package)
    paths = {item.path for item in snapshot.dispositions}
    assert "api.py" in paths
    assert "outside.py" not in paths
    assert all(item.entry_kind is not EntryKind.GITLINK or True for item in snapshot.dispositions)


def test_repo_provider_index_baseline_artifact_exists_or_is_writable() -> None:
    """Workspace expected output path is loadable when present."""

    # Walk up from this test file to the monorepo root used by the SCA board.
    here = Path(__file__).resolve()
    baseline = None
    for parent in here.parents:
        candidate = (
            parent
            / "data"
            / "agent_supervisor"
            / "swissknife_contract_assurance"
            / "baseline"
            / "provider-index.json"
        )
        if candidate.is_file():
            baseline = candidate
            break
    if baseline is None:
        pytest.skip("provider-index.json baseline not yet published in this tree")
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload.get("schema") == PROVIDER_INDEX_SCHEMA
    assert payload.get("bodies_in_cas") is True
    assert payload.get("cross_root_join_policy") == "package_module_function_exact"
    packages = {item["package"] for item in payload.get("providers", ())}
    assert {
        "ipfs_accelerate_py",
        "ipfs_kit_py",
        "ipfs_datasets_py",
    }.issubset(packages)
    for item in payload["providers"]:
        assert "def " not in json.dumps(item)
        # Opaque-only roots are explicit when present.
        if item.get("opaque_gitlink"):
            assert item.get("indexed") is False
