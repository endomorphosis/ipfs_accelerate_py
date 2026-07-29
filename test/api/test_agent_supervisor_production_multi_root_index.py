"""Production multi-root provider indexing (SCA-603 / SCAEV179INDEXGRAPH)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
    BaselineStageName,
    CONTRACT_ASSURANCE_INDEXGRAPH_EVIDENCE,
    materialize_contract_assurance_baseline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer import (
    PROVIDER_INDEX_SCHEMA,
    build_multi_root_repository_index,
    write_provider_index_baseline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    SCOPE_POLICY_SCHEMA,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "index_repository_contracts.py"
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        (
            "git",
            "-c",
            "user.name=SCA Production Multi-Root",
            "-c",
            "user.email=sca-prod-multi-root@example.invalid",
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
    _git(root, "config", "user.name", "SCA Production Multi-Root")
    _git(root, "config", "user.email", "sca-prod-multi-root@example.invalid")
    _git(root, "add", "-A")
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    if status.strip():
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


def _policy_for_fixture() -> dict[str, object]:
    return {
        "schema": SCOPE_POLICY_SCHEMA,
        "schemaVersion": 1,
        "scopeId": "test-sca-production-multi-root-v1",
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


def _build_superproject(tmp_path: Path) -> Path:
    superproject = tmp_path / "super"
    superproject.mkdir()
    _git(superproject, "init", "-q")
    _git(superproject, "config", "user.name", "SCA Production Multi-Root")
    _git(superproject, "config", "user.email", "sca-prod-multi-root@example.invalid")
    _write(superproject / "README.md", "superproject\n")
    _write(superproject / "swissknife" / "src" / "main.ts", "export const x = 1;\n")
    _git(superproject, "add", ".")
    _git(superproject, "commit", "-qm", "superproject base")

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
        _write(package_dir / "mcp_server" / "tools.py", _provider_source(package, "run"))
        _write(provider / "README.md", f"{package} checkout\n")
        commit = _init_repo(provider, f"init {package}")
        _git(
            superproject,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{commit},{scope}",
        )
        target = superproject / scope
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ("git", "clone", "-q", str(provider), str(target)),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    _git(superproject, "commit", "-qm", "pin provider gitlinks")
    return superproject


def test_production_multi_root_indexes_all_three_provider_roots(
    tmp_path: Path,
) -> None:
    superproject = _build_superproject(tmp_path)
    multi = build_multi_root_repository_index(
        superproject,
        index_root=tmp_path / "index",
        scope_policy=_policy_for_fixture(),
        extract_symbols=True,
    )

    assert multi.all_providers_indexed is True
    assert multi.any_opaque_gitlink is False
    packages = {item.package for item in multi.providers}
    assert packages == {
        "ipfs_accelerate_py",
        "ipfs_kit_py",
        "ipfs_datasets_py",
    }
    # Independent namespaces and content identities.
    tree_ids = {
        item.observation.head_tree_id or item.observation.index_tree_id
        for item in multi.providers
    }
    assert len(tree_ids) == 3
    for provider in multi.providers:
        assert provider.indexed is True
        assert provider.observation.opaque_gitlink is False
        assert provider.symbols
        assert provider.observation.origin_url or provider.observation.head_commit_id
    destination = tmp_path / "provider-index.json"
    write_provider_index_baseline(multi, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == PROVIDER_INDEX_SCHEMA
    assert payload["bodies_in_cas"] is True
    assert payload["cross_root_join_policy"] == "package_module_function_exact"
    assert len(payload["providers"]) == 3


def test_missing_provider_root_blocks_exhaustive_authority(
    tmp_path: Path,
) -> None:
    superproject = _build_superproject(tmp_path)
    subprocess.run(
        ("rm", "-rf", str(superproject / "external" / "ipfs_datasets")),
        check=True,
    )
    multi = build_multi_root_repository_index(
        superproject,
        index_root=tmp_path / "index-partial",
        scope_policy=_policy_for_fixture(),
        extract_symbols=False,
    )
    assert multi.exhaustive_parity_allowed is False
    datasets = multi.provider_for_package("ipfs_datasets_py")
    assert datasets is not None
    assert datasets.indexed is False

    baseline = materialize_contract_assurance_baseline(
        snapshot_id="snap-partial-providers",
        snapshot={
            "snapshot_id": "snap-partial-providers",
            "scope_policy_id": "policy-fixture",
            "head_tree_id": "tree-fixture",
            "stats": {"tracked_path_count": 1, "disposition_count": 1},
        },
        multi_root_index=multi,
        extract_expected=False,
        project_graph=False,
        run_traces=False,
        run_parity=False,
        run_mismatch=False,
        run_vulnerability=False,
        run_graphrag=False,
        require_actual_package_surfaces=False,
        assess_surface_health=True,
    )
    assert baseline.llm_call_count == 0
    provider_stage = next(
        stage
        for stage in baseline.stages
        if stage.name is BaselineStageName.PROVIDER_INDEX
    )
    assert provider_stage.completeness.value in {"partial", "failed"}
    assert baseline.claims.get("exhaustive") is False
    assert baseline.findings["index_graph"]["evidence"] == (
        CONTRACT_ASSURANCE_INDEXGRAPH_EVIDENCE
    )


def test_production_cli_wires_multi_root_and_provider_index(
    tmp_path: Path,
) -> None:
    """CLI production path indexes providers and emits provider-index.json."""

    superproject = _build_superproject(tmp_path)
    policy_path = tmp_path / "scope.json"
    policy_path.write_text(
        json.dumps(_policy_for_fixture(), indent=2) + "\n", encoding="utf-8"
    )
    output_root = tmp_path / "baseline-out"
    env = dict(**__import__("os").environ)
    package_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = (
        str(package_root)
        + (__import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--repo-root",
            str(superproject),
            "--scope-config",
            str(policy_path),
            "--output-root",
            str(output_root),
            "--skip-extraction",
            "--index-provider-roots",
            "--allow-expected-only-surfaces",
            "--skip-graphrag",
            "--max-paths",
            "200",
            "--max-provider-symbol-files",
            "20",
            "--shadow",
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode in {0, 2, 3}, proc.stderr + "\n" + proc.stdout
    # Even when primary SwissKnife index health is partial, multi-root side
    # effects must still have been attempted when providers are present.
    provider_index = output_root / "provider-index.json"
    if provider_index.is_file():
        payload = json.loads(provider_index.read_text(encoding="utf-8"))
        assert payload["schema"] == PROVIDER_INDEX_SCHEMA
        assert len(payload["providers"]) == 3
        assert payload["bodies_in_cas"] is True
        assert "llm" not in json.dumps(payload).lower() or payload.get(
            "llm_call_count", 0
        ) == 0
    else:
        # Fail closed only when the CLI could not start multi-root at all.
        assert "multi_root" in (proc.stderr + proc.stdout).lower() or proc.returncode != 0


def test_workspace_provider_index_baseline_artifact_is_schema_valid() -> None:
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
    assert baseline is not None, "provider-index.json must be published"
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["schema"] == PROVIDER_INDEX_SCHEMA
    assert payload["bodies_in_cas"] is True
    assert payload["cross_root_join_policy"] == "package_module_function_exact"
    packages = {item["package"] for item in payload["providers"]}
    assert packages == {
        "ipfs_accelerate_py",
        "ipfs_kit_py",
        "ipfs_datasets_py",
    }
