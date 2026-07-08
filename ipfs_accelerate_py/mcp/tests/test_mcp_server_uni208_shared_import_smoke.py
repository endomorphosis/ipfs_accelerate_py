from __future__ import annotations


def test_shared_core_and_github_operations_import_smoke() -> None:
    from ipfs_accelerate_py.shared import GitHubOperations, SharedCore

    core = SharedCore()
    ops = GitHubOperations(core)

    assert core is not None
    assert ops is not None
    assert type(ops).__name__ == "GitHubOperations"


def test_entrypoint_modules_import_smoke() -> None:
    from ipfs_accelerate_py.ai_inference_cli import main as legacy_main
    from ipfs_accelerate_py.cli_entry import main as unified_main

    assert callable(legacy_main)
    assert callable(unified_main)
