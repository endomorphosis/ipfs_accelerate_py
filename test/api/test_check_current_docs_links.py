"""Smoke tests for the allowlisted documentation link checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_current_docs_links.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_current_docs_links", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_module()


def test_github_slug_basic(checker) -> None:
    assert checker.github_slug("Version sources") == "version-sources"
    assert checker.github_slug("Code-owned blockers (do not paper over)") == (
        "code-owned-blockers-do-not-paper-over"
    )


def test_allowlist_files_exist(checker) -> None:
    files, missing = checker.expand_allowlist(checker.ALLOWLIST)
    assert not missing
    assert files
    assert all(path.is_file() for path in files)


def test_checker_main_passes_on_repo(checker) -> None:
    assert checker.main([]) == 0


def test_missing_target_is_reported(checker, tmp_path: Path) -> None:
    # Keep the source file under the repo so resolve_link stays in-tree.
    fixture_dir = REPO_ROOT / "test" / "_linkcheck_fixture"
    fixture_dir.mkdir(exist_ok=True)
    doc = fixture_dir / "doc.md"
    try:
        doc.write_text("# Title\n\nSee [missing](nope.md).\n", encoding="utf-8")
        errors = checker.check_file(doc, check_anchors=True)
        assert any("missing target" in err and "nope.md" in err for err in errors)
    finally:
        if doc.exists():
            doc.unlink()
        try:
            fixture_dir.rmdir()
        except OSError:
            pass
