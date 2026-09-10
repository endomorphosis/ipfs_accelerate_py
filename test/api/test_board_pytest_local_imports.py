"""Local operator imports must not disappear from affected-test selection."""

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    board_pytest_selection as selection,
)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return name


def selected(root, monkeypatch, dirty, *tests):
    monkeypatch.setattr(selection, "dirty_source_paths", lambda _: (dirty,))
    result = selection.select_affected_test_files(root, tests)
    # None means a conservative full run; either result must retain dependencies.
    return set(tests) if result is None else set(result)


@pytest.mark.parametrize("module", ["scripts.operator", "local_operator", "other_package.entry"])
def test_local_operator_dependency_is_selected_with_direct_consumer(tmp_path, monkeypatch, module):
    dependency = write(tmp_path, "ipfs_accelerate_py/worker.py", "VALUE = 2\n")
    write(tmp_path, module.replace(".", "/") + ".py", "from ipfs_accelerate_py.worker import VALUE\n")
    direct = write(tmp_path, "test/test_direct.py", "from ipfs_accelerate_py.worker import VALUE\n")
    indirect = write(tmp_path, "test/test_indirect.py", f"from {module} import VALUE\n")
    unrelated = write(tmp_path, "test/test_unrelated.py", "def test_other():\n    assert True\n")
    assert selected(tmp_path, monkeypatch, dependency, direct, indirect, unrelated) == {direct, indirect}


def test_package_initialization_is_a_dependency(tmp_path, monkeypatch):
    dependency = write(tmp_path, "ipfs_accelerate_py/worker.py", "VALUE = 2\n")
    write(tmp_path, "scripts/__init__.py", "from ipfs_accelerate_py.worker import VALUE\n")
    write(tmp_path, "scripts/operator.py", "VALUE = 1\n")
    direct = write(tmp_path, "test/test_direct.py", "from ipfs_accelerate_py.worker import VALUE\n")
    indirect = write(tmp_path, "test/test_indirect.py", "import scripts.operator\n")
    unrelated = write(tmp_path, "test/test_unrelated.py", "VALUE = 1\n")
    assert selected(tmp_path, monkeypatch, dependency, direct, indirect, unrelated) == {direct, indirect}


def test_oversized_local_operator_retains_uncertain_test(tmp_path, monkeypatch):
    dependency = write(tmp_path, "ipfs_accelerate_py/worker.py", "VALUE = 2\n")
    write(tmp_path, "scripts/operator.py", "# large\n" * 100)
    monkeypatch.setattr(selection, "_MAX_FILE_BYTES", 256)
    direct = write(tmp_path, "test/test_direct.py", "from ipfs_accelerate_py.worker import VALUE\n")
    indirect = write(tmp_path, "test/test_indirect.py", "from scripts import operator\n")
    unrelated = write(tmp_path, "test/test_unrelated.py", "VALUE = 1\n")
    assert selected(tmp_path, monkeypatch, dependency, direct, indirect, unrelated) == {direct, indirect}


def test_non_utf8_local_operator_retains_uncertain_test(tmp_path, monkeypatch):
    dependency = write(tmp_path, "ipfs_accelerate_py/worker.py", "VALUE = 2\n")
    path = write(tmp_path, "scripts/operator.py", "")
    (tmp_path / path).write_bytes(b"# coding: latin-1\nVALUE = 'caf\xe9'\n")
    direct = write(tmp_path, "test/test_direct.py", "from ipfs_accelerate_py.worker import VALUE\n")
    indirect = write(tmp_path, "test/test_indirect.py", "from scripts import operator\n")
    unrelated = write(tmp_path, "test/test_unrelated.py", "VALUE = 1\n")
    assert selected(tmp_path, monkeypatch, dependency, direct, indirect, unrelated) == {direct, indirect}


def test_external_symlink_is_uncertain_without_reading_its_source(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    dependency = write(root, "ipfs_accelerate_py/worker.py", "VALUE = 2\n")
    outside = tmp_path / "outside"
    write(outside, "operator.py", "raise AssertionError('outside source must not be read')\n")
    (root / "scripts").symlink_to(outside, target_is_directory=True)
    direct = write(root, "test/test_direct.py", "from ipfs_accelerate_py.worker import VALUE\n")
    indirect = write(root, "test/test_indirect.py", "from scripts import operator\n")
    unrelated = write(root, "test/test_unrelated.py", "VALUE = 1\n")
    original = Path.read_text

    def confined_read(path, *args, **kwargs):
        assert not path.resolve().is_relative_to(outside)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", confined_read)
    assert selected(root, monkeypatch, dependency, direct, indirect, unrelated) == {direct, indirect}


def test_relative_local_operator_import_keeps_its_dependency(tmp_path, monkeypatch):
    dependency = write(tmp_path, "scripts/helpers.py", "VALUE = 2\n")
    write(tmp_path, "scripts/operator.py", "from .helpers import VALUE\n")
    direct = write(tmp_path, "test/test_direct.py", "from scripts.helpers import VALUE\n")
    indirect = write(tmp_path, "test/test_indirect.py", "from scripts.operator import VALUE\n")
    unrelated = write(tmp_path, "test/test_unrelated.py", "VALUE = 1\n")
    assert selected(tmp_path, monkeypatch, dependency, direct, indirect, unrelated) == {direct, indirect}


@pytest.mark.parametrize("deleted,statement", [
    ("scripts/deleted.py", "import scripts.deleted\n"),
    ("scripts/deleted.py", "from scripts import deleted\n"),
    ("local_operator.py", "import local_operator\n"),
])
def test_deleted_local_module_cannot_be_hidden_by_another_selected_test(
    tmp_path, monkeypatch, deleted, statement,
):
    dependency = write(tmp_path, "changed.py", "VALUE = 2\n")
    write(tmp_path, "scripts/__init__.py", "# existing parent package\n")
    direct = write(tmp_path, "test/test_direct.py", "import changed\n")
    indirect = write(tmp_path, "test/test_indirect.py", statement)
    unrelated = write(tmp_path, "test/test_unrelated.py", "VALUE = 1\n")
    monkeypatch.setattr(selection, "dirty_source_paths", lambda _: (dependency, deleted))
    result = selection.select_affected_test_files(tmp_path, (direct, indirect, unrelated))
    assert result == frozenset({direct, indirect})
