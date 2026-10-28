"""Actual Native children keep temporary data and pytest cleanup command-local."""
from __future__ import annotations

import getpass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    build_validation_environment,
    validation_environment_for_runner,
)


@pytest.mark.parametrize("child_fails", [False, True])
def test_real_pytest_children_keep_temp_cleanup_inside_their_fresh_home(
    tmp_path: Path, child_fails: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    shared = tmp_path / "unrelated-shared-temp"
    shared.mkdir(mode=0o700)
    pytest_root = shared / ("pytest-of-" + getpass.getuser())
    garbage = pytest_root / "garbage-unrelated-campaign"
    garbage.mkdir(parents=True, mode=0o700)
    sentinel = garbage / "preserve-me.txt"
    sentinel.write_text("unrelated temporary work", encoding="utf-8")
    sentinel_before = (sentinel.read_bytes(), sentinel.stat().st_ino, sentinel.stat().st_mtime_ns)
    config = workspace / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    test_file = workspace / "test_child_temp.py"
    test_file.write_text(r'''import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def test_owned_temporary_roots(tmp_path):
    home = Path(os.environ["HOME"]).resolve()
    private_tmp = Path(os.environ["TMPDIR"]).resolve()
    nested = subprocess.run(
        [sys.executable, "-I", "-c",
         "import json,os,tempfile; from pathlib import Path; "
         "p=Path(tempfile.mkdtemp(prefix='nested-')); "
         "(p/'nested-write').write_text('owned'); "
         "print(json.dumps({'tmp':tempfile.gettempdir(),'created':str(p)}))"],
        check=True, capture_output=True, text=True,
    )
    observation = {
        "home": str(home), "temp": str(private_tmp), "pytest_tmp": str(tmp_path),
        "python_tmp": tempfile.gettempdir(), "nested": json.loads(nested.stdout),
        "temp_variables": {key: os.environ.get(key) for key in
                           ("TMPDIR", "TEMP", "TMP", "PYTEST_DEBUG_TEMPROOT")},
        "home_mode": stat.S_IMODE(home.stat().st_mode),
        "temp_mode": stat.S_IMODE(private_tmp.stat().st_mode),
    }
    (Path.cwd() / "observed.json").write_text(json.dumps(observation), encoding="utf-8")
    assert private_tmp.is_relative_to(home)
    assert tmp_path.resolve().is_relative_to(private_tmp)
    assert Path(tempfile.gettempdir()).resolve() == private_tmp
    assert Path(observation["nested"]["created"]).is_relative_to(private_tmp)
    assert all(value == str(private_tmp) for value in observation["temp_variables"].values())
    assert observation["home_mode"] == observation["temp_mode"] == 0o700
    assert not (Path.cwd() / "request_failure").exists(), "intentional child failure"
''', encoding="utf-8")
    if child_fails:
        (workspace / "request_failure").touch()
    runner = TodoImplementationDaemon._validation_command_runner
    environment = validation_environment_for_runner(build_validation_environment(), runner)
    # A direct caller cannot redirect temporary state or pytest retention to a
    # shared root. Native must choose fresh owned paths after sanitization.
    environment.update({key: str(shared) for key in
                        ("TMPDIR", "TEMP", "TMP", "PYTEST_DEBUG_TEMPROOT")})
    environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="", PYTEST_PLUGINS="")
    original_environment = dict(environment)
    command = "python -m pytest -q -p no:cacheprovider --confcutdir=. -c pytest.ini test_child_temp.py"
    homes = []
    for _ in range(2):
        result = runner(
            spec=SimpleNamespace(command=command, raw_command=command),
            workspace_path=workspace, timeout_seconds=45,
            environment=environment,
        )
        assert result["returncode"] == (1 if child_fails else 0), result["output"]
        observed = json.loads((workspace / "observed.json").read_text(encoding="utf-8"))
        home = Path(observed["home"])
        private_tmp = Path(observed["temp"])
        assert private_tmp.is_relative_to(home)
        assert Path(observed["pytest_tmp"]).is_relative_to(private_tmp)
        assert observed["python_tmp"] == str(private_tmp)
        assert Path(observed["nested"]["created"]).is_relative_to(private_tmp)
        assert observed["temp_mode"] == observed["home_mode"] == 0o700
        assert not home.exists(), "Native must clean command-owned state after every result"
        assert home not in homes, "commands must not share a temporary namespace"
        homes.append(home)
        assert sentinel.exists(), "pytest cleanup reached unrelated temporary work"
        assert (sentinel.read_bytes(), sentinel.stat().st_ino, sentinel.stat().st_mtime_ns) == sentinel_before
        assert sorted(pytest_root.iterdir()) == [garbage]
    assert environment == original_environment
