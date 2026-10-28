"""Actual Native children keep temporary data and pytest cleanup command-local."""
from __future__ import annotations

import getpass
import json
import tempfile
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


def test_native_chrome_starts_with_a_long_inherited_temporary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Chromium's ProcessSingleton socket is created under TMPDIR. A long
    # operator-selected parent must not force that socket past sun_path's
    # bound, even though ordinary filesystem/profile paths allow more bytes.
    long_parent = tmp_path / ("operator-selected-temporary-root-" * 4)
    long_parent.mkdir(mode=0o700)
    assert len(str(long_parent)) > 108
    monkeypatch.setenv("TMPDIR", str(long_parent))
    monkeypatch.setattr(tempfile, "tempdir", str(long_parent))
    workspace = tmp_path / "chrome-workspace"
    workspace.mkdir(mode=0o700)
    chrome = Path("/opt/google/chrome/chrome")
    assert chrome.is_file(), "actual pinned Chrome executable is required"
    probe = workspace / "browser_probe.py"
    probe.write_text(r'''import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

home = Path(os.environ["HOME"])
private_tmp = Path(tempfile.gettempdir())
profile = Path.cwd() / "a-long-synthetic-profile-path-to-keep-independent-of-the-socket-root" / "profile"
profile.mkdir(mode=0o700, parents=True)
command = ["/opt/google/chrome/chrome", "--headless", "--no-sandbox",
           "--disable-gpu", "--disable-breakpad", "--disable-crash-reporter",
           "--disable-background-networking", "--no-first-run", "--no-default-browser-check",
           "--user-data-dir=" + str(profile), "--dump-dom", "about:blank"]
result = subprocess.run(command, text=True, capture_output=True, timeout=15)
observation = {"returncode": result.returncode, "stdout": result.stdout,
               "stderr": result.stderr, "home": str(home), "tmp": str(private_tmp),
               "home_mode": stat.S_IMODE(home.stat().st_mode),
               "tmp_mode": stat.S_IMODE(private_tmp.stat().st_mode),
               "profile": str(profile), "python_tmp": tempfile.gettempdir()}
(Path.cwd() / "browser-observed.json").write_text(json.dumps(observation), encoding="utf-8")
assert result.returncode == 0, result.stderr
assert "<html>" in result.stdout and "<body>" in result.stdout
''', encoding="utf-8")
    runner = TodoImplementationDaemon._validation_command_runner
    environment = validation_environment_for_runner(build_validation_environment(), runner)
    command = "python browser_probe.py"
    result = runner(spec=SimpleNamespace(command=command, raw_command=command),
                    workspace_path=workspace, timeout_seconds=30, environment=environment)
    assert result["returncode"] == 0, result["output"]
    observed = json.loads((workspace / "browser-observed.json").read_text(encoding="utf-8"))
    home, private_tmp = Path(observed["home"]), Path(observed["tmp"])
    assert observed["returncode"] == 0
    assert len(str(Path(observed["profile"]))) > 108
    assert private_tmp.is_relative_to(home)
    assert len(str(private_tmp)) + len("/.com.google.Chrome.abcdef/SingletonSocket") < 108
    assert observed["home_mode"] == observed["tmp_mode"] == 0o700
    assert not home.exists(), "Native must retire the browser's owned temporary namespace"
    assert list(long_parent.iterdir()) == [], "Native must not leave data under the unbounded parent"
