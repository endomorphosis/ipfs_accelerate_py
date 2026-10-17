"""Pure observations of both barrier helpers; no gate files, sleep or native IO.

Only the pinned function AST is lifted to module scope. Its body/signature are
unchanged. Source text is read; neither barrier main nor authority is executed.
"""
import ast
import copy
from pathlib import Path
import stat

import pytest

SOURCE = Path(__file__).absolute().parent.parent / "v05"


def reader(name):
    tree = ast.parse((SOURCE / name).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "release_gate_observation")
    unit = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]))
    namespace = {"stat": stat}
    exec(compile(unit, str(SOURCE / name), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("name", ["inner_checker.py", "outer_start_barrier.py"])
def test_empty_publication_and_fstat_write_race_wait_until_exact_G(name):
    observe = reader(name)
    base = dict(mode=stat.S_IFREG | 0o600, nlink=1, now=1, deadline=180)
    assert observe(**base, size=0, data=b"") is False
    assert observe(**base, size=0, data=b"G") is False
    assert observe(**base, size=1, data=b"G") is True


@pytest.mark.parametrize("name", ["inner_checker.py", "outer_start_barrier.py"])
def test_waiting_never_resets_absolute_deadline_or_accepts_G_after_expiry(name):
    observe = reader(name)
    base = dict(mode=stat.S_IFREG | 0o600, nlink=1, deadline=180)
    for now in (0, 90, 179.999):
        assert observe(**base, size=0, data=b"", now=now) is False
    for size, data in ((0, b""), (1, b"G")):
        with pytest.raises(SystemExit, match="^admission barrier expired$"):
            observe(**base, size=size, data=data, now=180)


@pytest.mark.parametrize("name", ["inner_checker.py", "outer_start_barrier.py"])
@pytest.mark.parametrize("fault", ["invalid_byte", "invalid_size", "zero_stat_invalid_data", "symlink", "hardlink"])
def test_malformed_or_aliased_gate_observations_are_refused(name, fault):
    observe = reader(name)
    observation = dict(mode=stat.S_IFREG | 0o600, nlink=1, size=1, data=b"G", now=1, deadline=180)
    if fault == "invalid_byte":
        observation["data"] = b"X"
    elif fault == "invalid_size":
        observation.update(size=2, data=b"GG")
    elif fault == "zero_stat_invalid_data":
        observation.update(size=0, data=b"X")
    elif fault == "symlink":
        observation["mode"] = stat.S_IFLNK | 0o777
    elif fault == "hardlink":
        observation["nlink"] = 2
    with pytest.raises(SystemExit, match="^invalid admission barrier observation$"):
        observe(**observation)
