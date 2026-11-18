"""Implementation CLI defaults and explicit opt-out are consistent end to end."""

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    implementation_supervisor,
)


@pytest.mark.parametrize("module", [implementation_daemon, implementation_supervisor])
@pytest.mark.parametrize("arguments,expected", [
    ([], True), (["--implement"], True), (["--no-implement"], False),
])
def test_implementation_is_opt_out(module, arguments, expected):
    assert module.parse_args(arguments).implement is expected


@pytest.mark.parametrize("module", [implementation_daemon, implementation_supervisor])
def test_conflicting_modes_are_rejected(module):
    with pytest.raises(SystemExit) as error:
        module.parse_args(["--implement", "--no-implement"])
    assert error.value.code == 2
