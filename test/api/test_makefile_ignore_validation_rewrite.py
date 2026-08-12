"""Makefile ignore-check rewrite unblocks CI re-enable proposal gates."""

from __future__ import annotations

import shlex
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.validation.makefile_ignore_check import (
    main as makefile_ignore_main,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    _command_is_allowed,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    rewrite_shell_makefile_ignore_check,
    split_validation_commands,
)


def test_rewrite_shell_makefile_ignore_absent_check() -> None:
    shell = (
        "test -z \"$(rg -n 'ignore=tests/integration/test_swissknife_mobile_interop' "
        "Makefile || true)\""
    )
    rewritten = rewrite_shell_makefile_ignore_check(shell)
    assert "makefile_ignore_check" in rewritten
    assert "--absent" in rewritten
    assert "ignore=tests/integration/test_swissknife_mobile_interop" in rewritten
    argv = tuple(shlex.split(rewritten))
    assert _command_is_allowed(argv, (argv,))


def test_split_validation_commands_rewrites_ignore_check() -> None:
    raw = (
        "PYTHONPATH=src:external/ipfs_accelerate pytest "
        "tests/integration/test_swissknife_mobile_interop.py -q; "
        "test -z \"$(rg -n 'ignore=tests/integration/test_swissknife_mobile_interop' "
        "Makefile || true)\""
    )
    commands = split_validation_commands(raw)
    assert len(commands) == 2
    assert commands[0].startswith("PYTHONPATH=")
    assert "makefile_ignore_check" in commands[1]
    for command in commands:
        argv = tuple(shlex.split(command))
        assert _command_is_allowed(argv, (argv,)), command


def test_makefile_ignore_check_absent_and_present(tmp_path: Path) -> None:
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "test-ci:\n\tpytest --ignore=tests/integration/test_foo.py\n",
        encoding="utf-8",
    )
    assert (
        makefile_ignore_main(
            [
                "--makefile",
                str(makefile),
                "--absent",
                "ignore=tests/integration/test_foo.py",
            ]
        )
        == 1
    )
    assert (
        makefile_ignore_main(
            [
                "--makefile",
                str(makefile),
                "--absent",
                "ignore=tests/integration/test_bar.py",
            ]
        )
        == 0
    )
    assert (
        makefile_ignore_main(
            [
                "--makefile",
                str(makefile),
                "--present",
                "ignore=tests/integration/test_foo.py",
            ]
        )
        == 0
    )
