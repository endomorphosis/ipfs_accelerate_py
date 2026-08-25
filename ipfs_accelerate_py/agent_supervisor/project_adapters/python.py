"""Qualified Python ProjectAdapter (EAAEF-041)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .base import GenericProjectAdapter, ProjectSupport


PYTHON_TOOLCHAIN: Final[Mapping[str, str]] = MappingProxyType(
    {
        "python": "python3.12",
        "pytest": "python3.12 -m pytest",
        "ruff": "python3.12 -m ruff",
    }
)


class PythonProjectAdapter:
    """Compile locked Python toolchain argv.  Mutation is still not admitted here."""

    def inspect(self, root: Path | str) -> ProjectSupport:
        support = GenericProjectAdapter().inspect(root)
        if "python" not in support.languages:
            return support
        return support

    def focused_test_argv(self, paths: Sequence[str]) -> tuple[str, ...]:
        relative = [str(path) for path in paths if str(path).endswith(".py")]
        if not relative:
            raise ValueError("focused python tests require .py paths")
        return ("python3.12", "-m", "pytest", "-q", *relative)

    def static_argv(self, paths: Sequence[str]) -> tuple[str, ...]:
        relative = [str(path) for path in paths if str(path).endswith(".py")]
        return ("python3.12", "-m", "ruff", "check", *relative)

    def mutation_admitted(self, support: ProjectSupport) -> bool:
        # Qualification of the adapter is not live mutation admission.
        return False


def python_toolchain() -> Mapping[str, Any]:
    return PYTHON_TOOLCHAIN
