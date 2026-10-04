"""Packaging gates for supervisor contract-analysis integrations."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contract_analysis_is_a_discovered_python_package() -> None:
    marker = (
        REPO_ROOT
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "contract_analysis"
        / "__init__.py"
    )
    assert marker.is_file()

    packages = set(
        find_packages(
            where=str(REPO_ROOT),
            include=["ipfs_accelerate_py", "ipfs_accelerate_py.*"],
        )
    )
    assert "ipfs_accelerate_py.agent_supervisor.contract_analysis" in packages
