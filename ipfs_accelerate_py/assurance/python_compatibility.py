"""Canonical Python compatibility metadata for Accelerate (PCPR-036).

Align packaging metadata, classifiers, CI declarations, and installer
selection on the real Python floor. Declared versions must be versions
this tree actually targets. This table is not live CI execution, not
CPU/CUDA qualification, and not a closed PCPR release.

The stable command is ``ipfs-accelerate``. ``ipfs_accelerate`` remains
an explicit compatibility alias for the historical inference CLI.
"""

from __future__ import annotations

from typing import Final

TASK_ID: Final[str] = "PCPR-036"
GOAL_ID: Final[str] = "PCPR-G420"
INTERFACE: Final[str] = "PythonCompatibilityMetadata@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/python-compatibility-metadata@1"

PYTHON_FLOOR: Final[str] = "3.12"
REQUIRES_PYTHON: Final[str] = ">=3.12"
DECLARED_PYTHON_VERSIONS: Final[tuple[str, ...]] = ("3.12",)
FORBIDDEN_PYTHON_VERSIONS: Final[tuple[str, ...]] = ("3.8", "3.9", "3.10", "3.11")

PYPROJECT_REQUIRES_PYTHON_LINE: Final[str] = 'requires-python = ">=3.12"'
SETUP_PYTHON_REQUIRES: Final[str] = 'python_requires=">=3.12"'
REQUIRED_CLASSIFIER: Final[str] = "Programming Language :: Python :: 3.12"
FORBIDDEN_CLASSIFIERS: Final[tuple[str, ...]] = (
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
)

STABLE_COMMAND: Final[str] = "ipfs-accelerate"
COMPATIBILITY_COMMAND: Final[str] = "ipfs_accelerate"
STABLE_COMMAND_ENTRY: Final[str] = "ipfs_accelerate_py.cli_entry:main"
COMPATIBILITY_COMMAND_ENTRY: Final[str] = (
    "ipfs_accelerate_py.ai_inference_cli:main"
)

METADATA_RELPATHS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "setup.py",
    "README.md",
    "docs/guides/getting-started/installation.md",
    "docs/guides/troubleshooting/faq.md",
    "install/install.sh",
    "install/install.ps1",
    "install/INSTALLATION_GUIDE.md",
    "install/Dockerfile.cache",
)
CI_WORKFLOW_RELPATHS: Final[tuple[str, ...]] = (
    ".github/workflows/amd64-ci.yml",
    ".github/workflows/arm64-ci.yml",
    ".github/workflows/multiarch-ci.yml",
)
CI_PYTHON_VERSION_LINE: Final[str] = "PYTHON_VERSION: '3.12'"
CACHE_IMAGE_FROM_LINE: Final[str] = "FROM python:3.12-slim"


def classifier_for(version: str) -> str:
    return f"Programming Language :: Python :: {version}"


def source_declares_floor(source: str) -> bool:
    """True when source uses the canonical requires-python floor string."""

    return PYPROJECT_REQUIRES_PYTHON_LINE in source or SETUP_PYTHON_REQUIRES in source


def source_has_forbidden_classifiers(source: str) -> bool:
    return any(item in source for item in FORBIDDEN_CLASSIFIERS)


def source_has_required_classifier(source: str) -> bool:
    return REQUIRED_CLASSIFIER in source
