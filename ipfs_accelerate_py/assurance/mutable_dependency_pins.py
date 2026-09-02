"""Canonical immutable pins for Accelerate VCS dependencies (PCPR-035).

Replace mutable Git branches with immutable commits. Exact versions and
signed artifacts remain valid pin forms. This table is not live package
qualification, not a PyPI install, and not a closed PCPR release.

Kit and Datasets pins bind the portfolio source-authority gitlinks.
Transformers and model-manager pins bind Accelerate nested gitlinks.
libp2p and llama.cpp pins are in-tree declared freezes of a recorded
commit; sealed validation does not re-fetch remotes and does not claim
those packages are installed or live.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

TASK_ID: Final[str] = "PCPR-035"
GOAL_ID: Final[str] = "PCPR-G420"
INTERFACE: Final[str] = "MutableDependencyPins@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/mutable-dependency-pins@1"

PinKind = Literal["portfolio_gitlink", "accelerate_gitlink", "declared_freeze"]

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
GIT_URL_RE: Final = re.compile(
    r"(?:(?P<dist>[A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*)?"
    r"(?P<url>git\+https://[^\s\"'#,;\\]+)"
    r"(?:#(?P<fragment>[^\s\"';\\]+))?"
)
MUTABLE_REFS: Final[frozenset[str]] = frozenset(
    {"main", "master", "HEAD", "head", "develop", "trunk", "latest"}
)
QUALIFIED_PROFILE_RELPATHS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "requirements-hf-server.txt",
    "ipfs_accelerate_py/requirements.txt",
    "ipfs_accelerate_py/mcp/requirements-mcp.txt",
    "install/requirements_base.txt",
    "install/requirements_apple.txt",
    "install/requirements_cuda.txt",
    "install/requirements_github_ci.txt",
    "install/requirements_openvino.txt",
    "install/requirements_qualcomm.txt",
    "ipfs_accelerate_py/worker/skillset/libllama/requirements.txt",
)
RUNTIME_SPEC_RELPATHS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/mcplusplus_module/p2p/libp2p_runtime.py",
    "ipfs_accelerate_py/mcplusplus_module/p2p_transport.py",
    "scripts/dependency_installer.py",
    "scripts/comprehensive_dependency_installer.py",
    "scripts/setup_environment.py",
    "scripts/setup/install_p2p_cache_deps.sh",
    "scripts/auto-update.sh",
    "scripts/validation/test_cross_platform_cache.sh",
    "scripts/validation/validate_docker_cache_setup.sh",
)
EXCLUDED_RELPATH_PREFIXES: Final[tuple[str, ...]] = (
    "packaging/proof_context/",
    "test/",
    "tests/",
    "docs/",
)


@dataclass(frozen=True)
class DependencyPin:
    """One immutable VCS pin. Never a live install or release claim."""

    name: str
    distribution: str
    git_url: str
    commit: str
    pin_kind: PinKind
    subdirectory: str | None = None
    live: bool = False
    evidence_kind: str = "measured"

    def __post_init__(self) -> None:
        if COMMIT_RE.fullmatch(self.commit) is None:
            raise ValueError(f"{self.name} commit must be a 40-character lowercase git id")
        if self.live:
            raise ValueError(f"{self.name} pin cannot claim live qualification")
        if not self.git_url.startswith("git+https://"):
            raise ValueError(f"{self.name} git_url must be a git+https direct reference")

    def pep508(self, *, marker: str | None = None) -> str:
        spec = f"{self.distribution} @ {self.git_url}@{self.commit}"
        if self.subdirectory:
            spec += f"#subdirectory={self.subdirectory}"
        if marker:
            spec += f" ; {marker}"
        return spec

    def direct_url(self) -> str:
        spec = f"{self.git_url}@{self.commit}"
        if self.subdirectory:
            spec += f"#subdirectory={self.subdirectory}"
        return spec

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "distribution": self.distribution,
            "git_url": self.git_url,
            "commit": self.commit,
            "pin_kind": self.pin_kind,
            "subdirectory": self.subdirectory,
            "pep508": self.pep508(),
            "live": False,
            "evidence_kind": self.evidence_kind,
        }


PINS: Final[Mapping[str, DependencyPin]] = MappingProxyType(
    {
        "ipfs_kit_py": DependencyPin(
            name="ipfs_kit_py",
            distribution="ipfs_kit_py",
            git_url="git+https://github.com/endomorphosis/ipfs_kit_py.git",
            commit="a4fd25d944e6aaf25fb51e054a3f62bbcbea544d",
            pin_kind="portfolio_gitlink",
        ),
        "ipfs_datasets_py": DependencyPin(
            name="ipfs_datasets_py",
            distribution="ipfs_datasets_py",
            git_url="git+https://github.com/endomorphosis/ipfs_datasets_py.git",
            commit="d54a6494f50b5982758f8a6d2aafe24ccc7523b8",
            pin_kind="portfolio_gitlink",
        ),
        "ipfs_transformers_py": DependencyPin(
            name="ipfs_transformers_py",
            distribution="ipfs_transformers_py",
            git_url="git+https://github.com/endomorphosis/ipfs_transformers_py.git",
            commit="b397988ed9e3e656475c1cf4417b84efdb95daf3",
            pin_kind="accelerate_gitlink",
        ),
        "ipfs_model_manager_py": DependencyPin(
            name="ipfs_model_manager_py",
            distribution="ipfs_model_manager_py",
            git_url="git+https://github.com/endomorphosis/ipfs_model_manager_py.git",
            commit="f6151d2113f42e75ea7d83a1b2362fc97e55e44d",
            pin_kind="accelerate_gitlink",
        ),
        "libp2p": DependencyPin(
            name="libp2p",
            distribution="libp2p",
            git_url="git+https://github.com/libp2p/py-libp2p.git",
            commit="20d9527ff5a2e8368e77b964b340d0c7ded1a61f",
            pin_kind="declared_freeze",
        ),
        "gguf-py": DependencyPin(
            name="gguf-py",
            distribution="gguf",
            git_url="git+https://github.com/ggerganov/llama.cpp.git",
            commit="b96806d96061049a5b574269b049bf6241d63d46",
            pin_kind="declared_freeze",
            subdirectory="gguf-py",
        ),
    }
)


def pep508_spec(name: str, *, marker: str | None = None) -> str:
    """Return the canonical PEP 508 VCS spec for a named pin."""

    pin = PINS.get(name)
    if pin is None:
        raise KeyError(f"unknown dependency pin {name!r}")
    return pin.pep508(marker=marker)


def _ref_from_git_url(url: str) -> str | None:
    """Return the VCS ref after the last '@' that is not userinfo."""

    body = url
    if body.startswith("git+https://"):
        body = body[len("git+https://") :]
    if "@" not in body:
        return None
    # Userinfo is host-prefixed (user@host); the VCS ref is path-prefixed
    # (repo.git@ref). Split on the last '@'.
    _, _, ref = url.rpartition("@")
    if not ref or "/" in ref or ":" in ref:
        return None
    return ref


def parse_git_direct_references(text: str) -> tuple[dict[str, Any], ...]:
    """Parse git+https direct references from requirement-like text."""

    found: list[dict[str, Any]] = []
    for match in GIT_URL_RE.finditer(text):
        url = match.group("url")
        fragment = match.group("fragment")
        ref = _ref_from_git_url(url)
        mutable = ref is None or ref in MUTABLE_REFS or COMMIT_RE.fullmatch(ref) is None
        found.append(
            {
                "distribution": match.group("dist"),
                "url": url,
                "ref": ref,
                "fragment": fragment,
                "mutable": mutable,
                "span": match.span(),
            }
        )
    return tuple(found)


def mutable_git_references(text: str) -> tuple[dict[str, Any], ...]:
    """Return git direct references that still track a mutable branch or HEAD."""

    return tuple(item for item in parse_git_direct_references(text) if item["mutable"])


def expected_pin_substrings() -> tuple[str, ...]:
    """Exact URL@commit tokens that qualified profiles must use."""

    tokens: list[str] = []
    for pin in PINS.values():
        tokens.append(f"{pin.git_url}@{pin.commit}")
    return tuple(tokens)


def pin_table_mapping() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "pins": {name: pin.to_mapping() for name, pin in PINS.items()},
        "qualified_profile_relpaths": list(QUALIFIED_PROFILE_RELPATHS),
        "runtime_spec_relpaths": list(RUNTIME_SPEC_RELPATHS),
        "live": False,
        "evidence_kind": "measured",
    }
