from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements(req_path: Path) -> list[str]:
    if not req_path.exists():
        return []
    requirements: list[str] = []
    for line in req_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(stripped)
    return requirements


def _read_optional_deps(pyproject_path: Path) -> dict[str, list[str]]:
    if not pyproject_path.exists():
        return {}
    try:
        import tomllib  # py3.11+
    except Exception:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    data = tomllib.loads(pyproject_path.read_text())
    return (data.get("project", {}) or {}).get("optional-dependencies", {}) or {}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

install_requires = _read_requirements(this_directory / "requirements.txt")
extras_require = _read_optional_deps(this_directory / "pyproject.toml")

setup(
    name="ipfs_accelerate_py",
    version="0.0.45",
    packages=find_packages(include=["ipfs_accelerate_py", "ipfs_accelerate_py.*"]),
    include_package_data=True,
    description="A comprehensive framework for hardware-accelerated machine learning inference with IPFS network-based distribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Benjamin Barber",
    author_email="starworks5@gmail.com",
    url="https://github.com/endomorphosis/ipfs_accelerate_py",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    keywords="machine learning, IPFS, hardware-acceleration, inference, distributed computing, WebGPU, WebNN",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ipfs_accelerate=ipfs_accelerate_py.ai_inference_cli:main",
            "ipfs-accelerate=ipfs_accelerate_py.cli_entry:main",
        ]
    },
)
