from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False, stdout=sys.stdout, stderr=sys.stderr).returncode


def _detect_nvidia_cuda_version() -> tuple[int, int] | None:
    """Best-effort detect CUDA version reported by nvidia-smi.

    Returns:
        (major, minor) or None if not detectable.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        out = subprocess.check_output([nvidia_smi], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None

    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _select_torch_install_mode() -> str:
    """Select the 'most recent CUDA-supported' torch install mode.

    Modes:
      - 'skip': do nothing (keep whatever pip/setuptools resolves)
      - 'cpu': force CPU-only torch install (not recommended on NVIDIA hosts)
      - 'cu124': install CUDA 12.4 build
      - 'cu130-nightly': install CUDA 13.0 nightly build (needed for very new GPUs)

    Override with env var:
      IPFS_ACCELERATE_PY_TORCH_MODE=auto|skip|cpu|cu124|cu130-nightly
    """
    mode = os.environ.get("IPFS_ACCELERATE_PY_TORCH_MODE", "auto").strip().lower()
    if mode != "auto":
        return mode

    cuda_ver = _detect_nvidia_cuda_version()
    if not cuda_ver:
        return "skip"

    major, minor = cuda_ver
    # If the driver reports CUDA 13.x, prefer the cu130 nightly wheels.
    # This is particularly important for very new GPUs (e.g. GB10 / sm_121).
    if major >= 13:
        return "cu130-nightly"

    # Otherwise use the newest stable CUDA index we have in-repo.
    if major == 12 and minor >= 4:
        return "cu124"
    if major == 12:
        return "cu124"
    # Fallback: keep default resolution.
    return "skip"


def _maybe_install_torch() -> None:
    """Optionally install CUDA-enabled torch into the current environment.

    IMPORTANT:
      - This only runs for legacy `setup.py install` / `setup.py develop` flows.
      - For normal `pip install .` (PEP517/wheel), setuptools install hooks are not reliable.
        Use the provided helper scripts in `scripts/` for deterministic installs.
    """
    enabled = os.environ.get("IPFS_ACCELERATE_PY_SETUP_AUTO_TORCH", "1").strip() not in {"0", "false", "no"}
    if not enabled:
        return

    mode = _select_torch_install_mode()
    if mode in {"skip", ""}:
        return

    this_directory = Path(__file__).parent

    if mode == "cu130-nightly":
        req = this_directory / "install" / "requirements_torch_cu130_nightly.txt"
        if req.exists():
            _run([sys.executable, "-m", "pip", "install", "-U", "pip"])
            _run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "-r", str(req)])
        return

    if mode == "cu124":
        req = this_directory / "install" / "requirements_torch_cu124.txt"
        if req.exists():
            _run([sys.executable, "-m", "pip", "install", "-U", "pip"])
            _run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "-r", str(req)])
        return

    if mode == "cpu":
        _run([sys.executable, "-m", "pip", "install", "-U", "pip"])
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        return

    # Unknown value: do nothing.
    return


def _get_cmdclass():
    """Attach pip-based torch auto-install to legacy setuptools flows."""
    cmdclass = {}

    try:
        from setuptools.command.install import install as _install

        class install(_install):  # type: ignore
            def run(self):
                _maybe_install_torch()
                super().run()

        cmdclass["install"] = install
    except Exception:
        pass

    try:
        from setuptools.command.develop import develop as _develop

        class develop(_develop):  # type: ignore
            def run(self):
                _maybe_install_torch()
                super().run()

        cmdclass["develop"] = develop
    except Exception:
        pass

    return cmdclass


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
    cmdclass=_get_cmdclass(),
    entry_points={
        "console_scripts": [
            "ipfs_accelerate=ipfs_accelerate_py.ai_inference_cli:main",
            "ipfs-accelerate=ipfs_accelerate_py.cli_entry:main",
        ]
    },
)
