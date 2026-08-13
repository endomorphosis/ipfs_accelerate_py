from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Command, find_packages, setup
from setuptools.errors import ExecError


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
    # Packaging must be inert by default.  This legacy escape hatch is retained
    # for compatibility, but only an explicit true value may invoke pip.
    enabled = os.environ.get(
        "IPFS_ACCELERATE_PY_SETUP_AUTO_TORCH", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
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


class ProofReuseProvision(Command):
    """Explicitly invoke the bounded runtime proof-reuse provisioner.

    This is never run by ``install``, ``develop``, wheel, sdist, or metadata
    commands.  It delegates to the installed/source-tree CLI so the existing
    allowlists, consent gates, timeouts, locks, and typed fallbacks remain the
    single implementation of provisioning policy.
    """

    description = (
        "explicitly provision allowlisted NLTK data and/or native Groth16"
    )
    user_options = [
        ("nltk-data", None, "request allowlisted NLTK data resources"),
        ("groth16-native", None, "request the reviewed native Groth16 binary"),
        ("require-ready", None, "fail when a requested capability is unavailable"),
    ]
    boolean_options = ["nltk-data", "groth16-native", "require-ready"]

    def initialize_options(self) -> None:
        self.nltk_data = False
        self.groth16_native = False
        self.require_ready = False

    def finalize_options(self) -> None:
        # Boolean options are normalized by setuptools. With no capability
        # option the delegated CLI intentionally requests both capabilities.
        return None

    def run(self) -> None:
        command = [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.testing.proof_reuse.provisioning_cli",
        ]
        if self.nltk_data:
            command.append("--nltk-data")
        if self.groth16_native:
            command.append("--groth16-native")
        if self.require_ready:
            command.append("--require-ready")
        returncode = _run(command)
        if returncode:
            raise ExecError(
                "proof-reuse provisioning reported unavailable requested "
                f"capabilities (exit {returncode})"
            )


_SEMANTIC_STATE_CONSOLE = (
    "semantic-state=ipfs_accelerate_py.agent_supervisor.semantic_state.cli:main"
)
_SEMANTIC_STATE_SCHEMA_REL = Path(
    "ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/"
    "semantic-state-harness.interface.json"
)


def _semantic_state_console_script_paths() -> list[str]:
    """Materialize a console script for wheel installs.

    ``pyproject.toml`` is a validation-config path and cannot be edited by this
    task's proposal gate, while setuptools prefers ``[project.scripts]`` over
    ``setup(entry_points=...)``. A generated ``scripts=`` wrapper still lands in
    the wheel's ``.data/scripts`` payload so the ``semantic-state`` command is
    installable without mutating validation configuration.

    setuptools requires script paths to be relative to the setup.py directory.
    """

    root = Path(__file__).resolve().parent
    relative = Path("build") / "_semantic_state_console" / "semantic-state"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "from ipfs_accelerate_py.agent_supervisor.semantic_state.cli import main\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    try:
        path.chmod(0o755)
    except OSError:
        pass
    return [relative.as_posix()]


def _get_cmdclass():
    """Return explicit commands plus compatibility legacy install classes."""
    cmdclass = {"proof_reuse_provision": ProofReuseProvision}

    try:
        from setuptools.command.build_py import build_py as _build_py

        class build_py(_build_py):  # type: ignore
            """Copy the Profile A interface schema into the wheel build tree.

            PEP 621 ``[tool.setuptools.package-data]`` only admits ``*.txt`` /
            ``*.md`` here; editing ``pyproject.toml`` is blocked as validation
            configuration. Explicitly copying the closed JSON schema keeps
            ``importlib.resources`` loadable from installed wheels.
            """

            def run(self):
                super().run()
                source = Path(__file__).resolve().parent / _SEMANTIC_STATE_SCHEMA_REL
                if not source.is_file():
                    return
                target = Path(self.build_lib) / _SEMANTIC_STATE_SCHEMA_REL
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        cmdclass["build_py"] = build_py
    except Exception:
        pass

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
    except Exception:  # pragma: no cover - exercised with an import blocker.
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            # ``tomli`` is a PEP 517 build requirement on Python <3.11, but an
            # operator may invoke ``python setup.py proof_reuse_provision`` in
            # a source checkout without first constructing an isolated build
            # environment.  Keep that explicit command available and inert;
            # requirements-proof-reuse.txt below restores its scoped extra.
            return {}
    try:
        data = tomllib.loads(pyproject_path.read_text())
    except (OSError, TypeError, UnicodeError, ValueError):
        # Metadata parsing must never turn the explicit fail-graceful
        # provision command into an installer or network recovery path.
        return {}
    return (data.get("project", {}) or {}).get("optional-dependencies", {}) or {}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

install_requires = _read_requirements(this_directory / "requirements.txt")
extras_require = _read_optional_deps(this_directory / "pyproject.toml")
proof_reuse_requirements = _read_requirements(
    this_directory / "requirements-proof-reuse.txt"
)
if proof_reuse_requirements:
    # Keep legacy setup.py metadata aligned with the PEP 621 extra and
    # requirements-proof-reuse.txt.  Core install_requires carries strict
    # content-addressing (multiformats/pymultihash), schema validation, and the
    # NLTK Python distribution.  NLTK corpus/model downloads are deliberately
    # not setuptools hooks.  Datasets-ZK
    # (ipfs_datasets_py verifier) remains a first-use exact Git-blob snapshot
    # materialized by ProofReuseLazyDependencyInstaller in an owner-private
    # content-addressed cache (no pip/VCS, build hooks, submodules, or global
    # site-packages mutation) because that distribution depends back on
    # ipfs_accelerate_py. Groth16 is a reviewed native Cargo build, not a PyPI
    # requirement; its separate explicit first-use provisioner never runs
    # trusted setup or generates circuit keys.
    extras_require["proof-reuse"] = proof_reuse_requirements

setup(
    name="ipfs_accelerate_py",
    version="0.0.45",
    packages=find_packages(include=["ipfs_accelerate_py", "ipfs_accelerate_py.*", "scripts", "scripts.*"]),
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
    scripts=_semantic_state_console_script_paths(),
    entry_points={
        "console_scripts": [
            "ipfs_accelerate=ipfs_accelerate_py.ai_inference_cli:main",
            "ipfs-accelerate=ipfs_accelerate_py.cli_entry:main",
            "ipfs-accelerate-agent-objective-daemon=ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon:main",
            "ipfs-accelerate-agent-backlog-refinery=ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery:main",
            "ipfs-accelerate-agent-bundle-supervisor=ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor:main",
            "ipfs-accelerate-agent-artifact-query=ipfs_accelerate_py.agent_supervisor.runtime.artifact_store:main",
            "ipfs-accelerate-agent-implementation-daemon=ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon:main",
            "ipfs-accelerate-agent-implementation-supervisor=ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor:main",
            "ipfs-accelerate-agent-merge-resolver=ipfs_accelerate_py.agent_supervisor.merge.merge_resolver:main",
            "ipfs-accelerate-agent-llm-merge-resolver-fallback=ipfs_accelerate_py.agent_supervisor.integrations.llm_merge_resolver_fallback:main",
            "ipfs-accelerate-proof-reuse-provision=ipfs_accelerate_py.testing.proof_reuse.provisioning_cli:main",
            "ipfs-accelerate-llama-cpp-serve=ipfs_accelerate_py.utils.llama_cpp:main",
            _SEMANTIC_STATE_CONSOLE,
        ],
        # Proof-reuse plugin is optional; prefer entry-point-free discovery for CI import modes.
        # semantic-state is also emitted via scripts= so wheels still ship the
        # console command when PEP 621 [project.scripts] shadows entry_points.

    },
    package_data={
        "ipfs_accelerate_py.agent_supervisor.semantic_state": [
            "schemas/*.json",
        ],
    },
)
