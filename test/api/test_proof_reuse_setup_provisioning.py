"""Setup-facing proof-reuse provisioning stays explicit and fail-graceful."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from setuptools import Distribution
from setuptools.errors import ExecError

from ipfs_accelerate_py.testing.proof_reuse import provisioning_cli
from ipfs_accelerate_py.testing.proof_reuse.services import (
    DEFAULT_NLTK_DATA_RESOURCES,
)

ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
SETUP_PY = ACCELERATE_ROOT / "setup.py"
PYPROJECT = ACCELERATE_ROOT / "pyproject.toml"


def _setup_namespace() -> tuple[dict[str, Any], dict[str, Any]]:
    with patch("setuptools.setup") as setup:
        namespace = runpy.run_path(str(SETUP_PY))
    assert setup.call_args is not None
    return namespace, dict(setup.call_args.kwargs)


def _resolution(capability: str, *, available: bool) -> Any:
    action = "DEFERRED" if capability == "groth16_native" else "RUN"
    return SimpleNamespace(
        available=available,
        reason_code="available" if available else "capability_missing",
        installed=available,
        action=action,
        to_dict=lambda: {
            "available": available,
            "reason_code": "available" if available else "capability_missing",
            "capability": capability,
            "installed": available,
            "action": action,
            "diagnostics": {},
        },
    )


class _Installer:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls: list[tuple[str, Any]] = []

    def ensure_nltk_data(self, resources: tuple[str, ...]) -> Any:
        self.calls.append(("nltk_data", resources))
        return _resolution("nltk_data", available=self.available)

    def ensure_groth16_native_backend(self) -> Any:
        # No ``consent=True`` argument: the existing environment gates remain
        # authoritative even though invoking this CLI is itself explicit.
        self.calls.append(("groth16_native", None))
        return _resolution("groth16_native", available=self.available)

    def inspect_groth16_runtime(self) -> dict[str, Any]:
        self.calls.append(("inspect_groth16_runtime", None))
        return {
            "ready": self.available,
            "skip_authority": False,
            "trusted_setup_attempted": False,
        }

    def dependency_plan(self) -> dict[str, Any]:
        return {
            "lazy": True,
            "download_during_package_install": False,
            "build_during_package_install": False,
        }


def test_setup_and_project_publish_explicit_commands_and_nltk_metadata() -> None:
    _namespace, metadata = _setup_namespace()
    project = PYPROJECT.read_text(encoding="utf-8")

    assert "nltk>=3.8.1,<4" in metadata["install_requires"]
    assert "nltk>=3.8.1,<4" in metadata["extras_require"]["proof-reuse"]
    assert not any(
        requirement.strip().lower().startswith("groth16")
        for requirement in metadata["install_requires"]
    )
    assert "proof_reuse_provision" in metadata["cmdclass"]
    expected = (
        "ipfs-accelerate-proof-reuse-provision="
        "ipfs_accelerate_py.testing.proof_reuse.provisioning_cli:main"
    )
    assert expected in metadata["entry_points"]["console_scripts"]
    assert (
        'ipfs-accelerate-proof-reuse-provision = '
        '"ipfs_accelerate_py.testing.proof_reuse.provisioning_cli:main"'
    ) in project


def test_setup_optional_metadata_is_safe_without_toml_parser() -> None:
    real_import = __import__

    def import_without_toml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"tomllib", "tomli"}:
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=import_without_toml),
        patch(
            "subprocess.run",
            side_effect=AssertionError("setup metadata must remain process-free"),
        ),
    ):
        namespace, metadata = _setup_namespace()
        assert namespace["_read_optional_deps"](PYPROJECT) == {}

    assert namespace["_read_optional_deps"](Path("missing.toml")) == {}
    assert "proof_reuse_provision" in metadata["cmdclass"]
    assert metadata["extras_require"]["proof-reuse"]
    assert "nltk>=3.8.1,<4" in metadata["extras_require"]["proof-reuse"]


@pytest.mark.parametrize("configured", (None, "", "0", "false", "invalid"))
def test_legacy_setup_torch_install_is_inert_unless_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
) -> None:
    namespace, _metadata = _setup_namespace()
    if configured is None:
        monkeypatch.delenv("IPFS_ACCELERATE_PY_SETUP_AUTO_TORCH", raising=False)
    else:
        monkeypatch.setenv("IPFS_ACCELERATE_PY_SETUP_AUTO_TORCH", configured)

    namespace["_select_torch_install_mode"] = lambda: (_ for _ in ()).throw(
        AssertionError("default setup path must not inspect the accelerator")
    )
    namespace["_run"] = lambda _command: (_ for _ in ()).throw(
        AssertionError("default setup path must not invoke pip")
    )

    namespace["_maybe_install_torch"]()


def test_setup_command_delegates_without_implementing_install_policy() -> None:
    namespace, metadata = _setup_namespace()
    invocations: list[list[str]] = []
    command_type = metadata["cmdclass"]["proof_reuse_provision"]
    command = command_type(Distribution())
    command.run.__globals__["_run"] = (
        lambda invoked: invocations.append(list(invoked)) or 0
    )
    command.nltk_data = True
    command.groth16_native = False
    command.require_ready = False

    command.run()

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation[1:3] == ["-m", provisioning_cli.__name__]
    assert invocation[-1] == "--nltk-data"
    assert "--groth16-native" not in invocation


def test_setup_command_can_require_ready() -> None:
    _namespace, metadata = _setup_namespace()
    command_type = metadata["cmdclass"]["proof_reuse_provision"]
    command = command_type(Distribution())
    command.run.__globals__["_run"] = lambda _command: 2
    command.require_ready = True

    with pytest.raises(ExecError, match="exit 2"):
        command.run()


def test_cli_default_requests_both_capabilities_without_bypassing_consent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installer = _Installer()
    monkeypatch.setattr(
        provisioning_cli,
        "get_default_lazy_dependency_installer",
        lambda: installer,
    )

    returncode = provisioning_cli.main([])

    report = json.loads(capsys.readouterr().out)
    assert returncode == 0
    assert report["ready"] is True
    assert report["trusted_setup_attempted"] is False
    assert installer.calls == [
        ("nltk_data", DEFAULT_NLTK_DATA_RESOURCES),
        ("groth16_native", None),
        ("inspect_groth16_runtime", None),
    ]


@pytest.mark.parametrize(
    "runtime_version",
    ((3, 8, 20), (3, 9, 20), (3, 11, 9)),
)
def test_cli_pre_reviewed_python_floor_is_typed_without_runtime_import(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime_version: tuple[int, int, int],
) -> None:
    def forbidden_runtime_load() -> Any:
        raise AssertionError("unsupported Python must not import the proof runtime")

    monkeypatch.setattr(
        provisioning_cli,
        "get_default_lazy_dependency_installer",
        forbidden_runtime_load,
    )
    monkeypatch.setattr(
        provisioning_cli,
        "_runtime_nltk_policy",
        forbidden_runtime_load,
    )

    returncode = provisioning_cli.main(
        ["--groth16-native", "--require-ready"],
        runtime_version=runtime_version,
    )

    report = json.loads(capsys.readouterr().out)
    assert returncode == 2
    assert report["ready"] is False
    assert report["action"] == "RUN_OR_DEFERRED"
    assert report["trusted_setup_attempted"] is False
    assert report["network_attempted"] is False
    assert report["process_started"] is False
    failure = report["results"]["runtime"]
    assert failure["reason_code"] == "unsupported_python_runtime"
    assert failure["action"] == "RUN_OR_DEFERRED"
    assert failure["diagnostics"] == {
        "required_python": ">=3.12",
        "detected_python": ".".join(str(value) for value in runtime_version),
        "runtime_dependency_import_attempted": False,
    }
    assert report["results"]["groth16_native"]["action"] == "DEFERRED"


def test_cli_pre_reviewed_python_floor_isolated_import_stays_cold() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import sys

        source_root = sys.argv[1]
        sys.path.insert(0, source_root)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            blocked = (
                "ipfs_accelerate_py.testing.proof_reuse.lazy_dependencies",
                "ipfs_accelerate_py.testing.proof_reuse.services",
            )
            if name in blocked or any(name.startswith(item + ".") for item in blocked):
                raise AssertionError("proof runtime import attempted")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        from ipfs_accelerate_py.testing.proof_reuse import provisioning_cli

        raise SystemExit(
            provisioning_cli.main(
                ["--nltk-data", "--require-ready"],
                runtime_version=(3, 9, 19),
            )
        )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ACCELERATE_ROOT)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["results"]["runtime"]["reason_code"] == (
        "unsupported_python_runtime"
    )
    assert report["requested"]["nltk_data"] is True
    assert report["requested"]["groth16_native"] is False


def test_cli_selection_and_require_ready_are_typed_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installer = _Installer(available=False)
    monkeypatch.setattr(
        provisioning_cli,
        "get_default_lazy_dependency_installer",
        lambda: installer,
    )

    returncode = provisioning_cli.main(
        ["--nltk-data", "--nltk-resource", "punkt", "--require-ready"]
    )

    report = json.loads(capsys.readouterr().out)
    assert returncode == 2
    assert report["ready"] is False
    assert report["action"] == "RUN_OR_DEFERRED"
    assert report["results"]["nltk_data"]["reason_code"] == "capability_missing"
    assert installer.calls == [("nltk_data", ("punkt",))]


def test_cli_optional_boundary_exception_never_escapes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _BrokenInstaller(_Installer):
        def ensure_groth16_native_backend(self) -> Any:
            raise OSError("private path must not be emitted")

    installer = _BrokenInstaller()
    monkeypatch.setattr(
        provisioning_cli,
        "get_default_lazy_dependency_installer",
        lambda: installer,
    )

    returncode = provisioning_cli.main(["--groth16-native"])

    output = capsys.readouterr().out
    report = json.loads(output)
    assert returncode == 0
    assert report["ready"] is False
    failure = report["results"]["groth16_native"]
    assert failure["reason_code"] == "provisioner_exception"
    assert failure["action"] == "DEFERRED"
    assert failure["diagnostics"] == {"error_type": "OSError"}
    assert "private path" not in output


def test_importing_setup_does_not_import_or_run_provisioning_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(provisioning_cli, "main", lambda *_args: calls.append(True))

    _setup_namespace()

    assert calls == []
