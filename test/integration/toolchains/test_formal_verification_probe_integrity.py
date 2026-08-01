"""Exact probe and managed-artifact identity integrity (FVT-062 / FVT-G202).

``FormalVerificationProbeIntegrity@1``

Covers:

* Java identity is parsed only from the quoted java/openjdk version banner
  after hostile option variables are neutralized;
* bare names resolve only through PATH and dry-run executes nothing;
* Apalache uses ``version``, Isabelle uses ``version``, ProVerif uses a valid
  identity command (``-help``), and nonzero error banners cannot prove
  usability;
* TLC 1.8.0 binds SHA-256 ``e22f8ffb…``, release tag ``v1.8.0``, and revision
  ``30cc360``;
* genuine TLC help is recognized despite exit 1 only with required markers;
* returned launchers execute through the validated Java 17+ runtime;
* TLC and Apalache artifact plus launcher repair is staged, atomic, and
  rollback-safe; failed repair preserves a prior good install.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
STATE_MODEL_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "installers"
    / "state_model.py"
)
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)

INTERFACE = "FormalVerificationProbeIntegrity@1"
GOAL_ID = "FVT-G202"
TASK_ID = "FVT-062"

LOCKED_TLC_VERSION = "1.8.0"
LOCKED_TLC_RELEASE_TAG = "v1.8.0"
LOCKED_TLC_REVISION = "30cc360"
LOCKED_TLC_SHA256 = (
    "e22f8ffb4bacdea0a871f444dd94fe5fb0d8013b3388ae39e82e26f852c735d5"
)
LOCKED_APALACHE_VERSION = "0.58.3"
LOCKED_APALACHE_SHA256 = (
    "ba622db9538aebf942cc7a7815f942a6b2b419012707e16dfdc25a73ff95d0a5"
)

TLC_HELP_OUTPUT = """\
NAME
    TLC - provides model checking and simulation of TLA+ specifications - Version 2026.07.31
SYNOPSIS
    TLC [options] SPEC
DESCRIPTION
    The model checker (TLC) checks or simulates TLA+ specifications.
"""


def _ensure_import_paths() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    _ensure_import_paths()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def state_model():
    _ensure_import_paths()
    from ipfs_datasets_py.logic.backends.installers import state_model as module

    return module


@pytest.fixture(scope="module")
def certifier():
    return _load_module(CERTIFIER_PATH, "certify_formal_verification_toolchains")


@pytest.fixture(scope="module")
def lock_document() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing lock: {LOCK_PATH}"
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _tools_by_id(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry["tool_id"]): entry
        for entry in lock.get("tools") or ()
        if isinstance(entry, dict) and entry.get("tool_id")
    }


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_java(
    path: Path,
    version: str,
    *,
    runtime_exit: int = 0,
    runtime_output: str = "TLC runtime probe",
    reject_option_env: bool = False,
) -> Path:
    option_guard = ""
    if reject_option_env:
        option_guard = (
            'if [ -n "${JAVA_TOOL_OPTIONS+x}" ] || '
            '[ -n "${_JAVA_OPTIONS+x}" ] || '
            '[ -n "${JDK_JAVA_OPTIONS+x}" ]; then\n'
            "  echo 'hostile option env present' >&2\n"
            "  exit 91\n"
            "fi\n"
        )
    return _write_executable(
        path,
        (
            f"{option_guard}"
            'if [ "${1:-}" = "-version" ]; then\n'
            f'  echo \'openjdk version "{version}"\' >&2\n'
            "  exit 0\n"
            "fi\n"
            "cat <<'TLC_OUTPUT'\n"
            f"{runtime_output.rstrip()}\n"
            "TLC_OUTPUT\n"
            f"exit {runtime_exit}"
        ),
    )


# ---------------------------------------------------------------------------
# Expected outputs / lock bindings
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert LOCK_PATH.is_file()
    assert STATE_MODEL_PATH.is_file()
    assert CERTIFIER_PATH.is_file()
    assert Path(__file__).is_file()


def test_interface_constants(state_model, certifier) -> None:
    assert INTERFACE == "FormalVerificationProbeIntegrity@1"
    assert GOAL_ID == "FVT-G202"
    assert TASK_ID == "FVT-062"
    assert state_model.TLC_VERSION == LOCKED_TLC_VERSION
    assert state_model.TLC_RELEASE_TAG == LOCKED_TLC_RELEASE_TAG
    assert state_model.TLC_REVISION == LOCKED_TLC_REVISION
    assert state_model.TLC_SHA256 == LOCKED_TLC_SHA256
    assert state_model.APALACHE_VERSION == LOCKED_APALACHE_VERSION
    assert state_model.APALACHE_SHA256 == LOCKED_APALACHE_SHA256
    assert "JAVA_TOOL_OPTIONS" in state_model.JAVA_OPTION_ENV_VARS
    assert "_JAVA_OPTIONS" in certifier.JAVA_OPTION_ENV_VARS


def test_lock_binds_exact_probe_commands_and_tlc_identity(
    lock_document: dict[str, Any],
) -> None:
    tools = _tools_by_id(lock_document)
    inventory = lock_document["checksummed_release_inventory"]["tlc"]

    assert inventory["sha256"] == LOCKED_TLC_SHA256
    assert inventory["release_tag"] == LOCKED_TLC_RELEASE_TAG
    assert inventory["revision"] == LOCKED_TLC_REVISION

    tlc = tools["tlc"]
    probe = tlc["offline_probe"]
    assert probe["argv"] == ["-help"]
    assert set(probe["accepted_returncodes"]) == {0, 1}
    assert probe["artifact_sha256"] == LOCKED_TLC_SHA256
    assert probe["release_tag"] == LOCKED_TLC_RELEASE_TAG
    assert probe["revision"] == LOCKED_TLC_REVISION
    assert tlc["pins"][0]["sha256"] == LOCKED_TLC_SHA256
    assert tlc["pins"][0]["revision"] == LOCKED_TLC_REVISION
    assert tlc["pins"][0]["release_tag"] == LOCKED_TLC_RELEASE_TAG
    assert tlc["deployment_contract"]["release_tag"] == LOCKED_TLC_RELEASE_TAG
    assert tlc["deployment_contract"]["revision"] == LOCKED_TLC_REVISION

    assert tools["apalache"]["offline_probe"]["argv"] == ["version"]
    assert tools["apalache"]["offline_probe"]["artifact_sha256"] == (
        LOCKED_APALACHE_SHA256
    )
    assert tools["isabelle"]["offline_probe"]["argv"] == ["version"]
    assert tools["proverif"]["offline_probe"]["argv"] == ["-help"]
    assert tools["java"]["offline_probe"]["argv"] == ["-version"]


# ---------------------------------------------------------------------------
# Java identity: quoted banner only + hostile env neutralization
# ---------------------------------------------------------------------------


def test_java_major_version_parses_only_quoted_banner(state_model, certifier) -> None:
    hostile = (
        "Picked up JAVA_TOOL_OPTIONS: -Dreview.marker=17.0.0\n"
        'openjdk version "1.8.0_482"\n'
        "OpenJDK Runtime Environment (build 1.8.0_482-b08)\n"
    )
    assert state_model.java_major_version(hostile) == 8
    assert certifier.java_major_version(hostile) == 8
    assert state_model.java_major_version('java version "17.0.12"') == 17
    assert state_model.java_major_version("17.0.12 without quotes") is None
    assert certifier.parse_java_version_banner(
        'Picked up _JAVA_OPTIONS: x\njava version "21.0.1"\n'
    ) == "21.0.1"
    assert certifier.parse_java_version_banner("not a java banner") is None


@pytest.mark.parametrize(
    "variable",
    ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS"),
)
def test_java_probe_neutralizes_hostile_option_variables(
    variable: str,
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    java = _fake_java(
        tmp_path / "java",
        "17.0.12",
        reject_option_env=True,
    )
    monkeypatch.setenv(variable, "-Dreview.marker=99.0")

    probe = state_model.probe_java_runtime(
        java_executable=java,
        minimum_major=17,
    )

    assert probe.usable is True
    assert probe.major == 17
    assert probe.reason_code is None


def test_certifier_java_probe_rejects_unquoted_banner_and_strips_hostile_env(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    java = _write_executable(
        tmp_path / "java",
        (
            'if [ -n "${JAVA_TOOL_OPTIONS+x}" ]; then\n'
            "  echo 'Picked up JAVA_TOOL_OPTIONS: -Dleak=1' >&2\n"
            "  exit 0\n"
            "fi\n"
            'echo \'openjdk version "17.0.12"\' >&2\n'
            "exit 0\n"
        ),
    )
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Dhostile=1")
    monkeypatch.setattr(
        certifier,
        "resolve_executable",
        lambda _candidates: str(java),
    )

    result = certifier.probe_tool_identity(
        {
            "tool_id": "java",
            "availability": "host_support",
            "executable_candidates": ["java"],
            "offline_probe": {"argv": ["-version"]},
        },
        env=certifier.offline_env(
            {
                "PATH": str(tmp_path),
                "JAVA_TOOL_OPTIONS": "-Dhostile=1",
            }
        ),
    )

    assert result["identity_probed"] is True
    assert result["java_major"] == 17
    assert 'version "17.0.12"' in result["version_string"]

    monkeypatch.setattr(
        certifier,
        "bounded_run",
        lambda argv, **_kwargs: certifier.subprocess.CompletedProcess(
            argv,
            0,
            stdout="Picked up JAVA_TOOL_OPTIONS: -Dreview.marker=17.0\n",
            stderr="",
        ),
    )
    forged = certifier.probe_tool_identity(
        {
            "tool_id": "java",
            "availability": "host_support",
            "executable_candidates": ["java"],
            "offline_probe": {"argv": ["-version"]},
        },
        env=certifier.offline_env({"PATH": str(tmp_path)}),
    )
    assert forged["identity_probed"] is False
    assert forged["probe_error"] == "java_version_banner_unreadable"


# ---------------------------------------------------------------------------
# PATH resolution + dry-run executes nothing
# ---------------------------------------------------------------------------


def test_bare_names_resolve_only_through_path(
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local = _fake_java(tmp_path / "java", "21.0.1")
    monkeypatch.chdir(tmp_path)

    assert state_model.which_executable("java", path_env="/missing") is None
    assert state_model.which_executable("./java", path_env="/missing") == str(
        local.resolve()
    )
    assert state_model.which_executable(
        "java",
        path_env=str(tmp_path),
    ) == str(local.resolve())


def test_dry_run_executes_nothing(
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"
    java = _write_executable(
        tmp_path / "java",
        f"touch '{marker}'\nexit 99",
    )
    monkeypatch.setattr(
        state_model,
        "which_executable",
        lambda *args, **kwargs: pytest.fail("dry-run resolved an executable"),
    )
    monkeypatch.setattr(
        state_model,
        "probe_java_runtime",
        lambda *args, **kwargs: pytest.fail("dry-run probed the JVM"),
    )
    monkeypatch.setattr(
        state_model,
        "download_artifact",
        lambda *args, **kwargs: pytest.fail("dry-run downloaded an artifact"),
    )

    receipt = state_model.ensure_state_model_portfolio(
        yes=True,
        strict=True,
        dry_run=True,
        install_root=tmp_path / "install",
        java_executable=java,
    )

    assert receipt["tlc"]["phase"] == "dry_run"
    assert receipt["apalache"]["phase"] == "dry_run"
    assert receipt["java_runtime"]["reason_code"] == "dry_run_not_probed"
    assert not marker.exists()


# ---------------------------------------------------------------------------
# Nonzero error banners are not identity
# ---------------------------------------------------------------------------


def test_nonzero_error_banner_cannot_prove_usability(
    certifier,
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        certifier,
        "resolve_executable",
        lambda _candidates: "/bin/false",
    )
    monkeypatch.setattr(
        certifier,
        "bounded_run",
        lambda *_args, **_kwargs: certifier.subprocess.CompletedProcess(
            ["/bin/false"],
            42,
            "",
            "fatal: not a version 9.9.9\n",
        ),
    )
    result = certifier.probe_tool_identity(
        {
            "tool_id": "proverif",
            "availability": "managed_pin",
            "executable_candidates": ["proverif"],
            "offline_probe": {"argv": ["-help"]},
        },
        env={},
    )
    assert result["identity_probed"] is False
    assert result["installed"] is False
    assert result["probe_error"] == "identity_probe_nonzero:42"

    java17 = _fake_java(tmp_path / "java17" / "java", "17.0.12")
    apalache = _write_executable(
        tmp_path / "apalache-mc",
        'echo "Apalache 0.58.3" >&2\nexit 1',
    )
    probe = state_model.probe_apalache_runtime(
        str(apalache),
        java_executable=java17,
    )
    assert probe.usable is False
    assert probe.returncode == 1
    assert probe.reason_code == "runtime_probe_nonzero_exit"


# ---------------------------------------------------------------------------
# TLC help semantics + digest / revision binding
# ---------------------------------------------------------------------------


def test_tlc_help_accepts_exit_one_only_with_required_markers(
    state_model,
    tmp_path: Path,
) -> None:
    java17 = _fake_java(
        tmp_path / "java17" / "java",
        "17.0.12",
        runtime_exit=1,
        runtime_output=TLC_HELP_OUTPUT,
    )
    jar = tmp_path / state_model.TLC_JAR_NAME
    jar.write_bytes(b"fixture-identity-is-verified-separately")

    accepted = state_model.probe_tlc_runtime(
        jar_path=jar,
        java_executable=java17,
    )
    assert accepted.usable is True
    assert accepted.returncode == 1

    incomplete = _fake_java(
        tmp_path / "java17b" / "java",
        "17.0.12",
        runtime_exit=1,
        runtime_output="TLC 1.8.0 help without markers\n",
    )
    rejected = state_model.probe_tlc_runtime(
        jar_path=jar,
        java_executable=incomplete,
    )
    assert rejected.usable is False
    assert rejected.reason_code == "tlc_help_semantics_missing"


def test_tlc_certifier_probe_binds_digest_tag_and_revision(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
    lock_document: dict[str, Any],
) -> None:
    entry = _tools_by_id(lock_document)["tlc"]
    monkeypatch.setattr(
        certifier,
        "resolve_executable",
        lambda _candidates: "/managed/bin/tlc",
    )
    monkeypatch.setattr(
        certifier,
        "bounded_run",
        lambda argv, **_kwargs: certifier.subprocess.CompletedProcess(
            argv,
            1,
            stdout=TLC_HELP_OUTPUT,
            stderr="",
        ),
    )
    monkeypatch.setattr(
        certifier,
        "_managed_state_model_identity",
        lambda *_args, **_kwargs: {
            "usable": True,
            "artifact_sha256": LOCKED_TLC_SHA256,
            "release_tag": LOCKED_TLC_RELEASE_TAG,
            "revision": LOCKED_TLC_REVISION,
            "artifact_digest_verified": True,
            "payload_digest_verified": True,
            "launchers_structurally_valid": True,
            "manifest_valid": True,
        },
    )

    result = certifier.probe_tool_identity(
        entry,
        env=certifier.offline_env({"PATH": "/managed/bin"}),
    )
    assert result["identity_probed"] is True
    assert LOCKED_TLC_SHA256 in result["version_string"]
    assert LOCKED_TLC_REVISION in result["version_string"]
    assert LOCKED_TLC_RELEASE_TAG in result["version_string"]

    mismatched = json.loads(json.dumps(entry))
    mismatched["offline_probe"]["revision"] = "deadbeef"
    bad = certifier.probe_tool_identity(
        mismatched,
        env=certifier.offline_env({"PATH": "/managed/bin"}),
    )
    assert bad["identity_probed"] is False
    assert bad["probe_error"] == "tlc_help_or_managed_digest_identity_failed"


def test_tlc_banner_only_without_help_markers_is_not_identity(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
    lock_document: dict[str, Any],
) -> None:
    entry = _tools_by_id(lock_document)["tlc"]
    monkeypatch.setattr(
        certifier,
        "resolve_executable",
        lambda _candidates: "/managed/bin/tlc",
    )
    monkeypatch.setattr(
        certifier,
        "_managed_state_model_identity",
        lambda *_args, **_kwargs: {
            "usable": True,
            "artifact_sha256": LOCKED_TLC_SHA256,
            "release_tag": LOCKED_TLC_RELEASE_TAG,
            "revision": LOCKED_TLC_REVISION,
        },
    )
    monkeypatch.setattr(
        certifier,
        "bounded_run",
        lambda argv, **_kwargs: certifier.subprocess.CompletedProcess(
            argv,
            0,
            stdout="TLC 1.8.0\n",
            stderr="",
        ),
    )
    result = certifier.probe_tool_identity(
        entry,
        env=certifier.offline_env({"PATH": "/managed/bin"}),
    )
    assert result["identity_probed"] is False
    assert result["probe_error"] == "tlc_help_or_managed_digest_identity_failed"


# ---------------------------------------------------------------------------
# Launchers bind validated Java; repair is atomic and rollback-safe
# ---------------------------------------------------------------------------


def test_successful_tlc_install_binds_java17_and_revision_manifest(
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    java17 = _fake_java(
        tmp_path / "java17" / "java",
        "17.0.12",
        runtime_exit=1,
        runtime_output=TLC_HELP_OUTPUT,
    )
    install_root = tmp_path / "install"

    def fake_download(url: str, destination: Path, **kwargs: object):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"reviewed-tlc-fixture")
        return True, state_model.TLC_SHA256

    monkeypatch.setattr(state_model, "download_artifact", fake_download)
    monkeypatch.setattr(state_model, "verify_sha256", lambda *args: True)
    monkeypatch.setattr(
        state_model,
        "authorize_plugin_install",
        lambda *args, **kwargs: None,
    )

    receipt = state_model.ensure_tlc(
        yes=True,
        strict=False,
        force=True,
        install_root=install_root,
        java_executable=java17,
        test_mode=True,
    )

    launcher = install_root / "bin" / state_model.TLC_EXECUTABLE
    assert receipt.status == "installed"
    assert receipt.bindings["release_tag"] == LOCKED_TLC_RELEASE_TAG
    assert receipt.bindings["revision"] == LOCKED_TLC_REVISION
    assert f"exec '{java17.resolve()}' -cp " in launcher.read_text(
        encoding="utf-8"
    )
    identity = state_model.managed_tlc_identity(
        install_root,
        java_executable=java17,
    )
    assert identity["usable"] is True
    assert identity["release_tag"] == LOCKED_TLC_RELEASE_TAG
    assert identity["revision"] == LOCKED_TLC_REVISION
    assert identity["artifact_sha256"] == LOCKED_TLC_SHA256
    manifest = json.loads(
        Path(identity["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["release_tag"] == LOCKED_TLC_RELEASE_TAG
    assert manifest["revision"] == LOCKED_TLC_REVISION
    assert manifest["artifact_sha256"] == LOCKED_TLC_SHA256


def test_tlc_failed_runtime_validation_preserves_prior_good_install(
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    java17 = _fake_java(
        tmp_path / "java17" / "java",
        "17.0.12",
        runtime_exit=1,
    )
    install_root = tmp_path / "install"
    final_jar = (
        install_root
        / "tlc"
        / state_model.TLC_VERSION
        / state_model.TLC_JAR_NAME
    )
    final_jar.parent.mkdir(parents=True)
    final_jar.write_bytes(b"previous-valid-install")
    old_launcher = _write_executable(
        install_root / "bin" / state_model.TLC_EXECUTABLE,
        'echo "previous launcher"',
    )
    previous_launcher = old_launcher.read_bytes()
    previous_jar = final_jar.read_bytes()

    def fake_download(url: str, destination: Path, **kwargs: object):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"not-a-real-jar")
        return True, state_model.TLC_SHA256

    monkeypatch.setattr(state_model, "download_artifact", fake_download)
    monkeypatch.setattr(state_model, "verify_sha256", lambda *args: True)
    monkeypatch.setattr(
        state_model,
        "authorize_plugin_install",
        lambda *args, **kwargs: None,
    )

    receipt = state_model.ensure_tlc(
        yes=True,
        strict=False,
        force=True,
        install_root=install_root,
        java_executable=java17,
        test_mode=True,
    )

    assert receipt.status == "failed"
    assert receipt.phase == "runtime_validation"
    assert "post_install_usability_failed" in receipt.reason_codes
    assert final_jar.read_bytes() == previous_jar
    assert old_launcher.read_bytes() == previous_launcher


def test_atomic_publication_restores_prior_files_on_failure(
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "final" / "first"
    second = tmp_path / "final" / "second"
    staged_first = tmp_path / "staged" / "first"
    staged_second = tmp_path / "staged" / "second"
    for path, value in (
        (first, b"old-first"),
        (second, b"old-second"),
        (staged_first, b"new-first"),
        (staged_second, b"new-second"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    original_replace = Path.replace

    def fail_second(source: Path, target: Path):
        if source == staged_second:
            raise OSError("injected publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second)

    with pytest.raises(OSError, match="injected"):
        state_model._commit_staged_files(
            ((staged_first, first), (staged_second, second)),
            backup_dir=tmp_path / "backups",
        )

    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"


def test_apalache_publication_failure_restores_previous_install(
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    java17 = _fake_java(tmp_path / "java17" / "java", "17.0.12")
    install_root = tmp_path / "install"
    previous_install = install_root / f"apalache-{state_model.APALACHE_VERSION}"
    previous_install.mkdir(parents=True)
    sentinel = previous_install / "previous-release"
    sentinel.write_text("preserve me", encoding="utf-8")
    previous_launcher = _write_executable(
        install_root / "bin" / state_model.APALACHE_EXECUTABLE,
        'echo "previous launcher"',
    )
    previous_launcher_bytes = previous_launcher.read_bytes()

    def fake_download(url: str, destination: Path, **kwargs: object):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"reviewed-archive")
        return True, state_model.APALACHE_SHA256

    def fake_extract(archive: Path, destination: Path) -> None:
        _write_executable(
            destination / "apalache-0.58.3" / "bin" / "apalache-mc",
            'echo "Apalache 0.58.3"',
        )

    monkeypatch.setattr(state_model, "download_artifact", fake_download)
    monkeypatch.setattr(state_model, "verify_sha256", lambda *args: True)
    monkeypatch.setattr(state_model, "_safe_extract_tar", fake_extract)
    monkeypatch.setattr(
        state_model,
        "authorize_plugin_install",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        state_model,
        "_commit_staged_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("injected launcher publication failure")
        ),
    )

    with pytest.raises(OSError, match="injected launcher"):
        state_model.ensure_apalache(
            yes=True,
            strict=False,
            force=True,
            install_root=install_root,
            java_executable=java17,
            test_mode=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert previous_launcher.read_bytes() == previous_launcher_bytes


def test_ensure_apalache_blocks_java_below_17(
    state_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    java8 = _fake_java(tmp_path / "java8" / "java", "1.8.0_482")
    monkeypatch.setattr(
        state_model,
        "download_artifact",
        lambda *args, **kwargs: pytest.fail("download must not be attempted"),
    )

    receipt = state_model.ensure_apalache(
        yes=True,
        strict=False,
        force=True,
        install_root=tmp_path / "install",
        java_executable=java8,
        test_mode=True,
    )

    assert receipt.status == "blocked"
    assert receipt.phase == "java_support"
    assert "java_version_unsupported" in receipt.reason_codes
    assert receipt.bindings["minimum_java_major"] == 17


def test_managed_tlc_identity_requires_exact_digest_and_revision_manifest(
    state_model,
    tmp_path: Path,
) -> None:
    java17 = _fake_java(tmp_path / "java17" / "java", "17.0.12")
    root = tmp_path / "install"
    jar = root / "tlc" / state_model.TLC_VERSION / state_model.TLC_JAR_NAME
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"wrong-digest-bytes")
    identity = state_model.managed_tlc_identity(root, java_executable=java17)
    assert identity["artifact_digest_verified"] is False
    assert identity["usable"] is False
    assert identity["revision"] == LOCKED_TLC_REVISION
    assert identity["release_tag"] == LOCKED_TLC_RELEASE_TAG
