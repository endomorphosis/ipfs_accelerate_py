#!/opt/ipfs-task-tools/bin/python
"""Qualify the ordinary worker's mounted Python toolchain, inside its container.

Run with ``python -I -S -B /opt/qualify_runtime.py``. This script does not call
Docker, providers, or the network, and makes no Lean qualification claim.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


APPROVED_ROOT = Path("/opt/ipfs-validation-site-packages")
WORKSPACE = Path("/workspace")
PYTHON = Path("/opt/ipfs-task-tools/bin/python")
REPORT_PATH = WORKSPACE / "runtime-report.json"
INSTALLED_MANIFEST = Path("/opt/installed-files.json")
EXPECTED_WORKER_UID = 1000
WHEEL_LOCKS = {
    "arm64": "eed4f3a61cdba4930473acd3c4f28debe24966218cb74506fcd38fd6ec5f82d8",
    "amd64": "4691b6e4c25f6e11f57c0434bc794595912325edfdff3d5551b1ee4295316f1a",
}
EXPECTED_WHEEL_LOCK_SHA256 = WHEEL_LOCKS["arm64"]
EXPECTED_ARCHITECTURE = "arm64"
AMD64_RIPGREP_SHA256 = "be2fbcdd985161603bfa268c7647eec670b9e94374f0e972da708d7de670a8fb"
# Taken from the adjacent, source-reviewed wheels.lock.json. Do not discover
# expected versions from the installed packages being qualified.
EXPECTED_VERSIONS = {
    "pytest": "9.0.3",
    "pluggy": "1.6.0",
    "packaging": "26.2",
    "iniconfig": "2.3.0",
    "pygments": "2.19.2",
    "cryptography": "49.0.0",
    "cffi": "2.1.0",
    "pycparser": "3.0",
    "duckdb": "1.5.2",
}
IMPORT_NAMES = (
    "pytest", "pluggy", "packaging", "iniconfig", "pygments",
    "cryptography", "cffi", "pycparser", "duckdb",
)


class QualificationFailure(Exception):
    """A bounded, authored failure code, suitable for the public report."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise QualificationFailure(code)


def inside_approved(path: str | os.PathLike[str]) -> bool:
    return Path(path).resolve().is_relative_to(APPROVED_ROOT.resolve())


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def environment() -> dict[str, str]:
    # Do not inherit provider credentials, host plugin settings, or site paths.
    # -I ignores PYTHONPATH; the child explicitly inserts this same approved root.
    return {
        "PATH": (
            "/opt/ipfs-task-tools/bin:/usr/bin:/bin"
            if EXPECTED_ARCHITECTURE == "amd64" else "/usr/bin:/bin"
        ),
        "HOME": str(WORKSPACE),
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(APPROVED_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }


def run_command(argv: list[str], *, timeout: int) -> dict[str, object]:
    """Capture only command metadata, never command output or environment values."""
    with tempfile.TemporaryFile(dir="/tmp") as output:
        completed = subprocess.run(
            argv, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, env=environment(), cwd=WORKSPACE,
            timeout=timeout, check=False,
        )
        output.seek(0, os.SEEK_END)
        size = output.tell()
        require(size <= 1_048_576, "subprocess_output_limit_exceeded")
        output.seek(0)
        digest = hashlib.sha256(output.read()).hexdigest()
    return {
        "returncode": completed.returncode,
        "output_bytes": size,
        "output_sha256": digest,
    }


def check_identity_and_isolation() -> dict[str, object]:
    require(os.getuid() != 0 and os.geteuid() != 0, "root_execution_forbidden")
    require(os.getuid() == EXPECTED_WORKER_UID and os.geteuid() == EXPECTED_WORKER_UID, "unexpected_worker_uid")
    require(sys.version_info[:2] == (3, 12), "unexpected_python_version")
    require(bool(sys.flags.isolated), "python_isolation_required")
    require(bool(sys.flags.no_site), "python_no_site_required")
    require(bool(sys.flags.dont_write_bytecode), "python_no_bytecode_required")
    require(Path(sys.executable) == PYTHON, "unexpected_python_executable")
    require(APPROVED_ROOT.is_dir(), "approved_packages_missing")
    require(WORKSPACE.is_dir(), "workspace_missing")
    return {"uid": os.getuid(), "python_version": ".".join(map(str, sys.version_info[:3]))}


def check_package_closure() -> dict[str, object]:
    installed_files = check_installed_package_files()
    found: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(APPROVED_ROOT)]):
        raw_name = distribution.metadata.get("Name")
        require(bool(raw_name), "package_name_missing")
        name = canonical_name(str(raw_name))
        require(name not in found, "duplicate_package_distribution")
        require(inside_approved(distribution.locate_file("")), "distribution_outside_approved_root")
        found[name] = distribution.version
    require(found == EXPECTED_VERSIONS, "installed_package_versions_do_not_match_lock")
    sys.path.insert(0, str(APPROVED_ROOT))
    origins: dict[str, str] = {}
    for name in IMPORT_NAMES:
        spec = importlib.util.find_spec(name)
        require(spec is not None and bool(spec.origin), "package_import_spec_missing")
        require(inside_approved(str(spec.origin)), "package_spec_outside_approved_root")
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        require(bool(origin) and inside_approved(str(origin)), "package_import_outside_approved_root")
        origins[name] = str(Path(str(origin)).resolve())
    # These modules exercise the installed native extension closure rather than
    # merely accepting Python package metadata.
    for name in ("_cffi_backend", "cryptography.hazmat.bindings._rust", "_duckdb"):
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        require(bool(origin) and inside_approved(str(origin)), "native_import_outside_approved_root")
        origins[name] = str(Path(str(origin)).resolve())
    return {"installed_files": installed_files, "versions": found, "import_origins": origins}


def check_installed_package_files() -> dict[str, object]:
    """Verify the complete package tree before importing any approved package.

    The separately frozen, read-only installed manifest is supplied by the
    builder. Its generated tool/profile entries are outside this check; the
    Python placeholder is intentionally overlaid by the trusted host binary.
    """
    require(not INSTALLED_MANIFEST.is_symlink() and INSTALLED_MANIFEST.is_file(), "installed_manifest_missing_or_symlinked")
    require(INSTALLED_MANIFEST.stat().st_size <= 4 * 1024**2, "installed_manifest_size_limit_exceeded")
    raw = INSTALLED_MANIFEST.read_bytes()

    def unique_objects(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            require(key not in result, "installed_manifest_duplicate_key")
            result[key] = value
        return result

    manifest = json.loads(raw, object_pairs_hook=unique_objects)
    require(isinstance(manifest, dict), "installed_manifest_not_object")
    require(manifest.get("schema") == "legal-ir-ordinary-worker-installed-files@1", "installed_manifest_schema_mismatch")
    require(manifest.get("wheel_lock_sha256") == EXPECTED_WHEEL_LOCK_SHA256, "installed_manifest_lock_mismatch")
    files = manifest.get("files")
    require(isinstance(files, dict), "installed_manifest_files_missing")
    prefix = "opt/ipfs-validation-site-packages/"
    expected: dict[str, dict[str, object]] = {}
    expected_directories: set[str] = set()
    for path, item in files.items():
        require(isinstance(path, str), "installed_manifest_path_not_string")
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix):]
        parts = relative.split("/")
        require(bool(relative) and all(part not in ("", ".", "..") for part in parts), "installed_manifest_package_path_invalid")
        require("\\" not in relative and str(PurePosixPath(relative)) == relative, "installed_manifest_package_path_noncanonical")
        require(isinstance(item, dict) and item.get("kind") == "file", "installed_manifest_package_not_regular_file")
        require(type(item.get("bytes")) is int and 0 <= item["bytes"] <= 128 * 1024**2, "installed_manifest_file_size_invalid")
        require(isinstance(item.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None, "installed_manifest_file_digest_invalid")
        expected[relative] = item
        expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    require(0 < len(expected) <= 10_000, "installed_manifest_package_population_invalid")
    require(not APPROVED_ROOT.is_symlink(), "approved_package_root_symlinked")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    total_bytes = 0
    for directory, directories, filenames in os.walk(APPROVED_ROOT, followlinks=False):
        for name in [*directories, *filenames]:
            path = Path(directory) / name
            relative = path.relative_to(APPROVED_ROOT).as_posix()
            parts = relative.split("/")
            require(not any(part in ("sitecustomize", "usercustomize") for part in parts), "python_startup_hook_not_admitted")
            require(not name.endswith(".pth") and not name.startswith(("sitecustomize.", "usercustomize.")), "python_startup_hook_not_admitted")
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), "approved_package_tree_symlinked")
            if stat.S_ISDIR(mode):
                actual_directories.add(relative)
                require(relative in expected_directories, "unmanifested_package_directory")
                continue
            require(stat.S_ISREG(mode), "approved_package_tree_nonregular_file")
            require(relative in expected, "unmanifested_package_file")
            actual_files.add(relative)
            require(len(actual_files) <= 10_000, "installed_package_population_limit_exceeded")
            item = expected[relative]
            size = path.stat().st_size
            require(size == item["bytes"], "installed_package_file_size_mismatch")
            total_bytes += size
            require(total_bytes <= 512 * 1024**2, "installed_package_bytes_limit_exceeded")
            digest = hashlib.sha256()
            with path.open("rb") as installed_file:
                for block in iter(lambda: installed_file.read(1024**2), b""):
                    digest.update(block)
            require(digest.hexdigest() == item["sha256"], "installed_package_file_digest_mismatch")
    require(actual_files == set(expected), "installed_package_manifest_file_set_mismatch")
    require(actual_directories == expected_directories, "installed_package_manifest_directory_set_mismatch")
    return {"verified_files": len(actual_files), "verified_bytes": total_bytes,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "unmanifested_entries": 0, "python_startup_hooks": 0,
            "scope": "approved_package_root_only_host_python_overlay_excluded"}


def check_native_ed25519() -> dict[str, object]:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    message = b"legal-ir-worker-qualification-v1"
    signature = private_key.sign(message)
    require(len(signature) == 64, "ed25519_signature_length_invalid")
    public_key.verify(signature, message)
    rejected = False
    try:
        public_key.verify(signature, message + b"-changed")
    except InvalidSignature:
        rejected = True
    require(rejected, "ed25519_accepted_changed_message")
    return {"sign_verify": True, "changed_message_rejected": True}


def check_native_duckdb() -> dict[str, object]:
    import duckdb

    connection = duckdb.connect(":memory:", config={"threads": "1", "memory_limit": "128MB"})
    try:
        connection.execute("CREATE TABLE clauses (entity VARCHAR, value INTEGER)")
        connection.executemany("INSERT INTO clauses VALUES (?, ?)", [("actor", 3), ("actor", 5), ("term", 2)])
        rows = connection.execute(
            "SELECT entity, SUM(value) FROM clauses GROUP BY entity ORDER BY entity"
        ).fetchall()
        require(rows == [("actor", 8), ("term", 2)], "duckdb_parameterized_aggregation_invalid")
    finally:
        connection.close()
    return {"in_memory_parameterized_aggregation": True}


def check_filesystem() -> dict[str, object]:
    # mkstemp uses exclusive creation and never overwrites an existing path.
    for directory in (WORKSPACE, Path("/tmp")):
        fd, name = tempfile.mkstemp(prefix="legal-ir-owned-write-probe-", dir=directory)
        try:
            os.write(fd, b"owned qualification probe\n")
            os.fsync(fd)
            os.close(fd)
            fd = -1
            require(Path(name).read_bytes() == b"owned qualification probe\n", "writable_probe_readback_failed")
        finally:
            if fd >= 0:
                os.close(fd)
            Path(name).unlink(missing_ok=True)
    denied: list[str] = []
    for directory in (Path("/opt"), APPROVED_ROOT, Path("/usr")):
        # tempfile.mkstemp also guarantees the negative probe cannot replace an
        # existing host or image file, even if a mount was misconfigured.
        try:
            fd, name = tempfile.mkstemp(prefix="legal-ir-owned-denial-probe-", dir=directory)
        except OSError as error:
            require(error.errno in (errno.EROFS, errno.EACCES, errno.EPERM), "readonly_probe_unexpected_error")
            denied.append(str(directory))
        else:
            os.close(fd)
            Path(name).unlink(missing_ok=True)
            raise QualificationFailure("readonly_mount_was_writable")
    return {"writable": ["/workspace", "/tmp"], "writes_denied": denied}


def check_tools() -> dict[str, object]:
    require(PYTHON.is_file() and os.access(PYTHON, os.X_OK), "python_tool_missing")
    results: dict[str, object] = {}
    for name, arguments in (("git", ["--version"]), ("rg", ["--version"])):
        path = shutil.which(name, path=environment()["PATH"])
        require(path is not None, "required_tool_missing")
        resolved = Path(str(path)).resolve()
        if EXPECTED_ARCHITECTURE == "amd64" and name == "rg":
            require(resolved == Path("/opt/ipfs-task-tools/bin/rg"), "unexpected_image_ripgrep_path")
            require(hashlib.sha256(resolved.read_bytes()).hexdigest() == AMD64_RIPGREP_SHA256, "image_ripgrep_digest_mismatch")
        else:
            require(resolved.is_relative_to(Path("/usr")), "tool_outside_readonly_usr")
        result = run_command([str(path), *arguments], timeout=10)
        require(result["returncode"] == 0, "required_tool_version_failed")
        results[name] = {"path": str(path), **result}
    return {"python": str(PYTHON), **results}


TEST_SOURCE = '''from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import duckdb
import pytest


def test_ed25519_rejects_changed_message_and_wrong_key():
    key = Ed25519PrivateKey.generate()
    payload = b"independent-worker-subprocess-check"
    signature = key.sign(payload)
    key.public_key().verify(signature, payload)
    with pytest.raises(InvalidSignature):
        key.public_key().verify(signature, payload + b"changed")
    with pytest.raises(InvalidSignature):
        Ed25519PrivateKey.generate().public_key().verify(signature, payload)


def test_duckdb_join_aggregation_and_null_semantics():
    connection = duckdb.connect(":memory:", config={"threads": "1", "memory_limit": "128MB"})
    try:
        connection.execute("CREATE TABLE entities (id INTEGER, label VARCHAR)")
        connection.execute("CREATE TABLE spans (entity_id INTEGER, weight INTEGER)")
        connection.executemany("INSERT INTO entities VALUES (?, ?)", [(1, "actor"), (2, "term"), (3, "unused")])
        connection.executemany("INSERT INTO spans VALUES (?, ?)", [(1, 2), (1, 5), (2, 4)])
        result = connection.execute("SELECT e.label, COUNT(s.weight), SUM(s.weight) FROM entities e LEFT JOIN spans s ON e.id = s.entity_id GROUP BY e.label ORDER BY e.label").fetchall()
        assert result == [("actor", 2, 7), ("term", 1, 4), ("unused", 0, None)]
    finally:
        connection.close()
'''


def check_pytest_subprocess() -> dict[str, object]:
    test_directory = Path(tempfile.mkdtemp(prefix="legal-ir-qualification-tests-", dir=WORKSPACE))
    test_path = test_directory / "test_worker_native_runtime.py"
    junit_path = test_directory / "pytest-results.xml"
    test_path.write_text(TEST_SOURCE, encoding="utf-8")
    bootstrap = (
        "import sys; "
        f"sys.path.insert(0, {str(APPROVED_ROOT)!r}); "
        "import pytest; "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    result = run_command([
        str(PYTHON), "-I", "-S", "-B", "-c", bootstrap,
        "-q", "--disable-warnings", "-p", "no:cacheprovider",
        f"--junitxml={junit_path}", str(test_path),
    ], timeout=60)
    require(result["returncode"] == 0, "pytest_native_subprocess_failed")
    require(junit_path.is_file() and junit_path.stat().st_size <= 65_536, "pytest_result_missing_or_oversized")
    junit = ET.parse(junit_path).getroot()
    cases = junit.findall(".//testcase")
    require(len(cases) == 2, "pytest_executed_test_count_invalid")
    require({case.get("name") for case in cases} == {
        "test_ed25519_rejects_changed_message_and_wrong_key",
        "test_duckdb_join_aggregation_and_null_semantics",
    }, "pytest_executed_test_names_invalid")
    require(all(not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases), "pytest_cases_did_not_all_pass")
    return {"test_count": len(cases), "isolated_python": True, "plugin_autoload_disabled": True, **result}


def write_report(report: dict[str, object]) -> None:
    encoded = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    require(len(encoded) <= 32_768, "report_size_limit_exceeded")
    # Do not follow a pre-existing report symlink into another mounted tree.
    fd = os.open(REPORT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        raise
    sys.stdout.write(encoded.decode("utf-8"))
    sys.stdout.flush()


def main() -> int:
    global EXPECTED_WORKER_UID, EXPECTED_WHEEL_LOCK_SHA256, EXPECTED_ARCHITECTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-uid", type=int, default=1000)
    parser.add_argument("--architecture", choices=tuple(WHEEL_LOCKS), default="arm64")
    args = parser.parse_args()
    require(args.expected_uid > 0, "root_execution_forbidden")
    EXPECTED_WORKER_UID = args.expected_uid
    EXPECTED_WHEEL_LOCK_SHA256 = WHEEL_LOCKS[args.architecture]
    EXPECTED_ARCHITECTURE = args.architecture
    started = time.monotonic()
    checks: list[dict[str, object]] = []
    package_closure_ready = False
    for name, function, requires_packages in (
        ("identity_and_python_isolation", check_identity_and_isolation, False),
        ("approved_package_closure", check_package_closure, False),
        ("native_ed25519", check_native_ed25519, True),
        ("native_duckdb", check_native_duckdb, True),
        ("filesystem_permissions", check_filesystem, False),
        ("python_git_rg_tools", check_tools, False),
        ("isolated_pytest_subprocess", check_pytest_subprocess, True),
    ):
        if requires_packages and not package_closure_ready:
            checks.append({"name": name, "ok": False, "error": "approved_package_closure_not_qualified"})
            continue
        try:
            details = function()
        except QualificationFailure as error:
            checks.append({"name": name, "ok": False, "error": str(error)})
        except BaseException as error:
            # Exception arguments and tracebacks may contain arbitrary package
            # output. Report only the built-in/class name, bounded to 80 chars.
            checks.append({"name": name, "ok": False, "error_type": type(error).__name__[:80]})
        else:
            checks.append({"name": name, "ok": True, "details": details})
            if name == "approved_package_closure":
                package_closure_ready = True
    ok = all(check["ok"] is True for check in checks)
    report: dict[str, object] = {
        "schema": "legal-ir-ordinary-worker-runtime-qualification@1",
        "ok": ok,
        "checks": checks,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "provider_calls": 0,
        "network_calls": 0,
        "lean_qualified": False,
        "scope": "ordinary_worker_mounted_python_runtime_only",
    }
    write_report(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
