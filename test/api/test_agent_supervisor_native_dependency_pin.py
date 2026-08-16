"""Hermetic native dependency pin, sealing, and preload tests."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import struct
import subprocess
import sys
import sysconfig
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ipfs_accelerate_py import llm_router

AUTHORIZATION_ID = (
    "sha256:039bdbbff886311847200cfdb4d99a498b8836f11e49b139f3dce5d1f398c4ff"
)
REAL_DUCKDB_PAYLOAD_SHA256 = (
    "sha256:c378b8f61040764fdc904cf7c0643a005d547f491ab9303e6bd13c33aa353f2a"
)
REAL_DUCKDB_DEPENDENCY_ID = (
    "sha256:bf982f675cc4c4fa212066d706cd387c9821b3b69f5f8cc7c07169bc347b88b5"
)
REAL_PYTHON_EXECUTABLE_SHA256 = (
    "sha256:1a301bb1763139d48ae638d97b11edf56de6cd185e1b054eae6dc28c271c0c5f"
)
REAL_DUCKDB_SIZE = 54_278_072
REAL_DUCKDB_NEEDED = (
    "libdl.so.2",
    "libstdc++.so.6",
    "libm.so.6",
    "libgcc_s.so.1",
    "libpthread.so.0",
    "libc.so.6",
)


def _machine_code() -> int:
    codes = {"aarch64": 183, "x86_64": 62}
    try:
        return codes[os.uname().machine]
    except KeyError:
        pytest.skip("synthetic ELF fixture supports aarch64 and x86_64")


def _synthetic_elf(
    *,
    needed: tuple[str, ...] = ("libc.so.6",),
    forbidden_tag: int | None = None,
    include_interp: bool = False,
    executable_stack: bool = False,
    object_type: int = 3,
    string_table_address: int | None = None,
    dynamic_virtual_address: int | None = None,
) -> bytes:
    """Build a small, independently structured ELF64 policy fixture."""

    endian = "<" if sys.byteorder == "little" else ">"
    data_encoding = 1 if sys.byteorder == "little" else 2
    base_address = 0x400000
    string_offset = 0x200
    dynamic_offset = 0x300
    string_table = bytearray(b"\0")
    needed_offsets: list[int] = []
    for name in needed:
        needed_offsets.append(len(string_table))
        string_table.extend(name.encode("ascii") + b"\0")
    entries = [
        (5, string_table_address or base_address + string_offset),
        (10, len(string_table)),
        *((1, offset) for offset in needed_offsets),
    ]
    if forbidden_tag is not None:
        entries.append((forbidden_tag, 1))
    entries.append((0, 0))
    dynamic = b"".join(struct.pack(endian + "qQ", *entry) for entry in entries)
    program_count = 2 + int(include_interp) + int(executable_stack)
    total_size = 0x500
    ident = (
        b"\x7fELF"
        + bytes((2, data_encoding, 1, 3, 0))
        + b"\0" * 7
    )
    header = ident + struct.pack(
        endian + "HHIQQQIHHHHHH",
        object_type,
        _machine_code(),
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        program_count,
        0,
        0,
        0,
    )
    program_headers = [
        struct.pack(
            endian + "IIQQQQQQ",
            1,
            5,
            0,
            base_address,
            base_address,
            total_size,
            total_size,
            0x1000,
        ),
        struct.pack(
            endian + "IIQQQQQQ",
            2,
            6,
            dynamic_offset,
            dynamic_virtual_address or base_address + dynamic_offset,
            dynamic_virtual_address or base_address + dynamic_offset,
            len(dynamic),
            len(dynamic),
            8,
        ),
    ]
    if include_interp:
        program_headers.append(
            struct.pack(
                endian + "IIQQQQQQ",
                3,
                4,
                0x180,
                base_address + 0x180,
                base_address + 0x180,
                8,
                8,
                1,
            )
        )
    if executable_stack:
        program_headers.append(
            struct.pack(
                endian + "IIQQQQQQ",
                0x6474E551,
                7,
                0,
                0,
                0,
                0,
                0,
                16,
            )
        )
    payload = bytearray(total_size)
    payload[:64] = header
    payload[64 : 64 + 56 * len(program_headers)] = b"".join(program_headers)
    payload[string_offset : string_offset + len(string_table)] = string_table
    payload[dynamic_offset : dynamic_offset + len(dynamic)] = dynamic
    return bytes(payload)


def _source_path(root: Path) -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert isinstance(suffix, str)
    return root / f"_duckdb{suffix}"


def _write_source(root: Path, raw: bytes | None = None) -> Path:
    path = _source_path(root)
    path.write_bytes(raw if raw is not None else _synthetic_elf())
    path.chmod(0o775)
    return path


def _inspect(path: Path) -> llm_router.AgentSupervisorNativeDependencyPin:
    return llm_router.inspect_agent_supervisor_native_dependency_source(
        path,
        distribution_version="9.9.9",
        engine_version="v9.9.9",
    )


def _seal(
    path: Path,
    pin: llm_router.AgentSupervisorNativeDependencyPin,
) -> llm_router.AgentSupervisorNativeDependencyLaunch:
    return llm_router.seal_agent_supervisor_native_dependency(
        path,
        expected_pin=pin,
        accepted_authorization_id=AUTHORIZATION_ID,
    )


def _rebound_pin(
    pin: llm_router.AgentSupervisorNativeDependencyPin,
    **updates: object,
) -> llm_router.AgentSupervisorNativeDependencyPin:
    values = pin.as_dict()
    values.update(updates)
    values["dependency_id"] = llm_router._content_addressed_mapping(
        values,
        identity_field="dependency_id",
    )
    values["elf_dt_needed"] = tuple(values["elf_dt_needed"])
    return llm_router.AgentSupervisorNativeDependencyPin(**values)  # type: ignore[arg-type]


def _launch_mapping(
    launch: llm_router.AgentSupervisorNativeDependencyLaunch,
) -> dict[str, Any]:
    return json.loads(launch.to_json())


class _FakeCursor:
    def __init__(self, row: tuple[object, ...]) -> None:
        self._row = row

    def fetchone(self) -> tuple[object, ...]:
        return self._row


class _FakeConnection:
    def __init__(self, engine_version: str, *, close_raises: bool) -> None:
        self.engine_version = engine_version
        self.close_raises = close_raises
        self.closed = False

    def execute(self, query: str) -> _FakeCursor:
        if query == "SELECT version()":
            return _FakeCursor((self.engine_version,))
        if query == "SELECT 42":
            return _FakeCursor((42,))
        raise AssertionError(f"unexpected query: {query}")

    def close(self) -> None:
        self.closed = True
        if self.close_raises:
            raise RuntimeError("synthetic close failure")


class _FakeExtensionLoader:
    distribution_version = "9.9.9"
    engine_version = "v9.9.9"
    mutate_mode = False
    close_raises = False
    calls: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path
        self.calls.append((name, path))

    def create_module(self, spec: object) -> types.ModuleType:
        del spec
        module = types.ModuleType(self.name)
        module.__file__ = self.path
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        module.__version__ = self.distribution_version
        module.connect = lambda database: _FakeConnection(  # type: ignore[attr-defined]
            self.engine_version,
            close_raises=self.close_raises,
        )
        if self.mutate_mode:
            descriptor = int(self.path.rsplit("/", 1)[-1])
            os.fchmod(descriptor, 0o400)


def _install_fake_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeExtensionLoader.distribution_version = "9.9.9"
    _FakeExtensionLoader.engine_version = "v9.9.9"
    _FakeExtensionLoader.mutate_mode = False
    _FakeExtensionLoader.close_raises = False
    _FakeExtensionLoader.calls = []
    # These unit tests replace ExtensionFileLoader with a pure-Python fake, so
    # no native initialization occurred in a prior test.  Reset only the
    # private test-process sentinel; production exposes no reset path.
    monkeypatch.setattr(
        llm_router,
        "_AGENT_NATIVE_DEPENDENCY_PRELOAD_STARTED",
        False,
    )
    monkeypatch.setattr(
        llm_router.importlib.machinery,
        "ExtensionFileLoader",
        _FakeExtensionLoader,
    )
    monkeypatch.delitem(sys.modules, "_duckdb", raising=False)
    monkeypatch.delitem(sys.modules, "duckdb", raising=False)


def test_inspection_is_path_free_evidence_and_sealing_requires_acceptance(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    pin = _inspect(source)

    assert stat.S_IMODE(source.stat().st_mode) == 0o775
    assert pin.module_name == "_duckdb"
    assert pin.public_alias == pin.distribution_name == "duckdb"
    assert pin.extension_filename == source.name
    assert pin.elf_dt_needed == ("libc.so.6",)
    assert str(source) not in pin.to_json()
    assert pin.dependency_id == llm_router._content_addressed_mapping(
        pin.as_dict(),
        identity_field="dependency_id",
    )

    with pytest.raises(ValueError, match="externally accepted"):
        llm_router.seal_agent_supervisor_native_dependency(source)
    with pytest.raises(ValueError, match="authorization"):
        llm_router.seal_agent_supervisor_native_dependency(
            source,
            expected_pin=pin,
        )
    with pytest.raises(ValueError, match="authorization"):
        llm_router.seal_agent_supervisor_native_dependency(
            source,
            expected_pin=pin,
            accepted_authorization_id="evidence-only",
        )

    source.write_bytes(_synthetic_elf(needed=("libm.so.6",)))
    source.chmod(0o775)
    with pytest.raises(ValueError, match="does not match"):
        _seal(source, pin)


@pytest.mark.parametrize(
    "kind",
    ("leaf_symlink", "parent_symlink", "hardlink", "fifo", "empty"),
)
def test_source_inspection_is_stable_nofollow_evidence(
    tmp_path: Path,
    kind: str,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    real = _write_source(real_parent)
    if kind == "leaf_symlink":
        candidate = _source_path(tmp_path)
        candidate.symlink_to(real)
    elif kind == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(real_parent, target_is_directory=True)
        candidate = _source_path(alias)
    elif kind == "hardlink":
        candidate = _source_path(tmp_path)
        os.link(real, candidate)
    elif kind == "fifo":
        candidate = _source_path(tmp_path)
        os.mkfifo(candidate)
    else:
        candidate = _source_path(tmp_path)
        candidate.write_bytes(b"")

    with pytest.raises(ValueError):
        _inspect(candidate)


@pytest.mark.parametrize(
    "updates",
    (
        {"module_name": "duckdb"},
        {"public_alias": "fake_duckdb"},
        {"platform_name": "not-linux"},
        {"platform_machine": "not-aarch64"},
        {"python_cache_tag": "wrong-cache-tag"},
        {"python_soabi": "wrong-soabi"},
        {"python_executable_sha256": "sha256:" + "0" * 64},
        {"extension_filename": "_duckdb.wrong.so"},
        {"elf_class_bits": 32},
        {
            "elf_endianness": (
                "big" if sys.byteorder == "little" else "little"
            )
        },
        {"elf_abi_version": 1},
        {"elf_machine": 62 if os.uname().machine == "aarch64" else 183},
        {"payload_sha256": "sha256:" + "0" * 64},
        {"size_bytes": 1},
        {"elf_dt_needed": ["libm.so.6"]},
    ),
)
def test_sealing_denies_wrong_pin_identity(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    source = _write_source(tmp_path)
    pin = _inspect(source)
    wrong = _rebound_pin(pin, **updates)
    with pytest.raises(ValueError):
        _seal(source, wrong)


@pytest.mark.parametrize(
    "forbidden_tag",
    (15, 29, 0x6FFFFEFB, 0x6FFFFEFC, 0x7FFFFFFD, 0x7FFFFFFF),
)
def test_elf_parser_rejects_ambient_dynamic_loader_tags(
    forbidden_tag: int,
) -> None:
    with pytest.raises(ValueError, match="ambient loader path"):
        llm_router._agent_parse_native_dependency_elf(
            _synthetic_elf(forbidden_tag=forbidden_tag)
        )


@pytest.mark.parametrize(
    "case",
    (
        "not_elf",
        "interp",
        "executable_stack",
        "executable_object",
        "needed_path",
        "duplicate_needed",
        "overlapping_string_table",
        "mismatched_dynamic_mapping",
    ),
)
def test_elf_parser_rejects_malformed_or_ambient_layout(case: str) -> None:
    fixtures = {
        "not_elf": b"not-elf",
        "interp": _synthetic_elf(include_interp=True),
        "executable_stack": _synthetic_elf(executable_stack=True),
        "executable_object": _synthetic_elf(object_type=2),
        "needed_path": _synthetic_elf(needed=("../libc.so.6",)),
        "duplicate_needed": _synthetic_elf(
            needed=("libc.so.6", "libc.so.6")
        ),
        "overlapping_string_table": _synthetic_elf(
            string_table_address=0x400300
        ),
        "mismatched_dynamic_mapping": _synthetic_elf(
            dynamic_virtual_address=0x400310
        ),
    }
    with pytest.raises(ValueError):
        llm_router._agent_parse_native_dependency_elf(fixtures[case])


def test_launch_json_and_descriptor_binding_are_strict(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    pin = _inspect(source)
    launch = _seal(source, pin)
    second = _seal(source, pin)
    try:
        assert launch.pass_fds == (launch.descriptor.descriptor,)
        assert launch.bootstrap_arguments == (
            str(launch.descriptor.descriptor),
            launch.to_json(),
        )
        assert str(source) not in launch.to_json()
        assert (
            llm_router.parse_agent_supervisor_native_dependency_launch(
                _launch_mapping(launch)
            )
            == launch
        )
        assert (
            llm_router.verify_agent_supervisor_native_dependency_sealed_fd(
                launch
            )
            == f"/proc/self/fd/{launch.descriptor.descriptor}"
        )

        with pytest.raises(ValueError, match="canonical"):
            llm_router._agent_parse_native_dependency_launch_json(
                " " + launch.to_json()
            )
        duplicate = '{"schema":"duplicate",' + launch.to_json()[1:]
        with pytest.raises(ValueError, match="invalid"):
            llm_router._agent_parse_native_dependency_launch_json(duplicate)
        extra = _launch_mapping(launch)
        extra["extra"] = True
        with pytest.raises(ValueError, match="fields"):
            llm_router.parse_agent_supervisor_native_dependency_launch(extra)
        boolean_fd = _launch_mapping(launch)
        boolean_fd["descriptor"]["descriptor"] = True
        with pytest.raises(ValueError):
            llm_router.parse_agent_supervisor_native_dependency_launch(boolean_fd)

        substituted_binding = replace(
            launch.descriptor,
            descriptor=second.descriptor.descriptor,
        )
        substituted = replace(launch, descriptor=substituted_binding)
        with pytest.raises(ValueError, match="identity changed"):
            llm_router.verify_agent_supervisor_native_dependency_sealed_fd(
                substituted
            )
        with pytest.raises(ValueError, match="substituted"):
            llm_router.preload_agent_supervisor_native_dependency_from_bootstrap(
                str(second.descriptor.descriptor),
                launch.to_json(),
            )

        os.fchmod(launch.descriptor.descriptor, 0o400)
        with pytest.raises(ValueError):
            llm_router.verify_agent_supervisor_native_dependency_sealed_fd(
                launch
            )
    finally:
        os.close(launch.descriptor.descriptor)
        os.close(second.descriptor.descriptor)


def test_unsealed_fd_and_descriptor_mutations_are_denied(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    raw = source.read_bytes()
    pin = _inspect(source)
    launch = _seal(source, pin)
    unsealed = os.memfd_create(
        llm_router._AGENT_NATIVE_DEPENDENCY_MEMFD_NAME,
        os.MFD_ALLOW_SEALING,
    )
    try:
        os.write(unsealed, raw)
        os.fchmod(unsealed, llm_router._AGENT_NATIVE_DEPENDENCY_SEALED_MODE)
        metadata = os.fstat(unsealed)
        unsealed_binding = replace(
            launch.descriptor,
            descriptor=unsealed,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=metadata.st_mode,
            st_uid=metadata.st_uid,
            st_nlink=metadata.st_nlink,
            seals=launch.descriptor.seals,
        )
        with pytest.raises(ValueError):
            llm_router.verify_agent_supervisor_native_dependency_sealed_fd(
                replace(launch, descriptor=unsealed_binding)
            )
        for field, value in (
            ("st_dev", launch.descriptor.st_dev + 1),
            ("st_ino", launch.descriptor.st_ino + 1),
            ("st_mode", stat.S_IFREG | 0o400),
            ("st_uid", launch.descriptor.st_uid + 1),
            ("st_nlink", 1),
            ("size_bytes", launch.descriptor.size_bytes - 1),
            ("payload_sha256", "sha256:" + "0" * 64),
            ("seals", 0),
        ):
            with pytest.raises(ValueError):
                llm_router.verify_agent_supervisor_native_dependency_sealed_fd(
                    replace(
                        launch,
                        descriptor=replace(launch.descriptor, **{field: value}),
                    )
                )
    finally:
        os.close(unsealed)
        os.close(launch.descriptor.descriptor)


def test_synthetic_preload_uses_exact_loader_alias_and_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_loader(monkeypatch)
    source = _write_source(tmp_path)
    pin = _inspect(source)
    launch = _seal(source, pin)
    try:
        module = llm_router.preload_agent_supervisor_native_dependency_from_bootstrap(
            *launch.bootstrap_arguments
        )
        assert _FakeExtensionLoader.calls == [
            ("_duckdb", f"/proc/self/fd/{launch.descriptor.descriptor}")
        ]
        assert sys.modules["_duckdb"] is module
        assert sys.modules["duckdb"] is module
        assert module.connect(":memory:").execute("SELECT 42").fetchone() == (42,)
    finally:
        sys.modules.pop("_duckdb", None)
        sys.modules.pop("duckdb", None)
        os.close(launch.descriptor.descriptor)


@pytest.mark.parametrize(
    "variable",
    ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT"),
)
def test_preload_denies_ambient_loader_environment_before_loader_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    _install_fake_loader(monkeypatch)
    source = _write_source(tmp_path)
    pin = _inspect(source)
    launch = _seal(source, pin)
    monkeypatch.setenv(variable, str(tmp_path / "attacker.so"))
    try:
        with pytest.raises(ValueError, match="ambient loader environment"):
            llm_router.preload_agent_supervisor_native_dependency(launch)
        assert _FakeExtensionLoader.calls == []
        assert not llm_router._AGENT_NATIVE_DEPENDENCY_PRELOAD_STARTED
    finally:
        os.close(launch.descriptor.descriptor)


@pytest.mark.parametrize(
    "failure_kind",
    ("distribution", "engine", "postload_mode", "connection_close"),
)
def test_preload_failure_makes_process_terminal_without_a_second_loader_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    _install_fake_loader(monkeypatch)
    source = _write_source(tmp_path)
    pin = _inspect(source)
    launch = _seal(source, pin)
    retry_launch = _seal(source, pin)
    try:
        monkeypatch.setitem(sys.modules, "duckdb", types.ModuleType("duckdb"))
        with pytest.raises(ValueError, match="already present"):
            llm_router.preload_agent_supervisor_native_dependency(launch)
        monkeypatch.delitem(sys.modules, "duckdb")
        assert _FakeExtensionLoader.calls == []

        if failure_kind == "distribution":
            _FakeExtensionLoader.distribution_version = "0.0.0"
        elif failure_kind == "engine":
            _FakeExtensionLoader.engine_version = "v0.0.0"
        elif failure_kind == "postload_mode":
            _FakeExtensionLoader.mutate_mode = True
        elif failure_kind == "connection_close":
            _FakeExtensionLoader.close_raises = True
        else:  # pragma: no cover - closed pytest parameter set
            raise AssertionError(f"unknown failure kind: {failure_kind}")
        with pytest.raises(ValueError, match="failed closed"):
            llm_router.preload_agent_supervisor_native_dependency(launch)
        assert "_duckdb" not in sys.modules and "duckdb" not in sys.modules
        assert len(_FakeExtensionLoader.calls) == 1

        _FakeExtensionLoader.distribution_version = "9.9.9"
        _FakeExtensionLoader.engine_version = "v9.9.9"
        _FakeExtensionLoader.mutate_mode = False
        _FakeExtensionLoader.close_raises = False
        with pytest.raises(ValueError, match="process is terminal"):
            llm_router.preload_agent_supervisor_native_dependency(retry_launch)
        assert len(_FakeExtensionLoader.calls) == 1
    finally:
        sys.modules.pop("_duckdb", None)
        sys.modules.pop("duckdb", None)
        os.close(launch.descriptor.descriptor)
        os.close(retry_launch.descriptor.descriptor)


_ISOLATED_PRELOAD = r"""
import json
import sys

native_fd, launch_json, trusted_root = sys.argv[1:]
sys.path.insert(0, trusted_root)
from ipfs_accelerate_py.llm_router import (
    preload_agent_supervisor_native_dependency_from_bootstrap,
)

module = preload_agent_supervisor_native_dependency_from_bootstrap(
    native_fd,
    launch_json,
)
print(json.dumps({
    "module": module.__name__,
    "origin": module.__file__,
    "version": module.__version__,
    "aliases_identical": sys.modules["_duckdb"] is sys.modules["duckdb"],
    "query": module.connect(":memory:").execute("SELECT 42").fetchone()[0],
}, sort_keys=True))
"""


def test_real_aarch64_duckdb_loads_from_sealed_fd_under_isolated_python(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux" or not hasattr(os, "uname"):
        pytest.skip("Linux memfd integration only")
    if os.uname().machine != "aarch64" or sys.implementation.cache_tag != "cpython-312":
        pytest.skip("reviewed native fixture is CPython 3.12 aarch64")
    installed = importlib.util.find_spec("_duckdb")
    if installed is None or installed.origin is None:
        pytest.skip("reviewed DuckDB fixture is unavailable")
    source = Path(installed.origin)
    if (
        source.name != "_duckdb.cpython-312-aarch64-linux-gnu.so"
        or not source.is_file()
        or source.stat().st_size != REAL_DUCKDB_SIZE
        or not Path("/usr/bin/python3.12").is_file()
    ):
        pytest.skip("reviewed DuckDB/Python fixture is unavailable")

    pin = llm_router.inspect_agent_supervisor_native_dependency_source(
        source,
        distribution_version="1.5.2",
        engine_version="v1.5.2",
    )
    assert stat.S_IMODE(source.stat().st_mode) == 0o775
    assert pin.dependency_id == REAL_DUCKDB_DEPENDENCY_ID
    assert pin.payload_sha256 == REAL_DUCKDB_PAYLOAD_SHA256
    assert pin.python_executable_sha256 == REAL_PYTHON_EXECUTABLE_SHA256
    assert pin.size_bytes == REAL_DUCKDB_SIZE
    assert pin.elf_class_bits == 64
    assert pin.elf_endianness == "little"
    assert pin.elf_osabi == 3
    assert pin.elf_machine == 183
    assert pin.elf_dt_needed == REAL_DUCKDB_NEEDED

    launch = _seal(source, pin)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "hostile-imported"
    hostile.joinpath("sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('sitecustomize')\n"
    )
    for name in ("duckdb.py", "_duckdb.py"):
        hostile.joinpath(name).write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text({name!r})\n"
        )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(hostile),
            "PYTHONUSERBASE": str(hostile),
            "PYTHONSTARTUP": str(hostile / "sitecustomize.py"),
        }
    )
    hostile_loader_environment = dict(environment)
    hostile_loader_environment.update(
        {
            "LD_LIBRARY_PATH": str(hostile),
            "LD_PRELOAD": str(hostile / "attacker.so"),
        }
    )
    repository_root = Path(llm_router.__file__).resolve().parents[1]
    try:
        denied = subprocess.run(
            [
                "/usr/bin/python3.12",
                "-I",
                "-c",
                _ISOLATED_PRELOAD,
                *launch.bootstrap_arguments,
                str(repository_root),
            ],
            cwd=hostile,
            env=hostile_loader_environment,
            pass_fds=launch.pass_fds,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert denied.returncode != 0
        assert "ambient loader environment is forbidden" in denied.stderr
        sanitized_environment = {
            name: value
            for name, value in environment.items()
            if not name.startswith("LD_")
        }
        completed = subprocess.run(
            [
                "/usr/bin/python3.12",
                "-I",
                "-c",
                _ISOLATED_PRELOAD,
                *launch.bootstrap_arguments,
                str(repository_root),
            ],
            cwd=hostile,
            env=sanitized_environment,
            pass_fds=launch.pass_fds,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    finally:
        os.close(launch.descriptor.descriptor)
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed == {
        "aliases_identical": True,
        "module": "_duckdb",
        "origin": f"/proc/self/fd/{launch.descriptor.descriptor}",
        "query": 42,
        "version": "1.5.2",
    }
    assert not marker.exists()
