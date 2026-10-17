"""Future native-owner resource API qualification; authoring never executes it.

This stdlib-only driver compiles exact pinned loader utility bytes. It creates
explicit test-only package module views without executing a package initializer
or importing product, database, verifier, provider or model code. The real
importlib.resources API reads the actual six pinned C SQL files. No native
admission, custody, schema installation or original launch receipt is simulated.
Run only under the root's separately admitted finite original native owner.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import importlib.resources
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType

SCHEMA = "pinned-product-sql-resource-api-qualification-request@1"
RESULT_SCHEMA = "pinned-product-sql-resource-api-qualification-result@1"
PACKAGE = "ipfs_accelerate_py.agent_supervisor.task_sources"
KEYS = {"schema", "fixture_only", "repo_root", "controller_head", "loader_pin",
        "previous_loader_pin", "python_pin", "driver_pin", "package_initializer_pin",
        "catalog_source_pin", "sql_pins", "output"}
MAX_SOURCE = 8 * 1024**2
MAX_JSON = 2 * 1024**2
MAX_RESULT = 256 * 1024
PIN_FIELDS = {"path", "bytes", "sha256"}


def need(value, message):
    if not value:
        raise ValueError(message)


def decode(data):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate resource qualification JSON key")
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite qualification JSON")))


def regular(path, maximum=MAX_SOURCE, *, private=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "resource qualification file alias")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and 0 <= before.st_size <= maximum, "resource qualification file custody/bound")
        need(not private or (before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o600),
             "resource qualification private file differs")
        chunks, total = [], 0
        while block := os.read(fd, min(65536, maximum - total + 1)):
            chunks.append(block)
            total += len(block)
            need(total <= maximum, "resource qualification read bound")
        fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink",
                  "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda item: tuple(getattr(item, key) for key in fields)
        need(total == before.st_size and witness(before) == witness(os.fstat(fd)) == witness(path.lstat()),
             "resource qualification input changed")
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_pin(value, maximum=MAX_SOURCE):
    need(type(value) is dict and set(value) == PIN_FIELDS and type(value["bytes"]) is int,
         "resource qualification pin fields")
    data = regular(value["path"], maximum)
    need(len(data) == value["bytes"] and hashlib.sha256(data).hexdigest() == value["sha256"],
         "resource qualification pinned bytes differ")
    return data


def local_head(repo):
    marker = repo / ".git"
    if marker.is_file():
        text = regular(marker, 4096).decode().strip()
        need(text.startswith("gitdir: "), "resource qualification Git pointer")
        git = Path(text[8:])
        git = (git if git.is_absolute() else repo / git).resolve(strict=True)
    else:
        git = marker.resolve(strict=True)
    head = regular(git / "HEAD", 4096).decode().strip()
    if not head.startswith("ref: "):
        need(re.fullmatch("[0-9a-f]{40}", head), "resource qualification detached HEAD")
        return head
    ref = head[5:]
    need(ref.startswith("refs/") and ".." not in Path(ref).parts, "resource qualification Git ref")
    common = git
    if (git / "commondir").is_file():
        common = (git / regular(git / "commondir", 4096).decode().strip()).resolve(strict=True)
    for directory in (git, common):
        if (directory / ref).is_file():
            value = regular(directory / ref, 4096).decode().strip()
            need(re.fullmatch("[0-9a-f]{40}", value), "resource qualification loose HEAD")
            return value
    packed = regular(common / "packed-refs", MAX_JSON).decode().splitlines()
    found = [line.split()[0] for line in packed if line and not line.startswith(("#", "^"))
             and len(line.split()) == 2 and line.split()[1] == ref]
    need(len(found) == 1 and re.fullmatch("[0-9a-f]{40}", found[0]), "resource qualification packed HEAD")
    return found[0]


class NoProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "ipfs_accelerate_py" or fullname.startswith("ipfs_accelerate_py.")
                or fullname in {"duckdb", "_duckdb"} or fullname.startswith("duckdb.")):
            raise ImportError("resource API qualification must not import native product/backend")
        return None


def forbid_dispatch(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork", "os.forkpty",
                 "socket.connect", "socket.getaddrinfo"}:
        raise PermissionError("resource API qualification forbids process/network dispatch")
    if event == "socket.__new__":
        raise PermissionError("resource API qualification forbids sockets")


def utility_module(name, pin):
    data = read_pin(pin)
    module = ModuleType(name)
    module.__file__ = pin["path"]
    exec(compile(data, pin["path"], "exec", dont_inherit=True), module.__dict__)
    read_pin(pin)
    return module


def package_view(loader, repo, pins):
    # This is an explicit resource API test double, not an imported package.
    # find_spec constructs the genuine candidate loader; exec_module is NEVER called.
    spec = loader._PinnedProductFinder(repo, pins).find_spec(PACKAGE)
    need(spec is not None and spec.loader is not None and spec.submodule_search_locations is not None,
         "candidate test package resource spec differs")
    view = ModuleType(PACKAGE)
    view.__spec__, view.__loader__, view.__file__, view.__package__ = spec, spec.loader, spec.origin, PACKAGE
    view.__path__ = list(spec.submodule_search_locations)
    return view


def native_required_names(data):
    values = []
    for node in ast.parse(data).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "REQUIRED_PACKAGE_SQL_FILENAMES":
                values.append(ast.literal_eval(node.value))
        elif isinstance(node, ast.Assign) and any(isinstance(item, ast.Name)
                and item.id == "REQUIRED_PACKAGE_SQL_FILENAMES" for item in node.targets):
            values.append(ast.literal_eval(node.value))
    need(len(values) == 1 and type(values[0]) is tuple and len(values[0]) == 5
         and all(type(name) is str for name in values[0])
         and [int(name[:4]) for name in values[0]] == [1, 2, 3, 4, 5],
         "actual native catalog required migration literal differs")
    return list(values[0])


def write_new(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    need(len(raw) <= MAX_RESULT, "resource qualification result exceeds bound")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def run(request_path, expected_sha):
    need(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode is True,
         "resource qualification requires exact -I -S -B startup")
    raw = regular(request_path, MAX_JSON, private=True)
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "resource qualification request hash")
    request = decode(raw)
    need(type(request) is dict and set(request) == KEYS and request["schema"] == SCHEMA
         and request["fixture_only"] is True, "resource qualification closed request schema")
    repo, output = Path(request["repo_root"]), Path(request["output"])
    need(repo.is_absolute() and repo.resolve(strict=True) == repo
         and re.fullmatch("[0-9a-f]{40}", request["controller_head"])
         and local_head(repo) == request["controller_head"], "resource qualification actual C HEAD")
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
         and not output.exists() and not output.is_symlink() and not output.is_relative_to(repo),
         "resource qualification result must be NEW outside C")
    need(request["driver_pin"]["path"] == str(Path(__file__).resolve(strict=True))
         and request["python_pin"]["path"] == str(Path(sys.executable).resolve(strict=True)),
         "resource qualification actual driver/Python identity")
    read_pin(request["driver_pin"])
    read_pin(request["python_pin"], 128 * 1024**2)
    package = repo / "ipfs_accelerate_py/agent_supervisor/task_sources"
    need(request["package_initializer_pin"]["path"] == str(package / "__init__.py")
         and request["catalog_source_pin"]["path"] == str(package / "control_plane_migrations.py"),
         "resource qualification native package/catalog paths")
    initializer = read_pin(request["package_initializer_pin"])
    required = native_required_names(read_pin(request["catalog_source_pin"]))
    sql_pins = request["sql_pins"]
    need(type(sql_pins) is dict and len(sql_pins) == 6 and set(sql_pins)
         == {str(path) for path in (package / "sql").glob("*.sql")}, "actual six SQL source population")
    sql_bytes = {}
    for path, pin in sql_pins.items():
        need(path == pin["path"] and Path(path).parent == package / "sql", "SQL pin crosses actual native directory")
        sql_bytes[Path(path).name] = read_pin(pin, 1024**2)
    need(set(required).issubset(sql_bytes) and len(sql_bytes) == 6, "native required-five versus physical-six differs")
    sys.meta_path.insert(0, NoProductImports())
    sys.addaudithook(forbid_dispatch)
    candidate = utility_module("_resource_api_candidate_utilities", request["loader_pin"])
    previous = utility_module("_resource_api_previous_utilities", request["previous_loader_pin"])
    pins = candidate._package_pins(repo)
    need(type(pins) is dict and all(pins.get(path) == value for path, value in sql_pins.items())
         and pins.get(str(package / "__init__.py")) == request["package_initializer_pin"],
         "candidate full source/resource population differs")
    cases = []
    root = importlib.resources.files(package_view(candidate, repo, pins))
    directory = root.joinpath("sql")
    need(directory.is_dir() and not directory.is_file(), "candidate SQL directory not exposed")
    entries = sorted(directory.iterdir(), key=lambda item: item.name)
    need([item.name for item in entries] == sorted(sql_bytes) and all(item.is_file() for item in entries),
         "candidate resource enumeration differs from exact six actual SQL files")
    cases.append({"case": "candidate-real-stdlib-files-enumerates-exact-six", "passed": True})
    for item in entries:
        need(item.read_bytes() == sql_bytes[item.name]
             and item.read_text(encoding="utf-8") == sql_bytes[item.name].decode("utf-8"),
             "candidate binary/text resource bytes differ")
    cases.append({"case": "candidate-reads-all-six-exact-pinned-bytes", "passed": True})
    legacy = importlib.resources.files(package_view(previous, repo, pins)).joinpath("sql")
    need(legacy.is_dir() is False, "previous plain Loader unexpectedly exposes SQL resources")
    cases.append({"case": "previous-plain-loader-lacks-sql-resource-directory", "passed": True})

    def refusal(name, operation):
        try:
            operation()
        except (ValueError, FileNotFoundError, PermissionError, IsADirectoryError, io.UnsupportedOperation) as error:
            cases.append({"case": name, "passed": True, "refusal_type": type(error).__name__})
        else:
            raise ValueError("expected resource API refusal absent: " + name)

    try:
        candidate._PinnedProductFinder(repo, pins).find_spec(PACKAGE + ".sql")
    except ModuleNotFoundError:
        cases.append({"case": "sql-only-namespace-cannot-bypass-closed-reader", "passed": True})
    else:
        raise ValueError("SQL-only namespace unexpectedly exposes ambient resource discovery")
    actual_loader = package_view(candidate, repo, pins).__spec__.loader
    refusal("reader-wrong-package-name", lambda: actual_loader.get_resource_reader(PACKAGE + ".other"))
    refusal("resource-path-filesystem-fallback", lambda: actual_loader.get_resource_reader(PACKAGE).resource_path(entries[0].name))
    with entries[0].open("rb") as binary:
        view = binary.getbuffer()
        try:
            need(binary.writable() is False and view.readonly is True,
                 "candidate binary resource snapshot is writable")
        finally:
            view.release()
        cases.append({"case": "binary-resource-buffer-is-readonly", "passed": True})
        refusal("binary-resource-write", lambda: binary.write(b"fixture-only-memory-write"))
        refusal("binary-resource-truncate", lambda: binary.truncate())
        refusal("binary-resource-writelines", lambda: binary.writelines([b"fixture-only-memory-write"]))
    refusal("unregistered-file", lambda: directory.joinpath("unregistered-resource.sql").read_bytes())
    refusal("parent-path-escape", lambda: directory.joinpath("..").joinpath("control_plane_migrations.py").read_bytes())
    refusal("absolute-path", lambda: directory.joinpath("/etc/passwd").read_bytes())
    refusal("embedded-parent-path-escape", lambda: directory.joinpath("../control_plane_migrations.py").read_bytes())
    refusal("directory-open", lambda: directory.open("rb"))
    for mode in ("w", "wb", "a", "x", "r+"):
        refusal("write-mode-" + mode, lambda selected=mode: entries[0].open(selected))
    target = sorted(sql_pins)[0]
    corrupted = dict(pins)
    corrupted[target] = {**pins[target], "sha256": "0" * 64}
    refusal("copied-sql-pin-hash-mismatch", lambda: importlib.resources.files(
        package_view(candidate, repo, corrupted)).joinpath("sql").joinpath(Path(target).name).read_bytes())
    corrupted_initializer = dict(pins)
    key = str(package / "__init__.py")
    corrupted_initializer[key] = {**pins[key], "sha256": "0" * 64}
    refusal("copied-initializer-pin-hash-mismatch", lambda: importlib.resources.files(
        package_view(candidate, repo, corrupted_initializer)).joinpath("sql").joinpath(Path(target).name).read_bytes())
    # All C/source bytes must remain unchanged. No C file is modified by a negative control.
    for pin in [request["driver_pin"], request["loader_pin"], request["previous_loader_pin"],
                request["package_initializer_pin"], request["catalog_source_pin"], *sql_pins.values()]:
        read_pin(pin)
    need(candidate._package_pins(repo) == pins,
         "full actual product Python/SQL source population changed during resource qualification")
    need(local_head(repo) == request["controller_head"], "C HEAD changed during resource qualification")
    need(not any(name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py.")
         or name in {"duckdb", "_duckdb"} or name.startswith("duckdb.") for name in sys.modules),
         "product/backend module unexpectedly loaded")
    result = {"schema": RESULT_SCHEMA, "fixture_only": True, "passed": True,
        "test_package_view_only": True, "test_view_package_initializer_executed": False,
        "request_pin": {"path": str(request_path), "bytes": len(raw), "sha256": expected_sha},
        "driver_pin": request["driver_pin"], "loader_pin": request["loader_pin"],
        "previous_loader_pin": request["previous_loader_pin"], "controller_head": request["controller_head"],
        "native_catalog_source_pin": request["catalog_source_pin"],
        "actual_sql_resource_pins": sql_pins, "physical_sql_files": sorted(sql_bytes),
        "native_required_migration_names_from_literal_AST": required,
        "native_migration_catalog_executed": False, "native_schema_installed": False,
        "case_count": len(cases), "cases": cases, "source_hashes_unchanged": True,
        "product_backend_or_verifier_imported": False, "database_opened": False,
        "process_network_provider_or_model_dispatched": False, "symlink_fixture_created": False,
        "resource_api_fixture_qualified": True, "authentic_custody_qualified": False,
        "production_qualified": False, "historical_unknown_resolved": False,
        "native_resource_ack_created": False, "original_launch_receipt_created": False,
        "limitations": ["Package views are explicitly constructed metadata test doubles, not real package imports.",
            "This finite resource API test does not qualify genuine schema installation, A/B custody, ADMIN or settlement.",
            "Native catalog remains unexecuted; required-five migration names are a pinned source AST observation."]}
    result_pin = write_new(output, result)
    print(json.dumps({"fixture_only": True, "passed": True, "result_pin": result_pin,
                      "native_resource_ack_created": False}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    return run(args.request, args.request_sha256)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"resource_api_qualification_failed": True,
                          "error_type": type(error).__name__, "fixture_only": True,
                          "native_resource_ack_created": False}), file=sys.stderr, flush=True)
        raise SystemExit(2)
