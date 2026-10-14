"""Exercise the real CLI across independent processes and removed checkouts."""
import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_cli as cli


def invoke(*arguments):
    process = subprocess.run([sys.executable, "-m", cli.__name__, *map(str, arguments)],
        capture_output=True, timeout=30, check=False)
    return process


def result(*arguments):
    process = invoke(*arguments)
    assert process.returncode == 0, process.stderr.decode()
    return json.loads(process.stdout)


def fixture_tree(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_bytes(b"# This original comment is retained in the package.\n"
        b"def add(value):\n    # This detailed implementation comment is intentionally long.\n"
        b"    return value + 1\n")
    (source / "data.bin").write_bytes(b"\x00\xff\x01")
    (source / ".config").write_bytes(b"enabled=true\n")
    return source


def test_cli_create_inspect_read_delete_source_and_restore(tmp_path):
    source = fixture_tree(tmp_path)
    originals = {path.name: path.read_bytes() for path in source.iterdir()}
    package = tmp_path / "package"
    receipt = result("create", source, package)
    binding = ["--expected-manifest-sha256", receipt["manifest_sha256"]]
    verified = result("verify", package, *binding)
    assert verified["entries_verified"] == 3
    assert verified["proof_authority"] is False
    listing = result("list", package, *binding, "--limit", "2")
    assert len(listing["entries"]) == 2 and listing["truncated"] is True
    filtered = result("list", package, *binding, "--prefix", "calc")
    assert set(filtered["entries"]) == {"calc.py"}
    compact = result("read", package, "calc.py", *binding)
    assert "return value + 1" in compact["content"]
    assert "original comment" not in compact["content"]
    shutil.rmtree(source)
    restored = tmp_path / "restored"
    result("restore", package, restored, *binding)
    assert {p.name: p.read_bytes() for p in restored.iterdir()} == originals
    compact_root = tmp_path / "compact"
    result("restore", package, compact_root, *binding, "--representation", "compact")
    assert (compact_root / "calc.py").read_text() == compact["content"]
    assert (compact_root / "data.bin").read_bytes() == originals["data.bin"]


def test_cli_revision_materializes_authored_behavior_without_changing_parent(tmp_path):
    source = fixture_tree(tmp_path)
    original = (source / "calc.py").read_bytes()
    package = tmp_path / "package"
    captured = result("create", source, package)
    operations = tmp_path / "operations.json"
    operations.write_text(json.dumps([
        {"operation": "replace_symbol", "path": "calc.py", "symbol": "add",
         "expected_sha256": hashlib.sha256(original).hexdigest(),
         "replacement": "def add(value):\n    return value + 10\n"},
        {"operation": "write_file", "path": "run.py", "expected_sha256": None,
         "content": "from calc import add\nprint(add(2))\n"},
    ]))
    successor = tmp_path / "successor"
    revised = result("revise", package, successor, "--expected-manifest-sha256",
        captured["manifest_sha256"], "--operations", operations)
    # Read the package through the same explicit verification protocol; the
    # revision receipt uses its own parent/successor vocabulary.
    manifest_hash = hashlib.sha256((successor / "manifest.json").read_bytes()).hexdigest()
    result("verify", successor, "--expected-manifest-sha256", manifest_hash)
    restored = tmp_path / "restored"
    result("restore", successor, restored, "--expected-manifest-sha256", manifest_hash)
    run = subprocess.run([sys.executable, "-B", "run.py"], cwd=restored,
        capture_output=True, timeout=10, check=True)
    assert run.stdout == b"12\n"
    assert (source / "calc.py").read_bytes() == original
    assert captured["manifest_sha256"] == hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
    assert revised


@pytest.mark.parametrize("arguments", [("--limit", "0"), ("--limit", "129")])
def test_cli_listing_bounds_return_structured_failure(tmp_path, arguments):
    package = tmp_path / "package"
    receipt = result("create", fixture_tree(tmp_path), package)
    process = invoke("list", package, "--expected-manifest-sha256", receipt["manifest_sha256"], *arguments)
    assert process.returncode == 2 and not process.stdout
    assert json.loads(process.stderr)["schema"] == "repository-source-command-error@1"


def test_cli_bad_pin_and_existing_restore_never_overwrite(tmp_path):
    package = tmp_path / "package"
    receipt = result("create", fixture_tree(tmp_path), package)
    process = invoke("verify", package, "--expected-manifest-sha256", "0" * 64)
    assert process.returncode == 2 and not process.stdout
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"leave intact")
    process = invoke("restore", package, destination, "--expected-manifest-sha256", receipt["manifest_sha256"])
    assert process.returncode == 2
    assert (destination / "sentinel").read_bytes() == b"leave intact"


@pytest.mark.parametrize("content", [b'[{"operation":"x","operation":"y"}]', b'[NaN]', b'{}'])
def test_operation_file_rejects_ambiguous_json(tmp_path, content):
    path = tmp_path / "operations.json"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        cli._operations(path)


def test_operation_file_rejects_symlink_and_oversize(tmp_path):
    path = tmp_path / "operations.json"
    path.write_text("[]")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        cli._operations(link)
    with path.open("wb") as stream:
        stream.truncate(cli.MAX_OPERATIONS_BYTES + 1)
    with pytest.raises(ValueError):
        cli._operations(path)
