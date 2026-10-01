"""Retain only declared public task files, never supervisor or verifier state.

Capture is observational: the caller fences workers and stops the supervisor
before final export. Bytes are transported inertly, with no candidate imports
or execution. Hashes bind evidence; they are not signatures or verdicts.
"""
from __future__ import annotations

import asyncio
import base64
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat

from .terminal_deployment import PYTHON

SCHEMA = "terminal-declared-public-output-evidence@1"
FILES = {"bottle.py": 2 * 1024 * 1024, "report.jsonl": 256 * 1024}
MAX_ENVELOPE = 4 * 1024 * 1024
MAX_DIFF_BYTES = 64 * 1024
MAX_DIFF_LINES = 8000

# Deliberately standalone standard-library code. No import resolves against the
# candidate working directory; -I -S also disables user and site initialization.
CAPTURE_SCRIPT = r'''
import base64, hashlib, json, os, stat
root = "/app"
limits = {"bottle.py": 2097152, "report.jsonl": 262144}
identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
records = {}
try:
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
except OSError:
    directory = None
for name, maximum in limits.items():
    item = {"status": "unavailable", "reason": "app_directory_unavailable"}
    if directory is not None:
        try:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                item = {"status": "rejected", "reason": "not_independent_regular_file"}
            elif info.st_size > maximum:
                item = {"status": "rejected", "reason": "byte_bound_exceeded"}
            else:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                            or before.st_size > maximum or identity(info) != identity(before)):
                        raise ValueError("changed")
                    raw = stream.read(maximum + 1)
                    after = os.fstat(stream.fileno())
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (len(raw) > maximum or identity(before) != identity(after)
                        or identity(after) != identity(current) or not stat.S_ISREG(current.st_mode)
                        or current.st_nlink != 1):
                    raise ValueError("changed")
                item = {"status": "captured", "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "base64": base64.b64encode(raw).decode("ascii")}
        except FileNotFoundError:
            item = {"status": "missing"}
        except ValueError:
            item = {"status": "rejected", "reason": "changed_during_capture"}
        except OSError:
            item = {"status": "unavailable", "reason": "read_refused"}
    records[name] = item
if directory is not None:
    os.close(directory)
print(json.dumps({"schema": "terminal-declared-public-output-evidence@1", "files": records},
                 sort_keys=True, separators=(",", ":")))
'''


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _strict(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate public evidence field")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _read(path, maximum):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("canonical public evidence path required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise ValueError("bounded independent public evidence file required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(raw) > maximum or identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError("public evidence changed during read")
    return raw


def _write(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)
        os.fchmod(handle.fileno(), 0o444)
    return {"sha256": _sha(raw), "bytes": len(raw)}


def _root(logs_dir):
    logs_dir = Path(logs_dir).absolute()
    if logs_dir.resolve(strict=True) != logs_dir or not logs_dir.is_dir():
        raise ValueError("canonical existing benchmark log directory required")
    root = logs_dir / "public-output-evidence"
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("independent public evidence directory required")
    else:
        root.mkdir(mode=0o700)
    return root


def _validate_envelope(stdout):
    if not isinstance(stdout, str) or len(stdout.encode()) > MAX_ENVELOPE:
        raise ValueError("bounded public evidence envelope required")
    value = _strict(stdout)
    if (type(value) is not dict or set(value) != {"schema", "files"} or value["schema"] != SCHEMA
            or type(value["files"]) is not dict or set(value["files"]) != set(FILES)):
        raise ValueError("only exact declared public filenames may be captured")
    blobs, descriptors = {}, {}
    for name, item in value["files"].items():
        if type(item) is not dict:
            raise ValueError("public file descriptor required")
        if item.get("status") == "captured":
            if set(item) != {"status", "bytes", "sha256", "base64"}:
                raise ValueError("closed captured public file descriptor required")
            raw = base64.b64decode(item["base64"], validate=True)
            if (type(item["bytes"]) is not int or len(raw) != item["bytes"] or len(raw) > FILES[name]
                    or _sha(raw) != item["sha256"]):
                raise ValueError("captured public bytes differ from digest or bound")
            blobs[name] = raw
            descriptors[name] = {key: item[key] for key in ("status", "bytes", "sha256")}
        elif item == {"status": "missing"}:
            descriptors[name] = item
        elif (set(item) == {"status", "reason"} and (item["status"], item["reason"]) in {
                ("rejected", "not_independent_regular_file"), ("rejected", "byte_bound_exceeded"),
                ("rejected", "changed_during_capture"), ("unavailable", "app_directory_unavailable"),
                ("unavailable", "read_refused")}):
            descriptors[name] = item
        else:
            raise ValueError("unknown public capture status")
    return blobs, descriptors


async def _capture(environment, logs_dir, phase):
    root = _root(logs_dir)
    output = root / phase
    if output.exists():
        raise ValueError("public evidence stage already exists; never overwrite a trial")
    result = await asyncio.wait_for(environment.exec(
        command=shlex.join([PYTHON, "-I", "-S", "-c", CAPTURE_SCRIPT]),
        cwd="/", user="supervisor", timeout_sec=15), timeout=20)
    if result.return_code != 0:
        raise ValueError("public capture command failed")
    blobs, files = _validate_envelope(result.stdout)
    output.mkdir(mode=0o700)
    for name, raw in blobs.items():
        _write(output / name, raw)
    receipt = {"schema": SCHEMA, "phase": phase, "files": files,
        "implementation_sha256": _sha(Path(__file__).read_bytes()),
        "capture_script_sha256": _sha(CAPTURE_SCRIPT.encode()),
        "candidate_code_executed": False, "private_state_accessed": False,
        "verifier_state_accessed": False, "signed": False}
    digest = _write(output / "receipt.json", _json(receipt))
    return {**receipt, "receipt": str(output / "receipt.json"), "receipt_sha256": digest["sha256"]}


async def capture_public_inputs(environment, logs_dir) -> dict:
    """Capture the exact public baseline before the driver or worker starts."""
    return await _capture(environment, logs_dir, "before")


def _baseline(root):
    stage = root / "before"
    if not stage.exists():
        return None, {}
    receipt_raw = _read(stage / "receipt.json", 32_000)
    receipt = _strict(receipt_raw)
    if receipt.get("schema") != SCHEMA or receipt.get("phase") != "before" or set(receipt["files"]) != set(FILES):
        raise ValueError("public baseline receipt differs")
    blobs = {}
    for name, item in receipt["files"].items():
        if item.get("status") == "captured":
            raw = _read(stage / name, FILES[name])
            if len(raw) != item["bytes"] or _sha(raw) != item["sha256"]:
                raise ValueError("public baseline bytes differ from receipt")
            blobs[name] = raw
    return {**receipt, "receipt_sha256": _sha(receipt_raw)}, blobs


def _diff(before, after, name):
    left, right = before.decode("utf-8", errors="replace").splitlines(keepends=True), after.decode("utf-8", errors="replace").splitlines(keepends=True)
    if len(left) > MAX_DIFF_LINES or len(right) > MAX_DIFF_LINES:
        return None, {"status": "omitted", "reason": "line_bound_exceeded"}
    raw, truncated = bytearray(), False
    for line in difflib.unified_diff(left, right, fromfile="before/" + name, tofile="after/" + name, n=3):
        encoded = line.encode()
        if len(raw) + len(encoded) > MAX_DIFF_BYTES:
            raw.extend(encoded[:MAX_DIFF_BYTES - len(raw)])
            truncated = True
            break
        raw.extend(encoded)
    # Decoding for presentation may replace invalid UTF-8 or a final split
    # multibyte character. The independently retained source bytes are exact.
    return bytes(raw).decode("utf-8", errors="ignore").encode(), {"status": "captured", "truncated": truncated,
        "text_encoding": "utf-8 presentation; source bytes retained separately", "authoritative": False}


async def export_public_outputs(environment, logs_dir) -> dict:
    """Capture declared published output bytes after caller-enforced STOP.

    This function does not infer that STOP succeeded. Caller lifecycle receipts
    establish quiescence. Missing baselines or files are explicit observations.
    """
    root = _root(logs_dir)
    if (root / "receipt.json").exists():
        raise ValueError("final public evidence already exists; never overwrite a trial")
    baseline, before = _baseline(root)
    final = await _capture(environment, logs_dir, "after")
    diffs = {}
    diff_directory = None
    for name in FILES:
        old_status = (baseline or {}).get("files", {}).get(name, {}).get("status")
        new_status = final["files"][name]["status"]
        if old_status not in {"captured", "missing"} or new_status not in {"captured", "missing"}:
            diffs[name] = {"status": "unavailable", "reason": "baseline_or_output_unavailable"}
            continue
        old = before.get(name, b"")
        new = _read(root / "after" / name, FILES[name]) if new_status == "captured" else b""
        raw, descriptor = _diff(old, new, name)
        if raw is not None:
            if diff_directory is None:
                diff_directory = root / "diffs"
                diff_directory.mkdir(mode=0o700)
            descriptor.update(_write(diff_directory / (name + ".diff"), raw))
            descriptor["artifact"] = "diffs/" + name + ".diff"
        diffs[name] = descriptor
    instruction = Path(logs_dir).absolute() / "instruction.md"
    instruction_binding = None
    if instruction.exists():
        raw = _read(instruction, 256 * 1024)
        instruction_binding = {"sha256": _sha(raw), "bytes": len(raw), "signed": False}
    receipt = {"schema": SCHEMA, "status": "captured", "declared_files": list(FILES),
        "before_receipt_sha256": baseline["receipt_sha256"] if baseline else None,
        "after_receipt_sha256": final["receipt_sha256"], "files": final["files"], "diffs": diffs,
        "public_instruction": instruction_binding, "quiescence": "caller_lifecycle_receipt_required",
        "implementation_sha256": _sha(Path(__file__).read_bytes()),
        "candidate_code_executed": False, "private_state_accessed": False,
        "verifier_state_accessed": False, "signed": False}
    artifact = _write(root / "receipt.json", _json(receipt))
    return {**receipt, "receipt": str(root / "receipt.json"), "receipt_sha256": artifact["sha256"]}
