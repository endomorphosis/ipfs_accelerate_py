"""Exact readonly callable capsule copy beneath root's genuine prep owner.

No product import, copied source execution, native admission or subprocess.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat
import sys

HERE = Path(__file__).absolute().parent
PHASE = HERE.parent
RUNS = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs')
SOURCE_ROOT = RUNS / 'legal-ir-ordinary-handoff-native-20261017-r01/preparation/capsule'
SOURCE_MANIFEST_SHA = 'b2a78de35845425f533f7a361b3998f41628d472fafbbccb3aa346d7b50e0925'
ATTEMPT = RUNS / 'legal-ir-callable-handoff-20261017-r01/preparation'
CAPSULE = ATTEMPT / 'capsule'
RUNNER_MANIFEST_SHA = 'b56ff4909d5b8adb6ffe9521fbfc96e60376d6d41b19dde1b0bddb7775757c6e'
HANDOFF_MANIFEST_SHA = 'f7d06829c5c49b45171baf8a5d80c9f28b91fb49950a4b42f6ca11f117693b18'
MAX_FILES, MAX_PAYLOAD, MAX_SOURCE = 4096, 128 * 1024 * 1024, 16 * 1024 * 1024
RUNTIME = 'ipfs_accelerate_py/agent_supervisor/runtime/'
OVERRIDES = {
 RUNTIME + 'grok_cli_runner.py': PHASE / 'runner-v02/proposed' / RUNTIME / 'grok_cli_runner.py',
 RUNTIME + 'ordinary_success_handoff.py': PHASE / 'handoff-v02/proposed' / RUNTIME / 'ordinary_success_handoff.py',
 RUNTIME + 'ordinary_custody_hold.py': PHASE / 'handoff-v02/proposed' / RUNTIME / 'ordinary_custody_hold.py',
}

def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def canonical(path):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and '..' not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical unaliased absolute source/output required')
    return path


def metadata(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid, info.st_nlink)


def read_source(path, *, expected_size=None, expected_sha=None, output=None, maximum=MAX_SOURCE):
    path = canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, count, chunks = hashlib.sha256(), 0, []
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 <= before.st_size <= maximum,
             'bounded stable owned regular source required')
        if expected_size is not None:
            need(type(expected_size) is int and before.st_size == expected_size, 'source size differs from frozen inventory')
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            need(count <= before.st_size, 'source grew during copy/read')
            digest.update(chunk)
            if output is not None:
                output.write(chunk)
            elif expected_sha is None:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    need(count == before.st_size and metadata(before) == metadata(after) == metadata(path.lstat()),
         'source changed during copy/read')
    if expected_sha is not None:
        need(type(expected_sha) is str and len(expected_sha) == 64 and digest.hexdigest() == expected_sha,
             'source SHA differs from frozen inventory')
    return b''.join(chunks) if expected_sha is None and output is None else {'bytes': count, 'sha256': digest.hexdigest()}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path, raw, mode=0o444):
    path = canonical(path)
    need(path.parent.is_dir() and type(raw) is bytes and len(raw) <= 4 * 1024 * 1024, 'bounded new output required')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, 'wb') as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def load_inventory(path, expected):
    raw = read_source(path, maximum=4 * 1024 * 1024)
    need(sha(raw) == expected, "reviewed_callable_inventory_changed")
    inventory = strict_json(raw)
    need(inventory.get("schema") == "legal-ir-callable-capsule-source-inventory@1"
         and inventory.get("source_root") == str(SOURCE_ROOT)
         and inventory.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA
         and inventory.get("runner_manifest_sha256") == RUNNER_MANIFEST_SHA
         and inventory.get("handoff_manifest_sha256") == HANDOFF_MANIFEST_SHA,
         "exact_reviewed_source_generations_required")
    base_raw = read_source(SOURCE_ROOT / "SOURCE_MANIFEST.json", maximum=4 * 1024 * 1024)
    need(sha(base_raw) == SOURCE_MANIFEST_SHA, "actual_original_readonly_capsule_changed")
    base = strict_json(base_raw)
    need(base.get("schema") == "legal-ir-native-handoff-capsule@1"
         and base.get("capsule_root") == str(SOURCE_ROOT) and base.get("payload_file_count") == 2806,
         "actual_original_capsule_population_required")
    originals = {row["path"]: row for row in base["files"]}
    override_rows = {}
    for role, digest in (("runner-v02", RUNNER_MANIFEST_SHA), ("handoff-v02", HANDOFF_MANIFEST_SHA)):
        manifest_raw = read_source(PHASE / role / "SOURCE_MANIFEST.json", maximum=4 * 1024 * 1024)
        need(sha(manifest_raw) == digest, "frozen_callable_source_manifest_changed")
        for row in strict_json(manifest_raw)["files"]:
            relative = row["path"]
            if relative.startswith("proposed/") and relative[9:] in OVERRIDES:
                need(relative[9:] not in override_rows, "duplicate_callable_override")
                override_rows[relative[9:]] = row
    need(set(override_rows) == set(OVERRIDES)
         and RUNTIME + "ordinary_custody_hold.py" not in originals,
         "exact_three_callable_overrides_required")
    rows = inventory.get("files")
    need(type(rows) is list and len(rows) == inventory.get("payload_file_count") == 2807,
         "complete_2807_file_callable_capsule_required")
    names, total = [], 0
    for row in rows:
        need(type(row) is dict and set(row) == {"path", "source_path", "origin", "bytes", "sha256"},
             "closed_callable_source_row_required")
        relative = PurePosixPath(row["path"])
        need(not relative.is_absolute() and relative.as_posix() == row["path"] and relative.suffix == ".py"
             and relative.parts[0] == "ipfs_accelerate_py" and ".." not in relative.parts
             and ".git" not in relative.parts and "__pycache__" not in relative.parts,
             "canonical_python_only_capsule_required")
        if row["path"] in OVERRIDES:
            reference = override_rows[row["path"]]
            need(row["origin"] == "frozen_callable_override"
                 and row["source_path"] == str(OVERRIDES[row["path"]]), "exact_callable_override_path_required")
        else:
            reference = originals.get(row["path"])
            need(reference is not None and row["origin"] == "original_readonly_capsule"
                 and row["source_path"] == str(SOURCE_ROOT / row["path"]), "exact_original_capsule_path_required")
        need(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_SOURCE
             and row["bytes"] == reference["bytes"] and row["sha256"] == reference["sha256"],
             "actual_frozen_callable_source_row_differs")
        names.append(row["path"])
        total += row["bytes"]
    need(names == sorted(set(originals) | set(OVERRIDES))
         and total == inventory.get("payload_bytes") <= MAX_PAYLOAD, "unique_complete_bounded_callable_payload_required")
    return inventory, raw


def main():
    need(len(sys.argv) == 3, "reviewed_source_inventory_and_sha_required")
    need(canonical(os.environ["TMPDIR"]) == ATTEMPT, "actual_outer_preparation_owner_required")
    info = ATTEMPT.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
         "actual_private_preparation_attempt_required")
    inventory_path = canonical(sys.argv[1])
    need(inventory_path == HERE / "SOURCE_INVENTORY.json", "fixed_reviewed_inventory_path_required")
    inventory, raw_inventory = load_inventory(inventory_path, sys.argv[2])
    need(not os.path.lexists(CAPSULE), "fresh_callable_capsule_required")
    def guard():
        checked, raw = load_inventory(inventory_path, sys.argv[2])
        need(checked == inventory and raw == raw_inventory, "callable_inventory_changed")
        for row in inventory["files"]:
            read_source(row["source_path"], expected_size=row["bytes"], expected_sha=row["sha256"])
    guard()
    CAPSULE.mkdir(mode=0o700)
    copied = []
    for row in inventory["files"]:
        destination = canonical(CAPSULE / row["path"])
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            pin = read_source(row["source_path"], expected_size=row["bytes"], expected_sha=row["sha256"], output=stream)
            os.fchmod(stream.fileno(), 0o444)
            stream.flush()
            os.fsync(stream.fileno())
        read_source(destination, expected_size=row["bytes"], expected_sha=row["sha256"])
        copied.append({"path": row["path"], **pin})
    guard()
    manifest = {"schema": "legal-ir-callable-handoff-capsule@1", "capsule_root": str(CAPSULE),
        "source_inventory_sha256": sha(raw_inventory), "base_capsule_manifest_sha256": SOURCE_MANIFEST_SHA,
        "runner_source_manifest_sha256": RUNNER_MANIFEST_SHA, "handoff_source_manifest_sha256": HANDOFF_MANIFEST_SHA,
        "payload_file_count": len(copied), "payload_bytes": sum(row["bytes"] for row in copied), "files": copied,
        "provider_calls_performed": False, "production_hook_deployed": False, "canonical_tasks_accepted": 0,
        "acceptance_authority": False, "completion_authority": False}
    manifest_raw = wire(manifest)
    write_new(CAPSULE / "SOURCE_MANIFEST.json", manifest_raw)
    directories = [Path(path) for path, _dirs, _files in os.walk(CAPSULE, followlinks=False)]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        canonical(directory)
        fsync_directory(directory)
        directory.chmod(0o500)
    fsync_directory(ATTEMPT)
    guard()
    outcome = {"schema": "legal-ir-callable-capsule-preparation@1", "passed": True,
        "capsule_root": str(CAPSULE), "capsule_manifest_sha256": sha(manifest_raw),
        "source_inventory_sha256": sha(raw_inventory), "payload_file_count": len(copied),
        "payload_bytes": manifest["payload_bytes"], "all_files_fsynced_readonly": True,
        "original_native_preparation_closeout_pending": True, "actual_public_wrapper_qualified": False,
        "provider_calls_performed": False, "canonical_tasks_accepted": 0, "acceptance_authority": False}
    write_new(ATTEMPT / "capsule-preparation.json", wire(outcome), mode=0o600)
    print(json.dumps(outcome, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


