"""Detached archive mutation controls for the finite advisory join auditor.

This stdlib-only runner starts the auditor in fresh Python processes. Untouched
archive files share read-only hardlinks; each mutation target is unlinked from
its clone and recreated as an independent file before any chmod or write.
Full input byte/mode inventories are compared before and after all calls.
Symlinks are inert raw-target metadata; neither inventory nor cloning reads
their targets. The auditor continues to refuse symlink evidence paths.
No trainer, native owner, worker, inference or prover is started by this runner.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time

SCHEMA = "finite-advisory-join-detached-tamper-controls@1"
AUDIT_SCHEMA = "finite-repository-advisory-join-independent-audit@1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FILES = 200_000
MAX_LINK_BYTES = 16 * 1024


class TamperControlError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise TamperControlError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES,
              "bounded regular file required: " + str(path))
        parts, total = [], 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            parts.append(block)
            total += len(block)
            _need(total <= MAX_FILE_BYTES, "file grew beyond read bound")
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        _need(identity(before) == identity(after) == identity(current), "file changed during detached read")
        return b"".join(parts), before
    finally:
        os.close(descriptor)


def _write(path, value):
    _need(not path.exists() and not path.is_symlink(), "fresh output file required")
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _link(path):
    """Read only the link's own inode and raw target string, never its target."""
    before = path.stat(follow_symlinks=False)
    _need(stat.S_ISLNK(before.st_mode), "inert symlink disposition requires a symlink inode")
    raw = os.readlink(os.fsencode(path))
    after = path.stat(follow_symlinks=False)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_mode)
    _need(identity(before) == identity(after) and len(raw) == before.st_size
          and len(raw) <= MAX_LINK_BYTES, "raw symlink metadata changed or exceeds bound")
    return raw, {"kind": "symlink", "bytes": len(raw), "sha256": _sha(raw),
                 "target_raw_hex": raw.hex(), "target_sha256": _sha(raw),
                 "mode": stat.S_IMODE(before.st_mode), "target_dereferenced": False}


def _entries(root):
    """Enumerate lstat entries without descending or classifying link targets."""
    pending = [root]
    while pending:
        directory = pending.pop()
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with os.scandir(descriptor) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
            for entry in entries:
                path = directory / entry.name
                info = path.stat(follow_symlinks=False)
                yield path, info
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
        finally:
            os.close(descriptor)


def _inventory(root):
    rows = {}
    for path, info in _entries(root):
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISLNK(info.st_mode):
            _, row = _link(path)
        else:
            _need(stat.S_ISREG(info.st_mode), "input namespace contains a special file: " + str(path))
            raw, stable = _read(path)
            row = {"kind": "regular_file", "bytes": len(raw), "sha256": _sha(raw),
                   "mode": stat.S_IMODE(stable.st_mode)}
        rows[path.relative_to(root).as_posix()] = row
        _need(len(rows) <= MAX_FILES, "input namespace exceeds file count bound")
    _need(rows, "input namespace has no artifacts")
    return rows


def _clone_readonly(source, destination, original):
    """Share unchanged regular files; use byte copies only across filesystems.

    No permission change or write is performed through a hardlinked path.
    The fresh auditor is read-only. Full original hashes bracket all controls.
    """
    destination.mkdir(mode=0o700)
    shared, copied, links = 0, 0, 0
    for relative, expected in original.items():
        old = source / relative
        new = destination / relative
        new.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if expected["kind"] == "symlink":
            raw, row = _link(old)
            _need(row == expected, "original inert symlink metadata changed while cloning")
            os.symlink(raw, os.fsencode(new))
            _, cloned = _link(new)
            _need(cloned == expected, "cloned inert symlink raw target/mode differs")
            links += 1
            continue
        _need(expected["kind"] == "regular_file", "unsupported cloned entry disposition")
        old_info = old.stat(follow_symlinks=False)
        _need(stat.S_ISREG(old_info.st_mode) and old_info.st_size == expected["bytes"]
              and stat.S_IMODE(old_info.st_mode) == expected["mode"],
              "original archive file metadata changed while cloning controls")
        try:
            os.link(old, new, follow_symlinks=False)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            raw, old_info = _read(old)
            _need(len(raw) == expected["bytes"] and _sha(raw) == expected["sha256"],
                  "original archive changed during cross-filesystem byte copy")
            descriptor = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                new_info = os.fstat(descriptor)
                _need((old_info.st_dev, old_info.st_ino) != (new_info.st_dev, new_info.st_ino)
                      and new_info.st_nlink == 1, "fallback copy shares original inode")
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                # This descriptor was just proved independent of the source.
                os.fchmod(descriptor, expected["mode"])
            finally:
                os.close(descriptor)
            copied += 1
        else:
            new_info = new.stat(follow_symlinks=False)
            _need(stat.S_ISREG(new_info.st_mode)
                  and (old_info.st_dev, old_info.st_ino) == (new_info.st_dev, new_info.st_ino)
                  and new_info.st_size == expected["bytes"]
                  and stat.S_IMODE(new_info.st_mode) == expected["mode"],
                  "read-only hardlink clone differs from original regular file")
            shared += 1
    return {"strategy": "readonly_hardlinks_with_unlink_and_recreate_before_mutation",
            "hardlinked_readonly_files": shared, "cross_filesystem_byte_copies": copied,
            "inert_symlinks_recreated": links, "symlink_targets_dereferenced": False,
            "whole_tree_copy_fsync_performed": False}


def _mapped(root, locator):
    _need(type(locator) is str and locator and "\0" not in locator, "invalid control artifact locator")
    path = PurePosixPath(locator)
    _need(".." not in path.parts, "control artifact locator escapes archive")
    mappings = {"/results": root, "/opt/ipfs-supervisor/source": root / "source",
        "/opt/ipfs-supervisor/datasets": root / "datasets", "/opt/ipfs-supervisor/kit": root / "kit",
        "/opt/ipfs-supervisor/finite-handoffs": root / "handoffs"}
    if path.is_absolute():
        selected = None
        for prefix, destination in mappings.items():
            if path == PurePosixPath(prefix) or path.is_relative_to(prefix):
                selected = destination / path.relative_to(prefix)
                break
        _need(selected is not None, "control locator is outside copied container mounts")
        return selected
    return root / path


def _mutate(original_root, copied_root, relative, transform):
    source = original_root / relative
    target = copied_root / relative
    original, source_info = _read(source)
    before, target_info = _read(target)
    _need(original == before, "read-only clone mutation preimage differs from original")
    after, detail = transform(before)
    _need(type(after) is bytes and after != before and len(after) <= MAX_FILE_BYTES, "control must actually change bounded bytes")
    # Remove the clone directory entry first. Never truncate or chmod its
    # possibly shared inode. A new exclusive file supplies the mutation.
    current = target.stat(follow_symlinks=False)
    _need((current.st_dev, current.st_ino) == (target_info.st_dev, target_info.st_ino)
          and stat.S_ISREG(current.st_mode), "control target changed before detachment")
    target.unlink()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        current = os.fstat(descriptor)
        source_current = source.stat(follow_symlinks=False)
        _need((current.st_dev, current.st_ino) != (source_current.st_dev, source_current.st_ino)
              and current.st_nlink == 1, "recreated mutation target is not an independent inode")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(after)
            stream.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    retained, _ = _read(target)
    _need(retained == after and _read(source)[0] == original, "detached mutation changed original bytes")
    return {"relative_path": relative, "before_sha256": _sha(before), "after_sha256": _sha(after),
        "before_bytes": len(before), "after_bytes": len(after), "detached_inode_verified": True,
        "original_bytes_still_exact": True, "mutation": detail}


def _json_change(raw, mutate):
    value = json.loads(raw)
    detail = mutate(value)
    return _json_bytes(value), detail


def _signed_context(raw):
    def mutate(value):
        context = value["receipt"]["payload"]["semantic_context"]
        previous = context["domain_inputs"][-1]
        context["domain_inputs"][-1] = previous + 1
        return {"field": "receipt.payload.semantic_context.domain_inputs[4]", "before": previous, "after": previous + 1,
                "signature_rewritten": False}
    return _json_change(raw, mutate)


def _checkpoint(raw):
    def mutate(value):
        def change(node, path):
            if type(node) is dict:
                for key, child in node.items():
                    if type(child) is float:
                        _need(math.isfinite(child), "original checkpoint has nonfinite weight")
                        node[key] = child + .125
                        return {"field": path + "." + key, "before": child, "after": node[key]}
                    result = change(child, path + "." + key)
                    if result is not None:
                        return result
            elif type(node) is list:
                for position, child in enumerate(node):
                    if type(child) is float:
                        _need(math.isfinite(child), "original checkpoint has nonfinite weight")
                        node[position] = child + .125
                        return {"field": path + "[" + str(position) + "]", "before": child, "after": node[position]}
                    result = change(child, path + "[" + str(position) + "]")
                    if result is not None:
                        return result
            return None
        result = change(value["state"]["parameters"], "state.parameters")
        _need(result is not None, "actual checkpoint has no numerical parameter to mutate")
        result["retained_pin_rewritten"] = False
        return result
    return _json_change(raw, mutate)


def _replacement(raw):
    _need(b"return n + 2" in raw, "retained replacement is outside the authored fixture")
    changed = raw.replace(b"return n + 2", b"return n + 3", 1)
    return changed, {"field": "actual replacement source", "before": "return n + 2", "after": "return n + 3",
                     "retained_pin_rewritten": False}


def _bridge_revision(raw):
    def mutate(value):
        previous = value["task_revision"]
        _need(type(previous) is int and previous >= 1, "bridge lacks a real observed ready revision")
        value["task_revision"] = previous + 1
        return {"field": "task_revision", "before": previous, "after": previous + 1,
                "public_descriptor_rewritten": False}
    return _json_change(raw, mutate)


def _task_population(raw):
    def mutate(value):
        previous = value["graph"]["tasks"]
        _need(len(previous) == 2 and {row["task_key"] for row in previous} == {"FINITE-TYPE", "FINITE-OFFSET"},
              "original archive lacks the complete two-task population")
        value["graph"]["tasks"] = [row for row in previous if row["task_key"] != "FINITE-TYPE"]
        return {"field": "graph.tasks", "removed_original_task": "FINITE-TYPE", "before_count": 2, "after_count": 1,
                "signed_receipt_rewritten": False}
    return _json_change(raw, mutate)


def _audit(auditor, namespace, output):
    start = time.monotonic()
    with (output / "stdout.json").open("xb") as stdout, (output / "stderr.txt").open("xb") as stderr:
        process = subprocess.run([sys.executable, str(auditor), str(namespace)],
            stdout=stdout, stderr=stderr, timeout=300, check=False)
    stdout_raw, _ = _read(output / "stdout.json")
    stderr_raw, _ = _read(output / "stderr.txt")
    try:
        payload = json.loads(stdout_raw)
    except (ValueError, UnicodeError) as error:
        raise TamperControlError("fresh auditor did not return a JSON result") from error
    return {"returncode": process.returncode, "elapsed_seconds": time.monotonic() - start,
            "result": payload, "stdout_sha256": _sha(stdout_raw), "stderr_sha256": _sha(stderr_raw),
            "stderr_bytes": len(stderr_raw), "fresh_process": True}


def run(namespace, output, *, auditor=None):
    namespace = Path(namespace).absolute()
    output = Path(output).absolute()
    auditor = (Path(auditor) if auditor is not None else Path(__file__).with_name("audit_finite_repository_advisory_join.py")).absolute()
    _need(namespace.resolve(strict=True) == namespace and namespace.is_dir(), "canonical completed archive required")
    _need(auditor.resolve(strict=True) == auditor and auditor.is_file(), "canonical frozen auditor required")
    _need(not output.exists() and output.parent.resolve(strict=True) == output.parent
          and not output.is_relative_to(namespace), "fresh control output must be outside original archive")
    output.mkdir(mode=0o700)
    before = _inventory(namespace)
    _write(output / "original-before.json", before)
    auditor_raw, _ = _read(auditor)
    report = {"schema": SCHEMA, "status": "incomplete", "original_namespace": str(namespace),
        "output": str(output), "auditor": {"path": str(auditor), "sha256": _sha(auditor_raw), "bytes": len(auditor_raw)},
        "interpreter": str(Path(sys.executable).resolve()), "controls": [], "native_jobs_started": 0,
        "fitting_performed": False, "inference_performed": False, "provers_started": 0,
        "scope": "Fresh read-only auditor processes over readonly hardlink clones; each modified target is unlinked and recreated independently. Symlinks retain inert raw target metadata and are never dereferenced by this runner. Original bytes/modes/link targets are preserved; hardlink counts may change."}
    caught = None
    try:
        baseline_directory = output / "baseline"
        baseline_directory.mkdir(mode=0o700)
        baseline = _audit(auditor, namespace, baseline_directory)
        report["baseline"] = baseline
        _need(baseline["returncode"] == 0 and baseline["result"]["schema"] == AUDIT_SCHEMA
              and baseline["result"]["status"] == "verified", "original completed archive did not pass baseline auditor")
        generated = json.loads(_read(namespace / "native/generated-candidate.json")[0])
        replacement = _mapped(namespace, generated["artifacts"]["replacement"]["path"]).relative_to(namespace).as_posix()
        controls = (
            ("signed_context", "native/before-admission.json", _signed_context, "signature"),
            ("selected_checkpoint_weight", "native/private/advisory/successor-child-frozen-context/checkpoint.json", _checkpoint, "advisory retained bytes"),
            ("generated_proposal_replacement", replacement, _replacement, "raw artifact pin"),
            ("native_task_bridge_revision", "native/candidate-bridge.json", _bridge_revision, "bridge changed native task/revision"),
            ("original_task_population", "native/before-admission.json", _task_population, "signed admission joins"),
        )
        for label, relative, transform, expected_error in controls:
            directory = output / label
            directory.mkdir(mode=0o700)
            copied = directory / "copied-namespace"
            clone = _clone_readonly(namespace, copied, before)
            mutation = _mutate(namespace, copied, relative, transform)
            result = _audit(auditor, copied, directory)
            refused = (result["returncode"] != 0 and result["result"].get("schema") == AUDIT_SCHEMA
                       and result["result"].get("status") == "refused"
                       and expected_error in result["result"].get("error", ""))
            row = {"control": label, "clone": clone, "mutation": mutation, "audit": result,
                   "expected_refusal": expected_error, "refused_at_expected_boundary": refused}
            report["controls"].append(row)
            _write(directory / "control.json", row)
            _need(refused, "detached mutation did not refuse at expected boundary: " + label)
        _need(_read(auditor)[0] == auditor_raw, "frozen auditor changed during controls")
        report["status"] = "verified"
        report["baseline_verified"] = True
        report["all_detached_mutations_refused"] = True
    except BaseException as error:
        caught = error
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    finally:
        try:
            after = _inventory(namespace)
            _write(output / "original-after.json", after)
            report["original_namespace_bytes_and_modes_unchanged"] = before == after
            report["original_file_count"] = len(before)
            report["original_total_bytes"] = sum(row["bytes"] for row in before.values())
            report["original_regular_file_count"] = sum(row["kind"] == "regular_file" for row in before.values())
            report["original_inert_symlink_count"] = sum(row["kind"] == "symlink" for row in before.values())
            report["original_symlink_targets_unchanged"] = {
                name: row for name, row in before.items() if row["kind"] == "symlink"} == {
                name: row for name, row in after.items() if row["kind"] == "symlink"}
            _need(before == after, "original input archive changed during detached tamper controls")
        except BaseException as error:
            caught = error
            report["status"] = "failed"
            report["original_namespace_bytes_and_modes_unchanged"] = False
            report["preservation_error_type"] = type(error).__name__
            report["preservation_error"] = str(error)
        _write(output / "result.json", report)
    if caught is not None:
        raise TamperControlError("retained detached tamper qualification failed: " + str(caught)) from caught
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--auditor", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.namespace, args.output, auditor=args.auditor)
    except (TamperControlError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"schema": result["schema"], "status": result["status"], "control_count": len(result["controls"]),
                      "original_namespace_bytes_and_modes_unchanged": result["original_namespace_bytes_and_modes_unchanged"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
