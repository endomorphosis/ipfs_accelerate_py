"""Compile a private, self-contained repository read set without a provider call.

Mandatory literal instructions and whole original files remain intact. Lexical
query results only nominate optional whole original files; they establish neither
task coverage nor acceptance. Counts cover the complete rendered user input,
not hidden native history, provider framing, a session hard cap or billing.

For shared use, supply both --scheduler-state and --scheduler-policy-sha256 to
attach to the selected existing native scheduler without creating state or
changing its policy. Omission preserves native default scheduler behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

SCHEMA = "repository-read-set-cli@1"
MAX_REQUIREMENT_BYTES = 16384
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_REQUIRED_PATHS = 8
MAX_COUNTER = (1 << 63) - 1
MAX_SCHEDULER_STATE_BYTES = 8 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}\Z")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _absolute(value):
    path = Path(value)
    if ".." in path.parts or len(str(path).encode("utf-8")) > 4096:
        raise ValueError("bounded explicit path required")
    return Path(os.path.abspath(path))


def _directory(path):
    """Open every existing ancestor without following a symlink."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _relative(value):
    if (type(value) is not str or not value or len(value.encode("utf-8")) > 1024
            or "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or value.startswith("/")
            or any(part in {"", ".", "..", ".git"} for part in value.split("/"))):
        raise ValueError("canonical relative source path required")
    return value


def _capture_requirements(path):
    parent = _directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or not 1 <= before.st_size <= MAX_REQUIREMENT_BYTES):
                raise ValueError("bounded single-link requirements file required")
            chunks, size = [], 0
            while size <= MAX_REQUIREMENT_BYTES:
                chunk = os.read(descriptor, MAX_REQUIREMENT_BYTES + 1 - size)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            after = os.fstat(descriptor)
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            witness = lambda info: tuple(getattr(info, key) for key in (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))
            if size != before.st_size or witness(before) != witness(after) or witness(after) != witness(current):
                raise ValueError("requirements changed during capture")
            raw = b"".join(chunks)
            text = raw.decode("utf-8")
            if not text.strip() or "\x00" in text:
                raise ValueError("literal UTF-8 requirements required")
            return raw, text
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _output_parent(output, root):
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("output must be disjoint from source")
    parent = _directory(output.parent)
    try:
        try:
            os.stat(output.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return parent
        raise ValueError("fresh output directory required")
    except BaseException:
        os.close(parent)
        raise


def _scheduler_input(state, policy):
    if (state is None) != (policy is None):
        raise ValueError("paired existing scheduler state and policy required")
    if state is None:
        return None
    if type(policy) is not str or re.fullmatch(r"[0-9a-f]{64}", policy) is None:
        raise ValueError("exact existing scheduler policy SHA256 required")
    path = _absolute(state)
    parent = _directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            info = os.fstat(descriptor)
            visible = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or not 0 < info.st_size <= MAX_SCHEDULER_STATE_BYTES
                    or (info.st_dev, info.st_ino) != (visible.st_dev, visible.st_ino)):
                raise ValueError("bounded exact existing scheduler state required")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    return path


def _publish(output, root, prompt, receipt):
    """Write only our new private directory, using pinned directory descriptors."""
    parent = _output_parent(output, root)
    parent_info = os.fstat(parent)
    parent_identity = (parent_info.st_dev, parent_info.st_ino)
    directory, identity, created = None, None, []
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent)
        directory = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        info = os.fstat(directory)
        identity = (info.st_dev, info.st_ino)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("private output directory required")
        for name, raw in (("prompt.txt", prompt), ("receipt.json", receipt)):
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory)
            created.append(name)
            try:
                remaining = memoryview(raw)
                while remaining:
                    count = os.write(descriptor, remaining)
                    if count <= 0:
                        raise ValueError("output write refused")
                    remaining = remaining[count:]
                os.fsync(descriptor)
                written = os.fstat(descriptor)
                visible = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (not stat.S_ISREG(written.st_mode) or written.st_nlink != 1
                        or stat.S_IMODE(written.st_mode) != 0o600 or written.st_uid != os.getuid()
                        or written.st_size != len(raw)
                        or (written.st_dev, written.st_ino) != (visible.st_dev, visible.st_ino)):
                    raise ValueError("output custody changed")
            finally:
                os.close(descriptor)
        visible = os.stat(output.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISDIR(visible.st_mode) or (visible.st_dev, visible.st_ino) != identity:
            raise ValueError("output directory changed")
        closing = _directory(output.parent)
        try:
            current = os.fstat(closing)
            if (current.st_dev, current.st_ino) != parent_identity:
                raise ValueError("output parent changed")
        finally:
            os.close(closing)
        os.fsync(directory)
    except BaseException:
        if directory is not None:
            for name in created:
                try:
                    os.unlink(name, dir_fd=directory)
                except OSError:
                    pass
        if identity is not None:
            try:
                visible = os.stat(output.name, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISDIR(visible.st_mode) and (visible.st_dev, visible.st_ino) == identity:
                    os.rmdir(output.name, dir_fd=parent)
            except OSError:
                pass
        raise
    finally:
        if directory is not None:
            os.close(directory)
        os.close(parent)


def run_repository_read_set(*, root, query, required_paths, requirements_file,
        task_id, provider, model, max_input_tokens, output, limit=8,
        scheduler_state=None, scheduler_policy_sha256=None):
    """Compile through the actual native session and automatic offline resolver."""
    root, output, requirements_file = map(_absolute, (root, output, requirements_file))
    if any(type(value) is not str or _ID.fullmatch(value) is None for value in (task_id, provider, model)):
        raise ValueError("explicit bounded task and provider/model identifiers required")
    if type(query) is not str or not query.strip() or len(query.encode("utf-8")) > 4096:
        raise ValueError("bounded query required")
    if type(required_paths) not in {list, tuple} or len(required_paths) > MAX_REQUIRED_PATHS:
        raise ValueError("bounded required path population required")
    required_paths = tuple(_relative(path) for path in required_paths)
    if len(set(required_paths)) != len(required_paths):
        raise ValueError("unique required paths required")
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("bounded query population required")
    if type(max_input_tokens) is not int or not 1 <= max_input_tokens <= MAX_COUNTER:
        raise ValueError("positive local input budget required")
    scheduler_path = _scheduler_input(scheduler_state, scheduler_policy_sha256)
    descriptor = _directory(root)
    os.close(descriptor)
    parent = _output_parent(output, root)
    os.close(parent)
    required_raw, required_text = _capture_requirements(requirements_file)
    temporary_root = Path("/tmp")
    descriptor = _directory(temporary_root)
    os.close(descriptor)
    if (root == temporary_root or root in temporary_root.parents
            or any(os.path.lexists(path / ".git") for path in (temporary_root, *temporary_root.parents))):
        raise ValueError("private state must be outside Git")
    from ipfs_accelerate_py.agent_supervisor.runtime.automatic_prompt_tokenizer import resolve_prompt_tokenizer
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_read_set import compile_repository_read_set
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    index_factory = None
    if scheduler_path is not None:
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
        owner = open_existing_scheduler(scheduler_path, scheduler_policy_sha256)
        index_factory = lambda path: NativeRepositoryIndex(path, scheduler=owner)
    counter, provenance = resolve_prompt_tokenizer(provider=provider, model=model)
    with tempfile.TemporaryDirectory(prefix="repository-read-set-", dir=temporary_root) as temporary:
        with RepositorySourceSession(root, Path(temporary) / "state", representation="original",
                index_factory=index_factory) as session:
            before = session.status()
            result = compile_repository_read_set(session=session, generation=before["generation"], query=query,
                required_paths=required_paths, required_texts=(required_text,), task_id=task_id,
                provider=provider, model=model, workspace=root, token_counter=counter,
                max_input_tokens=max_input_tokens, limit=limit)
            after = session.status()
            if (after.get("source_current") is not True or after.get("refresh_required") is not False
                    or any(before.get(key) != after.get(key) for key in ("session_id", "generation",
                        "source_tree_sha256", "package_manifest_sha256", "datasets_head", "semantic_state_cid"))):
                raise ValueError("source generation changed")
            if _capture_requirements(requirements_file)[0] != required_raw:
                raise ValueError("literal requirements changed")
            prompt = result.selected_prompt.encode("utf-8")
            receipt = dict(schema=SCHEMA, read_set=result.receipt, tokenizer_provenance=provenance,
                resource_scheduler_binding=dict(mode="existing_pinned" if scheduler_path is not None else "native_default",
                    state_path_sha256=None if scheduler_path is None else _sha(str(scheduler_path).encode()),
                    policy_sha256=scheduler_policy_sha256, policy_reconfiguration_requested=False,
                    state_creation_requested=False if scheduler_path is not None else None),
                requirements_sha256=_sha(required_raw), requirements_bytes=len(required_raw),
                prompt_sha256=_sha(prompt), prompt_bytes=len(prompt), source_generation=after["generation"],
                source_tree_sha256=after["source_tree_sha256"],
                package_manifest_sha256=after["package_manifest_sha256"], source_current_at_close=True,
                source_guard_scope="cooperative snapshot checks; no writer lease", provider_invoked=False,
                native_history=False, task_acceptance=False, billing_total_verified=False,
                whole_session_token_savings=None, raw_source_authoritative=True,
                source_bodies_in_receipt=False, proof_authority=False, execution_authority=False,
                completion_authority=False)
            encoded = _json(receipt)
            if len(encoded) > MAX_RECEIPT_BYTES or len(prompt) > 256000:
                raise ValueError("bounded output required")
    _publish(output, root, prompt, encoded)
    return dict(schema=SCHEMA, status="written", prompt_sha256=_sha(prompt), prompt_bytes=len(prompt),
        receipt_sha256=_sha(encoded), receipt_bytes=len(encoded), provider_invoked=False,
        task_acceptance=False, source_bodies_exported_to_stdout=False)


class _Arguments(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("explicit bounded arguments required")


def main(argv=None):
    parser = _Arguments(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--required-path", action="append", default=[])
    parser.add_argument("--requirements-file", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scheduler-state")
    parser.add_argument("--scheduler-policy-sha256")
    try:
        arguments = vars(parser.parse_args(argv))
        arguments["required_paths"] = arguments.pop("required_path")
        result = run_repository_read_set(**arguments)
    except Exception:
        print(_json(dict(schema=SCHEMA, status="refused", reason="invalid_input_or_local_failure")).decode(), end="")
        return 2
    print(_json(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
