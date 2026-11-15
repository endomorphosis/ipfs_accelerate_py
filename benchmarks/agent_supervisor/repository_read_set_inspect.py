"""Make one live, guarded Grok inspection call from a bounded original read set.

Run from the explicit source root with the project available on Python's import
path. This command contacts the selected provider; repository_read_set remains
the offline preparation command. Tool-off settings are requested policy, not
verified native containment. No task acceptance or whole-session saving follows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile

from . import repository_read_set as preparation

SCHEMA = "repository-read-set-inspection-cli@1"
MAX_RESPONSE_BYTES = 1024 * 1024


def _publish(output, root, files):
    """Publish only a fresh private directory; never replace existing evidence."""
    parent = preparation._output_parent(output, root)
    parent_info = os.fstat(parent)
    directory, identity, created = None, None, []
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent)
        directory = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        info = os.fstat(directory)
        identity = info.st_dev, info.st_ino
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("private inspection output required")
        for name, raw in files.items():
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory)
            created.append(name)
            try:
                remaining = memoryview(raw)
                while remaining:
                    count = os.write(descriptor, remaining)
                    if count <= 0:
                        raise ValueError("inspection publication refused")
                    remaining = remaining[count:]
                os.fsync(descriptor)
                written = os.fstat(descriptor)
                visible = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (not stat.S_ISREG(written.st_mode) or written.st_nlink != 1
                        or stat.S_IMODE(written.st_mode) != 0o600 or written.st_uid != os.getuid()
                        or written.st_size != len(raw)
                        or (written.st_dev, written.st_ino) != (visible.st_dev, visible.st_ino)):
                    raise ValueError("inspection output custody changed")
            finally:
                os.close(descriptor)
        current = os.stat(output.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise ValueError("inspection output directory changed")
        closing = preparation._directory(output.parent)
        try:
            current = os.fstat(closing)
            if (current.st_dev, current.st_ino) != (parent_info.st_dev, parent_info.st_ino):
                raise ValueError("inspection output parent changed")
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
                current = os.stat(output.name, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == identity:
                    os.rmdir(output.name, dir_fd=parent)
            except OSError:
                pass
        raise
    finally:
        if directory is not None:
            os.close(directory)
        os.close(parent)


def run_repository_read_set_inspection(*, root, query, required_paths, requirements_file,
        task_id, provider, model, max_input_tokens, output, limit=8, timeout=1800,
        max_output_tokens=2048, scheduler_state=None, scheduler_policy_sha256=None,
        failure_publication_observer=None):
    """Retain closed paid-attempt evidence, then rethrow the original API error."""
    root, output, requirements_file = map(preparation._absolute, (root, output, requirements_file))
    if Path.cwd() != root:
        raise ValueError("inspection must start in its explicit source root")
    if (provider, model) != ("grok_cli", "grok-4.7"):
        raise ValueError("explicit supported Grok inspection route required")
    if type(task_id) is not str or preparation._ID.fullmatch(task_id) is None:
        raise ValueError("explicit task identity required")
    if type(query) is not str or not query.strip() or len(query.encode()) > 4096:
        raise ValueError("bounded query required")
    if type(required_paths) not in {list, tuple} or len(required_paths) > preparation.MAX_REQUIRED_PATHS:
        raise ValueError("bounded required paths required")
    required_paths = tuple(preparation._relative(path) for path in required_paths)
    if len(set(required_paths)) != len(required_paths):
        raise ValueError("unique required paths required")
    for value, maximum in ((limit, 32), (timeout, 1800), (max_output_tokens, 16384),
            (max_input_tokens, preparation.MAX_COUNTER)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("bounded inspection settings required")
    if failure_publication_observer is not None and not callable(failure_publication_observer):
        raise ValueError("explicit failure publication observer required")
    scheduler_path = preparation._scheduler_input(scheduler_state, scheduler_policy_sha256)
    descriptor = preparation._directory(root)
    os.close(descriptor)
    parent = preparation._output_parent(output, root)
    os.close(parent)
    required_raw, required_text = preparation._capture_requirements(requirements_file)
    temporary_root = Path("/tmp")
    descriptor = preparation._directory(temporary_root)
    os.close(descriptor)
    if (root == temporary_root or root in temporary_root.parents
            or any(os.path.lexists(path / ".git") for path in (temporary_root, *temporary_root.parents))):
        raise ValueError("private native state must remain outside Git")
    from ipfs_accelerate_py.agent_supervisor.runtime.automatic_prompt_tokenizer import resolve_prompt_tokenizer
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_read_set_provider import inspect_repository_read_set
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    index_factory = None
    if scheduler_path is not None:
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
        owner = open_existing_scheduler(scheduler_path, scheduler_policy_sha256)
        index_factory = lambda path: NativeRepositoryIndex(path, scheduler=owner)
    counter, provenance = resolve_prompt_tokenizer(provider=provider, model=model)
    last_receipt, observation_status, phase = None, "not_observed", "native_session_initialization"

    def requirements_guard():
        if preparation._capture_requirements(requirements_file)[0] != required_raw:
            raise ValueError("literal inspection requirements changed")

    def observe(value):
        nonlocal last_receipt, observation_status
        try:
            if type(value) is not dict:
                raise ValueError("closed inspection receipt required")
            raw = preparation._json(value)
            if len(raw) > preparation.MAX_RECEIPT_BYTES:
                raise ValueError("inspection receipt exceeds bound")
            last_receipt = json.loads(raw)
            observation_status = "retained"
        except Exception:
            observation_status = "invalid_observation"

    common = dict(schema=SCHEMA, task_id=task_id, provider=provider, model=model,
        tokenizer_provenance=provenance, requirements_sha256=preparation._sha(required_raw),
        requirements_bytes=len(required_raw), live_provider_operation_requested=True,
        resource_scheduler_binding=dict(mode="existing_pinned" if scheduler_path is not None else "native_default",
            state_path_sha256=None if scheduler_path is None else preparation._sha(str(scheduler_path).encode()),
            policy_sha256=scheduler_policy_sha256, policy_reconfiguration_requested=False,
            state_creation_requested=False if scheduler_path is not None else None),
        source_guard_scope="cooperative snapshot checks; no writer lease", native_history_complete=None,
        whole_session_token_savings=None, task_acceptance=False, billing_total_verified=False,
        source_bodies_in_receipt=False, proof_authority=False, execution_authority=False,
        completion_authority=False)
    try:
        with tempfile.TemporaryDirectory(prefix="repository-read-set-inspect-", dir=temporary_root) as temporary:
            with RepositorySourceSession(root, Path(temporary) / "state", representation="original",
                    index_factory=index_factory) as session:
                before = session.status()
                phase = "guarded_inspection"
                result = inspect_repository_read_set(session=session, generation=before["generation"],
                    query=query, required_paths=required_paths, required_texts=(required_text,),
                    task_id=task_id, provider=provider, model=model, workspace=root, token_counter=counter,
                    max_input_tokens=max_input_tokens, limit=limit, timeout=timeout,
                    max_output_tokens=max_output_tokens, requirements_guard=requirements_guard,
                    receipt_observer=observe)
                observe(result.receipt)
                if observation_status != "retained":
                    raise ValueError("closed inspection receipt unavailable")
                phase = "closing_requirements_check"
                requirements_guard()
                phase = "closing_source_check"
                after = session.status()
                if (after.get("source_current") is not True or after.get("refresh_required") is not False
                        or any(before.get(key) != after.get(key) for key in ("session_id", "generation",
                            "source_tree_sha256", "package_manifest_sha256", "datasets_head", "semantic_state_cid"))):
                    raise ValueError("inspection source generation changed")
                prompt, response = result.selected_prompt.encode(), result.response_text.encode()
                if len(prompt) > 256000 or not 0 < len(response) <= MAX_RESPONSE_BYTES:
                    raise ValueError("bounded inspection text required")
                receipt = dict(common, status="returned", inspection=last_receipt,
                    inspection_observation_status=observation_status, source_current_at_close=True,
                    source_generation=after["generation"], source_tree_sha256=after["source_tree_sha256"],
                    prompt_sha256=preparation._sha(prompt), prompt_bytes=len(prompt),
                    response_sha256=preparation._sha(response), response_bytes=len(response))
        encoded = preparation._json(receipt)
        if len(encoded) > preparation.MAX_RECEIPT_BYTES:
            raise ValueError("bounded inspection receipt required")
        phase = "output_publication"
        _publish(output, root, {"prompt.txt": prompt, "response.txt": response, "receipt.json": encoded})
    except BaseException:
        publication = dict(schema=SCHEMA, status="failed", reason="inspection_failed",
            failure_phase=phase, receipt_retained=False, receipt_sha256=None, receipt_bytes=None)
        try:
            failed = dict(common, status="failed", failure_reason="inspection_failed", failure_phase=phase,
                inspection=last_receipt, inspection_observation_status=observation_status,
                source_current_at_close=None, prompt_sha256=None, response_sha256=None)
            encoded = preparation._json(failed)
            if len(encoded) <= preparation.MAX_RECEIPT_BYTES:
                _publish(output, root, {"receipt.json": encoded})
                publication.update(receipt_retained=True, receipt_sha256=preparation._sha(encoded), receipt_bytes=len(encoded))
        except BaseException:
            pass  # Never mask the original provider/cancellation/post-return failure.
        if failure_publication_observer is not None:
            try:
                failure_publication_observer(publication)
            except BaseException:
                pass
        raise
    return dict(schema=SCHEMA, status="written", prompt_sha256=preparation._sha(prompt), prompt_bytes=len(prompt),
        response_sha256=preparation._sha(response), response_bytes=len(response),
        receipt_sha256=preparation._sha(encoded), receipt_bytes=len(encoded),
        live_provider_operation_requested=True, task_acceptance=False, raw_content_exported_to_stdout=False)


def main(argv=None):
    parser = preparation._Arguments(description=__doc__)
    for flag in ("root", "query", "requirements-file", "task-id", "provider", "model", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--required-path", action="append", default=[])
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--scheduler-state")
    parser.add_argument("--scheduler-policy-sha256")
    publication = None
    def remember(value):
        nonlocal publication
        publication = value
    try:
        arguments = vars(parser.parse_args(argv))
        arguments["required_paths"] = arguments.pop("required_path")
        result = run_repository_read_set_inspection(**arguments, failure_publication_observer=remember)
    except Exception:
        result = publication or dict(schema=SCHEMA, status="refused", reason="invalid_input_or_local_failure",
            receipt_retained=False)
        print(preparation._json(result).decode(), end="")
        return 2
    print(preparation._json(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
