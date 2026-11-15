"""Finite offline qualification of issued packet retention and historical audit.

The retained authored fixtures supply native owner initialization and claims.
Source indexing, signed public checks, stdio MCP edits, packet issuance, router
dispatch, artifact publication and the Terminal audit use their actual owners.
Only the provider response is scripted. This is not a live benchmark runner.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

from .terminal_capsule_columns_experiment import MAX_FIXTURE_BYTES, MAX_FIXTURE_FILES

SCHEMA = "terminal-world-packet-replay-offline-experiment@1"
MAX_REPORT_BYTES = 1024 * 1024
CLEANUP_SCOPE = "held_root_contents_only/v1"
_LOCK = threading.Lock()
FALSE = dict(live_provider_calls=0, model_calls=0, training_executed=False,
    live_terminal_bench_run=False, provider_usage_measured=False,
    token_savings_established=False, provider_ingestion_verified=False,
    source_currentness_verified=False, owner_signature_replayed=False,
    proof_authority=False, execution_authority=False, completion_authority=False,
    task_completed=False, raw_content_exported=False)
CHECKS = frozenset({"native_failed_passed_failed_checks", "two_source_generations",
    "guarded_full_file_mcp_edit", "public_check_bytes_unchanged",
    "stale_source_dispatch_refused", "newer_owner_dispatch_refused",
    "exact_router_input_capture", "both_full_artifacts_retained",
    "historical_inputs_verified_after_owner_close",
    "historical_inputs_verified_after_source_unavailable",
    "historical_replay_grants_no_authority", "exact_two_invocation_population"})


def _need(value, code):
    if not value:
        raise ValueError(code)


def _wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_fixture_directory(root_fd, parts, records):
    """Walk only held-root-relative directory entries under their exact pins."""
    descriptor = os.dup(root_fd)
    prefix = ()
    try:
        _need(_file_identity(os.fstat(descriptor)) == records[prefix], "private_fixture_identity_changed")
        for name in parts:
            following = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
            prefix += (name,)
            _need(_file_identity(os.fstat(descriptor)) == records[prefix], "private_fixture_identity_changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fixture_census(root_fd, root_identity, observation):
    """Complete bounded metadata census before any permission mutation."""
    _need(_file_identity(os.fstat(root_fd)) == root_identity, "private_fixture_identity_changed")
    records, directories, pending = {(): root_identity}, [()], [()]
    observation.update(private_fixture_observed_prefix_entries=0,
        private_fixture_observed_prefix_files=0, private_fixture_observed_prefix_bytes=0)
    while pending:
        parts = pending.pop()
        descriptor = _open_fixture_directory(root_fd, parts, records)
        try:
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    observation["private_fixture_observed_prefix_entries"] += 1
                    _need(observation["private_fixture_observed_prefix_entries"] <= MAX_FIXTURE_FILES,
                        "private_fixture_file_bound_exceeded")
                    info = entry.stat(follow_symlinks=False)
                    _need(info.st_uid == os.getuid(), "private_fixture_foreign_owner")
                    _need(not stat.S_ISLNK(info.st_mode), "unexpected_private_fixture_symlink")
                    relative = parts + (entry.name,)
                    records[relative] = _file_identity(info)
                    if stat.S_ISDIR(info.st_mode):
                        _need(info.st_dev == root_identity[0], "private_fixture_cross_device")
                        directories.append(relative); pending.append(relative)
                    else:
                        _need(stat.S_ISREG(info.st_mode), "unexpected_private_fixture_file_kind")
                        observation["private_fixture_observed_prefix_files"] += 1
                        observation["private_fixture_observed_prefix_bytes"] += info.st_size
                        _need(observation["private_fixture_observed_prefix_bytes"] <= MAX_FIXTURE_BYTES,
                            "private_fixture_byte_bound_exceeded")
        finally:
            os.close(descriptor)
    _need(_file_identity(os.fstat(root_fd)) == root_identity, "private_fixture_identity_changed")
    return records, directories


def _grant_owned_cleanup_directory_write(root_fd, root_identity, observation):
    """Add owner-write only to already censused owned directory inodes.

    Files, including hardlinks, never receive chmod. All directory/file records
    must still match the complete first census before the first mutation.
    """
    records, directories = _fixture_census(root_fd, root_identity, observation)
    replay, repeated = _fixture_census(root_fd, root_identity, {})
    _need(records == replay and directories == repeated, "private_fixture_identity_changed")
    changed = 0
    for parts in sorted(directories, key=lambda value: (len(value), value)):
        descriptor = _open_fixture_directory(root_fd, parts, records)
        try:
            info = os.fstat(descriptor)
            _need(info.st_uid == os.getuid() and stat.S_ISDIR(info.st_mode), "private_fixture_foreign_owner")
            if not info.st_mode & stat.S_IWUSR:
                os.fchmod(descriptor, stat.S_IMODE(info.st_mode) | stat.S_IWUSR)
                after = os.fstat(descriptor)
                _need((after.st_dev, after.st_ino, after.st_uid) == (info.st_dev, info.st_ino, info.st_uid)
                    and stat.S_ISDIR(after.st_mode)
                    and stat.S_IMODE(after.st_mode) == stat.S_IMODE(info.st_mode) | stat.S_IWUSR,
                    "private_fixture_identity_changed")
                records[parts] = _file_identity(after)
                changed += 1
                observation["private_fixture_cleanup_directories_made_owner_writable"] = changed
        finally:
            os.close(descriptor)
    replay, _ = _fixture_census(root_fd, records[()], {})
    _need(records == replay, "private_fixture_identity_changed")
    observation["private_fixture_cleanup_directories_made_owner_writable"] = changed
    return records


def _delete_fixture_contents(root_fd, parent_fd, root_name, records, observation):
    """Delete only the bounded censused contents under the retained root FD.

    The root's parent entry is never removed. A replacement root cannot redirect
    operations into its tree; namespace drift stops cleanup and qualification.
    Each child directory remains open while its censused contents are removed.
    """
    children = {}
    for parts in records:
        if parts:
            children.setdefault(parts[:-1], []).append(parts)

    def root_bound():
        actual = os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
        held = os.fstat(root_fd)
        _need(_file_identity(actual) == _file_identity(held) == records[()]
            and stat.S_ISDIR(held.st_mode) and held.st_uid == os.getuid(),
            "private_fixture_identity_changed")

    def remove_children(descriptor, parts):
        _need(_file_identity(os.fstat(descriptor)) == records[parts], "private_fixture_identity_changed")
        for relative in sorted(children.get(parts, ())):
            root_bound()
            _need(_file_identity(os.fstat(descriptor)) == records[parts], "private_fixture_identity_changed")
            name = relative[-1]
            actual = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            expected = records[relative]
            _need(_file_identity(actual) == expected and actual.st_uid == os.getuid()
                and not stat.S_ISLNK(actual.st_mode), "private_fixture_identity_changed")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            directory = stat.S_ISDIR(actual.st_mode)
            child_fd = os.open(name, flags | (os.O_DIRECTORY if directory else 0), dir_fd=descriptor)
            try:
                _need(_file_identity(os.fstat(child_fd)) == expected, "private_fixture_identity_changed")
                if directory:
                    remove_children(child_fd, relative)
                    with os.scandir(child_fd) as remaining:
                        _need(next(remaining, None) is None, "private_fixture_contents_changed")
                    root_bound()
                    _need(_file_identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False))
                        == _file_identity(os.fstat(child_fd)) == records[relative], "private_fixture_identity_changed")
                    # Remove only the now-empty censused child directory.
                    # No fresh name-based traversal follows it into a tree.
                    os.rmdir(name, dir_fd=descriptor)
                    observation["private_fixture_cleanup_removed_directories"] += 1
                else:
                    _need(stat.S_ISREG(actual.st_mode), "unexpected_private_fixture_file_kind")
                    root_bound()
                    _need(_file_identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False))
                        == _file_identity(os.fstat(child_fd)) == expected, "private_fixture_identity_changed")
                    os.unlink(name, dir_fd=descriptor)
                    observation["private_fixture_cleanup_removed_files"] += 1
                after = os.fstat(descriptor)
                before = records[parts]
                _need((after.st_dev, after.st_ino, after.st_uid, after.st_mode) == before[:4],
                    "private_fixture_identity_changed")
                records[parts] = _file_identity(after)
            finally:
                os.close(child_fd)
        with os.scandir(descriptor) as remaining:
            _need(next(remaining, None) is None, "private_fixture_contents_changed")

    root_bound()
    remove_children(root_fd, ())
    root_bound()


@contextmanager
def _world_private_fixture(parent):
    """World-only cleanup for native immutable source-package directories."""
    parent = Path(parent).absolute()
    _need(parent.resolve(strict=True) == parent and not parent.is_symlink(), "canonical_scratch_parent_required")
    parent_info = parent.stat()
    _need(stat.S_ISDIR(parent_info.st_mode) and parent_info.st_uid == os.getuid(), "owned_scratch_parent_required")
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    path, root_fd = None, None
    cleanup = dict(private_fixture_cleanup_complete=False, private_fixture_files=None,
        private_fixture_bytes=None, private_fixture_storage_bound_bytes=MAX_FIXTURE_BYTES,
        private_fixture_cleanup_directories_made_owner_writable=0,
        private_fixture_cleanup_scope=CLEANUP_SCOPE, private_fixture_root_removed=False,
        private_fixture_cleanup_removed_files=0, private_fixture_cleanup_removed_directories=0)
    codes = {"private_fixture_identity_changed", "private_fixture_foreign_owner", "private_fixture_cross_device",
        "private_fixture_file_bound_exceeded", "private_fixture_byte_bound_exceeded",
        "unexpected_private_fixture_symlink", "unexpected_private_fixture_file_kind",
        "private_fixture_native_contexts_not_closed", "private_fixture_contents_changed"}
    try:
        path = Path(tempfile.mkdtemp(prefix="world-packet-replay-", dir=parent))
        root_fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        initial = os.fstat(root_fd)
        _need(stat.S_IMODE(initial.st_mode) == 0o700 and initial.st_uid == os.getuid(), "private_fixture_identity_changed")
        yield path, cleanup
    finally:
        try:
            if path is not None and root_fd is not None:
                _need(cleanup.get("private_fixture_native_contexts_closed", True) is True,
                    "private_fixture_native_contexts_not_closed")
                _need((os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino)
                    == (parent_info.st_dev, parent_info.st_ino)
                    and parent.resolve(strict=True) == parent, "private_fixture_identity_changed")
                current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                _need(_file_identity(current) == _file_identity(os.fstat(root_fd))
                    and stat.S_ISDIR(current.st_mode) and current.st_uid == os.getuid()
                    and path.resolve(strict=True) == path, "private_fixture_identity_changed")
                records = _grant_owned_cleanup_directory_write(root_fd, _file_identity(current), cleanup)
                _need(_file_identity(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)) == records[()]
                    and path.resolve(strict=True) == path, "private_fixture_identity_changed")
                cleanup["private_fixture_files"] = cleanup["private_fixture_observed_prefix_files"]
                cleanup["private_fixture_bytes"] = cleanup["private_fixture_observed_prefix_bytes"]
                _delete_fixture_contents(root_fd, parent_fd, path.name, records, cleanup)
                _need(_file_identity(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False))
                    == _file_identity(os.fstat(root_fd)) == records[()]
                    and path.resolve(strict=True) == path, "private_fixture_identity_changed")
                with os.scandir(root_fd) as remaining:
                    cleanup["private_fixture_cleanup_complete"] = next(remaining, None) is None
        except (OSError, ValueError) as error:
            code = str(error) if type(error) is ValueError else ""
            cleanup["private_fixture_cleanup_reason"] = (code if code in codes
                else "owned_fixture_cleanup_io_unavailable" if isinstance(error, OSError)
                else "owned_fixture_cleanup_unavailable")
        finally:
            if root_fd is not None: os.close(root_fd)
            os.close(parent_fd)


def report_sha256(report):
    """FULL canonical report hash, including its internal seal."""
    return _sha(_wire(report))


def _artifact_root(path):
    path = Path(path).absolute()
    _need(path.resolve(strict=True) == path and not path.is_symlink(),
        "canonical_artifact_root_required")
    info = path.stat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o700, "private_owned_artifact_root_required")
    _need(not any(path.iterdir()), "fresh_artifact_root_required")
    return path


def _pins():
    """Declared authored-source population, not transitive runtime attestation."""
    root = Path(__file__).resolve().parents[3]
    names = (
        "benchmarks/agent_supervisor/container_coding/terminal_world_packet_replay_experiment.py",
        "benchmarks/agent_supervisor/container_coding/terminal_capsule_columns_experiment.py",
        "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
        "benchmarks/agent_supervisor/container_coding/terminal_world_packet_audit.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_packet.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_packet_replay.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_invocation_replay.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_artifact_retention.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_intent_world.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_session.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_mcp.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_policy.py",
        "test/api/test_agent_supervisor_local_planning_admission.py",
        "test/api/test_intent_requirement_observation.py",
        "test/api/test_intent_requirement_observation_native.py",
        "test/api/test_repository_context_compiler.py",
        "test/api/test_repository_source_mcp.py",
    )
    return dict(scope="listed_source_files_only_not_transitive_provenance",
        files={name: _sha((root / name).read_bytes()) for name in names})


@contextmanager
def _authored_scenario(fixture):
    # These are explicitly retained qualification fixtures, not task-source
    # imports or a production plugin loader. They use the native owner APIs.
    from test.api.test_agent_supervisor_local_planning_admission import scenario
    factory = getattr(scenario, "__wrapped__", None)
    if factory is None:
        factory = getattr(scenario, "_fixture_function", None)
    _need(callable(factory), "authored_fixture_factory_unavailable")
    with contextmanager(factory)(fixture) as case:
        yield case


@contextmanager
def _scripted_provider(workspace, state):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy

    def generate(prompt, **options):
        _need(options.get("provider") == "codex_cli" and options.get("model_name") == "pinned"
            and options.get("allow_local_fallback") is False
            and options.get("allow_cross_provider_fallback") is False, "authored_route_changed")
        state["inputs"].append(prompt)
        observation = {"exit_code": 0}
        binding = options.get("codex_repository_source_binding")
        if binding is not None:
            policy.validate_binding(binding, workspace=workspace, materialized=True)
            observation.update(codex_repository_source_configured=True,
                codex_repository_source_config_sha256=binding["config_sha256"])
        set_last_cli_observation("codex_cli", observation)
        if state["callback"] is not None:
            state["callback"](binding)
        return "Authored offline response; independent owner validation remains required."

    with ExitStack() as stack:
        stack.enter_context(patch.object(intelligence_index, "discover_available_providers", return_value=["codex_cli"]))
        stack.enter_context(patch.object(intelligence_index, "select_efficient_route",
            return_value=SimpleNamespace(provider="codex_cli", model_name="pinned",
                reasoning_effort="high", catalog_revision="authored-offline-seam")))
        stack.enter_context(patch.object(llm_router, "get_llm_provider", return_value=object()))
        stack.enter_context(patch.object(llm_router, "generate_text", side_effect=generate))
        yield


def _refused(operation, code):
    try:
        operation()
    except ValueError:
        return True
    raise ValueError(code)


def _prepare_packet(session, observation, *, task_id, generation, command):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
        prepare_repository_world_packet, repository_world_task_binding,
    )
    from test.api.test_repository_context_compiler import compile_context, compiler, obligation
    task = "Change answer() to return 2; keep the public check unchanged."
    effects, paths = ("repository_write", "validation"), ("answer.py",)
    required = (obligation("preserve-check", summary="Do not change test_answer.py or the declared task scope."),
        observation.reference)
    adapter = session.context_adapter(compiler())
    compiled = compile_context(adapter, required,
        generation=generation, query="answer", goal={"intent": task}, scope={"paths": paths},
        authority={"mode": "proposal", "allowed_paths": paths,
            "world_task_packet": repository_world_task_binding(task_id, task, effects, paths)},
        acceptance={"criteria": ("The existing public answer check passes.",)})
    return task, prepare_repository_world_packet(adapter=adapter,
        compilation=compiled, generation=generation, task_id=task_id, task_text=task,
        unresolved_obligations=("Independent task correctness remains unresolved.",),
        validation_commands=(command,), allowed_effects=effects, write_paths=paths,
        intent_world_observation=observation)


def _audit_projection(audit):
    return dict(status=audit.get("status"), audit_sha256=_sha(_wire(audit)),
        all_observed_coding_inputs_verified=audit.get("all_observed_coding_inputs_verified"),
        invocation_count=len(audit.get("invocations", [])), match_count=len(audit.get("matches", [])),
        read_error_count=len(audit.get("read_errors", [])),
        invocations=[{key: row.get(key) for key in ("invocation_id", "status", "model_input_verified")}
            for row in audit.get("invocations", [])],
        matches=[{key: row.get(key) for key in ("invocation_id", "model_prompt_sha256",
            "model_prompt_bytes", "model_input_verified", "verification_scope")}
            for row in audit.get("matches", [])],
        proof_authority=audit.get("proof_authority"), execution_authority=audit.get("execution_authority"),
        completion_authority=audit.get("completion_authority"), raw_prompts_exported=audit.get("raw_prompts_exported"))


def _cycle(fixture, artifacts, scratch_root):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_intent_world import capture_repository_owner_intent_world
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import provider_subprocess_environment
    from test.api.test_intent_requirement_observation import _admit
    from test.api.test_intent_requirement_observation_native import _owner, _claim, _run
    from test.api.test_repository_source_mcp import real_client
    from .terminal_context_audit import collect_terminal_context_audit

    receipts, packets = [], []
    state = dict(inputs=[], callback=None)
    checks = dict.fromkeys(CHECKS, False)
    previous = Path.cwd()
    with _authored_scenario(fixture) as scenario:
        admission, tasks = _admit(scenario)
        case = {**scenario, "admission": admission, "intent_manifest": admission["manifest"], "tasks": tasks}
        workspace, task_id = scenario["repository"], tasks["LOCAL-TASK"]
        original_check = (workspace / "test_answer.py").read_bytes()
        source_state = fixture / "source-state"
        os.chdir(workspace)
        try:
            with _scripted_provider(workspace, state), _owner(case, fixture) as owner, \
                    _claim(owner, fixture) as (_daemon, attempt):
                _need(attempt.task_cid == task_id, "native_claim_task_differs")
                failed = _run(owner, attempt, timeout=10)
                _need(failed["passed"] is False, "authored_initial_check_did_not_fail")
                command = shlex.join(owner.source.get_task(task_id).validations[0]["argv"])
                native_revision = owner.source.get_task(task_id).revision
                private = (owner.credentials.token, owner.identity.listen_uri,
                    str(owner.database), str(owner.server.typed_command_socket_path()))
                with RepositorySourceSession(workspace, source_state) as session:
                    def capture(generation):
                        return capture_repository_owner_intent_world(session=session, server=owner.server,
                            admission=admission, task_id=task_id, generation=generation)

                    def dispatch(packet, **options):
                        cleaned = provider_subprocess_environment(os.environ)
                        # Use the actual daemon boundary while retaining the
                        # private supervisor owner objects outside worker env.
                        with patch.dict(os.environ, cleaned, clear=True), redirect_stdout(io.StringIO()):
                            return router.run(prompt=task, provider="codex_cli", model="pinned", timeout=30,
                                max_output_tokens=128, world_task_packet=packet,
                                world_packet_artifact_dir=artifacts, **options)

                    first_observation = capture(1)
                    task, first = _prepare_packet(session, first_observation,
                        task_id=task_id, generation=1, command=command)
                    packets.append(first)

                    def edit(binding):
                        _need(type(binding) is dict, "real_source_helper_binding_missing")
                        with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
                            found, error = client.call("repository_context_query", generation=1, query="answer", limit=8)
                            _need(not error and any(row["path"] == "answer.py" and row["name"] == "answer.answer"
                                for row in found["candidates"]), "authored_definition_not_indexed")
                            read, error = client.call("repository_read", generation=1, path="answer.py")
                            _need(not error and read["content"].count("return 1") == 1, "guarded_full_file_read_failed")
                            edited, error = client.call("repository_replace_symbol", generation=1, path="answer.py",
                                symbol="answer", expected_sha256=read["source_sha256"],
                                replacement=read["content"].replace("return 1", "return 2"))
                            _need(not error and edited["canonical_changed"] is True, "guarded_authored_edit_failed")
                            checks["guarded_full_file_mcp_edit"] = True

                    state["callback"] = edit
                    _, receipt = dispatch(first, repository_source_policy=policy.POLICIES[2])
                    receipt["phase"] = "coding"
                    receipts.append(receipt)
                    state["callback"] = None
                    checks["stale_source_dispatch_refused"] = _refused(lambda: dispatch(first), "stale_source_dispatch_not_refused")
                    _need(len(state["inputs"]) == 1, "stale_packet_reached_provider")
                    _need(session.refresh(generation=1)["generation"] == 2, "source_generation_not_refreshed")
                    passed = _run(owner, attempt, timeout=10)
                    _need(passed["passed"] is True, "authored_successor_check_did_not_pass")
                    successor = capture(2)
                    task, second = _prepare_packet(session, successor,
                        task_id=task_id, generation=2, command=command)
                    packets.append(second)
                    _, receipt = dispatch(second)
                    receipt["phase"] = "coding"
                    receipts.append(receipt)
                    checks["two_source_generations"] = True
                    checks["public_check_bytes_unchanged"] = (workspace / "test_answer.py").read_bytes() == original_check
                    answer = workspace / "answer.py"
                    answer_before = answer.read_bytes()
                    observed_failure = []

                    def newer_failure(*args, **kwargs):
                        answer.write_text("def answer():\n    return 3\n")
                        try:
                            observed_failure.append(run_owner_local_task_validations(server=owner.server,
                                task_cid=task_id, attempt_id=attempt.attempt_id,
                                expected_revision=native_revision, timeout=10)["passed"])
                        finally:
                            answer.write_bytes(answer_before)
                        _need(session.status_summary()["source_current"] is True, "authored_source_not_restored")
                        return object()

                    with patch.object(llm_router, "get_llm_provider", side_effect=newer_failure):
                        checks["newer_owner_dispatch_refused"] = _refused(lambda: dispatch(second), "newer_owner_dispatch_not_refused")
                    checks["native_failed_passed_failed_checks"] = observed_failure == [False]
                    checks["exact_two_invocation_population"] = len(state["inputs"]) == len(receipts) == 2
                    checks["exact_router_input_capture"] = all(_sha(prompt.encode()) == row["model_prompt_sha256"]
                        and len(prompt.encode()) == row["model_prompt_bytes"] for prompt, row in zip(state["inputs"], receipts))
                    checks["both_full_artifacts_retained"] = all(row.get("world_packet_artifacts", {}).get("status") == "retained"
                        for row in receipts)
                    _need(owner.source.get_task(task_id).status == "in_progress", "native_task_was_completed")
                    for prompt in state["inputs"]:
                        _need(not any(value and value in prompt for value in private), "native_private_binding_exported")
        finally:
            os.chdir(previous)
    # The owner/claim/source contexts and IntentRepository are closed. Rename
    # the original checkout so historical reconstruction cannot rely on it.
    unavailable = fixture / "unavailable-live-checkout"
    workspace.rename(unavailable)
    _need(not workspace.exists(), "original_workspace_still_available")
    audit = collect_terminal_context_audit(state=artifacts / "absent-live-state", receipts=receipts,
        workspace_root=scratch_root, world_packet_artifact_root=artifacts, timeout_seconds=90)
    verified = (audit.get("all_observed_coding_inputs_verified") is True
        and len(audit.get("invocations", [])) == 2 and len(audit.get("matches", [])) == 2
        and all(row.get("status") == "verified_exact_model_input" for row in audit["invocations"]))
    checks["historical_inputs_verified_after_owner_close"] = verified
    checks["historical_inputs_verified_after_source_unavailable"] = verified
    checks["historical_replay_grants_no_authority"] = all(audit.get(name) is False
        for name in ("proof_authority", "execution_authority", "completion_authority", "raw_prompts_exported"))
    _need(all(checks.values()), "offline_cycle_qualification_failed")
    for row in receipts:
        _need(not any(value and value in _wire(row["world_packet_artifacts"]).decode() for value in private),
            "native_private_binding_exported")
    return dict(checks=checks, counters=dict(source_generations=2, native_signed_checks=3,
        real_child_mcp_edits=1, authored_provider_calls=2, stale_source_refusals=1,
        newer_owner_refusals=1, retained_successful_invocations=2),
        invocations=[dict(invocation_id=row["invocation_id"], model_prompt_sha256=row["model_prompt_sha256"],
            model_prompt_bytes=row["model_prompt_bytes"], retention=row["world_packet_artifacts"])
            for row in receipts], audit=_audit_projection(audit))


def run_experiment(*, scratch_root: Path, artifact_root: Path):
    """Execute only this declared authored qualification under caller admission."""
    _need(_LOCK.acquire(blocking=False), "offline_experiment_already_active")
    previous = Path.cwd()
    try:
        scratch_root = Path(scratch_root).absolute()
        artifacts = _artifact_root(artifact_root)
        before = _pins()
        with _world_private_fixture(scratch_root) as (fixture, cleanup):
            cleanup["private_fixture_native_contexts_closed"] = False
            _need(not artifacts.is_relative_to(fixture) and not fixture.is_relative_to(artifacts),
                "artifact_fixture_namespaces_overlap")
            cycle = _cycle(fixture, artifacts, scratch_root)
            cleanup["private_fixture_native_contexts_closed"] = True
        unchanged = before == _pins()
        qualified = unchanged and cleanup["private_fixture_cleanup_complete"]
        report = dict(schema=SCHEMA, status="qualified_offline" if qualified else "incomplete_offline",
            qualified=qualified, source_pins=before, source_pins_unchanged=unchanged,
            scope="authored native source/intent checks, issued packet dispatch, retention and historical input audit only",
            source_edit_profile="existing_guarded_full_file_read_and_explicit_authored_answer_symbol",
            tokenizer_measurement=None, whole_session_token_savings=None,
            **cycle, **cleanup, **FALSE)
        report["report_sha256"] = _sha(_wire(report))
        _need(len(_wire(report)) <= MAX_REPORT_BYTES, "offline_report_exceeds_bound")
        return report
    finally:
        os.chdir(previous)
        _LOCK.release()


def validate_report(report, *, expected_report_sha256):
    """Validate independent report identity and finite claims; no runtime rerun."""
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_artifact_retention import validate_world_packet_artifacts
    _need(type(report) is dict and len(_wire(report)) <= MAX_REPORT_BYTES, "bounded_offline_report_required")
    _need(report_sha256(report) == expected_report_sha256, "independent_report_pin_differs")
    _need(report.get("schema") == SCHEMA and report.get("report_sha256") == _sha(_wire(
        {key: value for key, value in report.items() if key != "report_sha256"})), "offline_report_seal_differs")
    _need(all(type(report.get(key)) is type(value) and report.get(key) == value for key, value in FALSE.items()),
        "offline_scope_or_authority_differs")
    _need(report.get("tokenizer_measurement") is None and report.get("whole_session_token_savings") is None,
        "offline_usage_claim_differs")
    _need(type(report.get("checks")) is dict and set(report["checks"]) == CHECKS
        and all(value is True for value in report["checks"].values()), "offline_checks_differ")
    counters = dict(source_generations=2, native_signed_checks=3,
        real_child_mcp_edits=1, authored_provider_calls=2, stale_source_refusals=1,
        newer_owner_refusals=1, retained_successful_invocations=2)
    _need(type(report.get("counters")) is dict and set(report["counters"]) == set(counters)
        and all(type(report["counters"][key]) is int and report["counters"][key] == value
            for key, value in counters.items()), "offline_population_differs")
    _need(type(report.get("invocations")) is list and len(report["invocations"]) == 2,
        "offline_invocation_population_differs")
    identities = []
    for row in report["invocations"]:
        _need(type(row) is dict and set(row) == {"invocation_id", "model_prompt_sha256", "model_prompt_bytes", "retention"}
            and type(row["invocation_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", row["invocation_id"])
            and type(row["model_prompt_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["model_prompt_sha256"])
            and type(row["model_prompt_bytes"]) is int and 0 < row["model_prompt_bytes"] <= 256000,
            "offline_invocation_identity_differs")
        identities.append(row["invocation_id"])
        retained = validate_world_packet_artifacts(row["retention"], invocation_id=row["invocation_id"])
        _need(retained["status"] == "retained",
            "offline_retention_missing")
    _need(len(set(identities)) == 2, "offline_duplicate_invocation_identity")
    _need(type(report.get("source_pins_unchanged")) is bool
        and type(report.get("private_fixture_cleanup_complete")) is bool, "offline_qualification_fields_differ")
    _need(report.get("private_fixture_native_contexts_closed") is True, "offline_native_context_close_missing")
    _need(report.get("private_fixture_cleanup_scope") == CLEANUP_SCOPE
        and report.get("private_fixture_root_removed") is False, "offline_cleanup_scope_differs")
    qualified = report["source_pins_unchanged"] and report["private_fixture_cleanup_complete"]
    _need(report.get("qualified") is qualified and report.get("status") == (
        "qualified_offline" if qualified else "incomplete_offline"), "offline_qualification_claim_differs")
    _need(report["audit"].get("all_observed_coding_inputs_verified") is True
        and type(report["audit"].get("invocation_count")) is int
        and type(report["audit"].get("match_count")) is int
        and report["audit"]["invocation_count"] == report["audit"]["match_count"] == 2,
        "offline_historical_audit_population_differs")
    audit = report["audit"]
    _need(all(audit.get(key) is False for key in ("proof_authority", "execution_authority",
        "completion_authority", "raw_prompts_exported")), "offline_audit_authority_differs")
    _need(type(audit.get("invocations")) is list and len(audit["invocations"]) == 2
        and [row.get("invocation_id") for row in audit["invocations"]] == identities
        and all(row.get("status") == "verified_exact_model_input" and row.get("model_input_verified") is True
            for row in audit["invocations"]), "offline_audit_invocation_join_differs")
    _need(type(audit.get("matches")) is list and len(audit["matches"]) == 2,
        "offline_audit_match_population_differs")
    expected = {row["invocation_id"]: row for row in report["invocations"]}
    _need([row.get("invocation_id") for row in audit["matches"]] == identities,
        "offline_audit_match_join_differs")
    for match in audit["matches"]:
        source = expected[match["invocation_id"]]
        _need(match.get("model_input_verified") is True
            and match.get("verification_scope") == "historical_world_packet_input_reconstruction"
            and match.get("model_prompt_sha256") == source["model_prompt_sha256"]
            and type(match.get("model_prompt_bytes")) is int
            and match["model_prompt_bytes"] == source["model_prompt_bytes"],
            "offline_exact_model_input_join_differs")
    return dict(valid=True, validation_scope="independently pinned finite report; no independent native experiment rerun", **FALSE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _need(args.output.parent.resolve(strict=True) == args.output.parent.absolute(), "canonical_output_parent_required")
    report = run_experiment(scratch_root=args.scratch_root, artifact_root=args.artifact_root)
    validate_report(report, expected_report_sha256=report_sha256(report))
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_wire(report) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(schema=SCHEMA, qualified=report["qualified"], report_sha256=report_sha256(report), **FALSE), sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
