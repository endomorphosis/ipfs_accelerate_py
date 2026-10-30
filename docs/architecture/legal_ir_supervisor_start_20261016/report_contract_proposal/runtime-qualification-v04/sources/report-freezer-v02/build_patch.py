"""Effect-free exact-baseline patch assembly; never imports product modules.

Root may assemble/apply the reviewed patches separately. This program only
reads protected sources and writes two fresh proposal patch files here.
"""
from pathlib import Path
import difflib
import hashlib

ROOT = Path(__file__).absolute().parent
A = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
DAEMON = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
LOCK = 'ipfs_accelerate_py/agent_supervisor/merge/git_checkout_transaction.py'
EXPECTED_DAEMON = '23addb0dfc23e2cb093e7f74abbb7dc84a5308c24259e1eb0e25d721c0a80bed'
EXPECTED_LOCK = 'bdf01fc5f0a9ebb5f3ce5fac0a2319d44aac25b99f804af98cddb6830ac2076a'


def replace_one(source, old, new):
    if source.count(old) != 1:
        raise ValueError('exact_single_source_anchor_required')
    return source.replace(old, new, 1)


def daemon_successor(source):
    prefix, marker, rest = source.partition("class PortalImplementationDaemon(AuthoritativeCompletionMixin):\n")
    body, end, suffix = rest.partition("class DatabaseImplementationDaemonError(RuntimeError):\n")
    if not marker or not end:
        raise ValueError("original_daemon_class_boundaries_required")
    return prefix + marker + _daemon_body_successor(body) + end + suffix


def _daemon_body_successor(source):
    source = replace_one(source, '        manual_completion_authority_revalidation_only: bool = False,\n        preserve_task_attempts: bool = False,\n        objective_path: Path | None = None,\n',
        '        manual_completion_authority_revalidation_only: bool = False,\n        preserve_task_attempts: bool = False,\n        report_only_candidate_freezing: bool = False,\n        objective_path: Path | None = None,\n')
    source = replace_one(source, '        self.preserve_task_attempts = preserve_task_attempts\n',
        '''        self.preserve_task_attempts = preserve_task_attempts
        if type(report_only_candidate_freezing) is not bool:
            raise ValueError("report_only_candidate_freezing must be boolean")
        self.report_only_candidate_freezing = report_only_candidate_freezing
        self._report_only_before_attempt = None
        self._report_only_dispatch = None
''')
    source = replace_one(source, '''        consume_attempt: bool = True,
    ) -> None:
        state.active_task_id = task.task_id
''', '''        consume_attempt: bool = True,
    ) -> None:
        if self.report_only_candidate_freezing:
            from .report_only_candidate_freezer import capture_before_attempt

            self._report_only_before_attempt = capture_before_attempt(
                self, state, task, attempt, consume_attempt=consume_attempt,
            )
            self._report_only_dispatch = None
        state.active_task_id = task.task_id
''')
    source = replace_one(source, '''        workspace_path: Path,
    ) -> None:
        """Start a fresh no-worker grace window immediately before Popen."""
''', '''        workspace_path: Path,
        baseline_ref: str = "",
        callable_invocation: object = None,
        control_plane_connection: object = None,
    ) -> None:
        """Start a fresh no-worker grace window immediately before Popen.

        The opt-in callable bridge supplies actual retained objects, never JSON.
        Legacy callsites remain unchanged and cannot enable report freezing.
        """
''')
    anchor = '''        state.active_phase_detail = "provider_launch_birth"
        state.heartbeat_at = timestamp
        state.last_progress_at = timestamp
        state.save(self.state_path)
'''
    source = replace_one(source, anchor, anchor + '''        if self.report_only_candidate_freezing:
            from .report_only_candidate_freezer import capture_dispatch

            self._report_only_dispatch = capture_dispatch(
                self, state, task, attempt, workspace_path, baseline_ref,
                callable_invocation, control_plane_connection,
            )
''')
    tail = '''        self._run_git(
            [
                "-c",
                "user.name=Implementation Daemon",
                "-c",
                "user.email=implementation-daemon@example.invalid",
                "commit",
                "-m",
                f"{task.task_id}: {task.title or 'implementation attempt'}",
                "-m",
                f"Attempt: {attempt}",
            ],
            cwd=worktree_path,
        )
        commit_ref = self._run_git(["rev-parse", "HEAD"], cwd=worktree_path).stdout.strip()
        result = {
            "committed": True,
            "commit": commit_ref,
            "status": status,
        }
        if submodule_results:
            result["submodule_results"] = submodule_results
        return result
'''
    replacement = '''        result = self._commit_staged_worktree_changes(
            worktree_path, task, attempt, status=status,
        )
        if submodule_results:
            result["submodule_results"] = submodule_results
        return result

    def _commit_staged_worktree_changes(
        self,
        worktree_path: Path,
        task: PortalTask,
        attempt: int,
        *,
        status: str,
        git_runner: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Original commit tail shared by normal and guarded report-only paths.

        The opt-in public method supplies its internal native bounded runner.
        This private tail supplies neither staging nor mutation permission.
        """
        run_git = self._run_git if git_runner is None else git_runner
        run_git(
            [
                "-c",
                "user.name=Implementation Daemon",
                "-c",
                "user.email=implementation-daemon@example.invalid",
                "commit",
                "-m",
                f"{task.task_id}: {task.title or 'implementation attempt'}",
                "-m",
                f"Attempt: {attempt}",
            ],
            cwd=worktree_path,
        )
        commit_ref = run_git(["rev-parse", "HEAD"], cwd=worktree_path).stdout.strip()
        return {"committed": True, "commit": commit_ref, "status": status}

    def freeze_report_only_candidate(
        self,
        *,
        outcome: object,
        freezer_owner: object,
        protected_source_sha256: dict[str, str],
        declared_external_storage: dict[str, int],
    ) -> dict[str, Any]:
        """Freeze the original report during the genuine active success callback.

        No task/baseline/counter binding is accepted from the caller. The
        original dispatch capture, current SELECT row, decision runtime and
        checkout lock must all agree. No validation or acceptance is issued.
        """
        from .report_only_candidate_freezer import freeze

        return freeze(
            self, outcome=outcome, freezer_owner=freezer_owner,
            protected_source_sha256=protected_source_sha256,
            declared_external_storage=declared_external_storage,
        )
'''
    return replace_one(source, tail, replacement)


def lock_successor(source):
    # Keep the original public resolver/timeout behavior and factor only its
    # identical lock body. Both paths share the same registry/held-depths.
    old = '''    directory = git_checkout_directory(Path(repo))
    key = str(directory)
'''
    new = '''    directory = git_checkout_directory(Path(repo))
    with checkout_git_directory_mutation(directory, timeout_seconds=timeout):
        yield directory


@contextmanager
def checkout_git_directory_mutation(
    directory: Path, *, timeout_seconds: float = 30.0,
) -> Iterator[Path]:
    """Hold the original checkout lock for an already observed Git directory.

    The caller must obtain this exact directory from the actual repository.
    This advisory lock supplies no repository/task/lease mutation authority.
    The native report freezer resolves it with an attached finite Git child.
    """
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("checkout Git timeout must be finite and nonnegative")
    directory = Path(directory)
    if (not directory.is_absolute() or str(directory) != os.path.abspath(directory)
            or any(path.is_symlink() for path in (directory, *directory.parents))
            or not directory.is_dir()):
        raise CheckoutGitMutationBlocked("checkout_git_directory_unaliased_required")
    key = str(directory)
'''
    return replace_one(source, old, new)


def main():
    raw_daemon = (A / DAEMON).read_bytes()
    if hashlib.sha256(raw_daemon).hexdigest() != EXPECTED_DAEMON:
        raise ValueError('original_daemon_source_changed')
    for relative, successor in ((DAEMON, daemon_successor), (LOCK, lock_successor)):
        raw = (A / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != (EXPECTED_DAEMON if relative == DAEMON else EXPECTED_LOCK):
            raise ValueError('original_source_changed')
        original = raw.decode()
        proposed = successor(original)
        patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True),
            proposed.splitlines(keepends=True), fromfile='a/' + relative, tofile='b/' + relative))
        name = 'implementation_daemon.patch' if relative == DAEMON else 'git_checkout_transaction.patch'
        with (ROOT / name).open('x') as stream:
            stream.write(patch)


if __name__ == '__main__':
    main()
