"""Source-only proposal materializer; never imports or executes product code."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
DEST = Path(__file__).resolve().parent
DAEMON = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
SUPERVISOR = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py'
RUNNER = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py'
CONFIGURED = 'ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py'


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f'expected unique source anchor, found {text.count(old)}: {old[:100]!r}')
    return text.replace(old, new, 1)


def guard_method(text: str, name: str, body: str) -> str:
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PortalImplementationDaemon')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    first = fn.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant) or not isinstance(first.value.value, str):
        raise ValueError(f'{name} missing docstring')
    lines = text.splitlines(keepends=True)
    lines.insert(first.end_lineno, '\n        if self.preserve_task_attempts:\n' + body)
    return ''.join(lines)


def proposed_sources() -> dict[str, str]:
    sources = {p: (BASE / p).read_text() for p in (DAEMON, SUPERVISOR, RUNNER, CONFIGURED)}
    d = sources[DAEMON]
    d = replace_once(d, '        manual_completion_authority_revalidation_only: bool = False,\n', '        manual_completion_authority_revalidation_only: bool = False,\n        preserve_task_attempts: bool = False,\n')
    d = replace_once(d, '        self.manual_completion_authority_revalidation_only = bool(\n', '        if not isinstance(preserve_task_attempts, bool):\n            raise ValueError("preserve_task_attempts must be boolean")\n        self.preserve_task_attempts = preserve_task_attempts\n        self.manual_completion_authority_revalidation_only = bool(\n')
    for name in (
        '_rearm_attempt_limited_retry_repairs',
        '_rearm_attempt_limited_stale_proposal_tasks',
        '_rearm_attempt_limited_obsoleted_validation_tasks',
        '_release_repaired_dependency_preflight_cooldowns',
        '_release_repaired_read_only_validation_cooldowns',
        '_release_obsolete_provider_repair_cooldowns',
        '_release_obsolete_venv_launcher_cooldowns',
    ):
        d = guard_method(d, name, '            return []\n')
    d = guard_method(d, '_reset_attempt_budgets_for_completed_retry_repairs', '            return [], []\n')
    d = guard_method(d, '_release_unfinished_active_attempt', '            return {\n                "consumed": True,\n                "released": False,\n                "attempt": max(0, int(attempt or 0)),\n                "task_id": str(task_id or "").strip(),\n                "reason": "preserve_task_attempts",\n            }\n')
    # A restore is refundable in preserve mode only on a pinned source path
    # that independently proves no provider execution; unknown/post-execution
    # deferrals cannot recycle an already consumed ordinal.
    d = replace_once(d, '    def _restore_task_attempt(\n        self,\n        state: PortalTaskState,\n        task: PortalTask,\n        attempt: int,\n    ) -> None:\n', '    def _restore_task_attempt(\n        self,\n        state: PortalTaskState,\n        task: PortalTask,\n        attempt: int,\n        *,\n        zero_provider_execution: bool = False,\n    ) -> None:\n')
    d = guard_method(d, '_restore_task_attempt', '            if zero_provider_execution is not True:\n                return\n')
    # This helper returns above unless two independent pre-dispatch refusal
    # observations agree; merely seeing a quota-like failure is insufficient.
    d = replace_once(d, '        self._restore_task_attempt(state, task, max(0, attempt - 1))\n        self._mark_implementation_finished(state, finished_at=finished_at)\n        state.selection_idle_reason = "provider_capacity_backoff"\n', '        self._restore_task_attempt(\n            state, task, max(0, attempt - 1), zero_provider_execution=True,\n        )\n        self._mark_implementation_finished(state, finished_at=finished_at)\n        state.selection_idle_reason = "provider_capacity_backoff"\n')
    d = replace_once(d, '                    self._restore_task_attempt(\n                        state,\n                        task,\n                        max(0, attempt - 1),\n                    )\n                    if not state.implementation_in_progress:\n', '                    self._restore_task_attempt(\n                        state,\n                        task,\n                        max(0, attempt - 1),\n                        zero_provider_execution=(\n                            ephemeral_result.get("provider_dispatched") is False\n                        ),\n                    )\n                    if not state.implementation_in_progress:\n')
    d = replace_once(d, '                self._restore_task_attempt(\n                    state,\n                    task,\n                    max(0, attempt - 1),\n                )\n                if not state.implementation_in_progress:\n', '                self._restore_task_attempt(\n                    state,\n                    task,\n                    max(0, attempt - 1),\n                    zero_provider_execution=True,\n                )\n                if not state.implementation_in_progress:\n')
    d = replace_once(d, '                        self._restore_task_attempt(\n                            state,\n                            task,\n                            max(0, attempt - 1),\n                        )\n                    finished_at = utc_now()\n', '                        self._restore_task_attempt(\n                            state,\n                            task,\n                            max(0, attempt - 1),\n                            zero_provider_execution=(provider_dispatched is False),\n                        )\n                    finished_at = utc_now()\n')
    d = replace_once(d, '        else:\n            self._restore_task_attempt(state, task, max(0, attempt - 1))\n        state.last_implementation_started_at = started_at\n', '        else:\n            self._restore_task_attempt(\n                state, task, max(0, attempt - 1),\n                zero_provider_execution=(\n                    provider_dispatched is False\n                    and (\n                        lifecycle_setup_deferral\n                        or lifecycle_race_exception\n                        or typed_pre_dispatch_deferral\n                        or submodule_setup_deferral\n                        or dependency_setup_deferral\n                    )\n                ),\n            )\n        state.last_implementation_started_at = started_at\n')
    d = replace_once(d, '            if self.manual_completion_authority_revalidation_only\n            else self._retry_budget_repair_runtime_revision()\n', '            if (\n                self.manual_completion_authority_revalidation_only\n                or self.preserve_task_attempts\n            )\n            else self._retry_budget_repair_runtime_revision()\n')
    # Keep the closed report semantics and ordinary provider route unchanged.
    d = replace_once(d, '            manual_completion_authority_revalidation_only=(\n                args.manual_completion_authority_revalidation_only\n            ),\n', '            manual_completion_authority_revalidation_only=(\n                args.manual_completion_authority_revalidation_only\n            ),\n            preserve_task_attempts=args.preserve_task_attempts,\n')
    option = '''    parser.add_argument(
        "--preserve-task-attempts",
        action="store_true",
        help=(
            "Preserve consumed task attempts: disable automatic exhaustion "
            "rearming, completed-repair resets, retry cooldown repairs, and "
            "unknown vanished-worker rollback. Only confirmed zero-provider "
            "prelaunch deferrals retain their non-consuming behavior."
        ),
    )
'''
    d = replace_once(d, '    parser.add_argument(\n        "--manual-completion-authority-revalidation-only",\n', option + '    parser.add_argument(\n        "--manual-completion-authority-revalidation-only",\n')
    # Database mode already retains its own authoritative attempts. Preserve
    # the explicit policy at its API boundary and in its portal execution bridge.
    d = replace_once(d, '        max_task_attempts: int = 0,\n        clock_ms:', '        max_task_attempts: int = 0,\n        preserve_task_attempts: bool = False,\n        clock_ms:')
    d = replace_once(d, '        self.max_task_attempts = max_task_attempts\n        self._clock_ms', '        if not isinstance(preserve_task_attempts, bool):\n            raise DatabaseImplementationAuthorityError("preserve_task_attempts must be boolean")\n        self.preserve_task_attempts = preserve_task_attempts\n        self.max_task_attempts = max_task_attempts\n        self._clock_ms')
    d = replace_once(d, '            max_task_attempts=int(args.max_task_attempts),\n', '            max_task_attempts=int(args.max_task_attempts),\n            preserve_task_attempts=args.preserve_task_attempts,\n')
    # Expose the active policy in both ordinary refresh receipt projections.
    d = replace_once(d, '                    "ordinary_provider_dispatch_allowed": (\n', '                    "preserve_task_attempts": self.preserve_task_attempts,\n                    "ordinary_provider_dispatch_allowed": (\n')
    d = replace_once(d, '\n            "ordinary_provider_dispatch_allowed": (\n', '\n            "preserve_task_attempts": self.preserve_task_attempts,\n            "ordinary_provider_dispatch_allowed": (\n')
    sources[DAEMON] = d

    s = sources[SUPERVISOR]
    s = replace_once(s, '    manual_completion_authority_revalidation_only: bool = False\n', '    manual_completion_authority_revalidation_only: bool = False\n    preserve_task_attempts: bool = False\n')
    s = replace_once(s, '    def __post_init__(self) -> None:\n        _validated_validation_max_workers', '    def __post_init__(self) -> None:\n        if not isinstance(self.preserve_task_attempts, bool):\n            raise ValueError("preserve_task_attempts must be boolean")\n        _validated_validation_max_workers')
    s = replace_once(s, '    normalized = dict(payload)\n', '    normalized = dict(payload)\n    preserve_task_attempts = payload.get("preserve_task_attempts", False)\n    if not isinstance(preserve_task_attempts, bool):\n        raise SupervisorSchedulerConfigError("preserve_task_attempts must be boolean")\n    normalized["preserve_task_attempts"] = preserve_task_attempts\n')
    s = replace_once(s, '    return options\n\n\n# --- restored expand_supervisor_scheduler_config_args', '    if profile.get("preserve_task_attempts") is True:\n        options.append("--preserve-task-attempts")\n    return options\n\n\n# --- restored expand_supervisor_scheduler_config_args')
    s = replace_once(s, '            if self.config.manual_completion_authority_revalidation_only:\n                command.append(', '            if self.config.preserve_task_attempts:\n                command.append("--preserve-task-attempts")\n            if self.config.manual_completion_authority_revalidation_only:\n                command.append(')
    s = replace_once(s, '        if option_values("--execution-slice-task-id") != set(\n', '        preserve_options = [\n            token for token in tokens\n            if token.startswith("--")\n            and "--preserve-task-attempts".startswith(token.partition("=")[0])\n        ]\n        expected_preserve_options = (\n            ["--preserve-task-attempts"] if self.config.preserve_task_attempts else []\n        )\n        if preserve_options != expected_preserve_options:\n            return False\n        if option_values("--execution-slice-task-id") != set(\n')
    s = replace_once(s, '    parser.add_argument(\n        "--manual-completion-authority-revalidation-only",\n', option + '    parser.add_argument(\n        "--manual-completion-authority-revalidation-only",\n')
    s = replace_once(s, '        worktree_reconciliation_enabled=args.worktree_reconciliation_enabled,\n', '        preserve_task_attempts=bool(getattr(args, "preserve_task_attempts", False)),\n        worktree_reconciliation_enabled=args.worktree_reconciliation_enabled,\n')
    sources[SUPERVISOR] = s

    r = sources[RUNNER]
    r = replace_once(r, '            manual_completion_authority_revalidation_only=bool(\n                getattr(parsed, "manual_completion_authority_revalidation_only", False)\n            ),\n', '            manual_completion_authority_revalidation_only=bool(\n                getattr(parsed, "manual_completion_authority_revalidation_only", False)\n            ),\n            preserve_task_attempts=bool(getattr(parsed, "preserve_task_attempts", False)),\n')
    r = replace_once(r, '        manual_completion_authority_revalidation_only=(\n            authority_revalidation_only\n        ),\n', '        manual_completion_authority_revalidation_only=(\n            authority_revalidation_only\n        ),\n        preserve_task_attempts=bool(getattr(parsed, "preserve_task_attempts", False)),\n')
    r = replace_once(r, '            max_task_attempts=int(getattr(parsed, "max_task_attempts", 0)),\n', '            max_task_attempts=int(getattr(parsed, "max_task_attempts", 0)),\n            preserve_task_attempts=bool(getattr(parsed, "preserve_task_attempts", False)),\n')
    # The explicit database constructor helper previously relies on its own
    # default cap; leave that unrelated behavior unchanged.
    r = replace_once(r, '        provider_fn=provider_fn,\n', '        preserve_task_attempts=bool(getattr(parsed, "preserve_task_attempts", False)),\n        provider_fn=provider_fn,\n')
    sources[RUNNER] = r

    c = sources[CONFIGURED]
    c = replace_once(c, '        "retry_budget_guardrail_enabled",\n', '        "preserve_task_attempts",\n        "retry_budget_guardrail_enabled",\n')
    c = replace_once(c, '    if payload.get("retry_budget_guardrail_enabled") is False:\n', '    if payload.get("preserve_task_attempts") is True:\n        args.append("--preserve-task-attempts")\n    if payload.get("retry_budget_guardrail_enabled") is False:\n')
    sources[CONFIGURED] = c
    return sources


def main() -> None:
    proposed = proposed_sources()
    patch_parts = []
    provenance = []
    for rel, after in proposed.items():
        before = (BASE / rel).read_text()
        ast.parse(after, filename=rel)
        out = DEST / 'proposed' / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(after)
        patch_parts.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/' + rel, tofile='b/' + rel))
        provenance.append({'path': rel, 'original_sha256': hashlib.sha256(before.encode()).hexdigest(), 'proposed_sha256': hashlib.sha256(after.encode()).hexdigest(), 'original_bytes': len(before.encode()), 'proposed_bytes': len(after.encode())})
    (DEST / 'preserve-task-attempts.patch').write_text(''.join(patch_parts))
    (DEST / 'BASE_PROVENANCE.json').write_text(json.dumps({'schema': 'legal-ir-source-patch-proposal@1', 'base_root': str(BASE), 'scope': 'source-only-not-executed', 'files': provenance}, indent=2) + '\n')


if __name__ == '__main__':
    main()
