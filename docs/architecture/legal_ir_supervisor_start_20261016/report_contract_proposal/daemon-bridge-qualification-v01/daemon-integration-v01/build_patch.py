"""Assemble an opt-in daemon proposal without importing or changing the daemon.

This authoring program reads exact frozen inputs and emits a fresh patch. It
does not start the supervisor, open its database, or run a provider or Git.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path('/home/barberb/lift_coding')
CONTROLLER = WORKSPACE / 'external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller'
RELATIVE = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
BASE_BUILDER = WORKSPACE / 'artifacts/legal-ir-callable-handoff-20261017-01/report-freezer-v02/build_patch.py'
ORIGINAL_SHA = '23addb0dfc23e2cb093e7f74abbb7dc84a5308c24259e1eb0e25d721c0a80bed'
FREEZER_BASE_SHA = '60aa2303b5221c5f685dd1b724cf80e10de758e30c5c31aa0996cf4c97abb60b'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_one(source, old, new):
    if source.count(old) != 1:
        raise ValueError('exact_single_source_anchor_required: ' + old[:100])
    return source.replace(old, new, 1)


def method_edit(source, name, edit):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'PortalImplementationDaemon')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == name)
    lines = source.splitlines(keepends=True)
    before = ''.join(lines[:method.lineno - 1])
    body = ''.join(lines[method.lineno - 1:method.end_lineno])
    after = ''.join(lines[method.end_lineno:])
    return before + edit(body) + after


def successor(source):
    # This narrow class-load guard prevents borrowed-state reads from importing
    # a legacy JSON projection into the supervisor's already-open database.
    anchor = '        elif routed == {}:\n            if not path.exists():\n'
    source = replace_one(source, anchor,
        '        elif routed == {}:\n'
        '            from .portal_task_state_control_plane import task_state_connection_is_borrowed\n'
        '            if task_state_connection_is_borrowed(path):\n'
        '                raise RuntimeError("borrowed task state requires its original authoritative row")\n'
        '            if not path.exists():\n')
    source = replace_one(source, '        self._report_only_before_attempt = None\n        self._report_only_dispatch = None\n',
        '        self._report_only_before_attempt = None\n'
        '        self._report_only_dispatch = None\n'
        '        self._report_only_native_host = None\n')
    source = replace_one(source,
        '                self, state, task, attempt, workspace_path, baseline_ref,\n'
        '                callable_invocation, control_plane_connection,\n',
        '                self, state, task, attempt, callable_invocation,\n'
        '                control_plane_connection,\n')

    def run_once(body):
        anchor = '        retained = self._retained_provider_capacity_deferral()\n'
        return replace_one(body, anchor,
            '        if self.report_only_candidate_freezing:\n'
            '            self._require_report_only_native_host().guard_pass_entry(self)\n'
            + anchor)
    source = method_edit(source, 'run_once', run_once)

    # Default CLI and legacy paths do not bind this object. A serialized packet,
    # task identifier, or duck-typed stand-in cannot supply a host binding.
    anchor = '    def _manual_completion_revalidation_store_path(self) -> Path:\n'
    source = replace_one(source, anchor,
        '    def _require_report_only_native_host(self):\n'
        '        from .report_only_native_host import require_report_only_native_host\n\n'
        '        return require_report_only_native_host(self)\n\n'
        '    def preflight_report_only_candidate(\n'
        '        self, state, *, task, preflight_owner, parent_phase,\n'
        '        control_plane_connection, protected_source_sha256, declared_external_storage,\n'
        '    ):\n'
        '        from .report_only_candidate_freezer import preflight_report_only_candidate\n\n'
        '        return preflight_report_only_candidate(\n'
        '            self, state, task=task, preflight_owner=preflight_owner,\n'
        '            parent_phase=parent_phase, control_plane_connection=control_plane_connection,\n'
        '            protected_source_sha256=protected_source_sha256,\n'
        '            declared_external_storage=declared_external_storage,\n'
        '        )\n\n'
        '    def report_only_preflight_workspace(self, state, *, task):\n'
        '        from .report_only_candidate_freezer import report_only_preflight_workspace\n\n'
        '        return report_only_preflight_workspace(self, state, task=task)\n\n'
        '    def report_only_preflight_storage_roots(self, state, *, task):\n'
        '        from .report_only_candidate_freezer import report_only_preflight_storage_roots\n\n'
        '        return report_only_preflight_storage_roots(self, state, task=task)\n\n' + anchor)

    def run_implementation(body):
        anchor = '        retained = self._retained_provider_capacity_deferral()\n'
        body = replace_one(body, anchor,
            '        if self.report_only_candidate_freezing:\n'
            '            self._require_report_only_native_host().require_attempt_profile(self, task, state)\n'
            + anchor)
        anchor = '            if authority_revalidation_only:\n                return self._run_manual_completion_authority_revalidation(\n'
        body = replace_one(body, anchor,
            '            if self.report_only_candidate_freezing:\n'
            '                self._require_report_only_native_host().capture_attempt_frame(\n'
            '                    self, task=task, state=state, attempt=attempt,\n'
            '                    task_claim_path=task_claim_path, task_claim_metadata=task_claim_metadata,\n'
            '                    acquired_resource_claims=acquired_resource_claims,\n'
            '                    lock_path=lock_path, lock_metadata=lock_metadata,\n'
            '                )\n' + anchor)
        # Report-only errors must never reach normal rollback/terminal handlers.
        anchor = '        except subprocess.TimeoutExpired as timeout_exc:\n'
        body = replace_one(body, anchor, anchor +
            '            if self.report_only_candidate_freezing:\n'
            '                return self._require_report_only_native_host().handle_attempt_exception(\n'
            '                    self, timeout_exc, task=task, state=state,\n'
            '                    lifecycle_record=self._active_worktree_lifecycle,\n'
            '                )\n')
        anchor = '        except Exception as exc:\n            if isinstance(exc, provider_capacity_custody.RetentionError):\n'
        body = replace_one(body, anchor,
            '        except Exception as exc:\n'
            '            if self.report_only_candidate_freezing:\n'
            '                return self._require_report_only_native_host().handle_attempt_exception(\n'
            '                    self, exc, task=task, state=state,\n'
            '                    lifecycle_record=self._active_worktree_lifecycle,\n'
            '                )\n'
            '            if isinstance(exc, provider_capacity_custody.RetentionError):\n')
        anchor = '        finally:\n            try:\n                if acquired_lock and not self._release_implementation_lock(\n'
        body = replace_one(body, anchor,
            '        finally:\n'
            '            if self.report_only_candidate_freezing:\n'
            '                self._require_report_only_native_host().before_original_finalizer(\n'
            '                    self, stage="implementation_claim_release", original_frame={\n'
            '                        "task": task, "state": state, "attempt": attempt,\n'
            '                        "task_claim_path": task_claim_path,\n'
            '                        "task_claim_metadata": task_claim_metadata,\n'
            '                        "acquired_task_claim": acquired_task_claim,\n'
            '                        "resource_claims": acquired_resource_claims,\n'
            '                        "lock_path": lock_path, "lock_metadata": lock_metadata,\n'
            '                        "acquired_lock": acquired_lock,\n'
            '                        "lifecycle": self._active_worktree_lifecycle,\n'
            '                    },\n'
            '                )\n'
            '            try:\n                if acquired_lock and not self._release_implementation_lock(\n')
        return body
    source = method_edit(source, '_run_implementation', run_implementation)

    def ephemeral(body):
        anchor = '            self._mark_implementation_started(\n'
        body = replace_one(body, anchor,
            '            if self.report_only_candidate_freezing:\n'
            '                self._require_report_only_native_host().preflight_candidate(\n'
            '                    self, state, task, command=tuple(command),\n'
            '                )\n'
            + anchor)
        anchor = '                            provider_environment = (\n                                self._implementation_process_environment(\n                                    task,\n                                    attempt=attempt,\n                                    checkpoint_dir=checkpoint_dir,\n                                )\n                            )\n'
        body = replace_one(body, anchor, anchor +
            '                            if self.report_only_candidate_freezing:\n'
            '                                return self._require_report_only_native_host().dispatch_provider(\n'
            '                                    self, state=state, task=task, attempt=attempt,\n'
            '                                    prompt=prompt, environment=provider_environment,\n'
            '                                    log_path=log_path, command=tuple(command),\n'
            '                                )\n')
        anchor = '            if retry_no_change_probe_only:\n                probe_reason = str(\n'
        body = replace_one(body, anchor,
            '            if self.report_only_candidate_freezing:\n'
            '                return self._require_report_only_native_host().require_candidate_result(self, completed)\n'
            + anchor)
        anchor = '        except subprocess.TimeoutExpired as timeout_exc:\n'
        body = replace_one(body, anchor, anchor +
            '            if self.report_only_candidate_freezing:\n'
            '                return self._require_report_only_native_host().handle_attempt_exception(\n'
            '                    self, timeout_exc, task=task, state=state, lifecycle_record=lifecycle_record,\n'
            '                )\n')
        anchor = '        except Exception as exc:\n            returncode = 1\n'
        body = replace_one(body, anchor,
            '        except Exception as exc:\n'
            '            if self.report_only_candidate_freezing:\n'
            '                return self._require_report_only_native_host().handle_attempt_exception(\n'
            '                    self, exc, task=task, state=state, lifecycle_record=lifecycle_record,\n'
            '                )\n'
            '            returncode = 1\n')
        return body
    source = method_edit(source, '_run_implementation_in_ephemeral_worktree', ephemeral)

    def pass_finalizer(body):
        anchor = '        finally:\n            retain_selected_intent = bool(\n'
        return replace_one(body, anchor,
            '        finally:\n'
            '            if self.report_only_candidate_freezing:\n'
            '                self._require_report_only_native_host().before_original_finalizer(\n'
            '                    self, stage="pass_dispatch_intent_reconciliation", original_frame={\n'
            '                        "selected": selected, "state": state,\n'
            '                        "implementation_result": implementation_result,\n'
            '                        "reusable_dispatch_intents": reusable_dispatch_intents,\n'
            '                    },\n'
            '                )\n'
            '            retain_selected_intent = bool(\n')
    source = method_edit(source, '_run_once', pass_finalizer)
    ast.parse(source)
    return source


def assemble():
    original = (CONTROLLER / RELATIVE).read_bytes()
    if sha(original) != ORIGINAL_SHA:
        raise ValueError('original_daemon_source_changed')
    # Frozen authoring helper is standard-library-only and its main is guarded.
    spec = importlib.util.spec_from_file_location('frozen_source_assembler', BASE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = module.daemon_successor(original.decode())
    if sha(base.encode()) != FREEZER_BASE_SHA:
        raise ValueError('qualified_freezer_base_changed')
    return original.decode(), successor(base)


def main():
    original, proposed = assemble()
    patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True),
        proposed.splitlines(keepends=True), fromfile='a/' + RELATIVE, tofile='b/' + RELATIVE))
    with (ROOT / 'implementation_daemon.patch').open('x') as stream:
        stream.write(patch)
    with (ROOT / 'ASSEMBLY.json').open('x') as stream:
        json.dump({'schema': 'legal-ir-daemon-bridge-assembly@1',
                   'original_sha256': ORIGINAL_SHA, 'freezer_base_sha256': FREEZER_BASE_SHA,
                   'proposed_sha256': sha(proposed.encode()), 'patch_sha256': sha(patch.encode()),
                   'runtime_enabled': False, 'qualification': 'source assembly only'}, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
