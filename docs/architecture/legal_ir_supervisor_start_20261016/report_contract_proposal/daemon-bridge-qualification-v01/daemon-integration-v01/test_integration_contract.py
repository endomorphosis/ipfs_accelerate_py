"""Source controls for the actual composed daemon, without a fake supervisor.

These checks establish ordering and preserved producer code only. They do not
establish successful provider dispatch, native custody, or task acceptance.
Run this file only inside the reviewed native qualification envelope.
"""
import ast
import unittest

from build_patch import assemble


def methods(source, name='PortalImplementationDaemon'):
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == name)
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def calls(node, name):
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)
            and getattr(child.func, 'attr', getattr(child.func, 'id', None)) == name]


def normalized(node):
    return ast.dump(node, include_attributes=False)


class ActualDaemonSourceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.proposed = assemble()
        cls.old = methods(cls.original)
        cls.new = methods(cls.proposed)

    def test_opt_in_default_is_false(self):
        ctor = self.new['__init__']
        defaults = dict(zip([arg.arg for arg in ctor.args.kwonlyargs], ctor.args.kw_defaults))
        self.assertIs(defaults['report_only_candidate_freezing'].value, False)

    def test_no_counter_producer_or_rollback_changes(self):
        for name in ('_record_task_attempt', '_restore_task_attempt', '_task_attempt'):
            self.assertEqual(normalized(self.old[name]), normalized(self.new[name]))

    def test_host_required_before_original_pass_work(self):
        method = self.new['run_once']
        guard = calls(method, 'guard_pass_entry')[0]
        retained = calls(method, '_retained_provider_capacity_deferral')[0]
        self.assertLess(guard.lineno, retained.lineno)

    def test_profile_refusal_precedes_original_attempt_effects(self):
        method = self.new['_run_implementation']
        guard = calls(method, 'require_attempt_profile')[0]
        for name in ('_require_plan_runtime_before_claim', '_try_acquire_implementation_dispatch_intent',
                     '_require_primary_provider_readiness'):
            self.assertLess(guard.lineno, min(call.lineno for call in calls(method, name)))

    def test_preflight_precedes_only_original_ordinal_consumption(self):
        method = self.new['_run_implementation_in_ephemeral_worktree']
        preflight = calls(method, 'preflight_candidate')[0]
        starts = calls(method, '_mark_implementation_started')
        self.assertEqual(len(starts), 1)
        self.assertLess(preflight.lineno, starts[0].lineno)
        self.assertEqual([keyword.arg for keyword in preflight.keywords], ['command'])

    def test_frame_retained_before_ephemeral_execution(self):
        method = self.new['_run_implementation']
        capture = calls(method, 'capture_attempt_frame')[0]
        self.assertLess(capture.lineno, min(call.lineno for call in
                                         calls(method, '_run_implementation_in_ephemeral_worktree')))
        self.assertIn('acquired_resource_claims', [keyword.arg for keyword in capture.keywords])

    def test_original_decision_callback_still_owns_dispatch(self):
        method = self.new['_run_implementation_in_ephemeral_worktree']
        callback = next(node for node in ast.walk(method)
                        if isinstance(node, ast.FunctionDef) and node.name == 'invoke_provider')
        dispatch = calls(callback, 'dispatch_provider')
        self.assertEqual(len(dispatch), 1)
        self.assertLess(dispatch[0].lineno, calls(callback, 'run_process_group_stream')[0].lineno)
        decision = next(call for call in calls(method, '_decision_runtime_mutation')
                        if any(isinstance(arg, ast.Name) and arg.id == 'invoke_provider' for arg in call.args))
        self.assertIsInstance(decision.args[-1], ast.Name)
        self.assertIn('provider_authorized', ast.unparse(callback.body[:3]))

    def test_inert_result_exits_before_legacy_acceptance(self):
        method = self.new['_run_implementation_in_ephemeral_worktree']
        result = calls(method, 'require_candidate_result')[0]
        for name in ('_board_completion_decision', '_mark_implementation_finished'):
            self.assertLess(result.lineno, min(call.lineno for call in calls(method, name)))
        parent_return = [node for node in ast.walk(method) if isinstance(node, ast.Return)
                         and node.value is result]
        self.assertEqual(len(parent_return), 1)

    def test_both_error_boundaries_precede_generic_terminal_effects(self):
        for name in ('_run_implementation', '_run_implementation_in_ephemeral_worktree'):
            handlers = [node for node in ast.walk(self.new[name]) if isinstance(node, ast.ExceptHandler)
                        and isinstance(node.type, ast.Name) and node.type.id == 'Exception'
                        and calls(node, 'handle_attempt_exception')]
            self.assertEqual(len(handlers), 1)
            first = handlers[0].body[0]
            self.assertIsInstance(first, ast.If)
            self.assertEqual(ast.unparse(first.test), 'self.report_only_candidate_freezing')
            self.assertIsInstance(first.body[0], ast.Return)

    def test_attempt_and_pass_finalizers_guard_first(self):
        for name, stage in (('_run_implementation', 'implementation_claim_release'),
                            ('_run_once', 'pass_dispatch_intent_reconciliation')):
            guarded = [node for node in ast.walk(self.new[name]) if isinstance(node, ast.Try)
                       and node.finalbody and calls(node.finalbody[0], 'before_original_finalizer')]
            self.assertEqual(len(guarded), 1)
            first = guarded[0].finalbody[0]
            self.assertEqual(ast.unparse(first.test), 'self.report_only_candidate_freezing')
            call = calls(first, 'before_original_finalizer')[0]
            self.assertEqual(next(keyword.value.value for keyword in call.keywords if keyword.arg == 'stage'), stage)

    def test_dispatch_capture_does_not_accept_caller_baseline(self):
        call = calls(self.new['_mark_provider_launch_boundary'], 'capture_dispatch')[0]
        self.assertEqual([ast.unparse(arg) for arg in call.args],
                         ['self', 'state', 'task', 'attempt', 'callable_invocation', 'control_plane_connection'])

    def test_empty_borrowed_row_does_not_migrate_legacy_json(self):
        load = methods(self.proposed, 'PortalTaskState')['load']
        guard = calls(load, 'task_state_connection_is_borrowed')[0]
        self.assertLess(guard.lineno, min(call.lineno for call in calls(load, 'read_text')))

    def test_acceptance_merge_and_validation_producers_preserved(self):
        for name in ('_board_completion_decision', '_mark_implementation_finished',
                     '_clear_active_execution_state', '_run_validation_commands'):
            self.assertEqual(normalized(self.old[name]), normalized(self.new[name]))

    def test_database_daemon_untouched(self):
        self.assertEqual({name: normalized(node) for name, node in methods(self.original, 'DatabaseImplementationDaemon').items()},
                         {name: normalized(node) for name, node in methods(self.proposed, 'DatabaseImplementationDaemon').items()})


if __name__ == '__main__':
    unittest.main()
