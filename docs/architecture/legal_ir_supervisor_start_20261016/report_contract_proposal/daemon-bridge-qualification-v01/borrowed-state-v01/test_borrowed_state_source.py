"""Cold source contracts only; no connection/owner/outcome is fabricated."""
import ast
from pathlib import Path
import unittest

HERE = Path(__file__).absolute().parent
RELATIVE = Path('ipfs_accelerate_py/agent_supervisor/todo_daemon/portal_task_state_control_plane.py')
BASE = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node, attribute):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attribute]


class BorrowedSourceControls(unittest.TestCase):
    def unchanged_sql(self, name):
        before = calls(function(BASE / RELATIVE, name), 'execute')
        after = calls(function(HERE / 'proposed' / RELATIVE, name), 'execute')
        self.assertEqual([ast.dump(n, include_attributes=False) for n in before],
                         [ast.dump(n, include_attributes=False) for n in after])

    def test_original_lane_save_sql_and_parameters(self):
        self.unchanged_sql('save_portal_task_state')

    def test_original_lane_load_sql_and_parameters(self):
        self.unchanged_sql('load_portal_task_state')

    def test_original_board_save_sql_and_parameters(self):
        self.unchanged_sql('_save_board_task_state')

    def test_original_board_load_sql_and_parameters(self):
        self.unchanged_sql('_load_board_task_state')

    def test_default_connect_body_preserved_after_reopen_guard(self):
        before = function(BASE / RELATIVE, '_connect')
        after = function(HERE / 'proposed' / RELATIVE, '_connect')
        self.assertEqual([ast.dump(n, include_attributes=False) for n in before.body],
                         [ast.dump(n, include_attributes=False) for n in after.body[1:]])

    def test_borrowed_context_never_owns_connection_lifetime(self):
        node = function(HERE / 'proposed' / RELATIVE, '_state_connection')
        borrowed_branch = next(n for n in node.body if isinstance(n, ast.If))
        text = ast.unparse(borrowed_branch)
        for forbidden in ('_connect(', '.close(', '.commit(', '.rollback(', 'reconnect', 'CREATE ', 'duckdb.connect('):
            self.assertNotIn(forbidden, text)
        self.assertIn('with binding.condition:', text)
        self.assertIn('_check_borrowed_locked(binding)', text)
        self.assertIn('yield binding.connection', text)

    def test_nonowning_scope_never_enters_or_exits_native_wrapper(self):
        node = function(HERE / 'proposed' / RELATIVE, 'borrow_task_state_connection')
        text = ast.unparse(node)
        for forbidden in ('open_duckdb_connection(', 'DuckDBConnection(', '.close(', '.commit(',
                          '.rollback(', 'reconnect_exclusive_owner(', 'with connection:', '_project_ducklake('):
            self.assertNotIn(forbidden, text)
        self.assertIn('yield connection', text)

    def test_source_profile_refuses_unreviewed_backend_objects(self):
        node = function(HERE / 'proposed' / RELATIVE, 'require_borrowed_task_state_source_profile')
        text = ast.unparse(node)
        self.assertIn('task_source is None', text)
        self.assertIn('plan_revision_store is None', text)
        self.assertIn('configured_parallel_execution_plan is None', text)
        self.assertIn('active_plan_binding is None', text)
        self.assertNotIn('getattr(', text)


if __name__ == '__main__':
    unittest.main()
