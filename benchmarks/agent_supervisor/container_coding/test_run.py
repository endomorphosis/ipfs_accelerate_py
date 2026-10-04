import unittest
from run import TASKS, prove, state_query


class StateTests(unittest.TestCase):
    def test_completion_requires_tests(self):
        prove(state_query(TASKS, {"sum"}, [], set()), "sat")
        prove(state_query(TASKS, {"sum"}, [], {"sum"}), "unsat")

    def test_dependencies_and_exclusive_lease(self):
        prove(state_query(TASKS, [], ["sum_followup"], []), "sat")
        prove(state_query(TASKS, ["sum"], ["sum_followup"], ["sum"]), "unsat")
        prove(state_query(TASKS, ["sum"], ["sum"], ["sum"]), "sat")

    def test_conflict_without_dependency_violation(self):
        tasks = [dict(t, deps=[]) for t in TASKS]
        prove(state_query(tasks, [], ["sum", "sum_followup"], []), "sat")

    def test_unknown_and_duplicate_leases_rejected(self):
        for active in (["unknown"], ["sum", "sum"]):
            with self.assertRaises(ValueError):
                state_query(TASKS, [], active, [])


if __name__ == "__main__":
    unittest.main()
