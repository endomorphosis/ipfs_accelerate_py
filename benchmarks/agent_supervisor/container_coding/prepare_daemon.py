"""Seed a disposable repo and invoke the native objective/subgoal/task generator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


TESTS = '''import unittest
from operations import sum_values, unique_values
class OperationsTests(unittest.TestCase):
    def test_sum(self):
        for values, expected in [([],0),([1,-2,4],3),([-4,-3],-7)]:
            self.assertEqual(sum_values(values), expected)
    def test_unique(self):
        for values, expected in [([],[]),([2,1,2],[2,1]),([3,3,3],[3])]:
            self.assertEqual(unique_values(values), expected)
if __name__ == '__main__': unittest.main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-with-router", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    repo = root / "repo"
    if repo.exists():
        parser.error("repository already exists; choose a fresh output directory")
    repo.mkdir(parents=True)
    (repo / "operations.py").write_text("def sum_values(values):\n    return None\n\ndef unique_values(values):\n    return None\n")
    (repo / "test_operations.py").write_text(TESTS)
    (repo / "README.md").write_text(
        "# Supervisor benchmark\nImplement sum_values and unique_values in operations.py. "
        "Preserve first-occurrence order for unique_values. Empty inputs and negative integers "
        "must work. Only operations.py is an implementation output. Do not change test_operations.py. "
        "Validate with python3 -m unittest -v test_operations.\n")
    (repo / ".gitignore").write_text("__pycache__/\n.runtime/\n")
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(repo), *argv], text=True)
    git("init", "-b", "main")
    git("config", "user.name", "Supervisor Benchmark")
    git("config", "user.email", "benchmark@localhost")
    git("add", ".")
    git("commit", "-m", "Seed failing coding benchmark")
    baseline = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_operations"],
                              cwd=repo, capture_output=True, text=True)
    (root / "baseline-tests.log").write_text(baseline.stdout + baseline.stderr)
    if baseline.returncode == 0:
        raise RuntimeError("broken seed unexpectedly passed")
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import ensure_objective_tracking_document
    obj = repo / "objectives.md"
    ensure_objective_tracking_document(
        obj, ultimate_goal=(repo / "README.md").read_text(),
        root_evidence=["sum_values implementation", "unique_values implementation"],
        goal_prefix="BENCH-G", root_goal_title="Repair benchmark operations")
    obj.write_text(obj.read_text().replace("Outputs: ipfs_accelerate_py/agent_supervisor, docs",
                                          "Outputs: operations.py").replace(
        "Validation: test -f " + str(obj), "Validation: python3 -m unittest -v test_operations"))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    cmd = [sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon",
           "--repo-root", str(repo), "--objective-path", "objectives.md", "--todo-path", "tasks.todo.md",
           "--discovery-dir", ".runtime/discovery", "--bundle-dir", ".runtime/bundles",
           "--dataset-dir", ".runtime/dataset", "--graph-path", ".runtime/graph.json",
           "--objective-generation-path", ".runtime/generation.json", "--task-prefix", "BENCH",
           "--goal-prefix", "BENCH-G", "--refine-objective-heap", "--max-refinement-children", "2",
           "--max-refinement-depth", "1", "--max-findings", "3", "--surplus-findings-per-goal", "1",
           "--no-persist-ast-dataset", "--no-todo-vector-index", "--no-reconcile-goal-completion",
           "--objective-generation-max-new-work", "3", "--objective-generation-max-open-work", "6"]
    if args.plan_with_router:
        cmd += ["--generate-plan-branches", "--plan-branch-count", "1", "--plan-router-provider",
                "grok_cli", "--plan-router-model", "grok-4.6", "--plan-router-timeout", "90"]
    generated = subprocess.run(cmd, cwd=repo, env=env, text=True, capture_output=True, timeout=420)
    (root / "generation.log").write_text(generated.stderr)
    (root / "generation.json").write_text(generated.stdout)
    generated.check_returncode()
    payload = json.loads(generated.stdout)
    git("add", "objectives.md", "tasks.todo.md", "data")
    # Workers receive tracked files through Git worktrees; ignored context is absent.
    context_files = sorted((repo / ".runtime/discovery").glob("*.md"))
    context_files += sorted((repo / ".runtime/bundles").glob("*.md"))
    git("add", "-f", *(str(path.relative_to(repo)) for path in context_files))
    git("commit", "-m", "Generate objective hierarchy and task packets with supervisor objective daemon")
    print(json.dumps({"repository": str(repo), "goals": payload["objective_goal_count"],
                      "tasks": payload["generated_count"], "refined_goals": payload["refined_goal_ids"]}))


if __name__ == "__main__":
    main()
