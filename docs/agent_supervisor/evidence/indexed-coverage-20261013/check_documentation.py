"""Retain local gates and explicit links for the context projection report."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

root = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
argv = [sys.executable, 'scripts/docs/run_documentation_gates.py']
run = subprocess.run(argv, cwd=root, capture_output=True)
(out / 'gates.stdout').write_bytes(run.stdout)
(out / 'gates.stderr').write_bytes(run.stderr)
spec = importlib.util.spec_from_file_location('doclinks', root / 'scripts/docs/check_current_docs_links.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
relative = ['docs/agent_supervisor/indexed_coverage_readiness.md', 'docs/agent_supervisor/evidence/indexed-coverage-20261013/README.md']
errors = [error for name in relative for error in module.check_file(root / name, check_anchors=True)]
diff = subprocess.run(['git', 'diff', '--check'], cwd=root, capture_output=True)
hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = dict(schema='indexed-coverage-documentation-checks@1', cwd=str(root),
    command=argv, gate_exit_code=run.returncode, additional_files=relative,
    explicit_link_errors=errors, git_diff_check_exit_code=diff.returncode,
    git_diff_check_output=(diff.stdout + diff.stderr).decode(),
    source_sha256={name: hash_file(root / name) for name in relative},
    helper_sha256=hash_file(Path(__file__)),
    passed=run.returncode == 0 and not errors and diff.returncode == 0)
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
raise SystemExit(0 if result['passed'] else 1)
