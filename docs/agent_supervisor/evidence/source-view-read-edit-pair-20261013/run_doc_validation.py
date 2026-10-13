import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

root = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
out = Path(__file__).parent / 'documentation-validation'
out.mkdir(exist_ok=False)
paths = [
    'docs/agent_supervisor/source_view_read_edit_pair.md',
    'docs/agent_supervisor/worker_source_views.md',
    'docs/agent_supervisor/reversible_codebase_views_plan.md',
    'docs/agent_supervisor/evidence/source-view-read-edit-pair-20261013/README.md',
]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
command = [sys.executable, 'scripts/docs/run_documentation_gates.py']
before = {p: sha(root / p) for p in paths}
result = subprocess.run(command, cwd=root, capture_output=True)
(out / 'stdout.txt').write_bytes(result.stdout)
(out / 'stderr.txt').write_bytes(result.stderr)
checker_path = root / 'scripts/docs/check_current_docs_links.py'
spec = importlib.util.spec_from_file_location('doc_links', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
errors = [error for p in paths for error in checker.check_file(root / p, check_anchors=True)]
after = {p: sha(root / p) for p in paths}
record = dict(schema='source-view-pair-documentation-validation@1',
    command=command, cwd=str(root), exit_code=result.returncode,
    workflow_blocks_passed=4 if result.returncode == 0 else None,
    extra_links_and_anchors=True, errors=errors, source_before=before,
    source_after=after, sources_unchanged=before == after,
    checker_sha256=sha(checker_path), runner_sha256=sha(Path(__file__)),
    stdout_sha256=sha(out / 'stdout.txt'), stderr_sha256=sha(out / 'stderr.txt'))
(out / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
if result.returncode or errors or before != after:
    raise SystemExit(1)
