"""Copy an explicit closed-evidence allowlist after the final live freeze check."""
import hashlib
import json
from pathlib import Path
import shutil

B = Path(__file__).resolve().parent
W = Path('/home/barberb/lift_coding/.worktrees/supervisor-tool-event-coverage-20261015')
DEST = W / 'docs/agent_supervisor/evidence/tool-event-coverage-20261015'
LABELS = ('capture-final01', 'routes-dev01', 'codec-final01', 'external-final01')
NAMES = (
    'analyze_closed.py', 'closed-arithmetic-analysis-01.json',
    'closed-coverage-analysis-01.json', 'null-status-protocol-audit.json',
    'authored-e2e/test_authored_pipeline.py', 'final_freeze_check.py',
    'final-freeze-check-01.json', 'qualification-final01.json', 'qualify.py',
    'run-tests.py', 'isolated-pruning-profile-design.md',
    'isolated-pruning-profile-audit.json', 'publish_evidence.py',
    'live/actual-public-diagnostic-01.json', 'live/closed_export.py',
    'live/environment.json', 'live/largest-eigenval-public-profile.json',
    'live/local-asset-readiness-draft01.json', 'live/prepare_draft.py',
    'live/recipe-draft.json', 'live/recipe.json', 'live/run.py',
    'live/scaffold-origin.json', 'live/source-freeze.json',
    'live/test_closed_export.py', 'live/test_run.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    final = json.loads((B / 'final-freeze-check-01.json').read_bytes())
    qualification = json.loads((B / 'qualification-final01.json').read_bytes())
    assert final['source_frozen_unchanged'] is True
    assert final['exact_container_absence_observations'] == [True, True]
    assert final['runtime_commit'] == qualification['runtime_commit']
    assert final['diagnostic_sha256'] == sha(B / 'live/actual-public-diagnostic-01.json')
    names = list(NAMES)
    names.extend(label + suffix for label in LABELS
                 for suffix in ('-result.json', '-source-pins.json', '.xml'))
    names.extend('live/' + step + suffix for step in ('bundle', 'verify', 'execute')
                 for suffix in ('-start.json', '-result.json'))
    assert len(names) == len(set(names))
    for name in names:
        path = B / name
        assert path.is_file() and not path.is_symlink()
        assert 'private' not in path.relative_to(B).parts
    DEST.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in sorted(names):
        source, target = B / name, DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        rows.append(dict(path=name, bytes=target.stat().st_size, sha256=sha(target)))
    manifest = dict(schema='tool-event-coverage-evidence-manifest@1',
        runtime_commit=final['runtime_commit'], files=rows,
        private_owner_artifacts_included=False, raw_provider_content_included=False,
        official_verifier_executed=False, source_freeze_released_after_final_check=True)
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(files=len(rows), destination=str(DEST),
        manifest_sha256=sha(DEST / 'manifest.json'))))


if __name__ == '__main__':
    main()
