"""Publish an explicit metadata/helper allowlist; never recurse over raw trials."""
from pathlib import Path
import hashlib
import json
import re
import shutil

R = Path(__file__).resolve().parent
Q = Path('/home/barberb/lift_coding/.worktrees/output-observer-queued-20261013')
DEST = Q / 'docs/agent_supervisor/evidence/numerical-live-comparison-20261013'
FILES = '''
pair-design.json
historical-recipe-erratum.json
pair-lock.json
pair-observation-final.json
three-arm-observation-final.json
environment.json
launch-recipe-candidates.json
run_trial.py
collect_pair.py
test_collect_pair.py
collect_three_arms.py
test_collect_three_arms.py
collector-qualification.json
collector-independent-review.json
preparation-independent-audit.json
runtime-preparation-independent-review.json
final-lock-independent-review.json
full-pair-independent-review.json
three-arm-independent-review.json
no-index-result-review.json
retained-assets-inspection.json
allocation-event-contract-diagnosis.json
output-observer-queued-qualification.json
output-observer-queued-postmerge-qualification.json
queued-output-observer-independent-review.json
live-preparation/bundle-result.json
live-preparation/verification-result.json
provider-readiness/route-audit.json
provider-readiness/grok-metadata-precheck-01.json
provider-readiness/grok-metadata-precheck.json
provider-readiness/public-provider-assets.json
collector-tests-01/exit.json
collector-tests-04/exit.json
collector-tests-04/command.json
collector-tests-04/stdout
three-arm-collector-tests-04/result.json
three-arm-collector-tests-04/command.json
three-arm-collector-tests-04/pytest.xml
output-observer-queued-tests-01/exit.json
output-observer-queued-tests-03/exit.json
output-observer-queued-tests-03/command.json
output-observer-queued-tests-03/junit.xml
live-trials/codex-full-01/launch-start.json
live-trials/codex-full-01/launch-result.json
live-trials/codex-no-index-01/launch-start.json
live-trials/codex-no-index-01/launch-result.json
next-native-baseline/execution-lock.json
next-native-baseline/release.json
next-native-baseline/source-pins.json
next-native-baseline/environment.json
next-native-baseline/environment-notes.json
next-native-baseline/recipe.json
next-native-baseline/run_native.py
next-native-baseline/host-cli-receipt.json
next-native-baseline/native-preparation-independent-review.json
next-native-baseline/native-lock-independent-review.json
next-native-baseline/native-codex-01/launch-start.json
next-native-baseline/native-codex-01/launch-result.json
'''.split()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def checked(path):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 * 1024 * 1024, path
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    assert not re.search(r'-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH)', text), path
    assert not re.search(r'\b(?:hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,})\b', text), path
    if path.suffix == '.json':
        json.loads(text)
    return raw

def main():
    assert not DEST.exists(), 'Evidence directory must be fresh'
    # Check everything before creating the destination. Explicit reviews are
    # required in addition to this narrow credential-pattern guard.
    rows = [(name, checked(R / name)) for name in FILES]
    report = checked(R / 'live-three-arm-report.md')
    DEST.mkdir(parents=True)
    for name, raw in rows:
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    note = '''\n## Evidence seal\n\nThis directory contains reviewed aggregate receipts and helper sources, not raw\nmodel transcripts, credentials, databases, hidden verifier contents or binary\nassets. Helpers that replay launches require the separately retained local trial\ninputs; this seal alone is not a runnable benchmark distribution. No trial is\nrerun by verifying its file hashes. Earlier failed development fixtures and the\nfailed Grok setup probe remain labeled in their qualification records.\n'''
    (DEST / 'README.md').write_text(report.decode() + note)
    (DEST / '.gitattributes').write_text('**/stdout -whitespace\n')
    shutil.copyfile(Path(__file__), DEST / 'seal_publication.py')
    files = {str(p.relative_to(DEST)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
             for p in sorted(DEST.rglob('*')) if p.is_file()}
    manifest = dict(schema='numerical-live-comparison-evidence@1', files=files,
        runtime_accelerate_commit='a8f57366ff4110a87f1f13d1e60992f83f689821',
        runtime_datasets_commit='28b9ef1f0b9da51e87b1e68ee565b43681397109',
        publication_base_commit='f7cf87104e079d1159226c137449e29bf98bfd40',
        live_attempts=3, all_live_attempts_retained=True,
        proof_obligations_discharged=0, causal_advantage_claimed=False,
        output_observer_fix_in_live_runtime=False,
        source_files={name:sha((Q / name).read_bytes()) for name in (
            'benchmarks/agent_supervisor/container_coding/terminal_output_observation.py',
            'benchmarks/agent_supervisor/container_coding/test_terminal_output_queued_observation.py')})
    (DEST / 'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    result = dict(files=len(files)+1, manifest_sha256=sha((DEST/'manifest.json').read_bytes()), destination=str(DEST))
    (R / 'publication-seal-result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
