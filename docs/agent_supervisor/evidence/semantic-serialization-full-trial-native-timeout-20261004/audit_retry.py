"""Bounded command/script comparison; never executes a benchmark or reads private bodies."""
from pathlib import Path
import datetime
import hashlib
import json

B = Path('/home/barberb/lift_coding/artifacts/semantic-identity-reuse-native-20261004')
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original = (B / 'full_trial.py').read_text()
retry = (B / 'full_trial_retry.py').read_text()
assert retry.count("O=B/'full-retry-02'\n") == 1
normalized_retry = retry.replace("O=B/'full-retry-02'\n", '').replace('O/', 'B/')
assert normalized_retry == original, 'Runner differs beyond destination-root relocation'
records = {}
qualified_sha = sha(B / 'qualification-01/qualification.json')
archive_audit = read(B / 'build-audit.json')
for phase in ('prepare', 'execute'):
    a_path = B / f'{phase}-command.json'
    b_path = B / 'full-retry-02' / f'{phase}-command.json'
    a, b = read(a_path), read(b_path)
    assert a.keys() == b.keys()
    assert a['argv'].count('--output') == b['argv'].count('--output') == 1
    ai, bi = a['argv'].index('--output') + 1, b['argv'].index('--output') + 1
    assert ai == bi
    assert a['argv'][ai] == str(B / 'trial-01')
    assert b['argv'][bi] == str(B / 'full-retry-02/trial-01')
    normalized = json.loads(json.dumps(b))
    normalized['argv'][bi] = a['argv'][ai]
    assert a == normalized, f'{phase} differs beyond --output value'
    assert a['qualification_sha256'] == qualified_sha
    assert a['qualified_archive'] == archive_audit['archive_sha256']
    assert a['qualified_manifest'] == archive_audit['manifest_sha256']
    assert a['input_review_sha256'] == sha(B / 'input-review.json')
    assert a['intent_requirement_contract_sha256'] == sha(B / 'intent-requirements.json')
    records[phase] = {
        'original_command_sha256': sha(a_path),
        'retry_command_sha256': sha(b_path),
        'only_changed_field': 'argv[--output value]',
        'normalized_records_equal': True,
        'cwd_equal': True,
        'declared_environment_equal': True,
        'archive_manifest_qualification_input_review_and_intent_digests_equal': True,
        'provider_model_reasoning_and_planning_equal': True,
        'resource_deadline_attempt_worker_and_retry_settings_equal': True,
    }

decl_path = B / 'full-retry-02/retry-declaration.json'
decl = read(decl_path)
assert decl['runner_sha256'] == sha(B / 'full_trial_retry.py')
assert decl['prior_trial_metadata_sha256'] == sha(B / 'full-trial-metadata.json')
assert decl['host_sample_sha256'] == sha(B / 'host-before-full-retry-02.json')
assert decl['prior_trial_retained'] is True
assert decl['changes_to_source_archive_or_limits'] is False
assert decl['additional_declared_trials'] == 1
assert decl['harbor_automatic_retries'] == 0

package = B / 'public-full-trial-01-evidence'
manifest = read(package / 'manifest.json')
listed = set()
for entry in manifest['files']:
    rel = entry['path']
    assert rel not in listed and not Path(rel).is_absolute() and '..' not in Path(rel).parts
    listed.add(rel)
    p = package / rel
    assert not p.is_symlink()
    assert p.stat().st_size == entry['bytes']
    assert sha(p) == entry['sha256']
actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file() and p != package / 'manifest.json'}
assert listed == actual

result = {
    'schema': 'full-trial-retry-equivalence-audit@1',
    'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer_scope': 'Read-only original/retry runner and command metadata; local audit output only.',
    'runner': {
        'original_sha256': sha(B / 'full_trial.py'),
        'retry_sha256': sha(B / 'full_trial_retry.py'),
        'normalized_source_exactly_equal': True,
        'permitted_change': 'Introduce the full-retry-02 destination root and relocate prepared trial/config and command/stdout/stderr/exit artifacts beneath it.',
        'qualification_and_fresh_source_task_checks_preserved': True,
    },
    'commands': records,
    'qualified_archive_sha256': archive_audit['archive_sha256'],
    'qualified_manifest_sha256': archive_audit['manifest_sha256'],
    'qualification_sha256': qualified_sha,
    'declaration_sha256': sha(decl_path),
    'declaration_runner_prior_receipt_and_later_host_sample_bindings_valid': True,
    'first_trial_package': {
        'manifest_sha256': sha(package / 'manifest.json'),
        'member_count': len(listed),
        'exact_closure_and_all_member_digests_valid': True,
    },
    'limitations': [
        'Equality of declared environment overrides does not prove equality of all inherited ambient environment variables or live host pressure.',
        'No full archive rescan or independent re-execution of source/task admission checks was performed by this audit.',
        'The later pre-retry host sample does not establish causation for the first trial failure.',
        'Command equivalence is not evidence of retry task success; retry runtime and cleanup evidence require a separate receipt.',
    ],
    'private_auth_verifier_or_model_bodies_read': False,
    'benchmark_or_docker_work_executed': False,
    'repository_or_runtime_input_mutations': False,
    'audit_script_sha256': sha(Path(__file__)),
    'valid': True,
}
(OUT / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'valid': True, 'audit_sha256': sha(OUT / 'audit.json'), 'first_trial_members': len(listed)}))
