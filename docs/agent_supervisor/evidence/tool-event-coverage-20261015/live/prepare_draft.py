"""Prepare a disabled successor public census; no packaging or Docker calls."""
from pathlib import Path
import hashlib
import json

A = Path(__file__).resolve().parent
PREVIOUS = Path('/home/barberb/lift_coding/artifacts/tool-event-census-20261015/live')
SOURCE = Path('/home/barberb/lift_coding/.worktrees/supervisor-tool-event-coverage-20261015')


def write(name, value):
    with (A / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def binding(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


prior = json.loads((PREVIOUS / 'recipe-draft.json').read_bytes())
old_source = prior['source']


def relocate(value):
    if type(value) is str:
        return value.replace(str(PREVIOUS), str(A)).replace(old_source, str(SOURCE))
    if type(value) is list:
        return [relocate(item) for item in value]
    if type(value) is dict:
        return {key: relocate(item) for key, item in value.items()}
    return value


recipe = relocate(prior)
recipe.update(schema='tool-event-public-diagnostic-launch-recipe@1',
    status='disabled_pending_qualification_and_source_freeze',
    expected_working_set_census_schema='router-semantic-working-set-census@3',
    source_base_commit='d0690f8454d0126d7cff6a1d99975ca252110d4f',
    inline_evidence_dispatch_enabled=False, semantic_pack_projection_dispatch_enabled=False)
recipe.pop('expected_working_set_census_schema', None)
recipe.update(working_set_census_only=False, semantic_working_set='legacy', candidate_rounds=1,
    doctor_coding_policy=None, grok_coding_max_turns=8, disable_intent_autoencoder=True,
    expected_progress_schema='native-grok-stream-progress@5',
    expected_tool_event_schema='native-grok-tool-events@2', max_coding_invocations=1,
    provider_fallback=False, automatic_retries=0, official_verifier_enabled=False)
write('recipe-draft.json', recipe)
write('environment.json', relocate(json.loads((PREVIOUS / 'environment.json').read_bytes())))
write('local-asset-readiness-draft01.json', relocate(json.loads((PREVIOUS / 'local-asset-readiness-draft01.json').read_bytes())))
with (A / 'largest-eigenval-public-profile.json').open('xb') as stream:
    stream.write((PREVIOUS / 'largest-eigenval-public-profile.json').read_bytes())
for name in ('run.py', 'closed_export.py', 'test_run.py', 'test_closed_export.py'):
    with (A / name).open('x') as stream:
        stream.write(relocate((PREVIOUS / name).read_text()).replace('native-grok-stream-progress@4', 'native-grok-stream-progress@5'))
write('scaffold-origin.json', dict(schema='tool-event-diagnostic-scaffold-origin@1',
    source_files=[binding(PREVIOUS / name) for name in ('run.py', 'closed_export.py', 'test_run.py',
        'test_closed_export.py', 'recipe-draft.json', 'environment.json', 'local-asset-readiness-draft01.json',
        'largest-eigenval-public-profile.json')],
    disabled=True, provider_calls=0, docker_calls=0, new_runtime_qualification_required=True,
    new_runtime_commit_required=True, new_archive_required=True,
    previous_result_or_private_state_reused=False))
print(json.dumps(dict(status='disabled_successor_draft_prepared', provider_calls=0, docker_calls=0)))
