"""Export only closed, already-inspected pilot metadata after service recovery."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
W = B.parent.parent
A = W / '.worktrees/ir-release-accelerate-20261002'
OUT = Path(sys.argv[1]).resolve()
RUNS = (
    ('headless-terminal', B),
    ('largest-eigenval', B.parent / 'terminal-suite-eigenval-pilot-20261004'),
    ('feal-differential-cryptanalysis', B.parent / 'terminal-suite-feal-pilot-20261004'),
)

def read(path):
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size < 16 * 1024 * 1024
    return json.loads(path.read_text())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')

assert not OUT.exists()
audit = read(B / 'build-audit.json')
rows = []
for task, root in RUNS:
    result = read(root / 'pilot-result.json')
    service = read(root / 'service-pause.json')
    assert result['task'] == task and result['trial_count'] == 1
    assert result['training_on_benchmark'] is False
    assert service['restored'] is True
    restored = [e for e in service['events'] if e['phase'] == 'restored']
    assert len(restored) == 1 and restored[0]['model_ready'] is True
    assert restored[0]['service']['ActiveState'] == 'active'
    assert restored[0]['service']['SubState'] == 'running'
    assert int(restored[0]['service']['MainPID']) > 0
    execute_exit = read(root / 'execute-exit.json')
    assert read(root / 'build-audit.json') == audit
    for name, expected in read(root / 'harness-pins.json').items():
        assert digest(root / name) == expected, name
    raw = root / (task + '-01') / 'receipt.json'
    # Byte verification only: do not parse the raw receipt a second time.
    assert digest(raw) == result['receipt_sha256']
    assert raw.stat().st_size == result['receipt_bytes']
    trial = result['trials'][0]
    rows.append(dict(task=task, trial=trial['trial'], reward=trial['reward'],
                     complete_single_trial_receipt=result['complete_single_trial_receipt'],
                     original_task_inputs_unchanged=result['original_task_inputs_unchanged'],
                     exact_trial_task_matches=trial['exact_trial_task_matches'],
                     execute_returncode=execute_exit['returncode'],
                     service_wrapper_returncode=service['returncode'],
                     task_completed=trial['supervisor']['task_completed'],
                     durations_seconds=trial['durations_seconds'], usage=trial['usage'],
                     doctor=trial['doctor'], service_restored=True))

OUT.mkdir(parents=True)
for name in ('build-audit.json', 'campaign-plan.json', 'suite-preflight.json',
             'inventory.json', 'broader-readiness.json',
             'pilot-followups.json',
             'preparation-test-observation.json', 'runner-test-observation.json'):
    value = read(B / name)
    shutil.copyfile(B / name, OUT / name)

for name in ('source-pins.json', 'build-command.json'):
    read(B / name)
    shutil.copyfile(B / name, OUT / name)

for task, root in RUNS:
    target = OUT / 'runs' / task
    target.mkdir(parents=True)
    for name in ('pilot-result.json', 'service-pause.json', 'pressure-exit.json',
                 'prepare-exit.json', 'execute-exit.json', 'harness-pins.json',
                 task + '-profile.json'):
        value = read(root / name)
        if name == 'pilot-result.json':
            value = dict(value, source_receipt_schema=value['schema'],
                         schema='terminal-suite-public-summary@1',
                         canonical_receipt=False)
            write(target / name, value)
        else:
            shutil.copyfile(root / name, target / name)
    samples = [json.loads(line) for line in (root / 'pressure.jsonl').read_text().splitlines()]
    pressure_exit = read(root / 'pressure-exit.json')
    assert samples and pressure_exit['samples'] == len(samples)
    assert pressure_exit['observational_only'] is True
    write(target / 'pressure-summary.json', dict(
        schema='pilot-pressure-summary@1', observational_only=True,
        samples=len(samples), sample_file_sha256=digest(root / 'pressure.jsonl'),
        first_sample_at=samples[0]['at'], last_sample_at=samples[-1]['at'],
        min_available_kib=min(s['memory']['MemAvailable_kib'] for s in samples),
        max_available_kib=max(s['memory']['MemAvailable_kib'] for s in samples),
        max_full_avg10=max(s['memory_psi_percent']['full']['avg10'] for s in samples),
        phases=sorted({s['phase'] for s in samples}),
    ))
    for name in ('stage.py', 'run_with_leanstral_paused.py', 'observe_pressure.py', 'source384-config.json'):
        shutil.copyfile(root / name, target / name)
    for name in ('stage-build-original.py', 'campaign-plan.json', 'stage-pre-diagnostics.py',
                 'harness-pins-pre-diagnostics.json', 'inspection-projection-change.json'):
        if (root / name).exists():
            shutil.copyfile(root / name, target / name)
    if (root / 'bounded_projection.py').exists():
        shutil.copyfile(root / 'bounded_projection.py', target / 'bounded_projection.py')
    if (root / 'archive-reuse.json').exists():
        write(target / 'archive-reuse.json', read(root / 'archive-reuse.json'))
    write(target / 'shared-provenance.json', dict(
        build_command='../../build-command.json', source_pins='../../source-pins.json',
        runtime_archive_omitted=True, original_host_paths_require_rebinding=True,
    ))

xml = B / 'root-controls04.xml'
tree = ET.parse(xml)
suites = list(tree.getroot().iter('testsuite'))
write(OUT / 'root-controls-summary.json', dict(
    schema='junit-bounded-summary@1', source_sha256=digest(xml),
    suites=[{k: s.attrib.get(k) for k in ('name', 'tests', 'failures', 'errors', 'skipped', 'time')} for s in suites],
    cases=[dict(classname=c.get('classname'), name=c.get('name'),
                skipped=c.find('skipped') is not None) for c in tree.getroot().iter('testcase')],
    overlapping_observer_test_sets_not_summed=True,
))
write(OUT / 'campaign-result.json', dict(
    schema='terminal-suite-bounded-pilot@1', dataset_revision='2fd12b88aafdd04a52c298e3940bcb189f9766d6',
    dataset_tasks=89, previously_run_tasks=1, newly_run_tasks=len(rows),
    remaining_unexecuted_tasks=88-len(rows), trials=rows,
    remaining_scope='Tasks outside the prior Bottle run and this current full-indexed pilot; older legacy runs are not counted here.',
    expansion_note='The preserved original campaign plan selected headless-terminal. During that pilot, largest-eigenval and FEAL were prepared as additional serial tasks, reusing frozen runtime assets and the original execution deadline; task state and indexes were fresh.',
    archive=audit, selected_arm='full', resource_profile='source384-5cpu-16gib-extended@1',
    workflow='Direct planning, indexed context, generic Doctor analysis, router residual work.',
    ir_scope='Frozen SecurityIR Source384 checkpoint inference; no claim of all four IR checkpoint inference.',
    provider='ipfs_accelerate_py.llm_router -> codex_cli', model='gpt-5.6-sol',
    reasoning_effort='high', cli_version='0.158.0',
    serial=True, attempts_per_task=1, training_on_benchmark=False,
    matched_baselines=False, benchmark_advantage_claimed=False,
    token_scope='Observed cumulative router sessions; cached tokens included in input.',
    token_or_dollar_ceiling_enforced=False, dollar_cost=None,
    first_trial_successor_status='not_exported_by_original_bounded_inspector',
    broad_readiness_is_static_metadata_only=True,
))
lines = [
    '# Broader Terminal-Bench full supervisor pilot', '',
    'Three additional original Terminal-Bench tasks were attempted in fresh Docker containers '
    'using the full indexed supervisor. These selected tasks exercise the broader task-profile '
    'adapter; they are not a representative or complete suite score.', '',
    '| Task | Official reward | Supervisor completed | Agent seconds | Agent setup seconds | Observed tokens |',
    '| --- | ---: | :---: | ---: | ---: | ---: |',
]
for row in rows:
    reward = row['reward'].get('reward') if isinstance(row['reward'], dict) else None
    durations = row['durations_seconds'] or {}
    tokens = row['usage'].get('total_tokens')
    lines.append(f"| `{row['task']}` | {reward if reward is not None else 'unknown'} | "
                 f"{row['task_completed']} | {durations.get('agent_execution', 0):.2f} | "
                 f"{durations.get('agent_setup', 0):.2f} | {tokens if tokens is not None else 'unknown'} |")
lines.extend([
    '',
    'Tokens are observed cumulative router-session totals, including cached input. '
    'Cached tokens are already included in input tokens and must not be added again. '
    'Planning and implementation sessions both count. Session invocations are not a count '
    'of underlying model API requests. Dollar costs are unknown; no hard token or dollar ceiling was enforced.',
    '',
    '`largest-eigenval` reached native execution but timed out within the selected work budget. '
    'Its zero reward is retained. The bounded summary does not identify the exact internal '
    'step responsible, so it does not establish whether the candidate algorithm itself was correct.',
    '',
    'Each run uses direct goal/task planning, signed admission, indexed repository context, '
    'semantic capsules, generic symbolic-doctor analysis, and residual coding through '
    '`ipfs_accelerate_py.llm_router` with `codex_cli`, `gpt-5.6-sol`, high reasoning, and CLI `0.158.0`. '
    'The frozen SecurityIR Source384 checkpoint is consumed at inference time. '
    'The generic doctor abstains on unsupported repairs. These runs do not demonstrate '
    'symbolic-only synthesis, all four IR checkpoints, or formally proven interpretation of task prose.',
    '',
    'The profile is `source384-5cpu-16gib-extended@1`: five CPUs, 16 GiB, '
    '840 seconds of supervisor work and 60 seconds reserved for cleanup. Coding provider calls '
    'are capped at 300 seconds by the worker contract; the outer agent timeout is 960 seconds. '
    'There is one worker and one attempt per task. Tasks are serial. Native task budgets differ; '
    'FEAL normally has 1,800 seconds, so this is explicitly a capped-budget evaluation. '
    'The one-hour execution pilot excludes initial preparation and can retain necessary service-restoration time.',
    '',
    'The full archive audit binds ' + str(audit['verified_members']) + ' members and ' + str(audit['repository_pins']) +
    ' repository files. Actual Docker sources are accelerate `' + audit['source_revisions']['source'] +
    '`, datasets `' + audit['source_revisions']['datasets'] + '`, and kit `' + audit['source_revisions']['kit'] +
    '`. The archive SHA-256 is `' + audit['archive_sha256'] + '`. Runtime/model assets are reused across '
    'the three tasks; task containers, repository indexes, task state, and model responses are fresh. '
    'No training is performed on benchmark inputs.',
    '',
    'Public instruction/profile bindings and original task-file hashes remain in the evidence. '
    'Public summaries have a distinct schema and are not canonical benchmark receipts. '
    'Each original receipt was parsed once by its bounded inspector; export verifies the same '
    'bytes by SHA-256 without parsing it again. Credentials, model bodies, hidden verifier bodies '
    'and raw runtime logs are omitted. Producer scripts retain historical host paths; this package '
    'is evidence, not a self-contained runtime archive.',
    '',
    'The first inspector exported refresh duration but not successor status; successful successor '
    'refresh is therefore not claimed for that task. Later inspectors retain bounded refresh metadata. '
    'FEAL refreshed context in 11.10 seconds with fresh native Source384 inference and current advice. '
    'That refresh preserves four indexed paths and records one declared new output outside the index scope; '
    'it does not establish that the new attack implementation was formalized. The advice has no proof, '
    'planning, dispatch, execution or completion authority and requires an independent manifest for admission. '
    'The FEAL inspector also records bounded native task/daemon status; its change and prior producer '
    'hash are retained. Leanstral is restored to active/running with HTTP readiness after every run. '
    'Pressure summaries are observational, with explicit sample coverage.',
    '',
    'All 88 remaining task configurations pass static Harbor configuration parsing. '
    'That does not qualify their runtime adapters. After this pilot, 85 tasks still lack trials '
    'under this current full indexed methodology, apart from any older legacy experiments. '
    'The public readiness inventory records profile, source-format, empty-source, service/VM, '
    'build/training, Git-state, external-data and filesystem-scope gaps. Their rewards remain unknown. '
    'The original remaining tasks total 41.29 hours of native agent budgets, before setup and verification.',
    '',
    'Local component checks include a retained JUnit summary of 132 passes and one existing '
    'offline UV control skip, plus separately authored observations of 148 runner checks and '
    '71 preparation/context checks. These sets overlap and are not summed. The observer records '
    'explicitly distinguish missing raw test artifacts and source-pin coverage. '
    'Later publication reconciliation does not change the source revision of these Docker trials.',
    '',
    'No matched native-Codex or no-index baseline was run in this pilot, and no efficiency '
    'advantage is claimed. Follow-ups include broader task adapters and a bound code/metadata '
    'partition for the doctor: its current AST-only inventory gate rejects instruction Markdown '
    'and profile JSON as well as unsupported code. The existing 32-task backlog is not closed by this pilot.',
])
(OUT / 'README.md').write_text('\n'.join(lines) + '\n')
with (OUT / 'scores.csv').open('x', newline='') as f:
    fields = ('task', 'official_reward', 'supervisor_completed', 'agent_seconds',
              'agent_setup_seconds', 'input_tokens', 'cached_input_tokens', 'output_tokens', 'total_tokens')
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(task=row['task'], official_reward=(row['reward'] or {}).get('reward'),
            supervisor_completed=row['task_completed'],
            agent_seconds=(row['durations_seconds'] or {}).get('agent_execution'),
            agent_setup_seconds=(row['durations_seconds'] or {}).get('agent_setup'),
            **{k: row['usage'].get(k) for k in fields[5:]}))
print(json.dumps({'output': str(OUT), 'runs': len(rows)}))
