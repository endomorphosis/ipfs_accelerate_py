"""Explicit preparation steps for a new capsule-budget pair; never generates text.

Run each operation separately after reviewing its output. The 'freeze' operation
only writes a lock; run_locked_trial.py is the separate generation entrypoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from campaign_common import (A, Q, LABELS, BUDGETS, SCRIPT_FILES, benchmark_source_pins,
    binding, check_binding, differences, disk_admission_snapshot, need, now, read,
    recipe, reporting_freeze_binding, safe_environment, sha, write_new, write_numbered)


def run_logged(argv, stem, *, timeout):
    path = A / 'live-preparation'
    need(not (path / (stem + '-result.json')).exists(), 'fresh operation output required')
    result = dict(schema='capsule-budget-preparation-operation@1', argv=argv, cwd=str(Q),
        started_at_utc=now(), environment_manifest=binding(A / 'live-environment.json'),
        provider_generation_calls=0)
    write_new(path / (stem + '-command.json'), result)
    start = time.monotonic()
    with (path / (stem + '.stdout')).open('xb') as out, (path / (stem + '.stderr')).open('xb') as err:
        try:
            completed = subprocess.run(argv, cwd=Q, env=safe_environment(), stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, timeout=timeout)
            result.update(returncode=completed.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            result.update(returncode=124, timed_out=True)
    result.update(elapsed_seconds=time.monotonic()-start,
        stdout=binding(path / (stem + '.stdout')), stderr=binding(path / (stem + '.stderr')))
    write_new(path / (stem + '-result.json'), result)
    need(result['returncode'] == 0, 'preparation operation failed: ' + stem)
    return result


def bundle():
    selected = recipe(); declared = read(A / 'bundle-command.json')
    argv = declared['argv']
    need(argv[:5] == [selected['python'], '-B', '-m',
        'benchmarks.agent_supervisor.container_coding.terminal_deployment', 'bundle'], 'bundle-only command required')
    need(argv[argv.index('--output')+1] == selected['runtime_directory'], 'isolated runtime output required')
    need(not Path(selected['runtime_directory']).exists(), 'fresh archive directory required')
    before = benchmark_source_pins()
    write_new(A / 'live-preparation/bundle-source-before.json', before)
    write_numbered(A / 'admission-snapshots', 'before-bundle', disk_admission_snapshot())
    run_logged(argv, 'bundle', timeout=1800)
    after = benchmark_source_pins()
    write_new(A / 'live-preparation/bundle-source-after.json', after)
    need(before == after, 'source changed during bundle; rebuild into a fresh campaign')


def verify():
    selected = recipe()
    run_logged([selected['python'], '-B', str(A / 'verify_runtime.py')], 'verify', timeout=1200)


def prepare():
    selected = recipe(); verified = read(A / 'live-preparation/verification-result.json')
    need(verified['all_archive_members_match_manifest'] is True
        and verified['current_source_capabilities_match'] is True, 'qualified new archive required')
    check_binding(verified['archive']); check_binding(verified['manifest'])
    for label in LABELS:
        row = selected['arms'][label]
        need(not Path(row['output']).exists(), 'fresh trial output required')
        argv = row['prepare_argv']
        need(argv[4] == 'prepare' and 'execute' not in argv, 'prepare-only operation required')
        run_logged(argv, label + '-prepare', timeout=900)


def freeze():
    selected = recipe()
    need(not (A / 'pair-lock.json').exists(), 'new pair lock required')
    reporting_inputs = reporting_freeze_binding(A / 'reporting-freeze.json')
    need(subprocess.run(['git', 'status', '--porcelain'], cwd=Q, capture_output=True, text=True,
        check=True).stdout == '', 'finish source changes before freezing')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Q, capture_output=True, text=True, check=True).stdout.strip()
    verified = read(A / 'live-preparation/verification-result.json')
    check_binding(verified['archive']); check_binding(verified['manifest'])
    readiness_path = A / 'provider-readiness/grok-metadata-precheck.json'
    ready = read(readiness_path)
    need(ready.get('provider_generation_calls') == 0 and ready.get('pinned_provider_profile') == selected['provider_profile']
        and ready.get('host_auth_unchanged') is True and ready.get('container_and_tmpfs_removed') is True
        and ready.get('host_temporary_auth_copy_removed') is True, 'current isolated metadata readiness required')
    observed = ready.get('observation', {})
    need(observed.get('status') == 'ready' and observed.get('selected_model_observed') is True
        and observed.get('text_generation_calls') == 0 and observed.get('worker_euid') == 1001,
        'isolated Grok model catalog readiness required')
    # Freeze imports must resolve the same Q/D/K sources used by child commands,
    # even when this script was invoked outside the checkout or without PYTHONPATH.
    environment = safe_environment()
    os.environ.clear(); os.environ.update(environment)
    sys.path[:0] = environment['PYTHONPATH'].split(os.pathsep)
    from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
    from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
    from benchmarks.agent_supervisor.container_coding.terminal_capsule_selection_policy import require_capsule_budget_archive
    need(Path(benchmark.__file__).resolve() == Q / 'benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py',
        'reporting import escaped the frozen checkout')
    manifest = read(verified['manifest']['path'])
    require_capsule_budget_archive(manifest, 'one-capsule@1')
    attempts = []; preps = []; configs = []
    for label in LABELS:
        row = selected['arms'][label]; out = Path(row['output'])
        need(not (out / 'invocation.json').exists() and not (out / 'launch-start.json').exists(), 'freeze must precede every execution')
        prep = read(out / 'preparation.json'); config = read(out / 'config.json')
        kwargs = config['agents'][0]['kwargs']
        need(prep['prepared'] is True and prep['provider_calls'] == 0 and prep['config_sha256'] == sha(out / 'config.json'),
             'qualified untouched dry-run required')
        need(prep['task'] == 'largest-eigenval' and prep['arm'] == kwargs['arm'] == 'full', 'task/full arm differs')
        need(prep['provider_profile'] == kwargs['provider_profile'] == selected['provider_profile'], 'Grok route differs')
        need(prep.get('capsule_budget', 'legacy') == kwargs.get('capsule_budget', 'legacy') == BUDGETS[label], 'capsule budget differs')
        defaults = dict(planning_mode='model', semantic_transport_schema='supervisor-semantic-router-input@1',
            coding_reply_mode='legacy', semantic_metadata_view='legacy', capsule_selection='lexical-query-symbol-name@1')
        for key in ('planning_mode','semantic_transport_schema','coding_reply_mode','semantic_metadata_view',
                    'capsule_selection','source_view_profile','source_edit_mode','numerical_formalization_profile','resource_profile'):
            need(kwargs.get(key, defaults.get(key)) == selected['common'][key], 'common control differs: ' + key)
        need(prep['archive_sha256'] == verified['archive']['sha256'] and prep['manifest_sha256'] == verified['manifest']['sha256'],
             'prepared archive differs from verification')
        need(prep['host_source_sha256'] == benchmark_source_pins(), 'prepared sources changed')
        task = Path(prep['dataset']) / prep['task']
        observation = controls.observe_controls(prep, config, current_task_hashes=benchmark._task_hashes(task))
        need(observation['status'] == 'observed' and observation['configuration_unchanged'] is True,
             'declared configuration or permitted-input hashes changed')
        attempts.append(dict(label=label, arm='full', capsule_budget=BUDGETS[label], directory=str(out),
            preparation=binding(out / 'preparation.json'), configuration=binding(out / 'config.json')))
        preps.append(prep); configs.append(config)
    difference_paths = differences(configs[0], configs[1])
    need(set(difference_paths) == set(selected['allowed_config_difference_paths']), 'unexpected full configuration difference')
    common_keys = ('task','arm','task_input_sha256','archive_sha256','manifest_sha256','host_source_sha256',
                   'provider_profile','model','reasoning_effort','cli_version','resource_profile','source384',
                   'intent_action_384','intent_action_384_readiness')
    need(all(preps[0].get(key) == preps[1].get(key) for key in common_keys), 'common preparation differs')
    comparison = controls.compare_controls(*[dict(status='observed',configuration_unchanged=True,
        declared=p['comparison_controls']) for p in preps])
    need(comparison['matches'] is False and comparison['differences'] == ['experimental_controls_sha256.capsule_budget'],
         'declared comparison differs beyond capsule treatment')
    rows = {row['path']: row for row in manifest['files']}
    host_only_tests = 0; archive_matches = 0
    for path, digest in preps[0]['host_source_sha256'].items():
        relative = Path(path).relative_to(Q); archived = rows.get('source/' + str(relative))
        if archived is None:
            need(relative.name.startswith('test_'), 'production source absent from archive')
            host_only_tests += 1
        else:
            need(archived['sha256'] == digest, 'host and archived source differ')
            archive_matches += 1
    audit = dict(schema='capsule-budget-prelaunch-control-audit@1', complete=True,
        actual_config_diff_paths=difference_paths, allowed_config_diff_paths=selected['allowed_config_difference_paths'],
        generic_control_comparison=comparison, expected_experimental_difference='capsule_budget',
        host_source_files=len(preps[0]['host_source_sha256']), archive_matched_host_files=archive_matches,
        host_only_test_files=host_only_tests, task_hashes_identical=True, source384_bindings_identical=True,
        provider_generation_calls=0, no_token_savings_claimed=True)
    audit_path = A / 'live-preparation/control-audit.json'; write_new(audit_path, audit)
    launch_inputs = {name: binding(A / name) for name in (*SCRIPT_FILES, 'recipe.json', 'live-environment.json', 'bundle-command.json')}
    lock = dict(schema='capsule-budget-live-pair-lock@1', frozen_before_launch=True, created_at_utc=now(),
        source_head=head, common={**selected['common'], 'task_input_sha256':preps[0]['task_input_sha256']},
        allowed_config_diff_paths=selected['allowed_config_difference_paths'], attempts=attempts,
        launch_inputs=launch_inputs, runtime=dict(archive=verified['archive'],manifest=verified['manifest']),
        verification=binding(A / 'live-preparation/verification-result.json'), readiness=binding(readiness_path),
        control_audit=binding(audit_path), host_source_sha256=preps[0]['host_source_sha256'],
        reporting_inputs=reporting_inputs,
        execution_authorization='Invoke each label explicitly; no automatic execution, provider retry or fallback.',
        collection_helper='Bound capsule-budget reader and exact imported helper closure; historical source/comment campaign stays unchanged.')
    write_new(A / 'pair-lock.json', lock)
    print(json.dumps(dict(pair_lock=binding(A / 'pair-lock.json'), provider_calls=0), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('snapshot','bundle','verify','prepare','freeze'))
    args = parser.parse_args()
    if args.operation == 'snapshot':
        observed = disk_admission_snapshot()
        path = write_numbered(A / 'admission-snapshots', 'host-disk', observed)
        print(json.dumps(dict(observation=observed, artifact=binding(path)), sort_keys=True))
    else:
        globals()[args.operation]()


if __name__ == '__main__':
    main()
