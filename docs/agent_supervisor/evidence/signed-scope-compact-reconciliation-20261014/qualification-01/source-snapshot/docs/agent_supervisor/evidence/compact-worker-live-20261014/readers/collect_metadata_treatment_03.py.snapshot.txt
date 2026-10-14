"""Separate metadata-policy treatment readout; original pair remains unchanged."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.representation_displacement import validate_representation_displacement

A = Path(__file__).resolve().parent

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, A / filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

r = module('pair_readout', 'collect_live_readout.py')
shape = module('native_shape', 'collect_native_shape.py')
MODE = 'common-bindings-instruction@1'
DIFFERENCES = {'/agents/0/kwargs/semantic_metadata_view', '/jobs_dir'}


def bound(binding):
    value, actual = r.read(binding['path'])
    r.need(actual['sha256'] == binding['sha256'], 'frozen metadata treatment input changed')
    return value, actual


def selected_prompt(call):
    selection = call.get('semantic_metadata_view')
    displacement = call.get('representation_displacement')
    if selection is None and displacement is None:
        return dict(status='unavailable', requested_mode=MODE, selected_mode=None)
    r.need(type(selection) is dict and type(displacement) is dict, 'incomplete projection observation')
    value = validate_representation_displacement(displacement,
        selected_input_sha256=call['model_prompt_sha256'], selected_input_bytes=call['model_prompt_bytes'])
    r.need(value['requested_mode'] == selection.get('requested_mode') == MODE, 'projection request differs')
    for key in ('selected_mode', 'fallback_reason', 'native_complete_bytes', 'candidate_complete_bytes',
                'selected_complete_bytes', 'native_complete_sha256', 'candidate_complete_sha256',
                'selected_complete_sha256'):
        r.need(value[key] == selection.get(key), 'selection/displacement observations disagree')
    return dict(status='observed', **{key:value[key] for key in (
        'requested_mode', 'selected_mode', 'fallback_reason', 'selection_basis', 'native_complete_bytes',
        'candidate_complete_bytes', 'selected_complete_bytes', 'native_complete_sha256',
        'candidate_complete_sha256', 'selected_complete_sha256', 'selected_complete_byte_reduction',
        'required_source_bytes_preserved')},
        fixed_codec_replayed_here=False, observation_scope='Closed recorded arithmetic and selected model-prompt identity; does not independently replay unavailable prompt bodies.')


def treatment(release_path, execution_lock_path=None):
    release, binding = r.read(release_path)
    r.need(release.get('schema') == 'compact-metadata-treatment-preparation-release@1'
        and release.get('frozen_before_execution') is True and release.get('original_pair_population_unchanged') is True
        and set(release['allowed_config_difference_paths']) == DIFFERENCES, 'treatment release scope differs')
    lock, lock_binding = bound(release['baseline_pair_lock'])
    original = r.collect(r.input_from_lock(lock, lock_binding))
    baseline = next(row for row in original['attempts'] if row['label'] == 'comments')
    r.need(baseline['directory'] == release['comparison_baseline_directory'], 'baseline not registered')
    config, cb = bound(release['files']['configuration']); prep, pb = bound(release['files']['preparation'])
    baseline_config, _ = r.read(baseline['configuration']['path'])
    baseline_prep, _ = r.read(baseline['preparation']['path'])
    changes = r.differences(baseline_config, config)
    r.need(set(changes) == DIFFERENCES, 'treatment changes more than metadata policy and output path')
    kwargs = config['agents'][0]['kwargs']
    r.need(kwargs.get('semantic_metadata_view') == MODE and prep.get('semantic_metadata_view') == MODE
        and kwargs.get('source_view_profile') == prep.get('source_view_profile') == 'python-comments@1'
        and kwargs.get('source_edit_mode') == prep.get('source_edit_mode') == 'mapped-span@2'
        and prep['config_sha256'] == cb['sha256'], 'treatment selection differs')
    r.need(release['host_source_sha256'] == lock['host_source_sha256'], 'runtime source closure differs')
    for key in ('task', 'task_input_sha256', 'archive_sha256', 'manifest_sha256', 'host_source_sha256',
                'provider_profile', 'model', 'reasoning_effort', 'cli_version', 'resource_profile'):
        r.need(prep.get(key) == baseline_prep.get(key), 'treatment common preparation differs')
    r.need(r.controls.build_controls(config, task_input_sha256=prep['task_input_sha256'], task=prep['task'],
        model=prep['model'], reasoning_effort=prep['reasoning_effort'], cli_version=prep['cli_version']) == prep['comparison_controls'], 'treatment controls differ')
    base = Path(release['treatment_directory'])
    r.need(base / 'preparation.json' == Path(pb['path']) and base / 'config.json' == Path(cb['path']), 'treatment directory differs')
    receipt, rb = r.read(base/'receipt.json', optional=True)
    launched, sb = r.read(base/'launch-start.json', optional=True)
    finished, fb = r.read(base/'launch-result.json', optional=True)
    execution = None
    if launched is not None or receipt is not None or finished is not None:
        r.need(execution_lock_path is not None and launched is not None, 'executed treatment lacks separate execution lock')
        execution, eb = r.read(execution_lock_path)
        r.need(execution.get('schema') == 'compact-metadata-live-execution-lock@1'
            and execution.get('frozen_before_launch') is True and execution.get('label') == 'comments-metadata', 'separate execution lock scope differs')
        files = execution['files']
        for key, expected in (('release',binding),('preparation',pb),('configuration',cb),
                              ('baseline_pair_lock',lock_binding),('baseline_launch_result',baseline['launch_result'])):
            r.need(files[key]['path'] == expected['path'] and files[key]['sha256'] == expected['sha256'], 'execution lock input differs')
        for key in ('environment','recipe','runtime_manifest'):
            r.need(files[key] == release['files'][key], 'execution release inputs differ')
            bound(files[key])
        r.need(files['archive'] == release['archive'] and r.source_binding(files['runner']['path']) == files['runner']['sha256'], 'execution archive or runner differs')
        r.need(launched.get('schema') == 'repeated-compact-live-launch@1'
            and launched.get('runner_sha256') == files['runner']['sha256']
            and launched.get('recipe_sha256') == files['recipe']['sha256']
            and launched.get('environment_manifest_sha256') == files['environment']['sha256'], 'native launch provenance differs')
        r.need(launched.get('metadata_release_sha256') == binding['sha256']
            and launched.get('execution_lock_sha256') == eb['sha256']
            and launched.get('baseline_pair_lock_sha256') == lock_binding['sha256']
            and launched.get('preparation_sha256') == pb['sha256']
            and launched.get('label') == 'comments-metadata', 'treatment launch bindings differ')
        execution = eb
    rows = []
    integrity = False
    if receipt is not None:
        r.need(receipt.get('schema') == r.comparison.SUPERVISOR and receipt.get('arm') == 'full', 'native supervisor receipt required')
        r.need(all(receipt.get(key) == prep.get(key) for key in ('task','model','provider_profile','reasoning_effort','cli_version')), 'native treatment route differs')
        controls = r.obj(receipt.get('comparison_controls'))
        flags = {'source_view_config_unchanged','source_edit_config_unchanged','semantic_metadata_config_unchanged'} | {key for key in receipt if key.endswith('_config_unchanged')}
        integrity = controls.get('status') == 'observed' and controls.get('declared') == prep['comparison_controls'] and controls.get('configuration_unchanged') is True and receipt.get('original_task_inputs_unchanged') is True and all(receipt.get(key) is True for key in flags)
        r.need(type(receipt.get('trials')) is list and len(receipt['trials']) <= 16, 'bounded treatment trials required')
        seen = {row['invocation_id'] for attempt in original['attempts'] for trial in attempt['trials'] for row in trial['usage']['invocations']}
        for raw in receipt['trials']:
            r.need(raw.get('exact_trial_task_matches') is True and raw.get('task') in (prep['task'],'terminal-bench/'+prep['task']), 'treatment task differs')
            report = r.obj(raw.get('supervisor')); measured = r.usage(report)
            provider = r.obj(report.get('provider_profile'))
            if report:
                r.need(provider.get('id') == prep['provider_profile'] and provider.get('model') == prep['model'] and provider.get('reasoning_effort') == prep['reasoning_effort'] and report.get('arm') == 'full', 'treatment supervisor route differs')
            for call in measured['invocations']:
                r.need(call['invocation_id'] not in seen and call['model'] == prep['model'] and call['provider'] == provider.get('provider'), 'treatment invocation duplicated or route differs')
                seen.add(call['invocation_id'])
            trial = r.comparison._trial(receipt, raw, source=rb)
            rows.append(dict(trial=r.tag(raw.get('trial')), official_reward=r.number(trial['official_reward']),
                outcome=r.tag(trial['outcome']), exception_type=r.tag(trial['exception_type']),
                supervisor_error_type=r.tag(trial['supervisor_error_type']), native_task_completed=trial['native_task_completed'],
                durations_seconds=trial['durations_seconds'], supervisor_phases_seconds=trial['supervisor_phases_seconds'],
                runtime_matches_preparation=trial.get('runtime_archive_sha256') == prep['archive_sha256'], usage=measured,
                source_use=r.source_usage(report, measured['invocations'], 'python-comments@1'),
                native_shape=shape.project(report,[row['invocation_id'] for row in measured['invocations']]),
                coding_input_selection=[selected_prompt(call) for call in report.get('provider_invocations',[]) if call.get('phase') == 'coding']))
    if finished is not None:
        r.need(finished.get('schema') == 'repeated-compact-live-launch-result@1'
            and finished.get('label') == 'comments-metadata'
            and finished.get('metadata_release_sha256') == binding['sha256']
            and finished.get('execution_lock_sha256') == execution['sha256']
            and finished.get('receipt_sha256') == rb.get('sha256'), 'launch result receipt differs')
        integrity = integrity and all(finished.get(key) is True for key in ('preparation_unchanged','configuration_unchanged','host_source_unchanged'))
    else:
        integrity = False
    return dict(schema='compact-metadata-treatment-readout@1', release=binding, execution_lock=execution,
        original_pair_lock=lock_binding, original_pair_population_unchanged=True,
        baseline_comments_attempt=baseline, preparation=pb, configuration=cb, configuration_difference_paths=changes,
        receipt=rb, launch_start=sb, launch_result=fb, launch_seconds=r.number(r.obj(finished).get('seconds')),
        launch_exit_code=shape.exit_code(r.obj(finished).get('exit_code')), trials=rows,
        status='awaiting_receipt' if receipt is None else 'observed', treatment_control_integrity=integrity,
        frozen_source_pin_count=len(lock['host_source_sha256']), frozen_source_pins_verified=True,
        reported_counter_scope='Per-invocation native final envelopes are retained even when complete usage is unknown; no comparison total is inferred.',
        limitations=['This sequential third treatment is outside the original pair and is not randomized.',
            'Reported prompt byte displacement does not establish complete token savings or task causality.',
            'Native planning and tool histories may differ; compare the retained call and task counts.',
            'Source helper observations do not establish ordinary source tool use, proof or final task correctness.'],
        provider_calls_by_reader=0, raw_model_transcripts_read=False, hidden_verifier_or_solution_contents_read=False,
        billing_total_verified=False, causal_advantage_claimed=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--execution-lock',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=treatment(args.release,args.execution_lock)
    result['implementation_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__).resolve(),Path(r.__file__).resolve(),Path(shape.__file__).resolve(),
        Path(__import__(validate_representation_displacement.__module__,fromlist=['']).__file__).resolve())}
    with args.output.open('x') as handle:json.dump(result,handle,sort_keys=True,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(path=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))
