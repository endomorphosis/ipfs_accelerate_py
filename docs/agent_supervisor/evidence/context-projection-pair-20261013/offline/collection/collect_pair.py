"""Adapt retained independent owner observations to the strict trace comparator.

This collector does no source/model execution. New records reference the original
immutable captures; owner replay receipts remain caller-supplied evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

P = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
sys.path.insert(0, str(P))
MODULE = P / 'benchmarks/agent_supervisor/container_coding/context_projection_trace_comparison.py'


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(wire(value) + b'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run, output = args.run.absolute(), args.output.absolute()
    assert run.is_dir() and not output.exists() and output.parent == run.parent
    qualification = load(run / 'qualification.json')
    assert all(qualification[key] is True for key in ('complete','container_removed','source_unchanged','runner_unchanged'))
    command = load(run / 'command.json')
    assert all(sha(Path(path).read_bytes()) == digest for path,digest in command['source_bindings'].items())
    assert all(sha(Path(path).read_bytes()) == digest for path,digest in command['runner_bindings'].items())
    native_prompt = (run / 'results/owner-input-prompt.txt').read_bytes()
    arms = {arm: load(run / 'results' / arm / 'worker-result.json') for arm in ('exact','projected')}
    owners = {arm: load(run / 'results' / arm / 'owner-qualification.json') for arm in arms}
    common = {'native_prompt':sha(native_prompt)}
    for path, digest in command['source_bindings'].items():
        if '/.benchmarks/' in path:
            common['public/' + Path(path).name] = digest
    assert owners['exact']['admitted_source_hashes'] == owners['projected']['admitted_source_hashes']
    common.update({'admitted/' + name: digest for name,digest in owners['exact']['admitted_source_hashes'].items()})
    operations = [
        ('bootstrap:eigen.py','source','eigen.py','compact','complete'),
        ('discovery:catalog','catalog',None,None,'complete'),
        ('baseline_read:eigen.py','source','eigen.py','compact','complete'),
        ('baseline_read:eval.py','source','eval.py','compact','complete'),
        ('pre_edit_expansion:eigen.py','source','eigen.py','original','complete'),
        ('stale_generation:eval.py','source','eval.py','compact','refused'),
        ('edited_refresh:eigen.py','source','eigen.py','compact','complete'),
        ('edited_read:eval.py','source','eval.py','compact','complete'),
        ('post_edit_expansion:eigen.py','source','eigen.py','original','complete'),
        ('restore_stale_generation:eigen.py','source','eigen.py','compact','refused'),
        ('restored_refresh:eigen.py','source','eigen.py','compact','complete')]
    source_hashes = owners['exact']['public_source_hashes']
    controls = dict(schema='context-projection-trace-controls@1', common_input_sha256=common,
        runtime_sha256={path:digest for path,digest in command['source_bindings'].items() if '/.worktrees/' in path},
        script_sha256=command['runner_bindings'][str(run.parent / 'worker_probe.py')], page_budget=8192,
        source_view_profile='identity@1', baseline_metadata_mode='legacy', treatment_metadata_mode='common-bindings@1',
        provider='codex_cli', coding_reply_mode='ordinary-completion@1', workspace=arms['exact']['workspace'],
        source_hashes=source_hashes, logical_operations=[dict(zip(('operation_id','kind','path','view','outcome'), row)) for row in operations],
        source_edit=dict(path='eigen.py',old='idx',new='i',occurrences=3,
            before_sha256=source_hashes['eigen.py'],after_sha256=owners['exact']['edited_source_hashes']['eigen.py']))
    replays = {}
    for arm, trace in arms.items():
        owner = owners[arm]
        assert all(owner[key] is True for key in ('complete','broker_stopped','descriptor_removed','source_restored',
            'workspace_git_clean','native_session_accounting_verified','actual_default_cli_newlines_included'))
        assert owner['remaining_worker_pids'] == []
        assert owner['owner_input_prompt_sha256'] == sha(native_prompt) and owner['owner_input_prompt_bytes'] == len(native_prompt)
        assert owner['public_source_hashes'] == source_hashes
        by_operation = {row['operation_id']: row for row in owner['source_read_replays'] + [owner['catalog_replay']]}
        assert all(row['complete'] is True and row['exact_owner_replay'] is True for row in by_operation.values())
        counts = load(run / 'results' / arm / 'session-observation.json')['counts']
        replays[arm] = dict(schema='source-view-trace-owner-replay@1', trace_sha256=sha(wire(trace)),
            controls_sha256=sha(wire(controls)), prompt_sha256=trace['prompt']['sha256'],
            operation_replays=[dict(operation_id=op['operation_id'], sha256=by_operation[op['operation_id']]['assembled_sha256'],
                bytes=by_operation[op['operation_id']]['assembled_bytes']) for op in controls['logical_operations'] if op['outcome']=='complete'],
            source_hashes=owner['public_source_hashes'], edited_source_hashes=owner['edited_source_hashes'], restored_source_hashes=owner['restored_source_hashes'],
            response_json_bytes=counts['response_bytes'], delivered_response_json_bytes=counts['delivered_response_bytes'],
            delivered_responses=counts['delivered_responses'], accounting_complete=True,
            source_replay_verified=True, source_edit_verified=True, cleanup_verified=True)
    import importlib
    module = importlib.import_module('benchmarks.agent_supervisor.container_coding.context_projection_trace_comparison')
    comparison = module.compare_traces(baseline=arms['exact'], treatment=arms['projected'], controls=controls,
        baseline_replay=replays['exact'], treatment_replay=replays['projected'])
    output.mkdir(mode=0o700)
    save(output / 'controls.json',controls)
    for arm,replay in replays.items():save(output / (arm + '-owner-replay.json'),replay)
    save(output / 'comparison.json',comparison)
    inputs = [run/'qualification.json',run/'command.json',run/'results/owner-input-prompt.txt']
    for arm in arms:
        inputs.extend(run/'results'/arm/name for name in ('worker-result.json','owner-qualification.json','session-observation.json','router-observation.json'))
    save(output / 'collection-provenance.json',dict(schema='source-view-trace-collection-provenance@1',
        run=str(run),collector_sha256=sha(Path(__file__).read_bytes()),comparator_sha256=sha(MODULE.read_bytes()),
        inputs_sha256={str(path):sha(path.read_bytes()) for path in inputs},
        shared_preflight=True,ordered_metadata_modes=['legacy','common-bindings@1'],source_view_profile='identity@1',cold_warm_speed_comparison=False,
        optional_catalog_discovery=True, input_paths_already_known=True,
        source_coding_task='Authored variable rename within permitted public inputs; not a Terminal-Bench solution',
        notes='Owner replay envelopes adapt independently qualified artifacts; hashes bind caller-supplied evidence, not authenticated authority. Exposure counts one prepared prompt plus every actual default CLI stdout/stderr once; no model billing, token, or reward claim.'))
    print(json.dumps(comparison,sort_keys=True))


if __name__ == '__main__':
    main()
