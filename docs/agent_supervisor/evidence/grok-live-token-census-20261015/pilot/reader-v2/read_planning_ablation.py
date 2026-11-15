"""Add declared-planning observations to the unchanged qualified pilot reader.

Only retained configuration/result reports are read. These reports cannot grant
execution or establish independent admission, semantics, billing or savings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
MODE = 'declared-contract@1'
SCHEMA = 'supervisor-declared-contract-planning@1'
FLAGS = ('instruction_interpreted', 'source_semantics_verified', 'semantic_alignment_verified',
         'proof_authority', 'execution_authority', 'completion_authority')
IDS = ('manifest_cid', 'graph_cid', 'request_cid', 'scan_cid', 'source_tree_id', 'plan_id')
RECEIPT_FIELDS = {'schema', 'planning_mode', *IDS, 'goals', 'tasks', 'pending_requirements',
    'formal_plan_consistent', 'provider_calls', 'model_calls', 'source_obligations_discharged',
    *FLAGS, 'interpretation_scope'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def baseline_reader():
    spec = importlib.util.spec_from_file_location('_qualified_reader_loader', BASE / 'qualified_reader_loader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_reader(BASE)


def planning_control(config):
    if type(config) is not dict or type(config.get('agents')) is not list or len(config['agents']) != 1:
        return None
    agent = config['agents'][0]
    if type(agent) is not dict or type(agent.get('kwargs')) is not dict:
        return None
    value = agent['kwargs'].get('planning_mode', 'model')
    return value if type(value) is str and value in ('model', MODE) else None


def declared_receipt(value):
    if type(value) is not dict or set(value) != RECEIPT_FIELDS:
        raise ValueError('invalid_declared_receipt')
    if (value['schema'] != SCHEMA or value['planning_mode'] != MODE
            or value['interpretation_scope'] != 'exact_signed_task_structure_only'
            or value['formal_plan_consistent'] is not True
            or any(value[key] is not False for key in FLAGS)
            or any(type(value[key]) is not int or value[key] != expected for key, expected in
                (('goals', 2), ('tasks', 1), ('provider_calls', 0), ('model_calls', 0), ('source_obligations_discharged', 0)))
            or type(value['pending_requirements']) is not int or not 1 <= value['pending_requirements'] <= 65535
            or any(type(value[key]) is not str or not 1 <= len(value[key]) <= 256
                or not value[key].isascii() or any(ord(c) < 33 or ord(c) > 126 for c in value[key]) for key in IDS)):
        raise ValueError('invalid_declared_receipt')
    return value


def observe_planning(report, mode, reader):
    result = dict(status='unavailable', declared_route_supported_by_retained_report=False,
        independent_admission_replay_performed=False, observation_only=True,
        proof_authority=False, execution_authority=False, completion_authority=False,
        planning_tokens_saved=None, total_token_savings_established=False)
    if mode is None:
        return dict(result, reason='invalid_configuration_control')
    if type(report) is not dict:
        return dict(result, reason='retained_report_missing')
    plan = report.get('planning')
    if mode == 'model':
        conflict = report.get('planning_mode') not in (None, 'model') or (
            type(plan) is dict and (plan.get('planning_mode') not in (None, 'model')
                or 'declared_contract_planning' in plan))
        return dict(result, status='invalid' if conflict else 'not_selected',
            reason='planning_control_conflict' if conflict else 'model_planning_selected')
    if type(plan) is not dict:
        return dict(result, reason='declared_plan_missing')
    if plan.get('qualified') is False:
        return dict(result, status='unqualified', qualified=False, reason='declared_plan_not_qualified')
    try:
        receipt = declared_receipt(plan.get('declared_contract_planning'))
        if (report.get('planning_mode') != MODE or plan.get('planning_mode') != MODE
                or plan.get('planning_strategy') != 'declared_contract' or plan.get('qualified') is not True
                or any(type(plan.get(key)) is not int or plan[key] != expected
                    for key, expected in (('goals', 2), ('tasks', 1), ('provider_calls', 0)))
                or any(plan.get(key) is not False for key in FLAGS)
                or any(key not in plan or plan[key] is not None
                    for key in ('provider_receipt', 'execution_receipt', 'model_request_sha256'))
                or plan.get('provider_observation') != {} or type(plan.get('provider_observation')) is not dict
                or type(plan.get('task_cids')) is not list or len(plan['task_cids']) != 1
                or type(plan['task_cids'][0]) is not str or not 1 <= len(plan['task_cids'][0]) <= 256
                or type(report.get('provider_invocations')) is not list):
            raise ValueError('inconsistent_declared_plan')
        invocations, _ = reader.distinct_receipts(report)
        if any(type(row.get('phase')) is not str or row['phase'] not in ('coding', 'review')
               for row in invocations):
            raise ValueError('contradictory_planning_invocation')
        raw = reader.canonical(receipt)
        return dict(result, status='validated_retained_declared_plan',
            declared_route_supported_by_retained_report=True, qualified=True, goals=2, tasks=1,
            provider_calls_reported=0, model_calls_reported=0, planning_native_receipts=0,
            receipt_schema=SCHEMA, receipt_sha256=digest(raw), receipt_bytes=len(raw),
            pending_requirements=receipt['pending_requirements'], formal_plan_consistent_reported=True,
            instruction_interpreted=False, source_semantics_verified=False, semantic_alignment_verified=False,
            source_obligations_discharged=0, interpretation_scope='exact_signed_task_structure_only')
    except (ValueError, TypeError, KeyError):
        return dict(result, status='invalid', reason='inconsistent_or_invalid_declared_evidence')


def planning_run(directory, baseline, reader):
    root = Path(directory)
    pins = {row['path']: row for row in baseline['artifact_bindings']}
    def same_read(path):
        bindings = []
        value = reader.read(path, bindings)
        if bindings and any(pins.get(row['path']) != row for row in bindings):
            raise ValueError('retained_artifact_changed_during_additive_read')
        return value
    mode = planning_control(same_read(root / 'config.json'))
    trials = []
    for row in baseline['trials']:
        name = row['trial']
        if Path(name).name != name or not name.startswith('largest-eigenval__'):
            raise ValueError('invalid_retained_trial_name')
        path = root / 'jobs/supervisor-full-largest-eigenval' / name / 'agent/supervisor-result.json'
        trials.append(dict(trial=name, planning=observe_planning(same_read(path), mode, reader)))
    return dict(planning_mode=mode, trials=trials)


def read_comparison(recipe_path, *, prior_directory=None):
    reader = baseline_reader()
    bindings = []
    recipe = reader.read(recipe_path, bindings)
    if recipe is None or recipe.get('schema') != 'doctor-dispatch-live-pilot@1':
        raise ValueError('known_pilot_recipe_required')
    pilot = reader.read_run(recipe['output'])
    prior = reader.read_run(prior_directory) if prior_directory is not None else None
    pilot_planning = planning_run(recipe['output'], pilot, reader)
    prior_planning = planning_run(prior_directory, prior, reader) if prior is not None else None
    changed = None if prior is None or pilot_planning['planning_mode'] == prior_planning['planning_mode'] else {
        'planning_mode': dict(pilot=pilot_planning['planning_mode'], prior=prior_planning['planning_mode'])}
    return dict(schema='planning-ablation-additive-readout@1',
        baseline=dict(pilot=pilot, prior=prior), planning=dict(pilot=pilot_planning, prior=prior_planning),
        differing_planning_control=changed, recipe_bindings=bindings,
        baseline_reader_sha256=digest((BASE / 'read_pilot.py').read_bytes()),
        baseline_reporting_freeze_sha256=digest((BASE / 'reporting-freeze.json').read_bytes()),
        additive_reader_sha256=digest(Path(__file__).read_bytes()),
        provider_calls_by_reader=0, causal_savings_claimed=False, total_token_savings_established=False,
        independent_owner_replay_performed=False, raw_prompts_or_provider_text_exported=False,
        limitations=['Declared planning copies a signed task structure; it does not infer intent or prove source semantics.',
            'Retained reports support a declared-route observation, not independently replayed admission or a billing total.',
            'Baseline usage, missingness, official reward and audit projections are unchanged.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args()
    print(json.dumps(read_comparison(args.recipe, prior_directory=args.prior), indent=2, sort_keys=True, allow_nan=False))
