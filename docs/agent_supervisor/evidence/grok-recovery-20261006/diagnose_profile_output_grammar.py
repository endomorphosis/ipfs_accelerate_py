"""Compare public profile declaration shape with the pinned canonical grammar.

No benchmark source, hidden verifier, provider body or credential is read.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')
EXPECTED = '987e0ad1f1c22ef70531449a6cbf49c786afe152'


def main():
    head = subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip()
    producer = 'ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py'
    original_producer = subprocess.check_output(['git', '-C', str(P), 'show', EXPECTED + ':' + producer])
    if (P / producer).read_bytes() != original_producer:
        raise SystemExit('public profile producer differs from trial revision')
    grammar_path = 'ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py'
    grammar_bytes = subprocess.check_output(['git', '-C', str(P), 'show', EXPECTED + ':' + grammar_path])
    grammar_ast = ast.parse(grammar_bytes)
    names = {'_SAFE_FALLBACK_BEHAVIORS', '_SAFE_OUTPUT_EFFECTS', '_SAFE_MEDIA_TYPES',
             '_GOAL_FIELDS', '_TASK_FIELDS', '_OUTPUT_FIELDS', '_VALIDATION_FIELDS',
             'PROMPT_GOAL_PROPOSAL_SCHEMA', 'PROMPT_GOAL_PLANNER_VERSION'}
    nodes = [node for node in grammar_ast.body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in names]
    if {node.targets[0].id for node in nodes} != names:
        raise SystemExit('expected historical grammar constants required')
    function = next(node for node in grammar_ast.body if isinstance(node, ast.FunctionDef) and node.name == '_proposal_schema')
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    historical = {}
    exec(compile(ast.Module(body=nodes + [function], type_ignores=[]), grammar_path, 'exec'), historical)
    profile_path = A / 'grok-tune-mjcf-03-profile.json'
    report_path = A / 'grok-tune-mjcf-03/jobs/supervisor-full-tune-mjcf/tune-mjcf__2MevWy5/agent/supervisor-result.json'
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from benchmarks.agent_supervisor.container_coding.terminal_task_profile import task_profile_spec
        from jsonschema import Draft202012Validator
        profile_bytes = profile_path.read_bytes()
        profile = json.loads(profile_bytes)
        spec = task_profile_spec(profile, policy_cid='synthetic-policy')
        schema = historical['_proposal_schema'](SimpleNamespace(budget=SimpleNamespace(
            max_goals=2, max_tasks=1, max_graph_depth=4,
            max_serialized_bytes=32768, max_provider_tokens=32768)))
        for row in spec['validations']:
            row.pop('policy_cid', None)
        for row in spec['acceptance']:
            row['evidence_cids'] = ['synthetic-evidence']
        errors = []
        for field, definition in (('outputs', 'output'), ('validations', 'validation'), ('acceptance', 'acceptance')):
            for row in spec[field]:
                errors.extend((field, error) for error in Draft202012Validator(schema['definitions'][definition]).iter_errors(row))
    if not (len(errors) == 1 and errors[0][0] == 'outputs'
            and list(errors[0][1].path) == ['media_type']
            and errors[0][1].validator == 'enum'
            and errors[0][1].instance == 'application/xml'):
        raise SystemExit('expected closed profile grammar mismatch was not reproduced')
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if not (report.get('error_phase') == 'prepare'
            and report.get('error') == {'type': 'ValueError', 'message': 'terminal task declaration row differs from canonical grammar'}
            and report.get('provider_invocations') == []):
        raise SystemExit('retained source-owned failure differs')
    value = dict(schema='terminal-public-profile-grammar-diagnosis@1',
        trial_name='grok-tune-mjcf-03', source_revision=EXPECTED,
        historical_grammar_loaded_from_exact_git_revision=True,
        profile_producer_matches_trial_source=True, observation_host_revision=head,
        grammar_source_sha256=hashlib.sha256(grammar_bytes).hexdigest(),
        official_reward=0, task_completed=False,
        observed_failure_phase='prepare', error_type='ValueError',
        reason_code='output_media_type_outside_canonical_enum',
        declaration_field='outputs', declaration_member='media_type',
        rejected_public_media_type='application/xml',
        canonical_allowed_media_type_count=len(errors[0][1].validator_value),
        mismatching_rows=1, row_counts={key: len(spec[key]) for key in ('outputs', 'validations', 'acceptance')},
        reproducer_uses_synthetic_policy_and_evidence=True,
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        retained_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        source_function='terminal_planner_contract.validate_task_contract', source_line=74,
        provider_invocation_receipts=0, native_usage='unknown',
        index_hydration_observed=False, source384_inference_observed=False,
        planning_admission_observed=False, native_start_observed=False,
        worker_cleanup_returncode=report.get('worker_cleanup_returncode'),
        fixture_scope_gap='authored_probe_used_python_and_text_outputs_not_xml',
        host_prepare_scope='benchmark_bundle_and_config_not_container_task_binder',
        hidden_verifier_or_source_bodies_read=False,
        raw_model_or_credential_data_exported=False,
        provider_calls=0, source_mutations=0)
    target = A / 'preparation-failure-observation-03.json'
    with target.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
