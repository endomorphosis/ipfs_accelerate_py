"""Provider-free catalog-shape rehearsal through the real process adapter.

Authored minimal inputs and output are private fixture data. The subprocess is
a test executable, not Grok. No benchmark input or hidden verifier is consumed.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import traceback

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-source-head', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.expected_source_head) or not re.fullmatch('[0-9]{2}', args.attempt):
        parser.error('exact source revision and two-digit attempt required')
    recipe = module('preflight_recipe', A / 'grok-container/recipes/run_fresh_grok.py')
    before = recipe.snapshot(args.expected_source_head)
    private = A / ('private-public-tune-preflight-' + args.attempt)
    target = A / 'qualification' / ('public-tune-profile-preflight-' + args.attempt + '.json')
    if private.exists() or target.exists():
        raise SystemExit('new preflight attempt required')
    private.mkdir(mode=0o700)
    os.environ['IPFS_ACCELERATE_LLM_ALLOCATION_DB'] = str(private / 'allocation.duckdb')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    started = time.monotonic()
    result = dict(schema='terminal-public-tune-profile-preflight@1',
        source_before=before, provider_calls=0, live_provider_invocations=0,
        task_inputs_used=False, hidden_verifier_or_solution_bodies_read=False,
        raw_model_or_credentials_exported=False, authored_fixture_private=True,
        task_qualification_claim=False, completion_authority=False, qualified=False)
    log = private / 'preflight.log'
    with log.open('x') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        os.chmod(log, 0o600)
        try:
            from pytest import MonkeyPatch
            from jsonschema import Draft202012Validator
            helper_path = P / 'test/api/test_terminal_planner_media_contract.py'
            helper = module('public_tune_fixture', helper_path)
            fixture = private / 'fixture'; fixture.mkdir(mode=0o700)
            prepared = helper._prepare_catalog_shape(fixture, 'tune-mjcf')
            signed_before = json.dumps(prepared['spec'], sort_keys=True)
            prompt = helper.request_text(prepared)
            canonical = helper.planning_response_format(prompt)['json_schema']['schema']
            hint = helper.contract.contract_from_prompt(prompt)
            wire, _, projection = helper.native_schema_projection(canonical, task_contract=hint)
            Draft202012Validator.check_schema(wire)
            value = helper._proposal(prepared)
            Draft202012Validator(canonical).validate(value)
            Draft202012Validator(wire).validate(value)
            adapter = private / 'adapter'; adapter.mkdir(mode=0o700)
            with MonkeyPatch.context() as patch:
                text, receipt = helper._invoke_authored_cli(prepared, adapter, patch, value)
            if json.loads(text) != value or json.dumps(prepared['spec'], sort_keys=True) != signed_before:
                raise ValueError('authored response or signed declaration changed')
            if receipt['provider_invocation_policy']['structured_output']['native_schema_projection'] != projection:
                raise ValueError('projection receipt differs')
            graph = helper.graph(prepared, text)
            verified = helper.local.verify_local_benchmark_admission(
                helper.local.admit_local_benchmark_plan(graph=graph, manifest=prepared['manifest']))
            if verified['receipt']['completion_authority'] is not False:
                raise ValueError('admission must not imply task completion')
            if not all(row['phase'] == 'post_execution' for row in verified['receipt']['pending_requirements']):
                raise ValueError('unexpected execution requirement phase')
            outputs = prepared['spec']['outputs']
            if any((Path(prepared['repository']) / row['path']).exists() for row in outputs):
                raise ValueError('fixture must not precreate declared task outputs')
            result.update(qualified=True, authored_process_adapter_runs=1,
                canonical_schema_validated=True, native_schema_validated=True,
                returned_proposal_unchanged=True, signed_declaration_unchanged=True,
                strict_graph_parser_passed=True, independent_signed_admission_passed=True,
                goals=len(graph.goals), tasks=len(graph.tasks),
                public_output_media_types=sorted({row['media_type'] for row in outputs}),
                pending_post_execution_requirements=len(verified['receipt']['pending_requirements']),
                native_schema_projection=projection,
                fixture_helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest())
        except BaseException as exc:
            result['error_type'] = type(exc).__name__ if type(exc).__name__ in {
                'ValueError', 'AssertionError', 'LocalPlanningError', 'PromptGoalProposalError',
                'LLMRouterError', 'TimeoutExpired', 'OSError', 'RuntimeError', 'SystemExit'} else 'other'
            traceback.print_exc()
    result['seconds'] = time.monotonic() - started
    result['source_after'] = recipe.snapshot(args.expected_source_head, strict=False)
    result['source_unchanged'] = result['source_after'] == before
    result['qualified'] = result['qualified'] and result['source_unchanged']
    with target.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['qualified'] else 1)


if __name__ == '__main__':
    main()
