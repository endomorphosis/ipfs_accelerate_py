"""Authored offline statistics fixtures; no providers, containers or real trial claims."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import traceback


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reporter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-template', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
    from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls
    spec = importlib.util.spec_from_file_location('authored_pair_reporter', args.reporter)
    reporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(reporter)
    sources = [args.reporter, Path(__file__), Path(comparison.__file__)]
    before = {str(p): sha(p) for p in sources}
    template = json.loads(args.config_template.read_text())
    identity = dict(task='largest-eigenval', model='gpt-6.1-sol', reasoning_effort='high', cli_version='0.160.0')
    archive = 'a' * 64
    sentinel = 'AUTHORED_PRIVATE_PROMPT_AND_REPLY_MUST_NOT_APPEAR'

    def invocation(name, phase, input_tokens, output_tokens):
        return dict(invocation_id=name, phase=phase, provider='codex_cli', model=identity['model'],
            status='provider_returned', raw_model_text=sentinel, prompt=sentinel,
            native_rollout_usage=dict(schema='native-codex-cumulative-usage@1', cache_included_in_input=True,
                task_complete_observed=True, dollar_cost=None, usage=dict(input_tokens=input_tokens,
                    output_tokens=output_tokens, total_tokens=input_tokens+output_tokens,
                    cached_input_tokens=input_tokens//2, cache_write_input_tokens=0,
                    reasoning_output_tokens=output_tokens//2)))

    def fixture(case, label, *, suffix='', reward=1, empty=False):
        path = args.output / case / (label + suffix)
        config = deepcopy(template); config['jobs_dir'] = str(path / 'jobs')
        config['tasks'][0]['path'] = '/authored-public-task-fixture'
        if label == 'treatment': config['agents'][0]['kwargs']['semantic_metadata_view'] = 'common-bindings-instruction@1'
        else: config['agents'][0]['kwargs'].pop('semantic_metadata_view', None)
        write(path / 'config.json', config)
        controls = build_controls(config, task_input_sha256={'instruction.md': 'b'*64}, **identity)
        prepared = dict(schema='terminal-full-supervisor-preparation@1', arm='full', prepared=True,
            config_sha256=sha(path/'config.json'), task_input_sha256={'instruction.md':'b'*64},
            archive_sha256=archive, comparison_controls=controls, **identity)
        trial_id = 'authored-' + label + suffix
        values = (100, 10, 200, 20) if label == 'baseline' else (80, 8, 160, 16)
        report = dict(task_completed=reward==1, provider_invocations=[
            invocation(trial_id+'-planning','planning',*values[:2]),
            invocation(trial_id+'-coding','coding',*values[2:])],
            intent_preplanning={'status':'authored_fixture', 'enabled':False, 'generated_plan':sentinel})
        native = dict(verifier_result={'rewards':{'reward':reward}}, exception_info=None, task_name=identity['task'],
            agent_result={'metadata':{'runtime_archive_sha256':archive}}, authored_fixture=True)
        native_path = Path(config['jobs_dir']) / config['job_name'] / trial_id / 'result.json'
        write(native_path, native)
        write(native_path.parent/'agent'/'supervisor-result.json', report)
        trial = dict(trial=trial_id, native_result_sha256=sha(native_path), reward={'reward':reward},
            supervisor=report, agent_context={'metadata':{'runtime_archive_sha256':archive}}, exception_type=None,
            exact_trial_task_matches=True)
        receipt = dict(schema=comparison.SUPERVISOR, arm='full', **identity,
            original_task_inputs_unchanged=True, complete_single_trial_receipt=not empty,
            source384_enabled=True, trial_count=0 if empty else 1, trials=[] if empty else [trial],
            comparison_controls={'status':'observed','configuration_unchanged':True,'declared':controls})
        value=dict(path=path,receipt=receipt,prepared=prepared,config=config,native_path=native_path,native=native)
        return value

    outcomes=[]
    def run(name, mutate=None, verify=None, *, duplicate=None, extra_baseline=False):
        a=fixture(name,'baseline'); b=fixture(name,'treatment'); group=[a,b]
        if extra_baseline:group.append(fixture(name,'baseline',suffix='-prior-failed',reward=0))
        if mutate: mutate(a,b,group)
        for item in group:
            write(item['path']/'config.json',item['config']);write(item['path']/'preparation.json',item['prepared'])
            write(item['path']/'comparison.json',item['receipt'])
            if item['receipt']['trials']:
                write(item['native_path'].parent/'agent'/'supervisor-result.json',item['receipt']['trials'][0]['supervisor'])
        output=args.output/name/'pair.json'
        argv=[str(args.reporter),'--source',str(args.source),'--output',str(output)]
        for item in group:
            argv += ['--treatment' if item is b else '--baseline',str(item['path']/'comparison.json')]
        saved=sys.argv;sys.argv=argv
        try:
            try:
                with redirect_stdout(io.StringIO()):reporter.main()
            except ValueError:
                if duplicate is not True:raise
                return outcomes.append({'case':name,'passed':True,'behavior':'duplicate_or_conflict_refused'})
            if duplicate:raise AssertionError('duplicate retained trial was accepted')
            result=json.loads(output.read_text())
            assert sentinel not in output.read_text(), 'private authored prompt/reply leaked'
            assert result['native_codex_harness_compared'] is False
            assert result['causal_claim'] is False
            if verify:verify(result)
            outcomes.append({'case':name,'passed':True})
        except Exception as error:
            outcomes.append({'case':name,'passed':False,'error_type':type(error).__name__,'error':str(error),
                             'traceback':traceback.format_exc()})
        finally:sys.argv=saved

    def total(result,label):return result['label_totals']['full-lexical-baseline' if label=='baseline' else 'full-instruction-dedup']
    def expect(value,message):
        if not value:raise AssertionError(message)
    def check_unqualified(result):
        expect(result['all_pair_checks_pass'] is False,'stale binding must fail pair checks')
        expect(result['observed_token_ratio'] is None,'stale binding must not expose a qualified ratio')
    def incomplete(result):
        expect(total(result,'treatment')['tokens']['total_tokens'] is None,'incomplete native total must stay unknown')
        expect(result['observed_token_ratio'] is None,'incomplete total cannot yield ratio')
    def changed_call(a,b,group):return b['receipt']['trials'][0]['supervisor']['provider_invocations'][1]
    def completed(result):
        expect(result['all_pair_checks_pass'] is True,'honest completed pair should qualify')
        expect(total(result,'baseline')['tokens']['total_tokens']==330,'baseline counted cache/reasoning twice')
        expect(total(result,'treatment')['tokens']['total_tokens']==264,'treatment arithmetic differs')
        expect(result['observed_token_ratio']==.8,'expected observed ratio0.8')
    run('completed_pair',verify=completed)
    run('failed_attempt_retained',extra_baseline=True,verify=lambda r:(
        expect(total(r,'baseline')['tokens']['total_tokens']==660,'failed attempt tokens lost'),
        expect(total(r,'baseline')['observed_official_passes']==1,'failure counted as a pass'),
        expect(r['observed_token_ratio']==.4,'all-attempt numerator wrong')))
    run('incomplete_session',mutate=lambda a,b,g:changed_call(a,b,g)['native_rollout_usage'].update(task_complete_observed=False),verify=incomplete)
    run('unreceipted_attempt',mutate=lambda a,b,g:b['receipt']['trials'][0]['supervisor'].update(unreceipted_provider_attempt=True),verify=incomplete)
    run('unknown_reward',mutate=lambda a,b,g:b['receipt']['trials'][0].update(reward=None),verify=lambda r:expect(r['observed_token_ratio'] is None,'unknown reward yielded ratio'))
    def zero(a,b,g):
        b['receipt']['trials'][0]['reward']['reward']=0;b['native']['verifier_result']['rewards']['reward']=0
        write(b['native_path'],b['native']);b['receipt']['trials'][0]['native_result_sha256']=sha(b['native_path'])
    run('zero_official_passes',mutate=zero,verify=lambda r:expect(r['observed_token_ratio'] is None,'zero passes yielded ratio'))
    def empty(a,b,g):
        b['receipt'].update(trials=[],trial_count=0,complete_single_trial_receipt=False)
    run('zero_trial_receipt',mutate=empty,verify=lambda r:(check_unqualified(r),expect(len(r['receipts_without_trials'])==1,'empty attempt lost'),expect(total(r,'treatment')['tokens']['total_tokens'] is None,'empty attempt manufactured zero')))
    def empty_plus_pass(a,b,g):
        g[2]['receipt'].update(trials=[],trial_count=0,complete_single_trial_receipt=False)
    run('zero_trial_plus_pass',extra_baseline=True,mutate=empty_plus_pass,verify=lambda r:(check_unqualified(r),expect(total(r,'baseline')['tokens']['total_tokens'] is None,'empty attempt hidden behind completed pass')))
    for kind in ['duplicate','conflicting']:
        def mutation(a,b,g,kind=kind):
            v=deepcopy(a['receipt']['trials'][0])
            if kind=='conflicting':v['native_result_sha256']='c'*64
            a['receipt']['trials'].append(v);a['receipt']['trial_count']=2
        run(kind+'_trial',mutate=mutation,duplicate=True)
    run('stale_config',mutate=lambda a,b,g:b['config'].update(quiet=True),verify=check_unqualified)
    run('stale_prepared_controls',mutate=lambda a,b,g:b['prepared']['comparison_controls'].update(configuration_sha256='c'*64),verify=check_unqualified)
    run('stale_receipt_controls',mutate=lambda a,b,g:b['receipt']['comparison_controls'].update(declared=deepcopy(a['prepared']['comparison_controls'])),verify=check_unqualified)
    run('stale_receipt_identity',mutate=lambda a,b,g:b['receipt'].update(model='authored-other-model'),verify=check_unqualified)
    run('stale_runtime_archive',mutate=lambda a,b,g:b['receipt']['trials'][0]['agent_context']['metadata'].update(runtime_archive_sha256='c'*64),verify=check_unqualified)
    run('stale_native_result',mutate=lambda a,b,g:b['native_path'].write_text('{"authored":"changed"}\n'),verify=check_unqualified)
    run('stale_exported_reward',mutate=lambda a,b,g:b['receipt']['trials'][0]['reward'].update(reward=0),verify=check_unqualified)
    for kind in ['total','cache','reasoning','boolean']:
        def mutate(a,b,g,kind=kind):
            usage=changed_call(a,b,g)['native_rollout_usage']['usage']
            if kind=='total':usage['total_tokens']+=1
            elif kind=='cache':usage['cached_input_tokens']=usage['input_tokens']+1
            elif kind=='reasoning':usage['reasoning_output_tokens']=usage['output_tokens']+1
            else:usage['input_tokens']=True
        run('incoherent_'+kind,mutate=mutate,verify=lambda r:(check_unqualified(r),incomplete(r)))
    after={str(p):sha(p) for p in sources}
    result=dict(schema='authored-offline-pair-reporter-fixtures@1',cases=outcomes,
        passed=sum(x['passed'] for x in outcomes),failed=sum(not x['passed'] for x in outcomes),
        source_before=before,source_after=after,source_unchanged=before==after,
        provider_calls=0,container_actions=0,real_benchmark_trials=0,
        scope='Synthetic receipt arithmetic and binding checks only; no provider, reward or performance claims.')
    write(args.output/'qualification.json',result)
    print(json.dumps({key:result[key] for key in ['passed','failed','source_unchanged']}))
    return 0 if result['failed']==0 and result['source_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
