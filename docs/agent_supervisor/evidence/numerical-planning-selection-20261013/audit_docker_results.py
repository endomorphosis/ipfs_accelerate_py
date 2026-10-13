"""Read retained local evidence; export hashes/counts, never private wire data."""
from collections import Counter
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
P = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
D = Path('/home/barberb/lift_coding/.worktrees/indexed-coverage-datasets-20261013')
SOURCE = Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval')


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def bind(path, label=None):
    return dict(filename=label or path.name, bytes=path.stat().st_size,
                sha256=sha(path.read_bytes()))


def main():
    checks, inputs = [], []

    def check(name, value):
        assert value, name
        checks.append(dict(name=name, passed=True))

    baseline = read(R/'baseline.json')
    historical = read(R/'baseline-summary-v3.json')
    baseline_v3 = next(v for v in baseline['rows'] if v['version'] == 3)
    check('historical_summary_exact', sha(raw(historical)) == baseline_v3['summary_sha256']
          and len(raw(historical)) == 7614 and historical['pending_obligations'] == [])
    release = read(R/'docker-release-02.json')
    check('current_released_sources_exact', all(sha(Path(p).read_bytes()) == d
          for p, d in release['source_bindings'].items()))
    check('current_released_four_runner_files_exact', len(release['runner_bindings']) == 4
          and all(sha(Path(p).read_bytes()) == d for p, d in release['runner_bindings'].items()))
    for module, expected in baseline['frozen_producer_sha256'].items():
        path = D/Path(*module.split('.')).with_suffix('.py')
        check('unchanged_grammar_logic_core:' + path.name, sha(path.read_bytes()) == expected)
    failed = R/'docker-selected-01'
    fq, fp = read(failed/'qualification.json'), read(failed/'results/probe-qualification.json')
    check('historical_failure_fixture_only', not fq['complete'] and fp['error_type'] == 'AttributeError'
          and fp['error'] == "'str' object has no attribute 'get'"
          and not (failed/'private-state/owner-launch.json').exists()
          and not list((failed/'private-worker-observations').glob('numerical-*-audit.json')))
    check('historical_failure_cleaned_and_source_unchanged', all(fq[k] is True
          for k in ('container_removed', 'source_unchanged', 'runner_unchanged')))
    inputs += [bind(failed/'qualification.json', 'failed-qualification.json'),
               bind(failed/'results/probe-qualification.json', 'failed-probe-qualification.json')]
    instruction = (SOURCE/'instruction.md').read_bytes()
    completed = []
    for name in ('docker-selected-02', 'docker-full-01'):
        directory = R/name
        if not (directory/'qualification.json').exists():
            continue
        q = read(directory/'qualification.json')
        probe = read(directory/'results/probe-qualification.json')
        worker = read(directory/'private-fixture-ipc/worker-result.json')
        audit = read(directory/'results/broker-audit.json')
        command = read(directory/'command.json')
        formal_root = directory/'private-state/preflight/numerical-formalization'
        receipt = read(formal_root/'receipt.json')
        summary = read(formal_root/'planning-summary.json')
        index = read(formal_root/'reports.json')
        reports = {p: read(formal_root/v['artifact']) for p, v in index.items()}
        rows = {p: report['pending_obligations'] for p, report in reports.items()}
        check(name + ':released_run_complete', q['complete'] and probe['complete']
              and command['release_sha256'] == sha((R/'docker-release-02.json').read_bytes())
              and all(q[k] for k in ('source_unchanged', 'runner_unchanged', 'container_removed')))
        check(name + ':runner_snapshot_exact', all(sha((directory/'runner-snapshot'/Path(p).name).read_bytes()) == d
              for p, d in release['runner_bindings'].items()))
        check(name + ':source_report_population_unchanged',
              receipt['source_hashes'] == baseline['source_sha256']
              and {p: sha(raw(v)) for p, v in reports.items()} == baseline_v3['report_hashes']
              and sum(map(len, rows.values())) == 105)
        for module, relative in (
                ('consumer', 'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py'),
                ('obligation_retrieval', 'ipfs_datasets_py/logic/software_verification/numerical_obligation_retrieval.py')):
            base = P if module == 'consumer' else D
            check(name + ':producer:' + module, receipt['producers'][module] == sha((base/relative).read_bytes()))
        check(name + ':complete_summary_identity', summary == receipt['summary']
              and sha(raw(summary)) == probe['summary_sha256']
              and sha(raw(summary)) == probe['formalization_delivery']['native_summary_sha256']
              == probe['formalization_delivery']['model_summary_sha256']
              and probe['planner_request_summary_exact']
              and probe['owner_compiled_worker_summary_exact']
              and probe['worker_received_summary_exact']
              and probe['formalization_delivery']['model_formalization_exact'])
        check(name + ':summary_authority_fence', all(summary[k] is False for k in
              ('proof_authority', 'completion_authority', 'execution_authority',
               'source_semantics_verified', 'loop_invariants_verified', 'termination_verified'))
              and summary['source_obligations_discharged'] == 0)
        hints = summary['pending_obligations']
        check(name + ':balanced_hint_coverage_and_omissions', len(raw(summary)) == 8064
              and len(hints) == 8 and summary['omitted_obligation_summaries'] == 97
              and len({h['path'] for h in hints}) == 2
              and len({(h['path'], f) for h in hints for f in h['function_ids']}) == 4
              and len({(h['path'], loop) for h in hints for loop in h.get('loop_region_ids', [])}) == 2
              and len({h['proof_phase'] for h in hints if 'proof_phase' in h}) == 4
              and sum('proof_phase' in h for h in hints) == 5)
        hydration = probe['native_ducklake_hydration']
        check(name + ':native_ducklake_hydration', hydration['ducklake_verified'] is True
              and hydration['catalog_count'] == hydration['ducklake_catalogs'] == hydration['ducklake_links'] == 3
              and hydration['ducklake_world_records'] == 1)
        requests = audit['requests']
        check(name + ':broker_exact_accounting',
              len(requests) == audit['accepted_requests'] == probe['broker_requests']
              and [x['ordinal'] for x in requests] == list(range(1, len(requests)+1))
              and len({x['request_sha256'] for x in requests}) == len(requests)
              and sum(x['response_bytes'] for x in requests) == audit['response_bytes'] == probe['broker_response_bytes']
              and all(x['delivered'] for x in requests)
              and audit['accounting_complete'] and audit['service_stopped']
              and audit['inflight_request'] is None)
        check(name + ':unchanged_quota_and_no_exhaustion',
              audit['session']['max_requests'] == 64 and audit['accepted_requests'] <= 64
              and audit['session']['max_total_response_bytes'] == 524288
              and audit['response_bytes'] < 524288)
        ledger = {x['request_sha256']: x for x in requests}
        check(name + ':negative_and_cleanup_coverage', len(probe['negative_checks']) == 7
              and probe['broker_stopped'] and probe['descriptor_and_socket_removed']
              and probe['source_restored'] and probe['private_read_denial_count'] == 5)
        check(name + ':no_provider_or_proof_claim', all(probe[k] == 0 for k in
              ('provider_calls', 'model_calls', 'solver_calls', 'source_obligations_discharged'))
              and probe['native_provider_tokens'] is None
              and probe['live_planner_delivery_verified'] is False
              and probe['official_verifier_executed'] is False
              and probe['full_live_supervisor_deployment_qualified'] is False)
        result = dict(run=name, mode=worker['mode'], complete=True,
            summary_bytes=len(raw(summary)), inline_hints=len(summary['pending_obligations']),
            broker_requests=len(requests), broker_response_bytes=audit['response_bytes'],
            response_status_counts=dict(sorted(Counter(v['status'] for v in requests).items())))
        if worker['mode'] == 'selected':
            selection = worker['selection']; manifest = selection['manifest']
            measured = read(directory/'results/retrieval-comparison.json')
            check('selection:manifest_exact_bindings', measured == selection['comparison']
                  and sha(raw(manifest)) == measured['selection_sha256']
                  and manifest['summary_sha256'] == sha(raw(summary))
                  and manifest['instruction_sha256'] == sha(instruction))
            # Independently implement the declared visible-only grouping, without
            # importing the producer selector or inspecting reports to choose IDs.
            hints = summary['pending_obligations']; chosen, unavailable = [], []
            for path in ('eigen.py', 'eval.py'):
                available = [h for h in hints if h['path'] == path and h['kind'] == 'eigenpair_residual']
                if available: chosen.append(('residual:' + path, available[0]))
                else: unavailable.append('residual:' + path)
            if not any(h['path'] == 'eigen.py' for _, h in chosen):
                available = [h for h in hints if h['path'] == 'eigen.py']
                if available: chosen.append(('target-entrypoint:eigen.py', available[0]))
                else: unavailable.append('target-entrypoint:eigen.py')
            for phase in ('activation_entry', 'normal_or_continue_backedge',
                          'exit_cause_and_post_state', 'whole_activation_progress'):
                available = [h for h in hints if h['path'] == 'eval.py' and h.get('proof_phase') == phase]
                if available: chosen.append(('loop-proof-phase:' + phase, available[0]))
                else: unavailable.append('loop-proof-phase:' + phase)
            check('selection:public_visible_policy_exact',
                  [(x['scenario'], x['path'], x['obligation_id']) for x in manifest['selected']]
                  == [(label, h['path'], h['obligation_id']) for label, h in chosen]
                  and manifest['not_selectable'] == unavailable
                  and manifest['full_report_read_for_selection'] is False
                  and manifest['source_body_read_for_selection'] is False)
            retained = {(r['path'], r['obligation_id']): r['row'] for r in selection['rows']}
            all_pages = []
            for case in measured['selected_cases']:
                selected = case['selection']; key = (selected['path'], selected['obligation_id'])
                expected = next(r for r in rows[key[0]] if r['obligation_id'] == key[1])
                check('selection:full_row:' + key[1], expected == retained[key]
                      and sha(raw(expected)) == case['full_row_sha256']
                      and len(raw(expected)) == case['full_row_bytes']
                      and expected['status'] == 'pending'
                      and all(expected[k] is False for k in ('proof_authority', 'execution_authority',
                          'completion_authority', 'loop_invariants_verified', 'termination_verified')))
                check('selection:whole_native_row_exact:' + key[1],
                      case['exact_id_pages'][0]['row_sha256'] == [sha(raw(expected))]
                      and case['full_row_equal'])
                for category in ('filter_pages', 'exact_id_pages'):
                    pages = case[category]; total_key = 'existing_filter' if category == 'filter_pages' else 'exact_id'
                    totals = case[total_key]
                    for page in pages:
                        entry = ledger[page['request_sha256']]
                        check('selection:owner_ledger:' + page['request_sha256'],
                              all(entry[k] == page[k] for k in ('response_sha256', 'response_bytes', 'returned_rows'))
                              and entry['delivered'] and entry['status'] == 'ok'
                              and page['framed_request_bytes'] == page['request_bytes']+4
                              and page['framed_response_bytes'] == page['response_bytes']+4
                              and page['cli_stdout_bytes'] == page['response_bytes']+1)
                    check('selection:page_sum:' + key[1] + ':' + category,
                          totals['requests'] == len(pages) and all(totals[k] == sum(p[k] for p in pages)
                          for k in ('request_bytes', 'response_bytes', 'framed_request_bytes',
                                    'framed_response_bytes', 'cli_stdout_bytes')))
                    all_pages += pages
                filter_hashes = [v for page in case['filter_pages'] for v in page['row_sha256']]
                native_filter = [v for v in rows[key[0]] if selected['function_id'] in v['function_ids']
                                 and selected['kind'] == v['kind']]
                check('selection:filter_native_prefix:' + key[1], filter_hashes ==
                      [sha(raw(v)) for v in native_filter[:len(filter_hashes)]]
                      and sha(raw(expected)) in case['filter_pages'][-1]['row_sha256']
                      and all(sha(raw(expected)) not in page['row_sha256'] for page in case['filter_pages'][:-1]))
            check('selection:measured_request_population', len(all_pages) == probe['measured_request_hashes_match_owner_ledger'])
            for category, total_key in (('filter_pages','existing_filter'), ('exact_id_pages','exact_id')):
                pages = [p for case in measured['selected_cases'] for p in case[category]]
                total = measured[total_key]
                check('selection:batch_sum:' + total_key, total['requests'] == len(pages)
                      and all(total[k] == sum(p[k] for p in pages) for k in
                      ('request_bytes','response_bytes','framed_request_bytes','framed_response_bytes','cli_stdout_bytes'))
                      and total['canonical_request_response_bytes'] == total['request_bytes']+total['response_bytes']
                      and total['common_summary_bytes'] == len(raw(summary))
                      and total['summary_plus_canonical_request_response_bytes'] == len(raw(summary))+total['canonical_request_response_bytes'])
            result.update(selected_obligations=len(chosen), existing_filter=measured['existing_filter'],
                exact_id=measured['exact_id'], missing_hint_scenarios=unavailable,
                legacy_summary_discovery_measured=measured['legacy_summary_discovery_measured'])
        else:
            all_request_bytes, all_response_bytes, pages = [], [], 0
            for path, expected in rows.items():
                responses = worker['pages'][path]
                check('full:rows:' + path, [v for response in responses for v in response['page']['rows']] == expected)
                cursor = None
                for response in responses:
                    request = dict(schema='supervisor-numerical-obligation-request@1',
                        session_id=response['session_id'], request_id=response['request_id'],
                        query=dict(path=path,function_id=None,kind=None,cursor=cursor,limit=4,max_bytes=8192))
                    entry = ledger[sha(raw(request))]
                    check('full:request:' + sha(raw(request)), entry['response_sha256'] == sha(raw(response))
                          and entry['response_bytes'] == len(raw(response)) and entry['status']=='ok')
                    all_request_bytes.append(len(raw(request))); all_response_bytes.append(len(raw(response)))
                    cursor = response['page']['next_cursor']; pages += 1
                check('full:exhausted:' + path, cursor is None)
            req, resp = sum(all_request_bytes), sum(all_response_bytes)
            check('full:totals', probe['page_count'] == pages
                  and probe['complete_request_envelope_utf8_bytes'] == req
                  and probe['complete_response_envelope_utf8_bytes'] == resp
                  and probe['full_scan_framed_request_bytes'] == req+4*pages
                  and probe['full_scan_framed_response_bytes'] == resp+4*pages
                  and probe['full_scan_cli_stdout_bytes'] == resp+pages
                  and probe['full_scan_summary_plus_canonical_request_response_bytes'] == len(raw(summary))+req+resp)
            result.update(full_scan_rows=105, full_scan_pages=pages, request_bytes=req,
                response_bytes=resp, summary_plus_request_response_bytes=len(raw(summary))+req+resp)
        inputs.extend([bind(directory/'qualification.json',name+'-qualification.json'),
                       bind(directory/'results/broker-audit.json',name+'-private-broker-audit.json'),
                       bind(directory/'private-fixture-ipc/worker-result.json',name+'-private-worker-result.json')])
        completed.append(result)
    output = dict(schema='numerical-planning-selection-docker-independent-audit@1',
        status='passed', check_count=len(checks), checks=checks, completed_runs=completed,
        full_scan_pending=not any(v['mode']=='full' for v in completed),
        historical_fixture_failure=dict(run='docker-selected-01',production_failure=False,
            phase='post-preflight authored planner-summary inspection before owner-worker launch',
            cause='Fixture called .get on an allowed string in a mixed summary list.',
            correction='Fixture now filters isinstance(value,dict) before schema lookup; production behavior unchanged.',
            provider_calls=0,model_calls=0,broker_requests=0),
        historical_summary=dict(bytes=7614,inline_hints=0),
        current_declared_source_bindings=len(release['source_bindings']),
        release=bind(R/'docker-release-02.json'),inputs=inputs,
        public_export_allowlist=['comparison-design.json','public_query_comparison.py',
            'fixture-integration-audit.json','docker-results-independent-audit.json',
            'docker-selected-02/results/retrieval-comparison.json'],
        private_exports_forbidden=['raw broker/session receipts','raw worker-selected rows',
            'raw prompts or request identifiers','owner databases','private state and diagnostics'],
        limitations=['Canonical IPC bytes, framed IPC bytes and CLI stdout bytes are separate layers; do not add them together.',
            'Exact-ID versus existing-filter arms receive identical compact hints. Historical-summary discovery and model selection quality are unmeasured.',
            'Authored public instruction selection is not an autonomous provider decision or a proved task relevance ranking.',
            'Production planner request compilation and exact worker delivery are observed; the authored graph is not proof of live planner consumption.',
            'Exhaustive scan is a separate inventory reference, not the selective comparator baseline.',
            'Rows remain pending; no execution witness, invariant, termination, benchmark reward or native token advantage follows.'],
        provider_calls=0,model_calls=0,source_executed=False,proof_authority=False,
        source_obligations_discharged=0,native_provider_tokens=None,official_benchmark_score=None)
    target=R/'docker-results-independent-audit.json'
    target.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(artifact=target.name,sha256=sha(target.read_bytes()),checks=len(checks),
        completed=[v['run'] for v in completed],full_scan_pending=output['full_scan_pending'])))


if __name__=='__main__':
    main()
