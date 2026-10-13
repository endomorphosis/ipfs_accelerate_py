"""Independent read-only audit; publish only aggregate results and hashes.

Reads retained private receipts in place. Never copies their rows, descriptors,
queries, prompts, catalogs or signatures into the public audit document.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path('/home/barberb/lift_coding')
P = WORKSPACE / '.worktrees/supervisor-comparison-contracts-20261011'
D = WORKSPACE / '.worktrees/indexed-coverage-datasets-20261013'
PUBLIC_SOURCE = WORKSPACE / '.benchmarks/terminal-bench-2/largest-eigenval/environment/src'
PROFILE = 'python-numerical-control-structure@3'
sys.path[:0] = [str(P), str(D)]


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def read(path, maximum=32_000_000):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError('bounded regular retained evidence required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        def identity(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
                    value.st_mode, value.st_uid, value.st_gid, value.st_nlink)
        if identity(os.fstat(fd)) != identity(info):
            raise ValueError('retained evidence changed before read')
        chunks, length = [], 0
        while length <= maximum:
            chunk = os.read(fd, min(1024*1024, maximum + 1 - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
        data = b''.join(chunks)
        if identity(os.fstat(fd)) != identity(info) or len(data) != info.st_size:
            raise ValueError('retained evidence changed during read')
        return data
    finally:
        os.close(fd)


def sha(path):
    return hashlib.sha256(read(path)).hexdigest()


def load(path):
    return json.loads(read(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docker-dir', '--run', dest='run', type=Path, default=ROOT/'docker-01')
    parser.add_argument('--release', type=Path, default=ROOT/'docker-release-01.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.absolute()
    if run.parent != ROOT or output.parent != ROOT or output.exists():
        raise ValueError('fresh audit output and qualified local run required')
    checks, evidence, details = [], [], []

    def check(name, condition, **extra):
        checks.append(dict(name=name, passed=bool(condition), **extra))
        if not condition:
            raise AssertionError(name)

    def retained(relative):
        path = run / relative
        evidence.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        return load(path)

    result = dict(schema='numerical-worker-docker-independent-evidence-audit@2',
        profile=PROFILE, status='failed', checks=checks, evidence_bindings=evidence,
        source_replayed_files=details, provider_calls=0, source_obligations_discharged=0,
        benchmark_source_executed=False, production_files_changed=False)
    try:
        qualification = retained('qualification.json')
        probe = retained('results/probe-qualification.json')
        command = retained('command.json')
        controls = retained('container-controls.json')
        session = retained('results/session-observation.json')
        router = retained('results/router-observation.json')
        audit = retained('results/broker-audit.json')
        worker = retained('private-fixture-ipc/worker-result.json')
        release = load(args.release)
        check('completed_owner_and_exact_probe', qualification['complete'] is True
              and qualification['owner_returncode'] == 0 and qualification['probe'] == probe)
        check('release_manifest_digest', sha(args.release) == command['release_sha256'])
        check('release_source_and_runner_inventory', command['source_bindings'] == release['source_bindings']
              and command['runner_bindings'] == release['runner_bindings'])
        permitted_inputs = {PUBLIC_SOURCE/'eigen.py', PUBLIC_SOURCE/'eval.py',
            PUBLIC_SOURCE.parent.parent/'instruction.md',
            WORKSPACE/'artifacts/terminal-suite-eigenval-pilot-20261004/largest-eigenval-profile.json'}
        for name, digest in release['source_bindings'].items():
            path = Path(name)
            check('source_binding_allowed', path in permitted_inputs or (path.suffix == '.py' and
                  (path.is_relative_to(P) or path.is_relative_to(D))))
            check('current_source_binding', sha(path) == digest)
        for name, digest in release['runner_bindings'].items():
            path = Path(name)
            check('runner_binding_allowed', path.parent == ROOT and path.suffix == '.py')
            check('runner_snapshot_binding', sha(path) == digest == sha(run/'runner-snapshot'/path.name))

        check('router_private_public_match', router == load(run/'private-fixture-ipc/router-observation.json'))
        owner_audits = list((run/'private-worker-observations').glob('numerical-*-audit.json'))
        check('broker_private_public_match', len(owner_audits) == 1 and audit == load(owner_audits[0]))
        formal_root = run/'private-state/preflight/numerical-formalization'
        formal = load(formal_root/'receipt.json')
        check('selected_control_profile', formal['profile'] == formal['summary']['profile'] == PROFILE)
        index = load(formal_root/'reports.json')
        check('exact_permitted_source_population', set(index) == set(worker['pages']) == {'eigen.py', 'eval.py'})
        from ipfs_datasets_py.logic.software_verification import numerical_logic_v3 as logic
        initial, all_responses, expected_rows = [], [], {}
        for number, name in enumerate(sorted(index)):
            ref = index[name]
            check('canonical_report_locator_' + name, ref['artifact'] == f'report-{number:04d}.json')
            report_path = formal_root/ref['artifact']
            check('native_report_artifact_hash_' + name, sha(report_path) == ref['sha256'])
            source = read(PUBLIC_SOURCE/name).decode()
            report = logic.validate_numerical_logic_report(load(report_path), source, path=name)
            public_report = ROOT/'public-02'/(name+'.report.json')
            check('final_public_report_join_' + name, report == load(public_report))
            evidence.append(dict(path=str(public_report.relative_to(ROOT)), sha256=sha(public_report)))
            check('public_source_binding_' + name, hashlib.sha256(source.encode()).hexdigest()
                  == report['source_sha256'] == probe['public_source_hashes'][name])
            expected = report['pending_obligations']
            expected_rows[name] = expected
            responses = worker['pages'][name]
            actual = [row for response in responses for row in response['page']['rows']]
            check('exact_source_replayed_rows_' + name, actual == expected)
            check('unique_obligation_ids_' + name, len({r['obligation_id'] for r in actual}) == len(actual))
            check('full_control_contexts_preserved_' + name, all(
                r['evaluation_guard_verified'] is False and r['reachability_verified'] is False
                and r['loop_invariants_verified'] is False and r['termination_verified'] is False
                and r['control_flow_semantics_verified'] is False
                and type(r['cfg_occurrences']) is list and type(r['loop_region_ids']) is list
                for r in actual))
            sizes = [len(wire(response)) for response in responses]
            check('initial_metrics_' + name, probe['per_file'][name] == dict(
                expected_rows=len(expected), retrieved_rows=len(actual), page_count=len(responses),
                complete_envelope_bytes=sizes, rows_sha256=hashlib.sha256(wire(actual)).hexdigest()))
            details.append(dict(path=name, source_sha256=report['source_sha256'],
                report_sha256=report['report_sha256'], row_count=len(actual),
                structural_status=report['structural']['status'],
                functions=report['summary']['functions'],
                structurally_complete_functions=report['summary']['structurally_complete_functions'],
                control_regions=report['summary']['control_region_count'],
                loop_regions=report['summary']['loop_region_count'],
                control_context_row_count=sum(bool(r['control_region_ids']) for r in actual),
                loop_context_row_count=sum(bool(r['loop_region_ids']) for r in actual),
                source_anchor_proof_scope_count=sum('proof_scope' in r for r in actual),
                proof_phases=sorted({r['proof_scope']['proof_phase'] for r in actual if 'proof_scope' in r}),
                initial_pages=len(responses), initial_response_bytes=sum(sizes), max_response_bytes=max(sizes),
                source_replay_exact=True))
            initial.extend(responses)
            all_responses.extend(responses)
        filter_checks = []
        for selected in worker['filters']:
            name, function, kind = selected['path'], selected['function_id'], selected['kind']
            expected = [r for r in expected_rows[name] if (function is None or function in r['function_ids'])
                        and (kind is None or kind == r['kind'])]
            actual = [r for response in selected['responses'] for r in response['page']['rows']]
            check('exact_filtered_rows', actual == expected)
            filter_checks.append(dict(path=name, function_id=function, kind=kind,
                                      matched_rows=len(actual), page_count=len(selected['responses'])))
            all_responses.extend(selected['responses'])
        check('filter_metrics', filter_checks == probe['filter_checks'])
        check('bounded_complete_response_envelopes', all(r['ok'] is True and len(wire(r)) <= 8192 for r in all_responses))
        rows_by_digest = {r['response_sha256']: r for r in audit['requests']}
        check('unique_owner_response_hashes', len(rows_by_digest) == len(audit['requests']))
        check('retained_responses_match_owner_accounting', all(
            (record := rows_by_digest.get(hashlib.sha256(wire(response)).hexdigest())) is not None
            and record['response_bytes'] == len(wire(response)) and record['delivered'] is True
            and record['status'] == 'ok' and record['returned_rows'] == len(response['page']['rows'])
            for response in all_responses))
        requests = audit['requests']
        counts = Counter(row['status'] for row in requests)
        negatives = {'wrong_peer_uid1000':'peer_refused','unrelated_peer_uid1001':'peer_refused',
            'forged_cursor':'retrieval_refused','crossed_path_cursor':'retrieval_refused',
            'crossed_filter_cursor':'retrieval_refused','stale_canonical':'stale_context','stale_allocated':'stale_context'}
        check('seven_negative_cases', len(probe['negative_checks']) == 7
              and {row['case']:row['reason_code'] for row in probe['negative_checks']} == negatives)
        check('worker_negative_subset', all(row in probe['negative_checks'] for row in worker['negatives'])
              and len(worker['negatives']) == 5)
        check('complete_owner_accounting', audit['accounting_complete'] is True and audit['service_stopped'] is True
              and audit['inflight_request'] is None and audit['accepted_requests'] == len(requests)
              and audit['response_bytes'] == sum(row['response_bytes'] for row in requests)
              and all(row['delivered'] is True for row in requests))
        check('expected_session_statuses', counts == Counter(ok=len(all_responses)+1,
            peer_refused=2, retrieval_refused=3, stale_context=2))
        check('final_qualification_exact_request_quota', len(requests) == 64
              and audit['response_bytes'] <= 524288)
        check('refusals_contain_no_rows', all(row['returned_rows'] == 0 for row in requests if row['status'] != 'ok'))
        check('session_collection_matches_owner', session['status'] == 'observed'
              and session['counts']['accepted_requests'] == audit['accepted_requests']
              and session['counts']['response_bytes'] == audit['response_bytes']
              and session['counts']['delivered_rows'] == sum(row['returned_rows'] for row in requests)
              and session['sessions'][0]['status_counts'] == dict(counts))
        dispatch = router['receipt']['numerical_obligation_retrieval']
        check('router_dispatch_bound_to_owner', dispatch['descriptor_sha256'] == audit['descriptor_sha256']
              and all(dispatch[key] == audit['session'][key] for key in (
                  'session_id','task_cid','task_id','source_tree_id','context_receipt_sha256',
                  'context_bundle_sha256','boundary_manifest_sha256')))
        calls = router['authored_generate_calls']
        check('authored_dispatch_prompt_binding', len(calls) == 1 and calls[0]['authored_provider_response'] is True
              and calls[0]['prompt_sha256'] == router['receipt']['model_prompt_sha256']
              and calls[0]['prompt_utf8_bytes'] == router['receipt']['model_prompt_bytes'])
        check('private_read_denials_and_worker_boundary', worker['uid'] == 1001
              and worker['no_new_privileges'] is True and worker['worker_effective_capabilities'] == 0
              and len(worker['private_read_denials']) == probe['private_read_denial_count'] == 5)
        modes = {name:stat.S_IMODE((run/name).lstat().st_mode) for name in
                 ('private-state','private-fixture-ipc','private-worker-observations')}
        check('private_parent_custody', stat.S_IMODE(run.lstat().st_mode) == 0o700
              and modes == {'private-state':0o700,'private-fixture-ipc':0o750,'private-worker-observations':0o750})
        check('descriptor_and_socket_removed', not list((run/'private-worker-observations').glob('numerical-*.sock'))
              and not (run/'private-worker-observations'/('numerical-'+audit['session']['session_id']+'.json')).exists())
        check('container_controls', controls['controls'] == dict(Memory=12*1024**3, NanoCpus=5_000_000_000,
            NetworkMode='none', PidsLimit=128, Privileged=False, ReadonlyRootfs=False))
        allowed_mounts = {'/opt/ipfs-supervisor/source':P,'/opt/ipfs-supervisor/datasets':D,
            '/opt/ipfs-supervisor/kit':WORKSPACE/'.worktrees/publication-main-20261004/kit',
            '/extensions':Path('/home/barberb/.duckdb/extensions/v1.5.5/linux_arm64'),
            '/qualification/owner_probe.py':ROOT/'owner_probe.py','/qualification/worker_probe.py':ROOT/'worker_probe.py',
            '/permitted/eigen.py':PUBLIC_SOURCE/'eigen.py','/permitted/eval.py':PUBLIC_SOURCE/'eval.py',
            '/permitted/instruction.md':PUBLIC_SOURCE.parent.parent/'instruction.md',
            '/permitted/profile.json':WORKSPACE/'artifacts/terminal-suite-eigenval-pilot-20261004/largest-eigenval-profile.json',
            '/results':run/'results'}
        check('exact_mount_allowlist', len(controls['mounts']) == len(allowed_mounts)
              and {row['Destination']:Path(row['Source']) for row in controls['mounts']} == allowed_mounts
              and all(row['Type'] == 'bind' and row['RW'] is (row['Destination'] == '/results') for row in controls['mounts']))
        remaining = subprocess.run(['docker','ps','-aq','--filter','id='+qualification['container_id']],
                                   check=True, capture_output=True, text=True, timeout=10)
        check('container_cleanup_rechecked', qualification['container_removed'] is True and not remaining.stdout.strip())
        check('provider_free_flags', all(value['provider_calls'] == value['model_calls'] == 0
              for value in (qualification,probe,worker,router)) and audit['retrieval_provider_calls'] == 0)
        check('no_benchmark_or_token_claim', qualification['official_terminal_bench_score'] is None
              and qualification['native_provider_tokens'] is None and qualification['paid_token_savings_measured'] is False
              and qualification['full_runtime_deployment_qualified'] is False
              and probe['official_verifier_executed'] is False and probe['source_obligations_discharged'] == 0)
        check('source_and_runner_fence', qualification['source_unchanged'] is qualification['runner_unchanged'] is True
              and all(sha(name) == digest for name, digest in release['source_bindings'].items())
              and all(sha(name) == digest for name, digest in release['runner_bindings'].items()))
        result.update(status='passed', metrics=dict(unique_obligations=sum(map(len,expected_rows.values())),
            initial_pages=len(initial), initial_canonical_response_bytes=sum(len(wire(row)) for row in initial),
            max_initial_response_bytes=max(len(wire(row)) for row in initial),
            retained_full_success_response_count=len(all_responses), complete_session_requests=len(requests),
            docker_filter_selections=len(worker['filters']),
            docker_filtered_success_pages=sum(len(r['responses']) for r in worker['filters']),
            configured_session_request_limit=64, configured_session_response_limit_bytes=524288,
            successful_session_pages=counts['ok'], complete_session_canonical_response_bytes=audit['response_bytes'],
            delivered_row_instances_including_repeated_filtered_reads=sum(row['returned_rows'] for row in requests),
            negative_cases=7, private_read_denials=5, live_provider_calls=0, provider_tokens=None))
    except Exception as error:
        result['error_type'] = type(error).__name__
        result['failed_check'] = next((row['name'] for row in reversed(checks) if not row['passed']), None)
    result['publishable_subset'] = [str((run/name).relative_to(ROOT)) for name in (
        'qualification.json','results/probe-qualification.json','results/session-observation.json',
        'container-controls.json','command.json','owner-command.json','runner-snapshot/owner_probe.py',
        'runner-snapshot/worker_probe.py','runner-snapshot/run_docker_qualification.py')] + [args.release.name, output.name]
    result['retain_private_only'] = ['private-state/**','private-fixture-ipc/**','private-worker-observations/**',
        'results/broker-audit.json','results/router-observation.json','owner.stdout','owner.stderr']
    result['qualifications'] = [
        'Actual generated owner entry, UID1000/1001 boundary, router and numerical IPC; provider selection and generate_text remain authored hooks.',
        'Full session counts include final recovery and any filtered repetitions explicitly counted in docker_filter_selections. Retained full response bodies exclude final recovery and seven refusals. Canonical bytes exclude four-byte frame headers.',
        'The final Docker qualification devotes the 64-request limit to complete source paging and negative/recovery checks. Successful filtered retrieval is covered by separate unit/integration qualifications, not by this Docker audit.',
        'Public source was parsed and replayed, never executed by this audit. Loop/guard/library semantics, invariants and termination remain unproved.',
        'Five private read denials are bounded custody evidence. Fixture outputs were writable during the controlled run; this is not a general hostile-worker or rootless-container proof.',
        'Container worktrees were removed. Source restoration and DuckLake hydration are retained owner observations; this audit does not independently reopen removed worktrees or publish catalog contents.',
        'No live model, learned inference, native full supervisor START, official benchmark reward or paid token improvement is qualified.']
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), output=str(output))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
