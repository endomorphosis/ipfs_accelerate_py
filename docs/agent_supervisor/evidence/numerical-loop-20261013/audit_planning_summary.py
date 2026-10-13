"""Read-only review of bounded planning metadata; publish aggregates, not private rows."""
import hashlib
import json
import os
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
W = Path('/home/barberb/lift_coding')
P = W/'.worktrees/supervisor-comparison-contracts-20261011'
D = W/'.worktrees/indexed-coverage-datasets-20261013'
os.environ.update(IPFS_KIT_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI='0')
sys.path[:0] = [str(P), str(D)]
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
from audit_docker_evidence import load, read, sha, wire


def main():
    root = R/'docker-03/private-state/preflight/numerical-formalization'
    receipt = load(root/'receipt.json')
    summary = receipt['summary']
    index = load(root/'reports.json')
    reports = {name: load(R/'public-02'/(name+'.report.json')) for name in sorted(index)}
    links = {name: load(root/ref['unit_links_artifact']) for name, ref in index.items()}
    checks = []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise AssertionError(name)

    before = sha(P/'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py')
    check('exact_public_sources', set(reports) == {'eigen.py', 'eval.py'})
    check('persisted_summary_exact', read(root/'planning-summary.json') == wire(summary))
    check('record_summary_hash_exact', receipt['record']['summary_sha256'] == hashlib.sha256(wire(summary)).hexdigest())
    check('selected_profile', receipt['profile'] == summary['profile'] == native.CONTROL_PROFILE)
    check('production_summary_reconstruction', native._summary(reports, summary['source_tree_id'], links,
        profile=native.CONTROL_PROFILE) == summary)
    check('whole_summary_byte_bound', len(wire(summary)) <= 8192)
    check('authority_and_verification_flags', all(summary[k] is False for k in (
        'proof_authority', 'completion_authority', 'execution_authority', 'source_semantics_verified',
        'whole_program_semantics_verified', 'reachability_verified', 'loop_invariants_verified',
        'termination_verified', 'control_flow_semantics_verified', 'autoencoder_coverage_changed',
        'source_unit_links_establish_learned_coverage')))
    check('no_learning_or_provider_claim', all(summary[k] == 0 for k in (
        'source_obligations_discharged','training_steps','provider_calls','model_inference_calls')))
    total_functions = sum(r['summary']['functions'] for r in reports.values())
    check('function_counts_exact', summary['functions'] == total_functions and
        summary['structurally_complete_functions'] == sum(r['summary']['structurally_complete_functions'] for r in reports.values()))
    omissions = []
    for total_key, list_key, omitted_key, expected in (
        ('functions','per_function','omitted_function_summaries', total_functions),
        ('pending_obligation_count','pending_obligations','omitted_obligation_summaries', sum(len(r['pending_obligations']) for r in reports.values())),
        ('compound_contract_count','compound_contracts','omitted_compound_contract_summaries', sum(len(r['compound_contracts']) for r in reports.values())),
        ('assertion_requirement_count','assertion_requirements','omitted_assertion_requirement_summaries', sum(len(r['assertion_requirements']) for r in reports.values())),
        ('source_unit_count','source_unit_links','omitted_source_unit_link_summaries', sum(r['unit_count'] for r in links.values())),
        ('control_reference_count','control_references','omitted_control_reference_summaries', len(reports)),
    ):
        check('exact_omissions_'+list_key, summary[total_key] == expected == len(summary[list_key]) + summary[omitted_key])
        omissions.append(dict(category=list_key,total=expected,retained=len(summary[list_key]),omitted=summary[omitted_key]))
    for row in summary['control_references']:
        report = reports[row['path']]
        regions = report['structural']['control_regions']
        loops = sorted(r['region_id'] for r in regions if r['kind'] in {'for_statement','while_statement'})
        controls = sorted((r['region_id'] for r in regions), key=lambda x:(x not in loops,x))
        functions = report['structural']['program']['functions']
        for key, values in (
            ('function_ids',sorted(f['function_id'] for f in functions)),
            ('cfg_ids',sorted(f['cfg']['graph_id'] for f in functions)),
            ('control_region_ids',controls), ('loop_region_ids',loops)):
            check('bounded_exact_control_inventory_'+row['path']+'_'+key,
                  row[key] == values[:4] and row['omitted_'+key] == max(0,len(values)-4))
        check('control_inventory_unverified_'+row['path'], all(row[k] is False for k in (
            'reachability_verified','loop_invariants_verified','termination_verified','control_flow_semantics_verified')))
    check('loops_visible_first', bool(summary['control_references'][0]['loop_region_ids']))
    for row in summary['assertion_requirements']:
        full = next(r for r in reports[row['path']]['assertion_requirements'] if r['requirement_id'] == row['requirement_id'])
        for key in ('control_region_ids','loop_region_ids','guarded_region_ids','evaluation_guard_verified',
                    'reachability_verified','loop_invariants_verified','termination_verified','control_flow_semantics_verified'):
            check('assertion_scope_'+key, row[key] == full[key])
        check('assertion_cfg_references', all(row[key] == sorted({r[field] for r in full['cfg_occurrences']})
            for key,field in (('cfg_ids','cfg_id'),('cfg_block_ids','block_id'),('cfg_command_ids','command_id'))))
    proof_rows = [row for report in reports.values() for row in report['pending_obligations'] if 'proof_scope' in row]
    check('full_proof_applicability_unresolved', all(row['evaluation_context_role'] == 'source_anchor_only'
        and row['logical_guard_scope'] == 'unresolved_proof_scope_applicability'
        and row['proof_scope']['phase_established'] is False
        and row['proof_scope']['applicability_verified'] is False for row in proof_rows))
    check('summary_interpretation_preserves_scope_distinction', 'Loop proof applicability phases are distinct from source-anchor evaluation contexts' in summary['interpretation'])
    check('summary_retrieval_paths_exact', summary['obligation_retrieval']['source_paths'] == sorted(reports)
        and summary['obligation_retrieval']['omitted_source_paths'] == 0)
    check('consumer_source_unchanged', sha(P/'ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py') == before)
    result = dict(schema='numerical-control-planning-summary-independent-review@1', status='passed',
        consumer_sha256=before, checks=checks, summary_sha256=hashlib.sha256(wire(summary)).hexdigest(),
        summary_bytes=len(wire(summary)), omissions=omissions,
        complete_functions=summary['structurally_complete_functions'], functions=total_functions,
        source_reports={name:dict(file_sha256=sha(R/'public-02'/(name+'.report.json')),
            structural_status=report['structural']['status']) for name,report in reports.items()},
        full_proof_scope_rows=len(proof_rows), full_proof_phases=sorted({r['proof_scope']['proof_phase'] for r in proof_rows}),
        findings=[
            'No correctness blocker found in selected v3 summary wiring.',
            'The 7614-byte planning summary retains both source inventories and three assertions, but omits all 105 pending rows, four function summaries, four source-unit links and two compound contracts; every omission is counted.',
            'Full pending rows, guards, CFG occurrences and loop-proof scopes remain available through the separately audited source-replayed retrieval. Compact planning metadata alone is insufficient to interpret or discharge the obligations.',
            'Four complete function structures do not imply complete module translation: eval.py retains explicit module-level fences.',
            'This deterministic grammar expansion changes no learned checkpoint coverage and establishes no runtime, loop-invariant, termination, proof, benchmark or token advantage.'],
        limits=['Retained owner observations are read in place; private rows and identifiers are not exported.',
                'The summary was reconstructed with the production consumer and final public reports; independent checks additionally compare counts, inventories, scope flags and omission accounting.'])
    out = R/'planning-summary-review.json'
    with out.open('x') as handle:
        json.dump(result,handle,sort_keys=True,indent=2);handle.write('\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),summary_bytes=result['summary_bytes'],output=str(out))))


if __name__ == '__main__':
    main()
