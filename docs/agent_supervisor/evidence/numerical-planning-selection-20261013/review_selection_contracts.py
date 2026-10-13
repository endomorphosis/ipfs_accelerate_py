"""Independent client and compact-summary checks using authored source data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as owner
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as client
from ipfs_datasets_py.logic.software_verification import numerical_logic, numerical_logic_v2, numerical_logic_v3 as logic
from ipfs_datasets_py.logic.software_verification import numerical_obligation_retrieval as pager

R = Path(__file__).resolve().parent
PUBLIC = Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src')
sha = lambda data: hashlib.sha256(data).hexdigest()
raw = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
checks = []


def check(name, condition):
    if not condition: raise AssertionError(name)
    checks.append(name)


def summary(source):
    reports = {'authored.py':logic.formalize_numerical_source(source,path='authored.py')}
    return reports, owner._summary(reports,owner._tree({'authored.py':sha(source.encode())},logic.PROFILE),profile=logic.PROFILE)


cases = {
    'upstream_function_omissions': '\n'.join(f'def fn{i}(value):\n    return inspect(value)\n' for i in range(20)),
    'upstream_assertion_omissions': 'def inspect(value):\n' + ''.join('    assert valid(value)\n' for _ in range(6)),
    'upstream_compound_omissions': 'import numpy as np\n' + '\n'.join(
        f'def selected{i}(matrix):\n    values, vectors = np.linalg.eig(matrix)\n    chosen = np.argmax(np.abs(values))\n    return values[chosen], vectors[:, chosen]\n'
        for i in range(9)),
    'loop_proof_hints': 'def scan(items):\n    for item in items:\n        assert valid(item)\n    return items\n',
}
summary_results = []
for name, source in cases.items():
    reports, compact = summary(source)
    report = reports['authored.py']
    for collection, omitted, count in (
        ('per_function','omitted_function_summaries','functions'),
        ('pending_obligations','omitted_obligation_summaries','pending_obligation_count'),
        ('assertion_requirements','omitted_assertion_requirement_summaries','assertion_requirement_count'),
        ('compound_contracts','omitted_compound_contract_summaries','compound_contract_count'),
        ('control_references','omitted_control_reference_summaries','control_reference_count')):
        check(name+':'+collection+'_accounting',len(compact[collection])+compact[omitted]==compact[count])
    check(name+':byte_bound',len(owner._bytes(compact))<=8192)
    check(name+':context_required',compact['planning_hint_contract']['full_context_required'] is True)
    check(name+':no_authority',compact['source_obligations_discharged']==0 and compact['proof_authority'] is False)
    by_id={r['obligation_id']:r for r in report['pending_obligations']}
    for hint in compact['pending_obligations']:
        row=by_id[hint['obligation_id']]
        check(name+':hint_identity:'+row['obligation_id'],hint['expression_id']==row['expression_id'] and hint['function_ids']==row['function_ids'])
        selected=pager.page_numerical_obligations(report=report,source_text=source,path='authored.py',obligation_id=hint['obligation_id'])
        check(name+':whole_row_retrieval:'+row['obligation_id'],selected['rows']==[row])
        if 'proof_scope' in row:
            check(name+':proof_applicability:'+row['obligation_id'],hint['scope']=='proof_applicability'
                and hint['proof_phase']==row['proof_scope']['proof_phase']
                and hint['proof_scope_control_region_id']==row['proof_scope']['control_region_id'])
    summary_results.append(dict(name=name,source_sha256=sha(source.encode()),bytes=len(owner._bytes(compact)),
        functions=compact['functions'],assertions=compact['assertion_requirement_count'],compounds=compact['compound_contract_count'],
        retained_hints=len(compact['pending_obligations'])))

source=cases['loop_proof_hints'];report=logic.formalize_numerical_source(source,path='authored.py')
row=next(r for r in report['pending_obligations'] if r['kind']=='loop_exit_reasoning')
query=dict(path='authored.py',function_id=row['function_ids'][0],kind=row['kind'],obligation_id=row['obligation_id'],cursor=None,limit=8,max_bytes=8192)
request=dict(schema=client.REQUEST_SCHEMA,session_id='a'*32,request_id='b'*32,query=query)
page=pager.page_numerical_obligations(report=report,source_text=source,path='authored.py',
    function_id=query['function_id'],kind=query['kind'],obligation_id=query['obligation_id'],max_bytes=7500)
body=dict(schema=client.RESPONSE_SCHEMA,session_id=request['session_id'],request_id=request['request_id'],ok=True,
    reason_code=None,page=dict(page,schema='supervisor-numerical-obligation-page@1'))
check('actual_dataset_singleton_accepted',client.validate_response(body,request=request)==body)
mutations={'offset':1,'remaining_count':1,'exhausted':False,'status':'more_available','matched_count':True,
           'returned_count':True,'next_cursor':{'offset':1}}
for field,value in mutations.items():
    bad=deepcopy(body);bad['page'][field]=value
    refused=False
    try:client.validate_response(bad,request=request)
    except client.NumericalObligationClientError:refused=True
    check('client_refuses_contradictory_'+field,refused)
for field,value in {'kind':'foreign','function_ids':['foreign'],'obligation_id':'foreign'}.items():
    bad=deepcopy(body);bad['page']['rows'][0][field]=value
    refused=False
    try:client.validate_response(bad,request=request)
    except client.NumericalObligationClientError:refused=True
    check('client_refuses_foreign_row_'+field,refused)
for value in ('\ud800','\udfff','é'*257,None):
    bad=deepcopy(request);bad['query']['obligation_id']=value
    refused=False
    try:client.validate_request(bad)
    except client.NumericalObligationClientError:refused=True
    check('client_refuses_malformed_id_'+str(len(checks)),refused)

baseline=json.loads((R/'baseline.json').read_bytes())
sources={p:(PUBLIC/p).read_text() for p in baseline['source_sha256']}
for version,old in enumerate((numerical_logic,numerical_logic_v2),1):
    reports={p:old.formalize_numerical_source(text,path=p) for p,text in sources.items()}
    compact=owner._summary(reports,owner._tree(baseline['source_sha256'],old.PROFILE),
        owner._function_unit_links(sources,reports),profile=old.PROFILE)
    expected=next(row for row in baseline['rows'] if row['version']==version)
    check(f'v{version}_summary_exact_historical_bytes',sha(raw(compact))==expected['summary_sha256'])

result=dict(schema='numerical-selection-independent-contract-review@1',passed=True,checks=len(checks),check_names=checks,
    authored_summary_cases=summary_results,producer_sha256={Path(m.__file__).name:sha(Path(m.__file__).read_bytes()) for m in (owner,client,pager,logic)},
    script_sha256=sha(Path(__file__).read_bytes()),source_executed=False,provider_calls=0,solver_calls=0,
    findings=[dict(issue='Selected replies admitted contradictory singleton accounting and mismatched conjunctive row filters.',status='resolved_and_independently_retested'),
        dict(issue='Compact omission counters lost upstream summary truncation.',status='resolved_and_independently_retested')],
    limitations=['This audit checks authored source data and local response validation, not a live provider or isolated UID transport.',
        'Worker text selectors intentionally reject NUL/newline; all native obligation IDs use the compatible hash-based spelling.',
        'Hints require full source-replayed rows before logical interpretation; retrieval grants no proof authority.'])
(R/'selection-contract-review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),authored_summary_cases=len(summary_results))))
