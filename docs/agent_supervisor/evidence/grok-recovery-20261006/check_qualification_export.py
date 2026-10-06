"""Artifact-only qualification exporter safety checks; no provider calls."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import tempfile

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('qualification_export', ROOT/'export_qualification.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PRIVATE = 'PRIVATE_SENTINEL_MUST_NOT_BE_EXPORTED'
checks = []

def write_case(root, label='fixture', *, exit_code=0, changed=False, errors=False):
    source = {'accelerate_head': 'a'*40, 'datasets_head': 'b'*40,
              'source_sha256': {'ipfs_accelerate_py/fixture.py': 'c'*64},
              'datasets_source_sha256': {'ipfs_datasets_py/fixture.py': 'd'*64},
              'accelerate_status': '', 'datasets_status': ''}
    after = copy.deepcopy(source)
    if changed:
        after['source_sha256']['ipfs_accelerate_py/fixture.py'] = 'e'*64
    case = ('<testcase classname="" name="fixture"><error>'+PRIVATE+'</error></testcase>' if errors else
            '<testcase classname="test.fixture" name="case"><system-out>'+PRIVATE+'</system-out><system-err>'+PRIVATE+'</system-err></testcase>')
    raw = ('<testsuites><testsuite tests="1" failures="0" errors="'+str(int(errors))+'" skipped="0">'+case+'</testsuite></testsuites>').encode()
    command = {'before':source, 'environment_overrides': {'API_KEY':PRIVATE},'argv':[PRIVATE]}
    result = {'after':after, 'exit_code':exit_code, 'source_unchanged':not changed,
              'seconds':1.2,'xml_sha256':hashlib.sha256(raw).hexdigest(),'log_sha256':'f'*64}
    (root/(label+'-command.json')).write_text(json.dumps(command))
    (root/(label+'-exit.json')).write_text(json.dumps(result))
    (root/(label+'.xml')).write_bytes(raw)
    return command,result,raw

def rejects(action, label):
    try: action()
    except (ValueError, TypeError, KeyError): checks.append(label)
    else: raise AssertionError(label)

with tempfile.TemporaryDirectory(prefix='grok-export-review-') as directory:
    root=Path(directory)
    write_case(root)
    value=module.collect(root,['fixture'])
    assert PRIVATE not in json.dumps(value)
    assert value['runs'][0]['counts']=={'passed':1,'failed':0,'errors':0,'skipped':0}
    assert value['runs'][0]['qualified'] is True
    checks.append('capture_and_command_credentials_omitted')
    write_case(root,'repeated')
    value=module.collect(root,['fixture','repeated'])
    assert value['repeated_cases_not_summed'] is True and 'counts' not in value
    assert len(value['runs'])==2 and all(row['counts']['passed']==1 for row in value['runs'])
    checks.append('repeated_cases_remain_per_run_no_sum')
    rejects(lambda:module.collect(root,['fixture','fixture']), 'duplicate_labels_rejected')
    rejects(lambda:module.collect(root,['../fixture']), 'path_escape_rejected')
    write_case(root,changed=True)
    row=module.collect(root,['fixture'])['runs'][0]
    assert row['source_unchanged'] is False and row['qualified'] is False
    assert row['source_sha256_before'] != row['source_sha256']
    checks.append('source_changed_preserves_both_snapshots')
    _,result,_=write_case(root,changed=True)
    result['source_unchanged']=True
    (root/'fixture-exit.json').write_text(json.dumps(result))
    rejects(lambda:module.collect(root,['fixture']), 'false_source_unchanged_claim_rejected')
    write_case(root,errors=True,exit_code=2)
    row=module.collect(root,['fixture'])['runs'][0]
    assert row['outcome']=='collection_error' and row['collection_errors']==1 and row['counts']['passed']==0
    assert row['source_unchanged'] is True and row['qualified'] is False
    checks.append('collection_error_preserved_without_capture')
    _,result,raw=write_case(root)
    (root/'fixture.xml').write_bytes(raw+b' ')
    rejects(lambda:module.collect(root,['fixture']), 'xml_hash_drift_rejected')
    _,result,raw=write_case(root)
    altered=raw.replace(b'tests="1"',b'tests="2"')
    result['xml_sha256']=hashlib.sha256(altered).hexdigest()
    (root/'fixture.xml').write_bytes(altered)
    (root/'fixture-exit.json').write_text(json.dumps(result))
    rejects(lambda:module.collect(root,['fixture']), 'declared_count_disagreement_rejected')
    for key,bad in [('seconds',float('nan')),('exit_code',True),('xml_sha256',PRIVATE),('log_sha256',PRIVATE)]:
        _,result,_=write_case(root)
        result[key]=bad
        (root/'fixture-exit.json').write_text(json.dumps(result))
        rejects(lambda:module.collect(root,['fixture']), 'invalid_'+key+'_rejected')
    _,result,_=write_case(root)
    result['after']['source_sha256']={'/foreign/secret.py':'a'*64}
    result['source_unchanged']=False
    (root/'fixture-exit.json').write_text(json.dumps(result))
    rejects(lambda:module.collect(root,['fixture']), 'noncanonical_source_path_rejected')

report={'schema':'grok-qualification-export-safety-review@1','checks':checks,
        'passed':len(checks),'provider_calls':0,'production_source_changed':False,
        'exporter_sha256':hashlib.sha256((ROOT/'export_qualification.py').read_bytes()).hexdigest()}
(ROOT/'qualification-export-safety-review.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
