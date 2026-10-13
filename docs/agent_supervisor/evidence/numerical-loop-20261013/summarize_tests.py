"""Count distinct final root-owned regressions without duplicating focused runs."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[];passed=[]
for repo,folder in [('datasets','datasets-final-02'),('accelerator','accelerator-final-01')]:
    path=R/folder/'junit.xml';tree=ET.parse(path)
    tests=tree.findall('.//testcase');ids=[];skipped=[];failed=[]
    for t in tests:
        node=repo+':'+t.attrib['classname']+'::'+t.attrib['name']
        assert node not in ids;ids.append(node)
        if t.find('failure') is not None or t.find('error') is not None:failed.append(node)
        elif t.find('skipped') is not None:skipped.append(dict(node=node,message=t.find('skipped').attrib.get('message','')))
        else:passed.append(node)
    assert not failed
    rows.append(dict(repository=repo,path=str(path.relative_to(R)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tests=len(tests),passed=len(tests)-len(skipped),skipped=skipped,failed=failed,exit=json.loads((R/folder/'exit.json').read_bytes())))
assert len(passed)==len(set(passed))
(R/'passed-node-ids.txt').write_text('\n'.join(sorted(passed))+'\n')
record=dict(schema='numerical-loop-regression-ledger@1',distinct_passed_tests=len(passed),failed_tests=0,skipped_tests=sum(len(r['skipped']) for r in rows),rows=rows,focused_agent_runs_counted_again=False)
(R/'test-ledger.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(record))
