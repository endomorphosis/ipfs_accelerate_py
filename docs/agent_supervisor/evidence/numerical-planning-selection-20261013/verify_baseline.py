"""Replay frozen public producers and historic no-ID pages after selection changes."""
from pathlib import Path
import hashlib,json,subprocess
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as owner
from ipfs_datasets_py.logic.software_verification import numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,numerical_source_v3,numerical_logic_v3,program,numerical_obligation_retrieval as pager
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384
R=Path(__file__).resolve().parent
PUBLIC=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src')
raw=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
sha=lambda value:hashlib.sha256(value).hexdigest()
def write(path,value):
    with path.open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
baseline=json.loads((R/'baseline.json').read_bytes())
sources={p.name:p.read_text() for p in [PUBLIC/'eigen.py',PUBLIC/'eval.py']}
hashes={p:sha(v.encode()) for p,v in sources.items()}
assert hashes==baseline['source_sha256']
modules=[numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,numerical_source_v3,numerical_logic_v3,program]
assert {m.__name__:sha(Path(m.__file__).read_bytes()) for m in modules}==baseline['frozen_producer_sha256']
assert structured_source_384._implementation()==baseline['structured_checkpoint_implementation']
rows=[]
for version,logic in enumerate([numerical_logic,numerical_logic_v2,numerical_logic_v3],1):
    old=baseline['rows'][version-1]
    reports={p:logic.formalize_numerical_source(v,path=p) for p,v in sources.items()}
    assert {p:sha(raw(v)) for p,v in reports.items()}==old['report_hashes']
    links=owner._function_unit_links(sources,reports)
    summary=owner._summary(reports,owner._tree(hashes,logic.PROFILE),links,profile=logic.PROFILE)
    if version<3: assert sha(raw(summary))==old['summary_sha256']
    else:
        assert len(raw(summary))<=8192
        assert len(summary['pending_obligations'])+summary['omitted_obligation_summaries']==105
        write(R/'qualified-summary-v3.json',summary)
    pages=[]
    for path,report in reports.items():
        first=pager.page_numerical_obligations(report=report,source_text=sources[path],path=path,limit=1)
        second=pager.page_numerical_obligations(report=report,source_text=sources[path],path=path,limit=1,cursor=first['next_cursor'])
        pages.append(dict(path=path,first=first,second=second))
    assert sha(raw(pages))==old['page_sha256']
    rows.append(dict(version=version,profile=logic.PROFILE,summary_sha256=sha(raw(summary)),summary_bytes=len(raw(summary)),inline_pending=len(summary['pending_obligations']),historical_pages_equal=True,report_bytes_equal=True,historical_summary_equal=sha(raw(summary))==old['summary_sha256']))
result=dict(schema='numerical-planning-selection-public-replay@1',complete=True,rows=rows,grammar_and_checkpoint_pins_equal=True,provider_calls=0,source_executed=False,native_provider_tokens=None,source_obligations_discharged=0,owner_sha256=sha(Path(owner.__file__).read_bytes()),pager_sha256=sha(Path(pager.__file__).read_bytes()))
write(R/'public-replay.json',result);print(json.dumps(result))
