"""Capture source-bound reports, old summaries and historical pager bytes."""
from pathlib import Path
import hashlib,json,subprocess
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as owner
from ipfs_datasets_py.logic.software_verification import numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,numerical_source_v3,numerical_logic_v3,program,numerical_obligation_retrieval as pager
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
D=Path('/home/barberb/lift_coding/.worktrees/indexed-coverage-datasets-20261013')
PUBLIC=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src')
raw=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
sha=lambda value:hashlib.sha256(value).hexdigest()
write=lambda path,value:path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
sources={p.name:p.read_text() for p in [PUBLIC/'eigen.py',PUBLIC/'eval.py']}
hashes={p:sha(v.encode()) for p,v in sources.items()}
rows=[]
for version,logic in enumerate([numerical_logic,numerical_logic_v2,numerical_logic_v3],1):
    reports={p:logic.formalize_numerical_source(v,path=p) for p,v in sources.items()}
    links=owner._function_unit_links(sources,reports)
    summary=owner._summary(reports,owner._tree(hashes,logic.PROFILE),links,profile=logic.PROFILE)
    write(R/f'baseline-summary-v{version}.json',summary)
    pages=[]
    for path,report in reports.items():
        first=pager.page_numerical_obligations(report=report,source_text=sources[path],path=path,limit=1)
        second=pager.page_numerical_obligations(report=report,source_text=sources[path],path=path,limit=1,cursor=first['next_cursor'])
        pages.append(dict(path=path,first=first,second=second))
    write(R/f'baseline-pages-v{version}.json',pages)
    rows.append(dict(version=version,profile=logic.PROFILE,report_hashes={p:sha(raw(v)) for p,v in reports.items()},summary_sha256=sha(raw(summary)),summary_bytes=len(raw(summary)),inline_pending=len(summary['pending_obligations']),page_sha256=sha(raw(pages))))
modules=[numerical_source,numerical_logic,numerical_source_v2,numerical_logic_v2,numerical_source_v3,numerical_logic_v3,program]
result=dict(schema='numerical-planning-selection-baseline@1',source_sha256=hashes,rows=rows,frozen_producer_sha256={m.__name__:sha(Path(m.__file__).read_bytes()) for m in modules},structured_checkpoint_implementation=structured_source_384._implementation(),owner_sha256=sha(Path(owner.__file__).read_bytes()),pager_sha256=sha(Path(pager.__file__).read_bytes()),accelerator_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),datasets_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=D,text=True).strip(),provider_calls=0,source_executed=False)
write(R/'baseline.json',result);print(json.dumps(rows))
