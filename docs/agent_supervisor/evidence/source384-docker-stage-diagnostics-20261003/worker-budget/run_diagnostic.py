"""Run only after root review/GO. Preserves the uninstrumented archive."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

B=Path(__file__).parent
ARCHIVE=B.parent/'source384-docker-reconstruction-20261003/bundle'
EXPECTED_ARCHIVE='43ba7c9bc830b9c6c684589db15ddad25a37339014a5287d41a962c10d98f9c3'
TASK=Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/fix-code-vulnerability')


def instrument_probe(original):
    runtime=(B/'timing_runtime.py').read_text()
    anchor="  result['producer']=_pins()\n"
    assert original.count(anchor)==1
    result=original.replace(anchor,anchor+'  exec('+repr(runtime)+',globals())\n  _diag_install()\n')
    anchor='finally:\n signal.setitimer(signal.ITIMER_REAL,0)\n'
    assert result.count(anchor)==1
    replacement='''finally:
 try:
  if '_diag_finish' in globals(): _diag_finish()
 except BaseException as diagnostic_error:
  result['diagnostic_collection_error']=type(diagnostic_error).__name__
 result['diagnostic_only']=True
 result['production_qualification_claimed']=False
 signal.setitimer(signal.ITIMER_REAL,0)
'''
    result=result.replace(anchor,replacement)
    compile(result,'<external-source384-stage-diagnostic>','exec')
    return result


async def run():
    from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
    from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualifier
    from ipfs_accelerate_py.agent_supervisor.runtime.source384_config import _regular_bytes
    go=json.loads((B/'run-admission.json').read_text())
    assert go['root_go_received'] is True and go['diagnostic_only'] is True
    assert not (B/'docker-01').exists(), 'diagnostic requires a fresh container/output'
    manifest=json.loads((ARCHIVE/'manifest.json').read_text())
    assert manifest['archive_sha256']==EXPECTED_ARCHIVE
    assert hashlib.sha256((ARCHIVE/'runtime.tar.gz').read_bytes()).hexdigest()==EXPECTED_ARCHIVE
    pins=[]
    for name in ('prepared-a-pins.json','frozen-d-pins.json'):
        pins.extend(json.loads((ARCHIVE.parent/name).read_text()))
    files={row['path']:row['sha256'] for row in manifest['files']}
    roots={'source':Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002'),
        'datasets':Path('/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002')}
    for row in pins:
        prefix,relative=row['path'].split('/',1)
        assert files[row['path']]==row['sha256'], 'archive owner differs from frozen producer'
        assert hashlib.sha256((roots[prefix]/relative).read_bytes()).hexdigest()==row['sha256'], 'live owner differs from frozen producer'
    (B/'producer-verification.json').write_text(json.dumps(dict(archive_sha256=EXPECTED_ARCHIVE,
        frozen_archive_and_live_pins_equal=True,files=pins),indent=2)+'\n')
    (B/'archive-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    original_probe=qualifier.CONTEXT_PROBE
    original_qualify=qualifier.qualify_context
    qualifier.CONTEXT_PROBE=instrument_probe(original_probe)
    (B/'instrumented-probe.py').write_text(qualifier.CONTEXT_PROBE)
    async def diagnostic_context(environment,**kwargs):
        try:
            return await original_qualify(environment,**kwargs)
        finally:
            collections=[]
            for name,remote,maximum in (
                ('phase-events.jsonl','/opt/ipfs-supervisor/state/source384-phase-events.jsonl',1024*1024),
                ('phase-summary.json','/opt/ipfs-supervisor/state/source384-phase-summary.json',32768),
            ):
                path=B/name
                try:
                    await asyncio.wait_for(environment.download_file(remote,path),30)
                    raw=_regular_bytes(path.resolve(strict=True),maximum)
                    collections.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                except BaseException as exc:
                    collections.append(dict(name=name,error_type=type(exc).__name__))
            (B/'diagnostic-collection.json').write_text(json.dumps(collections,indent=2)+'\n')
    qualifier.qualify_context=diagnostic_context
    start=time.monotonic()
    report=dict(schema='external-source384-docker-stage-diagnostic@1',diagnostic_only=True,
        production_qualification_claimed=False,benchmark_result=False,
        provider_calls=0,training_steps=0,official_verifier_executed=False,
        archive_sha256=EXPECTED_ARCHIVE,resource_profile='source384-5cpu-12gib@1',
        native_context_budget_seconds=90,probe_budget_seconds=270)
    try:
        raw=await asyncio.wait_for(deployment.qualify_original_container(task_dir=TASK,
            archive_dir=ARCHIVE,output=B/'docker-01',install_codex=False,
            resource_profile='source384-5cpu-12gib@1',source384_context=True),2300)
        (B/'raw-production-wrapper-return.json').write_text(json.dumps(raw,indent=2,sort_keys=True)+'\n')
        report['instrumented_call_returned']=True
    except BaseException as exc:
        traceback.print_exc()
        report.update(instrumented_call_returned=False,error_type=type(exc).__name__,error=str(exc)[:2048])
    finally:
        qualifier.CONTEXT_PROBE=original_probe
        qualifier.qualify_context=original_qualify
        report['seconds']=time.monotonic()-start
        (B/'diagnostic-result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


if __name__=='__main__':
    report=asyncio.run(run())
    raise SystemExit(0 if report['instrumented_call_returned'] else 1)
