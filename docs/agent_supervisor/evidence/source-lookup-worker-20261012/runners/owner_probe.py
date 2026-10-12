"""Actual UID1000 owner, native DuckDB/CAS and unchanged coding launcher."""
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path('/opt/ipfs-supervisor')
PRIVATE = ROOT / 'state'
CONTROL = ROOT / 'fixture-ipc'
sys.path[:0] = [str(ROOT / name) for name in ('source', 'datasets', 'kit')]
PROMPT = 'Read the admitted source files through the read-only source-lookup helper. Do not execute or edit them.'
FILES = {'module.py': ('# exact caf\u00e9 source \\"\t\r\n' * 180 + 'VALUE = 42\r\n').encode(),
         'data.bin': bytes(range(256)) * 4, 'empty.txt': b''}


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def write(path, value):
    Path(path).write_bytes(wire(value))


def git(root, *argv):
    return subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-C',str(root),*argv],stderr=subprocess.DEVNULL).decode().strip()


def capability():
    """Same authored native-owner lifecycle adapter as existing host fixtures."""
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        DEFAULT_QUACK_BETA_LIMITATIONS, ExtensionObservation, ParsedVersion,
        QuackCapabilityReport, QuackCapabilityStatus, default_compatibility_profile)
    profile = default_compatibility_profile()
    parts = tuple(map(int, duckdb.__version__.split('.')[:3]))
    return QuackCapabilityReport(status=QuackCapabilityStatus.COMPATIBLE, profile=profile,
        duckdb_importable=True, duckdb_version=duckdb.__version__,
        duckdb_version_parsed=ParsedVersion(*parts, raw=duckdb.__version__),
        platform_name='Linux', platform_machine='authored-fixture',
        extension=ExtensionObservation(name='quack',installed=True,loaded=True,
            install_path='/authored-lifecycle/quack.duckdb_extension',extension_version='authored'),
        extension_fingerprint='sha256:'+'ab'*32,
        observed_functions=('quack_serve','quack_query'),observed_surfaces=profile.required_surfaces,
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS)


def migrate(path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import install_control_plane_schema
    return install_control_plane_schema(path, application_version='0.0.45', tool_version='1.5.5',
                                        owner_id='owner:source-worker-qualification')


def main():
    assert os.getuid() == os.geteuid() == os.getgid() == 1000
    outcome = dict(schema='source-worker-docker-probe@1',complete=False,provider_calls=0,
        model_calls=0,solver_calls=0,native_provider_tokens=None,total_token_savings_measured=False,
        model_tool_choice_qualified=False,source_executed=False,automatic_terminal_binding_qualified=False,
        native_duckdb_real=True,quack_lifecycle_adapter='FakeQuackTransport + authored capability',
        original_cas_real=True,plain_prompt=True,semantic_repository_option_used=False)
    outcome['coding_reply_mode']='legacy'
    outcome['modern_sealed_semantic_route_qualified']=False
    outcome['legacy_route_reason']='Existing worker/router ordinary-completion requires semantic_repository; source-only fixture intentionally omits it.'
    server = session = process = None
    started = time.monotonic()
    try:
        assert all(not os.environ.get(key) for key in ('OPENAI_API_KEY','XAI_API_KEY','ANTHROPIC_API_KEY','HF_TOKEN'))
        from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as inventory
        from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_client as client
        from ipfs_accelerate_py.agent_supervisor.runtime.source_lookup_worker_broker import start_source_lookup_worker_broker
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport, build_server
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnerLiveness
        from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes, content_identity
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
        from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
        from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import preflight_committed_repository
        repository, workspace = ROOT/'repository', ROOT/'worktrees/task'
        assert not any(repository.iterdir())
        for name, data in FILES.items(): (repository/name).write_bytes(data)
        git(repository,'init','-b','main');git(repository,'config','core.autocrlf','false')
        git(repository,'add','-A');git(repository,'-c','user.name=Qualification','-c','user.email=qualification@example.invalid','commit','-qm','authored inert source')
        head,tree=git(repository,'rev-parse','HEAD'),git(repository,'rev-parse','HEAD^{tree}')
        plan=preflight_committed_repository(repository,repository_id='fixture:source-worker',expected_commit=head,expected_tree=tree)
        request=inventory.SourceInventoryRequest('source-worker-task','fixture:source-worker',head,tree,plan.population_cid)
        limits=inventory.SourceInventoryLimits(frame_bytes=4096)
        cas=ImmutableCAS(PRIVATE/'cas')
        retained=inventory.retain_source_inventory(repository,cas=cas,request=request,limits=limits)
        git(repository,'worktree','add','--detach','-q',str(workspace))
        body=dict(task_id=request.task_id,title='Read authored source',read_paths=sorted(name.encode().hex() for name in FILES))
        task_cid=content_identity({'authored_task':body})
        server=build_server(database_path=PRIVATE/'control.duckdb',state_dir=PRIVATE/'owner',
            repository_id=request.repository_id,store_id='source-worker-docker',transport=FakeQuackTransport(),
            capability_probe=capability,migrate=migrate,connection_factory=open_duckdb_connection,
            owner_liveness_probe=lambda _birth:OwnerLiveness.DEAD,typed_command_socket_path=PRIVATE/'native.sock')
        server.start()
        connection,gateway=server._connection,server._command_gateway
        connection.execute('INSERT INTO goals (goal_cid,goal_alias,objective_id,parent_goal_cid,ordinal,title,status,created_at,updated_at,revision,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            ['goal:source-worker','SOURCE-GOAL','objective:source-worker','',1,'Source read','open','','',1,'{}'])
        connection.execute('INSERT INTO tasks (task_cid,task_alias,goal_cid,plan_cid,objective_id,ordinal,status,revision,priority,created_at,updated_at,identity_json,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            [task_cid,'SOURCE-TASK','goal:source-worker','','objective:source-worker',1,'ready',3,'P0','','','{}',canonical_json_bytes(body).decode()])
        server.bind_current_source_lookup(repository=workspace,cas=cas,expected_inventory_cid=retained.inventory_cid,
            request=request,limits=limits,task_cid=task_cid,task_revision=3,expected_task_body_cid=content_identity(body),
            context_cid=cid_for_structured({'source_worker_context':task_cid}),
            allowed_paths=tuple(sorted(name.encode().hex() for name in FILES)),max_response_bytes=4096,max_slice_bytes=4096)
        session=start_source_lookup_worker_broker(server=server,canonical_repository=repository,workspace=workspace,
            boundary=ROOT/'container-boundary.json',prompt=PROMPT,timeout_seconds=300)
        descriptor=json.loads(session.descriptor_path.read_bytes())
        assert descriptor['source_refs']==sorted(descriptor['source_refs'],key=lambda ref:ref['raw_path_hex'])
        write(PRIVATE/'expected-source.json', {name:dict(base64=base64.b64encode(data).decode(),cid=cid_for_bytes(data)) for name,data in FILES.items()})
        write(CONTROL/'private-paths.json',[str(PRIVATE/'control.duckdb'),str(PRIVATE/'cas'),str(PRIVATE/'native.sock')])
        (CONTROL/'private-paths.json').chmod(0o440)
        argv=[str(ROOT/'bin/owner-worker'),'--provider','codex_cli','--model','source-ipc-authored',
              '--timeout','300','--max-output-tokens','128','--purpose','coding',*session.extra_argv]
        write(PRIVATE/'owner-launch.json',dict(argv=argv,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
            prompt_bytes=len(PROMPT.encode()),native_grant_id=session._grant.grant_id,source_inventory_cid=retained.inventory_cid))
        stderr=(PRIVATE/'owner-entry.stderr').open('w')
        process=subprocess.Popen(argv,cwd=workspace,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True)
        session.note_spawned(process)
        process.stdin.write(PROMPT);process.stdin.close()
        negatives,lines=[],[]
        for line in process.stdout:
            lines.append(line)
            try:event=json.loads(line)
            except ValueError:continue
            if type(event)is not dict or 'fixture_event' not in event:continue
            name=event['fixture_event']
            if name=='worker_ready':
                ref=next(ref for ref in descriptor['source_refs'] if bytes.fromhex(ref['raw_path_hex'])==b'module.py')
                payload=dict(schema=client.SOURCE_REQUEST_SCHEMA,binding_cid=descriptor['source_binding_cid'],
                    raw_path_hex=ref['raw_path_hex'],source_cid=ref['source_cid'],offset=0,max_bytes=4096,continuation=None)
                req=dict(schema=client.REQUEST_SCHEMA,session_id=descriptor['session_id'],request_id=uuid.uuid4().hex,payload=payload)
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as channel:
                    channel.settimeout(30);channel.connect(descriptor['socket_path']);client.send_frame(channel,req)
                    refused=client.receive_frame(channel)
                assert refused['reason_code']=='peer_refused' and refused['result']is None
                negatives.append(dict(case='wrong_uid1000',reason_code=refused['reason_code']))
                result=subprocess.run(['/usr/bin/sudo','-n','-u','benchmarkworker','--',str(ROOT/'bin/source-probe-client'),
                    client.DESCRIPTOR_FLAG,str(session.descriptor_path),client.DIGEST_FLAG,session.descriptor_sha256],
                    cwd=workspace,input=wire(payload)+b'\n',capture_output=True,timeout=30)
                write(PRIVATE/'unrelated-peer-diagnostic.json',dict(returncode=result.returncode,
                    stdout=result.stdout.decode(),stderr=result.stderr.decode(),
                    probe_scope='Actual unrelated UID1001 with no-new-privileges, executing exact protected helper'))
                refused=json.loads(result.stdout)
                assert result.returncode==2 and refused['reason_code']=='peer_refused' and refused['result']is None
                negatives.append(dict(case='unrelated_uid1001',reason_code=refused['reason_code']))
            elif name in ('mutate_canonical','restore_canonical','mutate_allocated','restore_allocated'):
                action,scope=name.split('_')
                path=(repository if scope=='canonical' else workspace)/'module.py'
                path.write_bytes(FILES['module.py']+(b'\n# authored stale overlay\n' if action=='mutate' else b''))
            else:raise AssertionError('unknown qualification event')
            (CONTROL/(name+'.ack')).write_text('ack')
        code=process.wait(timeout=30);stderr.close()
        (PRIVATE/'owner-entry.stdout').write_text(''.join(lines))
        assert code==0,'owner entry failed; diagnostic retained'
        worker=json.loads((CONTROL/'worker-result.json').read_bytes())
        router=json.loads((CONTROL/'router-observation.json').read_bytes())
        assert worker['uid']==1001 and len(worker['private_read_denials'])==3
        assert router['descriptor_environment_restored']is True and len(router['authored_generate_calls'])==1
        per_file,counts={},[]
        for name,expected in FILES.items():
            pages=worker['pages'][name]
            chunks=[(page['result']['content'].encode() if page['result']['encoding']=='utf-8'
                     else base64.b64decode(page['result']['content'],validate=True)) for page in pages]
            assert b''.join(chunks)==expected
            assert pages[0]['result']['offset']==0 and pages[-1]['result']['end_offset']==len(expected)
            for page in pages:
                assert page['response_bytes']==len(wire(page))+1<=32768
                assert page['response_wire_bytes']==page['response_bytes']+3
                assert 'binding' not in page['result'] and 'caller_grant_id' not in page['result']
                assert str(PRIVATE/'control.duckdb').encode()not in wire(page)
                counts.append(page['response_bytes'])
            per_file[name]=dict(source_bytes=len(expected),source_cid=cid_for_bytes(expected),page_count=len(pages),
                response_bytes=sum(page['response_bytes'] for page in pages),source_sha256=hashlib.sha256(expected).hexdigest())
        grant_id=session._grant.grant_id
        session.close()
        audit=json.loads(session.audit_path.read_bytes())
        assert audit['service_stopped'] and audit['accounting_complete'] and audit['cleanup_complete']
        assert grant_id in gateway._revoked_grants
        assert not session.socket_path.exists() and not session.descriptor_path.exists()
        assert audit['backend_accounting_scope']=='validated-source-success-envelopes'
        assert all((repository/name).read_bytes()==data==(workspace/name).read_bytes() for name,data in FILES.items())
        write(Path('/results/private-broker-audit.json'),audit)
        write(Path('/results/private-router-observation.json'),router)
        outcome.update(complete=True,owner_uid=1000,worker_uid=1001,per_file=per_file,
            exact_reconstructed_bytes=sum(map(len,FILES.values())),primary_page_count=len(counts),
            primary_response_bytes=sum(counts),largest_response_bytes=max(counts),
            private_read_denial_count=3,negative_checks=negatives+worker['negatives'],
            broker_requests=audit['accepted_requests'],broker_response_bytes=audit['response_bytes'],
            backend_success_response_bytes=audit['backend_response_bytes'],backend_success_wire_bytes=audit['backend_response_wire_bytes'],
            backend_cost_scope=audit['backend_accounting_scope'],handshake_error_byte_costs=None,
            stable_native_grant=True,original_grant_revoked=True,broker_stopped=True,source_restored=True,
            descriptor_and_socket_removed=True,authored_router_calls=1,source_refs_count=len(descriptor['source_refs']))
        outcome['distinct_helper_processes']=len({item['pid']for item in worker['helper_processes']})
        outcome['actual_helper_uid1001_verified']=all(item['uid']==1001 for item in worker['helper_processes'])
    except BaseException as error:
        outcome.update(error_type=type(error).__name__,error=str(error)[:500])
        (PRIVATE/'probe-failure.txt').write_text(traceback.format_exc())
    finally:
        if process is not None and process.poll()is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if session is not None:
            try:session.close()
            except Exception as error:outcome['cleanup_error_type']=type(error).__name__;outcome['complete']=False
        if server is not None:
            try:server.stop()
            except Exception as error:outcome['owner_stop_error_type']=type(error).__name__;outcome['complete']=False
        outcome['seconds']=time.monotonic()-started
        write(Path('/results/probe-qualification.json'),outcome)
        print(json.dumps({key:outcome.get(key)for key in('complete','error_type','seconds','primary_page_count')}),flush=True)
    return 0 if outcome['complete']else 1


if __name__=='__main__':
    raise SystemExit(main())
