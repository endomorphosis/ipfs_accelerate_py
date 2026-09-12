import hashlib,json,os
from pathlib import Path
import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as G


def inputs(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    home=tmp_path/'grok-home';home.mkdir()
    prompt=tmp_path/'prompt';prompt.write_text('inert test')
    binary=tmp_path/'inert';binary.write_text('#!/bin/sh\nexit 0\n');binary.chmod(0o700)
    config=tmp_path/'config';config.mkdir()
    env={'XAI_API_KEY':'test-primary-credential-not-real','OPENAI_API_KEY':'test-peer-secret-not-real','LD_PRELOAD':'poison','PYTHONPATH':'poison'}
    return dict(grok_command=[str(binary),'--tools',G._SEALED_GROK_TOOLS,'--disallowed-tools',G._SEALED_GROK_DISALLOWED_TOOLS,'--no-subagents','--disable-web-search'],grok_bin=binary,workspace=workspace,prompt_path=prompt,grok_home=home,base_env={'PATH':'/usr/bin:/bin'},child_env=env,denied_paths=(),mask_root=tmp_path/'provider-masks',docker_config=config,container_name='ipfs-accelerate-grok-123-'+('a'*32),cidfile=tmp_path/'cid',docker_bin='/usr/bin/docker',isolation_image=G._CODEX_TASK_TOOLCHAIN_IMAGE_ID)


def test_docker_profile_adds_only_exact_terminal_and_preserves_host_lists(tmp_path):
    data=inputs(tmp_path)
    command=G._docker_grok_command(**data,task_execution=True)
    assert command[command.index('--tools')+1]==G._SEALED_GROK_TOOLS+',run_terminal_cmd'
    denied=command[command.index('--disallowed-tools')+1].split(',')
    assert set(denied)==set(G._SEALED_GROK_DISALLOWED_TOOLS.split(','))-{'run_terminal_cmd','run_terminal_command'}
    assert {'web_fetch','web_search','task','Agent','call_mcp_tool'}<=set(denied)
    assert 'run_terminal_cmd' in G._SEALED_GROK_DISALLOWED_TOOLS
    assert 'run_terminal_cmd' not in G._SEALED_GROK_TOOLS
    assert '--runtime=runc' in command and '--network=bridge' in command
    assert command[command.index(G._CODEX_TASK_TOOLCHAIN_IMAGE_ID)+1:][:2]==['-I',str(G._GROK_CONTAINER_BOOTSTRAP)]


def test_container_launcher_is_nonsecret_local_and_maps_python_environment(tmp_path):
    data=inputs(tmp_path);command=G._docker_grok_command(**data,task_execution=True)
    wrapper=(tmp_path/'task-launchers/provider-command-env').read_text()
    bootstrap=(tmp_path/'task-launchers/task-bootstrap.py').read_text()
    assert wrapper.startswith('#!/opt/ipfs-task-tools/bin/python -I\n')
    assert '/proc/' not in wrapper
    for secret in ('test-primary-credential-not-real','test-peer-secret-not-real','poison'):
        assert secret not in wrapper+bootstrap+' '.join(command)
    assert 'OPENAI_API_KEY' not in data['child_env']
    assert data['child_env']['XAI_API_KEY']=='test-primary-credential-not-real'
    assert data['child_env']['GROK_CODEX_MCPS_ENABLED']=='0'
    assert data['child_env']['PATH']=='/opt/ipfs-task-tools/bin:/usr/bin:/bin'
    assert data['child_env'][G.PROVIDER_COMMAND_ENV_WRAPPER_ENV]==str(G._GROK_CONTAINER_COMMAND_WRAPPER)
    assert hashlib.sha256(wrapper.encode()).hexdigest() in bootstrap
    assert 'os.environ.clear()' in bootstrap
    assert 'readonly' in ' '.join(command)


def test_task_profile_rejects_other_image_before_launcher_creation(tmp_path):
    data=inputs(tmp_path);data['isolation_image']='sha256:'+('b'*64)
    with pytest.raises(ValueError,match='pinned task-toolchain'):
        G._docker_grok_command(**data,task_execution=True)
    assert not (tmp_path/'task-launchers').exists()


def test_task_profile_rejects_unmounted_declared_roots(tmp_path):
    data=inputs(tmp_path);outside=tmp_path/'outside';outside.mkdir()
    data['base_env']['IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT']=str(outside)
    with pytest.raises((ValueError,G.ProviderCommandEnvironmentError)):
        G._docker_grok_command(**data,task_execution=True)


def test_old_file_only_docker_builder_does_not_gain_terminal(tmp_path):
    data=inputs(tmp_path);command=G._docker_grok_command(**data)
    assert command[command.index('--tools')+1]==G._SEALED_GROK_TOOLS
    assert command[command.index('--disallowed-tools')+1]==G._SEALED_GROK_DISALLOWED_TOOLS
    assert not (tmp_path/'task-launchers').exists()


def tex_profile(tmp_path):
    root=tmp_path/'tinytex';(root/'bin/aarch64-linux').mkdir(parents=True);(root/'texmf-dist').mkdir()
    binary=root/'bin/aarch64-linux/pdflatex';binary.write_text('inert fixture');binary.chmod(0o700)
    wrapper=tmp_path/'tex-wrapper';wrapper.write_text('#!/bin/sh\nexit 0\n');wrapper.chmod(0o700)
    members={}
    for item in root.rglob('*'):
        members[str(item.relative_to(root))]=({'kind':'directory'} if item.is_dir() else {'kind':'file','bytes':item.stat().st_size,'sha256':hashlib.sha256(item.read_bytes()).hexdigest()})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'schema':'grok-tex-tree/v1','root':str(root),'members':members}))
    profile={'schema':'grok-docker-tex-toolchain/v1','root':str(root),'wrapper':str(wrapper),'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest()}
    return profile,root,manifest


@pytest.mark.parametrize('mutation',['file','extra','escape','manifest'])
def test_tex_profile_rejects_changed_bytes_members_and_escape(tmp_path,mutation):
    profile,root,manifest=tex_profile(tmp_path)
    assert G._grok_task_tex_toolchain({G._GROK_TEX_TOOLCHAIN_ENV:json.dumps(profile)})['root']==str(root)
    if mutation=='file': (root/'bin/aarch64-linux/pdflatex').write_text('changed')
    elif mutation=='extra': (root/'new').write_text('extra')
    elif mutation=='escape': (root/'bad-link').symlink_to('/usr/bin/python3')
    else: manifest.write_text('{}')
    with pytest.raises(ValueError):
        G._grok_task_tex_toolchain({G._GROK_TEX_TOOLCHAIN_ENV:json.dumps(profile)})


def test_effective_container_contract_separates_parent_formal_identity(tmp_path):
    data=inputs(tmp_path);data['child_env'][G.FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV]='parent-formal-proof'
    effective,approved=G._grok_task_container_environment(base_env=data['base_env'],child_env=data['child_env'],workspace=data['workspace'],grok_home=data['grok_home'])
    assert G.FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV not in effective
    assert G.FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV not in approved
    assert effective[G._GROK_PARENT_FORMAL_TOOLCHAIN_SHA256_ENV]=='parent-formal-proof'
    assert effective[G.PROVIDER_COMMAND_ENV_DIGEST_ENV]==G.provider_command_environment_sha256(approved)


def codex_inputs(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    auth=tmp_path/'auth.json';auth.write_text('{}\n');auth.chmod(0o600)
    binary=tmp_path/'codex';binary.write_text('#!/bin/sh\nexit 0\n');binary.chmod(0o700)
    config=tmp_path/'config';config.mkdir()
    return dict(codex_command=[str(binary),'exec','--ignore-user-config','--ignore-rules','--ephemeral','-s','workspace-write','-C',str(workspace),'-m','gpt-5.6-terra','-c','model_reasoning_effort="high"','-'],workspace=workspace,source_auth=auth,child_env=G._codex_task_container_environment(),docker_config=config,container_name='ipfs-accelerate-codex-1-'+('b'*32),cidfile=tmp_path/'cid',docker_bin='/usr/bin/docker',isolation_image=G._CODEX_TASK_TOOLCHAIN_IMAGE_ID)


def test_codex_optional_tex_uses_same_manifest_and_only_exact_ro_mounts(tmp_path):
    data=codex_inputs(tmp_path);profile,root,manifest=tex_profile(tmp_path)
    data['base_env']={G._GROK_TEX_TOOLCHAIN_ENV:json.dumps(profile),'XAI_API_KEY':'fake-secret-not-admitted','OPENAI_API_KEY':'fake-peer-not-admitted'}
    command=G._docker_codex_fallback_command(**data)
    mounts=[command[i+1] for i,x in enumerate(command[:-1]) if x=='--mount']
    for exact in (profile['root'],profile['wrapper']):
        assert any('src='+exact+',' in x and 'readonly' in x for x in mounts)
    assert not any('src='+str(tmp_path)+',' in x for x in mounts)
    assert data['child_env']['PATH'].startswith(str(root)+'/bin/aarch64-linux:')
    digest=G._grok_task_tex_toolchain(data['base_env'])['sha256']
    assert data['child_env'][G._GROK_TEX_TOOLCHAIN_SHA256_ENV]==digest
    assert data['child_env']['TEXMFVAR'].startswith('/tmp/')
    assert 'fake-secret' not in ' '.join(command) and 'fake-peer' not in ' '.join(command)
    assert 'XAI_API_KEY' not in data['child_env'] and 'OPENAI_API_KEY' not in data['child_env']
    assert 'Codex task formatting tools unavailable' in ' '.join(command)
    assert data['codex_command'][data['codex_command'].index('-s')+1]=='workspace-write'
    assert 'gpt-5.6-terra' in command and 'model_reasoning_effort="high"' in command


def test_codex_without_tex_keeps_existing_sealed_route(tmp_path):
    data=codex_inputs(tmp_path);original=dict(data['child_env'])
    command=G._docker_codex_fallback_command(**data)
    assert data['child_env']==original
    assert not any('TEXMF' in part for part in command)
    assert command[command.index(G._CODEX_TASK_TOOLCHAIN_IMAGE_ID)+1]=='-i'


def test_codex_tex_rejects_changed_manifest_content(tmp_path):
    data=codex_inputs(tmp_path);profile,root,manifest=tex_profile(tmp_path)
    (root/'bin/aarch64-linux/pdflatex').write_text('changed')
    with pytest.raises(ValueError,match='exact manifest'):
        G._docker_codex_fallback_command(**data,base_env={G._GROK_TEX_TOOLCHAIN_ENV:json.dumps(profile)})
    assert not data['cidfile'].exists()


def test_codex_tex_rejects_workspace_overlap(tmp_path):
    data=codex_inputs(tmp_path);profile,root,manifest=tex_profile(data['workspace'])
    with pytest.raises(ValueError,match='overlaps'):
        G._docker_codex_fallback_command(**data,base_env={G._GROK_TEX_TOOLCHAIN_ENV:json.dumps(profile)})
    assert not data['cidfile'].exists()


def research_profile(tmp_path):
    root=tmp_path/'research';(root/'python').mkdir(parents=True);(root/'bin').mkdir()
    for name in G._RESEARCH_REQUIRED_COMMANDS:
        p=root/'bin'/name;p.write_text('#!/bin/sh\nexit 0\n');p.chmod(0o700)
    for name in G._RESEARCH_REQUIRED_MODULES:
        p=root/'python'/name;p.mkdir();(p/'__init__.py').write_text('FIXTURE = True\n')
    members={str(p.relative_to(root)):({'kind':'directory'} if p.is_dir() else {'kind':'file','bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}) for p in root.rglob('*')}
    manifest=tmp_path/'research-manifest.json';manifest.write_text(json.dumps({'schema':'research-runtime-tree/v1','root':str(root),'members':members}))
    profile={'schema':'docker-research-toolchain/v1','root':str(root),'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    return profile,root,manifest


@pytest.mark.parametrize('mutation',['file','extra','escape','manifest','world_writable'])
def test_research_manifest_rejects_unadmitted_content_and_paths(tmp_path,mutation):
    profile,root,manifest=research_profile(tmp_path);env={G._RESEARCH_TOOLCHAIN_ENV:json.dumps(profile)}
    assert G._research_task_toolchain(env)['root']==str(root)
    if mutation=='file':(root/'python/torch/__init__.py').write_text('changed')
    elif mutation=='extra':(root/'extra').write_text('extra')
    elif mutation=='escape':(root/'escape').symlink_to('/usr/bin/python3')
    elif mutation=='manifest':manifest.write_text('{}')
    else:(root/'python/torch/__init__.py').chmod(0o666)
    with pytest.raises(ValueError):G._research_task_toolchain(env)


@pytest.mark.parametrize('provider',['grok','codex'])
def test_research_profile_maps_only_exact_ro_tree_and_sealed_preflight(tmp_path,provider):
    profile,root,manifest=research_profile(tmp_path)
    env={G._RESEARCH_TOOLCHAIN_ENV:json.dumps(profile),'OPENAI_API_KEY':'not-admitted','XAI_API_KEY':'not-admitted'}
    if provider=='grok':
        data=inputs(tmp_path);data['base_env']=env;command=G._docker_grok_command(**data,task_execution=True)
        bootstrap=(tmp_path/'task-launchers/task-bootstrap.py').read_text()
        wrapper=(tmp_path/'task-launchers/provider-command-env').read_text()
        assert 'importlib' in bootstrap and str(root/'python') in wrapper
    else:
        data=codex_inputs(tmp_path);command=G._docker_codex_fallback_command(**data,base_env=env)
        assert 'importlib' in ' '.join(command)
    mounts=[command[i+1] for i,x in enumerate(command[:-1]) if x=='--mount']
    assert any('src='+str(root)+',' in v and 'readonly' in v for v in mounts)
    assert not any('src='+str(tmp_path)+',' in v for v in mounts)
    assert data['child_env']['PATH'].startswith(str(root/'bin')+':')
    assert data['child_env']['PYTHONPATH'].startswith(str(root/'python')+':')
    assert data['child_env'][G._RESEARCH_TOOLCHAIN_SHA256_ENV]==G._research_task_toolchain(env)['sha256']
    assert 'OPENAI_API_KEY' not in data['child_env']
    assert 'not-admitted' not in ' '.join(command)


@pytest.mark.parametrize('provider',['grok','codex'])
def test_research_profile_rejects_writable_workspace_overlap(tmp_path,provider):
    data=inputs(tmp_path) if provider=='grok' else codex_inputs(tmp_path)
    profile,_,_=research_profile(data['workspace']);env={G._RESEARCH_TOOLCHAIN_ENV:json.dumps(profile)}
    with pytest.raises(ValueError,match='overlaps'):
        if provider=='grok':G._docker_grok_command(**{**data,'base_env':env},task_execution=True)
        else:G._docker_codex_fallback_command(**data,base_env=env)
    assert not data['cidfile'].exists()


def test_research_preflight_uses_only_sealed_paths_under_isolated_python(tmp_path):
    import subprocess,sys
    admitted=tmp_path/'admitted';admitted.mkdir();(admitted/'qualification_probe.py').write_text('VALUE = "admitted"\n')
    poison=tmp_path/'poison';poison.mkdir();(poison/'qualification_probe.py').write_text('raise RuntimeError("workspace poisoning")\n');(poison/'sitecustomize.py').write_text('raise RuntimeError("ambient poisoning")\n')
    code=G._research_module_preflight_source({'PYTHONPATH':str(admitted)},('qualification_probe',))+'assert sys.modules["qualification_probe"].VALUE == "admitted"\n'
    r=subprocess.run([sys.executable,'-I','-c',code],cwd=poison,env={**os.environ,'PYTHONPATH':str(poison)},capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    (admitted/'qualification_probe.py').unlink()
    r=subprocess.run([sys.executable,'-I','-c',code],cwd=poison,env={**os.environ,'PYTHONPATH':str(poison)},capture_output=True,text=True)
    assert r.returncode!=0 and 'ModuleNotFoundError' in r.stderr
