"""Pure syntax/format and real private-file refusal controls; no SDK grant.

The runtime is deliberately loaded under a negative-only alias. Such an alias
must refuse construction/binding, so none of these cases can authorize work.
Actual native topology, Popen behavior and successful host composition require
the separately authored genuine native fixture, never mocks or these controls.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import subprocess

import pytest

HERE = Path(__file__).absolute().parent
SOURCE = HERE/'proposed/ipfs_accelerate_py/agent_supervisor/runtime/native_admin_execution.py'


@pytest.fixture(scope='module')
def negative_runtime():
    spec = importlib.util.spec_from_file_location('negative_only_native_admin_alias', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tmp_path(request):
    # Avoid pytest tmp_path_factory and its persistent *_current symlinks.
    parent = Path(os.environ['TMPDIR'])/'cases'
    parent.mkdir(mode=0o700,exist_ok=True)
    path = parent/hashlib.sha256(request.node.nodeid.encode()).hexdigest()
    path.mkdir(mode=0o700,exist_ok=False)
    assert not parent.is_symlink() and not path.is_symlink()
    return path


def test_refusal_escapes_all_legacy_exception_families(negative_runtime):
    error = negative_runtime.NativeAdminRefusal('control')
    assert isinstance(error, BaseException)
    assert not isinstance(error, (Exception, OSError, ValueError, subprocess.SubprocessError))


def test_unbound_discriminator_is_false(negative_runtime):
    assert negative_runtime.native_admin_execution_is_bound() is False


def test_unbound_dispatch_does_not_evaluate_command_or_kwargs(negative_runtime):
    assert negative_runtime.run_bound_native_admin_if_present(None, shell=True) is None


def test_alias_constructor_refuses_before_project_import(negative_runtime):
    with pytest.raises(negative_runtime.NativeAdminRefusal, match='canonical_runtime'):
        negative_runtime.NativeAdminExecution(None,phase_lease=None,source_sha256={},
            allowed_roots=[],absolute_deadline=1)


def test_alias_binding_refuses(negative_runtime):
    with pytest.raises(negative_runtime.NativeAdminRefusal, match='canonical_runtime'):
        with negative_runtime.bind_native_admin_execution(None):
            pytest.fail('negative-only alias must never enter a productive scope')


@pytest.mark.parametrize('method', ['run','close','observation','release'])
def test_uninitialized_exact_class_cannot_act(negative_runtime,method):
    # Negative metadata forgery only: there is no owner/lease/Popen or grant.
    execution = object.__new__(negative_runtime.NativeAdminExecution)
    with pytest.raises(negative_runtime.NativeAdminRefusal, match='constructed_executor'):
        if method == 'run':
            execution.run(['git','rev-parse','--git-common-dir'])
        elif method == 'release':
            execution.release(artifacts_durable=True)
        else:
            getattr(execution,method)()


@pytest.mark.parametrize('copier',[copy.copy,copy.deepcopy,pickle.dumps])
def test_objects_reject_copy_and_serialization(negative_runtime,copier):
    with pytest.raises(negative_runtime.NativeAdminRefusal):
        copier(negative_runtime._Nonportable())


@pytest.mark.parametrize('function',['run_bound_native_admin_if_present','NativeAdminExecution.run'])
def test_unknown_kwargs_reach_a_bound_refusal_body(negative_runtime,function):
    target = negative_runtime
    for name in function.split('.'):
        target = getattr(target,name)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(target).parameters.values())


@pytest.mark.parametrize('argv',[
    ['git','rev-parse','--git-common-dir'],
    ['git','rev-parse','--verify','--quiet','main'],
    ['git','rev-parse','--verify','--quiet','refs/heads/missing-owned-fixture-ref'],
    ['git','-c','core.quotepath=false','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null',
     'rev-parse','--verify','HEAD^{commit}'],
    ['git','-c','core.quotepath=false','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null',
     'status','--porcelain=v1','-z','--untracked-files=all'],
])
def test_pure_original_readonly_argv_projection(negative_runtime,argv):
    projected = negative_runtime._git_arguments(argv)
    assert projected == ['/usr/bin/git',*argv[1:]]
    assert argv[0] == 'git'


@pytest.mark.parametrize('argv',[
    ['/tmp/git','rev-parse','--git-common-dir'],
    ['docker','info'],
    ['sh','-c','git rev-parse HEAD'],
    ['git','fetch','origin'],
    ['git','commit','-m','unexpected'],
    ['git','-c','credential.helper=unexpected','rev-parse','HEAD'],
    ['git','rev-parse','--verify','--quiet','--unsafe-option'],
    ['git','rev-parse','--verify','--quiet','branch\0suffix'],
])
def test_pure_argv_profile_refuses_other_producers(negative_runtime,argv):
    with pytest.raises(negative_runtime.NativeAdminRefusal):
        negative_runtime._git_arguments(argv)


def test_text_projection_preserves_universal_newlines(negative_runtime):
    assert negative_runtime._decode_output(b'a\r\nb\rc\n','utf-8','strict') == 'a\nb\nc\n'


def test_text_projection_preserves_decode_failure(negative_runtime):
    with pytest.raises(UnicodeDecodeError):
        negative_runtime._decode_output(b'\xff','utf-8','strict')


def test_stable_group_writable_owned_source_is_supported(negative_runtime,tmp_path):
    path = tmp_path/'source.py'
    path.write_bytes(b'owned source\n')
    try:
        os.chmod(path,0o664)
        negative_runtime._verify_pin(path,hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        os.chmod(path,0o600)


@pytest.mark.parametrize('kind',['symlink','hardlink','world_writable','digest_mismatch'])
def test_private_file_aliases_and_changes_refuse(negative_runtime,kind,tmp_path):
    original = tmp_path/'source.py'
    original.write_bytes(b'owned source\n')
    path,alias = original,tmp_path/'alias.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        if kind == 'symlink':
            alias = path.with_name('alias.py')
            alias.symlink_to(path)
            path = alias
        elif kind == 'hardlink':
            os.link(path,path.with_name('alias.py'))
        elif kind == 'world_writable':
            os.chmod(path,0o666)
        else:
            digest = '0'*64
        with pytest.raises((negative_runtime.NativeAdminRefusal,OSError)):
            negative_runtime._verify_pin(path,digest)
    finally:
        if os.path.lexists(alias):
            alias.unlink()
        os.chmod(original,0o600)


def test_import_dependencies_remain_standard_library():
    tree = ast.parse(SOURCE.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    for node in imports:
        names = [x.name.split('.')[0] for x in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]]
        assert all(name in {'contextlib','contextvars','hashlib','importlib','io','json','math','os','pathlib',
            're','selectors','signal','stat','subprocess','sys','threading','time','weakref'} for name in names)
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            pytest.fail('import time must not execute project/native/process calls')


def test_public_exports_name_only_the_narrow_execution_contract():
    tree = ast.parse(SOURCE.read_text())
    exports = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id == '__all__' for target in node.targets))
    assert set(exports) == {'NativeAdminExecution','NativeAdminRefusal','NativeAdminCustodyPending',
        'native_admin_execution_is_bound','run_bound_native_admin_if_present',
        'bind_native_admin_execution','acknowledge_stopped_native_admin_custody'}
