"""Cold/source controls only; no product import, native grant or OS child.

Isolated extracted source functions receive an import shim that can only deny
or report unbound. It is NOT a canonical runtime module/native executor and
does not establish a positive host/daemon/connection/lease qualification.
No shared module, subprocess global or actual producer is monkeypatched.
Run only in a later finite genuinely admitted source-control worker.
"""
import __future__
import ast
import builtins
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import stat
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent


@pytest.fixture
def tmp_path(request):
    """Owned regular directory; never pytest's framework symlink fixture."""
    attempt = Path(os.environ['TMPDIR'])
    assert attempt.is_absolute() and not any(item.is_symlink() for item in (attempt, *attempt.parents))
    metadata = attempt.lstat()
    assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) == 0o700
    cases = attempt/'cases'
    cases.mkdir(mode=0o700, exist_ok=True)
    metadata = cases.lstat()
    assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) == 0o700
    path = cases/hashlib.sha256(request.node.nodeid.encode()).hexdigest()
    path.mkdir(mode=0o700, exist_ok=False)
    return path


@pytest.fixture(scope='module')
def assembly():
    spec = importlib.util.spec_from_file_location('reviewed_bootstrap_source_builder_only', HERE/'build_patch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    originals, bases, proposed = module.assemble()
    return module, originals, bases, proposed


def hook_index(node):
    return next(index for index, item in enumerate(node.body)
                if isinstance(item, ast.ImportFrom) and item.module.endswith('.native_admin_execution'))


@pytest.mark.parametrize('site_index', range(4))
def test_exact_unbound_try_call_and_error_handlers(assembly, site_index):
    module, _originals, bases, proposed = assembly
    relative, name, _source, expected = module.SITES[site_index]
    old = module.definition(bases[relative], name)
    new = module.definition(proposed[relative], name)
    index = hook_index(new)
    branch = new.body[index+1]
    assert isinstance(branch, ast.If)
    assert ast.dump(branch.orelse[0], include_attributes=False) == ast.dump(old.body[index], include_attributes=False)
    assert module.ast_sha(old) == expected
    assert len(branch.body) == 1 and isinstance(branch.body[0], ast.Assign)
    assert branch.body[0].value.func.id == 'run_bound_native_admin_if_present'
    assert all(not isinstance(item, ast.Try) for item in branch.body)
    original_call = old.body[index].body[0].value
    restored_call = copy.deepcopy(branch.body[0].value)
    restored_call.func = copy.deepcopy(original_call.func)
    assert ast.dump(restored_call, include_attributes=False) == ast.dump(original_call, include_attributes=False)


@pytest.mark.parametrize('site_index', range(4))
def test_exact_every_other_function_statement_and_validator(assembly, site_index):
    module, _originals, bases, proposed = assembly
    relative, name, _source, _expected = module.SITES[site_index]
    old = module.definition(bases[relative], name)
    new = module.definition(proposed[relative], name)
    index = hook_index(new)
    new.body[index:index+2] = new.body[index+1].orelse
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)


def cold_function(module, source, name, *, bound, failure):
    node = copy.deepcopy(module.definition(source, name))
    imports = []
    launches = []

    def dispatch(*_args, **_kwargs):
        raise failure

    shim = SimpleNamespace(native_admin_execution_is_bound=lambda: bound,
                           run_bound_native_admin_if_present=dispatch)

    def local_import(qualified, *_args, **_kwargs):
        imports.append(qualified)
        if qualified == module.RUNTIME_MODULE:
            return shim
        raise AssertionError('unexpected_import_in_cold_source_probe')

    def forbidden_launch(*_args, **_kwargs):
        launches.append(True)
        raise AssertionError('legacy_process_reached_after_bound_refusal')

    namespace = dict(Path=Path, Sequence=tuple, os=os,
        _AGENT_CONTROL_PLANE_MAX_MANIFEST_BYTES=1048576,
        subprocess=SimpleNamespace(run=forbidden_launch, DEVNULL=subprocess.DEVNULL,
            PIPE=subprocess.PIPE, SubprocessError=subprocess.SubprocessError),
        __builtins__={**vars(builtins), '__import__': local_import})
    unit = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), '<cold-denial-only-source-probe>', 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[node.name], imports, launches


class ColdBoundRefusal(BaseException):
    """Negative-only marker; no executor, object association or native grant."""


@pytest.mark.parametrize('site_index', range(4))
@pytest.mark.parametrize('failure_kind', ('custody', 'os_error', 'subprocess_error'))
def test_bound_failure_cannot_fall_back_or_enter_legacy_catches(assembly, site_index, failure_kind):
    module, _originals, _bases, proposed = assembly
    relative, name, _source, _expected = module.SITES[site_index]
    failure = {'custody': ColdBoundRefusal('cold_bound_denial'),
               'os_error': OSError('cold_bound_os_error'),
               'subprocess_error': subprocess.SubprocessError('cold_bound_process_error')}[failure_kind]
    function, imports, launches = cold_function(module, proposed[relative], name, bound=True, failure=failure)
    if name == '_agent_git_output':
        arguments = (Path('/cold/no-repository'), ('rev-parse', '--verify', 'HEAD'))
    elif name.endswith('._git_ref_exists'):
        # A negative-only argument carrier, not an original daemon instance.
        arguments = (SimpleNamespace(repo_root=Path('/cold/no-repository')), 'HEAD')
    else:
        arguments = (Path('/cold/no-repository'),)
    with pytest.raises(type(failure)) as caught:
        function(*arguments)
    assert caught.value is failure
    assert imports == [module.RUNTIME_MODULE]
    assert launches == []


def test_unbound_route_keeps_single_original_argument_evaluation(assembly):
    module, _originals, _bases, proposed = assembly
    relative, name, _source, _expected = module.SITES[0]
    marker = ColdBoundRefusal('original_argv_evaluation_marker')
    evaluations = []

    class OriginalArguments:
        def __iter__(self):
            evaluations.append(True)
            raise marker

    function, imports, launches = cold_function(module, proposed[relative], name,
        bound=False, failure=AssertionError('unbound_dispatch_must_not_run'))
    with pytest.raises(ColdBoundRefusal) as caught:
        function(Path('/cold/no-repository'), OriginalArguments())
    assert caught.value is marker
    assert evaluations == [True]
    assert imports == [module.RUNTIME_MODULE]
    assert launches == []


def test_original_eager_generation_and_outer_error_handlers_are_preserved(assembly):
    module, originals, _bases, proposed = assembly
    name = '_capture_imported_control_plane_generation'
    assert module.ast_sha(module.definition(originals[module.DAEMON], name)) == module.ast_sha(
        module.definition(proposed[module.DAEMON], name))

    def capture_blocks(source):
        return [ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, ast.Try) and any(isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Name) and item.func.id == name for item in ast.walk(node))]

    assert len(capture_blocks(originals[module.DAEMON])) == 1
    assert capture_blocks(originals[module.DAEMON]) == capture_blocks(proposed[module.DAEMON])


@pytest.mark.parametrize('site_index', range(4))
def test_only_selected_definition_changes_in_each_complete_module(assembly, site_index):
    module, _originals, bases, proposed = assembly
    relative, name, _source, _expected = module.SITES[site_index]
    old_tree = ast.parse(bases[relative])
    new_tree = ast.parse(proposed[relative])
    old_parent, new_parent = old_tree, new_tree
    parts = name.split('.')
    for part in parts[:-1]:
        old_parent = next(node for node in old_parent.body if isinstance(node, ast.ClassDef) and node.name == part)
        new_parent = next(node for node in new_parent.body if isinstance(node, ast.ClassDef) and node.name == part)
    old_node = next(node for node in old_parent.body if isinstance(node, ast.FunctionDef) and node.name == parts[-1])
    index = next(i for i, node in enumerate(new_parent.body) if isinstance(node, ast.FunctionDef) and node.name == parts[-1])
    new_parent.body[index] = old_node
    assert ast.dump(new_tree, include_attributes=False) == ast.dump(old_tree, include_attributes=False)
