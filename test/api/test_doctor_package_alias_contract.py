"""Authored regular-package alias contracts and independent negative controls.

These tests validate the finite source environment and exact callee edit. Native
Lean/Z3 checks prove the emitted alias/signature statements, not Python import
machinery or whole-program behavior.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import PurePosixPath
import shutil
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import doctor_alias_contract as alias_module
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_alias_contract import (
    ImportedAliasContractError,
    ImportedAliasRepair,
    OPERATOR,
    PACKAGE_OPERATOR,
    discover_imported_alias_repair,
    read_imported_alias_sources,
)
from test.api.test_doctor_task_workflow import _provers


CALLER = '''"""An authored caller with one unresolved direct callee."""
from MODULE import transform as normalize

def answer(value, /, enabled=True):
    return transform(value, enabled=enabled)
'''
DONOR = '''"""An authored, function-only donor."""
def transform(value, /, scale=2, *, enabled=True):
    return value * scale if enabled else value
'''


def _sources(*, caller='pkg/caller.py', imported='pkg.helpers',
             donor='pkg/helpers.py', initializer='"""Inert package."""\n'):
    sources = {caller: CALLER.replace('MODULE', imported), donor: DONOR}
    for path in (caller, donor):
        for parent in PurePosixPath(path).parents:
            if str(parent) != '.':
                sources[str(parent / '__init__.py')] = initializer
    return sources


def _discover(sources=None, *, path='pkg/caller.py'):
    return discover_imported_alias_repair(
        sources=_sources() if sources is None else sources, path=path)


def _write_sources(root, sources):
    for name, text in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.mark.parametrize('caller,imported,donor', [
    ('caller.py', 'pkg.helpers', 'pkg/helpers.py'),
    ('pkg/caller.py', 'pkg.helpers', 'pkg/helpers.py'),
    ('pkg/caller.py', '.helpers', 'pkg/helpers.py'),
    ('pkg/nested/caller.py', '..helpers', 'pkg/helpers.py'),
    ('pkg/nested/caller.py', '.helpers', 'pkg/nested/helpers.py'),
    ('pkg/caller.py', '.nested.helpers', 'pkg/nested/helpers.py'),
    ('pkg/caller.py', 'other.helpers', 'other/helpers.py'),
])
def test_package_alias_resolves_exact_donor_and_changes_only_callee(tmp_path, caller, imported, donor):
    sources = _sources(caller=caller, imported=imported, donor=donor)
    contract = _discover(sources, path=caller)
    row = contract.to_dict()
    assert row['path'] == caller and row['donor_path'] == donor
    assert row['subject'] == 'answer'
    assert row['previous'] == 'transform' and row['replacement'] == 'normalize'
    assert row['target_binding'] == row['bindings']['normalize']
    assert contract.operator == row['schema'] == PACKAGE_OPERATOR
    assert row['source_hashes'] == {
        name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
    assert contract.sources() == sources
    before = sources[caller]
    after = before.replace('return transform(', 'return normalize(')
    assert before[row['offset']:row['end_offset']] == 'transform'
    contract.validate_candidate(after)
    assert _discover({**sources, caller: after}, path=caller) is None
    for different in (after.replace('enabled=enabled', 'enabled=False'),
                      after.replace('authored caller', 'altered caller'),
                      after.replace('normalize(value', 'normalize(2')):
        with pytest.raises(ImportedAliasContractError):
            contract.validate_candidate(different)
    # Independently exercise Python's package resolution for these authored,
    # inert fixtures. Production discovery must never import target source.
    _write_sources(tmp_path, {**sources, caller: after})
    module = caller.removesuffix('.py').replace('/', '.')
    run = subprocess.run([sys.executable, '-I', '-B', '-c',
        'import importlib, sys; sys.path.insert(0, sys.argv[1]); '
        'module = importlib.import_module(sys.argv[2]); '
        'assert module.answer(3) == 6; assert module.answer(3, enabled=False) == 3',
        str(tmp_path), module], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize('initializer', ['', 'pass\n', '"""Package documentation."""\n',
                                         '"""Package documentation."""\npass\npass\n'])
def test_regular_package_initializers_are_inert(initializer):
    assert _discover(_sources(initializer=initializer)) is not None


def test_flat_population_retains_existing_operator_and_package_metadata_is_bound():
    flat = _discover(_sources(caller='caller.py', imported='helpers', donor='helpers.py'),
                     path='caller.py')
    assert flat.operator == flat.to_dict()['schema'] == OPERATOR
    assert 'module_resolution' not in flat.to_dict()
    packaged = _discover()
    row = packaged.to_dict()
    resolution = row['module_resolution']
    assert resolution['schema'] == 'closed-python-package-resolution@1'
    assert resolution['module_paths']['pkg.helpers'] == 'pkg/helpers.py'
    assert resolution['module_paths']['pkg.caller'] == 'pkg/caller.py'
    assert resolution['package_paths'] == ['pkg/__init__.py']
    assert len(resolution['import_edges']) == 1
    forged = deepcopy(row)
    forged['module_resolution']['module_paths']['pkg.helpers'] = 'pkg/caller.py'
    with pytest.raises(ImportedAliasContractError):
        ImportedAliasRepair(json.dumps(forged), json.dumps(packaged.sources()))
    forged = deepcopy(row)
    forged['module_resolution']['package_paths'] = []
    with pytest.raises(ImportedAliasContractError):
        ImportedAliasRepair(json.dumps(forged), json.dumps(packaged.sources()))


@pytest.mark.parametrize('initializer', [
    'import sys\n',
    'from .helpers import transform\n',
    'from .helpers import transform as public_transform\n',
    'value = 1\n',
    'def marker(value):\n    return value\n',
    'def __getattr__(name):\n    return name\n',
    'print("import-time effect")\n',
    'if False:\n    import os\n',
    'raise RuntimeError("import-time effect")\n',
])
def test_package_init_effects_exports_and_hidden_branches_remain_unsupported(initializer):
    with pytest.raises(ImportedAliasContractError):
        _discover(_sources(initializer=initializer))


@pytest.mark.parametrize('removed', ['pkg/__init__.py', 'pkg/nested/__init__.py'])
def test_namespace_or_incomplete_parent_population_is_rejected(removed):
    sources = _sources(caller='pkg/nested/caller.py', imported='..helpers')
    del sources[removed]
    with pytest.raises(ImportedAliasContractError):
        _discover(sources, path='pkg/nested/caller.py')


@pytest.mark.parametrize('extra', ['pkg.py', 'pkg/nested.py'])
def test_module_and_regular_package_collision_is_rejected(extra):
    sources = _sources(caller='pkg/nested/caller.py', imported='..helpers')
    sources[extra] = 'def identity(value):\n    return value\n'
    with pytest.raises(ImportedAliasContractError):
        _discover(sources, path='pkg/nested/caller.py')


@pytest.mark.parametrize('name', [
    '/absolute.py', '../escape.py', 'pkg/../escape.py', 'pkg/./extra.py',
    'pkg//extra.py', 'pkg\\extra.py', 'pkg/extra.py/', 'pkg/extra.txt',
    'pkg/not-valid.py', 'pkg/123name.py', 'pkg/class.py', 'class/module.py',
    'pkg/extra\x00.py', '__init__.py',
])
def test_noncanonical_or_nonmodule_population_paths_are_rejected(name):
    sources = _sources()
    sources[name] = DONOR
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)


@pytest.mark.parametrize('segment', ['__init__', '__dict__', '__class__', '__path__'])
def test_special_package_metadata_name_cannot_be_a_nested_package_component(segment):
    caller = f'pkg/{segment}/caller.py'
    with pytest.raises(ImportedAliasContractError):
        _discover(_sources(caller=caller), path=caller)


@pytest.mark.parametrize('module', ['__dict__', '__class__', '__path__', '__spec__'])
def test_imported_submodule_cannot_overwrite_implicit_package_metadata(module):
    with pytest.raises(ImportedAliasContractError):
        _discover(_sources(imported='pkg.' + module, donor='pkg/' + module + '.py'))


@pytest.mark.parametrize('name', [None, 7, b'extra.py'])
def test_nonstring_population_names_are_rejected_before_sorting(name):
    sources = _sources()
    sources[name] = DONOR
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)


@pytest.mark.parametrize('depth', [8, 9])
def test_package_depth_bound_is_applied_to_complete_population(depth):
    caller = '/'.join(['pkg'] + [f'level{number}' for number in range(depth - 1)] + ['caller.py'])
    sources = _sources(caller=caller)
    if depth == 8:
        assert _discover(sources, path=caller) is not None
    else:
        with pytest.raises(ImportedAliasContractError):
            _discover(sources, path=caller)


@pytest.mark.parametrize('length', [64, 65])
def test_package_identifier_bounds_are_applied_to_each_segment(length):
    package = 'p' * length
    caller, donor = package + '/caller.py', package + '/helpers.py'
    sources = _sources(caller=caller, imported=package + '.helpers', donor=donor)
    if length == 64:
        assert _discover(sources, path=caller) is not None
    else:
        with pytest.raises(ImportedAliasContractError):
            _discover(sources, path=caller)


@pytest.mark.parametrize('root', ['sys', 'os', 'builtins', 'json', 'importlib'])
def test_builtin_and_standard_library_roots_cannot_be_local_packages(root):
    sources = _sources(caller='caller.py', imported=f'{root}.helpers', donor=f'{root}/helpers.py')
    with pytest.raises(ImportedAliasContractError):
        _discover(sources, path='caller.py')


@pytest.mark.parametrize('root', ['sys', 'os', 'builtins', 'json', 'importlib'])
def test_standard_library_population_conflicts_are_not_ignored(root):
    sources = _sources()
    sources[root + '.py'] = 'def identity(value):\n    return value\n'
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)


@pytest.mark.parametrize('caller,imported', [
    ('caller.py', '.helpers'),
    ('pkg/caller.py', '..helpers'),
    ('pkg/nested/caller.py', '...helpers'),
    ('pkg/caller.py', 'pkg'),
    ('pkg/caller.py', 'pkg.__init__'),
    ('pkg/caller.py', '.missing'),
    ('pkg/caller.py', 'pkg.helpers.nested'),
])
def test_missing_or_nonmodule_donors_and_relative_root_escape_are_rejected(caller, imported):
    sources = _sources(caller=caller, imported=imported)
    with pytest.raises(ImportedAliasContractError):
        _discover(sources, path=caller)


@pytest.mark.parametrize('replacement', [
    'import pkg.helpers as normalize',
    'from pkg.helpers import *',
    'from pkg.helpers import transform as normalize, transform as alternate',
    'from pkg.helpers import transform as normalize\nfrom pkg.helpers import transform as normalize',
    'from pkg.helpers import transform as normalize\nfrom pkg.helpers import transform as answer',
])
def test_dynamic_attribute_star_duplicate_and_ambiguous_import_shapes_abstain(replacement):
    sources = _sources()
    sources['pkg/caller.py'] = sources['pkg/caller.py'].replace(
        'from pkg.helpers import transform as normalize', replacement)
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)


@pytest.mark.parametrize('name', ['__name__', '__package__', '__builtins__', '__path__', '__spec__',
                                 '__dict__', '__class__', '__all__'])
def test_import_alias_cannot_rebind_implicit_module_or_package_globals(name):
    sources = _sources()
    sources['pkg/caller.py'] = sources['pkg/caller.py'].replace('as normalize', 'as ' + name)
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)


@pytest.mark.parametrize('before,after', [
    ('def answer(value, /, enabled=True):', 'def answer(normalize, /, enabled=True):'),
    ('def answer(value, /, enabled=True):', 'def answer(transform, /, enabled=True):'),
    ('return transform(value, enabled=enabled)', 'return helpers.transform(value, enabled=enabled)'),
    ('return transform(value, enabled=enabled)', 'return __import__("pkg.helpers")'),
    ('return transform(value, enabled=enabled)', 'return normalize(transform(value))'),
])
def test_package_resolution_does_not_expand_callee_or_argument_grammar(before, after):
    sources = _sources()
    sources['pkg/caller.py'] = sources['pkg/caller.py'].replace(before, after)
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)


def test_reexport_chains_and_import_cycles_are_not_resolved():
    sources = _sources()
    sources['pkg/other.py'] = 'def other(value):\n    return value\n'
    sources['pkg/helpers.py'] = 'from .other import other\n' + DONOR
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)
    sources['pkg/helpers.py'] = 'from .caller import answer\n' + DONOR
    with pytest.raises(ImportedAliasContractError):
        _discover(sources)


def test_nested_package_parent_sources_are_loaded_and_currentness_bound(tmp_path):
    sources = _sources(caller='pkg/nested/caller.py', imported='..helpers')
    contract = _discover(sources, path='pkg/nested/caller.py')
    _write_sources(tmp_path, sources)
    hashes = contract.to_dict()['source_hashes']
    assert read_imported_alias_sources(repository=tmp_path, source_hashes=hashes) == sources
    contract.assert_current(tmp_path)
    # Even another inert initializer must invalidate the existing source claim.
    (tmp_path / 'pkg/__init__.py').write_text('"""Changed package documentation."""\n')
    with pytest.raises(ImportedAliasContractError):
        contract.assert_current(tmp_path)
    with pytest.raises(ImportedAliasContractError):
        read_imported_alias_sources(repository=tmp_path, source_hashes=hashes)


@pytest.mark.parametrize('digest', [None, False, 7, '0' * 63, '0' * 65, 'G' * 64, 'A' * 64])
def test_loader_validates_all_digests_before_reading_any_source(tmp_path, monkeypatch, digest):
    hashes = _discover().to_dict()['source_hashes']
    hashes['pkg/helpers.py'] = digest

    def unexpected_read(*args, **kwargs):
        pytest.fail('invalid source hash inventory reached filesystem reading')

    monkeypatch.setattr(alias_module, '_read_current_source', unexpected_read)
    with pytest.raises(ImportedAliasContractError):
        read_imported_alias_sources(repository=tmp_path, source_hashes=hashes)


@pytest.mark.parametrize('tamper', ['digest', 'omitted_digest', 'source', 'omitted_source', 'donor_path'])
def test_contract_cannot_claim_unbound_parent_initializers_or_donor(tamper):
    contract = _discover()
    row, sources = deepcopy(contract.to_dict()), contract.sources()
    if tamper == 'digest':
        row['source_hashes']['pkg/__init__.py'] = '0' * 64
    elif tamper == 'omitted_digest':
        del row['source_hashes']['pkg/__init__.py']
    elif tamper == 'source':
        sources['pkg/__init__.py'] = 'pass\n'
    elif tamper == 'omitted_source':
        del sources['pkg/__init__.py']
    else:
        row['donor_path'] = 'pkg/other.py'
    with pytest.raises(ImportedAliasContractError):
        ImportedAliasRepair(json.dumps(row), json.dumps(sources))


@pytest.mark.parametrize('key,value', [('donor_path', 'pkg/caller.py'),
                                      ('module', 'other.helpers'), ('level', 0)])
def test_relative_resolution_edge_cannot_be_relabelled_without_reconstruction(key, value):
    contract = _discover(_sources(imported='.helpers'))
    row = contract.to_dict()
    assert row['module_resolution']['import_edges'][0][key] != value
    row['module_resolution']['import_edges'][0][key] = value
    with pytest.raises(ImportedAliasContractError):
        ImportedAliasRepair(json.dumps(row), json.dumps(contract.sources()))


@pytest.mark.parametrize('shape', ['parent_symlink', 'initializer_symlink', 'initializer_fifo',
                                   'initializer_hardlink'])
def test_package_source_loader_rejects_indirect_or_nonregular_initializers(tmp_path, shape):
    sources = _sources()
    contract = _discover(sources)
    root = tmp_path / 'repository'
    root.mkdir()
    _write_sources(root, sources)
    if shape == 'parent_symlink':
        moved = tmp_path / 'moved-package'
        (root / 'pkg').rename(moved)
        (root / 'pkg').symlink_to(moved, target_is_directory=True)
    else:
        initializer = root / 'pkg/__init__.py'
        initializer.unlink()
        if shape == 'initializer_symlink':
            outside = tmp_path / 'initializer.txt'
            outside.write_text(sources['pkg/__init__.py'])
            initializer.symlink_to(outside)
        elif shape == 'initializer_hardlink':
            outside = tmp_path / 'initializer.txt'
            outside.write_text(sources['pkg/__init__.py'])
            os.link(outside, initializer)
        else:
            os.mkfifo(initializer)
    with pytest.raises((ImportedAliasContractError, OSError)):
        read_imported_alias_sources(repository=root, source_hashes=contract.to_dict()['source_hashes'])
    with pytest.raises((ImportedAliasContractError, OSError)):
        contract.assert_current(root)


def test_package_projection_uses_real_provers_and_rejects_wrong_binding_and_signature(tmp_path):
    paths = _provers({})
    contract = _discover(_sources(caller='pkg/nested/caller.py', imported='..helpers'),
                         path='pkg/nested/caller.py')
    projection = contract.formal_projection('qualification:regular-package-alias-binding')
    path = tmp_path / 'PackageAlias.lean'

    def lean_result(source):
        path.write_text(source)
        return subprocess.run([str(paths['kernel_executable']), str(path)],
                              capture_output=True, text=True, timeout=30)

    def z3_result(source):
        return subprocess.run([shutil.which('z3'), '-in'], input=source,
                              capture_output=True, text=True, timeout=15)

    lean = lean_result(projection['lean'])
    assert lean.returncode == 0, lean.stdout + lean.stderr
    assert set(lean.stdout.strip().splitlines()) == set(projection['expected_axioms'])
    z3 = z3_result(projection['smt'])
    assert z3.returncode == 0 and z3.stdout.strip() == 'unsat'

    target = json.dumps(contract.to_dict()['target_binding'])
    alias = json.dumps(contract.to_dict()['replacement'])
    wrong_lean = projection['lean'].replace('= some ' + target, '= some "different-donor"')
    wrong_smt = projection['smt'].replace('(lookupBinding ' + alias + ') ' + target,
                                          '(lookupBinding ' + alias + ') "different-donor"')
    assert wrong_lean != projection['lean'] and wrong_smt != projection['smt']
    assert lean_result(wrong_lean).returncode != 0
    z3 = z3_result(wrong_smt)
    assert z3.returncode == 0 and z3.stdout.strip() == 'sat'

    wrong_lean = projection['lean'].replace(
        'def keywordParameters : List String := ["scale", "enabled"]',
        'def keywordParameters : List String := ["scale"]')
    wrong_smt = projection['smt'].replace('(= "enabled" "enabled")', '(= "enabled" "unbound")')
    assert wrong_lean != projection['lean'] and wrong_smt != projection['smt']
    assert lean_result(wrong_lean).returncode != 0
    z3 = z3_result(wrong_smt)
    assert z3.returncode == 0 and z3.stdout.strip() == 'sat'
