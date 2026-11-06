"""Normal real namespace imports with declared directory custody and raw bodies.

The fixture uses actual package files, PathFinder and Python's normal namespace
loader. It never manufactures a package object or inserts one into sys.modules.
"""
from contextlib import contextmanager
import hashlib
import importlib
import os
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker


@pytest.fixture
def normal_namespace_fixture(tmp_path):
    datasets = tmp_path / 'datasets'
    package = datasets / 'ipfs_datasets_py'
    logic = package / 'logic'
    namespace = logic / 'backends'
    namespace.mkdir(parents=True)
    initializers = [package / '__init__.py', logic / '__init__.py']
    for path in initializers:
        path.write_text('ordinary_package_initialized = True\n')
    source = namespace / 'process.py'
    source.write_text('namespace_child_executed = True\n')
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [*initializers, source]}
    info = namespace.lstat()
    profile = dict(fullname='ipfs_datasets_py.logic.backends', path=str(namespace),
        device=info.st_dev, inode=info.st_ino, initializer_absent=True)
    return dict(root=tmp_path, datasets=datasets, logic=logic, namespace=namespace,
        source=source, pins=pins, profile=profile)


@contextmanager
def cold_normal_imports(fixture, profiles):
    original_modules = {name: value for name, value in sys.modules.items()
        if name == 'ipfs_datasets_py' or name.startswith('ipfs_datasets_py.')}
    original_path, original_meta = list(sys.path), list(sys.meta_path)
    try:
        for name in original_modules:
            del sys.modules[name]
        guard = worker.PinnedSources(fixture['pins'], [fixture['root'] / 'candidate',
            fixture['datasets'], fixture['root'] / 'trusted'], profiles)
        sys.path.insert(0, str(fixture['datasets']))
        sys.meta_path.insert(0, guard)
        yield guard
    finally:
        for name in tuple(sys.modules):
            if name == 'ipfs_datasets_py' or name.startswith('ipfs_datasets_py.'):
                del sys.modules[name]
        sys.modules.update(original_modules)
        sys.path[:] = original_path
        sys.meta_path[:] = original_meta
        importlib.invalidate_caches()


def test_declared_real_namespace_keeps_normal_loader_and_raw_descendant_body(normal_namespace_fixture):
    fixture = normal_namespace_fixture
    with cold_normal_imports(fixture, [fixture['profile']]) as guard:
        actual = importlib.import_module('ipfs_datasets_py.logic.backends.process')
        namespace = sys.modules['ipfs_datasets_py.logic.backends']
        assert namespace.__spec__.origin is None
        assert list(namespace.__path__) == [str(fixture['namespace'])]
        assert type(namespace.__loader__).__name__ == 'NamespaceLoader'
        assert actual.namespace_child_executed is True and Path(actual.__file__) == fixture['source']
        assert guard.executed[str(fixture['source'])] == fixture['pins'][str(fixture['source'])]
        assert guard.normal_namespace_origins == {fixture['profile']['fullname']: fixture['profile']}
        assert str(fixture['namespace']) not in guard.executed
        assert guard.verify_normal_namespaces() == guard.normal_namespace_origins
        assert guard.denials == []


@pytest.mark.parametrize('locations', ['wrong', 'multiple'])
def test_normal_namespace_refuses_wrong_or_multiple_real_search_locations(normal_namespace_fixture, locations):
    fixture = normal_namespace_fixture
    other = fixture['root'] / 'other-logic'
    (other / 'backends').mkdir(parents=True)
    with cold_normal_imports(fixture, [fixture['profile']]) as guard:
        logic = importlib.import_module('ipfs_datasets_py.logic')
        logic.__path__[:] = [str(other)] if locations == 'wrong' else [str(fixture['logic']), str(other)]
        with pytest.raises(ImportError, match='search locations changed'):
            importlib.import_module('ipfs_datasets_py.logic.backends')
        assert guard.denials and guard.normal_namespace_origins == {}
        assert 'ipfs_datasets_py.logic.backends' not in sys.modules


def test_normal_namespace_initializer_appearing_after_declaration_is_permanent_refusal(normal_namespace_fixture):
    fixture = normal_namespace_fixture
    with cold_normal_imports(fixture, [fixture['profile']]) as guard:
        (fixture['namespace'] / '__init__.py').write_text('unexpected_body_executed = True\n')
        with pytest.raises(ImportError):
            importlib.import_module('ipfs_datasets_py.logic.backends')
        assert guard.denials and guard.normal_namespace_origins == {}
        with pytest.raises(ValueError, match='initializer absence changed'):
            guard.verify_normal_namespaces()


def test_unlisted_project_namespace_still_refuses_a_real_pathfinder_spec(normal_namespace_fixture):
    fixture = normal_namespace_fixture
    with cold_normal_imports(fixture, None) as guard:
        with pytest.raises(ImportError, match='ordinary normal source origin'):
            importlib.import_module('ipfs_datasets_py.logic.backends')
        assert guard.denials and guard.normal_namespace_origins == {}


def test_declared_namespace_requires_its_normal_descendant_source_pin(normal_namespace_fixture):
    fixture = normal_namespace_fixture
    fixture['pins'].pop(str(fixture['source']))
    with pytest.raises(ValueError, match='exact declared D backends namespace'):
        worker.PinnedSources(fixture['pins'], [fixture['root'] / 'candidate',
            fixture['datasets'], fixture['root'] / 'trusted'], [fixture['profile']])


def test_normal_namespace_descendant_source_drift_never_executes_body(normal_namespace_fixture):
    fixture = normal_namespace_fixture
    with cold_normal_imports(fixture, [fixture['profile']]) as guard:
        fixture['source'].write_text('namespace_child_executed = False\n')
        with pytest.raises(ImportError, match='raw first-party source changed'):
            importlib.import_module('ipfs_datasets_py.logic.backends.process')
        assert str(fixture['source']) not in guard.executed and guard.denials
        assert 'ipfs_datasets_py.logic.backends.process' not in sys.modules


def test_normal_namespace_directory_replacement_never_becomes_new_custody(normal_namespace_fixture):
    fixture = normal_namespace_fixture
    with cold_normal_imports(fixture, [fixture['profile']]) as guard:
        fixture['namespace'].rename(fixture['namespace'].with_name('preserved-original'))
        fixture['namespace'].mkdir()
        with pytest.raises(ImportError, match='namespace custody changed'):
            importlib.import_module('ipfs_datasets_py.logic.backends')
        assert guard.denials and guard.normal_namespace_origins == {}
