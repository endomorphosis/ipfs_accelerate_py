"""Container entrypoint for existing offline supervisor integration tests."""
import os
import sys

sys.path[:0] = ['/code/source', '/code/datasets', '/code/kit']
os.chdir('/tmp')

if '--preflight' in sys.argv:
    import importlib
    import importlib.metadata
    import json
    import duckdb
    report = {'schema': 'numerical-supervisor-container-preflight@1', 'packages': {}}
    for name in ('pytest', 'jsonschema', 'referencing', 'duckdb', 'pandas'):
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    for name in ('benchmarks.agent_supervisor.container_coding.test_terminal_numerical_selection_integration',):
        try:
            importlib.import_module(name)
            report[name] = {'imported': True}
        except Exception as exc:
            report[name] = {'imported': False, 'error': type(exc).__name__, 'message': str(exc)}
    try:
        with duckdb.connect(':memory:') as connection:
            connection.execute('SET autoinstall_known_extensions=false')
            connection.execute('SET autoload_known_extensions=false')
            report['extension_directory'] = connection.execute("SELECT current_setting('extension_directory')").fetchone()[0]
            connection.execute('LOAD ducklake')
        report['ducklake_loaded'] = True
    except Exception as exc:
        report['ducklake_loaded'] = False
        report['ducklake_error'] = str(exc)
    print(json.dumps(report, sort_keys=True))
else:
    import pytest
    raise SystemExit(pytest.main([
        '-q', '--color=no', '-x',
        '/code/source/benchmarks/agent_supervisor/container_coding/test_terminal_numerical_selection_integration.py',
        '--basetemp=/tmp/supervisor-pytest',
        '--junitxml=/qualification-output/junit.xml',
        '-o', 'cache_dir=/tmp/pytest-cache',
    ]))
