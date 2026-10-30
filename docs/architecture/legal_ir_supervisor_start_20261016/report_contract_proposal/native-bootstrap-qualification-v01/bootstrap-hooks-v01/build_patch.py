"""Effect-free source assembly for four explicitly bound bootstrap Git hooks.

Only stdlib and the source-pinned previous source assembler are loaded. No
product module, Git, database, lease, subprocess or provider is imported/run.
The source remains a disabled proposal; main() only writes review artifacts.
"""
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).absolute().parent
ORIGINAL_ROOT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
PREDECESSOR = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-disposition-20261017-01/daemon-integration-v04/build_patch.py')
PREDECESSOR_SHA = 'cb51877fca1785f8a54de8c7c01c5d9f4fe87c0465afa2ddd45283b364b7a47b'
DAEMON_BASE_SHA = '67f4959c391776a1361a98f2a5f8d137e86ea5651e9ced0ab96fca3eb91f15a0'
RUNTIME_MODULE = 'ipfs_accelerate_py.agent_supervisor.runtime.native_admin_execution'
DAEMON = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
SITES = (
    ('ipfs_accelerate_py/agent_implementation_route.py', '_agent_git_output',
     '56126c1bfb8db68750909559b76deee2ae61bf6f82636f1e170a339941d253f2',
     'a908aa2b9806e9499a1eb7fe1887fefa17b142f7ea715c4a61be08a63030e277'),
    ('ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py', 'git_common_dir',
     '8e0820ac71147059c76a81e1c0281e932e2c800838169f8392a89522c40ed87f',
     '560c08c136daf46fc986534ba0118667401c7b2688de9539b1dc93540b0d939b'),
    ('ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py', 'control_plane_git_common_dir',
     '863d7c16d193cf13fd37ca49f7fc34ce01b0d3c814b8e29be1397ed419eb22c1',
     '3d3d035accc2f417346acb9b3a99f96486221c516df6d99ee59bfee96bc85f9c'),
    (DAEMON, 'PortalImplementationDaemon._git_ref_exists',
     '23addb0dfc23e2cb093e7f74abbb7dc84a5308c24259e1eb0e25d721c0a80bed',
     'ed16eea639bcebcd641a0c4d233b0e2109e8560409dda34011f0d74fc7d952f8'),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def ast_sha(node):
    return sha(ast.dump(node, include_attributes=False).encode())


def definition(source, qualified_name):
    current = ast.parse(source)
    for part in qualified_name.split('.'):
        matches = [node for node in current.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == part]
        need(len(matches) == 1, 'one_exact_bootstrap_source_definition_required')
        current = matches[0]
    return current


def hook(source, qualified_name, expected_ast_sha):
    """Keep the original unbound try/call/catches and all validators intact.

    The presence check is only a discriminator, never native authority. Its
    lazy import adds no todo_daemon dependency to runtime. Bound dispatch is
    outside every original local catch, and cannot fall through to the legacy
    subprocess branch after failure or an unexpected None return.
    """
    node = definition(source, qualified_name)
    need(ast_sha(node) == expected_ast_sha, 'exact_original_bootstrap_function_required')
    tries = [item for item in node.body if isinstance(item, ast.Try)]
    need(len(tries) == 1, 'one_original_bootstrap_query_try_required')
    original_try = tries[0]
    need(len(original_try.body) == 1 and isinstance(original_try.body[0], ast.Assign),
         'one_original_bootstrap_query_assignment_required')
    assignment = original_try.body[0]
    need(isinstance(assignment.value, ast.Call)
         and isinstance(assignment.value.func, ast.Attribute)
         and isinstance(assignment.value.func.value, ast.Name)
         and assignment.value.func.value.id == 'subprocess'
         and assignment.value.func.attr == 'run', 'exact_original_subprocess_run_required')
    lines = source.splitlines(keepends=True)
    indent = ' ' * original_try.col_offset
    bound_assignment = ''.join(lines[assignment.lineno-1:assignment.end_lineno])
    need(bound_assignment.count('subprocess.run(') == 1, 'one_bootstrap_callee_spelling_required')
    bound_assignment = bound_assignment.replace('subprocess.run(', 'run_bound_native_admin_if_present(', 1)
    original_block = ''.join(lines[original_try.lineno-1:original_try.end_lineno])
    replacement = (
        indent + 'from ' + RUNTIME_MODULE + ' import (\n'
        + indent + '    native_admin_execution_is_bound,\n'
        + indent + '    run_bound_native_admin_if_present,\n'
        + indent + ')\n'
        + indent + 'if native_admin_execution_is_bound():\n'
        + bound_assignment
        + indent + 'else:\n'
        + ''.join('    ' + line if line.strip() else line for line in original_block.splitlines(keepends=True))
    )
    proposed = ''.join(lines[:original_try.lineno-1]) + replacement + ''.join(lines[original_try.end_lineno:])
    patched = definition(proposed, qualified_name)
    location = next(i for i, item in enumerate(node.body) if item is original_try)
    need(isinstance(patched.body[location], ast.ImportFrom)
         and patched.body[location].module == RUNTIME_MODULE,
         'canonical_runtime_import_required')
    branch = patched.body[location+1]
    need(isinstance(branch, ast.If) and len(branch.orelse) == 1
         and ast.dump(branch.orelse[0], include_attributes=False) == ast.dump(original_try, include_attributes=False),
         'original_unbound_query_and_error_handlers_must_be_identical')
    restored = list(patched.body)
    restored[location:location+2] = branch.orelse
    patched.body = restored
    need(ast_sha(patched) == expected_ast_sha, 'all_original_non_hook_function_semantics_must_be_identical')
    return proposed


def original_sources():
    values = {}
    for relative, _name, expected, _ast in SITES:
        raw = (ORIGINAL_ROOT/relative).read_bytes()
        need(sha(raw) == expected, 'exact_original_bootstrap_module_source_required')
        values[relative] = raw.decode()
    return values


def assemble():
    originals = original_sources()
    raw = PREDECESSOR.read_bytes()
    need(sha(raw) == PREDECESSOR_SHA, 'exact_frozen_daemon_V04_source_assembler_required')
    spec = importlib.util.spec_from_file_location('frozen_disposition_V04_source_assembler_only', PREDECESSOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original, _previous, daemon_base = module.assemble()
    need(original == originals[DAEMON] and sha(daemon_base.encode()) == DAEMON_BASE_SHA,
         'complete_frozen_disposition_daemon_V04_required')
    bases = dict(originals)
    bases[DAEMON] = daemon_base
    proposed = {relative: hook(bases[relative], name, expected_ast)
                for relative, name, _source, expected_ast in SITES}
    return originals, bases, proposed


def main():
    originals, bases, proposed = assemble()
    outputs = []
    for relative, name, _source, expected_ast in SITES:
        for label, before in (('original', originals[relative]), ('composed-base', bases[relative])):
            patch_name = relative.replace('/', '__') + '.' + label + '.patch'
            patch = ''.join(difflib.unified_diff(before.splitlines(True), proposed[relative].splitlines(True),
                fromfile='before/'+relative, tofile='proposed/'+relative))
            with (HERE/patch_name).open('x') as stream:
                stream.write(patch)
        outputs.append(dict(path=relative, definition=name, original_sha256=sha(originals[relative].encode()),
            composed_base_sha256=sha(bases[relative].encode()), proposed_sha256=sha(proposed[relative].encode()),
            proposed_bytes=len(proposed[relative].encode()), original_function_AST_sha256=expected_ast))
    report = dict(schema='legal-ir-native-bootstrap-hook-source-assembly@1', modules=outputs,
        runtime_module=RUNTIME_MODULE, runtime_enabled=False, tests_executed=False,
        actual_native_admission_qualified=False, original_four_unbound_branches_preserved=True,
        qualification='disabled source proposal only; no product import, execution or authority')
    with (HERE/'ASSEMBLY.json').open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
