"""Source-only successor: check durable denial before report-only dispatch.

No product import, Git, DB, native grant or job. Original checkout and frozen
predecessors stay unchanged. The guarded private kernel still has its original
workspace decorator; only public run_once is before that decorator boundary.
"""
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).absolute().parent
BASE = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-settlement-20261017-01/daemon-integration-v03')
BUILDER_SHA = 'f5199a384dfc8ab642582bb68fe78765f61b6958471597180aab4f25c270006e'
BASE_SHA = '4984370456bd304dd9d89550c05e2d364fd9f0ca4849db560cb10caddfe3f407'
RELATIVE = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def successor(source):
    if sha(source.encode()) != BASE_SHA:
        raise ValueError('exact_frozen_daemon_V03_required')
    for tail in ('guard_pass_entry(self)', 'require_attempt_profile(self, task, state)'):
        before = ('        if self.report_only_candidate_freezing:\n'
                  '            self._require_report_only_native_host().' + tail + '\n')
        after = ('        if self.report_only_candidate_freezing:\n'
                 '            from .report_only_disposition_gate import require_clear_report_only_disposition\n\n'
                 '            require_clear_report_only_disposition(self)\n'
                 '            self._require_report_only_native_host().' + tail + '\n')
        if source.count(before) != 1:
            raise ValueError('one_exact_original_report_only_entry_guard_required')
        source = source.replace(before, after, 1)
    ast.parse(source)
    return source


def assemble():
    path = BASE/'build_patch.py'
    if sha(path.read_bytes()) != BUILDER_SHA:
        raise ValueError('exact_original_source_assembler_required')
    spec = importlib.util.spec_from_file_location('frozen_V03_source_assembler_only', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original, _previous, base = module.assemble()
    return original, base, successor(base)


def main():
    original, base, proposed = assemble()
    for name, before in (('implementation_daemon.patch', original), ('disposition-entry-successor.patch', base)):
        raw = ''.join(difflib.unified_diff(before.splitlines(True), proposed.splitlines(True),
            fromfile='before/'+RELATIVE, tofile='proposed/'+RELATIVE))
        with (HERE/name).open('x') as stream:
            stream.write(raw)
    evidence = dict(schema='legal-ir-disposition-entry-source-assembly@1',
        original_sha256=sha(original.encode()), base_sha256=sha(base.encode()),
        proposed_sha256=sha(proposed.encode()), proposed_bytes=len(proposed.encode()),
        runtime_enabled=False, qualification='source assembly only')
    with (HERE/'ASSEMBLY.json').open('x') as stream:
        json.dump(evidence, stream, sort_keys=True, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
