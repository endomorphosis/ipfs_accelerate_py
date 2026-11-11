"""Authored closed custody, isolated lazy imports and rejected dependency drift."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qualified_reader_loader import load_reader
from test_safe_diagnostics import observed

ROOT = Path(__file__).resolve().parent


def custody():
    return dict(schema='database-native-provider-custody-observation@1',
        checks=[dict(stage='issuance', status='passed', reason_code='eligible'),
                dict(stage='native_exit', status='denied', reason_code='child_census_not_empty_or_unstable')],
        observation_only=True, completion_authority=False, retry_authority=False, settlement_authority=False)


class ClosureTests(unittest.TestCase):
    def test_later_custody_import_works_after_loader_restores_search_path_in_fresh_process(self):
        value = observed(); value['bridge']['diagnostic']['native_provider_custody_observation'] = custody()
        code = '''import importlib.util,json,sys
from pathlib import Path
root=Path(sys.argv[1]); before=list(sys.path)
spec=importlib.util.spec_from_file_location('isolated_loader',root/'qualified_reader_loader.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
reader=loader.load_reader(root)
assert sys.path==before and str(root) not in sys.path
result=reader.failure_observations(json.loads(sys.stdin.read()))
assert result['status']=='validated_retained_observation'
assert result['bridge']['diagnostic']['native_provider_custody_observation']['checks'][1]['reason_code']=='child_census_not_empty_or_unstable'
assert not any(n.startswith(('ipfs_accelerate_py','ipfs_datasets_py','ipfs_kit_py')) for n in sys.modules)
print('closed_custody_observation_retained')
'''
        result = subprocess.run([sys.executable, '-I', '-c', code, str(ROOT)],
            input=json.dumps(value), text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), 'closed_custody_observation_retained')

    def test_custody_invalid_fields_authority_and_stage_order_fail_closed(self):
        reader = load_reader(ROOT)
        for mutate in (lambda v:v.update(raw_message='PRIVATE'), lambda v:v.update(retry_authority=True),
            lambda v:v['checks'].reverse(), lambda v:v['checks'][1].update(reason_code='PRIVATE'),
            lambda v:v['checks'][0].update(status='denied'), lambda v:v.update(observation_only=1)):
            value = observed(); c = custody(); mutate(c)
            value['bridge']['diagnostic']['native_provider_custody_observation'] = c
            result = reader.failure_observations(value)
            self.assertEqual(result['status'], 'invalid')
            self.assertNotIn('PRIVATE', json.dumps(result))

    def test_missing_lazy_dependency_is_detected_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); original=json.loads((ROOT/'reporting-freeze.json').read_text())
            rows=[]
            for row in original['files']:
                if Path(row['path']).name=='diagnostic_custody_contract.py':continue
                path=root/Path(row['path']).name;path.write_bytes(Path(row['path']).read_bytes())
                rows.append({**row,'path':str(path)})
            (root/'reporting-freeze.json').write_text(json.dumps({**original,'files':rows}))
            with self.assertRaisesRegex(ValueError,'unbound_reader_dependency'):
                load_reader(root)

    def test_changed_contract_bytes_are_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);original=json.loads((ROOT/'reporting-freeze.json').read_text());rows=[]
            for row in original['files']:
                path=root/Path(row['path']).name;path.write_bytes(Path(row['path']).read_bytes()+b'\n')
                rows.append({**row,'path':str(path)})
            (root/'reporting-freeze.json').write_text(json.dumps({**original,'files':rows}))
            with self.assertRaisesRegex(ValueError,'reader_binding_changed'):
                load_reader(root)

    def test_foreign_cached_contract_module_is_rejected(self):
        old=sys.modules.get('diagnostic_custody_contract')
        from types import SimpleNamespace
        sys.modules['diagnostic_custody_contract']=SimpleNamespace(__file__='/untrusted/diagnostic_custody_contract.py')
        try:
            with self.assertRaisesRegex(ValueError,'foreign_reader_module'):
                load_reader(ROOT)
        finally:
            if old is None:sys.modules.pop('diagnostic_custody_contract',None)
            else:sys.modules['diagnostic_custody_contract']=old
