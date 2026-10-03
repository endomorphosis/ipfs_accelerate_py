import copy
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import codex_cache_candidate as c
import native_library_cache_candidate as library

class NativeLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.rows=copy.deepcopy(library.ROWS)
        for i,row in enumerate(self.rows):
            raw=(row['name']+'\x00').encode()*(i+1);p=self.root/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(raw);p.chmod(row['mode']);row.update(uid=os.getuid(),sha256=hashlib.sha256(raw).hexdigest(),expected_bytes=len(raw))

    def test_exact_seven_real_syscalls_preserve_bytes_identity(self):
        before={r['path']:c.identity((self.root/r['path']).stat()) for r in self.rows}
        result=c.advise_selected(self.root,self.rows,expected_count=7)
        self.assertEqual(result['advised_files'],7);self.assertEqual(result['selected_files'],7)
        self.assertEqual(result['body_read_bytes'],sum(r['expected_bytes'] for r in self.rows))
        self.assertEqual(before,{r['path']:c.identity((self.root/r['path']).stat()) for r in self.rows})
        for row in self.rows:self.assertEqual(hashlib.sha256((self.root/row['path']).read_bytes()).hexdigest(),row['sha256'])

    def test_corrupt_last_body_causes_zero_advice(self):
        p=self.root/self.rows[-1]['path'];raw=p.read_bytes();p.write_bytes(b'X'+raw[1:])
        with patch.object(c.os,'posix_fadvise') as hint,self.assertRaises(ValueError):c.advise_selected(self.root,self.rows,expected_count=7)
        hint.assert_not_called()

    def test_closed_population_rejects_extra_duplicate_unsafe_and_wrong_size(self):
        variants=[]
        rows=copy.deepcopy(self.rows);rows.append(rows[-1]);variants.append(rows)
        rows=copy.deepcopy(self.rows);rows[-1]=rows[0];variants.append(rows)
        rows=copy.deepcopy(self.rows);rows[-1]['path']='../auth.json';variants.append(rows)
        rows=copy.deepcopy(self.rows);rows[-1]['expected_bytes']+=1;variants.append(rows)
        rows=copy.deepcopy(self.rows);rows[-1]['mode']=0o664;variants.append(rows)
        with patch.object(c.os,'posix_fadvise') as hint:
            for rows in variants:
                with self.subTest(rows=rows),self.assertRaises(ValueError):c.advise_selected(self.root,rows,expected_count=7)
            hint.assert_not_called()

    def test_torch_symlink_hardlink_refused_before_reads(self):
        p=self.root/self.rows[-1]['path'];other=self.root/self.rows[-2]['path'];p.unlink();p.symlink_to(other)
        with patch.object(c.os,'read') as read,self.assertRaises(OSError):c.advise_selected(self.root,self.rows,expected_count=7)
        read.assert_not_called();p.unlink();os.link(other,p)
        with patch.object(c.os,'read') as read,self.assertRaises(ValueError):c.advise_selected(self.root,self.rows,expected_count=7)
        read.assert_not_called()

    def test_library_adapter_exact_source_bindings(self):
        path=Path(library.__file__).with_name('finite-library-source-pins.json');raw=path.read_bytes();v=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),library.SOURCE_PINS_SHA256)
        self.assertEqual(len(library.ROWS),7)
        for source,row in zip(v['extensions'],library.ROWS[:3]):
            self.assertEqual(row['sha256'],source['sha256']);self.assertEqual(row['expected_bytes'],source['bytes'])
            self.assertEqual(row['path'],'home/.duckdb/extensions/v1.5.5/linux_arm64/'+Path(source['path']).name)
        for source,row in zip(v['torch'],library.ROWS[3:]):
            self.assertEqual(row['sha256'],source['sha256']);self.assertEqual(row['expected_bytes'],source['bytes'])
            self.assertEqual(row['path'],'venv/lib/python3.12/site-packages/'+source['wheel_member'])
            self.assertEqual(row['mode'],source['mode'])

    def test_exact_isolated_library_loader_binds_all_dependencies(self):
        from run_diagnostic import library_cache_loader
        base=Path(c.__file__).with_name('cache_candidate.py');native=Path(c.__file__)
        small=self.root/'fixture.py';small.write_text("from codex_cache_candidate import advise_selected\ndef main():\n print('bound-native-loop')\n")
        args=[str(base),hashlib.sha256(base.read_bytes()).hexdigest(),str(native),hashlib.sha256(native.read_bytes()).hexdigest(),str(small),hashlib.sha256(small.read_bytes()).hexdigest()]
        def run(values):return subprocess.run([sys.executable,'-I','-S','-B','-c',library_cache_loader(*values)],capture_output=True,text=True,timeout=5)
        ok=run(args);self.assertEqual(ok.returncode,0,ok.stderr);self.assertEqual(ok.stdout.strip(),'bound-native-loop')
        for index in (1,3,5):
            values=list(args);values[index]='0'*64;bad=run(values)
            self.assertNotEqual(bad.returncode,0);self.assertNotIn('bound-native-loop',bad.stdout)

    def test_optional_snapshot_and_error_write_cannot_replace_context_outcome(self):
        # Execute the actual wrapper function with failing optional telemetry.
        path=Path(c.__file__).with_name('run_diagnostic.py')
        tree=ast.parse(path.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='contextualized')
        primary=ValueError('retained primary context failure')
        async def execute(**kwargs):raise OSError('optional snapshot unavailable')
        def write(*args):raise OSError('optional receipt storage unavailable')
        namespace=dict(deployment=SimpleNamespace(PYTHON='python',runtime_environment=lambda:{}),
            json=json,shlex=__import__('shlex'),write=write)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
        async def fails(environment,**kwargs):raise primary
        namespace['original_context']=fails
        with self.assertRaises(ValueError) as raised:asyncio.run(namespace['contextualized'](SimpleNamespace(exec=execute)))
        self.assertIs(raised.exception,primary)
        async def succeeds(environment,**kwargs):return {'qualified':True}
        namespace['original_context']=succeeds
        self.assertEqual(asyncio.run(namespace['contextualized'](SimpleNamespace(exec=execute))),{'qualified':True})

if __name__=='__main__':unittest.main()
