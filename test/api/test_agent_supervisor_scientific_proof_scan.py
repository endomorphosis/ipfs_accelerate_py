"""Bounded proof traversal unit fixtures; these establish no live health.

The same stdlib unittest class can run against a private AST-only helper subject.
Synthetic special-file metadata is explicitly modeled; no OS links/FIFO are made.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as executor


class TestScientificProofScan(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='legal-ir-private-proof-scan-')
        self.base=Path(self.temporary.name);self.root=self.base/'proof';self.root.mkdir(mode=0o700)

    def tearDown(self):
        self.temporary.cleanup()

    def walk(self,root=None,**limits):
        chosen=dict(max_entries=128,max_directories=64,max_pending=64);chosen.update(limits)
        return executor._bounded_proof_walk(self.root if root is None else root,**chosen)

    def test_nested_hidden_pathset_and_generation_directory_digest_match_rglob(self):
        (self.root/'.git').mkdir();(self.root/'.git'/'objects').mkdir()
        (self.root/'empty').mkdir();(self.root/'a').write_bytes(b'abc')
        (self.root/'.hidden').write_bytes(b'hidden');(self.root/'.git'/'objects'/'blob').write_bytes(b'obj')
        expected=list(self.root.rglob('*'));actual=list(self.walk())
        self.assertEqual({str(p.relative_to(self.root))for p in actual},
                         {str(p.relative_to(self.root))for p in expected})
        def digest(paths):
            rows=[('.',stat.S_IMODE(self.root.lstat().st_mode))]
            rows.extend((p.relative_to(self.root).as_posix(),stat.S_IMODE(p.lstat().st_mode))
                        for p in paths if p.is_dir())
            return hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()
        self.assertEqual(digest(actual),digest(expected))

    def test_missing_and_file_roots_remain_empty(self):
        file=self.root/'ordinary';file.write_bytes(b'x')
        for root in (self.root/'missing',file):
            with self.subTest(root_kind='file'if root==file else'missing'):
                self.assertEqual(list(self.walk(root)),[])

    def test_deep_empty_directories_hit_directory_cap(self):
        cursor=self.root
        for index in range(6):cursor=cursor/str(index);cursor.mkdir()
        with self.assertRaisesRegex(executor.ExecutorHold,'proof_scan_directory_bound'):
            list(self.walk(max_directories=3))

    def test_wide_empty_directories_hit_pending_cap(self):
        for index in range(6):(self.root/str(index)).mkdir()
        with self.assertRaisesRegex(executor.ExecutorHold,'proof_scan_pending_bound'):
            list(self.walk(max_pending=2))

    def test_entry_cap_is_incremental_and_closes_iterator(self):
        state=dict(read=0,closed=False)
        class Rows:
            def __enter__(self):return self
            def __exit__(self,*args):state['closed']=True
            def __iter__(self):return self
            def __next__(self):
                state['read']+=1
                if state['read']>4:raise AssertionError('unbounded iterator consumption')
                return SimpleNamespace(name='file-'+str(state['read']),
                    stat=lambda **kwargs:SimpleNamespace(st_mode=stat.S_IFREG|0o664))
        with patch.object(executor.os,'scandir',return_value=Rows()):
            with self.assertRaisesRegex(executor.ExecutorHold,'proof_scan_entry_bound'):
                list(self.walk(max_entries=3))
        self.assertEqual(state,dict(read=4,closed=True))

    def test_caps_require_positive_exact_integers(self):
        for field in ('max_entries','max_directories','max_pending'):
            for value in (0,-1,True,1.5):
                with self.subTest(field=field,value=value):
                    with self.assertRaisesRegex(executor.ExecutorHold,'proof_scan_bounds'):
                        list(self.walk(**{field:value}))

    def test_scandir_closes_before_descending(self):
        (self.root/'child').mkdir();(self.root/'child'/'data').write_bytes(b'x')
        original=executor.os.scandir;state=dict(active=0,peak=0)
        class Rows:
            def __init__(self,path):self.iterator=original(path)
            def __enter__(self):
                state['active']+=1;state['peak']=max(state['peak'],state['active']);return self.iterator
            def __exit__(self,*args):self.iterator.close();state['active']-=1
        with patch.object(executor.os,'scandir',side_effect=Rows):list(self.walk())
        self.assertEqual(state,dict(active=0,peak=1))

    def test_scandir_open_iteration_and_context_errors_keep_legacy_suppression(self):
        for fault in ('open','enter','iteration','exit'):
            with self.subTest(fault=fault):
                state=dict(read=0)
                class Rows:
                    def __enter__(self):
                        if fault=='enter':raise OSError('modeled scandir context entry')
                        return self
                    def __exit__(self,*args):
                        if fault=='exit':raise OSError('modeled scandir context exit')
                    def __iter__(self):return self
                    def __next__(self):
                        state['read']+=1
                        if state['read']==1:
                            return SimpleNamespace(name='prefix',
                                stat=lambda **kwargs:SimpleNamespace(st_mode=stat.S_IFREG|0o600))
                        if fault=='iteration':raise OSError('modeled mid-iterator fault')
                        raise StopIteration
                def scan(path):
                    if fault=='open':raise OSError('modeled scandir open fault')
                    return Rows()
                with patch.object(executor.os,'scandir',side_effect=scan):
                    self.assertEqual(list(self.walk()),[])

    def test_modeled_symlink_directory_is_yielded_without_recursion(self):
        calls=[]
        class Rows:
            def __enter__(self):return iter([SimpleNamespace(name='alias',
                stat=lambda **kwargs:SimpleNamespace(st_mode=stat.S_IFLNK|0o777))])
            def __exit__(self,*args):pass
        def scan(path):
            calls.append(Path(path))
            if Path(path)!=self.root:raise AssertionError('descended a modeled symlink')
            return Rows()
        with patch.object(executor.os,'scandir',side_effect=scan):paths=list(self.walk())
        self.assertEqual(paths,[self.root/'alias']);self.assertEqual(calls,[self.root])

    def test_budget_keeps_group_writable_data_and_byte_reserves(self):
        data=self.root/'data';data.write_bytes(b'abc');data.chmod(0o664)
        self.assertEqual(executor.check_proof_budget(self.root,5,reserve_bytes=2),{'files':1,'bytes':3})
        with self.assertRaisesRegex(executor.ExecutorHold,'proof_projected_storage_bound'):
            executor.check_proof_budget(self.root,5,reserve_bytes=3)
        self.assertEqual(stat.S_IMODE(data.lstat().st_mode),0o664)

    def test_budget_file_count_reserves_remain_exact(self):
        info=SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=0)
        item=SimpleNamespace(lstat=lambda:info)
        for count,reserve,accepted in ((19999,1,True),(20000,0,True),(20000,1,False),(20001,0,False)):
            with self.subTest(count=count,reserve=reserve):
                def walk(*args,**kwargs):return(iter(item for _ in range(count)))
                with patch.object(executor,'_bounded_proof_walk',side_effect=walk):
                    if accepted:
                        self.assertEqual(executor.check_proof_budget(self.root,1,reserve_files=reserve),
                                         {'files':count,'bytes':0})
                    else:
                        with self.assertRaisesRegex(executor.ExecutorHold,'proof_.*storage_bound'):
                            executor.check_proof_budget(self.root,1,reserve_files=reserve)

    def test_budget_retains_modeled_symlink_hardlink_and_fifo_refusals(self):
        data=self.root/'ordinary';data.write_bytes(b'')
        for mode,nlink,reason in ((stat.S_IFLNK|0o777,1,'proof_symlink'),
                                  (stat.S_IFREG|0o600,2,'proof_nonordinary_artifact'),
                                  (stat.S_IFIFO|0o600,1,'proof_nonordinary_artifact')):
            with self.subTest(mode=mode,nlink=nlink):
                info=SimpleNamespace(st_mode=mode,st_nlink=nlink,st_size=0)
                with patch.object(Path,'lstat',return_value=info):
                    with self.assertRaisesRegex(executor.ExecutorHold,reason):
                        executor.check_proof_budget(self.root,1)

    def test_budget_does_not_add_directory_mtime_snapshot_holds(self):
        (self.root/'ordinary').write_bytes(b'x');original=executor.os.scandir;made=[]
        class Rows:
            def __init__(self,path):self.path=Path(path);self.iterator=original(path)
            def __enter__(self):return self.iterator
            def __exit__(self,*args):
                self.iterator.close()
                if self.path==self_root and not made:
                    (self_root/'concurrent-empty').mkdir();made.append(True)
        self_root=self.root
        with patch.object(executor.os,'scandir',side_effect=Rows):
            self.assertEqual(executor.check_proof_budget(self.root,1),{'files':1,'bytes':1})
        self.assertTrue((self.root/'concurrent-empty').is_dir())
