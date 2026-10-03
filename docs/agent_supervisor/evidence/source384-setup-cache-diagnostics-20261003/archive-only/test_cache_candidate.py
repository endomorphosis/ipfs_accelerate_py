import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cache_candidate as c

class CacheCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'toolchains/lean/a'
        self.path.parent.mkdir(parents=True);self.path.write_bytes(b'unchanged\x00\xff'*1000)
        self.path.chmod(0o644)
        self.rows=[dict(path='toolchains/lean/a',bytes=self.path.stat().st_size,mode=0o644,
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())]

    def test_actual_syscalls_preserve_bytes_and_metadata(self):
        raw=self.path.read_bytes();before=c.identity(self.path.stat())
        result=c.advise(self.root,self.rows)
        self.assertEqual(c.identity(self.path.stat()),before)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual((result['advised_files'],result['body_reads'],result['body_writes']),(1,0,0))
        self.assertFalse(result['freed_bytes_claimed'])

    def test_empty_duplicate_outside_and_traversal_rejected(self):
        cases=[[],self.rows*2]
        for name in ('../a','/etc/passwd','source/../a','source//a','source/./a',
                     'home/.codex/auth.json','state/key','worktrees/a','app/bottle.py','source/a\\b'):
            cases.append([{**self.rows[0],'path':name}])
        with patch.object(c.os,'posix_fadvise') as hint:
            for rows in cases:
                with self.subTest(rows=rows),self.assertRaises(ValueError):c.advise(self.root,rows)
            hint.assert_not_called()

    def test_leaf_symlink_rejected_before_advice(self):
        self.path.unlink();self.path.symlink_to('/etc/passwd')
        with patch.object(c.os,'posix_fadvise') as hint,self.assertRaises(OSError):c.advise(self.root,self.rows)
        hint.assert_not_called()

    def test_ancestor_symlink_rejected_before_advice(self):
        self.path.unlink();self.path.parent.rmdir();self.path.parent.symlink_to('/tmp')
        with patch.object(c.os,'posix_fadvise') as hint,self.assertRaises(OSError):c.advise(self.root,self.rows)
        hint.assert_not_called()

    def test_size_mode_nonregular_and_hardlink_rejected(self):
        with patch.object(c.os,'posix_fadvise') as hint:
            for changed in ({'bytes':1},{'mode':0o755}):
                with self.assertRaises(ValueError):c.advise(self.root,[{**self.rows[0],**changed}])
            os.link(self.path,self.root/'hardlink')
            with self.assertRaises(ValueError):c.advise(self.root,self.rows)
            (self.root/'hardlink').unlink();self.path.unlink();os.mkfifo(self.path)
            with self.assertRaises(ValueError):c.advise(self.root,self.rows)
            hint.assert_not_called()

    def test_late_invalid_row_is_checked_before_any_advice(self):
        with patch.object(c.os,'posix_fadvise') as hint,self.assertRaises(OSError):
            c.advise(self.root,self.rows+[{**self.rows[0],'path':'source/missing'}])
        hint.assert_not_called()

    def test_sync_precedes_hint_and_errors_not_counted(self):
        calls=[]
        with patch.object(c.os,'fdatasync',side_effect=lambda fd:calls.append('sync')):
            with patch.object(c.os,'posix_fadvise',side_effect=lambda *args:calls.append('hint')):
                c.advise(self.root,self.rows)
        self.assertEqual(calls,['sync','hint'])
        with patch.object(c.os,'fdatasync',side_effect=OSError(5,'test')),patch.object(c.os,'posix_fadvise') as hint:
            result=c.advise(self.root,self.rows)
        self.assertEqual(result['advised_files'],0);self.assertEqual(len(result['errors']),1)
        hint.assert_not_called()

    def test_mutation_during_advice_refused(self):
        with patch.object(c.os,'posix_fadvise',side_effect=lambda *args:self.path.chmod(0o755)):
            with self.assertRaises(ValueError):c.advise(self.root,self.rows)

    def test_fixed_inventory_bounds(self):
        for rows in ([self.rows[0]]*(c.MAX_FILES+1),[{**self.rows[0],'bytes':c.MAX_BYTES+1}],
                     [{**self.rows[0],'mode':True}],[{**self.rows[0],'sha256':'bad'}]):
            with self.assertRaises(ValueError):c.checked_rows(rows)

    def test_manifest_reader_rejects_fifo_symlink_and_oversized(self):
        path=self.root/'manifest.json';os.mkfifo(path)
        with self.assertRaises(ValueError):c.manifest_bytes(path)
        path.unlink();path.symlink_to(self.path)
        with self.assertRaises(OSError):c.manifest_bytes(path)
        path.unlink();path.write_bytes(b'{}')
        self.assertEqual(c.manifest_bytes(path),b'{}')
        with patch.object(c,'MAX_MANIFEST',1),self.assertRaises(ValueError):c.manifest_bytes(path)

    def test_refusal_captures_exact_row_and_nofollow_metadata(self):
        self.path.unlink();self.path.parent.rmdir();self.path.parent.symlink_to('/tmp')
        with self.assertRaises(c.PathRefusal) as caught:c.advise(self.root,self.rows)
        detail=caught.exception.details
        self.assertEqual(detail['row_index'],0)
        self.assertEqual(detail['manifest_row'],self.rows[0])
        self.assertEqual(detail['entry_nofollow']['kind'],'symlink')
        self.assertEqual(detail['root_descriptor']['descriptor_target'],str(self.root))
        self.assertFalse(detail['advice_issued']);self.assertFalse(detail['body_read'])
        self.assertNotIn('symlink_target',detail['entry_nofollow'])

    def test_non_directory_ancestor_is_distinguished(self):
        self.path.unlink();self.path.parent.rmdir();self.path.parent.write_bytes(b'opaque')
        with self.assertRaises(c.PathRefusal) as caught:c.advise(self.root,self.rows)
        self.assertEqual(caught.exception.details['entry_nofollow']['kind'],'regular')

if __name__=='__main__':unittest.main()
